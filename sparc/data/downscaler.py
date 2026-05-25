"""LayerDownscaler — canonical seam for assigning source data onto fishnet cells.

All four spatial downscaling patterns used by the data collection pipeline are
unified behind this single interface.  Collect modules no longer own downscaling
logic; they call the appropriate adapter method and receive a ``pd.Series``.

Adapters
--------
from_raster_file(grid, raster_path, stat)
    Raster file (GeoTIFF) → zonal statistic per cell via ``rasterstats``.
    Used by: NLCD, LiDAR building heights.

from_raster_array(grid, array, transform, stat)
    In-memory NumPy array with an affine transform → zonal stat per cell.
    Used by: Landsat COG tiles (already loaded into memory).

from_grid_points(grid, lons, lats, values, method)
    Scattered point measurements → interpolated value per cell centroid.
    Bilinear interpolation with nearest-neighbour fallback for NaN cells.
    Used by: ERA5 (0.25° grid), CAPA station data.

from_vector_footprints(grid, footprints, value_col, agg)
    Vector polygon footprints → aggregated value per cell.
    Cells with no overlapping footprints receive 0.0.
    Used by: building footprints (height, coverage).

All adapters accept a :class:`~sparc.data.spatial_grid.SpatialGrid` as their
first argument and return a ``pd.Series`` indexed 0…N-1 matching the grid rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Union

import numpy as np
import pandas as pd

try:
    import geopandas as gpd

    HAS_GEO = True
except ImportError:
    HAS_GEO = False


StatName = str
AggName = Literal["mean", "sum", "max", "min", "count"]
InterpMethod = Literal["linear", "nearest", "cubic"]


class LayerDownscaler:
    """Adapters that place a source dataset onto 30 m fishnet cells.

    All methods accept a :class:`~sparc.data.spatial_grid.SpatialGrid` and
    return a ``pd.Series`` of length ``len(grid)`` indexed from 0.
    """

    # ------------------------------------------------------------------
    # Adapter A — raster file
    # ------------------------------------------------------------------

    def from_raster_file(
        self,
        grid: "object",
        raster_path: Union[str, Path],
        stat: StatName = "mean",
        nodata: float | None = None,
        all_touched: bool = False,
    ) -> pd.Series:
        """Zonal statistic from a GeoTIFF file onto each grid cell.

        Parameters
        ----------
        grid : SpatialGrid
            Analysis grid.
        raster_path : str | Path
            Path to a GeoTIFF raster.
        stat : str
            rasterstats statistic name: ``"mean"``, ``"sum"``, ``"majority"``, etc.
        nodata : float | None
            No-data sentinel value passed to rasterstats.
        all_touched : bool
            If True, include cells that touch the raster pixel boundary.

        Returns
        -------
        pd.Series — one value per grid row; NaN where the raster has no data.
        """
        try:
            import rasterstats
        except ImportError as exc:
            raise ImportError("rasterstats is required: pip install rasterstats") from exc

        cells = _cells(grid)
        results = rasterstats.zonal_stats(
            cells,
            str(raster_path),
            stats=[stat],
            nodata=nodata,
            all_touched=all_touched,
        )
        return pd.Series([r.get(stat) for r in results], name=stat, dtype=float)

    # ------------------------------------------------------------------
    # Adapter B — in-memory raster array
    # ------------------------------------------------------------------

    def from_raster_array(
        self,
        grid: "object",
        array: np.ndarray,
        transform: "object",
        stat: StatName = "mean",
        nodata: float | None = None,
        all_touched: bool = False,
    ) -> pd.Series:
        """Zonal statistic from an in-memory NumPy array with affine transform.

        Parameters
        ----------
        grid : SpatialGrid
        array : np.ndarray
            2-D raster data array.
        transform : affine.Affine
            Affine transform matching the array's CRS.
        stat : str
            rasterstats statistic name.
        nodata : float | None
        all_touched : bool

        Returns
        -------
        pd.Series — one value per grid row.
        """
        try:
            import rasterstats
        except ImportError as exc:
            raise ImportError("rasterstats is required: pip install rasterstats") from exc

        cells = _cells(grid)
        results = rasterstats.zonal_stats(
            cells,
            array,
            affine=transform,
            stats=[stat],
            nodata=nodata,
            all_touched=all_touched,
        )
        return pd.Series([r.get(stat) for r in results], name=stat, dtype=float)

    # ------------------------------------------------------------------
    # Adapter C — scattered grid points (ERA5, CAPA)
    # ------------------------------------------------------------------

    def from_grid_points(
        self,
        grid: "object",
        lons: np.ndarray,
        lats: np.ndarray,
        values: np.ndarray,
        method: InterpMethod = "linear",
    ) -> pd.Series:
        """Interpolate scattered point measurements to each cell centroid.

        Bilinear (``"linear"``) interpolation is tried first.  Any NaN cells
        (outside the convex hull of source points) are filled using nearest-
        neighbour, so the result is always fully populated.

        Parameters
        ----------
        grid : SpatialGrid
        lons, lats : array-like
            Source point coordinates in EPSG:4326.
        values : array-like
            Measurement values at each (lon, lat) point.
        method : "linear" | "nearest" | "cubic"
            Primary interpolation method.  Nearest fallback is always applied.

        Returns
        -------
        pd.Series — one interpolated value per grid row, no NaNs.
        """
        try:
            from scipy.interpolate import griddata
        except ImportError as exc:
            raise ImportError("scipy is required: pip install scipy") from exc

        centroids = _centroids(grid)  # (N, 2) lon/lat
        src_pts = np.column_stack([np.asarray(lons), np.asarray(lats)])
        vals = np.asarray(values, dtype=float)

        # griddata with "linear" or "cubic" requires enough points for a
        # Delaunay triangulation (≥3 in 2D). Fall back to nearest when the
        # primary method fails (e.g. only 1–2 source points supplied).
        try:
            interpolated = griddata(src_pts, vals, centroids, method=method)
        except Exception:
            interpolated = np.full(len(centroids), np.nan)

        # Fill NaN cells (outside hull) with nearest-neighbour
        nan_mask = np.isnan(interpolated)
        if nan_mask.any():
            nearest = griddata(src_pts, vals, centroids[nan_mask], method="nearest")
            interpolated[nan_mask] = nearest

        return pd.Series(interpolated, name="interpolated", dtype=float)

    # ------------------------------------------------------------------
    # Adapter D — vector footprints (buildings)
    # ------------------------------------------------------------------

    def from_vector_footprints(
        self,
        grid: "object",
        footprints: "gpd.GeoDataFrame",
        value_col: str,
        agg: AggName = "mean",
    ) -> pd.Series:
        """Aggregate polygon footprint values onto each grid cell.

        Cells with no overlapping footprints receive ``0.0``.

        Parameters
        ----------
        grid : SpatialGrid
        footprints : gpd.GeoDataFrame
            Polygon features with a numeric *value_col*.
        value_col : str
            Column in *footprints* to aggregate.
        agg : "mean" | "sum" | "max" | "min" | "count"
            Aggregation function name.

        Returns
        -------
        pd.Series — one aggregated value per grid row; 0.0 for empty cells.
        """
        if not HAS_GEO:
            raise ImportError("geopandas is required: pip install geopandas")

        cells = _cells(grid)
        n = len(cells)

        # Project footprints to match cells CRS
        if footprints.crs != cells.crs:
            footprints = footprints.to_crs(cells.crs)

        # Add row index to cells so we can group by cell after spatial join
        cells_indexed = cells.copy()
        cells_indexed["_cell_idx"] = np.arange(n)

        joined = gpd.sjoin(
            footprints[[value_col, "geometry"]],
            cells_indexed[["_cell_idx", "geometry"]],
            how="left",
            predicate="intersects",
        )

        if joined.empty or joined["_cell_idx"].isna().all():
            return pd.Series(np.zeros(n, dtype=float))

        agg_fn = {"mean": "mean", "sum": "sum", "max": "max", "min": "min", "count": "count"}
        grouped = joined.groupby("_cell_idx")[value_col].agg(agg_fn[agg])

        result = pd.Series(np.zeros(n, dtype=float), name=value_col)
        result.update(grouped)
        return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cells(grid: "object") -> "gpd.GeoDataFrame":
    """Extract the projected GeoDataFrame from a SpatialGrid."""
    if hasattr(grid, "cells_3857"):
        return grid.cells_3857  # type: ignore[union-attr]
    # Fallback: bare GeoDataFrame passed directly
    return grid  # type: ignore[return-value]


def _centroids(grid: "object") -> np.ndarray:
    """Extract (N, 2) lon/lat centroid array from a SpatialGrid."""
    if hasattr(grid, "centroids_4326"):
        return grid.centroids_4326  # type: ignore[union-attr]
    # Fallback: bare GeoDataFrame — compute centroids on the fly
    import geopandas as gpd
    gdf: gpd.GeoDataFrame = grid  # type: ignore[assignment]
    geo = gdf.geometry.centroid.to_crs("EPSG:4326")
    return np.column_stack([geo.x, geo.y])
