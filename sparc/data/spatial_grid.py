"""SpatialGrid — the analysis grid with an explicit CRS contract.

Wraps the 30 m analysis fishnet GeoDataFrame and exposes both the projected
view (EPSG:3857, metric units) and the geographic centroid view (EPSG:4326,
lon/lat) through a stable interface.  The geographic centroids are computed
once and cached, eliminating repeated `.to_crs()` calls across collect modules.

Usage
-----
::

    from sparc.data.spatial_grid import SpatialGrid
    from sparc.data.collect.boundary import BoundaryResult

    grid = SpatialGrid.from_boundary(boundary, resolution_m=30.0)

    # Projected cells for rasterio / rasterstats operations
    cells = grid.cells_3857

    # Geographic centroids (N, 2) for griddata / API bounding boxes
    lons_lats = grid.centroids_4326

"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import geopandas as gpd

    HAS_GEO = True
except ImportError:
    HAS_GEO = False

# Web Mercator — 30 m units are exact at mid-latitudes
_PROJECTED_CRS = "EPSG:3857"
_GEOGRAPHIC_CRS = "EPSG:4326"

# Default fishnet resolution used by the data collection pipeline
DEFAULT_RESOLUTION_M = 30.0


@dataclass
class SpatialGrid:
    """Analysis grid with explicit CRS context.

    Attributes
    ----------
    cells_3857 : gpd.GeoDataFrame
        Polygon cells in EPSG:3857 (Web Mercator, metric units).
        This is the working CRS for rasterio and rasterstats operations.
    _centroids_4326_cache : np.ndarray | None
        Internal cache.  Access via the ``centroids_4326`` property.
    """

    cells_3857: "gpd.GeoDataFrame"
    _centroids_4326_cache: Optional[np.ndarray] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Geographic centroid view — cached
    # ------------------------------------------------------------------

    @property
    def centroids_4326(self) -> np.ndarray:
        """Geographic centroids as (N, 2) array of [lon, lat] in EPSG:4326.

        Computed once on first access and cached for all subsequent calls.
        """
        if self._centroids_4326_cache is None:
            geo = self.cells_3857.geometry.centroid.to_crs(_GEOGRAPHIC_CRS)
            self._centroids_4326_cache = np.column_stack([geo.x, geo.y])
        return self._centroids_4326_cache

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.cells_3857)

    @property
    def bbox_4326(self) -> tuple[float, float, float, float]:
        """(minx, miny, maxx, maxy) bounding box in EPSG:4326."""
        bounds = self.cells_3857.to_crs(_GEOGRAPHIC_CRS).total_bounds
        return (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_boundary(
        cls,
        boundary: "object",
        resolution_m: float = DEFAULT_RESOLUTION_M,
    ) -> "SpatialGrid":
        """Build a SpatialGrid from a resolved study-area boundary.

        Parameters
        ----------
        boundary : BoundaryResult | gpd.GeoDataFrame
            Either a ``BoundaryResult`` (from ``sparc.data.collect.boundary``)
            or a raw GeoDataFrame with a geometry column.  Must be in or
            convertible to EPSG:4326.
        resolution_m : float
            Cell size in metres.  Default 30 m.

        Returns
        -------
        SpatialGrid

        Raises
        ------
        ImportError
            If geopandas is not installed.
        """
        if not HAS_GEO:
            raise ImportError("geopandas is required: pip install geopandas")

        # Accept either a BoundaryResult or a raw GeoDataFrame
        if hasattr(boundary, "gdf"):
            boundary_gdf: gpd.GeoDataFrame = boundary.gdf  # type: ignore[union-attr]
        else:
            boundary_gdf = boundary  # type: ignore[assignment]

        # Project boundary to EPSG:3857 for metric-unit fishnet creation
        boundary_3857 = boundary_gdf.to_crs(_PROJECTED_CRS)
        minx, miny, maxx, maxy = boundary_3857.total_bounds

        from sparc.data.processing import create_fishnet, ClipMethod, clip_to_boundary

        fishnet = create_fishnet(
            bounds=(minx, miny, maxx, maxy),
            resolution=resolution_m,
            crs=_PROJECTED_CRS,
        )

        # Clip to study boundary using centroid-within (default)
        fishnet = clip_to_boundary(
            fishnet,
            boundary_3857,
            method=ClipMethod.CENTROID_WITHIN,
        )

        return cls(cells_3857=fishnet.reset_index(drop=True))
