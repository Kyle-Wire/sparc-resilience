"""SpatialGrid — the analysis grid with an explicit CRS contract.

Wraps the analysis fishnet GeoDataFrame and exposes both the projected
view (project working CRS, metric units) and the geographic centroid view
(EPSG:4326, lon/lat) through a stable interface.  The geographic centroids
are computed once and cached, eliminating repeated `.to_crs()` calls across
collect modules.

The working CRS defaults to EPSG:3857 (Web Mercator) for backward
compatibility, but any projected CRS can be supplied via ``from_boundary()``.

Usage
-----
::

    from sparc.data.spatial_grid import SpatialGrid
    from sparc.data.collect.boundary import BoundaryResult

    # Default: EPSG:3857
    grid = SpatialGrid.from_boundary(boundary, resolution_m=30.0)

    # Custom working CRS (e.g., UTM zone 19N)
    grid = SpatialGrid.from_boundary(boundary, resolution_m=30.0, crs="EPSG:32619")

    # Projected cells for rasterio / rasterstats operations
    cells = grid.cells_proj           # preferred — CRS-neutral name
    cells = grid.cells_3857           # backward-compat alias (same object)

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

# Default projected CRS — used when no CRS is supplied to from_boundary()
_DEFAULT_PROJECTED_CRS = "EPSG:3857"
_GEOGRAPHIC_CRS = "EPSG:4326"

# Default fishnet resolution used by the data collection pipeline
DEFAULT_RESOLUTION_M = 30.0


@dataclass
class SpatialGrid:
    """Analysis grid with explicit CRS context.

    Attributes
    ----------
    cells_proj : gpd.GeoDataFrame
        Polygon cells in the project working CRS (metric units).
        Use this for rasterio and rasterstats operations.
    projected_crs : str
        EPSG code string of the working CRS, e.g. ``"EPSG:3857"`` or
        ``"EPSG:32619"``.  Set automatically by :meth:`from_boundary`.
    cells_3857 : property
        Backward-compatible alias for ``cells_proj``.  Returns the same
        GeoDataFrame regardless of the actual CRS.
    _centroids_4326_cache : np.ndarray | None
        Internal cache.  Access via the ``centroids_4326`` property.
    """

    cells_proj: "gpd.GeoDataFrame"
    projected_crs: str = _DEFAULT_PROJECTED_CRS
    _centroids_4326_cache: Optional[np.ndarray] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Backward-compat alias — callers using cells_3857 continue to work
    # ------------------------------------------------------------------

    @property
    def cells_3857(self) -> "gpd.GeoDataFrame":
        """Backward-compatible alias for :attr:`cells_proj`.

        Returns the projected fishnet GeoDataFrame regardless of the actual
        working CRS.  Prefer ``cells_proj`` in new code.
        """
        return self.cells_proj

    # ------------------------------------------------------------------
    # Geographic centroid view — cached
    # ------------------------------------------------------------------

    @property
    def centroids_4326(self) -> np.ndarray:
        """Geographic centroids as (N, 2) array of [lon, lat] in EPSG:4326.

        Computed once on first access and cached for all subsequent calls.
        """
        if self._centroids_4326_cache is None:
            geo = self.cells_proj.geometry.centroid.to_crs(_GEOGRAPHIC_CRS)
            self._centroids_4326_cache = np.column_stack([geo.x, geo.y])
        return self._centroids_4326_cache

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.cells_proj)

    @property
    def bbox_4326(self) -> tuple[float, float, float, float]:
        """(minx, miny, maxx, maxy) bounding box in EPSG:4326."""
        bounds = self.cells_proj.to_crs(_GEOGRAPHIC_CRS).total_bounds
        return (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_boundary(
        cls,
        boundary: "object",
        resolution_m: float = DEFAULT_RESOLUTION_M,
        crs: str = _DEFAULT_PROJECTED_CRS,
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
        crs : str
            Projected CRS for the fishnet, e.g. ``"EPSG:3857"`` (default) or
            ``"EPSG:32619"`` (UTM zone 19N).  Must be a projected (metric or
            feet-based) CRS — geographic CRS values are rejected.

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

        # Project boundary to the working CRS for metric-unit fishnet creation
        boundary_proj = boundary_gdf.to_crs(crs)
        minx, miny, maxx, maxy = boundary_proj.total_bounds

        from sparc.data.processing import create_fishnet, ClipMethod, clip_to_boundary

        fishnet = create_fishnet(
            bounds=(minx, miny, maxx, maxy),
            resolution=resolution_m,
            crs=crs,
        )

        # Clip to study boundary using centroid-within (default)
        fishnet = clip_to_boundary(
            fishnet,
            boundary_proj,
            method=ClipMethod.CENTROID_WITHIN,
        )

        return cls(cells_proj=fishnet.reset_index(drop=True), projected_crs=crs)
