"""Data processing utilities — fishnet creation, zonal statistics, boundary clipping."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    from shapely.geometry import box

    HAS_GEO = True
except ImportError:
    HAS_GEO = False

try:
    from rasterstats import zonal_stats as _zonal_stats

    HAS_RASTERSTATS = True
except ImportError:
    HAS_RASTERSTATS = False


def create_fishnet(
    bounds: Tuple[float, float, float, float],
    resolution: float,
    crs: str = "EPSG:4326",
) -> "gpd.GeoDataFrame":
    """Create a rectangular fishnet grid over *bounds* at the given *resolution*.

    Parameters
    ----------
    bounds : (minx, miny, maxx, maxy)
    resolution : cell size in CRS units (metres if projected, degrees if geographic).
    crs : coordinate reference system string.

    Returns
    -------
    GeoDataFrame with polygon cells and ``OBJECTID`` column.
    """
    if not HAS_GEO:
        raise ImportError("geopandas is required for fishnet creation")

    minx, miny, maxx, maxy = bounds
    xs = np.arange(minx, maxx, resolution)
    ys = np.arange(miny, maxy, resolution)
    cells = []
    for x in xs:
        for y in ys:
            cells.append(box(x, y, x + resolution, y + resolution))

    gdf = gpd.GeoDataFrame(geometry=cells, crs=crs)
    gdf["OBJECTID"] = range(1, len(gdf) + 1)
    # Centroid coords for downstream modelling
    gdf["centroid_x"] = gdf.geometry.centroid.x
    gdf["centroid_y"] = gdf.geometry.centroid.y
    return gdf


def clip_to_boundary(
    fishnet: "gpd.GeoDataFrame",
    boundary: "gpd.GeoDataFrame",
) -> "gpd.GeoDataFrame":
    """Clip fishnet cells to a study-area boundary polygon.

    Only cells whose centroid falls inside the boundary are retained.
    """
    if not HAS_GEO:
        raise ImportError("geopandas is required")
    boundary_union = boundary.geometry.unary_union
    mask = fishnet.geometry.centroid.within(boundary_union)
    return fishnet.loc[mask].reset_index(drop=True)


def run_zonal_stats(
    fishnet: "gpd.GeoDataFrame",
    raster_paths: List[str],
    stats: str = "mean",
) -> "gpd.GeoDataFrame":
    """Compute zonal statistics for each raster layer on *fishnet* cells.

    Parameters
    ----------
    fishnet : GeoDataFrame of polygon cells.
    raster_paths : list of raster file paths (GeoTIFF).
    stats : space-separated stat names (default ``"mean"``).

    Returns the fishnet with new columns ``<layer>_<stat>`` appended.
    """
    if not HAS_RASTERSTATS:
        raise ImportError("rasterstats is required — pip install rasterstats")
    if not HAS_GEO:
        raise ImportError("geopandas is required")

    for rpath in raster_paths:
        layer_name = Path(rpath).stem
        results = _zonal_stats(fishnet, rpath, stats=stats.split())
        for s in stats.split():
            fishnet[f"{layer_name}_{s}"] = [r.get(s) for r in results]

    return fishnet
