"""Data processing utilities — fishnet creation, zonal statistics, boundary clipping."""

from __future__ import annotations

import warnings
from enum import Enum
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


class ClipMethod(str, Enum):
    """How fishnet cells are filtered against the study-area boundary.

    CENTROID_WITHIN
        Retain cells whose centroid falls inside the boundary (original
        behaviour; fast but lossy at boundary edges).
    AREA_PROPORTION
        Retain cells whose intersection area with the boundary is at least
        *min_coverage* (0–1) of the cell's own area.
    INTERSECTION
        Clip each cell to the exact boundary intersection; returns trimmed
        geometries instead of the original full cells.
    """

    CENTROID_WITHIN = "centroid_within"
    AREA_PROPORTION = "area_proportion"
    INTERSECTION = "intersection"


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
    method: ClipMethod = ClipMethod.CENTROID_WITHIN,
    min_coverage: float = 0.0,
) -> "gpd.GeoDataFrame":
    """Clip fishnet cells to a study-area boundary polygon.

    Parameters
    ----------
    fishnet : GeoDataFrame of polygon cells.
    boundary : GeoDataFrame containing the study-area boundary polygon(s).
    method : ClipMethod
        How cells are filtered / trimmed.  See :class:`ClipMethod` for details.
        Default is ``CENTROID_WITHIN`` (original behaviour).
    min_coverage : float
        Only used with ``AREA_PROPORTION``.  Cells with
        ``intersection_area / cell_area < min_coverage`` are dropped.
        Values in [0, 1]; default 0.0 keeps any cell with non-zero overlap.

    Returns
    -------
    GeoDataFrame — filtered (or trimmed) fishnet cells.
    """
    if not HAS_GEO:
        raise ImportError("geopandas is required")

    boundary_union = boundary.geometry.unary_union

    if method is ClipMethod.CENTROID_WITHIN:
        mask = fishnet.geometry.centroid.within(boundary_union)
        return fishnet.loc[mask].reset_index(drop=True)

    if method is ClipMethod.AREA_PROPORTION:
        cell_area = fishnet.geometry.area
        overlap = fishnet.geometry.intersection(boundary_union).area
        coverage = overlap / cell_area.replace(0, float("nan"))
        mask = coverage >= min_coverage
        return fishnet.loc[mask].reset_index(drop=True)

    if method is ClipMethod.INTERSECTION:
        result = fishnet.copy()
        result["geometry"] = fishnet.geometry.intersection(boundary_union)
        # Drop cells with no intersection (empty geometry)
        result = result[~result.geometry.is_empty].reset_index(drop=True)
        return result

    raise ValueError(f"Unknown ClipMethod: {method!r}")


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
