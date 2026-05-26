"""
buildings.py — Building footprints, heights, and Sky View Factor.

Resolves building heights for the study area using a fallback chain:
  Tier 1 — Local LiDAR file (user-provided DSM or building height raster)
  Tier 2 — OpenStreetMap Overpass API (building ways with height/levels tags)
  Tier 3 — OSHB via Microsoft Planetary Computer STAC (raster, ~90m source)
  Tier 4 — Constant default (configurable, e.g. 5.0 m)

For each 30m fishnet cell the module computes:
  bldg_height_mean  — mean building height (m) of all footprints in cell
  bldg_coverage     — fraction of cell area covered by building footprints
  svf               — Sky View Factor via H:W ratio proxy:
                      SVF = W / sqrt(W² + H²)   where W = 30 m cell width
                      SVF = 1.0 for cells with no buildings (open sky)
                      SVF = 0.0 approaches for very tall dense canyons

Optional DSM override: if a DSM file path is supplied, true SVF is computed
via WhiteboxTools (if installed) instead of the H:W proxy.
"""

from __future__ import annotations

import math
import urllib.request
import urllib.parse
import json
from pathlib import Path
from typing import Optional

import numpy as np

HTTP_TIMEOUT = 60.0
CELL_WIDTH_M = 30.0  # 30m fishnet cell width used in SVF formula
_DEFAULT_HEIGHT_M = 5.0  # fallback for buildings without height data

# OpenStreetMap Overpass API
OSM_OVERPASS_URL = "https://overpass-api.de/api/interpreter"


# ---------------------------------------------------------------------------
# Public API — two-phase (download then assign to grid)
# ---------------------------------------------------------------------------

def download_buildings(
    bbox: tuple[float, float, float, float],
    *,
    lidar_path: Optional[Path] = None,
    dsm_path: Optional[Path] = None,
    default_height_m: float = 5.0,
    cache_dir: Optional[Path] = None,
) -> tuple[dict, int]:
    """Download building data for *bbox* without grid assignment.

    This is **Phase 1** of the two-phase collection pattern.  Building
    polygons are fetched from OSM (or OSHB), or a local LiDAR raster path
    is recorded, without requiring a fishnet.  Call
    :func:`assign_buildings_to_grid` once the fishnet has been created.

    Returns
    -------
    (raw_data, tier)
        ``raw_data`` is a dict consumed by :func:`assign_buildings_to_grid`.
        ``tier`` is 1–4 indicating which source was used.
    """
    cache_dir = cache_dir or _default_cache_dir()

    if lidar_path and Path(lidar_path).exists():
        return {"type": "raster", "path": Path(lidar_path), "dsm_path": dsm_path}, 1

    # Tier 2 — OSM Overpass
    try:
        buildings = _download_osm_buildings_raw(bbox, cache_dir)
        if buildings is not None:
            return {"type": "vector", "gdf": buildings, "dsm_path": dsm_path}, 2
    except Exception:
        pass

    # Tier 3 — OSHB (Planetary Computer)
    try:
        oshb_href = _get_oshb_href(bbox)
        if oshb_href is not None:
            return {"type": "raster_url", "href": oshb_href, "dsm_path": dsm_path}, 3
    except Exception:
        pass

    # Tier 4 — constant fallback
    return {"type": "constant", "default_height_m": default_height_m, "dsm_path": dsm_path}, 4


def assign_buildings_to_grid(
    fishnet_gdf: object,
    raw_data: dict,
    tier: int,
    *,
    default_height_m: float = 5.0,
) -> tuple[object, int]:
    """Apply downloaded building data to fishnet cells.

    This is **Phase 2** of the two-phase collection pattern.

    Parameters
    ----------
    fishnet_gdf : gpd.GeoDataFrame
        Analysis grid at any resolution.
    raw_data : dict
        Building raw data from :func:`download_buildings`.
    tier : int
        Tier code from :func:`download_buildings`.
    default_height_m : float
        Height for constant fallback (tier 4).

    Returns
    -------
    (GeoDataFrame, tier_used)
        Fishnet with ``bldg_height_mean``, ``bldg_coverage``, ``svf``; tier int.
    """
    btype = raw_data.get("type", "constant")
    dsm_path = raw_data.get("dsm_path")

    if btype == "raster":
        gdf = _from_raster(fishnet_gdf, raw_data["path"], "bldg_height_mean")
        gdf = _coverage_from_height(gdf)
    elif btype == "vector":
        gdf = _join_buildings_to_fishnet(fishnet_gdf, raw_data["gdf"])
    elif btype == "raster_url":
        gdf = _from_raster(fishnet_gdf, raw_data["href"], "bldg_height_mean")
        gdf = _coverage_from_height(gdf)
    else:
        gdf = fishnet_gdf.copy()  # type: ignore[union-attr]
        gdf["bldg_height_mean"] = raw_data.get("default_height_m", default_height_m)  # type: ignore[index]
        gdf["bldg_coverage"] = 0.3  # type: ignore[index]

    if dsm_path and Path(dsm_path).exists():
        gdf = _true_svf_from_dsm(gdf, Path(dsm_path))
    else:
        gdf = _compute_svf_proxy(gdf)

    return gdf, tier


# ---------------------------------------------------------------------------
# Legacy single-call API (kept for backward compatibility)
# ---------------------------------------------------------------------------

def fetch_buildings(
    fishnet_gdf: object,
    bbox: tuple[float, float, float, float],
    *,
    lidar_path: Optional[Path] = None,
    dsm_path: Optional[Path] = None,
    default_height_m: float = 5.0,
    cache_dir: Optional[Path] = None,
) -> tuple[object, int]:
    """Fetch building heights and compute SVF onto the analysis fishnet.

    Parameters
    ----------
    fishnet_gdf : gpd.GeoDataFrame
        30m analysis grid.  Returns with three new columns.
    bbox : (minx, miny, maxx, maxy)
        Study bounding box in EPSG:4326.
    lidar_path : Path, optional
        Local LiDAR building height raster (GeoTIFF).  Tier 1.
    dsm_path : Path, optional
        Local Digital Surface Model for true SVF override.
    default_height_m : float
        Constant fallback height when all tiers fail.
    cache_dir : Path, optional
        Cache directory for downloaded tiles.

    Returns
    -------
    (GeoDataFrame, tier_used)
        GeoDataFrame with ``bldg_height_mean``, ``bldg_coverage``, ``svf``;
        integer tier (1–4) indicating which source was used.
    """
    cache_dir = cache_dir or _default_cache_dir()
    raw_data, tier = download_buildings(
        bbox,
        lidar_path=lidar_path,
        dsm_path=dsm_path,
        default_height_m=default_height_m,
        cache_dir=cache_dir,
    )
    return assign_buildings_to_grid(fishnet_gdf, raw_data, tier, default_height_m=default_height_m)


# ---------------------------------------------------------------------------
# Tier 1 — LiDAR raster
# ---------------------------------------------------------------------------

def _from_raster(fishnet_gdf: object, raster_path: Path, col_name: str) -> object:
    import rasterstats
    results = rasterstats.zonal_stats(
        fishnet_gdf,  # type: ignore[arg-type]
        str(raster_path),
        stats=["mean"],
        nodata=0,
        all_touched=False,
    )
    gdf = fishnet_gdf.copy()  # type: ignore[union-attr]
    gdf[col_name] = [r.get("mean") or 0.0 for r in results]  # type: ignore[index]
    return gdf


def _coverage_from_height(fishnet_gdf: object) -> object:
    """Estimate coverage as fraction of cell with height > 0 (raster-based proxy)."""
    import rasterstats
    gdf = fishnet_gdf.copy()  # type: ignore[union-attr]
    # For raster-based heights we set coverage = 1.0 where height > 0, else 0
    heights = gdf["bldg_height_mean"].fillna(0.0)  # type: ignore[union-attr]
    gdf["bldg_coverage"] = (heights > 0).astype(float)  # type: ignore[index]
    return gdf


# ---------------------------------------------------------------------------
# Tier 2 — OSM Overpass + fallback chain
# ---------------------------------------------------------------------------

def _from_mlbuildings_or_fallback(
    fishnet_gdf: object,
    bbox: tuple[float, float, float, float],
    default_height_m: float,
    cache_dir: Path,
) -> tuple[object, int]:
    """Try OSM Overpass (Tier 2), OSHB (Tier 3), constant (Tier 4)."""
    try:
        gdf = _from_osm_overpass(fishnet_gdf, bbox, cache_dir)
        if gdf is not None:
            return gdf, 2
    except Exception:
        pass

    try:
        gdf = _from_oshb(fishnet_gdf, bbox, cache_dir)
        if gdf is not None:
            return gdf, 3
    except Exception:
        pass

    # Tier 4 — constant fallback
    gdf = fishnet_gdf.copy()  # type: ignore[union-attr]
    gdf["bldg_height_mean"] = default_height_m  # type: ignore[index]
    gdf["bldg_coverage"] = 0.3  # conservative urban estimate  # type: ignore[index]
    return gdf, 4


def _from_osm_overpass(
    fishnet_gdf: object,
    bbox: tuple[float, float, float, float],
    cache_dir: Path,
) -> Optional[object]:
    """Fetch building footprints from OSM Overpass API and join to fishnet."""
    buildings = _download_osm_buildings_raw(bbox, cache_dir)
    if buildings is None:
        return None
    return _join_buildings_to_fishnet(fishnet_gdf, buildings)


def _download_osm_buildings_raw(
    bbox: tuple[float, float, float, float],
    cache_dir: Path,
) -> Optional[object]:
    """Download OSM building polygons for *bbox* without fishnet assignment.

    Returns a GeoDataFrame of building polygons, or ``None`` if no buildings
    are found or the API is unreachable.  Results are cached as a parquet file.
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Polygon
    except ImportError:
        return None

    minx, miny, maxx, maxy = bbox
    safe_key = (
        f"osm_bldg_{minx:.4f}_{miny:.4f}_{maxx:.4f}_{maxy:.4f}"
        .replace(".", "p").replace("-", "m")
    )
    cache_path = cache_dir / f"{safe_key}.parquet"

    if cache_path.exists():
        try:
            return gpd.read_parquet(cache_path)
        except Exception:
            cache_path.unlink(missing_ok=True)

    query = (
        f"[out:json][timeout:90];\n"
        f"(way[\"building\"]({miny},{minx},{maxy},{maxx}););\n"
        f"out body;\n>;\nout skel qt;\n"
    )
    data_enc = urllib.parse.urlencode({"data": query}).encode()
    req = urllib.request.Request(
        OSM_OVERPASS_URL,
        data=data_enc,
        headers={
            "User-Agent": "SPARC-DataCollection/1.0",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    with urllib.request.urlopen(req, timeout=120.0) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    # Build node-coordinate lookup then reconstruct polygons from ways
    node_coords: dict[int, tuple[float, float]] = {}
    ways_list: list[dict] = []
    for elem in result.get("elements", []):
        t = elem.get("type")
        if t == "node":
            node_coords[elem["id"]] = (elem["lon"], elem["lat"])
        elif t == "way":
            ways_list.append(elem)

    records: list[dict] = []
    for way in ways_list:
        node_ids = way.get("nodes", [])
        coords = [node_coords[nid] for nid in node_ids if nid in node_coords]
        if len(coords) < 3:
            continue

        tags = way.get("tags", {})
        height: Optional[float] = None
        if "height" in tags:
            try:
                height = float(str(tags["height"]).split()[0])
            except (ValueError, IndexError):
                pass
        if height is None and "building:levels" in tags:
            try:
                height = float(tags["building:levels"]) * 3.0
            except (ValueError, TypeError):
                pass

        try:
            poly = Polygon(coords)
            if poly.is_valid and not poly.is_empty:
                records.append({"geometry": poly, "height": height})
        except Exception:
            continue

    if not records:
        return None

    buildings = gpd.GeoDataFrame(records, crs="EPSG:4326")
    try:
        buildings.to_parquet(cache_path)
    except Exception:
        pass
    return buildings


def _get_oshb_href(
    bbox: tuple[float, float, float, float],
) -> Optional[str]:
    """Return the OSHB raster asset URL for *bbox* from Planetary Computer.

    Returns ``None`` if no OSHB tile covers the area or the PC client is
    unavailable.
    """
    try:
        import pystac_client  # type: ignore[import]
        import planetary_computer  # type: ignore[import]
    except ImportError:
        return None

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    minx, miny, maxx, maxy = bbox
    items = list(catalog.search(
        collections=["oshb"],
        bbox=[minx, miny, maxx, maxy],
        max_items=4,
    ).items())
    if not items:
        return None
    item = items[0]
    return item.assets.get("data", item.assets.get("image", list(item.assets.values())[0])).href


def _from_oshb(
    fishnet_gdf: object,
    bbox: tuple[float, float, float, float],
    cache_dir: Path,
) -> Optional[object]:
    """Fetch OSHB building heights from Planetary Computer STAC."""
    href = _get_oshb_href(bbox)
    if href is None:
        return None
    gdf = _from_raster(fishnet_gdf, href, "bldg_height_mean")  # type: ignore[arg-type]
    return _coverage_from_height(gdf)


# ---------------------------------------------------------------------------
# Building → fishnet join
# ---------------------------------------------------------------------------

def _join_buildings_to_fishnet(
    fishnet_gdf: object,
    buildings_gdf: object,
) -> object:
    """Compute mean height and coverage fraction per fishnet cell."""
    import geopandas as gpd

    fishnet = fishnet_gdf.copy()  # type: ignore[union-attr]
    fish_proj = fishnet.to_crs("EPSG:3857")  # type: ignore[union-attr]
    bld_proj = buildings_gdf.to_crs("EPSG:3857")  # type: ignore[union-attr]

    # Spatial join — each building to the fishnet cell it falls within
    joined = gpd.sjoin(bld_proj, fish_proj, how="left", predicate="intersects")

    height_col = next(
        (c for c in ["height", "Height", "building_height", "bldg_height"] if c in joined.columns),
        None,
    )

    cell_idx = joined.index_right
    n_cells = len(fishnet)

    if height_col:
        # Fill NaN heights with default so cells with footprints but no height
        # tag produce a usable SVF estimate rather than NaN.
        heights_filled = joined[height_col].fillna(_DEFAULT_HEIGHT_M)
        mean_heights = (
            heights_filled.groupby(cell_idx)
            .mean()
            .reindex(range(n_cells), fill_value=0.0)
        )
    else:
        mean_heights = [0.0] * n_cells

    # Coverage = total footprint area / cell area
    cell_area = CELL_WIDTH_M ** 2  # m²
    joined["_fp_area"] = joined.geometry.area
    coverage = (
        joined.groupby(cell_idx)["_fp_area"]
        .sum()
        .reindex(range(n_cells), fill_value=0.0)
        / cell_area
    ).clip(0.0, 1.0)

    fishnet["bldg_height_mean"] = list(mean_heights)  # type: ignore[index]
    fishnet["bldg_coverage"] = list(coverage)         # type: ignore[index]
    return fishnet


# ---------------------------------------------------------------------------
# SVF computation
# ---------------------------------------------------------------------------

def _compute_svf_proxy(fishnet_gdf: object) -> object:
    """H:W ratio SVF proxy:  SVF = W / sqrt(W² + H²)."""
    gdf = fishnet_gdf.copy()  # type: ignore[union-attr]
    H = gdf["bldg_height_mean"].fillna(0.0).to_numpy(dtype=float)  # type: ignore[union-attr]
    W = CELL_WIDTH_M
    with np.errstate(divide="ignore", invalid="ignore"):
        svf = np.where(H > 0.0, W / np.sqrt(W**2 + H**2), 1.0)
    gdf["svf"] = svf.clip(0.0, 1.0)  # type: ignore[index]
    return gdf


def _true_svf_from_dsm(fishnet_gdf: object, dsm_path: Path) -> object:
    """True SVF from DSM using WhiteboxTools if available; falls back to proxy."""
    try:
        from whitebox import WhiteboxTools  # type: ignore[import]
        import rasterio
        import rasterstats
    except ImportError:
        return _compute_svf_proxy(fishnet_gdf)

    wbt = WhiteboxTools()
    wbt.verbose = False
    svf_out = dsm_path.parent / "svf_output.tif"
    wbt.sky_view_factor(
        dem=str(dsm_path),
        output=str(svf_out),
        azimuth_step=10.0,
    )
    if not svf_out.exists():
        return _compute_svf_proxy(fishnet_gdf)

    import rasterstats
    results = rasterstats.zonal_stats(
        fishnet_gdf,  # type: ignore[arg-type]
        str(svf_out),
        stats=["mean"],
        nodata=-9999,
    )
    gdf = fishnet_gdf.copy()  # type: ignore[union-attr]
    gdf["svf"] = [max(0.0, min(1.0, r.get("mean") or 1.0)) for r in results]  # type: ignore[index]
    return gdf


def _default_cache_dir() -> Path:
    p = Path.home() / ".sparc" / "cache" / "buildings"
    p.mkdir(parents=True, exist_ok=True)
    return p
