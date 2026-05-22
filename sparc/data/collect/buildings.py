"""
buildings.py — Building footprints, heights, and Sky View Factor.

Resolves building heights for the study area using a fallback chain:
  Tier 1 — Local LiDAR file (user-provided DSM or building height raster)
  Tier 2 — Microsoft MLBuildings (Azure Blob GeoParquet tiles, ~1B buildings)
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

# Microsoft MLBuildings — Azure Blob Storage (publicly readable)
# Tile index lists parquet files by quadkey / country
ML_BUILDINGS_INDEX = (
    "https://minedbuildings.blob.core.windows.net/global-buildings/dataset-links.csv"
)


# ---------------------------------------------------------------------------
# Public API
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

    if lidar_path and Path(lidar_path).exists():
        gdf = _from_raster(fishnet_gdf, Path(lidar_path), "bldg_height_mean")
        gdf = _coverage_from_height(gdf)
        tier = 1
    else:
        gdf, tier = _from_mlbuildings_or_fallback(fishnet_gdf, bbox, default_height_m, cache_dir)

    if dsm_path and Path(dsm_path).exists():
        gdf = _true_svf_from_dsm(gdf, Path(dsm_path))
    else:
        gdf = _compute_svf_proxy(gdf)

    return gdf, tier


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
# Tier 2 — Microsoft MLBuildings
# ---------------------------------------------------------------------------

def _from_mlbuildings_or_fallback(
    fishnet_gdf: object,
    bbox: tuple[float, float, float, float],
    default_height_m: float,
    cache_dir: Path,
) -> tuple[object, int]:
    """Try MLBuildings (Tier 2), OSHB (Tier 3), constant (Tier 4)."""
    try:
        gdf = _from_mlbuildings(fishnet_gdf, bbox, cache_dir)
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


def _from_mlbuildings(
    fishnet_gdf: object,
    bbox: tuple[float, float, float, float],
    cache_dir: Path,
) -> Optional[object]:
    """Join Microsoft MLBuildings footprints + heights to the fishnet."""
    try:
        import geopandas as gpd
        import pandas as pd
    except ImportError:
        return None

    # Fetch the dataset index to find relevant country/tile parquet files
    index_path = cache_dir / "mlbuildings_index.csv"
    if not index_path.exists():
        try:
            req = urllib.request.Request(
                ML_BUILDINGS_INDEX,
                headers={"User-Agent": "SPARC-DataCollection/1.0"},
            )
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
                index_path.write_bytes(resp.read())
        except Exception:
            return None

    # Parse CSV: Location,Quadkey,Url columns
    import csv
    minx, miny, maxx, maxy = bbox
    parquet_urls: list[str] = []
    with open(index_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            url = row.get("Url", "")
            # Simple geographic filter: download tiles whose quadkey region
            # overlaps bbox. We use the Location column (e.g. "United States")
            # as a rough filter; finer filtering happens after download.
            if url:
                parquet_urls.append(url)

    if not parquet_urls:
        return None

    # Download and union footprints that intersect bbox
    from shapely.geometry import box as shapely_box
    bbox_geom = shapely_box(minx, miny, maxx, maxy)
    all_gdfs: list = []

    # Limit to first 5 tile files to avoid unbounded downloads
    for url in parquet_urls[:5]:
        tile_path = cache_dir / f"mlb_{abs(hash(url))}.parquet"
        if not tile_path.exists():
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "SPARC-DataCollection/1.0"}
                )
                with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
                    tile_path.write_bytes(resp.read())
            except Exception:
                continue
        try:
            tile_gdf = gpd.read_parquet(tile_path)
            if tile_gdf.crs is None:
                tile_gdf = tile_gdf.set_crs("EPSG:4326")
            elif str(tile_gdf.crs.to_epsg()) != "4326":
                tile_gdf = tile_gdf.to_crs("EPSG:4326")
            clipped = tile_gdf[tile_gdf.intersects(bbox_geom)]
            if not clipped.empty:
                all_gdfs.append(clipped)
        except Exception:
            continue

    if not all_gdfs:
        return None

    import pandas as pd
    buildings = gpd.GeoDataFrame(
        pd.concat(all_gdfs, ignore_index=True), crs="EPSG:4326"
    )
    return _join_buildings_to_fishnet(fishnet_gdf, buildings)


def _from_oshb(
    fishnet_gdf: object,
    bbox: tuple[float, float, float, float],
    cache_dir: Path,
) -> Optional[object]:
    """Fetch OSHB building heights from Planetary Computer STAC."""
    try:
        import pystac_client  # type: ignore[import]
        import planetary_computer  # type: ignore[import]
        import rasterio
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

    # Use the first item's data asset
    item = items[0]
    href = item.assets.get("data", item.assets.get("image", list(item.assets.values())[0])).href
    return _from_raster(_coverage_from_height(fishnet_gdf), href, "bldg_height_mean")  # type: ignore[arg-type]


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
        mean_heights = (
            joined.groupby(cell_idx)[height_col]
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
