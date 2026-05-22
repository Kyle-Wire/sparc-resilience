"""
nlcd.py — NLCD variable fetch via MRLC WCS.

Fetches three NLCD layers from the Multi-Resolution Land Characteristics
Consortium (MRLC) Web Coverage Service, clips each to the study boundary,
resamples to the analysis fishnet, and returns a GeoDataFrame with columns:
  pct_impervious  — impervious surface fraction 0–100
  pct_canopy      — tree canopy cover 0–100
  land_cover      — numeric land cover class (NLCD code)

Land cover classes are kept as integers; waterbodies are NOT masked.
No API key is required — the MRLC WCS is publicly accessible.
"""

from __future__ import annotations

import io
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# MRLC WCS constants
# ---------------------------------------------------------------------------

MRLC_WCS_BASE = "https://www.mrlc.gov/geoserver/mrlc_display/wcs"
HTTP_TIMEOUT = 60.0

# MRLC layer identifiers (NLCD 2021)
_LAYER_IMPERVIOUS = "NLCD_2021_Impervious_L48_20230630"
_LAYER_CANOPY     = "NLCD_2021_TreeCanopy_L48_20221101"
_LAYER_LANDCOVER  = "NLCD_2021_Land_Cover_L48_20230630"

# NLCD string → numeric class mapping (NLCD Anderson Level I/II codes)
NLCD_CLASS_MAP: dict[str, int] = {
    "Open Water": 11,
    "Perennial Ice/Snow": 12,
    "Developed, Open Space": 21,
    "Developed, Low Intensity": 22,
    "Developed, Medium Intensity": 23,
    "Developed High Intensity": 24,
    "Barren Land": 31,
    "Deciduous Forest": 41,
    "Evergreen Forest": 42,
    "Mixed Forest": 43,
    "Dwarf Scrub": 51,
    "Shrub/Scrub": 52,
    "Grassland/Herbaceous": 71,
    "Sedge/Herbaceous": 72,
    "Lichens": 73,
    "Moss": 74,
    "Pasture/Hay": 81,
    "Cultivated Crops": 82,
    "Woody Wetlands": 90,
    "Emergent Herbaceous Wetlands": 95,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_nlcd(
    fishnet_gdf: object,
    bbox: tuple[float, float, float, float],
    *,
    cache_dir: Optional[Path] = None,
) -> object:
    """Fetch NLCD impervious, canopy, and land cover onto the analysis fishnet.

    Parameters
    ----------
    fishnet_gdf : gpd.GeoDataFrame
        30m analysis grid in its projected CRS.  Returned with three new columns.
    bbox : (minx, miny, maxx, maxy)
        Bounding box in EPSG:4326 used for the WCS request.
    cache_dir : Path, optional
        Directory to cache raw WCS GeoTIFF downloads.  Defaults to
        ``~/.sparc/cache/nlcd/``.

    Returns
    -------
    gpd.GeoDataFrame
        Input fishnet with ``pct_impervious``, ``pct_canopy``, and
        ``land_cover`` columns appended.
    """
    try:
        import geopandas as gpd
        import rasterio
        from rasterio.mask import mask as rio_mask
        from rasterio.warp import reproject, Resampling
        import rasterstats
    except ImportError as exc:
        raise ImportError(
            "rasterio and rasterstats are required for NLCD fetch: pip install rasterio rasterstats"
        ) from exc

    cache_dir = cache_dir or _default_cache_dir()

    layers = [
        (_LAYER_IMPERVIOUS, "pct_impervious", "mean"),
        (_LAYER_CANOPY,     "pct_canopy",     "mean"),
        (_LAYER_LANDCOVER,  "land_cover",     "majority"),
    ]

    gdf = fishnet_gdf  # type: ignore[assignment]
    for layer_id, col_name, stat in layers:
        tif_path = _fetch_wcs_tile(layer_id, bbox, cache_dir)
        gdf = _zonal_stat_onto_fishnet(gdf, tif_path, col_name, stat)

    # Convert land cover majority to int; fill NaN with -1 (no-data sentinel)
    if "land_cover" in gdf.columns:
        gdf["land_cover"] = gdf["land_cover"].fillna(-1).astype(int)

    return gdf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_wcs_tile(
    layer_id: str,
    bbox: tuple[float, float, float, float],
    cache_dir: Path,
) -> Path:
    """Download a WCS GeoTIFF for *layer_id* clipped to *bbox*."""
    minx, miny, maxx, maxy = bbox
    cache_key = f"{layer_id}_{minx:.4f}_{miny:.4f}_{maxx:.4f}_{maxy:.4f}"
    out_path = cache_dir / f"{cache_key}.tif"
    if out_path.exists():
        return out_path

    params = urllib.parse.urlencode({
        "service": "WCS",
        "version": "2.0.1",
        "request": "GetCoverage",
        "coverageId": layer_id,
        "subsettingCrs": "http://www.opengis.net/def/crs/EPSG/0/4326",
        "subset": [
            f"Long({minx},{maxx})",
            f"Lat({miny},{maxy})",
        ],
        "format": "image/tiff",
    }, doseq=True)
    url = f"{MRLC_WCS_BASE}?{params}"

    req = urllib.request.Request(url, headers={"User-Agent": "SPARC-DataCollection/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            data = resp.read()
    except Exception as exc:
        raise RuntimeError(f"MRLC WCS request failed for layer {layer_id}: {exc}") from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    return out_path


def _zonal_stat_onto_fishnet(
    gdf: object,
    raster_path: Path,
    col_name: str,
    stat: str,
) -> object:
    """Compute zonal statistic *stat* for each fishnet cell from *raster_path*."""
    import rasterstats

    results = rasterstats.zonal_stats(
        gdf,  # type: ignore[arg-type]
        str(raster_path),
        stats=[stat],
        nodata=None,
        all_touched=False,
    )
    values = [r.get(stat) for r in results]
    gdf = gdf.copy()  # type: ignore[union-attr]
    gdf[col_name] = values  # type: ignore[index]
    return gdf


def _default_cache_dir() -> Path:
    p = Path.home() / ".sparc" / "cache" / "nlcd"
    p.mkdir(parents=True, exist_ok=True)
    return p
