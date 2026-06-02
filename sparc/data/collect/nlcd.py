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
HTTP_TIMEOUT = 300.0  # 5 min — the server may return a full CONUS tile (~50–100 MB)

# MRLC layer identifiers (NLCD 2021, verified from WCS GetCapabilities 2025)
# Full IDs include the workspace prefix "mrlc_display__"
_LAYER_IMPERVIOUS = "mrlc_display__NLCD_2021_Impervious_L48"
_LAYER_CANOPY     = "mrlc_display__nlcd_tcc_conus_2021_v2021-4"
_LAYER_LANDCOVER  = "mrlc_display__NLCD_2021_Land_Cover_L48"

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
# Public API — two-phase (download then assign to grid)
# ---------------------------------------------------------------------------

def download_nlcd(
    bbox: tuple[float, float, float, float],
    *,
    cache_dir: Optional[Path] = None,
) -> dict[str, Path]:
    """Download NLCD tiles for *bbox* and return cached GeoTIFF paths.

    This is **Phase 1** of the two-phase collection pattern: raw rasters are
    fetched and clipped to the bounding box without requiring a fishnet.
    Call :func:`assign_nlcd_to_grid` afterwards to compute zonal statistics
    once the fishnet has been created at the desired resolution.

    Returns
    -------
    dict[str, Path]
        Mapping of ``column_name → clipped GeoTIFF path``.
        Keys: ``"pct_impervious"``, ``"pct_canopy"``, ``"land_cover"``.
    """
    cache_dir = cache_dir or _default_cache_dir()
    layers = [
        (_LAYER_IMPERVIOUS, "pct_impervious"),
        (_LAYER_CANOPY,     "pct_canopy"),
        (_LAYER_LANDCOVER,  "land_cover"),
    ]
    return {col: _fetch_wcs_tile(lid, bbox, cache_dir) for lid, col in layers}


def assign_nlcd_to_grid(
    fishnet_gdf: object,
    tif_paths: dict[str, Path],
) -> object:
    """Apply NLCD zonal statistics to fishnet cells.

    This is **Phase 2** of the two-phase collection pattern.

    Parameters
    ----------
    fishnet_gdf : gpd.GeoDataFrame
        Analysis grid at any resolution.
    tif_paths : dict[str, Path]
        Mapping from ``column_name → GeoTIFF path`` as returned by
        :func:`download_nlcd`.

    Returns
    -------
    gpd.GeoDataFrame
        Fishnet with ``pct_impervious``, ``pct_canopy``, and ``land_cover``
        columns appended.
    """
    col_stat_map = {
        "pct_impervious": "mean",
        "pct_canopy":     "mean",
        "land_cover":     "majority",
    }
    gdf = fishnet_gdf
    for col_name, stat in col_stat_map.items():
        if col_name in tif_paths:
            gdf = _zonal_stat_onto_fishnet(gdf, tif_paths[col_name], col_name, stat)
    if "land_cover" in gdf.columns:
        gdf["land_cover"] = gdf["land_cover"].fillna(-1).astype(int)  # type: ignore[union-attr]
    return gdf


# ---------------------------------------------------------------------------
# Legacy single-call API (kept for backward compatibility)
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

    tif_paths = download_nlcd(bbox, cache_dir=cache_dir)
    return assign_nlcd_to_grid(fishnet_gdf, tif_paths)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_wcs_tile(
    layer_id: str,
    bbox: tuple[float, float, float, float],
    cache_dir: Path,
) -> Path:
    """Download a WCS GeoTIFF for *layer_id* and clip it to *bbox*.

    The MRLC GeoServer WCS sometimes ignores the ``subset`` parameters and
    returns the full CONUS raster (~50–100 MB compressed, ~1.5 TiB
    uncompressed).  This function handles both cases:

    1. Streams the raw WCS response to a temp file in chunks so that it
       never needs to fit in RAM all at once.
    2. Opens the saved file with rasterio and reads **only the bbox window**,
       producing a small clipped GeoTIFF.  The raw download is then deleted.

    The clipped file is cached under a ``_clipped`` suffix so that subsequent
    calls (e.g. re-running a project) skip the download entirely.
    """
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.warp import transform_bounds

    minx, miny, maxx, maxy = bbox
    cache_key = f"{layer_id}_{minx:.4f}_{miny:.4f}_{maxx:.4f}_{maxy:.4f}"
    # Cached *clipped* tile — fast path
    clipped_path = cache_dir / f"{cache_key}_clipped.tif"
    if clipped_path.exists():
        return clipped_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_path = cache_dir / f"{cache_key}_raw.tif"

    # ---------- 1. Stream-download the WCS response to disk ----------------
    if not raw_path.exists():
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
                with open(raw_path, "wb") as fh:
                    while True:
                        chunk = resp.read(1 << 17)  # 128 KB chunks
                        if not chunk:
                            break
                        fh.write(chunk)
        except Exception as exc:
            raw_path.unlink(missing_ok=True)
            raise RuntimeError(f"MRLC WCS request failed for layer {layer_id}: {exc}") from exc

    # ---------- 2. Clip to bbox window using rasterio ----------------------
    try:
        with rasterio.open(raw_path) as src:
            # Transform the WGS-84 bbox to the raster's native CRS
            bounds_src = transform_bounds("EPSG:4326", src.crs, minx, miny, maxx, maxy)
            window = from_bounds(*bounds_src, src.transform)
            clipped_arr = src.read(1, window=window)
            clipped_transform = src.window_transform(window)

            with rasterio.open(
                clipped_path, "w",
                driver="GTiff",
                height=clipped_arr.shape[0],
                width=clipped_arr.shape[1],
                count=1,
                dtype=clipped_arr.dtype,
                crs=src.crs,
                transform=clipped_transform,
                compress="lzw",
            ) as dst:
                dst.write(clipped_arr, 1)
    except Exception as exc:
        raw_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to clip WCS tile for {layer_id}: {exc}") from exc

    # Remove the raw (possibly full-CONUS) file to save disk space
    raw_path.unlink(missing_ok=True)
    return clipped_path


def _zonal_stat_onto_fishnet(
    gdf: object,
    raster_path: Path,
    col_name: str,
    stat: str,
) -> object:
    """Sample raster values onto fishnet cells via centroid point lookup.

    Since the fishnet is at 30m resolution matching the NLCD rasters, each
    cell maps to exactly one pixel. Centroid-based sampling via rasterio is
    orders of magnitude faster than polygon zonal stats for large grids
    (O(N) array indexing vs. O(N * polygon_size) intersection).
    """
    import rasterio
    from rasterio.transform import rowcol
    import numpy as np

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        transform  = src.transform
        arr        = src.read(1)          # read full (already-clipped) tile
        nodata     = src.nodata

    gdf_proj = gdf.to_crs(raster_crs)    # type: ignore[union-attr]

    # Compute centroids in the original UTM projection (accurate), then reproject
    # to the raster CRS for the rowcol lookup. Avoids the geographic-CRS centroid warning.
    centroids_utm = gdf.geometry.centroid  # type: ignore[union-attr]
    centroids_gdf = centroids_utm.to_frame("geometry").set_crs(gdf.crs).to_crs(raster_crs)  # type: ignore[union-attr]
    xs = centroids_gdf.geometry.x.values
    ys = centroids_gdf.geometry.y.values

    rows, cols = rowcol(transform, xs, ys)
    rows = np.clip(np.array(rows, dtype=int), 0, arr.shape[0] - 1)
    cols = np.clip(np.array(cols, dtype=int), 0, arr.shape[1] - 1)

    values = arr[rows, cols].astype(float)
    if nodata is not None:
        values[values == nodata] = np.nan

    gdf = gdf.copy()  # type: ignore[union-attr]
    gdf[col_name] = values  # type: ignore[index]
    return gdf


def _default_cache_dir() -> Path:
    p = Path.home() / ".sparc" / "cache" / "nlcd"
    p.mkdir(parents=True, exist_ok=True)
    return p
