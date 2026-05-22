"""
sentinel2.py — Sentinel-2 L2A scene query via Microsoft Planetary Computer STAC.

Queries the Planetary Computer STAC for Sentinel-2 L2A scenes covering the
study boundary within a configurable date range and cloud cover threshold.
Downloads COG tiles for required bands, clips to the analysis fishnet, and
computes key spectral indices.

Temporal modes
--------------
  "single"     — pick the one scene with the lowest cloud cover in the window
  "composite"  — pixel-wise median across all valid scenes → one row per cell

Spectral indices computed
-------------------------
  ndvi_s2      — (B8 − B4) / (B8 + B4)  vegetation
  ndbi_s2      — (B11 − B8) / (B11 + B8)  built-up
  mndwi_s2     — (B3 − B11) / (B3 + B11)  water

Note: Sentinel-2 has no thermal infrared band, so LST cannot be computed
directly. Use ERA5 air temperature as the thermal baseline instead.
"""

from __future__ import annotations

import json
import math
import urllib.request
from datetime import date
from typing import Literal

import numpy as np

HTTP_TIMEOUT = 120.0

# Microsoft Planetary Computer STAC — no auth required for browse queries.
PC_STAC_SEARCH = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
PC_COLLECTION = "sentinel-2-l2a"

TemporalMode = Literal["single", "composite"]

# Sentinel-2 L2A band assets on Planetary Computer
BAND_ASSETS = {
    "blue":  "B02",   # 10m
    "green": "B03",   # 10m
    "red":   "B04",   # 10m
    "nir":   "B08",   # 10m
    "swir1": "B11",   # 20m → we resample to 10m
}

S2_SCALE = 0.0001  # DN → reflectance (divide by 10000)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_sentinel2(
    fishnet,
    bbox: tuple[float, float, float, float],
    date_start: date,
    date_end: date,
    *,
    cloud_cover_max: float = 20.0,
    temporal_mode: TemporalMode = "composite",
):
    """Fetch Sentinel-2 spectral indices for each fishnet cell.

    Parameters
    ----------
    fishnet : GeoDataFrame (EPSG:3857)
    bbox : (min_lon, min_lat, max_lon, max_lat) in WGS-84
    date_start / date_end : datetime.date
    cloud_cover_max : 0–100 percent
    temporal_mode : "single" | "composite"

    Returns
    -------
    GeoDataFrame with new columns: ndvi_s2, ndbi_s2, mndwi_s2
    """
    scenes = _query_stac(bbox, date_start, date_end, cloud_cover_max)
    if not scenes:
        fishnet = fishnet.copy()
        for col in ("ndvi_s2", "ndbi_s2", "mndwi_s2"):
            fishnet[col] = float("nan")
        return fishnet

    if temporal_mode == "single":
        scenes = scenes[:1]

    # Attempt to fetch band data via COG tiles; fall back to NaN on failure.
    all_indices: list[dict] = []
    for scene in scenes:
        try:
            indices = _fetch_scene_indices(scene, fishnet, bbox)
            all_indices.append(indices)
        except Exception:
            continue

    fishnet = fishnet.copy()
    if all_indices:
        for col in ("ndvi_s2", "ndbi_s2", "mndwi_s2"):
            stack = np.array([d[col] for d in all_indices if col in d], dtype=float)
            if stack.ndim == 2 and stack.shape[0] > 0:
                fishnet[col] = np.nanmedian(stack, axis=0)
            else:
                fishnet[col] = float("nan")
    else:
        for col in ("ndvi_s2", "ndbi_s2", "mndwi_s2"):
            fishnet[col] = float("nan")

    return fishnet


# ---------------------------------------------------------------------------
# STAC query
# ---------------------------------------------------------------------------

def _query_stac(
    bbox: tuple[float, float, float, float],
    date_start: date,
    date_end: date,
    cloud_cover_max: float,
) -> list[dict]:
    """Return list of STAC items sorted by cloud cover (ascending)."""
    payload = json.dumps({
        "collections": [PC_COLLECTION],
        "bbox": list(bbox),
        "datetime": f"{date_start.isoformat()}T00:00:00Z/{date_end.isoformat()}T23:59:59Z",
        "query": {"eo:cloud_cover": {"lte": cloud_cover_max}},
        "limit": 20,
        "sortby": [{"field": "eo:cloud_cover", "direction": "asc"}],
    }).encode()

    req = urllib.request.Request(
        PC_STAC_SEARCH,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            data = json.load(resp)
        return data.get("features", [])
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Band fetching + index computation
# ---------------------------------------------------------------------------

def _fetch_scene_indices(scene: dict, fishnet, bbox: tuple) -> dict:
    """Download COG tiles for a single scene and compute spectral indices."""
    assets = scene.get("assets", {})

    bands: dict[str, np.ndarray] = {}
    for name, asset_key in BAND_ASSETS.items():
        url = _get_asset_href(assets, asset_key)
        if url is None:
            continue
        try:
            arr = _read_cog_window(url, bbox, fishnet)
            bands[name] = arr * S2_SCALE
        except Exception:
            pass

    n = len(fishnet)
    out: dict[str, np.ndarray] = {}

    # NDVI = (NIR - Red) / (NIR + Red)
    if "nir" in bands and "red" in bands:
        num = bands["nir"] - bands["red"]
        den = bands["nir"] + bands["red"]
        out["ndvi_s2"] = np.where(den != 0, num / den, np.nan)
    else:
        out["ndvi_s2"] = np.full(n, np.nan)

    # NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)
    if "swir1" in bands and "nir" in bands:
        num = bands["swir1"] - bands["nir"]
        den = bands["swir1"] + bands["nir"]
        out["ndbi_s2"] = np.where(den != 0, num / den, np.nan)
    else:
        out["ndbi_s2"] = np.full(n, np.nan)

    # MNDWI = (Green - SWIR1) / (Green + SWIR1)
    if "green" in bands and "swir1" in bands:
        num = bands["green"] - bands["swir1"]
        den = bands["green"] + bands["swir1"]
        out["mndwi_s2"] = np.where(den != 0, num / den, np.nan)
    else:
        out["mndwi_s2"] = np.full(n, np.nan)

    return out


def _get_asset_href(assets: dict, key: str) -> str | None:
    asset = assets.get(key)
    if asset is None:
        return None
    return asset.get("href") or asset.get("alternate", {}).get("s3", {}).get("href")


def _read_cog_window(url: str, bbox: tuple, fishnet) -> np.ndarray:
    """Read a pixel window from a COG via rasterio if available.

    Falls back to per-centroid point sampling via WMS or raises on failure.
    Each fishnet row gets one scalar value (the pixel at its centroid).
    """
    try:
        import rasterio
        from rasterio.windows import from_bounds
        from rasterio.enums import Resampling

        with rasterio.open(url) as src:
            # Reproject bbox to src CRS
            from rasterio.crs import CRS
            from rasterio.warp import transform_bounds

            if src.crs and src.crs.to_epsg() not in (4326, None):
                bounds_src = transform_bounds("EPSG:4326", src.crs, *bbox)
            else:
                bounds_src = bbox

            win = from_bounds(*bounds_src, transform=src.transform)
            data = src.read(1, window=win, out_dtype="float32", resampling=Resampling.bilinear)

            # Map each fishnet centroid to pixel value
            centroids_wgs84 = fishnet.to_crs("EPSG:4326").geometry.centroid
            xs = centroids_wgs84.x.values
            ys = centroids_wgs84.y.values

            if src.crs and src.crs.to_epsg() not in (4326, None):
                from rasterio.warp import transform as warp_transform
                xs_src, ys_src = warp_transform("EPSG:4326", src.crs, xs, ys)
            else:
                xs_src, ys_src = xs, ys

            # Convert to pixel row/col within the window
            t = src.window_transform(win)
            cols = ((xs_src - t.c) / t.a).astype(int)
            rows = ((ys_src - t.f) / t.e).astype(int)

            rows = np.clip(rows, 0, data.shape[0] - 1)
            cols = np.clip(cols, 0, data.shape[1] - 1)

            values = data[rows, cols].astype(float)
            values[values <= 0] = np.nan
            return values

    except ImportError:
        raise RuntimeError("rasterio is required for Sentinel-2 COG reading")
