"""
landsat.py — USGS Landsat C2 scene query, index computation, and LST.

Queries the USGS Landsat Collection 2 Level-2 STAC catalog for scenes
covering the study boundary within a configurable date range and cloud cover
threshold.  Downloads COG tiles for required bands, clips to the analysis
fishnet, and computes all 13 spectral indices plus LST.

Temporal modes
--------------
  "single"     — pick the one scene with the lowest cloud cover in the window
  "composite"  — pixel-wise median across all valid scenes → one row per cell
  "panel"      — each scene becomes a dated row set; returns multi-date panel

Spectral indices computed
-------------------------
  ndvi, ndbi, mndwi, ndwi, savi, evi, ndbai, nbr, ndmi, ui, bsi, albedo, lst
"""

from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from typing import Literal, Optional

import numpy as np

HTTP_TIMEOUT = 120.0
# Primary: AWS Element84 Earth-Search (reliable, no auth required)
ELEMENT84_STAC = "https://earth-search.aws.element84.com/v1/search"
# Fallback: USGS Landsat Look STAC server
USGS_STAC      = "https://landsatlook.usgs.gov/stac-server/search"

# Collection names per endpoint
_ELEMENT84_COLLECTION = "landsat-c2-l2"
_USGS_COLLECTION      = "landsat-c2l2-sr"

TemporalMode = Literal["single", "composite", "panel"]

# Landsat 8/9 OLI band mapping (Collection 2, Level-2 SR/ST)
BAND_ASSETS = {
    "blue":   "SR_B2",
    "green":  "SR_B3",
    "red":    "SR_B4",
    "nir":    "SR_B5",
    "swir1":  "SR_B6",
    "swir2":  "SR_B7",
    "thermal": "ST_B10",  # Surface Temperature (Kelvin × 0.00341802 + 149.0)
}

LST_SCALE  = 0.00341802
LST_OFFSET = 149.0          # K; subtract 273.15 for °C


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_landsat(
    fishnet_gdf: object,
    bbox: tuple[float, float, float, float],
    window: "TemporalWindow",
    *,
    cloud_cover_max: float = 20.0,
    temporal_mode: TemporalMode = "composite",
    enabled_indices: Optional[list[str]] = None,
    cache_dir: Optional[Path] = None,
) -> object:
    """Fetch Landsat data aligned to CAPA anchor dates.

    Parameters
    ----------
    fishnet_gdf : gpd.GeoDataFrame
        30m analysis grid.
    bbox : (minx, miny, maxx, maxy)
        Study bounding box in EPSG:4326.
    window : TemporalWindow
        Temporal search contract.  ``window.anchor_dates`` are the CAPA
        measurement dates; Landsat searches within ``±window.tolerance_days``
        of each anchor.  If ``anchor_dates`` is empty, falls back to searching
        the full ``window.date_start`` → ``window.date_end`` range.
    cloud_cover_max : float
        Maximum cloud cover percentage (0–100).  Default 20%.
    temporal_mode : "single" | "composite" | "panel"
        How multiple scenes are collapsed.
    enabled_indices : list[str] | None
        Subset of index names to compute; None = all 13.
    cache_dir : Path, optional
        Cache directory for COG downloads.

    Returns
    -------
    gpd.GeoDataFrame
        Fishnet (or panel) with spectral columns.  NaN-filled when no scenes
        are available — never raises on empty STAC results.

    Raises
    ------
    TypeError
        If ``window`` is not a ``TemporalWindow`` instance.  The old bare
        ``date_start / date_end`` positional interface has been removed.
    """
    from ._temporal import TemporalWindow as _TemporalWindow
    if not isinstance(window, _TemporalWindow):
        raise TypeError(
            f"fetch_landsat() expects a TemporalWindow as the third argument, "
            f"got {type(window).__name__}. The bare date_start/date_end interface "
            f"has been replaced — use TemporalWindow.from_capa_dates()."
        )

    cache_dir = cache_dir or _default_cache_dir()
    enabled = set(enabled_indices or list(SPECTRAL_INDEX_FORMULAS.keys()) + ["lst"])

    if window.anchor_dates:
        scene_arrays, scene_dates = _fetch_anchored_scenes(
            bbox, window, enabled, cache_dir, cloud_cover_max, temporal_mode
        )
    else:
        # No CAPA anchors: fall back to full date-range search (degraded mode)
        scene_arrays, scene_dates = _fetch_full_range_scenes(
            bbox, window.date_start, window.date_end, enabled, cache_dir,
            cloud_cover_max, temporal_mode
        )

    if not scene_arrays:
        return _nan_fishnet(fishnet_gdf, enabled)

    if temporal_mode == "panel":
        return _build_panel(fishnet_gdf, scene_arrays, scene_dates)

    composite = _median_composite(scene_arrays)
    return _assign_arrays_to_fishnet(fishnet_gdf, composite)


def _nan_fishnet(fishnet_gdf: object, enabled: set) -> object:
    """Return fishnet with NaN columns for all requested indices."""
    try:
        gdf = fishnet_gdf.copy()  # type: ignore[union-attr]
    except Exception:
        gdf = fishnet_gdf
    for idx in list(enabled) + ["lst"]:
        gdf[idx] = float("nan")  # type: ignore[index]
    return gdf


def _fetch_anchored_scenes(
    bbox: tuple,
    window: "TemporalWindow",
    enabled: set,
    cache_dir: Path,
    cloud_max: float,
    temporal_mode: str,
) -> tuple[list, list]:
    """Fetch one best-match Landsat scene per CAPA anchor date."""
    scene_arrays: list[dict] = []
    scene_dates: list[date] = []

    for anchor in window.anchor_dates:
        start, end = window.search_range(anchor)
        try:
            scenes = _query_stac(bbox, start, end, cloud_max)
        except RuntimeError:
            scenes = []

        if not scenes:
            continue

        # Report date offset via TemporalWindow warnings
        available_dates = [_parse_date(s["datetime"]) for s in scenes]
        closest_date, _offset = window.closest_match(available_dates, anchor)

        # Pick the scene with the closest acquisition date to the anchor
        target = min(scenes, key=lambda s: abs((_parse_date(s["datetime"]) - anchor).days))
        bands = _download_bands(target, bbox, enabled, cache_dir)
        if bands is None:
            continue

        scene_arrays.append(_compute_indices(bands, enabled))
        scene_dates.append(closest_date or anchor)

    return scene_arrays, scene_dates


def _fetch_full_range_scenes(
    bbox: tuple,
    date_start: date,
    date_end: date,
    enabled: set,
    cache_dir: Path,
    cloud_max: float,
    temporal_mode: str,
) -> tuple[list, list]:
    """Fallback: fetch all Landsat scenes in the full date range."""
    try:
        scenes = _query_stac(bbox, date_start, date_end, cloud_max)
    except RuntimeError:
        return [], []

    if not scenes:
        return [], []

    if temporal_mode == "single":
        scenes = [min(scenes, key=lambda s: s["cloud_cover"])]

    scene_arrays: list[dict] = []
    scene_dates: list[date] = []
    for scene in scenes:
        acq_date = _parse_date(scene["datetime"])
        bands = _download_bands(scene, bbox, enabled, cache_dir)
        if bands is None:
            continue
        scene_arrays.append(_compute_indices(bands, enabled))
        scene_dates.append(acq_date)

    return scene_arrays, scene_dates


# ---------------------------------------------------------------------------
# STAC query
# ---------------------------------------------------------------------------

def _query_stac(
    bbox: tuple[float, float, float, float],
    date_start: date,
    date_end: date,
    cloud_max: float,
) -> list[dict]:
    """Search for Landsat 8/9 Collection 2 Level-2 scenes.

    Tries the AWS Element84 Earth-Search endpoint first (more reliable), then
    falls back to the USGS Landsat Look STAC server.  Cloud-cover filtering is
    done client-side after the response to avoid relying on optional STAC API
    query extensions that may not be enabled on all servers.
    """
    endpoints = [
        (ELEMENT84_STAC, _ELEMENT84_COLLECTION),
        (USGS_STAC,      _USGS_COLLECTION),
    ]

    last_error: Exception | None = None
    for url, collection in endpoints:
        try:
            scenes = _query_stac_endpoint(url, collection, bbox, date_start, date_end)
            # Post-filter by cloud cover (avoids relying on unsupported query extensions)
            return [s for s in scenes if s["cloud_cover"] <= cloud_max]
        except Exception as exc:
            last_error = exc
            continue

    raise RuntimeError(
        f"STAC query failed on all endpoints: {last_error}"
    ) from last_error


def _query_stac_endpoint(
    url: str,
    collection: str,
    bbox: tuple[float, float, float, float],
    date_start: date,
    date_end: date,
) -> list[dict]:
    """Send a minimal STAC search (no query/fields extensions) to one endpoint."""
    minx, miny, maxx, maxy = bbox
    payload = {
        "collections": [collection],
        "bbox": [minx, miny, maxx, maxy],
        "datetime": f"{date_start.isoformat()}T00:00:00Z/{date_end.isoformat()}T23:59:59Z",
        "limit": 200,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "SPARC-DataCollection/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"STAC query failed ({url}): {exc}") from exc

    items = data.get("features", [])
    scenes = []
    for item in items:
        props = item.get("properties", {})
        scenes.append({
            "id": item["id"],
            "datetime": props.get("datetime", ""),
            "cloud_cover": props.get("eo:cloud_cover", 100),
            "assets": item.get("assets", {}),
        })
    return scenes


# ---------------------------------------------------------------------------
# Band download
# ---------------------------------------------------------------------------

def _download_bands(
    scene: dict,
    bbox: tuple[float, float, float, float],
    enabled: set[str],
    cache_dir: Path,
) -> Optional[dict[str, np.ndarray]]:
    """Download required COG bands for one scene, clipped to bbox."""
    try:
        import rasterio
        from rasterio.windows import from_bounds
        from rasterio.warp import transform_bounds
    except ImportError:
        raise ImportError("rasterio is required for Landsat band download: pip install rasterio")

    bands: dict[str, np.ndarray] = {}
    assets = scene["assets"]
    minx, miny, maxx, maxy = bbox

    for band_name, asset_key in BAND_ASSETS.items():
        # Only download bands actually needed for enabled indices
        if not _band_needed(band_name, enabled):
            continue
        asset = assets.get(asset_key) or assets.get(asset_key.lower())
        if asset is None:
            continue
        href = asset.get("href", asset.get("url", ""))
        if not href:
            continue

        cache_key = f"{scene['id']}_{asset_key}"
        cache_path = cache_dir / f"{cache_key}.npy"

        if cache_path.exists():
            bands[band_name] = np.load(str(cache_path))
            continue

        try:
            with rasterio.open(href) as src:
                window = from_bounds(minx, miny, maxx, maxy, src.transform)
                arr = src.read(1, window=window).astype(np.float32)
                arr[arr <= 0] = np.nan  # mask fill values
                np.save(str(cache_path), arr)
                bands[band_name] = arr
        except Exception:
            continue

    # Scale thermal to LST in °C
    if "thermal" in bands:
        bands["lst_kelvin"] = bands.pop("thermal")
        bands["lst"] = (bands["lst_kelvin"] * LST_SCALE + LST_OFFSET) - 273.15

    return bands if bands else None


# ---------------------------------------------------------------------------
# Spectral index formulas
# ---------------------------------------------------------------------------

def _safe_norm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a - b) / (a + b) with NaN for zero denominator."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where((a + b) != 0, (a - b) / (a + b), np.nan)
    return result.astype(np.float32)


SPECTRAL_INDEX_FORMULAS: dict = {
    "ndvi":  lambda b: _safe_norm(b["nir"],   b["red"]),
    "ndbi":  lambda b: _safe_norm(b["swir1"], b["nir"]),
    "mndwi": lambda b: _safe_norm(b["green"], b["swir1"]),
    "ndwi":  lambda b: _safe_norm(b["green"], b["nir"]),
    "savi":  lambda b: ((b["nir"] - b["red"]) / (b["nir"] + b["red"] + 0.5)) * 1.5,
    "evi":   lambda b: 2.5 * (b["nir"] - b["red"]) / (b["nir"] + 6*b["red"] - 7.5*b["blue"] + 1),
    "ndbai": lambda b: _safe_norm(b["swir1"], b["swir2"]),
    "nbr":   lambda b: _safe_norm(b["nir"],   b["swir2"]),
    "ndmi":  lambda b: _safe_norm(b["nir"],   b["swir1"]),
    "ui":    lambda b: _safe_norm(b["swir2"], b["nir"]),
    "bsi":   lambda b: _safe_norm(b["red"] + b["swir1"], b["nir"] + b["blue"]),
    "albedo": lambda b: (
        0.356*b["blue"] + 0.130*b["red"] + 0.373*b["nir"] +
        0.085*b["swir1"] + 0.072*b["swir2"] - 0.0018
    ),
}


def _band_needed(band_name: str, enabled: set[str]) -> bool:
    """Return True if *band_name* is used by any enabled index."""
    usage = {
        "blue":    {"bsi", "evi", "albedo"},
        "green":   {"mndwi", "ndwi"},
        "red":     {"ndvi", "savi", "evi", "bsi", "albedo"},
        "nir":     {"ndvi", "ndbi", "mndwi", "ndwi", "savi", "evi", "nbr", "ndmi", "ui", "bsi", "albedo"},
        "swir1":   {"ndbi", "mndwi", "ndbai", "ndmi", "bsi", "albedo"},
        "swir2":   {"ndbai", "nbr", "ui", "albedo"},
        "thermal": {"lst"},
    }
    return bool(enabled & usage.get(band_name, set()))


def _compute_indices(
    bands: dict[str, np.ndarray],
    enabled: set[str],
) -> dict[str, np.ndarray]:
    indices: dict[str, np.ndarray] = {}
    for name, formula in SPECTRAL_INDEX_FORMULAS.items():
        if name not in enabled:
            continue
        try:
            indices[name] = formula(bands).astype(np.float32)
        except KeyError:
            pass  # required band not available for this scene
    if "lst" in enabled and "lst" in bands:
        indices["lst"] = bands["lst"]
    return indices


# ---------------------------------------------------------------------------
# Temporal aggregation
# ---------------------------------------------------------------------------

def _median_composite(scene_arrays: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Pixel-wise median across scenes for each index."""
    composite: dict[str, np.ndarray] = {}
    all_keys = {k for s in scene_arrays for k in s}
    for key in all_keys:
        stack = np.array([s[key] for s in scene_arrays if key in s], dtype=np.float32)
        composite[key] = np.nanmedian(stack, axis=0)
    return composite


def _assign_arrays_to_fishnet(
    fishnet_gdf: object,
    index_arrays: dict[str, np.ndarray],
) -> object:
    """Zonal-mean each index array onto fishnet cells."""
    import rasterstats
    import rasterio
    from rasterio.transform import from_bounds

    gdf = fishnet_gdf.copy()  # type: ignore[union-attr]
    bounds = gdf.to_crs("EPSG:4326").total_bounds  # type: ignore[union-attr]

    for name, arr in index_arrays.items():
        if arr is None or arr.size == 0:
            gdf[name] = float("nan")  # type: ignore[index]
            continue
        rows, cols = arr.shape
        transform = from_bounds(*bounds, cols, rows)
        results = rasterstats.zonal_stats(
            gdf,  # type: ignore[arg-type]
            arr,
            affine=transform,
            stats=["mean"],
            nodata=np.nan,
        )
        gdf[name] = [r.get("mean") for r in results]  # type: ignore[index]
    return gdf


def _build_panel(
    fishnet_gdf: object,
    scene_arrays: list[dict[str, np.ndarray]],
    scene_dates: list[date],
) -> object:
    """Build a multi-date panel GeoDataFrame (one copy per scene date)."""
    import pandas as pd
    import geopandas as gpd

    frames = []
    for arr_dict, acq_date in zip(scene_arrays, scene_dates):
        gdf = _assign_arrays_to_fishnet(fishnet_gdf, arr_dict)
        gdf["scene_date"] = acq_date.isoformat()  # type: ignore[index]
        frames.append(gdf)

    if not frames:
        return fishnet_gdf

    return gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True),
        crs=fishnet_gdf.crs,  # type: ignore[union-attr]
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _parse_date(dt_str: str) -> date:
    return date.fromisoformat(dt_str[:10])


def _default_cache_dir() -> Path:
    p = Path.home() / ".sparc" / "cache" / "landsat"
    p.mkdir(parents=True, exist_ok=True)
    return p
