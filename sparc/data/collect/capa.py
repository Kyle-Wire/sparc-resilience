"""
capa.py — CAPA Heat Watch traverse data for the SPARC pipeline.

Fetches NOAA/NIHHIS Heat Watch campaign traverse shapefiles from an OSF
repository and spatially interpolates them to the 30m fishnet at three
time-of-day windows:

  am (morning)   → aat_morning  (~06:00 local)
  af (afternoon) → aat_midday   (~12:00–14:00 local)
  pm (evening)   → aat_night    (~19:00–21:00 local)

The campaign date is parsed from the traverses ZIP filename:
  ``traverses_chw_{city}_{MMDDYY}.zip``

Data source
-----------
NOAA/NIHHIS Urban Heat Island Mapping Campaigns — hosted on OSF.io
  Campaigns: https://heat.gov/urban-heat-islands-mapping-campaign-program/
  OSF API:   https://api.osf.io/v2/nodes/{node_id}/files/osfstorage/
  Download:  https://osf.io/download/{file_guid}/
  Free and public — no API key required.

Configuration
-------------
In ``project.yml`` add::

    collect:
      capa_osf_node: "3xts6"   # OSF node ID for your campaign

The node ID is the 5-character code in the campaign URL:
  https://osf.io/{node_id}/files/

Returns NaN columns and an empty date list on any failure — never raises.
"""

from __future__ import annotations

import io
import json
import logging
import re
import tempfile
import urllib.request
import zipfile
from datetime import date
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from ._temporal import TemporalWindow

log = logging.getLogger(__name__)

HTTP_TIMEOUT = 60.0

OSF_API        = "https://api.osf.io/v2/nodes/{node_id}/files/osfstorage/"
# Waterbutler direct-download URL — faster and more reliable than osf.io/download/
# Retrieved from item["links"]["move"] in the API response (same base URL, GET = download)
OSF_WATERBUTLER = "https://files.osf.io/v1/resources/{node_id}/providers/osfstorage/{file_id}"

# Match _MMDDYY followed by any non-digit (covers both ZIP names and report filenames)
_DATE_RE = re.compile(r"_(\d{2})(\d{2})(\d{2})(?=[^\d]|$)", re.IGNORECASE)

# Raster prefix → output column
_WINDOWS: dict[str, str] = {
    "am": "aat_morning",
    "af": "aat_midday",
    "pm": "aat_night",
}

# TIF filename start patterns (lowercase) for old-format campaigns (individual files)
_WINDOW_TIF_PATTERNS: dict[str, list[str]] = {
    "am": ["am_t_f", "morning_area-wide_temperature", "morning_area_wide_temperature"],
    "af": ["af_t_f", "afternoon_area-wide_temperature", "afternoon_area_wide_temperature"],
    "pm": ["pm_t_f", "evening_area-wide_temperature", "evening_area_wide_temperature"],
}


# ---------------------------------------------------------------------------
# Public API — two-phase (download then assign to grid)
# ---------------------------------------------------------------------------

def download_capa(
    bbox: tuple[float, float, float, float],
    window_or_date_start: "date | TemporalWindow",
    date_end: Optional[date] = None,
    *,
    osf_node_id: Optional[str] = None,
    osf_folder_hint: Optional[str] = None,
) -> tuple[Optional[dict], list[date]]:
    """Download CAPA Heat Watch rasters and extract anchor dates.

    This is **Phase 1** of the two-phase collection pattern.  Temperature
    rasters are fetched from OSF without interpolating onto a fishnet.
    Call :func:`assign_capa_to_grid` afterwards once the fishnet has been
    created at the desired resolution.

    Parameters
    ----------
    bbox : (minx, miny, maxx, maxy) — kept for API symmetry; CAPA data covers
        the full campaign city regardless of bbox.
    window_or_date_start : TemporalWindow | date
    date_end : date | None  (legacy bare-date call style)
    osf_node_id : str | None
    osf_folder_hint : str | None

    Returns
    -------
    (raster_data | None, capa_dates)
        ``raster_data`` is the dict consumed by :func:`assign_capa_to_grid`.
        ``None`` when ``osf_node_id`` is absent or download fails.
    """
    if not osf_node_id:
        log.warning(
            "capa: osf_node_id not configured — no anchor dates. "
            "Add 'collect.capa_osf_node: <node_id>' to project.yml. "
            "Find your campaign at "
            "https://heat.gov/urban-heat-islands-mapping-campaign-program/"
        )
        return None, []
    try:
        source = _find_raster_source(osf_node_id, folder_hint=osf_folder_hint)
        log.info("capa: found raster source type=%s name='%s'", source["type"], source["name"])
        raster_data = _load_raster_data(source)
        campaign_date = _parse_campaign_date(source["name"])
        capa_dates = [campaign_date] if campaign_date is not None else []
        log.info("capa: downloaded rasters; campaign date %s", campaign_date)
        return raster_data, capa_dates
    except Exception as exc:
        log.warning("capa: download failed (%s) — data will be NaN", exc)
        return None, []


def assign_capa_to_grid(
    fishnet_gdf: object,
    raster_data: Optional[dict],
) -> object:
    """Interpolate downloaded CAPA rasters onto fishnet cell centroids.

    This is **Phase 2** of the two-phase collection pattern.

    Parameters
    ----------
    fishnet_gdf : gpd.GeoDataFrame
        Analysis grid at any resolution.
    raster_data : dict | None
        Raw raster data returned by :func:`download_capa`.  NaN-fills when
        ``None``.

    Returns
    -------
    gpd.GeoDataFrame
        Fishnet with ``aat_morning``, ``aat_midday``, ``aat_night``, and
        ``diurnal_aat`` columns appended.
    """
    if raster_data is None:
        return _nan_fishnet(fishnet_gdf)
    try:
        return _sample_rasters_to_fishnet(fishnet_gdf, raster_data)
    except Exception as exc:
        log.warning("capa: grid assignment failed (%s) — returning NaN", exc)
        return _nan_fishnet(fishnet_gdf)


# ---------------------------------------------------------------------------
# Legacy single-call API (kept for backward compatibility)
# ---------------------------------------------------------------------------

def fetch_capa(
    fishnet_gdf: object,
    bbox: tuple[float, float, float, float],
    window_or_date_start: "date | TemporalWindow",
    date_end: Optional[date] = None,
    *,
    osf_node_id: Optional[str] = None,
    osf_folder_hint: Optional[str] = None,
) -> tuple[object, list[date]]:
    """Fetch CAPA Heat Watch traverse data and populate the fishnet.

    Parameters
    ----------
    fishnet_gdf : gpd.GeoDataFrame
        30m analysis grid.  Returned with four new columns.
    bbox : (minx, miny, maxx, maxy)
        Study bounding box in EPSG:4326.  Not used for querying (CAPA data
        covers the whole campaign city); kept for API compatibility.
    window_or_date_start : TemporalWindow | date
        Either a :class:`TemporalWindow` (preferred) or a bare ``date``
        start value (legacy; *date_end* is then required).
    date_end : date | None
        Only used with the legacy bare-date call style.
    osf_node_id : str | None
        OSF project node ID for the Heat Watch campaign, e.g. ``"rk75w"``.
        Obtainable from the URL: https://osf.io/{node_id}/files/
        If None, returns NaN columns and empty anchor list.
    osf_folder_hint : str | None
        Optional folder name hint for multi-city project nodes (e.g. ``"tdsy7"``
        which contains dozens of cities). Set to a substring of the target city
        folder name so only that folder is scanned, e.g. ``"Boston"``.
        Not needed for single-campaign nodes like ``"rk75w"``.

    Returns
    -------
    (gdf, capa_dates)
        ``gdf`` — fishnet with ``aat_morning``, ``aat_midday``, ``aat_night``,
        and ``diurnal_aat`` columns.  NaN-filled on failure.
        ``capa_dates`` — campaign measurement dates; empty on failure.
    """
    if not osf_node_id:
        log.warning(
            "capa: osf_node_id not configured — returning NaN. "
            "Add 'collect.capa_osf_node: <node_id>' to project.yml. "
            "Find your campaign at "
            "https://heat.gov/urban-heat-islands-mapping-campaign-program/"
        )
        return _nan_fishnet(fishnet_gdf), []

    try:
        raster_data, capa_dates = download_capa(
            bbox, window_or_date_start, date_end,
            osf_node_id=osf_node_id,
            osf_folder_hint=osf_folder_hint,
        )
    except Exception as exc:
        log.warning("capa: download failed (%s) — returning NaN", exc)
        return _nan_fishnet(fishnet_gdf), []
    try:
        fishnet_out = assign_capa_to_grid(fishnet_gdf, raster_data)
    except Exception as exc:
        log.warning("capa: grid assignment failed (%s) — returning NaN", exc)
        fishnet_out = _nan_fishnet(fishnet_gdf)
    return fishnet_out, capa_dates


# ---------------------------------------------------------------------------
# OSF discovery + download
# ---------------------------------------------------------------------------

def _scan_storage(
    node_id: str,
    folder_id: str = "",
    depth: int = 0,
    top_level_filter: Optional[str] = None,
) -> list[dict]:
    """Recursively list every file in an OSF node's storage.

    Parameters
    ----------
    node_id          : OSF project node (e.g. ``"tdsy7"``)
    folder_id        : OSF file-storage folder ID (empty = root)
    depth            : current recursion depth (max 6)
    top_level_filter : If set, only recurse into *root-level* folders whose
                       name contains this substring (case-insensitive).  Useful
                       for multi-city project nodes like ``"tdsy7"`` where each
                       city lives in its own top-level folder.
    """
    if depth > 6:
        return []
    path: str = f"{folder_id}/" if folder_id else ""
    url: Optional[str] = OSF_API.format(node_id=node_id) + path
    files: list[dict] = []
    while url:
        raw = _http_get(url)
        payload = json.loads(raw.decode("utf-8"))
        for item in payload.get("data", []):
            kind: str = item["attributes"].get("kind", "file")
            name: str = item["attributes"]["name"]
            fid:  str = item["id"]
            if kind == "folder":
                # At depth 0, apply the city-folder filter if provided
                if depth == 0 and top_level_filter:
                    if top_level_filter.lower() not in name.lower():
                        continue  # skip this top-level folder
                files.extend(_scan_storage(node_id, fid, depth + 1, top_level_filter))
            else:
                files.append({"name": name, "id": fid})
        url = payload.get("links", {}).get("next")
    return files


def _find_raster_source(node_id: str, folder_hint: Optional[str] = None) -> dict:
    """Recursively search an OSF node for temperature raster data.

    Supports two formats:

    * **New (2021+)**: ``rasters_chw_*.zip`` containing ``am_t_f.tif``,
      ``af_t_f.tif``, ``pm_t_f.tif``.
    * **Old (2019-2020)**: Individual TIF files such as
      ``Morning_Area-wide_Temperature_*.tif`` inside a ``Surface Models``
      subfolder.

    Returns a source descriptor dict consumed by :func:`_load_raster_data`.
    """
    files = _scan_storage(node_id, top_level_filter=folder_hint)
    log.debug("capa: scanned %d files in OSF node %s (filter=%r)", len(files), node_id, folder_hint)

    # Priority 1: rasters_*.zip  (new format — single download)
    for f in files:
        nl = f["name"].lower()
        if nl.startswith("rasters_") and nl.endswith(".zip"):
            return {"type": "zip", "id": f["id"], "name": f["name"], "node_id": node_id}

    # Priority 1b: any zip that looks like it contains all raster data
    # (e.g. "All Data_Heat Watch New Orleans_110420.zip")
    for f in files:
        nl = f["name"].lower()
        if nl.endswith(".zip") and ("data" in nl or "raster" in nl or "heat" in nl):
            return {"type": "zip", "id": f["id"], "name": f["name"], "node_id": node_id}

    # Priority 2: individual temperature TIFs  (old format)
    tif_ids: dict[str, str] = {}
    for window, patterns in _WINDOW_TIF_PATTERNS.items():
        for f in files:
            nl = f["name"].lower()
            if nl.endswith(".tif") and any(nl.startswith(p) for p in patterns):
                tif_ids[window] = f["id"]
                break

    if tif_ids:
        # Find the best filename for date parsing (prefer one with _MMDDYY)
        date_name = next(
            (f["name"] for f in files if _DATE_RE.search(f["name"])),
            files[0]["name"] if files else "unknown",
        )
        return {"type": "tifs", "ids": tif_ids, "name": date_name, "node_id": node_id}

    raise FileNotFoundError(
        f"No raster data found in OSF node '{node_id}' after scanning {len(files)} files. "
        f"Expected a 'rasters_*.zip' (new format) or individual temperature TIFs (old format). "
        f"Check https://osf.io/{node_id}/files/"
    )


def _load_raster_data(source: dict) -> dict:
    """Download raster data from OSF Waterbutler.

    Returns
    -------
    ``{"_zip": bytes}`` for ZIP format, or
    ``{"am": bytes, "af": bytes, "pm": bytes}`` for individual TIFs.
    """
    node_id: str = source["node_id"]
    if source["type"] == "zip":
        url = OSF_WATERBUTLER.format(node_id=node_id, file_id=source["id"])
        log.info("capa: downloading rasters ZIP %s", source["name"])
        return {"_zip": _http_get(url)}
    else:
        tif_bytes: dict[str, bytes] = {}
        for window, fid in source["ids"].items():
            url = OSF_WATERBUTLER.format(node_id=node_id, file_id=fid)
            log.info("capa: downloading TIF for window '%s'", window)
            tif_bytes[window] = _http_get(url)
        return tif_bytes


def _http_get(url: str) -> bytes:
    req = urllib.request.Request(
        url, headers={"User-Agent": "SPARC-DataCollection/1.0"}
    )
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        return resp.read()


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

def _parse_campaign_date(filename: str) -> Optional[date]:
    """Extract the campaign date from any campaign filename.

    Searches for ``_MMDDYY`` anywhere in the filename::

        rasters_chw_brockton_092823.zip          → 2023-09-28
        Final Report_Heat Watch Boston_100419.pdf → 2019-10-04
        traverses_chw_mystic river day 1_112421  → 2021-11-24
    """
    m = _DATE_RE.search(filename)
    if not m:
        log.warning("capa: could not parse date from filename '%s'", filename)
        return None
    mm, dd, yy = int(m.group(1)), int(m.group(2)), int(m.group(3))
    year = 2000 + yy
    try:
        return date(year, mm, dd)
    except ValueError:
        log.warning("capa: invalid date in filename '%s'", filename)
        return None


# ---------------------------------------------------------------------------
# Raster sampling
# ---------------------------------------------------------------------------

def _sample_rasters_to_fishnet(fishnet_gdf: object, raster_data: dict) -> object:
    """Sample temperature rasters onto fishnet cell centroids.

    Parameters
    ----------
    raster_data
        Either ``{"_zip": bytes}`` (new format — ZIP extracted for ``am_t_f.tif`` etc.)
        or ``{"am": bytes, "af": bytes, "pm": bytes}`` (old format — individual TIF bytes).
        Values are kept in degrees Fahrenheit.
    """
    import geopandas as gpd
    import pathlib
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import transform as rio_transform

    gdf = fishnet_gdf.copy()  # type: ignore[union-attr]

    # Fishnet centroids in EPSG:4326
    centroids_proj = gdf.geometry.centroid  # type: ignore[union-attr]
    centroids_4326 = gpd.GeoSeries(centroids_proj, crs=gdf.crs).to_crs("EPSG:4326")  # type: ignore[union-attr]
    lons = centroids_4326.x.to_numpy(dtype=float)
    lats = centroids_4326.y.to_numpy(dtype=float)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)

        # Build {prefix: Path} regardless of source format
        tif_paths: dict[str, pathlib.Path] = {}
        if "_zip" in raster_data:
            # New format: extract ZIP; try both naming conventions:
            #   {prefix}_t_f.tif       (2022+ campaigns)
            #   {prefix}_t_f_ranger.tif (2020-2021 campaigns using Ranger interpolation)
            with zipfile.ZipFile(io.BytesIO(raster_data["_zip"])) as z:
                z.extractall(tmp_path)
            for prefix in _WINDOWS:
                # Try multiple naming conventions (oldest → newest campaigns)
                glob_patterns = [
                    f"{prefix}_t_f.tif",            # 2022+ standard
                    f"{prefix}_t_f_ranger.tif",     # 2020-2021 Ranger interpolation
                    f"*_{prefix}_temp_f.tif",       # 2024+ city-prefixed format
                    f"*_{prefix}_t_f.tif",          # variant with city prefix
                ]
                for pat in glob_patterns:
                    # rglob handles both root-level and nested subdirectory files
                    matches = list(tmp_path.rglob(pat))
                    if matches:
                        tif_paths[prefix] = matches[0]
                        break
        else:
            # Old format: write raw TIF bytes keyed by window prefix
            for prefix, tif_bytes in raster_data.items():
                p = tmp_path / f"{prefix}.tif"
                p.write_bytes(tif_bytes)
                tif_paths[prefix] = p

        for prefix, out_col in _WINDOWS.items():
            tif = tif_paths.get(prefix)
            if tif is None or not tif.exists():
                log.warning("capa: no TIF for window '%s'", prefix)
                gdf[out_col] = float("nan")  # type: ignore[index]
                continue

            with rasterio.open(tif) as src:
                raster_crs = src.crs
                if raster_crs and raster_crs != CRS.from_epsg(4326):
                    xs_r, ys_r = rio_transform("EPSG:4326", raster_crs, lons, lats)
                else:
                    xs_r, ys_r = lons, lats

                nodata = src.nodata
                vals = np.array(
                    [v[0] for v in src.sample(zip(xs_r, ys_r), indexes=1)],
                    dtype=float,
                )

            if nodata is not None:
                vals[vals == nodata] = np.nan
            vals[vals <= -999.0] = np.nan  # -9999 nodata per README

            gdf[out_col] = vals  # type: ignore[index]

    # Diurnal range: afternoon − evening (in °F)
    midday = gdf["aat_midday"].to_numpy(dtype=float, na_value=float("nan"))  # type: ignore[union-attr]
    night  = gdf["aat_night"].to_numpy(dtype=float, na_value=float("nan"))   # type: ignore[union-attr]
    gdf["diurnal_aat"] = midday - night  # type: ignore[index]

    return gdf


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

def _nan_fishnet(fishnet_gdf: object) -> object:
    """Return the fishnet with NaN columns for all four CAPA outputs."""
    try:
        gdf = fishnet_gdf.copy()  # type: ignore[union-attr]
    except Exception:
        gdf = fishnet_gdf
    for col in ("aat_morning", "aat_midday", "aat_night", "diurnal_aat"):
        gdf[col] = float("nan")  # type: ignore[index]
    return gdf
