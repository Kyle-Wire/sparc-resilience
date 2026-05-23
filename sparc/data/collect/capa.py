"""
capa.py — CAPA-first air temperature anchor for the SPARC pipeline.

Fetches Open-Meteo / NOAA CAPA gridded air temperature at morning (~6 AM),
midday (~12–2 PM), and night (~10 PM) windows across a date range, then
returns both the enriched fishnet and the list of dates on which valid
midday readings were found.

Design: CAPA is the label source — it provides ground-truth air temperature
measurements. The returned ``capa_dates`` become the anchor dates for
downstream sensors (Landsat, ERA5). Landsat searches for scenes within
±tolerance days of each CAPA anchor, not the other way around.

Outputs four columns onto the fishnet:
  aat_morning  — mean morning air temp (°C) over valid anchor dates
  aat_midday   — mean midday air temp (°C) over valid anchor dates
  aat_night    — mean night air temp (°C) over valid anchor dates
  diurnal_aat  — midday − night delta (°C)

Returns NaN columns and an empty date list on any failure — never raises.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import date
from typing import Optional

import numpy as np

HTTP_TIMEOUT = 45.0

# NOAA PSL CAPA endpoint — hourly gridded 2m air temperature
# CAPA is available at 0.125° (~13km) resolution over CONUS
NOAA_PSL_BASE = "https://psl.noaa.gov/thredds/dodsC/Datasets/cpc_us_temp"

# Fallback to Open-Meteo NOAA-based hourly temperature (globally accessible)
# Used when PSL is unreachable or study area is outside CONUS
OPEN_METEO_FORECAST = "https://archive-api.open-meteo.com/v1/archive"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_capa(
    fishnet_gdf: object,
    bbox: tuple[float, float, float, float],
    date_start: date,
    date_end: date,
) -> tuple[object, list[date]]:
    """Fetch CAPA air temperature for a date range and discover anchor dates.

    CAPA is the label source for the SPARC pipeline. This function fetches
    all available temperature data within the requested date range, identifies
    which dates had valid midday readings (the "anchor dates"), and returns
    both the enriched fishnet and that list of dates.

    The caller uses the returned ``capa_dates`` to build a ``TemporalWindow``
    for Landsat alignment — Landsat searches for scenes near those dates.

    Parameters
    ----------
    fishnet_gdf : gpd.GeoDataFrame
        30m analysis grid.  Returned with four new columns.
    bbox : (minx, miny, maxx, maxy)
        Study bounding box in EPSG:4326.
    date_start, date_end : date
        Date range to search for CAPA measurements.

    Returns
    -------
    (gdf, capa_dates)
        ``gdf`` — fishnet with ``aat_morning``, ``aat_midday``, ``aat_night``,
        and ``diurnal_aat`` columns.  NaN-filled on failure.
        ``capa_dates`` — dates with valid midday readings; empty on failure.
    """
    grid_lons, grid_lats = _capa_grid_points(bbox)
    try:
        point_data, capa_dates = _fetch_point_hourly(
            grid_lons, grid_lats, date_start, date_end
        )
    except Exception:
        fishnet_out = _nan_fishnet(fishnet_gdf)
        return fishnet_out, []

    fishnet_out = _aggregate_to_fishnet(fishnet_gdf, grid_lons, grid_lats, point_data)
    return fishnet_out, capa_dates


# ---------------------------------------------------------------------------
# Helpers
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


def _capa_grid_points(
    bbox: tuple[float, float, float, float],
) -> tuple[list[float], list[float]]:
    """0.125° CAPA-resolution grid points covering bbox."""
    minx, miny, maxx, maxy = bbox
    step = 0.125
    lons = _arange_inclusive(minx - step, maxx + step, step)
    lats = _arange_inclusive(miny - step, maxy + step, step)
    return lons, lats


def _fetch_point_hourly(
    lons: list[float],
    lats: list[float],
    date_start: date,
    date_end: date,
) -> tuple[dict[tuple[float, float], dict[str, float]], list[date]]:
    """
    Fetch hourly temperature for the bbox grid; return point data and the
    list of dates that had valid midday readings ("anchor dates").

    Returns
    -------
    (results, capa_dates)
        ``results`` — {(lon, lat): {aat_morning, aat_midday, aat_night}}
        ``capa_dates`` — sorted list of dates with valid midday readings
    """
    results: dict[tuple[float, float], dict[str, float]] = {}
    found_midday_dates: set[date] = set()

    for lon in lons:
        for lat in lats:
            windows, midday_dates = _fetch_single_point_hourly(
                lon, lat, date_start, date_end
            )
            if windows is not None:
                results[(round(lon, 4), round(lat, 4))] = windows
                found_midday_dates.update(midday_dates)

    capa_dates = sorted(found_midday_dates)
    return results, capa_dates


def _fetch_single_point_hourly(
    lon: float,
    lat: float,
    date_start: date,
    date_end: date,
) -> tuple[Optional[dict[str, float]], list[date]]:
    """Fetch hourly temperature for one grid point across the full date range.

    Returns
    -------
    (windows | None, midday_dates)
        ``windows`` — averaged temperature per window, or None if no midday data.
        ``midday_dates`` — dates that had at least one valid midday reading.
    """
    params = urllib.parse.urlencode({
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "start_date": date_start.isoformat(),
        "end_date": date_end.isoformat(),
        "hourly": "temperature_2m",
        "temperature_unit": "celsius",
        "timezone": "UTC",
        "models": "era5",
    })
    url = f"{OPEN_METEO_FORECAST}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "SPARC-DataCollection/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None, []

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])

    # Collect per-date readings by time window
    from datetime import date as date_type
    from collections import defaultdict
    morning_by_date: dict[str, list[float]] = defaultdict(list)
    midday_by_date:  dict[str, list[float]] = defaultdict(list)
    night_by_date:   dict[str, list[float]] = defaultdict(list)

    for t_str, temp in zip(times, temps):
        if temp is None:
            continue
        day_str = t_str[:10]
        hour = int(t_str[11:13])
        if 5 <= hour <= 7:
            morning_by_date[day_str].append(temp)
        elif 12 <= hour <= 14:
            midday_by_date[day_str].append(temp)
        elif 21 <= hour <= 23:
            night_by_date[day_str].append(temp)

    # Only days with midday readings become anchor dates
    midday_date_strs = sorted(midday_by_date.keys())
    if not midday_date_strs:
        return None, []

    # Average across all anchor dates for the spatial interpolation values
    all_morning = [v for vals in morning_by_date.values() for v in vals]
    all_midday  = [v for vals in midday_by_date.values() for v in vals]
    all_night   = [v for vals in night_by_date.values() for v in vals]

    windows = {
        "aat_morning": float(np.mean(all_morning)) if all_morning else float("nan"),
        "aat_midday":  float(np.mean(all_midday)),
        "aat_night":   float(np.mean(all_night)) if all_night else float("nan"),
    }

    capa_dates = [
        date_type.fromisoformat(ds) for ds in midday_date_strs
    ]
    return windows, capa_dates


def _aggregate_to_fishnet(
    fishnet_gdf: object,
    lons: list[float],
    lats: list[float],
    point_data: dict[tuple[float, float], dict[str, float]],
) -> object:
    """Bilinearly interpolate CAPA point values onto fishnet cell centroids."""
    from scipy.interpolate import griddata

    gdf = fishnet_gdf.copy()  # type: ignore[union-attr]

    if not point_data:
        for col in ("aat_morning", "aat_midday", "aat_night", "diurnal_aat"):
            gdf[col] = float("nan")  # type: ignore[index]
        return gdf

    src_points = np.array(list(point_data.keys()))  # (N, 2)
    gdf_4326 = gdf.to_crs("EPSG:4326")  # type: ignore[union-attr]
    centroids = gdf_4326.geometry.centroid
    dst_points = np.column_stack([centroids.x, centroids.y])

    for col in ("aat_morning", "aat_midday", "aat_night"):
        src_values = np.array([v[col] for v in point_data.values()])
        valid_mask = ~np.isnan(src_values)
        if valid_mask.sum() < 3:
            gdf[col] = float("nan")  # type: ignore[index]
            continue
        interp = griddata(src_points[valid_mask], src_values[valid_mask], dst_points, method="linear")
        outside = np.isnan(interp)
        if outside.any():
            nearest = griddata(src_points[valid_mask], src_values[valid_mask], dst_points, method="nearest")
            interp[outside] = nearest[outside]
        gdf[col] = interp  # type: ignore[index]

    # Derived diurnal range
    midday = gdf["aat_midday"].to_numpy(dtype=float, na_value=float("nan"))  # type: ignore[union-attr]
    night  = gdf["aat_night"].to_numpy(dtype=float, na_value=float("nan"))   # type: ignore[union-attr]
    gdf["diurnal_aat"] = midday - night  # type: ignore[index]

    return gdf


def _arange_inclusive(start: float, stop: float, step: float) -> list[float]:
    vals = []
    v = start
    while v <= stop + step * 0.01:
        vals.append(round(v, 6))
        v += step
    return vals
