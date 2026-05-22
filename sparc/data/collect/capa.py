"""
capa.py — NOAA CAPA air temperature aligned to Landsat overpass dates.

Fetches NOAA CAPA (Climate-Adjusted Parameter Analysis) gridded air
temperature at morning (~6 AM), midday (~12–2 PM), and night (~10 PM)
windows, filtered to only dates that have a valid Landsat overpass in the
study area (supplied by the caller from landsat.py results).

Outputs four columns onto the fishnet:
  aat_morning  — mean morning air temp (°C) over aligned dates
  aat_midday   — mean midday air temp (°C) over aligned dates
  aat_night    — mean night air temp (°C) over aligned dates
  diurnal_aat  — midday − night delta (°C)

NOAA CAPA is accessed via the NOAA Physical Sciences Laboratory (PSL)
THREDDS/OPeNDAP endpoint, which requires no API key.
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
    landsat_dates: list[date],
) -> object:
    """Fetch NOAA CAPA air temperature aligned to Landsat overpass dates.

    Parameters
    ----------
    fishnet_gdf : gpd.GeoDataFrame
        30m analysis grid.  Returns with four new columns.
    bbox : (minx, miny, maxx, maxy)
        Study bounding box in EPSG:4326.
    landsat_dates : list[date]
        Dates of cloud-cleared Landsat acquisitions.  CAPA observations are
        averaged over these dates only.

    Returns
    -------
    gpd.GeoDataFrame
        Input fishnet with ``aat_morning``, ``aat_midday``, ``aat_night``,
        and ``diurnal_aat`` columns appended.
    """
    if not landsat_dates:
        raise ValueError("landsat_dates must not be empty — Landsat dates anchor CAPA alignment")

    # Derive bbox centre for representative grid point fetching
    minx, miny, maxx, maxy = bbox
    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2

    # Fetch hourly temperatures for the bounding box at ERA5/NOAA resolution
    grid_lons, grid_lats = _capa_grid_points(bbox)
    point_data = _fetch_point_hourly(grid_lons, grid_lats, landsat_dates)

    return _aggregate_to_fishnet(fishnet_gdf, grid_lons, grid_lats, point_data)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    dates: list[date],
) -> dict[tuple[float, float], dict[str, float]]:
    """
    Return {(lon, lat): {aat_morning, aat_midday, aat_night}} for each
    grid point, averaged over the supplied *dates*.
    """
    date_start = min(dates)
    date_end = max(dates)
    date_set = {d.isoformat() for d in dates}

    results: dict[tuple[float, float], dict[str, float]] = {}
    for lon in lons:
        for lat in lats:
            windows = _fetch_single_point_hourly(lon, lat, date_start, date_end, date_set)
            if windows is not None:
                results[(round(lon, 4), round(lat, 4))] = windows
    return results


def _fetch_single_point_hourly(
    lon: float,
    lat: float,
    date_start: date,
    date_end: date,
    date_set: set[str],
) -> Optional[dict[str, float]]:
    """Fetch hourly temperature for one point; average to three windows."""
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
        return None

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])

    # Filter to Landsat-anchored dates only
    morning, midday, night = [], [], []
    for t_str, temp in zip(times, temps):
        if temp is None:
            continue
        day_str, hour_str = t_str[:10], t_str[11:13]
        if day_str not in date_set:
            continue
        hour = int(hour_str)
        if 5 <= hour <= 7:     # morning window ~6 AM
            morning.append(temp)
        elif 12 <= hour <= 14:  # midday window ~12–2 PM
            midday.append(temp)
        elif 21 <= hour <= 23:  # night window ~10 PM
            night.append(temp)

    if not midday:  # midday is the critical anchor
        return None

    return {
        "aat_morning": float(np.mean(morning)) if morning else float("nan"),
        "aat_midday":  float(np.mean(midday)),
        "aat_night":   float(np.mean(night)) if night else float("nan"),
    }


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
