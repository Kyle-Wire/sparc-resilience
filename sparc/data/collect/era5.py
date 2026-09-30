"""
era5.py — ERA5 background temperature via Open-Meteo.

Fetches hourly 2 m air temperature from the Open-Meteo Historical Weather
API (no API key required) for the bounding box of the study area, aggregates
to daily morning / midday / night means, and bilinearly downscales from the
ERA5 native ~0.25° grid to the analysis fishnet.

The ``era5_t2m`` column written to the fishnet is the daily mean temperature
in °C over the requested date range, spatially interpolated to each 30m cell.
It is subtracted from the CAPA air temperature to produce ``aat_residual``.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from ._temporal import TemporalWindow

HTTP_TIMEOUT = 30.0
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


# ---------------------------------------------------------------------------
# Public API — two-phase (download then assign to grid)
# ---------------------------------------------------------------------------

def download_era5(
    bbox: tuple[float, float, float, float],
    window_or_date_start: "date | TemporalWindow",
    date_end: Optional[date] = None,
) -> tuple[list[float], list[float], dict]:
    """Fetch ERA5 temperature at grid points covering *bbox*.

    This is **Phase 1** of the two-phase collection pattern.  Temperature values
    are fetched at ERA5 native 0.25° grid points without requiring a fishnet.
    Call :func:`assign_era5_to_grid` afterwards to downscale these values onto
    a fishnet created at the desired resolution.

    Parameters
    ----------
    bbox : (minx, miny, maxx, maxy) in EPSG:4326
    window_or_date_start : TemporalWindow | date
    date_end : date | None

    Returns
    -------
    (grid_lons, grid_lats, point_means)
        ERA5 grid point coordinates and mean daily temperatures (°C).
    """
    from ._temporal import TemporalWindow as _TW
    if isinstance(window_or_date_start, _TW):
        d_start, d_end = window_or_date_start.date_start, window_or_date_start.date_end
    else:
        d_start = window_or_date_start  # type: ignore[assignment]
        d_end = date_end  # type: ignore[assignment]

    grid_lons, grid_lats = _era5_grid_points_in_bbox(bbox)
    point_means = _fetch_point_means(grid_lons, grid_lats, d_start, d_end)
    return grid_lons, grid_lats, point_means


def assign_era5_to_grid(
    fishnet_gdf: object,
    grid_lons: list[float],
    grid_lats: list[float],
    point_means: dict,
) -> object:
    """Downscale ERA5 grid point temperatures to fishnet cells.

    This is **Phase 2** of the two-phase collection pattern.

    Parameters
    ----------
    fishnet_gdf : gpd.GeoDataFrame
        Analysis grid at any resolution.
    grid_lons, grid_lats : list[float]
        ERA5 grid point coordinates from :func:`download_era5`.
    point_means : dict
        Temperature values from :func:`download_era5`.

    Returns
    -------
    gpd.GeoDataFrame with ``era5_t2m`` column appended.
    """
    return _downscale_to_fishnet(fishnet_gdf, grid_lons, grid_lats, point_means)


# ---------------------------------------------------------------------------
# Legacy single-call API (kept for backward compatibility)
# ---------------------------------------------------------------------------

def fetch_era5(
    fishnet_gdf: object,
    bbox: tuple[float, float, float, float],
    window_or_date_start: "date | TemporalWindow",
    date_end: Optional[date] = None,
) -> object:
    """Fetch ERA5 2 m temperature and downscale onto the analysis fishnet.

    Parameters
    ----------
    fishnet_gdf : gpd.GeoDataFrame
        30m analysis grid.  Returned with ``era5_t2m`` column appended.
    bbox : (minx, miny, maxx, maxy)
        Study-area bounding box in EPSG:4326.
    window_or_date_start : TemporalWindow | date
        Either a :class:`TemporalWindow` (preferred — aligns ERA5 to CAPA
        anchor dates) or a bare ``date`` for the start of the averaging window
        (legacy; requires *date_end*).
    date_end : date | None
        End of the averaging window.  Only used when *window_or_date_start*
        is a bare ``date`` (legacy call style).

    Returns
    -------
    gpd.GeoDataFrame
        Input fishnet with ``era5_t2m`` (°C, daily mean) appended.
        In panel mode with anchor dates, an ``era5_t2m_<date>`` column is
        appended for each anchor date instead.
    """
    # Resolve date range from either a TemporalWindow or legacy bare dates
    from ._temporal import TemporalWindow as _TW
    if isinstance(window_or_date_start, _TW):
        d_start, d_end = window_or_date_start.date_start, window_or_date_start.date_end
    else:
        d_start = window_or_date_start  # type: ignore[assignment]
        d_end = date_end  # type: ignore[assignment]

    grid_lons, grid_lats, point_means = download_era5(bbox, d_start, d_end)
    return assign_era5_to_grid(fishnet_gdf, grid_lons, grid_lats, point_means)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _era5_grid_points_in_bbox(
    bbox: tuple[float, float, float, float],
) -> tuple[list[float], list[float]]:
    """Return lon/lat grid point centres covering *bbox* at ERA5 0.25° spacing."""
    minx, miny, maxx, maxy = bbox
    step = 0.25
    # Expand slightly so interpolation doesn't extrapolate at edges
    lons = _arange_inclusive(minx - step, maxx + step, step)
    lats = _arange_inclusive(miny - step, maxy + step, step)
    return lons, lats


def _fetch_point_means(
    lons: list[float],
    lats: list[float],
    date_start: date,
    date_end: date,
) -> dict[tuple[float, float], float]:
    """Fetch daily mean 2 m temperature for each (lon, lat) ERA5 grid point."""
    results: dict[tuple[float, float], float] = {}
    for lon in lons:
        for lat in lats:
            mean_t = _fetch_single_point(lon, lat, date_start, date_end)
            if mean_t is not None:
                results[(round(lon, 4), round(lat, 4))] = mean_t
    return results


def _fetch_single_point(
    lon: float,
    lat: float,
    date_start: date,
    date_end: date,
) -> Optional[float]:
    """Fetch mean 2 m temperature for one ERA5 grid point via Open-Meteo."""
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
    url = f"{OPEN_METEO_ARCHIVE}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "SPARC-DataCollection/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None  # Silently skip unreachable points; coverage check handles it

    temps = data.get("hourly", {}).get("temperature_2m", [])
    valid = [t for t in temps if t is not None]
    if not valid:
        return None
    return float(np.mean(valid))


def _downscale_to_fishnet(
    fishnet_gdf: object,
    lons: list[float],
    lats: list[float],
    point_means: dict[tuple[float, float], float],
) -> object:
    """Bilinearly interpolate ERA5 point means onto fishnet cell centroids."""
    if not point_means:
        gdf = fishnet_gdf.copy()  # type: ignore[union-attr]
        gdf["era5_t2m"] = float("nan")  # type: ignore[index]
        return gdf

    from scipy.interpolate import griddata

    src_points = np.array(list(point_means.keys()))   # (N, 2) lon/lat
    src_values = np.array(list(point_means.values())) # (N,)

    # Compute centroids in projected CRS to avoid geographic-CRS warnings,
    # then reproject the centroid points to lon/lat for griddata.
    import geopandas as gpd
    centroids_proj = fishnet_gdf.geometry.centroid  # type: ignore[union-attr]
    centroids_4326 = gpd.GeoSeries(centroids_proj, crs=fishnet_gdf.crs).to_crs("EPSG:4326")  # type: ignore[union-attr]
    dst_points = np.column_stack([centroids_4326.x, centroids_4326.y])

    interpolated = griddata(src_points, src_values, dst_points, method="linear")
    # Fall back to nearest for cells outside the convex hull of ERA5 points
    outside = np.isnan(interpolated)
    if outside.any():
        nearest = griddata(src_points, src_values, dst_points, method="nearest")
        interpolated[outside] = nearest[outside]

    gdf = fishnet_gdf.copy()  # type: ignore[union-attr]
    gdf["era5_t2m"] = interpolated  # type: ignore[index]
    return gdf


# ---------------------------------------------------------------------------
# ERA5 boundary conditions — hourly 3-window fetch
# ---------------------------------------------------------------------------

# CAPA Heat Watch traverse protocol: three one-hour traverses per campaign
# day, ~6–7 am, 3–4 pm and 7–8 pm LOCAL time.  The ERA5 value for each window
# is the mean of the listed local clock hours.  These are protocol defaults —
# check them against the actual traverse timestamps (``capa.parse_traverse_table``)
# for each campaign when available.
DEFAULT_BOUNDARY_WINDOWS: dict[str, tuple[int, ...]] = {
    "morning": (6, 7),
    "midday":  (15, 16),   # CAPA "afternoon" traverse; key kept as "midday"
    "evening": (19, 20),
}

# Per-window output variables of :func:`download_era5_boundary`.
#   t2m [°C], windspeed [m/s, scalar mean], winddir [deg, from vector mean],
#   rh [%], ssrd [W/m², Open-Meteo shortwave_radiation = preceding-hour mean
#   GHI], blh [m], u10 / v10 [m/s, vector-mean wind components].
ERA5_BOUNDARY_VARIABLES: tuple[str, ...] = (
    "t2m", "windspeed", "winddir", "rh", "ssrd", "blh", "u10", "v10",
)

_BASE_HOURLY_VARS = "temperature_2m,windspeed_10m,winddirection_10m,relativehumidity_2m"
_EXTENDED_HOURLY_VARS = _BASE_HOURLY_VARS + ",shortwave_radiation,boundary_layer_height"


def _normalize_windows(
    windows: "dict[str, tuple[int, ...]] | None",
) -> dict[str, tuple[int, ...]]:
    """Merge user windows over the CAPA-protocol defaults (keys fixed)."""
    out = dict(DEFAULT_BOUNDARY_WINDOWS)
    if windows:
        for key, hours in windows.items():
            if key == "afternoon":
                key = "midday"
            if key not in out:
                raise ValueError(
                    f"Unknown ERA5 window {key!r}; expected one of {sorted(out)}"
                )
            if isinstance(hours, int):
                hours = (hours,)
            hrs = tuple(int(h) for h in hours)
            if not hrs or any(h < 0 or h > 23 for h in hrs):
                raise ValueError(f"ERA5 window {key!r} hours must be in 0..23, got {hours!r}")
            out[key] = hrs
    return out


def download_era5_boundary(
    bbox: tuple[float, float, float, float],
    campaign_date: "date",
    timezone: Optional[str] = None,
    windows: "dict[str, tuple[int, ...]] | None" = None,
) -> tuple[list[float], list[float], dict]:
    """Fetch ERA5 hourly boundary conditions for 3 LOCAL-time windows on the campaign date.

    Fetches ``temperature_2m``, ``windspeed_10m``, ``winddirection_10m``,
    ``relativehumidity_2m``, ``shortwave_radiation`` and
    ``boundary_layer_height`` at ERA5 native grid points covering *bbox*.

    The request uses ``timezone=<IANA tz>`` (or ``"auto"``, which Open-Meteo
    resolves from the coordinates), so the hourly arrays are in local clock
    time — DST included — for the local campaign date.  (The former
    ``round(lon/15)`` UTC arithmetic ignored DST and, with ``timezone=UTC``,
    made the evening window wrap onto the *previous* local evening.)

    Parameters
    ----------
    bbox : (minx, miny, maxx, maxy) in EPSG:4326
    campaign_date : date
        The CAPA campaign date (local).  Only data from this single day is fetched.
    timezone : str | None
        IANA time zone of the city (e.g. ``"America/New_York"``).  ``None``
        → ``"auto"``.
    windows : dict[str, tuple[int, ...]] | None
        Local clock hours averaged per window, keys ``morning`` / ``midday``
        (alias ``afternoon``) / ``evening``.  Defaults to the CAPA Heat Watch
        traverse protocol — morning (6, 7), afternoon (15, 16), evening
        (19, 20) — see :data:`DEFAULT_BOUNDARY_WINDOWS`; verify against the
        traverse timestamps where available.

    Returns
    -------
    (grid_lons, grid_lats, boundary_data)
        ``boundary_data`` maps ``(lon, lat)`` to a dict with keys
        ``"morning"``, ``"midday"``, ``"evening"``, each holding a sub-dict
        with the keys in :data:`ERA5_BOUNDARY_VARIABLES`
        (``t2m, windspeed, winddir, rh, ssrd, blh, u10, v10``).
    """
    win = _normalize_windows(windows)
    tz = timezone or "auto"

    grid_lons, grid_lats = _era5_grid_points_in_bbox(bbox)

    boundary_data: dict = {}
    for lon in grid_lons:
        for lat in grid_lats:
            point_data = _fetch_boundary_point(
                lon, lat, campaign_date, windows=win, timezone=tz,
            )
            boundary_data[(round(lon, 4), round(lat, 4))] = point_data

    return grid_lons, grid_lats, boundary_data


def assign_era5_boundary_to_grid(
    fishnet_gdf: object,
    grid_lons: list[float],
    grid_lats: list[float],
    boundary_data: dict,
) -> object:
    """Bilinearly downscale 3-window ERA5 boundary conditions onto fishnet centroids.

    Appends 24 columns ``era5_{window}_{var}`` to *fishnet_gdf* for
    window ∈ (morning, midday, evening) and var ∈
    :data:`ERA5_BOUNDARY_VARIABLES`::

        era5_morning_t2m,  era5_morning_windspeed,  era5_morning_winddir,  era5_morning_rh,
        era5_morning_ssrd, era5_morning_blh,        era5_morning_u10,      era5_morning_v10,
        ... (same for midday and evening)

    Downscaling ``winddir`` linearly is only approximate across the 0/360°
    wrap; ``u10`` / ``v10`` are the preferred wind inputs.

    Falls back to NaN on any failure.

    Parameters
    ----------
    fishnet_gdf : gpd.GeoDataFrame
    grid_lons, grid_lats : list[float]
        ERA5 grid point coordinates from :func:`download_era5_boundary`.
    boundary_data : dict
        Boundary data from :func:`download_era5_boundary`.
    """
    _ERA5_BOUNDARY_COLS = [
        f"era5_{w}_{v}"
        for w in ("morning", "midday", "evening")
        for v in ERA5_BOUNDARY_VARIABLES
    ]

    gdf = fishnet_gdf.copy()  # type: ignore[union-attr]

    if not boundary_data:
        for col in _ERA5_BOUNDARY_COLS:
            gdf[col] = float("nan")  # type: ignore[index]
        return gdf

    try:
        from scipy.interpolate import griddata
        import geopandas as gpd

        centroids_proj = gdf.geometry.centroid  # type: ignore[union-attr]
        centroids_4326 = gpd.GeoSeries(
            centroids_proj, crs=gdf.crs  # type: ignore[union-attr]
        ).to_crs("EPSG:4326")
        dst_points = np.column_stack([centroids_4326.x, centroids_4326.y])

        src_keys = list(boundary_data.keys())          # list of (lon, lat)
        src_coords = np.array(src_keys)                # (M, 2)

        windows = ("morning", "midday", "evening")
        variables = ERA5_BOUNDARY_VARIABLES

        for window in windows:
            for var in variables:
                col_name = f"era5_{window}_{var}"
                try:
                    src_vals = np.array([
                        boundary_data[k].get(window, {}).get(var, float("nan"))
                        for k in src_keys
                    ], dtype=float)

                    if np.all(np.isnan(src_vals)):
                        gdf[col_name] = float("nan")  # type: ignore[index]
                        continue

                    valid_mask = ~np.isnan(src_vals)
                    interp = griddata(
                        src_coords[valid_mask],
                        src_vals[valid_mask],
                        dst_points,
                        method="linear",
                    )
                    outside = np.isnan(interp)
                    if outside.any():
                        nearest = griddata(
                            src_coords[valid_mask],
                            src_vals[valid_mask],
                            dst_points,
                            method="nearest",
                        )
                        interp[outside] = nearest[outside]
                    gdf[col_name] = interp  # type: ignore[index]
                except Exception:
                    gdf[col_name] = float("nan")  # type: ignore[index]

    except Exception:
        for col in _ERA5_BOUNDARY_COLS:
            if col not in gdf.columns:  # type: ignore[union-attr]
                gdf[col] = float("nan")  # type: ignore[index]

    return gdf


# ---------------------------------------------------------------------------
# ERA5 boundary helpers
# ---------------------------------------------------------------------------

def _fetch_boundary_point(
    lon: float,
    lat: float,
    campaign_date: "date",
    windows: "dict[str, tuple[int, ...]] | None" = None,
    timezone: str = "auto",
) -> dict:
    """Fetch hourly ERA5 boundary variables at one grid point for one LOCAL day.

    Hourly values are requested in local time (``timezone``) and each window
    averages the listed local clock hours of *campaign_date*.  Wind direction
    is the direction of the vector-mean wind; ``u10 = −ws·sin(wd)``,
    ``v10 = −ws·cos(wd)`` (meteorological convention: wd is where the wind
    blows FROM, degrees clockwise from north).  If the extended request
    (with ``shortwave_radiation`` / ``boundary_layer_height``) is rejected,
    the base variables are re-requested and ``ssrd`` / ``blh`` stay NaN.
    """
    win = _normalize_windows(windows)
    nan = float("nan")
    result = {w: {v: nan for v in ERA5_BOUNDARY_VARIABLES} for w in win}

    date_str = campaign_date.isoformat()

    def _request(hourly_vars: str) -> dict:
        params = urllib.parse.urlencode({
            "latitude":   f"{lat:.4f}",
            "longitude":  f"{lon:.4f}",
            "start_date": date_str,
            "end_date":   date_str,
            "hourly": hourly_vars,
            "temperature_unit": "celsius",
            "windspeed_unit": "ms",
            "timezone": timezone or "auto",
            "models": "era5",
        })
        url = f"{OPEN_METEO_ARCHIVE}?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "SPARC-DataCollection/1.0"})
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))

    try:
        data = _request(_EXTENDED_HOURLY_VARS)
    except urllib.error.HTTPError as exc:
        if exc.code != 400:
            return result
        # Variable rejected by the API → retry with the base variables only.
        try:
            data = _request(_BASE_HOURLY_VARS)
        except Exception:
            return result
    except Exception:
        return result

    hourly = data.get("hourly", {}) or {}
    t2m_arr = hourly.get("temperature_2m", []) or []
    ws_arr  = hourly.get("windspeed_10m", hourly.get("wind_speed_10m", [])) or []
    wd_arr  = hourly.get("winddirection_10m", hourly.get("wind_direction_10m", [])) or []
    rh_arr  = hourly.get("relativehumidity_2m", hourly.get("relative_humidity_2m", [])) or []
    sw_arr  = hourly.get("shortwave_radiation", []) or []
    blh_arr = hourly.get("boundary_layer_height", []) or []
    times   = hourly.get("time", []) or []

    # Map local clock hour → array indices on the campaign date.  With a
    # ``time`` array this is robust to DST days (23/25 entries); otherwise
    # fall back to index == hour.
    hour_to_idx: dict[int, list[int]] = {}
    if times:
        for i, ts in enumerate(times):
            ts = str(ts)
            if ts[:10] != date_str or len(ts) < 13:
                continue
            try:
                hour_to_idx.setdefault(int(ts[11:13]), []).append(i)
            except ValueError:
                continue
    else:
        n = max(len(t2m_arr), len(ws_arr), len(wd_arr), len(rh_arr))
        hour_to_idx = {h: [h] for h in range(min(n, 24))}

    def _vals(arr: list, idxs: list[int]) -> np.ndarray:
        out = [float(arr[i]) for i in idxs if i < len(arr) and arr[i] is not None]
        return np.asarray(out, dtype=float)

    def _mean(arr: list, idxs: list[int]) -> float:
        v = _vals(arr, idxs)
        return float(np.mean(v)) if v.size else nan

    for window, hours in win.items():
        idxs = [i for h in hours for i in hour_to_idx.get(h, [])]
        if not idxs:
            continue
        # Wind: vector components per hour (need paired ws/wd).
        pairs = [
            (float(ws_arr[i]), float(wd_arr[i]))
            for i in idxs
            if i < len(ws_arr) and i < len(wd_arr)
            and ws_arr[i] is not None and wd_arr[i] is not None
        ]
        if pairs:
            ws = np.array([p[0] for p in pairs])
            wd = np.deg2rad(np.array([p[1] for p in pairs]))
            u = -ws * np.sin(wd)
            v = -ws * np.cos(wd)
            u_m, v_m = float(u.mean()), float(v.mean())
            if abs(u_m) < 1e-12 and abs(v_m) < 1e-12:
                wd_m = float(np.rad2deg(wd[0]) % 360.0)
            else:
                wd_m = float(np.rad2deg(np.arctan2(-u_m, -v_m)) % 360.0)
            ws_m = float(ws.mean())
        else:
            u_m = v_m = wd_m = nan
            ws_m = _mean(ws_arr, idxs)
        result[window] = {
            "t2m":       _mean(t2m_arr, idxs),
            "windspeed": ws_m,
            "winddir":   wd_m,
            "rh":        _mean(rh_arr, idxs),
            "ssrd":      _mean(sw_arr, idxs),
            "blh":       _mean(blh_arr, idxs),
            "u10":       u_m,
            "v10":       v_m,
        }

    return result


def _arange_inclusive(start: float, stop: float, step: float) -> list[float]:
    """np.arange that reliably includes *stop* within floating-point tolerance."""
    vals = []
    v = start
    while v <= stop + step * 0.01:
        vals.append(round(v, 6))
        v += step
    return vals
