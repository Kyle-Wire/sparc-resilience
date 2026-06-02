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

def download_era5_boundary(
    bbox: tuple[float, float, float, float],
    campaign_date: "date",
) -> tuple[list[float], list[float], dict]:
    """Fetch ERA5 hourly boundary conditions at 3 time windows on the campaign date.

    Fetches ``temperature_2m``, ``windspeed_10m``, ``winddirection_10m``, and
    ``relativehumidity_2m`` at ERA5 native grid points covering *bbox* for
    morning (~6 am), midday (~1 pm), and evening (~8 pm) local solar time.

    UTC hours are approximated from the bbox centroid longitude::

        utc_offset = round(mean(minx, maxx) / 15)
        morning_utc = (6  - utc_offset) % 24
        midday_utc  = (13 - utc_offset) % 24
        evening_utc = (20 - utc_offset) % 24

    Parameters
    ----------
    bbox : (minx, miny, maxx, maxy) in EPSG:4326
    campaign_date : date
        The CAPA campaign date.  Only data from this single day is fetched.

    Returns
    -------
    (grid_lons, grid_lats, boundary_data)
        ``boundary_data`` maps ``(lon, lat)`` to a dict with keys
        ``"morning"``, ``"midday"``, ``"evening"``, each holding a sub-dict
        ``{"t2m": float, "windspeed": float, "winddir": float, "rh": float}``.
    """
    minx, miny, maxx, maxy = bbox
    centroid_lon = (minx + maxx) / 2.0
    utc_offset = round(centroid_lon / 15.0)
    morning_utc = (6  - utc_offset) % 24
    midday_utc  = (13 - utc_offset) % 24
    evening_utc = (20 - utc_offset) % 24

    grid_lons, grid_lats = _era5_grid_points_in_bbox(bbox)

    boundary_data: dict = {}
    for lon in grid_lons:
        for lat in grid_lats:
            point_data = _fetch_boundary_point(
                lon, lat, campaign_date,
                morning_utc, midday_utc, evening_utc,
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

    Appends 12 columns to *fishnet_gdf*::

        era5_morning_t2m,  era5_morning_windspeed,  era5_morning_winddir,  era5_morning_rh,
        era5_midday_t2m,   era5_midday_windspeed,   era5_midday_winddir,   era5_midday_rh,
        era5_evening_t2m,  era5_evening_windspeed,  era5_evening_winddir,  era5_evening_rh

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
        "era5_morning_t2m", "era5_morning_windspeed",
        "era5_morning_winddir", "era5_morning_rh",
        "era5_midday_t2m",  "era5_midday_windspeed",
        "era5_midday_winddir",  "era5_midday_rh",
        "era5_evening_t2m", "era5_evening_windspeed",
        "era5_evening_winddir", "era5_evening_rh",
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
        variables = ("t2m", "windspeed", "winddir", "rh")

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
    morning_utc: int,
    midday_utc: int,
    evening_utc: int,
) -> dict:
    """Fetch hourly ERA5 boundary variables at one grid point for one day."""
    _NAN_WINDOW = {"t2m": float("nan"), "windspeed": float("nan"),
                   "winddir": float("nan"), "rh": float("nan")}
    result = {
        "morning": dict(_NAN_WINDOW),
        "midday":  dict(_NAN_WINDOW),
        "evening": dict(_NAN_WINDOW),
    }

    date_str = campaign_date.isoformat()
    params = urllib.parse.urlencode({
        "latitude":   f"{lat:.4f}",
        "longitude":  f"{lon:.4f}",
        "start_date": date_str,
        "end_date":   date_str,
        "hourly": "temperature_2m,windspeed_10m,winddirection_10m,relativehumidity_2m",
        "temperature_unit": "celsius",
        "windspeed_unit": "ms",
        "timezone": "UTC",
        "models": "era5",
    })
    url = f"{OPEN_METEO_ARCHIVE}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "SPARC-DataCollection/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return result

    hourly = data.get("hourly", {})
    t2m_arr   = hourly.get("temperature_2m", [])
    ws_arr    = hourly.get("windspeed_10m", [])
    wd_arr    = hourly.get("winddirection_10m", [])
    rh_arr    = hourly.get("relativehumidity_2m", [])

    def _safe_get(arr: list, idx: int):
        if idx < len(arr) and arr[idx] is not None:
            return float(arr[idx])
        return float("nan")

    for window, utc_hour in (
        ("morning", morning_utc),
        ("midday",  midday_utc),
        ("evening", evening_utc),
    ):
        result[window] = {
            "t2m":       _safe_get(t2m_arr, utc_hour),
            "windspeed": _safe_get(ws_arr,  utc_hour),
            "winddir":   _safe_get(wd_arr,  utc_hour),
            "rh":        _safe_get(rh_arr,  utc_hour),
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
