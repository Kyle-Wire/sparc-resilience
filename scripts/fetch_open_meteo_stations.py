"""
fetch_open_meteo_stations.py

Discover nearby ASOS weather stations for a SPARC city study area, fetch their
historical temperature observations for the campaign date, and compute UHI anomaly
relative to ERA5 background.

Data sources:
  - Station discovery: NOAA NWS gridpoints/stations API (free, no key)
  - Observations:      Iowa Environmental Mesonet ASOS archive (free, no key)
  - ERA5 background:   Open-Meteo historical archive API (free, no key)

Usage:
    python scripts/fetch_open_meteo_stations.py --city-slug philadelphia_pa
    python scripts/fetch_open_meteo_stations.py --city-slug atlanta_ga --buffer-km 75
    python scripts/fetch_open_meteo_stations.py --city-slug chicago_il --matern-range-m 5000

Outputs:
    output/cities/{city_slug}/stations.parquet

Output columns:
    station_id          ASOS station ID (e.g. "PHL")
    station_name        Human-readable name
    lat, lon            WGS84 decimal degrees
    easting, northing   UTM metres (city's EPSG)
    in_bbox             bool: True if station is inside study bbox
    dist_to_bbox_m      Distance to nearest bbox edge (0.0 if inside)
    spatial_weight      Matérn-decay weight (1.0 in bbox, exp-decay outside)
    campaign_date       ISO date string from city data
    window              "morning" | "midday" | "evening"
    obs_temp_f          Observed temperature at window hour (°F); NaN if unavailable
    era5_temp_f         ERA5 temperature at station lat/lon from Open-Meteo (°F)
    uhi_anomaly_f       obs_temp_f - era5_temp_f (NaN if obs unavailable)
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
import urllib.error
import urllib.request
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

_WINDOW_HOUR = {"morning": 7.0, "midday": 14.0, "evening": 19.0}

# State code lookup from city slug suffix (e.g. "philadelphia_pa" → "PA")
_SLUG_TO_TIMEZONE = {
    "pa": "America/New_York",
    "nc": "America/New_York",
    "ga": "America/New_York",
    "vt": "America/New_York",
    "il": "America/Chicago",
    "la": "America/Chicago",
    "wa": "America/Los_Angeles",
    "nm": "America/Denver",
    "co": "America/Denver",
    "tx": "America/Chicago",
}


def _http_get(url: str, *, retries: int = 3, backoff: float = 2.0) -> str:
    """GET a URL with retry + exponential backoff. Returns decoded body."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "sparc-resilience/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode()
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                wait = backoff ** attempt
                log.warning("Rate-limited (429), sleeping %.1f s …", wait)
                time.sleep(wait)
            else:
                raise
        except urllib.error.URLError as e:
            if attempt < retries - 1:
                time.sleep(backoff)
            else:
                raise
    raise RuntimeError("Unreachable")


def discover_stations_nws(lat: float, lon: float) -> list[dict]:
    """Return NWS ASOS stations for the WFO grid cell containing (lat, lon).

    Each dict has keys: station_id, station_name, lat, lon.
    """
    # Step 1: resolve WFO / grid coordinates
    try:
        body = _http_get(f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}")
        props = json.loads(body).get("properties", {})
        wfo = props["gridId"]
        gx = props["gridX"]
        gy = props["gridY"]
    except Exception as e:
        log.error("NWS points API failed: %s", e)
        return []

    # Step 2: list stations for that WFO grid cell
    try:
        body = _http_get(
            f"https://api.weather.gov/gridpoints/{wfo}/{gx},{gy}/stations"
        )
        features = json.loads(body).get("features", [])
    except Exception as e:
        log.error("NWS stations API failed: %s", e)
        return []

    stations = []
    for f in features:
        p = f.get("properties", {})
        coords = f.get("geometry", {}).get("coordinates", [None, None])
        if coords[0] is None or coords[1] is None:
            continue
        stations.append(
            {
                "station_id": p.get("stationIdentifier", ""),
                "station_name": p.get("name", ""),
                "lat": float(coords[1]),
                "lon": float(coords[0]),
            }
        )
    log.info("NWS discovery: %d stations found near (%.4f, %.4f)", len(stations), lat, lon)
    return stations


def _iem_station_id(nws_id: str) -> str:
    """Convert NWS ICAO-style ID (KPHL) to IEM ID (PHL)."""
    if len(nws_id) == 4 and nws_id.startswith("K"):
        return nws_id[1:]
    return nws_id


def fetch_iem_obs(station_id: str, date_str: str, timezone: str) -> dict[str, float]:
    """Fetch hourly METAR observations from IEM ASOS for one station and date.

    Parameters
    ----------
    station_id : 3 or 4-char ASOS ID (K-prefix is stripped automatically)
    date_str   : "YYYY-MM-DD"
    timezone   : IANA timezone string, e.g. "America/New_York"

    Returns
    -------
    dict mapping hour (float 0-23) → temperature_f (float), empty if unavailable.
    """
    iem_id = _iem_station_id(station_id)
    y, m, d = date_str.split("-")
    tz_enc = timezone.replace("/", "%2F")

    url = (
        f"https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
        f"?station={iem_id}&data=tmpf"
        f"&year1={y}&month1={m}&day1={d}"
        f"&year2={y}&month2={m}&day2={d}"
        f"&tz={tz_enc}&format=onlycomma&latlon=no&missing=M&report_type=3"
    )

    try:
        time.sleep(0.3)  # gentle rate limiting
        raw = _http_get(url)
    except Exception as e:
        log.warning("IEM fetch failed for %s on %s: %s", iem_id, date_str, e)
        return {}

    obs: dict[str, float] = {}
    lines = [l for l in raw.strip().splitlines() if not l.startswith("#")]
    if len(lines) < 2:
        return obs
    # header: station,valid,tmpf
    for line in lines[1:]:
        parts = line.strip().split(",")
        if len(parts) < 3:
            continue
        valid_str = parts[1].strip()  # e.g. "2022-09-21 07:54"
        tmpf_str = parts[2].strip()
        if tmpf_str == "M" or not tmpf_str:
            continue
        try:
            # Parse HH:MM from valid_str
            time_part = valid_str.split(" ")[-1]  # "07:54"
            hh, mm = time_part.split(":")
            hour_frac = int(hh) + int(mm) / 60.0
            obs[f"{hour_frac:.4f}"] = float(tmpf_str)
        except (ValueError, IndexError):
            continue

    return obs


def closest_obs(obs: dict[str, float], target_hour: float) -> Optional[float]:
    """Return the observed temperature closest in time to target_hour."""
    if not obs:
        return None
    best_key = min(obs.keys(), key=lambda k: abs(float(k) - target_hour))
    diff = abs(float(best_key) - target_hour)
    if diff > 2.0:  # skip if more than 2 hours away
        return None
    return obs[best_key]


def fetch_era5_at_point(lat: float, lon: float, date_str: str, timezone: str) -> dict[int, float]:
    """Fetch hourly ERA5 temperature (°F) at a lat/lon point from Open-Meteo.

    Returns dict: {hour_int: temp_f} for all 24 hours.
    """
    tz_enc = timezone.replace("/", "%2F")
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat:.4f}&longitude={lon:.4f}"
        f"&start_date={date_str}&end_date={date_str}"
        f"&hourly=temperature_2m&temperature_unit=fahrenheit"
        f"&timezone={tz_enc}&format=json"
    )
    try:
        time.sleep(0.2)
        raw = _http_get(url)
        data = json.loads(raw)
        times = data["hourly"]["time"]      # ["2022-09-21T00:00", ...]
        temps = data["hourly"]["temperature_2m"]  # [°F, ...]
        result: dict[int, float] = {}
        for t_str, temp in zip(times, temps):
            hour = int(t_str.split("T")[1][:2])
            if temp is not None:
                result[hour] = float(temp)
        return result
    except Exception as e:
        log.warning("Open-Meteo ERA5 fetch failed at (%.4f, %.4f): %s", lat, lon, e)
        return {}


def _dist_to_bbox_m(
    easting: float,
    northing: float,
    bbox_min_e: float,
    bbox_min_n: float,
    bbox_max_e: float,
    bbox_max_n: float,
) -> float:
    """Distance in metres from a UTM point to the nearest bbox edge.

    Returns 0.0 if the point is inside the bbox.
    """
    dx = max(bbox_min_e - easting, 0.0, easting - bbox_max_e)
    dy = max(bbox_min_n - northing, 0.0, northing - bbox_max_n)
    return math.sqrt(dx * dx + dy * dy)


def _matern32_weight(dist_m: float, range_m: float) -> float:
    """Matérn-3/2 spatial weight: 1.0 at dist=0, decays with distance."""
    sqrt3 = math.sqrt(3.0)
    scaled = sqrt3 * dist_m / range_m
    return (1.0 + scaled) * math.exp(-scaled)


def process_city(
    city_slug: str,
    output_dir: str,
    buffer_km: float,
    matern_range_m: float,
) -> None:
    """Full pipeline for one city: discover stations, fetch obs + ERA5, save parquet."""

    # ── Load city geoparquet ──────────────────────────────────────────────────
    data_path = os.path.join(output_dir, "cities", city_slug, "data.geoparquet")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"City data not found: {data_path}")

    log.info("Loading %s …", data_path)
    gdf = gpd.read_parquet(data_path)
    utm_epsg = int(str(gdf.crs.to_epsg()))
    campaign_date = str(gdf["campaign_date"].iloc[0])

    # Bbox in UTM metres
    bbox = gdf.total_bounds  # (minx, miny, maxx, maxy) in UTM
    buffer_m = buffer_km * 1000.0
    bbox_buf = (
        bbox[0] - buffer_m,
        bbox[1] - buffer_m,
        bbox[2] + buffer_m,
        bbox[3] + buffer_m,
    )

    # City centroid in WGS84
    to_wgs84 = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)

    cx_utm = (bbox[0] + bbox[2]) / 2
    cy_utm = (bbox[1] + bbox[3]) / 2
    cx_lon, cx_lat = to_wgs84.transform(cx_utm, cy_utm)

    # Timezone from city slug
    state_code = city_slug.split("_")[-1].lower()
    timezone = _SLUG_TO_TIMEZONE.get(state_code, "America/New_York")
    log.info("City: %s | date: %s | timezone: %s | utm_epsg: %d",
             city_slug, campaign_date, timezone, utm_epsg)
    log.info("Bbox UTM: %.0f %.0f %.0f %.0f", *bbox)
    log.info("City centroid WGS84: lat=%.4f lon=%.4f", cx_lat, cx_lon)

    # ── Compute city-mean ERA5 background per window ──────────────────────────
    era5_city_mean: dict[str, float] = {}
    for w in ("morning", "midday", "evening"):
        col = f"era5_{w}_t2m"
        if col in gdf.columns:
            mean_c = float(gdf[col].mean())
            era5_city_mean[w] = mean_c * 9 / 5 + 32  # Celsius → Fahrenheit
    log.info("City ERA5 means (°F): morning=%.1f midday=%.1f evening=%.1f",
             era5_city_mean.get("morning", float("nan")),
             era5_city_mean.get("midday", float("nan")),
             era5_city_mean.get("evening", float("nan")))

    # ── Discover stations ─────────────────────────────────────────────────────
    log.info("Discovering NWS stations …")
    all_stations = discover_stations_nws(cx_lat, cx_lon)
    if not all_stations:
        log.error("No stations found — aborting")
        return

    # Project stations to UTM, filter by buffered bbox
    filtered = []
    for s in all_stations:
        e, n = to_utm.transform(s["lon"], s["lat"])
        if bbox_buf[0] <= e <= bbox_buf[2] and bbox_buf[1] <= n <= bbox_buf[3]:
            dist = _dist_to_bbox_m(e, n, *bbox)
            in_bbox = dist == 0.0
            weight = 1.0 if in_bbox else _matern32_weight(dist, matern_range_m)
            filtered.append({**s, "easting": e, "northing": n,
                              "in_bbox": in_bbox, "dist_to_bbox_m": dist,
                              "spatial_weight": weight})

    log.info("Stations within %.0f km buffer: %d / %d",
             buffer_km, len(filtered), len(all_stations))

    # ── Fetch observations and ERA5 for each station ──────────────────────────
    rows = []
    for i, s in enumerate(filtered):
        sid = s["station_id"]
        log.info("[%d/%d] %s (%s) lat=%.4f lon=%.4f weight=%.3f",
                 i + 1, len(filtered), sid, s["station_name"],
                 s["lat"], s["lon"], s["spatial_weight"])

        # IEM observations
        obs = fetch_iem_obs(sid, campaign_date, timezone)
        n_obs = len(obs)
        log.info("  IEM: %d hourly observations", n_obs)

        # Open-Meteo ERA5 background at station location
        era5_hourly = fetch_era5_at_point(s["lat"], s["lon"], campaign_date, timezone)

        # One row per window
        for window, window_hour in _WINDOW_HOUR.items():
            obs_temp = closest_obs(obs, window_hour)
            era5_temp = era5_hourly.get(int(round(window_hour)))

            uhi = (obs_temp - era5_temp) if (obs_temp is not None and era5_temp is not None) else None
            uhi_vs_city = (
                (obs_temp - era5_city_mean[window])
                if (obs_temp is not None and window in era5_city_mean) else None
            )

            rows.append({
                "station_id": sid,
                "station_name": s["station_name"],
                "lat": s["lat"],
                "lon": s["lon"],
                "easting": s["easting"],
                "northing": s["northing"],
                "utm_epsg": utm_epsg,
                "in_bbox": s["in_bbox"],
                "dist_to_bbox_m": s["dist_to_bbox_m"],
                "spatial_weight": s["spatial_weight"],
                "campaign_date": campaign_date,
                "window": window,
                "obs_temp_f": obs_temp,
                "era5_temp_f": era5_temp,
                "uhi_anomaly_f": uhi,
                "uhi_vs_city_mean_f": uhi_vs_city,
            })

    if not rows:
        log.warning("No station rows produced — check API connectivity")
        return

    # ── Save output ───────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    out_dir = os.path.join(output_dir, "cities", city_slug)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "stations.parquet")
    df.to_parquet(out_path, index=False)
    log.info("Saved → %s  (%d rows, %d stations)", out_path, len(df), df["station_id"].nunique())

    # Summary
    for w in ("morning", "midday", "evening"):
        wdf = df[df["window"] == w]
        obs_valid = wdf["obs_temp_f"].notna()
        n_valid = obs_valid.sum()
        if n_valid > 0:
            mean_uhi = wdf.loc[obs_valid, "uhi_anomaly_f"].mean()
            log.info("  %s: %d/%d stations have obs | mean UHI=%.2f°F",
                     w, n_valid, len(wdf), mean_uhi if not math.isnan(mean_uhi) else 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch ASOS station obs + ERA5 for SPARC city")
    parser.add_argument("--city-slug", required=True,
                        help="City slug, e.g. philadelphia_pa")
    parser.add_argument("--output-dir", default="output",
                        help="Root output directory (default: output)")
    parser.add_argument("--buffer-km", type=float, default=50.0,
                        help="Buffer in km beyond bbox for station search (default: 50)")
    parser.add_argument("--matern-range-m", type=float, default=6266.0,
                        help="Matérn-3/2 effective range in metres for outside-bbox stations "
                             "(default: 6266 from Stage 0 correlograms)")
    args = parser.parse_args()

    process_city(
        city_slug=args.city_slug,
        output_dir=args.output_dir,
        buffer_km=args.buffer_km,
        matern_range_m=args.matern_range_m,
    )


if __name__ == "__main__":
    main()
