"""Census ACS / TIGER auto-fetch for equity layer context.

V1 scope: given a project bounding box (minlon, minlat, maxlon, maxlat)
or a list of points, look up the covering counties via the Census
Geocoder and aggregate ACS 5-year demographics. No GIS dependencies
required — pure stdlib HTTP. Falls back gracefully when offline.

Why county-level (not tract): the geocoder returns one county per
point; tract-level joins need TIGER polygon downloads + spatial join,
which is heavier than a v1 needs. County rates (poverty, median income,
SVI proxy) are still meaningful equity context.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

ACS_VINTAGE = 2022                         # ACS 5-year ending year
ACS_BASE = f"https://api.census.gov/data/{ACS_VINTAGE}/acs/acs5"
GEOCODER_URL = "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
GEOCODER_BENCHMARK = "Public_AR_Current"
GEOCODER_VINTAGE = "Current_Current"
HTTP_TIMEOUT = 10.0

# ACS variable codes — population, income, poverty, housing.
_ACS_VARS = {
    "population":    "B01003_001E",
    "median_income": "B19013_001E",
    "poverty_count": "B17001_002E",
    "poverty_total": "B17001_001E",   # denominator for poverty rate
    "mobile_homes":  "B25024_010E",
    "housing_total": "B25024_001E",   # denominator for mobile-home share
}


@dataclass
class CensusContext:
    """Area-wide ACS demographics for the project bounding box."""

    counties: list[dict]       # [{state_fips, county_fips, name}]
    n_counties: int
    population: Optional[int]
    median_income: Optional[float]    # population-weighted across counties
    poverty_rate: Optional[float]     # 0-1
    mobile_home_share: Optional[float]
    vintage: int
    source: str = "Census ACS 5-year + Census Geocoder"
    notes: str = ""


# ----------------------------------------------------------------------
# Cache
# ----------------------------------------------------------------------

def _cache_dir() -> Path:
    p = Path.home() / ".sparc" / "cache" / "census"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cache_get(key: str) -> Optional[dict]:
    fp = _cache_dir() / f"{key}.json"
    if not fp.exists():
        return None
    try:
        with open(fp, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _cache_put(key: str, value: dict) -> None:
    try:
        with open(_cache_dir() / f"{key}.json", "w", encoding="utf-8") as fh:
            json.dump(value, fh)
    except OSError:
        pass


# ----------------------------------------------------------------------
# HTTP helpers
# ----------------------------------------------------------------------

def _http_json(url: str, *, timeout: float = HTTP_TIMEOUT) -> Optional[object]:
    """GET a JSON URL with a short timeout. Returns None on any failure."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "sparc-resilience/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (intentional HTTPS GET)
            data = resp.read().decode("utf-8")
        return json.loads(data)
    except Exception:
        return None


# ----------------------------------------------------------------------
# Geocoder: lat/lon -> {state, county}
# ----------------------------------------------------------------------

def lookup_county(lat: float, lon: float) -> Optional[dict]:
    """Return ``{state_fips, county_fips, county_name}`` for a coordinate."""
    cache_key = f"county_{lat:.4f}_{lon:.4f}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached or None

    qs = urllib.parse.urlencode({
        "x": lon,
        "y": lat,
        "benchmark": GEOCODER_BENCHMARK,
        "vintage": GEOCODER_VINTAGE,
        "layers": "Counties",
        "format": "json",
    })
    payload = _http_json(f"{GEOCODER_URL}?{qs}")
    if not isinstance(payload, dict):
        return None
    try:
        counties = payload["result"]["geographies"]["Counties"]
    except (KeyError, TypeError):
        _cache_put(cache_key, {})
        return None
    if not counties:
        _cache_put(cache_key, {})
        return None
    c = counties[0]
    out = {
        "state_fips": str(c.get("STATE", "")).zfill(2),
        "county_fips": str(c.get("COUNTY", "")).zfill(3),
        "county_name": c.get("NAME") or c.get("BASENAME") or "",
    }
    _cache_put(cache_key, out)
    return out


# ----------------------------------------------------------------------
# ACS county query
# ----------------------------------------------------------------------

def fetch_acs_counties(state_fips: str, county_fips_list: Iterable[str]) -> list[dict]:
    """Fetch ACS variables for one state's counties. Returns rows of
    ``{county_fips, county_name, **vars}``.
    """
    counties = sorted(set(county_fips_list))
    if not counties:
        return []
    cache_key = f"acs_{state_fips}_{'-'.join(counties)}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached.get("rows", [])

    vars_csv = ",".join(["NAME", *_ACS_VARS.values()])
    qs = urllib.parse.urlencode({
        "get": vars_csv,
        "for": f"county:{','.join(counties)}",
        "in":  f"state:{state_fips}",
    })
    payload = _http_json(f"{ACS_BASE}?{qs}")
    if not isinstance(payload, list) or len(payload) < 2:
        return []
    header, *rows = payload
    name_idx = header.index("NAME")
    var_idx = {alias: header.index(code) for alias, code in _ACS_VARS.items()}
    out = []
    for row in rows:
        county_code = row[header.index("county")]
        rec: dict[str, object] = {
            "county_fips": county_code,
            "county_name": row[name_idx],
        }
        for alias, idx in var_idx.items():
            try:
                rec[alias] = float(row[idx]) if row[idx] not in (None, "") else None
                if rec[alias] is not None and rec[alias] < 0:  # ACS sentinel
                    rec[alias] = None
            except (TypeError, ValueError):
                rec[alias] = None
        out.append(rec)
    _cache_put(cache_key, {"rows": out})
    return out


# ----------------------------------------------------------------------
# Top-level: bbox -> CensusContext
# ----------------------------------------------------------------------

def _bbox_sample_points(minlon: float, minlat: float,
                        maxlon: float, maxlat: float) -> list[tuple[float, float]]:
    """Sample 5 points (4 corners + center) — enough to catch all counties
    intersecting most project bboxes without flooding the geocoder."""
    cx = 0.5 * (minlon + maxlon)
    cy = 0.5 * (minlat + maxlat)
    return [
        (minlat, minlon),
        (minlat, maxlon),
        (maxlat, minlon),
        (maxlat, maxlon),
        (cy, cx),
    ]


def fetch_area_demographics(minlon: float, minlat: float,
                            maxlon: float, maxlat: float) -> CensusContext:
    """Resolve which US counties the bbox touches and aggregate ACS stats."""

    # 1. Find counties via geocoder (per-corner + center).
    counties_seen: dict[tuple[str, str], dict] = {}
    for lat, lon in _bbox_sample_points(minlon, minlat, maxlon, maxlat):
        c = lookup_county(lat, lon)
        if c and c.get("state_fips") and c.get("county_fips"):
            counties_seen[(c["state_fips"], c["county_fips"])] = c

    if not counties_seen:
        return CensusContext(
            counties=[], n_counties=0, population=None, median_income=None,
            poverty_rate=None, mobile_home_share=None, vintage=ACS_VINTAGE,
            notes="Census Geocoder returned no counties (offline, or bbox outside the US?).",
        )

    # 2. Group by state, fetch ACS in one call per state.
    by_state: dict[str, list[str]] = {}
    for st, co in counties_seen.keys():
        by_state.setdefault(st, []).append(co)

    rows: list[dict] = []
    for st, cos in by_state.items():
        rows.extend({**r, "state_fips": st} for r in fetch_acs_counties(st, cos))

    if not rows:
        return CensusContext(
            counties=list(counties_seen.values()),
            n_counties=len(counties_seen),
            population=None, median_income=None, poverty_rate=None,
            mobile_home_share=None, vintage=ACS_VINTAGE,
            notes="ACS query failed — counties identified but no data returned.",
        )

    # 3. Aggregate (population-weighted income, summed counts).
    total_pop = 0.0
    inc_num = 0.0
    inc_den = 0.0
    pov_num = 0.0
    pov_den = 0.0
    mh_num = 0.0
    mh_den = 0.0
    for r in rows:
        pop = r.get("population") or 0.0
        total_pop += pop or 0.0
        inc = r.get("median_income")
        if inc is not None and pop:
            inc_num += inc * pop
            inc_den += pop
        pn, pd = r.get("poverty_count"), r.get("poverty_total")
        if pn is not None and pd:
            pov_num += pn
            pov_den += pd
        mn, md = r.get("mobile_homes"), r.get("housing_total")
        if mn is not None and md:
            mh_num += mn
            mh_den += md

    return CensusContext(
        counties=[{"state_fips": r["state_fips"],
                   "county_fips": r["county_fips"],
                   "name": r.get("county_name", "")} for r in rows],
        n_counties=len(rows),
        population=int(total_pop) if total_pop else None,
        median_income=(inc_num / inc_den) if inc_den else None,
        poverty_rate=(pov_num / pov_den) if pov_den else None,
        mobile_home_share=(mh_num / mh_den) if mh_den else None,
        vintage=ACS_VINTAGE,
    )


def context_to_equity_layers(ctx: CensusContext) -> dict:
    """Convert a CensusContext into the layer dict used by
    ``combine_equity_layers``. Uniform per candidate (one area summary)
    — caller decides how to blend with other layers.

    Returns a payload shaped like the body of ``POST /decision/equity``
    (minus candidate_names / weights), with ``invert`` flags pre-set.
    """
    layers: dict[str, float] = {}
    invert: dict[str, bool] = {}
    if ctx.poverty_rate is not None:
        layers["poverty_rate"] = float(ctx.poverty_rate)
    if ctx.mobile_home_share is not None:
        layers["mobile_home_share"] = float(ctx.mobile_home_share)
    if ctx.median_income is not None:
        layers["median_income"] = float(ctx.median_income)
        invert["median_income"] = True   # higher income ⇒ lower vulnerability
    return {
        "layers": layers,
        "invert": invert,
        "context": asdict(ctx),
    }


# ----------------------------------------------------------------------
# Tract-level equity: TIGER spatial join → per-cell equity score
# ----------------------------------------------------------------------

import hashlib
import time
import warnings as _warnings

TIGER_TRACT_URL = (
    "https://tigerweb.geo.census.gov/arcgis/rest/services/"
    "TIGERweb/tigerWMS_Current/MapServer/8/query"
)
EQUITY_CACHE_TTL_DAYS = 7

# ACS variable codes for tract-level query
_TRACT_ACS_VARS = {
    "population":     "B01003_001E",   # total population
    "poverty_count":  "B17001_002E",   # below poverty
    "poverty_total":  "B17001_001E",   # poverty universe
    "nonhisp_white":  "B03002_003E",   # non-Hispanic white alone
}


def _tract_cache_key(bbox: tuple[float, float, float, float]) -> str:
    raw = f"{bbox[0]:.4f},{bbox[1]:.4f},{bbox[2]:.4f},{bbox[3]:.4f}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _is_cache_valid(cache_key: str) -> bool:
    fp = _cache_dir() / f"{cache_key}.json"
    if not fp.exists():
        return False
    age_days = (time.time() - fp.stat().st_mtime) / 86400
    return age_days < EQUITY_CACHE_TTL_DAYS


def _http_json_with_retry(url: str, *, timeout: float = 15.0) -> Optional[object]:
    """GET JSON with a single retry on 429 (rate limit) with 5-second back-off."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "sparc-resilience/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            _warnings.warn(
                "Census API rate limit hit (HTTP 429). Retrying in 5 seconds. "
                "Provide a Census API key in Settings to increase limits.",
                stacklevel=3,
            )
            time.sleep(5)
            try:
                req2 = urllib.request.Request(url, headers={"User-Agent": "sparc-resilience/1.0"})
                with urllib.request.urlopen(req2, timeout=timeout) as resp2:  # noqa: S310
                    return json.loads(resp2.read().decode("utf-8"))
            except Exception:
                return None
        return None
    except Exception:
        return None


def _fetch_tiger_tracts(
    bbox: tuple[float, float, float, float],
) -> list[dict]:
    """Fetch Census tract geometries (simplified polygon centroids) within bbox.

    Returns a list of ``{tract_geoid, state_fips, county_fips, tract_code,
    centroid_lon, centroid_lat}``. Uses the TIGER WMS MapServer query endpoint.
    Because the full polygon join requires geopandas, we fetch tract centroids
    and use point-in-bounding-box logic to assign project grid cells.
    """
    minlon, minlat, maxlon, maxlat = bbox
    # ArcGIS REST: geometry is xmin,ymin,xmax,ymax in WGS84 (EPSG:4326 = 4326)
    geom_str = f"{minlon},{minlat},{maxlon},{maxlat}"
    qs = urllib.parse.urlencode({
        "where": "1=1",
        "geometry": geom_str,
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "GEOID,STATE,COUNTY,TRACT,CENTLAT,CENTLON",
        "returnGeometry": "false",
        "outSR": "4326",
        "f": "json",
        "resultRecordCount": "2000",
    })
    payload = _http_json_with_retry(f"{TIGER_TRACT_URL}?{qs}")
    if not isinstance(payload, dict):
        return []
    features = payload.get("features") or []
    tracts = []
    for feat in features:
        attrs = feat.get("attributes") or {}
        geoid = str(attrs.get("GEOID") or "")
        if not geoid:
            continue
        try:
            clat = float(attrs.get("CENTLAT") or 0.0)
            clon = float(attrs.get("CENTLON") or 0.0)
        except (TypeError, ValueError):
            clat, clon = 0.0, 0.0
        tracts.append({
            "tract_geoid": geoid,
            "state_fips": str(attrs.get("STATE") or "").zfill(2),
            "county_fips": str(attrs.get("COUNTY") or "").zfill(3),
            "tract_code": str(attrs.get("TRACT") or ""),
            "centroid_lat": clat,
            "centroid_lon": clon,
        })
    return tracts


def _fetch_acs_tracts(
    state_fips: str,
    county_fips: str,
    census_api_key: Optional[str] = None,
) -> list[dict]:
    """Fetch ACS 5-year tract-level demographics for one county.

    Returns rows of ``{tract_code, poverty_rate, minority_pct}``.
    """
    cache_key = f"acs_tract_{state_fips}_{county_fips}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached.get("rows", [])

    vars_csv = ",".join(["NAME", *_TRACT_ACS_VARS.values()])
    params: dict[str, str] = {
        "get": vars_csv,
        "for": "tract:*",
        "in": f"state:{state_fips} county:{county_fips}",
    }
    if census_api_key:
        params["key"] = census_api_key
    qs = urllib.parse.urlencode(params)
    payload = _http_json_with_retry(f"{ACS_BASE}?{qs}")
    if not isinstance(payload, list) or len(payload) < 2:
        return []

    header, *rows = payload
    try:
        idx = {name: header.index(code) for name, code in _TRACT_ACS_VARS.items()}
        tract_idx = header.index("tract")
    except ValueError:
        return []

    out = []
    for row in rows:
        def _safe(name: str) -> Optional[float]:
            try:
                v = float(row[idx[name]])
                return v if v >= 0 else None
            except (TypeError, ValueError, IndexError):
                return None

        pop = _safe("population") or 0.0
        pov_c = _safe("poverty_count")
        pov_t = _safe("poverty_total") or pop or 1.0
        nhw = _safe("nonhisp_white")

        poverty_rate = (pov_c / pov_t) if pov_c is not None else None
        minority_pct = (1.0 - nhw / pop) if (nhw is not None and pop > 0) else None

        out.append({
            "tract_code": str(row[tract_idx]),
            "poverty_rate": poverty_rate,
            "minority_pct": minority_pct,
        })

    _cache_put(cache_key, {"rows": out})
    return out


def fetch_tract_equity(
    bbox: tuple[float, float, float, float],
    census_api_key: Optional[str] = None,
) -> list[dict]:
    """Fetch tract-level equity scores for a bounding box.

    Returns a list of dicts, one per tract, with keys:
    ``tract_geoid``, ``centroid_lat``, ``centroid_lon``,
    ``poverty_rate``, ``minority_pct``, ``equity_score``.

    ``equity_score`` = 0.5 × poverty_rate + 0.5 × minority_pct,
    min-max normalised across returned tracts to [0, 1].
    """
    cache_key = f"tract_equity_{_tract_cache_key(bbox)}"
    if _is_cache_valid(cache_key):
        cached = _cache_get(cache_key)
        if cached and cached.get("tracts"):
            return cached["tracts"]

    tracts = _fetch_tiger_tracts(bbox)
    if not tracts:
        return []

    # Group tracts by state+county, fetch ACS once per county
    county_acs: dict[tuple[str, str], dict[str, dict]] = {}
    for t in tracts:
        key = (t["state_fips"], t["county_fips"])
        if key not in county_acs:
            rows = _fetch_acs_tracts(t["state_fips"], t["county_fips"], census_api_key)
            county_acs[key] = {r["tract_code"]: r for r in rows}

    # Merge ACS data into tract records
    result = []
    for t in tracts:
        acs = county_acs.get((t["state_fips"], t["county_fips"]), {}).get(t["tract_code"], {})
        pov = acs.get("poverty_rate")
        min_pct = acs.get("minority_pct")
        # Raw composite (before normalisation)
        if pov is not None and min_pct is not None:
            raw_score = 0.5 * pov + 0.5 * min_pct
        elif pov is not None:
            raw_score = pov
        elif min_pct is not None:
            raw_score = min_pct
        else:
            raw_score = None
        result.append({
            "tract_geoid": t["tract_geoid"],
            "centroid_lat": t["centroid_lat"],
            "centroid_lon": t["centroid_lon"],
            "poverty_rate": pov,
            "minority_pct": min_pct,
            "_raw_score": raw_score,
        })

    # Min-max normalise equity_score across tracts that have data
    raw_scores = [r["_raw_score"] for r in result if r["_raw_score"] is not None]
    lo = min(raw_scores) if raw_scores else 0.0
    hi = max(raw_scores) if raw_scores else 1.0
    span = hi - lo if hi > lo else 1.0

    for r in result:
        rs = r.pop("_raw_score")
        r["equity_score"] = float((rs - lo) / span) if rs is not None else 0.5

    _cache_put(cache_key, {"tracts": result})
    return result


def _assign_tract_to_cells(
    cell_lats: list[float],
    cell_lons: list[float],
    tract_records: list[dict],
) -> list[str]:
    """Nearest-centroid assignment of each grid cell to a tract geoid.

    Pure stdlib — no geopandas required. Uses squared Euclidean distance
    in lon/lat space (acceptable for the sub-county distances involved).
    """
    if not tract_records:
        return [""] * len(cell_lats)

    import math

    t_lats = [t["centroid_lat"] for t in tract_records]
    t_lons = [t["centroid_lon"] for t in tract_records]
    t_ids = [t["tract_geoid"] for t in tract_records]

    result = []
    for clat, clon in zip(cell_lats, cell_lons):
        best_idx = 0
        best_dist = math.inf
        for i, (tlat, tlon) in enumerate(zip(t_lats, t_lons)):
            d = (clat - tlat) ** 2 + (clon - tlon) ** 2
            if d < best_dist:
                best_dist = d
                best_idx = i
        result.append(t_ids[best_idx])
    return result


def get_per_cell_equity(
    data_df: "pd.DataFrame",  # type: ignore[name-defined]
    census_api_key: Optional[str] = None,
) -> "np.ndarray":  # type: ignore[name-defined]
    """Return a per-cell equity score array (0–1) aligned to ``data_df`` rows.

    Derives lat/lon from the DataFrame's coordinate columns (tries common
    names: lat/lon, latitude/longitude, y/x, coord_y/coord_x). Falls back to
    uniform 0.5 if coordinates cannot be resolved.

    Parameters
    ----------
    data_df
        The project DataFrame (one row per grid cell).
    census_api_key
        Optional Census API key for higher rate limits.

    Returns
    -------
    np.ndarray of shape (n_cells,) with dtype float64, values in [0, 1].
    """
    import numpy as np

    n = len(data_df)
    fallback = np.full(n, 0.5, dtype=float)

    # Detect lat/lon columns
    _LAT_CANDIDATES = ["lat", "latitude", "LAT", "LATITUDE", "y", "Y", "coord_y", "COORD_Y"]
    _LON_CANDIDATES = ["lon", "lng", "longitude", "LON", "LNG", "LONGITUDE", "x", "X", "coord_x", "COORD_X"]
    lat_col = next((c for c in _LAT_CANDIDATES if c in data_df.columns), None)
    lon_col = next((c for c in _LON_CANDIDATES if c in data_df.columns), None)
    if lat_col is None or lon_col is None:
        _warnings.warn(
            "Cannot determine lat/lon columns for Census equity fetch. "
            f"Tried: {_LAT_CANDIDATES}, {_LON_CANDIDATES}. "
            "Using uniform equity_score=0.5.",
        )
        return fallback

    try:
        lats = data_df[lat_col].to_numpy(dtype=float)
        lons = data_df[lon_col].to_numpy(dtype=float)
    except Exception:
        return fallback

    finite = np.isfinite(lats) & np.isfinite(lons)
    if not finite.any():
        return fallback

    bbox = (
        float(lons[finite].min()),
        float(lats[finite].min()),
        float(lons[finite].max()),
        float(lats[finite].max()),
    )

    tracts = fetch_tract_equity(bbox, census_api_key=census_api_key)
    if not tracts:
        _warnings.warn(
            "Census TIGER returned no tract data for the project bounding box. "
            "Using uniform equity_score=0.5.",
        )
        return fallback

    # Build tract_geoid → equity_score lookup
    tract_lookup: dict[str, float] = {t["tract_geoid"]: t["equity_score"] for t in tracts}

    # Assign each cell to nearest tract centroid
    assigned = _assign_tract_to_cells(
        lats.tolist(), lons.tolist(), tracts
    )

    equity_arr = np.array(
        [tract_lookup.get(gid, 0.5) for gid in assigned],
        dtype=float,
    )
    return equity_arr


def get_per_cell_equity_df(
    data_df: "pd.DataFrame",  # type: ignore[name-defined]
    census_api_key: Optional[str] = None,
) -> "pd.DataFrame":  # type: ignore[name-defined]
    """Same as ``get_per_cell_equity`` but returns a DataFrame with full columns.

    Columns: ``cell_id``, ``equity_score``, ``poverty_rate``, ``minority_pct``,
    ``tract_geoid``.
    """
    import numpy as np
    import pandas as pd

    n = len(data_df)
    scores = get_per_cell_equity(data_df, census_api_key=census_api_key)

    # Rebuild tract assignment for metadata columns
    _LAT_CANDIDATES = ["lat", "latitude", "LAT", "LATITUDE", "y", "Y", "coord_y"]
    _LON_CANDIDATES = ["lon", "lng", "longitude", "LON", "LNG", "LONGITUDE", "x", "X", "coord_x"]
    lat_col = next((c for c in _LAT_CANDIDATES if c in data_df.columns), None)
    lon_col = next((c for c in _LON_CANDIDATES if c in data_df.columns), None)

    tract_geoids = [""] * n
    poverty_rates: list[Optional[float]] = [None] * n
    minority_pcts: list[Optional[float]] = [None] * n

    if lat_col and lon_col:
        try:
            lats = data_df[lat_col].to_numpy(dtype=float)
            lons = data_df[lon_col].to_numpy(dtype=float)
            finite = np.isfinite(lats) & np.isfinite(lons)
            if finite.any():
                bbox = (
                    float(lons[finite].min()), float(lats[finite].min()),
                    float(lons[finite].max()), float(lats[finite].max()),
                )
                tracts = fetch_tract_equity(bbox, census_api_key=census_api_key)
                if tracts:
                    tract_lookup_meta = {
                        t["tract_geoid"]: t for t in tracts
                    }
                    assigned = _assign_tract_to_cells(lats.tolist(), lons.tolist(), tracts)
                    for i, gid in enumerate(assigned):
                        tract_geoids[i] = gid
                        meta = tract_lookup_meta.get(gid, {})
                        poverty_rates[i] = meta.get("poverty_rate")
                        minority_pcts[i] = meta.get("minority_pct")
        except Exception:
            pass

    cell_ids = list(range(n))
    if "cell_id" in data_df.columns:
        cell_ids = data_df["cell_id"].tolist()

    return pd.DataFrame({
        "cell_id": cell_ids,
        "equity_score": scores,
        "poverty_rate": poverty_rates,
        "minority_pct": minority_pcts,
        "tract_geoid": tract_geoids,
    })
