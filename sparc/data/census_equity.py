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
