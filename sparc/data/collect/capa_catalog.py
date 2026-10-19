"""
CAPA Heat Watch city catalog.

Maps city names to their OSF node IDs and metadata for the
NOAA/NIHHIS Urban Heat Island Mapping Campaign (Heat Watch).

Each entry:
  osf_node    — 5-char OSF project ID (https://osf.io/{node_id}/files/)
  folder_hint — For multi-city nodes: partial city name used to filter
                the root-level subfolder.  None for dedicated per-city nodes.
  utm_epsg    — Recommended projected CRS for spatial operations.
  year        — Campaign year (omitted if unknown).
  state       — US state abbreviation.

Optional, where the catalog knows it (see :func:`availability`,
:func:`traverse_node`, :func:`campaign_date`):
  traverse_node        — OSF node holding the raw traverse points, when it
                         differs from or is confirmed apart from ``osf_node``.
  campaign_date        — ISO date of the campaign day.
  campaign_date_source — where that date comes from: "traverse timestamps"
                         (the readings' own clock), "filename_guess" or
                         "era5_matched" (circular for ERA5 features; see
                         configs/multicity_pilot.yml).
  availability         — {"traverse_points": bool | None, "rasters": bool | None}:
                         whether the node is known to hold the raw traverse
                         points (the measurements) and CAPA's interpolated
                         area-wide rasters (a land-cover model of them, not a
                         measurement); None = not known to the catalog.
  availability_source  — where the availability statement comes from.

Multi-city nodes:
  tdsy7  — 2019 NIHHIS campaign (~25 cities in one OSF project)
  3xvmg  — Virginia cities (Abingdon, Arlington, Farmville, Lynchburg, Petersburg)
  j6eqr  — Manhattan + Bronx, NY
  6rwbg  — Jersey City, Newark, Elizabeth, NJ
  ktr56  — San Francisco, CA + Las Vegas/Clark County, NV
  rbd3q  — Iowa City + Cedar Rapids, IA

Add new cities:
  Find the node ID at
    https://heat.gov/urban-heat-islands-mapping-campaign-program/
  or directly at https://osf.io/ (5-char code in the URL).
  Then add the entry here and run: sparc init --list-cities to verify.
"""
from __future__ import annotations

import difflib

# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

CAPA_CATALOG: dict[str, dict] = {

    # ── NEW ENGLAND ────────────────────────────────────────────────────────
    "Burlington, VT": {
        "osf_node": "tezb9", "folder_hint": None,
        "utm_epsg": "EPSG:32618", "state": "VT",
    },
    "Boston, MA": {
        "osf_node": "tdsy7", "folder_hint": "Boston",
        "utm_epsg": "EPSG:32619", "year": 2019, "state": "MA",
    },
    "Brockton, MA": {
        "osf_node": "rk75w", "folder_hint": None,
        "utm_epsg": "EPSG:32619", "year": 2023, "state": "MA",
    },
    "Providence, RI": {
        # 2020 RI Heat Watch (not the 2019 NIHHIS project tdsy7): the traverses are OSF wu9v7,
        # four runs on 2020-07-29 (am/af/pm, plus mi 00-01 on 07-30) dated by the readings' own clock.
        "osf_node": "wu9v7", "folder_hint": None, "traverse_node": "wu9v7",
        "utm_epsg": "EPSG:32619", "year": 2020, "state": "RI",
        "campaign_date": "2020-07-29", "campaign_date_source": "traverse timestamps",
        "availability": {"traverse_points": True, "rasters": None},
        "availability_source": "docs/results/canopy_identification.md (the wu9v7 download: am/af/pm/mi_trav "
                               "point shapefiles); rasters in wu9v7 not verified",
    },

    # ── MID-ATLANTIC ───────────────────────────────────────────────────────
    "New York, NY": {
        "osf_node": "tdsy7", "folder_hint": "New York",
        "utm_epsg": "EPSG:32618", "year": 2019, "state": "NY",
    },
    "Manhattan, NY": {
        "osf_node": "j6eqr", "folder_hint": "Manhattan",
        "utm_epsg": "EPSG:32618", "state": "NY",
    },
    "Bronx, NY": {
        "osf_node": "j6eqr", "folder_hint": "Bronx",
        "utm_epsg": "EPSG:32618", "state": "NY",
    },
    "Brooklyn, NY": {
        "osf_node": "c6xjv", "folder_hint": None,
        "utm_epsg": "EPSG:32618", "state": "NY",
    },
    "Jersey City, NJ": {
        "osf_node": "6rwbg", "folder_hint": "Jersey City",
        "utm_epsg": "EPSG:32618", "state": "NJ",
    },
    "Newark, NJ": {
        "osf_node": "6rwbg", "folder_hint": "Newark",
        "utm_epsg": "EPSG:32618", "state": "NJ",
    },
    "Elizabeth, NJ": {
        "osf_node": "6rwbg", "folder_hint": "Elizabeth",
        "utm_epsg": "EPSG:32618", "state": "NJ",
    },
    "Philadelphia, PA": {
        "osf_node": "3xts6", "folder_hint": None,
        "utm_epsg": "EPSG:32618", "year": 2022, "state": "PA",
    },
    "Scranton, PA": {
        "osf_node": "8cnq4", "folder_hint": "Scranton",
        "utm_epsg": "EPSG:32618", "state": "PA",
    },
    "Wilkes-Barre, PA": {
        "osf_node": "8cnq4", "folder_hint": "Wilkes-Barre",
        "utm_epsg": "EPSG:32618", "state": "PA",
    },
    "Wilmington, DE": {
        "osf_node": "yef5w", "folder_hint": None,
        "utm_epsg": "EPSG:32618", "state": "DE",
    },
    "Baltimore, MD": {
        "osf_node": "tdsy7", "folder_hint": "Baltimore",
        "utm_epsg": "EPSG:32618", "year": 2019, "state": "MD",
    },

    # ── SOUTH ATLANTIC ─────────────────────────────────────────────────────
    "Abingdon, VA": {
        "osf_node": "3xvmg", "folder_hint": "Abingdon",
        "utm_epsg": "EPSG:32617", "state": "VA",
    },
    "Arlington, VA": {
        "osf_node": "3xvmg", "folder_hint": "Arlington",
        "utm_epsg": "EPSG:32618", "state": "VA",
    },
    "Farmville, VA": {
        "osf_node": "3xvmg", "folder_hint": "Farmville",
        "utm_epsg": "EPSG:32617", "state": "VA",
    },
    "Lynchburg, VA": {
        "osf_node": "3xvmg", "folder_hint": "Lynchburg",
        "utm_epsg": "EPSG:32617", "state": "VA",
    },
    "Petersburg, VA": {
        "osf_node": "3xvmg", "folder_hint": "Petersburg",
        "utm_epsg": "EPSG:32618", "state": "VA",
    },
    "Richmond, VA": {
        "osf_node": "3xvmg", "folder_hint": "Richmond",
        "utm_epsg": "EPSG:32618", "year": 2021, "state": "VA",
    },
    "Virginia Beach, VA": {
        "osf_node": "3xvmg", "folder_hint": "Virginia Beach",
        "utm_epsg": "EPSG:32618", "year": 2021, "state": "VA",
    },
    "Salem, VA": {
        "osf_node": "3xvmg", "folder_hint": "Salem",
        "utm_epsg": "EPSG:32617", "year": 2021, "state": "VA",
    },
    "Raleigh, NC": {
        "osf_node": "rwjnx", "folder_hint": None,
        "utm_epsg": "EPSG:32617", "state": "NC",
    },
    "Charlotte, NC": {
        "osf_node": "86ume", "folder_hint": None,
        "utm_epsg": "EPSG:32617", "state": "NC",
    },
    "Asheville, NC": {
        "osf_node": "mtxy5", "folder_hint": None,
        "utm_epsg": "EPSG:32617", "state": "NC",
    },
    "Columbia, SC": {
        "osf_node": "nqwyr", "folder_hint": None,
        "utm_epsg": "EPSG:32617", "state": "SC",
    },
    "Charleston, SC": {
        "osf_node": "b4tfy", "folder_hint": None,
        "utm_epsg": "EPSG:32617", "state": "SC",
    },
    "Atlanta, GA": {
        "osf_node": "r74h2", "folder_hint": None,
        "utm_epsg": "EPSG:32616", "state": "GA",
    },
    "Jacksonville, FL": {
        "osf_node": "xpuhk", "folder_hint": None,
        "utm_epsg": "EPSG:32617", "state": "FL",
    },
    "Fort Lauderdale, FL": {
        "osf_node": "tdsy7", "folder_hint": "Fort Lauderdale",
        "utm_epsg": "EPSG:32617", "year": 2019, "state": "FL",
    },
    "Miami, FL": {
        "osf_node": "tdsy7", "folder_hint": "Miami",
        "utm_epsg": "EPSG:32617", "year": 2019, "state": "FL",
    },

    # ── SOUTH CENTRAL ──────────────────────────────────────────────────────
    "Louisville, KY": {
        "osf_node": "tdsy7", "folder_hint": "Louisville",
        "utm_epsg": "EPSG:32616", "year": 2019, "state": "KY",
    },
    "Jackson, MS": {
        "osf_node": "tdsy7", "folder_hint": "Jackson",
        "utm_epsg": "EPSG:32615", "year": 2019, "state": "MS",
    },
    "New Orleans, LA": {
        "osf_node": "eau82", "folder_hint": None,
        "utm_epsg": "EPSG:32615", "state": "LA",
    },
    "Dallas, TX": {
        "osf_node": "tdsy7", "folder_hint": "Dallas",
        "utm_epsg": "EPSG:32614", "year": 2019, "state": "TX",
    },
    "El Paso, TX": {
        "osf_node": "tdsy7", "folder_hint": "El Paso",
        "utm_epsg": "EPSG:32613", "year": 2019, "state": "TX",
    },
    "Austin, TX": {
        "osf_node": "h8cte", "folder_hint": None,
        "utm_epsg": "EPSG:32614", "state": "TX",
    },
    "Houston, TX": {
        "osf_node": "yqh5u", "folder_hint": None,
        "utm_epsg": "EPSG:32615", "state": "TX",
    },
    "Oklahoma City, OK": {
        "osf_node": "e6qfa", "folder_hint": None,
        "utm_epsg": "EPSG:32614", "state": "OK",
    },

    # ── MIDWEST ────────────────────────────────────────────────────────────
    "Chicago, IL": {
        "osf_node": "6d7p2", "folder_hint": None,
        "utm_epsg": "EPSG:32616", "state": "IL",
    },
    "Milwaukee, WI": {
        "osf_node": "khmnd", "folder_hint": None,
        "utm_epsg": "EPSG:32616", "state": "WI",
    },
    "Kansas City, MO": {
        "osf_node": "tdsy7", "folder_hint": "Kansas City",
        "utm_epsg": "EPSG:32615", "year": 2019, "state": "MO",
    },
    "Kansas City, KS": {
        "osf_node": "hxyrv", "folder_hint": None,
        "utm_epsg": "EPSG:32615", "state": "KS",
    },
    "Columbia, MO": {
        "osf_node": "ucmde", "folder_hint": None,
        "utm_epsg": "EPSG:32615", "state": "MO",
    },
    "Iowa City, IA": {
        "osf_node": "rbd3q", "folder_hint": "Iowa City",
        "utm_epsg": "EPSG:32615", "state": "IA",
    },
    "Cedar Rapids, IA": {
        "osf_node": "rbd3q", "folder_hint": "Cedar Rapids",
        "utm_epsg": "EPSG:32615", "state": "IA",
    },
    "Omaha, NE": {
        "osf_node": "z7xw5", "folder_hint": None,
        "utm_epsg": "EPSG:32615", "state": "NE",
    },
    "Anchorage, AK": {
        "osf_node": "tdsy7", "folder_hint": "Anchorage",
        "utm_epsg": "EPSG:32604", "year": 2019, "state": "AK",
    },

    # ── MOUNTAIN ───────────────────────────────────────────────────────────
    "Denver, CO": {
        "osf_node": "tdsy7", "folder_hint": "Denver",
        "utm_epsg": "EPSG:32613", "year": 2019, "state": "CO",
    },
    "Boulder, CO": {
        "osf_node": "ek7r5", "folder_hint": None,
        "utm_epsg": "EPSG:32613", "state": "CO",
    },
    "Longmont, CO": {
        "osf_node": "usp5x", "folder_hint": None,
        "utm_epsg": "EPSG:32613", "state": "CO",
    },
    "Salt Lake City, UT": {
        "osf_node": "7k2u9", "folder_hint": None,
        "utm_epsg": "EPSG:32612", "state": "UT",
    },
    "Albuquerque, NM": {
        "osf_node": "np4q9", "folder_hint": None,
        "utm_epsg": "EPSG:32613", "state": "NM",
    },
    "Las Cruces, NM": {
        "osf_node": "zd5mb", "folder_hint": None,
        "utm_epsg": "EPSG:32613", "state": "NM",
    },
    "Sedona, AZ": {
        "osf_node": "e5q7h", "folder_hint": None,
        "utm_epsg": "EPSG:32612", "state": "AZ",
    },
    "Las Vegas, NV": {
        "osf_node": "xsvjm", "folder_hint": None,   # xsvjm = Clark County NV (ktr56 is SF only)
        "utm_epsg": "EPSG:32611", "state": "NV",
    },
    "Boise, ID": {
        "osf_node": "tdsy7", "folder_hint": "Boise",
        "utm_epsg": "EPSG:32611", "year": 2019, "state": "ID",
    },

    # ── PACIFIC ────────────────────────────────────────────────────────────
    "Los Angeles, CA": {
        "osf_node": "tdsy7", "folder_hint": "Los Angeles",
        "utm_epsg": "EPSG:32611", "year": 2019, "state": "CA",
    },
    "San Bernardino, CA": {
        "osf_node": "tdsy7", "folder_hint": "San Bernardino",
        "utm_epsg": "EPSG:32611", "year": 2019, "state": "CA",
    },
    "Victorville, CA": {
        "osf_node": "tdsy7", "folder_hint": "Victorville",
        "utm_epsg": "EPSG:32611", "year": 2019, "state": "CA",
    },
    "San Diego, CA": {
        "osf_node": "m9g6t", "folder_hint": None,
        "utm_epsg": "EPSG:32611", "state": "CA",
    },
    "Sacramento, CA": {
        "osf_node": "tdsy7", "folder_hint": "Sacramento",
        "utm_epsg": "EPSG:32610", "year": 2019, "state": "CA",
    },
    "San Francisco, CA": {
        "osf_node": "ktr56", "folder_hint": "San Francisco",
        "utm_epsg": "EPSG:32610", "state": "CA",
    },
    "Portland, OR": {
        "osf_node": "e7js9", "folder_hint": None,
        "utm_epsg": "EPSG:32610", "state": "OR",
    },
    "Seattle, WA": {
        "osf_node": "mz79p", "folder_hint": None,
        "utm_epsg": "EPSG:32610", "state": "WA",
    },
}

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

AVAILABILITY_KEYS = ("traverse_points", "rasters")


def availability(entry: dict) -> dict:
    """``{"traverse_points": bool | None, "rasters": bool | None}`` for a catalog entry.

    True / False only where the catalog states it (``entry["availability"]``); None means not known
    (the node may or may not hold that kind of file).  Traverse points are the vehicle measurements;
    rasters are CAPA's interpolated area-wide maps, a land-cover model of those measurements."""
    av = (entry or {}).get("availability") or {}
    return {k: (None if av.get(k) is None else bool(av.get(k))) for k in AVAILABILITY_KEYS}


def traverse_node(entry: dict) -> tuple[str | None, str | None]:
    """(OSF node, folder hint) to search for the raw traverse points of a catalog entry."""
    entry = entry or {}
    if entry.get("traverse_node"):
        same = entry["traverse_node"] == entry.get("osf_node")
        return entry["traverse_node"], (entry.get("folder_hint") if same else entry.get("traverse_folder_hint"))
    return entry.get("osf_node"), entry.get("folder_hint")


def campaign_date(entry: dict) -> tuple[str | None, str | None]:
    """(ISO campaign date, its source) when the catalog records one, else (None, None)."""
    entry = entry or {}
    d = entry.get("campaign_date")
    return (str(d), entry.get("campaign_date_source") or "unknown") if d else (None, None)


def list_cities() -> list[str]:
    """Return all city names in the catalog, sorted alphabetically."""
    return sorted(CAPA_CATALOG.keys())


def lookup_city(name: str) -> tuple[str, dict] | tuple[None, None]:
    """Find a catalog entry by city name using fuzzy matching.

    Returns ``(canonical_name, entry)`` on a confident match,
    or ``(None, None)`` if no close match exists.

    Matching is case-insensitive and tolerates minor typos.
    """
    lower = name.strip().lower()

    # Exact match (case-insensitive)
    for key in CAPA_CATALOG:
        if key.lower() == lower:
            return key, CAPA_CATALOG[key]

    # Prefix / contains match — e.g. "Boston" → "Boston, MA"
    candidates = [k for k in CAPA_CATALOG if lower in k.lower()]
    if len(candidates) == 1:
        return candidates[0], CAPA_CATALOG[candidates[0]]
    if len(candidates) > 1:
        # Multiple prefix hits — return all so the caller can prompt
        return None, {"ambiguous": candidates}

    # Fuzzy fallback (case-insensitive)
    keys = list(CAPA_CATALOG.keys())
    keys_lower = [k.lower() for k in keys]
    close_lower = difflib.get_close_matches(lower, keys_lower, n=3, cutoff=0.6)
    close = [keys[keys_lower.index(c)] for c in close_lower]
    if len(close) == 1:
        return close[0], CAPA_CATALOG[close[0]]
    if close:
        return None, {"suggestions": close}

    return None, None
