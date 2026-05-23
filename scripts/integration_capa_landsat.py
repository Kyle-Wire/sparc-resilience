"""
Integration smoke test — CAPA-first temporal alignment with LIVE API calls.

Calls:
  Open-Meteo ERA5 archive  ->  real CAPA midday temperature + anchor dates
  AWS Element84 STAC       ->  real Landsat C2 L2 scene metadata

NO geopandas required: loads capa/landsat internals directly via importlib.util.

Study area : Austin TX inner-city (-97.74, 30.27)  -- one grid point
Date range : July 2023  (peak UHI month, dense Landsat coverage)
"""
from __future__ import annotations

import importlib.util
import sys
import types
import warnings
from datetime import date
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap: stub every parent package so sparc.data.__init__ (geopandas)
# is never executed.  Each leaf module is loaded directly from its .py file.
# ──────────────────────────────────────────────────────────────────────────────
def _stub(name: str) -> types.ModuleType:
    if name not in sys.modules:
        m = types.ModuleType(name)
        m.__path__ = []
        m.__package__ = name
        sys.modules[name] = m
    return sys.modules[name]


def _load(name: str, relpath: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, Path(relpath))
    mod  = importlib.util.module_from_spec(spec)    # type: ignore[arg-type]
    mod.__package__ = name.rsplit(".", 1)[0]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)                     # type: ignore[union-attr]
    return mod


for _pkg in ("sparc", "sparc.data", "sparc.data.collect"):
    _stub(_pkg)

_temporal_mod = _load("sparc.data.collect._temporal", "sparc/data/collect/_temporal.py")
_capa_mod     = _load("sparc.data.collect.capa",      "sparc/data/collect/capa.py")
_landsat_mod  = _load("sparc.data.collect.landsat",   "sparc/data/collect/landsat.py")

TemporalWindow      = _temporal_mod.TemporalWindow
_fetch_single_point = _capa_mod._fetch_single_point_hourly
_query_stac         = _landsat_mod._query_stac
_parse_date         = _landsat_mod._parse_date


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
LON, LAT       = -97.74, 30.27
BBOX           = (-97.80, 30.22, -97.62, 30.33)
START          = date(2023, 7,  1)
END            = date(2023, 7, 31)
STAC_CLOUD_MAX = 40.0


def sep(title: str) -> None:
    print(f"\n{'─'*62}")
    print(f"  {title}")
    print("─" * 62)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — Open-Meteo live call for one grid point
# ──────────────────────────────────────────────────────────────────────────────
sep("STEP 1 — Open-Meteo ERA5 archive  (live HTTP call)")
print(f"  point      : ({LON}, {LAT})  Austin TX downtown")
print(f"  date range : {START} -> {END}")
print()

windows, midday_dates = _fetch_single_point(LON, LAT, START, END)

if windows is None:
    print("  X  No midday data returned -- check network connectivity.")
    sys.exit(1)

print(f"  aat_morning  : {windows['aat_morning']:.2f} C  (mean across Jul 2023)")
print(f"  aat_midday   : {windows['aat_midday']:.2f} C")
print(f"  aat_night    : {windows['aat_night']:.2f} C")
print(f"  diurnal      : {windows['aat_midday'] - windows['aat_night']:.2f} C")
print(f"\n  Anchor dates with valid midday reading : {len(midday_dates)}")
for d in midday_dates[:8]:
    print(f"    {d}")
if len(midday_dates) > 8:
    print(f"    ... ({len(midday_dates) - 8} more)")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — Build TemporalWindow from CAPA anchor dates
# ──────────────────────────────────────────────────────────────────────────────
sep("STEP 2 — TemporalWindow.from_capa_dates()")
window = TemporalWindow.from_capa_dates(
    midday_dates,
    date_start=START,
    date_end=END,
    tolerance_days=3,
)
print(f"  anchor_dates   : {len(window.anchor_dates)} dates")
print(f"  tolerance_days : {window.tolerance_days}")
print()
print("  Sample STAC search windows (anchor +/- tolerance, clipped):")
for anchor in window.anchor_dates[:5]:
    s, e = window.search_range(anchor)
    print(f"    CAPA anchor {anchor}  ->  STAC search  {s} -> {e}")
if len(window.anchor_dates) > 5:
    print(f"    ... ({len(window.anchor_dates) - 5} more anchors)")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — Landsat STAC query (AWS Element84)
# ──────────────────────────────────────────────────────────────────────────────
sep("STEP 3 — Landsat C2 L2 STAC  (live AWS Element84 call)")
print(f"  bbox            : {BBOX}")
print(f"  full date range : {START} -> {END}")
print(f"  cloud_cover_max : {STAC_CLOUD_MAX}%")
print()

try:
    scenes = _query_stac(BBOX, START, END, STAC_CLOUD_MAX)
except Exception as exc:
    print(f"  X  STAC query failed: {exc}")
    scenes = []

print(f"  Scenes returned : {len(scenes)}")
if scenes:
    print()
    print(f"  {'Scene ID':<30}  {'Date':12}  {'Cloud%':>7}")
    print(f"  {'─'*30}  {'─'*12}  {'─'*7}")
    for s in scenes[:15]:
        sid    = s.get("id", "?")[:30]
        dt_str = s.get("datetime", "?")[:10]
        cloud  = s.get("cloud_cover", float("nan"))
        print(f"  {sid:<30}  {dt_str:12}  {cloud:>7.1f}%")
    if len(scenes) > 15:
        print(f"  ... ({len(scenes) - 15} more scenes)")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — Align CAPA anchors to real Landsat scenes
# ──────────────────────────────────────────────────────────────────────────────
sep("STEP 4 — CAPA <-> Landsat temporal alignment")
if not scenes:
    print("  No scenes available -- alignment not possible.")
else:
    scene_dates = [_parse_date(s["datetime"]) for s in scenes]
    unique_scene_dates = sorted(set(scene_dates))
    print(f"  Unique Landsat dates ({len(unique_scene_dates)}):")
    for d in unique_scene_dates:
        print(f"    {d}")
    print()
    print(f"  {'CAPA anchor':14}  {'Matched scene':14}  {'Offset':>6}  Status")
    print(f"  {'─'*14}  {'─'*14}  {'─'*6}  {'─'*30}")
    exact = near = beyond = missed = 0
    for anchor in window.anchor_dates:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            matched, offset = window.closest_match(scene_dates, anchor)
        if matched is None:
            status = "no scene in window"
            missed += 1
        elif offset == 0:
            status = "exact match"
            exact += 1
        elif offset <= window.tolerance_days:
            status = f"within tolerance (+{offset}d)"
            near += 1
        else:
            status = f"BEYOND tolerance ({offset}d gap)"
            beyond += 1
        matched_str = str(matched) if matched else "--"
        print(f"  {str(anchor):14}  {matched_str:14}  {offset:>6}d  {status}")
    print()
    print(f"  Exact matches    : {exact}")
    print(f"  Near  matches    : {near}  (within +/-{window.tolerance_days}d)")
    print(f"  Beyond tolerance : {beyond}")
    print(f"  No scene found   : {missed}")

print()
