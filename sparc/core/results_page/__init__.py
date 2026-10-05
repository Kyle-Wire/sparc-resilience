"""The standalone interactive results page of a finished core run (SPEC §6.9).

    python -m sparc.core.results_page <run dir> <core config> [--out results.html] [--placebo placebo.json]

:func:`build_results_page` reads the run directory written by ``run_core`` (manifest.json, predictions.parquet,
response_*.parquet, scenario_deltas.parquet, allocation.parquet and the stage files), plus the optional
post-run outputs (planner/, emulator.npz/.json, placebo.json and the uncertainty / simcheck / multiverse
sections merged into the manifest), and writes one self-contained HTML page: a map explorer (temperature,
land cover, cooling footprints, saturation, scenarios, budget plan, climate futures, planner layers, a design
tool and the CV folds) plus charts and tables for accuracy vs distance, area of influence, dose-response,
scenarios, uncertainty, climate, the causal audit, the budget plan and the validation studies.

Nothing here is tied to one city:

* the data and the CV folds are rebuilt with :func:`sparc.core.baselines.load_run` (the checkpoint is never
  unpickled), and the held-out / buffer cells of each fold travel as one u8 mask per fold;
* ``causal``, ``cv_distance``, ``optimize``, ``influence``, ``baselines``, ``climate`` and ``scenarios`` fall
  back to their stage files when the manifest lacks them (runs stopped before the manifest was finalised);
* the map's layer catalogue (:func:`layer_catalog`) and the variable names come from the config's physics
  roles and actionable levers;
* the Pareto caption is written from the frontier's numbers (:func:`pareto_caption`).

Page text comes from the config's optional ``report`` block::

    report:
      title: Providence Heat Model
      place: Providence, RI
      area: the Brown University / Providence study area
      caveats: ["..."]          # appended to the data-driven caveats
"""

from __future__ import annotations

import argparse
import base64
import copy
import html
import json
import re
from pathlib import Path

import numpy as np

__all__ = ["build_results_page", "collect", "render_page", "layer_catalog", "pareto_caption", "variable_names",
           "lever_info", "TEMPLATE", "main"]

TEMPLATE = Path(__file__).resolve().parent / "template.html"

#: display names of the physics roles (the variable a role names gets this label)
ROLE_NAMES = {"canopy": "Tree canopy", "impervious": "Impervious cover", "albedo": "Albedo", "ndvi": "NDVI",
              "elevation": "Elevation", "water_distance": "Distance to water"}
#: short names used in layer titles ("Canopy: footprint per +1 pp")
ROLE_SHORT = {"canopy": "Canopy", "impervious": "Impervious", "albedo": "Albedo", "ndvi": "NDVI",
              "elevation": "Elevation", "water_distance": "Distance to water"}
#: nouns used in running text ("Planned canopy increase")
ROLE_NOUNS = {"canopy": "canopy", "impervious": "impervious cover", "albedo": "albedo"}
#: map inputs shown in the Land cover theme, in this order (roles), with unit and decimals
INPUT_ROLES = (("canopy", "%", 0, "Share of the cell under tree canopy."),
               ("impervious", "%", 0, "Share of the cell that is paved or built."),
               ("ndvi", "", 2, "Vegetation greenness. In scenarios it follows canopy and impervious cover through "
                               "a fitted monotone model."),
               ("albedo", "", 3, "Surface reflectance (0–1)."))
_UNIT_LABELS = {"degf": "°F", "degc": "°C", "f": "°F", "c": "°C", "k": "K", "kelvin": "K"}
_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
          10: "ten"}


# ---------------------------------------------------------------------------
# encoding helpers
# ---------------------------------------------------------------------------

def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def _enc(v, kind: str = "u16") -> dict:
    """Quantise a per-point layer (0 = missing) with its range and percentiles."""
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    lo, hi = (float(np.nanmin(v)), float(np.nanmax(v))) if ok.any() else (0.0, 1.0)
    if hi <= lo:
        hi = lo + 1e-9
    top = 254 if kind == "u8" else 65534
    q = np.zeros(v.size, np.uint8 if kind == "u8" else np.uint16)
    q[ok] = 1 + np.rint((v[ok] - lo) / (hi - lo) * top).astype(q.dtype)
    p = [float(x) for x in np.nanpercentile(v[ok], [1, 2, 50, 98, 99])] if ok.any() else [0.0] * 5
    return {"kind": kind, "lo": lo, "hi": hi, "b64": _b64(q), "p": p,
            "mean": float(np.nanmean(v)) if ok.any() else None}


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _unit_label(units: str | None) -> str:
    """The template's ``U``: ``degF`` → ``°F`` (other unit strings as they are)."""
    if not units:
        return ""
    return _UNIT_LABELS.get(str(units).strip().lower(), str(units))


def _fmt(v, d: int = 2) -> str:
    """The template's ``fmt``: fixed decimals with thousands separators, "—" when not a finite number."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    return f"{f:,.{d}f}" if np.isfinite(f) else "—"


def _block_text(block_m) -> str:
    """``2000`` → ``2 km`` (the held-out block size in running text); ``held-out`` when unknown."""
    return f"{_km(block_m)} km" if block_m else "held-out"


def _number_word(n: int) -> str:
    return _WORDS.get(int(n), str(int(n)))


def _km(m) -> str:
    """``2000`` → ``2``, ``600`` → ``0.6`` (km)."""
    try:
        v = float(m) / 1000.0
    except (TypeError, ValueError):
        return "—"
    return f"{v:g}" if abs(v * 10 - round(v * 10)) < 1e-9 else f"{v:.2f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------------------
# variables and levers
# ---------------------------------------------------------------------------

def _roles(cfg) -> dict:
    return dict(((cfg.raw.get("physics") or {}).get("roles")) or {})


def lever_info(cfg) -> dict[str, dict]:
    """``{lever: {role, label, short, noun, unit, albedo_like, step, mult, decimals, direction}}``.

    ``albedo_like`` levers (role albedo, unit "albedo"/"reflectance", or a maximum ≤ 1) are reported per +0.01
    on the map and per +0.1 in the causal table; the others per +1 unit."""
    by_col = {col: role for role, col in _roles(cfg).items()}
    out = {}
    for var, spec in (cfg.actionable or {}).items():
        spec = spec or {}
        role = by_col.get(var)
        unit_raw = str(spec.get("unit") or "")
        hi = spec.get("max")
        albedo_like = bool(role == "albedo" or unit_raw.lower() in ("albedo", "reflectance")
                           or (hi is not None and float(hi) <= 1.0))
        label = spec.get("label") or ROLE_NAMES.get(role) or var.replace("_", " ")
        unit = "" if albedo_like else (unit_raw or "units")
        out[var] = {"role": role, "label": label, "short": ROLE_SHORT.get(role) or label,
                    "noun": ROLE_NOUNS.get(role) or str(label).lower(), "unit": unit, "albedo_like": albedo_like,
                    "step": 0.01 if albedo_like else 1.0, "decimals": 3 if albedo_like else 1,
                    "dose_decimals": 2 if albedo_like else 1, "edit_decimals": 2 if albedo_like else 0,
                    "direction": str(spec.get("direction", "increase"))}
    return out


def variable_names(cfg) -> dict[str, str]:
    """Display names of the variables (the template's ``vname``): physics roles, then lever labels."""
    out = {"target": "Air temperature residual"}
    for role, col in _roles(cfg).items():
        if col and role in ROLE_NAMES:
            out[str(col)] = ROLE_NAMES[role]
    for var, info in lever_info(cfg).items():
        out.setdefault(var, str(info["label"]))
    return out


# ---------------------------------------------------------------------------
# captions written from the numbers
# ---------------------------------------------------------------------------

def pareto_caption(optimize: dict | None, units: str = "°F") -> str | None:
    """The budget frontier's caption: how much more cooling a doubled budget buys, read from the points.

    The ratio is total planned cooling at 2× over 1× the budget (or between the two largest budgets when the
    frontier has no 1×/2× pair).  The wording follows the ratio; there is no canned conclusion."""
    o = optimize or {}
    pts = [p for p in ((o.get("pareto") or {}).get("points") or []) if isinstance(p, dict)]
    budget = o.get("budget")
    base = f"Total planned cooling ({units} summed over cells) as the budget scales."
    if not pts or not budget:
        return None
    try:
        xs = [float(p["budget"]) / float(budget) for p in pts]
        ys = [float(p["total_benefit"]) for p in pts]
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return base
    k1 = next((i for i, x in enumerate(xs) if abs(x - 1.0) < 1e-9), None)
    k2 = next((i for i, x in enumerate(xs) if abs(x - 2.0) < 1e-9), None)
    if k1 is not None and k2 is not None:
        a, b, step = k1, k2, "Doubling the budget from 1× to 2×"
    elif len(xs) >= 2:
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        a, b = order[-2], order[-1]
        step = f"Raising the budget from {xs[a]:g}× to {xs[b]:g}×"
    else:
        return base
    if not (ys[a] > 0) or not np.isfinite(ys[b]):
        return base
    r = ys[b] / ys[a]
    budget_ratio = xs[b] / xs[a]
    gain = (r - 1.0) / max(budget_ratio - 1.0, 1e-9)       # extra cooling per extra budget, vs the first budget
    if gain >= 0.85:
        verdict = "so returns barely diminish at this scale: plenty of high-benefit cells remain."
    elif gain >= 0.5:
        verdict = (f"so returns diminish: each extra unit of budget buys about {gain:.0%} of the cooling a unit of "
                   "the smaller budget bought, because the best cells are treated first.")
    elif gain > 0:
        verdict = (f"so returns diminish sharply: each extra unit of budget buys only about {gain:.0%} of the "
                   "cooling a unit of the smaller budget bought; most high-benefit cells are already in the plan.")
    else:
        verdict = "so the extra budget buys no further cooling: every cell that cools is already in the plan."
    return f"{base} {step} raises total cooling {r:.2f}×, {verdict}"


# ---------------------------------------------------------------------------
# the layer catalogue
# ---------------------------------------------------------------------------

def _hd_label(case: str) -> tuple[str, bool]:
    """``(label, is_today)`` of a planner hot-days case (``today``, ``ssp245_mid``, ``ssp585_late`` …)."""
    if case == "today":
        return "per summer (today)", True
    m = re.fullmatch(r"ssp(\d)(\d)(\d)_(mid|late)", case)
    if m:
        period = "2041–2060" if m.group(4) == "mid" else "2081–2100"
        return f"(SSP{m.group(1)}-{m.group(2)}.{m.group(3)}, {period})", False
    return f"({case.replace('_', ' ')})", False


def layer_catalog(cfg, layers: dict, *, units: str, background: float, block_m=None, n_folds: int = 5,
                  scen_layers=(), climate: bool = False, sites: bool = False, design: bool = False,
                  optimize_variable: str | None = None) -> list[dict]:
    """The map's layer list, generated from the config (roles, levers) and the layers present.

    Each entry is ``{group, key, name, unit, scale, center?, d, desc, mult?, zeroBlank?}``; keys starting with
    ``__`` are computed in the page (climate futures, logger sites, the design tool, the CV folds)."""
    U = units
    roles = _roles(cfg)
    levers = lever_info(cfg)
    folds_word = _number_word(n_folds)
    out: list[dict] = []

    def add(**o) -> None:
        if o["key"].startswith("__") or o["key"] in layers:
            out.append(o)

    bg = _fmt(background, 1)
    block = _block_text(block_m)
    add(group="Temperature", key="obs", name="Observed air temperature", unit=U, scale="div", center=background,
        d=1, desc=f"Measured air temperature. Colours are relative to the city median ({bg} {U}).")
    add(group="Temperature", key="pred", name="Predicted (held-out)", unit=U, scale="div", center=background, d=1,
        desc=f"Each cell is predicted by the fold model that never saw its {block} block, so this is what the model "
             "knows without local data.")
    add(group="Temperature", key="err", name="Error (observed − predicted)", unit=U, scale="div", center=0, d=2,
        desc="Red: warmer than predicted. Blue: cooler. Large coherent patches are local heat sources or sinks the "
             "inputs do not capture.")
    add(group="Temperature", key="hw", name="90% interval half-width", unit=U, scale="seq", d=2,
        desc="Cross-conformal interval: 90% of held-out cells fall within this distance of the prediction.")
    shown = set()
    for role, unit, d, desc in INPUT_ROLES:
        col = roles.get(role)
        if col and f"in_{col}" not in shown:
            shown.add(f"in_{col}")
            add(group="Land cover", key=f"in_{col}", name=ROLE_NAMES[role], unit=unit, scale="seq", d=d, desc=desc)
    for var, info in levers.items():
        if f"in_{var}" in shown:
            continue
        shown.add(f"in_{var}")
        add(group="Land cover", key=f"in_{var}", name=str(info["label"]), unit=info["unit"], scale="seq",
            d=3 if info["albedo_like"] else 1, desc=f"Today's {info['noun']} (an actionable lever).")
    # cooling effects: footprint per unit (and own-cell for canopy) per lever
    for var, info in levers.items():
        step = "0.01" if info["albedo_like"] else "1"
        per = f"+{step}" + (f" {info['unit']}" if info["unit"] else "")
        role = info["role"]
        if role == "canopy":
            desc = ("Total temperature change summed over every cell affected when this cell gains 1 percentage "
                    "point of canopy (through neighbourhood features and heat transport).")
        elif role == "impervious":
            desc = ("Total temperature change when this cell gains 1 pp of impervious cover. Removing pavement "
                    "reverses the sign.")
        elif info["albedo_like"]:
            desc = f"Total temperature change when this cell's {info['noun']} rises by 0.01."
        else:
            desc = (f"Total temperature change summed over every cell affected when this cell's {info['noun']} rises "
                    f"by {per.lstrip('+')} (through neighbourhood features and heat transport).")
        if var == optimize_variable:
            desc += " This is what the budget optimiser ranks."
        add(group="Cooling effects", key=f"fp_{var}", name=f"{info['short']}: footprint per {per}", unit=U,
            scale="div", center=0, d=4, desc=desc, **({"mult": 0.01} if info["albedo_like"] else {}))
        if role == "canopy":
            add(group="Cooling effects", key=f"own_{var}", name=f"{info['short']}: own-cell effect per {per}",
                unit=U, scale="div", center=0, d=5,
                desc="Change in this cell's own temperature only, with its neighbours held fixed. Much smaller than "
                     "the footprint: canopy cools its surroundings more than itself.")
    for var, info in levers.items():
        nm = info["short"] + (" removal" if info["direction"] == "decrease" else "")
        unit = info["unit"]
        add(group="Saturation", key=f"cls_{var}", name=f"{nm}: response shape", scale="cat",
            desc="Saturating: cooling levels off within the tested doses. Linear: no knee yet. Accelerating: little "
                 "effect at first, then a sharp rise before levelling off. Grey: too little headroom to tell.")
        add(group="Saturation", key=f"marg_{var}", name=f"{nm}: cooling per unit at today's level",
            unit=f"{U}/{unit or 'unit'}", scale="seq", d=3,
            desc="Initial slope of the fitted curve: the cooling the first unit of change would buy here. High values "
                 "are the best places to act first.")
        add(group="Saturation", key=f"A_{var}", name=f"{nm}: maximum cooling", unit=U, scale="seq", d=2,
            desc="Plateau of the fitted saturating curve (for linear cells, the cooling at the largest dose tested).")
        add(group="Saturation", key=f"d90_{var}", name=f"{nm}: dose for 90% of the maximum", unit=unit, scale="seq",
            d=info["dose_decimals"],
            desc="Neighbourhood dose that gets 90% of the achievable cooling. Small values mean the benefit saturates "
                 "quickly.")
    for s in scen_layers:
        add(group="Scenarios", key=s["key"], name=s["name"], unit=U, scale="div", center=0, d=2,
            desc=f"Re-predicted change in air temperature at every cell for this scenario (mean of the {folds_word} "
                 "fold models).")
    oinfo = levers.get(optimize_variable or "") or {}
    noun = oinfo.get("noun") or "lever"
    verb = "decrease" if oinfo.get("direction") == "decrease" else "increase"
    add(group="Budget plan", key="alloc_dose", name=f"Planned {noun} {verb}", unit=oinfo.get("unit") or "",
        scale="seq", d=1, zeroBlank=True,
        desc=f"Where the budget-optimal plan {'removes' if verb == 'decrease' else 'adds'} {noun}, and how much. "
             "Untreated cells are blank.")
    add(group="Budget plan", key="alloc_delta", name="Re-predicted ΔT of the plan", unit=U, scale="div", center=0, d=3,
        desc="The whole plan applied at once and re-predicted through the model (closed-loop check).")
    if climate:
        add(group="Climate futures", key="__clim_temp", name="Future air temperature", unit=U, scale="climtemp",
            center=background, d=1,
            desc="Observed temperature plus the CMIP6 multi-model median warming for the chosen pathway and period, "
                 "plus the chosen adaptation scenario's re-predicted change. Colours keep today's scale, so warming "
                 "shows as a shift to red.")
        add(group="Climate futures", key="__clim_thr", name="Cells at or above the heat threshold", scale="climthr",
            desc="Cells whose projected temperature reaches the threshold chosen in the Heat exposure table (Climate "
                 "futures section) under the chosen pathway, period and adaptation.")
    add(group="Heat stress", key="heat_index", name="Heat index (NWS), campaign afternoon", unit="°F", scale="seq",
        d=1, desc="Apparent temperature from the measured air temperature and the campaign's dewpoint (uniform "
                  "across the city, so each cell's relative humidity follows from its own temperature), using the "
                  "US National Weather Service heat index.")
    add(group="Heat stress", key="heat_cat", name="Heat-risk category (NWS)", scale="heatcat",
        desc="NWS heat-index categories: Caution 80–90 °F (fatigue with prolonged exposure), Extreme caution "
             "90–103 °F (heat cramps and exhaustion possible), Danger 103–125 °F (heat stroke possible), Extreme "
             "danger ≥ 125 °F (heat stroke highly likely).")
    add(group="Planner", key="people", name="Residents per cell (HRSL)", unit="people", scale="seq", d=1,
        desc="Residential population from Meta/CIESIN's High Resolution Settlement Layer (census-based, circa "
             "2010s), summed onto each cell. Daytime presence differs.")
    add(group="Planner", key="plantable", name="Room for more canopy", unit="pp", scale="seq", d=0,
        desc="Canopy a cell could still gain: open green and bare land (ESA WorldCover 2021) plus a fifth of its "
             "built-up area for street trees and lots, never above 100% minus today's canopy. The budget plan "
             "respects it.")
    hd = []
    for key in layers:
        m = re.fullmatch(r"hd_(.+?)_(today|ssp\d{3}_(?:mid|late)|[a-z0-9]+_[a-z0-9]+)", key)
        if m:
            try:
                thr = float(m.group(1))
            except ValueError:
                continue
            hd.append((thr, m.group(1), m.group(2), key))
    first_today = True
    for _thr, t, case, key in sorted(hd, key=lambda r: (r[0], r[2] != "today", r[2])):
        label, today = _hd_label(case)
        if today:
            name = f"Hot afternoons ≥ {t} {U} {label}"
            if first_today:
                desc = (f"Summer days (June–August, 1995–2014 airport record) whose afternoon reaches {t} {U} in this "
                        "cell: the airport's daily maximum plus the cell's campaign-afternoon difference from the "
                        "airport.")
                first_today = False
            else:
                desc = f"As above for {t} {U}."
        else:
            name = f"Hot afternoons ≥ {t} {U} {label}"
            desc = "As today, with each CMIP6 model's summer warming added to the airport record (median over models)."
        add(group="Planner", key=key, name=name, unit="days", scale="seq", d=1, desc=desc)
    if sites:
        add(group="Planner", key="__sites", name="Suggested logger sites", scale="sites",
            desc="Where new temperature loggers would most sharpen the canopy effect: low- and high-canopy cells at "
                 "matched impervious cover, favouring places where the fold models disagree most, at least 400 m "
                 "apart.")
    if design:
        add(group="Design", key="__design", name="Your design: change in air temperature", unit=U, scale="design",
            center=0, d=3,
            desc="Paint changes on the map with the brush; the change in afternoon air temperature updates as you go. "
                 "It comes from a linear emulator of the fitted model (fold-averaged), checked against the full model "
                 "on random patches; use the Scenarios theme for city-wide changes.")
    add(group="CV design", key="__folds", name="Spatial cross-validation folds", scale="folds",
        desc="For the chosen fold: blue cells are held out for testing, orange cells are dropped from training as a "
             "buffer, grey cells train the models. Every accuracy number on this page comes from this design.")
    return out


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------

def _stage_file(run: Path, name: str):
    p = run / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _causal_section(m: dict, run: Path) -> dict | None:
    """The manifest's ``causal`` summary, else built from ``causal.json`` (S6's own file)."""
    if m.get("causal"):
        return m["causal"]
    raw = _stage_file(run, "causal.json")
    if not isinstance(raw, dict):
        return None
    if "treatments" not in raw:                 # already a summary (older layout)
        return raw
    from sparc.core.report import _causal_summary

    return _causal_summary(raw) or None


def _identify_section(run: Path) -> dict | None:
    """Canopy identification (``python -m sparc.core.identify``): the lab's verdicts, the map designs on the
    real target and, when a campaign's traverses were analysed, their estimate.  Read from ``identify/``
    next to the run (or inside it)."""
    for d in (run / "identify", run.parent / "identify"):
        lab = _stage_file(d, "identify_lab.json")
        if not lab:
            continue
        keep = ("mean", "sd", "mean_se", "truth", "bias", "coverage", "excludes_zero", "advects")
        designs = {k: {"label": v["label"], "estimand": v["estimand"], "level": v["level"], "kind": v["kind"],
                       "status": v["verdict"]["status"], "reasons": v["verdict"]["reasons"],
                       "worlds": {w: {f: x.get(f) for f in keep} for w, x in v["generators"].items()},
                       **({"floor": v["floor"]} if v.get("floor") else {})}
                   for k, v in lab.get("designs", {}).items()}
        mp = _stage_file(d, "identify_map.json") or {}
        est = _stage_file(d, "identify_estimate.json")
        if est:
            km = est.get("kilometre") or {}
            est = {k: est.get(k) for k in ("source", "window", "run", "qa", "headline")} | {
                "runs": [{k: r.get(k) for k in ("run", "window", "n", "street")} for r in est.get("runs") or []],
                "kilometre": {k: km.get(k) for k in ("status", "reason", "winds", "result", "lab_status")}}
        kml = _stage_file(d, "identify_kmlab.json")
        kmd = ((kml or {}).get("designs") or {}).get("wind_shift")
        return {"n_reps": lab.get("n_reps"), "designs": designs, "map": mp.get("designs"), "estimate": est,
                "kmlab": {"winds": kmd.get("winds"), "status": kmd["verdict"]["status"], "reasons": kmd["verdict"]["reasons"],
                          "worlds": kmd["generators"]} if kmd else None}
    return None


def _corners(cfg, g) -> dict | None:
    """``[lat, lon]`` of the corner cell centres.  The run frame is ``data.reproject_to`` (metres) when the data
    were reprojected, else ``data.crs`` scaled to metres by its coordinate unit."""
    crs = cfg.data.get("reproject_to") or cfg.data.get("crs")
    if not crs:
        return None
    from pyproj import Transformer

    tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    s = 1.0 if cfg.data.get("reproject_to") else cfg.coord_scale
    x1, y1 = g.x0 + (g.nx - 1) * g.dx, g.y0 + (g.ny - 1) * g.dy

    def ll(xm, ym):
        lon, lat = tr.transform(xm / s, ym / s)
        return [lat, lon]

    return {"sw": ll(g.x0, g.y0), "se": ll(x1, g.y0), "nw": ll(g.x0, y1), "ne": ll(x1, y1)}


def _ids(ids) -> dict:
    """Point ids for the design tool's CSV: packed uint32 when they are integers that fit, else as text."""
    a = np.asarray(ids)
    if a.dtype.kind in "iu" and (a.size == 0 or (a.min() >= 0 and a.max() < 2 ** 32)):
        return {"ids": _b64(a.astype(np.uint32))}
    if a.dtype.kind == "f" and np.all(np.isfinite(a)) and np.all(a == np.round(a)) and \
            (a.size == 0 or (a.min() >= 0 and a.max() < 2 ** 32)):
        return {"ids": _b64(a.astype(np.uint32))}
    return {"ids_text": [str(x) for x in a.tolist()]}


def collect(run, cfg, placebo_path=None) -> dict:
    """Everything the page shows, as one JSON-ready dict (the template's ``DATA``)."""
    import pandas as pd

    from sparc.core.baselines import load_run

    run = Path(run)
    cfg = copy.deepcopy(cfg)                      # load_run applies the run's coarse / subsample settings to it
    placebo = None
    for cand in ([placebo_path] if placebo_path else []) + [run / "placebo.json"]:
        if cand and Path(cand).exists():
            placebo = json.loads(Path(cand).read_text(encoding="utf-8"))
            placebo = {k: placebo.get(k) for k in ("rows", "n_pass_model", "n_pass_causal", "n_placebos", "coarse_m",
                                                   "layer_correlation_with_original")}
            break

    data, folds, m, pred = load_run(run, cfg)
    g = data.grid
    L: dict = {
        "obs": _enc(pred["target"]),
        "pred": _enc(pred["pred"]),
        "err": _enc(pred["target"] - pred["pred"]),
        "hw": _enc(pred["pi_hi"] - pred["pred"]),
    }
    roles = _roles(cfg)
    levers = lever_info(cfg)
    for role in ("canopy", "impervious", "ndvi", "albedo"):
        col = roles.get(role)
        if col and col in data.frame:
            L["in_" + col] = _enc(data.frame[col], "u8")
    for v in levers:                              # the design tool clamps each lever around today's value
        if f"in_{v}" not in L and v in data.frame:
            L["in_" + v] = _enc(data.frame[v], "u8")
    fold = np.where(folds.fold_id >= 0, folds.fold_id, 255).astype(np.uint8)
    excl = [_b64((~folds.train_masks[k] & ~folds.test_masks[k]).astype(np.uint8)) for k in range(folds.n_folds)]

    canopy = roles.get("canopy")
    for v in cfg.actionable:
        f = run / f"response_{v}.parquet"
        if not f.exists():
            continue
        r = pd.read_parquet(f)
        L[f"fp_{v}"] = _enc(r["footprint_effect_per_unit"])
        if v == canopy:
            L[f"own_{v}"] = _enc(r["own_effect_per_unit"])
        L[f"marg_{v}"] = _enc(r["marginal_benefit_per_unit"])
        L[f"A_{v}"] = _enc(r["max_cooling_A"])
        L[f"d90_{v}"] = _enc(r["d90"])
        cm = r["curve_model"].astype(str).to_numpy()
        # three coloured shapes (a map carries at most three distinguishable
        # categorical colours); censored / insufficient cells are grey
        cls = np.where(r["censored"].to_numpy(bool), 0,
                       np.where(cm == "saturating", 1, np.where(cm == "linear", 2,
                                np.where(cm == "sigmoid", 3, 0)))).astype(np.uint8)
        L[f"cls_{v}"] = {"kind": "cat", "b64": _b64(cls),
                         "labels": ["too little headroom to tell", "saturating", "linear", "accelerating (S-shaped)"]}
    scen_layers = []
    if (run / "scenario_deltas.parquet").exists():
        d = pd.read_parquet(run / "scenario_deltas.parquet")
        for c in d.columns:
            if c != "id":
                key = f"sc_{len(scen_layers)}"
                L[key] = _enc(d[c])
                scen_layers.append({"key": key, "name": c})
    if (run / "allocation.parquet").exists():
        a = pd.read_parquet(run / "allocation.parquet")
        L["alloc_dose"] = _enc(a["dose"])
        L["alloc_delta"] = _enc(a["closed_loop_delta"])

    # planner layers (post.planner) and next-campaign sites
    pl_dir = run / "planner"
    sites = None
    if (pl_dir / "planner_cells.parquet").exists():
        pc = pd.read_parquet(pl_dir / "planner_cells.parquet")
        if np.array_equal(pc["id"].to_numpy(), data.ids):
            L["people"] = _enc(pc["people"])
            L["plantable"] = _enc(pc["plantable_canopy_pp"])
            for c in pc.columns:
                if c.startswith("hot_days_ge_"):
                    L["hd_" + c[len("hot_days_ge_"):]] = _enc(pc[c])
    # heat stress (NWS heat index) and plain verdicts per scenario
    from sparc.core import heat as heatmod

    people = None
    if (pl_dir / "planner_cells.parquet").exists() and "people" in L:
        people = pc["people"].to_numpy(float)
    heat_risk = None
    try:
        heat_risk = heatmod.run_heat_risk(run, people=people, config_dir=getattr(cfg, "base_dir", None))
    except Exception:                                     # noqa: BLE001 - the page builds without it
        heat_risk = None
    if heat_risk:
        hi = heatmod.heat_index_today(data.target_raw, data.target_units, heat_risk["dewpoint_C"])
        L["heat_index"] = _enc(hi)
        L["heat_cat"] = {"kind": "cat", "b64": _b64(heatmod.category_codes(hi).astype(np.uint8)),
                         "labels": [c[1] for c in heatmod.CATEGORIES]}
    verdicts = {r["scenario"]: heatmod.effect_verdict(r)
                for r in ((m.get("uncertainty") or {}).get("scenarios") or []) if r.get("scenario")}
    if (pl_dir / "logger_sites.csv").exists():
        ls = pd.read_csv(pl_dir / "logger_sites.csv")
        sites = {"cell": ls["cell"].astype(int).tolist(), "role": ls["role"].tolist(),
                 "canopy": ls["canopy"].round(1).tolist(), "impervious": ls["impervious"].round(1).tolist()}

    # design tool: linear emulator (post.emulator)
    design = None
    if (run / "emulator.npz").exists() and (run / "emulator.json").exists():
        em = np.load(run / "emulator.npz")
        meta = json.loads((run / "emulator.json").read_text(encoding="utf-8"))
        if np.array_equal(em["ids"], data.ids):
            lv = {}
            for var, d in meta["levers"].items():
                v = d["validation"]
                lv[var] = {
                    "own": _enc(em[f"{var}__own"]),
                    "chans": [{"sigma": c["sigma_cells"], "coef": _enc(em[c["coef"]]),
                               "w": None if c["unit_weight"] else _enc(em[c["weight"]])} for c in d["channels"]],
                    "dq": _enc(em[f"{var}__dq"]) if d["physics"] and f"{var}__dq" in em.files else None,
                    "bounds": d["bounds"], "direction": d["direction"],
                    "dose": float(d.get("design_dose", v["dose"])),
                    "val": {"dose": v["dose"], "patch_abs": v["patch_mean_abs_err_median"],
                            "patch_rel": v["patch_mean_rel_err_median"], "pass": v["patch_pass_rate"],
                            "p95": v["p95_cell_err_median"], "uniform_rel": v["uniform"]["rel_err"]},
                }
            K = em["physics_kernel"].astype("float32") if "physics_kernel" in em.files else None
            design = {"levers": lv, "kernel": None if K is None else
                      {"k": int((K.shape[0] - 1) // 2), "b64": _b64(K)}}

    def jl(name):
        return _stage_file(run, name)

    rep = cfg.raw.get("report") or {}
    units = data.target_units
    U = _unit_label(units)
    optimize = jl("optimize.json") or m.get("optimize")
    climate = m.get("climate") or jl("climate.json")
    cv = m.get("cv") or {}
    catalog = layer_catalog(cfg, L, units=U, background=float(data.background), block_m=cv.get("block_m"),
                            n_folds=int(folds.n_folds), scen_layers=scen_layers, climate=bool(climate),
                            sites=bool(sites), design=bool(design and design["levers"]),
                            optimize_variable=(optimize or {}).get("variable"))
    return _clean({
        "name": m["name"], "created": m["created_utc"], "commit": m.get("git_commit"), "n": m["n_points"],
        "timings": m.get("timings_s"), "qa": m.get("qa"), "influence": m.get("influence") or jl("influence.json"),
        "cv": m.get("cv"),
        "metrics": m.get("metrics"), "lambda_pde": m.get("lambda_pde"), "lambda_scores": m.get("lambda_scores"),
        "stacker": m.get("stacker"), "stacker_choice": m.get("stacker_choice"), "spatial_plus": m.get("spatial_plus"),
        "physics": m.get("physics"), "cv_distance": m.get("cv_distance") or jl("cv_distance.json"),
        "response": m.get("response"), "curves": jl("response_curves.json"),
        "scenarios": m.get("scenarios") or jl("scenarios.json"),
        "causal": _causal_section(m, run), "identify": _identify_section(run), "optimize": optimize, "climate": climate,
        "baselines": m.get("baselines") or jl("baselines.json"), "provenance": m.get("provenance"),
        "literature": m.get("literature"), "planner": m.get("planner"), "uncertainty": m.get("uncertainty"),
        "heat": heat_risk, "verdicts": verdicts,
        "simcheck": m.get("simcheck"), "multiverse": m.get("multiverse"), "placebo": placebo, "design": design,
        "sites": sites, "physics_advection": m.get("physics_advection"),
        "forcing": ((m.get("config") or {}).get("physics") or {}).get("forcing_info"),
        "optimize_meta": {k: (m.get("optimize") or {}).get(k) for k in ("constraint", "objective")},
        "actionable": cfg.actionable, "units": units, "background": float(data.background),
        "roles": roles, "var_names": variable_names(cfg), "levers": levers, "catalog": catalog,
        "unit_label": U, "block_txt": _block_text(cv.get("block_m")), "pareto_caption": pareto_caption(optimize, U),
        "geom": {"nx": g.nx, "ny": g.ny, "dx": g.dx, "n": int(data.n), "corners": _corners(cfg, g),
                 "ix": _b64(g.ix.astype(np.uint16)), "iy": _b64(g.iy.astype(np.uint16)), **_ids(data.ids)},
        "layers": L, "fold": _b64(fold), "excl": excl, "n_folds": int(folds.n_folds),
        "scen_layers": scen_layers, "caveats_extra": list(rep.get("caveats") or []),
    })


# ---------------------------------------------------------------------------
# render and build
# ---------------------------------------------------------------------------

def render_page(blob: dict, cfg) -> str:
    """The template with the page text and the ``DATA`` payload filled in."""
    rep = cfg.raw.get("report") or {}
    qa = blob.get("qa") or {}
    cv = blob.get("cv") or {}
    text = TEMPLATE.read_text(encoding="utf-8")
    subs = {
        "{{TITLE}}": html.escape(str(rep.get("title", "Urban Heat Model")), quote=False),
        "{{PLACE}}": html.escape(str(rep.get("place", cfg.name)), quote=False),
        "{{AREA}}": html.escape(str(rep.get("area", "the study area")), quote=False),
        "{{CELL}}": f"{_fmt(qa.get('cell_m'), 0)} m cell" if qa.get("cell_m") else "grid cell",
        "{{N_FOLDS}}": _number_word(int(blob.get("n_folds") or cv.get("n_folds") or 5)),
        "{{BLOCK}}": _block_text(cv.get("block_m")),
    }
    for k, v in subs.items():
        text = text.replace(k, v)
    payload = json.dumps(blob, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    return text.replace("/*__DATA__*/", payload)


def build_results_page(run_dir, cfg, out, placebo_path=None) -> Path:
    """Build the page of ``run_dir`` (``cfg``: a ``CoreConfig`` or a config path) into ``out``; returns ``out``.

    ``placebo_path``: a ``placebo.json`` of a placebo study (default: the run's own copy, if any)."""
    from sparc.core import runio
    from sparc.core.config import CoreConfig, load_core_config

    if not isinstance(cfg, CoreConfig):
        cfg = load_core_config(cfg)
    blob = collect(Path(run_dir), cfg, Path(placebo_path) if placebo_path else None)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    runio.write_text_atomic(out, render_page(blob, cfg))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sparc.core.results_page",
                                 description="Build the interactive results page for a finished core run.")
    ap.add_argument("run_dir")
    ap.add_argument("config")
    ap.add_argument("--out", default=None, help="output HTML (default: <run dir>/results.html)")
    ap.add_argument("--placebo", default=None, help="placebo.json of a placebo study")
    args = ap.parse_args(argv)
    run = Path(args.run_dir)
    out = build_results_page(run, args.config, Path(args.out) if args.out else run / "results.html",
                             placebo_path=args.placebo)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0
