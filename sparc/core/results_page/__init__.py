"""The standalone interactive results page of a finished core run (SPEC §6.9).

    python -m sparc.core.results_page <run dir> <core config> [--out results.html] [--placebo placebo.json]

:func:`build_results_page` reads the run directory written by ``run_core`` (manifest.json, predictions.parquet,
response_*.parquet, scenario_deltas.parquet, allocation.parquet and the stage files), plus the optional
post-run outputs (planner/, emulator.npz/.json, placebo.json and the uncertainty / simcheck / multiverse
sections merged into the manifest), and writes one self-contained HTML page: a map explorer (temperature,
land cover, cooling per step, diminishing returns, scenarios, budget plan, future climate, people and hot days, a
paint-a-design tool and the test areas) plus charts and tables for accuracy vs distance, area of influence,
dose-response, scenarios, uncertainty, climate, the causal audit, the budget plan and the validation studies.
Everything outside Model details is in plain words, as SPARC Studio says them (``sparc/core/glossary.json``).

Nothing here is tied to one city:

* the data and the CV folds are rebuilt with :func:`sparc.core.baselines.load_run` (the checkpoint is never
  unpickled), and the held-out / buffer cells of each fold travel as one u8 mask per fold;
* ``causal``, ``cv_distance``, ``optimize``, ``influence``, ``baselines``, ``climate`` and ``scenarios`` fall
  back to their stage files when the manifest lacks them (runs stopped before the manifest was finalised);
* the map's layer catalogue (:func:`layer_catalog`, in plain words) and the variable names come from the config's
  physics roles and actionable levers;
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
           "lever_info", "glossary_payload", "TEMPLATE", "main"]

TEMPLATE = Path(__file__).resolve().parent / "template.html"

#: display names of the physics roles (the variable a role names gets this label), in plain words (the glossary's
#: lever measures: "paving", "surface reflectivity")
ROLE_NAMES = {"canopy": "Tree canopy", "impervious": "Paving", "albedo": "Surface reflectivity", "ndvi": "Greenness",
              "elevation": "Elevation", "water_distance": "Distance to water"}
#: short names used in layer titles ("Tree canopy: total effect of +1 pt, nearby included")
ROLE_SHORT = {"canopy": "Tree canopy", "impervious": "Paving", "albedo": "Reflectivity", "ndvi": "Greenness",
              "elevation": "Elevation", "water_distance": "Distance to water"}
#: nouns used in running text ("Planned canopy increase")
ROLE_NOUNS = {"canopy": "canopy", "impervious": "paving", "albedo": "reflectivity"}
#: map inputs shown in the Land cover theme, in this order (roles), with unit and decimals
INPUT_ROLES = (("canopy", "%", 0, "Share of the place under tree canopy."),
               ("impervious", "%", 0, "Share of the place that is paved or built on."),
               ("ndvi", "", 2, "How green the plants in the place look from space, on a 0–1 scale. In scenarios it "
                               "follows the tree-canopy and paving changes."),
               ("albedo", "", 3, "How much sunlight the surfaces reflect, on a 0–1 scale (0 soaks up all of it, 1 "
                                 "reflects all of it)."))
_UNIT_LABELS = {"degf": "°F", "degc": "°C", "f": "°F", "c": "°C", "k": "K", "kelvin": "K"}
_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
          10: "ten"}


#: fixed glossary sections the page reads (trust words, curve shapes, heat categories, lever names, pathways, periods)
GLOSSARY_SECTIONS = ("trust", "trustRanges", "shapes", "heat", "levers", "emissions", "periods")


def glossary_payload(template_text: str | None = None) -> dict:
    """``DATA.glossary``: the plain words of ``sparc/core/glossary.json`` (shared with SPARC Studio) that the page uses.

    Terms: every term id the template names as a quoted string (``infotip("r2")``, ``data-tip="coverage"``,
    ``["R²", "r2"]``), with its label, tip and technical name (the longer ``more`` text is left out); plus the
    fixed sections in :data:`GLOSSARY_SECTIONS`."""
    from sparc.core.glossary import glossary

    g = glossary()
    text = TEMPLATE.read_text(encoding="utf-8") if template_text is None else template_text
    used = set(re.findall(r"[\"']([a-z][a-z0-9_]*)[\"']", text))
    terms = {k: {f: v[f] for f in ("label", "tip", "technical") if v.get(f)}
             for k, v in (g.get("terms") or {}).items() if k in used}
    return {"version": g.get("version"), "terms": terms,
            **{s: g[s] for s in GLOSSARY_SECTIONS if isinstance(g.get(s), dict)}}


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
    """``2000`` → ``2 km`` (the size of the test blocks in running text); ``test`` when unknown."""
    return f"{_km(block_m)} km" if block_m else "test"


def _unit_word(unit: str, one: bool = False) -> str:
    """A lever unit as a planner reads it (the glossary's unit word): ``pp`` → ``pts`` (``pt`` for one)."""
    if unit == "pp":
        return "pt" if one else "pts"
    return unit


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
    base = f"Total planned cooling ({units}·places, added up over every place) as the budget grows."
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
        verdict = "so returns barely diminish at this scale: plenty of places that cool well remain."
    elif gain >= 0.5:
        verdict = (f"so returns diminish: each extra unit of budget buys about {gain:.0%} of the cooling a unit of "
                   "the smaller budget bought, because the best places are changed first.")
    elif gain > 0:
        verdict = (f"so returns diminish sharply: each extra unit of budget buys only about {gain:.0%} of the "
                   "cooling a unit of the smaller budget bought; most of the places that cool well are already in "
                   "the plan.")
    else:
        verdict = "so the extra budget buys no further cooling: every place that cools is already in the plan."
    return f"{base} {step} raises total cooling {r:.2f}×, {verdict}"


# ---------------------------------------------------------------------------
# the layer catalogue
# ---------------------------------------------------------------------------

def _hd_label(case: str) -> tuple[str, bool]:
    """``(label, is_today)`` of a planner hot-days case (``today``, ``ssp245_mid``, ``ssp585_late`` …), in the
    glossary's words: ``per summer, moderate emissions, mid-century (2041–2060)``."""
    from sparc.core.glossary import section

    if case == "today":
        return "per summer, today", True
    m = re.fullmatch(r"(ssp\d{3})_(mid|late)", case)
    if m:
        years = "2041-2060" if m.group(2) == "mid" else "2081-2100"
        em = (section("emissions").get(m.group(1)) or {}).get("label") or m.group(1).upper()
        period = (section("periods").get(years) or {}).get("label") or years.replace("-", "–")
        return f"per summer, {em}, {period}", False
    return f"per summer ({case.replace('_', ' ')})", False


def layer_catalog(cfg, layers: dict, *, units: str, background: float, block_m=None, n_folds: int = 5,
                  scen_layers=(), climate: bool = False, sites: bool = False, design: bool = False,
                  optimize_variable: str | None = None) -> list[dict]:
    """The map's layer list, generated from the config (roles, levers) and the layers present, in plain words.

    Each entry is ``{group, key, name, unit, scale, center?, d, desc, mult?, zeroBlank?}``; keys starting with
    ``__`` are computed in the page (future climate, logger sites, the design tool, the test areas)."""
    U = units
    roles = _roles(cfg)
    levers = lever_info(cfg)
    folds_word = _number_word(n_folds)
    out: list[dict] = []

    def add(**o) -> None:
        if o["key"].startswith("__") or o["key"] in layers:
            out.append(o)

    block = _block_text(block_m)
    add(group="Temperature", key="obs", name="Measured air temperature", unit=U, scale="div", center=background,
        d=1, desc="Air temperature measured on the hot afternoon. Colours compare each place with the city's typical "
                  "(median) temperature, the middle of the scale.")
    add(group="Temperature", key="pred", name="Predicted air temperature", unit=U, scale="div", center=background,
        d=1, desc=f"Each place is predicted by a model trained without the {block} block around it, so this is what "
                  "the model knows without local readings.")
    add(group="Temperature", key="err", name="Prediction miss (measured − predicted)", unit=U, scale="div", center=0,
        d=2, desc="Orange: warmer than predicted. Purple: cooler than predicted. Large patches of one colour are local "
                  "heat sources or cool spots the map layers do not capture.")
    add(group="Temperature", key="hw", name="Prediction range (±)", unit=U, scale="seq", d=2,
        desc="How far the 90% prediction range reaches either side of the prediction: 9 in 10 test readings fall "
             "within it.")
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
        add(group="Land cover", key=f"in_{var}", name=str(info["label"]), unit=_unit_word(info["unit"]), scale="seq",
            d=3 if info["albedo_like"] else 1, desc=f"Today's {info['noun']} (something a plan can change).")
    # cooling per step: the total effect (nearby included) per unit, and the effect on the spot for canopy, per lever
    for var, info in levers.items():
        step = "0.01" if info["albedo_like"] else "1"
        per = f"+{step}" + (f" {_unit_word(info['unit'], one=True)}" if info["unit"] else "")
        role = info["role"]
        reach = "added up over every place it reaches"
        if role == "canopy":
            desc = (f"The temperature change, {reach}, when this place gains 1 percentage point of tree canopy "
                    "(the cooler air spreads to the neighbours).")
        elif role == "impervious":
            desc = (f"The temperature change, {reach}, when this place gains 1 percentage point of paving. Removing "
                    "paving reverses the sign.")
        elif info["albedo_like"]:
            desc = f"The temperature change, {reach}, when this place's {info['noun']} rises by 0.01 on its 0–1 scale."
        else:
            desc = f"The temperature change, {reach}, when this place's {info['noun']} rises by {per.lstrip('+')}."
        if var == optimize_variable:
            desc += " This is what the budget plan ranks places by."
        add(group="Cooling per step", key=f"fp_{var}", name=f"{info['short']}: total effect of {per}, nearby included",
            unit=U, scale="div", center=0, d=4, desc=desc, **({"mult": 0.01} if info["albedo_like"] else {}))
        if role == "canopy":
            add(group="Cooling per step", key=f"own_{var}", name=f"{info['short']}: effect of {per} on the spot",
                unit=U, scale="div", center=0, d=5,
                desc="The change in this place's own temperature only, with its neighbours left as they are. Much "
                     "smaller than the total: tree canopy cools its surroundings more than itself.")
    for var, info in levers.items():
        nm = info["short"] + (" removed" if info["direction"] == "decrease" else " added")
        unit = _unit_word(info["unit"])
        add(group="Diminishing returns", key=f"cls_{var}", name=f"{nm}: how the cooling builds up", scale="cat",
            desc="Each place's cooling, sorted by how it builds up as more is added. Diminishing returns: each extra "
                 "step cools less than the last. Steady returns: each step cools about the same. Slow start, then "
                 "strong: little at first, then a sharp gain. Grey: still cooling at the most the space allows (hit "
                 "the limit), or too little room to tell.")
        # per +0.01 for a 0–1 lever (as its total-effect layer), per +1 unit otherwise
        add(group="Diminishing returns", key=f"marg_{var}", name=f"{nm}: cooling from the first step",
            unit=f"{U} per +0.01" if info["albedo_like"] else f"{U} per {_unit_word(info['unit'], one=True) or 'unit'}",
            scale="seq", d=5 if info["albedo_like"] else 3,
            desc="How much the first step of change would cool here. High values are the best places to act first.",
            **({"mult": 0.01} if info["albedo_like"] else {}))
        add(group="Diminishing returns", key=f"A_{var}", name=f"{nm}: most cooling possible", unit=U, scale="seq", d=2,
            desc="The most cooling this place can get from the change, where its curve levels off (for places with "
                 "steady returns, the cooling at the largest amount tested).")
        add(group="Diminishing returns", key=f"d90_{var}", name=f"{nm}: amount for 90% of the cooling", unit=unit,
            scale="seq", d=info["dose_decimals"],
            desc="The amount of change (averaged over the neighbourhood) by which this place gets 90% of all the "
                 "cooling it can get. Small values mean the benefit levels off quickly.")
    for s in scen_layers:
        add(group="Scenarios", key=s["key"], name=s["name"], unit=U, scale="div", center=0, d=2,
            desc=f"The change in afternoon air temperature at every place with this scenario (the average of the "
                 f"{folds_word} models trained on different test areas).")
    oinfo = levers.get(optimize_variable or "") or {}
    noun = oinfo.get("noun") or "lever"
    verb = "decrease" if oinfo.get("direction") == "decrease" else "increase"
    add(group="Budget plan", key="alloc_dose", name=f"Planned {noun} {verb}", unit=_unit_word(oinfo.get("unit") or ""),
        scale="seq", d=1, zeroBlank=True,
        desc=f"Where the budget plan {'removes' if verb == 'decrease' else 'adds'} {noun}, and how much. Places it "
             "leaves as they are are blank.")
    add(group="Budget plan", key="alloc_delta", name="The plan's temperature change", unit=U, scale="div", center=0,
        d=3, desc="The whole plan run through the full model at once, so neighbouring changes and limits all count "
                  "(the checked benefit).")
    if climate:
        add(group="Future climate", key="__clim_temp", name="Future air temperature", unit=U, scale="climtemp",
            center=background, d=1,
            desc="Today's measured temperature plus the warming the climate models expect (their middle answer) for "
                 "the chosen emissions future and period, plus the chosen scenario's change. Colours keep today's "
                 "scale, so warming shows as a shift to orange.")
        add(group="Future climate", key="__clim_thr", name="Places at or above the heat threshold", scale="climthr",
            desc="Places whose future temperature reaches the threshold chosen under Heat exposure (in the Future "
                 "climate section), for the chosen emissions future, period and scenario.")
    add(group="Heat risk", key="heat_index", name="Feels-like temperature, measured afternoon", unit="°F",
        scale="seq", d=1,
        desc="Air temperature and humidity combined into how hot it feels (the US National Weather Service heat "
             "index), with the dew point measured that afternoon. The dew point is about the same across the city, "
             "so the humidity of each place follows from its own temperature.")
    add(group="Heat risk", key="heat_cat", name="Heat-risk level", scale="heatcat",
        desc="The US National Weather Service levels of the feels-like temperature: Caution 80–90 °F (tiredness with "
             "long exposure), Extreme caution 90–103 °F (heat cramps and exhaustion possible), Danger 103–125 °F "
             "(heat stroke possible), Extreme danger 125 °F or more (heat stroke highly likely).")
    add(group="People & hot days", key="people", name="Residents per place", unit="people", scale="seq", d=1,
        desc="People living in each place, from Meta and CIESIN's High Resolution Settlement Layer (based on "
             "censuses, around the 2010s). Daytime numbers differ.")
    add(group="People & hot days", key="plantable", name="Room for more canopy", unit="pts", scale="seq", d=0,
        desc="Tree canopy a place could still gain: open green and bare land (ESA WorldCover 2021) plus a fifth of "
             "its built-up area for street trees and lots, never above 100% minus today's canopy. The budget plan "
             "stays within it.")
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
        name = f"Hot afternoons ≥ {t} {U} {label}"
        if today:
            if first_today:
                desc = (f"Summer days (June–August, 1995–2014 airport record) whose afternoon reaches {t} {U} in this "
                        "place: the airport's daily high plus this place's difference from the airport on the "
                        "measured afternoon.")
                first_today = False
            else:
                desc = f"As above for {t} {U}."
        else:
            desc = ("As today, with each climate model's summer warming added to the airport record (the middle "
                    "answer across models).")
        add(group="People & hot days", key=key, name=name, unit="days", scale="seq", d=1, desc=desc)
    if sites:
        add(group="People & hot days", key="__sites", name="Where to measure next", scale="sites",
            desc="Where new temperature loggers would best pin down the tree-canopy effect: places with little and "
                 "with a lot of tree canopy but similar paving, favouring places where models trained on different "
                 "test areas disagree most, at least 400 m apart.")
    if design:
        add(group="Paint a design", key="__design", name="Your design: quick estimate of the change", unit=U,
            scale="design", center=0, d=3,
            desc="Paint changes on the map with the brush; the quick estimate of the change in afternoon air "
                 "temperature updates as you go. It is an instant, approximate answer, checked against the full "
                 "model on random patches (see What cools). For citywide changes use the Scenarios theme.")
    add(group="Test areas", key="__folds", name="How the model was tested", scale="folds",
        desc="For the chosen test area: purple places are kept out of training and used to test the model, red "
             "places are left out of training as a gap around them, grey places train it. Every accuracy number on "
             "this page comes from this design.")
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
                         "labels": ["Hit the limit, or too little room to tell", "Diminishing returns", "Steady returns",
                                    "Slow start, then strong"]}
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
        "glossary": glossary_payload(),
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
        "{{CELL}}": f"place, {_fmt(qa.get('cell_m'), 0)} m across" if qa.get("cell_m") else "place",
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
