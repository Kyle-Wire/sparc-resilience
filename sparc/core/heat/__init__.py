"""Heat stress and plain-language verdicts for a finished run.

Air temperature alone understates danger: the body sheds heat by sweating,
which humidity slows. The US National Weather Service heat index (Rothfusz
regression with its low- and high-humidity adjustments) turns air
temperature and relative humidity into an apparent temperature, and the NWS
groups it into four risk categories:

=================  ==========================  =================================================
category           heat index (°F)             what the NWS says can happen
=================  ==========================  =================================================
Caution            80 to < 90                  fatigue with prolonged exposure or activity
Extreme caution    90 to < 103                 heat cramps and heat exhaustion possible
Danger             103 to < 125                heat cramps or exhaustion likely, heat stroke possible
Extreme danger     ≥ 125                       heat stroke highly likely
=================  ==========================  =================================================

Within one city and one afternoon, moisture is close to uniform, so the
campaign's measured dewpoint (the airport station's, else ERA5's) applies to
every cell and each cell's relative humidity follows from its own
temperature. Future humidity is uncertain, so futures are given as a band
between two standard assumptions: **constant dewpoint** (moisture unchanged;
relative humidity falls as it warms; the lower bound) and **constant relative
humidity** (moisture rises with warming; the upper bound).

:func:`effect_verdict` turns a scenario's uncertainty components (from
``sparc core uncertainty``) into one of "Robust", "Direction only" or "Not
established", with the reasons, so a fragile number is never read as a solid
one.

This package sits below ``sparc/core`` so it is not part of the core code
fingerprint (it reads finished runs and never changes how a run is fitted).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

CATEGORIES: tuple[tuple[str, str, float, float], ...] = (
    ("below", "Below caution", -math.inf, 80.0),
    ("caution", "Caution", 80.0, 90.0),
    ("extreme_caution", "Extreme caution", 90.0, 103.0),
    ("danger", "Danger", 103.0, 125.0),
    ("extreme_danger", "Extreme danger", 125.0, math.inf),
)
CATEGORY_NOTES = {
    "below": "no heat-index risk category",
    "caution": "fatigue possible with prolonged exposure or activity",
    "extreme_caution": "heat cramps and heat exhaustion possible",
    "danger": "heat cramps or heat exhaustion likely; heat stroke possible with prolonged exposure",
    "extreme_danger": "heat stroke highly likely",
}
HUMIDITY_ASSUMPTIONS = ("constant_dewpoint", "constant_rh")
SOURCE = "US National Weather Service heat index (Rothfusz 1990 regression with NWS adjustments)"


# --------------------------------------------------------------------------- #
# Units and humidity                                                           #
# --------------------------------------------------------------------------- #
def is_fahrenheit(units: str | None) -> bool:
    u = (units or "").lower().replace("°", "").replace("deg", "").strip()
    return u in ("f", "fahrenheit", "degf")


def to_f(t, units: str | None) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    return t if is_fahrenheit(units) else t * 9.0 / 5.0 + 32.0


def delta_to_f(dt, units: str | None) -> np.ndarray:
    """A temperature *difference* in the target's units, in °F."""
    dt = np.asarray(dt, dtype=float)
    return dt if is_fahrenheit(units) else dt * 9.0 / 5.0


def f_to_c(t_f) -> np.ndarray:
    return (np.asarray(t_f, dtype=float) - 32.0) * 5.0 / 9.0


def rh_from_dewpoint(t_c, td_c) -> np.ndarray:
    """Relative humidity (%) from air temperature and dewpoint (°C), Magnus formula (as ``forcing``)."""
    t_c, td_c = np.asarray(t_c, dtype=float), np.asarray(td_c, dtype=float)

    def es(t):
        return 6.112 * np.exp(17.62 * t / (243.12 + t))

    return np.clip(100.0 * es(td_c) / es(t_c), 0.0, 100.0)


def heat_index_f(t_f, rh) -> np.ndarray:
    """NWS heat index (°F) for air temperature ``t_f`` (°F) and relative humidity ``rh`` (%), elementwise.

    Uses Steadman's simple form below 80 °F and the Rothfusz regression with
    the NWS low- and high-humidity adjustments above it, as the NWS heat index
    calculator does (and ``sparc.core.forcing.heat_index_F`` for one value).
    """
    t = np.asarray(t_f, dtype=float)
    r = np.asarray(rh, dtype=float)
    t, r = np.broadcast_arrays(t, r)
    simple = 0.5 * (t + 61.0 + (t - 68.0) * 1.2 + r * 0.094)
    full = (-42.379 + 2.04901523 * t + 10.14333127 * r - 0.22475541 * t * r - 6.83783e-3 * t ** 2
            - 5.481717e-2 * r ** 2 + 1.22874e-3 * t ** 2 * r + 8.5282e-4 * t * r ** 2 - 1.99e-6 * t ** 2 * r ** 2)
    low = (r < 13) & (t >= 80) & (t <= 112)
    full = np.where(low, full - ((13 - r) / 4.0) * np.sqrt(np.clip((17 - np.abs(t - 95.0)) / 17.0, 0, None)), full)
    high = (r > 85) & (t >= 80) & (t <= 87)
    full = np.where(high, full + ((r - 85) / 10.0) * ((87 - t) / 5.0), full)
    use_full = ((simple + t) / 2.0) >= 80.0
    return np.where(use_full, full, simple)


def category_codes(hi_f) -> np.ndarray:
    """0 = below caution … 4 = extreme danger, per cell."""
    hi = np.asarray(hi_f, dtype=float)
    edges = [c[3] for c in CATEGORIES[:-1]]            # 80, 90, 103, 125
    return np.digitize(hi, edges, right=False).astype(np.int8)


# --------------------------------------------------------------------------- #
# Campaign humidity                                                            #
# --------------------------------------------------------------------------- #
def campaign_dewpoint(forcing: dict | None) -> tuple[float | None, str | None]:
    """The campaign's dewpoint (°C) and where it came from: the airport station's, else ERA5's."""
    f = forcing or {}
    st = f.get("station") or {}
    if st.get("td_C") is not None:
        name = st.get("name") or st.get("station") or "station"
        return float(st["td_C"]), f"measured at {name} during the campaign hours"
    e5 = f.get("era5") or {}
    if e5.get("d2m_C") is not None:
        return float(e5["d2m_C"]), "ERA5 reanalysis for the campaign hours"
    return None, None


def load_forcing(cfg_raw: dict | None, config_dir: str | Path | None = None) -> dict | None:
    """The forcing JSON named by ``physics.forcing`` (path relative to the config folder), if readable."""
    ref = ((cfg_raw or {}).get("physics") or {}).get("forcing")
    if not ref or not isinstance(ref, str):
        return None
    p = Path(ref)
    # the config folder recorded in the run may have moved (a copied run): also try the working directory
    candidates = [p] if p.is_absolute() else (
        ([Path(config_dir) / p] if config_dir else []) + [Path.cwd() / p, Path.cwd() / "configs" / p])
    for c in candidates:
        try:
            return json.loads(c.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    return None


# --------------------------------------------------------------------------- #
# Heat risk                                                                    #
# --------------------------------------------------------------------------- #
def heat_index_today(temp, units: str | None, dewpoint_c: float) -> np.ndarray:
    """Per-cell heat index (°F) on the campaign afternoon."""
    t_f = to_f(temp, units)
    return heat_index_f(t_f, rh_from_dewpoint(f_to_c(t_f), dewpoint_c))


def _case(label: str, t_f: np.ndarray, rh: np.ndarray, people: np.ndarray | None, adapted: bool,
          humidity: str, warming_f: float | None) -> dict:
    hi = heat_index_f(t_f, rh)
    codes = category_codes(hi)
    row = {"case": label, "adapted": adapted, "humidity": humidity, "warming_F": warming_f,
           "cell_mean_hi": float(np.mean(hi)), "max_hi": float(np.max(hi)),
           "cells": {c[0]: int(np.sum(codes == i)) for i, c in enumerate(CATEGORIES)}}
    if people is not None:
        p = np.nan_to_num(np.asarray(people, dtype=float))
        tot = max(float(p.sum()), 1e-9)
        row["person_mean_hi"] = float(np.sum(p * hi) / tot)
        row["people"] = {c[0]: float(p[codes == i].sum()) for i, c in enumerate(CATEGORIES)}
        row["share_people"] = {k: v / tot for k, v in row["people"].items()}
        row["people_extreme_caution_or_worse"] = float(p[codes >= 2].sum())
        row["people_danger_or_worse"] = float(p[codes >= 3].sum())
    return row


def heat_risk(temp, units: str | None, dewpoint_c: float, *, people=None,
              futures: dict[str, float] | None = None, adaptation=None) -> dict:
    """Heat-index exposure today and in each future, with and without ``adaptation``.

    ``temp`` and ``adaptation`` (a per-cell change) are in the target's units;
    ``futures`` maps a label to the warming in the target's units (e.g. the
    CMIP6 median). Futures carry both humidity assumptions.
    """
    t_f = to_f(temp, units)
    rh_now = rh_from_dewpoint(f_to_c(t_f), dewpoint_c)
    a_f = delta_to_f(adaptation, units) if adaptation is not None else None
    cases = [_case("today", t_f, rh_now, people, False, "observed", 0.0)]
    if a_f is not None:
        t1 = t_f + a_f
        cases.append(_case("today", t1, rh_from_dewpoint(f_to_c(t1), dewpoint_c), people, True, "observed", 0.0))
    for label, w in (futures or {}).items():
        w_f = float(delta_to_f(w, units))
        for adapted in ((False, True) if a_f is not None else (False,)):
            t1 = t_f + w_f + (a_f if adapted else 0.0)
            for hum in HUMIDITY_ASSUMPTIONS:
                rh = rh_from_dewpoint(f_to_c(t1), dewpoint_c) if hum == "constant_dewpoint" else rh_now
                cases.append(_case(label, t1, rh, people, adapted, hum, w_f))
    return {
        "source": SOURCE,
        "dewpoint_C": float(dewpoint_c),
        "rh_today_range": [float(np.min(rh_now)), float(np.max(rh_now))],
        "categories": [{"id": c[0], "label": c[1], "lo_F": None if math.isinf(c[2]) else c[2],
                        "hi_F": None if math.isinf(c[3]) else c[3], "note": CATEGORY_NOTES[c[0]]}
                       for c in CATEGORIES],
        "humidity_assumptions": {
            "constant_dewpoint": "moisture unchanged as it warms (relative humidity falls): lower bound",
            "constant_rh": "relative humidity unchanged (moisture rises with warming): upper bound"},
        "cases": cases,
    }


def run_heat_risk(run_dir, *, people=None, package: str | None = None, config_dir=None) -> dict | None:
    """:func:`heat_risk` for a finished run directory (``None`` when the run has no campaign humidity).

    Reads the manifest's config (``physics.forcing``), the observed target from
    ``predictions.parquet``, CMIP6 futures from ``climate.json`` (median and
    10th–90th percentile warming) and the package from ``scenario_deltas.parquet``.
    """
    import pandas as pd

    run_dir = Path(run_dir)
    try:
        m = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    cfg_raw = m.get("config") or {}
    cdir = config_dir or (m.get("provenance") or {}).get("config_dir")
    forcing = load_forcing(cfg_raw, cdir)
    td, td_src = campaign_dewpoint(forcing)
    if td is None:
        return None
    pred_p = run_dir / "predictions.parquet"
    if not pred_p.exists():
        return None
    pred = pd.read_parquet(pred_p)
    col = "target" if "target" in pred else ("y" if "y" in pred else None)
    if col is None:
        return None
    temp = pred[col].to_numpy(float)
    units = ((cfg_raw.get("data") or {}).get("target_units") or (m.get("climate") or {}).get("units")
             or m.get("target_units") or "degF")
    clim = {}
    if (run_dir / "climate.json").exists():
        try:
            clim = json.loads((run_dir / "climate.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            clim = {}
    futures, spread = {}, {}
    for p in clim.get("projections") or []:
        w = p.get("warming") or {}
        if w.get("median") is None:
            continue
        label = f"{p.get('label') or p.get('experiment')} {p['period']}"
        futures[label] = float(w["median"])
        spread[label] = [w.get("p10"), w.get("p90")]
    adapt = None
    if (run_dir / "scenario_deltas.parquet").exists():
        deltas = pd.read_parquet(run_dir / "scenario_deltas.parquet")
        joint = [j.get("name") for j in (cfg_raw.get("joint_scenarios") or [])]
        pkg = package or (joint[0] if joint else None)
        if pkg and pkg in deltas and len(deltas) == len(temp):
            adapt = deltas[pkg].to_numpy(float)
            package = pkg
    if people is not None and len(people) != len(temp):
        people = None
    out = heat_risk(temp, units, td, people=people, futures=futures, adaptation=adapt)
    out.update({"dewpoint_source": td_src, "package": package if adapt is not None else None,
                "warming_spread": spread, "units": units})
    return out


# --------------------------------------------------------------------------- #
# Verdicts                                                                     #
# --------------------------------------------------------------------------- #
VERDICTS = {
    "robust": "Robust",
    "direction": "Direction only",
    "not_established": "Not established",
    "unknown": "Not assessed",
}


def effect_verdict(row: dict, *, sign_floor: float = 0.9, extrapolation_ceiling: float = 0.2,
                   extrapolation_limit: float = 0.5) -> dict:
    """One plain verdict for a scenario's uncertainty row (``uncertainty.json`` ``scenarios[]``).

    * **Not established**: the change is indistinguishable from what the
      pipeline reports when the simulation plants no effect (null artefact),
      or neither the estimation interval nor the analysis-choice variants
      agree on its sign.
    * **Robust**: the envelope (estimation ∪ specification ∪ attribution)
      excludes zero and the sign holds across analysis choices.
    * **Direction only**: the sign is stable but the plausible range still
      reaches zero.

    Extrapolation (more than ``extrapolation_ceiling`` of edited cells beyond
    the observed range) and disagreement with the causal check are reported
    as qualifiers, not hidden. When most edited cells (more than
    ``extrapolation_limit``) are beyond the observed range the model is
    extrapolating, and the change is not established whatever its interval.
    """
    reasons: list[str] = []
    qualifiers: list[str] = []
    est = row.get("estimate")
    if est is None:
        return {"verdict": "unknown", "label": VERDICTS["unknown"], "reasons": ["no estimate"], "qualifiers": []}
    nulls = [a for a in row.get("null_artifact") or [] if a.get("product", "rf") == "rf"]
    sign = row.get("sign_stability")
    e95 = row.get("estimation_95")
    env = row.get("envelope")
    e95_excl = bool(e95 and (e95[1] < 0 or e95[0] > 0))
    env_excl = bool(env and (env[1] < 0 or env[0] > 0))
    sign_ok = sign is None or sign >= sign_floor
    fx = row.get("frac_extrapolated")
    if fx is not None and fx > extrapolation_ceiling:
        qualifiers.append(f"{fx:.0%} of edited cells are beyond observed conditions")
    if row.get("model_within_causal") is False:
        qualifiers.append("outside the independent causal check's band")
    if fx is not None and fx > extrapolation_limit:
        reasons.append(f"{fx:.0%} of edited cells are beyond the conditions the model was trained on")
        verdict = "not_established"
    elif nulls and not all(a.get("distinguishable") for a in nulls):
        a = nulls[0]
        reasons.append(f"a simulation with no planted effect produces a similar change ({a['delta']:+.2f})")
        verdict = "not_established"
    elif env_excl and sign_ok:
        reasons.append("its plausible range excludes zero" + (
            f" and every analysis variant agrees on the sign ({sign:.0%})" if sign is not None else ""))
        verdict = "robust"
    elif sign is not None and sign >= sign_floor:
        reasons.append(f"every analysis variant agrees on the sign ({sign:.0%}), but the plausible range reaches zero")
        verdict = "direction"
    elif e95_excl:
        reasons.append("the estimation interval excludes zero, but analysis choices were not checked"
                       if sign is None else "the estimation interval excludes zero, but analysis choices disagree")
        verdict = "direction"
    else:
        reasons.append("its estimation interval includes zero")
        verdict = "not_established"
    return {"verdict": verdict, "label": VERDICTS[verdict], "reasons": reasons, "qualifiers": qualifiers}


def verdicts_for_run(run_dir) -> dict[str, dict]:
    """``{scenario name: effect_verdict}`` from a run's ``uncertainty.json`` (empty when absent)."""
    p = Path(run_dir) / "uncertainty.json"
    try:
        u = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {r["scenario"]: effect_verdict(r) for r in u.get("scenarios") or [] if r.get("scenario")}
