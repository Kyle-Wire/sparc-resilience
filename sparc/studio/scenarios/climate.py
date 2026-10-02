"""Climate × adaptation (SPEC §7.11, api.md §7.7): CMIP6 change factors × the observed field × scenarios.

The factors table (rows ``experiment, period, model, delta_K``) comes from, in order:

1. the run config's ``climate.table`` (resolved against its config dir);
2. the workspace CMIP6 cache for the run's site (``cmip6_<variable>_<lat>_<lon>.csv``, written by
   ``climate_stage`` and the ``input.cmip6`` job);
3. the per-model warming recorded in the run's ``climate.json`` (converted back to kelvin).

Without any of them the endpoints answer ``404 no_climate_factors`` with the ``input.cmip6`` action.
``explore`` is ``climate.summarize_projections(observed, factors, {name: Δ}, thresholds, to_units)`` with any
exact results, verified plans and configured scenarios as adaptations, plus residents at or above each
threshold (planner layers) under the chosen warming statistic (median, p10, p90 or one model).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from sparc.studio.errors import ApiError

log = logging.getLogger("sparc.studio.scenarios")

__all__ = ["load_factors", "factors_info", "explore", "to_units", "fetch_action"]


def to_units(ctx) -> float:
    return 1.8 if ctx.units.get("target") == "°F" else 1.0


def fetch_action(ctx) -> dict | None:
    if not ctx.project_id:
        return None
    return {"kind": "fetch_input", "label": "Fetch CMIP6 change factors", "method": "POST",
            "path": f"/api/projects/{ctx.project_id}/inputs/cmip6", "body": {}}


def load_factors(ctx, cache_dir: str | Path | None = None):
    """``(factors DataFrame, source path or label)`` or ``(None, None)``."""
    import pandas as pd

    cc = ctx.cfg_raw.get("climate") or {}
    cfg = ctx.cfg
    if cc.get("table") and cfg is not None:
        p = cfg.resolve_path(cc["table"])
        if p is not None and Path(p).is_file():
            return pd.read_csv(p), str(p)
    clim = ctx.manifest.get("climate") or {}
    site = clim.get("site") or {}
    var = cc.get("variable") or clim.get("variable") or "tasmax"
    if cache_dir is not None and site.get("lat") is not None:
        p = Path(cache_dir) / f"cmip6_{var}_{float(site['lat']):.3f}_{float(site['lon']):.3f}.csv"
        if p.is_file():
            return pd.read_csv(p), str(p)
    rows = []
    k = to_units(ctx)
    for proj in clim.get("projections") or []:
        for model, w in ((proj.get("warming") or {}).get("by_model") or {}).items():
            if w is not None:
                rows.append({"experiment": proj["experiment"], "period": proj["period"], "model": model,
                             "delta_K": float(w) / k})
    if rows:
        return pd.DataFrame(rows), "climate.json"
    return None, None


def _observed(ctx) -> np.ndarray:
    data = ctx.data
    if data is not None and getattr(data, "target_raw", None) is not None:
        return np.asarray(data.target_raw, dtype=np.float64)
    pred = ctx.predictions
    if pred is None or "target" not in pred:
        raise ApiError("output_missing", "the observed temperatures are not available",
                       detail={"output": "predictions", "produced_by": "stage:S2_S3", "expected_path": None})
    return pred["target"].to_numpy(np.float64)


def _thresholds(ctx, thresholds) -> list[float]:
    if thresholds:
        return [float(t) for t in thresholds]
    t = (ctx.cfg_raw.get("climate") or {}).get("thresholds")
    if t:
        return [float(x) for x in t]
    return [90.0, 95.0] if ctx.units.get("target") == "°F" else [32.0, 35.0]


def factors_info(ctx, cache_dir=None) -> dict:
    """``GET /api/runs/{rid}/climate/factors``."""
    from sparc.core.climate import summarize_projections

    f, src = load_factors(ctx, cache_dir)
    if f is None or not len(f):
        return {"present": False, "path": None, "experiments": [], "periods": [], "models": [], "warming": [],
                "action": fetch_action(ctx)}
    obs = _observed(ctx)
    s = summarize_projections(obs, f, {}, _thresholds(ctx, None), to_units=to_units(ctx))
    warming = [{"experiment": p["experiment"], "label": p["label"], "period": p["period"], "n_models": p["n_models"],
                **{k: p["warming"][k] for k in ("median", "p10", "p90", "min", "max", "by_model")}}
               for p in s["projections"]]
    return {"present": True, "path": src, "experiments": sorted(f["experiment"].astype(str).unique()),
            "periods": sorted(f["period"].astype(str).unique()), "models": sorted(f["model"].astype(str).unique()),
            "warming": warming, "action": None}


def _stat_value(w: dict, statistic: Any) -> float | None:
    if isinstance(statistic, dict):
        return (w.get("by_model") or {}).get(statistic.get("model"))
    return w.get(statistic or "median")


def explore(db, ctx, body: dict, cache_dir=None) -> dict:
    """``POST /api/runs/{rid}/climate/explore``."""
    from sparc.core.climate import summarize_projections
    from sparc.studio.runs import layers as L
    from sparc.studio.scenarios.compare import resolve_item

    f, _src = load_factors(ctx, cache_dir)
    if f is None or not len(f):
        raise ApiError("no_climate_factors", "this run has no CMIP6 change factors", action=fetch_action(ctx))
    exps, pers = body.get("experiments"), body.get("periods")
    if exps:
        f = f[f["experiment"].astype(str).isin([str(e) for e in exps])]
    if pers:
        f = f[f["period"].astype(str).isin([str(p) for p in pers])]
    obs = _observed(ctx)
    thr = _thresholds(ctx, body.get("thresholds"))
    adapt: dict[str, np.ndarray] = {}
    for ref in body.get("adaptations") or []:
        it = resolve_item(ctx, db, ref)
        if it.get("zero"):
            continue
        name = it["label"]
        while name in adapt:
            name += " ′"
        adapt[name] = np.asarray(it["delta"], dtype=np.float64)
    k = to_units(ctx)
    out = summarize_projections(obs, f, adapt, thr, to_units=k)
    statistic = body.get("statistic") or "median"
    for p in out["projections"]:
        p["warming"]["selected"] = _stat_value(p["warming"], statistic)
    people_exposure = None
    lay = L.people_layers(ctx) if ctx.data is not None else None
    if lay is not None and "people" in lay:
        people = np.nan_to_num(lay["people"].to_numpy(float))
        total = max(float(people.sum()), 1e-9)
        people_exposure = []

        def row(exp, label, period, name, w, delta):
            t = obs + (w or 0.0) + (0.0 if delta is None else delta)
            return {"experiment": exp, "label": label, "period": period, "variant": name, "warming": w,
                    "people_ge": {f"{th:g}": float(people[t >= th].sum()) for th in thr},
                    "share_people_ge": {f"{th:g}": float(people[t >= th].sum() / total) for th in thr}}

        people_exposure.append(row(None, "today", None, "no adaptation", 0.0, None))
        for name, d in adapt.items():
            people_exposure.append(row(None, "today", None, name, 0.0, d))
        for p in out["projections"]:
            w = p["warming"]["selected"]
            if w is None:
                continue
            people_exposure.append(row(p["experiment"], p["label"], p["period"], "no adaptation", w, None))
            for name, d in adapt.items():
                people_exposure.append(row(p["experiment"], p["label"], p["period"], name, w, d))
    from sparc.studio.runs.common import clean

    return clean({"present": True, "thresholds": out["thresholds"], "projections": out["projections"],
                  "adaptation": out["adaptation"], "people_exposure": people_exposure,
                  "units": ctx.units.get("target", "°F"), "statistic": statistic, "present_today": out["present"]})
