"""The generated layer catalog and per-layer arrays of a run (SPEC §6.3, api.md §6.2).

Layers are generated **from the config and the files present** - nothing is hard-coded to one city.  Each
entry is a ``LayerMeta`` (api.md §1) whose values come from a run file, in run row order:

=================  ===========================================================================================
Group              Keys
=================  ===========================================================================================
temperature        ``obs``, ``pred``, ``resid``, ``halfwidth``, ``halfwidth_adaptive``, ``dist_train_m``,
                   ``oof_<model>``, ``resid_<model>`` (``predictions.parquet``)
inputs             every predictor (levers and physics roles first), ``zone`` (cat)
cv                 ``fold`` (cat); per-fold class arrays are served by ``/folds/{k}.bin``
effects            per lever: ``fp_``, ``own_``, ``own_sd_``, ``fp_sd_``, ``marg_``, ``A_``, ``d90_``, ``ds_``,
                   ``infl_``, ``fitr2_``, ``headroom_``, ``cls_`` (cat) from ``response_<var>.parquet``
causal             ``cate_<t>``, ``mslope_<t>``, ``mslope_own_<t>`` (``causal_cells.parquet``)
scenarios          ``sc:<slug>`` (``scenario_deltas.parquet``), ``sc_sd:<slug>``, ``sc_ex:<slug>``
                   (``scenario_detail.npz``)
budget             ``alloc_dose`` (zero_blank), ``alloc_delta`` (``allocation.parquet``)
planner            ``people``, ``people_60_plus``, ``people_under_5``, ``lc_*`` (``planner.layers``),
                   ``plantable_pp``, every ``hot_days_ge_*`` column (``planner/planner_cells.parquet``)
studio_results     ``res:<res_id>:{delta, delta_sd, extrapolation, realized_<var>, abs}``
studio_plans       ``plan:<plid>:{dose, planned_benefit, closed_loop_delta}``
studio_compare     ``cmp:<cid>:<a>__<b>``
=================  ===========================================================================================

Studio layers read the on-disk formats of api.md §12.2 (``results/<res_id>/cells.parquet``,
``plans/<plid>/*.npy``, ``comparisons/<cid>/diff_<a>__<b>.npy``).  Values travel as little-endian Float32
(NaN = no data); categorical layers are Uint8 (255 = no data).  Arrays are cached in a 256 MB LRU.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from sparc.studio.errors import ApiError
from sparc.studio.runs.common import LRU, clean, file_stat, read_json_cached

log = logging.getLogger("sparc.studio.runs")

__all__ = ["LayerDef", "layer_defs", "layer_array", "layer_catalog", "layer_meta", "layer_etag", "layer_stats",
           "GROUP_LABELS", "CLS_LABELS", "resolve_layer", "lever_info"]

LAYER_CACHE = LRU(256 * 1024 ** 2)
GROUP_LABELS = OrderedDict([
    ("temperature", "Temperature"), ("inputs", "Inputs"), ("cv", "CV design"), ("effects", "Effects"),
    ("causal", "Causal"), ("scenarios", "Scenarios (configured)"), ("budget", "Budget"),
    ("planner", "Planner & people"), ("studio_results", "Studio results"), ("studio_plans", "Budget plans"),
    ("studio_compare", "Comparisons"),
])
CLS_LABELS = ["censored or too little headroom", "saturating", "linear", "S-shaped"]
_COOLER = "negative = cooler"
_BENEFIT = "positive = cooler"


@dataclass
class LayerDef:
    key: str
    group: str
    label: str
    unit: str
    scale: str
    fn: Callable[[], np.ndarray]
    source: dict | None = None
    center: float | str | None = None      # number, or "median" (the layer's own median)
    decimals: int = 2
    mult: float = 1.0
    zero_blank: bool = False
    labels: list[str] | None = None
    desc: str = ""
    sign_note: str | None = None
    dtype: str = "float32"
    sources: tuple[str, ...] = field(default_factory=tuple)    # files whose stat keys the ETag


def lever_info(ctx) -> dict[str, dict]:
    """``{var: {unit, label, direction, albedo_like, role}}`` of the actionable levers."""
    from sparc.core.catalog import unit_label

    raw = ctx.cfg_raw
    roles = {v: k for k, v in ((raw.get("physics") or {}).get("roles") or {}).items()}
    out = {}
    for var, spec in (raw.get("actionable") or {}).items():
        spec = spec or {}
        unit = unit_label(spec.get("unit")) or ""
        hi = spec.get("max")
        albedo_like = (roles.get(var) == "albedo") or (str(spec.get("unit") or "").lower() == "albedo") or (
            hi is not None and float(hi) <= 1.0)
        out[var] = {"unit": unit or "units", "label": spec.get("label") or var.replace("_", " "),
                    "direction": str(spec.get("direction", "increase")), "albedo_like": bool(albedo_like),
                    "role": roles.get(var)}
    return out


def _f32(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float32)


def _stat_key(ctx, rels) -> str:
    parts = []
    for r in rels:
        p = Path(r) if Path(r).is_absolute() else ctx.run_dir / r
        st = file_stat(p)
        parts.append(f"{r}:{st}")
    return "|".join(parts)


# ---------------------------------------------------------------------------
# the catalog
# ---------------------------------------------------------------------------

def layer_defs(ctx) -> "OrderedDict[str, LayerDef]":
    """Every layer of the run, in display order.

    The run's own layers are cached on the context; Studio layers (results, plans, comparisons) are listed
    afresh on every call, because they appear without the run's files changing."""
    base = ctx._get("layer_defs", lambda: _build_defs(ctx))
    out = OrderedDict(base)
    if ctx.n is not None:
        for d in _studio_layer_defs(ctx):
            out[d.key] = d
    return out


def _build_defs(ctx) -> "OrderedDict[str, LayerDef]":
    defs: "OrderedDict[str, LayerDef]" = OrderedDict()
    tu = ctx.units.get("target", "°F")
    n = ctx.n
    if n is None:
        return defs

    def add(d: LayerDef) -> None:
        defs[d.key] = d

    # ---------------------------------------------------------------- temperature
    pred = ctx.predictions
    if pred is not None:
        cols = set(pred.columns)

        def col(c):
            return lambda: _f32(ctx.predictions[c].to_numpy(float))

        if "target" in cols:
            add(LayerDef("obs", "temperature", "Observed temperature", tu, "div", col("target"),
                         {"file": "predictions.parquet", "column": "target"}, center="median", decimals=1,
                         desc="The measured temperature of each cell.", sources=("predictions.parquet",)))
        if "pred" in cols:
            add(LayerDef("pred", "temperature", "Predicted temperature (held out)", tu, "div", col("pred"),
                         {"file": "predictions.parquet", "column": "pred"}, center="median_obs", decimals=1,
                         desc="Out-of-fold stacked prediction: each cell predicted by a model that never saw its block.",
                         sources=("predictions.parquet",)))
        if {"target", "pred"} <= cols:
            add(LayerDef("resid", "temperature", "Residual (observed − predicted)", tu, "div",
                         lambda: _f32(ctx.predictions["target"].to_numpy(float) - ctx.predictions["pred"].to_numpy(float)),
                         {"file": "predictions.parquet", "column": "target-pred"}, center=0.0,
                         desc="Held-out error of the stack.", sign_note="positive = warmer than predicted",
                         sources=("predictions.parquet",)))
        if {"pi_hi", "pred"} <= cols:
            add(LayerDef("halfwidth", "temperature", "Interval half-width", tu, "seq",
                         lambda: _f32(ctx.predictions["pi_hi"].to_numpy(float) - ctx.predictions["pred"].to_numpy(float)),
                         {"file": "predictions.parquet", "column": "pi_hi-pred"},
                         desc="Half-width of the conformal prediction interval (global).",
                         sources=("predictions.parquet",)))
        if {"pi_hi_adaptive", "pred"} <= cols:
            add(LayerDef("halfwidth_adaptive", "temperature", "Interval half-width (distance-adaptive)", tu, "seq",
                         lambda: _f32(ctx.predictions["pi_hi_adaptive"].to_numpy(float)
                                      - ctx.predictions["pred"].to_numpy(float)),
                         {"file": "predictions.parquet", "column": "pi_hi_adaptive-pred"},
                         desc="Half-width of the interval that widens away from training data.",
                         sources=("predictions.parquet",)))
        if "dist_train_m" in cols:
            add(LayerDef("dist_train_m", "temperature", "Distance to training data", "m", "seq",
                         col("dist_train_m"), {"file": "predictions.parquet", "column": "dist_train_m"}, decimals=0,
                         desc="Distance from each cell to the nearest training point of its fold.",
                         sources=("predictions.parquet",)))
        for c in sorted(c for c in cols if c.startswith("oof_")):
            m = c[4:]
            add(LayerDef(f"oof_{m}", "temperature", f"Held-out prediction: {m}", tu, "div", col(c),
                         {"file": "predictions.parquet", "column": c}, center="median_obs", decimals=1,
                         desc=f"Out-of-fold prediction of the {m} base model.", sources=("predictions.parquet",)))
            if "target" in cols:
                add(LayerDef(f"resid_{m}", "temperature", f"Residual: {m}", tu, "div",
                             (lambda c=c: _f32(ctx.predictions["target"].to_numpy(float)
                                               - ctx.predictions[c].to_numpy(float))),
                             {"file": "predictions.parquet", "column": f"target-{c}"}, center=0.0,
                             sign_note="positive = warmer than predicted", sources=("predictions.parquet",)))

    # ---------------------------------------------------------------- inputs
    levers = lever_info(ctx)
    data = ctx.data
    if data is not None:
        raw = ctx.cfg_raw
        roles = (raw.get("physics") or {}).get("roles") or {}
        role_of = {v: k for k, v in roles.items()}
        cats = set((raw.get("encodings") or {}).get("categorical") or [])
        cols = list(data.frame.columns)
        order = [c for c in levers if c in cols] + [c for c in cols if c in role_of and c not in levers] + \
                [c for c in cols if c not in levers and c not in role_of]
        for c in order:
            lab = levers.get(c, {}).get("label") or c.replace("_", " ")
            unit = levers.get(c, {}).get("unit") or ("pp" if role_of.get(c) in ("canopy", "impervious") else "")
            if c in cats:
                vals = data.frame[c].to_numpy()
                levels = [str(v) for v in sorted(set(vals.tolist()), key=str)]
                if len(levels) <= 254:
                    look = {v: i for i, v in enumerate(levels)}
                    add(LayerDef(c, "inputs", lab, "", "cat",
                                 (lambda c=c, look=look: np.array([look.get(str(v), 255) for v in ctx.data.frame[c].to_numpy()],
                                                                  dtype=np.uint8)),
                                 {"file": "data", "column": c}, labels=levels, dtype="uint8", decimals=0))
                continue
            desc = f"Predictor {c}" + (f" (physics role {role_of[c]})" if c in role_of else "") + \
                   (" - an actionable lever" if c in levers else "")
            add(LayerDef(c, "inputs", lab, unit, "seq", (lambda c=c: _f32(ctx.data.frame[c].to_numpy(float))),
                         {"file": "data", "column": c}, desc=desc))
    g = ctx.grid
    if g is not None and g.zones and len(g.zones) <= 254:
        add(LayerDef("zone", "inputs", "Zone", "", "cat",
                     lambda: np.where(ctx.grid.zone >= 0, ctx.grid.zone, 255).astype(np.uint8),
                     {"file": "predictions.parquet", "column": "zone"}, labels=[str(z) for z in g.zones],
                     dtype="uint8", decimals=0, desc="Zone / neighbourhood code (reporting only).",
                     sources=("predictions.parquet",)))

    # ---------------------------------------------------------------- CV design
    if pred is not None and "fold" in pred.columns:
        k = int(np.nanmax(pred["fold"].to_numpy())) + 1 if len(pred) else 0
        add(LayerDef("fold", "cv", "CV fold", "", "cat",
                     lambda: np.asarray(ctx.predictions["fold"].to_numpy(), dtype=np.int64).clip(0, 255).astype(np.uint8),
                     {"file": "predictions.parquet", "column": "fold"}, labels=[f"Fold {i + 1}" for i in range(k)],
                     dtype="uint8", decimals=0, desc="Spatial CV fold that holds each cell out.",
                     sources=("predictions.parquet",)))
    elif ctx.folds is not None:
        k = ctx.folds.n_folds
        add(LayerDef("fold", "cv", "CV fold", "", "cat", lambda: ctx.folds.fold_id.astype(np.uint8), None,
                     labels=[f"Fold {i + 1}" for i in range(k)], dtype="uint8", decimals=0))

    # ---------------------------------------------------------------- effects per lever
    for var in ctx.response_vars():
        rel = f"response_{var}.parquet"
        info = levers.get(var) or {"unit": "units", "label": var, "albedo_like": False}
        lu = info["unit"]
        mult = 0.01 if info.get("albedo_like") else 1.0
        per = f"per +0.01 {lu}" if mult != 1.0 else f"per {lu}"
        df = ctx.parquet(rel)
        if df is None:
            continue
        cols = set(df.columns)

        def rc(c, rel=rel):
            return lambda: _f32(ctx.parquet(rel)[c].to_numpy(float))

        spec = [
            (f"fp_{var}", "footprint_effect_per_unit", f"Cooling footprint: {info['label']}", f"{tu}·cells {per}", "div",
             0.0, mult, "Total ΔT across the neighbourhood when this cell's lever rises by one unit.", _COOLER),
            (f"own_{var}", "own_effect_per_unit", f"Own-cell effect: {info['label']}", f"{tu} {per}", "div", 0.0, mult,
             "ΔT of the cell itself per unit of its own lever.", _COOLER),
            (f"own_sd_{var}", "own_effect_sd", f"Own-cell effect SD: {info['label']}", f"{tu} {per}", "seq", None, mult,
             "Fold-to-fold spread of the own-cell effect.", None),
            (f"fp_sd_{var}", "footprint_effect_sd", f"Footprint SD: {info['label']}", f"{tu}·cells {per}", "seq", None,
             mult, "Fold-to-fold spread of the footprint effect.", None),
            (f"marg_{var}", "marginal_benefit_per_unit", f"Marginal cooling: {info['label']}", f"{tu} {per}", "seq",
             None, mult, "Cooling per extra unit at today's level (neighbourhood adoption).", _BENEFIT),
            (f"A_{var}", "max_cooling_A", f"Maximum cooling: {info['label']}", tu, "seq", None, 1.0,
             "Ceiling of the fitted dose-response curve.", _BENEFIT),
            (f"d90_{var}", "d90", f"Dose for 90% of the cooling: {info['label']}", lu, "seq", None, 1.0,
             "Neighbourhood dose reaching 90% of the maximum cooling (within the tested doses).", None),
            (f"ds_{var}", "saturation_scale_ds", f"Saturation scale: {info['label']}", lu, "seq", None, 1.0,
             "Dose scale d_s of a saturating curve.", None),
            (f"infl_{var}", "inflection_dose", f"Inflection dose: {info['label']}", lu, "seq", None, 1.0,
             "Inflection of an S-shaped curve.", None),
            (f"fitr2_{var}", "fit_r2", f"Curve fit R²: {info['label']}", "1", "seq", None, 1.0,
             "R² of the fitted dose-response curve.", None),
            (f"headroom_{var}", "headroom", f"Headroom: {info['label']}", lu, "seq", None, 1.0,
             "Room to the lever's bound in its direction.", None),
        ]
        for key, c, label, unit, scale, center, m, desc, sign in spec:
            if c in cols:
                add(LayerDef(key, "effects", label, unit, scale, rc(c), {"file": rel, "column": c}, center=center,
                             mult=m, desc=desc, sign_note=sign, decimals=3 if scale == "div" else 2, sources=(rel,)))
        if "curve_model" in cols:
            add(LayerDef(f"cls_{var}", "effects", f"Curve shape: {info['label']}", "", "cat",
                         (lambda rel=rel: _curve_class(ctx.parquet(rel))), {"file": rel, "column": "curve_model"},
                         labels=list(CLS_LABELS), dtype="uint8", decimals=0, sources=(rel,),
                         desc="Fitted dose-response shape per cell (grey: censored or too little headroom)."))

    # ---------------------------------------------------------------- causal
    cc = ctx.parquet("causal_cells.parquet")
    if cc is not None:
        for c in cc.columns:
            m = re.match(r"^(cate|mslope_own|mslope):(.+)$", c)
            if not m:
                continue
            kind, t = m.groups()
            info = levers.get(t) or {"unit": "units", "label": t, "albedo_like": False}
            mult = 0.01 if info.get("albedo_like") else 1.0
            per = f"per +0.01 {info['unit']}" if mult != 1.0 else f"per {info['unit']}"
            label = {"cate": f"Causal effect (CATE): {info['label']}",
                     "mslope": f"Model adoption slope: {info['label']}",
                     "mslope_own": f"Model own-cell slope: {info['label']}"}[kind]
            desc = {"cate": "R-learner conditional average effect per unit of the treatment.",
                    "mslope": "The model's per-cell ΔT per unit when the whole area adopts the lever.",
                    "mslope_own": "The model's own-cell slope per unit."}[kind]
            add(LayerDef(f"{kind}_{t}", "causal", label, f"{tu} {per}", "div",
                         (lambda c=c: _f32(ctx.parquet("causal_cells.parquet")[c].to_numpy(float))),
                         {"file": "causal_cells.parquet", "column": c}, center=0.0, mult=mult, decimals=3,
                         desc=desc, sign_note=_COOLER, sources=("causal_cells.parquet",)))

    # ---------------------------------------------------------------- scenarios (configured)
    sd = ctx.parquet("scenario_deltas.parquet")
    if sd is not None:
        detail = ctx.scenario_detail()
        for s in ctx.configured_scenarios():
            nm, slug = s["name"], s["slug"]
            if nm not in sd.columns:
                continue
            add(LayerDef(f"sc:{slug}", "scenarios", nm, tu, "div",
                         (lambda nm=nm: _f32(ctx.parquet("scenario_deltas.parquet")[nm].to_numpy(float))),
                         {"file": "scenario_deltas.parquet", "column": nm}, center=0.0, decimals=2,
                         desc=f"ΔT of the configured scenario “{nm}”.", sign_note=_COOLER,
                         sources=("scenario_deltas.parquet",)))
            if detail and nm in detail["sd"]:
                add(LayerDef(f"sc_sd:{slug}", "scenarios", f"{nm}: fold SD", tu, "seq",
                             (lambda nm=nm: _f32(ctx.scenario_detail()["sd"][nm])),
                             {"file": "scenario_detail.npz", "column": f"sd ({nm})"}, decimals=3,
                             desc="Fold-to-fold spread of ΔT.", sources=("scenario_detail.npz",)))
            if detail and nm in detail["ex"]:
                add(LayerDef(f"sc_ex:{slug}", "scenarios", f"{nm}: extrapolation", "1", "seq",
                             (lambda nm=nm: _f32(ctx.scenario_detail()["ex"][nm])),
                             {"file": "scenario_detail.npz", "column": f"ex ({nm})"}, decimals=2,
                             desc="Extrapolation score (> 1 = outside observed conditions).",
                             sources=("scenario_detail.npz",)))

    # ---------------------------------------------------------------- budget
    al = ctx.parquet("allocation.parquet")
    if al is not None:
        var = (ctx.cfg_raw.get("optimize") or {}).get("variable")
        lu = (levers.get(var) or {}).get("unit", "units")
        if "dose" in al.columns:
            add(LayerDef("alloc_dose", "budget", f"Planned dose{': ' + var if var else ''}", lu, "seq",
                         lambda: _f32(ctx.parquet("allocation.parquet")["dose"].to_numpy(float)),
                         {"file": "allocation.parquet", "column": "dose"}, zero_blank=True, decimals=1,
                         desc="Budget allocation of S7 (cells without a dose are blank).",
                         sources=("allocation.parquet",)))
        if "closed_loop_delta" in al.columns:
            add(LayerDef("alloc_delta", "budget", "Allocation ΔT (closed loop)", tu, "div",
                         lambda: _f32(ctx.parquet("allocation.parquet")["closed_loop_delta"].to_numpy(float)),
                         {"file": "allocation.parquet", "column": "closed_loop_delta"}, center=0.0,
                         desc="ΔT of the whole allocation run through the model.", sign_note=_COOLER,
                         sources=("allocation.parquet",)))

    # ---------------------------------------------------------------- planner & people
    lay = people_layers(ctx)
    if lay is not None:
        for c in lay.columns:
            if c in ("id", "x_m", "y_m"):
                continue
            label = {"people": "Residents", "people_60_plus": "Residents aged 60+",
                     "people_under_5": "Residents under 5"}.get(c) or ("Land cover: " + c[3:] if c.startswith("lc_")
                                                                       else c)
            unit = "people" if c.startswith("people") else ("share" if c.startswith("lc_") else "")
            add(LayerDef(c, "planner", label, unit, "seq",
                         (lambda c=c: _f32(people_layers(ctx)[c].to_numpy(float))),
                         {"file": str((ctx.cfg_raw.get("planner") or {}).get("layers")), "column": c},
                         decimals=2 if c.startswith("lc_") else 1))
    pc = ctx.parquet("planner/planner_cells.parquet")
    if pc is not None and "plantable_canopy_pp" in pc.columns:
        add(LayerDef("plantable_pp", "planner", "Plantable canopy headroom", "pp", "seq",
                     lambda: _aligned(ctx, ctx.parquet("planner/planner_cells.parquet"), "plantable_canopy_pp"),
                     {"file": "planner/planner_cells.parquet", "column": "plantable_canopy_pp"}, decimals=1,
                     sources=("planner/planner_cells.parquet",)))
    elif lay is not None and data is not None:
        can = ((ctx.cfg_raw.get("physics") or {}).get("roles") or {}).get("canopy")
        if can and can in data.frame.columns:
            add(LayerDef("plantable_pp", "planner", "Plantable canopy headroom", "pp", "seq",
                         lambda: _plantable(ctx), {"file": "planner.layers", "column": "plantable"}, decimals=1))
    if pc is not None:
        for c in pc.columns:
            if c.startswith("hot_days_ge_"):
                add(LayerDef(c, "planner", "Hot days ≥ " + c[len("hot_days_ge_"):].replace("_", " "), "days/yr", "seq",
                             (lambda c=c: _aligned(ctx, ctx.parquet("planner/planner_cells.parquet"), c)),
                             {"file": "planner/planner_cells.parquet", "column": c}, decimals=1,
                             sources=("planner/planner_cells.parquet",)))

    return defs


def _curve_class(df) -> np.ndarray:
    cm = df["curve_model"].astype(str).to_numpy()
    cens = df["censored"].to_numpy(bool) if "censored" in df.columns else np.zeros(len(df), bool)
    return np.where(cens, 0, np.where(cm == "saturating", 1, np.where(cm == "linear", 2,
                                                                    np.where(cm == "sigmoid", 3, 0)))).astype(np.uint8)


def _aligned(ctx, df, col) -> np.ndarray:
    """A per-id table column in run row order."""
    if df is None or col not in df.columns:
        return np.full(ctx.n, np.nan, np.float32)
    if "id" in df.columns and len(df) == ctx.n and np.array_equal(df["id"].to_numpy().astype(str),
                                                                   np.asarray(ctx.grid.ids).astype(str)):
        return _f32(df[col].to_numpy(float))
    if "id" in df.columns:
        s = df.set_index(df["id"].astype(str))[col]
        return _f32(s.reindex(np.asarray(ctx.grid.ids).astype(str)).to_numpy(float))
    return _f32(df[col].to_numpy(float)) if len(df) == ctx.n else np.full(ctx.n, np.nan, np.float32)


def people_layers(ctx):
    """The config's ``planner.layers`` table aligned with the run (None without it)."""
    return ctx._get("people_layers", lambda: _load_people(ctx))


def _load_people(ctx):
    import pandas as pd

    rel = (ctx.cfg_raw.get("planner") or {}).get("layers")
    cfg = ctx.cfg
    if not rel or cfg is None:
        return None
    path = cfg.resolve_path(rel)
    if path is None or not Path(path).is_file():
        return None
    data = ctx.data
    try:
        if data is not None:
            from sparc.core.opendata import load_layers

            lay = load_layers(cfg, data)
            if lay is not None:
                lay = lay.reset_index(drop=False) if "id" not in lay.columns else lay
            return lay
        lay = pd.read_parquet(path)
        if (ctx.manifest_raw or {}).get("qa", {}).get("coarse"):
            return None                      # coarse runs need the data to re-aggregate the layers
        ids = np.asarray(ctx.grid.ids)
        return lay.set_index("id").reindex(ids).reset_index()
    except Exception as exc:
        log.warning("%s: planner layers unreadable: %s", ctx.run_id, exc)
        return None


def _plantable(ctx) -> np.ndarray:
    from sparc.core.planner import plantable_headroom

    can = ((ctx.cfg_raw.get("physics") or {}).get("roles") or {}).get("canopy")
    share = float((ctx.cfg_raw.get("planner") or {}).get("paved_plantable_share", 0.2))
    return _f32(plantable_headroom(ctx.data.frame[can].to_numpy(float), people_layers(ctx), share))


# ---------------------------------------------------------------------------
# Studio layers (api.md §12.2)
# ---------------------------------------------------------------------------

def _studio_layer_defs(ctx) -> list[LayerDef]:
    out: list[LayerDef] = []
    tu = ctx.units.get("target", "°F")
    sd = ctx.studio_dir
    res_root = sd / "results"
    if res_root.is_dir():
        for rdir in sorted(p for p in res_root.iterdir() if p.is_dir() and (p / "cells.parquet").exists()):
            rid = rdir.name
            summ = read_json_cached(rdir / "summary.json") or {}
            name = ((summ.get("scenario") or {}).get("name") or (summ.get("summary") or {}).get("scenario_id") or rid)
            try:
                import pyarrow.parquet as pq

                cols = pq.read_schema(rdir / "cells.parquet").names
            except Exception:
                continue
            for field_ in ["delta", "delta_sd", "extrapolation"] + [c for c in cols if c.startswith("realized_")]:
                if field_ not in cols:
                    continue
                unit, scale, center, sign = (tu, "div", 0.0, _COOLER) if field_ == "delta" else \
                    (tu, "seq", None, None) if field_ == "delta_sd" else ("1", "seq", None, None) \
                    if field_ == "extrapolation" else ("", "div", 0.0, None)
                out.append(LayerDef(f"res:{rid}:{field_}", "studio_results", f"{name}: {field_.replace('_', ' ')}",
                                    unit, scale, (lambda k=f"res:{rid}:{field_}": resolve_layer(ctx, k)),
                                    {"file": f"studio/results/{rid}/cells.parquet", "column": field_}, center=center,
                                    sign_note=sign, sources=(str(rdir / "cells.parquet"),)))
            if "delta" in cols and ctx.predictions is not None:
                out.append(LayerDef(f"res:{rid}:abs", "studio_results", f"{name}: temperature with the scenario", tu,
                                    "div", (lambda k=f"res:{rid}:abs": resolve_layer(ctx, k)),
                                    {"file": f"studio/results/{rid}/cells.parquet", "column": "target+delta"},
                                    center="median_obs", decimals=1,
                                    sources=(str(rdir / "cells.parquet"), "predictions.parquet")))
    pl_root = sd / "plans"
    if pl_root.is_dir():
        for pdir in sorted(p for p in pl_root.iterdir() if p.is_dir()):
            plid = pdir.name
            params = read_json_cached(pdir / "params.json") or {}
            name = params.get("name") or plid
            for field_, label, scale, sign in (("dose", "dose", "seq", None),
                                               ("planned_benefit", "planned cooling", "seq", _BENEFIT),
                                               ("closed_loop_delta", "closed-loop ΔT", "div", _COOLER)):
                if field_ == "closed_loop_delta":
                    rj = read_json_cached(pdir / "realised.json") or {}
                    if not rj.get("result_id") and not (pdir / "closed_loop_delta.npy").exists():
                        continue
                elif not (pdir / f"{field_}.npy").exists():
                    continue
                out.append(LayerDef(f"plan:{plid}:{field_}", "studio_plans", f"{name}: {label}",
                                    tu if field_ != "dose" else "", scale,
                                    (lambda k=f"plan:{plid}:{field_}": resolve_layer(ctx, k)),
                                    {"file": f"studio/plans/{plid}", "column": field_},
                                    center=0.0 if scale == "div" else None, zero_blank=field_ == "dose",
                                    sign_note=sign, sources=(str(pdir / f"{field_}.npy"), str(pdir / "realised.json"))))
    cmp_root = sd / "comparisons"
    if cmp_root.is_dir():
        for cdir in sorted(p for p in cmp_root.iterdir() if p.is_dir()):
            cid = cdir.name
            for f in sorted(cdir.glob("diff_*.npy")):
                pair = f.stem[len("diff_"):]
                out.append(LayerDef(f"cmp:{cid}:{pair}", "studio_compare", f"{cid}: {pair.replace('__', ' − ')}", tu,
                                    "div", (lambda k=f"cmp:{cid}:{pair}": resolve_layer(ctx, k)),
                                    {"file": f"studio/comparisons/{cid}/{f.name}", "column": None}, center=0.0,
                                    sign_note="negative = the first item is cooler", sources=(str(f),)))
    return out


def resolve_layer(ctx, key: str) -> np.ndarray:
    """Values of a Studio layer key (``res:``, ``plan:``, ``cmp:``) straight from its files."""
    import pandas as pd

    n = ctx.n
    sd = ctx.studio_dir
    parts = key.split(":")
    if parts[0] == "res" and len(parts) == 3:
        rid, field_ = parts[1], parts[2]
        path = sd / "results" / rid / "cells.parquet"
        if not path.exists():
            raise ApiError("unknown_layer", f"no result {rid!r} on this run")
        df = pd.read_parquet(path)
        if field_ == "abs":
            if "delta" not in df.columns or ctx.predictions is None:
                raise ApiError("unknown_layer", f"layer {key!r} needs the result's delta and the observations")
            return _f32(ctx.predictions["target"].to_numpy(float) + _aligned(ctx, df, "delta"))
        if field_ not in df.columns:
            raise ApiError("unknown_layer", f"result {rid!r} has no field {field_!r}")
        return _aligned(ctx, df, field_)
    if parts[0] == "plan" and len(parts) == 3:
        plid, field_ = parts[1], parts[2]
        pdir = sd / "plans" / plid
        if not pdir.is_dir():
            raise ApiError("unknown_layer", f"no plan {plid!r} on this run")
        if field_ in ("dose", "planned_benefit"):
            p = pdir / f"{field_}.npy"
            if not p.exists():
                raise ApiError("unknown_layer", f"plan {plid!r} has no {field_}")
            return _f32(np.load(p, allow_pickle=False))
        if field_ == "closed_loop_delta":
            if (pdir / "closed_loop_delta.npy").exists():
                return _f32(np.load(pdir / "closed_loop_delta.npy", allow_pickle=False))
            rj = read_json_cached(pdir / "realised.json") or {}
            if rj.get("result_id"):
                return resolve_layer(ctx, f"res:{rj['result_id']}:delta")
        raise ApiError("unknown_layer", f"plan {plid!r} has no {field_!r} layer")
    if parts[0] == "cmp" and len(parts) == 3:
        cid, pair = parts[1], parts[2]
        p = sd / "comparisons" / cid / f"diff_{pair}.npy"
        if not p.exists():
            raise ApiError("unknown_layer", f"no comparison layer {key!r}")
        a = _f32(np.load(p, allow_pickle=False))
        if n is not None and a.size != n:
            raise ApiError("unknown_layer", f"comparison layer {key!r} does not match the run's grid")
        return a
    raise ApiError("unknown_layer", f"unknown layer {key!r}")


# ---------------------------------------------------------------------------
# arrays, stats, metadata
# ---------------------------------------------------------------------------

def _def_for(ctx, key: str) -> LayerDef:
    defs = layer_defs(ctx)
    d = defs.get(key)
    if d is not None:
        return d
    if key.split(":")[0] in ("res", "plan", "cmp"):
        resolve_layer(ctx, key)              # raises unknown_layer
        return LayerDef(key, "studio_results", key, "", "div", lambda: resolve_layer(ctx, key))
    raise ApiError("unknown_layer", f"unknown layer {key!r}", detail={"key": key})


def layer_array(ctx, key: str) -> np.ndarray:
    """The values of ``key`` (float32, or uint8 for categorical layers), length n, run row order."""
    d = _def_for(ctx, key)
    ck = (ctx.run_id, ctx.key, key, _stat_key(ctx, d.sources))
    hit = LAYER_CACHE.get(ck)
    if hit is not None:
        return hit
    try:
        arr = d.fn()
    except ApiError:
        raise
    except KeyError as exc:
        raise ApiError("output_missing", f"layer {key!r} is not available: missing {exc}",
                       detail={"output": key, "produced_by": None, "expected_path": None})
    arr = np.ascontiguousarray(arr, dtype=np.uint8 if d.dtype == "uint8" else np.float32)
    if ctx.n is not None and arr.size != ctx.n:
        raise ApiError("output_missing", f"layer {key!r} has {arr.size} values for a grid of {ctx.n}",
                       detail={"output": key, "produced_by": None, "expected_path": None})
    LAYER_CACHE.put(ck, arr)
    return arr


def layer_etag(ctx, key: str) -> str:
    d = _def_for(ctx, key)
    raw = f"{ctx.run_id}|{key}|{_stat_key(ctx, d.sources) if d.sources else ctx.key}"
    return '"' + hashlib.sha1(raw.encode()).hexdigest() + '"'


def layer_stats(arr: np.ndarray, *, cat: bool = False) -> dict:
    v = np.asarray(arr, dtype=np.float64)
    ok = np.isfinite(v) & ((v != 255) if cat else True)
    vals = v[ok]
    if vals.size == 0:
        return {"n": 0, "lo": None, "hi": None, "mean": None, "p1": None, "p2": None, "p50": None, "p98": None,
                "p99": None}
    p = np.percentile(vals, [1, 2, 50, 98, 99])
    return {"n": int(vals.size), "lo": float(vals.min()), "hi": float(vals.max()), "mean": float(vals.mean()),
            "p1": float(p[0]), "p2": float(p[1]), "p50": float(p[2]), "p98": float(p[3]), "p99": float(p[4])}


def layer_meta(ctx, key: str) -> dict:
    d = _def_for(ctx, key)
    try:
        arr = layer_array(ctx, key)
    except ApiError:
        arr = np.full(ctx.n or 0, np.nan, np.float32)
    stats = layer_stats(arr, cat=d.scale == "cat")
    center = d.center
    if center == "median":
        center = stats["p50"]
    elif center == "median_obs":
        center = stats["p50"]
        if "obs" in layer_defs(ctx):
            try:
                center = layer_stats(layer_array(ctx, "obs"))["p50"]
            except ApiError:
                pass
    return clean({"key": d.key, "group": d.group, "label": d.label, "unit": d.unit, "scale": d.scale,
                  "center": center if d.scale == "div" else None, "decimals": int(d.decimals), "mult": float(d.mult),
                  "zero_blank": bool(d.zero_blank), "labels": d.labels, "desc": d.desc, "sign_note": d.sign_note,
                  "source": d.source, "dtype": d.dtype, "stats": stats})


def layer_catalog(ctx) -> dict:
    """``{groups: [{id, label, layers: LayerMeta[]}]}`` (cached per context)."""
    def build():
        groups: "OrderedDict[str, list]" = OrderedDict()
        for key, d in layer_defs(ctx).items():
            try:
                meta = layer_meta(ctx, key)
            except Exception as exc:       # one unreadable file never hides the other layers
                log.warning("%s: layer %s unavailable: %s", ctx.run_id, key, exc)
                continue
            groups.setdefault(d.group, []).append(meta)
        return {"groups": [{"id": g, "label": GROUP_LABELS.get(g, g), "layers": ls} for g, ls in groups.items()]}

    studio_sig = tuple(sorted(str(p) for p in ctx.studio_dir.glob("*/*") if p.is_dir())) if ctx.studio_dir.is_dir() \
        else ()
    return ctx._get(f"layer_catalog:{hash(studio_sig)}", build)


def fold_classes(ctx, k: int) -> np.ndarray:
    """Uint8 class of every cell for fold ``k``: 0 train, 1 test, 2 buffer, 255 outside the fold design."""
    folds = ctx.folds
    if folds is None:
        raise ApiError("output_missing", f"the CV design is not available: {ctx.folds_error or 'unknown'}",
                       detail={"output": "folds", "produced_by": "stage:S2_S3", "expected_path": None})
    if k < 0 or k >= folds.n_folds:
        raise ApiError("not_found", f"fold {k} does not exist (the run has {folds.n_folds} folds)")
    out = np.full(folds.fold_id.size, 2, dtype=np.uint8)
    out[folds.train_masks[k]] = 0
    out[folds.test_masks[k]] = 1
    out[folds.fold_id < 0] = 255
    return out


def json_dumps(obj) -> str:
    return json.dumps(clean(obj), separators=(",", ":"), allow_nan=False)
