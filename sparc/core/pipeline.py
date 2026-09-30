"""Run the core pipeline end to end (S0 → S7) and write a run directory.

Outputs (``cfg.output.dir``/<run name>/):

* ``manifest.json`` — config, versions, timings, QA and every summary metric
* ``predictions.parquet`` — per point: target, OOF base predictions, stacked
  prediction, conformal interval
* ``influence.json`` — S1 ranges, anisotropy, target ACF
* ``physics.json`` — per-fold fitted physics parameters
* ``response_<var>.parquet`` + ``response_curves.json`` — S4 maps and curves
* ``scenarios.json`` (+ ``scenario_deltas.parquet``) — S5
* ``causal.json`` — S6 estimates, sensitivity and model-vs-causal audit
* ``optimize.json`` (+ ``allocation.parquet``) — S7
* ``report.md`` — human-readable summary
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from sparc.core.config import CoreConfig, load_core_config
from sparc.core.cv import make_spatial_folds
from sparc.core.data import CoreData, load_core_data, prepare_frame
from sparc.core.ensemble import fit_ensemble
from sparc.core.features import build_context
from sparc.core.mediators import MediatorChain

log = logging.getLogger(__name__)

ALL_STAGES = ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7")


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(_jsonable(obj), indent=2), encoding="utf-8")


@dataclass
class CoreResult:
    cfg: CoreConfig
    data: CoreData
    run_dir: Path | None
    influence: object = None
    ensemble: object = None
    responses: dict = field(default_factory=dict)
    scenarios: list = field(default_factory=list)
    causal: dict = field(default_factory=dict)
    optimize: dict = field(default_factory=dict)
    manifest: dict = field(default_factory=dict)


def _block_and_buffer(cfg: CoreConfig, influence, data: CoreData) -> tuple[float, float]:
    cv = cfg.raw["cv"]
    block = cv.get("block_m", "auto")
    if block in (None, "auto"):
        block = influence.block_size_m if influence is not None else 10 * data.grid.dx
        # keep ≥ 3 blocks per fold on small study areas (coordinates, not the
        # target: data.y is ΔT)
        extent = min(np.ptp(data.x), np.ptp(data.y_coord))
        block = float(min(block, extent / 3.0))
    if float(block) < 3.0 * data.grid.dx:
        # A block smaller than a few cells is random-point CV in disguise:
        # neighbours of every test point sit in training and scores leak.
        log.warning("cv.block_m=%.1f m is below 3 grid cells (%.1f m); raising it", float(block), 3.0 * data.grid.dx)
        block = 3.0 * data.grid.dx
    buf = cv.get("buffer_m", "auto")
    if buf in (None, "auto"):
        buf = float(block) / 3.0
    return float(block), float(buf)


def run_core(cfg: CoreConfig | str | Path, stages=ALL_STAGES, fast: bool = False, frame: pd.DataFrame | None = None,
             write: bool = True) -> CoreResult:
    """Run the requested stages.  ``fast`` shrinks the problem (8k-point
    window, 3 folds, fewer epochs) for smoke runs and CI."""
    if not isinstance(cfg, CoreConfig):
        cfg = load_core_config(cfg)
    if fast:
        if frame is None and not cfg.data.get("subsample"):
            cfg.raw["data"]["subsample"] = 8000
        cfg.raw["cv"]["n_folds"] = min(3, int(cfg.raw["cv"]["n_folds"]))
        cfg.raw["stacker"]["epochs"] = min(200, int(cfg.raw["stacker"]["epochs"]))
        cfg.raw["stacker"]["tune_lambda"] = [0.0, 0.1]
        cfg.raw["influence"]["n_perm"] = min(9, int(cfg.raw["influence"].get("n_perm", 19)))
    stages = set(stages)
    timings: dict[str, float] = {}
    t0 = time.time()

    # ------------------------------------------------------------------ S0
    data = prepare_frame(frame, cfg) if frame is not None else load_core_data(cfg)
    timings["S0"] = time.time() - t0
    run_dir = None
    if write:
        run_dir = cfg.output_dir / (cfg.name + ("_fast" if fast else ""))
        run_dir.mkdir(parents=True, exist_ok=True)
    result = CoreResult(cfg=cfg, data=data, run_dir=run_dir)
    log.info("S0: %d points, grid %s, cell %.2f m", data.n, data.grid.shape, data.grid.dx)

    # ------------------------------------------------------------------ S1
    from sparc.core.influence import compute_influence

    t = time.time()
    influence = compute_influence(data, cfg.raw["influence"], seed=int(cfg.raw["cv"].get("seed", 0)))
    result.influence = influence
    ranges = dict(influence.ranges_m)
    timings["S1"] = time.time() - t
    if run_dir:
        _write_json(run_dir / "influence.json", influence.to_dict())

    if not ({"S2", "S3", "S4", "S5", "S6", "S7"} & stages):
        return _finish(result, timings, fast)

    # --------------------------------------------------------------- S2+S3
    t = time.time()
    block, buf = _block_and_buffer(cfg, influence, data)
    folds = make_spatial_folds(data.coords, n_folds=int(cfg.raw["cv"]["n_folds"]), block_m=block,
                               buffer_m=buf, seed=int(cfg.raw["cv"].get("seed", 42)))
    ctx = build_context(data.frame, data, ranges, cfg)
    ens = fit_ensemble(ctx, folds, cfg, ranges_m=ranges, intercept_range_m=influence.target_resid_range_m,
                       L_init=influence.L_prior_m, seed=int(cfg.raw["cv"].get("seed", 0)))
    result.ensemble = ens
    timings["S2_S3"] = time.time() - t
    if run_dir:
        out = pd.DataFrame({"id": data.ids, "x_m": data.x, "y_m": data.y_coord, "fold": folds.fold_id,
                            "target": data.target_raw, "dT": data.y, "dT_pred": ens.oof_pred,
                            "pred": ens.oof_pred + data.background,
                            "pi_lo": ens.oof_pred + data.background - ens.halfwidth,
                            "pi_hi": ens.oof_pred + data.background + ens.halfwidth})
        for c in ens.oof_base.columns:
            out[f"oof_{c}"] = ens.oof_base[c].to_numpy(float) + data.background
        out.to_parquet(run_dir / "predictions.parquet", index=False)
        if ens.has_physics:
            _write_json(run_dir / "physics.json", [st.physics.params for st in ens.stacks])

    if not ({"S4", "S5", "S6", "S7"} & stages):
        return _finish(result, timings, fast, folds=folds)

    # --------------------------------------------------------------- S4/S5
    from sparc.core.response import ResponseEngine
    from sparc.core.scenarios import ScenarioEngine, specs_from_config

    t = time.time()
    mediators = MediatorChain(cfg.mediators).fit(data.frame) if cfg.mediators else None
    engine = ScenarioEngine(data, cfg, ens, ranges, mediators)
    resp = ResponseEngine(engine, influence_scales=cfg.raw["influence"].get("scales", (0.5, 1.0, 2.0)))
    if "S4" in stages or "S6" in stages or "S7" in stages:
        for var in cfg.actionable:
            vr = resp.sweep(var)
            result.responses[var] = vr
            if run_dir:
                vr.maps.assign(id=data.ids, x_m=data.x, y_m=data.y_coord).to_parquet(
                    run_dir / f"response_{var}.parquet", index=False)
        if run_dir:
            _write_json(run_dir / "response_curves.json",
                        {v: {"summary": r.summary, "curve": r.curve.to_dict(orient="list")}
                         for v, r in result.responses.items()})
    timings["S4"] = time.time() - t

    t = time.time()
    if "S5" in stages:
        deltas = {}
        for spec in specs_from_config(cfg):
            res = engine.run(spec)
            result.scenarios.append(res.summary())
            deltas[spec.name] = res.delta
        if run_dir and deltas:
            _write_json(run_dir / "scenarios.json", result.scenarios)
            pd.DataFrame(deltas).assign(id=data.ids).to_parquet(run_dir / "scenario_deltas.parquet", index=False)
    timings["S5"] = time.time() - t

    # ------------------------------------------------------------------ S6
    t = time.time()
    if "S6" in stages and cfg.raw["causal"].get("enabled", True) and cfg.raw["causal"].get("treatments"):
        from sparc.core.causal import run_causal_validation

        effects = {}
        for tr in cfg.raw["causal"]["treatments"]:
            if tr in result.responses:
                x = data.frame[tr].to_numpy(float)
                pos = x[x > 0] if (x > 0).mean() < 0.95 else x
                t_grid = np.quantile(pos, np.linspace(0.1, 0.9, 7))
                effects[tr] = resp.model_effects(tr, result.responses[tr], t_grid=t_grid)
        result.causal = run_causal_validation(data, cfg.raw["causal"], folds, ranges, model_effects=effects,
                                              seed=int(cfg.raw["cv"].get("seed", 0)))
        if run_dir:
            _write_json(run_dir / "causal.json", _strip_arrays(result.causal))
    timings["S6"] = time.time() - t

    # ------------------------------------------------------------------ S7
    t = time.time()
    ocfg = cfg.raw["optimize"]
    var = ocfg.get("variable")
    if "S7" in stages and ocfg.get("enabled", True) and var in result.responses and ocfg.get("budget"):
        from sparc.core.optimize import optimise_allocation

        eq = None
        if ocfg.get("equity_column") and frame is None:
            raw = pd.read_csv(cfg.data_path, encoding="utf-8-sig")
            if ocfg["equity_column"] in raw.columns:
                eq = raw[ocfg["equity_column"]].to_numpy(float)[: data.n]
        opt = optimise_allocation(engine, result.responses[var], float(ocfg["budget"]),
                                  cost_per_unit=cfg.actionable[var].get("cost_per_unit", ocfg.get("cost_per_unit", 1.0)),
                                  equity_scores=eq, equity_focus=float(ocfg.get("equity_focus", 0.0)))
        result.optimize = opt
        if run_dir:
            _write_json(run_dir / "optimize.json", {k: v for k, v in opt.items() if k not in ("dose", "closed_loop_delta")})
            pd.DataFrame({"id": data.ids, "x_m": data.x, "y_m": data.y_coord, "dose": opt.get("dose"),
                          "closed_loop_delta": opt.get("closed_loop_delta")}).to_parquet(
                run_dir / "allocation.parquet", index=False)
    timings["S7"] = time.time() - t
    return _finish(result, timings, fast, folds=folds)


def _strip_arrays(d):
    if isinstance(d, dict):
        return {k: _strip_arrays(v) for k, v in d.items() if not str(k).startswith("tau_hat")
                and not str(k).startswith("per_point")}
    if isinstance(d, list):
        return [_strip_arrays(v) for v in d]
    return d


def _finish(result: CoreResult, timings: dict, fast: bool, folds=None) -> CoreResult:
    from sparc.core.report import build_manifest, render_report

    result.manifest = build_manifest(result, timings, fast, folds)
    if result.run_dir:
        _write_json(result.run_dir / "manifest.json", result.manifest)
        (result.run_dir / "report.md").write_text(render_report(result), encoding="utf-8")
    return result
