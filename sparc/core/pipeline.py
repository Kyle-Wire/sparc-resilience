"""Run the core pipeline end to end (S0 → S7) and write a run directory.

Outputs (``cfg.output.dir``/<run name>/):

* ``manifest.json`` — config, versions, timings, QA and every summary metric
* ``predictions.parquet`` — per point: target, OOF base predictions, stacked
  prediction, conformal interval
* ``influence.json`` — S1 ranges, anisotropy, target ACF
* ``physics.json`` — per-fold fitted physics parameters
* ``response_<var>.parquet`` + ``response_curves.json`` — S4 maps and curves
* ``scenarios.json`` (+ ``scenario_deltas.parquet``) — S5
* ``cv_distance.json`` — optional skill-vs-distance CV curve
* ``climate.json`` — CMIP6 projections × adaptation (threshold exposure, offsets)
* ``causal.json`` — S6 estimates, sensitivity and model-vs-causal audit
* ``optimize.json`` (+ ``allocation.parquet``) — S7
* ``report.md`` — human-readable summary
* ``checkpoint.pkl`` — fitted state after each expensive stage (``resume=True``)
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
    cv_distance: dict = field(default_factory=dict)
    baselines: dict = field(default_factory=dict)
    climate: dict = field(default_factory=dict)
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


CHECKPOINT = "checkpoint.pkl"


def _fingerprint(cfg: CoreConfig, fast: bool, frame: pd.DataFrame | None) -> str:
    """Identity of a run for resuming: effective config, input data and the
    core source code (a code change invalidates saved fits; a docs commit
    does not)."""
    import hashlib

    h = hashlib.sha256()
    h.update(json.dumps(cfg.raw, sort_keys=True, default=str).encode())
    h.update(str(bool(fast)).encode())
    if frame is not None:
        h.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    else:
        st = Path(cfg.data_path).stat()
        h.update(f"{cfg.data_path}:{st.st_size}:{st.st_mtime_ns}".encode())
    for src in sorted(Path(__file__).parent.glob("*.py")):
        h.update(src.read_bytes())
    return h.hexdigest()[:16]


def _save_checkpoint(run_dir: Path | None, state: dict) -> None:
    if run_dir is None:
        return
    import pickle

    tmp = run_dir / (CHECKPOINT + ".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(run_dir / CHECKPOINT)
    log.info("checkpoint saved (%s)", ", ".join(sorted(state.get("done", ()))))


def _load_checkpoint(run_dir: Path | None, fingerprint: str) -> dict:
    if run_dir is None or not (run_dir / CHECKPOINT).exists():
        return {}
    import pickle

    with open(run_dir / CHECKPOINT, "rb") as fh:
        state = pickle.load(fh)
    if state.get("fingerprint") != fingerprint:
        log.warning("checkpoint in %s is from a different config/data/code version — ignoring it", run_dir)
        return {}
    log.info("resuming from checkpoint (done: %s)", ", ".join(sorted(state.get("done", ()))))
    return state


def run_core(cfg: CoreConfig | str | Path, stages=ALL_STAGES, fast: bool = False, frame: pd.DataFrame | None = None,
             write: bool = True, resume: bool = False, cv_curve: bool | None = None,
             coarse: float | None = None) -> CoreResult:
    """Run the requested stages.  ``fast`` shrinks the problem (8k-point
    window, 3 folds, fewer epochs) for smoke runs and CI.  With ``write``,
    fitted state is checkpointed after S3, the CV distance curve, S4, S5 and
    S6; ``resume`` reuses a checkpoint whose fingerprint matches.
    ``cv_curve`` overrides ``cv.distance_curve.enabled``; ``coarse`` (metres)
    overrides ``data.coarse_m`` — the full extent averaged onto coarser cells,
    the resolution of the validation studies."""
    if not isinstance(cfg, CoreConfig):
        cfg = load_core_config(cfg)
    if coarse:
        cfg.raw["data"]["coarse_m"] = float(coarse)
    if cv_curve is not None:
        cfg.raw["cv"].setdefault("distance_curve", {})["enabled"] = bool(cv_curve)
    if fast:
        if frame is None and not cfg.data.get("subsample"):
            cfg.raw["data"]["subsample"] = 8000
        cfg.raw["cv"]["n_folds"] = min(3, int(cfg.raw["cv"]["n_folds"]))
        cfg.raw["stacker"]["epochs"] = min(200, int(cfg.raw["stacker"]["epochs"]))
        cfg.raw["stacker"]["tune_lambda"] = [0.0, 0.1]
        cfg.raw["influence"]["n_perm"] = min(9, int(cfg.raw["influence"].get("n_perm", 19)))
        if cfg.raw["cv"].get("baselines", True) is True:
            cfg.raw["cv"]["baselines"] = ["hgb_xy", "hgb_focal", "idw"]     # no GP fits in smoke runs
    stages = set(stages)
    timings: dict[str, float] = {}
    t0 = time.time()

    # ------------------------------------------------------------------ S0
    data = prepare_frame(frame, cfg) if frame is not None else load_core_data(cfg)
    timings["S0"] = time.time() - t0
    run_dir = None
    if write:
        cm = cfg.data.get("coarse_m")
        run_dir = cfg.output_dir / (cfg.name + ("_fast" if fast else "") + (f"_coarse{float(cm):g}" if cm else ""))
        run_dir.mkdir(parents=True, exist_ok=True)
    result = CoreResult(cfg=cfg, data=data, run_dir=run_dir)
    log.info("S0: %d points, grid %s, cell %.2f m", data.n, data.grid.shape, data.grid.dx)
    fp = _fingerprint(cfg, fast, frame) if run_dir is not None else ""
    state = _load_checkpoint(run_dir, fp) if resume else {}
    done = set(state.get("done", ()))
    state = {**state, "fingerprint": fp, "done": done}

    # ------------------------------------------------------------------ S1
    t = time.time()
    if "S3" in done:
        influence = state["influence"]
    else:
        from sparc.core.influence import compute_influence

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
    ctx = build_context(data.frame, data, ranges, cfg)
    if "S3" in done:
        folds, ens = state["folds"], state["ensemble"]
    else:
        block, buf = _block_and_buffer(cfg, influence, data)
        folds = make_spatial_folds(data.coords, n_folds=int(cfg.raw["cv"]["n_folds"]), block_m=block,
                                   buffer_m=buf, seed=int(cfg.raw["cv"].get("seed", 42)))
        ens = fit_ensemble(ctx, folds, cfg, ranges_m=ranges, intercept_range_m=influence.target_resid_range_m,
                           L_init=influence.L_prior_m, seed=int(cfg.raw["cv"].get("seed", 0)))
        state.update(influence=influence, folds=folds, ensemble=ens)
        done.add("S3")
        _save_checkpoint(run_dir, state)
    result.ensemble = ens
    if data.zones is not None and ens.halfwidth_adaptive is not None:
        from sparc.core.ensemble import interval_diagnostics

        ens.metrics["stacker"]["interval_diagnostics"] = interval_diagnostics(
            data.y, ens.oof_pred, ens.halfwidth, ens.halfwidth_adaptive, folds.fold_id, ens.dist_train,
            groups={"zone": data.zones})
    timings["S2_S3"] = time.time() - t
    if run_dir:
        out = pd.DataFrame({"id": data.ids, "x_m": data.x, "y_m": data.y_coord, "fold": folds.fold_id,
                            "target": data.target_raw, "dT": data.y, "dT_pred": ens.oof_pred,
                            "pred": ens.oof_pred + data.background,
                            "pi_lo": ens.oof_pred + data.background - ens.halfwidth,
                            "pi_hi": ens.oof_pred + data.background + ens.halfwidth})
        if ens.halfwidth_adaptive is not None:
            out["pi_lo_adaptive"] = ens.oof_pred + data.background - ens.halfwidth_adaptive
            out["pi_hi_adaptive"] = ens.oof_pred + data.background + ens.halfwidth_adaptive
            out["dist_train_m"] = ens.dist_train
        if data.zones is not None:
            out["zone"] = data.zones
        for c in ens.oof_base.columns:
            out[f"oof_{c}"] = ens.oof_base[c].to_numpy(float) + data.background
        out.to_parquet(run_dir / "predictions.parquet", index=False)
        if ens.has_physics:
            _write_json(run_dir / "physics.json", [st.physics.params for st in ens.stacks])

    # ------------------------------------------------- reference baselines
    bl = cfg.raw["cv"].get("baselines", True)
    if bl:
        t = time.time()
        if "baselines" in done:
            result.baselines = state["baselines"]
        else:
            from sparc.core.baselines import BASELINES, compare_baselines

            result.baselines = compare_baselines(data.X.to_numpy(float), data.coords, data.y, ens.oof_pred, folds,
                                                 models=BASELINES if bl is True else tuple(bl),
                                                 seed=int(cfg.raw["cv"].get("seed", 0)), XF=ctx.XF.to_numpy(float))
            state["baselines"] = result.baselines
            done.add("baselines")
            _save_checkpoint(run_dir, state)
        log.info("baselines: %s", result.baselines["verdict"])
        timings["baselines"] = time.time() - t
        if run_dir:
            _write_json(run_dir / "baselines.json", result.baselines)

    # ------------------------------------------------- CV distance diagnostic
    dcfg = cfg.raw["cv"].get("distance_curve") or {}
    if dcfg.get("enabled"):
        t = time.time()
        if "cv_curve" in done:
            result.cv_distance = state["cv_distance"]
        else:
            from sparc.core.diagnostics import cv_distance_curve

            result.cv_distance = cv_distance_curve(ctx, data, cfg, influence, ens, folds,
                                                   block_sizes=dcfg.get("block_m", (0, 500, 1000)),
                                                   seed=int(cfg.raw["cv"].get("seed", 42)),
                                                   baselines=cfg.raw["cv"].get("baselines", True),
                                                   main_baselines=result.baselines or None)
            state["cv_distance"] = result.cv_distance
            done.add("cv_curve")
            _save_checkpoint(run_dir, state)
        timings["cv_curve"] = time.time() - t
        if run_dir:
            _write_json(run_dir / "cv_distance.json", result.cv_distance)

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
        if "S4" in done:
            result.responses = state["responses"]
        else:
            for var in cfg.actionable:
                result.responses[var] = resp.sweep(var)
            state["responses"] = result.responses
            done.add("S4")
            _save_checkpoint(run_dir, state)
        if run_dir:
            for var, vr in result.responses.items():
                vr.maps.assign(id=data.ids, x_m=data.x, y_m=data.y_coord).to_parquet(
                    run_dir / f"response_{var}.parquet", index=False)
            _write_json(run_dir / "response_curves.json",
                        {v: {"summary": r.summary, "curve": r.curve.to_dict(orient="list")}
                         for v, r in result.responses.items()})
    timings["S4"] = time.time() - t

    t = time.time()
    if "S5" in stages:
        if "S5" in done:
            result.scenarios, deltas = state["scenarios"], state["scenario_deltas"]
        else:
            deltas = {}
            for spec in specs_from_config(cfg):
                res = engine.run(spec)
                result.scenarios.append(res.summary())
                deltas[spec.name] = res.delta
            state.update(scenarios=result.scenarios, scenario_deltas=deltas)
            done.add("S5")
            _save_checkpoint(run_dir, state)
        if run_dir and deltas:
            _write_json(run_dir / "scenarios.json", result.scenarios)
            pd.DataFrame(deltas).assign(id=data.ids).to_parquet(run_dir / "scenario_deltas.parquet", index=False)
        if (cfg.raw.get("climate") or {}).get("enabled"):
            if "climate" in done:
                result.climate = state["climate"]
            else:
                result.climate = climate_stage(cfg, data, result.scenarios, deltas)
                state["climate"] = result.climate
                done.add("climate")
                _save_checkpoint(run_dir, state)
            if run_dir:
                _write_json(run_dir / "climate.json", result.climate)
    timings["S5"] = time.time() - t

    # ------------------------------------------------------------------ S6
    t = time.time()
    if "S6" in stages and cfg.raw["causal"].get("enabled", True) and cfg.raw["causal"].get("treatments"):
        if "S6" in done:
            result.causal = state["causal"]
        else:
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
            state["causal"] = result.causal
            done.add("S6")
            _save_checkpoint(run_dir, state)
        if result.scenarios:
            causal_crosscheck(result.scenarios, result.causal)
        if run_dir:
            _write_json(run_dir / "causal.json", _strip_arrays(result.causal))
            if result.scenarios:
                _write_json(run_dir / "scenarios.json", result.scenarios)
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


def _site_latlon(cfg: CoreConfig, data: CoreData) -> tuple[float, float]:
    from pyproj import Transformer

    crs = cfg.data.get("crs")
    if not crs:
        raise ValueError("climate: set climate.site: [lat, lon] (the data have no CRS)")
    tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    s = cfg.coord_scale
    lon, lat = tr.transform(float(np.mean(data.x)) / s, float(np.mean(data.y_coord)) / s)
    return float(lat), float(lon)


def default_adaptation(scenarios: list[dict], max_extrapolated: float = 0.2) -> list[str]:
    """Packages (several variables) plus, per variable, the largest dose that
    stays within observed conditions."""
    names, best = [], {}
    for sc in scenarios:
        real = sc.get("mean_realized") or {}
        if sc.get("frac_extrapolated", 1.0) > max_extrapolated:
            continue
        if len(real) > 1:
            names.append(sc["name"])
        elif len(real) == 1:
            (v, d), = real.items()
            if v not in best or abs(d) > abs(best[v][1]):
                best[v] = (sc["name"], d)
    return names + [n for n, _ in best.values()]


def climate_stage(cfg: CoreConfig, data: CoreData, scenarios: list[dict], deltas: dict) -> dict:
    """CMIP6 change factors × observed field × adaptation scenarios."""
    from sparc.core.climate import cmip6_change_factors, summarize_projections

    cc = cfg.raw["climate"]
    lat, lon = (float(cc["site"][0]), float(cc["site"][1])) if cc.get("site") else _site_latlon(cfg, data)
    if cc.get("source", "table") == "table":
        path = cfg.resolve_path(cc.get("table"))
        if path is None or not path.exists():
            raise FileNotFoundError(f"climate.table not found: {path} (write one with `sparc core climate`)")
        factors = pd.read_csv(path)
    else:
        cache = cfg.output_dir / str(cc.get("cache", "cache"))
        factors = cmip6_change_factors(lat, lon, cache, experiments=tuple(cc["experiments"]),
                                       periods={k: tuple(v) for k, v in cc["periods"].items()},
                                       months=tuple(cc["months"]), variable=cc["variable"])
        cache.mkdir(parents=True, exist_ok=True)
        factors.to_csv(cache / f"cmip6_{cc['variable']}_{lat:.3f}_{lon:.3f}.csv", index=False)
    factors = factors[factors["experiment"].isin(cc["experiments"]) & factors["period"].isin(list(cc["periods"]))]
    units = (data.target_units or "").lower()
    fahrenheit = units in ("degf", "f", "°f", "fahrenheit")
    thresholds = cc.get("thresholds") or ([90.0, 95.0] if fahrenheit else [32.0, 35.0])
    names = cc.get("adaptation") or default_adaptation(scenarios)
    adapt = {n: deltas[n] for n in names if n in deltas}
    out = summarize_projections(np.asarray(data.target_raw, float), factors, adapt, thresholds,
                                to_units=1.8 if fahrenheit else 1.0)
    out.update({"site": {"lat": lat, "lon": lon}, "source": cc.get("source", "table"),
                "variable": cc["variable"], "months": list(cc["months"]),
                "baseline": str(factors["baseline"].iloc[0]) if "baseline" in factors and len(factors) else "1995-2014",
                "n_models": int(factors["model"].nunique()), "units": data.target_units})
    log.info("climate: %d models; median %s warming %s", out["n_models"],
             "2041-2060 SSP2-4.5", next((f"{p['warming']['median']:+.2f}" for p in out["projections"]
                                         if p["experiment"] == "ssp245" and p["period"] == "2041-2060"), "n/a"))
    return out


def causal_crosscheck(scenarios: list[dict], causal: dict) -> list[dict]:
    """Attach a linear causal estimate to every scenario whose edited
    variables all have a spillover estimate: Σ_j (θ_own+θ_nbr)_j × mean
    realised change_j, with a 95% band from the SEs (independence assumed).
    It is a local-slope extrapolation — a cross-check on magnitude and sign,
    not a replacement for the model's scenario."""
    tr = (causal or {}).get("treatments") or {}
    for sc in scenarios:
        real = sc.get("mean_realized") or {}
        if not real or any(not isinstance((tr.get(v) or {}).get("spillover"), dict) for v in real):
            continue
        est, var = 0.0, 0.0
        for v, d in real.items():
            sp = tr[v]["spillover"]
            th, se = sp.get("theta_sum"), sp.get("se_sum")
            if th is None or se is None or not np.isfinite(th) or not np.isfinite(se):
                break
            est += float(th) * float(d)
            var += (float(se) * float(d)) ** 2
        else:
            half = 1.96 * float(np.sqrt(var))
            sc["causal_linear"] = {"delta": est, "lo": est - half, "hi": est + half,
                                   "model_within": bool(est - half <= sc["mean_delta"] <= est + half)}
    return scenarios


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
