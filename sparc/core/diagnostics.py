"""Diagnostics that support (not feed) the main pipeline.

Skill-vs-distance CV curve
--------------------------
The main run scores models on square spatial blocks at least as large as the
target's residual correlation range, with a buffer of block/3 (Roberts et al.
2017).  That answers "how well do we predict a neighbourhood this far from any
training data?" — deliberately strict.  The curve re-runs the same base models
and stacker on smaller blocks and on random points, so the report shows how
skill decays with distance from training data:

* random points (1-cell blocks, no buffer) — a *leaky* reference: every test
  point has training neighbours one cell away, so the score mostly measures
  interpolation (this is what an unblocked CV reports);
* intermediate blocks — within-city gap filling at that distance;
* the main blocks — extrapolation to an unseen neighbourhood, and the closest
  single-city preview of transfer to a new city.

Stacking, model selection and intervals always use the main blocks; the
curve is reporting only.

Effect-recovery benchmark
-------------------------
Accuracy says nothing about whether a model attributes the temperature to the
right causes — and scenarios, saturation and the optimiser all run on
attributed effects.  :func:`effect_recovery` compares every base model, the
stack and the stacked footprint map against the planted truths of the
synthetic city (:mod:`sparc.core.synthetic`); :func:`run_benchmark` runs it
with Spatial+ on and off.
"""

from __future__ import annotations

import logging

import numpy as np

from sparc.core.cv import SpatialFolds, make_spatial_folds
from sparc.core.ensemble import FittedEnsemble, fit_ensemble

log = logging.getLogger(__name__)


def _r2(y: np.ndarray, p: np.ndarray) -> float:
    return float(1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def curve_row(label: str, folds: SpatialFolds, ens: FittedEnsemble, y: np.ndarray, main: bool = False) -> dict:
    """One row of the curve from a fitted cross-fitted ensemble."""
    n = y.size
    kept = [float(tr.sum()) / max(n - te.sum(), 1) for tr, te in zip(folds.train_masks, folds.test_masks)]
    fold_r2 = [_r2(y[te], ens.oof_pred[te]) for te in folds.test_masks]
    s = ens.metrics["stacker"]
    return {
        "label": label,
        "main": main,
        "block_m": float(folds.block_m),
        "buffer_m": float(folds.buffer_m),
        "n_blocks": int(folds.n_blocks),
        "train_fraction_kept": float(np.mean(kept)),
        "stacker": {"rmse": s["rmse"], "r2": s["r2"], "interval_coverage": s.get("interval_coverage")},
        "stacker_choice": ens.stacker_choice or ("convex base only" if ens.lambda_pde is None
                                                 else f"residual, λ_PDE={ens.lambda_pde:g}"),
        "fold_r2_min": float(np.min(fold_r2)),
        "fold_r2_max": float(np.max(fold_r2)),
        "models": {k: {"rmse": v["rmse"], "r2": v["r2"]} for k, v in ens.metrics.items()
                   if k not in ("stacker", "base_mean")},
    }


def _baseline_cells(b: dict | None) -> dict:
    if not b:
        return {}
    return {k: {"rmse": r["rmse"], "r2": r["r2"], "delta_rmse": r["delta_rmse"], "stack_better": r["stack_better"],
                "baseline_better": r["baseline_better"]} for k, r in b["rows"].items()}


def cv_distance_curve(ctx, data, cfg, influence, main_ensemble: FittedEnsemble, main_folds: SpatialFolds,
                      block_sizes=(0, 500, 1000), seed: int = 42, baselines: bool | list = True,
                      main_baselines: dict | None = None) -> dict:
    """Refit the base models + stacker on each extra partition and tabulate
    out-of-fold skill against block size.  ``0`` means random points.

    Sizes below 3 grid cells, above a third of the study-area extent, or equal
    to the main block are skipped (the main run supplies its own row)."""
    n_folds = main_folds.n_folds
    dx = data.grid.dx
    extent = min(np.ptp(data.x), np.ptp(data.y_coord))
    ranges = dict(influence.ranges_m)
    rows = []
    for b in block_sizes:
        b = float(b)
        if b <= 0:
            block, buf, label = dx, 0.0, "random points (leaky reference)"
        else:
            if b < 3 * dx or b > extent / 3.0:
                log.info("cv curve: skipping %.0f m blocks (outside [%.0f, %.0f] m for this study area)",
                         b, 3 * dx, extent / 3.0)
                continue
            if abs(b - main_folds.block_m) < 0.5 * dx:
                continue
            block, buf, label = b, b / 3.0, f"{b:g} m blocks"
        log.info("cv curve: fitting %s", label)
        folds = make_spatial_folds(data.coords, n_folds=n_folds, block_m=block, buffer_m=buf, seed=seed)
        ens = fit_ensemble(ctx, folds, cfg, ranges_m=ranges, intercept_range_m=influence.target_resid_range_m,
                           L_init=influence.L_prior_m, seed=seed)
        rows.append(curve_row(label, folds, ens, data.y))
        if baselines:
            from sparc.core.baselines import BASELINES, compare_baselines

            rows[-1]["baselines"] = _baseline_cells(compare_baselines(
                data.X.to_numpy(float), data.coords, data.y, ens.oof_pred, folds, seed=seed,
                models=BASELINES if baselines is True else tuple(baselines), XF=ctx.XF.to_numpy(float)))
        log.info("cv curve: %s → stacker R² %.3f (RMSE %.3f)", label, rows[-1]["stacker"]["r2"],
                 rows[-1]["stacker"]["rmse"])
    rows.append(curve_row(f"{main_folds.block_m:.0f} m blocks (main)", main_folds, main_ensemble, data.y, main=True))
    if baselines and main_baselines:
        rows[-1]["baselines"] = _baseline_cells(main_baselines)
    rows.sort(key=lambda r: r["block_m"])
    return {"rows": rows, "n_folds": int(n_folds),
            "note": "Stacking, model selection and intervals use the main blocks; this curve is reporting only."}


def effect_recovery(result, city, variable: str = "canopy", dose: float = 5.0) -> dict:
    """Share of the planted effect each model recovers on the synthetic city.

    ``share`` = model mean Δŷ / true mean ΔT for a uniform +``dose`` edit
    (1 = unbiased, < 1 = attenuated); ``corr`` = spatial correlation of the
    per-cell responses with the truth."""
    from sparc.core.mediators import MediatorChain
    from sparc.core.scenarios import Intervention, ScenarioEngine, ScenarioSpec

    data, cfg, ens = result.data, result.cfg, result.ensemble
    med = MediatorChain(cfg.mediators).fit(data.frame) if cfg.mediators else None
    eng = ScenarioEngine(data, cfg, ens, result.influence.ranges_m, med)
    spec = ScenarioSpec("uniform", [Intervention(variable, "add", float(dose))])
    new, _ = eng.apply(spec)
    ctx_new, _ = eng.scenario_context(new)
    truth = city.true_response(float(dose))
    tm = float(np.mean(truth))

    def row(delta: np.ndarray) -> dict:
        return {"share": float(np.mean(delta) / tm), "corr": float(np.corrcoef(delta, truth)[0, 1]),
                "mean_delta": float(np.mean(delta))}

    models: dict[str, list] = {}
    for st in ens.stacks:
        for m in st.base:
            models.setdefault(m.name, []).append(m.predict(ctx_new) - m.predict(eng.base_ctx))
        if st.physics is not None:
            models.setdefault("physics", []).append(st.physics.predict(ctx_new) - st.physics.predict(eng.base_ctx))
    out = {"variable": variable, "dose": float(dose), "true_mean_delta": tm,
           "models": {k: row(np.mean(v, axis=0)) for k, v in models.items()},
           "stack": row(eng.run(spec).delta)}
    vr = (result.responses or {}).get(variable)
    if vr is not None:
        fp = vr.maps["footprint_effect_per_unit"].to_numpy(float)
        tfp = city.true_footprint()
        out["footprint"] = {"share": float(np.mean(fp) / np.mean(tfp)), "corr": float(np.corrcoef(fp, tfp)[0, 1])}
    out["oof"] = {k: {"rmse": v["rmse"], "r2": v["r2"]} for k, v in ens.metrics.items()}
    return out


def run_benchmark(seed: int = 0, spatial_plus_ab: bool = True, epochs: int = 150, n: int = 96) -> dict:
    """Synthetic-city benchmark (S0–S4, canopy only, no causal stage)."""
    from sparc.core.config import core_config_from_dict
    from sparc.core.pipeline import run_core
    from sparc.core.synthetic import make_synthetic_city, synthetic_city_config

    city = make_synthetic_city(n=n, seed=seed)
    runs = {}
    for sp in ((True, False) if spatial_plus_ab else (True,)):
        cfg = core_config_from_dict(synthetic_city_config())
        cfg.raw["models"]["spatial_plus"] = ["mgwr"] if sp else []
        cfg.raw["stacker"]["epochs"] = epochs
        cfg.raw["stacker"]["tune_lambda"] = [0.0, 0.1]
        cfg.raw["actionable"] = {"canopy": {"min": 0, "max": 100, "doses": [0, 5, 10, 15, 20, 30, 40]}}
        cfg.raw["causal"]["enabled"] = False
        cfg.raw["cv"]["baselines"] = False
        res = run_core(cfg, stages=("S0", "S1", "S2", "S3", "S4"), frame=city.frame, write=False)
        runs["spatial_plus" if sp else "standard"] = effect_recovery(res, city)
        log.info("benchmark (%s): stack share %.2f, footprint share %.2f",
                 "spatial+" if sp else "standard", runs["spatial_plus" if sp else "standard"]["stack"]["share"],
                 runs["spatial_plus" if sp else "standard"].get("footprint", {}).get("share", float("nan")))
    return {"seed": seed, "n": n, "runs": runs}


def benchmark_markdown(bench: dict) -> str:
    L = ["| setting | model | effect share | spatial corr | OOF RMSE | OOF R² |", "|---|---|---|---|---|---|"]
    for name, r in bench["runs"].items():
        rows = list(r["models"].items()) + [("stack", r["stack"])]
        for k, v in rows:
            o = r["oof"].get("stacker" if k == "stack" else k, {})
            L.append(f"| {name} | {k} | {v['share']:.2f} | {v['corr']:.2f} | {o.get('rmse', float('nan')):.3f} | "
                     f"{o.get('r2', float('nan')):.3f} |")
        if "footprint" in r:
            L.append(f"| {name} | stacked footprint map | {r['footprint']['share']:.2f} | {r['footprint']['corr']:.2f} | | |")
    return "\n".join(L)
