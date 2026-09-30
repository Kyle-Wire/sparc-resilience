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
        "stacker_choice": "convex base only" if ens.lambda_pde is None else f"residual, λ_PDE={ens.lambda_pde:g}",
        "fold_r2_min": float(np.min(fold_r2)),
        "fold_r2_max": float(np.max(fold_r2)),
        "models": {k: {"rmse": v["rmse"], "r2": v["r2"]} for k, v in ens.metrics.items()
                   if k not in ("stacker", "base_mean")},
    }


def cv_distance_curve(ctx, data, cfg, influence, main_ensemble: FittedEnsemble, main_folds: SpatialFolds,
                      block_sizes=(0, 500, 1000), seed: int = 42) -> dict:
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
        log.info("cv curve: %s → stacker R² %.3f (RMSE %.3f)", label, rows[-1]["stacker"]["r2"],
                 rows[-1]["stacker"]["rmse"])
    rows.append(curve_row(f"{main_folds.block_m:.0f} m blocks (main)", main_folds, main_ensemble, data.y, main=True))
    rows.sort(key=lambda r: r["block_m"])
    return {"rows": rows, "n_folds": int(n_folds),
            "note": "Stacking, model selection and intervals use the main blocks; this curve is reporting only."}
