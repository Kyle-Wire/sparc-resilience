"""Reference models scored on exactly the same spatial folds as the stack.

A reader's first question about a new spatial model is "does it beat the
standard tools?"  Four baselines answer it, each fitted per fold on the same
(buffered) training points and scored on the same held-out blocks:

* ``regression_kriging`` — ridge regression on the covariates plus ordinary
  kriging of its residuals (Matérn GP, hyper-parameters fitted on ≤ 2,000
  training points).  The geostatistics workhorse.
* ``hgb_xy`` — gradient-boosted trees on covariates *and* coordinates: the
  common "random forest with x, y" machine-learning baseline.
* ``hgb`` — the same trees on covariates only (no location).
* ``idw`` — inverse-distance interpolation of the training targets (12
  nearest, power 2): no covariates at all.
* ``hgb_focal`` — the trees on the stack's own inputs (covariates + the
  multi-scale neighbourhood means at the S1 influence ranges).  An ablation:
  it separates what the area-of-influence features buy from what the
  geographically weighted / physics models and the stacker add.

Every comparison is *paired*: per CV block b, D_b = Σ_{i∈b}(e²_baseline −
e²_stack), the mean difference ΔMSE = ΣD_b / n and its block-clustered SE
(blocks are the independent units).  ``stack_better`` is true when ΔMSE > 2 SE.
If the stack does not win, the report says so.
"""

from __future__ import annotations

import logging

import numpy as np

from sparc.core import progress

log = logging.getLogger(__name__)

BASELINES = ("regression_kriging", "hgb_xy", "hgb", "idw", "hgb_focal")
LABELS = {"regression_kriging": "regression-kriging (ridge + Matérn GP on residuals)",
          "hgb_focal": "gradient boosting on the stack's inputs (covariates + neighbourhood features)",
          "hgb_xy": "gradient boosting, covariates + x,y", "hgb": "gradient boosting, covariates only",
          "idw": "inverse-distance interpolation (no covariates)"}


def _standardise(X: np.ndarray, tr: np.ndarray) -> np.ndarray:
    mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0)
    return (X - mu) / np.where(sd > 0, sd, 1.0)


def _regression_kriging(X, coords, y, tr, te, seed, max_gp: int = 2000):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    from sklearn.linear_model import RidgeCV

    Z = _standardise(X, tr)
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Z[tr], y[tr])
    r = y[tr] - ridge.predict(Z[tr])
    rng = np.random.default_rng(seed)
    sub = tr if tr.size <= max_gp else rng.choice(tr, max_gp, replace=False)
    rs = y[sub] - ridge.predict(Z[sub])
    c0 = coords[tr].mean(axis=0)
    span = float(np.ptp(coords[tr], axis=0).max())
    kern = (ConstantKernel(max(float(np.var(r)), 1e-6), (1e-4, 1e3))
            * Matern(length_scale=span / 10.0, length_scale_bounds=(span / 1000.0, span * 2.0), nu=1.5)
            + WhiteKernel(max(0.1 * float(np.var(r)), 1e-6), (1e-6, 1e3)))
    gp = GaussianProcessRegressor(kern, normalize_y=True, random_state=seed, n_restarts_optimizer=0)
    gp.fit(coords[sub] - c0, rs)
    return ridge.predict(Z[te]) + gp.predict(coords[te] - c0)


def _hgb(F, y, tr, te, seed):
    from sklearn.ensemble import HistGradientBoostingRegressor

    m = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
                                      l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
                                      n_iter_no_change=30, random_state=seed)
    return m.fit(F[tr], y[tr]).predict(F[te])


def _idw(coords, y, tr, te, k: int = 12, power: float = 2.0):
    from scipy.spatial import cKDTree

    d, j = cKDTree(coords[tr]).query(coords[te], k=min(k, tr.size))
    d = np.maximum(d, 1e-6)
    w = d ** -power
    return (w * y[tr][j]).sum(axis=1) / w.sum(axis=1)


def baseline_oof(X: np.ndarray, coords: np.ndarray, y: np.ndarray, folds, models=BASELINES,
                 seed: int = 0, XF: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Out-of-fold predictions of each baseline (NaN where a point is in no test
    fold).  ``hgb_focal`` needs ``XF`` (covariates + focal features) and is
    skipped without it."""
    models = [m for m in models if m != "hgb_focal" or XF is not None]
    X = np.asarray(X, float)
    coords = np.asarray(coords, float)
    out = {m: np.full(y.size, np.nan) for m in models}
    for k, (tr, te) in enumerate(folds.split()):
        for m in models:
            progress.check_cancel()
            with progress.task("baseline_model", key=m, unit=f"baseline_fit:{m}") as sp:
                if m == "regression_kriging":
                    p = _regression_kriging(X, coords, y, tr, te, seed + k)
                elif m == "hgb_xy":
                    p = _hgb(np.column_stack([X, coords]), y, tr, te, seed + k)
                elif m == "hgb":
                    p = _hgb(X, y, tr, te, seed + k)
                elif m == "idw":
                    p = _idw(coords, y, tr, te)
                elif m == "hgb_focal":
                    p = _hgb(np.asarray(XF, float), y, tr, te, seed + k)
                else:
                    raise ValueError(f"unknown baseline {m!r}")
                out[m][te] = p
                if te.size:
                    sp.metrics["heldout_rmse"] = float(np.sqrt(np.mean((p - y[te]) ** 2)))
        progress.tick(k + 1, folds.n_folds, unit="baseline_fold")
        log.info("baselines: fold %d/%d done", k + 1, folds.n_folds)
    return out


def paired_comparison(y: np.ndarray, stack: np.ndarray, other: np.ndarray, block_id: np.ndarray) -> dict:
    """ΔMSE = MSE(other) − MSE(stack) with a block-clustered SE (> 0: stack better)."""
    ok = np.isfinite(stack) & np.isfinite(other)
    y, stack, other, b = y[ok], stack[ok], other[ok], block_id[ok]
    d = (other - y) ** 2 - (stack - y) ** 2
    n = d.size
    blocks, inv = np.unique(b, return_inverse=True)
    D = np.bincount(inv, weights=d)
    nb = np.bincount(inv).astype(float)
    dm = float(D.sum() / n)
    B = blocks.size
    se = float(np.sqrt(B / max(B - 1, 1) * np.sum((D - nb * dm) ** 2)) / n) if B > 1 else float("nan")
    rmse_o = float(np.sqrt(np.mean((other - y) ** 2)))
    rmse_s = float(np.sqrt(np.mean((stack - y) ** 2)))
    return {"rmse": rmse_o, "r2": float(1.0 - np.mean((other - y) ** 2) / np.var(y)),
            "stack_rmse": rmse_s, "delta_mse": dm, "delta_mse_se": se,
            "delta_rmse": rmse_o - rmse_s, "z": dm / se if se and np.isfinite(se) and se > 0 else None,
            "stack_better": bool(np.isfinite(se) and dm > 2.0 * se),
            "baseline_better": bool(np.isfinite(se) and dm < -2.0 * se),
            "frac_blocks_stack_better": float(np.mean(D > 0)), "n_blocks": int(B)}


def compare_baselines(X, coords, y, stack_oof, folds, models=BASELINES, seed: int = 0, XF=None) -> dict:
    """Fit every baseline on ``folds`` and compare each with the stack's OOF predictions."""
    oof = baseline_oof(X, coords, y, folds, models=models, seed=seed, XF=XF)
    rows = {m: {"label": LABELS.get(m, m), **paired_comparison(y, stack_oof, p, folds.block_id)}
            for m, p in oof.items()}
    best = min(rows, key=lambda m: rows[m]["rmse"])
    verdict = ("stack better than every baseline (ΔMSE > 2 SE)" if all(r["stack_better"] for r in rows.values())
               else f"best baseline {best} is better than the stack" if rows[best]["baseline_better"]
               else f"stack not distinguishable from {best} (|ΔMSE| ≤ 2 SE)" if not rows[best]["stack_better"]
               else "stack better than the best baseline; not better than every baseline")
    if not rows[best]["stack_better"]:
        progress.warn("baselines.stack_not_better", verdict, best_baseline=best,
                      delta_rmse=rows[best]["delta_rmse"], block_m=float(folds.block_m))
    return {"block_m": float(folds.block_m), "buffer_m": float(folds.buffer_m), "rows": rows,
            "best_baseline": best, "verdict": verdict}


def load_run(run_dir, cfg):
    """(data, folds, manifest, predictions) of a finished run, rebuilt without
    the checkpoint: the data from the config (coarse cells if the run was
    coarse) and the folds from the manifest's CV settings (deterministic).

    Runs made from an in-memory table (placebo children) are rebuilt from
    their ``input_frame.parquet`` instead of ``data.path``: the table they
    were fitted on differs from the file (shifted, rotated or added layers)."""
    import json
    from pathlib import Path

    import pandas as pd

    from sparc.core.cv import make_spatial_folds
    from sparc.core.data import load_core_data, prepare_frame

    run_dir = Path(run_dir)
    m = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    co = (m.get("qa") or {}).get("coarse")
    if co:
        cfg.raw["data"]["coarse_m"] = float(co["cell_m"])
    if m.get("qa", {}).get("subsample_window_n") and not cfg.data.get("subsample"):
        cfg.raw["data"]["subsample"] = int(m["qa"]["subsample_window_n"])
    frame_file = (m.get("provenance") or {}).get("input_frame") or "input_frame.parquet"
    if (run_dir / frame_file).exists():
        data = prepare_frame(pd.read_parquet(run_dir / frame_file), cfg)
    else:
        data = load_core_data(cfg)
    pred = pd.read_parquet(run_dir / "predictions.parquet")
    if not np.array_equal(pred["id"].to_numpy(), data.ids):
        raise ValueError(f"{run_dir}: predictions do not match the config's data")
    cv = m["cv"]
    folds = make_spatial_folds(data.coords, n_folds=int(cv["n_folds"]), block_m=float(cv["block_m"]),
                               buffer_m=float(cv["buffer_m"]), seed=int(cfg.raw["cv"].get("seed", 42)))
    if not np.array_equal(folds.fold_id, pred["fold"].to_numpy()):
        raise ValueError(f"{run_dir}: rebuilt folds differ from the run's (different cv.seed?)")
    return data, folds, m, pred


def baselines_for_run(run_dir, cfg, models=BASELINES) -> dict:
    """Score the baselines against a finished run and record them in its
    manifest (``baselines``, via :func:`sparc.core.runio.update_manifest`)
    and ``baselines.json``."""
    from pathlib import Path

    from sparc.core import runio
    from sparc.core.features import build_context

    run_dir = Path(run_dir)
    with progress.run_dir_scope(run_dir):
        with progress.task("load_run"):
            data, folds, m, pred = load_run(run_dir, cfg)
        ranges = (m.get("influence") or {}).get("ranges_m") or {}
        XF = build_context(data.frame, data, ranges, cfg).XF.to_numpy(float) if ranges else None
        res = compare_baselines(data.X.to_numpy(float), data.coords, data.y, pred["dT_pred"].to_numpy(float),
                                folds, models=models, seed=int(cfg.raw["cv"].get("seed", 0)), XF=XF)
        runio.write_json_atomic(run_dir / "baselines.json", res, indent=1)
        progress.artifact(run_dir / "baselines.json", role="baselines")
        runio.update_manifest(run_dir, {"baselines": res}, source="baselines")
        progress.artifact(run_dir / "manifest.json", role="manifest")
    return res
