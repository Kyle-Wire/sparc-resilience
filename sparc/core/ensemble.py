"""S2 + S3 — cross-fitted stacking.

One spatial-block partition is shared by everything:

1. For each fold k, every base model is fitted on the training part of fold k
   and predicts *all* points.  The out-of-fold matrix Ẑ takes, for each point,
   the prediction of the models that never saw it.
2. For each fold k the physics-informed stacker is trained on the rows outside
   fold k (their Ẑ rows) and predicts fold k.  This "cross-fitted stacking"
   leaks only second-order information (the other folds' Ẑ came from models
   that saw fold k) — far cheaper than full nesting.
3. Predictions for any (edited) context use the fold-k stack for points in
   fold k — the honest prediction — and the spread across folds as an
   epistemic uncertainty for *differences* (scenario deltas).

There is deliberately no full-data refit: full-data tree/GWR fits are
in-sample at every point and unlike the OOF features the stacker learned from.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sparc.core.base_models import FeatureContext, PhysicsBaseModel, build_base_models
from sparc.core.cv import SpatialFolds
from sparc.core.stacker import PhysicsInformedStacker, StackerInputs, cross_conformal_halfwidth

log = logging.getLogger(__name__)


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    ok = np.isfinite(p) & np.isfinite(y)
    y, p = y[ok], p[ok]
    res = y - p
    return {
        "rmse": float(np.sqrt(np.mean(res**2))),
        "mae": float(np.mean(np.abs(res))),
        "r2": float(1.0 - np.sum(res**2) / np.sum((y - y.mean()) ** 2)),
        "bias": float(-res.mean()),
        "n": int(ok.sum()),
    }


@dataclass
class FoldStack:
    base: list                       # fitted non-physics base models
    physics: PhysicsBaseModel | None
    stacker: PhysicsInformedStacker | None = None


@dataclass
class FittedEnsemble:
    folds: SpatialFolds
    stacks: list[FoldStack]
    base_names: list[str]
    oof_base: pd.DataFrame
    oof_pred: np.ndarray
    halfwidth: np.ndarray
    metrics: dict
    lambda_pde: float | None            # None: the neural residual is off (convex base only)
    lambda_scores: dict
    timings: dict = field(default_factory=dict)
    physics_selection: dict | None = None
    stacker_info: list | None = None
    stacker_choice: str = ""

    @property
    def has_physics(self) -> bool:
        return self.stacks[0].physics is not None

    # ------------------------------------------------------------ predict
    def fold_predictions(self, ctx: FeatureContext, phys_override: np.ndarray | None = None,
                         F_only: bool = False) -> np.ndarray:
        """(K, n) predictions of every fold's full stack on ``ctx``.

        ``phys_override`` (K, n) replaces the physics predictions (used for
        own-only perturbations, where the non-local physics response is
        applied analytically)."""
        out = np.empty((len(self.stacks), ctx.n))
        for k, st in enumerate(self.stacks):
            Z = np.column_stack([m.predict(ctx) for m in st.base]) if st.base else np.zeros((ctx.n, 0))
            if phys_override is not None:
                phys = phys_override[k]
            else:
                phys = st.physics.predict(ctx) if st.physics is not None else None
            feats = ctx.XF.to_numpy(float)
            out[k] = st.stacker.predict(StackerInputs(Z=Z, phys=phys, feats=feats))
        return out

    def physics_predictions(self, ctx: FeatureContext) -> np.ndarray | None:
        if not self.has_physics:
            return None
        return np.vstack([st.physics.predict(ctx) for st in self.stacks])

    def honest(self, fold_preds: np.ndarray) -> np.ndarray:
        """Pick, for each point, the prediction of the stack that never saw it."""
        return fold_preds[self.folds.fold_id, np.arange(fold_preds.shape[1])]

    def predict(self, ctx: FeatureContext) -> dict:
        fp = self.fold_predictions(ctx)
        return {"honest": self.honest(fp), "folds": fp}


def _select_advection(ctx, folds, stacks, fold_base_preds, pcfg, L_init, seed) -> dict:
    """Refit physics with v = 0 per fold and keep advection only if it lowers
    the held-out RMSE by more than its fold-to-fold noise (one SE)."""
    y = ctx.meta["y"]
    cfg0 = dict(pcfg)
    cfg0["fit_advection"] = False
    d, alt = [], []
    for k, (tr, te) in enumerate(folds.split()):
        m0 = PhysicsBaseModel(ctx.grid, cfg0, L_init=L_init, seed=seed + k).fit(ctx, tr)
        p0 = m0.predict(ctx)
        p1 = fold_base_preds[k]["physics"]
        r0 = float(np.sqrt(np.mean((p0[te] - y[te]) ** 2)))
        r1 = float(np.sqrt(np.mean((p1[te] - y[te]) ** 2)))
        d.append(r1 - r0)
        alt.append((m0, p0))
    d = np.asarray(d)
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
    keep = bool(d.mean() < -se)
    if not keep:
        for k, (m0, p0) in enumerate(alt):
            stacks[k].physics = m0
            fold_base_preds[k]["physics"] = p0
    log.info("physics advection %s (ΔRMSE adv − no-adv = %.4f ± %.4f)", "kept" if keep else "dropped", d.mean(), se)
    return {"kept": keep, "delta_rmse_mean": float(d.mean()), "delta_rmse_se": se,
            "per_fold_delta_rmse": [float(v) for v in d]}


def fit_ensemble(ctx: FeatureContext, folds: SpatialFolds, cfg, *, ranges_m: dict[str, float],
                 intercept_range_m: float, L_init: float | None, seed: int = 0) -> FittedEnsemble:
    """Fit base models per fold, build Ẑ, tune λ_PDE and fit the stackers."""
    y = ctx.meta["y"]
    n = ctx.n
    K = folds.n_folds
    t0 = time.time()
    stacks: list[FoldStack] = []
    fold_base_preds: list[dict[str, np.ndarray]] = []
    base_names: list[str] = []
    for k, (tr, _te) in enumerate(folds.split()):
        models = build_base_models(cfg.raw["models"], grid=ctx.grid, cfg_physics=cfg.raw["physics"],
                                   ranges_m=ranges_m, intercept_range_m=intercept_range_m,
                                   block_m=folds.block_m, L_init=L_init, seed=seed + k)
        preds, fitted, phys = {}, [], None
        for m in models:
            t = time.time()
            try:
                m.fit(ctx, tr)
                p = m.predict(ctx)
            except Exception as exc:  # a failed fold is an error, never mean-filled
                raise RuntimeError(f"base model {m.name!r} failed on fold {k}: {exc}") from exc
            if not np.all(np.isfinite(p)):
                raise RuntimeError(f"base model {m.name!r} produced non-finite predictions on fold {k}")
            preds[m.name] = p
            log.info("fold %d: %s fitted in %.1fs", k, m.name, time.time() - t)
            if isinstance(m, PhysicsBaseModel):
                phys = m
            else:
                fitted.append(m)
        base_names = [m.name for m in models]
        stacks.append(FoldStack(base=fitted, physics=phys))
        fold_base_preds.append(preds)
    t_base = time.time() - t0

    physics_selection = None
    pcfg = cfg.raw["physics"]
    fa = pcfg.get("fit_advection", "auto")
    adv_on = (pcfg.get("wind") is not None) if fa in ("auto", None) else bool(fa)
    if "physics" in base_names and adv_on and pcfg.get("select_advection", True):
        physics_selection = _select_advection(ctx, folds, stacks, fold_base_preds, pcfg, L_init, seed)

    oof = pd.DataFrame({name: np.empty(n) for name in base_names})
    for k in range(K):
        te = folds.test_masks[k]
        for name in base_names:
            oof.loc[te, name] = fold_base_preds[k][name][te]

    other = [nm for nm in base_names if nm != "physics"]
    Z_oof = oof[other].to_numpy(float) if other else np.zeros((n, 0))
    phys_oof = oof["physics"].to_numpy(float) if "physics" in base_names else None
    feats = ctx.XF.to_numpy(float)
    inp = StackerInputs(Z=Z_oof, phys=phys_oof, feats=feats)

    sigma_q = None
    if stacks[0].physics is not None:
        q = stacks[0].physics.model.source_raster(ctx.frame)
        sigma_q = float(np.nanstd(q[ctx.grid.mask]))

    scfg = cfg.raw["stacker"]
    lambdas = list(scfg.get("tune_lambda") or [scfg.get("lambda_pde", 1.0)])
    if stacks[0].physics is None:
        lambdas = [0.0]
    # Candidates, simplest first: equal-weight mean, convex (NNLS) blend, and
    # the blend plus the neural residual at each λ_PDE.  Anything beyond the
    # mean has to earn its place on the outer (spatially blocked) folds.
    candidates = [("mean", None), ("nnls", None)] if scfg.get("allow_residual_off", True) else []
    candidates += [("nnls", lam) for lam in lambdas]
    labels = {"mean": "equal-weight mean of base models", "nnls": "convex (NNLS) blend of base models"}
    t1 = time.time()
    scores, best = {}, None
    for base_mode, lam in candidates:
        stackers, pred = [], np.empty(n)
        for k, (tr, _te) in enumerate(folds.split()):
            st = PhysicsInformedStacker(scfg, ctx.grid, physics=(stacks[k].physics.model if stacks[k].physics else None),
                                        sigma_q=sigma_q, lambda_pde=(0.0 if lam is None else lam),
                                        seed=int(scfg.get("seed", 0)) + k, base_mode=base_mode)
            st.fit(inp, y, tr, groups=folds.block_id, buffer_m=folds.buffer_m, residual=lam is not None)
            pk = st.predict(inp)
            pred[folds.test_masks[k]] = pk[folds.test_masks[k]]
            stackers.append(st)
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        key = base_mode if lam is None else str(lam)
        key = "off" if key == "nnls" else key
        choice = labels[base_mode] if lam is None else f"{labels[base_mode]} + neural residual (λ_PDE={lam:g})"
        scores[key] = rmse
        log.info("stacker %s: OOF RMSE %.4f", choice, rmse)
        # ties go to the simpler candidate (earlier in the list)
        if best is None or rmse < best[0] * (1.0 - 1e-3):
            best = (rmse, lam, stackers, pred, choice)
    _, lam_best, stackers, oof_pred, stacker_choice = best
    for k in range(K):
        stacks[k].stacker = stackers[k]
    t_stack = time.time() - t1

    cov = float(scfg.get("coverage", 0.9))
    hw = cross_conformal_halfwidth(y, oof_pred, folds.fold_id, cov)
    metrics = {name: _metrics(y, oof[name].to_numpy(float)) for name in base_names}
    metrics["base_mean"] = _metrics(y, oof[base_names].mean(axis=1).to_numpy(float))
    metrics["stacker"] = _metrics(y, oof_pred)
    metrics["stacker"]["interval_coverage"] = float(np.mean(np.abs(y - oof_pred) <= hw))
    metrics["stacker"]["interval_target"] = cov
    metrics["stacker"]["interval_mean_halfwidth"] = float(np.mean(hw))
    wnames = other + (["physics"] if phys_oof is not None and str(scfg.get("physics_mode", "feature")) != "backbone" else [])
    stacker_info = [st.summary(wnames) for st in stackers]
    return FittedEnsemble(folds=folds, stacks=stacks, base_names=base_names, oof_base=oof, oof_pred=oof_pred,
                          halfwidth=hw, metrics=metrics, lambda_pde=(None if lam_best is None else float(lam_best)),
                          lambda_scores=scores,
                          timings={"base_models_s": t_base, "stackers_s": t_stack},
                          physics_selection=physics_selection, stacker_info=stacker_info,
                          stacker_choice=stacker_choice)
