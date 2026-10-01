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

Progress (``sparc.core.progress``): ``task fold[k/K]`` › ``task base_model[name]``
(unit ``base_fit:<name>``, metrics ``fit_s``, ``heldout_rmse``, ``heldout_r2``),
``task advection_check`` › ``task adv_refit[k/K]``, ``task
stacker_candidate[c/C]`` › ``task stacker_fold[k/K]`` (unit
``stacker_fit:<mean|nnls|residual>``), and every :meth:`FittedEnsemble.fold_predictions`
call ticks ``k/K`` in unit ``engine_pass`` (the last tick carries ``pass_s``).
Each of these is also a cancellation point.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sparc.core import progress
from sparc.core.base_models import FeatureContext, PhysicsBaseModel, build_base_models
from sparc.core.cv import SpatialFolds
from sparc.core.stacker import (
    PhysicsInformedStacker,
    StackerInputs,
    cross_conformal_adaptive,
    cross_conformal_halfwidth,
)

log = logging.getLogger(__name__)


def _heldout(y: np.ndarray, p: np.ndarray, te: np.ndarray) -> dict:
    """Held-out RMSE and R² of one fold's predictions (task metrics; not stored)."""
    if te.size < 2:
        return {}
    r = y[te] - p[te]
    sst = float(np.sum((y[te] - y[te].mean()) ** 2))
    return {"heldout_rmse": float(np.sqrt(np.mean(r ** 2))),
            "heldout_r2": float(1.0 - np.sum(r ** 2) / sst) if sst > 0 else None}


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
    halfwidth_adaptive: np.ndarray | None = None
    dist_train: np.ndarray | None = None      # distance (m) from each point to its fold's training data

    @property
    def has_physics(self) -> bool:
        return self.stacks[0].physics is not None

    # ------------------------------------------------------------ predict
    def fold_predictions(self, ctx: FeatureContext, phys_override: np.ndarray | None = None,
                         F_only: bool = False) -> np.ndarray:
        """(K, n) predictions of every fold's full stack on ``ctx``.

        ``phys_override`` (K, n) replaces the physics predictions (used for
        own-only perturbations, where the non-local physics response is
        applied analytically).

        One call is one ``engine_pass``: it ticks ``k/K`` after each fold
        (the ``k == K`` tick carries ``pass_s``, the seconds of the whole
        pass) and checks for a cancel request before each fold — the single
        hook that gives every scenario, sweep and emulator computation
        progress and a cancellation point."""
        K = len(self.stacks)
        out = np.empty((K, ctx.n))
        t0 = time.perf_counter()
        for k, st in enumerate(self.stacks):
            progress.check_cancel()
            Z = np.column_stack([m.predict(ctx) for m in st.base]) if st.base else np.zeros((ctx.n, 0))
            if phys_override is not None:
                phys = phys_override[k]
            else:
                phys = st.physics.predict(ctx) if st.physics is not None else None
            feats = ctx.XF.to_numpy(float)
            out[k] = st.stacker.predict(StackerInputs(Z=Z, phys=phys, feats=feats))
            if k + 1 < K:
                progress.tick(k + 1, K, unit="engine_pass")
            else:
                progress.tick(K, K, unit="engine_pass", pass_s=round(time.perf_counter() - t0, 4))
        return out

    def physics_predictions(self, ctx: FeatureContext) -> np.ndarray | None:
        if not self.has_physics:
            return None
        return np.vstack([st.physics.predict(ctx) for st in self.stacks])

    def honest(self, fold_preds: np.ndarray) -> np.ndarray:
        """Pick, for each point, the prediction of the stack that never saw it.
        For *baseline predictions and CV only* — decision quantities (scenario
        Δ, sensitivities, rankings) use :meth:`decision`."""
        return fold_preds[self.folds.fold_id, np.arange(fold_preds.shape[1])]

    @staticmethod
    def decision(fold_values: np.ndarray) -> np.ndarray:
        """Fold-averaged value of a decision quantity (Δ, sensitivity).  The
        honest per-cell pick would stitch five models together: 2 km seams in
        Δ maps, and fold-specific parameters (e.g. the physics gain) leaking
        into rankings."""
        return np.asarray(fold_values, float).mean(axis=0)

    @staticmethod
    def jackknife_sd(fold_values: np.ndarray) -> np.ndarray:
        """Delete-a-group jackknife SE from the K fold models (each trained
        without one group): sd·√(K−1).  The raw fold spread understates the
        estimation uncertainty because the folds share K−2 groups."""
        v = np.asarray(fold_values, float)
        return v.std(axis=0) * np.sqrt(max(v.shape[0] - 1, 1))

    def predict(self, ctx: FeatureContext) -> dict:
        fp = self.fold_predictions(ctx)
        return {"honest": self.honest(fp), "folds": fp}


def training_distance(coords: np.ndarray, folds: SpatialFolds) -> np.ndarray:
    """Distance (m) from every point to the nearest training point of the
    fold model that predicts it out of fold — how far it extrapolates."""
    from scipy.spatial import cKDTree

    out = np.zeros(len(coords))
    for k, (tr, te) in enumerate(folds.split()):
        if tr.size and te.size:
            out[te] = cKDTree(coords[tr]).query(coords[te], k=1)[0]
    return out


def interval_diagnostics(y, pred, hw, hw_ad, fold_id, dist, groups: dict | None = None) -> dict:
    """Coverage of the global and adaptive intervals overall, per fold, per
    distance-to-training quartile and per extra grouping (e.g. zones).
    Pooled coverage is nearly guaranteed by construction; the conditional
    breakdowns are what show whether the intervals are honest where it
    matters."""
    r = np.abs(np.asarray(y) - np.asarray(pred))

    def cov(mask):
        return {"n": int(mask.sum()), "global": float(np.mean(r[mask] <= hw[mask])) if mask.any() else None,
                "adaptive": float(np.mean(r[mask] <= hw_ad[mask])) if mask.any() else None,
                "halfwidth_global": float(np.mean(hw[mask])) if mask.any() else None,
                "halfwidth_adaptive": float(np.mean(hw_ad[mask])) if mask.any() else None}

    out = {"overall": cov(np.ones_like(r, dtype=bool)),
           "by_fold": {str(int(k)): cov(fold_id == k) for k in np.unique(fold_id)}}
    qs = np.quantile(dist, [0.25, 0.5, 0.75])
    edges = [-np.inf, *qs, np.inf]
    out["by_distance"] = {f"{lo if np.isfinite(lo) else 0:.0f}-{hi if np.isfinite(hi) else dist.max():.0f} m":
                          cov((dist > lo) & (dist <= hi)) for lo, hi in zip(edges[:-1], edges[1:])}
    for name, g in (groups or {}).items():
        out[f"by_{name}"] = {str(v): cov(g == v) for v in np.unique(g)}
    return out


def _select_advection(ctx, folds, stacks, fold_base_preds, pcfg, L_init, seed) -> dict:
    """Refit physics with v = 0 per fold and keep advection only if it lowers
    the held-out RMSE by more than its fold-to-fold noise (one SE)."""
    y = ctx.meta["y"]
    cfg0 = dict(pcfg)
    cfg0["fit_advection"] = False
    d, alt = [], []
    K = folds.n_folds
    for k, (tr, te) in enumerate(folds.split()):
        progress.check_cancel()
        with progress.task("adv_refit", k=k + 1, n=K, unit="adv_refit") as sp:
            m0 = PhysicsBaseModel(ctx.grid, cfg0, L_init=L_init, seed=seed + k).fit(ctx, tr)
            p0 = m0.predict(ctx)
            p1 = fold_base_preds[k]["physics"]
            r0 = float(np.sqrt(np.mean((p0[te] - y[te]) ** 2)))
            r1 = float(np.sqrt(np.mean((p1[te] - y[te]) ** 2)))
            sp.metrics.update(heldout_rmse_no_adv=r0, heldout_rmse_adv=r1)
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
    fold_model_s: dict[str, list[float]] = {}
    for k, (tr, te) in enumerate(folds.split()):
        progress.check_cancel()
        with progress.task("fold", k=k + 1, n=K):
            models = build_base_models(cfg.raw["models"], grid=ctx.grid, cfg_physics=cfg.raw["physics"],
                                       ranges_m=ranges_m, intercept_range_m=intercept_range_m,
                                       block_m=folds.block_m, L_init=L_init, seed=seed + k)
            preds, fitted, phys = {}, [], None
            for m in models:
                progress.check_cancel()
                with progress.task("base_model", key=m.name, unit=f"base_fit:{m.name}") as sp:
                    t = time.time()
                    try:
                        m.fit(ctx, tr)
                        p = m.predict(ctx)
                    except Exception as exc:  # a failed fold is an error, never mean-filled
                        raise RuntimeError(f"base model {m.name!r} failed on fold {k}: {exc}") from exc
                    if not np.all(np.isfinite(p)):
                        raise RuntimeError(f"base model {m.name!r} produced non-finite predictions on fold {k}")
                    fit_s = time.time() - t
                    sp.metrics.update(fit_s=round(fit_s, 3), **_heldout(y, p, te))
                preds[m.name] = p
                fold_model_s.setdefault(m.name, []).append(round(fit_s, 3))
                log.info("fold %d: %s fitted in %.1fs", k, m.name, fit_s)
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
        with progress.task("advection_check"):
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
    candidate_s: dict[str, float] = {}
    C = len(candidates)
    for c, (base_mode, lam) in enumerate(candidates, start=1):
        progress.check_cancel()
        ckey = base_mode if lam is None else f"residual:{lam:g}"
        unit = f"stacker_fit:{base_mode if lam is None else 'residual'}"
        tc = time.time()
        with progress.task("stacker_candidate", k=c, n=C, key=ckey) as csp:
            stackers, pred = [], np.empty(n)
            for k, (tr, _te) in enumerate(folds.split()):
                progress.check_cancel()
                with progress.task("stacker_fold", k=k + 1, n=K, unit=unit):
                    st = PhysicsInformedStacker(scfg, ctx.grid,
                                                physics=(stacks[k].physics.model if stacks[k].physics else None),
                                                sigma_q=sigma_q, lambda_pde=(0.0 if lam is None else lam),
                                                seed=int(scfg.get("seed", 0)) + k, base_mode=base_mode)
                    st.fit(inp, y, tr, groups=folds.block_id, buffer_m=folds.buffer_m, residual=lam is not None)
                    pk = st.predict(inp)
                    pred[folds.test_masks[k]] = pk[folds.test_masks[k]]
                    stackers.append(st)
            rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
            csp.metrics["rmse"] = rmse
        candidate_s[ckey] = round(time.time() - tc, 3)
        progress.metric("candidate_rmse", rmse, candidate=ckey)
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
    dist = training_distance(ctx.coords, folds)
    hw_ad = cross_conformal_adaptive(y, oof_pred, folds.fold_id, np.log1p(dist / ctx.grid.dx), cov)
    metrics = {name: _metrics(y, oof[name].to_numpy(float)) for name in base_names}
    metrics["base_mean"] = _metrics(y, oof[base_names].mean(axis=1).to_numpy(float))
    metrics["stacker"] = _metrics(y, oof_pred)
    metrics["stacker"]["interval_coverage"] = float(np.mean(np.abs(y - oof_pred) <= hw))
    metrics["stacker"]["interval_target"] = cov
    metrics["stacker"]["interval_mean_halfwidth"] = float(np.mean(hw))
    metrics["stacker"]["interval_diagnostics"] = interval_diagnostics(y, oof_pred, hw, hw_ad, folds.fold_id, dist)
    wnames = other + (["physics"] if phys_oof is not None and str(scfg.get("physics_mode", "feature")) != "backbone" else [])
    stacker_info = [st.summary(wnames) for st in stackers]
    s_m = metrics["stacker"]
    progress.metric("stacker_rmse", s_m["rmse"])
    progress.metric("stacker_r2", s_m["r2"])
    progress.metric("interval_coverage", s_m["interval_coverage"])
    progress.metric("interval_halfwidth", s_m["interval_mean_halfwidth"])
    return FittedEnsemble(folds=folds, stacks=stacks, base_names=base_names, oof_base=oof, oof_pred=oof_pred,
                          halfwidth=hw, metrics=metrics, lambda_pde=(None if lam_best is None else float(lam_best)),
                          lambda_scores=scores,
                          timings={"base_models_s": t_base, "stackers_s": t_stack, "fold_model_s": fold_model_s,
                                   "stacker_candidates_s": candidate_s},
                          physics_selection=physics_selection, stacker_info=stacker_info,
                          stacker_choice=stacker_choice, halfwidth_adaptive=hw_ad, dist_train=dist)
