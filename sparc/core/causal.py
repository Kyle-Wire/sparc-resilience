r"""S6 — causal validation of a trained predictive model's implied effects.

This module does not run a parallel scenario engine.  It estimates the
causal quantities a model's scenario output *implies*, using only
design-based / orthogonalised estimators with spatial cross-fitting.  It
then audits the model's slopes and curves against them (roadmap §3 P6).
Every nuisance model is fitted on the spatial-block training folds of
:func:`sparc.core.cv.make_spatial_folds` and predicted on the held-out
blocks, so each point gets exactly one out-of-fold nuisance prediction.

Estimators
----------
**Partially linear DML** (Robinson 1988; Chernozhukov et al. 2018,
*Econometrics Journal* 21:C1).  The model is

    Y = Σ_k θ_k·T_k + g(W, B(s)) + ε,    E[ε | T, W, s] = 0,

where B(s) is a Gaussian RBF basis of the coordinates
(:func:`spatial_basis`) that absorbs smooth spatial confounding.  Let
R_Y = Y − Ê[Y|W,B] and R_T = T − Ê[T|W,B] be the cross-fitted residuals.
Then θ̂ = (R_TᵀR_T)⁻¹R_TᵀR_Y.  Standard errors use the cluster-robust
sandwich with the spatial blocks as clusters (Cameron, Gelbach & Miller
2011; Bester, Conley & Hansen 2011):
V = (R_TᵀR_T)⁻¹ (Σ_g S_gS_gᵀ) (R_TᵀR_T)⁻¹ with S_g = Σ_{i∈g} R_T,i·ε̂_i.
Under effect heterogeneity τ(x), the single-treatment θ̂ is the
overlap-weighted average θ = E[ν²τ]/E[ν²] with ν = R_T (Aronow & Samii
2016).  The weights ν²/E[ν²] are returned so users can see *where* the
estimate comes from.

**Nuisance learners.**  The default (``learner='hgb'``) is additive when
a spatial basis is supplied: g(W, s) = f(W) + h(s).  f is
HistGradientBoosting on W; h is linear in the RBF basis, with a ridge
penalty on the unstandardised basis only and α chosen by spatial-block
inner CV.  Trees fitted directly on [W, basis] (``'hgb_joint'``) can
intersect RBF features and localise far below the basis bandwidth.  On
the interference fixture they learn neighbourhood means of T and Y and
absorb most of the spillover (θ̂_nbr ≈ −0.005 to −0.007 against a
planted −0.03).
Standardising the basis inflates RBF columns centred inside held-out
blocks and makes the fit explode there.  Residual spatial confounding
remains when held-out holes (block + 2·buffer) are large relative to the
basis scale.  It biases θ̂ towards the confounded estimate and is not
reflected in the SEs, so ``hole_scale_ratio`` is reported (keep it ≲ 1).

**Spillover via exposure mapping** (Hudgens & Halloran 2008; Aronow &
Samii 2017).  The exposure T̄_i is the self-excluded, mask-normalised
Gaussian neighbourhood mean of T with σ = radius/2 (:func:`exposure_mapping`).
DML is run with treatments [T, T̄].  θ_own is the effect of a point's own
treatment with neighbours held fixed.  θ_sum = θ_own + θ_nbr is the effect
of *everyone* adopting +d: the kernel weights sum to 1, so T̄ also shifts by
d.  θ_sum is well identified even when T and T̄ are strongly collinear.
The spatial-basis scale must be ≥ 2× the exposure radius, otherwise the
basis absorbs the spillover channel; this is enforced.

**CATE** — R-learner (Nie & Wager 2021, *Biometrika* 108:299).  τ̂(x)
minimises Σ_i (R_Y,i − τ(x_i)·R_T,i)², i.e. a weighted regression of
R_Y/R_T on x with weights R_T².  It is fitted with gradient boosting,
cross-fitted over the spatial folds.  **BLP calibration** (Chernozhukov,
Demirer, Duflo & Fernández-Val 2018, "Generic ML") regresses R_Y on
[R_T, R_T·(s − s̄)].  An interaction coefficient ≈ 1 means the
heterogeneity signal s (for example a model's per-point slope) is
calibrated; ≈ 0 means it has no causal content.

**Doubly-robust dose-response** (Kennedy, Ma, McHugh & Small 2017, *JRSS-B*
79:1229).  The pseudo-outcome is

    ξ_i = (Y_i − μ̂(T_i, X_i)) / π̂(T_i|X_i) · ∫π̂(T_i|x)dP(x) + ∫μ̂(T_i, x)dP(x),

and θ(t) = E[Y(t)] is its local-linear regression on T.  μ̂ is a
gradient-boosted outcome model.  π̂ is a conditional treatment density:
a hurdle (P(T=0|x) plus a truncated Gaussian for T>0) when there is a
point mass at zero, otherwise a heteroscedastic Gaussian.  The marginals
∫…dP(x) are computed on a 50-point t-grid over a ≤2000-point subsample of
x and interpolated, so the cost is O(50·n), not O(n²).  The confidence
intervals come from a spatial-block bootstrap of the second stage, with
the cross-fitted pseudo-outcomes held fixed.  NOTE: Kennedy's estimator
assumes no interference.  Under spillovers θ(t) is an *own-exposure*
curve (neighbours at their observed values).  Compare it with a model's
OWN-ONLY partial-dependence curve, never with "set T = t everywhere".

**Sensitivity.**  E-value (VanderWeele & Ding 2017, *Ann. Intern. Med.*
167:268): the effect is standardised as d = θ·Δ/σ(R_Y), converted with
RR ≈ exp(0.91·|d|), and E = RR + √(RR(RR−1)).  Robustness value (Cinelli &
Hazlett 2020, *JRSS-B* 82:39): with f = √(R²/(1−R²)),
RV_q = ½(√(f_q⁴ + 4f_q²) − f_q²).  The significance version inflates the
critical value by the cluster design effect se_cluster/se_iid and uses
G−1 degrees of freedom.  Chernozhukov, Cinelli, Newey, Sharma &
Syrgkanis (2022), "Long story short: omitted variable bias in causal
machine learning", gives the DML-native extension of these bounds.

**DAG audit** (optional).  MC³ (Madigan & York 1995) with a BGe score,
run on spatial-block bootstrap resamples.  It flags expert edges with low
posterior support and non-expert edges with high support.  It never drives
predictions.

Everything here depends only on numpy / scipy / pandas / scikit-learn.
The optional DAG audit imports ``sparc.causal.mc3`` lazily.
"""

from __future__ import annotations

import contextlib
import io
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from sparc.core import operators as ops
from sparc.core.cv import SpatialFolds
from sparc.core.grid import Grid

logger = logging.getLogger(__name__)

__all__ = [
    "DMLResult",
    "DRCurve",
    "spatial_basis",
    "exposure_mapping",
    "dml_plr",
    "spillover_dml",
    "r_learner_cate",
    "blp_calibration",
    "dr_dose_response",
    "e_value",
    "robustness_value",
    "audit",
    "curve_audit",
    "dag_audit",
    "run_causal_validation",
]

DEFAULT_RADIUS_M = 150.0
MIN_AUTO_BASIS_SCALE_M = 1000.0

# Nuisance targets are noisy (R² ≈ 0.5) and cross-fitted folds can be small;
# a lower learning rate and smaller trees than sklearn's defaults cut the
# out-of-fold error noticeably (fixed 200 iterations, no early stopping).
_HGB_DEFAULTS: dict[str, Any] = {
    "max_iter": 200,
    "learning_rate": 0.05,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 50,
    "l2_regularization": 0.0,
    "early_stopping": False,
}

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _as_2d(W: Any, n: int | None = None) -> np.ndarray:
    """Float (n, p) array from None / 1-D / 2-D / DataFrame input."""
    if W is None:
        if n is None:
            raise ValueError("_as_2d(None) needs n")
        return np.zeros((n, 0), dtype=np.float64)
    if isinstance(W, (pd.DataFrame, pd.Series)):
        W = W.to_numpy(dtype=np.float64)
    arr = np.asarray(W, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if n is not None and arr.shape[0] != n:
        raise ValueError(f"expected {n} rows, got {arr.shape[0]}")
    return arr


def _hgb_regressor(seed: int, params: dict | None = None):
    from sklearn.ensemble import HistGradientBoostingRegressor

    kw = {**_HGB_DEFAULTS, **(params or {})}
    kw.setdefault("random_state", seed)
    return HistGradientBoostingRegressor(**kw)


def _hgb_classifier(seed: int, params: dict | None = None):
    from sklearn.ensemble import HistGradientBoostingClassifier

    kw = {**_HGB_DEFAULTS, **(params or {})}
    kw.setdefault("random_state", seed)
    return HistGradientBoostingClassifier(**kw)


class _BlockRidge:
    """Partially penalised ridge: y = a + F·w + P·c with ‖c‖² penalised only.

    The last ``n_pen`` columns (the RBF basis; None → all columns) carry an
    identity penalty and are NOT standardised.  The leading columns (the
    controls W) and the intercept are unpenalised and projected out.  Set
    ``standardize=True`` for a classic ridge on raw controls.  Two failure
    modes of an off-the-shelf standardised RidgeCV on a spatial basis
    motivate this design:

    * RBF columns whose centres fall inside a held-out block have tiny
      training variance.  Standardising inflates them, which leaves them
      practically unpenalised, and the fit explodes inside the hole.
    * GCV/LOO picks near-zero penalties on autocorrelated data.  Here α is
      chosen by grouped (spatial-block) inner cross-validation, on a grid
      relative to the top squared singular value; all α values share one
      SVD per inner split.
    """

    def __init__(self, n_pen: int | None = None, standardize: bool = False, n_inner: int = 3, seed: int = 0,
                 rel_alphas: np.ndarray | None = None):
        self.n_pen = n_pen                   # trailing penalised columns (None → all)
        self.n_free = 0
        self.standardize = standardize
        self.n_inner = n_inner
        self.seed = seed
        self.rel_alphas = np.logspace(-8, 1, 28) if rel_alphas is None else np.asarray(rel_alphas, float)

    def _split(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        F = np.column_stack([np.ones(len(X)), X[:, : self.n_free]])
        P = X[:, self.n_free:]
        if self.standardize and P.shape[1]:
            P = (P - self.mu_) / self.sd_
        return F, P

    @staticmethod
    def _path(F: np.ndarray, P: np.ndarray, y: np.ndarray, alphas: np.ndarray | None):
        """Coefficient paths (w: (nf, A), c: (kp, A)) for every α, plus S."""
        Q, R = np.linalg.qr(F)
        Pr = P - Q @ (Q.T @ P)
        yr = y - Q @ (Q.T @ y)
        U, S, Vt = np.linalg.svd(Pr, full_matrices=False)
        if alphas is None:
            return None, None, S
        D = S[None, :] / (S[None, :] ** 2 + alphas[:, None])               # (A, r)
        C = Vt.T @ (D * (U.T @ yr)[None, :]).T                              # (kp, A)
        Wc = np.linalg.lstsq(R, Q.T @ (y[:, None] - P @ C), rcond=None)[0]  # (nf, A)
        return Wc, C, S

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, p = X.shape
        self.n_free = 0 if self.n_pen is None else max(p - int(self.n_pen), 0)
        Pm = X[:, self.n_free:]
        self.mu_ = Pm.mean(axis=0)
        sd = Pm.std(axis=0)
        self.sd_ = np.where(sd > 0, sd, 1.0)
        F, P = self._split(X)
        if P.shape[1] == 0:
            self.w_ = np.linalg.lstsq(F, y, rcond=None)[0]
            self.c_ = np.zeros(0)
            self.alpha_ = 0.0
            return self
        _, _, S_full = self._path(F, P, y, None)
        alphas = self.rel_alphas * max(float(S_full[0]) ** 2, 1e-12)
        g = np.arange(n) if groups is None else np.asarray(groups)
        ug = np.unique(g)
        if ug.size < self.n_inner:
            ug, g = np.arange(n), np.arange(n)
        rng = np.random.default_rng(self.seed)
        fold_of = dict(zip(rng.permutation(ug).tolist(), (np.arange(ug.size) % self.n_inner).tolist()))
        inner = np.fromiter((fold_of[v] for v in g.tolist()), dtype=np.int64, count=n)
        errs = np.zeros(alphas.size)
        for f in range(self.n_inner):
            va = inner == f
            if va.sum() == 0 or (~va).sum() < F.shape[1] + 5:
                continue
            Wc, C, _ = self._path(F[~va], P[~va], y[~va], alphas)
            pred = F[va] @ Wc + P[va] @ C                                   # (nv, A)
            errs += ((y[va, None] - pred) ** 2).sum(axis=0)
        j = int(np.argmin(errs))
        self.alpha_ = float(alphas[j])
        Wc, C, _ = self._path(F, P, y, alphas[j:j + 1])
        self.w_, self.c_ = Wc[:, 0], C[:, 0]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        F, P = self._split(np.asarray(X, dtype=np.float64))
        out = F @ self.w_
        if self.c_.size:
            out = out + P @ self.c_
        return out


def _ridge(seed: int = 0, n_pen: int | None = None, standardize: bool = False) -> _BlockRidge:
    return _BlockRidge(n_pen=n_pen, standardize=standardize, seed=seed)


def _fit(model: Any, X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None):
    """Fit, passing spatial groups to the internal learners that use them."""
    if isinstance(model, (_BlockRidge, _SpatialAdditiveRegressor, _SpatialAdditiveClassifier)):
        return model.fit(X, y, groups=groups)
    return model.fit(X, y)


class _SpatialAdditiveRegressor:
    """g(Z, s) = f(Z) + h(s), with f boosted trees on Z and h linear in the RBF basis B.

    The last ``n_basis`` columns of X are the basis.  The fit runs three
    stages: (1) a partially penalised ridge on [Z, B], with Z free and B
    penalised (:class:`_BlockRidge`), which captures linear Z and the
    smooth spatial field jointly; (2) HGB on Z for the stage-1 residual,
    the nonlinear part of f; (3) a penalised ridge on B for the remaining
    residual.

    Why not trees on [Z, B]?  Trees can intersect many RBF features and
    localise far below the basis bandwidth.  They then learn neighbourhood
    means of T and Y, which absorbs the spillover channel and the
    short-range identifying variation.  Keeping h linear in B restricts
    the spatial adjustment to scales ≳ the basis bandwidth.
    """

    def __init__(self, n_basis: int, seed: int = 0, hgb_params: dict | None = None):
        self.n_basis = int(n_basis)
        self.seed = seed
        self.hgb_params = hgb_params

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        nb = self.n_basis
        Z, B = X[:, : X.shape[1] - nb], X[:, X.shape[1] - nb:]
        self.r1_ = _ridge(self.seed, n_pen=nb).fit(X, y, groups=groups)
        r = y - self.r1_.predict(X)
        self.f_ = None
        if Z.shape[1] > 0:
            self.f_ = _hgb_regressor(self.seed, self.hgb_params).fit(Z, r)
            r = r - self.f_.predict(Z)
        self.r2_ = _ridge(self.seed + 1).fit(B, r, groups=groups)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        nb = self.n_basis
        Z, B = X[:, : X.shape[1] - nb], X[:, X.shape[1] - nb:]
        out = self.r1_.predict(X) + self.r2_.predict(B)
        if self.f_ is not None:
            out = out + self.f_.predict(Z)
        return out


class _SpatialAdditiveClassifier:
    """P(z=1 | Z, s): gradient-boosted classifier on [Z, ĥ(s)], where
    ĥ = ridge(B → z) is a single smooth spatial offset feature.  Trees on a
    single smooth feature cannot localise below its scale."""

    def __init__(self, n_basis: int, seed: int = 0, hgb_params: dict | None = None):
        self.n_basis = int(n_basis)
        self.seed = seed
        self.hgb_params = hgb_params

    def _features(self, X: np.ndarray) -> np.ndarray:
        nb = self.n_basis
        if nb == 0:
            return X
        Z, B = X[:, : X.shape[1] - nb], X[:, X.shape[1] - nb:]
        return np.column_stack([Z, self.h_.predict(B)])

    def fit(self, X: np.ndarray, z: np.ndarray, groups: np.ndarray | None = None):
        if self.n_basis:
            self.h_ = _ridge(self.seed).fit(X[:, X.shape[1] - self.n_basis:], z.astype(np.float64), groups=groups)
        self.c_ = _hgb_classifier(self.seed, self.hgb_params).fit(self._features(X), z)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.c_.predict_proba(self._features(X))


def _make_learner(learner: Any, seed: int, hgb_params: dict | None, n_basis: int = 0):
    """Nuisance regressor factory.

    * ``'hgb'``: HistGradientBoosting on W.  With a basis (the last
      ``n_basis`` columns) it becomes the additive f(W) + h(s) learner
      (:class:`_SpatialAdditiveRegressor`), so the spatial adjustment stays
      smooth.
    * ``'hgb_joint'``: HistGradientBoosting on [W, basis] jointly.  Not
      recommended with spillovers (see above).
    * ``'ridge'``: linear in W (unpenalised) plus a penalised basis
      (:class:`_BlockRidge`); a standardised ridge when there is no basis.
    * any sklearn-style estimator: cloned and fitted on [W, basis].
    """
    if isinstance(learner, str):
        key = learner.lower()
        if key == "hgb":
            if n_basis > 0:
                return _SpatialAdditiveRegressor(n_basis, seed, hgb_params)
            return _hgb_regressor(seed, hgb_params)
        if key == "hgb_joint":
            return _hgb_regressor(seed, hgb_params)
        if key == "ridge":
            # controls unpenalised + penalised basis; plain standardised ridge without a basis
            return _ridge(seed, n_pen=n_basis) if n_basis > 0 else _ridge(seed, standardize=True)
        raise ValueError(f"unknown learner {learner!r}; use 'hgb', 'hgb_joint', 'ridge' or an estimator")
    from sklearn.base import clone

    return clone(learner)


class _ConstantModel:
    """Predicts a stored constant (fallback when there are no features / one class)."""

    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.value)


def _check_partition(folds: SpatialFolds, n: int) -> None:
    cover = np.zeros(n, dtype=np.int64)
    for te in folds.test_masks:
        te = np.asarray(te, dtype=bool)
        if te.size != n:
            raise ValueError(f"fold masks have length {te.size}, data has {n} points")
        cover += te
    if not np.all(cover == 1):
        raise ValueError("folds.test_masks must partition the points (each point in exactly one test fold)")


def _crossfit_predict(
    X: np.ndarray,
    targets: np.ndarray,
    folds: SpatialFolds,
    learner: Any = "hgb",
    seed: int = 0,
    hgb_params: dict | None = None,
    n_basis: int = 0,
) -> np.ndarray:
    """Out-of-fold predictions of every column of ``targets`` from ``X``.

    Nuisance models are fitted on ``folds.train_masks[k]`` (which carry the
    spatial buffer) and predicted on ``folds.test_masks[k]``.  The last
    ``n_basis`` columns of X are the spatial basis.
    """
    n = X.shape[0]
    targets = targets.reshape(n, -1)
    _check_partition(folds, n)
    out = np.full(targets.shape, np.nan)
    for k in range(folds.n_folds):
        tr = np.asarray(folds.train_masks[k], dtype=bool)
        te = np.asarray(folds.test_masks[k], dtype=bool)
        if not te.any():
            continue
        if tr.sum() < 10:
            raise ValueError(f"fold {k}: only {int(tr.sum())} training points after buffering")
        for j in range(targets.shape[1]):
            if X.shape[1] == 0:
                model: Any = _ConstantModel(targets[tr, j].mean())
            else:
                model = _make_learner(learner, seed + 7919 * j + 31 * k, hgb_params, n_basis)
                _fit(model, X[tr], targets[tr, j], folds.block_id[tr])
            out[te, j] = model.predict(X[te])
    return out


def _cluster_ids(block_id: np.ndarray) -> tuple[np.ndarray, int]:
    _, inv = np.unique(np.asarray(block_id), return_inverse=True)
    return inv.astype(np.int64), int(inv.max()) + 1 if inv.size else 0


def _sandwich(X: np.ndarray, eps: np.ndarray, groups: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """(cluster-robust CR1, iid) covariance of an OLS fit of eps-residual model on X."""
    n, k = X.shape
    A_inv = np.linalg.pinv(X.T @ X)
    sigma2 = float(eps @ eps) / max(n - k, 1)
    V_iid = sigma2 * A_inv
    if groups is None:
        psi = X * eps[:, None]
        meat = psi.T @ psi
        V_cl = n / max(n - k, 1) * A_inv @ meat @ A_inv        # HC1
        return V_cl, V_iid
    g, G = _cluster_ids(groups)
    S = np.zeros((G, k))
    np.add.at(S, g, X * eps[:, None])
    meat = S.T @ S
    corr = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1)) if G > 1 else 1.0
    V_cl = corr * A_inv @ meat @ A_inv
    return V_cl, V_iid


def _corr2(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = float((a @ a) * (b @ b))
    return float((a @ b) ** 2 / den) if den > 0 else 0.0


def _jsonable(obj: Any, key: str | None = None) -> Any:
    """Recursively convert numpy types to builtins (keeps ``tau_hat*`` arrays)."""
    if key is not None and str(key).startswith("tau_hat"):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v, str(k)) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return _jsonable(obj.to_dict())
    return obj


# ---------------------------------------------------------------------------
# 1. Spatial basis
# ---------------------------------------------------------------------------


def spatial_basis(
    coords: np.ndarray,
    scale_m: float,
    max_centres: int = 400,
    seed: int = 0,
    radius_m: float | None = None,
    min_centres: int = 4,
    max_fit_points: int = 20000,
) -> np.ndarray:
    """Gaussian RBF features exp(−|s − c_j|²/(2·scale²)) on k-means centres.

    The number of centres is ≈ covered area / scale², clipped to
    [``min_centres``, ``max_centres``].  The area is the count of occupied
    cells at resolution scale/2, so ragged footprints are not over-counted.
    The bandwidth equals ``scale_m``.  A regression on these features
    absorbs confounders that vary smoothly at scales ≳ ``scale_m``.

    If ``radius_m`` (an exposure radius) is given, the scale is raised to
    ≥ 2·radius so the basis cannot absorb the spillover channel.
    """
    c = np.asarray(coords, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] != 2:
        raise ValueError("coords must be (n, 2)")
    scale = float(scale_m)
    if radius_m is not None and scale < 2.0 * float(radius_m):
        logger.info("spatial_basis: scale %.0f m < 2×radius (%.0f m); raised to %.0f m",
                    scale, float(radius_m), 2.0 * float(radius_m))
        scale = 2.0 * float(radius_m)
    if not scale > 0:
        raise ValueError("scale_m must be positive")
    c = c - c.mean(axis=0)
    res = scale / 2.0
    cell = np.floor((c - c.min(axis=0)) / res).astype(np.int64)
    n_cells = np.unique(cell[:, 1] * (int(cell[:, 0].max()) + 1) + cell[:, 0]).size
    area = n_cells * res**2
    n_unique = np.unique(c, axis=0).shape[0]
    k = int(np.clip(round(area / scale**2), min_centres, max_centres))
    k = max(1, min(k, n_unique))
    rng = np.random.default_rng(seed)
    fit_pts = c if len(c) <= max_fit_points else c[rng.choice(len(c), max_fit_points, replace=False)]
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, n_init=2, max_iter=100, random_state=seed).fit(fit_pts)
    centres = km.cluster_centers_
    out = np.empty((len(c), k), dtype=np.float64)
    cc = (centres**2).sum(axis=1)
    inv2s2 = 1.0 / (2.0 * scale**2)
    for start in range(0, len(c), 10000):
        blk = c[start:start + 10000]
        d2 = (blk**2).sum(axis=1)[:, None] + cc[None, :] - 2.0 * blk @ centres.T
        out[start:start + 10000] = np.exp(-np.maximum(d2, 0.0) * inv2s2)
    logger.debug("spatial_basis: %d centres, bandwidth %.0f m, area %.2f km²", k, scale, area / 1e6)
    return out


# ---------------------------------------------------------------------------
# 2. Exposure mapping
# ---------------------------------------------------------------------------


def exposure_mapping(values: np.ndarray, grid: Grid, radius_m: float) -> np.ndarray:
    """Self-excluded Gaussian neighbourhood mean T̄ at every point.

    σ = radius/2 (in cells: σ/grid.dx).  The mean is mask-normalised over
    cells with data, so edges and holes are handled.  Where the
    self-excluded mean is undefined (isolated cells), the plain mean
    (which includes self) is used, and after that the point's own value.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.size != grid.n_points:
        raise ValueError(f"values has {v.size} entries, grid has {grid.n_points} points")
    raster = grid.rasterize(v)
    mask = np.isfinite(raster)
    sigma_cells = float(radius_m) / 2.0 / float(grid.dx)
    excl = ops.masked_gaussian(raster, mask, sigma_cells, exclude_self=True)
    out = grid.sample(excl)
    bad = ~np.isfinite(out)
    if bad.any():
        incl = grid.sample(ops.masked_gaussian(raster, mask, sigma_cells, exclude_self=False))
        out = np.where(bad, incl, out)
        bad = ~np.isfinite(out)
        if bad.any():
            out = np.where(bad, v, out)
        logger.debug("exposure_mapping: %d points used the non-excluded fallback", int(bad.sum()))
    return out


# ---------------------------------------------------------------------------
# 3. Partially linear DML
# ---------------------------------------------------------------------------


@dataclass
class DMLResult:
    """Cross-fitted partially linear model fit (see module docstring)."""

    theta: np.ndarray
    se_cluster: np.ndarray
    se_iid: np.ndarray
    ci95: np.ndarray                 # (k, 2) cluster-robust, t_{G−1}
    n: int
    n_blocks: int
    resid_Y: np.ndarray
    resid_T: np.ndarray              # (n, k)
    partial_r2: np.ndarray           # (k,) FWL partial R² of each treatment with Y
    overlap_weights: np.ndarray      # (n,) R_T[:,0]² / mean(R_T[:,0]²)
    names: list[str]
    cov_cluster: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 0)))
    cov_iid: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 0)))
    nuisance_r2: dict = field(default_factory=dict)
    n_controls: int = 0
    learner: str = "hgb"

    @property
    def t_cluster(self) -> np.ndarray:
        return self.theta / np.where(self.se_cluster > 0, self.se_cluster, np.nan)

    @property
    def sd_resid_y(self) -> float:
        return float(np.std(self.resid_Y))

    def summary(self) -> dict:
        """JSON-serialisable summary (no per-point arrays)."""
        ow = self.overlap_weights
        return _jsonable({
            "names": list(self.names),
            "theta": self.theta,
            "se_cluster": self.se_cluster,
            "se_iid": self.se_iid,
            "ci95": self.ci95,
            "t_cluster": self.t_cluster,
            "partial_r2": self.partial_r2,
            "n": self.n,
            "n_blocks": self.n_blocks,
            "n_controls": self.n_controls,
            "sd_resid_y": self.sd_resid_y,
            "sd_resid_t": np.std(self.resid_T, axis=0),
            "cov_cluster": self.cov_cluster,
            "nuisance_r2": self.nuisance_r2,
            "overlap_weight_ess": float(ow.sum() ** 2 / max((ow**2).sum(), 1e-300)),
            "learner": self.learner,
        })

    to_dict = summary


def dml_plr(
    Y: np.ndarray,
    T: np.ndarray,
    W: Any,
    folds: SpatialFolds,
    basis: np.ndarray | None = None,
    extra_treatments: np.ndarray | None = None,
    learner: Any = "hgb",
    seed: int = 0,
    names: Sequence[str] | None = None,
    hgb_params: dict | None = None,
) -> DMLResult:
    """Partially linear DML, Y = Σ θ_k·T_k + g(W, basis) + ε, with spatial cross-fitting.

    Parameters
    ----------
    Y, T : (n,) outcome and main treatment (T may be (n, k)).
    W : controls (n, p) — array, DataFrame or None.  Never include
        mediators / descendants of the treatment.
    folds : spatial folds.  Nuisances are fitted on ``train_masks[k]`` and
        predicted on ``test_masks[k]``.  ``block_id`` defines the SE clusters.
    basis : optional spatial basis (:func:`spatial_basis`) appended to W.
    extra_treatments : additional treatment columns (e.g. exposure T̄).
    learner : 'hgb' (HistGradientBoosting on W, plus a smooth additive term
        linear in the basis when one is given), 'hgb_joint' (trees on
        [W, basis]; can absorb short-range spatial structure), 'ridge', or
        an sklearn estimator (fitted on [W, basis]).
    hgb_params : overrides of the HistGradientBoosting settings.
    """
    Yv = np.asarray(Y, dtype=np.float64).ravel()
    n = Yv.size
    Tm = _as_2d(T, n)
    if extra_treatments is not None:
        Tm = np.column_stack([Tm, _as_2d(extra_treatments, n)])
    k = Tm.shape[1]
    if names is None:
        names = ["T"] + [f"T_extra{j}" for j in range(1, k)] if k > 1 else ["T"]
    names = list(names)
    if len(names) != k:
        raise ValueError(f"{len(names)} names for {k} treatments")
    Wm = _as_2d(W, n)
    Bm = _as_2d(basis, n) if basis is not None else np.zeros((n, 0))
    X = np.column_stack([Wm, Bm])
    if not (np.all(np.isfinite(Yv)) and np.all(np.isfinite(Tm))):
        raise ValueError("dml_plr: Y and T must be finite")

    targets = np.column_stack([Yv, Tm])
    pred = _crossfit_predict(X, targets, folds, learner, seed, hgb_params, n_basis=Bm.shape[1])
    RY = Yv - pred[:, 0]
    RT = Tm - pred[:, 1:]

    A = RT.T @ RT
    theta = np.linalg.lstsq(A, RT.T @ RY, rcond=None)[0]
    eps = RY - RT @ theta
    V_cl, V_iid = _sandwich(RT, eps, folds.block_id)
    se_cl = np.sqrt(np.clip(np.diag(V_cl), 0, None))
    se_iid = np.sqrt(np.clip(np.diag(V_iid), 0, None))
    _, G = _cluster_ids(folds.block_id)
    tcrit = float(stats.t.ppf(0.975, max(G - 1, 1)))
    ci = np.column_stack([theta - tcrit * se_cl, theta + tcrit * se_cl])

    pr2 = np.empty(k)
    for j in range(k):
        if k == 1:
            pr2[j] = _corr2(RY, RT[:, 0])
        else:
            others = np.delete(RT, j, axis=1)
            ry = RY - others @ np.linalg.lstsq(others, RY, rcond=None)[0]
            rt = RT[:, j] - others @ np.linalg.lstsq(others, RT[:, j], rcond=None)[0]
            pr2[j] = _corr2(ry, rt)

    v0 = RT[:, 0] ** 2
    ow = v0 / max(float(v0.mean()), 1e-300)

    def _r2(target, resid):
        vt = float(np.var(target))
        return float(1.0 - np.var(resid) / vt) if vt > 0 else float("nan")

    nuis = {"Y": _r2(Yv, RY)}
    for j, nm in enumerate(names):
        nuis[nm] = _r2(Tm[:, j], RT[:, j])
    learner_name = learner if isinstance(learner, str) else type(learner).__name__
    res = DMLResult(theta=theta, se_cluster=se_cl, se_iid=se_iid, ci95=ci, n=n, n_blocks=G, resid_Y=RY,
                    resid_T=RT, partial_r2=pr2, overlap_weights=ow, names=names, cov_cluster=V_cl,
                    cov_iid=V_iid, nuisance_r2=nuis, n_controls=int(X.shape[1]), learner=str(learner_name))
    logger.info("dml_plr: %s", ", ".join(f"{nm}={t:+.4g}±{s:.2g}" for nm, t, s in zip(names, theta, se_cl)))
    return res


# ---------------------------------------------------------------------------
# 4. Spillover DML
# ---------------------------------------------------------------------------


def _hole_scale_ratio(folds: SpatialFolds, scale: float | None) -> float | None:
    """(block + 2·buffer) / basis scale: how far the smooth spatial term must
    extrapolate into a held-out block, relative to its own length scale."""
    if not scale:
        return None
    ratio = (float(folds.block_m) + 2.0 * float(folds.buffer_m)) / float(scale)
    if ratio > 1.0:
        logger.warning("held-out holes (block %.0f m + 2×buffer %.0f m) exceed the spatial-basis scale %.0f m; "
                       "residual spatial confounding may bias θ̂ towards the unadjusted estimate",
                       folds.block_m, folds.buffer_m, scale)
    return ratio


def spillover_dml(
    Y: np.ndarray,
    T: np.ndarray,
    W: Any,
    grid: Grid,
    radius_m: float,
    folds: SpatialFolds,
    basis_scale_m: float | None,
    coords: np.ndarray | None = None,
    learner: Any = "hgb",
    seed: int = 0,
    hgb_params: dict | None = None,
    max_centres: int = 400,
) -> dict:
    """Own / neighbour / total effects from DML with treatments [T, T̄].

    T̄ = :func:`exposure_mapping` (T, grid, radius_m).  The spatial basis
    scale is raised to ≥ 2·radius_m (logged) so it cannot absorb the
    spillover.  ``basis_scale_m=None`` disables the basis.

    Returns θ_own (compare with a model's own-only slope), θ_nbr, and
    θ_sum = θ_own + θ_nbr with SE √(c'Vc), c = (1, 1).  θ_sum is the
    comparator for a model's "neighbourhood adoption" slope: everyone
    +d shifts T̄ by d as well.  JSON-serialisable.
    """
    Tv = np.asarray(T, dtype=np.float64).ravel()
    Tbar = exposure_mapping(Tv, grid, radius_m)
    basis = None
    scale = None
    if basis_scale_m is not None:
        scale = float(basis_scale_m)
        if scale < 2.0 * radius_m:
            logger.info("spillover_dml: basis scale %.0f m < 2×radius %.0f m; using %.0f m",
                        scale, radius_m, 2.0 * radius_m)
            scale = 2.0 * float(radius_m)
        xy = grid.point_coords() if coords is None else np.asarray(coords, dtype=np.float64)
        basis = spatial_basis(xy, scale, max_centres=max_centres, seed=seed)
    res = dml_plr(Y, Tv, W, folds, basis=basis, extra_treatments=Tbar, learner=learner, seed=seed,
                  names=["own", "nbr"], hgb_params=hgb_params)
    c = np.array([1.0, 1.0])
    th_sum = float(c @ res.theta)
    se_sum = float(np.sqrt(max(c @ res.cov_cluster @ c, 0.0)))
    se_sum_iid = float(np.sqrt(max(c @ res.cov_iid @ c, 0.0)))
    tcrit = float(stats.t.ppf(0.975, max(res.n_blocks - 1, 1)))
    out = {
        "theta_own": float(res.theta[0]),
        "se_own": float(res.se_cluster[0]),
        "theta_nbr": float(res.theta[1]),
        "se_nbr": float(res.se_cluster[1]),
        "cov_own_nbr": float(res.cov_cluster[0, 1]),
        "theta_sum": th_sum,
        "se_sum": se_sum,
        "se_sum_iid": se_sum_iid,
        "ci95_own": [float(v) for v in res.ci95[0]],
        "ci95_nbr": [float(v) for v in res.ci95[1]],
        "ci95_sum": [th_sum - tcrit * se_sum, th_sum + tcrit * se_sum],
        "corr_T_Tbar": float(np.corrcoef(Tv, Tbar)[0, 1]),
        "corr_resid_T_Tbar": float(np.corrcoef(res.resid_T[:, 0], res.resid_T[:, 1])[0, 1]),
        "radius_m": float(radius_m),
        "basis_scale_m": scale,
        "hole_scale_ratio": _hole_scale_ratio(folds, scale),
        "n": res.n,
        "n_blocks": res.n_blocks,
        "partial_r2": [float(v) for v in res.partial_r2],
        "nuisance_r2": res.nuisance_r2,
    }
    logger.info("spillover_dml: own %+.4g±%.2g  nbr %+.4g±%.2g  sum %+.4g±%.2g (radius %.0f m)",
                out["theta_own"], out["se_own"], out["theta_nbr"], out["se_nbr"], th_sum, se_sum, radius_m)
    return out


# ---------------------------------------------------------------------------
# 5. CATE (R-learner) and BLP calibration
# ---------------------------------------------------------------------------


def r_learner_cate(
    Y: np.ndarray | None,
    T: np.ndarray | None,
    W: Any,
    folds: SpatialFolds,
    resid_Y: np.ndarray,
    resid_T: np.ndarray,
    feature_matrix: Any = None,
    seed: int = 0,
    hgb_params: dict | None = None,
) -> np.ndarray:
    """R-learner τ̂(x) (Nie & Wager 2021), cross-fitted over spatial folds.

    Minimises Σ (R_Y − τ(x)R_T)² with gradient boosting on target
    R_Y/R_T and weights R_T².  The gradients are R_T²·τ − R_T·R_Y, so
    small residuals do not blow up the fit.  ``feature_matrix`` defaults
    to W.  Y and T are accepted for API symmetry and are not needed once
    the residuals are given.
    """
    ry = np.asarray(resid_Y, dtype=np.float64).ravel()
    rt = np.asarray(resid_T, dtype=np.float64)
    rt = rt[:, 0] if rt.ndim == 2 else rt.ravel()
    n = ry.size
    F = _as_2d(W if feature_matrix is None else feature_matrix, n)
    w = rt**2
    ok = w > 1e-10 * max(float(w.mean()), 1e-300)
    target = np.zeros(n)
    target[ok] = ry[ok] / rt[ok]
    params = {"max_iter": 100, "learning_rate": 0.05, "max_leaf_nodes": 15,
              "min_samples_leaf": max(50, n // 100), "l2_regularization": 1.0, **(hgb_params or {})}
    _check_partition(folds, n)
    tau = np.full(n, np.nan)
    for k in range(folds.n_folds):
        tr = np.asarray(folds.train_masks[k], dtype=bool) & ok
        te = np.asarray(folds.test_masks[k], dtype=bool)
        if not te.any():
            continue
        if F.shape[1] == 0:
            tau[te] = float((rt[tr] * ry[tr]).sum() / w[tr].sum())
            continue
        model = _hgb_regressor(seed + 101 * k, params)
        model.fit(F[tr], target[tr], sample_weight=w[tr])
        tau[te] = model.predict(F[te])
    logger.info("r_learner_cate: τ̂ quantiles 5/50/95%% = %s",
                np.array2string(np.quantile(tau, [0.05, 0.5, 0.95]), precision=4))
    return tau


def blp_calibration(
    resid_Y: np.ndarray,
    resid_T: np.ndarray,
    s_model: np.ndarray,
    block_id: np.ndarray | None = None,
) -> dict:
    """Best-linear-predictor calibration test of a heterogeneity signal s.

    OLS R_Y = β₁·R_T + β₂·R_T·(s − s̄) + e.  β₁ estimates the (overlap-
    weighted) average effect.  β₂ ≈ 1 means s is a calibrated predictor of
    τ(x); β₂ ≈ 0 means s carries no causal heterogeneity.  The SEs are
    cluster-robust over ``block_id`` (HC1 otherwise).  Returns
    ``coef``/``se``/``p`` for β₂ and ``ate_coef``/``ate_se`` for β₁.
    """
    ry = np.asarray(resid_Y, dtype=np.float64).ravel()
    rt = np.asarray(resid_T, dtype=np.float64)
    rt = rt[:, 0] if rt.ndim == 2 else rt.ravel()
    s = np.asarray(s_model, dtype=np.float64).ravel()
    if not np.std(s) > 0:
        logger.warning("blp_calibration: heterogeneity signal is constant; interaction not identified")
        th = float(rt @ ry / max(float(rt @ rt), 1e-300))
        return {"coef": None, "se": None, "p": None, "p_vs_one": None, "ate_coef": th, "ate_se": None,
                "ate_p": None, "sd_s": 0.0, "n": int(ry.size)}
    X = np.column_stack([rt, rt * (s - s.mean())])
    beta = np.linalg.lstsq(X, ry, rcond=None)[0]
    eps = ry - X @ beta
    V_cl, _ = _sandwich(X, eps, block_id)
    se = np.sqrt(np.clip(np.diag(V_cl), 0, None))
    if block_id is not None:
        _, G = _cluster_ids(block_id)
        dof = max(G - 1, 1)
    else:
        dof = max(ry.size - 2, 1)
    tstat = beta / np.where(se > 0, se, np.nan)
    p = 2.0 * stats.t.sf(np.abs(tstat), dof)
    p_one = float(2.0 * stats.t.sf(abs((beta[1] - 1.0) / se[1]), dof)) if se[1] > 0 else float("nan")
    return _jsonable({
        "coef": float(beta[1]),
        "se": float(se[1]),
        "p": float(p[1]),
        "p_vs_one": p_one,
        "ate_coef": float(beta[0]),
        "ate_se": float(se[0]),
        "ate_p": float(p[0]),
        "sd_s": float(np.std(s)),
        "n": int(ry.size),
    })


# ---------------------------------------------------------------------------
# 6. Doubly-robust dose-response (Kennedy et al. 2017)
# ---------------------------------------------------------------------------


@dataclass
class DRCurve:
    """Doubly-robust dose-response θ(t) = E[Y(t)] with block-bootstrap CI."""

    t_grid: np.ndarray
    theta: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    ess: float                      # Kish ESS of the (clipped) density ratios
    clipped_frac: float
    bandwidth: float
    se: np.ndarray = field(default_factory=lambda: np.zeros(0))
    n: int = 0
    n_boot: int = 0
    hurdle: bool = False
    p_zero: float = 0.0
    ratio_quantiles: dict = field(default_factory=dict)
    note: str = ("own-exposure curve (Kennedy et al. 2017 assume no interference): "
                 "compare with a model's OWN-ONLY partial-dependence curve")

    def to_dict(self) -> dict:
        return _jsonable({
            "t_grid": self.t_grid, "theta": self.theta, "lo": self.lo, "hi": self.hi, "se": self.se,
            "ess": self.ess, "clipped_frac": self.clipped_frac, "bandwidth": self.bandwidth,
            "n": self.n, "n_boot": self.n_boot, "hurdle": self.hurdle, "p_zero": self.p_zero,
            "ratio_quantiles": self.ratio_quantiles, "note": self.note,
        })


class _TreatmentDensity:
    """π(t|x): hurdle (P0(x) mass at 0 + truncated Gaussian for t > 0) or
    heteroscedastic Gaussian.  Mean and variance come from gradient boosting."""

    def __init__(self, mean_model, var_model, var_floor: float, zero_model=None, p0_const: float = 0.0,
                 hurdle: bool = False, lower: float | None = None):
        self.mean_model = mean_model
        self.var_model = var_model
        self.var_floor = float(var_floor)
        self.zero_model = zero_model
        self.p0_const = float(p0_const)
        self.hurdle = hurdle
        self.lower = lower

    def p0(self, X: np.ndarray) -> np.ndarray:
        if not self.hurdle:
            return np.zeros(len(X))
        if self.zero_model is None or X.shape[1] == 0:
            p = np.full(len(X), self.p0_const)
        else:
            p = self.zero_model.predict_proba(X)[:, 1]
        return np.clip(p, 0.01, 0.99)

    def mean_sd(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m = self.mean_model.predict(X)
        v = np.maximum(self.var_model.predict(X), self.var_floor)
        return m, np.sqrt(v)

    def _cont(self, t: np.ndarray, m: np.ndarray, s: np.ndarray) -> np.ndarray:
        dens = stats.norm.pdf(t, loc=m, scale=s)
        if self.lower is not None:
            dens = dens / np.clip(stats.norm.sf(self.lower, loc=m, scale=s), 1e-3, None)
        return dens

    def pdf(self, t: np.ndarray, X: np.ndarray, is_zero: np.ndarray) -> np.ndarray:
        m, s = self.mean_sd(X)
        dens = self._cont(t, m, s)
        if self.hurdle:
            p0 = self.p0(X)
            return np.where(is_zero, p0, (1.0 - p0) * dens)
        return dens

    def marginal(self, t_grid: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float]:
        """(∫π(t|x)dP(x) on t_grid for the continuous part, ∫P0(x)dP(x))."""
        m, s = self.mean_sd(X)
        dens = self._cont(t_grid[:, None], m[None, :], s[None, :])
        if self.hurdle:
            p0 = self.p0(X)
            return (dens * (1.0 - p0)[None, :]).mean(axis=1), float(p0.mean())
        return dens.mean(axis=1), 0.0


def _fit_treatment_density(
    T: np.ndarray,
    X: np.ndarray,
    groups: np.ndarray,
    is_zero: np.ndarray,
    hurdle: bool,
    seed: int,
    hgb_params: dict | None,
    n_basis: int = 0,
) -> _TreatmentDensity:
    """Fit π(t|x) on one training fold.  The variance model is fitted on
    out-of-sample squared residuals (a 2-way split of the training blocks),
    so it does not inherit the mean model's in-sample overfit.  The last
    ``n_basis`` columns of X are the spatial basis (smooth adjustment only)."""
    lower = None
    if hurdle:
        cont = ~is_zero
        p0_const = float(is_zero.mean())
        zero_model = None
        if X.shape[1] > 0 and 0 < is_zero.sum() < len(is_zero):
            zero_model = _SpatialAdditiveClassifier(n_basis, seed + 1, hgb_params).fit(
                X, is_zero.astype(int), groups=groups)
        if np.all(T[cont] > 0):
            lower = 0.0
    else:
        cont = np.ones(len(T), dtype=bool)
        p0_const, zero_model = 0.0, None
    Tc, Xc, gc = T[cont], X[cont], groups[cont]

    def _fit_mean(idx):
        if X.shape[1] == 0:
            return _ConstantModel(Tc[idx].mean())
        return _fit(_make_learner("hgb", seed + 2, hgb_params, n_basis), Xc[idx], Tc[idx], gc[idx])

    all_idx = np.arange(len(Tc))
    mean_model = _fit_mean(all_idx)
    # 2-way split of the training blocks for out-of-sample residuals.
    ub = np.unique(gc)
    rng = np.random.default_rng(seed + 3)
    half = set(rng.permutation(ub)[: len(ub) // 2].tolist())
    part = np.array([g in half for g in gc], dtype=bool)
    if part.sum() >= 20 and (~part).sum() >= 20 and X.shape[1] > 0:
        r = np.empty(len(Tc))
        r[part] = Tc[part] - _fit_mean(np.flatnonzero(~part)).predict(Xc[part])
        r[~part] = Tc[~part] - _fit_mean(np.flatnonzero(part)).predict(Xc[~part])
    else:
        r = Tc - mean_model.predict(Xc)
    r2 = r**2
    var_floor = 0.1 * float(r2.mean())
    if X.shape[1] == 0:
        var_model: Any = _ConstantModel(r2.mean())
    else:
        var_model = _fit(_make_learner("hgb", seed + 4, {**(hgb_params or {}), "min_samples_leaf": 50}, n_basis),
                         Xc, r2, gc)
    return _TreatmentDensity(mean_model, var_model, var_floor, zero_model, p0_const, hurdle, lower)


def _silverman(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    iqr = float(np.subtract(*np.percentile(x, [75, 25]))) / 1.34 if x.size > 1 else 0.0
    spread = min(sd, iqr) if iqr > 0 else sd
    h = 0.9 * spread * max(x.size, 1) ** (-0.2)
    return float(h) if h > 0 else 1.0


class _LocalLinear:
    """Weighted local-linear smoother with precomputed Gaussian kernel moments,
    so bootstrap refits are five matrix–vector products."""

    def __init__(self, t_obs: np.ndarray, t_eval: np.ndarray, h: float):
        d = t_obs[None, :] - t_eval[:, None]
        K = np.exp(-0.5 * (d / h) ** 2)
        self.A0, self.A1, self.A2 = K, K * d, K * d * d

    def fit(self, z: np.ndarray, w: np.ndarray) -> np.ndarray:
        wz = w * z
        S0, S1, S2 = self.A0 @ w, self.A1 @ w, self.A2 @ w
        T0, T1 = self.A0 @ wz, self.A1 @ wz
        den = S0 * S2 - S1**2
        with np.errstate(invalid="ignore", divide="ignore"):
            ll = (S2 * T0 - S1 * T1) / den
            nw = T0 / S0
        good = np.abs(den) > 1e-10 * np.maximum(S0 * S2, 1e-300)
        return np.where(good, ll, nw)


def dr_dose_response(
    Y: np.ndarray,
    T: np.ndarray,
    W: Any,
    folds: SpatialFolds,
    t_grid: np.ndarray | None = None,
    n_grid: int = 25,
    hurdle: bool = True,
    n_boot: int = 200,
    seed: int = 0,
    bandwidth: float | None = None,
    n_marg: int = 50,
    max_marg_sample: int = 2000,
    ratio_clip: tuple[float, float] = (0.02, 50.0),
    zero_tol: float = 1e-9,
    hgb_params: dict | None = None,
    basis: np.ndarray | None = None,
    exposure: np.ndarray | None = None,
) -> DRCurve:
    """Kennedy et al. (2017) doubly-robust continuous-treatment dose-response.

    Steps: (1) cross-fitted μ̂(t,x) and π̂(t|x) on the spatial folds.
    (2) The pseudo-outcome ξ uses marginals ∫…dP(x) on an ``n_marg``-point
    t-grid over a ≤ ``max_marg_sample`` subsample of x, interpolated at
    T_i.  Density ratios are clipped to ``ratio_clip``.  (3) ξ is smoothed
    on T by local-linear regression with a Gaussian kernel, using the
    Silverman bandwidth of T>0.  With a hurdle, θ(0) is the mean of ξ over
    the T=0 points and the smoother uses T>0 only.  (4) The 95% percentile
    CI comes from a spatial-block bootstrap of step (3), with ξ fixed.

    The evaluation grid is restricted to [q10, q90] of the positive
    treatment values, plus 0 when the hurdle is active.  θ(t) is an
    OWN-exposure curve (no-interference assumption).  An optional spatial
    ``basis`` enters every nuisance as a smooth additive term only.

    ``exposure`` (the neighbour exposure T̄ from :func:`exposure_mapping`)
    is added to the covariates.  Under interference T̄ confounds the own
    curve: T correlates with its neighbours' treatment, which also moves
    Y.  Adjusting for T̄ makes θ(t) the own-exposure curve with neighbours
    at their observed values, which is exactly the counterpart of a model's
    own-only partial dependence.  Without it, spillover leaks into θ(t).
    For example, on the synthetic city (random zero-canopy cells) the
    unadjusted curve jumps by ~2.7 °F between t = 0 and t = 8.7.
    """
    Yv = np.asarray(Y, dtype=np.float64).ravel()
    Tv = np.asarray(T, dtype=np.float64).ravel()
    n = Yv.size
    Wm = _as_2d(W, n)
    if exposure is not None:
        Wm = np.column_stack([Wm, _as_2d(exposure, n)])
    n_basis = 0
    if basis is not None:
        Bm = _as_2d(basis, n)
        n_basis = Bm.shape[1]
        Wm = np.column_stack([Wm, Bm])
    _check_partition(folds, n)
    is_zero = np.abs(Tv) <= zero_tol
    p_zero = float(is_zero.mean())
    use_hurdle = bool(hurdle and p_zero >= 0.05 and (~is_zero).sum() >= 20)
    cont = ~is_zero if use_hurdle else np.ones(n, dtype=bool)
    Tc = Tv[cont]
    rng = np.random.default_rng(seed)
    S = rng.choice(n, size=min(n, max_marg_sample), replace=False)
    XS = Wm[S]
    t_marg = np.linspace(Tc.min(), Tc.max(), n_marg)
    if np.ptp(t_marg) == 0:
        raise ValueError("dr_dose_response: treatment has no variation")
    grid_mu = np.r_[0.0, t_marg] if use_hurdle else t_marg
    XT = np.column_stack([Tv, Wm])

    mu_i = np.full(n, np.nan)
    mu_marg_i = np.full(n, np.nan)
    pi_i = np.full(n, np.nan)
    pi_marg_i = np.full(n, np.nan)
    for k in range(folds.n_folds):
        tr = np.asarray(folds.train_masks[k], dtype=bool)
        te = np.asarray(folds.test_masks[k], dtype=bool)
        if not te.any():
            continue
        mu = _fit(_make_learner("hgb", seed + 11 * k, hgb_params, n_basis), XT[tr], Yv[tr], folds.block_id[tr])
        mu_i[te] = mu.predict(XT[te])
        # ∫μ(t, x)dP(x) on the grid, over the global x subsample (chunked over t).
        mm = np.empty(len(grid_mu))
        for c0 in range(0, len(grid_mu), 10):
            tg_c = grid_mu[c0:c0 + 10]
            rows = np.column_stack([np.repeat(tg_c, len(S)), np.tile(XS, (len(tg_c), 1))])
            mm[c0:c0 + 10] = mu.predict(rows).reshape(len(tg_c), len(S)).mean(axis=1)
        mu_marg_i[te] = np.interp(Tv[te], grid_mu, mm)
        dens = _fit_treatment_density(Tv[tr], Wm[tr], folds.block_id[tr], is_zero[tr], use_hurdle,
                                      seed + 13 * k, hgb_params, n_basis)
        pi_i[te] = dens.pdf(Tv[te], Wm[te], is_zero[te])
        pm, p0m = dens.marginal(t_marg, XS)
        pi_marg_i[te] = np.where(is_zero[te] & use_hurdle, p0m, np.interp(Tv[te], t_marg, pm))

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_raw = pi_marg_i / pi_i
    ratio_raw = np.where(np.isfinite(ratio_raw), ratio_raw, ratio_clip[1])
    lo_c, hi_c = ratio_clip
    clipped_frac = float(np.mean((ratio_raw < lo_c) | (ratio_raw > hi_c)))
    ratio = np.clip(ratio_raw, lo_c, hi_c)
    ess = float(ratio.sum() ** 2 / (ratio**2).sum())
    xi = (Yv - mu_i) * ratio + mu_marg_i

    # Evaluation grid.
    q10, q90 = np.quantile(Tc, [0.10, 0.90])
    if t_grid is None:
        t_pos = np.linspace(q10, q90, n_grid)
    else:
        tg = np.asarray(t_grid, dtype=np.float64).ravel()
        tol = 1e-9 * max(1.0, abs(q90))
        t_pos = tg[(tg >= q10 - tol) & (tg <= q90 + tol) & ~(use_hurdle & (np.abs(tg) <= zero_tol))]
        dropped = tg.size - t_pos.size - int(use_hurdle and np.any(np.abs(tg) <= zero_tol))
        if dropped > 0:
            logger.info("dr_dose_response: %d t_grid points outside [q10, q90] = [%.3g, %.3g] dropped",
                        dropped, q10, q90)
    include_zero = use_hurdle and (t_grid is None or np.any(np.abs(np.asarray(t_grid)) <= zero_tol))
    h = float(bandwidth) if bandwidth is not None else _silverman(Tc)

    smoother = _LocalLinear(Tc, t_pos, h)
    xi_c = xi[cont]
    xi_z = xi[is_zero] if include_zero else None

    def _curve(w_all: np.ndarray) -> np.ndarray:
        parts = []
        if include_zero:
            wz = w_all[is_zero]
            parts.append(np.array([float((wz * xi_z).sum() / max(wz.sum(), 1e-300))]))
        if t_pos.size:
            parts.append(smoother.fit(xi_c, w_all[cont]))
        return np.concatenate(parts) if parts else np.zeros(0)

    theta = _curve(np.ones(n))
    t_out = np.r_[0.0, t_pos] if include_zero else t_pos

    blk, G = _cluster_ids(folds.block_id)
    boots = np.empty((n_boot, t_out.size))
    brng = np.random.default_rng(seed + 12345)
    for b in range(n_boot):
        counts = np.bincount(brng.integers(0, G, size=G), minlength=G).astype(np.float64)
        boots[b] = _curve(counts[blk])
    if n_boot > 1:
        lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
        se = np.nanstd(boots, axis=0, ddof=1)
    else:
        lo = hi = se = np.full(t_out.size, np.nan)
    rq = dict(zip(["q01", "q50", "q99"], np.quantile(ratio_raw, [0.01, 0.5, 0.99]).tolist()))
    logger.info("dr_dose_response: %d grid points, h=%.3g, ESS=%.0f, clipped=%.1f%%, hurdle=%s",
                t_out.size, h, ess, 100 * clipped_frac, use_hurdle)
    return DRCurve(t_grid=t_out, theta=theta, lo=lo, hi=hi, ess=ess, clipped_frac=clipped_frac,
                   bandwidth=h, se=se, n=n, n_boot=int(n_boot), hurdle=use_hurdle, p_zero=p_zero,
                   ratio_quantiles=rq)


# ---------------------------------------------------------------------------
# 7. Sensitivity
# ---------------------------------------------------------------------------


def _evalue_from_d(d: float) -> tuple[float, float]:
    rr = math.exp(0.91 * abs(d))
    return rr, rr + math.sqrt(rr * (rr - 1.0))


def e_value(
    theta: float,
    contrast: float,
    sd_resid_y: float,
    ci: Sequence[float] | None = None,
) -> dict:
    """E-value for a continuous outcome (VanderWeele & Ding 2017).

    d = θ·Δ/σ(R_Y) (standardised effect of the contrast Δ), RR ≈ exp(0.91|d|),
    and E = RR + √(RR(RR−1)).  If ``ci`` is given, the CI bound closest to
    the null is converted the same way.  A CI that crosses 0 gives E = 1.
    """
    sd = float(sd_resid_y)
    if not sd > 0:
        raise ValueError("sd_resid_y must be positive")
    d = float(theta) * float(contrast) / sd
    rr, ev = _evalue_from_d(d)
    out = {"d": d, "rr": rr, "e_value": ev, "contrast": float(contrast), "sd_resid_y": sd, "e_value_ci": None}
    if ci is not None:
        lo, hi = float(min(ci)), float(max(ci))
        if lo <= 0.0 <= hi:
            out["e_value_ci"] = 1.0
        else:
            bound = lo if abs(lo) < abs(hi) else hi
            out["e_value_ci"] = _evalue_from_d(bound * float(contrast) / sd)[1]
            out["ci_bound"] = bound
    return out


def _rv(f: float) -> float:
    return 0.5 * (math.sqrt(f**4 + 4.0 * f**2) - f**2) if f > 0 else 0.0


def robustness_value(
    partial_r2: float,
    t_cluster: float,
    se_cluster: float,
    se_iid: float,
    n_blocks: int,
    n: int,
    p: int,
    q: float = 1.0,
    alpha: float = 0.05,
) -> dict:
    """Cinelli & Hazlett (2020) robustness value, with clustered degrees of freedom.

    f = √(R²/(1−R²)), where R² is the treatment's partial R² with the
    outcome.  f_q = q·f and RV_q = ½(√(f_q⁴ + 4f_q²) − f_q²).  RV_q is the
    partial R² (with both treatment and outcome) that a confounder needs
    to shrink the effect by 100q%.

    Significance version: f_{q,α} = q·f − t*_{α,G−1}·(se_cluster/se_iid)/√(n−p−1),
    floored at 0.  The ratio se_cluster/se_iid (floored at 1) is the
    cluster design effect.  RV_{q,α} = ½(√(f⁴_{q,α} + 4f²_{q,α}) − f²_{q,α}).
    In the CH edge case f_q > 1/f*, the minimal confounder has
    R²_{Y~Z|D,X} < R²_{D~Z|X} and RV_{q,α} = (f_q² − f*²)/(1 + f_q²).

    Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis (2022), "Long story
    short: omitted variable bias in causal machine learning" (NBER
    w30302), extend these bounds to DML with nonparametric nuisances.
    """
    r2 = float(np.clip(partial_r2, 0.0, 1.0 - 1e-12))
    f = math.sqrt(r2 / (1.0 - r2))
    fq = q * f
    rv_q = _rv(fq)
    dof_t = max(int(n_blocks) - 1, 1)
    dof_r = max(int(n) - int(p) - 1, 1)
    tcrit = float(stats.t.ppf(1.0 - alpha / 2.0, dof_t))
    deff = float(se_cluster / se_iid) if se_iid and se_iid > 0 else 1.0
    deff = max(deff, 1.0)
    f_crit = tcrit * deff / math.sqrt(dof_r)
    fqa = max(fq - f_crit, 0.0)
    rv_qa = _rv(fqa)
    if fqa > 0 and f_crit > 0 and fq > 1.0 / f_crit:
        rv_qa = (fq**2 - f_crit**2) / (1.0 + fq**2)
    return {
        "partial_r2": r2,
        "f": f,
        "q": float(q),
        "alpha": float(alpha),
        "rv_q": rv_q,
        "rv_q_alpha": rv_qa,
        "f_crit": f_crit,
        "t_crit": tcrit,
        "design_effect": deff,
        "t_cluster": float(t_cluster) if t_cluster is not None and np.isfinite(t_cluster) else None,
        "significant_cluster": bool(t_cluster is not None and np.isfinite(t_cluster) and abs(t_cluster) > tcrit),
        "dof_blocks": dof_t,
    }


# ---------------------------------------------------------------------------
# 8. Model-vs-causal audit
# ---------------------------------------------------------------------------


def audit(
    model_slope: float,
    model_se: float | None,
    theta: float,
    theta_se: float,
    rel_tol: float = 0.25,
    z_tol: float = 2.0,
) -> dict:
    """Compare a model's implied slope s with a causal estimate θ.

    z = (s − θ)/√(SE_θ² + SE_s²).  ``flag`` is set when the difference is
    both statistically clear (|z| > z_tol) and practically material
    (|s − θ| > rel_tol·max(|θ|, |s|)).  The verdict is "sign conflict" when
    flagged and the signs disagree, "magnitude differs" when flagged, and
    "consistent" otherwise.
    """
    s, th = float(model_slope), float(theta)
    se_s = float(model_se or 0.0)
    se_t = float(theta_se or 0.0)
    den = math.sqrt(se_s**2 + se_t**2)
    diff = s - th
    z = diff / den if den > 0 else (0.0 if diff == 0 else math.copysign(math.inf, diff))
    scale = max(abs(th), abs(s))
    rel = abs(diff) / scale if scale > 0 else 0.0
    flag = bool(abs(z) > z_tol and abs(diff) > rel_tol * scale)
    sign_agree = bool(np.sign(s) == np.sign(th) or s == 0.0 or th == 0.0)
    verdict = "consistent"
    if flag:
        verdict = "sign conflict" if not sign_agree else "magnitude differs"
    return {"model": s, "model_se": se_s, "causal": th, "causal_se": se_t, "diff": diff,
            "z": z if math.isfinite(z) else None, "rel_diff": rel, "flag": flag,
            "sign_agree": sign_agree, "sign_flag": not sign_agree, "verdict": verdict}


def curve_audit(
    model_t: Sequence[float],
    model_y: Sequence[float],
    dr: DRCurve | dict,
    tol: float = 2.0,
    center: bool = True,
) -> dict:
    """Compare a model's own-only PD curve with the DR curve on the DR grid.

    The model curve is interpolated at the DR grid points inside its range.
    With ``center`` both curves are centred over the common points, so
    shapes are compared and a level offset is ignored.  The statistic is
    max_t |Δ(t)| / half-width of the DR 95% CI at t.  The level CI is
    wider than a centred-curve CI, so this ratio is conservative.  Flag if
    it exceeds ``tol``.
    """
    d = dr.to_dict() if isinstance(dr, DRCurve) else dr
    tg = np.asarray(d["t_grid"], dtype=np.float64)
    th = np.asarray(d["theta"], dtype=np.float64)
    lo = np.asarray([np.nan if v is None else v for v in d["lo"]], dtype=np.float64)
    hi = np.asarray([np.nan if v is None else v for v in d["hi"]], dtype=np.float64)
    mt = np.asarray(model_t, dtype=np.float64)
    my = np.asarray(model_y, dtype=np.float64)
    order = np.argsort(mt)
    mt, my = mt[order], my[order]
    inside = (tg >= mt.min()) & (tg <= mt.max())
    if inside.sum() < 2:
        return {"n_common": int(inside.sum()), "flag": False, "verdict": "no overlap",
                "max_ratio": None, "max_abs_diff": None}
    m_at = np.interp(tg[inside], mt, my)
    c_at = th[inside]
    if center:
        m_at = m_at - m_at.mean()
        c_at = c_at - c_at.mean()
    diff = m_at - c_at
    hw = np.maximum(0.5 * (hi[inside] - lo[inside]), 1e-12)
    ratio = np.abs(diff) / hw
    ratio = np.where(np.isfinite(ratio), ratio, 0.0)
    j = int(np.argmax(ratio))
    flag = bool(ratio[j] > tol)
    return _jsonable({
        "n_common": int(inside.sum()), "max_abs_diff": float(np.max(np.abs(diff))),
        "max_ratio": float(ratio[j]), "t_at_max": float(tg[inside][j]), "centered": center,
        "flag": flag, "verdict": "shape differs" if flag else "consistent",
    })


# ---------------------------------------------------------------------------
# 9. Optional DAG audit (MC³ on block-bootstrap resamples)
# ---------------------------------------------------------------------------


def dag_audit(
    frame: pd.DataFrame,
    node_names: Sequence[str],
    expert_edges: Iterable[tuple[str, str]],
    block_id: np.ndarray,
    n_boot: int = 5,
    n_iter: int = 2000,
    seed: int = 0,
    low: float = 0.3,
    high: float = 0.7,
) -> dict | None:
    """Posterior edge-inclusion audit of the expert DAG.

    For each of ``n_boot`` spatial-block bootstrap resamples, the data are
    standardised and ``sparc.causal.mc3.run_mc3`` is run (2 chains, BGe
    score, prior from ``PhysicsInformedGraphPrior.from_config``).  The
    edge probabilities are averaged over resamples.  The result lists
    expert edges with support < ``low`` and non-expert edges with support
    > ``high``.  It returns None (with a warning) when the MC³ module
    cannot be imported.
    """
    try:
        from sparc.causal.mc3 import PhysicsInformedGraphPrior, run_mc3
    except Exception as exc:  # pragma: no cover - depends on optional stack
        logger.warning("dag_audit skipped: cannot import sparc.causal.mc3 (%s)", exc)
        return None
    nodes = list(node_names)
    edges = [(str(p), str(c)) for p, c in expert_edges]
    prior = PhysicsInformedGraphPrior.from_config(nodes, {"edges": [{"parent": p, "child": c} for p, c in edges]})
    df = frame[nodes].astype(float).reset_index(drop=True)
    blk, G = _cluster_ids(block_id)
    members = [np.flatnonzero(blk == g) for g in range(G)]
    rng = np.random.default_rng(seed)
    probs = []
    for b in range(n_boot):
        chosen = rng.integers(0, G, size=G)
        idx = np.concatenate([members[g] for g in chosen])
        sub = df.iloc[idx].reset_index(drop=True)
        sd = sub.std(ddof=0).replace(0.0, 1.0)
        sub = (sub - sub.mean()) / sd
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = run_mc3(sub, nodes, prior, n_iter=int(n_iter), n_chains=2, min_iter=int(n_iter),
                          seed=int(seed + b))
        logger.debug("dag_audit resample %d MC³ log:\n%s", b, buf.getvalue()[-2000:])
        probs.append(np.asarray(res.edge_inclusion_probs, dtype=np.float64))
    P = np.mean(probs, axis=0)
    Psd = np.std(probs, axis=0)
    idx = {nm: i for i, nm in enumerate(nodes)}
    expert_set = {(p, c) for p, c in edges if p in idx and c in idx}
    low_support = [{"parent": p, "child": c, "prob": float(P[idx[p], idx[c]])}
                   for p, c in sorted(expert_set) if P[idx[p], idx[c]] < low]
    unexpected = [{"parent": nodes[i], "child": nodes[j], "prob": float(P[i, j])}
                  for i in range(len(nodes)) for j in range(len(nodes))
                  if i != j and (nodes[i], nodes[j]) not in expert_set and P[i, j] > high]
    logger.info("dag_audit: %d expert edges with low support, %d unexpected edges", len(low_support), len(unexpected))
    return _jsonable({
        "node_names": nodes, "edge_probs": P, "edge_probs_sd": Psd,
        "expert_edges": [list(e) for e in sorted(expert_set)],
        "expert_low_support": low_support, "unexpected_high_support": unexpected,
        "n_boot": int(n_boot), "n_iter": int(n_iter), "thresholds": {"low": low, "high": high},
    })


# ---------------------------------------------------------------------------
# 10. Orchestration
# ---------------------------------------------------------------------------


def _basis_scale(cfg_value: Any, range_m: float) -> float:
    if isinstance(cfg_value, (int, float)) and not isinstance(cfg_value, bool) and cfg_value > 0:
        return float(cfg_value)
    return max(MIN_AUTO_BASIS_SCALE_M, 2.0 * float(range_m or 0.0))


def _controls_for(t: str, frame: pd.DataFrame, cfg: dict) -> list[str]:
    conf = (cfg.get("confounders") or {}).get(t)
    excl = set((cfg.get("exclude_controls") or {}).get(t) or [])
    if conf is None:
        conf = [c for c in frame.columns if c != t]
        logger.warning("causal: no confounders listed for %s; using all other predictors minus "
                       "exclude_controls (%s) — make sure no mediator is included", t, sorted(excl))
    ctrl = [c for c in conf if c != t and c not in excl]
    missing = [c for c in ctrl if c not in frame.columns]
    if missing:
        logger.warning("causal: controls %s for %s are not in the data; dropped", missing, t)
    return [c for c in ctrl if c in frame.columns]


def _default_expert_edges(cfg: dict, treatments: list[str], target: str) -> list[tuple[str, str]]:
    edges: list[tuple[str, str]] = []
    tset = set(treatments)
    for t in treatments:
        edges.append((t, target))
        for c in (cfg.get("confounders") or {}).get(t) or []:
            if c not in tset:
                edges.append((c, t))
                edges.append((c, target))
    return sorted(set(edges))


def run_causal_validation(
    data: Any,
    cfg_causal: dict,
    folds: SpatialFolds,
    ranges_m: dict[str, float],
    model_effects: dict | None = None,
    seed: int = 0,
    hgb_params: dict | None = None,
) -> dict:
    """S6: causal validation of every configured treatment.

    Per treatment t: controls W = confounders[t] − exclude_controls[t]
    (mediators and descendants of t must never be controls).  The
    spatial-basis scale is ``spatial_basis_scale_m``, or
    max(1000 m, 2·range_t) when set to 'auto'.  Results:

    * ``dml`` — single-treatment θ (overlap-weighted average effect).
    * ``spillover`` — θ_own / θ_nbr / θ_sum with radius = ranges_m[t] (150 m default).
    * ``cate`` — R-learner τ̂ summary plus per-point ``tau_hat`` (numpy),
      and a BLP self-calibration of τ̂.
    * ``dr_curve`` — Kennedy DR own-exposure dose-response, adjusted for the
      neighbour exposure T̄ (neighbours at their observed values).
    * ``sensitivity`` — E-value (contrast[t] or 1 SD of T) and robustness values.
    * ``audit`` (if ``model_effects[t]`` is given) — adoption slope vs θ_sum,
      own slope vs θ_own, own-only PD curve vs the DR curve, and BLP
      calibration of a per-point model slope.

    The output is JSON-serialisable except arrays under ``tau_hat*`` keys.
    """
    cfg = dict(cfg_causal or {})
    if not cfg.get("enabled", True):
        return {"enabled": False}
    frame: pd.DataFrame = data.frame
    Y = np.asarray(data.y, dtype=np.float64)
    coords = np.asarray(data.coords, dtype=np.float64)
    grid: Grid = data.grid
    n_boot = int(cfg.get("n_boot", 200))
    model_effects = model_effects or {}
    treatments = [t for t in (cfg.get("treatments") or []) if t in frame.columns]
    for t in cfg.get("treatments") or []:
        if t not in frame.columns:
            logger.warning("causal: treatment %s not in data; skipped", t)
    out: dict[str, Any] = {"enabled": True, "treatments": {}, "flags": [], "n": int(len(Y)),
                           "n_blocks": int(folds.n_blocks), "n_folds": int(folds.n_folds)}

    for t in treatments:
        logger.info("causal: validating %s", t)
        T = frame[t].to_numpy(dtype=np.float64)
        ctrl = _controls_for(t, frame, cfg)
        W = frame[ctrl].to_numpy(dtype=np.float64) if ctrl else np.zeros((len(T), 0))
        rng_t = float(ranges_m.get(t, 0.0) or 0.0)
        radius = rng_t if rng_t > 0 else DEFAULT_RADIUS_M
        scale = _basis_scale(cfg.get("spatial_basis_scale_m", "auto"), rng_t)
        basis = spatial_basis(coords, scale, seed=seed)
        res_t: dict[str, Any] = {"controls": ctrl, "basis_scale_m": scale, "n_basis": int(basis.shape[1]),
                                 "radius_m": radius}

        dml = dml_plr(Y, T, W, folds, basis=basis, seed=seed, names=[t], hgb_params=hgb_params)
        res_t["dml"] = dml.summary()
        res_t["hole_scale_ratio"] = _hole_scale_ratio(folds, scale)

        logger.info("causal: %s spillover (radius %.0f m)", t, radius)
        res_t["spillover"] = spillover_dml(Y, T, W, grid, radius, folds, basis_scale_m=scale, coords=coords,
                                           seed=seed, hgb_params=hgb_params)

        logger.info("causal: %s CATE (R-learner)", t)
        feats = np.column_stack([W, coords - coords.mean(axis=0)])
        tau = r_learner_cate(Y, T, W, folds, dml.resid_Y, dml.resid_T[:, 0], feature_matrix=feats,
                             seed=seed, hgb_params=None)
        qs = [0.05, 0.25, 0.5, 0.75, 0.95]
        res_t["cate"] = {
            "quantiles": dict(zip([f"q{int(q * 100):02d}" for q in qs], np.quantile(tau, qs).tolist())),
            "mean": float(tau.mean()),
            "sd": float(tau.std()),
            "features": ctrl + ["x", "y"],
            "blp_self": blp_calibration(dml.resid_Y, dml.resid_T[:, 0], tau, folds.block_id),
            "tau_hat": tau,
        }

        logger.info("causal: %s DR dose-response (adjusted for T̄ at %.0f m)", t, radius)
        dr = dr_dose_response(Y, T, W, folds, n_boot=n_boot, seed=seed, hgb_params=hgb_params, basis=basis,
                              exposure=exposure_mapping(T, grid, radius))
        res_t["dr_curve"] = {**dr.to_dict(), "exposure_radius_m": radius}

        contrast = (cfg.get("contrast") or {}).get(t)
        contrast = float(contrast) if contrast is not None else float(np.std(T))
        ev = e_value(float(dml.theta[0]), contrast, dml.sd_resid_y, ci=dml.ci95[0])
        rv = robustness_value(float(dml.partial_r2[0]), float(dml.t_cluster[0]), float(dml.se_cluster[0]),
                              float(dml.se_iid[0]), dml.n_blocks, dml.n, dml.n_controls)
        res_t["sensitivity"] = {"e_value": ev, "robustness": rv}

        me = model_effects.get(t)
        if me:
            aud: dict[str, Any] = {}
            sp = res_t["spillover"]
            if me.get("adoption_slope") is not None:
                aud["adoption_vs_theta_sum"] = audit(me["adoption_slope"], me.get("adoption_se"),
                                                     sp["theta_sum"], sp["se_sum"])
            if me.get("own_slope") is not None:
                aud["own_vs_theta_own"] = audit(me["own_slope"], me.get("own_se"), sp["theta_own"], sp["se_own"])
            pd_curve = me.get("own_pd_curve")
            if pd_curve and len(pd_curve.get("t", [])) >= 2:
                aud["own_pd_vs_dr_curve"] = curve_audit(pd_curve["t"], pd_curve["y"], dr)
            if me.get("per_point_slope") is not None:
                aud["blp_calibration"] = blp_calibration(dml.resid_Y, dml.resid_T[:, 0],
                                                         np.asarray(me["per_point_slope"], dtype=np.float64),
                                                         folds.block_id)
            res_t["audit"] = aud
            for name, a in aud.items():
                if a.get("flag"):
                    out["flags"].append({"treatment": t, "check": name, "verdict": a.get("verdict")})
        out["treatments"][t] = res_t

    dag_cfg = cfg.get("dag_audit")
    if dag_cfg:
        dc = dag_cfg if isinstance(dag_cfg, dict) else {}
        target = "target"
        nodes = list(dict.fromkeys(treatments + [c for t in treatments for c in _controls_for(t, frame, cfg)]))
        df = frame[nodes].copy()
        df[target] = Y
        if dc.get("edges"):
            edges = [(e["parent"], e["child"]) if isinstance(e, dict) else (e[0], e[1]) for e in dc["edges"]]
        else:
            edges = _default_expert_edges(cfg, treatments, target)
        out["dag_audit"] = dag_audit(df, nodes + [target], edges, folds.block_id,
                                     n_boot=int(dc.get("n_boot", 5)), n_iter=int(dc.get("n_iter", 2000)), seed=seed)
    return _jsonable(out)
