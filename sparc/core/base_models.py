"""S2 — base models.

Every model follows the same small interface so the cross-fitting engine and
the scenario engine can treat them uniformly::

    model.fit(ctx, train_idx) -> model
    model.predict(ctx) -> np.ndarray          # predictions at every point of ctx

``ctx`` is a :class:`FeatureContext`: the raw predictor frame, the encoded +
focal feature matrix, coordinates and the raster grid.  Scenarios build a new
context from edited inputs and call ``predict`` on already-fitted models, so
``predict`` must depend only on the context (never on cached training rows).

Models
------
* :class:`OLSModel` — ridge regression on encoded + focal features.
* :class:`MGWRModel` — multiscale GWR by backfitting, computed *exactly* on the
  raster with FFT convolutions: for term j with training mask m and partial
  residual r_j,
      β_j = [K_bj ∗ (m·x_j·r_j) + κ·β_j^glob] / [K_bj ∗ (m·x_j²) + κ]
  with a Gaussian kernel of bandwidth b_j per term (initialised from the S1
  influence range and chosen by inner spatial-block CV).  κ is a pseudo-count
  (≈ 20 effective points) that shrinks towards the global coefficient where
  there is no local data (e.g. inside held-out blocks).  Besides the raw
  predictors it takes each variable's focal feature at the S1 influence range
  (scale 1): with raw inputs alone a local regression can only express the
  cell's *own* effect, and the neighbourhood effect ends up in the intercept
  surface, which no scenario can move.
* :class:`GWRFModel` — geographically weighted random forest (GRF-style):
  local random forests at k-means anchors, blended with a global gradient-
  boosting model (Georganos et al., 2019).
* :class:`GAMModel` — additive cubic splines per feature plus a Gaussian RBF
  spatial smooth, ridge-penalised with the penalty chosen by block CV.

Spatial+ (Dupont, Wood & Augustin 2022) — GAM and MGWR, ``spatial_plus=True``
---------------------------------------------------------------------------
A penalised spatial term (RBF smooth, MGWR intercept surface) competes with
the covariate terms for every spatially smooth part of a covariate's effect
and takes a share of it, so fitted effects — and scenario magnitudes — are
attenuated.  Spatial+ gives the covariate terms only the part of each
covariate (here: of each covariate *basis function*) that the model's own
spatial term cannot represent: x̃ = x − f̂_x(s), with f̂_x a spatial smooth of
the covariate on the same basis / bandwidth.  f̂_x is fitted on the baseline
covariates of all cells (no target information) and then frozen, so a
scenario edit Δx passes through the covariate terms in full while the spatial
term stays put.

Opt-in via ``models.spatial_plus`` (a list of ``mgwr`` / ``gam``, or ``true``).
On the synthetic-city benchmark (``sparc core benchmark``) Spatial+ raised
MGWR's recovered share of the planted canopy effect from 0.50 to 0.69 at
unchanged accuracy, but on Providence it worsened MGWR's held-out accuracy
(scale-dependent covariate–temperature relations violate Spatial+'s premise),
so it is off by default.  For the GAM it did not help (0.35 → 0.36) and cost
accuracy: the GAM's attenuation comes from its prediction-tuned ridge
shrinkage, not from its spatial smooth (without any spatial basis it still
recovers only 0.36).
* :class:`PhysicsBaseModel` — wraps :class:`sparc.core.physics.PhysicsModel`.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sparc.core.grid import Grid

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature context
# ---------------------------------------------------------------------------


@dataclass
class FeatureContext:
    frame: pd.DataFrame          # raw predictors (physics roles live here)
    X: pd.DataFrame              # encoded predictors (no focal)
    F: pd.DataFrame              # focal features (may be empty)
    coords: np.ndarray           # (n, 2) metres
    grid: Grid
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.frame)

    @property
    def XF(self) -> pd.DataFrame:
        if self.F is None or self.F.shape[1] == 0:
            return self.X
        return pd.concat([self.X, self.F], axis=1)


def _block_groups(coords: np.ndarray, block_m: float) -> np.ndarray:
    from sparc.core.cv import block_ids

    return block_ids(coords, block_m)


def _choose_ridge_alpha(Z: np.ndarray, y: np.ndarray, groups: np.ndarray, alphas, n_splits: int = 3) -> float:
    """Pick a ridge penalty by grouped (spatial-block) CV — LOO on
    autocorrelated 30 m data selects penalties that are far too small."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold

    uniq = np.unique(groups)
    if uniq.size < n_splits:
        return float(alphas[len(alphas) // 2])
    gkf = GroupKFold(n_splits=n_splits)
    best, best_alpha = np.inf, float(alphas[0])
    for a in alphas:
        err = 0.0
        for tr, te in gkf.split(Z, y, groups):
            m = Ridge(alpha=a).fit(Z[tr], y[tr])
            err += float(np.sum((m.predict(Z[te]) - y[te]) ** 2))
        if err < best:
            best, best_alpha = err, float(a)
    return best_alpha


class _Standardizer:
    def fit(self, Z: np.ndarray):
        self.mu = np.nanmean(Z, axis=0)
        sd = np.nanstd(Z, axis=0)
        self.sd = np.where(sd > 1e-12, sd, 1.0)
        return self

    def transform(self, Z: np.ndarray) -> np.ndarray:
        return np.nan_to_num((Z - self.mu) / self.sd)


# ---------------------------------------------------------------------------
# OLS
# ---------------------------------------------------------------------------


class OLSModel:
    name = "ols"

    def __init__(self, block_m: float = 500.0):
        self.block_m = block_m

    def fit(self, ctx: FeatureContext, train_idx: np.ndarray) -> "OLSModel":
        from sklearn.linear_model import Ridge

        Z = ctx.XF.to_numpy(float)
        self.cols = list(ctx.XF.columns)
        self.std = _Standardizer().fit(Z[train_idx])
        Zs = self.std.transform(Z[train_idx])
        y = ctx.meta["y"][train_idx]
        groups = _block_groups(ctx.coords[train_idx], self.block_m)
        self.alpha = _choose_ridge_alpha(Zs, y, groups, np.logspace(-2, 3, 6))
        self.model = Ridge(alpha=self.alpha).fit(Zs, y)
        return self

    def predict(self, ctx: FeatureContext) -> np.ndarray:
        Z = ctx.XF[self.cols].to_numpy(float)
        return self.model.predict(self.std.transform(Z))


# ---------------------------------------------------------------------------
# MGWR via FFT backfitting
# ---------------------------------------------------------------------------


def _gauss_hat(py: int, px: int, sigma_cells: float) -> np.ndarray:
    ky = np.fft.fftfreq(py)
    kx = np.fft.rfftfreq(px)
    KY, KX = np.meshgrid(ky, kx, indexing="ij")
    # Unnormalised (peak-1) Gaussian so convolutions count effective points.
    norm = 2.0 * math.pi * sigma_cells**2
    return norm * np.exp(-2.0 * (math.pi**2) * sigma_cells**2 * (KX**2 + KY**2))


class _Convolver:
    """Cached FFT Gaussian smoothing on a fixed padded raster."""

    def __init__(self, shape: tuple[int, int], pad: int):
        self.ny, self.nx = shape
        self.pad = pad
        self.py = self.ny + 2 * pad + ((self.ny + 2 * pad) % 2)
        self.px = self.nx + 2 * pad + ((self.nx + 2 * pad) % 2)
        self._hats: dict[float, np.ndarray] = {}

    def __call__(self, raster: np.ndarray, sigma_cells: float) -> np.ndarray:
        if not np.isfinite(sigma_cells):
            s = float(np.sum(raster))
            return np.full((self.ny, self.nx), s)
        key = round(float(sigma_cells), 6)
        if key not in self._hats:
            self._hats[key] = _gauss_hat(self.py, self.px, sigma_cells)
        buf = np.zeros((self.py, self.px))
        buf[: self.ny, : self.nx] = raster
        out = np.fft.irfft2(np.fft.rfft2(buf) * self._hats[key], s=(self.py, self.px))
        return out[: self.ny, : self.nx]


class MGWRModel:
    """Multiscale GWR by backfitting on the raster (exact kernel sums via FFT)."""

    name = "mgwr"

    def __init__(
        self,
        ranges_m: dict[str, float] | None = None,
        intercept_range_m: float = 500.0,
        kappa: float = 20.0,
        n_iter: int = 15,
        candidates: tuple = (0.5, 1.0, 2.0, 4.0, math.inf),
        tune: bool = True,
        tol: float = 1e-5,
        focal_scales: tuple = (1.0,),
        spatial_plus: bool = True,
    ):
        self.ranges_m = dict(ranges_m or {})
        self.focal_scales = tuple(float(s) for s in focal_scales)
        self.spatial_plus = spatial_plus
        self.smooths: list | None = None
        self.intercept_range_m = float(intercept_range_m)
        self.kappa = float(kappa)
        self.n_iter = int(n_iter)
        self.candidates = candidates
        self.tune = tune
        self.tol = tol

    # -- helpers -------------------------------------------------------
    def _rasters(self, ctx: FeatureContext) -> list[np.ndarray]:
        Z = self.std.transform(ctx.XF[self.cols].to_numpy(float))
        rs = [np.ones(ctx.grid.shape)]
        for j in range(Z.shape[1]):
            r = np.nan_to_num(ctx.grid.rasterize(Z[:, j]))
            if self.smooths is not None:       # Spatial+: frozen baseline smooth
                r = (r - self.smooths[j]) * ctx.grid.mask
            rs.append(r)
        return rs

    def _backfit(self, xs, y_r, m, sigmas, beta_glob, conv, n_iter):
        p = len(xs)
        betas = [np.full(xs[0].shape, beta_glob[j]) for j in range(p)]
        denoms = [conv(m * xs[j] ** 2, sigmas[j]) + self.kappa for j in range(p)]
        fit = sum(betas[j] * xs[j] for j in range(p))
        prev = np.inf
        for _ in range(n_iter):
            for j in range(p):
                partial = (y_r - fit + betas[j] * xs[j]) * m
                num = conv(m * xs[j] * partial, sigmas[j]) + self.kappa * beta_glob[j]
                new = num / denoms[j]
                fit = fit + (new - betas[j]) * xs[j]
                betas[j] = new
            sse = float(np.sum(((y_r - fit) * m) ** 2))
            if abs(prev - sse) <= self.tol * max(prev, 1e-12):
                break
            prev = sse
        return betas

    def _sigma_cells(self, name: str, mult: float, dx: float) -> float:
        var = name.split("__f")[0] if "__f" in name else name
        base = self.intercept_range_m if name == "__intercept__" else self.ranges_m.get(var, self.intercept_range_m)
        if not np.isfinite(mult):
            return math.inf
        return max(0.5, mult * base / 2.0 / dx)

    # -- API -----------------------------------------------------------
    def fit(self, ctx: FeatureContext, train_idx: np.ndarray) -> "MGWRModel":
        g = ctx.grid
        wanted = {f"__f{s:g}" for s in self.focal_scales}
        self.cols = list(ctx.X.columns) + [c for c in ctx.F.columns if any(c.endswith(w) for w in wanted)]
        Z = ctx.XF[self.cols].to_numpy(float)
        self.std = _Standardizer().fit(Z[train_idx])
        y = ctx.meta["y"]
        names = ["__intercept__"] + self.cols
        max_range = max([self.intercept_range_m] + list(self.ranges_m.values()))
        pad = int(math.ceil(3.0 * 4.0 * max_range / 2.0 / g.dx)) + 2
        conv = _Convolver(g.shape, min(pad, max(g.shape)))
        self.smooths = None
        if self.spatial_plus:
            # Normalised Gaussian smooth of each regressor at the intercept
            # bandwidth over all observed cells (covariates only), frozen.
            m_all = g.mask.astype(float)
            sig = self._sigma_cells("__intercept__", 1.0, g.dx)
            den = np.maximum(conv(m_all, sig), 1e-9)
            Zs = self.std.transform(Z)
            self.smooths = [conv(np.nan_to_num(g.rasterize(Zs[:, j])) * m_all, sig) / den * m_all
                            for j in range(Zs.shape[1])]
        xs = self._rasters(ctx)
        m = np.isfinite(g.rasterize(np.ones(ctx.n), subset=train_idx)).astype(float)
        y_r = np.nan_to_num(g.rasterize(y, subset=train_idx))
        # Global OLS coefficients (shrinkage target), on the same (Spatial+
        # residualised) regressors the local fits use.
        design = np.column_stack([np.ones(len(train_idx)), self._point_regressors(ctx)[train_idx]])
        self.beta_glob = np.linalg.lstsq(design, y[train_idx], rcond=None)[0]
        mults = {n: 1.0 for n in names}

        if self.tune and len(train_idx) > 200:
            # Inner 2-fold spatial-block split of the training cells.
            from sparc.core.cv import block_ids

            blk = block_ids(ctx.coords[train_idx], max(self.intercept_range_m, 3 * g.dx))
            rng = np.random.default_rng(0)
            half = rng.permutation(np.unique(blk))[: max(1, np.unique(blk).size // 2)]
            inner_tr = train_idx[~np.isin(blk, half)]
            inner_te = train_idx[np.isin(blk, half)]
            m_in = np.isfinite(g.rasterize(np.ones(ctx.n), subset=inner_tr)).astype(float)
            y_in = np.nan_to_num(g.rasterize(y, subset=inner_tr))
            te_rows, te_cols = g.iy[inner_te], g.ix[inner_te]

            def score(ms):
                sig = [self._sigma_cells(n, ms[n], g.dx) for n in names]
                b = self._backfit(xs, y_in, m_in, sig, self.beta_glob, conv, max(5, self.n_iter // 3))
                pred = sum(b[j][te_rows, te_cols] * xs[j][te_rows, te_cols] for j in range(len(names)))
                return float(np.mean((pred - y[inner_te]) ** 2))

            for n in names:  # one coordinate-wise pass
                best_mult, best_s = mults[n], score(mults)
                for c in self.candidates:
                    if c == mults[n]:
                        continue
                    trial = dict(mults)
                    trial[n] = c
                    s = score(trial)
                    if s < best_s:
                        best_s, best_mult = s, c
                mults[n] = best_mult

        self.multipliers = mults
        self.sigmas = [self._sigma_cells(n, mults[n], g.dx) for n in names]
        self.bandwidths_m = {n: (float("inf") if not np.isfinite(s) else 2.0 * s * g.dx) for n, s in zip(names, self.sigmas)}
        self.betas = self._backfit(xs, y_r, m, self.sigmas, self.beta_glob, conv, self.n_iter)
        self.names = names
        return self

    def _point_regressors(self, ctx: FeatureContext) -> np.ndarray:
        """Standardised regressors at the points (minus the frozen Spatial+ smooth)."""
        Z = self.std.transform(ctx.XF[self.cols].to_numpy(float))
        if self.smooths is not None:
            g = ctx.grid
            Z = Z - np.column_stack([sm[g.iy, g.ix] for sm in self.smooths])
        return Z

    def coefficient_rasters(self) -> dict[str, np.ndarray]:
        return dict(zip(self.names, self.betas))

    def predict(self, ctx: FeatureContext) -> np.ndarray:
        g = ctx.grid
        # Point-level x (not rasterised) so collisions keep their own values.
        Z = self._point_regressors(ctx)
        pred = self.betas[0][g.iy, g.ix].copy()
        for j in range(1, len(self.names)):
            pred += self.betas[j][g.iy, g.ix] * Z[:, j - 1]
        return pred


# ---------------------------------------------------------------------------
# Geographically weighted random forest (GRF-lite)
# ---------------------------------------------------------------------------


class GWRFModel:
    name = "gwrf"

    def __init__(self, n_anchors: int = 48, local_n: int = 3000, n_trees: int = 50, n_nearest: int = 3,
                 local_weight: float = 0.5, seed: int = 0, n_jobs: int = 4):
        self.n_anchors = n_anchors
        self.local_n = local_n
        self.n_trees = n_trees
        self.n_nearest = n_nearest
        self.local_weight = local_weight
        self.seed = seed
        self.n_jobs = n_jobs

    def fit(self, ctx: FeatureContext, train_idx: np.ndarray) -> "GWRFModel":
        from scipy.spatial import cKDTree
        from sklearn.cluster import KMeans
        from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

        self.cols = list(ctx.XF.columns)
        Z = ctx.XF.to_numpy(float)
        y = ctx.meta["y"]
        Ztr, ytr, ctr = Z[train_idx], y[train_idx], ctx.coords[train_idx]
        self.global_model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                                          min_samples_leaf=40, l2_regularization=1.0,
                                                          random_state=self.seed).fit(Ztr, ytr)
        k = int(min(self.n_anchors, max(1, len(train_idx) // 400)))
        km = KMeans(n_clusters=k, n_init=3, random_state=self.seed).fit(ctr)
        self.anchors = km.cluster_centers_
        tree = cKDTree(ctr)
        self.local_models = []
        for a in self.anchors:
            _, nn = tree.query(a, k=min(self.local_n, len(train_idx)))
            rf = RandomForestRegressor(n_estimators=self.n_trees, min_samples_leaf=5, max_features=0.6,
                                       n_jobs=self.n_jobs, random_state=self.seed).fit(Ztr[nn], ytr[nn])
            self.local_models.append(rf)
        if k > 1:
            d, _ = cKDTree(self.anchors).query(self.anchors, k=2)
            self.anchor_bw = float(np.median(d[:, 1]))
        else:
            self.anchor_bw = 1e12
        return self

    def predict(self, ctx: FeatureContext) -> np.ndarray:
        from scipy.spatial import cKDTree

        Z = ctx.XF[self.cols].to_numpy(float)
        g_pred = self.global_model.predict(Z)
        kk = min(self.n_nearest, len(self.anchors))
        d, idx = cKDTree(self.anchors).query(ctx.coords, k=kk)
        if kk == 1:
            d, idx = d[:, None], idx[:, None]
        w = np.exp(-0.5 * (d / self.anchor_bw) ** 2) + 1e-12
        w = w / w.sum(axis=1, keepdims=True)
        local = np.zeros(ctx.n)
        for a, rf in enumerate(self.local_models):
            rows, slot = np.nonzero(idx == a)
            if rows.size:
                local[rows] += w[rows, slot] * rf.predict(Z[rows])
        return self.local_weight * local + (1.0 - self.local_weight) * g_pred


# ---------------------------------------------------------------------------
# GAM: additive splines + spatial RBF smooth
# ---------------------------------------------------------------------------


class GAMModel:
    name = "gam"

    def __init__(self, n_knots: int = 8, spatial_scale_m: float = 800.0, max_centres: int = 300, block_m: float = 500.0,
                 seed: int = 0, spatial_plus: bool = False):
        self.n_knots = n_knots
        self.spatial_scale_m = spatial_scale_m
        self.max_centres = max_centres
        self.block_m = block_m
        self.seed = seed
        self.spatial_plus = spatial_plus
        self.W = None

    def _parts(self, ctx: FeatureContext) -> tuple[np.ndarray, np.ndarray]:
        Z = ctx.XF[self.cols].to_numpy(float)
        Bx = self.splines.transform(np.clip(Z, self.lo, self.hi))
        d2 = ((ctx.coords[:, None, :] - self.centres[None, :, :]) ** 2).sum(-1)
        return Bx, np.exp(-0.5 * d2 / self.spatial_scale_m**2)

    def _basis(self, ctx: FeatureContext) -> np.ndarray:
        Bx, S = self._parts(ctx)
        if self.W is not None:              # Spatial+: frozen baseline covariate smooth
            Bx = Bx - S @ self.W
        return np.hstack([Bx, S])

    def fit(self, ctx: FeatureContext, train_idx: np.ndarray) -> "GAMModel":
        from sklearn.cluster import KMeans
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import SplineTransformer

        self.cols = list(ctx.XF.columns)
        Z = ctx.XF.to_numpy(float)[train_idx]
        self.lo, self.hi = np.nanmin(Z, axis=0), np.nanmax(Z, axis=0)
        self.splines = SplineTransformer(n_knots=self.n_knots, degree=3, knots="quantile",
                                         extrapolation="constant").fit(Z)
        c = ctx.coords[train_idx]
        area = np.ptp(c[:, 0]) * np.ptp(c[:, 1])
        k = int(np.clip(area / self.spatial_scale_m**2, 4, self.max_centres))
        self.centres = KMeans(n_clusters=k, n_init=2, random_state=self.seed).fit(c).cluster_centers_
        self.W = None
        if self.spatial_plus:
            # Smooth of every spline column on the RBF basis over all cells
            # (covariates only — no target information), then frozen.
            Bx, S = self._parts(ctx)
            G = S.T @ S
            lam = 1e-3 * float(np.trace(G)) / G.shape[0]
            self.W = np.linalg.solve(G + lam * np.eye(G.shape[0]), S.T @ Bx)
        B_all = self._basis(ctx)
        B = B_all[train_idx]
        self.std = _Standardizer().fit(B)
        Bs = self.std.transform(B)
        y = ctx.meta["y"][train_idx]
        groups = _block_groups(c, self.block_m)
        self.alpha = _choose_ridge_alpha(Bs, y, groups, np.logspace(-1, 4, 6))
        self.model = Ridge(alpha=self.alpha).fit(Bs, y)
        return self

    def predict(self, ctx: FeatureContext) -> np.ndarray:
        return self.model.predict(self.std.transform(self._basis(ctx)))


# ---------------------------------------------------------------------------
# Physics wrapper
# ---------------------------------------------------------------------------


class PhysicsBaseModel:
    name = "physics"

    def __init__(self, grid: Grid, cfg_physics: dict, L_init: float | None = None, seed: int = 0):
        self.grid = grid
        self.cfg_physics = cfg_physics
        self.L_init = L_init
        self.seed = seed

    def fit(self, ctx: FeatureContext, train_idx: np.ndarray) -> "PhysicsBaseModel":
        from sparc.core.physics import PhysicsModel

        self.model = PhysicsModel(ctx.grid, self.cfg_physics, L_init=self.L_init, seed=self.seed)
        self.model.fit(ctx.frame, ctx.meta["y"], train_idx, ctx.coords)
        return self

    def predict(self, ctx: FeatureContext) -> np.ndarray:
        return self.model.predict(ctx.frame, ctx.coords)

    @property
    def params(self) -> dict:
        return self.model.params


def build_base_models(cfg_models: dict, *, grid: Grid, cfg_physics: dict, ranges_m: dict[str, float],
                      intercept_range_m: float, block_m: float, L_init: float | None, seed: int = 0) -> list:
    """Instantiate the enabled base models (fresh, unfitted)."""
    models: list = []
    sp_cfg = cfg_models.get("spatial_plus", [])
    sp_set = {"mgwr", "gam"} if sp_cfg is True else set() if not sp_cfg else {str(m).lower() for m in sp_cfg}
    if cfg_models.get("ols", True):
        models.append(OLSModel(block_m=block_m))
    if cfg_models.get("mgwr", True):
        models.append(MGWRModel(ranges_m=ranges_m, intercept_range_m=intercept_range_m,
                                spatial_plus="mgwr" in sp_set))
    if cfg_models.get("gwrf", True):
        models.append(GWRFModel(seed=seed))
    if cfg_models.get("gam", True):
        models.append(GAMModel(spatial_scale_m=max(2 * intercept_range_m, 300.0), block_m=block_m, seed=seed,
                               spatial_plus="gam" in sp_set))
    if cfg_models.get("physics", True) and cfg_physics.get("enabled", True) and (cfg_physics.get("roles") or {}):
        models.append(PhysicsBaseModel(grid, cfg_physics, L_init=L_init, seed=seed))
    return models
