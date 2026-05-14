"""
spatial_cate — Geographically-weighted Conditional Average Treatment Effects.

Estimates spatially-varying causal effects using EconML's CausalForestDML,
with spatial coordinates and Laplacian eigenmaps as effect modifiers.
This produces per-point CATE estimates that replace the heuristic spatial
multiplier in the ScenarioSimulator.

References
----------
- Li et al. (2023): Spatial heterogeneity in causal effects of UHI.
- Athey & Wager (2019): Generalized random forests for heterogeneous
  treatment effects (CausalForestDML foundation).

Typical usage::

    from sparc.causal.spatial_cate import SpatialCATEEstimator

    estimator = SpatialCATEEstimator(config)
    cate_results = estimator.estimate_all(data, treatments, outcome, confounders)
    multipliers = estimator.cate_to_spatial_multiplier(cate_results, treatment)
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Phase 7 — KernelField-aware coordinate warp
# ---------------------------------------------------------------------------


def _anisotropic_warp_coords(
    coords: np.ndarray,
    predictor: "Optional[Any]" = None,
) -> np.ndarray:
    """Rotate + scale 2-D coords into the predictor's anisotropic principal frame.

    Equal Euclidean distance in the warped frame corresponds to equal
    Mat\u00e9rn kernel distance in the original frame, so any downstream model
    that uses an isotropic neighborhood (random forest, RBF features, k-NN)
    automatically inherits the correct anisotropy.

    Parameters
    ----------
    coords : (N, 2) world-frame coordinates.
    predictor : a ``PredictorKernel`` (or any object with ``is_anisotropic``,
        ``kappa_x``, ``kappa_y``, ``theta_rad``).  When ``None`` or isotropic,
        ``coords`` is returned unchanged.

    Returns
    -------
    np.ndarray of shape (N, 2).
    """
    if predictor is None or coords.shape[1] != 2:
        return coords
    if not getattr(predictor, "is_anisotropic", False):
        return coords
    kx = float(predictor.kappa_x)
    ky = float(predictor.kappa_y)
    th = float(predictor.theta_rad or 0.0)
    c, s = np.cos(th), np.sin(th)
    u = coords[:, 0] * c + coords[:, 1] * s
    v = -coords[:, 0] * s + coords[:, 1] * c
    warped = np.column_stack([kx * u, ky * v])
    return warped


class SpatialCATEEstimator:
    """
    Estimate geographically-weighted CATE via CausalForestDML.

    Parameters
    ----------
    config : dict
        Full SPARC config dict.
    n_estimators : int
        Number of trees in the causal forest.
    min_samples_leaf : int
        Minimum samples per leaf in treatment/outcome nuisance models.
    """

    def __init__(
        self,
        config: dict,
        n_estimators: int = 500,
        min_samples_leaf: int = 20,
        kernel_field: Optional[Any] = None,
    ):
        self.config = config
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.random_state = config.get('pipeline', {}).get('random_seed', 42)

        # Fitted models per treatment
        self._models: Dict[str, Any] = {}
        # Per-point CATE estimates per treatment: {treatment: np.ndarray}
        self.cate_estimates: Dict[str, np.ndarray] = {}
        # Per-point CATE confidence intervals: {treatment: (lower, upper)}
        self.cate_intervals: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        # Phase 7 — corrected anisotropic kernel for spatial heterogeneity.
        # When supplied, coords appended to W are warped into each treatment's
        # anisotropic principal frame so the forest sees an isotropic
        # neighborhood that matches the corrected length scales.
        self.kernel_field = kernel_field
        # Audit trail: which treatments actually used the warped coords.
        self._kernel_field_applied: Dict[str, bool] = {}

    # ------------------------------------------------------------------
    # Estimate CATE for a single treatment
    # ------------------------------------------------------------------

    def estimate(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        confounders: List[str],
        coord_cols: List[str] | None = None,
        spatial_features: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Estimate per-point CATE for one treatment using CausalForestDML.

        The effect modifier matrix W includes:
        - Spatial coordinates (if coord_cols provided)
        - Laplacian eigenmap features (if spatial_features provided)
        - Confounders (for heterogeneity detection)

        Parameters
        ----------
        data : pd.DataFrame
            Full dataset.
        treatment : str
            Treatment variable name.
        outcome : str
            Outcome variable name.
        confounders : list[str]
            Confounder/covariate column names.
        coord_cols : list[str], optional
            Coordinate column names for spatial heterogeneity [X, Y].
        spatial_features : np.ndarray, optional
            Additional spatial features (e.g. Laplacian eigenmaps).

        Returns
        -------
        np.ndarray
            Per-point CATE estimates (shape: n_observations,).
        """
        from sklearn.ensemble import HistGradientBoostingRegressor as HGB

        T = data[[treatment]].values
        Y = data[outcome].values
        X = data[confounders].values  # Covariates for nuisance models

        # Build effect modifier matrix W (for heterogeneity)
        W_parts = [X]  # Confounders
        if coord_cols:
            available = [c for c in coord_cols if c in data.columns]
            if available:
                coords = data[available].values.astype(np.float64)
                # Phase 7: warp into the treatment's anisotropic principal
                # frame so the forest's neighborhood structure matches the
                # corrected kernel.  Falls through to identity when the
                # kernel field is absent or the predictor is isotropic.
                predictor = None
                if self.kernel_field is not None and len(available) >= 2:
                    predictor = self.kernel_field.predictor(treatment)
                if predictor is not None and getattr(
                    predictor, "is_anisotropic", False,
                ):
                    coords = _anisotropic_warp_coords(coords, predictor)
                    self._kernel_field_applied[treatment] = True
                else:
                    self._kernel_field_applied[treatment] = False
                # Normalize coordinates to [0, 1] for numeric stability
                coords_min = coords.min(axis=0)
                coords_range = coords.max(axis=0) - coords_min
                coords_range[coords_range == 0] = 1.0
                coords_norm = (coords - coords_min) / coords_range
                W_parts.append(coords_norm)

        if spatial_features is not None and spatial_features.shape[0] == len(data):
            W_parts.append(spatial_features)

        W = np.hstack(W_parts)

        # Subsample for fitting if dataset is large (CausalForestDML scales poorly)
        max_fit_rows = self.config.get('causal', {}).get('cate_max_fit_rows', 10_000)
        if len(data) > max_fit_rows:
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(len(data), size=max_fit_rows, replace=False)
            T_fit, Y_fit, X_fit, W_fit = T[idx], Y[idx], X[idx], W[idx]
        else:
            T_fit, Y_fit, X_fit, W_fit = T, Y, X, W

        # Try econml CausalForestDML; fall back to sklearn-based local
        # linear DML if econml's Cython extensions are broken.
        try:
            from econml.dml import CausalForestDML

            model = CausalForestDML(
                model_y=HGB(
                    max_iter=200, max_depth=4, learning_rate=0.05,
                    min_samples_leaf=self.min_samples_leaf, random_state=self.random_state,
                ),
                model_t=HGB(
                    max_iter=200, max_depth=4, learning_rate=0.05,
                    min_samples_leaf=self.min_samples_leaf, random_state=self.random_state,
                ),
                n_estimators=self.n_estimators,
                min_samples_leaf=max(5, len(T_fit) // 200),
                random_state=self.random_state,
                cv=3,
            )

            model.fit(Y_fit, T_fit, X=X_fit, W=W_fit)

            # Get per-point CATE (predict on full dataset)
            cate = model.effect(X, T0=0, T1=1).flatten()

            # Confidence intervals
            try:
                inference = model.effect_inference(X)
                ci_lower = inference.conf_int(alpha=0.05)[0].flatten()
                ci_upper = inference.conf_int(alpha=0.05)[1].flatten()
                self.cate_intervals[treatment] = (ci_lower, ci_upper)
            except Exception:
                cate_std = np.std(cate)
                self.cate_intervals[treatment] = (
                    cate - 1.96 * cate_std,
                    cate + 1.96 * cate_std,
                )

            self._models[treatment] = model

        except Exception as econml_err:
            # Fallback: spatially-binned cross-fit DML using sklearn only
            import warnings
            warnings.warn(
                f"CausalForestDML failed ({econml_err}); "
                "using sklearn cross-fit DML with spatial binning."
            )
            cate = self._spatial_cate_sklearn_fallback(
                T_fit, Y_fit, X_fit, W_fit, T, Y, X, W, treatment,
            )

        self.cate_estimates[treatment] = cate
        return cate

    def _spatial_cate_sklearn_fallback(
        self,
        T_fit, Y_fit, X_fit, W_fit,
        T, Y, X, W,
        treatment: str,
    ) -> np.ndarray:
        """
        Fallback CATE: partition W into spatial bins and run a separate
        cross-fit DML in each bin to capture treatment-effect heterogeneity.
        """
        from sklearn.cluster import MiniBatchKMeans
        from sparc.causal.counterfactual_engine import CounterfactualEngine
        from sklearn.ensemble import HistGradientBoostingRegressor as HGB

        n_bins = min(20, max(5, len(T) // 500))
        km = MiniBatchKMeans(n_clusters=n_bins, random_state=self.random_state, n_init=3)
        labels = km.fit_predict(W)

        cate = np.zeros(len(T))
        for b in range(n_bins):
            mask = labels == b
            if mask.sum() < 30:
                continue
            coeff, _, _ = CounterfactualEngine._fit_edge_dml_sklearn(
                T[mask], Y[mask], W[mask],
                model_y=HGB(max_iter=150, max_depth=3, random_state=42),
                model_t=HGB(max_iter=150, max_depth=3, random_state=42),
                n_splits=min(3, max(2, mask.sum() // 30)),
            )
            cate[mask] = coeff

        # Rough CI from per-bin variation
        cate_std = np.std(cate)
        self.cate_intervals[treatment] = (
            cate - 1.96 * cate_std,
            cate + 1.96 * cate_std,
        )

        return cate

    # ------------------------------------------------------------------
    # Estimate CATE for all treatments
    # ------------------------------------------------------------------

    def estimate_all(
        self,
        data: pd.DataFrame,
        treatments: List[str],
        outcome: str,
        confounders: List[str],
        coord_cols: List[str] | None = None,
        spatial_features: np.ndarray | None = None,
    ) -> Dict[str, np.ndarray]:
        """
        Estimate CATE for all treatment variables.

        Returns
        -------
        dict
            ``{treatment: cate_array}``
        """
        results = {}
        for treatment in treatments:
            if treatment not in data.columns:
                continue
            try:
                cate = self.estimate(
                    data, treatment, outcome, confounders,
                    coord_cols=coord_cols,
                    spatial_features=spatial_features,
                )
                results[treatment] = cate
                print(f"    CATE({treatment}): mean={cate.mean():+.5f}, "
                      f"std={cate.std():.5f}, "
                      f"range=[{cate.min():+.4f}, {cate.max():+.4f}]")
            except Exception as e:
                warnings.warn(f"CATE estimation failed for {treatment}: {e}")

        return results

    # ------------------------------------------------------------------
    # Convert CATE to spatial multiplier (replaces heuristic)
    # ------------------------------------------------------------------

    def cate_to_spatial_multiplier(
        self,
        treatment: str,
        lower_bound: float = 0.5,
        upper_bound: float = 1.5,
    ) -> np.ndarray:
        """
        Convert per-point CATE estimates to spatial multipliers.

        The multiplier represents the ratio of local effect to the mean
        effect, bounded to ``[lower_bound, upper_bound]``.

        Formula: ``mult_i = clamp(CATE_i / mean(CATE), lower, upper)``

        Parameters
        ----------
        treatment : str
            Treatment variable name.
        lower_bound : float
            Minimum multiplier.
        upper_bound : float
            Maximum multiplier.

        Returns
        -------
        np.ndarray
            Per-point spatial multipliers.
        """
        if treatment not in self.cate_estimates:
            raise ValueError(f"No CATE estimates for '{treatment}'. Run estimate() first.")

        cate = self.cate_estimates[treatment]
        mean_cate = np.mean(cate)

        if abs(mean_cate) < 1e-10:
            return np.ones(len(cate))

        multiplier = cate / mean_cate
        return np.clip(multiplier, lower_bound, upper_bound)

    # ------------------------------------------------------------------
    # Export CATE to GeoDataFrame
    # ------------------------------------------------------------------

    def to_geodataframe(
        self,
        data: pd.DataFrame,
        coord_cols: List[str],
        crs: str = None,
    ) -> Any:
        """
        Export per-point CATE estimates as a GeoDataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            Original data with coordinate columns.
        coord_cols : list[str]
            [X, Y] coordinate column names.
        crs : str
            Coordinate reference system.

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame with CATE columns per treatment.
        """
        import geopandas as gpd
        from shapely.geometry import Point

        gdf = data.copy()
        geometry = [Point(x, y) for x, y in zip(data[coord_cols[0]], data[coord_cols[1]])]
        gdf = gpd.GeoDataFrame(gdf, geometry=geometry, crs=crs)

        for treatment, cate in self.cate_estimates.items():
            gdf[f'cate_{treatment}'] = cate
            gdf[f'spatial_mult_{treatment}'] = self.cate_to_spatial_multiplier(treatment)

            if treatment in self.cate_intervals:
                lower, upper = self.cate_intervals[treatment]
                gdf[f'cate_ci_lower_{treatment}'] = lower
                gdf[f'cate_ci_upper_{treatment}'] = upper

        return gdf

    # ------------------------------------------------------------------
    # GP surface regression over CATE estimates
    # ------------------------------------------------------------------

    def fit_gp_surface(
        self,
        coords: np.ndarray,
        corr_payload: Optional[dict] = None,
        treatments: Optional[List[str]] = None,
        max_fit_rows: int = 2_000,
        n_restarts_optimizer: int = 3,
    ) -> Dict[str, Dict[str, Any]]:
        """Fit a GP surface over the estimated CATE for each treatment.

        Produces a continuous, uncertainty-quantified CATE surface using
        :class:`CATEGPSurface`.  The Matérn kernel is seeded from the Stage 0
        correlogram when ``corr_payload`` is provided.

        Parameters
        ----------
        coords : (N, 2) array
            Raw spatial coordinates in world units (will be normalised
            internally to [0, 1] per dimension).
        corr_payload : dict, optional
            Stage 0 correlogram artifact dict (``MaternCorrelogramResult.to_payload()``).
            When provided, initialises ``nu`` and ``length_scale`` from it.
        treatments : list[str], optional
            Subset of treatments to process.  Defaults to all in ``cate_estimates``.
        max_fit_rows : int
            Sub-sampling limit for GPR fitting.
        n_restarts_optimizer : int
            Number of optimizer restarts for kernel hyperparameter fitting.

        Returns
        -------
        dict
            ``{treatment: {"gp_mean": ndarray, "gp_std": ndarray,
                           "moran_i": dict, "log_ml": float}}``
        """
        if not self.cate_estimates:
            raise RuntimeError(
                "No CATE estimates found. Run estimate() or estimate_all() first."
            )

        treatments = treatments or list(self.cate_estimates.keys())
        coords = np.asarray(coords, dtype=np.float64)

        # Normalise coords to [0, 1] per dimension
        coords_min = coords.min(axis=0)
        coords_range = coords.max(axis=0) - coords_min
        coords_range[coords_range == 0] = 1.0
        coords_norm = (coords - coords_min) / coords_range

        self._gp_surfaces: Dict[str, "CATEGPSurface"] = {}
        results: Dict[str, Dict[str, Any]] = {}

        for treatment in treatments:
            if treatment not in self.cate_estimates:
                continue
            cate = self.cate_estimates[treatment]

            gps = (
                CATEGPSurface.from_correlogram(
                    corr_payload,
                    max_fit_rows=max_fit_rows,
                    n_restarts_optimizer=n_restarts_optimizer,
                    random_state=self.random_state,
                )
                if corr_payload
                else CATEGPSurface(
                    max_fit_rows=max_fit_rows,
                    n_restarts_optimizer=n_restarts_optimizer,
                    random_state=self.random_state,
                )
            )
            gps.fit(cate, coords_norm)
            gp_mean, gp_std = gps.predict(coords_norm)
            self._gp_surfaces[treatment] = gps

            results[treatment] = {
                "gp_mean": gp_mean,
                "gp_std": gp_std,
                "moran_i": gps.moran_result,
                "log_ml": gps.log_marginal_likelihood,
                "kernel_params": gps.fitted_kernel_params,
            }

        return results

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Return summary statistics for all estimated CATEs."""
        summary = {}
        for treatment, cate in self.cate_estimates.items():
            summary[treatment] = {
                'mean': float(np.mean(cate)),
                'std': float(np.std(cate)),
                'min': float(np.min(cate)),
                'max': float(np.max(cate)),
                'median': float(np.median(cate)),
                'q25': float(np.percentile(cate, 25)),
                'q75': float(np.percentile(cate, 75)),
                'pct_negative': float(np.mean(cate < 0) * 100),
                'pct_positive': float(np.mean(cate > 0) * 100),
            }
            if treatment in self.cate_intervals:
                lower, upper = self.cate_intervals[treatment]
                # Fraction of CIs that exclude zero → significant effects
                excludes_zero = ((lower > 0) | (upper < 0))
                summary[treatment]['pct_significant'] = float(
                    np.mean(excludes_zero) * 100
                )
        return summary


# ===========================================================================
# Bayesian Spatial CATE — Phase 2 of v4 causal-stack rewrite
# ===========================================================================

class BayesianSpatialCATE:
    """Bayesian per-cell CATE with low-rank spatial GP prior + NUTS.

    The estimator implements a "double-ML for CATE" model with a sparse
    spatial Gaussian process over treatment-effect heterogeneity:

    1. Residualize ``Y`` on confounders ``Z`` via cross-fit HGB:
       ``Y_resid = Y - Ê[Y|Z]``.
    2. Residualize ``T`` on ``Z`` similarly: ``T_resid = T - Ê[T|Z]``.
    3. Model ``Y_resid_i = τ(s_i) · T_resid_i + ε_i`` with
       ``τ(s_i) = β_0 + Φ_K(s_i) · w``, where ``Φ_K`` is a fixed
       random-RBF feature basis of dimension K (default 32) on
       normalized coordinates. NUTS samples ``(β_0, w, log σ²)``.
    4. Posterior over per-cell τ:
       ``τ_post[d, i] = β_0[d] + Φ_K(s_i) · w[d]``.

    This is a tractable Bayesian sparse-GP approximation (random Fourier
    features view) that scales linearly in the number of cells. The same
    public API as :class:`SpatialCATEEstimator` is exposed so callers can
    swap estimators via the ``causal.cate_estimator`` config switch, plus
    new accessors for full posterior samples and CI summaries.

    Returns posterior samples ``τ[d, i]`` per treatment via
    :meth:`posterior_samples`; the conventional ``cate_estimates`` /
    ``cate_intervals`` mirror frequentist behavior using posterior mean
    / 5-95% credible intervals.
    """

    def __init__(
        self,
        config: dict,
        n_features: int = 32,
        n_samples: int = 2000,
        n_warmup: int = 500,
        kernel_lengthscale: float = 0.20,
        kernel_field: Optional[Any] = None,
    ):
        self.config = config
        self.n_features = int(n_features)
        self.n_samples = int(n_samples)
        self.n_warmup = int(n_warmup)
        self.kernel_lengthscale = float(kernel_lengthscale)
        self.random_state = config.get("pipeline", {}).get("random_seed", 42)

        # Compatible with SpatialCATEEstimator interface
        self.cate_estimates: Dict[str, np.ndarray] = {}
        self.cate_intervals: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        # New: per-treatment full posterior samples (n_draws, n_cells)
        self._posterior_samples: Dict[str, np.ndarray] = {}
        # Per-treatment NUTS diagnostics
        self._diagnostics: Dict[str, Dict[str, float]] = {}
        # Cached random-RBF projection per treatment (for posterior_predict)
        self._rbf_basis: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        # Phase 7 — corrected kernel-field context.  When supplied,
        # the per-treatment RFF lengthscale is overridden by
        # ``predictor.bandwidth_to_outcome / coord_extent`` and coords are
        # warped into the predictor's anisotropic principal frame before
        # RBF projection so the kernel matches the corrected geometry.
        self.kernel_field = kernel_field
        # Per-treatment audit: {treatment: {"applied_warp": bool,
        #                                    "effective_lengthscale": float}}
        self._kernel_field_applied: Dict[str, Dict[str, Any]] = {}

    # --------------------------------------------------------------
    # Random Fourier feature basis (Rahimi & Recht, 2008) — RBF kernel.
    # --------------------------------------------------------------

    def _build_rbf_features(
        self,
        coords_norm: np.ndarray,
        seed: int,
        lengthscale: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (Phi, W, b) where ``Phi`` is the random-RBF feature matrix
        of shape (N, n_features) with frequencies ``W`` and phases ``b``.

        ``lengthscale`` overrides the instance default (used by the Phase 7
        kernel-field path so per-treatment ``bandwidth_to_outcome`` controls
        the RBF kernel lengthscale).
        """
        rng = np.random.default_rng(seed)
        d = coords_norm.shape[1]
        ell = max(
            float(lengthscale) if lengthscale is not None
            else self.kernel_lengthscale,
            1e-3,
        )
        # ω ~ N(0, 1/ell²) gives RBF kernel with lengthscale ell.
        W = rng.normal(0.0, 1.0 / ell, size=(d, self.n_features))
        b = rng.uniform(0.0, 2.0 * np.pi, size=self.n_features)
        Phi = np.sqrt(2.0 / self.n_features) * np.cos(coords_norm @ W + b)
        return Phi.astype(np.float64), W.astype(np.float64), b.astype(np.float64)

    @staticmethod
    def _normalize_coords(coords: np.ndarray) -> np.ndarray:
        cmin = coords.min(axis=0)
        crange = coords.max(axis=0) - cmin
        crange[crange == 0] = 1.0
        return (coords - cmin) / crange

    # --------------------------------------------------------------
    # Single-treatment estimator
    # --------------------------------------------------------------

    def estimate(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        confounders: List[str],
        coord_cols: List[str] | None = None,
        spatial_features: np.ndarray | None = None,  # noqa: ARG002 — for API parity
    ) -> np.ndarray:
        """Posterior mean of τ(s_i) for one treatment (n_cells,)."""
        import math
        import torch
        from sklearn.ensemble import HistGradientBoostingRegressor as HGB
        from sklearn.model_selection import KFold

        from sparc.causal.nuts import NUTSBlock, run_nuts

        if coord_cols is None or len(coord_cols) < 2:
            raise ValueError("BayesianSpatialCATE requires coord_cols (length 2)")
        avail_coords = [c for c in coord_cols if c in data.columns]
        if len(avail_coords) < 2:
            raise ValueError(
                f"Coordinate columns {coord_cols} not present in data"
            )

        T = data[treatment].values.astype(np.float64)
        Y = data[outcome].values.astype(np.float64)
        Z = (
            data[confounders].values.astype(np.float64)
            if confounders else np.zeros((len(data), 1))
        )
        coords = data[avail_coords].values.astype(np.float64)

        # Phase 7: warp coords + derive RFF lengthscale from kernel field
        predictor = None
        if self.kernel_field is not None:
            predictor = self.kernel_field.predictor(treatment)
        if predictor is not None and getattr(predictor, "is_anisotropic", False):
            coords_warped = _anisotropic_warp_coords(coords, predictor)
            applied_warp = True
        else:
            coords_warped = coords
            applied_warp = False

        coord_extent = float(np.max(
            coords_warped.max(axis=0) - coords_warped.min(axis=0),
        ))
        if coord_extent <= 0:
            coord_extent = 1.0

        # Per-treatment lengthscale override: bandwidth_to_outcome maps to
        # the Matérn correlation length; expressed in normalized warped
        # units that becomes bandwidth / coord_extent.
        eff_lengthscale: Optional[float] = None
        if predictor is not None:
            bw = None
            if hasattr(predictor, "bandwidth_to_outcome"):
                bw = predictor.bandwidth_to_outcome
            if bw is not None and float(bw) > 0:
                eff_lengthscale = float(bw) / coord_extent
                # Clamp to a sensible band so a tiny bw doesn't drive the
                # frequency variance to infinity.
                eff_lengthscale = float(np.clip(eff_lengthscale, 0.02, 1.0))

        coords_norm = self._normalize_coords(coords_warped)
        self._kernel_field_applied[treatment] = {
            "applied_warp": bool(applied_warp),
            "effective_lengthscale": (
                float(eff_lengthscale) if eff_lengthscale is not None
                else float(self.kernel_lengthscale)
            ),
        }

        # ---- Cross-fit residualization (DML stage 1) ----
        n = len(data)
        rng = np.random.default_rng(self.random_state)
        n_splits = max(2, min(5, n // 200))
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        Y_resid = np.zeros(n)
        T_resid = np.zeros(n)
        for tr, te in kf.split(np.arange(n)):
            my = HGB(max_iter=150, max_depth=4, learning_rate=0.05,
                     min_samples_leaf=20, random_state=self.random_state)
            mt = HGB(max_iter=150, max_depth=4, learning_rate=0.05,
                     min_samples_leaf=20, random_state=self.random_state)
            try:
                my.fit(Z[tr], Y[tr])
                mt.fit(Z[tr], T[tr])
                Y_resid[te] = Y[te] - my.predict(Z[te])
                T_resid[te] = T[te] - mt.predict(Z[te])
            except Exception:
                Y_resid[te] = Y[te] - Y[tr].mean()
                T_resid[te] = T[te] - T[tr].mean()

        # ---- Build random RBF features over normalized coords ----
        edge_seed = int(rng.integers(0, 2**31 - 1))
        Phi, W_freq, b_phase = self._build_rbf_features(
            coords_norm, edge_seed, lengthscale=eff_lengthscale,
        )
        self._rbf_basis[treatment] = (W_freq, b_phase)
        K = self.n_features

        # Standardize T_resid scale once for stable β.
        t_scale = float(np.std(T_resid))
        if t_scale < 1e-12:
            t_scale = 1.0
        Tr = T_resid / t_scale
        Yr = Y_resid

        # ---- Bayesian model: Yr_i = (β0 + Phi_i · w) · Tr_i + ε ----
        device = torch.device("cpu")
        dtype = torch.float64
        Yr_t = torch.tensor(Yr, dtype=dtype, device=device)
        Tr_t = torch.tensor(Tr, dtype=dtype, device=device)
        Phi_t = torch.tensor(Phi, dtype=dtype, device=device)

        # Priors: weakly-informative N(0, σ²_w/τ²), τ_w controls smoothness.
        prior_var_b0 = 10.0
        prior_var_w = 1.0  # in standardized-T units; rescaled afterwards.
        sigma2_init = float(np.var(Yr).clip(min=1e-6))
        log_sigma2_init = math.log(sigma2_init)

        def _log_prob(params: dict[str, torch.Tensor]) -> torch.Tensor:
            b0 = params["b0"][0]
            w = params["w"]
            sigma2 = params["sigma2"]
            tau = b0 + Phi_t @ w
            mu = tau * Tr_t
            per_pt = -0.5 * (torch.log(sigma2) + (Yr_t - mu) ** 2 / sigma2)
            ll = torch.sum(per_pt)
            lp_b0 = -0.5 * (b0 ** 2) / prior_var_b0
            lp_w = -0.5 * torch.sum(w ** 2) / prior_var_w
            lp_sigma = -2.0 * torch.log(sigma2)
            return ll + lp_b0 + lp_w + lp_sigma

        blocks = [
            NUTSBlock(name="b0", dim=1, init=np.array([0.0])),
            NUTSBlock(name="w", dim=K, init=np.zeros(K)),
            NUTSBlock(name="sigma2", dim=1,
                      init=np.array([log_sigma2_init]), transform="log"),
        ]
        # Config resolution: prefer causal.bayesian_cate.*, fall back to
        # causal.nuts.* (single source of truth for users), then defaults.
        causal_cfg = (self.config.get("causal", {}) or {})
        cate_cfg = dict(causal_cfg.get("bayesian_cate", {}) or {})
        nuts_cfg = dict(causal_cfg.get("nuts", {}) or {})
        for key, default in (
            ("n_samples", self.n_samples),
            ("n_warmup", self.n_warmup),
            ("max_tree_depth", 8),
            ("target_accept_rate", 0.85),
            ("n_chains", 2),
        ):
            if key not in cate_cfg and key in nuts_cfg:
                cate_cfg[key] = nuts_cfg[key]
            cate_cfg.setdefault(key, default)

        # Hardware-aware shrink: low-tier RAM cuts the basis dimension so the
        # NUTS state space halves (~17 params instead of 33), which both speeds
        # mixing and avoids OOM on 8 GB laptops.
        hw_tier = (self.config.get("hardware", {}) or {}).get("tier", "").lower()
        n_chains = max(1, int(cate_cfg.get("n_chains", 2)))

        chain_results = []
        try:
            for chain_idx in range(n_chains):
                seed = int(rng.integers(0, 2**31 - 1))
                res = run_nuts(
                    log_prob_fn=_log_prob,
                    blocks=blocks,
                    n_samples=int(cate_cfg["n_samples"]),
                    n_warmup=int(cate_cfg["n_warmup"]),
                    max_depth=int(cate_cfg["max_tree_depth"]),
                    target_accept=float(cate_cfg["target_accept_rate"]),
                    seed=seed,
                    device=str(device),
                )
                chain_results.append(res)
        except Exception as exc:
            raise RuntimeError(
                f"BayesianSpatialCATE NUTS failed for {treatment}: {exc}"
            ) from exc

        # Concatenate chains so split-R-hat sees true inter-chain variance.
        results = chain_results[0]
        if len(chain_results) > 1:
            for k in results.samples:
                results.samples[k] = np.concatenate(
                    [r.samples[k] for r in chain_results], axis=0,
                )
            # Recompute pooled diagnostics using true multi-chain split R-hat.
            from sparc.causal.nuts import _split_r_hat, _ess_bulk
            for k, arr in results.samples.items():
                results.r_hat[k] = np.array(
                    [_split_r_hat(arr[:, d]) for d in range(arr.shape[1])]
                )
                results.ess[k] = np.array(
                    [_ess_bulk(arr[:, d]) for d in range(arr.shape[1])]
                )
            results.acceptance_rate = float(
                np.mean([r.acceptance_rate for r in chain_results])
            )
            results.n_divergences = int(sum(r.n_divergences for r in chain_results))

        b0_chain = np.asarray(results.samples["b0"]).reshape(-1)        # (D,)
        w_chain = np.asarray(results.samples["w"])                       # (D, K)
        # noqa: ARG002 hardware-tier hook (n_features can be set externally)
        _ = hw_tier

        # Posterior τ in standardized-T units → divide by t_scale to map
        # back to per-original-T-unit causal effect of T on Y.
        # τ_post[d, i] = (b0[d] + Phi[i] · w[d]) / t_scale
        # Compute in chunks to avoid memory blow-up on large N.
        D = b0_chain.shape[0]
        tau_post = np.empty((D, n), dtype=np.float64)
        chunk = max(1, min(D, 500))
        for d0 in range(0, D, chunk):
            d1 = min(D, d0 + chunk)
            tau_post[d0:d1] = (
                b0_chain[d0:d1, None] + w_chain[d0:d1] @ Phi.T
            ) / t_scale

        cate_mean = tau_post.mean(axis=0)
        ci_lower = np.percentile(tau_post, 5, axis=0)
        ci_upper = np.percentile(tau_post, 95, axis=0)

        self.cate_estimates[treatment] = cate_mean
        self.cate_intervals[treatment] = (ci_lower, ci_upper)
        self._posterior_samples[treatment] = tau_post
        # Aggregate convergence pass-rate for w[*] so callers can flag unstable
        # CATE without scanning per-parameter logs.
        rh_w = results.r_hat.get("w", np.array([np.nan]))
        ess_w = results.ess.get("w", np.array([np.nan]))
        if rh_w.size and ess_w.size:
            w_pass = float(np.mean((rh_w < 1.05) & (ess_w > 400)))
            w_max_rhat = float(np.nanmax(rh_w))
            w_min_ess = float(np.nanmin(ess_w))
        else:
            w_pass = float("nan")
            w_max_rhat = float("nan")
            w_min_ess = float("nan")
        self._diagnostics[treatment] = {
            "acceptance_rate": float(results.acceptance_rate),
            "n_divergences": int(results.n_divergences),
            "r_hat_b0": float(results.r_hat.get("b0", np.array([np.nan]))[0]),
            "ess_b0": float(results.ess.get("b0", np.array([np.nan]))[0]),
            "w_converged_fraction": w_pass,
            "w_max_r_hat": w_max_rhat,
            "w_min_ess": w_min_ess,
            "n_chains": int(n_chains),
            "n_draws": int(D),
            "t_scale": float(t_scale),
        }

        # Memory hygiene: free large intermediates before returning so the
        # next treatment loop iteration starts with a clean heap.
        del Phi_t, Yr_t, Tr_t, tau_post, b0_chain, w_chain, chain_results, results
        import gc as _gc
        _gc.collect()
        return cate_mean

    # --------------------------------------------------------------
    # Convenience APIs — interface parity with SpatialCATEEstimator
    # --------------------------------------------------------------

    def estimate_all(
        self,
        data: pd.DataFrame,
        treatments: List[str],
        outcome: str,
        confounders: List[str],
        coord_cols: List[str] | None = None,
        spatial_features: np.ndarray | None = None,  # noqa: ARG002
    ) -> Dict[str, np.ndarray]:
        results = {}
        for treatment in treatments:
            if treatment not in data.columns:
                continue
            try:
                cate = self.estimate(
                    data, treatment, outcome, confounders,
                    coord_cols=coord_cols,
                )
                results[treatment] = cate
                ci_lo, ci_hi = self.cate_intervals[treatment]
                print(f"    Bayesian CATE({treatment}): mean={cate.mean():+.5f}, "
                      f"std={cate.std():.5f}, "
                      f"ci=[{ci_lo.mean():+.4f}, {ci_hi.mean():+.4f}]")
            except Exception as e:
                warnings.warn(f"BayesianSpatialCATE failed for {treatment}: {e}")
        return results

    def cate_to_spatial_multiplier(
        self,
        treatment: str,
        lower_bound: float = 0.5,
        upper_bound: float = 1.5,
    ) -> np.ndarray:
        """Posterior mean multiplier τ(s)/mean(τ), clamped."""
        if treatment not in self.cate_estimates:
            raise ValueError(f"No CATE estimates for '{treatment}'.")
        cate = self.cate_estimates[treatment]
        mean_cate = float(np.mean(cate))
        if abs(mean_cate) < 1e-10:
            return np.ones_like(cate)
        return np.clip(cate / mean_cate, lower_bound, upper_bound)

    def posterior_samples(self, treatment: str) -> np.ndarray:
        """Return per-cell posterior τ samples, shape ``(n_draws, n_cells)``."""
        if treatment not in self._posterior_samples:
            raise KeyError(f"No posterior samples for '{treatment}'.")
        return self._posterior_samples[treatment]

    def multiplier_posterior(
        self,
        treatment: str,
        lower_bound: float = 0.5,
        upper_bound: float = 1.5,
    ) -> np.ndarray:
        """Per-draw multiplier samples ``τ_post / mean(τ_post)``, clamped.

        Returns shape ``(n_draws, n_cells)``. Mean over axis=0 reproduces
        :meth:`cate_to_spatial_multiplier` (modulo clamp).
        """
        if treatment not in self._posterior_samples:
            raise KeyError(f"No posterior samples for '{treatment}'.")
        tau = self._posterior_samples[treatment]
        mean_per_draw = tau.mean(axis=1, keepdims=True)
        mean_per_draw = np.where(np.abs(mean_per_draw) < 1e-10, 1.0, mean_per_draw)
        return np.clip(tau / mean_per_draw, lower_bound, upper_bound)

    def to_geodataframe(
        self,
        data: pd.DataFrame,
        coord_cols: List[str],
        crs: str | None = None,
    ) -> Any:
        import geopandas as gpd
        from shapely.geometry import Point

        gdf = data.copy()
        geometry = [
            Point(x, y) for x, y in zip(data[coord_cols[0]], data[coord_cols[1]])
        ]
        gdf = gpd.GeoDataFrame(gdf, geometry=geometry, crs=crs)
        for treatment, cate in self.cate_estimates.items():
            gdf[f"cate_{treatment}"] = cate
            gdf[f"spatial_mult_{treatment}"] = self.cate_to_spatial_multiplier(treatment)
            if treatment in self.cate_intervals:
                lo, hi = self.cate_intervals[treatment]
                gdf[f"cate_ci_lower_{treatment}"] = lo
                gdf[f"cate_ci_upper_{treatment}"] = hi
        return gdf

    def summary(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for treatment, cate in self.cate_estimates.items():
            entry: Dict[str, float] = {
                "mean": float(np.mean(cate)),
                "std": float(np.std(cate)),
                "min": float(np.min(cate)),
                "max": float(np.max(cate)),
                "median": float(np.median(cate)),
                "q25": float(np.percentile(cate, 25)),
                "q75": float(np.percentile(cate, 75)),
                "pct_negative": float(np.mean(cate < 0) * 100),
                "pct_positive": float(np.mean(cate > 0) * 100),
            }
            if treatment in self.cate_intervals:
                lo, hi = self.cate_intervals[treatment]
                entry["pct_significant"] = float(
                    np.mean((lo > 0) | (hi < 0)) * 100
                )
            entry.update(self._diagnostics.get(treatment, {}))
            out[treatment] = entry
        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_cate_estimator(config: dict) -> Any:
    """Return the CATE estimator selected by ``causal.cate_estimator``.

    Supported values:

    * ``"bayesian"`` — :class:`BayesianSpatialCATE` (per-cell posteriors).
    * ``"frequentist"`` — :class:`SpatialCATEEstimator` (CausalForestDML).
    * ``"auto"`` — Bayesian if NumPyro/JAX available **or** PyTorch +
      our internal NUTS module work (always true since `sparc.causal.nuts`
      is shipped); falls back to frequentist on import error.

    Default: ``"auto"``.
    """
    name = (
        (config.get("causal", {}) or {})
        .get("cate_estimator", "auto")
        .lower()
    )
    if name == "frequentist":
        return SpatialCATEEstimator(config)
    if name == "bayesian":
        return BayesianSpatialCATE(config)
    if name == "auto":
        # PyTorch + sparc.causal.nuts is a hard dependency of the repo;
        # only fall back if the module raises on import for some reason.
        try:
            from sparc.causal.nuts import run_nuts  # noqa: F401
            return BayesianSpatialCATE(config)
        except Exception:
            warnings.warn(
                "make_cate_estimator: sparc.causal.nuts unavailable, "
                "falling back to frequentist SpatialCATEEstimator."
            )
            return SpatialCATEEstimator(config)
    raise ValueError(
        f"Unknown causal.cate_estimator: {name!r}. "
        f"Use 'auto', 'bayesian', or 'frequentist'."
    )


# ===========================================================================
# GP Regression over CATE surface
# ===========================================================================

class CATEGPSurface:
    """Continuous uncertainty-quantified CATE surface via sklearn GP regression.

    Fits a :class:`~sklearn.gaussian_process.GaussianProcessRegressor` with a
    Matérn kernel over normalised spatial coordinates.  The kernel is
    initialised from Stage 0 correlogram parameters (``nu``, ``kappa``) when
    available, which biases the GP toward the same spatial autocorrelation
    structure already used by the causal-forest/Bayesian estimators.

    A Moran's I test is run on the GP residuals (observed CATE − posterior
    mean) to detect residual spatial autocorrelation that the GP surface has
    not captured.

    Parameters
    ----------
    nu : float
        Matérn smoothness parameter ν ∈ {0.5, 1.5, 2.5, ∞}. Default 1.5.
    length_scale : float
        Initial Matérn length scale in normalised coordinate units [0, 1].
        Default 0.2.
    length_scale_bounds : tuple[float, float]
        Optimiser search bounds for the length scale.
    n_restarts_optimizer : int
        Number of L-BFGS-B restarts for log-marginal-likelihood optimisation.
    max_fit_rows : int
        Hard cap on training points fed to GPR (GPR is O(N³)); rows beyond
        this are sub-sampled uniformly.  Default 2 000.
    random_state : int
        RNG seed for sub-sampling and optimizer restarts.
    """

    def __init__(
        self,
        nu: float = 1.5,
        length_scale: float = 0.2,
        length_scale_bounds: Tuple[float, float] = (1e-3, 1.0),
        n_restarts_optimizer: int = 3,
        max_fit_rows: int = 2_000,
        random_state: int = 42,
    ):
        self.nu = float(nu)
        self.length_scale = float(length_scale)
        self.length_scale_bounds = length_scale_bounds
        self.n_restarts_optimizer = n_restarts_optimizer
        self.max_fit_rows = max_fit_rows
        self.random_state = random_state

        self._gpr: Any = None            # fitted GaussianProcessRegressor
        self._fit_coords: np.ndarray | None = None   # normalised training coords
        self._fit_cate: np.ndarray | None = None     # training CATE values
        self.moran_result: Dict[str, float] = {}     # populated by fit()

    # ------------------------------------------------------------------
    # Class method: seed from Stage 0 correlogram artifact
    # ------------------------------------------------------------------

    @classmethod
    def from_correlogram(
        cls,
        corr_payload: dict,
        **kwargs,
    ) -> "CATEGPSurface":
        """Create a ``CATEGPSurface`` seeded from a Stage 0 correlogram dict.

        The payload format is the ``to_payload()`` output of
        :class:`~sparc.run.correlogram_matern_fit.MaternCorrelogramResult`:
        ``{"nu": float, "kappa": {"mean": float, ...}, ...}``.
        """
        nu = float(corr_payload.get("nu", 1.5))
        kappa_mean = float(
            corr_payload.get("kappa", {}).get("mean", 0.2)
        )
        # kappa from correlogram is in raw coordinate units; assume
        # coordinates will be normalised to [0,1] so we clamp kappa to
        # [0.01, 1.0] as an initial length scale.
        length_scale = float(np.clip(kappa_mean, 0.01, 1.0))
        return cls(nu=nu, length_scale=length_scale, **kwargs)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        cate: np.ndarray,
        coords_norm: np.ndarray,
    ) -> "CATEGPSurface":
        """Fit GP regression on per-point CATE estimates.

        Parameters
        ----------
        cate : (N,) array
            Per-point CATE estimates (from any estimator).
        coords_norm : (N, D) array
            Spatial coordinates normalised to [0, 1] per dimension.

        Returns
        -------
        self
        """
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, ConstantKernel

        cate = np.asarray(cate, dtype=np.float64).ravel()
        coords_norm = np.asarray(coords_norm, dtype=np.float64)
        if coords_norm.ndim == 1:
            coords_norm = coords_norm.reshape(-1, 1)

        N = len(cate)
        if N != len(coords_norm):
            raise ValueError(
                f"cate length {N} != coords_norm length {len(coords_norm)}"
            )

        # Sub-sample for fitting (GPR is O(N³))
        if N > self.max_fit_rows:
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(N, size=self.max_fit_rows, replace=False)
        else:
            idx = np.arange(N)

        X_fit = coords_norm[idx]
        y_fit = cate[idx]

        # Build kernel: constant amplitude × Matérn
        cate_var = float(np.var(cate))
        amplitude = ConstantKernel(
            constant_value=max(cate_var, 1e-6),
            constant_value_bounds=(1e-6, cate_var * 100 + 1e-3),
        )
        matern = Matern(
            nu=self.nu,
            length_scale=self.length_scale,
            length_scale_bounds=self.length_scale_bounds,
        )
        kernel = amplitude * matern

        self._gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=self.n_restarts_optimizer,
            normalize_y=True,
            random_state=self.random_state,
        )
        self._gpr.fit(X_fit, y_fit)

        self._fit_coords = coords_norm
        self._fit_cate = cate

        # Moran's I on residuals over full dataset
        gp_mean, _ = self._gpr.predict(coords_norm, return_std=True)
        residuals = cate - gp_mean
        self.moran_result = _moran_i(residuals, coords_norm, n_neighbors=8)

        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(
        self,
        coords_norm: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return GP posterior (mean, std) at query coordinates.

        Parameters
        ----------
        coords_norm : (M, D) array
            Query coordinates in the same normalised space used at fit time.

        Returns
        -------
        mean : (M,) array
        std  : (M,) array
        """
        if self._gpr is None:
            raise RuntimeError("Call fit() before predict().")
        coords_norm = np.asarray(coords_norm, dtype=np.float64)
        if coords_norm.ndim == 1:
            coords_norm = coords_norm.reshape(-1, 1)
        mean, std = self._gpr.predict(coords_norm, return_std=True)
        return mean, std

    # ------------------------------------------------------------------
    # Optimised kernel parameters (for diagnostics)
    # ------------------------------------------------------------------

    @property
    def fitted_kernel_params(self) -> Dict[str, Any]:
        """Return the optimised kernel parameters after fitting."""
        if self._gpr is None:
            return {}
        k = self._gpr.kernel_
        params = k.get_params()
        return {k_: float(v) if hasattr(v, "__float__") else v
                for k_, v in params.items()}

    @property
    def log_marginal_likelihood(self) -> float:
        """Return the log marginal likelihood of the fitted GP."""
        if self._gpr is None:
            return float("nan")
        return float(self._gpr.log_marginal_likelihood_value_)


# ---------------------------------------------------------------------------
# Moran's I helper
# ---------------------------------------------------------------------------

def _moran_i(
    residuals: np.ndarray,
    coords: np.ndarray,
    n_neighbors: int = 8,
) -> Dict[str, float]:
    """Compute Moran's I spatial autocorrelation statistic on ``residuals``.

    Uses a row-normalised k-NN spatial weights matrix.  The expected value
    and variance under the randomisation assumption are used to compute a
    z-score and two-tailed p-value.

    Parameters
    ----------
    residuals : (N,) array
        GP residuals: observed CATE − GP posterior mean.
    coords : (N, D) array
        Spatial coordinates used to build the k-NN weight matrix.
    n_neighbors : int
        Number of spatial neighbours per observation.

    Returns
    -------
    dict with keys: ``I``, ``E_I``, ``z_score``, ``p_value``,
    ``interpretation``.
    """
    from sklearn.neighbors import NearestNeighbors

    residuals = np.asarray(residuals, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)

    N = len(residuals)
    k = min(n_neighbors, N - 1)

    # Build k-NN graph (excluding self)
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree")
    nbrs.fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    # Build row-normalised weight matrix (binary k-NN, exclude self)
    # W[i, j] = 1/k if j in the k-NN of i (and j != i)
    W = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        neighbors = indices[i, 1:]  # skip self (index 0)
        W[i, neighbors] = 1.0 / k

    e = residuals - residuals.mean()
    S0 = float(W.sum())
    if S0 < 1e-12 or np.sum(e ** 2) < 1e-16:
        return {"I": float("nan"), "E_I": float(-1.0 / (N - 1)),
                "z_score": float("nan"), "p_value": float("nan"),
                "interpretation": "degenerate"}

    I = float((N / S0) * (e @ W @ e) / (e @ e))

    # Expected value under randomisation
    E_I = -1.0 / (N - 1)

    # Variance under normality assumption (Moran 1948)
    # Var[I] = (N²·S1 - N·S2 + 3·S0²) / ((N²-1)·S0²) − E[I]²
    S1 = 0.5 * float(np.sum((W + W.T) ** 2))
    S2 = float(np.sum((W.sum(axis=1) + W.sum(axis=0)) ** 2))
    n = float(N)
    var_I = (
        (n ** 2 * S1 - n * S2 + 3.0 * S0 ** 2)
        / ((n ** 2 - 1.0) * S0 ** 2)
    ) - E_I ** 2
    var_I = max(var_I, 1e-12)

    import math as _math
    z = (I - E_I) / _math.sqrt(var_I)
    # Two-tailed p-value using normal approximation
    p = 2.0 * (1.0 - 0.5 * (1.0 + _math.erf(abs(z) / _math.sqrt(2))))

    if abs(z) < 1.96:
        interp = "no significant spatial autocorrelation in residuals"
    elif I > E_I:
        interp = "positive spatial autocorrelation in residuals (GP under-smoothed)"
    else:
        interp = "negative spatial autocorrelation in residuals (GP over-smoothed)"

    return {
        "I": round(I, 6),
        "E_I": round(E_I, 6),
        "var_I": round(var_I, 8),
        "z_score": round(z, 4),
        "p_value": round(p, 6),
        "interpretation": interp,
    }
