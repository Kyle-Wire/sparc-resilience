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
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


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
                coords = data[available].values
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
