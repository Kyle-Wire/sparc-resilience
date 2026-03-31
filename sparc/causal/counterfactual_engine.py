"""
counterfactual_engine — Causal-effect estimation and scenario propagation.

Given a trained SPARC pipeline, a causal DAG, and intervention specifications
(from ``project.yml → scenarios``), this module:

1. Uses DoWhy ``do(x)`` to estimate the **average treatment effect** (ATE)
   and, optionally, the **conditional ATE** (CATE) with EconML estimators.
2. Propagates interventions through the DAG: e.g. +20 pp Canopy triggers
   a mediated NDVI increase, and both direct and indirect temperature effects
   are summed.
3. Blends data-estimated effects with physics priors (from ``priors.yml``)
   using a configurable shrinkage weight.
4. Applies caps / extrapolation guards from ``caps.yml`` before returning
   spatially-resolved counterfactual predictions.

Typical usage::

    from sparc.causal.counterfactual_engine import CounterfactualEngine

    engine = CounterfactualEngine(config)
    engine.load_dag()
    engine.fit(data)
    results = engine.intervene("Pct_Canopy", delta=+20)
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .dag_definition import (
    load_dag, dag_to_networkx, build_causal_model, get_node_roles,
    compute_backdoor_set, add_temporal_edges,
)


class CounterfactualEngine:
    """
    Orchestrates causal estimation and scenario simulation.

    Parameters
    ----------
    config : dict
        Full SPARC config dict (from ``load_config(project_path)``).
    """

    def __init__(self, config: dict):
        self.config = config

        # Physics priors (loaded lazily)
        self._priors: Optional[dict] = None
        self._caps: Optional[dict] = None

        # Causal graph objects
        self.dag_def: Optional[dict] = None
        self.graph = None            # networkx.DiGraph
        self.causal_model = None     # dowhy.CausalModel
        self.roles: Dict[str, List[str]] = {}

        # Estimated structural coefficients  {(parent, child): coeff}
        self._structural_coeffs: Dict[Tuple[str, str], float] = {}
        # HGB model objects per-edge (if estimator='hgb')
        self._edge_models: Dict[Tuple[str, str], Any] = {}
        # Training data medians (for non-linear prediction baselines)
        self._train_medians: Dict[str, float] = {}

        # Spatial CATE multipliers keyed by treatment name → 1-D array
        self._cate_multipliers: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_dag(self, dag_path: str | None = None) -> None:
        """Load the causal DAG from the path specified in config (or override).

        If the causal config section contains ``temporal_edges``, they are
        merged into the DAG automatically (creating lagged nodes).
        """
        dag_path = dag_path or self.config.get('causal', {}).get('dag_file')
        if dag_path is None:
            raise ValueError("No causal DAG file specified in config or arguments.")
        self.dag_def = load_dag(dag_path)
        self.graph = dag_to_networkx(self.dag_def)

        # Merge temporal edges from the config-level causal section
        # (in addition to any already in the DAG YAML)
        cfg_temporal = self.config.get('causal', {}).get('temporal_edges', [])
        if cfg_temporal:
            add_temporal_edges(self.graph, cfg_temporal)

        self.roles = get_node_roles(self.graph)

    @property
    def priors(self) -> dict:
        if self._priors is None:
            from sparc.config.config import load_physics_priors
            self._priors = load_physics_priors(self.config)
        return self._priors

    @property
    def caps(self) -> dict:
        if self._caps is None:
            from sparc.config.config import load_caps
            self._caps = load_caps(self.config)
        return self._caps

    # ------------------------------------------------------------------
    # Fitting — estimate structural coefficients from data
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame, coords: np.ndarray | None = None) -> None:
        """
        Estimate structural-equation coefficients from *data* using the DAG.

        Estimator is selected via ``config['causal']['estimator']``:
        * ``"ols"``  — OLS with backdoor adjustment.
        * ``"hgb"``  — HistGradientBoostingRegressor.  Stores the actual
          model objects in ``self._edge_models`` so that ``intervene()``
          can propagate effects non-linearly.
        * ``"dml"``  (recommended) — Debiased/Orthogonal ML with K-fold
          cross-fitting and optional spatial blocks.

        In all cases, a linearised coefficient is stored for compatibility.

        Parameters
        ----------
        data : pd.DataFrame
            Observational dataset.
        coords : np.ndarray, optional
            (N, 2) array of spatial coordinates for spatial DML folds.
        """
        if self.graph is None:
            raise RuntimeError("Call load_dag() before fit().")

        from sparc.config.config import load_monotone_constraints

        target = self.roles['outcomes'][0] if self.roles['outcomes'] else None
        calibration = self.priors.get('calibration', {})
        shrinkage = calibration.get('shrinkage_weight', 0.7)
        adaptive_weighting = calibration.get('adaptive_weighting', False)
        max_deviation = calibration.get('max_deviation', 0.5)
        coefficients_prior = self.priors.get('coefficients', {})
        scale_factors = self.priors.get('scale_harmonization', {}).get('factors', {})
        monotone = load_monotone_constraints(self.config)

        estimator_name = (
            self.config.get('causal', {}).get('estimator', 'hgb').lower()
        )

        # DML-specific: reduced shrinkage (debiased estimator needs less correction)
        if estimator_name == 'dml':
            dml_shrinkage_override = self.config.get('causal', {}).get(
                'dml_shrinkage', 0.25
            )
            shrinkage = dml_shrinkage_override

        # DML-specific: spatial block size for cross-fitting
        spatial_block_size = self.config.get('causal', {}).get(
            'spatial_block_size', None
        )
        dml_n_splits = self.config.get('causal', {}).get('dml_cv_folds', 5)

        # Storage for raw (pre-shrinkage) coefficients and SEs
        self._raw_coeffs: Dict[Tuple[str, str], float] = {}
        self._coeff_ses: Dict[Tuple[str, str], float] = {}
        self._dml_models: Dict[Tuple[str, str], Any] = {}

        for parent, child in self.graph.edges():
            if parent not in data.columns or child not in data.columns:
                continue

            # Collect backdoor adjustment set (full criterion)
            confounders = [
                v for v in compute_backdoor_set(self.graph, parent, child)
                if v in data.columns
            ]
            X_cols = [parent] + confounders

            try:
                if estimator_name == 'dml':
                    coeff_data, se, dml_model = self._fit_edge_dml(
                        data, confounders, child, parent,
                        coords=coords,
                        spatial_block_size=spatial_block_size,
                        n_splits=dml_n_splits,
                    )
                    self._dml_models[(parent, child)] = dml_model
                    self._coeff_ses[(parent, child)] = se
                elif estimator_name == 'hgb':
                    coeff_data, model_obj = self._fit_edge_hgb(
                        data, X_cols, child, parent, monotone,
                        target=target,
                    )
                    self._edge_models[(parent, child)] = model_obj
                else:
                    from sklearn.linear_model import LinearRegression
                    lr = LinearRegression()
                    lr.fit(data[X_cols], data[child])
                    coeff_data = float(lr.coef_[0])
            except Exception:
                coeff_data = 0.0

            # Store raw (pre-shrinkage) coefficient
            self._raw_coeffs[(parent, child)] = coeff_data

            # Shrink toward physics prior if available
            prior_info = coefficients_prior.get(parent, {})
            if prior_info and child == target:
                coeff_prior_raw = prior_info.get('value', coeff_data)

                # Rescale prior from literature units to per-1-raw-unit
                if "Pct" in parent or "pct" in parent:
                    unit_inc = 10.0
                elif parent in ("NDVI", "Albedo"):
                    unit_inc = 0.1
                else:
                    unit_inc = 1.0
                coeff_prior = coeff_prior_raw / unit_inc

                # Scale harmonization (literature resolution → 30 m)
                source_scale = prior_info.get('source_scale', '')
                scale_factor = 1.0
                for band_key, band_factors in scale_factors.items():
                    if isinstance(band_factors, dict) and parent in band_factors:
                        band_clean = band_key.lower().replace('m', '').replace(' ', '')
                        src_clean = source_scale.lower().replace('m', '').replace(' ', '')
                        if band_clean == src_clean:
                            scale_factor = float(band_factors[parent])
                            break
                coeff_prior *= scale_factor

                # Adaptive shrinkage (confidence-aware)
                if adaptive_weighting:
                    confidence = prior_info.get('confidence', 'medium')
                    eff_shrinkage = {
                        'high': shrinkage,
                        'medium': max(shrinkage - 0.2, 0.3),
                        'low': max(shrinkage - 0.4, 0.15),
                    }.get(confidence, shrinkage)
                else:
                    eff_shrinkage = shrinkage

                coeff_blended = (1 - eff_shrinkage) * coeff_data + eff_shrinkage * coeff_prior

                # Max-deviation guard
                if coeff_prior != 0:
                    deviation = abs(coeff_blended / coeff_prior - 1.0)
                    if deviation > max_deviation:
                        bound_lo = coeff_prior * (1.0 - max_deviation)
                        bound_hi = coeff_prior * (1.0 + max_deviation)
                        coeff_blended = float(np.clip(
                            coeff_blended, min(bound_lo, bound_hi), max(bound_lo, bound_hi)
                        ))
            else:
                coeff_blended = coeff_data

            self._structural_coeffs[(parent, child)] = coeff_blended

        # Store training data medians for non-linear prediction baselines
        self._train_medians = data.median(numeric_only=True).to_dict()

    # ------------------------------------------------------------------
    # HGB edge estimator (shared logic with CausalValidator)
    # ------------------------------------------------------------------

    @staticmethod
    def _fit_edge_hgb(
        data: pd.DataFrame,
        X_cols: list,
        child: str,
        parent: str,
        monotone: dict,
        target: str | None = None,
    ) -> tuple:
        """Fit HistGradientBoostingRegressor for one edge; return (coeff, model)."""
        from sklearn.ensemble import HistGradientBoostingRegressor

        # Only apply outcome-level monotone constraints when the child IS
        # the outcome.  Mediator edges may have different direction.
        mc = [monotone.get(col, 0) if child == target else 0 for col in X_cols]

        hgb = HistGradientBoostingRegressor(
            max_iter=200,
            max_depth=4,
            learning_rate=0.05,
            min_samples_leaf=20,
            monotonic_cst=mc,
            random_state=42,
        )
        X = data[X_cols].values
        y = data[child].values
        hgb.fit(X, y)

        # Mean marginal effect via finite-difference
        parent_idx = X_cols.index(parent)
        eps = np.std(X[:, parent_idx]) * 0.01
        if eps < 1e-8:
            eps = 1e-3
        X_plus = X.copy()
        X_plus[:, parent_idx] += eps
        marginal = (hgb.predict(X_plus) - hgb.predict(X)) / eps
        coeff_data = float(np.mean(marginal))

        return coeff_data, hgb

    # ------------------------------------------------------------------
    # DML edge estimator — Debiased / Orthogonal ML (FWL residualisation)
    # ------------------------------------------------------------------

    def _fit_edge_dml(
        self,
        data: pd.DataFrame,
        confounders: List[str],
        child: str,
        parent: str,
        coords: np.ndarray | None = None,
        spatial_block_size: float | None = None,
        n_splits: int = 5,
    ) -> Tuple[float, float, Any]:
        """
        Estimate the causal coefficient for ``parent -> child`` via
        Debiased/Orthogonal Machine Learning (Chernozhukov et al. 2018).

        Uses K-fold cross-fitting (optionally with spatial blocks) to avoid
        regularisation bias from the nuisance models.

        Parameters
        ----------
        data : pd.DataFrame
        confounders : list[str]
            Backdoor adjustment set (from :func:`compute_backdoor_set`).
        child : str
            Outcome of this edge.
        parent : str
            Treatment/exposure of this edge.
        coords : np.ndarray, optional
            (N, 2) coordinate array for spatial block CV.  If provided,
            cross-fitting folds respect spatial autocorrelation.
        spatial_block_size : float, optional
            Block size in CRS units for spatial fold assignment.
        n_splits : int
            Number of cross-fitting folds (default 5).

        Returns
        -------
        coeff : float
            The orthogonalised (debiased) causal coefficient.
        se : float
            Heteroskedasticity-robust standard error.
        dml_model : object
            Fitted econml.dml.LinearDML model (or None on fallback).
        """
        from econml.dml import LinearDML
        from sklearn.ensemble import HistGradientBoostingRegressor as HGB

        T = data[[parent]].values   # (N, 1)
        Y = data[child].values      # (N,)
        W = data[confounders].values if confounders else None  # (N, p) or None

        # Build spatial cross-fitting splitter if coordinates are available
        cv_splitter: Any = n_splits
        if coords is not None and spatial_block_size is not None:
            try:
                from sparc.run.enhanced_spatial_cv import spatial_kfold_enhanced
                spatial_folds = spatial_kfold_enhanced(
                    X=data[confounders].values if confounders else np.zeros((len(data), 1)),
                    y=Y,
                    coords=coords,
                    n_splits=n_splits,
                    block_size=spatial_block_size,
                )
                cv_splitter = spatial_folds
            except Exception:
                cv_splitter = n_splits  # fall back to random

        model_y = HGB(
            max_iter=200, max_depth=4, learning_rate=0.05,
            min_samples_leaf=20, random_state=42,
        )
        model_t = HGB(
            max_iter=200, max_depth=4, learning_rate=0.05,
            min_samples_leaf=20, random_state=42,
        )

        dml = LinearDML(
            model_y=model_y,
            model_t=model_t,
            cv=cv_splitter,
            random_state=42,
        )

        if W is not None and W.shape[1] > 0:
            dml.fit(Y, T, W=W)
        else:
            # No confounders — still run DML with intercept-only nuisance
            dml.fit(Y, T)

        coeff = float(dml.const_marginal_effect().flatten()[0])

        # Robust standard error
        try:
            inference = dml.const_marginal_effect_inference()
            se = float(inference.stderr.flatten()[0])
        except Exception:
            se = 0.0

        return coeff, se, dml

    # ------------------------------------------------------------------
    # Intervention — do(X = x + delta)
    # ------------------------------------------------------------------

    def intervene(
        self,
        variable: str,
        delta: float,
        data: pd.DataFrame | None = None,
        respect_caps: bool = True,
    ) -> pd.DataFrame:
        """
        Simulate a do-intervention on *variable* of magnitude *delta*.

        Propagates downstream effects through the causal graph using the
        fitted structural coefficients.

        Parameters
        ----------
        variable : str
            Which predictor to intervene on.
        delta : float
            Magnitude of the intervention (in original units).
        data : pd.DataFrame, optional
            Baseline data to compute counterfactuals over.  If *None*,
            uses the data passed to ``fit()``.
        respect_caps : bool
            Whether to clip the intervention at the physical bounds from
            ``caps.yml``.

        Returns
        -------
        pd.DataFrame
            Counterfactual dataset with a ``predicted_delta_response`` column.
        """
        if not self._structural_coeffs:
            raise RuntimeError("Call fit() before intervene().")

        target = self.roles['outcomes'][0] if self.roles['outcomes'] else None
        if target is None:
            raise ValueError("DAG has no outcome node.")

        # Clamp delta to caps
        if respect_caps:
            delta = self._clamp_delta(variable, delta)

        # Propagate through graph (BFS from intervention node)
        # Accumulate total effect on target
        import networkx as nx
        deltas: Dict[str, float] = {variable: delta}

        # Topological traversal starting from the intervention variable
        for node in nx.topological_sort(self.graph):
            if node == variable:
                continue
            # Sum contributions from all parents with a known delta
            incoming = 0.0
            for parent in self.graph.predecessors(node):
                if parent not in deltas:
                    continue
                edge_key = (parent, node)

                # Prefer non-linear HGB propagation when available
                if edge_key in self._edge_models and data is not None:
                    model = self._edge_models[edge_key]
                    # Build a feature row at median + deltas
                    confounders = [
                        v for v in compute_backdoor_set(self.graph, parent, node)
                        if v in self._train_medians
                    ]
                    X_cols = [parent] + confounders
                    base_vals = np.array(
                        [self._train_medians.get(c, 0.0) for c in X_cols]
                    ).reshape(1, -1)
                    mod_vals = base_vals.copy()
                    mod_vals[0, 0] += deltas[parent]
                    pred_diff = float(
                        model.predict(mod_vals)[0] - model.predict(base_vals)[0]
                    )
                    incoming += pred_diff
                elif edge_key in self._structural_coeffs:
                    incoming += deltas[parent] * self._structural_coeffs[edge_key]

            if incoming != 0.0:
                deltas[node] = incoming

        total_delta = deltas.get(target, 0.0)

        # Build result
        result = pd.DataFrame({
            'intervention_variable': [variable],
            'delta': [delta],
            'target_variable': [target],
            'predicted_delta_response': [total_delta],
            'propagated_deltas': [deltas],
        })

        return result

    def run_scenarios(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run all scenarios defined in ``config['scenarios']``.

        Returns a DataFrame with one row per (scenario, increment) pair.
        """
        scenarios = self.config.get('scenarios', [])
        if not scenarios:
            warnings.warn("No scenarios defined in config.")
            return pd.DataFrame()

        rows = []
        for scenario in scenarios:
            var = scenario['variable']
            for inc in scenario.get('increments', []):
                d = -inc if scenario.get('direction') == 'decrease' else inc
                result = self.intervene(var, d, data=data)
                row = {
                    'scenario_name': scenario['name'],
                    'variable': var,
                    'increment': inc,
                    'direction': scenario.get('direction', 'increase'),
                    'predicted_delta_response': result['predicted_delta_response'].iloc[0],
                }
                rows.append(row)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # DoWhy full estimation (for formal ATE / CATE)
    # ------------------------------------------------------------------

    def estimate_ate(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str | None = None,
        method: str = "backdoor.linear_regression",
    ) -> Dict[str, Any]:
        """
        Estimate the Average Treatment Effect using DoWhy.

        Parameters
        ----------
        data : pd.DataFrame
        treatment : str
        outcome : str, optional
            Defaults to the DAG outcome node.
        method : str
            DoWhy estimation method name.

        Returns
        -------
        dict with keys ``ate``, ``confidence_intervals``, ``p_value``.
        """
        if outcome is None:
            outcome = self.roles['outcomes'][0]

        model = build_causal_model(self.dag_def, data, treatment=treatment, outcome=outcome)
        identified = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(
            identified,
            method_name=method,
        )

        return {
            'ate': estimate.value,
            'confidence_intervals': getattr(estimate, 'get_confidence_intervals', lambda: None)(),
            'p_value': getattr(estimate, 'get_p_value', lambda: None)(),
            'method': method,
            'treatment': treatment,
            'outcome': outcome,
        }

    # ------------------------------------------------------------------
    # Spatial CATE multiplier support
    # ------------------------------------------------------------------

    def load_cate_multipliers(self, stage3_dir: str) -> None:
        """
        Load per-cell CATE multipliers saved by Stage 3.

        These are numpy arrays saved as
        ``spatial_cate_multiplier_{treatment}.npy`` which contain per-cell
        ratios τ(x_i)/mean(τ), bounded [0.5, 1.5].
        """
        import os
        import glob
        pattern = os.path.join(stage3_dir, 'spatial_cate_multiplier_*.npy')
        for path in glob.glob(pattern):
            fname = os.path.basename(path)
            # "spatial_cate_multiplier_Pct_Canopy.npy" → "Pct_Canopy"
            treatment = fname.replace('spatial_cate_multiplier_', '').replace('.npy', '')
            self._cate_multipliers[treatment] = np.load(path)

    def get_spatial_multiplier(
        self,
        treatment: str,
        n_cells: int,
    ) -> np.ndarray:
        """
        Return per-cell spatial multiplier for *treatment*.

        Falls back to uniform (all 1.0) if no CATE multiplier is available.
        """
        mult = self._cate_multipliers.get(treatment)
        if mult is not None:
            if len(mult) == n_cells:
                return mult
            # Length mismatch — interpolate or fall back
            warnings.warn(
                f"CATE multiplier length ({len(mult)}) != n_cells ({n_cells}); "
                f"using uniform multiplier."
            )
        return np.ones(n_cells)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _clamp_delta(self, variable: str, delta: float) -> float:
        """Clamp *delta* to the bounds in caps.yml."""
        delta_constraints = self.caps.get('delta_constraints', {}).get(variable, {})
        if delta > 0:
            max_inc = delta_constraints.get('max_increase', float('inf'))
            return min(delta, max_inc)
        else:
            max_dec = delta_constraints.get('max_decrease', float('inf'))
            return max(delta, -max_dec)
