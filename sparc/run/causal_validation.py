#!/usr/bin/env python3
"""
Stage 3 — Causal Validation
============================

Uses the causal DAG (``project.yml → causal.dag_file``) and the
``causal.counterfactual_engine.CounterfactualEngine`` to:

1. Load the DAG and classify variables (treatment / mediator / confounder).
2. Estimate structural-equation coefficients from the training data
   (OLS + backdoor adjustment via DoWhy).
3. Blend data-estimated effects with physics priors (shrinkage).
4. Run refutation tests (placebo treatment, random common cause) when
   DoWhy is available.
5. Produce ``scenario_coefficients.json`` consumed by Stage 4
   (the ScenarioSimulator).

The stage is deliberately *fast* — no heavy model retraining, just
coefficient estimation and diagnostic tests on tabular data.

Usage
-----
Called automatically by ``sparc run --stage 3`` or ``--stage all``.
Can also be invoked directly::

    python -m run.causal_validation
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Ensure repo root is on path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from sparc.config.config import load_config, load_monotone_constraints
from sparc.data.data_utils import load_and_preprocess_data
from sparc.run.pipeline_paths import get_paths


def _ensure_gdal_home() -> None:
    """
    On Windows, GDAL/fiona may try to resolve HOME to the *system*
    profile (``C:\\WINDOWS\\system32\\config\\systemprofile\\…``) which
    is not writable by normal users.  Set CPL_TMPDIR and HOME to a
    user-writable temp directory so driver config creation succeeds.
    """
    if sys.platform != 'win32':
        return
    import tempfile
    tmp = tempfile.gettempdir()  # always user-writable
    for var in ('CPL_TMPDIR', 'GDAL_PAM_PROXY_DIR'):
        if var not in os.environ:
            os.environ[var] = tmp
    # Fiona/GDAL use HOME for config; redirect if it points to system profile
    home = os.environ.get('HOME', '')
    if not home or 'systemprofile' in home.lower() or not os.path.isdir(home):
        os.environ['HOME'] = os.path.expanduser('~')


# ---------------------------------------------------------------------------
# CausalValidator — the Stage 3 workhorse
# ---------------------------------------------------------------------------

class CausalValidator:
    """
    Orchestrates causal validation between the DAG, physics priors,
    and data-estimated structural coefficients.

    Parameters
    ----------
    config : dict
        Full SPARC config dict (from ``load_config()``).
    """

    def __init__(self, config: dict):
        self.config = config
        self.paths = get_paths()

        # Will be populated by load_dag / fit
        self.dag_def: Optional[dict] = None
        self.graph = None               # networkx.DiGraph
        self.roles: Dict[str, List[str]] = {}
        self.structural_coeffs: Dict[tuple, float] = {}
        self._edge_models: Dict[tuple, Any] = {}  # HGB model objects per-edge
        self.ate_results: Dict[str, dict] = {}
        self.refutation_results: Dict[str, dict] = {}

        # Phase 1: Causal discovery
        self.discovery_report: Optional[dict] = None

        # Phase 2: Multi-estimator ATE
        self.ate_ipw_results: Dict[str, dict] = {}
        self.ate_dr_results: Dict[str, dict] = {}
        self.ate_gps_results: Dict[str, dict] = {}
        self.ate_matching_results: Dict[str, dict] = {}
        self.propensity_diagnostics: Dict[str, dict] = {}

        # Phase 3: Assumption diagnostics
        self.dag_diagnostics: Optional[dict] = None

        # Phase 4: Spatial CATE
        self.spatial_cate_results: Dict[str, dict] = {}
        self._spatial_cate_estimator: Optional[Any] = None

        # Phase 5: Elasticity, relative importance, bootstrap
        self.elasticity: Dict[str, float] = {}
        self.relative_importance: Dict[str, float] = {}
        self.bootstrap_cis: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Step 1: Load DAG
    # ------------------------------------------------------------------

    def load_dag(self) -> None:
        """Load and validate the causal DAG from project config."""
        from sparc.causal.dag_definition import load_dag, dag_to_networkx, get_node_roles

        dag_path = self.config.get('causal', {}).get('dag_file')
        if not dag_path:
            raise ValueError(
                "No causal DAG file specified.  Add  causal.dag_file  to project.yml."
            )
        if not os.path.exists(dag_path):
            raise FileNotFoundError(f"DAG file not found: {dag_path}")

        self.dag_def = load_dag(dag_path)
        self.graph = dag_to_networkx(self.dag_def)
        self.roles = get_node_roles(self.graph)

        print(f"  DAG loaded: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
        print(f"    Treatments:  {self.roles['treatments']}")
        print(f"    Mediators:   {self.roles['mediators']}")
        print(f"    Confounders: {self.roles['confounders']}")
        print(f"    Outcome:     {self.roles['outcomes']}")

    # ------------------------------------------------------------------
    # Step 1b: Causal discovery — validate expert DAG against data
    # ------------------------------------------------------------------

    def run_causal_discovery(self, data: pd.DataFrame, output_dir: str) -> None:
        """
        Run data-driven causal discovery algorithms and compare the
        learned DAG(s) against the expert DAG.

        Requires ``causal.discovery.enabled: true`` in project.yml.
        Writes ``dag_discovery_report.json`` to *output_dir*.
        """
        disc_cfg = self.config.get('causal', {}).get('discovery', {})
        if not disc_cfg.get('enabled', False):
            print("\n  Causal discovery disabled -- skipping.")
            return

        try:
            from sparc.causal.causal_discovery import CausalDiscoveryValidator
        except ImportError:
            print("  causal-learn not installed -- skipping causal discovery.")
            return

        node_names = [n['name'] for n in self.dag_def.get('nodes', [])]
        cols = [c for c in node_names if c in data.columns]
        if len(cols) < 3:
            print("  Too few DAG columns in data -- skipping discovery.")
            return

        methods = disc_cfg.get('methods', ['pc_stable', 'lingam', 'ges'])
        alpha = disc_cfg.get('alpha', 0.05)
        consensus_threshold = disc_cfg.get('consensus_threshold', 2)

        print(f"\n  Running causal discovery (methods={methods}, alpha={alpha})...")

        validator = CausalDiscoveryValidator(
            data[cols], node_names=cols,
        )

        # Run requested algorithms
        if 'pc_stable' in methods:
            validator.run_pc_stable(alpha=alpha)
        if 'lingam' in methods:
            validator.run_lingam()
        if 'ges' in methods:
            validator.run_ges()

        # Consensus DAG
        validator.consensus_dag(min_votes=consensus_threshold)

        # Compare with expert
        if disc_cfg.get('compare_expert', True) and self.graph is not None:
            comparison_all = validator.compare_with_expert(self.graph)
            # Report consensus comparison if available, else first method
            comp_key = 'consensus' if 'consensus' in comparison_all else next(iter(comparison_all), None)
            if comp_key:
                comparison = comparison_all[comp_key]
                print(f"    [{comp_key}] SHD = {comparison.get('shd', 'N/A')}")
                print(f"    [{comp_key}] Edge precision = {comparison.get('precision', 0):.2f}, "
                      f"recall = {comparison.get('recall', 0):.2f}, "
                      f"F1 = {comparison.get('f1', 0):.2f}")

                if comparison.get('missing_edges'):
                    print(f"    Missing edges (in expert, not data): "
                          f"{comparison['missing_edges']}")
                if comparison.get('extra_edges'):
                    print(f"    Extra edges (in data, not expert): "
                          f"{comparison['extra_edges']}")
                # Also print per-method summary
                for mname, mcomp in comparison_all.items():
                    if mname != comp_key:
                        print(f"    [{mname}] SHD={mcomp.get('shd','N/A')}, "
                              f"F1={mcomp.get('f1',0):.2f}, "
                              f"edges={mcomp.get('n_learned_edges',0)}")

        # d-separation validation
        dsep_results = {}
        if self.graph is not None:
            dsep_results = validator.validate_dseparation(
                self.graph, data[cols], alpha=alpha,
            )
            n_tested = dsep_results.get('total', 0)
            n_held = dsep_results.get('passed', 0)
            rate = n_held / n_tested if n_tested > 0 else 0
            print(f"    d-separation: {n_held}/{n_tested} implications held "
                  f"({rate:.0%})")

        # Save report
        self.discovery_report = validator.generate_report()
        # Inject d-separation results into the report dict
        if self.graph is not None and dsep_results:
            self.discovery_report['dseparation'] = dsep_results
        os.makedirs(output_dir, exist_ok=True)
        try:
            from sparc.run.pipeline_paths import get_result_store
            get_result_store().save_json(3, 'dag_discovery_report.json', self.discovery_report)
        except Exception:
            report_path = os.path.join(output_dir, 'dag_discovery_report.json')
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(self.discovery_report, f, indent=2, default=str)
        print(f"    Discovery report saved")

    # ------------------------------------------------------------------
    # Step 1c: DAG assumption diagnostics (positivity, SUTVA)
    # ------------------------------------------------------------------

    def run_dag_diagnostics(self, data: pd.DataFrame) -> None:
        """
        Run assumption diagnostics: positivity, SUTVA (spatial
        autocorrelation of treatment), conditional exchangeability proxies.
        """
        try:
            from sparc.causal.dag_validator import (
                check_positivity,
                check_sutva_spatial,
            )
        except ImportError:
            print("  dag_validator extensions not available -- skipping diagnostics.")
            return

        target = self.roles['outcomes'][0] if self.roles['outcomes'] else None
        coord_cols = self.config.get('variables', {}).get('coordinates', [])

        print("\n  Running DAG assumption diagnostics...")

        diagnostics: Dict[str, Any] = {}

        for treatment in self.roles['treatments']:
            if treatment not in data.columns:
                continue

            confounders = [
                c for c in self.roles.get('confounders', [])
                if c in data.columns
            ]

            # Positivity
            pos = check_positivity(data, treatment, confounders)
            diagnostics.setdefault(treatment, {})['positivity'] = pos
            ps_stats = pos.get('propensity_stats', {})
            n_violations = len(pos.get('violations', []))
            print(f"    {treatment} positivity: "
                  f"violation_rate={pos.get('violation_rate', 0):.2%}, "
                  f"ps_range=[{ps_stats.get('min', 0):.3f}, {ps_stats.get('max', 0):.3f}], "
                  f"violations={n_violations} strata")

            # SUTVA (spatial autocorrelation of treatment)
            if len(coord_cols) >= 2:
                sutva = check_sutva_spatial(data, treatment, coord_cols)
                diagnostics[treatment]['sutva'] = sutva
                mi = sutva.get('morans_i', 0)
                print(f"    {treatment} SUTVA (Moran's I): I={mi:.4f}, "
                      f"p={sutva.get('p_value', 1):.4f}")

        self.dag_diagnostics = diagnostics

    # ------------------------------------------------------------------
    # Step 2: Estimate structural coefficients
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame, coords: np.ndarray | None = None) -> None:
        """
        Estimate structural-equation coefficients for every edge in the DAG.

        Strategy is configurable via ``causal.estimator`` in project.yml:

        * ``"ols"``  (default) — OLS with backdoor adjustment.
        * ``"hgb"``  — HistGradientBoostingRegressor (monotone-constrained).
        * ``"dml"``  (recommended) — Debiased / Orthogonal ML with K-fold
          cross-fitting and optional spatial blocks.

        In all cases, the estimated coefficient is shrunk toward the physics
        prior and checked against monotone constraints.

        Parameters
        ----------
        data : pd.DataFrame
        coords : np.ndarray, optional
            (N, 2) coordinate array for spatial DML folds.
        """
        from sparc.config.config import load_physics_priors
        from sklearn.linear_model import LinearRegression
        from sparc.causal.dag_definition import compute_backdoor_set

        if self.graph is None:
            raise RuntimeError("Call load_dag() before fit().")

        target = self.roles['outcomes'][0] if self.roles['outcomes'] else None
        priors_data = load_physics_priors(self.config)
        coeff_priors = priors_data.get('coefficients', {})
        calibration = priors_data.get('calibration', {})
        shrinkage = calibration.get('shrinkage_weight', 0.7)
        adaptive_weighting = calibration.get('adaptive_weighting', False)
        max_deviation = calibration.get('max_deviation', 0.5)

        # Scale harmonization factors (literature resolution -> 30m target)
        scale_harm = priors_data.get('scale_harmonization', {})
        scale_factors = scale_harm.get('factors', {})

        monotone = load_monotone_constraints(self.config)

        estimator_name = (
            self.config.get('causal', {}).get('estimator', 'hgb').lower()
        )

        # DML: reduced shrinkage (debiased estimator needs less prior correction)
        if estimator_name == 'dml':
            shrinkage = self.config.get('causal', {}).get('dml_shrinkage', 0.25)

        print(f"\n  Estimating structural coefficients "
              f"(estimator={estimator_name}, shrinkage={shrinkage:.2f})...")

        # Lazily initialised for HGB / DML
        self._edge_models: Dict[tuple, Any] = {}
        self._raw_coeffs: Dict[tuple, float] = {}
        self._coeff_ses: Dict[tuple, float] = {}
        self._dml_models: Dict[tuple, Any] = {}

        # DML spatial params
        spatial_block_size = self.config.get('causal', {}).get('spatial_block_size', None)
        dml_n_splits = self.config.get('causal', {}).get('dml_cv_folds', 5)

        for parent, child in self.graph.edges():
            if parent not in data.columns or child not in data.columns:
                continue

            # Backdoor adjustment: full backdoor criterion (Pearl 2009)
            confounders = [
                n for n in compute_backdoor_set(self.graph, parent, child)
                if n in data.columns
            ]
            X_cols = [parent] + confounders

            try:
                if estimator_name == 'dml':
                    from sparc.causal.counterfactual_engine import CounterfactualEngine
                    coeff_data, se, dml_model = CounterfactualEngine._fit_edge_dml(
                        self, data, confounders, child, parent,
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
                    lr = LinearRegression()
                    lr.fit(data[X_cols], data[child])
                    coeff_data = float(lr.coef_[0])
            except Exception as exc:
                print(f"    WARNING: fit failed for {parent}->{child}: {exc}")
                coeff_data = 0.0

            # Store raw (pre-shrinkage) coefficient
            self._raw_coeffs[(parent, child)] = coeff_data

            # Shrink toward physics prior (only for edges → outcome)
            prior_info = coeff_priors.get(parent, {})
            if prior_info and child == target:
                coeff_prior_raw = prior_info.get('value', coeff_data)

                # ── Rescale prior from literature units to per-1-raw-unit ──
                # priors.yml stores e.g. -0.280 "°F per +10 pp" which must
                # be divided by the unit increment to match the regression
                # coefficient scale (per 1 pp or per 1.0 NDVI).
                if "Pct" in parent or "pct" in parent:
                    unit_inc = 10.0      # literature reports per +10 pp
                elif parent in ("NDVI", "Albedo"):
                    unit_inc = 0.1       # literature reports per +0.1
                else:
                    unit_inc = 1.0
                coeff_prior = coeff_prior_raw / unit_inc

                # ── Scale harmonization (literature resolution → 30 m) ──
                source_scale = prior_info.get('source_scale', '')
                scale_factor = 1.0
                for band_key, band_factors in scale_factors.items():
                    if isinstance(band_factors, dict) and parent in band_factors:
                        # Match source_scale to band key (e.g. "100-500m" ↔ "100m-500m")
                        band_clean = band_key.lower().replace('m', '').replace(' ', '')
                        src_clean = source_scale.lower().replace('m', '').replace(' ', '')
                        if band_clean == src_clean:
                            scale_factor = float(band_factors[parent])
                            break
                coeff_prior *= scale_factor

                # ── Adaptive shrinkage (confidence-aware) ──
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

                # ── Max-deviation guard ──
                if coeff_prior != 0:
                    deviation = abs(coeff_blended / coeff_prior - 1.0)
                    if deviation > max_deviation:
                        # Clamp toward prior so deviation stays within bounds
                        bound_lo = coeff_prior * (1.0 - max_deviation)
                        bound_hi = coeff_prior * (1.0 + max_deviation)
                        coeff_blended = float(np.clip(coeff_blended, min(bound_lo, bound_hi), max(bound_lo, bound_hi)))
            else:
                coeff_blended = coeff_data

            # Monotone-constraint check (sign correction)
            if child == target and parent in monotone:
                expected_sign = monotone[parent]
                if expected_sign != 0:
                    actual_sign = 1 if coeff_blended > 0 else (-1 if coeff_blended < 0 else 0)
                    if actual_sign != 0 and actual_sign != expected_sign:
                        print(f"    WARNING: {parent}->{child} coeff={coeff_blended:.4f} "
                              f"violates monotone constraint (expected sign={expected_sign})")

            self.structural_coeffs[(parent, child)] = coeff_blended
            tag = f" [{estimator_name}]"
            se_str = ""
            if (parent, child) in self._coeff_ses:
                se_str = f"  SE={self._coeff_ses[(parent, child)]:.5f}"
            print(f"    {parent:>25s} -> {child:<10s}  coeff={coeff_blended:+.5f}  "
                  f"(raw={coeff_data:+.5f}){se_str}{tag}")

    # ------------------------------------------------------------------
    # HGB edge estimator (monotone-constrained non-linear)
    # ------------------------------------------------------------------

    @staticmethod
    def _fit_edge_hgb(
        data: pd.DataFrame,
        X_cols: List[str],
        child: str,
        parent: str,
        monotone: Dict[str, int],
        target: Optional[str] = None,
        n_grid: int = 200,
    ) -> tuple:
        """
        Fit a HistGradientBoostingRegressor for a single DAG edge and
        extract a *linearised* marginal effect (mean ∂child/∂parent).

        Returns ``(coeff_data, model)`` where *model* is the fitted HGB
        estimator (stored for later non-linear prediction).
        """
        from sklearn.ensemble import HistGradientBoostingRegressor

        # Build monotone_cst array aligned to X_cols.
        # Only apply outcome-level constraints when the child IS the
        # outcome variable.  For mediator/cross-predictor edges the
        # relationship direction may differ (e.g. Canopy→NDVI is
        # positive even though Canopy→Temperature is negative).
        mc = []
        for col in X_cols:
            if child == target:
                mc.append(monotone.get(col, 0))
            else:
                mc.append(0)

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

        # Marginal effect: mean of ∂f/∂parent via finite-difference
        parent_idx = X_cols.index(parent)
        eps = np.std(X[:, parent_idx]) * 0.01  # 1% of std
        if eps < 1e-8:
            eps = 1e-3

        X_plus = X.copy()
        X_plus[:, parent_idx] += eps
        pred_base = hgb.predict(X)
        pred_plus = hgb.predict(X_plus)
        marginal = (pred_plus - pred_base) / eps
        coeff_data = float(np.mean(marginal))

        return coeff_data, hgb

    # ------------------------------------------------------------------
    # Step 3: Formal ATE estimation via DoWhy
    # ------------------------------------------------------------------

    def estimate_ate_all(self, data: pd.DataFrame) -> None:
        """
        Estimate Average Treatment Effect for every treatment → outcome
        edge using DoWhy (if available).

        When ``causal.estimate_cate`` is true in the config, also computes
        Conditional Average Treatment Effects (CATE) via EconML's
        ``LinearDML`` estimator, which provides per-treatment heterogeneous
        effect estimates.
        """
        target = self.roles['outcomes'][0] if self.roles['outcomes'] else None
        if target is None:
            print("  No outcome node -- skipping ATE estimation.")
            return

        try:
            from sparc.causal.dag_definition import build_causal_model
        except ImportError:
            print("  DoWhy not available -- skipping formal ATE estimation.")
            return

        estimate_cate = self.config.get('causal', {}).get('estimate_cate', False)

        print(f"\n  Estimating ATEs via DoWhy (outcome={target})...")

        for treatment in self.roles['treatments']:
            if treatment not in data.columns:
                continue
            try:
                model = build_causal_model(
                    self.dag_def, data,
                    treatment=treatment, outcome=target,
                )
                identified = model.identify_effect(proceed_when_unidentifiable=True)
                estimate = model.estimate_effect(
                    identified,
                    method_name="backdoor.linear_regression",
                )
                ate_value = float(estimate.value)
                self.ate_results[treatment] = {
                    'ate': ate_value,
                    'method': 'backdoor.linear_regression',
                }
                print(f"    ATE({treatment} -> {target}) = {ate_value:+.5f}")

                # ── EconML CATE (optional) ────────────────────────
                if estimate_cate:
                    self._estimate_cate(data, treatment, target)

            except Exception as e:
                print(f"    ATE({treatment} -> {target}) FAILED: {e}")
                self.ate_results[treatment] = {'ate': None, 'error': str(e)}

    # ------------------------------------------------------------------
    # EconML CATE estimation
    # ------------------------------------------------------------------

    def _estimate_cate(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
    ) -> None:
        """
        Estimate heterogeneous treatment effects using EconML's LinearDML.

        Results are stored in ``self.ate_results[treatment]['cate_*']``.
        Falls back to sklearn-only cross-fit DML if econml is broken.
        """
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor as HGB
        except ImportError:
            print(f"      CATE skipped for {treatment} (sklearn not available)")
            return

        predictor_list = self.config.get('predictors', {})
        if isinstance(predictor_list, dict):
            predictor_list = predictor_list.get('base_model', [])
        confounders = [
            c for c in predictor_list
            if c != treatment and c in data.columns
        ]
        # Override with proper backdoor set when DAG is available
        if self.graph:
            from sparc.causal.dag_definition import compute_backdoor_set
            confounders = [
                c for c in compute_backdoor_set(self.graph, treatment, outcome)
                if c in data.columns
            ]
        if not confounders:
            return

        try:
            T = data[[treatment]].values
            Y = data[outcome].values
            X = data[confounders].values

            # Try econml LinearDML first
            try:
                from econml.dml import LinearDML

                dml = LinearDML(
                    model_y=HGB(max_iter=100, max_depth=3, random_state=42),
                    model_t=HGB(max_iter=100, max_depth=3, random_state=42),
                    random_state=42,
                    cv=3,
                )
                dml.fit(Y, T, X=X)

                cate_mean = float(dml.const_marginal_effect(X).mean())
                cate_std = float(dml.const_marginal_effect(X).std())
                method = 'LinearDML'

            except Exception:
                # Fallback: manual cross-fit DML with sklearn only
                from sparc.causal.counterfactual_engine import CounterfactualEngine
                coeff, se, _ = CounterfactualEngine._fit_edge_dml_sklearn(
                    T, Y, X,
                    model_y=HGB(max_iter=100, max_depth=3, random_state=42),
                    model_t=HGB(max_iter=100, max_depth=3, random_state=42),
                    n_splits=3,
                )
                cate_mean = coeff
                cate_std = se
                method = 'CrossFitDML-sklearn'

            self.ate_results[treatment]['cate_mean'] = cate_mean
            self.ate_results[treatment]['cate_std'] = cate_std
            self.ate_results[treatment]['cate_method'] = method

            print(f"      CATE({treatment}): mean={cate_mean:+.5f}  "
                  f"std={cate_std:.5f}  [{method}]")

        except Exception as e:
            print(f"      CATE({treatment}) failed: {e}")

    # ------------------------------------------------------------------
    # Step 3b: IPW ATE estimation (propensity-score weighted)
    # ------------------------------------------------------------------

    def estimate_ate_ipw(self, data: pd.DataFrame) -> None:
        """
        Estimate ATE via Inverse Probability Weighting for each treatment.

        Continuous treatments are binarised at the median.  Propensity
        scores are estimated via logistic regression.  Results stored in
        ``self.ate_ipw_results`` and propensity diagnostics in
        ``self.propensity_diagnostics``.
        """
        ipw_cfg = self.config.get('causal', {})
        if not ipw_cfg.get('ipw_enabled', True):
            return

        target = self.roles['outcomes'][0] if self.roles['outcomes'] else None
        if target is None:
            return

        from sklearn.linear_model import LogisticRegression

        print(f"\n  Estimating ATEs via IPW (outcome={target})...")

        for treatment in self.roles['treatments']:
            if treatment not in data.columns:
                continue

            confounders = [
                c for c in self.roles.get('confounders', [])
                if c in data.columns
            ]
            if not confounders:
                predictor_list = self.config.get('predictors', {})
                if isinstance(predictor_list, dict):
                    predictor_list = predictor_list.get('base_model', [])
                confounders = [
                    c for c in predictor_list
                    if c != treatment and c != target and c in data.columns
                ]
            # Override with proper backdoor set when DAG is available
            if self.graph:
                from sparc.causal.dag_definition import compute_backdoor_set
                confounders = [
                    c for c in compute_backdoor_set(self.graph, treatment, target)
                    if c in data.columns
                ]

            if not confounders:
                print(f"    IPW({treatment}): no confounders -- skipped")
                continue

            try:
                # Binarise treatment at median
                t_median = float(data[treatment].median())
                T_bin = (data[treatment] > t_median).astype(int).values
                Y = data[target].values
                X_conf = data[confounders].values

                # Propensity score via logistic regression
                ps_model = LogisticRegression(
                    max_iter=500, solver='lbfgs', random_state=42,
                )
                ps_model.fit(X_conf, T_bin)
                ps = ps_model.predict_proba(X_conf)[:, 1]

                # Trimming: discard extreme PS
                trim_lo, trim_hi = 0.05, 0.95
                mask = (ps >= trim_lo) & (ps <= trim_hi)
                n_trimmed = int((~mask).sum())
                ps_t = ps[mask]
                T_t = T_bin[mask]
                Y_t = Y[mask]

                # Stabilised IPW ATE
                p_treat = T_t.mean()
                w1 = (T_t * p_treat) / ps_t
                w0 = ((1 - T_t) * (1 - p_treat)) / (1 - ps_t)
                ate_ipw = float((w1 * Y_t).sum() / w1.sum()
                                - (w0 * Y_t).sum() / w0.sum())

                self.ate_ipw_results[treatment] = {
                    'ate_ipw': ate_ipw,
                    'n_trimmed': n_trimmed,
                    'binarization_threshold': t_median,
                }

                # Propensity diagnostics
                self.propensity_diagnostics[treatment] = {
                    'ps_mean': float(ps.mean()),
                    'ps_std': float(ps.std()),
                    'ps_min': float(ps.min()),
                    'ps_max': float(ps.max()),
                    'extreme_fraction': float((~mask).mean()),
                    'n_trimmed': n_trimmed,
                    'n_total': len(ps),
                }

                # Compare with backdoor ATE
                backdoor_ate = self.ate_results.get(treatment, {}).get('ate')
                disc_str = ""
                if backdoor_ate and backdoor_ate != 0:
                    disc_pct = abs(ate_ipw - backdoor_ate) / abs(backdoor_ate) * 100
                    disc_str = f", Delta={disc_pct:.1f}%"
                    if disc_pct > 30:
                        disc_str += " [!] HIGH"

                print(f"    IPW({treatment} -> {target}) = {ate_ipw:+.5f}  "
                      f"(trimmed={n_trimmed}{disc_str})")

            except Exception as e:
                print(f"    IPW({treatment}) FAILED: {e}")
                self.ate_ipw_results[treatment] = {
                    'ate_ipw': None, 'error': str(e),
                }

    # ------------------------------------------------------------------
    # Step 3b-GPS: Generalized Propensity Score for continuous treatments
    # ------------------------------------------------------------------

    def estimate_ate_gps(self, data: pd.DataFrame) -> None:
        """
        Estimate the average dose-response slope for each continuous
        treatment via Generalized Propensity Scores (Hirano & Imbens 2004).

        Unlike binary IPW, this keeps treatments continuous:
        1. Model E[T | X] via HistGradientBoosting on the backdoor set.
        2. GPS = Normal pdf of T evaluated at the fitted conditional mean
           and residual standard deviation.
        3. Stabilised weights = f_marginal(T) / GPS, trimmed at the 1st/99th
           percentile of the weight distribution.
        4. Weighted OLS of Y on T yields the GPS-weighted ATE.
        """
        gps_cfg = self.config.get('causal', {})
        if not gps_cfg.get('gps_enabled', True):
            return

        target = self.roles['outcomes'][0] if self.roles['outcomes'] else None
        if target is None:
            return

        from sklearn.ensemble import HistGradientBoostingRegressor as HGB
        from scipy.stats import norm

        print(f"\n  Estimating ATEs via GPS (continuous, outcome={target})...")

        for treatment in self.roles['treatments']:
            if treatment not in data.columns:
                continue

            # Backdoor adjustment set
            if self.graph:
                from sparc.causal.dag_definition import compute_backdoor_set
                confounders = [
                    c for c in compute_backdoor_set(self.graph, treatment, target)
                    if c in data.columns
                ]
            else:
                confounders = [
                    c for c in self.roles.get('confounders', [])
                    if c in data.columns
                ]
            if not confounders:
                print(f"    GPS({treatment}): no confounders -- skipped")
                continue

            try:
                T = data[treatment].values.astype(np.float64)
                Y = data[target].values.astype(np.float64)
                X_conf = data[confounders].values.astype(np.float64)

                # Step 1: Conditional treatment model  E[T | X]
                t_model = HGB(max_iter=200, max_depth=4, random_state=42)
                t_model.fit(X_conf, T)
                t_hat = t_model.predict(X_conf)
                residuals = T - t_hat
                sigma = float(np.std(residuals))
                if sigma < 1e-12:
                    print(f"    GPS({treatment}): zero residual variance -- skipped")
                    continue

                # Step 2: GPS = f(T | X) = Normal(T; t_hat, sigma)
                gps = norm.pdf(T, loc=t_hat, scale=sigma)
                gps = np.clip(gps, 1e-12, None)  # avoid division by zero

                # Step 3: Marginal density of T
                f_marginal = norm.pdf(T, loc=float(T.mean()), scale=float(T.std()))
                f_marginal = np.clip(f_marginal, 1e-12, None)

                # Stabilised weights
                weights = f_marginal / gps

                # Trim extreme weights (1st/99th percentile)
                lo, hi = np.percentile(weights, [1, 99])
                weights = np.clip(weights, lo, hi)

                # Step 4: Weighted OLS  Y ~ T  (intercept + slope)
                T_design = np.column_stack([np.ones_like(T), T])
                W_sqrt = np.sqrt(weights)
                Tw = T_design * W_sqrt[:, None]
                Yw = Y * W_sqrt
                beta, _, _, _ = np.linalg.lstsq(Tw, Yw, rcond=None)
                ate_gps = float(beta[1])  # slope = dY/dT per unit

                # Effective sample size
                ess = float(weights.sum() ** 2 / (weights ** 2).sum())

                self.ate_gps_results[treatment] = {
                    'ate_gps': ate_gps,
                    'ess': ess,
                    'sigma_residual': sigma,
                    'weight_mean': float(weights.mean()),
                    'weight_max': float(weights.max()),
                }

                # Compare with backdoor ATE
                backdoor_ate = self.ate_results.get(treatment, {}).get('ate')
                disc_str = ""
                if backdoor_ate and backdoor_ate != 0:
                    disc_pct = abs(ate_gps - backdoor_ate) / abs(backdoor_ate) * 100
                    disc_str = f", Delta={disc_pct:.1f}%"
                    if disc_pct > 30:
                        disc_str += " [!] HIGH"

                print(f"    GPS({treatment} -> {target}) = {ate_gps:+.5f}/unit  "
                      f"(ESS={ess:.0f}{disc_str})")

            except Exception as e:
                print(f"    GPS({treatment}) FAILED: {e}")
                self.ate_gps_results[treatment] = {
                    'ate_gps': None, 'error': str(e),
                }

    # ------------------------------------------------------------------
    # Step 3b-Match: Covariate matching cross-check
    # ------------------------------------------------------------------

    def estimate_ate_matching(self, data: pd.DataFrame) -> None:
        """
        Estimate ATE via nearest-neighbour covariate matching for each
        treatment (Abadie & Imbens 2006).

        Treatments are binarised at a *meaningful* threshold (one
        standard deviation above/below mean) rather than the median,
        to create clear treatment/control contrast.  Matching is on
        Mahalanobis distance with caliper = 0.25 SD.
        """
        match_cfg = self.config.get('causal', {})
        if not match_cfg.get('matching_enabled', True):
            return

        target = self.roles['outcomes'][0] if self.roles['outcomes'] else None
        if target is None:
            return

        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler

        print(f"\n  Estimating ATEs via Matching (outcome={target})...")

        for treatment in self.roles['treatments']:
            if treatment not in data.columns:
                continue

            # Backdoor adjustment set
            if self.graph:
                from sparc.causal.dag_definition import compute_backdoor_set
                confounders = [
                    c for c in compute_backdoor_set(self.graph, treatment, target)
                    if c in data.columns
                ]
            else:
                confounders = [
                    c for c in self.roles.get('confounders', [])
                    if c in data.columns
                ]
            if not confounders:
                print(f"    Match({treatment}): no confounders -- skipped")
                continue

            try:
                T_raw = data[treatment].values.astype(np.float64)
                Y = data[target].values.astype(np.float64)
                X_conf = data[confounders].values.astype(np.float64)

                # Binarise at ±0.5 SD from mean (exclude middle band)
                t_mean = float(T_raw.mean())
                t_std = float(T_raw.std())
                if t_std < 1e-12:
                    print(f"    Match({treatment}): zero variance -- skipped")
                    continue

                hi_mask = T_raw >= (t_mean + 0.5 * t_std)
                lo_mask = T_raw <= (t_mean - 0.5 * t_std)
                keep = hi_mask | lo_mask
                if keep.sum() < 20:
                    print(f"    Match({treatment}): too few obs after binarisation -- skipped")
                    continue

                T_bin = hi_mask[keep].astype(int)
                Y_k = Y[keep]
                X_k = X_conf[keep]

                # Standardise covariates for distance calculation
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_k)

                treated_idx = np.where(T_bin == 1)[0]
                control_idx = np.where(T_bin == 0)[0]

                if len(treated_idx) < 5 or len(control_idx) < 5:
                    print(f"    Match({treatment}): insufficient treated/control -- skipped")
                    continue

                # Caliper = 0.25 (in standardised Mahalanobis distance)
                caliper = 0.25 * np.sqrt(X_scaled.shape[1])

                # Match treated → control
                nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
                nn.fit(X_scaled[control_idx])
                distances, indices = nn.kneighbors(X_scaled[treated_idx])

                # Apply caliper
                within_caliper = distances.ravel() <= caliper
                matched_treated = treated_idx[within_caliper]
                matched_control = control_idx[indices.ravel()[within_caliper]]

                n_matched = len(matched_treated)
                if n_matched < 5:
                    print(f"    Match({treatment}): too few matches within caliper -- skipped")
                    continue

                # ATT (Average Treatment effect on the Treated)
                att = float(Y_k[matched_treated].mean() - Y_k[matched_control].mean())

                # Abadie-Imbens SE (simple version)
                diffs = Y_k[matched_treated] - Y_k[matched_control]
                se = float(np.std(diffs, ddof=1) / np.sqrt(n_matched))

                self.ate_matching_results[treatment] = {
                    'att_matching': att,
                    'se': se,
                    'n_matched': n_matched,
                    'n_treated': int(hi_mask.sum()),
                    'n_control': int(lo_mask.sum()),
                    'caliper': caliper,
                }

                # Compare with backdoor ATE
                backdoor_ate = self.ate_results.get(treatment, {}).get('ate')
                disc_str = ""
                if backdoor_ate and backdoor_ate != 0:
                    disc_pct = abs(att - backdoor_ate) / abs(backdoor_ate) * 100
                    disc_str = f", Delta={disc_pct:.1f}%"
                    if disc_pct > 30:
                        disc_str += " [!] HIGH"

                print(f"    Match({treatment} -> {target}) ATT = {att:+.5f}  "
                      f"(SE={se:.5f}, n_matched={n_matched}{disc_str})")

            except Exception as e:
                print(f"    Match({treatment}) FAILED: {e}")
                self.ate_matching_results[treatment] = {
                    'att_matching': None, 'error': str(e),
                }

    # ------------------------------------------------------------------
    # Step 3c: Doubly-Robust ATE estimation (continuous)
    # ------------------------------------------------------------------

    def estimate_ate_dr(self, data: pd.DataFrame) -> None:
        """
        Estimate ATE via doubly-robust estimation using EconML's
        ``LinearDML`` which is inherently DR for continuous treatments.
        Consistent if *either* the treatment model or the outcome model
        is correctly specified (Chernozhukov et al. 2018).

        Unlike the old binary LinearDRLearner approach, this keeps
        treatments continuous — no information-destroying median split.
        """
        dr_cfg = self.config.get('causal', {})
        if not dr_cfg.get('doubly_robust', True):
            return

        target = self.roles['outcomes'][0] if self.roles['outcomes'] else None
        if target is None:
            return

        try:
            from sklearn.ensemble import HistGradientBoostingRegressor as HGB
        except ImportError:
            print("  sklearn not available -- skipping DR estimation.")
            return

        print(f"\n  Estimating ATEs via Doubly-Robust/DML (continuous, outcome={target})...")

        for treatment in self.roles['treatments']:
            if treatment not in data.columns:
                continue

            # Use proper backdoor adjustment set
            if self.graph:
                from sparc.causal.dag_definition import compute_backdoor_set
                confounders = [
                    c for c in compute_backdoor_set(self.graph, treatment, target)
                    if c in data.columns
                ]
            else:
                confounders = [
                    c for c in self.roles.get('confounders', [])
                    if c in data.columns
                ]

            W_cols = [
                c for c in confounders
                if c != treatment and c != target
            ]
            if not W_cols:
                predictor_list = self.config.get('predictors', {})
                if isinstance(predictor_list, dict):
                    predictor_list = predictor_list.get('base_model', [])
                W_cols = [
                    c for c in predictor_list
                    if c != treatment and c != target and c in data.columns
                ]

            if not W_cols:
                print(f"    DR({treatment}): no confounders -- skipped")
                continue

            try:
                T = data[treatment].values.reshape(-1, 1).astype(np.float64)
                Y = data[target].values.astype(np.float64)
                W = data[W_cols].values.astype(np.float64)

                # Try econml LinearDML; fall back to sklearn cross-fit DML
                try:
                    from econml.dml import LinearDML

                    dml = LinearDML(
                        model_y=HGB(max_iter=200, max_depth=4, random_state=42),
                        model_t=HGB(max_iter=200, max_depth=4, random_state=42),
                        discrete_treatment=False,
                        cv=3,
                        random_state=42,
                    )
                    dml.fit(Y, T, W=W)

                    ate_dr = float(dml.const_marginal_ate())
                    inf = dml.const_marginal_ate_inference()
                    se_dr = float(np.mean(inf.stderr_mean))
                    dr_method = 'LinearDML (continuous DR)'

                except Exception:
                    from sparc.causal.counterfactual_engine import CounterfactualEngine
                    ate_dr, se_dr, _ = CounterfactualEngine._fit_edge_dml_sklearn(
                        T, Y, W,
                        model_y=HGB(max_iter=200, max_depth=4, random_state=42),
                        model_t=HGB(max_iter=200, max_depth=4, random_state=42),
                        n_splits=3,
                    )
                    dr_method = 'CrossFitDML-sklearn (DR fallback)'

                self.ate_dr_results[treatment] = {
                    'ate_dr': ate_dr,
                    'se': se_dr,
                    'method': dr_method,
                }

                # Compare with backdoor
                backdoor_ate = self.ate_results.get(treatment, {}).get('ate')
                disc_str = ""
                if backdoor_ate and backdoor_ate != 0:
                    disc_pct = abs(ate_dr - backdoor_ate) / abs(backdoor_ate) * 100
                    disc_str = f", Delta={disc_pct:.1f}%"

                print(f"    DR({treatment} -> {target}) = {ate_dr:+.5f}  "
                      f"(SE={se_dr:.5f}{disc_str})")

            except Exception as e:
                print(f"    DR({treatment}) FAILED: {e}")
                self.ate_dr_results[treatment] = {
                    'ate_dr': None, 'error': str(e),
                }

    # ------------------------------------------------------------------
    # Step 3d: Spatial CATE via CausalForestDML
    # ------------------------------------------------------------------

    def run_spatial_cate(
        self,
        data: pd.DataFrame,
        output_dir: str,
        *,
        artifact_source: str = "frequentist_dml",
        global_anchor: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Estimate spatially-varying CATE for each treatment via
        ``CausalForestDML`` and export a GeoPackage with per-cell effects.

        Parameters
        ----------
        artifact_source
            Tag written into the registry metadata for each CATE
            multiplier (`frequentist_dml`, `bayesian_posterior_mean`,
            or `uniform_fallback`). Used by the desktop UI to label
            the source of the spatial heterogeneity.
        global_anchor
            Optional `{treatment: ATE}` dict (e.g. NUTS posterior mean).
            When provided and CausalForestDML fails or has no backdoor
            set, a uniform multiplier matching the global ATE is
            persisted so the scenario simulator and Budget Optimizer
            still see a usable artifact.
        """
        if not self.config.get('causal', {}).get('estimate_cate', True):
            return

        # Track per-treatment provenance so produce_scenario_coefficients
        # can stamp each `.npy` with the right `metadata.source` tag.
        if not hasattr(self, "_spatial_cate_sources"):
            self._spatial_cate_sources: Dict[str, str] = {}

        try:
            from sparc.causal.spatial_cate import SpatialCATEEstimator
        except ImportError:
            print("  spatial_cate module not available -- skipping.")
            return

        target = self.roles['outcomes'][0] if self.roles['outcomes'] else None
        if target is None:
            return

        coord_cols = self.config.get('variables', {}).get('coordinates', [])
        if len(coord_cols) < 2:
            print("  No coordinate columns -- skipping spatial CATE.")
            return

        predictor_list = self.config.get('predictors', {})
        if isinstance(predictor_list, dict):
            predictor_list = predictor_list.get('base_model', [])

        print(f"\n  Estimating spatial CATE (CausalForestDML)...")

        estimator = SpatialCATEEstimator(self.config)

        for treatment in self.roles['treatments']:
            if treatment not in data.columns:
                continue

            # Use proper backdoor set for spatial CATE
            if self.graph:
                from sparc.causal.dag_definition import compute_backdoor_set
                confounders = [
                    c for c in compute_backdoor_set(self.graph, treatment, target)
                    if c in data.columns
                ]
            else:
                confounders = [
                    c for c in predictor_list
                    if c != treatment and c != target and c in data.columns
                ]
            if not confounders:
                # Empty backdoor set is common in Bayesian-mode runs
                # where the DAG isn't fully populated. Emit a uniform
                # multiplier anchored at the global ATE so the
                # scenario simulator sees a registered artifact.
                anchor = (global_anchor or {}).get(treatment, 1.0)
                try:
                    n = len(data)
                    fallback = np.full(n, float(anchor) if np.isfinite(anchor) else 1.0)
                    self.spatial_cate_results[treatment] = {
                        'cate': fallback,
                        'cate_mean': float(np.mean(fallback)),
                        'cate_std': 0.0,
                        'multiplier': fallback,
                    }
                    self._spatial_cate_sources[treatment] = "uniform_fallback"
                    print(f"    {treatment}: empty backdoor → uniform fallback (anchor={anchor:+.4f})")
                except Exception as exc:
                    print(f"    {treatment}: fallback construction failed: {exc}")
                continue

            try:
                cate_array = estimator.estimate(
                    data=data,
                    treatment=treatment,
                    outcome=target,
                    confounders=confounders,
                    coord_cols=coord_cols,
                )
                self.spatial_cate_results[treatment] = {
                    'cate': cate_array,
                    'cate_mean': float(np.mean(cate_array)),
                    'cate_std': float(np.std(cate_array)),
                }

                # Spatial multiplier for scenario simulator
                mult = estimator.cate_to_spatial_multiplier(treatment)
                self.spatial_cate_results[treatment]['multiplier'] = mult
                self._spatial_cate_sources[treatment] = artifact_source

                summary = estimator.summary().get(treatment, {})
                sig_frac = summary.get('pct_significant', 0) / 100.0
                print(f"    {treatment}: mean CATE={summary.get('mean', 0):+.5f}, "
                      f"significant={sig_frac:.0%}")

            except Exception as e:
                print(f"    Spatial CATE({treatment}) FAILED: {e}")
                # Fall back to a uniform multiplier so downstream stages
                # still find a registered artifact for this treatment.
                anchor = (global_anchor or {}).get(treatment, 1.0)
                try:
                    n = len(data)
                    fallback = np.full(n, float(anchor) if np.isfinite(anchor) else 1.0)
                    self.spatial_cate_results[treatment] = {
                        'cate': fallback,
                        'cate_mean': float(np.mean(fallback)),
                        'cate_std': 0.0,
                        'multiplier': fallback,
                    }
                    self._spatial_cate_sources[treatment] = "uniform_fallback"
                except Exception:
                    pass

        # Export GeoPackage
        try:
            proj_crs = self.config.get('crs', {}).get('target_projected', 'EPSG:26919')
            gdf = estimator.to_geodataframe(data, coord_cols, crs=proj_crs)
            if gdf is not None and len(gdf) > 0:
                os.makedirs(output_dir, exist_ok=True)

                # Fix Windows GDAL/fiona permissions: ensure HOME points to
                # a writable directory so driver configs can be created.
                _ensure_gdal_home()

                try:
                    from sparc.run.pipeline_paths import get_result_store
                    get_result_store().save_geodataframe(3, 'spatial_cate_maps.gpkg', gdf)
                except Exception:
                    gpkg_path = os.path.join(output_dir, 'spatial_cate_maps.gpkg')
                    gdf.to_file(gpkg_path, driver='GPKG')
                print(f"    Spatial CATE maps saved")
        except Exception as e:
            print(f"    GeoPackage export failed: {e}")

        self._spatial_cate_estimator = estimator

    # ------------------------------------------------------------------
    # Step 3d-DR: Dose-Response Curves (non-linear treatment effects)
    # ------------------------------------------------------------------

    def compute_dose_response(self, data: pd.DataFrame, output_dir: str) -> None:
        """
        Estimate dose-response curves for each continuous treatment.

        Uses kernel-weighted local linear regression at multiple dose
        levels to capture non-linearities (e.g., canopy effect may
        saturate above 50%).  This directly addresses Yang/Yin Ch 11
        (GPS) by characterising the full dose-response function, not
        just a single ATE.

        Results stored in ``self.dose_response_results``.
        """
        if not self.config.get('causal', {}).get('dose_response', True):
            return

        target = self.roles['outcomes'][0] if self.roles.get('outcomes') else None
        if target is None:
            return

        from sklearn.ensemble import HistGradientBoostingRegressor as HGB

        print(f"\n  Computing dose-response curves (outcome={target})...")

        self.dose_response_results: Dict[str, dict] = {}

        for treatment in self.roles.get('treatments', []):
            if treatment not in data.columns:
                continue

            # Backdoor set
            if self.graph:
                from sparc.causal.dag_definition import compute_backdoor_set
                confounders = [
                    c for c in compute_backdoor_set(self.graph, treatment, target)
                    if c in data.columns
                ]
            else:
                confounders = [
                    c for c in self.roles.get('confounders', [])
                    if c in data.columns
                ]

            if not confounders:
                print(f"    DoseResp({treatment}): no confounders -- skipped")
                continue

            try:
                T = data[treatment].values.astype(np.float64)
                Y = data[target].values.astype(np.float64)
                X_conf = data[confounders].values.astype(np.float64)

                # Residualise Y and T on confounders (partial regression)
                y_model = HGB(max_iter=150, max_depth=3, random_state=42)
                y_model.fit(X_conf, Y)
                Y_resid = Y - y_model.predict(X_conf)

                t_model = HGB(max_iter=150, max_depth=3, random_state=42)
                t_model.fit(X_conf, T)
                T_resid = T - t_model.predict(X_conf)

                # Evaluate dose-response at n_doses evenly spaced points
                n_doses = 20
                dose_lo, dose_hi = np.percentile(T, [5, 95])
                dose_levels = np.linspace(dose_lo, dose_hi, n_doses)

                # Kernel bandwidth (Silverman's rule on T)
                bw = 1.06 * float(T.std()) * len(T) ** (-0.2)
                bw = max(bw, (dose_hi - dose_lo) / n_doses)

                marginal_effects = []
                dose_means = []

                for d in dose_levels:
                    # Gaussian kernel weights centered at dose d
                    w = np.exp(-0.5 * ((T - d) / bw) ** 2)
                    w /= w.sum() + 1e-12

                    # Weighted local linear regression: Y_resid ~ T_resid
                    w_sqrt = np.sqrt(w)
                    Tw = np.column_stack([np.ones_like(T_resid), T_resid]) * w_sqrt[:, None]
                    Yw = Y_resid * w_sqrt
                    try:
                        beta, _, _, _ = np.linalg.lstsq(Tw, Yw, rcond=None)
                        marginal_effects.append(float(beta[1]))
                    except Exception:
                        marginal_effects.append(float(np.nan))

                    dose_means.append(float(np.sum(w * Y_resid)))

                marginal_effects = np.array(marginal_effects)
                dose_means = np.array(dose_means)

                # Detect non-linearity: compare slope variation to constant ATE
                valid = ~np.isnan(marginal_effects)
                if valid.sum() >= 5:
                    me_valid = marginal_effects[valid]
                    me_range = float(me_valid.max() - me_valid.min())
                    me_mean = float(np.abs(me_valid).mean())
                    nonlinearity = me_range / (me_mean + 1e-12)
                    is_nonlinear = nonlinearity > 0.5
                else:
                    nonlinearity = 0.0
                    is_nonlinear = False

                self.dose_response_results[treatment] = {
                    'dose_levels': dose_levels.tolist(),
                    'marginal_effects': marginal_effects.tolist(),
                    'dose_means': dose_means.tolist(),
                    'bandwidth': bw,
                    'nonlinearity_ratio': nonlinearity,
                    'is_nonlinear': is_nonlinear,
                }

                nl_flag = " [NON-LINEAR]" if is_nonlinear else ""
                print(f"    DoseResp({treatment}): range=[{dose_lo:.1f}, {dose_hi:.1f}], "
                      f"nonlinearity={nonlinearity:.2f}{nl_flag}")

            except Exception as e:
                print(f"    DoseResp({treatment}) FAILED: {e}")

        # Save dose-response data
        if self.dose_response_results:
            os.makedirs(output_dir, exist_ok=True)
            dr_path = os.path.join(output_dir, 'dose_response_curves.json')
            try:
                from sparc.run.pipeline_paths import get_result_store
                get_result_store().save_json(3, 'dose_response_curves.json', self.dose_response_results)
            except Exception:
                with open(dr_path, 'w', encoding='utf-8') as f:
                    json.dump(self.dose_response_results, f, indent=2, default=str)
            try:
                from sparc.registry.run_registry import register_path
                register_path(dr_path, stage="3", artifact_id="dose_response_curves",
                              format="json", producer="causal_validation",
                              consumers=["server:/results/dose_response"])
            except Exception:
                pass
            print(f"  Dose-response curves saved")

    # ------------------------------------------------------------------
    # Step 3e: Elasticity coefficients
    # ------------------------------------------------------------------

    def compute_elasticity(self, data: pd.DataFrame) -> None:
        """
        Compute elasticity for each treatment→outcome edge:

            ε = ATE × X̄ / Ȳ

        Interpretation: % change in outcome per 1% change from mean
        treatment level.
        """
        target = self.roles['outcomes'][0] if self.roles['outcomes'] else None
        if target is None:
            return

        y_mean = float(data[target].mean())
        if abs(y_mean) < 1e-10:
            return

        print("\n  Computing elasticity coefficients...")

        for treatment in self.roles['treatments']:
            if treatment not in data.columns:
                continue

            ate_info = self.ate_results.get(treatment, {})
            ate_val = ate_info.get('ate')
            if ate_val is None:
                continue

            x_mean = float(data[treatment].mean())
            elasticity = ate_val * x_mean / y_mean
            self.elasticity[treatment] = elasticity
            print(f"    eps({treatment}) = {elasticity:+.4f}")

    # ------------------------------------------------------------------
    # Step 3f: Relative importance
    # ------------------------------------------------------------------

    def compute_relative_importance(self, data: pd.DataFrame) -> None:
        """
        Proportional contribution of each treatment:

            RI_j = |ATE_j × σ_j| / Σ|ATE_k × σ_k|
        """
        target = self.roles['outcomes'][0] if self.roles['outcomes'] else None
        if target is None:
            return

        raw_scores: Dict[str, float] = {}
        for treatment in self.roles['treatments']:
            if treatment not in data.columns:
                continue
            ate_val = self.ate_results.get(treatment, {}).get('ate')
            if ate_val is None:
                continue
            sigma = float(data[treatment].std())
            raw_scores[treatment] = abs(ate_val * sigma)

        total = sum(raw_scores.values())
        if total < 1e-12:
            return

        print("\n  Relative importance:")
        for var, score in raw_scores.items():
            ri = score / total
            self.relative_importance[var] = ri
            print(f"    RI({var}) = {ri:.3f}")

    # ------------------------------------------------------------------
    # Step 3g: Enhanced bootstrap confidence intervals
    # ------------------------------------------------------------------

    def run_enhanced_bootstrap(
        self,
        data: pd.DataFrame,
        n_bootstrap: Optional[int] = None,
    ) -> None:
        """
        Bootstrap ATE confidence intervals (configurable iterations).

        Uses ``causal.bootstrap_n`` from config (default 100).
        """
        if n_bootstrap is None:
            n_bootstrap = self.config.get('causal', {}).get('bootstrap_n', 100)

        target = self.roles['outcomes'][0] if self.roles['outcomes'] else None
        if target is None:
            return

        # Only run if we have more than a trivial sample
        if len(data) < 100:
            return

        from sklearn.linear_model import LinearRegression

        print(f"\n  Bootstrap ATEs (n={n_bootstrap})...")

        rng = np.random.RandomState(42)

        for treatment in self.roles['treatments']:
            if treatment not in data.columns:
                continue

            # Use proper backdoor set for bootstrap
            if self.graph:
                from sparc.causal.dag_definition import compute_backdoor_set
                confounders = [
                    n for n in compute_backdoor_set(self.graph, treatment,
                                                    self.roles['outcomes'][0])
                    if n in data.columns
                ]
            else:
                confounders = []

            X_cols = [treatment] + confounders
            available = [c for c in X_cols if c in data.columns]
            if treatment not in available:
                continue

            boot_ates = []
            for _ in range(n_bootstrap):
                idx = rng.choice(len(data), size=len(data), replace=True)
                d_boot = data.iloc[idx]
                try:
                    lr = LinearRegression()
                    lr.fit(d_boot[available], d_boot[target])
                    coef_idx = available.index(treatment)
                    boot_ates.append(float(lr.coef_[coef_idx]))
                except Exception:
                    continue

            if boot_ates:
                ci_lo = float(np.percentile(boot_ates, 2.5))
                ci_hi = float(np.percentile(boot_ates, 97.5))
                self.bootstrap_cis[treatment] = {
                    'ci_lower': ci_lo,
                    'ci_upper': ci_hi,
                    'se': float(np.std(boot_ates)),
                    'n_bootstrap': len(boot_ates),
                }
                print(f"    {treatment}: ATE 95% CI = [{ci_lo:+.5f}, {ci_hi:+.5f}]")

    # ------------------------------------------------------------------
    # Step 4: Refutation tests
    # ------------------------------------------------------------------

    def run_refutations(self, data: pd.DataFrame, n_subset: int = 5000) -> None:
        """
        Run DoWhy refutation tests for each treatment → outcome pair.

        Tests run:
        1. Placebo treatment (permutation)
        2. Random common cause
        3. Data subset (bootstrap stability)
        4. Unobserved common cause (add random confounder)

        Also computes an **E-value** for each ATE estimate, quantifying
        how strong an unmeasured confounder would need to be to explain
        away the observed effect.
        """
        target = self.roles['outcomes'][0] if self.roles['outcomes'] else None
        if target is None:
            return

        try:
            from sparc.causal.dag_definition import build_causal_model
        except ImportError:
            print("  DoWhy not available -- skipping refutations.")
            return

        # Subsample for speed
        if len(data) > n_subset:
            data_sub = data.sample(n=n_subset, random_state=42)
        else:
            data_sub = data

        print(f"\n  Running refutation tests (n={len(data_sub)})...")

        for treatment in self.roles['treatments']:
            if treatment not in data_sub.columns:
                continue
            try:
                model = build_causal_model(
                    self.dag_def, data_sub,
                    treatment=treatment, outcome=target,
                )
                identified = model.identify_effect(proceed_when_unidentifiable=True)
                estimate = model.estimate_effect(
                    identified,
                    method_name="backdoor.linear_regression",
                )

                ref_results: Dict[str, Any] = {}
                ate_val = float(estimate.value)

                # 1. Placebo treatment
                try:
                    placebo = model.refute_estimate(
                        identified, estimate,
                        method_name="placebo_treatment_refuter",
                        placebo_type="permute",
                    )
                    placebo_pass = abs(float(placebo.new_effect)) < abs(ate_val) * 0.5
                    ref_results['placebo_pass'] = placebo_pass
                except Exception:
                    ref_results['placebo_pass'] = None

                # 2. Random common cause
                try:
                    rcc = model.refute_estimate(
                        identified, estimate,
                        method_name="random_common_cause",
                    )
                    rcc_pass = abs(float(rcc.new_effect) - ate_val) < abs(ate_val) * 0.2
                    ref_results['random_common_cause_pass'] = rcc_pass
                except Exception:
                    ref_results['random_common_cause_pass'] = None

                # 3. Data subset validation (bootstrap stability)
                try:
                    subset_ref = model.refute_estimate(
                        identified, estimate,
                        method_name="data_subset_refuter",
                        subset_fraction=0.8,
                        num_simulations=5,
                    )
                    subset_pass = (
                        abs(float(subset_ref.new_effect) - ate_val)
                        < abs(ate_val) * 0.3
                    )
                    ref_results['data_subset_pass'] = subset_pass
                except Exception:
                    ref_results['data_subset_pass'] = None

                # 4. Add unobserved common cause
                try:
                    ucc = model.refute_estimate(
                        identified, estimate,
                        method_name="add_unobserved_common_cause",
                        confounders_effect_on_treatment="binary_flip",
                        confounders_effect_on_outcome="linear",
                        effect_strength_on_treatment=0.01,
                        effect_strength_on_outcome=0.02,
                    )
                    ucc_pass = (
                        abs(float(ucc.new_effect) - ate_val)
                        < abs(ate_val) * 0.25
                    )
                    ref_results['unobserved_common_cause_pass'] = ucc_pass
                except Exception:
                    ref_results['unobserved_common_cause_pass'] = None

                # 5. E-value computation
                ref_results['e_value'] = self._compute_e_value(
                    data_sub, treatment, target,
                )

                # 6. Cinelli-Hazlett sensitivity (partial R^2 robustness)
                if self.graph:
                    from sparc.causal.dag_definition import compute_backdoor_set
                    confounders_for_ch = [
                        n for n in compute_backdoor_set(self.graph, treatment, target)
                        if n in data_sub.columns
                    ]
                else:
                    confounders_for_ch = []
                ch = self._cinelli_hazlett_robustness(
                    data_sub, treatment, target, confounders_for_ch,
                )
                if ch:
                    ref_results['cinelli_hazlett'] = ch

                self.refutation_results[treatment] = ref_results

                # Print summary
                parts = []
                for key in ('placebo_pass', 'random_common_cause_pass',
                            'data_subset_pass', 'unobserved_common_cause_pass'):
                    val = ref_results.get(key)
                    label = key.replace('_pass', '')
                    parts.append(
                        f"{label}={'PASS' if val else 'FAIL' if val is not None else 'N/A'}"
                    )
                e_val = ref_results.get('e_value')
                e_str = f"e-value={e_val:.2f}" if e_val is not None else "e-value=N/A"
                print(f"    {treatment}: {', '.join(parts)}, {e_str}")

            except Exception as e:
                print(f"    {treatment}: refutation failed -- {e}")

    # ------------------------------------------------------------------
    # E-value computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_e_value(
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
    ) -> Optional[float]:
        """
        Compute the E-value for the treatment→outcome relationship.

        The E-value quantifies the minimum strength of association that an
        unmeasured confounder would need to have with *both* the treatment
        and the outcome (conditional on measured covariates) to explain away
        the observed effect.

        Uses the approximate formula: E = RR + sqrt(RR * (RR - 1)), where
        RR is an approximation based on the standardised mean difference.

        Returns None on failure.
        """
        try:
            t = data[treatment].values
            y = data[outcome].values

            # Standardised mean difference (Cohen's d for continuous treatment)
            t_std = np.std(t)
            if t_std < 1e-10:
                return None

            from sklearn.linear_model import LinearRegression
            lr = LinearRegression()
            lr.fit(t.reshape(-1, 1), y)
            beta = float(lr.coef_[0])

            # Convert to approximate risk ratio via Hasselblad-Hedges
            # RR ≈ exp(0.91 * d) where d = beta / SD(Y)
            y_std = np.std(y)
            if y_std < 1e-10:
                return None
            d = abs(beta * t_std) / y_std
            rr = np.exp(0.91 * d)

            if rr < 1.0:
                rr = 1.0 / rr  # Ensure RR >= 1

            e_value = float(rr + np.sqrt(rr * (rr - 1.0)))
            return e_value
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Cinelli-Hazlett sensitivity analysis
    # ------------------------------------------------------------------

    @staticmethod
    def _cinelli_hazlett_robustness(
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        confounders: List[str],
    ) -> Optional[Dict[str, float]]:
        """
        Compute the partial R² robustness value (Cinelli & Hazlett 2020).

        Returns the maximum R²_{Y~U|T,C} that an unmeasured confounder U
        could explain while still leaving ATE != 0.
        """
        try:
            from sklearn.linear_model import LinearRegression

            avail = [c for c in confounders if c in data.columns]
            T = data[treatment].values.reshape(-1, 1)
            Y = data[outcome].values

            # Full model: Y ~ T + C
            X_full = data[[treatment] + avail].values
            lr_full = LinearRegression().fit(X_full, Y)
            r2_full = lr_full.score(X_full, Y)

            # Reduced model: Y ~ C only
            if avail:
                X_red = data[avail].values
                lr_red = LinearRegression().fit(X_red, Y)
                r2_reduced = lr_red.score(X_red, Y)
            else:
                r2_reduced = 0.0

            # Partial R² of T given C
            partial_r2_t = (r2_full - r2_reduced) / (1.0 - r2_reduced + 1e-12)

            # Robustness value: R²_{Y~D|X} — how much of residual variance
            # T explains.  This is an upper bound on how strong an
            # unmeasured confounder must be to nullify T's effect.
            return {
                'partial_r2_treatment': float(partial_r2_t),
                'r2_full': float(r2_full),
                'r2_reduced': float(r2_reduced),
                'robustness_value': float(np.sqrt(partial_r2_t)),
            }
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Step 4b: DAG sensitivity analysis (contested edge directions)
    # ------------------------------------------------------------------

    def run_dag_sensitivity(
        self,
        data: pd.DataFrame,
        edge_parent: str = 'Pct_Canopy',
        edge_child: str = 'Pct_Impervious',
        coords: np.ndarray | None = None,
    ) -> Dict[str, Any]:
        """
        Test sensitivity of causal estimates to a contested DAG edge.

        Runs the full coefficient estimation under three DAG variants
        (original, reversed, removed) and reports how each treatment's
        structural coefficient changes.

        Parameters
        ----------
        data : pd.DataFrame
        edge_parent, edge_child : str
            The contested edge to vary.
        coords : np.ndarray, optional
            Spatial coordinates for DML folds.

        Returns
        -------
        dict
            ``{variant_name: {(parent, child): coeff}}`` for comparison.
        """
        from sparc.causal.dag_definition import (
            generate_dag_variants, dag_to_networkx, compute_backdoor_set,
        )
        from sparc.config.config import load_monotone_constraints
        from sklearn.linear_model import LinearRegression

        print(f"\n  DAG Sensitivity Analysis: varying {edge_parent} -> {edge_child}")
        print("  " + "-" * 56)

        variants = generate_dag_variants(self.dag_def, edge_parent, edge_child)
        results: Dict[str, Dict] = {}

        target = self.roles['outcomes'][0] if self.roles['outcomes'] else None
        estimator_name = self.config.get('causal', {}).get('estimator', 'hgb').lower()

        for vname, vdef in variants.items():
            try:
                G_v = dag_to_networkx(vdef)
            except ValueError as exc:
                print(f"    {vname}: SKIPPED ({exc})")
                continue

            coeffs_v: Dict[tuple, float] = {}

            for parent, child in G_v.edges():
                if parent not in data.columns or child not in data.columns:
                    continue
                confounders = [
                    v for v in compute_backdoor_set(G_v, parent, child)
                    if v in data.columns
                ]
                X_cols = [parent] + confounders
                try:
                    if estimator_name == 'dml':
                        from sparc.causal.counterfactual_engine import CounterfactualEngine
                        coeff, _se, _model = CounterfactualEngine._fit_edge_dml(
                            self, data, confounders, child, parent,
                            coords=coords,
                        )
                    elif estimator_name == 'hgb':
                        monotone = load_monotone_constraints(self.config)
                        coeff, _ = self._fit_edge_hgb(
                            data, X_cols, child, parent, monotone, target=target,
                        )
                    else:
                        lr = LinearRegression()
                        lr.fit(data[X_cols], data[child])
                        coeff = float(lr.coef_[0])
                except Exception:
                    coeff = 0.0

                coeffs_v[(parent, child)] = coeff

            results[vname] = coeffs_v

        # Print comparison table
        # Focus on treatment→outcome edges
        treatments = self.roles.get('treatments', [])
        print(f"\n  {'Edge':<35s}", end="")
        for vname in results:
            print(f"  {vname:>12s}", end="")
        print()
        print("  " + "-" * (35 + 14 * len(results)))

        for t in treatments:
            edge_key = (t, target)
            print(f"  {t + ' -> ' + target:<35s}", end="")
            for vname, coeffs_v in results.items():
                c = coeffs_v.get(edge_key, float('nan'))
                print(f"  {c:>+12.5f}", end="")
            print()

        self.dag_sensitivity_results = results
        return results

    # ------------------------------------------------------------------
    # Step 4d: CI propagation through mediation paths (delta method)
    # ------------------------------------------------------------------

    def compute_mediation_effects(self, data: pd.DataFrame) -> Dict[str, dict]:
        """
        Decompose total effects into direct and indirect (mediated)
        components with confidence intervals via the delta method.

        For each treatment T and mediator M on the path T → M → Y:
            - Direct effect:   β(T→Y)
            - Indirect effect: β(T→M) × β(M→Y)
            - Total effect:    direct + indirect
            - SE(indirect) via Sobel test: sqrt(β_TM² σ_MY² + β_MY² σ_TM²)

        Results stored in ``self.mediation_results``.
        """
        target = self.roles['outcomes'][0] if self.roles.get('outcomes') else None
        if not target or not self.graph:
            return {}

        mediators = self.roles.get('mediators', [])
        if not mediators:
            return {}

        print(f"\n  Computing mediation decomposition (outcome={target})...")
        self.mediation_results: Dict[str, dict] = {}

        for treatment in self.roles.get('treatments', []):
            direct_key = (treatment, target)
            direct_coeff = self.structural_coeffs.get(direct_key, 0.0)
            direct_se = self._coeff_ses.get(direct_key, float('nan'))

            indirect_total = 0.0
            indirect_se_sq = 0.0
            indirect_paths: List[dict] = []

            for mediator in mediators:
                # Check if path T → M → Y exists in the graph
                edge_tm = (treatment, mediator)
                edge_my = (mediator, target)

                if edge_tm not in self.structural_coeffs or edge_my not in self.structural_coeffs:
                    continue

                beta_tm = self.structural_coeffs[edge_tm]
                beta_my = self.structural_coeffs[edge_my]
                se_tm = self._coeff_ses.get(edge_tm, float('nan'))
                se_my = self._coeff_ses.get(edge_my, float('nan'))

                # Indirect effect via this mediator
                indirect = beta_tm * beta_my
                indirect_total += indirect

                # Sobel SE: sqrt(β_TM² σ_MY² + β_MY² σ_TM²)
                if not (np.isnan(se_tm) or np.isnan(se_my)):
                    sobel_var = (beta_tm ** 2) * (se_my ** 2) + (beta_my ** 2) * (se_tm ** 2)
                    sobel_se = float(np.sqrt(sobel_var))
                    indirect_se_sq += sobel_var
                else:
                    sobel_se = float('nan')

                indirect_paths.append({
                    'mediator': mediator,
                    'beta_T_M': beta_tm,
                    'beta_M_Y': beta_my,
                    'indirect_effect': indirect,
                    'sobel_se': sobel_se,
                    'significant': (
                        abs(indirect / sobel_se) > 1.96
                        if not np.isnan(sobel_se) and sobel_se > 0
                        else None
                    ),
                })

            total_effect = direct_coeff + indirect_total

            # Total SE (assuming independence of direct and indirect)
            total_se = float('nan')
            if not np.isnan(direct_se) and indirect_se_sq > 0:
                total_se = float(np.sqrt(direct_se ** 2 + indirect_se_sq))

            self.mediation_results[treatment] = {
                'direct_effect': direct_coeff,
                'direct_se': direct_se,
                'indirect_total': indirect_total,
                'indirect_se': float(np.sqrt(indirect_se_sq)) if indirect_se_sq > 0 else float('nan'),
                'total_effect': total_effect,
                'total_se': total_se,
                'indirect_paths': indirect_paths,
                'mediation_proportion': (
                    abs(indirect_total / total_effect) if abs(total_effect) > 1e-12 else 0.0
                ),
            }

            # Print
            med_pct = self.mediation_results[treatment]['mediation_proportion'] * 100
            se_str = f"  SE={total_se:.5f}" if not np.isnan(total_se) else ""
            print(f"    {treatment} -> {target}:  "
                  f"direct={direct_coeff:+.5f}  indirect={indirect_total:+.5f}  "
                  f"total={total_effect:+.5f}{se_str}  "
                  f"({med_pct:.1f}% mediated)")

            for path_info in indirect_paths:
                sig = ""
                if path_info['significant'] is True:
                    sig = " *"
                elif path_info['significant'] is False:
                    sig = " (ns)"
                print(f"      via {path_info['mediator']}: "
                      f"{path_info['indirect_effect']:+.5f}  "
                      f"(SE={path_info['sobel_se']:.5f}){sig}")

        return self.mediation_results

    # ------------------------------------------------------------------
    # Step 4d-LS: Location-stratified treatment effects
    # ------------------------------------------------------------------

    def compute_location_stratified_effects(self, data: pd.DataFrame) -> Dict[str, dict]:
        """
        Estimate treatment effects stratified by the Location variable.

        Runs OLS with backdoor adjustment within each location stratum
        to reveal spatial heterogeneity in treatment effects across
        distinct neighborhoods/land-use zones.

        Results stored in ``self.location_stratified_results``.
        """
        location_col = self.config.get('variables', {}).get('location', 'Location')
        if location_col not in data.columns:
            return {}

        target = self.roles['outcomes'][0] if self.roles.get('outcomes') else None
        if not target:
            return {}

        locations = sorted(data[location_col].dropna().unique())
        if len(locations) < 2:
            return {}

        from sklearn.linear_model import LinearRegression

        print(f"\n  Location-stratified effects ({len(locations)} locations, outcome={target})...")
        self.location_stratified_results: Dict[str, dict] = {}

        for treatment in self.roles.get('treatments', []):
            if treatment not in data.columns:
                continue

            # Backdoor set
            if self.graph:
                from sparc.causal.dag_definition import compute_backdoor_set
                confounders = [
                    c for c in compute_backdoor_set(self.graph, treatment, target)
                    if c in data.columns
                ]
            else:
                confounders = [
                    c for c in self.roles.get('confounders', [])
                    if c in data.columns
                ]

            strata: Dict[str, dict] = {}

            for loc in locations:
                mask = data[location_col] == loc
                df_loc = data[mask]

                if len(df_loc) < 30:
                    continue

                X_cols = [treatment] + confounders
                X_cols_valid = [c for c in X_cols if c in df_loc.columns]

                try:
                    lr = LinearRegression()
                    lr.fit(df_loc[X_cols_valid], df_loc[target])
                    coeff = float(lr.coef_[0])  # treatment coefficient

                    # Residual-based SE
                    resid = df_loc[target].values - lr.predict(df_loc[X_cols_valid])
                    mse = float(np.mean(resid ** 2))
                    XtX_inv = np.linalg.pinv(
                        df_loc[X_cols_valid].values.T @ df_loc[X_cols_valid].values
                    )
                    se = float(np.sqrt(mse * XtX_inv[0, 0]))

                    strata[str(loc)] = {
                        'coefficient': coeff,
                        'se': se,
                        'n': int(len(df_loc)),
                        'significant': abs(coeff / se) > 1.96 if se > 0 else None,
                    }
                except Exception:
                    continue

            if strata:
                # Overall coefficient for comparison
                overall_coeff = self.structural_coeffs.get((treatment, target), 0.0)

                self.location_stratified_results[treatment] = {
                    'strata': strata,
                    'overall_coefficient': overall_coeff,
                }

                # Print compact summary
                vals = [s['coefficient'] for s in strata.values()]
                print(f"    {treatment}: overall={overall_coeff:+.5f}  "
                      f"strata range=[{min(vals):+.5f}, {max(vals):+.5f}]  "
                      f"({len(strata)} locations)")

        return self.location_stratified_results

    # ------------------------------------------------------------------
    # Step 4e: Physics-prior diagnostic comparison table
    # ------------------------------------------------------------------

    def build_coefficient_comparison_table(self) -> Dict[str, dict]:
        """
        Build a comparison table showing, for each treatment → outcome edge:
        - Literature prior (raw from priors.yml, rescaled to per-unit)
        - Raw data-estimated coefficient (pre-shrinkage)
        - Prior-blended coefficient (post-shrinkage, stored in structural_coeffs)
        - GPS estimate (if available)
        - DR/DML estimate (if available)
        - Matching estimate (if available)
        - Discrepancy metrics

        Prints a nicely formatted table and returns the data as a dict.
        """
        from sparc.config.config import load_physics_priors

        priors_data = load_physics_priors(self.config)
        coeff_priors = priors_data.get('coefficients', {})
        scale_harm = priors_data.get('scale_harmonization', {})
        scale_factors = scale_harm.get('factors', {})

        target = self.roles['outcomes'][0] if self.roles.get('outcomes') else None
        if not target:
            return {}

        table: Dict[str, dict] = {}

        print("\n  ╔═══════════════════════════════════════════════════════════"
              "═══════════════════════════════════════════╗")
        print("  ║                     COEFFICIENT COMPARISON TABLE"
              "                                            ║")
        print("  ╠═══════════════════════════════════════════════════════════"
              "═══════════════════════════════════════════╣")
        print(f"  ║ {'Treatment':<18s} ║ {'Lit Prior':>10s} ║ {'Raw Data':>10s} ║ "
              f"{'Blended':>10s} ║ {'GPS':>10s} ║ {'DR/DML':>10s} ║ "
              f"{'Match':>10s} ║ {'Data/Lit':>8s} ║")
        print("  ╠══════════════════╬════════════╬════════════╬"
              "════════════╬════════════╬════════════╬"
              "════════════╬══════════╣")

        for treatment in self.roles.get('treatments', []):
            edge_key = (treatment, target)

            # Literature prior rescaled to per-unit
            prior_info = coeff_priors.get(treatment, {})
            prior_raw = prior_info.get('value', None)
            if prior_raw is not None:
                if "Pct" in treatment or "pct" in treatment:
                    unit_inc = 10.0
                elif treatment in ("NDVI", "Albedo"):
                    unit_inc = 0.1
                else:
                    unit_inc = 1.0
                prior_per_unit = prior_raw / unit_inc

                # Scale harmonization
                source_scale = prior_info.get('source_scale', '')
                for band_key, band_factors in scale_factors.items():
                    if isinstance(band_factors, dict) and treatment in band_factors:
                        band_clean = band_key.lower().replace('m', '').replace(' ', '')
                        src_clean = source_scale.lower().replace('m', '').replace(' ', '')
                        if band_clean == src_clean:
                            prior_per_unit *= float(band_factors[treatment])
                            break
            else:
                prior_per_unit = None

            raw_coeff = self._raw_coeffs.get(edge_key)
            blended = self.structural_coeffs.get(edge_key)
            gps_val = self.ate_gps_results.get(treatment, {}).get('ate_gps')
            dr_val = self.ate_dr_results.get(treatment, {}).get('ate_dr')
            match_val = self.ate_matching_results.get(treatment, {}).get('att_matching')

            # Ratio of data estimate to literature
            if prior_per_unit and prior_per_unit != 0 and raw_coeff is not None:
                ratio = raw_coeff / prior_per_unit
                ratio_str = f"{ratio:>7.2f}×"
            else:
                ratio_str = "   N/A  "

            def fmt(v: object) -> str:
                return f"{v:>+10.5f}" if v is not None else "       N/A"

            print(f"  ║ {treatment:<18s} ║ {fmt(prior_per_unit)} ║ {fmt(raw_coeff)} ║ "
                  f"{fmt(blended)} ║ {fmt(gps_val)} ║ {fmt(dr_val)} ║ "
                  f"{fmt(match_val)} ║ {ratio_str} ║")

            table[treatment] = {
                'literature_prior_per_unit': prior_per_unit,
                'raw_data_coeff': raw_coeff,
                'blended_coeff': blended,
                'gps_ate': gps_val,
                'dr_ate': dr_val,
                'matching_att': match_val,
                'data_to_literature_ratio': float(ratio_str.replace('×', '').strip()) if 'N/A' not in ratio_str else None,
                'confidence': prior_info.get('confidence', 'unknown'),
            }

        print("  ╚══════════════════╩════════════╩════════════╩"
              "════════════╩════════════╩════════════╩"
              "════════════╩══════════╝")

        return table

    # ------------------------------------------------------------------
    # Step 5: Produce scenario_coefficients.json
    # ------------------------------------------------------------------

    def produce_scenario_coefficients(self, output_dir: str) -> dict:
        """
        Write ``scenario_coefficients.json`` with DAG-adjusted coefficients
        for the ScenarioSimulator (Stage 4).

        Returns the dict that was written.
        """
        target = self.roles['outcomes'][0] if self.roles['outcomes'] else None
        monotone = load_monotone_constraints(self.config)

        coefficients = {}
        for (parent, child), coeff in self.structural_coeffs.items():
            if child == target:
                ate_info = self.ate_results.get(parent, {})
                refute_info = self.refutation_results.get(parent, {})
                ipw_info = self.ate_ipw_results.get(parent, {})
                dr_info = self.ate_dr_results.get(parent, {})
                boot_info = self.bootstrap_cis.get(parent, {})

                entry = {
                    'structural_coeff': coeff,
                    # Backdoor ATE
                    'ate_backdoor': ate_info.get('ate'),
                    'cate_mean': ate_info.get('cate_mean'),
                    'cate_std': ate_info.get('cate_std'),
                    # IPW ATE
                    'ate_ipw': ipw_info.get('ate_ipw'),
                    # GPS ATE (continuous)
                    'ate_gps': self.ate_gps_results.get(parent, {}).get('ate_gps'),
                    # Matching ATT
                    'att_matching': self.ate_matching_results.get(parent, {}).get('att_matching'),
                    # Doubly-robust ATE
                    'ate_doubly_robust': dr_info.get('ate_dr'),
                    # Monotone
                    'monotone_constraint': monotone.get(parent, 0),
                    # Refutation tests
                    'placebo_pass': refute_info.get('placebo_pass'),
                    'rcc_pass': refute_info.get('random_common_cause_pass'),
                    'data_subset_pass': refute_info.get('data_subset_pass'),
                    'ucc_pass': refute_info.get('unobserved_common_cause_pass'),
                    'e_value': refute_info.get('e_value'),
                    # Bootstrap CI
                    'bootstrap_ci_lower': boot_info.get('ci_lower'),
                    'bootstrap_ci_upper': boot_info.get('ci_upper'),
                    'bootstrap_se': boot_info.get('se'),
                    # Elasticity & relative importance
                    'elasticity': self.elasticity.get(parent),
                    'relative_importance': self.relative_importance.get(parent),
                }

                # Estimator agreement check
                ates = [
                    v for v in [
                        ate_info.get('ate'),
                        ipw_info.get('ate_ipw'),
                        self.ate_gps_results.get(parent, {}).get('ate_gps'),
                        self.ate_matching_results.get(parent, {}).get('att_matching'),
                        dr_info.get('ate_dr'),
                    ]
                    if v is not None
                ]
                if len(ates) >= 2:
                    max_disc = max(abs(a - b) for a in ates for b in ates)
                    ref_ate = abs(ates[0]) if ates[0] != 0 else 1e-6
                    entry['estimator_agreement'] = (max_disc / ref_ate) < 0.30
                    entry['max_discrepancy_pct'] = float(max_disc / ref_ate * 100)

                coefficients[parent] = entry

        # Also include mediator propagation coefficients
        mediator_coeffs = {}
        for (parent, child), coeff in self.structural_coeffs.items():
            if child != target and child in (self.roles.get('mediators', [])):
                mediator_coeffs.setdefault(child, {})[parent] = coeff

        output = {
            'metadata': {
                'stage': 'causal_validation',
                'target': target,
                'n_treatments': len(self.roles.get('treatments', [])),
                'n_edges': len(self.structural_coeffs),
                'dag_file': self.config.get('causal', {}).get('dag_file', ''),
                'estimator': self.config.get('causal', {}).get('estimator', 'ols'),
                'ipw_enabled': bool(self.ate_ipw_results),
                'gps_enabled': bool(self.ate_gps_results),
                'matching_enabled': bool(self.ate_matching_results),
                'doubly_robust_enabled': bool(self.ate_dr_results),
                'spatial_cate_enabled': bool(self.spatial_cate_results),
            },
            'direct_effects': coefficients,
            'mediator_propagation': mediator_coeffs,
            'all_structural_coefficients': {
                f"{p}->{c}": v for (p, c), v in self.structural_coeffs.items()
            },
        }

        # Add optional sections
        if self.propensity_diagnostics:
            output['propensity_diagnostics'] = self.propensity_diagnostics

        if self.dag_diagnostics:
            output['assumption_diagnostics'] = self.dag_diagnostics

        if self.discovery_report:
            output['discovery_summary'] = {
                'shd': self.discovery_report.get('expert_comparison', {}).get('shd'),
                'edge_f1': self.discovery_report.get('expert_comparison', {}).get('f1'),
                'dsep_pass_rate': self.discovery_report.get('dseparation', {}).get('pass_rate'),
            }

        if hasattr(self, 'mediation_results') and self.mediation_results:
            output['mediation_decomposition'] = self.mediation_results

        if hasattr(self, 'dose_response_results') and self.dose_response_results:
            output['dose_response'] = {
                var: {
                    'nonlinearity_ratio': info.get('nonlinearity_ratio'),
                    'is_nonlinear': info.get('is_nonlinear'),
                }
                for var, info in self.dose_response_results.items()
            }

        os.makedirs(output_dir, exist_ok=True)

        # Use ResultStore if available, falling back to direct I/O
        try:
            from sparc.run.pipeline_paths import get_result_store
            store = get_result_store()
        except Exception:
            store = None

        out_path = os.path.join(output_dir, 'scenario_coefficients.json')
        if store is not None:
            store.save_json(3, 'scenario_coefficients.json', output)
        else:
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, default=str)

        try:
            from sparc.registry.run_registry import register_path
            register_path(out_path, stage="3", artifact_id="scenario_coefficients",
                          format="json", producer="causal_validation",
                          consumers=["server:/results/scenarios",
                                     "stage4:scenario_simulator"])
        except Exception:
            pass

        # Persist HGB edge models (if any) for non-linear propagation in Stage 4
        if self._edge_models:
            if store is not None:
                store.save_pickle(3, 'edge_models.joblib', self._edge_models)
            else:
                import joblib
                models_path = os.path.join(output_dir, 'edge_models.joblib')
                joblib.dump(self._edge_models, models_path)
            print(f"  HGB edge models saved ({len(self._edge_models)} edges)")

        # Save propensity diagnostics separately
        if self.propensity_diagnostics:
            if store is not None:
                store.save_json(3, 'propensity_diagnostics.json', self.propensity_diagnostics)
            else:
                ps_path = os.path.join(output_dir, 'propensity_diagnostics.json')
                with open(ps_path, 'w', encoding='utf-8') as f:
                    json.dump(self.propensity_diagnostics, f, indent=2, default=str)

        # Save spatial CATE multipliers for Stage 4
        if self.spatial_cate_results:
            sources = getattr(self, "_spatial_cate_sources", {})
            for treatment, cate_info in self.spatial_cate_results.items():
                mult = cate_info.get('multiplier')
                if mult is not None:
                    npy_name = f'spatial_cate_multiplier_{treatment}.npy'
                    art_meta = {
                        "artifact_id": f"cate_multiplier::{treatment}",
                        "variable": treatment,
                        "source": sources.get(treatment, "frequentist_dml"),
                        "consumers": [
                            "sparc.interventions.scenario_simulator",
                            "sparc.causal.counterfactual_engine",
                            "sparc.decision.optimizer",
                        ],
                    }
                    if art_meta["source"] == "uniform_fallback":
                        art_meta["fallback"] = True
                    if store is not None:
                        store.save_numpy(3, npy_name, mult, metadata=art_meta)
                    else:
                        mult_path = os.path.join(output_dir, npy_name)
                        np.save(mult_path, mult)

        print(f"\n  Scenario coefficients saved: {out_path}")
        print(f"    Direct effects on {target}: {len(coefficients)} variables")
        print(f"    Mediator propagation rules: {sum(len(v) for v in mediator_coeffs.values())} edges")

        return output


# ---------------------------------------------------------------------------
# Main entry point (called by sparc __main__.py, Stage 3)
# ---------------------------------------------------------------------------

def main(approval_gate=None) -> dict:
    """
    Run the full Stage 3 Causal Validation pipeline.

    Parameters
    ----------
    approval_gate : callable, optional
        If provided, called after MC³ with the result dict.  Should
        block until the user approves (or raise to abort).

    Returns
    -------
    dict
        The scenario_coefficients payload.
    """
    print("\n" + "=" * 60)
    print("  Stage 3: Causal Validation")
    print("=" * 60)

    config = load_config()
    paths = get_paths()

    # Determine output directory
    stage3_dir = str(paths.stage3_dir)
    os.makedirs(stage3_dir, exist_ok=True)

    # Check DAG file exists
    dag_file = config.get('causal', {}).get('dag_file')
    if not dag_file or not os.path.exists(dag_file):
        print(f"  WARNING: No causal DAG file found ({dag_file}).")
        print("  Skipping causal validation -- Stage 4 will use physics priors only.")
        return {}

    # Load data
    print("\n  Loading data...")
    data = load_and_preprocess_data(
        raw_data_path=config['paths']['raw_csv_path'],
        identifier_col=config['variables']['identifier'],
        target_col=config['variables']['target'],
        coords_cols=config['variables']['coordinates'],
        predictor_cols=config['predictors']['base_model'],
        initial_crs=config['crs']['initial'],
        target_crs=config['crs']['target_projected'],
    )
    print(f"  Data loaded: {len(data):,} rows, {len(data.columns)} columns")

    # Run validator
    validator = CausalValidator(config)

    validator.load_dag()

    # Check inference mode: "bayesian" skips V1 frequentist analyses
    causal_cfg = config.get('causal', {})
    use_bayesian = causal_cfg.get('inference', '').lower() == 'bayesian'

    # Phase 1: Causal discovery — always run (validates expert DAG)
    validator.run_causal_discovery(data, stage3_dir)

    # Phase 3: Assumption diagnostics (positivity, SUTVA) — always run
    validator.run_dag_diagnostics(data)

    if use_bayesian:
        # =============================================================
        # V2 Bayesian path: MC³ + NUTS — skip V1 DML/IPW/GPS/etc.
        # =============================================================
        print("\n  --- V2 Bayesian Causal Analysis (MC³ + NUTS) ---")
        print("  Skipping V1 frequentist estimators (DML, IPW, GPS, matching, DR)")

        # Minimal structural coefficient estimation (needed for scenario_coefficients.json skeleton)
        coord_cols = config.get('variables', {}).get('coordinates', [])
        coords = data[coord_cols].values if len(coord_cols) >= 2 and all(c in data.columns for c in coord_cols) else None
        validator.fit(data, coords=coords)

        # Produce scenario_coefficients.json (V1 skeleton — will be augmented by Bayesian results)
        result = validator.produce_scenario_coefficients(stage3_dir)

        try:
            from sparc.run.v2_bayesian_causal import run_bayesian_causal

            # Load neural model if available from Stage 2
            neural_model = None
            v2_neural_dir = paths.stage2_dir / "v2_neural"
            meta_info_path = v2_neural_dir / "meta_info.json"
            if meta_info_path.exists():
                try:
                    import torch
                    from sparc.models.neural_meta import SPARCMetaLearner
                    with open(meta_info_path) as _mf:
                        mi = json.load(_mf)
                    neural_model = SPARCMetaLearner(
                        n_base_models=mi["n_base_models"],
                        n_physics_features=mi.get("n_physics_extended", mi["n_physics_features"]),
                        d_spatial=mi["d_spatial"],
                        hidden_dim=mi["hidden_dim"],
                        thresholds=mi.get("thresholds", [0.25, 0.50, 0.75]),
                    )
                    state = torch.load(
                        v2_neural_dir / "neural_meta.pt",
                        map_location="cpu",
                        weights_only=True,
                    )
                    neural_model.load_state_dict(state)
                    neural_model.eval()
                    print(f"  Loaded V2 neural model from {v2_neural_dir}")
                except Exception as e:
                    print(f"  Could not load V2 neural model: {e}")
                    neural_model = None

            bayesian_dir = os.path.join(stage3_dir, "bayesian")
            bayesian_result = run_bayesian_causal(
                data=data,
                dag_def=validator.dag_def,
                neural_model=neural_model,
                config=config,
                output_dir=bayesian_dir,
                approval_gate=approval_gate,
            )

            mc3_summary = bayesian_result.get("mc3_summary", {})
            print(f"  MC³: {mc3_summary.get('n_accepted', 0)} / "
                  f"{mc3_summary.get('n_total', 0)} accepted "
                  f"(rate={mc3_summary.get('acceptance_rate', 0):.2%})")

            nuts_summary = bayesian_result.get("nuts_results")
            if nuts_summary:
                print(f"  NUTS: acceptance={nuts_summary.get('acceptance_rate', 0):.1%}, "
                      f"divergences={nuts_summary.get('n_divergences', 0)}")
                beta_mean = nuts_summary.get("beta_mean", [])
                treatments = nuts_summary.get("treatments", [])
                for t, b in zip(treatments, beta_mean):
                    print(f"    {t}: beta = {b:+.5f}")

            # Merge bayesian results into the scenario coefficients
            result["bayesian_causal"] = bayesian_result

            # Re-save scenario_coefficients.json with Bayesian data
            sc_path = os.path.join(stage3_dir, "scenario_coefficients.json")
            with open(sc_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"  Updated scenario_coefficients.json with Bayesian results")

            # ---------------------------------------------------------
            # Phase 5: spatial CATE is now ALWAYS estimated, including
            # in Bayesian-mode runs. Previously this was skipped, which
            # left the Budget Optimizer with no CATE variables and made
            # the scenario simulator fall back to a uniform multiplier.
            # The CausalForestDML estimator runs alongside the Bayesian
            # global ATE; per-cell heterogeneity comes from the forest
            # while the posterior mean serves as the global anchor.
            # ---------------------------------------------------------
            if causal_cfg.get('estimate_cate', True):
                try:
                    nuts_res = bayesian_result.get("nuts_results") or {}
                    posterior_means = dict(zip(
                        nuts_res.get("treatments", []) or [],
                        nuts_res.get("beta_mean", []) or [],
                    ))
                    validator.run_spatial_cate(
                        data,
                        stage3_dir,
                        artifact_source="bayesian_posterior_mean",
                        global_anchor=posterior_means,
                    )
                    # Re-emit scenario_coefficients.json + multipliers so
                    # the freshly-computed CATE arrays are persisted.
                    validator.produce_scenario_coefficients(stage3_dir)
                except Exception as e:
                    print(f"  Bayesian-mode spatial CATE failed: {e}")
                    import traceback
                    traceback.print_exc()

        except Exception as e:
            print(f"  V2 Bayesian causal analysis failed: {e}")
            import traceback
            traceback.print_exc()

    else:
        # =============================================================
        # V1 Frequentist path: DML/OLS/HGB + multi-estimator ATE
        # =============================================================

        # Phase 2: Structural coefficient estimation
        coord_cols = config.get('variables', {}).get('coordinates', [])
        coords = data[coord_cols].values if len(coord_cols) >= 2 and all(c in data.columns for c in coord_cols) else None
        validator.fit(data, coords=coords)

        # Phase 2b: DAG sensitivity analysis (Canopy <-> Impervious direction)
        validator.run_dag_sensitivity(data, coords=coords)

        # ATE estimation — backdoor (existing)
        validator.estimate_ate_all(data)

        # ATE estimation — IPW (new)
        validator.estimate_ate_ipw(data)

        # ATE estimation — GPS for continuous treatments
        validator.estimate_ate_gps(data)

        # ATE estimation — covariate matching
        validator.estimate_ate_matching(data)

        # ATE estimation — doubly-robust (new)
        validator.estimate_ate_dr(data)

        # Phase 4: Spatial CATE via CausalForestDML
        validator.run_spatial_cate(data, stage3_dir)

        # Phase 4b: Dose-response curves (non-linear treatment effects)
        validator.compute_dose_response(data, stage3_dir)

        # Phase 5: Elasticity, relative importance, bootstrap
        validator.compute_elasticity(data)
        validator.compute_relative_importance(data)
        validator.run_enhanced_bootstrap(data)

        # Phase 5b: Mediation decomposition with CI propagation
        validator.compute_mediation_effects(data)

        # Phase 5c: Location-stratified effects
        validator.compute_location_stratified_effects(data)

        # Refutation tests (existing + Cinelli-Hazlett)
        validator.run_refutations(data)

        # Phase 6: Causal evaluation diagnostics (cumulative curves, calibration)
        try:
            from sparc.evaluation.causal_evaluation import run_causal_diagnostics
            print("\n  Running causal evaluation diagnostics...")
            run_causal_diagnostics(validator, data, stage3_dir)
        except Exception as e:
            print(f"  Causal diagnostics skipped: {e}")

        # Phase 7: Physics prior diagnostic comparison table
        validator.build_coefficient_comparison_table()

        result = validator.produce_scenario_coefficients(stage3_dir)

    # Save a human-readable summary
    summary_path = os.path.join(stage3_dir, 'causal_validation_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("CAUSAL VALIDATION SUMMARY\n")
        f.write("=" * 50 + "\n\n")

        # Discovery report
        if validator.discovery_report:
            f.write("CAUSAL DISCOVERY\n")
            comp = validator.discovery_report.get('expert_comparison', {})
            f.write(f"  SHD: {comp.get('shd', 'N/A')}\n")
            f.write(f"  Edge precision: {comp.get('precision', 'N/A')}\n")
            f.write(f"  Edge recall:    {comp.get('recall', 'N/A')}\n")
            f.write(f"  Edge F1:        {comp.get('f1', 'N/A')}\n")
            dsep = validator.discovery_report.get('dseparation', {})
            if dsep:
                f.write(f"  d-sep pass rate: {dsep.get('pass_rate', 'N/A')}\n")
            f.write("\n")

        # Assumption diagnostics
        if validator.dag_diagnostics:
            f.write("ASSUMPTION DIAGNOSTICS\n")
            for var, diag in validator.dag_diagnostics.items():
                pos = diag.get('positivity', {})
                sutva = diag.get('sutva', {})
                f.write(f"  {var}:\n")
                f.write(f"    Positivity: violation_rate={pos.get('violation_rate', 0):.2%}, "
                        f"violations={len(pos.get('violations', []))}\n")
                if sutva:
                    f.write(f"    SUTVA (Moran's I): I={sutva.get('morans_i', 0):.4f}, "
                            f"p={sutva.get('p_value', 1):.4f}\n")
            f.write("\n")

        f.write("DAG STRUCTURE\n")
        f.write(f"  Treatments:  {validator.roles['treatments']}\n")
        f.write(f"  Mediators:   {validator.roles['mediators']}\n")
        f.write(f"  Confounders: {validator.roles['confounders']}\n")
        f.write(f"  Outcome:     {validator.roles['outcomes']}\n\n")

        f.write("STRUCTURAL COEFFICIENTS (parent -> child)\n")
        for (parent, child), coeff in validator.structural_coeffs.items():
            f.write(f"  {parent:>25s} -> {child:<15s}  {coeff:+.5f}\n")

        f.write("\nATE ESTIMATES\n")
        f.write(f"  {'Variable':>25s}  {'Backdoor':>10s}  {'IPW':>10s}  "
                f"{'GPS':>10s}  {'Match':>10s}  {'DR':>10s}  {'Agreement':>10s}\n")
        f.write(f"  {'-'*25}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}\n")
        for var in validator.roles.get('treatments', []):
            bd = validator.ate_results.get(var, {}).get('ate')
            ipw = validator.ate_ipw_results.get(var, {}).get('ate_ipw')
            gps = validator.ate_gps_results.get(var, {}).get('ate_gps')
            match = validator.ate_matching_results.get(var, {}).get('att_matching')
            dr = validator.ate_dr_results.get(var, {}).get('ate_dr')
            bd_s = f"{bd:+.5f}" if bd is not None else "N/A"
            ipw_s = f"{ipw:+.5f}" if ipw is not None else "N/A"
            gps_s = f"{gps:+.5f}" if gps is not None else "N/A"
            match_s = f"{match:+.5f}" if match is not None else "N/A"
            dr_s = f"{dr:+.5f}" if dr is not None else "N/A"

            # Check agreement across all available estimators
            ates = [v for v in [bd, ipw, gps, match, dr] if v is not None]
            if len(ates) >= 2 and ates[0] != 0:
                max_d = max(abs(a - b) for a in ates for b in ates)
                agree = "YES" if (max_d / abs(ates[0])) < 0.30 else "NO"
            else:
                agree = "N/A"

            f.write(f"  {var:>25s}  {bd_s:>10s}  {ipw_s:>10s}  "
                    f"{gps_s:>10s}  {match_s:>10s}  {dr_s:>10s}  {agree:>10s}\n")

        # CATE summary
        if any(validator.ate_results.get(v, {}).get('cate_mean') is not None
               for v in validator.roles.get('treatments', [])):
            f.write("\nCATE (LinearDML)\n")
            for var in validator.roles.get('treatments', []):
                info = validator.ate_results.get(var, {})
                if info.get('cate_mean') is not None:
                    f.write(f"  {var:>25s}  mean={info['cate_mean']:+.5f}  "
                            f"std={info['cate_std']:.5f}\n")

        # Spatial CATE
        if validator.spatial_cate_results:
            f.write("\nSPATIAL CATE (CausalForestDML)\n")
            for var, info in validator.spatial_cate_results.items():
                cate_arr = info.get('cate')
                if cate_arr is not None:
                    f.write(f"  {var:>25s}  mean={np.mean(cate_arr):+.5f}  "
                            f"std={np.std(cate_arr):.5f}\n")

        # Elasticity
        if validator.elasticity:
            f.write("\nELASTICITY COEFFICIENTS\n")
            for var, eps in validator.elasticity.items():
                f.write(f"  {var:>25s}  eps = {eps:+.4f}\n")

        # Relative importance
        if validator.relative_importance:
            f.write("\nRELATIVE IMPORTANCE\n")
            for var, ri in sorted(
                validator.relative_importance.items(),
                key=lambda x: x[1], reverse=True,
            ):
                f.write(f"  {var:>25s}  RI = {ri:.3f}\n")

        # Bootstrap CIs
        if validator.bootstrap_cis:
            f.write("\nBOOTSTRAP 95% CONFIDENCE INTERVALS\n")
            for var, ci in validator.bootstrap_cis.items():
                f.write(f"  {var:>25s}  [{ci['ci_lower']:+.5f}, {ci['ci_upper']:+.5f}]  "
                        f"SE={ci['se']:.5f}\n")

        f.write("\nREFUTATION TESTS\n")
        for var, info in validator.refutation_results.items():
            test_keys = [
                ('placebo_pass', 'placebo'),
                ('random_common_cause_pass', 'rcc'),
                ('data_subset_pass', 'subset'),
                ('unobserved_common_cause_pass', 'unobs_cc'),
            ]
            parts = []
            for key, label in test_keys:
                val = info.get(key)
                parts.append(f"{label}={'PASS' if val else 'FAIL' if val is not None else 'N/A'}")
            e_val = info.get('e_value')
            e_str = f"e-value={e_val:.2f}" if e_val is not None else "e-value=N/A"

            # Cinelli-Hazlett robustness
            rv = info.get('cinelli_hazlett', {})
            rv_str = ""
            if rv:
                rv_str = f", RV={rv.get('robustness_value', 0):.3f}"

            f.write(f"  {var:>25s}  {', '.join(parts)}, {e_str}{rv_str}\n")

    # Register summary in ResultStore manifest
    try:
        from sparc.run.pipeline_paths import get_result_store
        store = get_result_store()
        # Re-read the text we just wrote so it appears in the manifest
        with open(summary_path, 'r', encoding='utf-8') as _sf:
            store.save_text(3, 'causal_validation_summary.txt', _sf.read())
    except Exception:
        pass  # Direct file was already written above

    print(f"  Summary saved: {summary_path}")
    print("\n  Stage 3 complete.\n")

    return result


if __name__ == "__main__":
    main()
