"""
ScenarioSimulator — Config-driven physics-constrained scenario predictor.

This module replaces the legacy ``physics_constrained_scenarios_v2.py`` with a
class that reads **all** hard-coded values (paths, features, district names,
physics coefficients, increments, CRS, etc.) from the unified project config
produced by ``config.config.load_config()``.

Spatial heterogeneity approach (preserved from v2):
  - Compute a spatial multiplier from baseline temperature z-scores and
    inter-model prediction disagreement.
  - Constrain multiplier to preserve the physics sign.
  - Results: physics-correct deltas that vary spatially (0.5× to 1.5×).

Usage::

    from sparc.interventions.scenario_simulator import ScenarioSimulator

    sim = ScenarioSimulator(config)
    sim.load_models()
    summary_df, results_gdf = sim.run()
    summary_df.to_csv("scenario_summary.csv")
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Ensure stdout can handle Unicode on Windows (cp1252 console).
if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Optional imports — gracefully degrade
try:
    import geopandas as gpd
    from shapely.geometry import Point

    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

try:
    from sparc.interventions.extrapolation_guard import (
        compute_extrapolation_score,
        classify_prediction_confidence,
    )

    EXTRAPOLATION_GUARD_AVAILABLE = True
except ImportError:
    EXTRAPOLATION_GUARD_AVAILABLE = False


class ScenarioSimulator:
    """
    Config-driven physics-constrained scenario predictor.

    Parameters
    ----------
    config : dict
        Full pipeline config dict (from ``load_config()``).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, config: dict):
        self.config = config

        # Paths
        self.data_path = Path(config["paths"]["raw_csv_path"])
        self.model_dir = Path(config["output"]["base_dir"]) / config["output"]["stage_dirs"].get("stage_2", "Stage_2_Spatial_CV")
        self.output_dir = Path(config["output"]["base_dir"]) / config["output"]["stage_dirs"].get("stage_4", "Stage_4_Scenarios")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Features / coords
        self.features: List[str] = config["predictors"]["base_model"]
        self.coord_cols: List[str] = config["variables"]["coordinates"]
        self.target_col: str = config["variables"]["target"]

        # CRS
        self.model_crs: str = config.get("crs", {}).get("target_projected", "EPSG:4326")

        # District / area-of-interest mapping (optional)
        self.area_column: str = config.get("data", {}).get("area_column", "Location")
        self.area_names: Dict[int, str] = {
            int(k): v
            for k, v in config.get("data", {}).get("area_names", {}).items()
        }

        # Physics priors
        physics = config.get("physics", {})
        self.literature_weight: float = physics.get("literature_weight", 0.5)

        # Load caps (constraints) into config for easy access
        from sparc.config.config import load_caps
        self.config['caps'] = load_caps(config)

        # Build per-variable priors from config
        self._build_physics_priors(physics)

        # Scenarios (from project.yml)
        self.scenarios: List[dict] = config.get("scenarios", [])
        self.interaction_scenarios: List[dict] = config.get("interaction_scenarios", [])

        # Models (populated by load_models())
        self._models: Dict[str, Any] = {}
        self._meta_model: Any = None
        self._feature_scaler: Any = None  # loaded from feature_scaler.pkl

        # MGWR local coefficients (populated by load_models())
        self._mgwr_coefficients_raw: Optional[np.ndarray] = None   # (n_points, n_features) raw-unit
        self._mgwr_scaler_scale: Optional[np.ndarray] = None       # σ per feature (for scale-invariant check)
        self._mgwr_feature_map: Dict[str, int] = {}                # feature_name → column index

        # Causal coefficients (populated by load_causal_coefficients())
        self._causal_coefficients: Optional[Dict[str, Any]] = None

        # GWRF condition curves — aggregate PDP + saturation fits (populated
        # by _load_condition_curves() during load_models())
        self._condition_curves: Dict[str, dict] = {}
        self._condition_curve_min_r2: float = 0.5  # quality gate for saturation path

        # Base-model consensus weights (populated by _compute_base_model_weights())
        self._base_model_weights: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Physics priors
    # ------------------------------------------------------------------

    def _build_physics_priors(self, physics: dict) -> None:
        """
        Construct ``self.physics_priors`` from the ``priors.yml`` file.

        The YAML schema is::

            coefficients:
              Pct_Canopy:
                value: -0.280
                units: "°F per +10 pp"
                uncertainty: 0.20
                ...

        These literature values are used for sign checking and as
        regularisation guardrails.  The ``value`` field is stored as
        ``lit_coef``; the ``ols_coef`` is left at 0 unless supplied
        separately in the config.
        """
        from sparc.config.config import load_physics_priors

        priors_data = load_physics_priors(self.config)

        # priors.yml stores literature coefficients under 'coefficients'
        coefficients_section = priors_data.get("coefficients", {})

        # Also accept legacy flat dicts if present
        ols_coeffs = priors_data.get("ols_coefficients", physics.get("ols_coefficients", {}))

        self.physics_priors: Dict[str, dict] = {}
        all_vars = set(list(coefficients_section.keys()) + list(ols_coeffs.keys()))

        for var in all_vars:
            # Literature coefficient from priors.yml nested structure
            lit_entry = coefficients_section.get(var, {})
            if isinstance(lit_entry, dict):
                lit_c = lit_entry.get("value", 0.0)
            else:
                lit_c = float(lit_entry)

            ols_c = ols_coeffs.get(var, 0.0)
            # When no OLS prior exists (ols_c == 0 and var not in the
            # ols_coefficients dict), use the literature value directly
            # instead of blending with zero (which would halve it).
            if ols_c == 0.0 and var not in ols_coeffs:
                blended = lit_c
            else:
                blended = (1 - self.literature_weight) * ols_c + self.literature_weight * lit_c
            direction = "cooling" if blended < 0 else "warming"

            # Infer unit increment from the units string or defaults
            if "Pct" in var or "pct" in var:
                unit_inc = 10  # literature reports per +10 pp
            elif var in ("NDVI", "Albedo"):
                unit_inc = 0.1  # literature reports per +0.1
            elif isinstance(lit_entry, dict) and "unit_increment" in lit_entry:
                unit_inc = float(lit_entry["unit_increment"])
            else:
                unit_inc = priors_data.get("unit_increments", {}).get(var, 1.0)

            self.physics_priors[var] = {
                "coefficient": blended,       # literature value per unit_increment
                "unit_increment": unit_inc,
                "direction": direction,
                "ols_coef": ols_c,
                "lit_coef": lit_c,
                "uncertainty": lit_entry.get("uncertainty", 0.2) if isinstance(lit_entry, dict) else 0.2,
            }

    # ------------------------------------------------------------------
    # MGWR coefficient conversion
    # ------------------------------------------------------------------

    def _extract_mgwr_coefficients(self) -> None:
        """
        Extract per-point MGWR local coefficients from the fitted GWR model
        and convert from scaled-feature units to raw-feature units.

        After this call, ``self._mgwr_coefficients_raw[i, j]`` gives
        ``∂AAT_z / ∂X_j_raw`` at point *i*, where X_j_raw is the j-th
        feature in its original (unscaled) units.
        """
        gwr_model = self._models.get("gwr")
        if gwr_model is None or not hasattr(gwr_model, "coefficients_"):
            print("   Warning: GWR model has no local coefficients — MGWR spatial blend disabled")
            return

        scaled_coeffs = gwr_model.coefficients_                     # (n_points, n_features) — scaled units
        scaler = getattr(gwr_model, "scaler", None)
        feature_names = getattr(gwr_model, "feature_names_", None)  # may be generic

        if scaler is None or not hasattr(scaler, "scale_"):
            print("   Warning: GWR model has no scaler — cannot convert coefficients to raw units")
            return

        # Convert: raw_coeff = scaled_coeff / std_j
        # Because the model was fit on (X – μ) / σ, so ∂y/∂X_raw = β_scaled / σ
        raw_coeffs = scaled_coeffs / scaler.scale_[np.newaxis, :]  # broadcast (n, p) / (p,)
        self._mgwr_coefficients_raw = raw_coeffs
        self._mgwr_scaler_scale = scaler.scale_  # keep σ for scale-invariant reliability check

        # Build name→index map.  feature_names_ may be generic (feature_0 etc.)
        # so we rely on the feature order from feature_info.json / self.features.
        for j, feat in enumerate(self.features):
            self._mgwr_feature_map[feat] = j

        n_pts = raw_coeffs.shape[0]
        print(f"   MGWR local coefficients extracted: {n_pts} points × {raw_coeffs.shape[1]} features")
        for feat in self.features:
            j = self._mgwr_feature_map[feat]
            rc = raw_coeffs[:, j]
            nz = rc != 0
            nz_count = int(nz.sum())
            mean_abs_nz = float(np.mean(np.abs(rc[nz]))) if nz.any() else 0.0
            scaled_rel = mean_abs_nz * scaler.scale_[j]
            print(f"      {feat:30s}  mean={rc.mean():+.6f}  nonzero={nz_count}/{n_pts}  reliability={scaled_rel:.4f}  (dz/draw)")

    # ------------------------------------------------------------------
    # Diminishing returns
    # ------------------------------------------------------------------

    @staticmethod
    def _diminishing_return(delta: np.ndarray, threshold: float) -> np.ndarray:
        """
        Apply diminishing-return scaling for feature changes beyond
        *threshold*.

        Within ``[-threshold, +threshold]`` the mapping is linear.
        Beyond that, a square-root taper reduces the effective magnitude::

            effective = sign(Δ) × [ threshold + √(|Δ| - threshold) × √threshold ]

        This ensures the first ``threshold`` units of change have full
        effect, but each additional unit has progressively less impact —
        capturing the empirical observation that interventions have
        diminishing marginal returns.
        """
        sign = np.sign(delta)
        abs_d = np.abs(delta)
        effective = np.where(
            abs_d <= threshold,
            abs_d,
            threshold + np.sqrt(np.maximum(abs_d - threshold, 0.0)) * np.sqrt(threshold),
        )
        return sign * effective

    def _get_diminishing_threshold(self, variable: str) -> float:
        """Return the per-variable diminishing-return threshold.

        Reads from ``caps.yml → diminishing_return_thresholds`` when
        available; otherwise falls back to hardcoded defaults for
        backward compatibility.
        """
        # Try config-driven thresholds first
        caps = self.config.get('caps', {})
        cfg_thresholds = caps.get('diminishing_return_thresholds', {})
        if variable in cfg_thresholds:
            return float(cfg_thresholds[variable])
        if 'default' in cfg_thresholds:
            return float(cfg_thresholds['default'])

        # Legacy hardcoded fallback
        _DEFAULTS = {
            "Pct_Canopy": 10.0,
            "Pct_Impervious": 10.0,
            "NDVI": 0.10,
            "Albedo": 0.10,
            "Elevation_m": 5.0,
            "Distance_from_water_m": 100.0,
        }
        return _DEFAULTS.get(variable, 10.0)

    # ------------------------------------------------------------------
    # MGWR-based delta computation
    # ------------------------------------------------------------------

    def _compute_mgwr_direct_delta(
        self, variable: str, effective_change: np.ndarray,
        baseline_values: Optional[np.ndarray] = None,
        modified_values: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, str]:
        """
        Per-point direct temperature delta using a four-tier strategy:

        1. **MGWR local coefficients** — if the GWR model produced
           reliable spatially-varying coefficients (mean |β| ≥
           ``mgwr_reliability_threshold``), use them directly for full
           spatial heterogeneity.
        2. **Saturation curve** — if MGWR is unreliable or unavailable
           but a GWRF condition curve with R² ≥
           ``_condition_curve_min_r2`` exists, use the non-linear PDP
           response.
        3. **MGWR blend** — if MGWR has weak signal and no saturation
           curve is available, blend MGWR with physics literature.
        4. **Physics literature** — pure fallback.

        Returns
        -------
        (delta, method) : (ndarray, str)
            ``delta`` — per-point temperature change.
            ``method`` — ``'mgwr'``, ``'saturation_curve'``,
            ``'mgwr_blend'``, or ``'physics_lit'``.
        """
        # Read reliability threshold from caps.yml (with hardcoded fallback)
        caps = self.config.get('caps', {})
        RELIABILITY_THRESHOLD = float(
            caps.get('mgwr_reliability_threshold', 0.01)
        )

        # --- Physics literature coefficient per raw unit ---------------
        prior = self.physics_priors.get(variable)
        if prior and prior.get("unit_increment", 1.0) != 0:
            lit_per_unit = prior["lit_coef"] / prior["unit_increment"]
        else:
            lit_per_unit = 0.0

        # --- Tier 1: MGWR local coefficients (spatial heterogeneity) ---
        j = self._mgwr_feature_map.get(variable)
        has_mgwr = (
            j is not None
            and self._mgwr_coefficients_raw is not None
        )

        if has_mgwr:
            local_coeff = self._mgwr_coefficients_raw[:, j]  # (n_points,)
            n = min(len(effective_change), len(local_coeff))
            local_coeff = local_coeff[:n]

            # Scale-invariant reliability: convert raw mean|β| back to
            # per-standard-deviation space so the threshold is comparable
            # across variables with very different raw units (e.g.
            # Pct_Canopy in 0–100 vs Albedo in 0–1).
            scaler_std = (
                self._mgwr_scaler_scale[j]
                if hasattr(self, '_mgwr_scaler_scale')
                   and self._mgwr_scaler_scale is not None
                else 1.0
            )
            # Exclude exact-zero coefficients from the mean — zeros from
            # sign-constraint projection are not evidence of no effect.
            nonzero = local_coeff != 0
            if nonzero.any():
                mean_abs_raw = float(np.mean(np.abs(local_coeff[nonzero])))
            else:
                mean_abs_raw = 0.0
            mean_abs_scaled = mean_abs_raw * scaler_std  # ≈ per-σ effect

            if mean_abs_scaled >= RELIABILITY_THRESHOLD:
                # MGWR has signal — use local coefficients, but anchor
                # magnitude to the DML structural coefficient when available.
                # MGWR bandwidth smoothing can attenuate the coefficient
                # magnitude (e.g. Canopy MGWR=-0.0007 vs DML=-0.022) while
                # the *spatial pattern* (where effects are stronger/weaker)
                # remains informative.  Strategy: rescale so the mean |β|
                # matches the DML estimate, preserving relative variation.
                causal_coef = self.get_causal_coefficient(variable)
                if causal_coef is not None and mean_abs_raw > 1e-12:
                    causal_abs = abs(causal_coef)
                    attenuation = causal_abs / mean_abs_raw
                    # Only rescale when MGWR is substantially attenuated
                    # (< 50% of DML magnitude); otherwise trust MGWR.
                    if attenuation > 2.0:
                        zero_mask = local_coeff == 0
                        local_coeff = local_coeff * attenuation
                        # Infill zero-coefficient cells with the mean of
                        # the rescaled non-zero coefficients.  This keeps
                        # the infill in the same MGWR-spatial / DML-anchored
                        # frame rather than substituting a separate estimate.
                        if zero_mask.any() and (~zero_mask).any():
                            local_coeff[zero_mask] = float(np.mean(local_coeff[~zero_mask]))
                        return local_coeff * effective_change[:n], 'mgwr_causal_anchored'
                return local_coeff * effective_change[:n], 'mgwr'

        # --- Tier 2: Saturation curve (GWRF condition curve) -----------
        curve = self._condition_curves.get(variable)
        if (
            curve is not None
            and curve['r2'] >= self._condition_curve_min_r2
            and baseline_values is not None
            and modified_values is not None
        ):
            try:
                from sparc.interventions.extrapolation_guard import (
                    predict_scenario_with_saturation,
                )
                condition_curve_dict = {
                    'grid_values': curve['grid_values'],
                    'pdp_values': curve['pdp_values'],
                }
                delta = predict_scenario_with_saturation(
                    variable_name=variable,
                    baseline_values=baseline_values,
                    modified_values=modified_values,
                    condition_curve=condition_curve_dict,
                    spatial_multiplier=None,
                )
                return delta, 'saturation_curve'
            except Exception as e:
                warnings.warn(
                    f"Saturation curve failed for {variable}: {e}; "
                    f"falling back to coefficient extrapolation."
                )

        # --- Tier 3: MGWR unreliable — blend with physics literature ---
        if has_mgwr:
            alpha = mean_abs_scaled / RELIABILITY_THRESHOLD       # 0 → 1
            blended = alpha * local_coeff + (1.0 - alpha) * lit_per_unit
            return blended * effective_change[:n], 'mgwr_blend'

        # --- Tier 4: No MGWR — pure physics literature fallback --------
        return lit_per_unit * effective_change, 'physics_lit'

    def _compute_indirect_effects(
        self,
        treatment: str,
        effective_change: np.ndarray,
        engine: Any,
        target: str,
        skip_variables: Optional[set] = None,
    ) -> np.ndarray:
        """
        Compute indirect temperature effects through DAG mediator pathways.

        For a path ``treatment → m₁ → … → mₖ → outcome``:
        - Intermediate edges use the CounterfactualEngine's fitted
          structural coefficients (global, estimated by OLS + backdoor).
        - The **final** edge ``mₖ → outcome`` uses the per-point MGWR
          local coefficient, so the indirect effect inherits full spatial
          heterogeneity.

        This avoids double-counting: the direct ``treatment → outcome``
        effect is handled separately by ``_compute_mgwr_direct_delta()``.

        Parameters
        ----------
        skip_variables : set, optional
            Variables whose direct/indirect effects are accounted for
            elsewhere (e.g. co-intervened variables in a joint scenario).
            Any mediator path passing through one of these nodes is
            skipped to prevent double-counting.
        """
        import networkx as nx

        n = len(effective_change)
        indirect_delta = np.zeros(n)

        graph = getattr(engine, "graph", None)
        if graph is None:
            return indirect_delta

        # Find all simple paths of length > 2 (i.e. at least one mediator)
        try:
            all_paths = list(nx.all_simple_paths(graph, treatment, target))
        except (nx.NetworkXError, nx.NodeNotFound):
            return indirect_delta

        for path in all_paths:
            if len(path) <= 2:
                continue  # direct edge, handled elsewhere

            # Skip paths that pass through co-intervened variables to
            # avoid double-counting their direct/indirect contributions.
            if skip_variables:
                intermediates = set(path[1:-1])  # exclude treatment & outcome
                if intermediates & skip_variables:
                    continue

            # Propagate through intermediate edges using structural coefficients
            current_delta = effective_change.copy()
            for step in range(len(path) - 2):
                parent, child = path[step], path[step + 1]
                coeff = engine._structural_coeffs.get((parent, child), 0.0)
                current_delta = current_delta * coeff

            # Final edge (last-mediator → outcome): use MGWR local coefficient
            final_mediator = path[-2]
            j = self._mgwr_feature_map.get(final_mediator)
            if j is not None and self._mgwr_coefficients_raw is not None:
                local_coeff = self._mgwr_coefficients_raw[:, j]
                if len(current_delta) < len(local_coeff):
                    indirect_delta += local_coeff[:len(current_delta)] * current_delta
                else:
                    indirect_delta += local_coeff * current_delta
            else:
                # Fallback: use DAG structural coefficient for final edge
                fallback_coeff = engine._structural_coeffs.get(
                    (final_mediator, target), 0.0
                )
                indirect_delta += current_delta * fallback_coeff

        return indirect_delta

    def _compute_temporal_lagged_effects(
        self,
        treatment: str,
        effective_change: np.ndarray,
        engine: Any,
        target: str,
    ) -> Dict[int, np.ndarray]:
        """
        Compute time-lagged effects from temporal edges in the DAG.

        For each temporal edge ``treatment_t-{lag} → target`` (or paths
        through mediators), estimate the delayed effect that would manifest
        ``lag`` time steps after the intervention.

        Parameters
        ----------
        treatment : str
            The intervened variable name (unlagged).
        effective_change : np.ndarray
            Per-point magnitude of the intervention.
        engine : CounterfactualEngine
            Fitted causal engine with structural coefficients.
        target : str
            Outcome variable name.

        Returns
        -------
        dict[int, np.ndarray]
            ``{lag: delta_array}`` — delayed effects keyed by time lag.
        """
        import networkx as nx

        n = len(effective_change)
        lagged_effects: Dict[int, np.ndarray] = {}

        graph = getattr(engine, 'graph', None)
        if graph is None:
            return lagged_effects

        # Find all lagged nodes derived from this treatment
        lagged_nodes = [
            node for node, attrs in graph.nodes(data=True)
            if attrs.get('temporal_source') == treatment
               and attrs.get('temporal_lag') is not None
        ]

        for lagged_node in lagged_nodes:
            lag = graph.nodes[lagged_node].get('temporal_lag', 1)

            # Find paths from the lagged node to the target
            try:
                paths = list(nx.all_simple_paths(graph, lagged_node, target))
            except (nx.NetworkXError, nx.NodeNotFound):
                continue

            lag_delta = np.zeros(n)
            for path in paths:
                current_delta = effective_change.copy()

                # Propagate through path edges
                for step in range(len(path) - 1):
                    parent, child = path[step], path[step + 1]
                    # Check for edge-specific lag coefficient
                    edge_data = graph.edges.get((parent, child), {})

                    # Use MGWR local coefficient for the final edge if available
                    if step == len(path) - 2:
                        j = self._mgwr_feature_map.get(child)
                        if child == target and j is None:
                            # Final edge targets outcome: use the mediator's
                            # column in MGWR coefficients
                            j = self._mgwr_feature_map.get(parent)

                        # Try MGWR spatial coefficients
                        if j is not None and self._mgwr_coefficients_raw is not None:
                            local_coeff = self._mgwr_coefficients_raw[:, j]
                            current_delta = local_coeff[:n] * current_delta[:n]
                        else:
                            coeff = engine._structural_coeffs.get(
                                (parent, child), 0.0
                            )
                            current_delta = current_delta * coeff
                    else:
                        coeff = engine._structural_coeffs.get(
                            (parent, child), 0.0
                        )
                        current_delta = current_delta * coeff

                lag_delta += current_delta[:n]

            if np.any(lag_delta != 0):
                if lag in lagged_effects:
                    lagged_effects[lag] += lag_delta
                else:
                    lagged_effects[lag] = lag_delta

        return lagged_effects

    def _apply_physics_guardrails(
        self, variable: str, delta: np.ndarray,
        change_direction: float = 1.0,
    ) -> np.ndarray:
        """
        Apply physics-based sign checking and magnitude capping.

        Parameters
        ----------
        variable : str
            Name of the intervention variable.
        delta : ndarray
            Per-point predicted temperature delta.
        change_direction : float
            ``+1.0`` if the intervention *increases* the variable,
            ``-1.0`` if it *decreases* it.

        Strategy is read from ``caps.yml → sign_violation_strategy``:

        * ``"dampen"`` (default) — multiply wrong-sign cells by
          ``sign_violation_dampen_factor`` (default 0.50).
        * ``"zero"`` — zero-out wrong-sign cells.
        * ``"keep"`` — trust the model, no sign correction.
        """
        caps = self.config.get('caps', {})
        MAX_Z_DELTA = float(caps.get('max_delta_magnitude', 3.0))

        caps = self.config.get('caps', {})
        strategy = caps.get('sign_violation_strategy', 'dampen')
        dampen_factor = float(caps.get('sign_violation_dampen_factor', 0.50))

        prior = self.physics_priors.get(variable)
        if prior is not None and strategy != 'keep':
            physics_sign = -1.0 if prior["direction"] == "cooling" else 1.0
            expected_sign = physics_sign * change_direction
            wrong_sign_mask = np.sign(delta) != expected_sign
            wrong_sign_mask &= (np.abs(delta) > 0.01)

            if strategy == 'zero':
                delta = np.where(wrong_sign_mask, 0.0, delta)
            else:  # 'dampen' (default)
                delta = np.where(wrong_sign_mask, delta * dampen_factor, delta)

        return np.clip(delta, -MAX_Z_DELTA, MAX_Z_DELTA)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_models(self) -> None:
        """Load serialised base models and meta-ensemble from ``model_dir``."""
        print("="*70)
        print("PHYSICS-CONSTRAINED SCENARIO PREDICTOR")
        print("With Extended Increments + Spatial Heterogeneity")
        print("="*70)

        print("\n1. Loading models...")
        model_files = {
            "meta": self.model_dir / "standard_meta_ensemble.pkl",
            "ols": self.model_dir / "base_models_full" / "ols_model_full.pkl",
            "gwr": self.model_dir / "base_models_full" / "gwr_model_full.pkl",
            "gwrf": self.model_dir / "base_models_full" / "gwrf_model_full.pkl",
            "ggpgam": self.model_dir / "base_models_full" / "ggpgam_model_full.pkl",
        }

        for name, path in model_files.items():
            if not path.exists():
                raise FileNotFoundError(f"Model file not found: {path}")
            if name == "meta":
                self._meta_model = joblib.load(path)
            else:
                self._models[name] = joblib.load(path)
        print("   All models loaded")

        # Extract MGWR local coefficients from the GWR model
        self._extract_mgwr_coefficients()

        # Load feature scaler (saved by enhanced_spatial_cv)
        scaler_path = self.model_dir / "feature_scaler.pkl"
        if scaler_path.exists():
            self._feature_scaler = joblib.load(scaler_path)
            print(f"   Feature scaler loaded from {scaler_path}")
        else:
            print("   Warning: feature_scaler.pkl not found — scenario features will NOT be scaled")

        # Load causal coefficients (from Stage 3, if available)
        self.load_causal_coefficients()

        # Load GWRF condition curves (from Stage 2 spatial intelligence)
        self._load_condition_curves()

        # Derive base-model consensus weights from meta-ensemble importance
        self._compute_base_model_weights()

    # ------------------------------------------------------------------
    # Condition curve loading (PDP saturation fits from Stage 2)
    # ------------------------------------------------------------------

    def _load_condition_curves(self) -> None:
        """Load GWRF aggregate PDP + saturation curves for scenario use.

        The JSON file is produced by Stage 2b when
        ``gwrf_model.export_condition_curves()`` runs.  Each entry
        contains ``pdp.grid_values``, ``pdp.pdp_values``, and
        ``curve_fit`` metadata used by
        :func:`extrapolation_guard.predict_scenario_with_saturation`.
        """
        import json

        # Canonical location: Stage_2_Spatial_CV/gwrf_pdp/
        base_dir = Path(self.config['output']['base_dir'])
        stage2_name = self.config.get('output', {}).get('stage_dirs', {}).get(
            'stage_2', 'Stage_2_Spatial_CV'
        )
        curves_path = base_dir / stage2_name / 'gwrf_pdp' / 'gwrf_condition_curves.json'

        # Fallback: legacy spatial_intelligence location
        if not curves_path.exists():
            curves_path = base_dir / 'spatial_intelligence' / 'gwrf_pdp' / 'gwrf_condition_curves.json'

        # Fallback: legacy Stage 3 location
        if not curves_path.exists():
            stage3_name = self.config.get('output', {}).get('stage_dirs', {}).get(
                'stage_3', 'Stage_3_Causal_Validation'
            )
            curves_path = base_dir / stage3_name / 'gwrf_condition_curves.json'

        if not curves_path.exists():
            print("   No GWRF condition curves found — Stage 4 will use coefficient extrapolation.")
            return

        with open(curves_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        loaded = 0
        skipped = 0
        for var_name, entry in raw.items():
            # Validate the entry has the data we need
            pdp = entry.get('pdp', {})
            curve_fit = entry.get('curve_fit', {})
            grid_vals = pdp.get('grid_values')
            pdp_vals = pdp.get('pdp_values')
            r2 = curve_fit.get('r2', 0.0)

            if grid_vals is None or pdp_vals is None:
                skipped += 1
                continue

            self._condition_curves[var_name] = {
                'grid_values': grid_vals,
                'pdp_values': pdp_vals,
                'pdp_std': pdp.get('pdp_std'),
                'curve_fit': curve_fit,
                'r2': r2,
            }
            loaded += 1

        usable = sum(
            1 for c in self._condition_curves.values()
            if c['r2'] >= self._condition_curve_min_r2
        )
        print(f"   Condition curves loaded: {loaded} variables "
              f"({usable} usable with R\u00b2 >= {self._condition_curve_min_r2}, "
              f"{skipped} skipped)")

    # ------------------------------------------------------------------
    # Base-model consensus weights (from meta-ensemble feature importance)
    # ------------------------------------------------------------------

    def _compute_base_model_weights(self) -> None:
        """Extract per-base-model weights from the meta-ensemble LightGBM
        feature importances.  These weights are used by the consensus
        delta approach: each base model's individual delta is weighted
        and summed rather than passing perturbed features through the
        full stacker (which is dominated by spatial features).

        ⚠ IMPORTANT: These are *predictive* weights (LightGBM gain
        importance), NOT *causal* weights.  GWRF dominance (~79%) reflects
        predictive accuracy, not causal attribution.  For causal
        inference, use Mode 3 (``run_with_causal_dag``) or the DML-derived
        coefficients from Stage 3 instead.

        Weights are the *gain* importances of ols_pred, gwr_pred,
        gwrf_pred, ggpgam_pred, normalised so they sum to 1.0 with a
        minimum floor of 0.05 per model.
        """
        MODEL_KEYS = ("ols", "gwr", "gwrf", "ggpgam")
        PRED_FEATURES = {k: f"{k}_pred" for k in MODEL_KEYS}
        FLOOR = 0.05

        try:
            imp_df = self._meta_model.get_feature_importance()
            imp_map = dict(zip(imp_df['feature'], imp_df['importance']))

            raw = {k: imp_map.get(feat, 0.0) for k, feat in PRED_FEATURES.items()}
            total = sum(raw.values())

            if total <= 0:
                # Fallback: equal weights
                self._base_model_weights = {k: 1.0 / len(MODEL_KEYS) for k in MODEL_KEYS}
            else:
                normed = {k: v / total for k, v in raw.items()}
                # Apply floor then re-normalise
                floored = {k: max(v, FLOOR) for k, v in normed.items()}
                ftot = sum(floored.values())
                self._base_model_weights = {k: v / ftot for k, v in floored.items()}

            wstr = ", ".join(f"{k}={v:.3f}" for k, v in self._base_model_weights.items())
            print(f"   Base-model consensus weights (PREDICTIVE, not causal): {wstr}")
            print(f"   ⚠ These weights reflect predictive power, not causal attribution.")
            print(f"     For causal inference, prefer Mode 3 (run_with_causal_dag).")

        except Exception as e:
            warnings.warn(f"Could not compute base-model weights ({e}); using equal weights.")
            self._base_model_weights = {k: 1.0 / len(MODEL_KEYS) for k in MODEL_KEYS}

    # ------------------------------------------------------------------
    # Causal coefficient loading (from Stage 3)
    # ------------------------------------------------------------------

    def load_causal_coefficients(self) -> None:
        """
        Load ``scenario_coefficients.json`` produced by Stage 3 (Causal
        Validation).  If found, the DAG-adjusted coefficients are stored
        and used by ``run_with_causal_dag()`` to override or blend with
        physics priors.
        """
        stage_dirs = self.config.get('output', {}).get('stage_dirs', {})
        stage3_name = stage_dirs.get('stage_3', 'Stage_3_Causal_Validation')
        coeff_path = Path(self.config['output']['base_dir']) / stage3_name / 'scenario_coefficients.json'

        if not coeff_path.exists():
            print("   No scenario_coefficients.json found — will use physics priors only.")
            return

        import json
        with open(coeff_path, 'r', encoding='utf-8') as f:
            self._causal_coefficients = json.load(f)

        n_direct = len(self._causal_coefficients.get('direct_effects', {}))
        n_med = sum(len(v) for v in self._causal_coefficients.get('mediator_propagation', {}).values())
        print(f"   Causal coefficients loaded: {n_direct} direct effects, {n_med} mediator rules")

    def get_causal_coefficient(self, variable: str) -> Optional[float]:
        """Return the DAG-adjusted structural coefficient for a treatment variable."""
        if self._causal_coefficients is None:
            return None
        effects = self._causal_coefficients.get('direct_effects', {})
        entry = effects.get(variable, {})
        return entry.get('structural_coeff')

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _predict_baseline(self, df: pd.DataFrame, verbose: bool = False) -> Tuple:
        """Run the full prediction pipeline and return (final, ols, gwr, gwrf, ggpgam)."""
        X = df[self.features].values
        coords = df[self.coord_cols].values

        preds = {}
        for name in ("ols", "gwr", "gwrf", "ggpgam"):
            model = self._models[name]
            if verbose:
                print(f"   Running {name.upper()}...")
            if name == "ols":
                preds[name] = model.predict(X)
            else:
                raw = model.predict(X, coords)
                preds[name] = raw[0] if isinstance(raw, tuple) else raw

        if verbose:
            print("   Running meta-ensemble...")
        final = self._meta_model.predict(preds, coords=coords, original_X=X)
        return final, preds["ols"], preds["gwr"], preds["gwrf"], preds["ggpgam"]

    # ------------------------------------------------------------------
    # Base-model delta consensus
    # ------------------------------------------------------------------

    def _predict_consensus_delta(
        self,
        df: pd.DataFrame,
        variable: str,
        modified_values: np.ndarray,
        baseline_base_preds: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """Compute a weighted-consensus temperature delta from base models.

        Instead of feeding perturbed features through the full meta-
        ensemble stacker (which is dominated by spatial features that
        don't change), this method:

        1. Runs each base model on *modified* features to get per-model
           predictions.
        2. Subtracts the *baseline* per-model predictions to get
           per-model deltas.
        3. Combines them using the ``_base_model_weights`` derived from
           the meta-ensemble's own feature-importance gain scores.

        Parameters
        ----------
        df : pd.DataFrame
            Original (unmodified) data.
        variable : str
            The scenario variable being changed.
        modified_values : ndarray
            The perturbed values for *variable*.
        baseline_base_preds : dict, optional
            Pre-computed ``{model_name: ndarray}`` baseline predictions.
            If None, baseline predictions are computed on the fly.

        Returns
        -------
        ndarray
            Per-point weighted-consensus delta.
        """
        df_mod = df.copy()
        df_mod[variable] = modified_values

        X_mod = df_mod[self.features].values
        coords = df_mod[self.coord_cols].values

        # Baseline per-model predictions (reuse cache when available)
        if baseline_base_preds is None:
            X_base = df[self.features].values
            coords_base = df[self.coord_cols].values
            baseline_base_preds = {}
            for name in ("ols", "gwr", "gwrf", "ggpgam"):
                model = self._models[name]
                if name == "ols":
                    baseline_base_preds[name] = model.predict(X_base)
                else:
                    raw = model.predict(X_base, coords_base)
                    baseline_base_preds[name] = raw[0] if isinstance(raw, tuple) else raw

        # Per-model modified predictions & deltas
        consensus_delta = np.zeros(len(df))
        for name in ("ols", "gwr", "gwrf", "ggpgam"):
            model = self._models[name]
            if name == "ols":
                mod_pred = model.predict(X_mod)
            else:
                raw = model.predict(X_mod, coords)
                mod_pred = raw[0] if isinstance(raw, tuple) else raw

            delta_m = mod_pred - baseline_base_preds[name]
            w = self._base_model_weights.get(name, 0.25)
            consensus_delta += w * delta_m

        return consensus_delta

    def _predict_joint_consensus_delta(
        self,
        df: pd.DataFrame,
        modified_df: pd.DataFrame,
        baseline_base_preds: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """Consensus delta when *multiple* variables are changed jointly.

        Works identically to :meth:`_predict_consensus_delta`, but
        accepts a fully modified DataFrame instead of a single variable.
        """
        X_mod = modified_df[self.features].values
        coords = modified_df[self.coord_cols].values

        if baseline_base_preds is None:
            X_base = df[self.features].values
            coords_base = df[self.coord_cols].values
            baseline_base_preds = {}
            for name in ("ols", "gwr", "gwrf", "ggpgam"):
                model = self._models[name]
                if name == "ols":
                    baseline_base_preds[name] = model.predict(X_base)
                else:
                    raw = model.predict(X_base, coords_base)
                    baseline_base_preds[name] = raw[0] if isinstance(raw, tuple) else raw

        consensus_delta = np.zeros(len(df))
        for name in ("ols", "gwr", "gwrf", "ggpgam"):
            model = self._models[name]
            if name == "ols":
                mod_pred = model.predict(X_mod)
            else:
                raw = model.predict(X_mod, coords)
                mod_pred = raw[0] if isinstance(raw, tuple) else raw

            delta_m = mod_pred - baseline_base_preds[name]
            w = self._base_model_weights.get(name, 0.25)
            consensus_delta += w * delta_m

        return consensus_delta

    # ------------------------------------------------------------------
    # Spatial heterogeneity
    # ------------------------------------------------------------------

    def _load_cate_multipliers(self, stage3_dir: str) -> Dict[str, np.ndarray]:
        """
        Load CATE-derived spatial multipliers from Stage 3 output.

        Returns dict mapping treatment name → ndarray of per-cell multipliers.
        Falls back to empty dict if files not found.
        """
        import glob
        multipliers: Dict[str, np.ndarray] = {}
        pattern = os.path.join(stage3_dir, 'spatial_cate_multiplier_*.npy')
        for path in glob.glob(pattern):
            fname = os.path.basename(path)
            treatment = fname.replace('spatial_cate_multiplier_', '').replace('.npy', '')
            multipliers[treatment] = np.load(path)
        return multipliers

    @staticmethod
    def _compute_spatial_multiplier(
        baseline_pred: np.ndarray,
        ols_pred: np.ndarray,
        gwr_pred: np.ndarray,
        gwrf_pred: np.ndarray,
        ggpgam_pred: np.ndarray,
    ) -> np.ndarray:
        """
        Build a spatial multiplier (0.5–1.5) from inter-model disagreement
        and baseline temperature z-score.

        This is the heuristic fallback.  When CATE-derived multipliers are
        available (from Stage 3 ``SpatialCATEEstimator``), those are used
        instead — see ``run()`` for the logic.
        """
        pred_range = (
            np.maximum.reduce([ols_pred, gwr_pred, gwrf_pred, ggpgam_pred])
            - np.minimum.reduce([ols_pred, gwr_pred, gwrf_pred, ggpgam_pred])
        )
        pred_range_norm = pred_range / (pred_range.mean() + 1e-12)
        temp_zscore = (baseline_pred - baseline_pred.mean()) / (baseline_pred.std() + 1e-12)
        multiplier = 1.0 + 0.2 * temp_zscore + 0.1 * (pred_range_norm - 1.0)
        return np.clip(multiplier, 0.5, 1.5)

    # ------------------------------------------------------------------
    # Physics delta
    # ------------------------------------------------------------------

    def _physics_delta(
        self, variable: str, actual_change: np.ndarray, spatial_mult: np.ndarray
    ) -> np.ndarray:
        """Temperature delta (°unit) with spatial heterogeneity.

        If Stage 3 causal coefficients are loaded, the DAG-adjusted
        structural coefficient is blended with the physics prior.

        When the Stage 3 estimator is ``dml`` (debiased ML), the causal
        coefficient receives higher blend weight (default 0.8) because
        DML provides doubly-robust estimates that are more trustworthy
        than the OLS-based physics-prior blend.
        """
        prior = self.physics_priors.get(variable)
        if prior is None:
            warnings.warn(f"No physics prior for {variable} — returning zero delta")
            return np.zeros_like(actual_change)

        coef = prior["coefficient"]
        unit = prior["unit_increment"]

        # Blend with causal coefficient when available
        causal_coef = self.get_causal_coefficient(variable)
        if causal_coef is not None:
            # Higher blend weight when DML estimator was used (more trustworthy)
            estimator = 'ols'
            if self._causal_coefficients is not None:
                estimator = self._causal_coefficients.get(
                    'metadata', {}
                ).get('estimator', 'ols')
            if estimator == 'dml':
                blend_w = self.config.get('causal', {}).get('dag_blend_weight', 0.8)
            else:
                blend_w = self.config.get('causal', {}).get('dag_blend_weight', 0.5)
            # causal_coef is per-unit-of-variable; physics coef is per unit_increment
            causal_coef_per_unit_inc = causal_coef * unit
            coef = blend_w * causal_coef_per_unit_inc + (1 - blend_w) * coef

        return coef * (actual_change / unit) * spatial_mult

    # ------------------------------------------------------------------
    # Interaction (multi-variable) scenarios
    # ------------------------------------------------------------------

    def run_interaction_scenarios(
        self,
        baseline_df: pd.DataFrame,
        baseline_pred: np.ndarray,
        spatial_multiplier: np.ndarray,
        target_mask: "pd.Series",
        results_df: pd.DataFrame,
        cate_multipliers: Dict[str, np.ndarray],
        verbose: bool = True,
    ) -> List[dict]:
        """Run interaction scenarios that modify multiple variables simultaneously.

        Each interaction scenario applies all leg deltas at once and sums the
        physics-predicted response deltas.

        Returns a list of summary dicts (same schema as single-variable rows).
        """
        rows: List[dict] = []
        for ix_sc in self.interaction_scenarios:
            name = ix_sc.get("name", "interaction")
            legs = ix_sc.get("legs", [])
            if not legs:
                continue

            if verbose:
                leg_desc = ", ".join(f"{l['variable']}Δ{l['delta']:+g}" for l in legs)
                print(f"\n--- Interaction: {name} ({leg_desc}) ---")

            combined_delta = np.zeros(target_mask.sum(), dtype=float)
            label_parts: List[str] = []

            for leg in legs:
                var_name = leg["variable"]
                delta = float(leg.get("delta", 0))
                if delta == 0 or var_name not in baseline_df.columns:
                    continue

                original = baseline_df.loc[target_mask, var_name].values
                modified = original + delta

                # Clip to physics bounds from scenario config
                sc_cfg = next((s for s in self.scenarios if s["variable"] == var_name), {})
                min_val = sc_cfg.get("min_val")
                max_val = sc_cfg.get("max_val")
                if min_val is not None:
                    modified = np.maximum(modified, min_val)
                if max_val is not None:
                    modified = np.minimum(modified, max_val)

                actual_change = modified - original

                if var_name in cate_multipliers:
                    full_cate = cate_multipliers[var_name]
                    if len(full_cate) == len(baseline_pred):
                        sm = full_cate[target_mask]
                    else:
                        sm = spatial_multiplier[target_mask]
                else:
                    sm = spatial_multiplier[target_mask]

                combined_delta += self._physics_delta(var_name, actual_change, sm)
                sign = "plus" if delta >= 0 else "minus"
                label_parts.append(f"{var_name}_{sign}_{str(abs(delta)).replace('.', 'p')}")

            scenario_label = "ix_" + "__".join(label_parts)
            scenario_pred = baseline_pred.copy()
            scenario_pred[target_mask] = baseline_pred[target_mask] + combined_delta
            results_df[f"pred_{scenario_label}"] = scenario_pred

            # Per-area summaries
            for area_id, area_name in self.area_names.items():
                mask = baseline_df[self.area_column] == area_id if self.area_column else pd.Series(True, index=baseline_df.index)
                if mask.sum() == 0:
                    continue
                rows.append({
                    "scenario": name,
                    "label": scenario_label,
                    "area": area_name,
                    "n_cells": int(mask.sum()),
                    "baseline_mean": float(baseline_pred[mask].mean()),
                    "scenario_mean": float(scenario_pred[mask].mean()),
                    "delta_mean": float((scenario_pred[mask] - baseline_pred[mask]).mean()),
                })

            if not self.area_names:
                rows.append({
                    "scenario": name,
                    "label": scenario_label,
                    "area": "all",
                    "n_cells": int(target_mask.sum()),
                    "baseline_mean": float(baseline_pred[target_mask].mean()),
                    "scenario_mean": float(scenario_pred[target_mask].mean()),
                    "delta_mean": float(combined_delta.mean()),
                })

            if verbose:
                print(f"   Combined delta mean: {combined_delta.mean():.4f}")

        return rows

    # ------------------------------------------------------------------
    # Core run
    # ------------------------------------------------------------------

    def run(self, verbose: bool = True) -> Tuple[pd.DataFrame, Optional[Any]]:
        """
        Execute all scenarios and return (summary_df, results_gdf).

        Parameters
        ----------
        verbose : bool
            Print progress to stdout.

        Returns
        -------
        summary_df : pd.DataFrame
            One row per (scenario, increment, area) combination.
        results_gdf : gpd.GeoDataFrame | pd.DataFrame
            Spatially-resolved results with prediction columns.
        """
        # Load data
        if verbose:
            print("\n2. Loading baseline data...")
        baseline_df = pd.read_csv(self.data_path)
        if verbose:
            print(f"   {len(baseline_df)} cells loaded")

        # Target mask (all rows if no area_column)
        if self.area_column and self.area_column in baseline_df.columns and self.area_names:
            target_mask = baseline_df[self.area_column].isin(self.area_names.keys())
        else:
            target_mask = pd.Series(True, index=baseline_df.index)

        if verbose:
            print(f"   Target cells: {target_mask.sum()}")

        # Baseline predictions
        if verbose:
            print("\n3. Computing spatial heterogeneity multipliers...")
        (baseline_pred, ols_base, gwr_base, gwrf_base, ggpgam_base) = self._predict_baseline(baseline_df, verbose=verbose)

        spatial_multiplier = self._compute_spatial_multiplier(
            baseline_pred, ols_base, gwr_base, gwrf_base, ggpgam_base
        )

        # Try to load CATE-derived multipliers from Stage 3
        cate_multipliers: Dict[str, np.ndarray] = {}
        stage3_dir = self.config.get('causal', {}).get('stage3_dir')
        if not stage3_dir:
            # Fall back to paths module
            try:
                from sparc.run.pipeline_paths import get_paths
                stage3_dir = str(get_paths().stage3_dir)
            except Exception:
                pass
        if stage3_dir and os.path.isdir(stage3_dir):
            cate_multipliers = self._load_cate_multipliers(stage3_dir)
            if cate_multipliers and verbose:
                print(f"   CATE multipliers loaded for: {list(cate_multipliers.keys())}")

        # Load elasticity from scenario_coefficients.json
        elasticity_map: Dict[str, float] = {}
        if stage3_dir:
            coeff_path = os.path.join(stage3_dir, 'scenario_coefficients.json')
            if os.path.isfile(coeff_path):
                import json as _json
                with open(coeff_path, 'r') as _f:
                    sc = _json.load(_f)
                for var, info in sc.get('direct_effects', {}).items():
                    if info.get('elasticity') is not None:
                        elasticity_map[var] = info['elasticity']

        if verbose:
            print(f"   Spatial multiplier range: {spatial_multiplier.min():.2f} to {spatial_multiplier.max():.2f}")

        results_df = baseline_df.copy()
        results_df["pred_baseline"] = baseline_pred
        results_df["spatial_multiplier"] = spatial_multiplier

        # Print priors table
        if verbose:
            self._print_priors_table()

        # Run scenarios
        if verbose:
            print("\n" + "="*70)
            print("5. RUNNING SCENARIOS (With Spatial Heterogeneity)")
            print("="*70)

        summary_rows: List[dict] = []

        for scenario in self.scenarios:
            var_name = scenario["variable"]
            direction = scenario.get("direction", "increase")
            increments = scenario.get("increments", [])
            min_val = scenario.get("min_val", None)
            max_val = scenario.get("max_val", None)
            unit = scenario.get("unit", "")

            if verbose:
                print(f"\n--- Variable: {var_name} ---")

            for increment in increments:
                if direction == "decrease":
                    scenario_label = f"{var_name}_minus_{str(increment).replace('.', 'p')}"
                    delta = -increment
                else:
                    scenario_label = f"{var_name}_plus_{str(increment).replace('.', 'p')}"
                    delta = increment

                if verbose:
                    print(f"\n   Scenario: {scenario_label}")

                original_values = baseline_df.loc[target_mask, var_name].values.copy()
                modified_values = original_values + delta

                # Clip to physical bounds if specified
                if min_val is not None:
                    modified_values = np.maximum(modified_values, min_val)
                if max_val is not None:
                    modified_values = np.minimum(modified_values, max_val)

                actual_change = modified_values - original_values
                # Use CATE-derived multiplier if available, else heuristic
                if var_name in cate_multipliers:
                    full_cate_mult = cate_multipliers[var_name]
                    if len(full_cate_mult) == len(baseline_pred):
                        target_spatial_mult = full_cate_mult[target_mask]
                    else:
                        target_spatial_mult = spatial_multiplier[target_mask]
                else:
                    target_spatial_mult = spatial_multiplier[target_mask]

                temp_delta_target = self._physics_delta(var_name, actual_change, target_spatial_mult)

                scenario_pred = baseline_pred.copy()
                scenario_pred[target_mask] = baseline_pred[target_mask] + temp_delta_target
                results_df[f"pred_{scenario_label}"] = scenario_pred

                # Extrapolation guard
                if EXTRAPOLATION_GUARD_AVAILABLE:
                    self._run_extrapolation_guard(
                        baseline_df, results_df, var_name, modified_values,
                        target_mask, temp_delta_target, baseline_pred, scenario_label, verbose
                    )

                # Per-area summaries
                for area_id, area_name in self.area_names.items():
                    mask = baseline_df[self.area_column] == area_id
                    if mask.sum() == 0:
                        continue
                    district_in_target = np.isin(
                        baseline_df.index[target_mask],
                        baseline_df.index[mask],
                    )
                    d_change = actual_change[district_in_target].mean()
                    d_delta = temp_delta_target[district_in_target].mean()
                    d_std = temp_delta_target[district_in_target].std()
                    d_mult = target_spatial_mult[district_in_target].mean()

                    if verbose:
                        print(f"      {area_name}: {d_delta:+.4f} +/- {d_std:.4f} (var: {d_change:+.2f}{unit}, mult: {d_mult:.2f})")

                    summary_rows.append({
                        "Variable": var_name,
                        "Scenario": scenario_label,
                        "Increment": increment,
                        "Direction": direction,
                        "Area": area_name,
                        "N_Cells": mask.sum(),
                        "Avg_Var_Change": d_change,
                        "Avg_Temp_Delta": d_delta,
                        "Temp_Delta_Std": d_std,
                        "Avg_Spatial_Multiplier": d_mult,
                        "Physics_Coef": self.physics_priors.get(var_name, {}).get("coefficient", 0.0),
                        "Elasticity": elasticity_map.get(var_name),
                        "CATE_multiplier_used": var_name in cate_multipliers,
                    })

        # --- Interaction (multi-variable) scenarios ---
        if self.interaction_scenarios:
            if verbose:
                print(f"\n{'='*70}")
                print(f"6. RUNNING INTERACTION SCENARIOS ({len(self.interaction_scenarios)})")
                print(f"{'='*70}")
            ix_rows = self.run_interaction_scenarios(
                baseline_df, baseline_pred, spatial_multiplier,
                target_mask, results_df, cate_multipliers, verbose=verbose,
            )
            summary_rows.extend(ix_rows)

        summary_df = pd.DataFrame(summary_rows)

        # Save summary CSV via ResultStore (with CSV fallback)
        try:
            from sparc.run.pipeline_paths import get_result_store
            store = get_result_store()
            store.save_dataframe(4, "scenario_summary.csv", summary_df, fmt="csv")
        except Exception:
            summary_path = self.output_dir / "scenario_summary.csv"
            summary_df.to_csv(summary_path, index=False)
        if verbose:
            print(f"\n   Summary saved")

        # Build GeoDataFrame if possible
        results_gdf = self._to_geodataframe(results_df, baseline_df)
        if GEOPANDAS_AVAILABLE and isinstance(results_gdf, gpd.GeoDataFrame):
            try:
                from sparc.run.pipeline_paths import get_result_store
                store = get_result_store()
                store.save_geodataframe(4, "scenario_results.gpkg", results_gdf)
            except Exception:
                try:
                    gpkg_path = self.output_dir / "scenario_results.gpkg"
                    results_gdf.to_file(gpkg_path, driver="GPKG")
                except Exception as e:
                    warnings.warn(f"Could not write GeoPackage: {e}")
            if verbose:
                print(f"   GeoPackage saved")

        return summary_df, results_gdf

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_geodataframe(self, results_df: pd.DataFrame, baseline_df: pd.DataFrame):
        """Convert results to GeoDataFrame if geopandas is available."""
        if not GEOPANDAS_AVAILABLE:
            return results_df
        try:
            x_col, y_col = self.coord_cols
            geometry = [Point(xy) for xy in zip(baseline_df[x_col], baseline_df[y_col])]
            gdf = gpd.GeoDataFrame(results_df, geometry=geometry, crs=self.model_crs)
            if self.area_column in gdf.columns and self.area_names:
                gdf["Area_Name"] = gdf[self.area_column].map(self.area_names)
            return gdf
        except Exception as e:
            warnings.warn(f"Could not create GeoDataFrame: {e}")
            return results_df

    def _run_extrapolation_guard(
        self, baseline_df, results_df, var_name, modified_values,
        target_mask, temp_delta_target, baseline_pred, scenario_label, verbose
    ):
        """Run extrapolation guard scoring for a single scenario.

        Uses the full multivariate feature matrix (all predictors with
        the perturbed column substituted) for Mahalanobis distance
        computation, rather than a single-column approximation.
        """
        try:
            X_train = baseline_df[self.features].values
            X_scenario = baseline_df[self.features].values.copy()
            var_idx = self.features.index(var_name)
            X_scenario[target_mask, var_idx] = modified_values

            # Full multivariate Mahalanobis on the perturbed feature vectors
            extrap_scores = compute_extrapolation_score(
                X_scenario[target_mask], X_train
            )
            confidence_labels = classify_prediction_confidence(
                extrap_scores, temp_delta_target,
                float(np.ptp(baseline_df[var_name].values))
            )
            conf_col = f"confidence_{scenario_label}"
            results_df[conf_col] = "N/A"
            results_df.loc[target_mask, conf_col] = confidence_labels

            n_spec = np.sum(confidence_labels == "SPECULATIVE")
            n_low = np.sum(confidence_labels == "LOW")
            if verbose and (n_spec > 0 or n_low > 0):
                print(f"      [GUARD] {n_spec} SPECULATIVE + {n_low} LOW confidence cells")
        except Exception as e:
            if verbose:
                print(f"      [GUARD] Extrapolation check failed: {e}")

    # ------------------------------------------------------------------
    # DAG-integrated scenario simulation (MGWR spatial blend)
    # ------------------------------------------------------------------

    def run_with_causal_dag(
        self,
        data: pd.DataFrame,
        verbose: bool = True,
    ) -> Tuple[pd.DataFrame, Optional[Any]]:
        """
        Execute scenarios using **MGWR local coefficients** for spatial
        heterogeneity and the **causal DAG** for indirect (mediated) effects.

        Architecture (per scenario increment):

        1. Compute ``actual_change`` per point (with physical bounds).
        2. Apply diminishing returns (square-root taper beyond threshold).
        3. **Direct effect** — per-point MGWR local coefficient × effective Δ.
        4. **Indirect effects** — DAG structural coefficients propagate Δ
           through mediator nodes; the *final* edge to the outcome still
           uses the per-point MGWR coefficient so the indirect effect
           inherits spatial heterogeneity.
        5. **Physics guardrails** — sign check + magnitude cap (±3 z-score).
        6. Per-point and per-area summaries.

        Returns
        -------
        summary_df : pd.DataFrame
        results_gdf : GeoDataFrame | DataFrame
        """
        from sparc.causal.counterfactual_engine import CounterfactualEngine

        # ── Load DAG and fit structural coefficients ──────────────────
        engine = CounterfactualEngine(self.config)
        dag_file = self.config.get('causal', {}).get('dag_file')
        if not dag_file or not Path(dag_file).exists():
            warnings.warn("No causal DAG file found — falling back to physics-only run().")
            return self.run(verbose=verbose)

        engine.load_dag(dag_file)
        engine.fit(data)
        target_var = engine.roles['outcomes'][0]

        if verbose:
            print("\nCausal DAG loaded and structural coefficients fitted.")
            n_edges = len(engine._structural_coeffs)
            print(f"   {n_edges} structural coefficients estimated")
            for (p, c), v in sorted(engine._structural_coeffs.items()):
                print(f"      {p:30s} -> {c:15s}  coeff={v:+.6f}")

        # ── Baseline predictions ──────────────────────────────────────
        if verbose:
            print("\nComputing baseline predictions...")
        (baseline_pred, ols_base, gwr_base, gwrf_base, ggpgam_base) = self._predict_baseline(
            data, verbose=verbose
        )

        results_df = data.copy()
        results_df["pred_baseline"] = baseline_pred

        # Print physics priors table
        if verbose:
            self._print_priors_table()

        # ── Scenario loop ─────────────────────────────────────────────
        if verbose:
            mgwr_status = "ENABLED" if self._mgwr_coefficients_raw is not None else "DISABLED (fallback to DAG)"
            print(f"\n{'='*70}")
            print(f"RUNNING SCENARIOS — MGWR Spatial Blend: {mgwr_status}")
            print(f"{'='*70}")

        summary_rows: List[dict] = []

        for scenario in self.scenarios:
            var_name = scenario["variable"]
            direction = scenario.get("direction", "increase")
            increments = scenario.get("increments", [])
            min_val = scenario.get("min_val", None)
            max_val = scenario.get("max_val", None)
            unit = scenario.get("unit", "")

            if verbose:
                print(f"\n{'─'*50}")
                print(f"Variable: {var_name}  (direction={direction})")
                print(f"{'─'*50}")

            for increment in increments:
                delta_signed = -increment if direction == "decrease" else increment
                label = f"{var_name}_{'minus' if delta_signed < 0 else 'plus'}_{str(increment).replace('.', 'p')}"

                # 1. Per-point actual change with physical bounds
                original_vals = data[var_name].values.copy()
                modified_vals = original_vals + delta_signed
                if min_val is not None:
                    modified_vals = np.maximum(modified_vals, min_val)
                if max_val is not None:
                    modified_vals = np.minimum(modified_vals, max_val)
                actual_change = modified_vals - original_vals

                # 2. Diminishing returns
                threshold = self._get_diminishing_threshold(var_name)
                effective_change = self._diminishing_return(actual_change, threshold)

                # 3. Direct effect — saturation curve or MGWR/physics blend
                direct_delta, direct_method = self._compute_mgwr_direct_delta(
                    var_name, effective_change,
                    baseline_values=original_vals,
                    modified_values=modified_vals,
                )

                # 3b. Physics guardrails on direct effect (direction-aware)
                change_dir = 1.0 if np.mean(actual_change) >= 0 else -1.0
                direct_delta = self._apply_physics_guardrails(
                    var_name, direct_delta, change_direction=change_dir,
                )

                # 4. Indirect effects — DAG mediator pathways + MGWR final step
                indirect_delta = self._compute_indirect_effects(
                    var_name, effective_change, engine, target_var,
                )

                # 4b. Temporal lagged effects — if DAG has time-lagged edges
                lagged_effects = self._compute_temporal_lagged_effects(
                    var_name, effective_change, engine, target_var,
                )

                # 5. Total = guardrailed direct + indirect (no second guardrail)
                total_delta = direct_delta + indirect_delta
                # Final magnitude cap only (no sign check — already handled)
                _max_mag = float(self.config.get('caps', {}).get('max_delta_magnitude', 3.0))
                total_delta = np.clip(total_delta, -_max_mag, _max_mag)

                # 6. Also compute the old-style global DAG delta for comparison
                try:
                    dag_result = engine.intervene(var_name, delta_signed, data=data)
                    dag_global_delta = dag_result['predicted_delta_response'].iloc[0]
                except Exception:
                    dag_global_delta = float('nan')

                # Store per-point results
                results_df[f"direct_{label}"] = direct_delta
                results_df[f"indirect_{label}"] = indirect_delta
                results_df[f"total_{label}"] = total_delta
                results_df[f"pred_{label}"] = baseline_pred + total_delta

                # Store time-lagged effects (delayed response columns)
                for lag, lag_delta in sorted(lagged_effects.items()):
                    results_df[f"lagged_t+{lag}_{label}"] = lag_delta
                    if verbose:
                        print(f"    Lagged t+{lag}: {float(np.mean(lag_delta)):+.4f}")

                # 6b. Extrapolation guard — confidence flags (multivariate)
                if EXTRAPOLATION_GUARD_AVAILABLE:
                    try:
                        X_train = data[self.features].values
                        X_scenario = X_train.copy()
                        var_idx = self.features.index(var_name)
                        X_scenario[:, var_idx] = modified_vals
                        extrap_scores = compute_extrapolation_score(
                            X_scenario, X_train
                        )
                        # max_observed_change = range of this variable in training data
                        max_obs_change = float(np.ptp(data[var_name].values))
                        confidence_labels = classify_prediction_confidence(
                            extrap_scores, actual_change,
                            max_obs_change,
                        )
                        results_df[f"confidence_{label}"] = confidence_labels
                        n_spec = int(np.sum(confidence_labels == "SPECULATIVE"))
                        n_low = int(np.sum(confidence_labels == "LOW"))
                        if verbose and (n_spec > 0 or n_low > 0):
                            print(f"      [GUARD] {n_spec} SPECULATIVE + {n_low} LOW confidence cells")
                    except Exception as e:
                        if verbose:
                            print(f"      [GUARD] Extrapolation check failed: {e}")

                # Summary statistics
                direct_mean = float(np.mean(direct_delta))
                indirect_mean = float(np.mean(indirect_delta))
                total_mean = float(np.mean(total_delta))
                total_std = float(np.std(total_delta))
                mgwr_j = self._mgwr_feature_map.get(var_name)
                mgwr_coeff_mean = (
                    float(np.mean(self._mgwr_coefficients_raw[:, mgwr_j]))
                    if mgwr_j is not None and self._mgwr_coefficients_raw is not None
                    else float('nan')
                )
                # Physics literature per raw unit
                _prior = self.physics_priors.get(var_name, {})
                _unit_inc = _prior.get("unit_increment", 1.0)
                lit_per_unit = _prior.get("lit_coef", 0.0) / _unit_inc if _unit_inc else 0.0
                physics_coef = _prior.get("coefficient", float('nan'))
                physics_dir = _prior.get("direction", "n/a")

                # Determine which coefficient source dominated
                _mgwr_abs = abs(mgwr_coeff_mean) if not np.isnan(mgwr_coeff_mean) else 0.0
                coeff_source = direct_method  # from _compute_mgwr_direct_delta return

                if verbose:
                    print(f"\n  {label}:")
                    print(f"    Raw Δ:  mean={np.mean(actual_change):+.2f}{unit}  "
                          f"Eff Δ: mean={np.mean(effective_change):+.2f}")
                    print(f"    Direct:   {direct_mean:+.4f}  (source: {coeff_source}, "
                          f"MGWR={mgwr_coeff_mean:+.6f}, lit/unit={lit_per_unit:+.4f})")
                    print(f"    Indirect: {indirect_mean:+.4f}")
                    print(f"    Total:    {total_mean:+.4f} ± {total_std:.4f}  (z-score units)")
                    print(f"    DAG-only: {dag_global_delta:+.4f}  (global, for comparison)")
                    print(f"    Physics:  {physics_coef:+.4f} per {_prior.get('unit_increment', '?')}{unit}  ({physics_dir})")

                # Per-area summaries
                for area_id, area_name in self.area_names.items():
                    if self.area_column not in data.columns:
                        continue
                    mask = data[self.area_column] == area_id
                    if mask.sum() == 0:
                        continue
                    n_cells = int(mask.sum())
                    a_total = total_delta[mask.values]
                    a_direct = direct_delta[mask.values]
                    a_indirect = indirect_delta[mask.values]
                    a_change = actual_change[mask.values]

                    if verbose:
                        print(f"    {area_name:20s}  total={a_total.mean():+.4f}±{a_total.std():.4f}"
                              f"  direct={a_direct.mean():+.4f}  indirect={a_indirect.mean():+.4f}"
                              f"  ΔVar={a_change.mean():+.2f}{unit}  (n={n_cells})")

                    summary_rows.append({
                        "Variable": var_name,
                        "Scenario": label,
                        "Increment": increment,
                        "Direction": direction,
                        "Area": area_name,
                        "N_Cells": n_cells,
                        "Avg_Var_Change": float(a_change.mean()),
                        "Direct_Delta_Mean": float(a_direct.mean()),
                        "Indirect_Delta_Mean": float(a_indirect.mean()),
                        "Total_Delta_Mean": float(a_total.mean()),
                        "Total_Delta_Std": float(a_total.std()),
                        "DAG_Global_Delta": dag_global_delta,
                        "MGWR_Coeff_Mean": mgwr_coeff_mean,
                        "Lit_Per_Unit": lit_per_unit,
                        "Coeff_Source": coeff_source,
                        "Physics_Lit_Coeff": physics_coef,
                        "Physics_Direction": physics_dir,
                    })

                # If no area names (global summary)
                if not self.area_names:
                    summary_rows.append({
                        "Variable": var_name,
                        "Scenario": label,
                        "Increment": increment,
                        "Direction": direction,
                        "Area": "All",
                        "N_Cells": len(data),
                        "Avg_Var_Change": float(np.mean(actual_change)),
                        "Direct_Delta_Mean": direct_mean,
                        "Indirect_Delta_Mean": indirect_mean,
                        "Total_Delta_Mean": total_mean,
                        "Total_Delta_Std": total_std,
                        "DAG_Global_Delta": dag_global_delta,
                        "MGWR_Coeff_Mean": mgwr_coeff_mean,
                        "Lit_Per_Unit": lit_per_unit,
                        "Coeff_Source": coeff_source,
                        "Physics_Lit_Coeff": physics_coef,
                        "Physics_Direction": physics_dir,
                    })

        # ── Joint (multi-variable) scenarios ─────────────────────────
        joint_scenarios = self.config.get('joint_scenarios', [])
        if joint_scenarios and verbose:
            print(f"\n{'='*70}")
            print(f"RUNNING JOINT SCENARIOS ({len(joint_scenarios)} defined)")
            print(f"{'='*70}")

        for joint in joint_scenarios:
            joint_name = joint.get('name', 'joint_scenario')
            interventions = joint.get('interventions', [])
            auto_propagate = joint.get('auto_propagate_dag', False)

            if verbose:
                print(f"\n{'─'*50}")
                print(f"Joint Scenario: {joint_name}")
                for iv in interventions:
                    print(f"  {iv['variable']} {iv.get('direction', 'increase')} {iv['increment']}")
                print(f"{'─'*50}")

            # 1. Apply all variable changes simultaneously
            modified_data = data.copy()
            actual_changes: Dict[str, np.ndarray] = {}

            for iv in interventions:
                vn = iv['variable']
                inc = iv['increment']
                direction = iv.get('direction', 'increase')
                delta = -inc if direction == 'decrease' else inc

                orig = data[vn].values.copy()
                mod = orig + delta

                # Physical bounds from the matching single-variable scenario
                single_sc = next(
                    (s for s in self.scenarios if s['variable'] == vn), {}
                )
                bmin = single_sc.get('min_val')
                bmax = single_sc.get('max_val')
                if bmin is not None:
                    mod = np.maximum(mod, bmin)
                if bmax is not None:
                    mod = np.minimum(mod, bmax)

                modified_data[vn] = mod
                actual_changes[vn] = mod - orig

            # 2. DAG auto-propagation (induced changes on correlated vars)
            if auto_propagate and engine is not None:
                for iv in interventions:
                    src = iv['variable']
                    # Find direct DAG children that are NOT in the explicit
                    # intervention list and NOT the outcome
                    for child in engine.graph.successors(src):
                        if child == target_var:
                            continue
                        already_explicit = any(
                            i['variable'] == child for i in interventions
                        )
                        if already_explicit:
                            continue
                        edge_coeff = engine._structural_coeffs.get(
                            (src, child), 0.0
                        )
                        if edge_coeff == 0.0:
                            continue
                        induced = actual_changes[src] * edge_coeff
                        modified_data[child] = data[child].values + induced
                        actual_changes[child] = induced
                        if verbose:
                            print(f"  Auto-propagated: {src} → {child}  "
                                  f"(coeff={edge_coeff:+.4f}, "
                                  f"mean Δ={induced.mean():+.3f})")

            # 3. Enforce combined constraint: Canopy + Impervious ≤ 100
            if ('Pct_Canopy' in modified_data.columns
                    and 'Pct_Impervious' in modified_data.columns):
                total_cover = (
                    modified_data['Pct_Canopy'].values
                    + modified_data['Pct_Impervious'].values
                )
                excess = np.maximum(total_cover - 100.0, 0.0)
                if np.any(excess > 0):
                    # Reduce impervious first (more actionable), then canopy
                    imp_vals = modified_data['Pct_Impervious'].values
                    reduction = np.minimum(excess, imp_vals)
                    modified_data['Pct_Impervious'] = imp_vals - reduction
                    remaining = excess - reduction
                    if np.any(remaining > 0):
                        modified_data['Pct_Canopy'] = (
                            modified_data['Pct_Canopy'].values - remaining
                        )
                    if verbose:
                        n_clipped = int(np.sum(excess > 0))
                        print(f"  Combined constraint: clipped {n_clipped} "
                              f"cells (Canopy+Impervious > 100%)")
                    # Recalculate actual changes after clipping
                    for vn in ('Pct_Canopy', 'Pct_Impervious'):
                        if vn in actual_changes:
                            actual_changes[vn] = (
                                modified_data[vn].values - data[vn].values
                            )

            # 4. Sum direct + indirect effects from all changed variables
            total_joint_delta = np.zeros(len(data))
            for vn, ac in actual_changes.items():
                threshold = self._get_diminishing_threshold(vn)
                eff_change = self._diminishing_return(ac, threshold)
                direct, _ = self._compute_mgwr_direct_delta(
                    vn, eff_change,
                    baseline_values=data[vn].values,
                    modified_values=modified_data[vn].values,
                )
                chdir = 1.0 if np.mean(ac) >= 0 else -1.0
                direct = self._apply_physics_guardrails(
                    vn, direct, change_direction=chdir,
                )
                # In joint scenarios, skip indirect paths that pass
                # through other co-intervened variables — those variables'
                # own direct effects already capture that contribution.
                co_intervened = set(actual_changes.keys()) - {vn}
                indirect = self._compute_indirect_effects(
                    vn, eff_change, engine, target_var,
                    skip_variables=co_intervened,
                ) if engine is not None else np.zeros(len(data))

                total_joint_delta += direct + indirect

            _max_mag = float(self.config.get('caps', {}).get('max_delta_magnitude', 3.0))
            total_joint_delta = np.clip(total_joint_delta, -_max_mag, _max_mag)

            # Store results
            label = joint_name.replace(' ', '_')
            results_df[f"total_{label}"] = total_joint_delta
            results_df[f"pred_{label}"] = baseline_pred + total_joint_delta

            joint_mean = float(np.mean(total_joint_delta))
            joint_std = float(np.std(total_joint_delta))

            if verbose:
                print(f"\n  {joint_name}:")
                for vn, ac in actual_changes.items():
                    print(f"    {vn}: mean Δ = {ac.mean():+.2f}")
                print(f"    Total joint Δ: {joint_mean:+.4f} ± {joint_std:.4f}")

            # Per-area summaries
            for area_id, area_name in self.area_names.items():
                if self.area_column not in data.columns:
                    continue
                mask = data[self.area_column] == area_id
                if mask.sum() == 0:
                    continue
                a_delta = total_joint_delta[mask.values]
                summary_rows.append({
                    "Variable": "JOINT:" + label,
                    "Scenario": label,
                    "Increment": 0,
                    "Direction": "joint",
                    "Area": area_name,
                    "N_Cells": int(mask.sum()),
                    "Avg_Var_Change": 0.0,
                    "Direct_Delta_Mean": float('nan'),
                    "Indirect_Delta_Mean": float('nan'),
                    "Total_Delta_Mean": float(a_delta.mean()),
                    "Total_Delta_Std": float(a_delta.std()),
                    "DAG_Global_Delta": float('nan'),
                    "MGWR_Coeff_Mean": float('nan'),
                    "Lit_Per_Unit": float('nan'),
                    "Coeff_Source": "joint",
                    "Physics_Lit_Coeff": float('nan'),
                    "Physics_Direction": "n/a",
                })

            if not self.area_names:
                summary_rows.append({
                    "Variable": "JOINT:" + label,
                    "Scenario": label,
                    "Increment": 0,
                    "Direction": "joint",
                    "Area": "All",
                    "N_Cells": len(data),
                    "Avg_Var_Change": 0.0,
                    "Direct_Delta_Mean": float('nan'),
                    "Indirect_Delta_Mean": float('nan'),
                    "Total_Delta_Mean": joint_mean,
                    "Total_Delta_Std": joint_std,
                    "DAG_Global_Delta": float('nan'),
                    "MGWR_Coeff_Mean": float('nan'),
                    "Lit_Per_Unit": float('nan'),
                    "Coeff_Source": "joint",
                    "Physics_Lit_Coeff": float('nan'),
                    "Physics_Direction": "n/a",
                })

        # ── Save results ──────────────────────────────────────────────
        summary_df = pd.DataFrame(summary_rows)
        try:
            from sparc.run.pipeline_paths import get_result_store
            store = get_result_store()
            store.save_dataframe(4, "scenario_summary_dag.csv", summary_df, fmt="csv")
            store.save_dataframe(4, "scenario_summary.csv", summary_df, fmt="csv")
        except Exception:
            summary_path = self.output_dir / "scenario_summary_dag.csv"
            summary_df.to_csv(summary_path, index=False)
        if verbose:
            print(f"\n{'='*70}")
            print(f"DAG scenario summary saved")

        results_gdf = self._to_geodataframe(results_df, data)
        if GEOPANDAS_AVAILABLE and isinstance(results_gdf, gpd.GeoDataFrame):
            try:
                from sparc.run.pipeline_paths import get_result_store
                store = get_result_store()
                store.save_geodataframe(4, "scenario_results_dag.gpkg", results_gdf)
                store.save_geodataframe(4, "scenario_results.gpkg", results_gdf)
            except Exception:
                try:
                    gpkg = self.output_dir / "scenario_results_dag.gpkg"
                    results_gdf.to_file(gpkg, driver="GPKG")
                    results_gdf.to_file(self.output_dir / "scenario_results.gpkg", driver="GPKG")
                except Exception as e:
                    warnings.warn(f"Could not write GeoPackage: {e}")
            if verbose:
                print(f"GeoPackage saved")

        # ── Generate interpretation maps ──────────────────────────────
        try:
            self._generate_interpretation_maps(results_df, data, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"  [MAPS] Map generation failed: {e}")

        return summary_df, results_gdf

    # ------------------------------------------------------------------
    # Ensemble re-prediction mode (Step 3)
    # ------------------------------------------------------------------

    def _predict_modified(
        self,
        df: pd.DataFrame,
        variable: str,
        modified_values: np.ndarray,
    ) -> np.ndarray:
        """Run the full ensemble on a *modified* feature matrix.

        Creates a copy of *df*, replaces *variable* with
        *modified_values*, then predicts through all four base models
        and the meta-ensemble.

        Returns
        -------
        ndarray
            Per-point ensemble predictions on the perturbed features.
        """
        df_mod = df.copy()
        df_mod[variable] = modified_values

        X_mod = df_mod[self.features].values
        coords = df_mod[self.coord_cols].values

        preds = {}
        for name in ("ols", "gwr", "gwrf", "ggpgam"):
            model = self._models[name]
            if name == "ols":
                preds[name] = model.predict(X_mod)
            else:
                raw = model.predict(X_mod, coords)
                preds[name] = raw[0] if isinstance(raw, tuple) else raw

        final = self._meta_model.predict(preds, coords=coords, original_X=X_mod)
        return final

    def run_with_model_reprediction(
        self,
        data: pd.DataFrame,
        verbose: bool = True,
    ) -> Tuple[pd.DataFrame, Optional[Any]]:
        """Execute scenarios using **base-model delta consensus**.

        Rather than re-running perturbed features through the full
        meta-ensemble stacker (whose spatial features dominate and
        suppress physical-variable deltas), this method:

        1. Computes baseline predictions (meta-ensemble for absolute
           accuracy, *plus* per-base-model predictions for delta
           extraction).
        2. For each scenario increment it runs each base model on the
           modified features, computes per-model deltas, and combines
           them using the meta-ensemble's own feature-importance
           weights.
        3. Applies physics guardrails, extrapolation guards, and
           optionally compares with the DAG coefficient path.

        Returns
        -------
        summary_df : pd.DataFrame
        results_gdf : GeoDataFrame | DataFrame
        """
        from sparc.causal.counterfactual_engine import CounterfactualEngine

        # ── Load DAG (for comparison column) ─────────────────────────
        engine = None
        dag_file = self.config.get('causal', {}).get('dag_file')
        if dag_file and Path(dag_file).exists():
            try:
                engine = CounterfactualEngine(self.config)
                engine.load_dag(dag_file)
                engine.fit(data)
            except Exception as e:
                warnings.warn(f"DAG loading failed ({e}); comparison column unavailable.")
                engine = None
        target_var = engine.roles['outcomes'][0] if engine else self.target_col

        # ── Baseline predictions ─────────────────────────────────────
        if verbose:
            print("\nComputing baseline predictions (all models)...")
        baseline_pred, ols_bl, gwr_bl, gwrf_bl, ggpgam_bl = self._predict_baseline(
            data, verbose=verbose,
        )
        baseline_base_preds: Dict[str, np.ndarray] = {
            "ols": ols_bl, "gwr": gwr_bl, "gwrf": gwrf_bl, "ggpgam": ggpgam_bl,
        }

        results_df = data.copy()
        results_df["pred_baseline"] = baseline_pred

        if verbose:
            self._print_priors_table()
            wstr = ", ".join(f"{k}={v:.3f}" for k, v in self._base_model_weights.items())
            print(f"\n   Consensus weights: {wstr}")
            print(f"\n{'='*70}")
            print("RUNNING SCENARIOS — Base-Model Delta Consensus")
            print(f"{'='*70}")

        summary_rows: List[dict] = []

        for scenario in self.scenarios:
            var_name = scenario["variable"]
            direction = scenario.get("direction", "increase")
            increments = scenario.get("increments", [])
            min_val = scenario.get("min_val", None)
            max_val = scenario.get("max_val", None)
            unit = scenario.get("unit", "")

            if verbose:
                print(f"\n{'─'*50}")
                print(f"Variable: {var_name}  (direction={direction})")
                print(f"{'─'*50}")

            for increment in increments:
                delta_signed = -increment if direction == "decrease" else increment
                label = f"{var_name}_{'minus' if delta_signed < 0 else 'plus'}_{str(increment).replace('.', 'p')}"

                # 1. Modified values with physical bounds
                original_vals = data[var_name].values.copy()
                modified_vals = original_vals + delta_signed
                if min_val is not None:
                    modified_vals = np.maximum(modified_vals, min_val)
                if max_val is not None:
                    modified_vals = np.minimum(modified_vals, max_val)
                actual_change = modified_vals - original_vals

                # 2. Base-model delta consensus
                repred_delta = self._predict_consensus_delta(
                    data, var_name, modified_vals,
                    baseline_base_preds=baseline_base_preds,
                )

                # 3. Physics guardrails on re-prediction delta
                change_dir = 1.0 if np.mean(actual_change) >= 0 else -1.0
                repred_delta = self._apply_physics_guardrails(
                    var_name, repred_delta, change_direction=change_dir,
                )

                # 4. Extrapolation guard — confidence flags
                confidence_labels = None
                if EXTRAPOLATION_GUARD_AVAILABLE:
                    try:
                        X_train = data[self.features].values
                        X_scenario = data[self.features].values.copy()
                        var_idx = self.features.index(var_name)
                        X_scenario[:, var_idx] = modified_vals
                        extrap_scores = compute_extrapolation_score(
                            X_scenario, X_train
                        )
                        confidence_labels = classify_prediction_confidence(
                            extrap_scores, actual_change,
                            np.std(baseline_pred),
                        )
                        results_df[f"confidence_{label}"] = confidence_labels
                    except Exception as e:
                        if verbose:
                            print(f"      [GUARD] Extrapolation check failed: {e}")

                # 5. DAG coefficient-based delta for comparison
                dag_global_delta = float('nan')
                if engine is not None:
                    try:
                        dag_result = engine.intervene(var_name, delta_signed, data=data)
                        dag_global_delta = dag_result['predicted_delta_response'].iloc[0]
                    except Exception:
                        pass

                # Store per-point results
                results_df[f"repred_delta_{label}"] = repred_delta
                results_df[f"pred_{label}"] = baseline_pred + repred_delta

                # Static PNG map
                map_path = self._save_scenario_map(
                    data, repred_delta, label,
                    title=f"{var_name} {direction} {increment}{unit}  |  "
                          f"mean Δ = {np.mean(repred_delta):+.4f}",
                )
                if map_path and verbose:
                    print(f"    Map saved: {map_path.name}")

                # Summary
                repred_mean = float(np.mean(repred_delta))
                repred_std = float(np.std(repred_delta))

                # 6. Mode 1 vs Mode 2 convergence check
                convergence_ok = True
                if not np.isnan(dag_global_delta) and abs(repred_mean) > 1e-6:
                    ratio = dag_global_delta / repred_mean if repred_mean != 0 else float('inf')
                    # Flag when modes disagree by more than 2× or opposite sign
                    if ratio < 0 or abs(ratio - 1.0) > 1.0:
                        convergence_ok = False

                if verbose:
                    n_spec = int(np.sum(confidence_labels == "SPECULATIVE")) if confidence_labels is not None else 0
                    print(f"\n  {label}:")
                    print(f"    Raw Δ:       mean={np.mean(actual_change):+.2f}{unit}")
                    print(f"    Repred Δ:    {repred_mean:+.4f} ± {repred_std:.4f}")
                    print(f"    DAG coeff Δ: {dag_global_delta:+.4f}  (global, for comparison)")
                    if not convergence_ok:
                        print(f"    ⚠ CONVERGENCE WARNING: Mode 1 and Mode 2 diverge "
                              f"(ratio={ratio:.2f}). Interpret Mode 2 with caution.")
                    if n_spec > 0:
                        print(f"    [GUARD] {n_spec} SPECULATIVE cells")

                # Per-area summaries
                for area_id, area_name in self.area_names.items():
                    if self.area_column not in data.columns:
                        continue
                    mask = data[self.area_column] == area_id
                    if mask.sum() == 0:
                        continue
                    n_cells = int(mask.sum())
                    a_repred = repred_delta[mask.values]
                    a_change = actual_change[mask.values]

                    if verbose:
                        print(f"    {area_name:20s}  repred={a_repred.mean():+.4f}±{a_repred.std():.4f}"
                              f"  ΔVar={a_change.mean():+.2f}{unit}  (n={n_cells})")

                    summary_rows.append({
                        "Variable": var_name,
                        "Scenario": label,
                        "Increment": increment,
                        "Direction": direction,
                        "Area": area_name,
                        "N_Cells": n_cells,
                        "Avg_Var_Change": float(a_change.mean()),
                        "Repred_Delta_Mean": float(a_repred.mean()),
                        "Repred_Delta_Std": float(a_repred.std()),
                        "DAG_Global_Delta": dag_global_delta,
                        "Mode_Convergence": convergence_ok,
                        "Method": "base_model_consensus",
                    })

                # Global summary fallback if no area names
                if not self.area_names:
                    summary_rows.append({
                        "Variable": var_name,
                        "Scenario": label,
                        "Increment": increment,
                        "Direction": direction,
                        "Area": "All",
                        "N_Cells": len(data),
                        "Avg_Var_Change": float(np.mean(actual_change)),
                        "Repred_Delta_Mean": repred_mean,
                        "Repred_Delta_Std": repred_std,
                        "DAG_Global_Delta": dag_global_delta,
                        "Mode_Convergence": convergence_ok,
                        "Method": "base_model_consensus",
                    })

        # ── Joint scenarios (consensus mode) ─────────────────────────
        joint_scenarios = self.config.get('joint_scenarios', [])
        if joint_scenarios and verbose:
            print(f"\n{'='*70}")
            print(f"JOINT SCENARIOS — Base-Model Delta Consensus ({len(joint_scenarios)} defined)")
            print(f"{'='*70}")

        for joint in joint_scenarios:
            joint_name = joint.get('name', 'joint_scenario')
            interventions = joint.get('interventions', [])

            if verbose:
                print(f"\n{'─'*50}")
                print(f"Joint Scenario: {joint_name}")
                for iv in interventions:
                    print(f"  {iv['variable']} {iv.get('direction', 'increase')} {iv['increment']}")
                print(f"{'─'*50}")

            # 1. Build a jointly-modified DataFrame
            modified_data = data.copy()
            actual_changes: Dict[str, np.ndarray] = {}

            for iv in interventions:
                vn = iv['variable']
                inc = iv['increment']
                direction = iv.get('direction', 'increase')
                delta = -inc if direction == 'decrease' else inc

                orig = data[vn].values.copy()
                mod = orig + delta

                single_sc = next(
                    (s for s in self.scenarios if s['variable'] == vn), {}
                )
                bmin = single_sc.get('min_val')
                bmax = single_sc.get('max_val')
                if bmin is not None:
                    mod = np.maximum(mod, bmin)
                if bmax is not None:
                    mod = np.minimum(mod, bmax)

                modified_data[vn] = mod
                actual_changes[vn] = mod - orig

            # 2. Canopy + Impervious <= 100 constraint
            if ('Pct_Canopy' in modified_data.columns
                    and 'Pct_Impervious' in modified_data.columns):
                total_cover = (
                    modified_data['Pct_Canopy'].values
                    + modified_data['Pct_Impervious'].values
                )
                excess = np.maximum(total_cover - 100.0, 0.0)
                if np.any(excess > 0):
                    imp_vals = modified_data['Pct_Impervious'].values
                    reduction = np.minimum(excess, imp_vals)
                    modified_data['Pct_Impervious'] = imp_vals - reduction
                    remaining = excess - reduction
                    if np.any(remaining > 0):
                        modified_data['Pct_Canopy'] = (
                            modified_data['Pct_Canopy'].values - remaining
                        )
                    if verbose:
                        n_clipped = int(np.sum(excess > 0))
                        print(f"  Combined constraint: clipped {n_clipped} "
                              f"cells (Canopy+Impervious > 100%)")
                    for vn in ('Pct_Canopy', 'Pct_Impervious'):
                        if vn in actual_changes:
                            actual_changes[vn] = (
                                modified_data[vn].values - data[vn].values
                            )

            # 3. Consensus delta across all simultaneous changes
            joint_delta = self._predict_joint_consensus_delta(
                data, modified_data,
                baseline_base_preds=baseline_base_preds,
            )

            # 4. Physics guardrails per-variable (sign-check the net delta
            #    using the dominant variable's expected direction)
            dominant_var = max(actual_changes, key=lambda v: abs(actual_changes[v].mean()))
            dominant_dir = 1.0 if np.mean(actual_changes[dominant_var]) >= 0 else -1.0
            joint_delta = self._apply_physics_guardrails(
                dominant_var, joint_delta, change_direction=dominant_dir,
            )
            _max_mag = float(self.config.get('caps', {}).get('max_delta_magnitude', 3.0))
            joint_delta = np.clip(joint_delta, -_max_mag, _max_mag)

            # Store per-point results
            label = joint_name.replace(' ', '_')
            results_df[f"repred_delta_{label}"] = joint_delta
            results_df[f"pred_{label}"] = baseline_pred + joint_delta

            # Static PNG map for joint scenario
            joint_map = self._save_scenario_map(
                data, joint_delta, f"JOINT_{label}",
                title=f"Joint: {joint_name}  |  mean Δ = {np.mean(joint_delta):+.4f}",
            )
            if joint_map and verbose:
                print(f"    Map saved: {joint_map.name}")

            joint_mean = float(np.mean(joint_delta))
            joint_std = float(np.std(joint_delta))

            if verbose:
                print(f"\n  {joint_name}:")
                for vn, ac in actual_changes.items():
                    print(f"    {vn}: mean Δ = {ac.mean():+.2f}")
                print(f"    Consensus Δ: {joint_mean:+.4f} ± {joint_std:.4f}")

            for area_id, area_name in self.area_names.items():
                if self.area_column not in data.columns:
                    continue
                mask = data[self.area_column] == area_id
                if mask.sum() == 0:
                    continue
                a_delta = joint_delta[mask.values]
                summary_rows.append({
                    "Variable": "JOINT:" + label,
                    "Scenario": label,
                    "Increment": 0,
                    "Direction": "joint",
                    "Area": area_name,
                    "N_Cells": int(mask.sum()),
                    "Avg_Var_Change": 0.0,
                    "Repred_Delta_Mean": float(a_delta.mean()),
                    "Repred_Delta_Std": float(a_delta.std()),
                    "DAG_Global_Delta": float('nan'),
                    "Method": "base_model_consensus_joint",
                })

            if not self.area_names:
                summary_rows.append({
                    "Variable": "JOINT:" + label,
                    "Scenario": label,
                    "Increment": 0,
                    "Direction": "joint",
                    "Area": "All",
                    "N_Cells": len(data),
                    "Avg_Var_Change": 0.0,
                    "Repred_Delta_Mean": joint_mean,
                    "Repred_Delta_Std": joint_std,
                    "DAG_Global_Delta": float('nan'),
                    "Method": "base_model_consensus_joint",
                })

        # ── Save results ──────────────────────────────────────────────
        summary_df = pd.DataFrame(summary_rows)
        try:
            from sparc.run.pipeline_paths import get_result_store
            store = get_result_store()
            store.save_dataframe(4, "scenario_summary_reprediction.csv", summary_df, fmt="csv")
            store.save_dataframe(4, "scenario_summary.csv", summary_df, fmt="csv")
        except Exception:
            summary_path = self.output_dir / "scenario_summary_reprediction.csv"
            summary_df.to_csv(summary_path, index=False)
        if verbose:
            print(f"\n{'='*70}")
            print(f"Re-prediction scenario summary saved")

        results_gdf = self._to_geodataframe(results_df, data)
        if GEOPANDAS_AVAILABLE and isinstance(results_gdf, gpd.GeoDataFrame):
            try:
                from sparc.run.pipeline_paths import get_result_store
                store = get_result_store()
                store.save_geodataframe(4, "scenario_results_reprediction.gpkg", results_gdf)
                store.save_geodataframe(4, "scenario_results.gpkg", results_gdf)
            except Exception:
                try:
                    gpkg = self.output_dir / "scenario_results_reprediction.gpkg"
                    results_gdf.to_file(gpkg, driver="GPKG")
                    results_gdf.to_file(self.output_dir / "scenario_results.gpkg", driver="GPKG")
                except Exception as e:
                    warnings.warn(f"Could not write GeoPackage: {e}")
            if verbose:
                print(f"GeoPackage saved")

        return summary_df, results_gdf

    # ------------------------------------------------------------------
    # Hybrid: Model re-prediction (direct) + DAG (indirect)
    # ------------------------------------------------------------------

    def run_with_hybrid_reprediction(
        self,
        data: pd.DataFrame,
        verbose: bool = True,
    ) -> Tuple[pd.DataFrame, Optional[Any]]:
        """Execute scenarios using a **hybrid** strategy.

        **Direct effect** — base-model consensus delta.  Each base model
        (OLS, GWR, GWRF, GGPGAM) predicts on the modified feature matrix
        and the per-model deltas are combined using meta-ensemble weights.
        This captures *non-linear* learned responses the coefficient
        approach may miss.

        **Indirect effects** — DAG structural-coefficient propagation
        through mediator paths.  Intermediate edges use global structural
        coefficients (OLS/DML backdoor-adjusted); the final edge to the
        outcome uses per-point MGWR local coefficients for spatial
        heterogeneity.

        This avoids double-counting because ``_predict_consensus_delta``
        only modifies the *intervened* variable in the feature matrix
        (mediator columns stay at baseline), while
        ``_compute_indirect_effects`` explicitly propagates through
        mediator paths of length > 2.

        Returns
        -------
        summary_df : pd.DataFrame
        results_gdf : GeoDataFrame | DataFrame
        """
        from sparc.causal.counterfactual_engine import CounterfactualEngine

        # ── Load DAG ──────────────────────────────────────────────────
        engine = None
        dag_file = self.config.get('causal', {}).get('dag_file')
        if dag_file and Path(dag_file).exists():
            try:
                engine = CounterfactualEngine(self.config)
                engine.load_dag(dag_file)
                engine.fit(data)
            except Exception as e:
                warnings.warn(f"DAG loading failed ({e}); indirect effects unavailable.")
                engine = None
        target_var = engine.roles['outcomes'][0] if engine else self.target_col

        if verbose:
            if engine is not None:
                n_edges = len(engine._structural_coeffs)
                print(f"\n  Hybrid mode: DAG loaded ({n_edges} structural coefficients)")
                for (p, c), v in sorted(engine._structural_coeffs.items()):
                    print(f"      {p:30s} -> {c:15s}  coeff={v:+.6f}")
            else:
                print("\n  Hybrid mode: no DAG — indirect effects will be zero")

        # ── Baseline predictions ──────────────────────────────────────
        if verbose:
            print("\nComputing baseline predictions (all models)...")
        baseline_pred, ols_bl, gwr_bl, gwrf_bl, ggpgam_bl = self._predict_baseline(
            data, verbose=verbose,
        )
        baseline_base_preds: Dict[str, np.ndarray] = {
            "ols": ols_bl, "gwr": gwr_bl, "gwrf": gwrf_bl, "ggpgam": ggpgam_bl,
        }

        results_df = data.copy()
        results_df["pred_baseline"] = baseline_pred

        if verbose:
            self._print_priors_table()
            wstr = ", ".join(f"{k}={v:.3f}" for k, v in self._base_model_weights.items())
            print(f"\n   Consensus weights: {wstr}")
            print(f"\n{'='*70}")
            print("RUNNING SCENARIOS — Hybrid (Model Consensus Direct + DAG Indirect)")
            print(f"{'='*70}")

        summary_rows: List[dict] = []

        # ── Single-variable scenarios ─────────────────────────────────
        for scenario in self.scenarios:
            var_name = scenario["variable"]
            direction = scenario.get("direction", "increase")
            increments = scenario.get("increments", [])
            min_val = scenario.get("min_val", None)
            max_val = scenario.get("max_val", None)
            unit = scenario.get("unit", "")

            if verbose:
                print(f"\n{'─'*50}")
                print(f"Variable: {var_name}  (direction={direction})")
                print(f"{'─'*50}")

            for increment in increments:
                delta_signed = -increment if direction == "decrease" else increment
                label = f"{var_name}_{'minus' if delta_signed < 0 else 'plus'}_{str(increment).replace('.', 'p')}"

                # 1. Modified values with physical bounds
                original_vals = data[var_name].values.copy()
                modified_vals = original_vals + delta_signed
                if min_val is not None:
                    modified_vals = np.maximum(modified_vals, min_val)
                if max_val is not None:
                    modified_vals = np.minimum(modified_vals, max_val)
                actual_change = modified_vals - original_vals

                # 2. Diminishing returns
                threshold = self._get_diminishing_threshold(var_name)
                effective_change = self._diminishing_return(actual_change, threshold)

                # 3. DIRECT: base-model consensus delta (nonlinear)
                direct_delta = self._predict_consensus_delta(
                    data, var_name, modified_vals,
                    baseline_base_preds=baseline_base_preds,
                )
                change_dir = 1.0 if np.mean(actual_change) >= 0 else -1.0
                direct_delta = self._apply_physics_guardrails(
                    var_name, direct_delta, change_direction=change_dir,
                )

                # 4. INDIRECT: DAG mediator pathways + MGWR final edge
                indirect_delta = np.zeros(len(data))
                if engine is not None:
                    indirect_delta = self._compute_indirect_effects(
                        var_name, effective_change, engine, target_var,
                    )

                # 5. Total = direct + indirect, magnitude-capped
                total_delta = direct_delta + indirect_delta
                _max_mag = float(self.config.get('caps', {}).get('max_delta_magnitude', 3.0))
                total_delta = np.clip(total_delta, -_max_mag, _max_mag)

                # 6. Extrapolation guard
                confidence_labels = None
                if EXTRAPOLATION_GUARD_AVAILABLE:
                    try:
                        X_train = data[self.features].values
                        X_scenario = X_train.copy()
                        var_idx = self.features.index(var_name)
                        X_scenario[:, var_idx] = modified_vals
                        extrap_scores = compute_extrapolation_score(X_scenario, X_train)
                        confidence_labels = classify_prediction_confidence(
                            extrap_scores, actual_change, np.std(baseline_pred),
                        )
                        results_df[f"confidence_{label}"] = confidence_labels
                    except Exception as e:
                        if verbose:
                            print(f"      [GUARD] Extrapolation check failed: {e}")

                # 7. DAG-only global delta for comparison
                dag_global_delta = float('nan')
                if engine is not None:
                    try:
                        dag_result = engine.intervene(var_name, delta_signed, data=data)
                        dag_global_delta = dag_result['predicted_delta_response'].iloc[0]
                    except Exception:
                        pass

                # Store per-point results
                results_df[f"direct_{label}"] = direct_delta
                results_df[f"indirect_{label}"] = indirect_delta
                results_df[f"total_{label}"] = total_delta
                results_df[f"pred_{label}"] = baseline_pred + total_delta

                # Map
                map_path = self._save_scenario_map(
                    data, total_delta, f"HYBRID_{label}",
                    title=f"{var_name} {direction} {increment}{unit}  |  "
                          f"mean Δ = {np.mean(total_delta):+.4f} (hybrid)",
                )
                if map_path and verbose:
                    print(f"    Map saved: {map_path.name}")

                # Summary stats
                direct_mean = float(np.mean(direct_delta))
                indirect_mean = float(np.mean(indirect_delta))
                total_mean = float(np.mean(total_delta))
                total_std = float(np.std(total_delta))

                if verbose:
                    n_spec = int(np.sum(confidence_labels == "SPECULATIVE")) if confidence_labels is not None else 0
                    print(f"\n  {label}:")
                    print(f"    Raw Δ:       mean={np.mean(actual_change):+.2f}{unit}  "
                          f"Eff Δ: mean={np.mean(effective_change):+.2f}")
                    print(f"    Direct  (model consensus): {direct_mean:+.4f}")
                    print(f"    Indirect (DAG mediators):   {indirect_mean:+.4f}")
                    print(f"    Total:    {total_mean:+.4f} ± {total_std:.4f}")
                    print(f"    DAG-only: {dag_global_delta:+.4f}  (global, for comparison)")
                    if n_spec > 0:
                        print(f"    [GUARD] {n_spec} SPECULATIVE cells")

                # Per-area summaries
                for area_id, area_name in self.area_names.items():
                    if self.area_column not in data.columns:
                        continue
                    mask = data[self.area_column] == area_id
                    if mask.sum() == 0:
                        continue
                    n_cells = int(mask.sum())
                    a_total = total_delta[mask.values]
                    a_direct = direct_delta[mask.values]
                    a_indirect = indirect_delta[mask.values]
                    a_change = actual_change[mask.values]

                    if verbose:
                        print(f"    {area_name:20s}  total={a_total.mean():+.4f}±{a_total.std():.4f}"
                              f"  (direct={a_direct.mean():+.4f}, indirect={a_indirect.mean():+.4f})"
                              f"  ΔVar={a_change.mean():+.2f}{unit}  (n={n_cells})")

                    summary_rows.append({
                        "Variable": var_name,
                        "Scenario": label,
                        "Increment": increment,
                        "Direction": direction,
                        "Area": area_name,
                        "N_Cells": n_cells,
                        "Avg_Var_Change": float(a_change.mean()),
                        "Direct_Delta_Mean": float(a_direct.mean()),
                        "Indirect_Delta_Mean": float(a_indirect.mean()),
                        "Total_Delta_Mean": float(a_total.mean()),
                        "Total_Delta_Std": float(a_total.std()),
                        "DAG_Global_Delta": dag_global_delta,
                        "Direct_Method": "model_consensus",
                        "Indirect_Method": "dag_mgwr" if engine else "none",
                        "Method": "hybrid",
                    })

                if not self.area_names:
                    summary_rows.append({
                        "Variable": var_name,
                        "Scenario": label,
                        "Increment": increment,
                        "Direction": direction,
                        "Area": "All",
                        "N_Cells": len(data),
                        "Avg_Var_Change": float(np.mean(actual_change)),
                        "Direct_Delta_Mean": direct_mean,
                        "Indirect_Delta_Mean": indirect_mean,
                        "Total_Delta_Mean": total_mean,
                        "Total_Delta_Std": total_std,
                        "DAG_Global_Delta": dag_global_delta,
                        "Direct_Method": "model_consensus",
                        "Indirect_Method": "dag_mgwr" if engine else "none",
                        "Method": "hybrid",
                    })

        # ── Joint scenarios (hybrid mode) ─────────────────────────────
        joint_scenarios = self.config.get('joint_scenarios', [])
        if joint_scenarios and verbose:
            print(f"\n{'='*70}")
            print(f"JOINT SCENARIOS — Hybrid ({len(joint_scenarios)} defined)")
            print(f"{'='*70}")

        for joint in joint_scenarios:
            joint_name = joint.get('name', 'joint_scenario')
            interventions = joint.get('interventions', [])

            if verbose:
                print(f"\n{'─'*50}")
                print(f"Joint Scenario: {joint_name}")
                for iv in interventions:
                    print(f"  {iv['variable']} {iv.get('direction', 'increase')} {iv['increment']}")
                print(f"{'─'*50}")

            # Build jointly-modified DataFrame
            modified_data = data.copy()
            actual_changes: Dict[str, np.ndarray] = {}
            for iv in interventions:
                vn = iv['variable']
                inc = iv['increment']
                d = iv.get('direction', 'increase')
                delta = -inc if d == 'decrease' else inc
                orig = data[vn].values.copy()
                mod = orig + delta
                single_sc = next((s for s in self.scenarios if s['variable'] == vn), {})
                bmin = single_sc.get('min_val')
                bmax = single_sc.get('max_val')
                if bmin is not None:
                    mod = np.maximum(mod, bmin)
                if bmax is not None:
                    mod = np.minimum(mod, bmax)
                modified_data[vn] = mod
                actual_changes[vn] = mod - orig

            # Canopy + Impervious <= 100 constraint
            if ('Pct_Canopy' in modified_data.columns
                    and 'Pct_Impervious' in modified_data.columns):
                total_cover = modified_data['Pct_Canopy'].values + modified_data['Pct_Impervious'].values
                excess = np.maximum(total_cover - 100.0, 0.0)
                if np.any(excess > 0):
                    imp_vals = modified_data['Pct_Impervious'].values
                    reduction = np.minimum(excess, imp_vals)
                    modified_data['Pct_Impervious'] = imp_vals - reduction
                    remaining = excess - reduction
                    if np.any(remaining > 0):
                        modified_data['Pct_Canopy'] = modified_data['Pct_Canopy'].values - remaining
                    if verbose:
                        n_clipped = int(np.sum(excess > 0))
                        print(f"  Combined constraint: clipped {n_clipped} cells")
                    for vn in ('Pct_Canopy', 'Pct_Impervious'):
                        if vn in actual_changes:
                            actual_changes[vn] = modified_data[vn].values - data[vn].values

            # Direct: joint consensus delta
            joint_direct = self._predict_joint_consensus_delta(
                data, modified_data, baseline_base_preds=baseline_base_preds,
            )
            dominant_var = max(actual_changes, key=lambda v: abs(actual_changes[v].mean()))
            dominant_dir = 1.0 if np.mean(actual_changes[dominant_var]) >= 0 else -1.0
            joint_direct = self._apply_physics_guardrails(
                dominant_var, joint_direct, change_direction=dominant_dir,
            )

            # Indirect: sum DAG indirect for each intervened variable
            joint_indirect = np.zeros(len(data))
            if engine is not None:
                co_intervened = set(actual_changes.keys())
                for vn, ac in actual_changes.items():
                    threshold = self._get_diminishing_threshold(vn)
                    eff = self._diminishing_return(ac, threshold)
                    skip = co_intervened - {vn}
                    joint_indirect += self._compute_indirect_effects(
                        vn, eff, engine, target_var, skip_variables=skip,
                    )

            joint_total = joint_direct + joint_indirect
            _max_mag = float(self.config.get('caps', {}).get('max_delta_magnitude', 3.0))
            joint_total = np.clip(joint_total, -_max_mag, _max_mag)

            label = joint_name.replace(' ', '_')
            results_df[f"direct_{label}"] = joint_direct
            results_df[f"indirect_{label}"] = joint_indirect
            results_df[f"total_{label}"] = joint_total
            results_df[f"pred_{label}"] = baseline_pred + joint_total

            joint_map = self._save_scenario_map(
                data, joint_total, f"HYBRID_JOINT_{label}",
                title=f"Joint: {joint_name}  |  mean Δ = {np.mean(joint_total):+.4f} (hybrid)",
            )
            if joint_map and verbose:
                print(f"    Map saved: {joint_map.name}")

            if verbose:
                print(f"\n  {joint_name}:")
                for vn, ac in actual_changes.items():
                    print(f"    {vn}: mean Δ = {ac.mean():+.2f}")
                print(f"    Direct  (consensus): {float(np.mean(joint_direct)):+.4f}")
                print(f"    Indirect (DAG):      {float(np.mean(joint_indirect)):+.4f}")
                print(f"    Total:               {float(np.mean(joint_total)):+.4f}")

            for area_id, area_name in self.area_names.items():
                if self.area_column not in data.columns:
                    continue
                mask = data[self.area_column] == area_id
                if mask.sum() == 0:
                    continue
                a_delta = joint_total[mask.values]
                summary_rows.append({
                    "Variable": "JOINT:" + label,
                    "Scenario": label,
                    "Increment": 0,
                    "Direction": "joint",
                    "Area": area_name,
                    "N_Cells": int(mask.sum()),
                    "Avg_Var_Change": 0.0,
                    "Direct_Delta_Mean": float(joint_direct[mask.values].mean()),
                    "Indirect_Delta_Mean": float(joint_indirect[mask.values].mean()),
                    "Total_Delta_Mean": float(a_delta.mean()),
                    "Total_Delta_Std": float(a_delta.std()),
                    "DAG_Global_Delta": float('nan'),
                    "Direct_Method": "model_consensus_joint",
                    "Indirect_Method": "dag_mgwr" if engine else "none",
                    "Method": "hybrid_joint",
                })

            if not self.area_names:
                summary_rows.append({
                    "Variable": "JOINT:" + label,
                    "Scenario": label,
                    "Increment": 0,
                    "Direction": "joint",
                    "Area": "All",
                    "N_Cells": len(data),
                    "Avg_Var_Change": 0.0,
                    "Direct_Delta_Mean": float(np.mean(joint_direct)),
                    "Indirect_Delta_Mean": float(np.mean(joint_indirect)),
                    "Total_Delta_Mean": float(np.mean(joint_total)),
                    "Total_Delta_Std": float(np.std(joint_total)),
                    "DAG_Global_Delta": float('nan'),
                    "Direct_Method": "model_consensus_joint",
                    "Indirect_Method": "dag_mgwr" if engine else "none",
                    "Method": "hybrid_joint",
                })

        # ── Save results ──────────────────────────────────────────────
        summary_df = pd.DataFrame(summary_rows)
        # ── Save results ──────────────────────────────────────────────
        summary_df = pd.DataFrame(summary_rows)
        try:
            from sparc.run.pipeline_paths import get_result_store
            store = get_result_store()
            store.save_dataframe(4, "scenario_summary_hybrid.csv", summary_df, fmt="csv")
            store.save_dataframe(4, "scenario_summary.csv", summary_df, fmt="csv")
        except Exception:
            summary_path = self.output_dir / "scenario_summary_hybrid.csv"
            summary_df.to_csv(summary_path, index=False)
        if verbose:
            print(f"\n{'='*70}")
            print(f"Hybrid scenario summary saved")

        results_gdf = self._to_geodataframe(results_df, data)
        if GEOPANDAS_AVAILABLE and isinstance(results_gdf, gpd.GeoDataFrame):
            try:
                from sparc.run.pipeline_paths import get_result_store
                store = get_result_store()
                store.save_geodataframe(4, "scenario_results_hybrid.gpkg", results_gdf)
                store.save_geodataframe(4, "scenario_results.gpkg", results_gdf)
            except Exception:
                try:
                    gpkg = self.output_dir / "scenario_results_hybrid.gpkg"
                    results_gdf.to_file(gpkg, driver="GPKG")
                    results_gdf.to_file(self.output_dir / "scenario_results.gpkg", driver="GPKG")
                except Exception as e:
                    warnings.warn(f"Could not write GeoPackage: {e}")
            if verbose:
                print(f"GeoPackage saved")

        return summary_df, results_gdf

    # ------------------------------------------------------------------
    # Scenario Interpretation Maps
    # ------------------------------------------------------------------

    def _generate_interpretation_maps(
        self,
        results_df: pd.DataFrame,
        data: pd.DataFrame,
        verbose: bool = True,
    ) -> None:
        """Generate policy-oriented interpretation maps from DAG scenario results.

        Creates three types of maps:
        1. **Per-scenario maps** — one per (variable, increment) showing
           the spatially-varying total cooling effect.
        2. **Headline maps** — one per variable at a moderate dose,
           styled for presentation (where to plant trees / remove pavement /
           brighten surfaces).
        3. **Dashboard** — a single multi-panel figure comparing all three
           intervention types side-by-side at their moderate dose.

        Maps use a "cooling benefit" framing: negative z-score deltas
        (= cooling) are shown as positive benefit in warm colours so that
        darker/warmer colours = "intervene here first".
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.colors import Normalize
            from matplotlib import cm
        except ImportError:
            if verbose:
                print("  [MAPS] matplotlib not available — skipping map generation")
            return

        maps_dir = self.output_dir / "scenario_maps"
        maps_dir.mkdir(parents=True, exist_ok=True)

        x_col, y_col = self.coord_cols[0], self.coord_cols[1]
        x = data[x_col].values
        y = data[y_col].values

        response_units = self.config.get("project", {}).get("response_units", "z-score")

        # ── Friendly labels for policy communication ──────────────
        friendly = {
            "Pct_Canopy": {
                "action": "Plant Trees",
                "short": "Canopy",
                "icon": "🌳",
                "description": "Increase tree canopy cover",
            },
            "Pct_Impervious": {
                "action": "Remove Pavement",
                "short": "Impervious",
                "icon": "🏗️",
                "description": "Reduce impervious surface",
            },
            "Albedo": {
                "action": "Brighten Surfaces",
                "short": "Albedo",
                "icon": "☀️",
                "description": "Increase surface reflectivity",
            },
        }

        # ── Pick a moderate increment per variable for headline maps ──
        moderate_labels = {}
        for scenario in self.scenarios:
            var = scenario["variable"]
            increments = scenario.get("increments", [])
            direction = scenario.get("direction", "increase")
            if not increments:
                continue
            # Pick the increment closest to 1/3 of the range
            mid_target = increments[0] + (increments[-1] - increments[0]) / 3
            moderate_inc = min(increments, key=lambda i: abs(i - mid_target))
            delta_signed = -moderate_inc if direction == "decrease" else moderate_inc
            label = f"{var}_{'minus' if delta_signed < 0 else 'plus'}_{str(moderate_inc).replace('.', 'p')}"
            moderate_labels[var] = {
                "label": label,
                "increment": moderate_inc,
                "direction": direction,
                "unit": scenario.get("unit", ""),
            }

        if verbose:
            print(f"\n{'='*70}")
            print("GENERATING INTERPRETATION MAPS")
            print(f"{'='*70}")

        # ── 1. Per-scenario total-effect maps (all increments) ────────
        total_cols = [c for c in results_df.columns if c.startswith("total_")]
        for col in total_cols:
            scenario_label = col.replace("total_", "")
            values = results_df[col].values

            # Cooling benefit = -delta (negative delta = temperature drop = benefit)
            benefit = -values

            # Determine variable name for title
            var_name = scenario_label.split("_plus_")[0].split("_minus_")[0]
            info = friendly.get(var_name, {"action": var_name, "short": var_name})

            # Parse increment from label
            if "_plus_" in scenario_label:
                inc_str = scenario_label.split("_plus_")[1].replace("p", ".")
                sign_word = "+"
            elif "_minus_" in scenario_label:
                inc_str = scenario_label.split("_minus_")[1].replace("p", ".")
                sign_word = "−"
            else:
                inc_str = ""
                sign_word = ""

            title = f"{info['action']}: {sign_word}{inc_str} — Cooling Benefit"

            # Colour scale: 0 to max benefit (sequential, not diverging)
            vmin_b = float(np.percentile(benefit, 2))
            vmax_b = float(np.percentile(benefit, 98))
            # If overall benefit is positive (cooling), anchor at 0
            if vmax_b > 0:
                vmin_b = max(vmin_b, 0.0)

            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            sc = ax.scatter(
                x, y, c=benefit, cmap="YlOrRd", s=0.5,
                vmin=vmin_b, vmax=vmax_b, edgecolors="none",
            )
            cbar = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
            cbar.set_label(f"Cooling Benefit ({response_units})", fontsize=10)
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=8)
            fig.tight_layout()

            out_path = maps_dir / f"{scenario_label}_cooling.png"
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)

        if verbose:
            print(f"  {len(total_cols)} per-scenario cooling maps saved")

        # ── 2. Headline maps (moderate dose, presentation quality) ────
        headline_data = {}  # var → benefit array (for dashboard)
        for var, meta in moderate_labels.items():
            col_name = f"total_{meta['label']}"
            if col_name not in results_df.columns:
                continue

            values = results_df[col_name].values
            benefit = -values  # cooling benefit

            info = friendly.get(var, {"action": var, "short": var, "description": var})
            inc = meta["increment"]
            unit = meta["unit"]
            dir_word = "+" if meta["direction"] == "increase" else "−"

            headline_data[var] = benefit

            # Percentile stats for annotation
            p50 = float(np.median(benefit))
            p90 = float(np.percentile(benefit, 90))
            mean_b = float(np.mean(benefit))

            # Colour scale: anchor at 0, go up to 98th percentile
            vmax_b = float(np.percentile(benefit, 98))
            if vmax_b <= 0:
                vmax_b = float(np.percentile(np.abs(benefit), 98))

            fig, ax = plt.subplots(1, 1, figsize=(11, 9))
            sc = ax.scatter(
                x, y, c=benefit, cmap="YlOrRd", s=0.6,
                vmin=0, vmax=vmax_b, edgecolors="none",
            )
            cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
            cbar.set_label(f"Cooling Benefit ({response_units})", fontsize=11)

            ax.set_title(
                f"Where to {info['action']}\n"
                f"Scenario: {dir_word}{inc} {unit}",
                fontsize=14, fontweight="bold",
            )

            # Annotation box with summary stats
            stats_text = (
                f"Mean benefit: {mean_b:.3f} {response_units}\n"
                f"Median: {p50:.3f}\n"
                f"90th pctl: {p90:.3f}\n"
                f"(darker = more cooling)"
            )
            ax.text(
                0.02, 0.02, stats_text,
                transform=ax.transAxes, fontsize=9,
                verticalalignment="bottom",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
            )

            ax.set_xlabel(x_col, fontsize=9)
            ax.set_ylabel(y_col, fontsize=9)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=8)
            fig.tight_layout()

            out_path = maps_dir / f"headline_{var}_cooling.png"
            fig.savefig(out_path, dpi=250, bbox_inches="tight")
            plt.close(fig)

            if verbose:
                print(f"  Headline map: {out_path.name}  "
                      f"(mean={mean_b:.4f}, 90th={p90:.4f})")

        # ── 3. Dashboard — all interventions side by side ─────────────
        if len(headline_data) >= 2:
            n_panels = len(headline_data)
            fig, axes = plt.subplots(
                1, n_panels, figsize=(6 * n_panels + 1, 7),
                constrained_layout=True,
            )
            if n_panels == 1:
                axes = [axes]

            # Shared colour scale across all panels
            all_benefits = np.concatenate(list(headline_data.values()))
            shared_vmax = float(np.percentile(all_benefits[all_benefits > 0], 98)) if np.any(all_benefits > 0) else 0.1

            for ax, (var, benefit) in zip(axes, headline_data.items()):
                info = friendly.get(var, {"action": var, "short": var})
                meta = moderate_labels[var]
                dir_word = "+" if meta["direction"] == "increase" else "−"

                sc = ax.scatter(
                    x, y, c=benefit, cmap="YlOrRd", s=0.3,
                    vmin=0, vmax=shared_vmax, edgecolors="none",
                )
                ax.set_title(
                    f"{info['action']}\n({dir_word}{meta['increment']} {meta['unit']})",
                    fontsize=12, fontweight="bold",
                )
                ax.set_aspect("equal")
                ax.tick_params(labelsize=7)
                mean_b = float(np.mean(benefit))
                ax.text(
                    0.02, 0.02, f"mean = {mean_b:.3f}",
                    transform=ax.transAxes, fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3"),
                )

            # Shared colorbar
            cbar = fig.colorbar(
                sc, ax=axes, shrink=0.85, pad=0.02,
                label=f"Cooling Benefit ({response_units})",
            )

            fig.suptitle(
                "Where Interventions Help Most — Spatial Cooling Benefit",
                fontsize=15, fontweight="bold", y=1.02,
            )

            dashboard_path = maps_dir / "dashboard_cooling_benefit.png"
            fig.savefig(dashboard_path, dpi=250, bbox_inches="tight")
            plt.close(fig)

            if verbose:
                print(f"  Dashboard: {dashboard_path.name}")

        # ── 4. Joint scenario maps ────────────────────────────────────
        joint_cols = [c for c in results_df.columns
                      if c.startswith("total_") and c.replace("total_", "") not in
                      [tc.replace("total_", "") for tc in total_cols
                       if any(tc.replace("total_", "").startswith(s["variable"])
                              for s in self.scenarios)]]
        # Simpler: find joint labels
        single_var_prefixes = tuple(s["variable"] for s in self.scenarios)
        for col in results_df.columns:
            if not col.startswith("total_"):
                continue
            scenario_label = col.replace("total_", "")
            # Skip single-variable scenarios (already mapped above)
            if any(scenario_label.startswith(p) for p in single_var_prefixes):
                continue

            benefit = -results_df[col].values
            title_name = scenario_label.replace("_", " ")

            vmax_b = float(np.percentile(benefit, 98)) if np.any(benefit > 0) else 0.1
            fig, ax = plt.subplots(1, 1, figsize=(11, 9))
            sc = ax.scatter(
                x, y, c=benefit, cmap="YlOrRd", s=0.6,
                vmin=0, vmax=max(vmax_b, 0.01), edgecolors="none",
            )
            cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
            cbar.set_label(f"Cooling Benefit ({response_units})", fontsize=11)
            ax.set_title(
                f"Joint Scenario: {title_name}\nCombined Cooling Benefit",
                fontsize=14, fontweight="bold",
            )
            mean_b = float(np.mean(benefit))
            p90 = float(np.percentile(benefit, 90))
            stats_text = (
                f"Mean benefit: {mean_b:.3f} {response_units}\n"
                f"90th pctl: {p90:.3f}\n"
                f"(darker = more cooling)"
            )
            ax.text(
                0.02, 0.02, stats_text,
                transform=ax.transAxes, fontsize=9,
                verticalalignment="bottom",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
            )
            ax.set_xlabel(x_col, fontsize=9)
            ax.set_ylabel(y_col, fontsize=9)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=8)
            fig.tight_layout()

            out_path = maps_dir / f"joint_{scenario_label}_cooling.png"
            fig.savefig(out_path, dpi=250, bbox_inches="tight")
            plt.close(fig)

            if verbose:
                print(f"  Joint map: {out_path.name}  (mean={mean_b:.4f})")

        # ── 5. Existing conditions vs. cooling benefit + priority maps ─
        #
        # For each intervention variable, compare:
        #   - Current value (where canopy/impervious/albedo already is)
        #   - Cooling benefit (from the causal model)
        #   - Headroom (how much room for the intervention)
        #   - Priority = benefit × headroom (actionable priority score)
        #
        # This answers: "Are we recommending planting trees where they
        # already exist, or where there's actually room to plant more?"

        # Define headroom logic per variable
        headroom_config = {
            "Pct_Canopy": {
                # Headroom = how much canopy CAN be added (up to max_val)
                "current_label": "Current Canopy Cover (%)",
                "headroom_label": "Room to Add Canopy (%)",
                "priority_label": "Priority: Where to Plant Trees",
                "cmap_current": "Greens",
                "max_val": 100.0,
                "direction": "increase",  # benefit comes from increasing
            },
            "Pct_Impervious": {
                # Headroom = how much impervious CAN be removed (down to min_val)
                "current_label": "Current Impervious Surface (%)",
                "headroom_label": "Removable Pavement (%)",
                "priority_label": "Priority: Where to Remove Pavement",
                "cmap_current": "Greys",
                "min_val": 0.0,
                "direction": "decrease",  # benefit comes from decreasing
            },
            "Albedo": {
                # Headroom = how much albedo CAN be increased
                "current_label": "Current Albedo (reflectance)",
                "headroom_label": "Room to Increase Albedo",
                "priority_label": "Priority: Where to Brighten Surfaces",
                "cmap_current": "YlGn",
                "max_val": 0.70,
                "direction": "increase",
            },
        }

        priority_data = {}  # var -> priority array (for combined dashboard)

        for var, meta in moderate_labels.items():
            if var not in data.columns or var not in headroom_config:
                continue

            hcfg = headroom_config[var]
            col_name = f"total_{meta['label']}"
            if col_name not in results_df.columns:
                continue

            current_vals = data[var].values.astype(float)
            benefit = -results_df[col_name].values  # cooling benefit (positive = good)
            info = friendly.get(var, {"action": var, "short": var})

            # Compute headroom
            if hcfg["direction"] == "increase":
                max_v = hcfg.get("max_val", float(current_vals.max()))
                headroom = max_v - current_vals
            else:
                min_v = hcfg.get("min_val", float(current_vals.min()))
                headroom = current_vals - min_v

            # Normalise headroom to [0, 1]
            headroom = np.clip(headroom, 0, None)
            headroom_max = headroom.max() if headroom.max() > 0 else 1.0
            headroom_norm = headroom / headroom_max

            # Normalise benefit to [0, 1]
            benefit_clipped = np.clip(benefit, 0, None)
            benefit_max = np.percentile(benefit_clipped, 98) if benefit_clipped.max() > 0 else 1.0
            benefit_norm = np.clip(benefit_clipped / benefit_max, 0, 1)

            # Priority = benefit × headroom (both matter)
            priority = benefit_norm * headroom_norm
            priority_data[var] = priority

            dir_word = "+" if meta["direction"] == "increase" else "-"

            # --- 4-panel figure: Current | Headroom | Benefit | Priority ---
            fig, axes = plt.subplots(2, 2, figsize=(16, 14), constrained_layout=True)

            # Panel 1: Current conditions
            ax = axes[0, 0]
            sc1 = ax.scatter(
                x, y, c=current_vals, cmap=hcfg["cmap_current"], s=0.4,
                edgecolors="none",
            )
            plt.colorbar(sc1, ax=ax, shrink=0.7, pad=0.02, label=hcfg["current_label"])
            ax.set_title(f"A) {hcfg['current_label']}", fontsize=12, fontweight="bold")
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7)
            mean_cur = float(current_vals.mean())
            ax.text(0.02, 0.02, f"mean = {mean_cur:.1f}", transform=ax.transAxes,
                    fontsize=8, bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3"))

            # Panel 2: Headroom (room for intervention)
            ax = axes[0, 1]
            sc2 = ax.scatter(
                x, y, c=headroom, cmap="Blues", s=0.4, edgecolors="none",
            )
            plt.colorbar(sc2, ax=ax, shrink=0.7, pad=0.02, label=hcfg["headroom_label"])
            ax.set_title(f"B) {hcfg['headroom_label']}", fontsize=12, fontweight="bold")
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7)
            mean_hr = float(headroom.mean())
            ax.text(0.02, 0.02, f"mean = {mean_hr:.2f}", transform=ax.transAxes,
                    fontsize=8, bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3"))

            # Panel 3: Cooling benefit (from causal model)
            ax = axes[1, 0]
            vmax_b = float(np.percentile(benefit_clipped, 98)) if benefit_clipped.max() > 0 else 0.1
            sc3 = ax.scatter(
                x, y, c=benefit_clipped, cmap="YlOrRd", s=0.4,
                vmin=0, vmax=vmax_b, edgecolors="none",
            )
            plt.colorbar(sc3, ax=ax, shrink=0.7, pad=0.02,
                         label=f"Cooling Benefit ({response_units})")
            ax.set_title(
                f"C) Causal Cooling Benefit ({dir_word}{meta['increment']} {meta['unit']})",
                fontsize=12, fontweight="bold",
            )
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7)
            mean_b = float(benefit_clipped.mean())
            ax.text(0.02, 0.02, f"mean = {mean_b:.3f}", transform=ax.transAxes,
                    fontsize=8, bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3"))

            # Panel 4: PRIORITY (benefit × headroom)
            ax = axes[1, 1]
            p98 = float(np.percentile(priority, 98)) if priority.max() > 0 else 0.1
            sc4 = ax.scatter(
                x, y, c=priority, cmap="hot_r", s=0.4,
                vmin=0, vmax=p98, edgecolors="none",
            )
            plt.colorbar(sc4, ax=ax, shrink=0.7, pad=0.02, label="Priority Score")
            ax.set_title(f"D) {hcfg['priority_label']}", fontsize=12, fontweight="bold")
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7)
            # Annotate: high priority = high benefit AND room to act
            ax.text(0.02, 0.02,
                    "Priority = Benefit x Headroom\n(dark = act here first)",
                    transform=ax.transAxes, fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.85, boxstyle="round,pad=0.3"))

            fig.suptitle(
                f"{info['action']}: Existing Conditions vs. Intervention Priority",
                fontsize=15, fontweight="bold", y=1.01,
            )

            out_path = maps_dir / f"priority_{var}.png"
            fig.savefig(out_path, dpi=250, bbox_inches="tight")
            plt.close(fig)

            if verbose:
                # Correlation between current conditions and benefit
                corr = float(np.corrcoef(current_vals, benefit_clipped)[0, 1])
                high_priority_pct = float((priority > np.percentile(priority, 75)).mean() * 100)
                print(f"  Priority map: {out_path.name}")
                print(f"    Corr(current, benefit) = {corr:+.3f}  "
                      f"({'benefit where already high' if corr > 0.3 else 'benefit where currently low' if corr < -0.3 else 'spatially independent'})")
                print(f"    Top-25% priority cells: {high_priority_pct:.0f}% of area")

        # --- Combined priority dashboard (all variables) ---
        if len(priority_data) >= 2:
            n_p = len(priority_data)
            fig, axes = plt.subplots(1, n_p, figsize=(6 * n_p + 1, 7), constrained_layout=True)
            if n_p == 1:
                axes = [axes]

            for ax, (var, priority) in zip(axes, priority_data.items()):
                info = friendly.get(var, {"action": var, "short": var})
                p98 = float(np.percentile(priority, 98)) if priority.max() > 0 else 0.1
                sc = ax.scatter(
                    x, y, c=priority, cmap="hot_r", s=0.3,
                    vmin=0, vmax=p98, edgecolors="none",
                )
                ax.set_title(f"{info['action']}", fontsize=12, fontweight="bold")
                ax.set_aspect("equal")
                ax.tick_params(labelsize=7)

            cbar = fig.colorbar(sc, ax=axes, shrink=0.85, pad=0.02, label="Priority Score")
            fig.suptitle(
                "Intervention Priority: Cooling Benefit x Headroom",
                fontsize=15, fontweight="bold", y=1.02,
            )

            fig.savefig(maps_dir / "dashboard_priority.png", dpi=250, bbox_inches="tight")
            plt.close(fig)

            if verbose:
                print(f"  Priority dashboard: dashboard_priority.png")

        # ── 6. Baseline + post-intervention absolute temperature maps ──
        baseline = results_df.get("pred_baseline")
        if baseline is not None:
            baseline_vals = baseline.values

            # Shared colour range across baseline and all post-intervention maps
            temp_vmin = float(np.percentile(baseline_vals, 1))
            temp_vmax = float(np.percentile(baseline_vals, 99))

            # 5a. Baseline temperature surface
            fig, ax = plt.subplots(1, 1, figsize=(11, 9))
            sc = ax.scatter(
                x, y, c=baseline_vals, cmap="RdYlBu_r", s=0.6,
                vmin=temp_vmin, vmax=temp_vmax, edgecolors="none",
            )
            cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
            cbar.set_label(f"Predicted Temperature ({response_units})", fontsize=11)
            ax.set_title(
                "Baseline Predicted Temperature\n(Current Conditions)",
                fontsize=14, fontweight="bold",
            )
            stats_text = (
                f"Mean: {baseline_vals.mean():.3f}\n"
                f"Range: [{baseline_vals.min():.3f}, {baseline_vals.max():.3f}]\n"
                f"(warmer = red, cooler = blue)"
            )
            ax.text(
                0.02, 0.02, stats_text,
                transform=ax.transAxes, fontsize=9,
                verticalalignment="bottom",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
            )
            ax.set_xlabel(x_col, fontsize=9)
            ax.set_ylabel(y_col, fontsize=9)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=8)
            fig.tight_layout()
            fig.savefig(maps_dir / "baseline_temperature.png", dpi=250, bbox_inches="tight")
            plt.close(fig)

            if verbose:
                print(f"  Baseline temperature map saved")

            # 5b. Post-intervention temperature maps (headline scenarios)
            for var, meta in moderate_labels.items():
                pred_col = f"pred_{meta['label']}"
                if pred_col not in results_df.columns:
                    continue

                post_vals = results_df[pred_col].values
                info = friendly.get(var, {"action": var, "short": var})
                dir_word = "+" if meta["direction"] == "increase" else "−"

                fig, ax = plt.subplots(1, 1, figsize=(11, 9))
                sc = ax.scatter(
                    x, y, c=post_vals, cmap="RdYlBu_r", s=0.6,
                    vmin=temp_vmin, vmax=temp_vmax, edgecolors="none",
                )
                cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
                cbar.set_label(f"Predicted Temperature ({response_units})", fontsize=11)
                ax.set_title(
                    f"Post-Intervention Temperature: {info['action']}\n"
                    f"Scenario: {dir_word}{meta['increment']} {meta['unit']}",
                    fontsize=14, fontweight="bold",
                )
                delta_vals = post_vals - baseline_vals
                stats_text = (
                    f"Mean temp: {post_vals.mean():.3f}\n"
                    f"Mean change: {delta_vals.mean():+.3f}\n"
                    f"Max cooling: {delta_vals.min():+.3f}\n"
                    f"(same scale as baseline)"
                )
                ax.text(
                    0.02, 0.02, stats_text,
                    transform=ax.transAxes, fontsize=9,
                    verticalalignment="bottom",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
                )
                ax.set_xlabel(x_col, fontsize=9)
                ax.set_ylabel(y_col, fontsize=9)
                ax.set_aspect("equal")
                ax.tick_params(labelsize=8)
                fig.tight_layout()

                out_path = maps_dir / f"post_intervention_{var}_temperature.png"
                fig.savefig(out_path, dpi=250, bbox_inches="tight")
                plt.close(fig)

                if verbose:
                    print(f"  Post-intervention map: {out_path.name}  "
                          f"(mean Δ={delta_vals.mean():+.4f})")

            # 5c. Before/After dashboard — baseline vs post-intervention
            n_vars = len(moderate_labels)
            if n_vars >= 1:
                fig, axes = plt.subplots(
                    2, n_vars, figsize=(6 * n_vars + 1, 12),
                    constrained_layout=True,
                )
                if n_vars == 1:
                    axes = axes.reshape(2, 1)

                for col_idx, (var, meta) in enumerate(moderate_labels.items()):
                    pred_col = f"pred_{meta['label']}"
                    if pred_col not in results_df.columns:
                        continue

                    post_vals = results_df[pred_col].values
                    info = friendly.get(var, {"action": var, "short": var})
                    dir_word = "+" if meta["direction"] == "increase" else "−"

                    # Top row: baseline
                    ax_top = axes[0, col_idx]
                    sc = ax_top.scatter(
                        x, y, c=baseline_vals, cmap="RdYlBu_r", s=0.3,
                        vmin=temp_vmin, vmax=temp_vmax, edgecolors="none",
                    )
                    ax_top.set_title("Current", fontsize=11, fontweight="bold")
                    ax_top.set_aspect("equal")
                    ax_top.tick_params(labelsize=6)
                    if col_idx == 0:
                        ax_top.set_ylabel("BEFORE", fontsize=12, fontweight="bold")

                    # Bottom row: post-intervention
                    ax_bot = axes[1, col_idx]
                    sc = ax_bot.scatter(
                        x, y, c=post_vals, cmap="RdYlBu_r", s=0.3,
                        vmin=temp_vmin, vmax=temp_vmax, edgecolors="none",
                    )
                    delta_mean = float(np.mean(post_vals - baseline_vals))
                    ax_bot.set_title(
                        f"{info['action']} ({dir_word}{meta['increment']})\n"
                        f"mean Δ = {delta_mean:+.3f}",
                        fontsize=11, fontweight="bold",
                    )
                    ax_bot.set_aspect("equal")
                    ax_bot.tick_params(labelsize=6)
                    if col_idx == 0:
                        ax_bot.set_ylabel("AFTER", fontsize=12, fontweight="bold")

                # Shared colorbar
                cbar = fig.colorbar(
                    sc, ax=axes, shrink=0.6, pad=0.02,
                    label=f"Predicted Temperature ({response_units})",
                )

                fig.suptitle(
                    "Before & After — Predicted Temperature Surface",
                    fontsize=15, fontweight="bold", y=1.02,
                )

                dashboard_path = maps_dir / "dashboard_before_after.png"
                fig.savefig(dashboard_path, dpi=250, bbox_inches="tight")
                plt.close(fig)

                if verbose:
                    print(f"  Before/After dashboard: {dashboard_path.name}")

        if verbose:
            print(f"\n  All maps saved to: {maps_dir}")

    # ------------------------------------------------------------------
    # Static PNG map generation (legacy helper)
    # ------------------------------------------------------------------

    def _save_scenario_map(
        self,
        data: pd.DataFrame,
        values: np.ndarray,
        label: str,
        title: str,
        cmap: str = "RdBu_r",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> Optional[Path]:
        """Save a static PNG map of a scenario delta (or prediction).

        Parameters
        ----------
        data : pd.DataFrame
            Must contain coordinate columns for plotting.
        values : ndarray
            Per-point values to colour-code.
        label : str
            Filename-safe label for the scenario (used in filename).
        title : str
            Human-readable title placed above the map.
        cmap : str
            Matplotlib colourmap name.
        vmin, vmax : float, optional
            Explicit colour limits.  If omitted, symmetric limits are
            derived from the 2nd/98th percentile of *values*.

        Returns
        -------
        Path or None
            Path to the saved PNG, or None if matplotlib is unavailable.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        maps_dir = self.output_dir / "scenario_maps"
        maps_dir.mkdir(parents=True, exist_ok=True)

        # Coordinates: prefer projected coords
        x_col, y_col = self.coord_cols[0], self.coord_cols[1]
        x = data[x_col].values
        y = data[y_col].values

        # Symmetric colour limits for deltas
        if vmin is None or vmax is None:
            abs_max = max(abs(np.percentile(values, 2)),
                         abs(np.percentile(values, 98)),
                         1e-6)
            vmin = -abs_max
            vmax = abs_max

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        sc = ax.scatter(x, y, c=values, cmap=cmap, s=0.5,
                        vmin=vmin, vmax=vmax, edgecolors="none")
        cbar = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)

        response_units = self.config.get("response_units", "")
        cbar.set_label(f"Delta ({response_units})" if response_units else "Delta")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_aspect("equal")
        fig.tight_layout()

        out_path = maps_dir / f"{label}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def _print_priors_table(self) -> None:
        """Print a formatted table of physics priors (literature values)."""
        w = self.literature_weight
        print(f"\n{'='*70}")
        print("PHYSICS PRIORS (Literature Coefficients)")
        print(f"{'='*70}")
        print(f"\n   Literature weight: {w*100:.0f}%")
        print(f"\n   {'Variable':25s} {'Lit Coef':>10s}  {'Blended':>10s}  {'Direction':>10s}  {'per':>10s}")
        print("   " + "-"*70)
        for var, prior in self.physics_priors.items():
            lit = prior["lit_coef"]
            hybrid = prior["coefficient"]
            direction = prior["direction"]
            unit = prior["unit_increment"]
            if "Pct" in var:
                unit_str = f"{unit:.0f}pp"
            else:
                unit_str = str(unit)
            print(f"   {var:25s} {lit:+.4f}      {hybrid:+.4f}      {direction:>10s}  per {unit_str}")
        if self._mgwr_coefficients_raw is not None:
            print(f"\n   MGWR local coefficients: LOADED ({self._mgwr_coefficients_raw.shape[0]} points)")
        else:
            print(f"\n   MGWR local coefficients: NOT AVAILABLE")

    # ==================================================================
    # Monte-Carlo Uncertainty Propagation (Steps 12-13)
    # ==================================================================

    def run_with_uncertainty(
        self,
        data: pd.DataFrame,
        n_mc: int = 50,
        coeff_cv: float = 0.15,
        verbose: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Wrap ``run_with_causal_dag()`` in a Monte-Carlo loop that
        perturbs structural coefficients and physics priors to produce
        credible intervals for every scenario delta.

        Parameters
        ----------
        data : pd.DataFrame
            Baseline spatial data.
        n_mc : int
            Number of Monte-Carlo draws (default 50).
        coeff_cv : float
            Coefficient of variation for Gaussian noise applied to each
            structural coefficient per draw (default 0.15 = ±15 %).
        verbose : bool
            Print progress every 10 iterations.

        Returns
        -------
        summary_df : pd.DataFrame
            Scenario summary with added ``*_q05``, ``*_q50``, ``*_q95``
            columns for each delta column.
        mc_meta : dict
            Metadata about the MC run (n_mc, columns perturbed, etc.).
        """
        import copy

        if verbose:
            print(f"\n{'='*70}")
            print(f"MONTE-CARLO UNCERTAINTY PROPAGATION  (n_mc={n_mc})")
            print(f"{'='*70}")

        # Store originals
        orig_coeffs = dict(self.causal_coefficients) if self.causal_coefficients else {}
        orig_priors = copy.deepcopy(self.physics_priors)

        # Collect per-cell delta columns across MC draws
        delta_cols: Dict[str, List[np.ndarray]] = {}

        for i in range(n_mc):
            if verbose and (i % 10 == 0):
                print(f"  MC draw {i+1}/{n_mc} ...")

            # Perturb structural coefficients
            perturbed_coeffs = {}
            for key, val in orig_coeffs.items():
                noise = np.random.normal(1.0, coeff_cv)
                perturbed_coeffs[key] = val * noise
            self.causal_coefficients = perturbed_coeffs

            # Perturb physics priors (literature coefficients)
            perturbed_priors = copy.deepcopy(orig_priors)
            for var, pinfo in perturbed_priors.items():
                noise = np.random.normal(1.0, coeff_cv)
                pinfo['lit_coef'] *= noise
                pinfo['coefficient'] *= noise
            self.physics_priors = perturbed_priors

            # Run the DAG-based simulation
            try:
                results_df, _, _ = self.run_with_causal_dag(
                    data, verbose=False,
                )
                # Collect all "total_*" columns
                for col in results_df.columns:
                    if col.startswith('total_'):
                        delta_cols.setdefault(col, []).append(
                            results_df[col].values.copy()
                        )
            except Exception as e:
                if verbose:
                    print(f"    MC draw {i+1} failed: {e}")

        # Restore originals
        self.causal_coefficients = orig_coeffs
        self.physics_priors = orig_priors

        # Compute quantiles
        if verbose:
            print(f"\n  Computing credible intervals ...")

        # Run once more with original coefficients for the median baseline
        results_df, summary_df, _ = self.run_with_causal_dag(
            data, verbose=False,
        )

        for col, draws in delta_cols.items():
            stacked = np.stack(draws, axis=0)  # (n_mc, n_cells)
            results_df[f"{col}_q05"] = np.percentile(stacked, 5, axis=0)
            results_df[f"{col}_q50"] = np.percentile(stacked, 50, axis=0)
            results_df[f"{col}_q95"] = np.percentile(stacked, 95, axis=0)

        # Augment summary with aggregated quantiles
        for col, draws in delta_cols.items():
            means = [float(np.mean(d)) for d in draws]
            if summary_df is not None and col.replace('total_', '') != '':
                # Append MC columns to summary
                pass  # per-area breakdown is complex; store global MC stats

        mc_meta = {
            'n_mc': n_mc,
            'coeff_cv': coeff_cv,
            'n_successful_draws': len(next(iter(delta_cols.values()), [])),
            'delta_columns_tracked': list(delta_cols.keys()),
        }

        if verbose:
            print(f"  MC complete: {mc_meta['n_successful_draws']}/{n_mc} "
                  f"successful draws across {len(delta_cols)} delta columns.")

        # Save MC results
        mc_path = self.output_dir / "scenario_mc_uncertainty.csv"
        mc_cols = [c for c in results_df.columns
                   if c.endswith(('_q05', '_q50', '_q95'))]
        if mc_cols:
            id_cols = [c for c in results_df.columns
                       if c in ('OBJECTID', 'Area_Name')]
            results_df[id_cols + mc_cols].to_csv(mc_path, index=False)
            if verbose:
                print(f"  MC quantiles saved: {mc_path}")

        return summary_df, mc_meta

    # ==================================================================
    # Monte-Carlo Uncertainty — Base-Model Consensus
    # ==================================================================

    def run_with_consensus_uncertainty(
        self,
        data: pd.DataFrame,
        n_mc: int = 50,
        weight_cv: float = 0.20,
        verbose: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Monte-Carlo uncertainty around the base-model consensus path.

        Each draw perturbs the base-model consensus weights by
        multiplicative Gaussian noise (mean 1.0, CV = *weight_cv*),
        re-normalises them, and re-runs the single-variable scenario
        loop to produce per-cell delta draws.  Credible intervals
        (q05 / q50 / q95) are added to the results.

        Parameters
        ----------
        data : pd.DataFrame
            Baseline spatial data.
        n_mc : int
            Number of Monte-Carlo draws (default 50).
        weight_cv : float
            Coefficient of variation for the Gaussian noise applied to
            each base-model weight per draw (default 0.20 = ±20 %).
        verbose : bool
            Print progress.

        Returns
        -------
        summary_df : pd.DataFrame
            Scenario summary with ``*_q05``, ``*_q50``, ``*_q95`` columns.
        mc_meta : dict
            Metadata about the MC run.
        """

        if verbose:
            print(f"\n{'='*70}")
            print(f"MC UNCERTAINTY — Base-Model Consensus  (n_mc={n_mc}, weight_cv={weight_cv})")
            print(f"{'='*70}")

        orig_weights = dict(self._base_model_weights)

        # ── Baseline predictions (constant across all draws) ─────────
        _, ols_bl, gwr_bl, gwrf_bl, ggpgam_bl = self._predict_baseline(data)
        baseline_base_preds: Dict[str, np.ndarray] = {
            "ols": ols_bl, "gwr": gwr_bl, "gwrf": gwrf_bl, "ggpgam": ggpgam_bl,
        }

        # Pre-compute modified features per scenario+increment
        # so we don't recompute them every MC draw.
        scenario_specs: List[dict] = []
        for scenario in self.scenarios:
            var_name = scenario["variable"]
            direction = scenario.get("direction", "increase")
            for increment in scenario.get("increments", []):
                delta_signed = -increment if direction == "decrease" else increment
                label = f"{var_name}_{'minus' if delta_signed < 0 else 'plus'}_{str(increment).replace('.', 'p')}"
                original_vals = data[var_name].values.copy()
                modified_vals = original_vals + delta_signed
                bmin = scenario.get("min_val")
                bmax = scenario.get("max_val")
                if bmin is not None:
                    modified_vals = np.maximum(modified_vals, bmin)
                if bmax is not None:
                    modified_vals = np.minimum(modified_vals, bmax)
                scenario_specs.append({
                    "label": label,
                    "var_name": var_name,
                    "modified_vals": modified_vals,
                    "actual_change": modified_vals - original_vals,
                })

        # Pre-compute per-model deltas (these don't change across draws)
        model_deltas: Dict[str, Dict[str, np.ndarray]] = {}
        for spec in scenario_specs:
            var_name = spec["var_name"]
            modified_vals = spec["modified_vals"]
            df_mod = data.copy()
            df_mod[var_name] = modified_vals
            X_mod = df_mod[self.features].values
            coords = df_mod[self.coord_cols].values

            per_model = {}
            for name in ("ols", "gwr", "gwrf", "ggpgam"):
                model = self._models[name]
                if name == "ols":
                    mod_pred = model.predict(X_mod)
                else:
                    raw = model.predict(X_mod, coords)
                    mod_pred = raw[0] if isinstance(raw, tuple) else raw
                per_model[name] = mod_pred - baseline_base_preds[name]
            model_deltas[spec["label"]] = per_model

        # ── MC draws ─────────────────────────────────────────────────
        delta_draws: Dict[str, List[np.ndarray]] = {s["label"]: [] for s in scenario_specs}

        for i in range(n_mc):
            if verbose and (i % 10 == 0):
                print(f"  MC draw {i+1}/{n_mc} ...")

            # Perturb weights
            noisy = {k: v * np.random.normal(1.0, weight_cv) for k, v in orig_weights.items()}
            noisy = {k: max(v, 0.01) for k, v in noisy.items()}
            wtot = sum(noisy.values())
            draw_weights = {k: v / wtot for k, v in noisy.items()}

            for spec in scenario_specs:
                lbl = spec["label"]
                delta = np.zeros(len(data))
                for name in ("ols", "gwr", "gwrf", "ggpgam"):
                    delta += draw_weights[name] * model_deltas[lbl][name]

                # Apply guardrails
                var_name = spec["var_name"]
                change_dir = 1.0 if np.mean(spec["actual_change"]) >= 0 else -1.0
                delta = self._apply_physics_guardrails(
                    var_name, delta, change_direction=change_dir,
                )
                delta_draws[lbl].append(delta)

        # Restore original weights
        self._base_model_weights = orig_weights

        # ── Compute deterministic deltas + quantiles ─────────────────
        if verbose:
            print(f"\n  Computing deterministic deltas & credible intervals ...")

        # Build results_df directly from the cached model_deltas
        # instead of re-running run_with_model_reprediction.
        baseline_pred = self._predict_baseline(data)[0]
        results_df = data.copy()
        results_df["pred_baseline"] = baseline_pred

        deterministic_summary_rows: List[dict] = []
        mc_summary_rows: List[dict] = []
        for spec in scenario_specs:
            lbl = spec["label"]
            var_name = spec["var_name"]

            # Deterministic consensus delta (original weights)
            det_delta = np.zeros(len(data))
            for name in ("ols", "gwr", "gwrf", "ggpgam"):
                det_delta += orig_weights[name] * model_deltas[lbl][name]
            change_dir = 1.0 if np.mean(spec["actual_change"]) >= 0 else -1.0
            det_delta = self._apply_physics_guardrails(
                var_name, det_delta, change_direction=change_dir,
            )
            results_df[f"repred_delta_{lbl}"] = det_delta

            # Per-area deterministic summary rows
            for area_id, area_name in self.area_names.items():
                if self.area_column not in data.columns:
                    continue
                mask = data[self.area_column] == area_id
                if mask.sum() == 0:
                    continue
                a_delta = det_delta[mask.values]
                deterministic_summary_rows.append({
                    "Variable": var_name,
                    "Scenario": lbl,
                    "Increment": float(abs(spec["actual_change"].mean())),
                    "Direction": "decrease" if np.mean(spec["actual_change"]) < 0 else "increase",
                    "Area": area_name,
                    "N_Cells": int(mask.sum()),
                    "Avg_Var_Change": float(spec["actual_change"][mask.values].mean()),
                    "Repred_Delta_Mean": float(a_delta.mean()),
                    "Repred_Delta_Std": float(a_delta.std()),
                    "Method": "base_model_consensus",
                })
            if not self.area_names:
                deterministic_summary_rows.append({
                    "Variable": var_name,
                    "Scenario": lbl,
                    "Increment": float(abs(spec["actual_change"].mean())),
                    "Direction": "decrease" if np.mean(spec["actual_change"]) < 0 else "increase",
                    "Area": "All",
                    "N_Cells": len(data),
                    "Avg_Var_Change": float(spec["actual_change"].mean()),
                    "Repred_Delta_Mean": float(det_delta.mean()),
                    "Repred_Delta_Std": float(det_delta.std()),
                    "Method": "base_model_consensus",
                })

            # MC quantiles
            stacked = np.stack(delta_draws[lbl], axis=0)  # (n_mc, n_cells)
            results_df[f"repred_delta_{lbl}_q05"] = np.percentile(stacked, 5, axis=0)
            results_df[f"repred_delta_{lbl}_q50"] = np.percentile(stacked, 50, axis=0)
            results_df[f"repred_delta_{lbl}_q95"] = np.percentile(stacked, 95, axis=0)

            means_per_draw = [float(np.mean(d)) for d in delta_draws[lbl]]
            mc_summary_rows.append({
                "Scenario": lbl,
                "Variable": var_name,
                "MC_Mean": float(np.mean(means_per_draw)),
                "MC_Std": float(np.std(means_per_draw)),
                "MC_q05": float(np.percentile(means_per_draw, 5)),
                "MC_q50": float(np.percentile(means_per_draw, 50)),
                "MC_q95": float(np.percentile(means_per_draw, 95)),
            })

        summary_df = pd.DataFrame(deterministic_summary_rows)

        mc_summary = pd.DataFrame(mc_summary_rows)

        mc_meta = {
            'n_mc': n_mc,
            'weight_cv': weight_cv,
            'n_successful_draws': n_mc,
            'scenario_labels': [s["label"] for s in scenario_specs],
        }

        if verbose:
            print(f"\n  MC complete: {n_mc} draws across "
                  f"{len(scenario_specs)} scenarios.\n")
            for _, row in mc_summary.iterrows():
                print(f"    {row['Scenario']:40s}  "
                      f"Δ = {row['MC_q50']:+.4f}  "
                      f"[{row['MC_q05']:+.4f}, {row['MC_q95']:+.4f}]")

        # Save MC results
        mc_path = self.output_dir / "scenario_mc_consensus.csv"
        mc_cols = [c for c in results_df.columns
                   if c.endswith(('_q05', '_q50', '_q95'))]
        if mc_cols:
            id_cols = [c for c in results_df.columns if c in ('OBJECTID', 'Area_Name')]
            valid_id_cols = [c for c in id_cols if c in results_df.columns]
            results_df[valid_id_cols + mc_cols].to_csv(mc_path, index=False)
            if verbose:
                print(f"  MC quantiles saved: {mc_path}")

        mc_summary_path = self.output_dir / "scenario_mc_consensus_summary.csv"
        mc_summary.to_csv(mc_summary_path, index=False)
        if verbose:
            print(f"  MC summary saved: {mc_summary_path}")

        return summary_df, mc_meta
