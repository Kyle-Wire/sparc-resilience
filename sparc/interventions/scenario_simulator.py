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


def _stage4_store():
    """Return the active ArtifactStore (None when running outside pipeline)."""
    try:
        from sparc.registry.store import get_active_store
        return get_active_store()
    except Exception:
        return None


def _persist_scenario_summary(
    summary_df: pd.DataFrame,
    *,
    artifact_id: str,
    output_dir: Path,
    disk_filename: str,
    also_register_as: Optional[str] = None,
) -> None:
    """Write a scenario summary table to artifacts.db when active, disk otherwise.

    ``also_register_as`` lets each mode (dag / hybrid / reprediction)
    additionally register itself under the canonical ``scenario_summary``
    id so consumers don't need to know the mode.
    """
    store = _stage4_store()
    if store is not None:
        try:
            store.write_table("4", artifact_id, summary_df,
                              producer="scenario_simulator")
            if also_register_as and also_register_as != artifact_id:
                store.write_table("4", also_register_as, summary_df,
                                  producer="scenario_simulator")
            return
        except Exception:
            pass
    try:
        from sparc.run.pipeline_paths import get_result_store
        rstore = get_result_store()
        rstore.save_dataframe(4, disk_filename, summary_df, fmt="csv")
        if also_register_as:
            rstore.save_dataframe(4, "scenario_summary.csv", summary_df, fmt="csv")
    except Exception:
        summary_df.to_csv(output_dir / disk_filename, index=False)


def _persist_scenario_geo(
    results_gdf,
    *,
    artifact_id: str,
    output_dir: Path,
    disk_filename: str,
    also_register_as: Optional[str] = None,
) -> None:
    """Write a scenario GeoDataFrame to artifacts.db (WKB table) or disk gpkg."""
    if not GEOPANDAS_AVAILABLE or not isinstance(results_gdf, gpd.GeoDataFrame):
        return
    store = _stage4_store()
    if store is not None:
        try:
            crs = str(results_gdf.crs) if results_gdf.crs is not None else None
            store.write_table(
                "4", artifact_id, results_gdf,
                geometry_col="geometry", crs=crs,
                producer="scenario_simulator",
            )
            if also_register_as and also_register_as != artifact_id:
                store.write_table(
                    "4", also_register_as, results_gdf,
                    geometry_col="geometry", crs=crs,
                    producer="scenario_simulator",
                )
            return
        except Exception:
            pass
    try:
        from sparc.run.pipeline_paths import get_result_store
        rstore = get_result_store()
        rstore.save_geodataframe(4, disk_filename, results_gdf)
        if also_register_as:
            rstore.save_geodataframe(4, "scenario_results.gpkg", results_gdf)
    except Exception:
        try:
            results_gdf.to_file(output_dir / disk_filename, driver="GPKG")
            if also_register_as:
                results_gdf.to_file(output_dir / "scenario_results.gpkg",
                                    driver="GPKG")
        except Exception as e:
            warnings.warn(f"Could not write GeoPackage: {e}")


def spatial_gini_coefficient(values: np.ndarray) -> float:
    """
    Compute the Gini coefficient of absolute values.

    Measures spatial inequality of effect magnitudes — higher Gini
    means the effect is concentrated in fewer locations.

    Parameters
    ----------
    values : (N,) array of effect magnitudes

    Returns
    -------
    gini : float in [0, 1] — 0 = perfect equality, 1 = all effect at one point
    """
    arr = np.abs(values).flatten()
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2 or arr.sum() < 1e-12:
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    index = np.arange(1, n + 1)
    return float(((2.0 * index - n - 1) * arr).sum() / (n * arr.sum()))


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
        try:
            from sparc.run.disk_policy import disk_writes_enabled
            if disk_writes_enabled():
                self.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
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

        # Build name→index map.  Use the GWR model's actual feature_names_
        # (which may be a subset of self.features) to avoid index-out-of-bounds.
        gwr_features = feature_names if feature_names else self.features
        for j, feat in enumerate(gwr_features):
            self._mgwr_feature_map[feat] = j

        n_pts = raw_coeffs.shape[0]
        print(f"   MGWR local coefficients extracted: {n_pts} points × {raw_coeffs.shape[1]} features")
        for feat in gwr_features:
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
        # Hardcoded per-variable defaults removed in SPARC v4 — declare
        # diminishing_return_thresholds in caps.yml. Returning a neutral
        # 10.0 keeps legacy behavior for unconfigured variables.
        return 10.0

    # ------------------------------------------------------------------
    # Bayesian per-cell coefficient β(s) loader
    # ------------------------------------------------------------------

    def _get_bayesian_beta(self, treatment: str) -> Optional[np.ndarray]:
        """Return per-cell Bayesian local treatment effect β(s) for *treatment*.

        Reads the ``cate_mean`` column from the ``("3","cate_summary")``
        long-format table (sorted by ``cell_id``). This is the Stage-3 NUTS
        posterior mean of the spatial CATE — in v4 it replaces every place
        the simulator previously used MGWR local coefficients to compute an
        effect, because MGWR coefficients are correlation-based and β(s) is
        the causal quantity.

        Returns ``None`` if the artifact is not registered (e.g. Stage 3 was
        not run or did not estimate CATE for *treatment*). Results are
        cached on ``self._bayesian_beta_cache``.
        """
        cache = getattr(self, "_bayesian_beta_cache", None)
        if cache is None:
            cache = {}
            self._bayesian_beta_cache = cache
        if treatment in cache:
            return cache[treatment]

        beta: Optional[np.ndarray] = None
        try:
            from sparc.registry.store import get_active_store
            store = get_active_store()
            if store is not None and store.has("3", "cate_summary"):
                df = store.read_table("3", "cate_summary")
                if df is not None and "treatment" in df.columns and "cate_mean" in df.columns:
                    sub = df[df["treatment"] == treatment]
                    if not sub.empty:
                        if "cell_id" in sub.columns:
                            sub = sub.sort_values("cell_id")
                        beta = np.asarray(sub["cate_mean"].to_numpy(), dtype=float)
        except Exception:
            beta = None

        cache[treatment] = beta
        return beta

    # ------------------------------------------------------------------
    # Direct-effect delta computation
    # ------------------------------------------------------------------

    def _compute_mgwr_direct_delta(
        self, variable: str, effective_change: np.ndarray,
        baseline_values: Optional[np.ndarray] = None,
        modified_values: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, str]:
        """
        Per-point direct delta using a four-tier strategy:

        1. **PDE alpha field** (V3) — if a learned spatial alpha field is
           available, modulate the global structural coefficient by it.
        2. **Bayesian β(s)** (v4) — per-cell posterior-mean coefficient
           from Stage 3 ``cate_summary.cate_mean`` (NUTS spatial CATE).
           This is the canonical causal local effect; MGWR/GWR local
           coefficients are no longer used here because they are
           correlation-based.
        3. **Saturation curve** — GWRF condition curve PDP (when reliable).
        4. **Physics literature** — pure prior fallback.

        Returns
        -------
        (delta, method) : (ndarray, str)
            ``delta`` — per-point change in outcome.
            ``method`` — ``'pde_alpha_field*'``, ``'bayesian_beta'``,
            ``'saturation_curve'``, or ``'physics_lit'``.
        """
        # --- Physics literature coefficient per raw unit ---------------
        prior = self.physics_priors.get(variable)
        if prior and prior.get("unit_increment", 1.0) != 0:
            lit_per_unit = prior["lit_coef"] / prior["unit_increment"]
        else:
            lit_per_unit = 0.0

        # --- Tier 0: V3 alpha field (PDE-learned spatial heterogeneity) ---
        if self._alpha_field is not None:
            n = min(len(effective_change), len(self._alpha_field))
            alpha_s = self._alpha_field[:n]
            alpha_mean = float(np.mean(alpha_s))
            if alpha_mean > 1e-12:
                alpha_norm = alpha_s / alpha_mean  # centered at 1.0

                # beta_global from NUTS posterior or counterfactual engine
                beta_global = self.get_causal_coefficient(variable)
                if beta_global is None:
                    beta_global = lit_per_unit

                # PDP saturation modulation (if available)
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
                        pdp_delta = predict_scenario_with_saturation(
                            variable_name=variable,
                            baseline_values=baseline_values[:n],
                            modified_values=modified_values[:n],
                            condition_curve=condition_curve_dict,
                            spatial_multiplier=None,
                        )
                        # PDP slope: normalize by effective_change to get
                        # per-unit marginal response
                        eff_n = effective_change[:n]
                        pdp_slope_raw = np.where(
                            np.abs(eff_n) > 1e-8,
                            pdp_delta / eff_n,
                            0.0,
                        )
                        # Normalize PDP slope to a pure spatial shape modifier
                        # (mean ≈ 1). Use absolute values so the sign comes
                        # solely from beta_global — avoids double-counting the
                        # direction already present in the PDP curve.
                        # Only use active points (non-zero eff_inc) for the mean.
                        abs_pdp = np.abs(pdp_slope_raw)
                        active_mask = np.abs(eff_n) > 1e-8
                        if active_mask.any():
                            abs_pdp_mean = float(np.mean(abs_pdp[active_mask]))
                        else:
                            abs_pdp_mean = 0.0
                        if abs_pdp_mean > 1e-10:
                            pdp_slope = abs_pdp / abs_pdp_mean
                        else:
                            pdp_slope = np.ones_like(pdp_slope_raw)
                        # Full formula: beta_global × increment × PDP_slope × alpha_norm
                        delta = beta_global * eff_n * pdp_slope * alpha_norm
                        gini = spatial_gini_coefficient(delta)
                        print(f"   [Tier 0] alpha+PDP delta for {variable}: "
                              f"Gini={gini:.4f}, mean={np.mean(delta):.4f}")
                        print(f"     PDP diag: raw_slope_mean={np.mean(pdp_slope_raw):.6f}, "
                              f"|raw|_mean={abs_pdp_mean:.6f}, "
                              f"norm_slope_mean={np.mean(pdp_slope):.4f}, "
                              f"beta_global={beta_global:.6f}")
                        return delta, 'pde_alpha_field_pdp'
                    except Exception:
                        pass  # fall through to non-PDP alpha path

                # Without PDP: beta_global × alpha_norm × effective_change
                delta = beta_global * alpha_norm * effective_change[:n]
                gini = spatial_gini_coefficient(delta)
                print(f"   [Tier 0] alpha delta for {variable}: "
                      f"Gini={gini:.4f}, mean={np.mean(delta):.4f}")
                return delta, 'pde_alpha_field'

        # --- Tier 1: Bayesian per-cell coefficient β(s) ---------------
        # Stage-3 NUTS posterior mean from cate_summary.cate_mean. This is
        # the canonical causal local effect; v4 dropped MGWR direct/blend.
        beta_s = self._get_bayesian_beta(variable)
        if beta_s is not None and beta_s.size > 0:
            n = min(len(effective_change), len(beta_s))
            return beta_s[:n] * effective_change[:n], 'bayesian_beta'

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

        # --- Tier 3: Pure physics literature fallback -----------------
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

            # Final edge (last-mediator → outcome): use alpha field (V3),
            # Bayesian β(s) (v4), or DAG structural coefficient.
            # MGWR/GWR coefficients are no longer used here — they are
            # correlation-based and have been superseded by Stage-3 β(s).
            final_mediator = path[-2]

            if self._alpha_field is not None:
                # V3: use normalized alpha field for spatial modulation
                n_alpha = min(len(current_delta), len(self._alpha_field))
                alpha_s = self._alpha_field[:n_alpha]
                alpha_mean = float(np.mean(alpha_s))
                if alpha_mean > 1e-12:
                    alpha_norm = alpha_s / alpha_mean
                    indirect_delta[:n_alpha] += alpha_norm * current_delta[:n_alpha]
                    continue

            beta_final = self._get_bayesian_beta(final_mediator)
            if beta_final is not None and beta_final.size > 0:
                m = min(len(current_delta), len(beta_final))
                indirect_delta[:m] += beta_final[:m] * current_delta[:m]
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

                    # Use Bayesian β(s) for the final edge if available;
                    # otherwise fall back to the DAG structural coefficient.
                    if step == len(path) - 2:
                        final_node = child if child == target else parent
                        beta_final = self._get_bayesian_beta(final_node)
                        if beta_final is not None and beta_final.size > 0:
                            m = min(n, len(beta_final))
                            current_delta = beta_final[:m] * current_delta[:m]
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

        # Check whether V2 neural artifacts exist — if so, V1 pkl models
        # are optional (bayesian / V2 scenario modes don't need them).
        v2_dir = self.model_dir / "v2_neural"
        has_v2 = (v2_dir / "neural_meta.pt").exists() and (v2_dir / "meta_info.json").exists()
        if not has_v2:
            try:
                from sparc.registry.store import get_active_store
                _store = get_active_store()
                if _store is not None and _store.has("2", "v2_neural_meta_info"):
                    # Even if disk pt is gone, db-resident meta is sufficient
                    # for V2-aware modes (disk pt restored separately if needed).
                    has_v2 = (v2_dir / "neural_meta.pt").exists() or _store.has("2", "v2_neural_meta_state")
            except Exception:
                pass

        model_files = {
            "meta": self.model_dir / "standard_meta_ensemble.pkl",
            "ols": self.model_dir / "base_models_full" / "ols_model_full.pkl",
            "gwr": self.model_dir / "base_models_full" / "gwr_model_full.pkl",
            "gwrf": self.model_dir / "base_models_full" / "gwrf_model_full.pkl",
            "ggpgam": self.model_dir / "base_models_full" / "ggpgam_model_full.pkl",
        }

        v1_loaded = 0
        for name, path in model_files.items():
            if not path.exists():
                if has_v2:
                    continue  # V1 models optional when V2 is available
                raise FileNotFoundError(f"Model file not found: {path}")
            if name == "meta":
                self._meta_model = joblib.load(path)
            else:
                self._models[name] = joblib.load(path)
            v1_loaded += 1

        if v1_loaded > 0:
            print(f"   V1 models loaded: {v1_loaded}")
        elif has_v2:
            print("   V1 models not found — using V2 neural pipeline")
        else:
            raise FileNotFoundError("No V1 or V2 models found in model_dir")

        # Extract MGWR local coefficients from the GWR model (if loaded)
        self._extract_mgwr_coefficients()

        # Load feature info (actual features used during training)
        _fi = None
        try:
            from sparc.registry.store import get_active_store
            _store = get_active_store()
            if _store is not None and _store.has("2", "feature_info"):
                _fi = _store.read_any("2", "feature_info")
        except Exception:
            _fi = None
        if _fi is None:
            feature_info_path = self.model_dir / "feature_info.json"
            if feature_info_path.exists():
                import json as _json_fi
                with open(feature_info_path) as _fi_f:
                    _fi = _json_fi.load(_fi_f)
        if _fi is not None:
            trained_features = _fi.get("feature_names", self.features)
            if trained_features != self.features:
                print(f"   Overriding config features ({len(self.features)}) with trained features ({len(trained_features)}): {trained_features}")
                self.features = trained_features

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

        # Load V2 neural meta-learner if available
        self._v2_neural_model = None
        self._v2_process_net = None
        self._v2_source_net = None
        self._v2_meta_info = None
        self._alpha_field = None
        self._alpha_field_full = None  # Phase 3 (v4): (N, n_treatments) tensor
        self._alpha_field_coords = None
        self._cardinal_neighbors = None
        self._grid_spacing = None
        v2_dir = self.model_dir / "v2_neural"

        # Resolve meta_info: prefer artifacts.db.
        meta_info_loaded = None
        try:
            from sparc.registry.store import get_active_store
            _store = get_active_store()
            if _store is not None and _store.has("2", "v2_neural_meta_info"):
                meta_info_loaded = _store.read_any("2", "v2_neural_meta_info")
        except Exception:
            meta_info_loaded = None

        meta_disk_exists = (v2_dir / "meta_info.json").exists()
        ckpt_disk_exists = (v2_dir / "neural_meta.pt").exists()

        if meta_info_loaded is not None or (ckpt_disk_exists and meta_disk_exists):
            try:
                if meta_info_loaded is None:
                    import json as _json
                    with open(v2_dir / "meta_info.json") as _f:
                        meta_info_loaded = _json.load(_f)
                self._v2_meta_info = meta_info_loaded
                print(f"   V2 neural meta-learner artifacts found (OOF R²={self._v2_meta_info.get('oof_r2', '?'):.4f})")
            except Exception as _e:
                print(f"   Warning: V2 neural meta_info failed to load: {_e}")

            # Load V3 alpha field if available (store-first, disk-fallback)
            try:
                from sparc.registry.store import get_active_store
                _store = get_active_store()
            except Exception:
                _store = None

            alpha_loaded = False
            if _store is not None:
                try:
                    if _store.has("2", "v2_alpha_field"):
                        raw_alpha = _store.read_any("2", "v2_alpha_field")
                        # Phase 3 (v4): blob may now be shape (N, n_treatments)
                        # under schema_version=2.  Keep the full tensor for the
                        # resolver and collapse to (N,) for legacy callers
                        # that index alpha[:n] as a 1-D array.
                        raw_alpha = np.asarray(raw_alpha)
                        if raw_alpha.ndim == 2 and raw_alpha.shape[-1] >= 1:
                            self._alpha_field_full = raw_alpha
                            self._alpha_field = (
                                raw_alpha[:, 0]
                                if raw_alpha.shape[-1] == 1
                                else raw_alpha.mean(axis=-1)
                            )
                        else:
                            self._alpha_field_full = raw_alpha.reshape(-1, 1)
                            self._alpha_field = raw_alpha.reshape(-1)
                        if _store.has("2", "v2_alpha_field_coords"):
                            self._alpha_field_coords = _store.read_any("2", "v2_alpha_field_coords")
                        alpha_loaded = True
                        n_heads = (
                            self._alpha_field_full.shape[-1]
                            if self._alpha_field_full is not None else 1
                        )
                        print(
                            f"   V3 alpha field loaded from artifacts.db "
                            f"({len(self._alpha_field)} points, {n_heads} treatment heads)"
                        )
                except Exception as _e:
                    print(f"   Warning: alpha_field load from store failed: {_e}")
                    self._alpha_field = None
                    self._alpha_field_full = None

            if not alpha_loaded:
                alpha_path = v2_dir / "alpha_field.npy"
                alpha_coords_path = v2_dir / "alpha_field_coords.npy"
                if alpha_path.exists():
                    try:
                        loaded = np.asarray(np.load(alpha_path))
                        # Disk artifact stays 1-D; populate full tensor as
                        # the broadcast (N, 1) view for the resolver.
                        if loaded.ndim == 1:
                            self._alpha_field = loaded
                            self._alpha_field_full = loaded.reshape(-1, 1)
                        else:
                            self._alpha_field_full = loaded
                            self._alpha_field = (
                                loaded[:, 0]
                                if loaded.shape[-1] == 1
                                else loaded.mean(axis=-1)
                            )
                        if alpha_coords_path.exists():
                            self._alpha_field_coords = np.load(alpha_coords_path)
                        print(f"   V3 alpha field loaded from disk ({len(self._alpha_field)} points)")
                    except Exception as _e:
                        print(f"   Warning: alpha_field.npy failed to load: {_e}")
                        self._alpha_field = None
                        self._alpha_field_full = None

            # Load PDE forward solver artifacts (source term net, cardinal neighbors, grid spacing)
            self._load_pde_solver_artifacts(v2_dir)

    # ------------------------------------------------------------------
    # PDE forward solver artifact loading
    # ------------------------------------------------------------------

    def _load_pde_solver_artifacts(self, v2_dir: Path) -> None:
        """Load SourceTermNet, cardinal neighbors, and grid spacing for PDE solving.

        Prefers artifacts.db (active ArtifactStore) and falls back to disk
        artifacts written by v2_neural_training.
        """
        import torch

        try:
            from sparc.registry.store import get_active_store
            _store = get_active_store()
        except Exception:
            _store = None

        # SourceTermNet (state dict in store, full file on disk)
        source_loaded = False
        if _store is not None:
            try:
                if _store.has("2", "v2_source_term_state"):
                    from sparc.models.process_rate_net import SourceTermNet
                    meta_info = self._v2_meta_info or {}
                    n_physics = meta_info.get("n_physics_original", 3)
                    self._v2_source_net = SourceTermNet(n_inputs=n_physics)
                    state = _store.read_any("2", "v2_source_term_state")
                    self._v2_source_net.load_state_dict(state)
                    self._v2_source_net.eval()
                    source_loaded = True
                    print(f"   SourceTermNet loaded from artifacts.db ({n_physics} inputs)")
            except Exception as _e:
                print(f"   Warning: SourceTermNet load from store failed: {_e}")
                self._v2_source_net = None

        if not source_loaded:
            stn_path = v2_dir / "source_term_net.pt"
            if stn_path.exists():
                try:
                    from sparc.models.process_rate_net import SourceTermNet
                    meta_info = self._v2_meta_info or {}
                    n_physics = meta_info.get("n_physics_original", 3)
                    self._v2_source_net = SourceTermNet(n_inputs=n_physics)
                    self._v2_source_net.load_state_dict(
                        torch.load(stn_path, map_location="cpu", weights_only=True)
                    )
                    self._v2_source_net.eval()
                    print(f"   SourceTermNet loaded from disk ({n_physics} inputs)")
                except Exception as _e:
                    print(f"   Warning: source_term_net.pt failed to load: {_e}")
                    self._v2_source_net = None

        # Cardinal neighbors (store-first)
        if _store is not None:
            try:
                if _store.has("2", "v2_cardinal_neighbors"):
                    self._cardinal_neighbors = _store.read_any("2", "v2_cardinal_neighbors")
            except Exception as _e:
                print(f"   Warning: cardinal_neighbors load from store failed: {_e}")
        if self._cardinal_neighbors is None:
            cn_path = v2_dir / "cardinal_neighbors.npy"
            if cn_path.exists():
                try:
                    self._cardinal_neighbors = np.load(cn_path)
                except Exception as _e:
                    print(f"   Warning: cardinal_neighbors.npy failed to load: {_e}")

        # Grid spacing (store-first)
        if _store is not None:
            try:
                if _store.has("2", "v2_grid_spacing"):
                    self._grid_spacing = _store.read_any("2", "v2_grid_spacing")
            except Exception as _e:
                print(f"   Warning: grid_spacing load from store failed: {_e}")
        if self._grid_spacing is None:
            gs_path = v2_dir / "grid_spacing.npy"
            if gs_path.exists():
                try:
                    self._grid_spacing = np.load(gs_path)
                except Exception as _e:
                    print(f"   Warning: grid_spacing.npy failed to load: {_e}")

        if self._cardinal_neighbors is not None and self._grid_spacing is not None:
            print(f"   PDE solver infrastructure loaded (N={len(self._cardinal_neighbors)})")

    @property
    def _pde_solver_available(self) -> bool:
        """True if all PDE forward solver artifacts are loaded."""
        return (
            self._v2_source_net is not None
            and self._alpha_field is not None
            and self._cardinal_neighbors is not None
            and self._grid_spacing is not None
        )

    def _pde_joint_delta(
        self,
        baseline_pred_z: np.ndarray,
        modified_physics_feats: np.ndarray,
        target_mask: "pd.Series",
    ) -> tuple[np.ndarray, dict]:
        """Compute joint scenario delta via PDE forward solve.

        Parameters
        ----------
        baseline_pred_z : (N,) — baseline predictions in z-space.
        modified_physics_feats : (N, n_physics) — z-scored physics features
            after all intervention legs have been applied.
        target_mask : boolean mask for target cells.

        Returns
        -------
        delta_raw : (N_target,) — predicted temperature change in original units.
        solve_info : dict with convergence diagnostics.
        """
        import torch
        from sparc.physics.pde_solver import poisson_solve

        device = torch.device("cpu")
        n = len(baseline_pred_z)

        # Compute post-intervention source term
        phys_t = torch.tensor(modified_physics_feats, dtype=torch.float32, device=device)
        with torch.no_grad():
            S_new = self._v2_source_net(phys_t).squeeze(-1)  # (N,)

        # Prepare solver inputs
        T_init = torch.tensor(baseline_pred_z, dtype=torch.float32, device=device)
        alpha = torch.tensor(self._alpha_field[:n], dtype=torch.float32, device=device)
        neighbor_idx = torch.tensor(self._cardinal_neighbors[:n], dtype=torch.long, device=device)
        h = torch.tensor(self._grid_spacing[:n], dtype=torch.float32, device=device)

        # Solve α∇²T = S
        T_post, info = poisson_solve(
            T_init=T_init,
            source=S_new,
            alpha=alpha,
            neighbor_idx=neighbor_idx,
            h=h,
        )

        # Delta in z-space → original units
        delta_z = (T_post - T_init).numpy()
        meta = self._v2_meta_info or {}
        y_std = meta.get("y_std", 1.0)
        delta_raw = delta_z * y_std

        # Sanity guard: no scenario should predict > 20 units change
        mean_abs = float(np.abs(delta_raw).mean())
        if mean_abs > 20.0:
            import warnings
            warnings.warn(
                f"PDE solver delta suspiciously large (mean |Δ| = {mean_abs:.1f}). "
                "Check scale consistency.",
                stacklevel=2,
            )

        # Extract target cells
        if hasattr(target_mask, 'values'):
            mask_arr = target_mask.values
        else:
            mask_arr = np.asarray(target_mask)
        delta_target = delta_raw[mask_arr[:n]]

        return delta_target, info

    def _interaction_via_pde(
        self,
        baseline_df: pd.DataFrame,
        baseline_pred: np.ndarray,
        target_mask: "pd.Series",
        legs: list[dict],
        verbose: bool = True,
    ) -> tuple[np.ndarray, dict]:
        """Compute joint interaction delta via PDE forward solve.

        Constructs modified physics features by applying all legs, z-scores
        them, runs SourceTermNet, then calls ``_pde_joint_delta``.
        """
        meta = self._v2_meta_info or {}
        feature_names = meta.get("feature_names", [])
        y_mean = meta.get("y_mean", 0.0)
        y_std = meta.get("y_std", 1.0)

        # Load feature scaling stats (store-first, disk-fallback)
        v2_dir = self.model_dir / "v2_neural"
        feat_mean = None
        feat_std = None
        try:
            from sparc.registry.store import get_active_store
            _store = get_active_store()
        except Exception:
            _store = None
        if _store is not None:
            try:
                if _store.has("2", "v2_feature_scaling"):
                    fs = _store.read_any("2", "v2_feature_scaling")
                    feat_mean = fs.get("feat_mean")
                    feat_std = fs.get("feat_std")
            except Exception:
                feat_mean = None
                feat_std = None
        if feat_mean is None or feat_std is None:
            scaling = np.load(v2_dir / "feature_scaling.npz")
            feat_mean = scaling["feat_mean"]
            feat_std = scaling["feat_std"]

        # Build modified raw feature matrix
        raw_feats = baseline_df[feature_names].values.copy()  # (N, n_feats)
        for leg in legs:
            var = leg["variable"]
            delta = float(leg.get("delta", 0))
            if delta == 0 or var not in feature_names:
                continue
            j = feature_names.index(var)
            raw_feats[:, j] += delta

            # Clip to physics bounds
            sc_cfg = next((s for s in self.scenarios if s["variable"] == var), {})
            if sc_cfg.get("min_val") is not None:
                raw_feats[:, j] = np.maximum(raw_feats[:, j], sc_cfg["min_val"])
            if sc_cfg.get("max_val") is not None:
                raw_feats[:, j] = np.minimum(raw_feats[:, j], sc_cfg["max_val"])

        # Z-score the modified features
        modified_z = (raw_feats - feat_mean) / feat_std

        # Baseline z-scored predictions
        baseline_z = (baseline_pred - y_mean) / y_std

        return self._pde_joint_delta(baseline_z, modified_z, target_mask)

    # ------------------------------------------------------------------
    # Condition curve loading (PDP saturation fits from Stage 2)
    # ------------------------------------------------------------------

    def _load_condition_curves(self) -> None:
        """Load saturation / dose-response curves for scenario use.

        **Source priority (highest → lowest):**

        1. GWRF classical PDP JSON (``gwrf_condition_curves.json``) — produced
           by Stage 2b ``gwrf_model.export_condition_curves()`` when enabled.
        2. Neural meta-learner PDP CSVs (``v2_neural/pdp/pdp_{feat}.csv``) —
           produced every run by ``_export_v2_outputs()``; these are
           PDE-informed because the full PDE-physics training (α∇²T−S≈0,
           energy balance, Fourier flux, etc.) shaped the model's response
           surface.  Saturation emerges naturally as the model output flattens
           where α forces PDE equilibrium.

        Combined with the α field in Tier 0, the neural PDPs give the richest
        saturation signal: ``beta_global × eff_change × pdp_slope × alpha_norm``,
        where ``pdp_slope`` captures nonlinear dose-response from the
        PDE-trained model and ``alpha_norm`` captures where (spatially) the
        effect is strongest.
        """
        import csv
        import json

        base_dir = Path(self.config['output']['base_dir'])
        stage2_name = self.config.get('output', {}).get('stage_dirs', {}).get(
            'stage_2', 'Stage_2_Spatial_CV'
        )

        # ---- Source 1: GWRF condition curve JSON ----
        curves_path = base_dir / stage2_name / 'gwrf_pdp' / 'gwrf_condition_curves.json'
        if not curves_path.exists():
            curves_path = base_dir / 'spatial_intelligence' / 'gwrf_pdp' / 'gwrf_condition_curves.json'
        if not curves_path.exists():
            stage3_name = self.config.get('output', {}).get('stage_dirs', {}).get(
                'stage_3', 'Stage_3_Causal_Validation'
            )
            curves_path = base_dir / stage3_name / 'gwrf_condition_curves.json'

        gwrf_loaded = 0
        gwrf_skipped = 0
        if curves_path.exists():
            with open(curves_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            for var_name, entry in raw.items():
                pdp = entry.get('pdp', {})
                curve_fit = entry.get('curve_fit', {})
                grid_vals = pdp.get('grid_values')
                pdp_vals = pdp.get('pdp_values')
                r2 = curve_fit.get('r2', 0.0)
                if grid_vals is None or pdp_vals is None:
                    gwrf_skipped += 1
                    continue
                self._condition_curves[var_name] = {
                    'grid_values': grid_vals,
                    'pdp_values': pdp_vals,
                    'pdp_std': pdp.get('pdp_std'),
                    'curve_fit': curve_fit,
                    'r2': r2,
                    'source': 'gwrf',
                }
                gwrf_loaded += 1

        # ---- Source 2: Neural meta-learner PDP (PDE-informed) ----
        # These are generated every run and capture the trained model's
        # nonlinear, PDE-constrained response surface.  Only loaded for
        # variables not already covered by the GWRF JSON above.
        # Prefer artifacts.db: pdp tables are registered as
        # ``v2_neural_pdp::<feature>`` under stage 2.
        neural_loaded = 0
        neural_skipped = 0
        loaded_from_store = False
        try:
            from sparc.registry.store import get_active_store
            _store = get_active_store()
            if _store is not None:
                manifest_stages = getattr(_store.registry.manifest, "stages", {}) or {}
                stage2 = manifest_stages.get("2", {}) or {}
                for art_id in stage2.keys():
                    if not art_id.startswith("v2_neural_pdp::"):
                        continue
                    feat_col = art_id.split("::", 1)[1]
                    if feat_col in self._condition_curves:
                        continue
                    try:
                        df = _store.read_any("2", art_id)
                    except Exception:
                        neural_skipped += 1
                        continue
                    if df is None or len(df) == 0 or feat_col not in df.columns:
                        neural_skipped += 1
                        continue
                    grid_vals = [float(v) for v in df[feat_col].tolist()]
                    pdp_vals = [float(v) for v in df["mean_prediction"].tolist()]
                    if {"q10", "q90"}.issubset(df.columns):
                        pdp_std = [
                            (float(q90) - float(q10)) / 2.56
                            for q10, q90 in zip(df["q10"].tolist(), df["q90"].tolist())
                        ]
                    else:
                        pdp_std = None
                    self._condition_curves[feat_col] = {
                        'grid_values': grid_vals,
                        'pdp_values': pdp_vals,
                        'pdp_std': pdp_std,
                        'curve_fit': {},
                        'r2': 1.0,
                        'source': 'neural_pde',
                    }
                    neural_loaded += 1
                    loaded_from_store = True
        except Exception:
            pass

        neural_pdp_dir = base_dir / stage2_name / 'v2_neural' / 'pdp'
        if not loaded_from_store and neural_pdp_dir.exists():
            for csv_path in sorted(neural_pdp_dir.glob('pdp_*.csv')):
                try:
                    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)
                    if not rows:
                        neural_skipped += 1
                        continue
                    # First column is the feature name (variable), others are
                    # mean_prediction, q10, q90
                    feat_col = reader.fieldnames[0] if reader.fieldnames else None
                    if feat_col is None or 'mean_prediction' not in (reader.fieldnames or []):
                        neural_skipped += 1
                        continue
                    # Skip if GWRF already provided this variable
                    if feat_col in self._condition_curves:
                        continue
                    grid_vals = [float(r[feat_col]) for r in rows]
                    pdp_vals = [float(r['mean_prediction']) for r in rows]
                    # Approximate std from q10/q90 interval (≈ 1.28σ each tail)
                    if 'q10' in rows[0] and 'q90' in rows[0]:
                        pdp_std = [
                            (float(r['q90']) - float(r['q10'])) / 2.56
                            for r in rows
                        ]
                    else:
                        pdp_std = None
                    self._condition_curves[feat_col] = {
                        'grid_values': grid_vals,
                        'pdp_values': pdp_vals,
                        'pdp_std': pdp_std,
                        'curve_fit': {},
                        # Neural model's own response surface — accept unconditionally
                        'r2': 1.0,
                        'source': 'neural_pde',
                    }
                    neural_loaded += 1
                except Exception:
                    neural_skipped += 1

        total = gwrf_loaded + neural_loaded
        if total == 0:
            print("   No condition curves found (GWRF or neural PDE) — "
                  "Stage 4 will use coefficient extrapolation.")
            return

        usable = sum(
            1 for c in self._condition_curves.values()
            if c['r2'] >= self._condition_curve_min_r2
        )
        print(f"   Condition curves loaded: {total} variables "
              f"({gwrf_loaded} GWRF, {neural_loaded} neural-PDE) — "
              f"{usable} usable with R\u00b2 \u2265 {self._condition_curve_min_r2}")

    # ------------------------------------------------------------------
    # Base-model consensus weights (from meta-ensemble feature importance)
    # ------------------------------------------------------------------

    def _compute_base_model_weights(self) -> None:
        """Compute per-base-model consensus weights.

        With the neural meta-learner, feature-importance extraction is not
        directly available, so we default to equal weights across the four
        base models.  Weights are used by the consensus delta approach:
        each base model's individual delta is weighted and summed.

        For causal inference, use Mode 3 (``run_with_causal_dag``) or the
        DML-derived coefficients from Stage 3 instead.
        """
        MODEL_KEYS = ("ols", "gwr", "gwrf", "ggpgam")

        self._base_model_weights = {k: 1.0 / len(MODEL_KEYS) for k in MODEL_KEYS}

        wstr = ", ".join(f"{k}={v:.3f}" for k, v in self._base_model_weights.items())
        print(f"   Base-model consensus weights (equal): {wstr}")
        print(f"     For causal inference, prefer Mode 3 (run_with_causal_dag).")

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

        if self._meta_model is not None:
            if verbose:
                print("   Running meta-ensemble...")
            final = self._meta_model.predict(preds, coords=coords, original_X=X)
        else:
            # V1 meta stacker not available — weighted average of base models
            if verbose:
                print("   Meta-ensemble not available — using weighted base-model average")
            weights = self._base_model_weights
            final = sum(weights.get(n, 0.25) * preds[n] for n in preds)
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

        When PDE solver artifacts are available, uses a Poisson forward solve
        (``poisson_solve``) for joint scenarios to capture cross-variable
        spatial interactions.  Otherwise, falls back to additive algebraic deltas.

        Returns a list of summary dicts (same schema as single-variable rows).
        """
        rows: List[dict] = []
        use_pde = self._pde_solver_available and len(self.features) > 0
        if use_pde and verbose:
            print("   PDE forward solver available — using for joint scenarios")

        for ix_sc in self.interaction_scenarios:
            name = ix_sc.get("name", "interaction")
            legs = ix_sc.get("legs", [])
            if not legs:
                continue

            if verbose:
                leg_desc = ", ".join(f"{l['variable']}Δ{l['delta']:+g}" for l in legs)
                print(f"\n--- Interaction: {name} ({leg_desc}) ---")

            # Build modified feature matrix for PDE path
            pde_attempted = False
            if use_pde:
                try:
                    combined_delta, pde_info = self._interaction_via_pde(
                        baseline_df, baseline_pred, target_mask, legs, verbose,
                    )
                    pde_attempted = True
                    if verbose:
                        print(f"   [PDE] converged={pde_info['converged']}, "
                              f"iters={pde_info['iterations']}, "
                              f"max_res={pde_info['max_residual']:.2e}")
                except Exception as _e:
                    if verbose:
                        print(f"   [PDE] failed ({_e}), falling back to algebraic")
                    pde_attempted = False

            if not pde_attempted:
                # Algebraic fallback: sum individual leg deltas
                combined_delta = np.zeros(target_mask.sum(), dtype=float)

                for leg in legs:
                    var_name = leg["variable"]
                    delta = float(leg.get("delta", 0))
                    if delta == 0 or var_name not in baseline_df.columns:
                        continue

                    original = baseline_df.loc[target_mask, var_name].values
                    modified = original + delta

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

            label_parts: List[str] = []
            for leg in legs:
                var_name = leg["variable"]
                delta = float(leg.get("delta", 0))
                if delta == 0 or var_name not in baseline_df.columns:
                    continue
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

        _persist_scenario_summary(
            summary_df,
            artifact_id="scenario_summary",
            output_dir=self.output_dir,
            disk_filename="scenario_summary.csv",
        )
        if verbose:
            print(f"\n   Summary saved")

        # Build GeoDataFrame if possible
        results_gdf = self._to_geodataframe(results_df, baseline_df)
        _persist_scenario_geo(
            results_gdf,
            artifact_id="scenario_results",
            output_dir=self.output_dir,
            disk_filename="scenario_results.gpkg",
        )
        if verbose and GEOPANDAS_AVAILABLE and isinstance(results_gdf, gpd.GeoDataFrame):
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
                # Bayesian β(s) mean (Stage-3 NUTS posterior) — the actual
                # coefficient driving direct-effect computation in v4.
                _beta_s = self._get_bayesian_beta(var_name)
                bayesian_coeff_mean = (
                    float(np.mean(_beta_s)) if _beta_s is not None and _beta_s.size > 0
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
                          f"β(s)={bayesian_coeff_mean:+.6f}, "
                          f"MGWR(diag)={mgwr_coeff_mean:+.6f}, lit/unit={lit_per_unit:+.4f})")
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
                        "Bayesian_Coeff_Mean": bayesian_coeff_mean,
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
                        "Bayesian_Coeff_Mean": bayesian_coeff_mean,
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
                    "Bayesian_Coeff_Mean": float('nan'),
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
                    "Bayesian_Coeff_Mean": float('nan'),
                    "MGWR_Coeff_Mean": float('nan'),
                    "Lit_Per_Unit": float('nan'),
                    "Coeff_Source": "joint",
                    "Physics_Lit_Coeff": float('nan'),
                    "Physics_Direction": "n/a",
                })

        # ── Save results ──────────────────────────────────────────────
        summary_df = pd.DataFrame(summary_rows)
        _persist_scenario_summary(
            summary_df,
            artifact_id="scenario_summary_dag",
            output_dir=self.output_dir,
            disk_filename="scenario_summary_dag.csv",
            also_register_as="scenario_summary",
        )
        if verbose:
            print(f"\n{'='*70}")
            print(f"DAG scenario summary saved")

        results_gdf = self._to_geodataframe(results_df, data)
        _persist_scenario_geo(
            results_gdf,
            artifact_id="scenario_results_dag",
            output_dir=self.output_dir,
            disk_filename="scenario_results_dag.gpkg",
            also_register_as="scenario_results",
        )
        if verbose and GEOPANDAS_AVAILABLE and isinstance(results_gdf, gpd.GeoDataFrame):
            print(f"GeoPackage saved")

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
        _persist_scenario_summary(
            summary_df,
            artifact_id="scenario_summary_reprediction",
            output_dir=self.output_dir,
            disk_filename="scenario_summary_reprediction.csv",
            also_register_as="scenario_summary",
        )
        if verbose:
            print(f"\n{'='*70}")
            print(f"Re-prediction scenario summary saved")

        results_gdf = self._to_geodataframe(results_df, data)
        _persist_scenario_geo(
            results_gdf,
            artifact_id="scenario_results_reprediction",
            output_dir=self.output_dir,
            disk_filename="scenario_results_reprediction.gpkg",
            also_register_as="scenario_results",
        )
        if verbose and GEOPANDAS_AVAILABLE and isinstance(results_gdf, gpd.GeoDataFrame):
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
        _persist_scenario_summary(
            summary_df,
            artifact_id="scenario_summary_hybrid",
            output_dir=self.output_dir,
            disk_filename="scenario_summary_hybrid.csv",
            also_register_as="scenario_summary",
        )
        if verbose:
            print(f"\n{'='*70}")
            print(f"Hybrid scenario summary saved")

        results_gdf = self._to_geodataframe(results_df, data)
        _persist_scenario_geo(
            results_gdf,
            artifact_id="scenario_results_hybrid",
            output_dir=self.output_dir,
            disk_filename="scenario_results_hybrid.gpkg",
            also_register_as="scenario_results",
        )
        if verbose and GEOPANDAS_AVAILABLE and isinstance(results_gdf, gpd.GeoDataFrame):
            print(f"GeoPackage saved")

        return summary_df, results_gdf

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
        mc_cols = [c for c in results_df.columns
                   if c.endswith(('_q05', '_q50', '_q95'))]
        if mc_cols:
            id_cols = [c for c in results_df.columns
                       if c in ('OBJECTID', 'Area_Name')]
            mc_df = results_df[id_cols + mc_cols].copy()
            store = _stage4_store()
            if store is not None:
                try:
                    store.write_table("4", "scenario_mc_uncertainty", mc_df,
                                      producer="scenario_simulator")
                except Exception:
                    store = None
            if store is None:
                mc_path = self.output_dir / "scenario_mc_uncertainty.csv"
                mc_df.to_csv(mc_path, index=False)
                if verbose:
                    print(f"  MC quantiles saved: {mc_path}")
            elif verbose:
                print(f"  MC quantiles saved to artifacts.db (scenario_mc_uncertainty)")

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
        mc_cols = [c for c in results_df.columns
                   if c.endswith(('_q05', '_q50', '_q95'))]
        store = _stage4_store()
        if mc_cols:
            id_cols = [c for c in results_df.columns if c in ('OBJECTID', 'Area_Name')]
            valid_id_cols = [c for c in id_cols if c in results_df.columns]
            mc_df = results_df[valid_id_cols + mc_cols].copy()
            wrote = False
            if store is not None:
                try:
                    store.write_table("4", "scenario_mc_consensus", mc_df,
                                      producer="scenario_simulator")
                    wrote = True
                except Exception:
                    wrote = False
            if not wrote:
                mc_path = self.output_dir / "scenario_mc_consensus.csv"
                mc_df.to_csv(mc_path, index=False)
                if verbose:
                    print(f"  MC quantiles saved: {mc_path}")
            elif verbose:
                print(f"  MC quantiles saved to artifacts.db (scenario_mc_consensus)")

        wrote_summary = False
        if store is not None:
            try:
                store.write_table("4", "scenario_mc_consensus_summary", mc_summary,
                                  producer="scenario_simulator")
                wrote_summary = True
            except Exception:
                wrote_summary = False
        if not wrote_summary:
            mc_summary_path = self.output_dir / "scenario_mc_consensus_summary.csv"
            mc_summary.to_csv(mc_summary_path, index=False)
            if verbose:
                print(f"  MC summary saved: {mc_summary_path}")
        elif verbose:
            print(f"  MC summary saved to artifacts.db (scenario_mc_consensus_summary)")

        return summary_df, mc_meta

    # ------------------------------------------------------------------
    # V2 Bayesian scenario simulation
    # ------------------------------------------------------------------

    def run_bayesian_scenarios(
        self,
        data=None,
        n_posterior_samples: int = 200,
        verbose: bool = True,
    ):
        """DEPRECATED — removed in SPARC v4.

        The legacy V2-Bayesian scenario path has been superseded by the
        unified CausalEffectResolver + per-edge NUTS posterior +
        Bayesian Spatial CATE in mode_3_full_ensemble / mode_4_hybrid.
        Credible intervals are now native to all four scenario modes.
        """
        raise RuntimeError(
            "scenario_mode='bayesian' (run_bayesian_scenarios) was removed "
            "in SPARC v4. Use 'mode_3_full_ensemble' or 'auto' — credible "
            "intervals are native to all modes via per-edge NUTS + "
            "Bayesian Spatial CATE. See MANUAL.md for migration notes."
        )

    # ------------------------------------------------------------------
    # Spatial modulation (PDP slope × alpha_norm)
    # ------------------------------------------------------------------

    def _get_spatial_modulation(
        self,
        variable: str,
        effective_inc: np.ndarray,
        baseline_values: np.ndarray,
        modified_values: np.ndarray,
    ) -> np.ndarray:
        """Return per-point PDP_slope × alpha_norm factor, or ones."""
        n = len(effective_inc)
        modulation = np.ones(n)

        # Alpha normalisation
        if self._alpha_field is not None:
            n_a = min(n, len(self._alpha_field))
            alpha_s = self._alpha_field[:n_a]
            alpha_mean = float(np.mean(alpha_s))
            if alpha_mean > 1e-12:
                modulation[:n_a] *= alpha_s / alpha_mean
                print(f"     [SPATIAL_MOD] {variable}: alpha_mean_raw={alpha_mean:.6f}, "
                      f"alpha_norm_mean={np.mean(modulation[:n_a]):.4f}")

        # PDP slope
        curve = self._condition_curves.get(variable)
        if (
            curve is not None
            and curve.get('r2', 0) >= self._condition_curve_min_r2
            and baseline_values is not None
            and modified_values is not None
        ):
            try:
                from sparc.interventions.extrapolation_guard import (
                    predict_scenario_with_saturation,
                )
                pdp_delta = predict_scenario_with_saturation(
                    variable_name=variable,
                    baseline_values=baseline_values[:n],
                    modified_values=modified_values[:n],
                    condition_curve={
                        'grid_values': curve['grid_values'],
                        'pdp_values': curve['pdp_values'],
                    },
                    spatial_multiplier=None,
                )
                pdp_slope_raw = np.where(
                    np.abs(effective_inc) > 1e-8,
                    pdp_delta / effective_inc,
                    0.0,
                )
                # Normalize to pure spatial shape modifier (mean ≈ 1)
                # using absolute values so beta controls the sign.
                # Only use points with non-zero effective increment for the
                # normalization mean — saturated points (eff_inc ≈ 0) get
                # pdp_slope = 0, which is fine because delta_t = β×0×mod = 0.
                abs_pdp = np.abs(pdp_slope_raw)
                active_mask = np.abs(effective_inc) > 1e-8
                if active_mask.any():
                    abs_pdp_mean = float(np.mean(abs_pdp[active_mask]))
                else:
                    abs_pdp_mean = 0.0
                if abs_pdp_mean > 1e-10:
                    pdp_slope = abs_pdp / abs_pdp_mean
                else:
                    pdp_slope = np.ones_like(pdp_slope_raw)
                modulation *= pdp_slope
                print(f"     [SPATIAL_MOD] {variable}: pdp_delta_mean={np.mean(pdp_delta):.6f}, "
                      f"pdp_slope_raw_mean={np.mean(pdp_slope_raw):.6f}, "
                      f"|raw|_mean={abs_pdp_mean:.10f}, "
                      f"pdp_slope_mean={np.mean(pdp_slope):.4f}, "
                      f"final_mod_mean={np.mean(modulation):.4f}")
            except Exception as e:
                print(f"     [SPATIAL_MOD] {variable}: PDP failed: {e}")

        return modulation

        return modulation

