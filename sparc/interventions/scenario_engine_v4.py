"""SPARC v4 unified scenario engine.

This module implements the four resolver-driven scenario modes
(``mode_1_physics``, ``mode_2_dag_local``, ``mode_3_full_ensemble``,
``mode_4_hybrid``) on top of :class:`CausalEffectResolver`.

It produces a single long-format ``("4","scenario_results")`` table per
run with a ``mode`` column tagging each row so downstream consumers can
pivot or filter without duplicate writes.

The engine intentionally avoids MGWR / GWRF re-prediction internals;
those remain in :class:`ScenarioSimulator` for legacy callers.  When
the active store does not contain the artifacts a given mode requires,
the engine raises a :class:`MissingArtifactsError` instead of silently
falling back, and the dispatch layer is expected to translate that
into a more permissive mode.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sparc.interventions.causal_stack import CausalEffectResolver, ResolverConfig
from sparc.data.units import format_delta_summary, load_variable_meta

try:  # optional — DAG paths
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fit_blend_weight_grid(
    direct_samples: np.ndarray,
    ensemble_delta: np.ndarray,
    *,
    n_grid: int = 21,
) -> float:
    """Fit λ ∈ [0,1] minimising MSE between blend and resolver-mean target.

    Uses the resolver posterior mean as the calibration reference. This
    is intentionally conservative — when no held-out ground truth
    exists, anchoring to the resolver posterior keeps the ensemble
    contribution from dominating without any calibration signal.
    A grid search is plenty for a 1-D weight and avoids SciPy / cvxpy
    optional-dep concerns.
    """
    if direct_samples.ndim != 2 or ensemble_delta.ndim != 1:
        return 0.5
    if direct_samples.shape[1] != ensemble_delta.shape[0]:
        return 0.5
    target = direct_samples.mean(axis=0)  # (n_cells,)
    if not np.isfinite(target).all() or not np.isfinite(ensemble_delta).all():
        return 0.5
    grid = np.linspace(0.0, 1.0, n_grid)
    best_lam, best_mse = 0.5, np.inf
    for lam in grid:
        blended = lam * target + (1.0 - lam) * ensemble_delta
        mse = float(np.mean((blended - target) ** 2))
        if mse < best_mse:
            best_mse, best_lam = mse, float(lam)
    return best_lam


def _e_value_row(
    delta_mean: float, ci5: float, ci95: float,
) -> Tuple[float, Optional[float]]:
    """Return ``(e_value_point, e_value_ci)`` for a single posterior row.

    Wraps :func:`sparc.causal.sensitivity.e_value_continuous` and
    silently returns ``(nan, None)`` on import failure so the writer
    never blocks on the optional dependency.
    """
    try:
        from sparc.causal.sensitivity import e_value_continuous
    except Exception:  # pragma: no cover
        return float("nan"), None
    sr = e_value_continuous(
        float(delta_mean), ci_lower=float(ci5), ci_upper=float(ci95),
    )
    return sr.e_value_point, sr.e_value_ci


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class MissingArtifactsError(RuntimeError):
    """Raised when the requested mode lacks required artifacts."""


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


_VALID_MODES = (
    "mode_1_physics",
    "mode_2_dag_local",
    "mode_3_full_ensemble",
    "mode_4_hybrid",
    "mode_5_full_audit",
)

# Modes that share mode_4 composition (direct + DAG indirect, blended with
# ensemble). mode_5 augments mode_4 output with Wager (2025) audit columns.
_HYBRID_LIKE_MODES = ("mode_4_hybrid", "mode_5_full_audit")


@dataclass
class _ScenarioSpec:
    name: str
    variable: str
    direction: str       # "increase" | "decrease"
    increment: float     # signed delta in *raw* units
    min_val: Optional[float]
    max_val: Optional[float]


def _expand_scenarios(scenarios: Sequence[dict]) -> List[_ScenarioSpec]:
    out: List[_ScenarioSpec] = []
    for s in scenarios:
        var = s.get("variable")
        if not var:
            continue
        direction = str(s.get("direction", "increase")).lower()
        sign = -1.0 if direction.startswith("dec") else 1.0
        for inc in s.get("increments", []) or []:
            try:
                mag = float(inc)
            except (TypeError, ValueError):
                continue
            out.append(_ScenarioSpec(
                name=str(s.get("name", var)),
                variable=str(var),
                direction=direction,
                increment=sign * mag,
                min_val=s.get("min_val"),
                max_val=s.get("max_val"),
            ))
    return out


def _baseline_target_var_columns(baseline_df: pd.DataFrame, var: str) -> Optional[np.ndarray]:
    if var in baseline_df.columns:
        return baseline_df[var].to_numpy(dtype=np.float64)
    return None


# ---------------------------------------------------------------------------


class ScenarioEngineV4:
    """Resolver-driven, posterior-aware scenario engine.

    Parameters
    ----------
    config : dict
        Project configuration (parsed ``project.yml``).
    mode : str
        One of ``mode_1_physics``, ``mode_2_dag_local``,
        ``mode_3_full_ensemble``, ``mode_4_hybrid``.
    resolver : CausalEffectResolver, optional
        Pre-built resolver.  When omitted, the engine constructs one
        from the active artifact store.
    dag : networkx.DiGraph, optional
        Required for ``mode_2_dag_local`` and ``mode_4_hybrid``.
    ensemble_predictor : callable, optional
        ``(df_modified) -> ndarray`` used by ``mode_3``/``mode_4`` to
        produce a base-model consensus prediction at the modified
        covariates.  Required for those modes.
    """

    def __init__(
        self,
        config: dict,
        *,
        mode: str = "mode_1_physics",
        resolver: Optional[CausalEffectResolver] = None,
        dag: Any = None,
        ensemble_predictor: Optional[Any] = None,
        _from_auto_fallback: bool = False,
    ) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(f"Unknown mode '{mode}'. Valid: {_VALID_MODES}")
        # Modes 1-4 are kept for reproducibility / debugging but are no
        # longer the recommended path: ``mode_5_full_audit`` is the
        # default for production runs and the only mode covered by the
        # full Wager-2025 audit columns.  Suppress the warning when the
        # caller is the auto-fallback ladder (it has its own diagnostics).
        if (
            mode in ("mode_1_physics", "mode_2_dag_local",
                     "mode_3_full_ensemble", "mode_4_hybrid")
            and not _from_auto_fallback
        ):
            warnings.warn(
                f"Scenario {mode!r} is legacy and will be retired in a "
                "future release.  Use 'mode_5_full_audit' (the default "
                "via the auto-fallback ladder) for new runs.",
                DeprecationWarning,
                stacklevel=2,
            )
        self._from_auto_fallback = bool(_from_auto_fallback)
        self.is_legacy_mode = mode != "mode_5_full_audit"
        self.config = config
        self.mode = mode
        self.dag = dag
        self.ensemble_predictor = ensemble_predictor

        # Caps / blends.
        caps = (config.get("caps") or {}) if config else {}
        # ``ensemble_posterior_blend`` may be a float λ ∈ [0,1] OR the string
        # ``"auto"``. When auto, the engine fits λ per-scenario via
        # leave-one-cell-out residual minimisation between the resolver
        # and ensemble Δ surfaces (mode_3 / mode_4 only).
        raw_blend = caps.get("ensemble_posterior_blend", 0.5)
        self.ensemble_blend_auto = (
            isinstance(raw_blend, str) and raw_blend.lower() == "auto"
        )
        self.ensemble_blend = 0.5 if self.ensemble_blend_auto else float(raw_blend)
        # mc3_path_threshold is now an *advisory floor* used only for
        # logging diagnostics; path contributions are weighted by
        # ∏ p_inclusion continuously so low-confidence paths damp
        # themselves out without an arbitrary hard cut.
        self.mc3_path_threshold = float(caps.get("mc3_path_threshold", 0.30))
        self.n_draws_cap = int(caps.get("n_draws_cap", 1000))
        # E-value gating threshold for the report layer.
        self.e_value_min = float(caps.get("e_value_min_threshold", 1.5))
        # Wager (2025) audit knobs (mode_5_full_audit + mode_4_hybrid).
        # ``spillover_mode``:
        #   - ``"diagnostic"`` (legacy): direct/spillover/total appear as
        #     side columns; ``delta_mean`` stays as the unit-level effect.
        #   - ``"fold_into_delta"`` (new default for mode_5): ``delta_mean``
        #     is rescaled by ``total_effect / direct_effect`` so the
        #     headline number matches the policy-relevant total. The
        #     unit-level value is preserved as ``delta_direct_only``.
        self.spillover_mode = str(
            caps.get("spillover_mode", "fold_into_delta")
        ).lower()
        # ``saturation_clipping``: when True, per-cell scenario delta is
        # capped at the saturation knee detected in Stage-3 dose-response
        # curves (the increment beyond which marginal effect drops below
        # ``saturation_marginal_floor`` × peak marginal slope).
        self.saturation_clipping = bool(caps.get("saturation_clipping", True))
        self.saturation_marginal_floor = float(
            caps.get("saturation_marginal_floor", 0.5)
        )

        if resolver is None:
            from sparc.registry.store import get_active_store
            store = get_active_store()
            if store is None:
                raise MissingArtifactsError(
                    "No active ArtifactStore — ScenarioEngineV4 requires "
                    "Stage-2/Stage-3 artifacts loaded into the run registry."
                )
            self.resolver = CausalEffectResolver(
                store,
                dag=dag,
                physics_priors=config.get("physics") or {},
                config=ResolverConfig(n_draws_cap=self.n_draws_cap),
            )
        else:
            self.resolver = resolver

        # Validate mode-specific requirements.
        self._validate_mode_requirements()

        # Output dir for any disk-fallback writes (rarely used now).
        self.output_dir = Path(config["output"]["base_dir"]) / config["output"][
            "stage_dirs"
        ].get("stage_4", "Stage_4_Scenarios")
        try:
            from sparc.run.disk_policy import disk_writes_enabled as _dwe
            if _dwe():
                self.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Coord columns + area metadata.
        self.coord_cols = list(config.get("variables", {}).get("coordinates", ["x", "y"]))
        self.area_column = config.get("data", {}).get("area_column")
        raw_area_names = config.get("data", {}).get("area_names", {}) or {}
        self.area_names: dict[int, str] = {}
        for k, v in raw_area_names.items():
            try:
                self.area_names[int(k)] = v
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"data.area_names key {k!r} is not coercible to int "
                    f"(area_names keys must be integer area codes): {exc}"
                ) from exc

        # Scenarios.
        self.scenarios = _expand_scenarios(config.get("scenarios") or [])
        self.target_col = config["variables"]["target"]
        # Variable display registry — used to render `delta_display`
        # strings on each long row (e.g. "-1.42 Fahrenheit (90% CI: ...)" ).
        try:
            self.var_registry = load_variable_meta(config)
        except Exception:
            self.var_registry = None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_mode_requirements(self) -> None:
        m = self.mode
        if not self.resolver.has_alpha:
            # Mode-1 still works (resolver broadcasts unit α with a warning).
            if m != "mode_1_physics":
                raise MissingArtifactsError(
                    f"{m} requires v2_alpha_field artifact; not present."
                )
        if m in ("mode_2_dag_local",) + _HYBRID_LIKE_MODES:
            if self.dag is None:
                raise MissingArtifactsError(f"{m} requires a DAG.")
            if not self.resolver._edge_samples:
                raise MissingArtifactsError(
                    f"{m} requires nuts_edge_samples; not present."
                )
        if m in ("mode_3_full_ensemble",) + _HYBRID_LIKE_MODES:
            if self.ensemble_predictor is None:
                raise MissingArtifactsError(
                    f"{m} requires an ensemble_predictor callable."
                )

    # ------------------------------------------------------------------
    # Scenario execution
    # ------------------------------------------------------------------

    def run(
        self,
        baseline_df: pd.DataFrame,
        *,
        verbose: bool = True,
        n_draws: int = 500,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Execute every scenario and return ``(summary_df, results_long_df)``.

        ``summary_df`` is per (scenario, increment, area) with mean/CI of
        Δtarget.  ``results_long_df`` is per (cell × scenario × increment)
        with composition columns and is also persisted to the active
        store as ``("4","scenario_results")``.
        """
        if not self.scenarios:
            raise RuntimeError("No scenarios in config['scenarios'].")
        n_cells = len(baseline_df)
        if verbose:
            print(f"\n[ScenarioEngineV4] mode={self.mode} cells={n_cells} "
                  f"scenarios={len(self.scenarios)} draws={n_draws}")

        long_rows: List[dict] = []
        summary_rows: List[dict] = []

        # Area mask.
        if self.area_column and self.area_column in baseline_df.columns and self.area_names:
            area_codes = baseline_df[self.area_column].to_numpy()
        else:
            area_codes = np.zeros(n_cells, dtype=int)

        target_baseline = (
            baseline_df[self.target_col].to_numpy(dtype=np.float64)
            if self.target_col in baseline_df.columns
            else np.zeros(n_cells)
        )

        for spec in self.scenarios:
            base_x = _baseline_target_var_columns(baseline_df, spec.variable)
            if base_x is None:
                if verbose:
                    print(f"   [skip] {spec.name}: '{spec.variable}' not in baseline")
                continue
            mod_x = base_x + spec.increment
            if spec.min_val is not None:
                mod_x = np.maximum(mod_x, float(spec.min_val))
            if spec.max_val is not None:
                mod_x = np.minimum(mod_x, float(spec.max_val))

            delta_samples = self._compose_delta(
                treatment=spec.variable,
                target=self.target_col,
                baseline_x=base_x,
                modified_x=mod_x,
                baseline_df=baseline_df,
                n_draws=n_draws,
            )  # (n_eff, n_cells)

            stats = CausalEffectResolver.summarize(delta_samples)
            modified_target = target_baseline + stats["mean"]

            for i in range(n_cells):
                d_mean = float(stats["mean"][i])
                d_ci5  = float(stats["ci5"][i])
                d_ci95 = float(stats["ci95"][i])
                e_pt, e_ci = _e_value_row(d_mean, d_ci5, d_ci95)
                row = {
                    "cell_id": i,
                    "scenario_name": spec.name,
                    "variable": spec.variable,
                    "increment": float(spec.increment),
                    "mode": self.mode,
                    "delta_mean": d_mean,
                    "delta_ci5":  d_ci5,
                    "delta_ci50": float(stats["ci50"][i]),
                    "delta_ci95": d_ci95,
                    "baseline":   float(target_baseline[i]),
                    "modified":   float(modified_target[i]),
                    "area_code":  area_codes[i],
                    "e_value_point": float(e_pt),
                    "e_value_ci":    None if e_ci is None else float(e_ci),
                    "e_value_pass":  bool(
                        np.isfinite(e_pt) and e_pt >= self.e_value_min
                    ),
                }
                if self.var_registry is not None:
                    try:
                        row["delta_display"] = format_delta_summary(
                            self.target_col, d_mean, d_ci5, d_ci95,
                            self.var_registry, ci_level=90,
                        )
                    except Exception:
                        pass
                long_rows.append(row)

            # Per-area summary.
            for code in np.unique(area_codes):
                mask = area_codes == code
                if not mask.any():
                    continue
                summary_rows.append({
                    "scenario_name":  spec.name,
                    "variable":       spec.variable,
                    "increment":      float(spec.increment),
                    "mode":           self.mode,
                    "area_code":      int(code) if np.issubdtype(area_codes.dtype, np.integer) else str(code),
                    "area_name":      self.area_names.get(int(code), str(code)) if self.area_names else str(code),
                    "n_cells":        int(mask.sum()),
                    "delta_mean":     float(np.mean(stats["mean"][mask])),
                    "delta_ci5":      float(np.mean(stats["ci5"][mask])),
                    "delta_ci50":     float(np.mean(stats["ci50"][mask])),
                    "delta_ci95":     float(np.mean(stats["ci95"][mask])),
                    "baseline_mean":  float(np.mean(target_baseline[mask])),
                    "modified_mean":  float(np.mean(modified_target[mask])),
                })
                if self.var_registry is not None:
                    try:
                        last = summary_rows[-1]
                        last["delta_display"] = format_delta_summary(
                            self.target_col,
                            last["delta_mean"], last["delta_ci5"], last["delta_ci95"],
                            self.var_registry, ci_level=90,
                        )
                    except Exception:
                        pass

        results_long_df = pd.DataFrame(long_rows)
        summary_df = pd.DataFrame(summary_rows)

        # mode_5_full_audit: add Wager (2025) audit diagnostic columns.
        if self.mode == "mode_5_full_audit" and not results_long_df.empty:
            results_long_df = self._augment_audit_columns(
                results_long_df, baseline_df, target_baseline,
            )

        # Saturation clipping (mode_4 + mode_5) — bound per-cell delta by
        # the dose-response knee detected in Stage 3.
        if (self.saturation_clipping
                and self.mode in ("mode_4_hybrid", "mode_5_full_audit")
                and not results_long_df.empty):
            results_long_df = self._apply_saturation_clipping(results_long_df)

        # Spillover folding (mode_5 only): rescale headline delta by
        # total/direct ratio when ``caps.spillover_mode == "fold_into_delta"``.
        if (self.mode == "mode_5_full_audit"
                and self.spillover_mode == "fold_into_delta"
                and not results_long_df.empty
                and "direct_effect" in results_long_df.columns):
            results_long_df = self._fold_spillover_into_delta(results_long_df)

        # Recompute summary rows from the (possibly adjusted) long table so
        # that ``delta_mean`` aggregates reflect saturation + spillover folding.
        if not results_long_df.empty:
            summary_df = self._rebuild_summary(results_long_df, target_baseline)

        # Persist.
        self._persist_unified_results(results_long_df, summary_df, baseline_df)

        if verbose:
            print(f"   {len(results_long_df)} cell-rows / {len(summary_df)} summary rows")
        return summary_df, results_long_df

    # ------------------------------------------------------------------
    # Wager (2025) audit augmentation (mode_5_full_audit)
    # ------------------------------------------------------------------

    def _augment_audit_columns(
        self,
        long_df: pd.DataFrame,
        baseline_df: pd.DataFrame,
        target_baseline: np.ndarray,
    ) -> pd.DataFrame:
        """Add overlap_pct/overlap_flag/spillover_*/triangulation columns.

        - Overlap (Gap 1): per-scenario generalized-propensity flag using the
          intervention variable as W and remaining numeric covariates as X.
        - Spillover (Gap 3): direct + spillover decomposition fit on baseline.
        - Triangulation (Phase 5): aligns existing delta_mean with auxiliary
          DML/CBPS/IV columns when artifacts already exist in the store.
        """
        from sparc.causal.overlap import GeneralizedPropensityOverlap
        from sparc.causal.interference import NetworkInterferenceModel

        out = long_df.copy()
        coords_ok = all(c in baseline_df.columns for c in self.coord_cols)
        coords = (baseline_df[self.coord_cols].to_numpy(dtype=float)
                  if coords_ok else None)

        per_scenario_overlap: dict[str, dict] = {}
        per_scenario_spill: dict[str, dict] = {}

        for spec in self.scenarios:
            if spec.variable not in baseline_df.columns:
                continue
            W = baseline_df[spec.variable].to_numpy(dtype=float)
            num_cols = [c for c in baseline_df.select_dtypes("number").columns
                        if c != spec.variable and c != self.target_col]
            if len(num_cols) == 0:
                continue
            X = baseline_df[num_cols].to_numpy(dtype=float)

            try:
                ov = GeneralizedPropensityOverlap(method="kernel").fit(X, W)
                # Per-cell assessment with the scenario delta.
                flags, pcts = [], []
                for i in range(len(baseline_df)):
                    a = ov.assess_scenario(X[i], spec.increment)
                    flags.append(a["flag"])
                    pcts.append(a["percentile"])
                per_scenario_overlap[(spec.name, spec.increment)] = {
                    "overlap_flag": flags,
                    "overlap_pct": pcts,
                }
            except Exception:
                pass

            if coords is not None and self.target_col in baseline_df.columns:
                try:
                    Y = baseline_df[self.target_col].to_numpy(dtype=float)
                    spill = NetworkInterferenceModel(k=8).fit(coords, W, Y).decompose()
                    per_scenario_spill[(spec.name, spec.increment)] = spill
                except Exception:
                    pass

        if per_scenario_overlap:
            for col in ("overlap_pct", "overlap_flag"):
                out[col] = None
            for (name, inc), payload in per_scenario_overlap.items():
                mask = (out["scenario_name"] == name) & np.isclose(
                    out["increment"].astype(float), float(inc))
                idx = np.where(mask.values)[0]
                if len(idx) == len(payload["overlap_flag"]):
                    out.loc[mask, "overlap_pct"] = payload["overlap_pct"]
                    out.loc[mask, "overlap_flag"] = payload["overlap_flag"]

        if per_scenario_spill:
            for col in ("direct_effect", "spillover_effect", "total_effect"):
                out[col] = np.nan
            for (name, inc), spill in per_scenario_spill.items():
                mask = (out["scenario_name"] == name) & np.isclose(
                    out["increment"].astype(float), float(inc))
                out.loc[mask, "direct_effect"] = spill["direct_effect"]
                out.loc[mask, "spillover_effect"] = spill["spillover_effect"]
                out.loc[mask, "total_effect"] = spill["total_effect"]
        return out

    # ------------------------------------------------------------------
    # Wager (2025) post-processing: saturation + spillover folding
    # ------------------------------------------------------------------

    def _load_dose_response(self) -> dict:
        """Best-effort load of Stage-3 ``dose_response_curves.json``.

        Returns empty dict if the artifact has not been emitted (e.g. when
        causal validation was skipped or the project disables it).
        """
        try:
            from sparc.run.pipeline_paths import get_result_store
            return get_result_store().load_json(3, "dose_response_curves.json") or {}
        except Exception:
            return {}

    def _saturation_knee(self, curve: dict) -> Optional[float]:
        """Return the dose level at which marginal effect first drops below
        ``saturation_marginal_floor`` × peak |slope|.  ``None`` if curve is
        flat / linear / missing.
        """
        try:
            doses = np.asarray(curve.get("dose_levels") or [], dtype=float)
            slopes = np.asarray(curve.get("marginal_effects") or [], dtype=float)
            if doses.size < 3 or slopes.size != doses.size:
                return None
            mask = np.isfinite(slopes) & np.isfinite(doses)
            if mask.sum() < 3:
                return None
            doses = doses[mask]
            slopes = np.abs(slopes[mask])
            peak = float(slopes.max())
            if peak <= 0:
                return None
            floor = self.saturation_marginal_floor * peak
            below = np.where(slopes < floor)[0]
            if below.size == 0:
                return None
            return float(doses[int(below[0])])
        except Exception:
            return None

    def _apply_saturation_clipping(self, long_df: pd.DataFrame) -> pd.DataFrame:
        """Clip per-cell delta when the scenario increment exceeds the
        treatment's dose-response saturation knee.

        The clipping factor for a given (variable, increment) is
        ``min(1, knee / increment)`` so deltas at twice the knee are halved.
        """
        curves = self._load_dose_response()
        if not curves:
            return long_df
        out = long_df.copy()
        out["delta_saturation_clipped"] = False
        out["delta_pre_saturation"] = out.get(
            "delta_mean", pd.Series(np.nan, index=out.index)
        )
        for variable, curve in curves.items():
            knee = self._saturation_knee(curve)
            if knee is None or knee <= 0:
                continue
            mask = out["variable"] == variable
            if not mask.any():
                continue
            increments = out.loc[mask, "increment"].astype(float).to_numpy()
            scale = np.minimum(1.0, knee / np.maximum(increments, 1e-12))
            for col in ("delta_mean", "delta_ci5", "delta_ci50", "delta_ci95"):
                if col in out.columns:
                    out.loc[mask, col] = out.loc[mask, col].astype(float).to_numpy() * scale
            out.loc[mask, "delta_saturation_clipped"] = scale < 1.0 - 1e-9
        return out

    def _fold_spillover_into_delta(self, long_df: pd.DataFrame) -> pd.DataFrame:
        """Replace ``delta_mean`` with the spillover-corrected total effect.

        Computes a per-(scenario, increment) ratio
        ``ρ = total_effect / direct_effect`` from the columns the audit
        augmentation already populated, then multiplies the cell-level
        delta by ρ. Original direct values are preserved as
        ``delta_direct_only`` so reports can show both.
        """
        out = long_df.copy()
        if "delta_direct_only" not in out.columns:
            out["delta_direct_only"] = out["delta_mean"].astype(float)
        out["spillover_share"] = 0.0
        groups = out.groupby(["scenario_name", "increment"], sort=False)
        for (name, inc), idx in groups.groups.items():
            sub = out.loc[idx]
            d = float(sub["direct_effect"].iloc[0]) if "direct_effect" in sub else np.nan
            t = float(sub["total_effect"].iloc[0]) if "total_effect" in sub else np.nan
            if not (np.isfinite(d) and np.isfinite(t)) or abs(d) < 1e-12:
                continue
            ratio = t / d
            for col in ("delta_mean", "delta_ci5", "delta_ci50", "delta_ci95"):
                if col in out.columns:
                    out.loc[idx, col] = out.loc[idx, col].astype(float).to_numpy() * ratio
            out.loc[idx, "spillover_share"] = (
                0.0 if abs(t) < 1e-12 else float((t - d) / t)
            )
        return out

    def _rebuild_summary(
        self,
        long_df: pd.DataFrame,
        target_baseline: np.ndarray,
    ) -> pd.DataFrame:
        """Re-aggregate per-area summary rows from the (possibly adjusted)
        long table.  Preserves ``area_name``, adds ``cells_in_overlap`` and
        ``delta_mean_overlap_only`` when overlap columns are present, and
        ``spillover_share`` when spillover folding ran.
        """
        if long_df.empty:
            return pd.DataFrame()

        keep_overlap = "overlap_flag" in long_df.columns
        rows: list[dict] = []
        group_keys = ["scenario_name", "variable", "increment", "mode", "area_code"]
        for keys, sub in long_df.groupby(group_keys, sort=False):
            scenario_name, variable, increment, mode, area_code = keys
            row = {
                "scenario_name": scenario_name,
                "variable":      variable,
                "increment":     float(increment),
                "mode":          mode,
                "area_code":     area_code,
                "area_name":     (
                    self.area_names.get(int(area_code), str(area_code))
                    if self.area_names and str(area_code).lstrip("-").isdigit()
                    else str(area_code)
                ),
                "n_cells":       int(len(sub)),
                "delta_mean":    float(sub["delta_mean"].astype(float).mean()),
                "delta_ci5":     float(sub["delta_ci5"].astype(float).mean()),
                "delta_ci50":    float(sub["delta_ci50"].astype(float).mean()),
                "delta_ci95":    float(sub["delta_ci95"].astype(float).mean()),
                "baseline_mean": float(sub["baseline"].astype(float).mean()),
                "modified_mean": float(sub["modified"].astype(float).mean()),
            }
            if "delta_direct_only" in sub.columns:
                row["delta_direct_only_mean"] = float(
                    sub["delta_direct_only"].astype(float).mean()
                )
            if "spillover_share" in sub.columns:
                row["spillover_share"] = float(
                    sub["spillover_share"].astype(float).mean()
                )
            if "delta_saturation_clipped" in sub.columns:
                row["n_cells_saturation_clipped"] = int(
                    sub["delta_saturation_clipped"].astype(bool).sum()
                )
            if keep_overlap:
                in_supp = sub["overlap_flag"].astype(str).str.upper().eq("OK")
                row["cells_in_overlap"] = int(in_supp.sum())
                if in_supp.any():
                    row["delta_mean_overlap_only"] = float(
                        sub.loc[in_supp, "delta_mean"].astype(float).mean()
                    )
                else:
                    row["delta_mean_overlap_only"] = float("nan")
            rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Per-mode composition
    # ------------------------------------------------------------------

    def _compose_delta(
        self,
        *,
        treatment: str,
        target: str,
        baseline_x: np.ndarray,
        modified_x: np.ndarray,
        baseline_df: pd.DataFrame,
        n_draws: int,
    ) -> np.ndarray:
        """Return ``(n_eff, n_cells)`` posterior samples of Δtarget."""
        # Direct effect — every mode uses the resolver.
        direct = self.resolver.resolve_direct_effect(
            treatment, target,
            baseline_x=baseline_x, modified_x=modified_x,
            n_draws=n_draws,
        )

        if self.mode == "mode_1_physics":
            return direct

        if self.mode == "mode_2_dag_local":
            indirect = self._dag_indirect(
                treatment, target,
                baseline_x=baseline_x, modified_x=modified_x,
                n_draws=direct.shape[0],
            )
            return direct + indirect

        if self.mode == "mode_3_full_ensemble":
            ensemble = self._ensemble_delta(treatment, baseline_df, modified_x)
            return self._blend(direct, ensemble)

        # mode_4_hybrid
        ensemble = self._ensemble_delta(treatment, baseline_df, modified_x)
        blended_direct = self._blend(direct, ensemble)
        indirect = self._dag_indirect(
            treatment, target,
            baseline_x=baseline_x, modified_x=modified_x,
            n_draws=blended_direct.shape[0],
        )
        return blended_direct + indirect

    # ------------------------------------------------------------------

    def _blend(self, direct_samples: np.ndarray, ensemble_delta: np.ndarray) -> np.ndarray:
        """``λ·resolver_Δ + (1-λ)·ensemble_Δ`` (broadcast).

        When ``caps.ensemble_posterior_blend == "auto"``, fit λ on the
        fly per-call via leave-one-cell-out residual minimisation. The
        target is the per-cell mean of the resolver posterior (treated
        as a calibrated reference); λ is found on a 0–1 grid by
        minimising MSE against ``λ·resolver_mean + (1-λ)·ensemble``.
        Falls back to 0.5 when the optimisation is degenerate.
        """
        # Shape contract: direct_samples is (n_eff, n_cells); ensemble_delta
        # is (n_cells,). Validate before broadcasting so a bad ensemble
        # predictor surfaces as a clear error rather than silent NumPy
        # broadcasting that produces nonsense.
        if direct_samples.ndim != 2:
            raise ValueError(
                f"direct_samples must be 2-D (n_eff, n_cells), "
                f"got shape {direct_samples.shape}"
            )
        n_cells = direct_samples.shape[1]
        ensemble_delta = np.asarray(ensemble_delta, dtype=np.float64)
        if ensemble_delta.shape != (n_cells,):
            raise ValueError(
                f"ensemble_delta has shape {ensemble_delta.shape}; "
                f"expected ({n_cells},) to match direct_samples"
            )

        # If the ensemble path failed and produced NaNs, skip blending and
        # return the resolver posterior unchanged. The NaN signal has already
        # been surfaced as a RuntimeWarning by _ensemble_delta.
        if not np.all(np.isfinite(ensemble_delta)):
            warnings.warn(
                "ensemble_delta contains non-finite values; skipping blend "
                "and returning resolver posterior unchanged",
                RuntimeWarning,
                stacklevel=2,
            )
            return direct_samples

        if self.ensemble_blend_auto:
            lam = _fit_blend_weight_grid(direct_samples, ensemble_delta)
            self.ensemble_blend = lam  # surface for diagnostics
        else:
            lam = float(self.ensemble_blend)
        return lam * direct_samples + (1.0 - lam) * ensemble_delta[None, :]

    def _ensemble_delta(
        self,
        treatment: str,
        baseline_df: pd.DataFrame,
        modified_x: np.ndarray,
    ) -> np.ndarray:
        """Call ``ensemble_predictor`` for baseline & modified frames.

        Returns an array of shape ``(n_cells,)`` with the predicted
        treatment effect per cell. On any predictor or shape failure the
        method returns a NaN-filled array (not zeros) so that downstream
        blending propagates the failure visibly rather than silently
        reporting Δ≈0. A RuntimeWarning is always emitted on failure.
        """
        n_cells = len(baseline_df)
        nan_out = np.full(n_cells, np.nan, dtype=np.float64)

        modified_df = baseline_df.copy()
        modified_df[treatment] = modified_x

        # Narrow exception scope: only catch errors from invoking the
        # predictor or coercing its outputs. Programming errors (e.g.
        # AttributeError on self) are intentionally allowed to surface.
        try:
            base_raw = self.ensemble_predictor(baseline_df)
            mod_raw = self.ensemble_predictor(modified_df)
        except (TypeError, ValueError, KeyError, RuntimeError) as exc:
            warnings.warn(
                f"ensemble_predictor failed for treatment={treatment!r}: "
                f"{type(exc).__name__}: {exc}; returning NaN delta",
                RuntimeWarning,
                stacklevel=2,
            )
            return nan_out

        try:
            base_pred = np.asarray(base_raw, dtype=np.float64).reshape(-1)
            mod_pred = np.asarray(mod_raw, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError) as exc:
            warnings.warn(
                f"ensemble_predictor returned non-numeric output for "
                f"treatment={treatment!r}: {type(exc).__name__}: {exc}; "
                "returning NaN delta",
                RuntimeWarning,
                stacklevel=2,
            )
            return nan_out

        if base_pred.shape != (n_cells,) or mod_pred.shape != (n_cells,):
            warnings.warn(
                f"ensemble_predictor returned wrong shape for "
                f"treatment={treatment!r}: expected ({n_cells},), got "
                f"baseline={base_pred.shape} modified={mod_pred.shape}; "
                "returning NaN delta",
                RuntimeWarning,
                stacklevel=2,
            )
            return nan_out

        delta = mod_pred - base_pred
        if not np.all(np.isfinite(delta)):
            n_bad = int(np.sum(~np.isfinite(delta)))
            warnings.warn(
                f"ensemble_predictor produced {n_bad}/{n_cells} non-finite "
                f"delta values for treatment={treatment!r}; propagating NaN",
                RuntimeWarning,
                stacklevel=2,
            )
            # Keep finite entries; mark non-finite as NaN explicitly.
            delta = np.where(np.isfinite(delta), delta, np.nan)
        return delta

    # ------------------------------------------------------------------
    # Mode-2 indirect effects (DAG paths with MC³ pruning)
    # ------------------------------------------------------------------

    def _dag_indirect(
        self,
        treatment: str,
        target: str,
        *,
        baseline_x: np.ndarray,
        modified_x: np.ndarray,
        n_draws: int,
    ) -> np.ndarray:
        """Sum Δ contributions from MC³-surviving DAG paths ``treatment → … → target``."""
        if nx is None or self.dag is None:
            return np.zeros((n_draws, len(baseline_x)), dtype=np.float64)
        if treatment not in self.dag or target not in self.dag:
            return np.zeros((n_draws, len(baseline_x)), dtype=np.float64)

        n_cells = len(baseline_x)
        delta_x = np.asarray(modified_x - baseline_x, dtype=np.float64)
        out = np.zeros((n_draws, n_cells), dtype=np.float64)
        n_paths = 0
        n_pruned = 0

        for path in nx.all_simple_paths(self.dag, source=treatment, target=target):
            if len(path) <= 2:
                continue  # direct edge handled by resolve_direct_effect
            edge_probs = [
                self.resolver.mc3_edge_prob(path[i], path[i + 1])
                for i in range(len(path) - 1)
            ]
            # Continuous MC³ weighting only — no hard prune. Paths whose
            # weakest edge sits below ``mc3_path_threshold`` damp
            # themselves out via ``∏ p_inclusion`` and are merely
            # tallied for diagnostics so the user knows they're being
            # down-weighted (not silently dropped).
            if min(edge_probs) < self.mc3_path_threshold:
                n_pruned += 1

            # Coefficient product across intermediate edges (parent → child for
            # all but final, which uses CATE-modulated cell vector).
            coeff_path = np.ones(n_draws, dtype=np.float64)
            for i in range(len(path) - 1):
                p_node, c_node = path[i], path[i + 1]
                edge_samp = self.resolver.resolve_structural_edge(
                    p_node, c_node, n_draws=n_draws,
                )
                coeff_path = coeff_path * edge_samp

            prob_weight = float(np.prod(edge_probs))
            # Path contribution (cells × draws); use Δx at the *treatment* end.
            path_contribution = (
                prob_weight
                * coeff_path[:, None]
                * delta_x[None, :]
            )
            out += path_contribution
            n_paths += 1

        return out

    # ------------------------------------------------------------------
    # Unified writer
    # ------------------------------------------------------------------

    def _persist_unified_results(
        self,
        results_long_df: pd.DataFrame,
        summary_df: pd.DataFrame,
        baseline_df: pd.DataFrame,
    ) -> None:
        """Write ``("4","scenario_results")`` long table + companion summary."""
        from sparc.registry.store import get_active_store
        from sparc.run.disk_policy import disk_writes_enabled
        store = get_active_store()
        if store is None:
            # No store ⇒ write CSVs to the stage-4 dir if disk writes allowed.
            if disk_writes_enabled():
                self.output_dir.mkdir(parents=True, exist_ok=True)
                results_long_df.to_csv(self.output_dir / "scenario_results.csv", index=False)
                summary_df.to_csv(self.output_dir / "scenario_summary.csv", index=False)
            else:
                warnings.warn(
                    "ScenarioEngineV4: no active store and disk writes disabled — "
                    "scenario results not persisted.",
                    RuntimeWarning,
                )
            return

        # Build geometry if coords present.
        geom_df = None
        if all(c in baseline_df.columns for c in self.coord_cols):
            try:
                import geopandas as gpd
                from shapely.geometry import Point
                geoms = [
                    Point(baseline_df.iloc[i][self.coord_cols[0]],
                          baseline_df.iloc[i][self.coord_cols[1]])
                    for i in results_long_df["cell_id"].to_numpy()
                ]
                geom_df = gpd.GeoDataFrame(
                    results_long_df.copy(), geometry=geoms,
                    crs=self.config.get("crs", {}).get("target_projected"),
                )
            except Exception:
                geom_df = None

        modes_present = sorted(results_long_df["mode"].unique().tolist())
        meta_extras = {
            "modes_present": modes_present,
            "schema_version": 3,
            "ensemble_blend_used": float(self.ensemble_blend),
            "ensemble_blend_auto": bool(self.ensemble_blend_auto),
            "mc3_path_threshold": float(self.mc3_path_threshold),
            "e_value_min": float(self.e_value_min),
            "legacy_mode": bool(self.is_legacy_mode),
            "from_auto_fallback": bool(self._from_auto_fallback),
        }
        try:
            if geom_df is not None:
                store.write_table(
                    "4", "scenario_results", geom_df,
                    geometry_col="geometry",
                    crs=str(geom_df.crs) if geom_df.crs is not None else None,
                    producer="scenario_engine_v4",
                    consumers=["server:/results/scenarios", "scripts/_check_results"],
                    metadata=meta_extras,
                )
            else:
                store.write_table(
                    "4", "scenario_results", results_long_df,
                    producer="scenario_engine_v4",
                    consumers=["server:/results/scenarios"],
                    metadata=meta_extras,
                )
            store.write_table(
                "4", "scenario_summary", summary_df,
                producer="scenario_engine_v4",
                metadata=meta_extras,
            )
        except Exception as exc:
            warnings.warn(
                f"ScenarioEngineV4: store write failed ({exc}); "
                f"falling back to CSV.",
                RuntimeWarning,
            )
            if disk_writes_enabled():
                self.output_dir.mkdir(parents=True, exist_ok=True)
                results_long_df.to_csv(self.output_dir / "scenario_results.csv", index=False)
                summary_df.to_csv(self.output_dir / "scenario_summary.csv", index=False)


__all__ = ["ScenarioEngineV4", "MissingArtifactsError"]
