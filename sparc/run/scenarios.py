"""Stage-4 add-ons that consume Stage-3 Wager-2025 artifacts.

Three responsibilities:

* :func:`run_policy_replay`   — turn ``policy_recommendations.json`` into a
  named scenario whose results sit alongside the user-defined scenarios.
* :func:`run_budget_optimization` — call :mod:`sparc.scenario.budget` over
  the cell-level scenario delta surface and emit allocation + Pareto
  artifacts.
* :func:`mirror_audit_artifacts_to_stage4` — register Stage-3 audit
  artifacts (CBPS, refutations, …) under Stage-4 so the Stage-5 report
  builder finds them as "current run" outputs.

Each function is best-effort: a missing artifact triggers a friendly skip
message rather than aborting the whole pipeline.
"""

from __future__ import annotations

import os
import json
from typing import Any, Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------- helpers

def _load_json(stage: int, name: str) -> Optional[dict]:
    """Best-effort load via the ResultStore."""
    try:
        from sparc.run.pipeline_paths import get_result_store
        return get_result_store().load_json(stage, name)
    except Exception:
        return None


def _save_json(stage: int, name: str, payload: Any) -> Optional[str]:
    try:
        from sparc.run.pipeline_paths import get_result_store
        return str(get_result_store().save_json(stage, name, payload))
    except Exception:
        return None


def _save_dataframe(stage: int, name: str, df: pd.DataFrame) -> Optional[str]:
    try:
        from sparc.run.pipeline_paths import get_result_store
        return str(get_result_store().save_dataframe(stage, name, df))
    except Exception:
        return None


# ===================================================================== A4
def run_policy_replay(
    *,
    config: dict,
    data: pd.DataFrame,
    summary_df: Optional[pd.DataFrame] = None,
    results_long_df: Optional[pd.DataFrame] = None,
) -> Optional[pd.DataFrame]:
    """Generate a synthetic ``"Optimal Targeted (Wager-EWM)"`` scenario.

    Reads ``policy_recommendations.json`` (Stage 3, Gap 8) and combines it
    with the cell-level mean delta from ``scenario_results`` to produce a
    per-treatment summary row stating the *expected* welfare from rolling
    out the learned EWM rule. The result is saved as
    ``scenario_results_policy_replay`` under Stage 4 and returned for
    downstream use.

    The replay does *not* re-invoke the resolver; it re-uses the cell-level
    deltas already computed by Stage 4 for each user-defined scenario.
    For each treatment ``t`` the replayed delta is the average of
    ``delta_mean`` over recommended cells (``rec[i] == 1``) drawn from the
    *largest-increment* user scenario for that treatment. This gives the
    apples-to-apples "what happens if we apply this scenario only where
    EWM says to" number that the council/equity reports need.
    """
    rec_payload = _load_json(3, "policy_recommendations.json")
    if not rec_payload or not isinstance(rec_payload, dict):
        return None
    per_cell = rec_payload.get("per_cell_recommendations") or {}
    if not per_cell:
        return None
    if results_long_df is None or results_long_df.empty:
        return None

    rows: list[dict] = []
    for treatment, rec in per_cell.items():
        rec_arr = np.asarray(rec, dtype=int)
        if rec_arr.sum() == 0:
            continue
        # Pick the user scenario for this treatment with the largest
        # |increment| as the proxy "fully-deployed" intensity.
        cand = results_long_df[results_long_df["variable"] == treatment]
        if cand.empty:
            continue
        cand = cand.copy()
        cand["abs_inc"] = cand["increment"].astype(float).abs()
        biggest_inc = float(cand["abs_inc"].max())
        full_rows = cand[cand["abs_inc"] == biggest_inc]
        # Map cell_id → delta_mean; restrict to recommended cells.
        cell_idx = full_rows["cell_id"].astype(int).to_numpy()
        deltas = full_rows["delta_mean"].astype(float).to_numpy()
        if len(cell_idx) != len(rec_arr):
            # Recommendation length mismatch — fall back to averaging
            # over the intersection by index position.
            n = min(len(cell_idx), len(rec_arr))
            cell_idx = cell_idx[:n]
            deltas = deltas[:n]
            rec_use = rec_arr[:n]
        else:
            rec_use = rec_arr
        treated_mask = rec_use.astype(bool)
        if not treated_mask.any():
            continue
        treated_delta = float(np.mean(deltas[treated_mask]))
        all_delta = float(np.mean(deltas))
        rows.append({
            "scenario_name": "Optimal Targeted (Wager-EWM)",
            "variable":      treatment,
            "increment":     biggest_inc,
            "n_recommended": int(treated_mask.sum()),
            "n_total":       int(len(treated_mask)),
            "fraction_treated": float(treated_mask.mean()),
            "delta_mean_targeted":   treated_delta,
            "delta_mean_full_rollout": all_delta,
            "qini_uplift":   None if abs(all_delta) < 1e-12
                              else float(treated_delta / all_delta),
            "mode": "policy_replay",
        })

    if not rows:
        return None

    out = pd.DataFrame(rows)
    _save_dataframe(4, "scenario_results_policy_replay.parquet", out)
    print(f"  [policy replay] {len(out)} treatment(s) summarised → "
          f"scenario_results_policy_replay")
    for r in rows:
        share = 100.0 * r["fraction_treated"]
        print(f"    {r['variable']}: treat {share:.1f}% of cells → "
              f"Δ̄_targeted={r['delta_mean_targeted']:+.4f} "
              f"(vs full rollout {r['delta_mean_full_rollout']:+.4f})")
    return out


# ===================================================================== A5
def run_budget_optimization(
    *,
    config: dict,
    results_long_df: pd.DataFrame,
) -> Optional[dict]:
    """Allocate a budget across cells using the cell-level scenario delta.

    Reads ``scenarios.budget`` from project.yml:

    .. code-block:: yaml

        scenarios:
          budget:
            total: 1000000          # null → unconstrained Pareto only
            cost_per_unit:
              Pct_Canopy:    250
              Pct_Impervious: 180
              Albedo:         90
            solver: greedy_2opt     # greedy | greedy_2opt | milp
            multipliers: [0.25, 0.5, 0.75, 1.0, 1.5]

    For each treatment present in ``results_long_df``, picks the largest
    user increment (the "fully deployed" surface), uses ``-delta_mean``
    as benefit (so cooling = positive benefit when target is temperature
    z-score), runs :mod:`sparc.scenario.budget` and persists allocation
    + Pareto frontier per treatment. Returns the assembled summary.
    """
    if results_long_df is None or results_long_df.empty:
        return None
    bcfg = (
        ((config.get("scenarios") if isinstance(config.get("scenarios"), dict)
          else None) or {}).get("budget")
        if isinstance(config.get("scenarios"), dict)
        else None
    )
    # ``scenarios:`` in project.yml is a *list* of scenario specs, so the
    # budget block is conventionally placed alongside rather than nested.
    if bcfg is None:
        bcfg = (config.get("budget") or {})
    if not bcfg:
        bcfg = {}

    cost_per_unit = (bcfg.get("cost_per_unit") or {})
    solver = str(bcfg.get("solver", "greedy_2opt")).lower()
    multipliers = bcfg.get("multipliers") or [0.25, 0.5, 0.75, 1.0, 1.5]
    total = bcfg.get("total")  # may be None → unconstrained

    try:
        from sparc.scenario.budget import optimize, pareto_sweep
        from sparc.scenario.benefit_surface import BenefitSurface
    except Exception as exc:
        print(f"  [budget] sparc.scenario.budget unavailable: {exc}")
        return None

    summary: dict[str, Any] = {"per_treatment": {}, "config": dict(bcfg)}
    allocations_long: list[dict] = []
    for treatment, sub in results_long_df.groupby("variable", sort=False):
        unit_cost = float(cost_per_unit.get(treatment, 1.0))
        surface = BenefitSurface.from_long_df(sub, unit_cost=unit_cost)
        if surface.benefits.max() == 0:
            continue
        # If a budget is set, run a single optimization at it AND a sweep.
        # If no budget set, do a unitless sweep with multipliers × max-cost.
        if total is None:
            base_budget = float(np.sum(surface.costs))  # 100% cost = treat all
        else:
            base_budget = float(total)
        try:
            res = optimize(
                **surface.to_optimize_args(), budget=base_budget,
                solver=solver,
            )
            sweep = pareto_sweep(
                **surface.to_optimize_args(), budget=base_budget,
                multipliers=tuple(multipliers),
                solver=solver,
            )
        except Exception as exc:
            print(f"  [budget] {treatment} skipped: {exc}")
            continue

        summary["per_treatment"][treatment] = {
            "base_budget": base_budget,
            "optimum": res.to_dict(),
            "pareto":  sweep.to_dict(),
            "n_cells_treated": res.n_treated,
        }
        # Annotate `optimum` with a display-string for the headline benefit,
        # expressed in *target* units.  Benefits are abs(delta_mean), so the
        # cumulative total_benefit approximates |total target change|.
        try:
            from sparc.data.units import format_value, load_variable_meta
            _var_reg = load_variable_meta(config)
            target_col = config.get("variables", {}).get("target")
            if target_col is not None:
                tot = float(res.total_benefit)
                # Display as negative change (cooling lowers the target).
                summary_value = -tot
                summary["per_treatment"][treatment]["optimum"][
                    "total_benefit_display"
                ] = format_value(summary_value, _var_reg.get(target_col))
        except Exception:
            pass
        for i, x in enumerate(res.allocation):
            allocations_long.append({
                "treatment": treatment,
                "cell_id": int(surface.cell_ids[i]),
                "allocation": float(x),
                "benefit": float(surface.benefits[i]),
                "cost": float(surface.costs[i]),
            })

        share_treated = 100.0 * res.n_treated / max(1, len(benefit))
        print(f"  [budget] {treatment}: solver={res.solver}  "
              f"benefit={res.total_benefit:.4f}  cost={res.total_cost:,.0f}  "
              f"({share_treated:.1f}% of cells treated)")

    if not summary["per_treatment"]:
        return None
    _save_json(4, "budget_optimization.json", summary)
    if allocations_long:
        _save_dataframe(4, "budget_allocation.parquet", pd.DataFrame(allocations_long))
    print(f"  [budget] {len(summary['per_treatment'])} treatment(s) optimised → "
          f"budget_optimization.json + budget_allocation.parquet")
    return summary


# ===================================================================== A6
_AUDIT_PASSTHROUGH_FILES = (
    "overlap_diagnostics.json",
    "rate_qini.json",
    "spillover_decomposition.json",
    "cbps_estimates.json",
    "policy_recommendations.json",
    "did_estimates.json",
    "wager2025_summary.json",
    "dose_response_curves.json",
    "mediation_effects.json",
    "coefficient_comparison.json",
    "location_stratified_effects.json",
    "refutations.json",
    "iv_estimates.json",
)


def mirror_audit_artifacts_to_stage4(stage3_dir: str) -> list[str]:
    """Re-register every Stage-3 audit artifact under Stage 4.

    Stage 5 builds its narrative from Stage-4 artifacts (the "current run"
    semantics in the registry).  By re-exporting Stage-3 outputs here we
    keep that contract without requiring report code to know which stage
    a given diagnostic originally came from.
    """
    mirrored: list[str] = []
    if not stage3_dir or not os.path.isdir(stage3_dir):
        return mirrored
    for fname in _AUDIT_PASSTHROUGH_FILES:
        src = os.path.join(stage3_dir, fname)
        if not os.path.exists(src):
            continue
        try:
            with open(src, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception:
            continue
        out = _save_json(4, fname, payload)
        if out:
            mirrored.append(fname)
    if mirrored:
        print(f"  [audit→stage4] mirrored {len(mirrored)} artifact(s) for report builder")
    return mirrored


# ===================================================================== main
def main(ctx: Any = None, *, fast_mode: bool = False) -> None:
    """Pipeline entry-point for Stage 4 (scenarios).

    Called by :mod:`sparc.run.stages` via the stage runner protocol
    ``main(ctx, fast_mode=<bool>)``.

    Parameters
    ----------
    ctx:
        ``RunContext`` from the pipeline orchestrator.  May be ``None`` in
        standalone / test invocations.
    fast_mode:
        When ``True``, skip heavy budget-optimisation passes.
    """
    try:
        from sparc.run.pipeline_paths import get_result_store
        store = get_result_store()
        stage3_dir = store.stage_dir(3)
    except Exception:
        stage3_dir = ""

    mirror_audit_artifacts_to_stage4(stage3_dir)

    if not fast_mode:
        try:
            from sparc.registry.artifact import ArtifactRegistry
            reg = ArtifactRegistry.current()
            policy = reg.load_json(3, "policy_recommendations") or {}
            if policy:
                run_policy_replay(config=policy, data=None)  # type: ignore[arg-type]
        except Exception:
            pass

        try:
            run_budget_optimization()
        except Exception:
            pass
