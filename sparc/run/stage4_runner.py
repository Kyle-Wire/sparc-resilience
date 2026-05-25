"""sparc/run/stage4_runner.py — Stage 4: Scenario Simulation (DAG + Physics).

Entry point for ``PipelineOrchestrator`` when executing stage 4.

``main(ctx, *, fast_mode)`` is the canonical runner signature expected by
``PipelineOrchestrator._real_runner("4")``.  It wraps the scenario execution
logic using the dependencies carried on ``RunContext`` rather than relying on
global env vars or closure references.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sparc.run.orchestrator import RunContext


def main(ctx: "RunContext", *, fast_mode: bool = False) -> None:
    """Execute Stage 4: Scenario Simulation.

    Reads scenarios from ``ctx.config`` and delegates engine dispatch to
    ``ScenarioEngineSelector``.  Returns immediately when no scenarios are
    configured or when ``auto_run_scenarios_at_stage_4`` is disabled.

    Parameters
    ----------
    ctx : RunContext
        Pipeline-wide dependencies: config, paths, project_path.
    fast_mode : bool
        When True, skip optional heavy add-ons (MC uncertainty, sensitivity
        analysis, Wager-2025 post-processing).
    """
    config = ctx.config
    paths = ctx.paths

    scenarios = config.get("scenarios", [])
    auto_run = config.get("auto_run_scenarios_at_stage_4", True)

    if not scenarios:
        print("\n>>> Stage 4: No scenarios defined in project.yml — skipping.")
        return

    if not auto_run:
        print(
            "\n>>> Stage 4: Scenarios defined but `auto_run_scenarios_at_stage_4` "
            "is false — use the Scenario Runner page to launch."
        )
        return

    ctx.checkpoint(0, "Stage 4: scenario simulation starting")
    _run_scenario_engine(config, paths, fast_mode=fast_mode)
    ctx.checkpoint(100, "Stage 4: scenario simulation complete")


# ---------------------------------------------------------------------------
# Implementation — extracted from sparc/__main__._run_scenarios
# ---------------------------------------------------------------------------


def _run_scenario_engine(config: dict, paths, *, fast_mode: bool = False) -> tuple:
    """Execute scenario engine and post-processing.

    Returns (summary_df, results_gdf).
    """
    from sparc.interventions.scenario_simulator import ScenarioSimulator
    from sparc.run.scenario_engine_selector import ScenarioEngineSelector
    import pandas as pd

    sim = ScenarioSimulator(config)
    sim.load_models()

    csv_path = config["paths"]["raw_csv_path"]
    data = pd.read_csv(csv_path)

    dag_file = config.get("causal", {}).get("dag_file")
    has_dag = bool(dag_file and Path(dag_file).exists())

    selector = ScenarioEngineSelector(config, sim, data)

    requested_mode = config.get("pipeline", {}).get("scenario_mode", "auto")
    # _force_full_audit can be set via config (replaces the __main__ args-closure reference)
    force_full_audit = bool(config.get("_force_full_audit", False))
    scenario_mode = selector.resolve_mode(
        requested_mode, has_dag, force_full_audit=force_full_audit
    )

    force_legacy = bool(config.get("_force_legacy_scenarios"))
    summary_df, results_gdf = selector.run(
        scenario_mode, has_dag=has_dag, force_legacy=force_legacy
    )

    print(f"  Scenario summary: {len(summary_df)} rows  (mode={scenario_mode})")

    if not fast_mode:
        _run_conservation_checks(config, data, results_gdf)
        _run_mc_uncertainty(config, sim, data)
        _run_sensitivity_analysis(config, sim, data, paths)
        _run_wager_addons(config, data, summary_df, results_gdf, paths)

    return summary_df, results_gdf


def _run_conservation_checks(config: dict, data, results_gdf) -> None:
    """Run physics conservation checks on scenario results (best-effort)."""
    try:
        import numpy as np
        from sparc.interventions.physics_priors import ConservationChecker

        checker = ConservationChecker()
        for scenario in config.get("scenarios", []):
            var = scenario["variable"]
            for inc in scenario.get("increments", []):
                direction = scenario.get("direction", "increase")
                delta_signed = -inc if direction == "decrease" else inc
                col_label = (
                    f"total_{var}_{'minus' if delta_signed < 0 else 'plus'}_"
                    f"{str(inc).replace('.', 'p')}"
                )
                if hasattr(results_gdf, "columns") and col_label in results_gdf.columns:
                    deltas = {var: np.full(len(data), delta_signed)}
                    target_deltas = results_gdf[col_label].values
                    checker.check(data, deltas, target_deltas=target_deltas, verbose=True)
    except Exception as exc:
        print(f"  [CONSERVATION] Check skipped ({exc})")


def _run_mc_uncertainty(config: dict, sim, data) -> None:
    """Run Monte-Carlo uncertainty propagation (optional)."""
    run_mc = config.get("pipeline", {}).get("run_mc_uncertainty", False)
    n_mc = config.get("pipeline", {}).get("n_mc_draws", 50)
    if run_mc:
        print(f"\n  [Mode 2] MC uncertainty — Base-Model Consensus (n={n_mc})")
        try:
            mc_summary, mc_meta = sim.run_with_consensus_uncertainty(
                data, n_mc=n_mc, verbose=True,
            )
            print(f"  MC meta: {mc_meta}")
        except Exception as exc:
            print(f"  MC uncertainty propagation failed ({exc})")
    else:
        print(
            f"\n  [Mode 2] MC uncertainty skipped "
            f"(set run_mc_uncertainty: true, n_mc_draws: {n_mc} to enable)"
        )


def _run_sensitivity_analysis(config: dict, sim, data, paths) -> None:
    """Run global sensitivity analysis (optional)."""
    run_sa = config.get("pipeline", {}).get("run_sensitivity_analysis", False)
    sa_method = config.get("pipeline", {}).get("sensitivity_method", "morris")
    if run_sa:
        try:
            from sparc.evaluation.sensitivity import SensitivityAnalyzer

            sa = SensitivityAnalyzer(config, sim)
            sa_result = sa.run(data, method=sa_method, verbose=True)
            sa_out = paths.output_dir / f"sensitivity_{sa_method}.csv"
            sa_result["summary_df"].to_csv(sa_out, index=False)
            print(f"  Sensitivity analysis saved: {sa_out}")
        except Exception as exc:
            print(f"  Sensitivity analysis failed ({exc})")
    else:
        print(
            f"\n  [SA] Sensitivity analysis skipped "
            f"(set run_sensitivity_analysis: true to enable)"
        )


def _run_wager_addons(config: dict, data, summary_df, results_gdf, paths) -> None:
    """Run Stage-4 Wager-2025 add-ons (all best-effort)."""
    try:
        from sparc.run.scenarios import (
            mirror_audit_artifacts_to_stage4,
            run_policy_replay,
            run_budget_optimization,
        )

        stage_dirs = (config.get("output") or {}).get("stage_dirs") or {}
        base = (config.get("output") or {}).get("base_dir", "output")
        s3 = stage_dirs.get("stage_3", "Stage_3_Causal_Validation")
        stage3_dir = os.path.join(str(paths.output_dir), s3)

        mirror_audit_artifacts_to_stage4(stage3_dir)
        run_policy_replay(
            config=config,
            data=data,
            summary_df=summary_df,
            results_long_df=results_gdf,
        )
        run_budget_optimization(
            config=config,
            results_long_df=results_gdf,
        )
    except Exception as exc:
        print(f"  [stage4 add-ons] failed ({exc}); continuing")
