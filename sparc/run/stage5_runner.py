"""sparc/run/stage5_runner.py — Stage 5: Interpretation Report + Decision Ranking.

Entry point for ``PipelineOrchestrator`` when executing stage 5.

``main(ctx, *, fast_mode)`` is the canonical runner signature expected by
``PipelineOrchestrator._real_runner("5")``.  It wraps report generation and
the decision ranking stage using ``RunContext`` rather than relying on global
state or a ``Namespace`` constructed in ``__main__``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sparc.run.orchestrator import RunContext

from sparc.run.stage_result import StageResult


def main(ctx: "RunContext", *, fast_mode: bool = False) -> StageResult:
    """Execute Stage 5: report generation and decision ranking.

    Parameters
    ----------
    ctx : RunContext
        Pipeline-wide dependencies: config, paths, project_path.
    fast_mode : bool
        When True, skip optional heavy report formats (e.g. PDF).

    Returns
    -------
    StageResult
        ``.ok`` is ``True`` only when all executed steps succeeded.
    """
    ctx.checkpoint(0, "Stage 5: report generation starting")
    try:
        report_result = _run_report(ctx)
        if report_result is None:
            report_result = StageResult(ok=True, stage=5)
    except Exception as exc:  # pragma: no cover — helpers should handle internally
        report_result = StageResult(ok=False, stage=5, error=str(exc))
    ctx.checkpoint(60, "Stage 5: report complete")

    decision_result = StageResult(ok=True, stage=5)
    if not fast_mode:
        try:
            decision_result = _run_decision(ctx)
            if decision_result is None:
                decision_result = StageResult(ok=True, stage=5)
        except Exception as exc:  # pragma: no cover
            decision_result = StageResult(ok=False, stage=5, error=str(exc))
    ctx.checkpoint(100, "Stage 5: decision ranking complete")

    if not report_result.ok:
        return report_result
    return decision_result


# ---------------------------------------------------------------------------
# Named helpers (patchable in tests)
# ---------------------------------------------------------------------------


def _run_report(ctx: "RunContext") -> StageResult:
    """Generate audience reports for Stage 5.

    Wraps ``sparc.report.audience.generate_audience_report`` for each
    registered audience.  Exceptions are caught and logged; the returned
    :class:`StageResult` carries ``ok=False`` on failure so the caller knows.
    """
    try:
        from sparc.report.audience import AUDIENCES, FORMATS, generate_audience_report
        from sparc.report.stakeholder_index import write_stakeholder_index

        config = ctx.config
        paths = ctx.paths

        out_dir = Path(paths.output_dir) / "Stage_5_Reports"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"Generating report(s) for: "
            f"{config.get('project', {}).get('name', 'Unnamed')}"
        )
        print(f"  Output: {out_dir}")
        print(f"  Audiences: {', '.join(AUDIENCES)}  ·  Format: html")

        written: list[Path] = []
        for aud in AUDIENCES:
            try:
                payload = generate_audience_report(
                    run_dir=Path(paths.output_dir),
                    config=config,
                    audience=aud,
                    fmt="html",
                )
            except Exception as exc:
                print(f"  [{aud}] FAILED: {exc}")
                continue
            dest = out_dir / f"report_{aud}.html"
            dest.write_text(payload, encoding="utf-8")
            written.append(dest)
            print(f"  [{aud}] → {dest.name}")

        if written:
            try:
                write_stakeholder_index(out_dir, written)
            except Exception as exc:
                print(f"  [index] skipped ({exc})")

        return StageResult(ok=True, stage=5)

    except Exception as exc:
        print(f"\n>>> Stage 5 report generation failed ({exc})")
        return StageResult(ok=False, stage=5, error=str(exc))


def _run_decision(ctx: "RunContext") -> StageResult:
    """Run the decision ranking stage (best-effort).

    Reads ``scenario_results.json`` from the Stage 4 output directory and
    produces a ranked intervention list under ``Stage_5_Reports/``.
    """
    try:
        from sparc.run.decision_stage import run_decision_stage
        from sparc.decision import propose_candidates_from_scenarios

        config = ctx.config
        paths = ctx.paths

        s4_path = paths.output_dir / "Stage_4_Scenarios" / "scenario_results.json"
        # Allow stage_dirs override
        stage_dirs = (config.get("output") or {}).get("stage_dirs") or {}
        s4_name = stage_dirs.get("stage_4", "Stage_4_Scenarios")
        s4_path = Path(paths.output_dir) / s4_name / "scenario_results.json"

        if not s4_path.exists():
            print("  [Decision] No scenario_results.json — decision ranking skipped.")
            return

        with open(s4_path, encoding="utf-8") as fh:
            s4_data = json.load(fh)

        scenarios_list = (
            s4_data if isinstance(s4_data, list) else s4_data.get("scenarios", [])
        )
        candidates = propose_candidates_from_scenarios(scenarios_list)

        if not candidates:
            print("  [Decision] No candidates proposed — skipping.")
            return StageResult(ok=True, stage=5)

        stage5_dir = Path(paths.output_dir) / "Stage_5_Reports"
        stage5_dir.mkdir(exist_ok=True, parents=True)

        dec_cfg = config.get("decision", {}) or {}
        run_decision_stage(
            candidates=candidates,
            budget=dec_cfg.get("budget"),
            robustness_lambda=float(dec_cfg.get("robustness_lambda", 0.0)),
            minimise=bool(dec_cfg.get("minimise", False)),
            output_dir=str(stage5_dir),
        )
        return StageResult(ok=True, stage=5)

    except Exception as exc:
        print(f"  [Decision] Stage 5 decision ranking skipped: {exc}")
        return StageResult(ok=False, stage=5, error=str(exc))
