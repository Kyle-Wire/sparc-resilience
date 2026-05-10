"""Central reasoning orchestrator for the SPARC pipeline.

The orchestrator owns three responsibilities:

1. **Stage dispatch** — call the appropriate stage entry point.
2. **Stage scoring** — extract a small handful of headline diagnostic metrics
   from the stage's return value (and best-effort from ``artifacts.db``) into a
   versioned :class:`~sparc.run.decisions.StageDecision`.
3. **Decision persistence** — write the :class:`StageDecision` as a struct
   artifact in ``artifacts.db`` so downstream stages and the desktop UI can
   read the reasoning trace.

In v1 the orchestrator's reasoning logic is intentionally thin: it only
records what already happened and produces no ``next_actions`` (except for
Stage 2 once the surrogate inner loop is wired in). Subsequent PRs grow the
reasoning surface (S3→S2 refutation gating, S2→S1 GWEN re-runs).

Both ``sparc.server.stream._execute_stage`` (WebSocket entry point) and
``sparc.__main__.cmd_run`` (CLI entry point) delegate stage execution to this
module so a single reasoner governs every run.

See ``docs/prd/prd-reasoning-engine-spine.md``.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

from sparc.run.decisions import (
    DecisionRecord,
    MetricEvaluation,
    NextAction,
    ProvenanceInfo,
    StageDecision,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_stage(
    stage: int,
    config: dict,
    *,
    fast: bool = False,
    skip_gwen: bool = False,
    project_path: Optional[str] = None,
    approval_gate: Optional[Callable[[dict], None]] = None,
    max_revisions: int = 2,  # reserved for future inter-stage feedback loops
) -> tuple[Any, StageDecision]:
    """Execute a single pipeline stage and return ``(raw_result, decision)``.

    This is the canonical dispatch path. ``raw_result`` is whatever the stage's
    ``main()`` function returns (callers may ``state.store_result`` it);
    ``decision`` is the :class:`StageDecision` that has already been persisted
    to ``artifacts.db`` (best-effort).

    Stage 4 (scenarios) has substantial caller-side logic (DAG load, v4 engine,
    Monte Carlo) and is **not** dispatched from here; callers should run their
    Stage 4 flow directly and then use :func:`score_stage` and
    :func:`persist_stage_decision` to emit a decision.
    """
    started = time.perf_counter()
    raw_result = _dispatch_stage(
        stage,
        config,
        fast=fast,
        skip_gwen=skip_gwen,
        project_path=project_path,
        approval_gate=approval_gate,
    )
    elapsed = time.perf_counter() - started

    decision = score_stage(stage, raw_result, elapsed_seconds=elapsed)
    persist_stage_decision(decision)
    return raw_result, decision


def score_stage(
    stage: int,
    result: Any,
    *,
    source: str = "legacy_dispatch",
    elapsed_seconds: Optional[float] = None,
    extra_decisions: Optional[list[DecisionRecord]] = None,
    extra_next_actions: Optional[list[NextAction]] = None,
    search_log: Optional[dict[str, Any]] = None,
    baseline_comparison: Optional[dict[str, Any]] = None,
) -> StageDecision:
    """Build a :class:`StageDecision` from a stage's return value.

    Metric extraction is best-effort: it inspects ``result`` for known keys
    and silently skips anything missing. When ``result`` is empty (some stages
    return ``None`` or only a status flag) the metrics dict will be empty —
    that's an acceptable v1 outcome and downstream stages can still read the
    decision envelope.
    """
    metrics = _extract_metrics_for_stage(stage, result)

    # Status policy: "passed" by default; a stage can flip this by returning a
    # dict with ``{"status": "failed"}``. Future inter-stage rules live here.
    status: str = "passed"
    if isinstance(result, dict):
        raw_status = result.get("status")
        if isinstance(raw_status, str) and raw_status in {"passed", "failed", "needs_revision"}:
            status = raw_status

    decisions = list(extra_decisions or [])
    if elapsed_seconds is not None:
        decisions.append(
            DecisionRecord(
                rule="stage_timing",
                evidence={"elapsed_seconds": round(float(elapsed_seconds), 4)},
                action="accept",
            )
        )

    next_actions = list(extra_next_actions or [])

    provenance = ProvenanceInfo(
        source=source,
        search_log=search_log,
        baseline_comparison=baseline_comparison,
    )

    return StageDecision(
        stage=stage,
        status=status,  # type: ignore[arg-type]
        metrics=metrics,
        decisions=decisions,
        next_actions=next_actions,
        provenance=provenance,
    )


def persist_stage_decision(decision: StageDecision) -> bool:
    """Write the decision as a struct artifact. Returns ``True`` on success."""
    try:
        from sparc.registry.store import get_active_store

        store = get_active_store()
        if store is None:
            return False
        store.write_struct(
            stage=decision.stage,
            artifact_id=f"stage_{decision.stage}_decision",
            payload=decision.to_artifact_struct(),
            producer="sparc.run.orchestrator",
        )
        return True
    except Exception as exc:  # noqa: BLE001
        # Decisions are observability — never let persistence failures
        # break a pipeline run.
        print(f"[orchestrator] decision persistence failed for stage {decision.stage}: {exc}")
        return False


# ---------------------------------------------------------------------------
# Internal: dispatch
# ---------------------------------------------------------------------------


def _dispatch_stage(
    stage: int,
    config: dict,
    *,
    fast: bool,
    skip_gwen: bool,
    project_path: Optional[str],
    approval_gate: Optional[Callable[[dict], None]],
) -> Any:
    """Call the appropriate stage entry point. Mirrors legacy dispatch."""
    if stage == 0:
        from sparc.run.correlogram_analysis import main as run_correlogram

        return run_correlogram(fast_mode=fast)

    if stage == 1:
        if skip_gwen:
            return None
        from sparc.run.gwen_variable_selection import main as run_gwen

        return run_gwen(config_path=project_path, fast_mode=fast)

    if stage == 2:
        from sparc.run.enhanced_spatial_cv import main as run_spatial_cv

        return run_spatial_cv(fast_mode=fast)

    if stage == 3:
        from sparc.run.causal_validation import main as run_causal_validation

        if approval_gate is not None:
            return run_causal_validation(approval_gate=approval_gate)
        return run_causal_validation()

    if stage == 4:
        # Stage 4 has caller-side scenario orchestration. Returning None here
        # keeps run_stage() usable for stage-4 timing/decision emission while
        # leaving the actual scenario dispatch to the caller.
        return None

    raise ValueError(f"Unknown stage: {stage!r}")


# ---------------------------------------------------------------------------
# Internal: metric extraction
# ---------------------------------------------------------------------------


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


def _extract_metrics_for_stage(stage: int, result: Any) -> dict[str, MetricEvaluation]:
    """Best-effort metric extraction. Reads from ``result`` first, then from
    the active artifact store as a fallback."""
    metrics: dict[str, MetricEvaluation] = {}

    if not isinstance(result, dict):
        result = {}

    if stage == 0:
        # correlogram_analysis.main returns a comprehensive dict; key names
        # are best-effort.
        n_predictors = _coerce_float(result.get("n_predictors") or result.get("n_variables"))
        if n_predictors is not None:
            metrics["n_predictors_analyzed"] = MetricEvaluation(
                name="n_predictors_analyzed",
                value=n_predictors,
                direction="higher_is_better",
            )
        kappa_mean = _coerce_float(result.get("mean_kappa"))
        if kappa_mean is not None:
            metrics["mean_kappa"] = MetricEvaluation(
                name="mean_kappa",
                value=kappa_mean,
                direction="higher_is_better",
            )

    elif stage == 1:
        n_features = _coerce_float(result.get("n_features_selected"))
        if n_features is not None:
            metrics["n_features_selected"] = MetricEvaluation(
                name="n_features_selected",
                value=n_features,
                direction="higher_is_better",
            )

    elif stage == 2:
        # Try canonical keys first; fall back to artifact store.
        for key, name, direction in (
            ("oof_r2", "oof_r2", "higher_is_better"),
            ("oof_rmse", "oof_rmse", "lower_is_better"),
            ("residual_morans_i", "residual_morans_i", "lower_is_better"),
            ("meta_r2", "meta_r2", "higher_is_better"),
        ):
            v = _coerce_float(result.get(key))
            if v is not None:
                metrics[name] = MetricEvaluation(
                    name=name,
                    value=v,
                    direction=direction,  # type: ignore[arg-type]
                )

        if not metrics:
            for k, v in _stage2_metrics_from_store().items():
                metrics[k] = v

    elif stage == 3:
        for key, name, direction in (
            ("mc3_acceptance_rate", "mc3_acceptance_rate", "higher_is_better"),
            ("nuts_rhat_max", "nuts_rhat_max", "lower_is_better"),
            ("nuts_divergences", "nuts_divergences", "lower_is_better"),
        ):
            v = _coerce_float(result.get(key))
            if v is not None:
                metrics[name] = MetricEvaluation(
                    name=name,
                    value=v,
                    direction=direction,  # type: ignore[arg-type]
                )

    elif stage == 4:
        n_scenarios = _coerce_float(result.get("n_scenarios"))
        if n_scenarios is not None:
            metrics["n_scenarios"] = MetricEvaluation(
                name="n_scenarios",
                value=n_scenarios,
                direction="higher_is_better",
            )
        mean_delta = _coerce_float(result.get("mean_predicted_delta"))
        if mean_delta is not None:
            metrics["mean_predicted_delta"] = MetricEvaluation(
                name="mean_predicted_delta",
                value=mean_delta,
                direction="higher_is_better",
            )

    return metrics


def _stage2_metrics_from_store() -> dict[str, MetricEvaluation]:
    """Fallback Stage 2 metric extraction from artifacts.db."""
    out: dict[str, MetricEvaluation] = {}
    try:
        from sparc.registry.store import get_active_store

        store = get_active_store()
        if store is None:
            return out
        # Best-effort read of the canonical Stage 2 results struct.
        for candidate_id in ("final_ensemble_results", "ensemble_results"):
            try:
                if not store.has(2, candidate_id):
                    continue
                payload = store.read_struct(2, candidate_id)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            for src_key, name, direction in (
                ("test_r2", "oof_r2", "higher_is_better"),
                ("test_rmse", "oof_rmse", "lower_is_better"),
                ("residual_morans_i", "residual_morans_i", "lower_is_better"),
            ):
                v = _coerce_float(payload.get(src_key))
                if v is not None:
                    out[name] = MetricEvaluation(
                        name=name,
                        value=v,
                        direction=direction,  # type: ignore[arg-type]
                    )
            if out:
                break
    except Exception:
        return out
    return out


__all__ = ["run_stage", "score_stage", "persist_stage_decision"]
