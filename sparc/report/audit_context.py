"""Aggregate every Stage-3+4 audit artifact into one typed dict.

The Stage-5 report builders (technical / planner / public / council /
scientist / equity / auditor) all share this single context object.
A dedicated module keeps the audience templates declarative — they
consume :func:`build_audit_context` rather than re-loading JSON.

Schema is best-effort: any missing artifact maps to ``None`` so
templates can guard with ``{% if x %}…{% endif %}``.
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict


class AuditContext(TypedDict, total=False):
    # Stage-3 Wager-2025 gaps
    overlap_diagnostics:        Optional[dict]
    rate_qini:                  Optional[dict]
    spillover_decomposition:    Optional[dict]
    cbps_estimates:             Optional[dict]
    policy_recommendations:     Optional[dict]
    did_estimates:              Optional[dict]
    iv_estimates:               Optional[dict]
    wager2025_summary:          Optional[dict]
    refutations:                Optional[dict]
    dose_response_curves:       Optional[dict]
    mediation_effects:          Optional[dict]
    coefficient_comparison:     Optional[dict]
    location_stratified_effects: Optional[dict]
    # Stage-4 add-ons
    scenario_results_policy_replay: Optional[dict]
    budget_optimization:        Optional[dict]
    # Top-level scenario summary
    scenario_summary:           Optional[list]


def _load_json(stage: int, name: str) -> Optional[Any]:
    try:
        from sparc.run.pipeline_paths import get_result_store
        return get_result_store().load_json(stage, name)
    except Exception:
        return None


def build_audit_context() -> AuditContext:
    """Aggregate every Stage-3/4 diagnostic artifact into one dict.

    All artifacts are loaded from the active :class:`ResultStore`. Missing
    artifacts are silently mapped to ``None`` — callers should branch on
    truthiness.
    """
    s3 = {
        "overlap_diagnostics":         _load_json(3, "overlap_diagnostics.json"),
        "rate_qini":                   _load_json(3, "rate_qini.json"),
        "spillover_decomposition":     _load_json(3, "spillover_decomposition.json"),
        "cbps_estimates":              _load_json(3, "cbps_estimates.json"),
        "policy_recommendations":      _load_json(3, "policy_recommendations.json"),
        "did_estimates":               _load_json(3, "did_estimates.json"),
        "iv_estimates":                _load_json(3, "iv_estimates.json"),
        "wager2025_summary":           _load_json(3, "wager2025_summary.json"),
        "refutations":                 _load_json(3, "refutations.json"),
        "dose_response_curves":        _load_json(3, "dose_response_curves.json"),
        "mediation_effects":           _load_json(3, "mediation_effects.json"),
        "coefficient_comparison":      _load_json(3, "coefficient_comparison.json"),
        "location_stratified_effects": _load_json(3, "location_stratified_effects.json"),
    }
    s4 = {
        "scenario_results_policy_replay": _load_json(4, "scenario_results_policy_replay.parquet"),
        "budget_optimization":            _load_json(4, "budget_optimization.json"),
    }
    # ResultStore.load_json may not return parquet; try alternate readers.
    if s4["scenario_results_policy_replay"] is None:
        try:
            from sparc.run.pipeline_paths import get_result_store
            df = get_result_store().load_dataframe(4, "scenario_results_policy_replay.parquet")
            s4["scenario_results_policy_replay"] = df.to_dict("records") if df is not None else None
        except Exception:
            s4["scenario_results_policy_replay"] = None

    out: AuditContext = {**s3, **s4}  # type: ignore[typeddict-item]
    return out


def gap_completion(ctx: AuditContext) -> dict[str, bool]:
    """Boolean roll-up for each Wager-2025 gap (10 gaps)."""
    return {
        "1_overlap":            bool(ctx.get("overlap_diagnostics")),
        "2_rate_qini":          bool(ctx.get("rate_qini")),
        "3_spillover":          bool(ctx.get("spillover_decomposition")),
        "4_cbps":               bool(ctx.get("cbps_estimates")),
        "5_did":                bool(ctx.get("did_estimates")),
        "6_iv":                 bool(ctx.get("iv_estimates")),
        "7_refutation":         bool(ctx.get("refutations")),
        "8_policy":             bool(ctx.get("policy_recommendations")),
        "9_dose_response":      bool(ctx.get("dose_response_curves")),
        "10_location_stratified": bool(ctx.get("location_stratified_effects")),
    }
