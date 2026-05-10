"""Tests for sparc.run.decisions — versioned StageDecision schema."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from sparc.run.decisions import (
    STAGE_DECISION_SCHEMA_VERSION,
    DecisionRecord,
    MetricEvaluation,
    NextAction,
    ProvenanceInfo,
    StageDecision,
)


def _minimal_decision(stage: int = 2) -> StageDecision:
    return StageDecision(
        stage=stage,
        status="passed",
        metrics={
            "r2": MetricEvaluation(
                name="r2",
                value=0.83,
                threshold=0.7,
                passed=True,
                direction="higher_is_better",
            ),
        },
        decisions=[
            DecisionRecord(
                rule="improvement_gate",
                evidence={"baseline": 0.812, "candidate": 0.829, "delta_pct": 2.1},
                action="accept",
            ),
        ],
        next_actions=[NextAction(kind="none", reason="ok")],
        provenance=ProvenanceInfo(source="surrogate_search"),
    )


def test_default_schema_version_is_1():
    d = _minimal_decision()
    assert d.schema_version == 1
    assert STAGE_DECISION_SCHEMA_VERSION == 1


def test_json_round_trip_preserves_all_fields():
    d = _minimal_decision()
    payload = d.to_artifact_struct()
    # Must be JSON-clean
    blob = json.dumps(payload)
    rehydrated = StageDecision.from_artifact_struct(json.loads(blob))
    assert rehydrated == d


def test_invalid_status_rejected():
    with pytest.raises(ValidationError):
        StageDecision(
            stage=2,
            status="weird",  # not in literal
            provenance=ProvenanceInfo(source="legacy_dispatch"),
        )


def test_invalid_decision_action_rejected():
    with pytest.raises(ValidationError):
        DecisionRecord(rule="x", evidence={}, action="maybe")


def test_metric_passed_threshold_flips_correctly():
    higher = MetricEvaluation(
        name="r2", value=0.6, threshold=0.7, passed=False, direction="higher_is_better"
    )
    assert higher.passed is False

    lower = MetricEvaluation(
        name="rmse",
        value=0.05,
        threshold=0.1,
        passed=True,
        direction="lower_is_better",
    )
    assert lower.passed is True


def test_provenance_search_log_optional():
    p = ProvenanceInfo(source="correlogram")
    assert p.search_log is None
    assert p.baseline_comparison is None


def test_metrics_dict_default_empty_and_addable():
    d = StageDecision(
        stage=0,
        status="passed",
        provenance=ProvenanceInfo(source="legacy_dispatch"),
    )
    assert d.metrics == {}
    assert d.decisions == []
    assert d.next_actions == []
