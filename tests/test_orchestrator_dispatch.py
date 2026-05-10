"""Tests for sparc.run.orchestrator dispatch + scoring + persistence."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from sparc.run.decisions import StageDecision
from sparc.run.orchestrator import (
    persist_stage_decision,
    run_stage,
    score_stage,
)


# ---------------------------------------------------------------------------
# score_stage
# ---------------------------------------------------------------------------


def test_score_stage_extracts_stage2_metrics_from_result_dict():
    result = {
        "oof_r2": 0.83,
        "oof_rmse": 0.12,
        "residual_morans_i": -0.04,
        "meta_r2": 0.86,
    }
    decision = score_stage(2, result, source="legacy_dispatch", elapsed_seconds=1.234)
    assert decision.stage == 2
    assert decision.status == "passed"
    assert set(decision.metrics) == {"oof_r2", "oof_rmse", "residual_morans_i", "meta_r2"}
    assert decision.metrics["oof_r2"].value == pytest.approx(0.83)
    assert decision.metrics["oof_rmse"].direction == "lower_is_better"
    assert decision.provenance.source == "legacy_dispatch"
    # Timing decision was appended
    assert any(d.rule == "stage_timing" for d in decision.decisions)


def test_score_stage_handles_empty_or_none_result():
    d = score_stage(0, None)
    assert isinstance(d, StageDecision)
    assert d.metrics == {}
    assert d.status == "passed"


def test_score_stage_respects_explicit_failed_status():
    d = score_stage(3, {"status": "failed"})
    assert d.status == "failed"


def test_score_stage_ignores_non_numeric_metric_values():
    d = score_stage(2, {"oof_r2": "n/a", "oof_rmse": float("nan"), "meta_r2": 0.5})
    assert "oof_r2" not in d.metrics
    assert "oof_rmse" not in d.metrics
    assert d.metrics["meta_r2"].value == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# persist_stage_decision
# ---------------------------------------------------------------------------


def test_persist_stage_decision_no_active_store_returns_false():
    """When no artifact store is active, persistence is a no-op (returns False)."""
    with patch("sparc.registry.store.get_active_store", return_value=None):
        decision = score_stage(0, None)
        assert persist_stage_decision(decision) is False


def test_persist_stage_decision_writes_struct_when_store_active():
    decision = score_stage(2, {"oof_r2": 0.7})
    captured = {}

    class _FakeStore:
        def write_struct(self, *, stage, artifact_id, payload, producer=None):
            captured["stage"] = stage
            captured["artifact_id"] = artifact_id
            captured["payload"] = payload
            captured["producer"] = producer

    with patch("sparc.registry.store.get_active_store", return_value=_FakeStore()):
        ok = persist_stage_decision(decision)

    assert ok is True
    assert captured["stage"] == 2
    assert captured["artifact_id"] == "stage_2_decision"
    assert captured["producer"] == "sparc.run.orchestrator"
    assert captured["payload"]["schema_version"] == 1
    assert captured["payload"]["stage"] == 2


# ---------------------------------------------------------------------------
# run_stage dispatch
# ---------------------------------------------------------------------------


def test_run_stage_dispatches_to_correlogram_main(monkeypatch):
    sentinel = {"n_predictors": 7, "mean_kappa": 1.4}

    def fake_main(fast_mode=False):
        fake_main.called_with = fast_mode
        return sentinel

    fake_main.called_with = None  # type: ignore[attr-defined]

    import sparc.run.correlogram_analysis as corr_mod
    monkeypatch.setattr(corr_mod, "main", fake_main)

    # Disable persistence
    monkeypatch.setattr("sparc.registry.store.get_active_store", lambda: None)

    raw, decision = run_stage(0, config={}, fast=True)

    assert raw is sentinel
    assert fake_main.called_with is True  # type: ignore[attr-defined]
    assert decision.stage == 0
    # Metrics from the result dict should have been picked up.
    assert "n_predictors_analyzed" in decision.metrics
    assert decision.metrics["n_predictors_analyzed"].value == pytest.approx(7.0)
    assert "mean_kappa" in decision.metrics


def test_run_stage_skip_gwen_short_circuits_stage_1(monkeypatch):
    monkeypatch.setattr("sparc.registry.store.get_active_store", lambda: None)
    raw, decision = run_stage(1, config={}, skip_gwen=True)
    assert raw is None
    assert decision.stage == 1


def test_run_stage_unknown_stage_raises():
    with pytest.raises(ValueError):
        run_stage(99, config={})
