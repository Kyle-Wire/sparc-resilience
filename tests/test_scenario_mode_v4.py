"""Tests for Phase 5a/5f/5h: scenario_mode renumbering, alias translation,
auto-mode policy, and bayesian deprecation."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Schema enum
# ---------------------------------------------------------------------------


def test_scenario_mode_enum_renumbered():
    schema_path = Path(__file__).resolve().parent.parent / "sparc" / "config" / "project_schema.json"
    schema = json.loads(schema_path.read_text())
    enum = schema["properties"]["pipeline"]["properties"]["scenario_mode"]["enum"]
    assert enum == [
        "auto",
        "mode_1_physics",
        "mode_2_dag_local",
        "mode_3_full_ensemble",
        "mode_4_hybrid",
    ]
    # Legacy keys must be absent from the enum.
    for legacy in ("physics", "dag_coefficient", "model_reprediction", "hybrid", "bayesian"):
        assert legacy not in enum


# ---------------------------------------------------------------------------
# Auto-mode policy
# ---------------------------------------------------------------------------


class _StubStore:
    def __init__(self, present):
        self._present = set(present)

    def has(self, stage, artifact_id):
        return (str(stage), artifact_id) in self._present


def test_auto_mode_no_artifacts_returns_mode_1(monkeypatch):
    from sparc import __main__ as cli

    monkeypatch.setattr(
        "sparc.registry.store.get_active_store",
        lambda: _StubStore([]),
    )
    assert cli._resolve_auto_scenario_mode(has_dag=False) == "mode_1_physics"
    assert cli._resolve_auto_scenario_mode(has_dag=True) == "mode_1_physics"


def test_auto_mode_alpha_only_returns_mode_1(monkeypatch):
    from sparc import __main__ as cli

    monkeypatch.setattr(
        "sparc.registry.store.get_active_store",
        lambda: _StubStore([("2", "v2_alpha_field")]),
    )
    assert cli._resolve_auto_scenario_mode(has_dag=False) == "mode_1_physics"


def test_auto_mode_alpha_plus_ensemble_returns_mode_3(monkeypatch):
    from sparc import __main__ as cli

    monkeypatch.setattr(
        "sparc.registry.store.get_active_store",
        lambda: _StubStore([
            ("2", "v2_alpha_field"),
            ("2", "v2_neural_ensemble"),
        ]),
    )
    assert cli._resolve_auto_scenario_mode(has_dag=False) == "mode_3_full_ensemble"


def test_auto_mode_dag_plus_nuts_plus_alpha_returns_mode_2(monkeypatch):
    from sparc import __main__ as cli

    monkeypatch.setattr(
        "sparc.registry.store.get_active_store",
        lambda: _StubStore([
            ("2", "v2_alpha_field"),
            ("3", "nuts_edge_samples"),
        ]),
    )
    assert cli._resolve_auto_scenario_mode(has_dag=True) == "mode_2_dag_local"


def test_auto_mode_full_stack_returns_mode_4_hybrid(monkeypatch):
    from sparc import __main__ as cli

    monkeypatch.setattr(
        "sparc.registry.store.get_active_store",
        lambda: _StubStore([
            ("2", "v2_alpha_field"),
            ("2", "v2_neural_ensemble"),
            ("3", "nuts_edge_samples"),
        ]),
    )
    assert cli._resolve_auto_scenario_mode(has_dag=True) == "mode_4_hybrid"


# ---------------------------------------------------------------------------
# Bayesian deprecation
# ---------------------------------------------------------------------------


def test_run_bayesian_scenarios_raises_runtime_error():
    """Phase 5h: legacy method body replaced with deprecation stub."""
    from sparc.interventions.scenario_simulator import ScenarioSimulator

    # Construct minimal sim (don't load models).
    sim = ScenarioSimulator.__new__(ScenarioSimulator)
    with pytest.raises(RuntimeError, match="scenario_mode='bayesian'.*removed"):
        sim.run_bayesian_scenarios(data=None, verbose=False)
