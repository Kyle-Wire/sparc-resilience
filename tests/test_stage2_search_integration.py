"""Tests for sparc.run.inner_loops.stage2_search — Stage 2 wiring."""

from __future__ import annotations

import numpy as np
import pytest

from sparc.run.decisions import StageDecision
from sparc.run.inner_loops.stage2_search import run_stage2_bandwidth_searches


@pytest.fixture
def synthetic_xy():
    rng = np.random.default_rng(11)
    n = 150
    coords = rng.uniform(0, 10_000, size=(n, 2))
    cx, cy = 5000.0, 5000.0
    y = np.exp(-((coords[:, 0] - cx) ** 2 + (coords[:, 1] - cy) ** 2) / (2 * 1500.0 ** 2))
    y += rng.normal(0, 0.05, size=n)
    X = rng.normal(size=(n, 3))
    return X, y, coords


def _base_config_gwr_only():
    return {
        "models": {"gwr": {"bandwidth": 1000.0}},
        "pipeline": {
            "use_surrogate_search": True,
            "surrogate_search": {
                "n_trials": 6,
                "n_verify": 2,
                "improvement_threshold_pct": 0.5,
                "bo_random_seed": 0,
                "composite_lambdas": {"phys": None, "mono": None, "spatial": None},
            },
        },
    }


def test_search_skipped_when_no_correlogram_bandwidths(synthetic_xy):
    X, y, coords = synthetic_xy
    cfg = _base_config_gwr_only()
    decision = run_stage2_bandwidth_searches(
        X=X, y=y, coords=coords, base_config=cfg, correlogram_bandwidths=None,
    )
    assert isinstance(decision, StageDecision)
    assert decision.stage == 2
    assert decision.provenance.source == "correlogram"
    assert any(d.rule == "surrogate_search_skipped" for d in decision.decisions)
    # Original config was not mutated.
    assert cfg["models"]["gwr"]["bandwidth"] == 1000.0


def test_search_skipped_when_no_eligible_models(synthetic_xy):
    X, y, coords = synthetic_xy
    cfg = {"models": {}, "pipeline": {"use_surrogate_search": True}}
    decision = run_stage2_bandwidth_searches(
        X=X, y=y, coords=coords, base_config=cfg,
        correlogram_bandwidths={"a": 1000.0, "b": 800.0},
    )
    assert decision.stage == 2
    assert decision.provenance.source == "correlogram"
    assert any(
        d.evidence.get("reason") == "no_eligible_models" for d in decision.decisions
    )


def test_search_runs_end_to_end_for_gwr(synthetic_xy):
    X, y, coords = synthetic_xy
    cfg = _base_config_gwr_only()

    decision = run_stage2_bandwidth_searches(
        X=X, y=y, coords=coords, base_config=cfg,
        correlogram_bandwidths={"f1": 800.0, "f2": 1200.0, "f3": 1500.0},
    )

    assert decision.stage == 2
    # GWR was searched → at least one DecisionRecord with rule="improvement_gate".
    gate_decisions = [d for d in decision.decisions if d.rule == "improvement_gate"]
    assert len(gate_decisions) == 1
    assert gate_decisions[0].evidence["model"] == "gwr"

    # Per-model metrics emitted.
    assert "gwr_chosen_score" in decision.metrics
    assert "gwr_baseline_score" in decision.metrics
    assert "gwr_delta_pct" in decision.metrics

    # Either the search promoted or it fell back. Either way, base_config now
    # has a numeric GWR bandwidth.
    bw = cfg["models"]["gwr"]["bandwidth"]
    assert isinstance(bw, float)
    assert bw > 0

    # Decision is JSON-serializable (artifact-store contract).
    payload = decision.to_artifact_struct()
    import json
    json.dumps(payload)


def test_search_handles_gwrf_with_k_neighbors(synthetic_xy):
    X, y, coords = synthetic_xy
    cfg = {
        "models": {
            "gwr": {"bandwidth": 1000.0},
            "gwrf": {"bandwidth": 1000.0, "k_neighbors": 16},
        },
        "pipeline": {
            "use_surrogate_search": True,
            "surrogate_search": {
                "n_trials": 6,
                "n_verify": 2,
                "improvement_threshold_pct": 0.5,
                "bo_random_seed": 0,
            },
        },
    }
    decision = run_stage2_bandwidth_searches(
        X=X, y=y, coords=coords, base_config=cfg,
        correlogram_bandwidths={"f1": 800.0, "f2": 1200.0},
    )

    # Both models searched.
    gate_decisions = [d for d in decision.decisions if d.rule == "improvement_gate"]
    models_seen = {d.evidence["model"] for d in gate_decisions}
    assert models_seen == {"gwr", "gwrf"}

    # GWRF k_neighbors was applied to config and is an int.
    assert isinstance(cfg["models"]["gwrf"]["k_neighbors"], int)
