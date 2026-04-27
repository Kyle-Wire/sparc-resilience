"""Tests for SPARC v4.1 robustness gating.

Covers:
- Item 1: E-value columns persisted on the scenario_results long table.
- Item 2: Continuous MC³ weighting (no hard prune; low-prob paths damp).
- Item 3: Counterfactual CI coverage harness.
- Item 4: ``ensemble_posterior_blend: auto`` learns λ per scenario.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sparc.evaluation.coverage import counterfactual_coverage
from sparc.interventions.scenario_engine_v4 import ScenarioEngineV4
from sparc.registry.run_registry import RunRegistry, set_active_registry
from sparc.registry.store import ArtifactStore

# Reuse the shared fixture helpers from the v4 engine test module.
from tests.test_scenario_engine_v4 import (  # type: ignore[import-not-found]
    _baseline_df, _build_dag, _config, _populate_store,
)


@pytest.fixture()
def stage4(tmp_path):
    reg = RunRegistry(tmp_path / "registry")
    set_active_registry(reg)
    store = ArtifactStore(reg)
    _populate_store(store)
    cfg = _config(tmp_path)
    yield cfg, store
    set_active_registry(None)


# ---------------------------------------------------------------------------
# Item 1: E-value gating on persisted scenario rows
# ---------------------------------------------------------------------------


def test_scenario_long_table_has_e_value_columns(stage4):
    cfg, store = stage4
    cfg["caps"]["e_value_min_threshold"] = 1.5
    engine = ScenarioEngineV4(cfg, mode="mode_1_physics")
    engine.run(_baseline_df(), n_draws=60, verbose=False)
    df = store.read_table("4", "scenario_results")
    for col in ("e_value_point", "e_value_ci", "e_value_pass"):
        assert col in df.columns, f"missing {col}"
    # E-values should be non-negative real numbers (or NaN).
    finite = df["e_value_point"].dropna()
    assert (finite >= 0).all()
    # ``e_value_pass`` round-trips through sqlite as 0/1; coerce and
    # assert all entries are valid booleans.
    passes = df["e_value_pass"].astype(bool)
    assert set(passes.unique()).issubset({True, False})


def test_scenario_metadata_carries_v41_diagnostics(stage4):
    cfg, store = stage4
    engine = ScenarioEngineV4(cfg, mode="mode_1_physics")
    engine.run(_baseline_df(), n_draws=40, verbose=False)
    entry = store.registry.lookup("4", "scenario_results")
    assert entry is not None
    md = entry.metadata or {}
    assert md.get("schema_version") == 3
    assert "ensemble_blend_used" in md
    assert "e_value_min" in md
    assert "mc3_path_threshold" in md


# ---------------------------------------------------------------------------
# Item 2: Continuous MC³ weighting (no hard prune)
# ---------------------------------------------------------------------------


def test_dag_indirect_no_hard_prune_low_prob_path_still_contributes(stage4):
    """Setting mc3_path_threshold above the path's min(prob) used to drop
    the contribution to zero. Now it should only damp via ∏ p_inclusion.
    """
    cfg, _ = stage4
    dag = _build_dag()

    # Threshold above the Canopy→NDVI edge prob (0.85) AND the
    # NDVI→LST edge prob (0.95). Min of the path is 0.85, so a hard
    # prune at 0.99 would zero the indirect contribution; continuous
    # weighting should still produce nonzero deltas.
    cfg["caps"]["mc3_path_threshold"] = 0.99

    base = _baseline_df()
    e_direct = ScenarioEngineV4(cfg, mode="mode_1_physics")
    _, long_direct = e_direct.run(base, n_draws=60, verbose=False)

    e_dag = ScenarioEngineV4(cfg, mode="mode_2_dag_local", dag=dag)
    _, long_dag = e_dag.run(base, n_draws=60, verbose=False)

    diff = (long_dag["delta_mean"].to_numpy()
            - long_direct["delta_mean"].to_numpy())
    # Continuous weighting should leave a measurable contribution.
    assert np.abs(diff).mean() > 1e-7, (
        "indirect path was hard-pruned; continuous MC³ weighting failed"
    )


# ---------------------------------------------------------------------------
# Item 4: Auto λ stacking
# ---------------------------------------------------------------------------


def test_ensemble_blend_auto_learns_weight(stage4):
    cfg, _ = stage4
    cfg["caps"]["ensemble_posterior_blend"] = "auto"

    def ens_pred(df):
        # Strong (overconfident) linear surrogate.
        return -0.4 * df["Pct_Canopy"].to_numpy() + 35.0

    engine = ScenarioEngineV4(
        cfg, mode="mode_3_full_ensemble", ensemble_predictor=ens_pred,
    )
    assert engine.ensemble_blend_auto is True
    _, long_df = engine.run(_baseline_df(), n_draws=40, verbose=False)
    # After at least one scenario, the surfaced λ must be a valid weight.
    assert 0.0 <= engine.ensemble_blend <= 1.0


def test_ensemble_blend_explicit_float_overrides_auto(stage4):
    cfg, _ = stage4
    cfg["caps"]["ensemble_posterior_blend"] = 0.7

    def ens_pred(df):
        return -0.1 * df["Pct_Canopy"].to_numpy() + 35.0

    engine = ScenarioEngineV4(
        cfg, mode="mode_3_full_ensemble", ensemble_predictor=ens_pred,
    )
    assert engine.ensemble_blend_auto is False
    assert engine.ensemble_blend == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Item 3: Counterfactual CI coverage harness
# ---------------------------------------------------------------------------


def test_counterfactual_coverage_returns_levels(stage4):
    cfg, _ = stage4

    def ens_pred(df):
        return -0.1 * df["Pct_Canopy"].to_numpy() + 35.0

    engine = ScenarioEngineV4(
        cfg, mode="mode_3_full_ensemble", ensemble_predictor=ens_pred,
    )
    report = counterfactual_coverage(
        engine, _baseline_df(),
        ensemble_predictor=ens_pred,
        holdout_frac=0.3, seed=7, n_draws=40,
    )
    for lvl in (50, 80, 90, 95):
        assert f"nominal_{lvl}" in report
        assert f"empirical_{lvl}" in report
    assert report["n_holdout"] >= 1
    assert report["n_scenarios"] >= 1
    # Empirical coverage at any level must be in [0, 1].
    for lvl in (50, 80, 90, 95):
        emp = report[f"empirical_{lvl}"]
        if emp == emp:  # not NaN
            assert 0.0 <= emp <= 1.0


def test_counterfactual_coverage_skips_when_no_predictor(stage4):
    cfg, _ = stage4
    engine = ScenarioEngineV4(cfg, mode="mode_1_physics")
    report = counterfactual_coverage(
        engine, _baseline_df(),
        ensemble_predictor=None, holdout_frac=0.2, seed=0,
    )
    assert report.get("skipped_reason") == "no ensemble_predictor available"
