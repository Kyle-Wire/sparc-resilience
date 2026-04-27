"""Tests for the v4 unified scenario engine (Phase 5b-e + 5g)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sparc.interventions.scenario_engine_v4 import (
    ScenarioEngineV4,
    MissingArtifactsError,
)
from sparc.registry.run_registry import RunRegistry, set_active_registry
from sparc.registry.store import ArtifactStore


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _populate_store(store, *, n_cells=30, n_draws=120, treatments=("Pct_Canopy",)):
    rng = np.random.default_rng(0)
    alpha = rng.uniform(0.1, 0.5, size=(n_cells, len(treatments)))
    coords = rng.uniform(0, 100, size=(n_cells, 2))
    store.write_blob(
        "2", "v2_alpha_field", alpha,
        serializer="pickle", producer="v2_neural_training",
        metadata={
            "schema_version": 2,
            "treatments": list(treatments),
            "n_treatments": len(treatments),
            "shape": list(alpha.shape),
        },
    )
    store.write_blob(
        "2", "v2_alpha_field_coords", coords,
        serializer="pickle", producer="v2_neural_training",
    )

    edge_samples = {(treatments[0], "LST"): rng.normal(-0.05, 0.005, size=n_draws),
                    (treatments[0], "NDVI"): rng.normal(0.01, 0.001, size=n_draws),
                    ("NDVI", "LST"):       rng.normal(-0.03, 0.003, size=n_draws)}
    store.write_blob(
        "3", "nuts_edge_samples", edge_samples,
        serializer="pickle", producer="v2_bayesian_causal",
    )
    store.write_table(
        "3", "nuts_edge_summary",
        pd.DataFrame([
            {"parent": treatments[0], "child": "LST",  "mean": -0.05, "std": 0.005},
            {"parent": treatments[0], "child": "NDVI", "mean":  0.01, "std": 0.001},
            {"parent": "NDVI",        "child": "LST",  "mean": -0.03, "std": 0.003},
        ]),
        producer="v2_bayesian_causal",
    )

    nodes = list(treatments) + ["NDVI", "LST"]
    mat = np.full((len(nodes), len(nodes)), 0.05)
    np.fill_diagonal(mat, 0.0)
    # Strong direct edges to LST.
    for src in (*treatments, "NDVI"):
        mat[nodes.index(src), nodes.index("LST")] = 0.95
    # Treatment → NDVI (used for indirect path).
    mat[nodes.index(treatments[0]), nodes.index("NDVI")] = 0.85
    wide = pd.DataFrame(mat, index=nodes, columns=nodes).reset_index(names="source")
    store.write_table("3", "edge_inclusion_probs", wide, producer="v2_bayesian_causal")

    cate = {t: rng.normal(loc=1.0, scale=0.05, size=(n_draws, n_cells)) for t in treatments}
    store.write_blob("3", "cate_samples", cate, serializer="pickle", producer="causal_validation")
    store.write_table(
        "3", "cate_summary",
        pd.DataFrame([
            {"cell_id": i, "treatment": t,
             "cate_mean": float(cate[t][:, i].mean()),
             "cate_std":  float(cate[t][:, i].std()),
             "cate_ci5":  float(np.percentile(cate[t][:, i], 5)),
             "cate_ci50": float(np.percentile(cate[t][:, i], 50)),
             "cate_ci95": float(np.percentile(cate[t][:, i], 95))}
            for t in treatments for i in range(n_cells)
        ]),
        producer="causal_validation",
    )
    return alpha, coords, edge_samples


def _baseline_df(n_cells=30):
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "x": rng.uniform(0, 100, n_cells),
        "y": rng.uniform(0, 100, n_cells),
        "Pct_Canopy": rng.uniform(5, 60, n_cells),
        "NDVI": rng.uniform(0.1, 0.7, n_cells),
        "LST": rng.uniform(28.0, 38.0, n_cells),
    })


def _config(stage_dir: Path) -> dict:
    return {
        "paths": {"raw_csv_path": str(stage_dir / "x.csv")},
        "output": {
            "base_dir": str(stage_dir),
            "stage_dirs": {"stage_4": "Stage_4_Scenarios"},
        },
        "variables": {"target": "LST", "coordinates": ["x", "y"]},
        "crs": {"target_projected": "EPSG:4326"},
        "scenarios": [
            {"name": "Canopy Increase", "variable": "Pct_Canopy",
             "direction": "increase", "increments": [10, 20],
             "min_val": 0, "max_val": 100},
        ],
        "physics": {"literature_weight": 0.3},
        "caps": {"ensemble_posterior_blend": 0.5,
                 "mc3_path_threshold": 0.30,
                 "n_draws_cap": 200},
    }


@pytest.fixture()
def stage4_setup(tmp_path):
    reg = RunRegistry(tmp_path / "registry")
    set_active_registry(reg)
    store = ArtifactStore(reg)
    _populate_store(store)
    cfg = _config(tmp_path)
    yield cfg, store
    set_active_registry(None)


# ---------------------------------------------------------------------------
# mode_1_physics
# ---------------------------------------------------------------------------


def test_mode_1_physics_runs_and_persists(stage4_setup):
    cfg, store = stage4_setup
    engine = ScenarioEngineV4(cfg, mode="mode_1_physics")
    summary, results_long = engine.run(_baseline_df(), n_draws=80, verbose=False)

    assert "mode" in results_long.columns
    assert (results_long["mode"] == "mode_1_physics").all()
    assert {"delta_mean", "delta_ci5", "delta_ci50", "delta_ci95"}.issubset(results_long.columns)
    # Persistence
    assert store.has("4", "scenario_results")
    assert store.has("4", "scenario_summary")
    # Two increments × 30 cells = 60 rows.
    assert len(results_long) == 60


def test_mode_1_physics_works_without_alpha(tmp_path):
    """Resolver gracefully degrades when α is absent (mode_1 only)."""
    reg = RunRegistry(tmp_path / "reg")
    set_active_registry(reg)
    cfg = _config(tmp_path)
    try:
        # No artifacts populated.
        engine = ScenarioEngineV4(cfg, mode="mode_1_physics")
        summary, long_df = engine.run(_baseline_df(), n_draws=20, verbose=False)
        assert len(long_df) > 0
    finally:
        set_active_registry(None)


# ---------------------------------------------------------------------------
# mode_2_dag_local
# ---------------------------------------------------------------------------


def _build_dag():
    import networkx as nx
    g = nx.DiGraph()
    g.add_node("Pct_Canopy", node_type="treatment")
    g.add_node("NDVI", node_type="mediator")
    g.add_node("LST", node_type="outcome")
    g.add_edge("Pct_Canopy", "LST")
    g.add_edge("Pct_Canopy", "NDVI")
    g.add_edge("NDVI", "LST")
    return g


def test_mode_2_dag_local_includes_indirect(stage4_setup):
    cfg, _ = stage4_setup
    dag = _build_dag()

    engine_direct = ScenarioEngineV4(cfg, mode="mode_1_physics")
    s1, long1 = engine_direct.run(_baseline_df(), n_draws=60, verbose=False)

    engine_dag = ScenarioEngineV4(cfg, mode="mode_2_dag_local", dag=dag)
    s2, long2 = engine_dag.run(_baseline_df(), n_draws=60, verbose=False)

    # Same shape, but DAG path adds indirect contribution → not identical.
    assert long1.shape == long2.shape
    diff = (long1["delta_mean"].to_numpy() - long2["delta_mean"].to_numpy())
    assert np.abs(diff).mean() > 1e-6  # indirect path moved the values
    assert (long2["mode"] == "mode_2_dag_local").all()


def test_mode_2_requires_dag_and_nuts(stage4_setup):
    cfg, _ = stage4_setup
    with pytest.raises(MissingArtifactsError, match="requires a DAG"):
        ScenarioEngineV4(cfg, mode="mode_2_dag_local")


# ---------------------------------------------------------------------------
# mode_3_full_ensemble
# ---------------------------------------------------------------------------


def test_mode_3_blends_with_ensemble(stage4_setup):
    cfg, _ = stage4_setup
    base = _baseline_df()

    # Ensemble predictor: simple linear surrogate.
    def ens_pred(df):
        return -0.1 * df["Pct_Canopy"].to_numpy() + 35.0

    engine = ScenarioEngineV4(
        cfg, mode="mode_3_full_ensemble",
        ensemble_predictor=ens_pred,
    )
    s, long_df = engine.run(base, n_draws=40, verbose=False)
    assert (long_df["mode"] == "mode_3_full_ensemble").all()
    assert long_df["delta_mean"].abs().mean() > 0


def test_mode_3_requires_ensemble_predictor(stage4_setup):
    cfg, _ = stage4_setup
    with pytest.raises(MissingArtifactsError, match="ensemble_predictor"):
        ScenarioEngineV4(cfg, mode="mode_3_full_ensemble")


# ---------------------------------------------------------------------------
# mode_4_hybrid
# ---------------------------------------------------------------------------


def test_mode_4_hybrid_combines_ensemble_direct_and_dag_indirect(stage4_setup):
    cfg, _ = stage4_setup
    dag = _build_dag()
    def ens_pred(df):
        return -0.1 * df["Pct_Canopy"].to_numpy() + 35.0

    engine = ScenarioEngineV4(
        cfg, mode="mode_4_hybrid",
        dag=dag, ensemble_predictor=ens_pred,
    )
    s, long_df = engine.run(_baseline_df(), n_draws=40, verbose=False)
    assert (long_df["mode"] == "mode_4_hybrid").all()
    # Sanity: CIs ordered.
    assert (long_df["delta_ci5"] <= long_df["delta_ci50"]).all()
    assert (long_df["delta_ci50"] <= long_df["delta_ci95"]).all()


# ---------------------------------------------------------------------------
# Unified writer schema
# ---------------------------------------------------------------------------


def test_persisted_long_table_has_required_columns(stage4_setup):
    cfg, store = stage4_setup
    engine = ScenarioEngineV4(cfg, mode="mode_1_physics")
    engine.run(_baseline_df(), n_draws=30, verbose=False)
    df = store.read_table("4", "scenario_results")
    expected = {"cell_id", "scenario_name", "variable", "increment", "mode",
                "delta_mean", "delta_ci5", "delta_ci50", "delta_ci95",
                "baseline", "modified"}
    assert expected.issubset(df.columns)


def test_invalid_mode_rejected(stage4_setup):
    cfg, _ = stage4_setup
    with pytest.raises(ValueError, match="Unknown mode"):
        ScenarioEngineV4(cfg, mode="bogus_mode")
