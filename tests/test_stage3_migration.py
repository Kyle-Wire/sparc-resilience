"""Stage 3 (v2_bayesian_causal) artifacts.db migration tests.

Round-trip the structs/tables/blobs produced after the Phase C5
migration:
- mc3_summary struct
- edge_inclusion_probs table (wide matrix with ``source`` index)
- edge_confidence table (long form with confidence label)
- median_probability_dag struct
- edge_inclusion_probs_npy blob (numpy matrix)
- nuts_summary struct (acceptance, R-hat, ESS, treatments)
- parameter_posteriors table (per-treatment quantiles)
- convergence_diagnostics table
- bma_coefficients table
- nuts_beta blob (posterior chain)
- nuts_sigma2 blob (auxiliary chain)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sparc.registry.run_registry import RunRegistry, set_active_registry
from sparc.registry.store import ArtifactStore


@pytest.fixture()
def store(tmp_path):
    reg = RunRegistry(tmp_path)
    set_active_registry(reg)
    yield ArtifactStore(reg)
    set_active_registry(None)


def test_mc3_summary_struct(store):
    payload = {
        "n_accepted": 12_345,
        "n_total": 40_000,
        "acceptance_rate": 12_345 / 40_000,
        "best_score": -1234.5,
        "node_names": ["A", "B", "C"],
    }
    store.write_struct("3", "mc3_summary", payload, producer="test")
    out = store.read_any("3", "mc3_summary")
    assert out["n_accepted"] == 12_345
    assert out["node_names"] == ["A", "B", "C"]


def test_edge_inclusion_probs_table(store):
    nodes = ["A", "B", "C"]
    matrix = np.array([[0.0, 0.9, 0.1],
                       [0.05, 0.0, 0.7],
                       [0.0, 0.2, 0.0]])
    df = pd.DataFrame(matrix, index=nodes, columns=nodes).reset_index(names="source")
    store.write_table("3", "edge_inclusion_probs", df, producer="test")
    out = store.read_any("3", "edge_inclusion_probs")
    assert "source" in out.columns
    assert set(nodes).issubset(out.columns)
    assert float(out.loc[0, "B"]) == pytest.approx(0.9)


def test_edge_confidence_table(store):
    df = pd.DataFrame([
        {"source": "A", "target": "B", "inclusion_prob": 0.95, "confidence": "strong"},
        {"source": "B", "target": "C", "inclusion_prob": 0.72, "confidence": "moderate"},
        {"source": "A", "target": "C", "inclusion_prob": 0.10, "confidence": "absent"},
    ])
    store.write_table("3", "edge_confidence", df, producer="test")
    out = store.read_any("3", "edge_confidence")
    assert len(out) == 3
    assert "confidence" in out.columns


def test_median_probability_dag_struct(store):
    payload = {
        "nodes": ["A", "B", "C"],
        "edges": [
            {"source": "A", "target": "B", "probability": 0.91},
            {"source": "B", "target": "C", "probability": 0.72},
        ],
    }
    store.write_struct("3", "median_probability_dag", payload, producer="test")
    out = store.read_any("3", "median_probability_dag")
    assert len(out["edges"]) == 2
    assert out["edges"][0]["probability"] == pytest.approx(0.91)


def test_edge_inclusion_probs_blob(store):
    arr = np.random.default_rng(0).random((6, 6)).astype(np.float64)
    store.write_blob("3", "edge_inclusion_probs_npy", arr,
                     serializer="pickle", producer="test")
    out = store.read_any("3", "edge_inclusion_probs_npy")
    np.testing.assert_array_equal(out, arr)


def test_nuts_summary_struct(store):
    payload = {
        "acceptance_rate": 0.73,
        "n_divergences": 4,
        "treatments": ["Pct_Canopy", "Albedo"],
        "beta_mean": [-0.12, 0.04],
        "beta_std": [0.03, 0.02],
        "r_hat": {"beta": [1.005, 1.002], "sigma2": [1.001]},
        "ess": {"beta": [4321.0, 5012.0], "sigma2": [3300.0]},
        "alpha_weighted": False,
    }
    store.write_struct("3", "nuts_summary", payload, producer="test")
    out = store.read_any("3", "nuts_summary")
    assert out["treatments"] == ["Pct_Canopy", "Albedo"]
    assert out["r_hat"]["beta"][0] == pytest.approx(1.005)


def test_parameter_posteriors_table(store):
    df = pd.DataFrame({
        "treatment": ["Pct_Canopy", "Albedo"],
        "treatment_std": [12.3, 0.05],
        "mean": [-0.12, 0.04],
        "std": [0.03, 0.02],
        "ci_5": [-0.17, 0.01],
        "ci_25": [-0.14, 0.025],
        "median": [-0.12, 0.04],
        "ci_75": [-0.10, 0.055],
        "ci_95": [-0.07, 0.07],
        "mean_per_std": [-1.5, 0.8],
        "std_per_std": [0.4, 0.3],
        "ci_5_per_std": [-2.1, 0.2],
        "ci_95_per_std": [-0.9, 1.4],
    })
    store.write_table("3", "parameter_posteriors", df, producer="test")
    out = store.read_any("3", "parameter_posteriors")
    assert list(out["treatment"]) == ["Pct_Canopy", "Albedo"]


def test_convergence_diagnostics_table(store):
    df = pd.DataFrame([
        {"parameter": "beta[Pct_Canopy]", "r_hat": 1.005, "ess": 4321.0, "converged": True},
        {"parameter": "beta[Albedo]", "r_hat": 1.002, "ess": 5012.0, "converged": True},
        {"parameter": "sigma2[0]", "r_hat": 1.001, "ess": 3300.0, "converged": True},
    ])
    store.write_table("3", "convergence_diagnostics", df, producer="test")
    out = store.read_any("3", "convergence_diagnostics")
    assert out["converged"].all()


def test_bma_coefficients_table(store):
    df = pd.DataFrame([
        {
            "treatment": "Pct_Canopy",
            "treatment_std": 12.3,
            "posterior_mean": -0.12,
            "posterior_std": 0.03,
            "edge_inclusion_prob": 0.92,
            "bma_effect": -0.1104,
            "bma_ci_5": -0.156,
            "bma_ci_95": -0.064,
            "bma_effect_per_std": -1.38,
            "bma_ci_5_per_std": -1.93,
            "bma_ci_95_per_std": -0.83,
        },
    ])
    store.write_table("3", "bma_coefficients", df, producer="test")
    out = store.read_any("3", "bma_coefficients")
    assert "bma_effect" in out.columns


def test_nuts_beta_blob_roundtrip(store):
    # (n_samples=200, n_treatments=3)
    chain = np.random.default_rng(1).standard_normal((200, 3)).astype(np.float64)
    store.write_blob("3", "nuts_beta", chain, serializer="pickle", producer="test")
    out = store.read_any("3", "nuts_beta")
    np.testing.assert_array_equal(out, chain)


def test_nuts_aux_chain_blob_roundtrip(store):
    chain = np.abs(np.random.default_rng(2).standard_normal(200)).astype(np.float64)
    store.write_blob("3", "nuts_sigma2", chain, serializer="pickle", producer="test")
    out = store.read_any("3", "nuts_sigma2")
    np.testing.assert_array_equal(out, chain)
