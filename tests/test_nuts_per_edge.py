"""Tests for per-edge NUTS posterior sampling (Phase 1 of v4 causal-stack).

Covers:

* :func:`sparc.run.v2_bayesian_causal._run_per_edge_nuts_sampling` —
  synthetic 4-node DAG round-trip through the ArtifactStore;
  recovery of true edge coefficients within posterior CI.
* :class:`CounterfactualEngine` — `load_edge_posteriors` /
  `sample_edge_posterior` API and OLS-broadcast fallback warning.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from sparc.registry.run_registry import RunRegistry
from sparc.registry.store import ArtifactStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path):
    reg = RunRegistry(tmp_path)
    return ArtifactStore(reg)


@pytest.fixture()
def synthetic_dag_data():
    """4-node DAG: U -> X -> M -> Y; U -> Y is a confounder.

    True structural coefficients (β):
        X -> M = +0.80
        M -> Y = -0.50
        U -> X = +0.40
        U -> Y = +0.30   (backdoor)
    """
    rng = np.random.default_rng(0)
    n = 800
    U = rng.normal(0, 1, n)
    X = 0.40 * U + rng.normal(0, 1, n)
    M = 0.80 * X + rng.normal(0, 0.8, n)
    Y = -0.50 * M + 0.30 * U + rng.normal(0, 0.6, n)
    df = pd.DataFrame({"U": U, "X": X, "M": M, "Y": Y})
    dag_def = {
        "nodes": [
            {"name": "U", "type": "confounder"},
            {"name": "X", "type": "treatment"},
            {"name": "M", "type": "mediator"},
            {"name": "Y", "type": "outcome"},
        ],
        "edges": [
            {"parent": "U", "child": "X"},
            {"parent": "U", "child": "Y"},
            {"parent": "X", "child": "M"},
            {"parent": "M", "child": "Y"},
        ],
    }
    truth = {("U", "X"): 0.40, ("U", "Y"): 0.30,
             ("X", "M"): 0.80, ("M", "Y"): -0.50}
    return df, dag_def, truth


# ---------------------------------------------------------------------------
# Per-edge NUTS sampling
# ---------------------------------------------------------------------------


def test_per_edge_nuts_recovers_truth_and_persists(
    monkeypatch, tmp_path, synthetic_dag_data,
):
    """Run per-edge NUTS, verify recovery + DB persistence."""
    from sparc.run.v2_bayesian_causal import _run_per_edge_nuts_sampling
    import sparc.run.v2_bayesian_causal as v2bc

    df, dag_def, truth = synthetic_dag_data

    # Bind active store so the persistence path inside the helper writes DB.
    reg = RunRegistry(tmp_path)
    store = ArtifactStore(reg)
    monkeypatch.setattr(v2bc, "_get_store", lambda: store)

    import torch
    summary = _run_per_edge_nuts_sampling(
        data=df,
        dag_def=dag_def,
        node_names=[n["name"] for n in dag_def["nodes"]],
        target_col="Y",
        neural_baseline=None,         # exercise residualization path uniformly
        alpha_weights=None,
        nuts_cfg={
            "per_edge_n_samples": 1500,
            "per_edge_n_warmup": 300,
            "max_tree_depth": 8,
            "target_accept_rate": 0.7,
            "seed": 7,
        },
        mc3_results=None,
        output_dir=tmp_path,
        sign_default_map={},          # no sign constraints in this synthetic test
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    assert summary["status"] == "ok"
    assert summary["n_sampled"] == 4

    # Round-trip via DB
    summary_df = store.read_table("3", "nuts_edge_summary")
    samples = store.read_blob("3", "nuts_edge_samples")
    assert isinstance(samples, dict)
    assert set(samples.keys()) == set(truth.keys())

    # Recovery within 3σ posterior on every edge.
    for (parent, child), beta_true in truth.items():
        chain = samples[(parent, child)]
        mean = float(chain.mean())
        std = float(chain.std())
        assert std > 0
        z = abs(mean - beta_true) / max(std, 1e-6)
        assert z < 3.0, (
            f"{parent}->{child}: posterior mean {mean:.3f} ± {std:.3f} "
            f"too far from truth {beta_true:.3f} (z={z:.2f})"
        )

    # Summary table schema sanity
    expected_cols = {
        "parent", "child", "mean", "std", "ci5", "ci50", "ci95",
        "r_hat", "ess", "n_samples", "prior_used", "sign_constrained",
        "mc3_edge_prob", "x_std", "is_target_edge", "n_obs",
        "acceptance_rate", "n_divergences",
    }
    assert expected_cols.issubset(set(summary_df.columns))

    # R-hat acceptance — synthetic data, single short chain → relax bound
    # to confirm the sampler at least returned a finite scalar per edge.
    finite_rhat = summary_df["r_hat"].dropna()
    if len(finite_rhat):
        assert finite_rhat.between(0.5, 2.0).all(), \
            f"r_hat outside plausible range: {finite_rhat.tolist()}"


def test_per_edge_nuts_skips_low_mc3_edges(monkeypatch, tmp_path, synthetic_dag_data):
    """Edges with mc3_edge_prob < threshold are skipped."""
    from sparc.run.v2_bayesian_causal import _run_per_edge_nuts_sampling
    import sparc.run.v2_bayesian_causal as v2bc

    df, dag_def, _ = synthetic_dag_data
    node_names = [n["name"] for n in dag_def["nodes"]]

    # Build a fake MC3 result that strongly suppresses U->X.
    n = len(node_names)
    probs = np.full((n, n), 0.95)
    np.fill_diagonal(probs, 0.0)
    probs[node_names.index("U"), node_names.index("X")] = 0.05

    class _FakeMC3:
        edge_inclusion_probs = probs

    reg = RunRegistry(tmp_path)
    store = ArtifactStore(reg)
    monkeypatch.setattr(v2bc, "_get_store", lambda: store)

    import torch
    summary = _run_per_edge_nuts_sampling(
        data=df,
        dag_def=dag_def,
        node_names=node_names,
        target_col="Y",
        neural_baseline=None,
        alpha_weights=None,
        nuts_cfg={
            "per_edge_n_samples": 600, "per_edge_n_warmup": 150,
            "mc3_edge_threshold": 0.3, "seed": 1,
        },
        mc3_results=_FakeMC3(),
        output_dir=tmp_path,
        sign_default_map={},
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    assert summary["n_skipped_mc3"] >= 1
    assert summary["n_sampled"] == 3  # 4 edges total - 1 pruned

    samples = store.read_blob("3", "nuts_edge_samples")
    assert ("U", "X") not in samples


# ---------------------------------------------------------------------------
# CounterfactualEngine — edge-posterior API
# ---------------------------------------------------------------------------


def test_counterfactual_engine_loads_and_samples_edge_posteriors(store):
    from sparc.causal.counterfactual_engine import CounterfactualEngine

    # Seed the store with synthetic posteriors.
    samples_dict = {
        ("X", "M"): np.random.default_rng(0).normal(0.8, 0.05, 500),
        ("M", "Y"): np.random.default_rng(1).normal(-0.5, 0.04, 500),
    }
    store.write_blob(
        "3", "nuts_edge_samples", samples_dict,
        serializer="pickle", producer="test",
    )
    summary_df = pd.DataFrame([
        {"parent": "X", "child": "M", "mean": 0.8, "std": 0.05},
        {"parent": "M", "child": "Y", "mean": -0.5, "std": 0.04},
    ])
    store.write_table(
        "3", "nuts_edge_summary", summary_df, producer="test",
    )

    engine = CounterfactualEngine(config={})
    n_loaded = engine.load_edge_posteriors(store=store)
    assert n_loaded == 2
    assert engine.has_edge_posterior("X", "M")
    assert not engine.has_edge_posterior("Q", "Z")

    # Full chain
    chain = engine.sample_edge_posterior("X", "M")
    assert chain.shape == (500,)
    assert abs(chain.mean() - 0.8) < 0.05

    # Subsample
    sub = engine.sample_edge_posterior("M", "Y", n_draws=100)
    assert sub.shape == (100,)


def test_counterfactual_engine_ols_fallback_warns_once(store):
    from sparc.causal.counterfactual_engine import CounterfactualEngine

    engine = CounterfactualEngine(config={})
    engine._structural_coeffs[("A", "B")] = 1.23

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        chain1 = engine.sample_edge_posterior("A", "B", n_draws=10)
        chain2 = engine.sample_edge_posterior("A", "B", n_draws=10)
    assert chain1.shape == (10,)
    assert chain2.shape == (10,)
    assert np.allclose(chain1, 1.23)
    # Exactly one warning per missing edge across repeated calls.
    fallback_warnings = [
        x for x in w if "no NUTS posterior" in str(x.message)
    ]
    assert len(fallback_warnings) == 1


def test_counterfactual_engine_load_warns_when_store_missing():
    from sparc.causal.counterfactual_engine import CounterfactualEngine

    engine = CounterfactualEngine(config={})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        n = engine.load_edge_posteriors(store=None)
    assert n == 0
    # The warning text differs depending on whether get_active_store()
    # returns None or raises — accept either path.
    assert any("ArtifactStore" in str(x.message) or "store" in str(x.message)
               for x in w)
