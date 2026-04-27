"""Tests for CausalEffectResolver — Phase 4 of v4 causal-stack rewrite."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from sparc.interventions.causal_stack import (
    CausalEffectResolver,
    ResolverConfig,
)
from sparc.registry.run_registry import RunRegistry
from sparc.registry.store import ArtifactStore


# ---------------------------------------------------------------------------
# Fixture: synthetic store populated with α / NUTS / CATE / MC³ artifacts
# ---------------------------------------------------------------------------


def _populate_store(store, *, n_cells=40, n_draws=200, treatments=("T1", "T2")):
    rng = np.random.default_rng(0)

    # α field — (N, n_treatments) under schema_version=2.
    alpha = rng.uniform(0.05, 0.5, size=(n_cells, len(treatments)))
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

    # NUTS edge samples & summary.
    edge_samples = {}
    summary_rows = []
    for t in treatments:
        s = rng.normal(0.5, 0.05, size=n_draws)
        edge_samples[(t, "Y")] = s
        summary_rows.append({
            "parent": t, "child": "Y",
            "mean": float(s.mean()), "std": float(s.std()),
        })
    edge_samples[("M", "Y")] = rng.normal(0.2, 0.03, size=n_draws)
    summary_rows.append({"parent": "M", "child": "Y", "mean": 0.2, "std": 0.03})

    store.write_blob(
        "3", "nuts_edge_samples", edge_samples,
        serializer="pickle", producer="v2_bayesian_causal",
    )
    store.write_table(
        "3", "nuts_edge_summary", pd.DataFrame(summary_rows),
        producer="v2_bayesian_causal",
    )

    # MC³ inclusion probs (wide format).
    nodes = list(treatments) + ["M", "Y"]
    mat = np.zeros((len(nodes), len(nodes)))
    for i, src in enumerate(nodes):
        for j, tgt in enumerate(nodes):
            if i == j:
                continue
            if (src in treatments and tgt == "Y") or (src == "M" and tgt == "Y"):
                mat[i, j] = 0.92
            else:
                mat[i, j] = 0.05
    wide = pd.DataFrame(mat, index=nodes, columns=nodes).reset_index(names="source")
    store.write_table(
        "3", "edge_inclusion_probs", wide,
        producer="v2_bayesian_causal",
    )

    # CATE samples & summary.
    cate_samples = {}
    cate_rows = []
    for t in treatments:
        # Heterogeneous τ that depends on coord X.
        base = 0.3 + 0.4 * (coords[:, 0] / 100.0)
        tau = rng.normal(loc=base, scale=0.05, size=(n_draws, n_cells))
        cate_samples[t] = tau
        for i in range(n_cells):
            cate_rows.append({
                "cell_id": i, "treatment": t,
                "cate_mean": float(tau[:, i].mean()),
                "cate_std": float(tau[:, i].std()),
                "cate_ci5": float(np.percentile(tau[:, i], 5)),
                "cate_ci50": float(np.percentile(tau[:, i], 50)),
                "cate_ci95": float(np.percentile(tau[:, i], 95)),
            })
    store.write_blob(
        "3", "cate_samples", cate_samples,
        serializer="pickle", producer="causal_validation",
    )
    store.write_table(
        "3", "cate_summary", pd.DataFrame(cate_rows),
        producer="causal_validation",
    )

    return {
        "alpha": alpha,
        "coords": coords,
        "edge_samples": edge_samples,
        "cate_samples": cate_samples,
        "treatments": list(treatments),
    }


@pytest.fixture()
def populated_store(tmp_path):
    reg = RunRegistry(tmp_path)
    store = ArtifactStore(reg)
    info = _populate_store(store)
    return store, info


# ---------------------------------------------------------------------------
# Construction / introspection
# ---------------------------------------------------------------------------


def test_resolver_loads_all_artifacts(populated_store):
    store, info = populated_store
    res = CausalEffectResolver(store)

    assert res.has_alpha
    assert set(res.available_treatments) >= set(info["treatments"])
    # Edge posteriors — both treatments + mediator edge present.
    assert res.has_posterior(("T1", "Y"))
    assert res.has_posterior(("T2", "Y"))
    assert res.has_posterior(("M", "Y"))
    # MC³ probs roundtrip
    assert res.mc3_edge_prob("T1", "Y") > 0.5
    assert res.mc3_edge_prob("M", "Y") > 0.5
    assert res.mc3_edge_prob("T1", "T2") < 0.5  # noise edge
    # No missing artifacts.
    assert res.missing_artifacts() == []


def test_resolver_missing_artifacts_warn_once(tmp_path):
    """Empty store → one warning per missing artifact, recorded in tally."""
    reg = RunRegistry(tmp_path)
    store = ArtifactStore(reg)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = CausalEffectResolver(store)
    labels = [str(x.message) for x in w]
    # All 6 expected artifacts should be reported missing.
    for art in [
        "v2_alpha_field", "nuts_edge_samples", "nuts_edge_summary",
        "edge_inclusion_probs", "cate_samples", "cate_summary",
    ]:
        assert any(art in s for s in labels), f"missing warning for {art}"
    assert len(res.missing_artifacts()) >= 6
    # Calling resolve methods on an empty resolver still works (fallback).
    delta = res.resolve_direct_effect(
        "T1", "Y",
        baseline_x=np.zeros(10), modified_x=np.ones(10),
        n_draws=50,
    )
    assert delta.shape == (50, 10)


# ---------------------------------------------------------------------------
# Edge resolution
# ---------------------------------------------------------------------------


def test_resolve_structural_edge_uses_posterior(populated_store):
    store, _ = populated_store
    res = CausalEffectResolver(store)
    samples = res.resolve_structural_edge("T1", "Y", n_draws=100)
    assert samples.shape == (100,)
    assert 0.4 < samples.mean() < 0.6  # synthetic mean ~0.5


def test_resolve_structural_edge_fallback(populated_store):
    """Unknown edge → constant broadcast + one-shot warning."""
    store, _ = populated_store
    res = CausalEffectResolver(store)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        samples = res.resolve_structural_edge("NOPE", "Y", n_draws=50)
    assert samples.shape == (50,)
    # Constant broadcast (variance ~0)
    assert samples.std() < 1e-9
    assert any("edge_posterior::NOPE->Y" in str(x.message) for x in w)


# ---------------------------------------------------------------------------
# CATE summary
# ---------------------------------------------------------------------------


def test_summarize_axis0():
    arr = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0]])
    s = CausalEffectResolver.summarize(arr)
    assert np.allclose(s["mean"], [3.0, 4.0, 5.0])
    assert s["ci5"].shape == (3,)
    assert s["ci95"].shape == (3,)


# ---------------------------------------------------------------------------
# Direct-effect composition
# ---------------------------------------------------------------------------


def test_resolve_direct_effect_shape_and_mean(populated_store):
    store, info = populated_store
    res = CausalEffectResolver(store, config=ResolverConfig(n_draws_cap=150, shrinkage_weight=0.0))

    n = info["alpha"].shape[0]
    baseline = np.zeros(n)
    modified = np.ones(n)
    delta = res.resolve_direct_effect("T1", "Y", baseline, modified, n_draws=150)

    assert delta.shape == (150, n)
    # Mean magnitude scales with β·α·multiplier·Δx.  All Δx = 1, multiplier ~1.
    expected = info["edge_samples"][("T1", "Y")].mean() * info["alpha"][:, 0].mean() / np.abs(info["alpha"][:, 0]).mean()
    # Per-cell means dominated by α and β posterior; sanity bound.
    cell_mean = delta.mean(axis=0)
    assert cell_mean.shape == (n,)
    assert cell_mean.mean() > 0  # positive direction
    assert np.isfinite(delta).all()


def test_resolve_direct_effect_draw_harmonization(populated_store):
    """n_eff ≤ min(n_nuts, n_cate, n_draws_cap)."""
    store, _ = populated_store
    res = CausalEffectResolver(store, config=ResolverConfig(n_draws_cap=80))
    delta = res.resolve_direct_effect(
        "T1", "Y",
        baseline_x=np.zeros(20), modified_x=np.ones(20),
        n_draws=10_000,
    )
    assert delta.shape[0] == 80


def test_resolve_direct_effect_per_treatment_alpha(populated_store):
    """Different treatment → different α slice → different delta magnitude."""
    store, info = populated_store
    res = CausalEffectResolver(store, config=ResolverConfig(shrinkage_weight=0.0))
    n = info["alpha"].shape[0]
    bx = np.zeros(n); mx = np.ones(n)
    d1 = res.resolve_direct_effect("T1", "Y", bx, mx, n_draws=100).mean(axis=0)
    d2 = res.resolve_direct_effect("T2", "Y", bx, mx, n_draws=100).mean(axis=0)
    # Different α columns → not identical.
    assert not np.allclose(d1, d2)


def test_summarize_round_trip(populated_store):
    store, info = populated_store
    res = CausalEffectResolver(store)
    n = info["alpha"].shape[0]
    delta = res.resolve_direct_effect(
        "T1", "Y", np.zeros(n), np.ones(n), n_draws=80,
    )
    s = res.summarize(delta)
    assert all(s[k].shape == (n,) for k in ("mean", "ci5", "ci50", "ci95"))
    assert (s["ci5"] <= s["ci50"]).all()
    assert (s["ci50"] <= s["ci95"]).all()


# ---------------------------------------------------------------------------
# Alpha alignment
# ---------------------------------------------------------------------------


def test_align_alpha_to_rows_kdtree(populated_store):
    store, info = populated_store
    res = CausalEffectResolver(store)
    # Pick a small subset of coords and assert alignment retrieves matching rows.
    target = info["coords"][:5] + 1e-3  # tiny jitter
    aligned = res.align_alpha_to_rows(target)
    assert aligned.shape == (5, info["alpha"].shape[1])
    assert np.allclose(aligned, info["alpha"][:5])
