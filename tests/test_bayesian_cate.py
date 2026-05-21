"""Tests for Bayesian Spatial CATE — Phase 2 of v4 causal-stack rewrite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sparc.causal.spatial_cate import (
    BayesianSpatialCATE,
    SpatialCATEEstimator,
    make_cate_estimator,
)
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
def synthetic_spatial_cate_data():
    """Synthetic spatial dataset with known heterogeneous treatment effect.

    True CATE varies smoothly with the X coordinate:
        τ(x, y) = -0.2 + 0.6 * x   (so τ is negative on left, positive on right)

    Y = τ(s) * T + 0.3 * Z + ε
    """
    rng = np.random.default_rng(0)
    n = 600
    coords_x = rng.uniform(0, 1, n)
    coords_y = rng.uniform(0, 1, n)
    Z = rng.normal(0, 1, n)
    T = 0.5 * Z + rng.normal(0, 1, n)
    tau_true = -0.2 + 0.6 * coords_x
    Y = tau_true * T + 0.3 * Z + rng.normal(0, 0.3, n)
    df = pd.DataFrame({
        "T": T, "Y": Y, "Z": Z,
        "POINT_X": coords_x, "POINT_Y": coords_y,
    })
    return df, tau_true


# ---------------------------------------------------------------------------
# Estimator behavior
# ---------------------------------------------------------------------------


def test_bayesian_cate_recovers_spatial_pattern(synthetic_spatial_cate_data):
    df, tau_true = synthetic_spatial_cate_data
    est = BayesianSpatialCATE(
        config={"pipeline": {"random_seed": 1}},
        n_features=24,
        n_samples=800,
        n_warmup=300,
    )
    cate = est.estimate(
        data=df,
        treatment="T",
        outcome="Y",
        confounders=["Z"],
        coord_cols=["POINT_X", "POINT_Y"],
    )
    assert cate.shape == (len(df),)

    # Posterior samples available
    tau_post = est.posterior_samples("T")
    assert tau_post.ndim == 2
    assert tau_post.shape[1] == len(df)

    # Spatial gradient: cells with X near 0 should have lower CATE than
    # cells with X near 1 (recovering the +0.6·x slope).
    left = df["POINT_X"] < 0.25
    right = df["POINT_X"] > 0.75
    assert cate[right].mean() > cate[left].mean(), (
        f"Failed to recover spatial CATE gradient: "
        f"left={cate[left].mean():.3f}, right={cate[right].mean():.3f}"
    )

    # CIs envelope the true value on average (≥ 60% coverage of 90% CI on
    # a noisy small sample is reasonable).
    lo, hi = est.cate_intervals["T"]
    coverage = float(np.mean((tau_true >= lo) & (tau_true <= hi)))
    assert coverage > 0.6, f"90% CI coverage too low: {coverage:.2f}"


def test_bayesian_cate_multiplier_posterior(synthetic_spatial_cate_data):
    df, _ = synthetic_spatial_cate_data
    est = BayesianSpatialCATE(
        config={"pipeline": {"random_seed": 0}},
        n_features=16, n_samples=400, n_warmup=150,
    )
    est.estimate(df, "T", "Y", ["Z"], coord_cols=["POINT_X", "POINT_Y"])

    mult_post = est.multiplier_posterior("T")
    assert mult_post.ndim == 2
    assert mult_post.shape[1] == len(df)
    assert mult_post.min() >= 0.5 - 1e-9
    assert mult_post.max() <= 1.5 + 1e-9


def test_make_cate_estimator_factory():
    bay = make_cate_estimator({"causal": {"cate_estimator": "bayesian"}})
    assert isinstance(bay, BayesianSpatialCATE)

    freq = make_cate_estimator({"causal": {"cate_estimator": "frequentist"}})
    assert isinstance(freq, SpatialCATEEstimator)

    auto = make_cate_estimator({"causal": {"cate_estimator": "auto"}})
    assert isinstance(auto, BayesianSpatialCATE)  # nuts module always present

    default = make_cate_estimator({})
    assert isinstance(default, BayesianSpatialCATE)

    with pytest.raises(ValueError):
        make_cate_estimator({"causal": {"cate_estimator": "garbage"}})


# ---------------------------------------------------------------------------
# Persistence: ("3","cate_summary") + ("3","cate_samples")
# ---------------------------------------------------------------------------


def test_persist_cate_artifacts_v2_bayesian(store, synthetic_spatial_cate_data):
    """Round-trip: BayesianSpatialCATE results land in DB with full posteriors."""
    from sparc.run.causal_validation import CausalValidator

    df, _ = synthetic_spatial_cate_data
    n = len(df)

    # Synthesize spatial_cate_results as if run_spatial_cate already ran.
    rng = np.random.default_rng(0)
    n_draws = 200
    tau_post = rng.normal(0.3, 0.05, size=(n_draws, n))
    cate_mean = tau_post.mean(axis=0)
    mult = np.clip(cate_mean / cate_mean.mean(), 0.5, 1.5)

    validator = CausalValidator.__new__(CausalValidator)
    validator.spatial_cate_results = {
        "T": {
            "cate": cate_mean,
            "cate_mean": float(cate_mean.mean()),
            "cate_std": float(cate_mean.std()),
            "multiplier": mult,
            "posterior_samples": tau_post,
            "diagnostics": {"acceptance_rate": 0.85},
        },
    }

    # Bind the active registry so get_active_store() resolves to our store.
    from sparc.registry.run_registry import set_active_registry
    set_active_registry(store.registry)
    try:
        validator._persist_cate_artifacts_v2(
            sources={"T": "bayesian_posterior_mean"},
            output_dir=str(store.registry.output_dir),
        )
    finally:
        set_active_registry(None)

    # Round-trip
    summary_df = store.read_table("3", "cate_summary")
    samples = store.read_blob("3", "cate_samples")

    assert set(summary_df["treatment"].unique()) == {"T"}
    assert summary_df.shape[0] == n
    expected_cols = {
        "cell_id", "treatment", "cate_mean", "cate_std",
        "cate_ci5", "cate_ci50", "cate_ci95",
        "multiplier_mean", "multiplier_ci5", "multiplier_ci95", "source",
    }
    assert expected_cols.issubset(set(summary_df.columns))

    # Posterior samples preserved
    assert "T" in samples
    assert samples["T"].shape == (n_draws, n)
    assert np.allclose(samples["T"], tau_post)

    # Source tag carried through
    assert (summary_df["source"] == "bayesian_posterior_mean").all()

    # Multiplier CIs differ from point estimate (heterogeneous posterior)
    assert summary_df["multiplier_ci5"].std() > 0
    assert summary_df["multiplier_ci95"].std() > 0


def test_persist_cate_artifacts_v2_frequentist(store):
    """Frequentist (single-draw) path: degenerate posterior is still persisted."""
    from sparc.run.causal_validation import CausalValidator

    n = 50
    cate = np.linspace(-0.2, 0.4, n)
    mult = np.clip(cate / cate.mean(), 0.5, 1.5)
    validator = CausalValidator.__new__(CausalValidator)
    validator.spatial_cate_results = {
        "X": {
            "cate": cate, "cate_mean": float(cate.mean()),
            "cate_std": float(cate.std()), "multiplier": mult,
            # No posterior_samples
        },
    }

    from sparc.registry.run_registry import set_active_registry
    set_active_registry(store.registry)
    try:
        validator._persist_cate_artifacts_v2(
            sources={"X": "frequentist_dml"},
            output_dir=str(store.registry.output_dir),
        )
    finally:
        set_active_registry(None)

    summary = store.read_table("3", "cate_summary")
    samples = store.read_blob("3", "cate_samples")
    assert summary.shape[0] == n
    assert samples["X"].shape == (1, n)
    # Frequentist CIs collapse to the point estimate.
    assert np.allclose(summary["cate_mean"].values, summary["cate_ci5"].values)
    assert np.allclose(summary["cate_mean"].values, summary["cate_ci95"].values)


# ---------------------------------------------------------------------------
# CB-1: CATEGPSurface.sample_posterior
# ---------------------------------------------------------------------------

def test_cate_gp_surface_sample_posterior_shape():
    """sample_posterior returns shape (N, n_samples) and spans > 1 posterior std."""
    from sparc.causal.spatial_cate import CATEGPSurface

    rng = np.random.default_rng(42)
    n = 80
    coords = rng.uniform(0, 1, (n, 2))
    cate = -0.2 + 0.6 * coords[:, 0] + rng.normal(0, 0.05, n)

    surf = CATEGPSurface(n_restarts_optimizer=0).fit(cate, coords)

    n_samples = 10
    samples = surf.sample_posterior(n_samples, coords)

    assert samples.shape == (n, n_samples), (
        f"Expected ({n}, {n_samples}), got {samples.shape}"
    )
    # Samples should span more than 1 posterior std across draws
    sample_range = samples.max(axis=1) - samples.min(axis=1)
    _, std = surf.predict(coords)
    assert sample_range.mean() > std.mean() * 0.5


def test_cate_gp_surface_sample_posterior_raises_before_fit():
    from sparc.causal.spatial_cate import CATEGPSurface

    surf = CATEGPSurface()
    with pytest.raises(RuntimeError, match="fit\\(\\)"):
        surf.sample_posterior(5, np.zeros((3, 2)))
