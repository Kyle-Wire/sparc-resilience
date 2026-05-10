"""End-to-end smoke test for the narrow bandwidth search.

A 200-point synthetic spatial dataset; tiny pseudo-surrogate; tiny pseudo
full-model; verify the loop produces a valid ``BandwidthSearchResult`` with a
populated trace and a JSON-serializable :class:`DecisionRecord`.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from sparc.run.inner_loops.bandwidth_search import (
    CompositeLambdas,
    CorrelogramPrior,
    composite_objective,
    run_narrow_bandwidth_search,
)


@pytest.fixture
def synthetic_dataset():
    rng = np.random.default_rng(7)
    n = 200
    coords = rng.uniform(0, 10_000, size=(n, 2))
    # Smooth Gaussian spatial trend + small noise.
    cx, cy = 5000.0, 5000.0
    y = np.exp(-((coords[:, 0] - cx) ** 2 + (coords[:, 1] - cy) ** 2) / (2 * 1500.0 ** 2))
    y += rng.normal(0, 0.05, size=n)
    return coords, y


def _make_predictor(bandwidth: float, coords: np.ndarray, y: np.ndarray):
    """Toy 'model': predict y by Gaussian-kernel-smoothing in space."""
    from sklearn.neighbors import NearestNeighbors

    n = coords.shape[0]
    nbrs = NearestNeighbors(n_neighbors=min(15, n - 1)).fit(coords)
    dists, idx = nbrs.kneighbors(coords)
    weights = np.exp(-(dists ** 2) / (2 * (bandwidth ** 2) + 1e-12))
    weights /= weights.sum(axis=1, keepdims=True) + 1e-12
    preds = np.einsum("ij,ij->i", weights, y[idx])
    return preds


def test_smoke_runs_end_to_end(synthetic_dataset):
    coords, y = synthetic_dataset
    lambdas = CompositeLambdas(phys=0.0, mono=0.0, spatial=0.1)

    def score_with_bandwidth(bw: float) -> float:
        preds = _make_predictor(bw, coords, y)
        return composite_objective(
            predictions=preds,
            y=y,
            coords=coords,
            lambdas=lambdas,
        )

    baseline_bw = 800.0
    baseline_score = score_with_bandwidth(baseline_bw)

    def surrogate(hparams):
        # Surrogate is just the noisy real scorer.
        return score_with_bandwidth(float(hparams["bandwidth"])) + 0.001

    def full_model(hparams):
        return score_with_bandwidth(float(hparams["bandwidth"]))

    prior = CorrelogramPrior(
        bandwidth_mean=baseline_bw,
        bandwidth_hdi_lo=400.0,
        bandwidth_hdi_hi=2400.0,
    )

    result = run_narrow_bandwidth_search(
        model_name="gwr",
        prior=prior,
        correlogram_baseline_score=baseline_score,
        surrogate_scorer=surrogate,
        full_model_scorer=full_model,
        n_trials=6,
        n_verify=2,
        improvement_threshold_pct=0.5,
        random_seed=42,
    )

    # Loop completed and produced a populated trace.
    assert len(result.trials) >= 6
    assert 1 <= len(result.verifications) <= 2
    assert "bandwidth" in result.chosen_hparams

    # Decision record is JSON-serializable end-to-end.
    payload = {
        "decision": result.decision.model_dump(mode="json"),
        "provenance": result.provenance.model_dump(mode="json"),
    }
    json.dumps(payload)
