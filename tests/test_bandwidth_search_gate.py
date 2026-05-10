"""Improvement-gate semantics for the narrow bandwidth search.

Two paths must both work:

1. **Promotion** — a candidate that beats the correlogram baseline by at least
   ``improvement_threshold_pct`` must be returned with ``promoted=True`` and
   ``provenance.source == "surrogate_search"``.
2. **Fallback** — a candidate within ±threshold of baseline must trigger a
   fallback to correlogram defaults with ``promoted=False`` and
   ``provenance.source == "correlogram"``.
"""

from __future__ import annotations

import pytest

from sparc.run.inner_loops.bandwidth_search import (
    CorrelogramPrior,
    run_narrow_bandwidth_search,
)


_PRIOR = CorrelogramPrior(
    bandwidth_mean=1000.0,
    bandwidth_hdi_lo=500.0,
    bandwidth_hdi_hi=2000.0,
)


def test_promotion_path_winning_candidate_promoted():
    """A surrogate that loves bw=900 + a full-model that confirms a 5% win
    should promote that candidate."""

    target_bw = 900.0
    baseline = 1.0  # correlogram-baseline composite score

    def surrogate(hparams):
        # Quadratic well centered at target_bw — BO will home in on it.
        return 0.5 + ((hparams["bandwidth"] - target_bw) / 500.0) ** 2

    def full_model(hparams):
        # Beats baseline by ~5% at target_bw.
        return 0.95 + 0.5 * ((hparams["bandwidth"] - target_bw) / 500.0) ** 2

    result = run_narrow_bandwidth_search(
        model_name="gwr",
        prior=_PRIOR,
        correlogram_baseline_score=baseline,
        surrogate_scorer=surrogate,
        full_model_scorer=full_model,
        n_trials=12,
        n_verify=3,
        improvement_threshold_pct=1.0,
        random_seed=0,
    )

    assert result.promoted is True
    assert result.decision.action == "accept"
    assert result.provenance.source == "surrogate_search"
    assert result.delta_pct >= 1.0
    # Winning bandwidth should be near the well.
    assert abs(result.chosen_hparams["bandwidth"] - target_bw) < 400.0
    # Baseline comparison evidence is recorded.
    bc = result.provenance.baseline_comparison
    assert bc is not None
    assert bc["outcome"] == "promoted"
    assert bc["delta_pct"] >= 1.0


def test_fallback_path_no_meaningful_improvement():
    """Surrogate suggests improvement, full-model verification reveals none —
    must fall back to correlogram defaults."""

    baseline = 1.0

    def surrogate(hparams):
        # Deceives BO into thinking other bandwidths are better.
        return 0.1 + 0.0001 * abs(hparams["bandwidth"] - 1500.0)

    def full_model(hparams):
        # In reality every candidate is within ±0.3% of baseline → fallback.
        return 0.999  # 0.1% improvement, well below 1.0% threshold

    result = run_narrow_bandwidth_search(
        model_name="gwr",
        prior=_PRIOR,
        correlogram_baseline_score=baseline,
        surrogate_scorer=surrogate,
        full_model_scorer=full_model,
        n_trials=10,
        n_verify=3,
        improvement_threshold_pct=1.0,
        random_seed=0,
    )

    assert result.promoted is False
    assert result.decision.action == "fallback"
    assert result.provenance.source == "correlogram"
    # Chosen hparams must equal the correlogram baseline.
    assert result.chosen_hparams == result.correlogram_baseline
    assert result.chosen_hparams["bandwidth"] == pytest.approx(1000.0)
    bc = result.provenance.baseline_comparison
    assert bc is not None
    assert bc["outcome"] == "fallback"


def test_gwrf_search_includes_k_neighbors_dim():
    baseline = 1.0

    def surrogate(hparams):
        assert "k_neighbors" in hparams
        assert isinstance(hparams["k_neighbors"], int)
        return 0.5

    def full_model(hparams):
        return 1.0  # no improvement

    result = run_narrow_bandwidth_search(
        model_name="gwrf",
        prior=CorrelogramPrior(
            bandwidth_mean=1000.0,
            bandwidth_hdi_lo=500.0,
            bandwidth_hdi_hi=2000.0,
            k_neighbors_default=16,
        ),
        correlogram_baseline_score=baseline,
        surrogate_scorer=surrogate,
        full_model_scorer=full_model,
        n_trials=6,
        n_verify=2,
        random_seed=0,
    )

    assert "k_neighbors" in result.correlogram_baseline
    assert result.correlogram_baseline["k_neighbors"] == 16
    assert "k_neighbors" in result.chosen_hparams


def test_invalid_n_trials_raises():
    with pytest.raises(ValueError):
        run_narrow_bandwidth_search(
            model_name="gwr",
            prior=_PRIOR,
            correlogram_baseline_score=1.0,
            surrogate_scorer=lambda h: 0.5,
            full_model_scorer=lambda h: 1.0,
            n_trials=1,
        )
