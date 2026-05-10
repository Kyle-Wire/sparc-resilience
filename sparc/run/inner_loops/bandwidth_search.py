"""Narrow Bayesian-Optimization inner loop for Stage 2 hyperparameters.

Per-base-model bandwidth (and ``k_neighbors`` where applicable) search with:

* **Prior from the Stage 0 correlogram** — the κ posterior mean and HDI seed
  the BO's Gaussian prior over log-bandwidth.
* **Surrogate-assisted exploration** — ``gp_minimize`` over a fast surrogate
  scoring function (caller-provided).
* **Trust-region verification** — the top-K BO candidates are re-scored with
  the *full* base model on spatial folds. Surrogate scores never determine
  the winner; verified scores do.
* **Improvement gate** — the verified best is promoted only if it beats the
  correlogram-derived baseline by at least ``improvement_threshold_pct``.
  Otherwise the loop falls back to correlogram defaults — this guarantees
  the search cannot regress today's pipeline quality.

The composite objective combines OOF MSE with PDE-residual, monotonicity, and
absolute residual Moran's I penalties (λ weights match ``sparc_joint_loss()``
defaults). All λ values are caller-provided so this module stays decoupled
from the training-stage configuration.

This module deliberately accepts callable abstractions for surrogate scoring
and full-model scoring so it can be unit-tested without instantiating GWR /
GWRF / GGPGAM models. See ``tests/test_bandwidth_search_gate.py`` and
``tests/test_bandwidth_search_smoke.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Optional, Sequence

import numpy as np

from sparc.run.decisions import DecisionRecord, ProvenanceInfo


ModelName = Literal["gwr", "gwrf", "ggpgam"]


# ---------------------------------------------------------------------------
# Composite objective
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompositeLambdas:
    """Penalty weights for the composite objective.

    Defaults mirror ``sparc_joint_loss()``'s typical Stage 2 inheritance.
    """

    phys: float = 0.0
    mono: float = 0.0
    spatial: float = 0.1


def composite_objective(
    *,
    predictions: np.ndarray,
    y: np.ndarray,
    coords: np.ndarray,
    lambdas: CompositeLambdas,
    pde_residual: Optional[float] = None,
    monotonicity_penalty: Optional[float] = None,
    morans_i_residuals: Optional[float] = None,
) -> float:
    """Return the composite score (smaller-is-better).

    .. math::

       \\mathcal{L}(\\theta) = \\text{MSE}_{OOF}(\\theta)
                              + \\lambda_{phys} \\cdot R_{PDE}(\\theta)
                              + \\lambda_{mono} \\cdot P_{mono}(\\theta)
                              + \\lambda_{spatial} \\cdot |I_{Moran,resid}|
    """
    pred = np.asarray(predictions, dtype=np.float64).reshape(-1)
    truth = np.asarray(y, dtype=np.float64).reshape(-1)
    if pred.shape != truth.shape:
        raise ValueError(
            f"predictions {pred.shape} and y {truth.shape} must align"
        )
    residuals = pred - truth
    mse = float(np.mean(residuals ** 2))

    score = mse

    if lambdas.phys != 0.0 and pde_residual is not None:
        score += float(lambdas.phys) * float(pde_residual)

    if lambdas.mono != 0.0 and monotonicity_penalty is not None:
        score += float(lambdas.mono) * float(monotonicity_penalty)

    if lambdas.spatial != 0.0:
        if morans_i_residuals is None and coords is not None and coords.size > 0:
            morans_i_residuals = _morans_i_knn(residuals, np.asarray(coords))
        if morans_i_residuals is not None and math.isfinite(morans_i_residuals):
            score += float(lambdas.spatial) * abs(float(morans_i_residuals))

    return score


def _morans_i_knn(residuals: np.ndarray, coords: np.ndarray, k: int = 8) -> float:
    """Lightweight k-NN Moran's I — mirrors ``calculate_fold_spatial_autocorr``.

    Used as a fallback when callers don't pass an explicit Moran's I term.
    Returns 0.0 on degenerate input rather than raising.
    """
    res = np.asarray(residuals, dtype=np.float64).reshape(-1)
    pts = np.asarray(coords, dtype=np.float64)
    n = res.shape[0]
    if n < 10 or pts.shape[0] != n:
        return 0.0
    try:
        from sklearn.neighbors import NearestNeighbors
    except Exception:
        return 0.0
    k_eff = max(1, min(k, n - 1))
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1).fit(pts)
    _, idx = nbrs.kneighbors(pts)
    centered = res - res.mean()
    denom = float(np.sum(centered ** 2))
    if denom == 0.0:
        return 0.0
    weight = 1.0 / k_eff
    total_sim = 0.0
    total_w = 0.0
    for i in range(n):
        for j in range(1, k_eff + 1):
            ni = idx[i, j]
            total_sim += weight * centered[i] * centered[ni]
            total_w += weight
    if total_w == 0.0:
        return 0.0
    return float((n / total_w) * (total_sim / denom))


# ---------------------------------------------------------------------------
# Search space + prior
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorrelogramPrior:
    """Per-model prior derived from the Stage 0 correlogram κ posterior."""

    bandwidth_mean: float
    bandwidth_hdi_lo: float
    bandwidth_hdi_hi: float
    k_neighbors_default: Optional[int] = None  # GWRF only

    def __post_init__(self) -> None:
        if self.bandwidth_mean <= 0:
            raise ValueError("bandwidth_mean must be positive")
        if not (0 < self.bandwidth_hdi_lo <= self.bandwidth_mean <= self.bandwidth_hdi_hi):
            raise ValueError(
                "bandwidth_mean must lie within [hdi_lo, hdi_hi] and all must be positive: "
                f"got mean={self.bandwidth_mean}, hdi=[{self.bandwidth_hdi_lo}, "
                f"{self.bandwidth_hdi_hi}]"
            )


def build_search_space(
    model_name: ModelName,
    prior: CorrelogramPrior,
    *,
    k_neighbors_bounds: tuple[int, int] = (4, 64),
) -> tuple[list[Any], list[float | int]]:
    """Return ``(skopt_dimensions, x0)`` for the BO search.

    ``x0`` is the correlogram-default initial point (used to anchor the BO
    so the very first sample is the baseline configuration).
    """
    from skopt.space import Integer, Real  # local import — heavy dep

    dims: list[Any] = [
        Real(
            prior.bandwidth_hdi_lo,
            prior.bandwidth_hdi_hi,
            prior="log-uniform",
            name="bandwidth",
        ),
    ]
    x0: list[float | int] = [float(prior.bandwidth_mean)]

    if model_name == "gwrf":
        k_default = (
            prior.k_neighbors_default
            if prior.k_neighbors_default is not None
            else int(round(0.5 * (k_neighbors_bounds[0] + k_neighbors_bounds[1])))
        )
        k_default = max(k_neighbors_bounds[0], min(k_neighbors_bounds[1], k_default))
        dims.append(Integer(k_neighbors_bounds[0], k_neighbors_bounds[1], name="k_neighbors"))
        x0.append(int(k_default))

    return dims, x0


# ---------------------------------------------------------------------------
# Search result + decision record builder
# ---------------------------------------------------------------------------


@dataclass
class SearchTrial:
    """One trial in the BO search trace."""

    iteration: int
    hparams: dict[str, Any]
    surrogate_score: float


@dataclass
class VerificationResult:
    """One trust-region verification outcome."""

    rank: int  # 1 = best surrogate score
    hparams: dict[str, Any]
    surrogate_score: float
    full_model_score: float


@dataclass
class BandwidthSearchResult:
    """Output of :func:`run_narrow_bandwidth_search`."""

    model_name: ModelName
    chosen_hparams: dict[str, Any]
    correlogram_baseline: dict[str, Any]
    correlogram_baseline_score: float
    chosen_score: float
    delta_pct: float
    promoted: bool
    decision: DecisionRecord
    provenance: ProvenanceInfo
    trials: list[SearchTrial] = field(default_factory=list)
    verifications: list[VerificationResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_narrow_bandwidth_search(
    *,
    model_name: ModelName,
    prior: CorrelogramPrior,
    correlogram_baseline_score: float,
    surrogate_scorer: Callable[[Mapping[str, Any]], float],
    full_model_scorer: Callable[[Mapping[str, Any]], float],
    n_trials: int = 50,
    n_verify: int = 3,
    improvement_threshold_pct: float = 1.0,
    random_seed: int = 42,
    k_neighbors_bounds: tuple[int, int] = (4, 64),
) -> BandwidthSearchResult:
    """Run the narrow surrogate-assisted BO search for one base model.

    Parameters
    ----------
    model_name
        ``"gwr"``, ``"gwrf"``, or ``"ggpgam"``.
    prior
        Correlogram-derived bandwidth posterior summary for this model.
    correlogram_baseline_score
        Composite score of the model fit at correlogram defaults
        (smaller-is-better). The improvement gate compares against this.
    surrogate_scorer
        Maps a hparams dict → composite score (smaller-is-better) using a
        fast surrogate. Called ``n_trials`` times.
    full_model_scorer
        Maps a hparams dict → composite score (smaller-is-better) using
        the *full* base model on spatial folds. Called at most ``n_verify``
        times during trust-region verification.
    n_trials
        Number of BO iterations (default 50).
    n_verify
        Number of top BO candidates to verify with the full model (default 3).
    improvement_threshold_pct
        Minimum relative improvement (in percent) over the correlogram
        baseline required to promote the search winner (default 1.0).
    random_seed
        Reproducibility seed for the BO acquisition function.
    k_neighbors_bounds
        ``(low, high)`` integer bounds for GWRF's ``k_neighbors`` dimension.

    Returns
    -------
    BandwidthSearchResult
        Full search outcome including BO trace, verification results, and a
        :class:`DecisionRecord` documenting the promote/fallback decision.
    """
    if n_trials < 2:
        raise ValueError("n_trials must be >= 2")
    if n_verify < 1:
        raise ValueError("n_verify must be >= 1")
    if improvement_threshold_pct < 0:
        raise ValueError("improvement_threshold_pct must be >= 0")

    from skopt import gp_minimize

    dims, x0 = build_search_space(
        model_name, prior, k_neighbors_bounds=k_neighbors_bounds
    )
    dim_names = [d.name for d in dims]

    trials: list[SearchTrial] = []

    def _vec_to_hparams(x: Sequence[Any]) -> dict[str, Any]:
        return {name: _coerce_value(v) for name, v in zip(dim_names, x)}

    def _objective(x: Sequence[Any]) -> float:
        hparams = _vec_to_hparams(x)
        score = float(surrogate_scorer(hparams))
        trials.append(
            SearchTrial(
                iteration=len(trials) + 1,
                hparams=hparams,
                surrogate_score=score,
            )
        )
        if not math.isfinite(score):
            # Penalize NaN/inf so BO steers away.
            return 1e12
        return score

    # gp_minimize requires n_initial_points >= 1. We always pass x0 as the
    # correlogram-default anchor, then let the GP explore.
    n_initial_points = max(5, min(10, n_trials // 2))
    gp_minimize(
        _objective,
        dims,
        x0=x0,
        n_calls=n_trials,
        n_initial_points=n_initial_points,
        random_state=random_seed,
    )

    # Trust-region verification: take top-K trials by surrogate score and
    # re-score them with the full model.
    sorted_trials = sorted(trials, key=lambda t: t.surrogate_score)
    verifications: list[VerificationResult] = []
    seen_keys: set[tuple] = set()
    for rank, trial in enumerate(sorted_trials, start=1):
        if len(verifications) >= n_verify:
            break
        key = tuple(sorted(trial.hparams.items()))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        full_score = float(full_model_scorer(trial.hparams))
        verifications.append(
            VerificationResult(
                rank=len(verifications) + 1,
                hparams=trial.hparams,
                surrogate_score=trial.surrogate_score,
                full_model_score=full_score,
            )
        )

    # Improvement gate.
    correlogram_defaults = _vec_to_hparams(x0)
    if not verifications:
        # Should never happen given n_trials >= 2, but be defensive.
        return _fallback_result(
            model_name=model_name,
            correlogram_defaults=correlogram_defaults,
            correlogram_baseline_score=correlogram_baseline_score,
            reason="no_verifications",
            trials=trials,
            verifications=verifications,
        )

    best_verified = min(verifications, key=lambda v: v.full_model_score)

    # Smaller-is-better: a positive delta_pct means we improved.
    baseline = float(correlogram_baseline_score)
    if baseline == 0.0:
        delta_pct = 0.0 if best_verified.full_model_score == 0.0 else float("inf")
    else:
        delta_pct = 100.0 * (baseline - best_verified.full_model_score) / abs(baseline)

    promoted = delta_pct >= improvement_threshold_pct

    if promoted:
        decision = DecisionRecord(
            rule="improvement_gate",
            evidence={
                "model": model_name,
                "baseline_score": baseline,
                "candidate_score": best_verified.full_model_score,
                "delta_pct": delta_pct,
                "threshold_pct": improvement_threshold_pct,
                "winning_hparams": best_verified.hparams,
            },
            action="accept",
        )
        provenance = ProvenanceInfo(
            source="surrogate_search",
            search_log={
                "model": model_name,
                "n_trials": len(trials),
                "n_verify": len(verifications),
                "trials": [
                    {"iter": t.iteration, "hparams": t.hparams, "score": t.surrogate_score}
                    for t in trials
                ],
                "verifications": [
                    {
                        "rank": v.rank,
                        "hparams": v.hparams,
                        "surrogate_score": v.surrogate_score,
                        "full_model_score": v.full_model_score,
                    }
                    for v in verifications
                ],
            },
            baseline_comparison={
                "baseline_score": baseline,
                "candidate_score": best_verified.full_model_score,
                "delta_pct": delta_pct,
                "threshold_pct": improvement_threshold_pct,
                "outcome": "promoted",
            },
        )
        chosen_hparams = best_verified.hparams
        chosen_score = best_verified.full_model_score
    else:
        decision = DecisionRecord(
            rule="improvement_gate",
            evidence={
                "model": model_name,
                "baseline_score": baseline,
                "candidate_score": best_verified.full_model_score,
                "delta_pct": delta_pct,
                "threshold_pct": improvement_threshold_pct,
                "winning_hparams": best_verified.hparams,
            },
            action="fallback",
        )
        provenance = ProvenanceInfo(
            source="correlogram",
            search_log={
                "model": model_name,
                "n_trials": len(trials),
                "n_verify": len(verifications),
            },
            baseline_comparison={
                "baseline_score": baseline,
                "candidate_score": best_verified.full_model_score,
                "delta_pct": delta_pct,
                "threshold_pct": improvement_threshold_pct,
                "outcome": "fallback",
            },
        )
        chosen_hparams = correlogram_defaults
        chosen_score = baseline

    return BandwidthSearchResult(
        model_name=model_name,
        chosen_hparams=chosen_hparams,
        correlogram_baseline=correlogram_defaults,
        correlogram_baseline_score=baseline,
        chosen_score=chosen_score,
        delta_pct=delta_pct,
        promoted=promoted,
        decision=decision,
        provenance=provenance,
        trials=trials,
        verifications=verifications,
    )


def _fallback_result(
    *,
    model_name: ModelName,
    correlogram_defaults: dict[str, Any],
    correlogram_baseline_score: float,
    reason: str,
    trials: list[SearchTrial],
    verifications: list[VerificationResult],
) -> BandwidthSearchResult:
    decision = DecisionRecord(
        rule="improvement_gate",
        evidence={"model": model_name, "reason": reason},
        action="fallback",
    )
    provenance = ProvenanceInfo(
        source="correlogram",
        baseline_comparison={"reason": reason, "outcome": "fallback"},
    )
    return BandwidthSearchResult(
        model_name=model_name,
        chosen_hparams=correlogram_defaults,
        correlogram_baseline=correlogram_defaults,
        correlogram_baseline_score=correlogram_baseline_score,
        chosen_score=correlogram_baseline_score,
        delta_pct=0.0,
        promoted=False,
        decision=decision,
        provenance=provenance,
        trials=trials,
        verifications=verifications,
    )


def _coerce_value(v: Any) -> Any:
    """Convert numpy scalar types to Python native types for clean JSON."""
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


__all__ = [
    "ModelName",
    "CompositeLambdas",
    "CorrelogramPrior",
    "SearchTrial",
    "VerificationResult",
    "BandwidthSearchResult",
    "build_search_space",
    "composite_objective",
    "run_narrow_bandwidth_search",
]
