"""Stage 2 surrogate-assisted bandwidth search orchestration.

This module wires the narrow Bayesian-Optimization inner loop
(:func:`sparc.run.inner_loops.bandwidth_search.run_narrow_bandwidth_search`)
into Stage 2's enhanced spatial CV.

For each base model that has a configurable bandwidth (GWR, GWRF), the search:

1. Builds a :class:`CorrelogramPrior` from the median per-variable bandwidth
   produced by Stage 0's correlogram.
2. Runs the BO loop with a fast spatial-smoother surrogate (subsampled for
   speed) and a full-data verification scorer.
3. Applies the winning hyperparameters to ``base_config['models'][model]`` so
   :meth:`EnhancedSpatialCV.create_optimized_models` picks them up automatically.

The whole pipeline is gated by ``pipeline.use_surrogate_search`` (default
``false``). When the flag is off this module is never invoked and Stage 2
behavior is byte-identical to the legacy path.

The v1 surrogate is a Gaussian-kernel smoother in coordinate space — *not* the
differentiable neural surrogates from the V2 training path. Hoisting the
neural surrogates earlier in Stage 2 is deferred (PRD open question #1); the
kernel smoother is a faithful, cheap stand-in that responds to bandwidth
changes the same way the real models do, so the safety machinery
(verification + improvement gate) is exercised honestly.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional

import numpy as np

from sparc.run.decisions import (
    DecisionRecord,
    MetricEvaluation,
    NextAction,
    ProvenanceInfo,
    StageDecision,
)
from sparc.run.inner_loops.bandwidth_search import (
    CompositeLambdas,
    CorrelogramPrior,
    ModelName,
    run_narrow_bandwidth_search,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_stage2_bandwidth_searches(
    *,
    X: np.ndarray,
    y: np.ndarray,
    coords: np.ndarray,
    base_config: dict,
    correlogram_bandwidths: Optional[dict[str, float]] = None,
) -> StageDecision:
    """Run the per-model surrogate-assisted bandwidth search.

    Mutates ``base_config['models'][model]`` in place to apply the chosen
    hyperparameters (winner or correlogram fallback). Returns the Stage 2
    :class:`StageDecision` aggregating all per-model search outcomes.

    When the search infrastructure cannot run (no correlogram data, no
    eligible models), returns a minimal ``passed`` decision with provenance
    ``"correlogram"`` and a single explanatory :class:`DecisionRecord`.

    Parameters
    ----------
    X, y, coords
        Stage 2's working dataset (after feature selection).
    base_config
        The active project config dict. Mutated in place.
    correlogram_bandwidths
        Optional ``{var_name: bandwidth}`` from
        :meth:`EnhancedSpatialCV.get_variable_bandwidths`. When ``None`` or
        empty, the search is skipped and the returned decision falls back to
        correlogram defaults.
    """
    cfg_search = (
        (base_config.get("pipeline") or {}).get("surrogate_search") or {}
    )
    n_trials = int(cfg_search.get("n_trials", 50))
    n_verify = int(cfg_search.get("n_verify", 3))
    threshold_pct = float(cfg_search.get("improvement_threshold_pct", 1.0))
    bo_seed = int(cfg_search.get("bo_random_seed", 42))

    lambdas = _resolve_lambdas(cfg_search.get("composite_lambdas") or {})

    eligible_models = _eligible_models(base_config)
    if not correlogram_bandwidths or not eligible_models:
        reason = (
            "no_correlogram_bandwidths"
            if not correlogram_bandwidths
            else "no_eligible_models"
        )
        return _empty_decision(reason)

    # Anchor the correlogram-prior bandwidth at the median across variables;
    # the HDI is a wide log-window around it so the BO has room to explore.
    bw_values = sorted(float(v) for v in correlogram_bandwidths.values() if v > 0)
    if not bw_values:
        return _empty_decision("no_positive_bandwidths")
    bw_mean = float(np.median(bw_values))
    bw_lo = max(1.0, float(bw_values[0]) * 0.5)
    bw_hi = float(bw_values[-1]) * 2.0
    if bw_lo >= bw_mean:
        bw_lo = bw_mean * 0.5
    if bw_hi <= bw_mean:
        bw_hi = bw_mean * 2.0

    per_model_decisions: list[DecisionRecord] = []
    per_model_metrics: dict[str, MetricEvaluation] = {}
    per_model_search_logs: dict[str, dict] = {}
    per_model_baseline_cmps: dict[str, dict] = {}

    for model_name in eligible_models:
        prior = _build_prior_for_model(
            model_name=model_name,
            bw_mean=bw_mean,
            bw_lo=bw_lo,
            bw_hi=bw_hi,
            base_config=base_config,
        )
        if prior is None:
            continue

        scorer_factory = _make_scorer_factory(
            X=X, y=y, coords=coords, lambdas=lambdas, model_name=model_name,
        )
        baseline_score = scorer_factory.full({
            "bandwidth": prior.bandwidth_mean,
            **({"k_neighbors": prior.k_neighbors_default} if prior.k_neighbors_default else {}),
        })

        result = run_narrow_bandwidth_search(
            model_name=model_name,
            prior=prior,
            correlogram_baseline_score=baseline_score,
            surrogate_scorer=scorer_factory.surrogate,
            full_model_scorer=scorer_factory.full,
            n_trials=n_trials,
            n_verify=n_verify,
            improvement_threshold_pct=threshold_pct,
            random_seed=bo_seed,
        )

        # Apply the chosen hyperparameters to base_config.
        _apply_hparams_to_config(model_name, result.chosen_hparams, base_config)

        per_model_decisions.append(result.decision)
        per_model_metrics[f"{model_name}_chosen_score"] = MetricEvaluation(
            name=f"{model_name}_chosen_score",
            value=float(result.chosen_score),
            direction="lower_is_better",
        )
        per_model_metrics[f"{model_name}_baseline_score"] = MetricEvaluation(
            name=f"{model_name}_baseline_score",
            value=float(result.correlogram_baseline_score),
            direction="lower_is_better",
        )
        per_model_metrics[f"{model_name}_delta_pct"] = MetricEvaluation(
            name=f"{model_name}_delta_pct",
            value=float(result.delta_pct),
            threshold=threshold_pct,
            passed=bool(result.promoted),
            direction="higher_is_better",
        )
        if result.provenance.search_log is not None:
            per_model_search_logs[model_name] = result.provenance.search_log
        if result.provenance.baseline_comparison is not None:
            per_model_baseline_cmps[model_name] = result.provenance.baseline_comparison

    # Provenance for the whole stage: surrogate_search if any model was
    # promoted, otherwise correlogram fallback.
    any_promoted = any(d.action == "accept" for d in per_model_decisions)
    provenance = ProvenanceInfo(
        source="surrogate_search" if any_promoted else "correlogram",
        search_log={"per_model": per_model_search_logs} if per_model_search_logs else None,
        baseline_comparison=(
            {"per_model": per_model_baseline_cmps} if per_model_baseline_cmps else None
        ),
    )

    return StageDecision(
        stage=2,
        status="passed",
        metrics=per_model_metrics,
        decisions=per_model_decisions,
        next_actions=[NextAction(kind="none", reason="stage2_search_complete")],
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _resolve_lambdas(cfg: dict) -> CompositeLambdas:
    """Resolve composite-objective lambdas. ``null`` in YAML means "use
    sparc_joint_loss() default", which we encode as the dataclass defaults.
    """
    defaults = CompositeLambdas()

    def _val(key: str, default: float) -> float:
        v = cfg.get(key)
        if v is None:
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    return CompositeLambdas(
        phys=_val("phys", defaults.phys),
        mono=_val("mono", defaults.mono),
        spatial=_val("spatial", defaults.spatial),
    )


def _eligible_models(base_config: dict) -> list[ModelName]:
    """Return the models that have a configurable bandwidth and are enabled."""
    models_cfg = base_config.get("models") or {}
    eligible: list[ModelName] = []
    if "gwr" in models_cfg:
        eligible.append("gwr")
    if "gwrf" in models_cfg:
        eligible.append("gwrf")
    # GGPGAM has no scalar bandwidth in the v1 search surface (its smoothness
    # is governed by `lam` and `n_splines`); skipped for v1.
    return eligible


def _build_prior_for_model(
    *,
    model_name: ModelName,
    bw_mean: float,
    bw_lo: float,
    bw_hi: float,
    base_config: dict,
) -> Optional[CorrelogramPrior]:
    """Construct the per-model :class:`CorrelogramPrior`."""
    k_default: Optional[int] = None
    if model_name == "gwrf":
        gwrf_cfg = (base_config.get("models") or {}).get("gwrf") or {}
        try:
            k_default = int(gwrf_cfg.get("k_neighbors", 16))
        except (TypeError, ValueError):
            k_default = 16
    try:
        return CorrelogramPrior(
            bandwidth_mean=bw_mean,
            bandwidth_hdi_lo=bw_lo,
            bandwidth_hdi_hi=bw_hi,
            k_neighbors_default=k_default,
        )
    except ValueError as exc:
        print(f"[stage2_search] could not build prior for {model_name}: {exc}")
        return None


def _apply_hparams_to_config(
    model_name: ModelName, hparams: dict[str, Any], base_config: dict,
) -> None:
    """Mutate ``base_config['models'][model_name]`` in place."""
    models_cfg = base_config.setdefault("models", {})
    model_cfg = models_cfg.setdefault(model_name, {})
    if "bandwidth" in hparams:
        model_cfg["bandwidth"] = float(hparams["bandwidth"])
    if "k_neighbors" in hparams:
        model_cfg["k_neighbors"] = int(hparams["k_neighbors"])


def _empty_decision(reason: str) -> StageDecision:
    return StageDecision(
        stage=2,
        status="passed",
        metrics={},
        decisions=[
            DecisionRecord(
                rule="surrogate_search_skipped",
                evidence={"reason": reason},
                action="fallback",
            )
        ],
        next_actions=[NextAction(kind="none", reason=f"skipped:{reason}")],
        provenance=ProvenanceInfo(source="correlogram"),
    )


# ---------------------------------------------------------------------------
# Surrogate / full-model scorer
# ---------------------------------------------------------------------------


class _SpatialSmootherScorer:
    """Gaussian-kernel spatial smoother scorer.

    Acts as a stand-in for the per-model V2 differentiable surrogate. The
    surrogate path uses a subsampled k-NN neighborhood for speed; the full
    path uses the entire dataset for honest verification.

    For GWRF (which has a ``k_neighbors`` dim) the scorer respects it by
    capping the effective neighborhood; this lets the BO observe a meaningful
    response surface in that dimension.
    """

    def __init__(
        self,
        *,
        X: np.ndarray,
        y: np.ndarray,
        coords: np.ndarray,
        lambdas: CompositeLambdas,
        model_name: ModelName,
        surrogate_subsample: int = 200,
        rng_seed: int = 0,
    ) -> None:
        self._X = np.asarray(X, dtype=np.float64)
        self._y = np.asarray(y, dtype=np.float64).reshape(-1)
        self._coords = np.asarray(coords, dtype=np.float64)
        self._lambdas = lambdas
        self._model_name = model_name

        rng = np.random.default_rng(rng_seed)
        n = self._coords.shape[0]
        if n > surrogate_subsample:
            self._sub_idx = rng.choice(n, size=surrogate_subsample, replace=False)
        else:
            self._sub_idx = np.arange(n)

    # -- callables -------------------------------------------------------

    def surrogate(self, hparams: dict[str, Any]) -> float:
        return self._score(hparams, indices=self._sub_idx)

    def full(self, hparams: dict[str, Any]) -> float:
        return self._score(hparams, indices=None)

    # -- internals -------------------------------------------------------

    def _score(self, hparams: dict[str, Any], indices: Optional[np.ndarray]) -> float:
        bw = float(hparams["bandwidth"])
        if not math.isfinite(bw) or bw <= 0:
            return 1e12

        coords = self._coords if indices is None else self._coords[indices]
        y = self._y if indices is None else self._y[indices]
        n = coords.shape[0]
        if n < 5:
            return 1e12

        k_neighbors = self._resolve_k(hparams, n)

        try:
            from sklearn.neighbors import NearestNeighbors
        except Exception:
            return 1e12

        nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(coords)
        dists, idx = nbrs.kneighbors(coords)
        # Leave-one-out: drop self (column 0).
        dists = dists[:, 1:]
        idx = idx[:, 1:]
        if dists.shape[1] == 0:
            return 1e12

        weights = np.exp(-(dists ** 2) / (2 * bw * bw))
        wsum = weights.sum(axis=1, keepdims=True)
        # Avoid div-by-zero on isolated points.
        wsum = np.where(wsum > 0, wsum, 1.0)
        weights = weights / wsum
        preds = np.einsum("ij,ij->i", weights, y[idx])
        residuals = preds - y
        mse = float(np.mean(residuals ** 2))
        score = mse

        if self._lambdas.spatial != 0.0:
            morans = _quick_morans_i(residuals, coords, k=min(8, n - 1))
            if math.isfinite(morans):
                score += float(self._lambdas.spatial) * abs(morans)

        if not math.isfinite(score):
            return 1e12
        return float(score)

    def _resolve_k(self, hparams: dict[str, Any], n: int) -> int:
        if self._model_name == "gwrf" and "k_neighbors" in hparams:
            k = int(hparams["k_neighbors"])
        else:
            k = 16
        # Need self + k neighbors; cap to dataset size.
        return max(2, min(k + 1, n))


def _make_scorer_factory(
    *,
    X: np.ndarray,
    y: np.ndarray,
    coords: np.ndarray,
    lambdas: CompositeLambdas,
    model_name: ModelName,
) -> _SpatialSmootherScorer:
    return _SpatialSmootherScorer(
        X=X, y=y, coords=coords, lambdas=lambdas, model_name=model_name,
    )


def _quick_morans_i(residuals: np.ndarray, coords: np.ndarray, k: int = 8) -> float:
    res = np.asarray(residuals, dtype=np.float64).reshape(-1)
    pts = np.asarray(coords, dtype=np.float64)
    n = res.shape[0]
    if n < 10:
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


__all__ = ["run_stage2_bandwidth_searches"]
