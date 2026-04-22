"""Monte Carlo decision uncertainty.

Given a set of intervention candidates with mean effect ``μ`` and posterior
std ``σ``, repeatedly sample effect draws, re-rank, and report selection
probabilities and value-at-risk per candidate.

This makes the optimizer's uncertainty *visible* — instead of one ranked
list, the user sees how often each candidate ends up in the budget under
posterior uncertainty.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional, Sequence

from .optimizer import InterventionCandidate, _EPS


@dataclass
class UncertaintyResult:
    candidate: str
    selection_probability: float    # P(selected under MC sampling)
    rank_mean: float                # mean rank (1 = best)
    rank_p10: float
    rank_p90: float
    effect_p10: float               # 10th-percentile causal effect
    effect_p50: float
    effect_p90: float


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    frac = pos - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def monte_carlo_decision(
    candidates: Sequence[InterventionCandidate],
    *,
    budget: Optional[float] = None,
    robustness_lambda: float = 0.0,
    minimise: bool = False,
    n_draws: int = 500,
    seed: Optional[int] = 42,
) -> list[UncertaintyResult]:
    """Repeatedly sample effects, re-rank, and report stability metrics.

    For each Monte-Carlo draw ``k = 1..n_draws`` we sample
    ``μ̃_c ~ Normal(μ_c, σ_c)`` per candidate, recompute the score
    ``w_c · sign·(μ̃_c − λσ_c) / (cost_c + ε)``, sort, and (if a budget
    is supplied) greedy-select.  The returned list is in the same order
    as the input ``candidates``.
    """
    rng = random.Random(seed)
    sign = -1.0 if minimise else 1.0
    n = len(candidates)
    if n == 0:
        return []

    rank_samples: list[list[int]] = [[] for _ in range(n)]
    effect_samples: list[list[float]] = [[] for _ in range(n)]
    selection_counts = [0] * n

    for _ in range(max(1, int(n_draws))):
        scored: list[tuple[int, float, float]] = []
        for idx, c in enumerate(candidates):
            sigma = max(c.effect_std, 0.0)
            mu_draw = c.mean_effect + rng.gauss(0.0, sigma) if sigma > 0 else c.mean_effect
            risk_adj = mu_draw - robustness_lambda * sigma
            score = c.equity_weight * (sign * risk_adj) / (c.cost + _EPS)
            scored.append((idx, score, mu_draw))

        # Rank: 1 = best score (largest).
        scored.sort(key=lambda t: t[1], reverse=True)
        for rank_pos, (idx, _score, mu_draw) in enumerate(scored, start=1):
            rank_samples[idx].append(rank_pos)
            effect_samples[idx].append(mu_draw)

        if budget is not None and budget > 0:
            running = 0.0
            for idx, _score, _mu in scored:
                cost = candidates[idx].cost
                if running + cost <= budget:
                    selection_counts[idx] += 1
                    running += cost

    draws = max(1, int(n_draws))
    out: list[UncertaintyResult] = []
    for idx, c in enumerate(candidates):
        ranks = rank_samples[idx]
        effects = effect_samples[idx]
        out.append(
            UncertaintyResult(
                candidate=c.name,
                selection_probability=selection_counts[idx] / draws,
                rank_mean=sum(ranks) / len(ranks) if ranks else float("nan"),
                rank_p10=_percentile([float(r) for r in ranks], 0.10),
                rank_p90=_percentile([float(r) for r in ranks], 0.90),
                effect_p10=_percentile(effects, 0.10),
                effect_p50=_percentile(effects, 0.50),
                effect_p90=_percentile(effects, 0.90),
            )
        )
    return out
