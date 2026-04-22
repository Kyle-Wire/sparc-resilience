"""Negative-control validation for causal claims.

A *negative control* is a treatment-outcome relationship that *should
not* exist according to the DAG. If the pipeline still detects a
sizeable effect on a negative-control outcome, something is wrong:
unmeasured confounding, leakage, mis-specified DAG, or just noise that
the model is over-fitting.

This module supports two complementary tests:

1. **Permutation test on CATE values.** Shuffle the cell-level CATE
   array and recompute the mean. Repeat *N* times to build a null
   distribution; report the empirical two-sided p-value of the actual
   mean against that null. A small p-value means the spatial
   heterogeneity is unlikely under the null of "no real treatment
   effect".

2. **Placebo treatment scoring.** Given any numeric column the user
   knows is unrelated to the outcome (e.g. random ID, parcel index),
   estimate its bivariate slope on the outcome. The slope distribution
   across placebos should be tightly centred on zero; large magnitudes
   indicate residual confounding or pipeline leakage.

Both helpers are pure-Python so the FastAPI layer can apply them
without re-running stage 3.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class PermutationResult:
    n: int
    mean_observed: float
    mean_null: float
    std_null: float
    p_value: float
    z_score: float
    n_permutations: int
    passed: bool          # True if p > 0.05 (the null cannot be rejected)


@dataclass
class PlaceboResult:
    placebo_label: str
    slope: float
    intercept: float
    abs_slope: float
    n: int


def permutation_test_cate(
    cate_values: Sequence[float],
    *,
    n_permutations: int = 1000,
    seed: Optional[int] = 42,
) -> PermutationResult:
    """Two-sided permutation test for the mean of *cate_values*.

    The null distribution is constructed by sign-flipping each value
    independently with probability 0.5 — the appropriate null for an
    effect array centred on zero under "no treatment effect".
    """
    rng = random.Random(seed)
    vals = [float(v) for v in cate_values if math.isfinite(float(v))]
    n = len(vals)
    if n == 0:
        raise ValueError("cate_values is empty")

    observed = statistics.fmean(vals)
    null_means: list[float] = []
    for _ in range(max(1, n_permutations)):
        flipped = [v if rng.random() < 0.5 else -v for v in vals]
        null_means.append(statistics.fmean(flipped))

    null_mean = statistics.fmean(null_means)
    null_std = statistics.pstdev(null_means) if len(null_means) > 1 else 0.0
    # Two-sided p-value: fraction of |null| >= |observed|.
    extreme = sum(1 for m in null_means if abs(m) >= abs(observed))
    p_value = (extreme + 1) / (len(null_means) + 1)
    z_score = (observed - null_mean) / null_std if null_std > 0 else float("nan")

    return PermutationResult(
        n=n,
        mean_observed=observed,
        mean_null=null_mean,
        std_null=null_std,
        p_value=p_value,
        z_score=z_score,
        n_permutations=n_permutations,
        passed=p_value > 0.05,   # negative-control passes when no signal
    )


def placebo_slopes(
    outcome: Sequence[float],
    placebos: dict[str, Sequence[float]],
) -> list[PlaceboResult]:
    """Fit OLS slope of ``outcome ~ placebo`` for each placebo column.

    ``placebos`` is ``{label: values}``. The returned list lets the UI
    show that all placebo slopes cluster near zero (good) or that some
    have suspicious magnitude (bad).
    """
    y = [float(v) for v in outcome]
    n = len(y)
    if n < 2:
        return []
    y_mean = sum(y) / n

    out: list[PlaceboResult] = []
    for label, xs in placebos.items():
        x = [float(v) for v in xs][:n]
        if len(x) != n:
            continue
        x_mean = sum(x) / n
        var_x = sum((xi - x_mean) ** 2 for xi in x)
        if var_x <= 1e-12:
            continue
        cov_xy = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        slope = cov_xy / var_x
        intercept = y_mean - slope * x_mean
        out.append(
            PlaceboResult(
                placebo_label=str(label),
                slope=slope,
                intercept=intercept,
                abs_slope=abs(slope),
                n=n,
            )
        )
    return out
