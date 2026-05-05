"""CATE ranking validation: RATE / TOC / QINI.

Wager (2025) audit Gap 2. Implements:
    - TOC(q): Targeting Operating Characteristic curve.
    - QINI(q): cumulative incremental gain at quantile q.
    - RATE: AUC of TOC against the random-targeting baseline (Yadlowsky
      et al. 2025, Eq. 2.3).

Inputs are priority *scores* (e.g., posterior-mean CATE estimates) and
*doubly-robust* pseudo-outcomes (AIPW Y_i^*).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ._audit import mark_addressed

GAP_2_IMPLEMENTED = True


def _sorted_by_score(scores: np.ndarray, dr_scores: np.ndarray):
    order = np.argsort(-scores)  # descending: highest priority first
    return scores[order], dr_scores[order]


def compute_toc(scores: np.ndarray,
                dr_scores: np.ndarray,
                grid: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Targeting Operating Characteristic.

    For each quantile q, TOC(q) = E[Y* | top q] - E[Y*]. Plot against q
    to inspect prioritisation quality.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    dr = np.asarray(dr_scores, dtype=float).ravel()
    if scores.shape != dr.shape:
        raise ValueError("scores and dr_scores must align")

    _, dr_sorted = _sorted_by_score(scores, dr)
    n = len(dr_sorted)
    if grid is None:
        grid = np.linspace(0.05, 1.0, 20)

    overall = dr_sorted.mean()
    toc = []
    for q in grid:
        k = max(1, int(round(q * n)))
        toc.append(dr_sorted[:k].mean() - overall)
    return {"q": grid, "toc": np.array(toc)}


def compute_qini(scores: np.ndarray,
                 dr_scores: np.ndarray,
                 grid: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Cumulative gain (QINI) at quantile q: q * (mean top-q - mean overall)."""
    out = compute_toc(scores, dr_scores, grid=grid)
    qini = out["q"] * out["toc"]
    return {"q": out["q"], "qini": qini}


def compute_rate(scores: np.ndarray,
                 dr_scores: np.ndarray,
                 n_bootstrap: int = 500,
                 alpha: float = 0.05,
                 random_state: Optional[int] = None) -> Dict[str, object]:
    """Rank-Average Treatment Effect AUC, with bootstrap CI.

    RATE = ∫₀¹ TOC(q) dq.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    dr = np.asarray(dr_scores, dtype=float).ravel()

    grid = np.linspace(1.0 / len(scores), 1.0, len(scores))
    toc_full = compute_toc(scores, dr, grid=grid)["toc"]
    rate_auc = float(np.trapz(toc_full, grid))

    rng = np.random.default_rng(random_state)
    boots = np.empty(n_bootstrap)
    n = len(scores)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        toc_b = compute_toc(scores[idx], dr[idx], grid=grid)["toc"]
        boots[b] = float(np.trapz(toc_b, grid))
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))

    mark_addressed(2)
    return {
        "rate_auc": rate_auc,
        "rate_ci": (lo, hi),
        "rate_se": float(np.std(boots, ddof=1)),
        "toc_grid": grid,
        "toc": toc_full,
    }


def aipw_pseudo_outcomes(Y: np.ndarray,
                         W: np.ndarray,
                         mu1: np.ndarray,
                         mu0: np.ndarray,
                         e: np.ndarray) -> np.ndarray:
    """Augmented inverse-propensity-weighted pseudo-outcomes.

    Y_i^* = mu1(X_i) - mu0(X_i)
            + W_i * (Y_i - mu1(X_i)) / e(X_i)
            - (1 - W_i) * (Y_i - mu0(X_i)) / (1 - e(X_i))
    """
    Y, W, mu1, mu0, e = (np.asarray(a, dtype=float).ravel()
                         for a in (Y, W, mu1, mu0, e))
    e = np.clip(e, 1e-3, 1 - 1e-3)
    return (mu1 - mu0
            + W * (Y - mu1) / e
            - (1 - W) * (Y - mu0) / (1 - e))
