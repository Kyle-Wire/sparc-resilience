"""Sequential unconfoundedness / g-formula.

Wager (2025) audit Gap 6. Implements:
    - :class:`SequentialIPW`  — Sequential inverse-propensity weighting
      (Theorem 14.3) with per-period propensities multiplied across time.
    - :class:`SequentialAIPW` — Sequential AIPW with backward regression
      (Theorem 14.4), preferred for treatment-confounder feedback.
    - Hernán–Robins pathology detector: when an intermediate state is both
      a mediator *and* a time-varying confounder, naive stratification is
      biased; this module emits an explicit warning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

from ._audit import mark_addressed

GAP_6_IMPLEMENTED = True


@dataclass
class _Period:
    L: np.ndarray   # confounders at time t (n × p_t)
    A: np.ndarray   # treatment at time t   (n,)


def _ensure_history(history: Sequence[dict]) -> List[_Period]:
    out = []
    for t, p in enumerate(history):
        L = np.atleast_2d(np.asarray(p["L"], dtype=float))
        if L.shape[0] == 1 and L.shape[1] != len(p["A"]):
            L = L.T
        A = np.asarray(p["A"], dtype=float).ravel()
        if L.shape[0] != len(A):
            raise ValueError(f"period {t}: L rows ({L.shape[0]}) != A length ({len(A)})")
        out.append(_Period(L=L, A=A))
    return out


def detect_treatment_confounder_feedback(history: Sequence[dict]) -> bool:
    """Detect Hernán-Robins pathology: a confounder L_{t+1} is influenced
    by A_t and itself influences A_{t+1}. Returns True if such a pattern
    is detected via cross-period correlations.
    """
    periods = _ensure_history(history)
    if len(periods) < 2:
        return False
    for t in range(len(periods) - 1):
        A_t = periods[t].A
        L_next = periods[t + 1].L
        A_next = periods[t + 1].A
        # Check (1) A_t -> L_{t+1} (correlation) AND (2) L_{t+1} -> A_{t+1}.
        corr_A_to_L = np.max([abs(np.corrcoef(A_t, L_next[:, j])[0, 1])
                              for j in range(L_next.shape[1])])
        corr_L_to_A = np.max([abs(np.corrcoef(L_next[:, j], A_next)[0, 1])
                              for j in range(L_next.shape[1])])
        if corr_A_to_L > 0.15 and corr_L_to_A > 0.15:
            return True
    return False


class SequentialIPW:
    """Sequential IPW (Theorem 14.3)."""

    def __init__(self, trim: float = 0.02):
        self.trim = trim
        self._propensities = None

    def fit(self, history: Sequence[dict], Y: np.ndarray) -> "SequentialIPW":
        periods = _ensure_history(history)
        Y = np.asarray(Y, dtype=float).ravel()
        # Build cumulative confounder histories.
        propensities = []
        cum_L = None
        for p in periods:
            X_t = p.L if cum_L is None else np.hstack([cum_L, p.L])
            clf = LogisticRegression(max_iter=200).fit(X_t, p.A.astype(int))
            e_t = np.clip(clf.predict_proba(X_t)[:, 1], self.trim, 1 - self.trim)
            propensities.append((clf, e_t, X_t))
            cum_L = X_t
        self._propensities = propensities
        self._Y = Y
        self._periods = periods
        mark_addressed(6)
        return self

    def evaluate_regime(self, regime: Sequence[float]) -> float:
        if self._propensities is None:
            raise RuntimeError("fit() must be called first")
        if len(regime) != len(self._periods):
            raise ValueError("regime length must match number of periods")
        n = len(self._Y)
        weights = np.ones(n)
        indicator = np.ones(n, dtype=bool)
        for t, ((_, e_t, _), p, a) in enumerate(zip(
                self._propensities, self._periods, regime)):
            a = float(a)
            indicator &= np.isclose(p.A, a)
            prob = e_t if a == 1 else 1 - e_t
            weights *= 1.0 / prob
        # Truncate weights at 99th percentile to avoid blow-up.
        cap = np.quantile(weights[indicator], 0.99) if indicator.any() else 1.0
        weights = np.clip(weights, 0, cap)
        if not indicator.any():
            return float("nan")
        return float((weights[indicator] * self._Y[indicator]).sum()
                     / weights[indicator].sum())


class SequentialAIPW:
    """Sequential AIPW with backward regression (Theorem 14.4) — preferred."""

    def __init__(self, trim: float = 0.02):
        self.trim = trim
        self._fitted = False

    def fit(self, history: Sequence[dict], Y: np.ndarray) -> "SequentialAIPW":
        self._periods = _ensure_history(history)
        self._Y = np.asarray(Y, dtype=float).ravel()
        # Fit propensities per period using cumulative history.
        self._propensities = []
        cum = None
        for p in self._periods:
            X_t = p.L if cum is None else np.hstack([cum, p.L])
            clf = LogisticRegression(max_iter=200).fit(X_t, p.A.astype(int))
            self._propensities.append((clf, X_t))
            cum = np.hstack([X_t, p.A.reshape(-1, 1)])

        if detect_treatment_confounder_feedback(history):
            import warnings
            warnings.warn(
                "Hernán-Robins treatment-confounder feedback detected: "
                "naive stratification will be biased. Sequential AIPW handles "
                "this correctly; please rely on this estimate, not on "
                "covariate-adjusted regression.",
                stacklevel=2,
            )
        mark_addressed(6)
        self._fitted = True
        return self

    def evaluate_regime(self, regime: Sequence[float]) -> float:
        if not self._fitted:
            raise RuntimeError("fit() must be called first")
        if len(regime) != len(self._periods):
            raise ValueError("regime length must match number of periods")
        Y = self._Y
        n = len(Y)

        # Backward iteration: build outcome regressions for the chosen regime.
        # Q_T = Y; Q_t = E[Q_{t+1} | history_t, A_t = a_t].
        Q = Y.copy()
        cumulative = None
        # First build cumulative-history features as we'd see them at each t.
        cum_features = []
        cum = None
        for t, p in enumerate(self._periods):
            X_t = p.L if cum is None else np.hstack([cum, p.L])
            cum_features.append(X_t)
            cum = np.hstack([X_t, p.A.reshape(-1, 1)])

        # Backward sweep.
        for t in reversed(range(len(self._periods))):
            p = self._periods[t]
            X_t = cum_features[t]
            Xt_full = np.hstack([X_t, p.A.reshape(-1, 1)])
            reg = LinearRegression().fit(Xt_full, Q)
            # Counterfactual: replace A_t with target regime value.
            Xt_cf = np.hstack([X_t,
                               np.full((n, 1), float(regime[t]))])
            Q = reg.predict(Xt_cf)

        # Now Q is the AIPW backward-regression baseline.
        # Apply IPW correction across periods that align with the regime.
        weights = np.ones(n)
        indicator = np.ones(n, dtype=bool)
        for t, ((clf, X_t), p, a) in enumerate(zip(
                self._propensities, self._periods, regime)):
            e_t = np.clip(clf.predict_proba(X_t)[:, 1], self.trim, 1 - self.trim)
            indicator &= np.isclose(p.A, float(a))
            prob = e_t if float(a) == 1 else 1 - e_t
            weights *= 1.0 / prob

        cap = np.quantile(weights[indicator], 0.99) if indicator.any() else 1.0
        weights = np.clip(weights, 0, cap)

        # Doubly-robust score.
        if indicator.any():
            ipw_term = float((weights[indicator]
                              * (Y[indicator] - Q[indicator])).sum()
                             / max(indicator.sum(), 1))
        else:
            ipw_term = 0.0
        return float(Q.mean() + ipw_term)
