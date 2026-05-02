"""Generalized-propensity overlap diagnostics for continuous treatments.

Wager (2025) audit Gap 1. For continuous treatments the binary-propensity
positivity assumption is replaced by a *conditional density* ``f(W|X)``: a
scenario is in-domain if the requested ``(x_baseline, w_target)`` falls in
the high-density region of the joint training distribution.

Two estimators are supported:
    1. Kernel conditional density on residualized W (Hayfield-Racine
       bandwidth). Robust under near-Gaussian residuals.
    2. Quantile-bin fallback: bin W into deciles, fit a multinomial logistic
       regression of bin label on X, and read off the predicted probability
       of the target bin.

Both produce a ``percentile`` (0-100) and a ``flag`` in
{HIGH, LOW, EXTRAPOLATED}.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

from ._audit import mark_addressed

GAP_1_IMPLEMENTED = True

_FLAG_HIGH = "HIGH"
_FLAG_LOW = "LOW"
_FLAG_EXTRAPOLATED = "EXTRAPOLATED"


def _silverman_bandwidth(x: np.ndarray) -> float:
    n = len(x)
    if n <= 1:
        return 1.0
    sd = float(np.std(x, ddof=1))
    iqr = float(np.subtract(*np.percentile(x, [75, 25])))
    spread = min(sd, iqr / 1.349) if iqr > 0 else sd
    if spread <= 0:
        spread = 1.0
    return 1.06 * spread * n ** (-1 / 5)


@dataclass
class OverlapAssessment:
    density: float
    percentile: float
    flag: str

    def as_dict(self) -> Dict[str, float | str]:
        return {"density": self.density,
                "percentile": self.percentile,
                "flag": self.flag}


class GeneralizedPropensityOverlap:
    """Conditional density f(W|X) for continuous-treatment overlap checks."""

    def __init__(self,
                 method: str = "kernel",
                 n_bins: int = 10,
                 low_pct: float = 10.0,
                 extrapolation_pct: float = 1.0):
        if method not in ("kernel", "binning"):
            raise ValueError(f"unknown method: {method}")
        self.method = method
        self.n_bins = n_bins
        self.low_pct = low_pct
        self.extrapolation_pct = extrapolation_pct
        self._scaler: Optional[StandardScaler] = None
        self._lin: Optional[LinearRegression] = None
        self._bw: Optional[float] = None
        self._resid_train: Optional[np.ndarray] = None
        self._density_train: Optional[np.ndarray] = None
        self._bin_edges: Optional[np.ndarray] = None
        self._bin_clf: Optional[LogisticRegression] = None
        self._x_mean: Optional[np.ndarray] = None
        self._x_std: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ fit

    def fit(self, X: np.ndarray, W: np.ndarray) -> "GeneralizedPropensityOverlap":
        X = np.atleast_2d(X).astype(float)
        if X.shape[0] == 1 and X.shape[1] != len(W):
            X = X.T
        W = np.asarray(W, dtype=float).ravel()
        if X.shape[0] != len(W):
            raise ValueError("X and W length mismatch")

        self._scaler = StandardScaler().fit(X)
        Xs = self._scaler.transform(X)
        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0, ddof=1) + 1e-12

        if self.method == "kernel":
            self._lin = LinearRegression().fit(Xs, W)
            resid = W - self._lin.predict(Xs)
            self._resid_train = resid
            self._bw = _silverman_bandwidth(resid)
            # Pre-compute density at every training point for percentile look-up.
            self._density_train = self._kernel_density(resid)
        else:
            quantiles = np.linspace(0, 1, self.n_bins + 1)
            edges = np.quantile(W, quantiles)
            edges[0] -= 1e-9
            edges[-1] += 1e-9
            self._bin_edges = edges
            labels = np.clip(np.searchsorted(edges, W, side="right") - 1,
                             0, self.n_bins - 1)
            self._bin_clf = LogisticRegression(max_iter=500,
                                               multi_class="multinomial")
            self._bin_clf.fit(Xs, labels)
            self._density_train = self._bin_density(Xs, W)

        mark_addressed(1)
        return self

    # --------------------------------------------------------------- assess

    def assess_scenario(self,
                        x_baseline: np.ndarray,
                        delta_w: float) -> Dict[str, float | str]:
        if self._scaler is None or self._density_train is None:
            raise RuntimeError("fit() must be called before assess_scenario()")
        x_baseline = np.atleast_1d(x_baseline).astype(float)
        if x_baseline.ndim == 1:
            x_baseline = x_baseline.reshape(1, -1)

        # Use baseline-mean W as anchor; assess the density at (x, w_baseline+delta).
        w_anchor = float(self._mean_w_at(x_baseline))
        w_target = w_anchor + float(delta_w)

        if self.method == "kernel":
            xs = self._scaler.transform(x_baseline)
            pred = self._lin.predict(xs)[0]
            resid = w_target - pred
            density = float(self._kernel_density(np.array([resid]))[0])
        else:
            xs = self._scaler.transform(x_baseline)
            density = float(self._bin_density(xs, np.array([w_target]))[0])

        percentile = float((self._density_train <= density).mean() * 100.0)
        flag = self._flag(percentile)
        return OverlapAssessment(density=density,
                                 percentile=percentile,
                                 flag=flag).as_dict()

    # ---------------------------------------------------------- internals

    def _flag(self, percentile: float) -> str:
        if percentile <= self.extrapolation_pct:
            return _FLAG_EXTRAPOLATED
        if percentile <= self.low_pct:
            return _FLAG_LOW
        return _FLAG_HIGH

    def _kernel_density(self, residuals: np.ndarray) -> np.ndarray:
        if self._resid_train is None or self._bw is None:
            raise RuntimeError("kernel state not initialised")
        bw = self._bw
        train = self._resid_train
        diffs = (residuals[:, None] - train[None, :]) / bw
        kvals = np.exp(-0.5 * diffs ** 2) / np.sqrt(2 * np.pi)
        return kvals.mean(axis=1) / bw

    def _bin_density(self, Xs: np.ndarray, W: np.ndarray) -> np.ndarray:
        if self._bin_clf is None or self._bin_edges is None:
            raise RuntimeError("binning state not initialised")
        bins = np.clip(
            np.searchsorted(self._bin_edges, W, side="right") - 1,
            0, self.n_bins - 1,
        )
        proba = self._bin_clf.predict_proba(Xs)
        widths = np.diff(self._bin_edges)
        return np.array([proba[i, b] / max(widths[b], 1e-12)
                         for i, b in enumerate(bins)])

    def _mean_w_at(self, x: np.ndarray) -> float:
        if self.method == "kernel":
            xs = self._scaler.transform(x)
            return float(self._lin.predict(xs)[0])
        # Binning: weighted average of bin midpoints.
        xs = self._scaler.transform(x)
        proba = self._bin_clf.predict_proba(xs)[0]
        midpoints = 0.5 * (self._bin_edges[:-1] + self._bin_edges[1:])
        return float(np.dot(proba, midpoints))


def assess_dataframe(model: GeneralizedPropensityOverlap,
                     X_baseline: np.ndarray,
                     delta_w: np.ndarray) -> Dict[str, np.ndarray]:
    """Vectorised wrapper for scenario tables."""
    X_baseline = np.atleast_2d(X_baseline).astype(float)
    delta_w = np.asarray(delta_w, dtype=float).ravel()
    if X_baseline.shape[0] != len(delta_w):
        if X_baseline.shape[0] == 1:
            X_baseline = np.repeat(X_baseline, len(delta_w), axis=0)
        else:
            raise ValueError("row mismatch between X_baseline and delta_w")
    out_d, out_p, out_f = [], [], []
    for row, dw in zip(X_baseline, delta_w):
        a = model.assess_scenario(row, dw)
        out_d.append(a["density"])
        out_p.append(a["percentile"])
        out_f.append(a["flag"])
    return {"density": np.array(out_d),
            "percentile": np.array(out_p),
            "flag": np.array(out_f, dtype=object)}
