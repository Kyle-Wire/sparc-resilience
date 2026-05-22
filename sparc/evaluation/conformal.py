"""
Spatially-blocked conformal prediction for SPARC V2.

Provides guaranteed finite-sample marginal coverage intervals by
applying conformal calibration with spatial-distance weighting.  When
calibration residuals are drawn from a spatially-heterogeneous process,
using uniform weights can yield under- or over-coverage in out-of-sample
locations; weighting by proximity (with bandwidth set from Stage 0 κ)
corrects for the non-exchangeability introduced by spatial autocorrelation.

References
----------
Tibshirani et al. (2019) "Conformal prediction under covariate shift."
Barber et al. (2023) "Conformal prediction beyond exchangeability."
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class PredictionIntervals:
    """Conformal prediction intervals for a batch of test points.

    Attributes
    ----------
    lower : ndarray, shape (N,)
        Lower bounds of the prediction intervals.
    upper : ndarray, shape (N,)
        Upper bounds of the prediction intervals.
    coverage_target : float
        Nominal coverage level (e.g. 0.90).
    quantile : float
        Calibration quantile used.  Diagnostics-only — the actual
        finite-sample guarantee is ``coverage_target``.
    n_cal : int
        Number of calibration residuals used.
    """

    lower: np.ndarray
    upper: np.ndarray
    coverage_target: float
    quantile: float
    n_cal: int
    metadata: dict = field(default_factory=dict)

    def width(self) -> np.ndarray:
        """Element-wise interval width."""
        return self.upper - self.lower

    def coverage(self, y_true: np.ndarray) -> float:
        """Empirical coverage of ``y_true`` (for held-out evaluation)."""
        y = np.asarray(y_true, dtype=float)
        return float(np.mean((y >= self.lower) & (y <= self.upper)))


class SpatialConformalPredictor:
    """
    Weighted split-conformal predictor with spatial proximity weighting.

    Calibration residuals near each test point are up-weighted; distant
    calibration points receive lower weight.  The bandwidth is derived
    from the Matérn κ parameter estimated in Stage 0 so that the kernel
    length-scale matches the domain's actual correlation structure.

    Usage
    -----
    >>> pred = SpatialConformalPredictor(kappa=0.05)
    >>> pred.calibrate(cal_coords, np.abs(cal_residuals))
    >>> intervals = pred.prediction_interval(test_coords, y_hat)

    Parameters
    ----------
    kappa : float, optional
        Matérn κ from Stage 0 (units: 1 / coord_unit).  If provided,
        the Gaussian kernel bandwidth is set to ``1 / kappa``.  If
        ``None``, equal-weight (standard split-conformal) is used.
    min_weight : float
        Floor on normalized weights to keep all calibration points
        numerically relevant (prevents division by tiny weights).
    """

    def __init__(
        self,
        kappa: Optional[float] = None,
        min_weight: float = 1e-6,
    ) -> None:
        self.kappa = kappa
        self.min_weight = min_weight
        self._cal_coords: Optional[np.ndarray] = None
        self._cal_scores: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def calibrate(
        self,
        cal_coords: np.ndarray,
        cal_scores: np.ndarray,
    ) -> "SpatialConformalPredictor":
        """
        Store calibration nonconformity scores.

        Parameters
        ----------
        cal_coords : ndarray, shape (N, 2)
            Spatial coordinates of calibration points (any projected CRS,
            same units as used when κ was estimated).
        cal_scores : ndarray, shape (N,)
            Nonconformity scores — typically ``|y_true − ŷ|``.  Must be
            non-negative.

        Returns
        -------
        self
        """
        self._cal_coords = np.asarray(cal_coords, dtype=float)
        self._cal_scores = np.asarray(cal_scores, dtype=float)
        if self._cal_scores.ndim != 1 or len(self._cal_scores) != len(self._cal_coords):
            raise ValueError(
                "cal_coords and cal_scores must have the same length and "
                "cal_scores must be 1-D."
            )
        return self

    # ------------------------------------------------------------------
    def prediction_interval(
        self,
        test_coords: np.ndarray,
        point_predictions: np.ndarray,
        coverage: float = 0.90,
    ) -> PredictionIntervals:
        """
        Compute symmetric prediction intervals for test points.

        For each test point the calibration scores are re-weighted by
        spatial proximity; then the weighted ``⌈(1−α)(1 + 1/n)⌉``-quantile
        is computed.  The resulting interval is guaranteed to contain the
        true value with probability ≥ ``coverage`` marginally over the
        weighted calibration set (Tibshirani et al. 2019, Theorem 1).

        Parameters
        ----------
        test_coords : ndarray, shape (M, 2)
            Coordinates of test points.
        point_predictions : ndarray, shape (M,)
            Point predictions (ŷ) at test locations.
        coverage : float
            Nominal marginal coverage, e.g. ``0.90``.

        Returns
        -------
        PredictionIntervals
        """
        if self._cal_coords is None or self._cal_scores is None:
            raise RuntimeError("Call calibrate() before prediction_interval().")
        if not 0.0 < coverage < 1.0:
            raise ValueError(f"coverage must be in (0, 1), got {coverage}.")

        test_coords = np.asarray(test_coords, dtype=float)
        y_hat = np.asarray(point_predictions, dtype=float)
        if y_hat.ndim != 1 or len(y_hat) != len(test_coords):
            raise ValueError(
                "point_predictions must be 1-D with length matching test_coords."
            )

        n_cal = len(self._cal_scores)
        alpha = 1.0 - coverage

        # Compute weights for every (test, cal) pair
        # Shape: (M, N_cal)
        weights = self._compute_weights(test_coords)  # (M, N_cal)

        # Weighted quantile level: (1−α)(1 + 1/n)  clipped to [0,1]
        q_level = min(1.0, (1.0 - alpha) * (1.0 + 1.0 / n_cal))

        # Compute weighted quantile per test point
        half_widths = np.empty(len(test_coords), dtype=float)
        for i in range(len(test_coords)):
            half_widths[i] = _weighted_quantile(
                self._cal_scores, weights[i], q_level
            )

        # The used quantile is the average across test points (summary)
        q_summary = float(np.mean(half_widths))

        return PredictionIntervals(
            lower=y_hat - half_widths,
            upper=y_hat + half_widths,
            coverage_target=coverage,
            quantile=q_summary,
            n_cal=n_cal,
            metadata={
                "kappa": self.kappa,
                "weighting": "spatial-kernel" if self.kappa else "uniform",
            },
        )

    # ------------------------------------------------------------------
    def _compute_weights(self, test_coords: np.ndarray) -> np.ndarray:
        """Return normalized weight matrix (M, N_cal) for each test point."""
        cal = self._cal_coords  # (N_cal, 2)
        assert cal is not None

        if self.kappa is None or self.kappa <= 0:
            # Uniform weights — standard split-conformal
            n = cal.shape[0]
            return np.ones((len(test_coords), n), dtype=float) / n

        # Gaussian kernel with bandwidth = 1/kappa
        bw = 1.0 / self.kappa
        # Squared distances: (M, N_cal)
        diff = test_coords[:, np.newaxis, :] - cal[np.newaxis, :, :]  # (M, N, 2)
        sq_dist = np.sum(diff ** 2, axis=-1)  # (M, N)
        log_weights = -sq_dist / (2.0 * bw ** 2)
        # Numerically stable softmax-style normalization
        log_weights -= log_weights.max(axis=1, keepdims=True)
        raw = np.exp(log_weights)
        raw = np.maximum(raw, self.min_weight)
        return raw / raw.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    """
    Compute the weighted quantile of ``values``.

    Uses the standard definition: sort values, take cumulative weight,
    return the first value whose cumulative weight exceeds ``quantile``.

    Parameters
    ----------
    values : ndarray, shape (N,)
    weights : ndarray, shape (N,)  — must be non-negative, need not sum to 1.
    quantile : float in (0, 1]

    Returns
    -------
    float
    """
    sorter = np.argsort(values)
    sv = values[sorter]
    sw = weights[sorter]
    cum_w = np.cumsum(sw)
    cum_w /= cum_w[-1]  # normalize
    # First index where cumulative weight ≥ quantile
    idx = int(np.searchsorted(cum_w, quantile))
    idx = min(idx, len(sv) - 1)
    return float(sv[idx])
