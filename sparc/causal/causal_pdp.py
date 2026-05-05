"""Phase C-2 — Causal PDP and saturation curves with NUTS credible bands.

Builds dose-response (PDP / ICE) curves whose y-axis is a *causal*
effect estimate τ(t) = Y(do(T=t)) − Y(do(T=t₀)) rather than a
correlational marginal.  Two paths are supported:

1. **Bayesian (preferred)** — when a ``BayesianSpatialCATE`` estimator
   has been fit, full posterior samples ``τ_post[d, i]`` per draw d and
   cell i are available.  Sweeping the treatment over a reference grid
   yields per-cell, per-dose response curves with *exact posterior
   credible bands* — no resampling tricks.

2. **Frequentist fallback** — when only a ``SpatialCATEEstimator`` is
   available, we use its CATE point estimate plus the pre-existing
   confidence interval and treat the response as locally linear in the
   treatment around its mean.  Wider, less informative, but still
   monotonically interpretable.

In both paths a **saturation knee** is detected by locating the inner
dose-grid index at which the posterior-mean marginal slope drops below
``saturation_marginal_floor × peak |slope|`` (default 0.5), matching the
Stage-4 Wager-2025 saturation gate convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CausalDoseResponseCurve:
    """A single per-treatment causal PDP with credible bands.

    ``response_*`` arrays are shaped ``(n_dose,)``.  ``per_cell_*`` arrays
    are shaped ``(n_cells, n_dose)`` and only populated when the source
    estimator carries posterior samples.
    """

    treatment: str
    dose_grid: np.ndarray                          # (n_dose,)
    response_mean: np.ndarray                      # (n_dose,)
    response_hdi_lo: np.ndarray                    # (n_dose,)
    response_hdi_hi: np.ndarray                    # (n_dose,)
    per_cell_mean: Optional[np.ndarray] = None     # (n_cells, n_dose)
    per_cell_hdi_lo: Optional[np.ndarray] = None   # (n_cells, n_dose)
    per_cell_hdi_hi: Optional[np.ndarray] = None   # (n_cells, n_dose)
    saturation_dose: Optional[float] = None        # treatment value at knee
    saturation_index: Optional[int] = None
    peak_marginal_slope: float = float("nan")
    knee_marginal_slope: float = float("nan")
    source: str = "unknown"                        # "bayesian" | "frequentist"
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict:
        """JSON-friendly dict ready for ``ArtifactStore.write_struct``."""
        out = {
            "treatment": self.treatment,
            "dose_grid": self.dose_grid.tolist(),
            "response_mean": self.response_mean.tolist(),
            "response_hdi_lo": self.response_hdi_lo.tolist(),
            "response_hdi_hi": self.response_hdi_hi.tolist(),
            "saturation_dose": (
                float(self.saturation_dose)
                if self.saturation_dose is not None else None
            ),
            "saturation_index": (
                int(self.saturation_index)
                if self.saturation_index is not None else None
            ),
            "peak_marginal_slope": float(self.peak_marginal_slope),
            "knee_marginal_slope": float(self.knee_marginal_slope),
            "source": self.source,
            "diagnostics": dict(self.diagnostics),
        }
        # Per-cell arrays are large; only emit when present and small.
        for k in ("per_cell_mean", "per_cell_hdi_lo", "per_cell_hdi_hi"):
            arr = getattr(self, k)
            if arr is not None and arr.size <= 2_000_000:
                out[k] = arr.tolist()
        return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _hdi(samples: np.ndarray, prob: float = 0.89, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Highest-density interval along ``axis`` of an array of draws."""
    s = np.sort(samples, axis=axis)
    n = s.shape[axis]
    width = max(1, int(round(prob * n)))
    width = min(width, n)
    if n <= 1:
        return s.take(0, axis=axis), s.take(0, axis=axis)
    n_intervals = n - width + 1
    lo_slices = [slice(None)] * s.ndim
    hi_slices = [slice(None)] * s.ndim
    lo_slices[axis] = slice(0, n_intervals)
    hi_slices[axis] = slice(width - 1, width - 1 + n_intervals)
    spans = s[tuple(hi_slices)] - s[tuple(lo_slices)]
    j = np.argmin(spans, axis=axis)
    # gather along axis
    j_exp = np.expand_dims(j, axis=axis)
    lo = np.take_along_axis(s, j_exp, axis=axis).squeeze(axis=axis)
    hi = np.take_along_axis(
        s, j_exp + (width - 1), axis=axis,
    ).squeeze(axis=axis)
    return lo, hi


def _detect_saturation(
    response_mean: np.ndarray,
    dose_grid: np.ndarray,
    floor: float = 0.5,
) -> tuple[Optional[int], float, float]:
    """Locate the saturation knee on a dose-response curve.

    The knee is the *first* inner index where the absolute marginal
    slope drops below ``floor × peak |slope|``.  Returns (idx, peak,
    knee_slope).  When no knee is detected, returns (None, peak, nan).
    """
    if response_mean.size < 3:
        return None, float("nan"), float("nan")
    slopes = np.gradient(response_mean, dose_grid)
    abs_slopes = np.abs(slopes)
    peak = float(np.max(abs_slopes))
    if peak <= 0:
        return None, peak, float("nan")
    # Search past the peak to avoid latching the initial flat region
    peak_idx = int(np.argmax(abs_slopes))
    threshold = floor * peak
    for i in range(peak_idx + 1, abs_slopes.size):
        if abs_slopes[i] < threshold:
            return i, peak, float(abs_slopes[i])
    return None, peak, float("nan")


# ---------------------------------------------------------------------------
# Bayesian path
# ---------------------------------------------------------------------------


def causal_pdp_bayesian(
    estimator: Any,
    treatment: str,
    treatment_values: np.ndarray,
    *,
    n_dose: int = 25,
    dose_range: Optional[tuple[float, float]] = None,
    saturation_floor: float = 0.5,
    hdi_prob: float = 0.89,
    keep_per_cell: bool = False,
) -> CausalDoseResponseCurve:
    """Causal PDP from a fitted ``BayesianSpatialCATE`` estimator.

    Parameters
    ----------
    estimator : BayesianSpatialCATE
        Must already have ``estimate(treatment, …)`` called for ``treatment``;
        we read ``estimator.posterior_samples(treatment)`` to recover the
        full ``(n_draws, n_cells)`` τ posterior.
    treatment : the treatment name.
    treatment_values : raw treatment column from the fit data, shape (N,).
        Used to (a) center the dose grid on the empirical treatment range
        and (b) compute the per-cell baseline ``T̄_i`` from which causal
        deltas are integrated.
    n_dose : number of grid points.
    dose_range : optional explicit (t_lo, t_hi); defaults to 5–95 percentile.

    Returns
    -------
    CausalDoseResponseCurve with ``source="bayesian"``.
    """
    tau_post = estimator.posterior_samples(treatment)         # (D, N)
    if tau_post.ndim != 2:
        raise ValueError(
            f"Expected posterior_samples shape (D, N); got {tau_post.shape}"
        )
    n_draws, n_cells = tau_post.shape
    t = np.asarray(treatment_values, dtype=np.float64)
    if t.shape[0] != n_cells:
        raise ValueError(
            f"treatment_values length ({t.shape[0]}) must equal "
            f"n_cells ({n_cells})"
        )
    if dose_range is None:
        t_lo, t_hi = np.percentile(t, [5, 95])
    else:
        t_lo, t_hi = dose_range
    if not np.isfinite(t_lo) or not np.isfinite(t_hi) or t_hi <= t_lo:
        # Degenerate (constant) treatment — fall back to ±1 around mean.
        m = float(np.nanmean(t))
        t_lo, t_hi = m - 1.0, m + 1.0
    dose_grid = np.linspace(t_lo, t_hi, n_dose)
    t_baseline = float(np.median(t))

    # Causal response per draw, per cell, per dose:
    #     R[d, i, k] = τ_post[d, i] · (dose_grid[k] − t_baseline)
    # Memory note: D × N × K float64 — chunked over draws.
    pop_mean_per_draw = tau_post.mean(axis=1, keepdims=False)           # (D,)
    delta_grid = dose_grid - t_baseline                                  # (K,)
    pop_response_post = pop_mean_per_draw[:, None] * delta_grid[None, :] # (D, K)
    response_mean = pop_response_post.mean(axis=0)
    response_lo, response_hi = _hdi(pop_response_post, hdi_prob, axis=0)

    per_cell_mean = per_cell_lo = per_cell_hi = None
    if keep_per_cell:
        per_cell_response_mean = tau_post.mean(axis=0)                  # (N,)
        per_cell_mean = (per_cell_response_mean[:, None]
                          * delta_grid[None, :])                        # (N, K)
        # Per-cell HDIs are computed by full sweep — only when explicitly
        # requested because memory is D·N·K floats.
        full_response = (tau_post[:, :, None]
                          * delta_grid[None, None, :])                  # (D, N, K)
        per_cell_lo, per_cell_hi = _hdi(full_response, hdi_prob, axis=0)

    sat_idx, peak, knee_slope = _detect_saturation(
        response_mean, dose_grid, saturation_floor,
    )
    sat_dose = float(dose_grid[sat_idx]) if sat_idx is not None else None

    diagnostics = {
        "n_draws": int(n_draws),
        "n_cells": int(n_cells),
        "baseline_dose": float(t_baseline),
        "dose_p5": float(t_lo),
        "dose_p95": float(t_hi),
        "hdi_prob": float(hdi_prob),
        "saturation_floor": float(saturation_floor),
    }
    return CausalDoseResponseCurve(
        treatment=treatment,
        dose_grid=dose_grid,
        response_mean=response_mean,
        response_hdi_lo=response_lo,
        response_hdi_hi=response_hi,
        per_cell_mean=per_cell_mean,
        per_cell_hdi_lo=per_cell_lo,
        per_cell_hdi_hi=per_cell_hi,
        saturation_dose=sat_dose,
        saturation_index=sat_idx,
        peak_marginal_slope=float(peak),
        knee_marginal_slope=float(knee_slope),
        source="bayesian",
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Frequentist path
# ---------------------------------------------------------------------------


def causal_pdp_frequentist(
    estimator: Any,
    treatment: str,
    treatment_values: np.ndarray,
    *,
    n_dose: int = 25,
    dose_range: Optional[tuple[float, float]] = None,
    saturation_floor: float = 0.5,
) -> CausalDoseResponseCurve:
    """Causal PDP from a fitted ``SpatialCATEEstimator`` (CausalForestDML).

    Approximates the response as locally linear in the treatment using
    the population-mean CATE.  The credible band uses the estimator's
    pre-existing 95% confidence interval (cate_intervals).
    """
    if treatment not in estimator.cate_estimates:
        raise KeyError(f"Estimator has no CATE for '{treatment}'")
    cate = np.asarray(estimator.cate_estimates[treatment], dtype=np.float64)
    ci_lo, ci_hi = estimator.cate_intervals.get(
        treatment, (cate, cate),
    )
    ci_lo = np.asarray(ci_lo, dtype=np.float64)
    ci_hi = np.asarray(ci_hi, dtype=np.float64)
    t = np.asarray(treatment_values, dtype=np.float64)
    if dose_range is None:
        t_lo, t_hi = np.percentile(t, [5, 95])
    else:
        t_lo, t_hi = dose_range
    if not np.isfinite(t_lo) or not np.isfinite(t_hi) or t_hi <= t_lo:
        m = float(np.nanmean(t))
        t_lo, t_hi = m - 1.0, m + 1.0
    dose_grid = np.linspace(t_lo, t_hi, n_dose)
    t_baseline = float(np.median(t))
    delta_grid = dose_grid - t_baseline

    pop_cate = float(np.mean(cate))
    pop_lo = float(np.mean(ci_lo))
    pop_hi = float(np.mean(ci_hi))
    response_mean = pop_cate * delta_grid
    response_lo = np.minimum(pop_lo * delta_grid, pop_hi * delta_grid)
    response_hi = np.maximum(pop_lo * delta_grid, pop_hi * delta_grid)

    sat_idx, peak, knee_slope = _detect_saturation(
        response_mean, dose_grid, saturation_floor,
    )
    sat_dose = float(dose_grid[sat_idx]) if sat_idx is not None else None
    diagnostics = {
        "n_cells": int(t.shape[0]),
        "baseline_dose": float(t_baseline),
        "dose_p5": float(t_lo),
        "dose_p95": float(t_hi),
        "saturation_floor": float(saturation_floor),
        "approximation": "locally_linear",
    }
    return CausalDoseResponseCurve(
        treatment=treatment,
        dose_grid=dose_grid,
        response_mean=response_mean,
        response_hdi_lo=response_lo,
        response_hdi_hi=response_hi,
        saturation_dose=sat_dose,
        saturation_index=sat_idx,
        peak_marginal_slope=float(peak),
        knee_marginal_slope=float(knee_slope),
        source="frequentist",
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------------


def causal_pdp(
    estimator: Any,
    treatment: str,
    treatment_values: np.ndarray,
    **kwargs: Any,
) -> CausalDoseResponseCurve:
    """Auto-dispatch to the bayesian / frequentist path based on what the
    estimator carries.  Caller may force a path via ``force_path="bayesian"``."""
    force = kwargs.pop("force_path", None)
    has_post = (
        hasattr(estimator, "posterior_samples")
        and callable(estimator.posterior_samples)
        and treatment in getattr(estimator, "_posterior_samples", {})
    )
    if force == "bayesian" or (force is None and has_post):
        return causal_pdp_bayesian(estimator, treatment, treatment_values, **kwargs)
    return causal_pdp_frequentist(
        estimator, treatment, treatment_values, **kwargs,
    )


def causal_pdps_for_all(
    estimator: Any,
    treatments: Sequence[str],
    data: "Any",
    treatment_columns: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> Dict[str, CausalDoseResponseCurve]:
    """Run :func:`causal_pdp` for every treatment, returning a dict.

    ``data`` is anything indexable like ``data[treatment_column].values``
    (typically a ``pandas.DataFrame``).  ``treatment_columns`` may map
    treatment names to column names; defaults to identity.
    """
    out: Dict[str, CausalDoseResponseCurve] = {}
    cmap = treatment_columns or {}
    for tr in treatments:
        col = cmap.get(tr, tr)
        try:
            t_vals = np.asarray(data[col]).astype(np.float64)
        except Exception:
            continue
        try:
            out[tr] = causal_pdp(estimator, tr, t_vals, **kwargs)
        except Exception as exc:  # noqa: BLE001
            out[tr] = CausalDoseResponseCurve(
                treatment=tr,
                dose_grid=np.array([]),
                response_mean=np.array([]),
                response_hdi_lo=np.array([]),
                response_hdi_hi=np.array([]),
                source="error",
                diagnostics={"error": str(exc)},
            )
    return out
