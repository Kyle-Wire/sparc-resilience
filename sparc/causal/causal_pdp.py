"""Phase C-2 — Causal PDP and saturation curves with NUTS credible bands.

Builds dose-response (PDP / ICE) curves whose y-axis is a *causal*
effect estimate τ(t) = Y(do(T=t)) − Y(do(T=t₀)) rather than a
correlational marginal.

Dose-response construction (``method = "tau_by_level_integral"``)
------------------------------------------------------------------
Each cell i carries a CATE τ_i — the local marginal effect of the
treatment at its *own* treatment level T_i.  The population marginal
effect at dose s is estimated by Nadaraya–Watson kernel regression of
τ_i on T_i::

    m̂(s) = Ê[τ | T ≈ s] = Σ_i K_h(T_i − s) τ_i / Σ_i K_h(T_i − s)

(Gaussian kernel, Silverman bandwidth on T), and the response is the
integral of the marginal effect from the median dose t₀::

    R(d) = ∫_{t₀}^{d} m̂(s) ds        (trapezoid rule; negative for d < t₀)

If the marginal effect declines with dose, R(d) is concave and the
saturation knee can be detected.  (The previous construction,
mean(τ)·(d − t₀), was linear by construction and could never show one.)

Two paths are supported:

1. **Bayesian (preferred)** — when a ``BayesianSpatialCATE`` estimator
   has been fit, full posterior samples ``τ_post[d, i]`` per draw d and
   cell i are available; R(d) is computed per posterior draw, giving
   posterior credible bands.

2. **Frequentist fallback** — when only a ``SpatialCATEEstimator`` is
   available, R(d) is computed from the CATE point estimates, and the
   band from the same integral applied to the per-cell interval bounds.

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
    method: str = "tau_by_level_integral"          # dose-response construction

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
            "method": self.method,
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


def _silverman_bandwidth(t: np.ndarray) -> float:
    """Silverman's rule-of-thumb bandwidth for a 1-D sample."""
    t = np.asarray(t, dtype=np.float64)
    t = t[np.isfinite(t)]
    n = t.size
    if n < 2:
        return 1.0
    sd = float(np.std(t, ddof=1))
    q75, q25 = np.percentile(t, [75, 25])
    iqr = float(q75 - q25)
    spread = min(sd, iqr / 1.34) if iqr > 0 else sd
    if not np.isfinite(spread) or spread <= 0:
        spread = sd if (np.isfinite(sd) and sd > 0) else 1.0
    return float(0.9 * spread * n ** (-0.2))


def _kernel_weights(t: np.ndarray, grid: np.ndarray, bandwidth: float) -> np.ndarray:
    """Column-normalised Gaussian kernel weights, shape (N, K).

    ``W[:, k]`` sums to 1, so ``tau @ W`` is the Nadaraya–Watson estimate
    Ê[τ | T ≈ grid[k]].  Grid points with no kernel mass fall back to the
    nearest observations (weights computed with log-sum-exp stabilisation).
    """
    z = (np.asarray(t, dtype=np.float64)[:, None] - grid[None, :]) / bandwidth
    logw = -0.5 * z * z
    logw -= logw.max(axis=0, keepdims=True)
    w = np.exp(logw)
    return w / w.sum(axis=0, keepdims=True)


def _integrated_response(
    m_aug: np.ndarray,
    aug_grid: np.ndarray,
    t0_index: int,
) -> np.ndarray:
    """Cumulative trapezoid integral of ``m_aug`` along the last axis,
    re-anchored so that the value at ``aug_grid[t0_index]`` is zero."""
    dx = np.diff(aug_grid)
    seg = 0.5 * (m_aug[..., 1:] + m_aug[..., :-1]) * dx
    cum = np.concatenate(
        [np.zeros(m_aug.shape[:-1] + (1,)), np.cumsum(seg, axis=-1)], axis=-1,
    )
    return cum - cum[..., t0_index:t0_index + 1]


def _augmented_grid(dose_grid: np.ndarray, t0: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Dose grid with t₀ inserted.  Returns (aug_grid, dose_idx, t0_idx)."""
    aug = np.unique(np.concatenate([dose_grid, [t0]]))
    dose_idx = np.searchsorted(aug, dose_grid)
    t0_idx = int(np.searchsorted(aug, t0))
    return aug, dose_idx, t0_idx


def tau_by_level_response(
    tau: np.ndarray,
    treatment_values: np.ndarray,
    dose_grid: np.ndarray,
    t0: float,
    bandwidth: Optional[float] = None,
) -> tuple[np.ndarray, float]:
    """R(d) = ∫_{t₀}^{d} Ê[τ | T≈s] ds evaluated on ``dose_grid``.

    ``tau`` may be (N,) or (D, N) (one row per posterior draw); the result
    is (K,) or (D, K) respectively.  Returns (response, bandwidth).
    """
    t = np.asarray(treatment_values, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    finite = np.isfinite(t) & np.all(np.isfinite(np.atleast_2d(tau)), axis=0)
    t = t[finite]
    tau = tau[..., finite]
    if bandwidth is None:
        bandwidth = _silverman_bandwidth(t)
    aug, dose_idx, t0_idx = _augmented_grid(np.asarray(dose_grid, dtype=np.float64), float(t0))
    if t.size == 0:
        shape = tau.shape[:-1] + (len(dose_grid),)
        return np.zeros(shape), float(bandwidth)
    W = _kernel_weights(t, aug, bandwidth)          # (N, K_aug)
    m_aug = tau @ W                                   # (K_aug,) or (D, K_aug)
    resp_aug = _integrated_response(m_aug, aug, t0_idx)
    return resp_aug[..., dose_idx], float(bandwidth)


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
    delta_grid = dose_grid - t_baseline                                  # (K,)

    # Population causal response per posterior draw:
    #     R[d, k] = ∫_{t0}^{dose_k} Ê[τ_post[d, ·] | T ≈ s] ds
    # (kernel regression of τ on T, integrated by the trapezoid rule)
    tau_post = np.asarray(tau_post, dtype=np.float64)
    pop_response_post, bandwidth = tau_by_level_response(
        tau_post, t, dose_grid, t_baseline,
    )                                                                    # (D, K)
    response_mean = pop_response_post.mean(axis=0)
    response_lo, response_hi = _hdi(pop_response_post, hdi_prob, axis=0)

    per_cell_mean = per_cell_lo = per_cell_hi = None
    if keep_per_cell:
        # Per-cell curve = population curve + the cell's CATE deviation from
        # the dose-level mean at its own dose, applied linearly:
        #     R_i(d) = R(d) + (τ_i − m̂(T_i)) · (d − t0)
        # (reduces to τ_i · (d − t0) when m̂ is flat).
        aug, _, _ = _augmented_grid(dose_grid, t_baseline)
        W = _kernel_weights(t, aug, bandwidth)                           # (N, Ka)
        m_aug = tau_post @ W                                             # (D, Ka)
        m_at_cell = np.stack(
            [np.interp(t, aug, m_aug[d]) for d in range(n_draws)], axis=0,
        )                                                                # (D, N)
        resid = tau_post - m_at_cell                                     # (D, N)
        # Memory: D·N·K floats — only when explicitly requested.
        full_response = (pop_response_post[:, None, :]
                         + resid[:, :, None] * delta_grid[None, None, :])  # (D, N, K)
        per_cell_mean = full_response.mean(axis=0)                       # (N, K)
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
        "method": "tau_by_level_integral",
        "kernel_bandwidth": float(bandwidth),
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

    Response R(d) = ∫_{t0}^{d} Ê[τ | T≈s] ds from the per-cell CATE point
    estimates (see module docstring).  The band applies the same integral
    to the per-cell lower/upper bounds of the estimator's 95% confidence
    interval (``cate_intervals``).
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

    n = t.shape[0]
    ci_lo = np.broadcast_to(ci_lo, (n,)) if ci_lo.ndim == 0 else ci_lo
    ci_hi = np.broadcast_to(ci_hi, (n,)) if ci_hi.ndim == 0 else ci_hi
    stacked = np.stack([cate, ci_lo, ci_hi], axis=0)                    # (3, N)
    responses, bandwidth = tau_by_level_response(
        stacked, t, dose_grid, t_baseline,
    )                                                                    # (3, K)
    response_mean = responses[0]
    response_lo = np.minimum(responses[1], responses[2])
    response_hi = np.maximum(responses[1], responses[2])

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
        "method": "tau_by_level_integral",
        "kernel_bandwidth": float(bandwidth),
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
