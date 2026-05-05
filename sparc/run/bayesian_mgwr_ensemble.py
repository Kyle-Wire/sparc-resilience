"""Phase 7 item 3 — Bayesian-MGWR posterior ensemble.

Given a fitted classical MGWR (`GWRModel`), an attached `KernelField`,
and the Phase-1 ``correlogram_matern_fit`` artifact (which carries
posterior κ samples per variable), this module:

1. Draws ``n_ensemble`` joint κ samples (one per predictor, drawn
   independently from the per-variable κ posterior with replacement).
2. For each draw, refits a fresh `GWRModel` with a κ-perturbed
   `KernelField` (per-predictor ``bandwidth_to_outcome = 1/κ_draw``).
3. Stacks the per-fit ``coefficients_`` matrices into
   ``(n_ensemble, n_samples, n_features)``.
4. Reports per-cell posterior mean, std, and 89% HDI for each local β.

Persisted as artifact ``stage=2, gwr_local_beta_posterior``.

Notes
-----
- Only the **bandwidth** (= 1/κ) is perturbed across ensemble members.
  Anisotropy (κ_x, κ_y, θ) and ν are held at their posterior means
  because the Stage-0 anisotropy artifact does not currently carry
  full per-component posterior samples.  Adding those would be a
  natural next step.
- Refits are independent so the ensemble parallelises trivially with
  ``joblib`` when needed; for typical n_ensemble ≤ 16 the serial path
  is fine.
- Designed to be invoked **after** the main MGWR fit so it does not
  interfere with the deterministic coefficient surface used by the
  scenario simulator.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import replace
from typing import Optional

import numpy as np

from sparc.models.gwr import GWRModel
from sparc.models.kernel_field import KernelField, PredictorKernel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Posterior sampling utilities
# ---------------------------------------------------------------------------


def _hdi(samples: np.ndarray, prob: float = 0.89) -> tuple[float, float]:
    """Highest-density interval over a 1D array of posterior draws."""
    s = np.sort(np.asarray(samples).ravel())
    n = s.size
    if n == 0:
        return (float("nan"), float("nan"))
    if n == 1:
        return (float(s[0]), float(s[0]))
    width = max(1, int(round(prob * n)))
    width = min(width, n)
    n_intervals = n - width + 1
    spans = s[width - 1: width - 1 + n_intervals] - s[:n_intervals]
    j = int(np.argmin(spans))
    return (float(s[j]), float(s[j + width - 1]))


def _sample_kappa_per_predictor(
    matern_payload: dict,
    feature_names: list[str],
    n_ensemble: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Draw ``n_ensemble`` κ values per predictor from the Stage-0 posterior.

    Falls back to a Gaussian around the posterior mean (sigma=10% of mean)
    when no explicit posterior samples are stored for a variable.
    """
    per_var_fits = (matern_payload or {}).get("per_variable_fits", {}) or {}
    out: dict[str, np.ndarray] = {}
    for name in feature_names:
        fit = per_var_fits.get(name) or {}
        kappa_block = fit.get("kappa") or {}
        samples = kappa_block.get("samples") or []
        if samples:
            arr = np.asarray(samples, dtype=np.float64)
            arr = arr[arr > 0]
            if arr.size > 0:
                idx = rng.integers(0, arr.size, size=n_ensemble)
                out[name] = arr[idx]
                continue
        # Fallback: jitter the mean
        mean = kappa_block.get("mean")
        if mean is not None and mean > 0:
            out[name] = np.clip(
                rng.normal(loc=float(mean), scale=0.1 * float(mean),
                           size=n_ensemble),
                1e-9, None,
            )
        else:
            out[name] = np.full(n_ensemble, np.nan)
    return out


def _perturb_kernel_field(
    kernel_field: KernelField,
    kappa_draw: dict[str, float],
) -> KernelField:
    """Return a deep-copied KernelField with per-predictor κ replaced by
    the supplied draw and ``bandwidth_to_outcome`` derived as 1/κ."""
    kf = deepcopy(kernel_field)
    new_predictors: list[PredictorKernel] = []
    for p in kf.predictors:
        k = kappa_draw.get(p.name)
        if k is None or not np.isfinite(k) or k <= 0:
            new_predictors.append(p)
            continue
        new_predictors.append(replace(
            p,
            kappa=float(k),
            bandwidth_to_outcome=float(1.0 / k),
        ))
    kf.predictors = new_predictors
    return kf


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_bayesian_mgwr_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    coords: np.ndarray,
    *,
    base_model: GWRModel,
    kernel_field: KernelField,
    matern_payload: dict,
    feature_names: list[str],
    n_ensemble: int = 16,
    seed: int = 42,
    persist: bool = False,
    producer: str = "stage2_bayesian_mgwr_ensemble",
) -> dict:
    """Run the Bayesian-MGWR posterior ensemble.

    Parameters
    ----------
    X, y, coords : the same arrays used for the deterministic GWR fit.
    base_model : the deterministic ``GWRModel`` already fit to (X, y);
        its hyperparameters (alpha, min_points, kernel-family setup)
        are inherited by every ensemble member.
    kernel_field : the deterministic ``KernelField``; deep-copied per
        ensemble member with its ``bandwidth_to_outcome``/κ replaced by
        the sampled draw.
    matern_payload : the ``correlogram_matern_fit`` artifact payload
        (``per_variable_fits[name].kappa.samples`` is consulted).
    feature_names : ordered names for the columns of X (must match the
        kernel field predictor names).
    n_ensemble : number of κ draws.
    persist : when True, writes ``stage=2, gwr_local_beta_posterior``.

    Returns
    -------
    dict with keys:
      - ``per_predictor.{name}.{mean, std, hdi89_lo, hdi89_hi, kappa_draws}``
        each a (n_samples,) array (mean/std/HDI) or (n_ensemble,) for
        kappa_draws
      - ``ensemble_beta``: stacked (n_ensemble, n_samples, n_features)
      - ``summary``: scalar diagnostics (mean across cells of std,
        coverage rate vs. base model, etc.)
    """
    if X.shape[0] != y.shape[0] or coords.shape[0] != y.shape[0]:
        raise ValueError("X, y, coords must share the leading dimension")
    if X.shape[1] != len(feature_names):
        raise ValueError(
            f"X has {X.shape[1]} columns but feature_names lists "
            f"{len(feature_names)}"
        )
    rng = np.random.default_rng(seed)
    kappa_per_pred = _sample_kappa_per_predictor(
        matern_payload, feature_names, n_ensemble, rng,
    )
    n_samples, n_features = X.shape

    ensemble_beta = np.zeros(
        (n_ensemble, n_samples, n_features), dtype=np.float64,
    )

    # Inherit the fitted base model's hyperparameters
    alpha = float(getattr(base_model, "alpha", 0.1))
    min_points = int(getattr(base_model, "min_points", 50))
    use_constrained = bool(getattr(base_model, "use_constrained_regression", True))
    physics_priors = getattr(base_model, "physics_priors", None)
    sign_constraints = getattr(base_model, "_sign_constraints", None)

    for e in range(n_ensemble):
        kappa_draw = {n: kappa_per_pred[n][e] for n in feature_names}
        kf_e = _perturb_kernel_field(kernel_field, kappa_draw)
        m = GWRModel(
            kernel_field=kf_e,
            alpha=alpha,
            min_points=min_points,
            use_constrained_regression=use_constrained,
            physics_priors=physics_priors,
            sign_constraints=sign_constraints,
        )
        m.feature_names_ = list(feature_names)
        try:
            m.fit(X, y, coords=coords)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Bayesian-MGWR ensemble member %d failed: %s", e, exc)
            ensemble_beta[e] = np.nan
            continue
        if m.coefficients_ is None:
            ensemble_beta[e] = np.nan
            continue
        ensemble_beta[e] = m.coefficients_

    # Per-predictor summaries
    per_predictor: dict[str, dict] = {}
    valid_mask = ~np.all(
        np.isnan(ensemble_beta.reshape(n_ensemble, -1)), axis=1,
    )
    n_valid = int(valid_mask.sum())
    valid_eb = ensemble_beta[valid_mask]
    for k, name in enumerate(feature_names):
        col = valid_eb[:, :, k]                                # (n_valid, N)
        mean = np.nanmean(col, axis=0)                         # (N,)
        std = np.nanstd(col, axis=0)
        # Per-cell HDI (vectorise: sort along ensemble axis once)
        sorted_col = np.sort(col, axis=0)                      # (n_valid, N)
        n_e = sorted_col.shape[0]
        prob = 0.89
        width = max(1, int(round(prob * n_e)))
        width = min(width, n_e)
        if n_e >= 2:
            n_intervals = n_e - width + 1
            spans = sorted_col[width - 1: width - 1 + n_intervals, :] \
                - sorted_col[:n_intervals, :]
            j = np.argmin(spans, axis=0)                       # (N,)
            idx = np.arange(n_samples)
            hdi_lo = sorted_col[j, idx]
            hdi_hi = sorted_col[j + width - 1, idx]
        else:
            hdi_lo = mean.copy()
            hdi_hi = mean.copy()
        per_predictor[name] = {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "hdi89_lo": hdi_lo.tolist(),
            "hdi89_hi": hdi_hi.tolist(),
            "kappa_draws": kappa_per_pred[name].tolist(),
        }

    # Coverage diagnostic: fraction of cells whose 89% HDI contains
    # the deterministic base β.
    coverage: dict[str, float] = {}
    if base_model.coefficients_ is not None and base_model.coefficients_.shape == (
        n_samples, n_features,
    ):
        for k, name in enumerate(feature_names):
            base_beta = base_model.coefficients_[:, k]
            lo = np.array(per_predictor[name]["hdi89_lo"])
            hi = np.array(per_predictor[name]["hdi89_hi"])
            covered = (base_beta >= lo) & (base_beta <= hi)
            coverage[name] = float(np.mean(covered))

    payload = {
        "n_ensemble": int(n_ensemble),
        "n_valid_members": n_valid,
        "feature_names": list(feature_names),
        "per_predictor": per_predictor,
        "summary": {
            "mean_posterior_std": {
                name: float(np.nanmean(per_predictor[name]["std"]))
                for name in feature_names
            },
            "coverage_of_base_model": coverage,
        },
    }

    if persist:
        try:
            from sparc.registry.store import get_active_store
            store = get_active_store()
            if store is not None:
                store.write_struct(
                    "2", "gwr_local_beta_posterior", payload,
                    producer=producer,
                    consumers=[
                        "report:/stage2/local_beta_posterior",
                        "audit:/stage2/bayesian_mgwr",
                    ],
                )
        except Exception as e:  # noqa: BLE001
            payload["persist_error"] = str(e)

    return payload
