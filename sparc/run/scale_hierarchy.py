"""
Phase 6 — Scale-Hierarchy / Fractal Signature Diagnostic.

Computes per-variable scale-covariance signatures so the pipeline can
flag fields whose generating process is fractal / self-similar (in
which case the stationary Markov assumption underlying the Matérn
kernel of Phase 1 breaks down).

Outputs (per variable):
  - spectral exponent β  (radially-averaged 2D power spectrum slope)
  - Hurst exponent H = (β − 2) / 2  (2D-fBm convention; the radially
    averaged 2D PSD of an fBm field with Hurst H scales as k^{−(2H+2)})
  - lacunarity curve     (box-counting at a geometric series of scales)
  - κ-scale-drift        (κ posterior re-fit at multiple lag resolutions
                          — fractal fields show systematic κ drift)
  - stationarity-class label ∈
        {stationary_markov, fractal_self_similar,
         non_stationary, ambiguous}

Bayesian fit uses ``sparc.causal.nuts.run_nuts`` (no numpyro dep) over
an intercept + slope + Student-t scale linear regression in log-log
space, so the fractal flag carries calibrated uncertainty.

Persisted as artifact ``stage=0, scale_hierarchy``.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gridding (irregular scattered → uniform raster for the FFT)
# ---------------------------------------------------------------------------


def grid_scatter(
    coords: np.ndarray,
    values: np.ndarray,
    *,
    n_grid: int = 64,
) -> np.ndarray:
    """Bin scattered (x, y, v) onto an n_grid×n_grid raster (mean per cell).

    Empty cells receive the global mean so the FFT does not pick up a
    sharp zero-rim.
    """
    coords = np.asarray(coords, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if coords.shape[0] != values.shape[0]:
        raise ValueError("coords and values must have the same length")
    if coords.shape[1] != 2:
        raise ValueError("coords must be (N, 2)")
    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)
    # Avoid degenerate grids
    span_x = max(xmax - xmin, 1e-9)
    span_y = max(ymax - ymin, 1e-9)
    ix = np.clip(((coords[:, 0] - xmin) / span_x * n_grid).astype(int),
                 0, n_grid - 1)
    iy = np.clip(((coords[:, 1] - ymin) / span_y * n_grid).astype(int),
                 0, n_grid - 1)
    raster = np.full((n_grid, n_grid), np.nan, dtype=np.float64)
    counts = np.zeros((n_grid, n_grid), dtype=np.int64)
    sums = np.zeros((n_grid, n_grid), dtype=np.float64)
    for i in range(coords.shape[0]):
        sums[iy[i], ix[i]] += values[i]
        counts[iy[i], ix[i]] += 1
    mask = counts > 0
    raster[mask] = sums[mask] / counts[mask]
    # Fill empty cells with the field mean (avoids FFT artefacts)
    fill = float(np.mean(values))
    raster[~mask] = fill
    return raster


# ---------------------------------------------------------------------------
# Radially-averaged 2D power spectrum
# ---------------------------------------------------------------------------


def radial_power_spectrum(
    raster: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the radially-averaged 2D power spectrum of a real raster.

    Returns ``(k, P_k)`` where ``k`` is in cycles/cell and ``P_k`` is the
    azimuthal mean of |F(k)|² in each radial bin.
    """
    raster = np.asarray(raster, dtype=np.float64)
    raster = raster - float(np.mean(raster))
    H, W = raster.shape
    F = np.fft.fftshift(np.fft.fft2(raster))
    P = (np.abs(F) ** 2) / (H * W)
    cy, cx = H // 2, W // 2
    yy, xx = np.indices(P.shape)
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_int = rr.astype(int)
    r_max = min(cy, cx)
    bins = np.arange(1, r_max + 1)  # skip DC
    P_k = np.zeros_like(bins, dtype=np.float64)
    for i, r in enumerate(bins):
        mask = r_int == r
        if mask.any():
            P_k[i] = float(P[mask].mean())
    # Normalise k to cycles/cell (so β is interpretable)
    k = bins.astype(np.float64) / float(min(H, W))
    valid = P_k > 0
    return k[valid], P_k[valid]


# ---------------------------------------------------------------------------
# Bayesian log-log slope (β posterior via NUTS, Student-t errors)
# ---------------------------------------------------------------------------


def _bayesian_loglog_slope(
    k: np.ndarray,
    P_k: np.ndarray,
    *,
    n_warmup: int = 500,
    n_samples: int = 1000,
    seed: int = 0,
) -> dict:
    """Fit ``log P_k = α − β log k + ε`` with Student-t(ν=4) errors via NUTS.

    Returns a dict with posterior summaries for β (and a Hurst H).
    Falls back to a closed-form OLS estimate when NUTS is unavailable.
    """
    log_k = np.log(k)
    log_P = np.log(P_k)
    n = len(log_k)

    # OLS fallback / initialiser
    A = np.column_stack([np.ones(n), -log_k])
    coef, _, _, _ = np.linalg.lstsq(A, log_P, rcond=None)
    alpha_init, beta_init = float(coef[0]), float(coef[1])
    resid = log_P - (alpha_init - beta_init * log_k)
    sigma_init = float(np.std(resid)) if np.std(resid) > 0 else 0.5

    try:
        import torch
        from sparc.causal.nuts import NUTSBlock, _ess_bulk, _split_r_hat, run_nuts

        log_k_t = torch.tensor(log_k, dtype=torch.float64)
        log_P_t = torch.tensor(log_P, dtype=torch.float64)
        nu_t = 4.0  # Student-t degrees of freedom (heavy tail for outliers)

        def log_prob(params):
            alpha = params["alpha"][0]
            beta = params["beta"][0]
            log_sigma = params["log_sigma"][0]
            sigma = torch.exp(log_sigma)
            mu = alpha - beta * log_k_t
            r = (log_P_t - mu) / sigma
            # Student-t log-likelihood
            ll = (-((nu_t + 1.0) / 2.0)
                  * torch.log(1.0 + (r ** 2) / nu_t)).sum()
            ll = ll - n * torch.log(sigma)
            # Priors: weak normals
            lp = -0.5 * (alpha ** 2) / 100.0 \
                 - 0.5 * (beta ** 2) / 100.0 \
                 - 0.5 * (log_sigma ** 2) / 4.0
            return ll + lp

        blocks = [
            NUTSBlock(name="alpha", dim=1,
                      init=np.array([alpha_init]), transform="identity"),
            NUTSBlock(name="beta", dim=1,
                      init=np.array([beta_init]), transform="identity"),
            NUTSBlock(name="log_sigma", dim=1,
                      init=np.array([math.log(max(sigma_init, 1e-3))]),
                      transform="identity"),
        ]
        res = run_nuts(
            log_prob_fn=log_prob, blocks=blocks,
            n_samples=n_samples, n_warmup=n_warmup,
            max_depth=8, target_accept=0.80, seed=seed, device="cpu",
        )
        beta_s = res.samples["beta"][:, 0]
        alpha_s = res.samples["alpha"][:, 0]
        # 2D fBm convention: P(k) ∝ k^{−(2H+2)} ⇒ H = (β − 2) / 2
        H_s = (beta_s - 2.0) / 2.0
        return {
            "method": "nuts",
            "n_samples": int(beta_s.shape[0]),
            "alpha": {"mean": float(np.mean(alpha_s)),
                      "hdi_89": _hdi(alpha_s, 0.89)},
            "beta": {"mean": float(np.mean(beta_s)),
                     "std": float(np.std(beta_s)),
                     "hdi_89": _hdi(beta_s, 0.89),
                     "r_hat": float(_split_r_hat(beta_s)),
                     "ess": float(_ess_bulk(beta_s))},
            "H": {"mean": float(np.mean(H_s)),
                  "hdi_89": _hdi(H_s, 0.89)},
            "fit_residual_rmse": float(np.sqrt(np.mean(
                (log_P - (alpha_init - beta_init * log_k)) ** 2))),
        }
    except Exception as e:  # pragma: no cover — fallback
        logger.info("Bayesian log-log fit fell back to OLS: %s", e)
        H_init = (beta_init - 2.0) / 2.0
        return {
            "method": "ols",
            "n_samples": 0,
            "alpha": {"mean": alpha_init, "hdi_89": [alpha_init, alpha_init]},
            "beta": {"mean": beta_init, "std": 0.0,
                     "hdi_89": [beta_init, beta_init],
                     "r_hat": float("nan"), "ess": float("nan")},
            "H": {"mean": H_init, "hdi_89": [H_init, H_init]},
            "fit_residual_rmse": float(np.sqrt(np.mean(
                (log_P - (alpha_init - beta_init * log_k)) ** 2))),
        }


def _hdi(samples: np.ndarray, prob: float) -> list[float]:
    samples = np.sort(np.asarray(samples).ravel())
    n = samples.size
    if n == 0:
        return [float("nan"), float("nan")]
    width = int(round(prob * n))
    width = max(1, min(width, n))
    n_intervals = n - width + 1
    spans = samples[width - 1: width - 1 + n_intervals] - samples[:n_intervals]
    j = int(np.argmin(spans))
    return [float(samples[j]), float(samples[j + width - 1])]


# ---------------------------------------------------------------------------
# Lacunarity (gliding-box, mass-fraction variance over the mean²)
# ---------------------------------------------------------------------------


def lacunarity_curve(raster: np.ndarray,
                     box_sizes: Optional[list[int]] = None) -> dict:
    """Compute lacunarity Λ(r) over a geometric series of box sizes.

    Λ(r) = Var(B(r)) / Mean(B(r))² + 1, where B(r) is the sum of the
    raster inside a sliding r×r window.  Higher Λ at small scales
    indicates more clustered / self-similar gappiness.
    """
    raster = np.asarray(raster, dtype=np.float64)
    H, W = raster.shape
    if box_sizes is None:
        # Geometric series capped by half-extent
        m = min(H, W)
        box_sizes = []
        r = 2
        while r <= m // 2:
            box_sizes.append(r)
            r *= 2
    out_r: list[int] = []
    out_lac: list[float] = []
    # Cumulative-sum trick for fast sliding box sums
    csum = np.cumsum(np.cumsum(raster, axis=0), axis=1)
    for r in box_sizes:
        if r >= H or r >= W:
            continue
        # Sliding sums via integral image
        S = (csum[r:, r:] - csum[:-r, r:] - csum[r:, :-r] + csum[:-r, :-r])
        m1 = float(np.mean(S))
        v = float(np.var(S))
        lac = (v / (m1 ** 2 + 1e-12)) + 1.0
        out_r.append(int(r))
        out_lac.append(float(lac))
    return {"box_sizes": out_r, "lacunarity": out_lac}


# ---------------------------------------------------------------------------
# κ scale-drift (re-bin the empirical correlogram at multiple resolutions)
# ---------------------------------------------------------------------------


def kappa_scale_drift(
    coords: np.ndarray,
    values: np.ndarray,
    *,
    n_resolutions: int = 3,
    max_pairs: int = 5000,
    seed: int = 0,
) -> dict:
    """Estimate κ at multiple lag resolutions to detect scale drift.

    For each resolution we estimate κ as the inverse e-folding length of
    the empirical autocovariance.  Returned as a per-resolution list of
    (lag_step, kappa_hat).  A monotone trend in κ across resolutions is
    the fractal signature predicted by the user.
    """
    rng = np.random.default_rng(seed)
    coords = np.asarray(coords, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    n = coords.shape[0]
    if n < 50:
        return {"resolutions": [], "kappa_per_resolution": [],
                "monotone_drift": False, "drift_score": 0.0}
    # Subsample pairs
    pair_count = min(max_pairs, n * (n - 1) // 2)
    i_idx = rng.integers(0, n, size=pair_count)
    j_idx = rng.integers(0, n, size=pair_count)
    keep = i_idx != j_idx
    i_idx, j_idx = i_idx[keep], j_idx[keep]
    d = np.linalg.norm(coords[i_idx] - coords[j_idx], axis=1)
    v_centered = values - float(np.mean(values))
    cov = v_centered[i_idx] * v_centered[j_idx]
    var = float(np.var(values))
    if var <= 0:
        return {"resolutions": [], "kappa_per_resolution": [],
                "monotone_drift": False, "drift_score": 0.0}

    d_max = float(np.percentile(d, 80))
    res_kappas: list[tuple[int, float]] = []
    base_n_lags = 8
    for r_idx in range(n_resolutions):
        n_lags = base_n_lags * (2 ** r_idx)
        if n_lags > pair_count // 4:
            continue
        edges = np.linspace(0.0, d_max, n_lags + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        rho = np.zeros(n_lags, dtype=np.float64)
        for b in range(n_lags):
            mask = (d >= edges[b]) & (d < edges[b + 1])
            if mask.sum() >= 5:
                rho[b] = float(np.mean(cov[mask])) / var
        # Fit κ as -slope of log|ρ| over lags where ρ > 0
        pos = (rho > 0.05) & (centers > 0)
        if pos.sum() >= 3:
            x = centers[pos]
            y = np.log(rho[pos])
            slope, _ = np.polyfit(x, y, 1)
            kappa_hat = float(-slope)
            if kappa_hat <= 0:
                continue
            res_kappas.append((int(n_lags), kappa_hat))

    if len(res_kappas) < 2:
        return {"resolutions": [r[0] for r in res_kappas],
                "kappa_per_resolution": [r[1] for r in res_kappas],
                "monotone_drift": False, "drift_score": 0.0}

    kappas = np.array([k for _, k in res_kappas])
    # Drift score: |max-min| / median; monotone if signs of consecutive diffs agree
    diffs = np.diff(kappas)
    monotone = bool(np.all(diffs > 0) or np.all(diffs < 0))
    drift_score = float((kappas.max() - kappas.min()) / max(np.median(kappas), 1e-12))
    return {
        "resolutions": [r for r, _ in res_kappas],
        "kappa_per_resolution": [k for _, k in res_kappas],
        "monotone_drift": monotone,
        "drift_score": drift_score,
    }


# ---------------------------------------------------------------------------
# Stationarity-class label
# ---------------------------------------------------------------------------


def classify_stationarity(
    *,
    beta_mean: float,
    beta_hdi: list[float],
    fit_rmse: float,
    drift_score: float,
    monotone_drift: bool,
) -> str:
    """Stationarity classifier from spectral β + κ scale-drift signals.

    - ``stationary_markov``: β posterior near zero; κ does not drift
      across resolutions.
    - ``fractal_self_similar``: β ∈ [2, 4] (i.e. H ∈ [0, 1]) with low
      fit residual AND monotone κ drift across resolutions.
    - ``non_stationary``: β > 4 (strong large-scale trend dominating).
    - ``ambiguous``: anything else.
    """
    fractal_lo, fractal_hi = beta_hdi[0], beta_hdi[1]
    in_fractal = (fractal_lo >= 1.5) and (fractal_hi <= 4.5) \
        and (2.0 <= beta_mean <= 4.0)
    if beta_mean > 4.0 and fractal_lo > 3.5:
        return "non_stationary"
    if in_fractal and fit_rmse < 1.0 and monotone_drift and drift_score > 0.15:
        return "fractal_self_similar"
    if not in_fractal and abs(beta_mean) < 1.0:
        return "stationary_markov"
    return "ambiguous"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_scale_hierarchy(
    coords: np.ndarray,
    variable_data: dict[str, np.ndarray],
    *,
    n_grid: int = 64,
    nuts_warmup: int = 500,
    nuts_samples: int = 1000,
    seed: int = 0,
    persist: bool = False,
    producer: str = "stage0_scale_hierarchy",
) -> dict:
    """Compute Phase-6 scale-hierarchy diagnostics for each variable.

    Parameters
    ----------
    coords : (N, 2) array of x, y coordinates (shared across variables).
    variable_data : ``{name: (N,) values}`` mapping.
    n_grid : raster resolution for the FFT.
    persist : when True, writes ``stage=0, scale_hierarchy`` to the
              active artifact store.
    """
    coords = np.asarray(coords, dtype=np.float64)
    per_variable: dict[str, dict] = {}

    for name, vals in variable_data.items():
        vals = np.asarray(vals, dtype=np.float64)
        try:
            raster = grid_scatter(coords, vals, n_grid=n_grid)
            k, P_k = radial_power_spectrum(raster)
            if k.size < 3:
                raise ValueError(
                    f"Power spectrum for '{name}' has too few bins ({k.size})"
                )
            slope = _bayesian_loglog_slope(
                k, P_k,
                n_warmup=nuts_warmup, n_samples=nuts_samples, seed=seed,
            )
            lac = lacunarity_curve(raster)
            drift = kappa_scale_drift(coords, vals, seed=seed)
            label = classify_stationarity(
                beta_mean=slope["beta"]["mean"],
                beta_hdi=slope["beta"]["hdi_89"],
                fit_rmse=slope["fit_residual_rmse"],
                drift_score=drift["drift_score"],
                monotone_drift=drift["monotone_drift"],
            )
            per_variable[name] = {
                "spectral": {
                    "k": k.tolist(),
                    "P_k": P_k.tolist(),
                    "slope": slope,
                },
                "lacunarity": lac,
                "kappa_scale_drift": drift,
                "stationarity_class": label,
            }
        except Exception as e:  # noqa: BLE001
            logger.warning("Scale hierarchy failed for '%s': %s", name, e)
            per_variable[name] = {"error": str(e),
                                  "stationarity_class": "ambiguous"}

    payload = {
        "per_variable": per_variable,
        "n_grid": int(n_grid),
        "n_variables": len(per_variable),
        "summary": {
            "stationarity_counts": _count_classes(per_variable),
        },
    }

    if persist:
        try:
            from sparc.registry.store import get_active_store
            store = get_active_store()
            if store is not None:
                store.write_struct(
                    "0", "scale_hierarchy", payload,
                    producer=producer,
                    consumers=["report:/stage0/scale_hierarchy",
                               "audit:/stage0/stationarity"],
                )
        except Exception as e:  # noqa: BLE001
            payload["persist_error"] = str(e)

    return payload


def _count_classes(per_variable: dict) -> dict[str, int]:
    counts: dict[str, int] = {}
    for v in per_variable.values():
        c = v.get("stationarity_class", "ambiguous")
        counts[c] = counts.get(c, 0) + 1
    return counts
