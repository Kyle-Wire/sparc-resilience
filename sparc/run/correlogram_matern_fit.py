"""
Bayesian Matérn fit for SPARC's empirical correlogram.

Fits a Matérn(κ, ν, σ², τ²) covariance model to the empirical Moran's I
curve produced by ``SpatialAutocorrelationAnalyzer.compute_correlogram()``.

The Matérn correlation function is

    ρ(h; κ, ν) = (2^(1-ν) / Γ(ν)) · (κh)^ν · K_ν(κh)

with closed-form simplifications at ν ∈ {0.5, 1.5, 2.5} (matern12, matern32,
matern52).  We assume the empirical Moran's I curve approximates the spatial
correlation function up to a multiplicative sill (σ²) and an additive nugget
(τ²) at h=0:

    I_obs(h) ≈ σ² · ρ(h; κ, ν) + τ² · 𝟙[h≈0] + ε(h)

NUTS sampling (via ``sparc.causal.nuts.run_nuts``) returns a posterior over
(κ, ν, σ², τ²) plus standard convergence diagnostics (R-hat, ESS).

A least-squares fallback is provided for very small datasets via
``method="lsq"``.

This module is the foundation for Phase 1 of the kernel-field roadmap:
``κ_correlogram`` is the quantity compared to the PDE-derived ``κ_PDE``
in ``sparc.physics.kappa_estimator``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from scipy import optimize, special

from sparc.causal.nuts import NUTSBlock, run_nuts

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Matérn kernel  (closed-form for ν ∈ {0.5, 1.5, 2.5}; generic via K_ν)
# ---------------------------------------------------------------------------


def matern_correlation_np(h: np.ndarray, kappa: float, nu: float) -> np.ndarray:
    """Matérn correlation, NumPy / SciPy path.

    Returns ρ(h) ∈ [0, 1] with ρ(0) = 1.  Uses closed-form expressions
    at ν ∈ {0.5, 1.5, 2.5}; generic ν uses ``scipy.special.kv``.
    """
    h = np.asarray(h, dtype=np.float64)
    out = np.ones_like(h)
    nz = h > 0
    z = kappa * h[nz]

    if math.isclose(nu, 0.5, abs_tol=1e-3):
        out[nz] = np.exp(-z)
    elif math.isclose(nu, 1.5, abs_tol=1e-3):
        out[nz] = (1.0 + z) * np.exp(-z)
    elif math.isclose(nu, 2.5, abs_tol=1e-3):
        out[nz] = (1.0 + z + z * z / 3.0) * np.exp(-z)
    else:
        log_pref = (1.0 - nu) * math.log(2.0) - special.gammaln(nu)
        log_zpow = nu * np.log(z)
        log_kv = np.log(np.maximum(special.kv(nu, z), 1e-300))
        out[nz] = np.exp(log_pref + log_zpow + log_kv)
    return np.clip(out, 0.0, 1.0)


def matern_correlation_torch(
    h: torch.Tensor, kappa: torch.Tensor, nu: float
) -> torch.Tensor:
    """Differentiable Matérn correlation for NUTS log-prob.

    Restricted to ν ∈ {0.5, 1.5, 2.5} so we have closed-form gradients;
    the ν prior in :func:`fit_matern_bayes` is a 3-way categorical so this
    restriction is honoured by construction.  ``h`` is the lag distance
    tensor, ``kappa`` is a scalar tensor (positive).
    """
    z = kappa * h
    out = torch.ones_like(h)
    if math.isclose(nu, 0.5, abs_tol=1e-3):
        out = torch.exp(-z)
    elif math.isclose(nu, 1.5, abs_tol=1e-3):
        out = (1.0 + z) * torch.exp(-z)
    elif math.isclose(nu, 2.5, abs_tol=1e-3):
        out = (1.0 + z + z * z / 3.0) * torch.exp(-z)
    else:
        raise ValueError(
            f"matern_correlation_torch only supports ν ∈ {{0.5, 1.5, 2.5}}, got {nu}"
        )
    # Enforce ρ(0) = 1 exactly (h could be ≥0 with floating-point fuzz).
    out = torch.where(h <= 0, torch.ones_like(out), out)
    return out


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class MaternFitResult:
    """Posterior summary for a single-variable Matérn fit.

    All posterior arrays are ``np.ndarray`` of shape ``(n_samples,)``.
    Means/HDIs are scalars; arrays of length 2 for HDIs.
    """

    method: str                              # "bayes" | "lsq"
    kappa_mean: float
    kappa_hdi: tuple[float, float]
    kappa_samples: np.ndarray
    nu: float                                # selected ν (categorical for bayes; fixed for lsq)
    sigma2_mean: float
    sigma2_hdi: tuple[float, float]
    tau2_mean: float
    tau2_hdi: tuple[float, float]
    fit_rmse: float
    r_hat: dict[str, float] = field(default_factory=dict)
    ess: dict[str, float] = field(default_factory=dict)
    converged: bool = True
    n_samples_used: int = 0
    nu_log_marginals: dict[float, float] = field(default_factory=dict)

    def to_payload(self) -> dict:
        """Convert to JSON-friendly dict for ``ArtifactStore.write_struct``.

        Posterior samples are kept (thinned) for downstream consumers.
        """
        return {
            "method": self.method,
            "kappa": {
                "mean": float(self.kappa_mean),
                "hdi89": [float(self.kappa_hdi[0]), float(self.kappa_hdi[1])],
                "samples": [float(x) for x in self.kappa_samples.tolist()],
            },
            "nu": float(self.nu),
            "nu_log_marginals": {str(k): float(v) for k, v in self.nu_log_marginals.items()},
            "sigma2": {
                "mean": float(self.sigma2_mean),
                "hdi89": [float(self.sigma2_hdi[0]), float(self.sigma2_hdi[1])],
            },
            "tau2": {
                "mean": float(self.tau2_mean),
                "hdi89": [float(self.tau2_hdi[0]), float(self.tau2_hdi[1])],
            },
            "fit_rmse": float(self.fit_rmse),
            "r_hat": {k: float(v) for k, v in self.r_hat.items()},
            "ess": {k: float(v) for k, v in self.ess.items()},
            "converged": bool(self.converged),
            "n_samples_used": int(self.n_samples_used),
        }


# ---------------------------------------------------------------------------
# Heuristic initial estimate from raw correlogram
# ---------------------------------------------------------------------------


def _heuristic_kappa_init(lags: np.ndarray, morans: np.ndarray) -> float:
    """Cheap log-linear fit ρ ≈ exp(-κh) → κ ≈ -slope.

    Used as the centre of the κ prior so NUTS starts in a sensible region.
    Returns a strictly-positive value; falls back to 1/median(lags) on
    degeneracy.
    """
    pos = (morans > 1e-3) & (lags > 0)
    if pos.sum() >= 3:
        log_rho = np.log(morans[pos])
        slope, _ = np.polyfit(lags[pos], log_rho, 1)
        if slope < 0:
            return float(-slope)
    med = float(np.median(lags[lags > 0])) if (lags > 0).any() else 1.0
    return 1.0 / max(med, 1e-9)


# ---------------------------------------------------------------------------
# 89% HDI
# ---------------------------------------------------------------------------


def _hdi(samples: np.ndarray, prob: float = 0.89) -> tuple[float, float]:
    s = np.sort(np.asarray(samples).ravel())
    n = len(s)
    if n < 2:
        v = float(s[0]) if n == 1 else 0.0
        return (v, v)
    k = max(1, int(math.floor(prob * n)))
    widths = s[k:] - s[: n - k]
    i = int(np.argmin(widths))
    return float(s[i]), float(s[i + k])


# ---------------------------------------------------------------------------
# Bayesian fit
# ---------------------------------------------------------------------------


def fit_matern_bayes(
    lags: np.ndarray,
    morans: np.ndarray,
    *,
    n_samples: int = 2000,
    n_warmup: int = 1000,
    n_chains: int = 2,
    seed: int = 42,
    nu_grid: tuple[float, ...] = (0.5, 1.5, 2.5),
) -> MaternFitResult:
    """Posterior over (κ, σ², τ²) for each ν ∈ ``nu_grid``; pick ν by marginal.

    The full Matérn ν is treated as a 3-way categorical (the closed-form
    half-integers).  For each fixed ν we run NUTS over (κ, σ², τ²) and
    estimate the log-marginal-likelihood by harmonic mean of the chain's
    log-probs (cheap and adequate for ν selection at this granularity).

    Parameters
    ----------
    lags : (L,) lag distances (must be > 0 for non-trivial bins).
    morans : (L,) empirical Moran's I values at each lag.

    Returns
    -------
    MaternFitResult.  ``method == "bayes"``.
    """
    lags = np.asarray(lags, dtype=np.float64)
    morans = np.asarray(morans, dtype=np.float64)
    if lags.shape != morans.shape:
        raise ValueError("lags and morans must have the same shape")
    L = len(lags)
    if L < 4:
        raise ValueError(f"Need at least 4 lag points to fit Matérn; got {L}")

    kappa_init = _heuristic_kappa_init(lags, morans)
    sigma2_init = float(np.clip(np.max(morans), 1e-3, 2.0))
    tau2_init = float(np.clip(np.var(morans - sigma2_init * np.exp(-kappa_init * lags)), 1e-6, 1.0))
    obs_var = float(np.var(morans).clip(min=1e-6))

    # Priors (in transformed/unconstrained space the NUTS sampler uses):
    #   κ ~ LogNormal(log(kappa_init), 1.0)
    #   σ² ~ HalfNormal(0, 1.0) implemented via Half-Normal on log σ²:
    #        we use σ² = exp(log_sigma2) with prior -0.5*log_sigma2² (proper).
    #   τ² ~ HalfNormal(0, 0.25)
    # ε observation noise: N(0, obs_var * weight_h) — weight grows with lag
    #        because empirical Moran's I has more variance at long lags
    #        (fewer pairs).  This matches the heteroscedastic reality without
    #        needing to count pairs explicitly.
    h_t = torch.tensor(lags, dtype=torch.float64)
    y_t = torch.tensor(morans, dtype=torch.float64)
    weight = torch.tensor(1.0 + lags / max(lags.max(), 1e-9), dtype=torch.float64)
    log_kappa_prior_mean = math.log(max(kappa_init, 1e-9))

    def make_log_prob(nu_val: float):
        def _log_prob(params: dict[str, torch.Tensor]) -> torch.Tensor:
            kappa = params["kappa"]
            sigma2 = params["sigma2"]
            tau2 = params["tau2"]
            obs_sigma2 = params["obs_sigma2"]
            rho = matern_correlation_torch(h_t, kappa, nu_val)
            mu = sigma2 * rho + tau2 * (h_t == 0).double()
            sigma2_obs = obs_sigma2 * weight
            ll = -0.5 * torch.sum(
                torch.log(sigma2_obs) + (y_t - mu) ** 2 / sigma2_obs
            )
            # Priors
            log_k = torch.log(kappa)
            lp_kappa = -0.5 * ((log_k - log_kappa_prior_mean) ** 2)
            lp_sigma2 = -0.5 * (sigma2 ** 2)
            lp_tau2 = -0.5 * (tau2 ** 2) / (0.25 ** 2)
            lp_obs = -2.0 * torch.log(obs_sigma2)  # weak Jeffreys-ish
            return ll + lp_kappa + lp_sigma2 + lp_tau2 + lp_obs
        return _log_prob

    nu_results: dict[float, dict] = {}
    for nu_val in nu_grid:
        blocks = [
            NUTSBlock(name="kappa", dim=1,
                      init=np.array([math.log(max(kappa_init, 1e-9))]),
                      transform="log"),
            NUTSBlock(name="sigma2", dim=1,
                      init=np.array([math.log(max(sigma2_init, 1e-9))]),
                      transform="log"),
            NUTSBlock(name="tau2", dim=1,
                      init=np.array([math.log(max(tau2_init, 1e-9))]),
                      transform="log"),
            NUTSBlock(name="obs_sigma2", dim=1,
                      init=np.array([math.log(max(obs_var, 1e-9))]),
                      transform="log"),
        ]
        chain_results = []
        try:
            _nuts_device = "cuda" if torch.cuda.is_available() else "cpu"
            if _nuts_device == "cuda":
                logger.debug("Matérn NUTS: using cuda")
            for chain_idx in range(max(1, n_chains)):
                res = run_nuts(
                    log_prob_fn=make_log_prob(nu_val),
                    blocks=blocks,
                    n_samples=n_samples,
                    n_warmup=n_warmup,
                    max_depth=8,
                    target_accept=0.80,
                    seed=seed + chain_idx,
                    device=_nuts_device,
                )
                chain_results.append(res)
        except Exception as exc:
            logger.warning("Matérn NUTS failed at ν=%.1f: %s", nu_val, exc)
            continue

        # Pool chains
        pooled = {k: np.concatenate([r.samples[k] for r in chain_results], axis=0)
                  for k in chain_results[0].samples}
        log_probs = np.concatenate([r.log_probs for r in chain_results], axis=0)
        # Harmonic-mean log-marginal estimate (Newton-Raftery): adequate for
        # ν selection over a 3-point grid.  Stable in log space.
        lp_max = float(np.max(log_probs))
        log_marg = lp_max - math.log(np.mean(np.exp(lp_max - log_probs)))
        # Per-chain split-R-hat from the first chain (chains pooled below).
        from sparc.causal.nuts import _split_r_hat, _ess_bulk
        r_hat = {k: float(_split_r_hat(pooled[k][:, 0])) for k in pooled}
        ess = {k: float(_ess_bulk(pooled[k][:, 0])) for k in pooled}
        nu_results[nu_val] = dict(
            samples=pooled, log_marg=log_marg, r_hat=r_hat, ess=ess,
            n_chains=len(chain_results),
        )

    if not nu_results:
        raise RuntimeError("All Matérn NUTS chains failed across ν grid")

    # Pick best ν by log-marginal
    nu_best = max(nu_results, key=lambda k: nu_results[k]["log_marg"])
    best = nu_results[nu_best]

    kappa_s = best["samples"]["kappa"][:, 0]
    sigma2_s = best["samples"]["sigma2"][:, 0]
    tau2_s = best["samples"]["tau2"][:, 0]

    # Posterior-mean fit RMSE
    rho_hat = matern_correlation_np(lags, float(np.mean(kappa_s)), nu_best)
    mu_hat = float(np.mean(sigma2_s)) * rho_hat
    mu_hat[lags == 0] += float(np.mean(tau2_s))
    fit_rmse = float(np.sqrt(np.mean((morans - mu_hat) ** 2)))

    converged = all(v < 1.05 for v in best["r_hat"].values()) and all(
        v > 100 for v in best["ess"].values()
    )

    return MaternFitResult(
        method="bayes",
        kappa_mean=float(np.mean(kappa_s)),
        kappa_hdi=_hdi(kappa_s),
        kappa_samples=kappa_s,
        nu=float(nu_best),
        sigma2_mean=float(np.mean(sigma2_s)),
        sigma2_hdi=_hdi(sigma2_s),
        tau2_mean=float(np.mean(tau2_s)),
        tau2_hdi=_hdi(tau2_s),
        fit_rmse=fit_rmse,
        r_hat=best["r_hat"],
        ess=best["ess"],
        converged=converged,
        n_samples_used=int(len(kappa_s)),
        nu_log_marginals={k: float(v["log_marg"]) for k, v in nu_results.items()},
    )


# ---------------------------------------------------------------------------
# Least-squares fallback
# ---------------------------------------------------------------------------


def fit_matern_lsq(
    lags: np.ndarray,
    morans: np.ndarray,
    *,
    nu: float = 1.5,
) -> MaternFitResult:
    """Bounded LSQ fit of (κ, σ², τ²) at fixed ν.

    Used when ``correlogram.matern.method = "lsq"`` or as an automatic
    fallback when the Bayesian fit raises.  HDIs are degenerate (point
    estimates) — downstream consumers must check ``method == "lsq"``
    before trusting uncertainty quantification.
    """
    lags = np.asarray(lags, dtype=np.float64)
    morans = np.asarray(morans, dtype=np.float64)
    kappa_init = _heuristic_kappa_init(lags, morans)
    sigma2_init = float(np.clip(np.max(morans), 1e-3, 2.0))
    tau2_init = 0.05

    def residual(theta):
        kappa, sigma2, tau2 = theta
        rho = matern_correlation_np(lags, kappa, nu)
        mu = sigma2 * rho
        mu[lags == 0] += tau2
        return mu - morans

    res = optimize.least_squares(
        residual,
        x0=[kappa_init, sigma2_init, tau2_init],
        bounds=([1e-6, 1e-6, 0.0], [1e3, 5.0, 1.0]),
        max_nfev=200,
    )
    kappa, sigma2, tau2 = res.x
    fit_rmse = float(np.sqrt(np.mean(res.fun ** 2)))
    return MaternFitResult(
        method="lsq",
        kappa_mean=float(kappa),
        kappa_hdi=(float(kappa), float(kappa)),
        kappa_samples=np.array([kappa], dtype=np.float64),
        nu=float(nu),
        sigma2_mean=float(sigma2),
        sigma2_hdi=(float(sigma2), float(sigma2)),
        tau2_mean=float(tau2),
        tau2_hdi=(float(tau2), float(tau2)),
        fit_rmse=fit_rmse,
        r_hat={},
        ess={},
        converged=bool(res.success),
        n_samples_used=1,
        nu_log_marginals={float(nu): 0.0},
    )


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------


def fit_matern(
    lags: np.ndarray,
    morans: np.ndarray,
    *,
    method: str = "bayes",
    n_samples: int = 800,
    n_warmup: int = 400,
    n_chains: int = 2,
    seed: int = 42,
) -> MaternFitResult:
    """Dispatch to Bayesian or LSQ Matérn fit based on ``method``.

    Falls back to LSQ if the Bayesian path raises (e.g. CUDA OOM in user
    environments where the NUTS sampler somehow ends up on GPU).
    """
    method = method.lower()
    if method == "lsq":
        return fit_matern_lsq(lags, morans)
    if method != "bayes":
        raise ValueError(f"Unknown method {method!r}; expected 'bayes' or 'lsq'")
    try:
        return fit_matern_bayes(
            lags, morans,
            n_samples=n_samples, n_warmup=n_warmup, n_chains=n_chains, seed=seed,
        )
    except Exception as exc:
        logger.warning("Bayesian Matérn fit failed (%s); falling back to LSQ", exc)
        return fit_matern_lsq(lags, morans)
