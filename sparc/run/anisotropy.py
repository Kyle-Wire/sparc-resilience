"""
Bayesian anisotropic Matérn fit for SPARC's directional correlogram.

Given a (lag × angle) Moran's I matrix from
:meth:`SpatialAutocorrelationAnalyzer.compute_directional_correlogram`,
we fit an anisotropic Matérn correlation model

    κ_eff(φ) = √( (κ_x · cos(φ − θ))² + (κ_y · sin(φ − θ))² )
    ρ(h, φ) = Matérn(h ; κ_eff(φ), ν)
    I_obs(h, φ) ≈ σ² · ρ(h, φ) + τ² · 𝟙[h≈0] + ε(h, φ)

via NUTS over (κ_x, κ_y, θ, σ², τ², obs_σ²) at fixed ν.  ν is selected
from the half-integer grid {0.5, 1.5, 2.5} by harmonic-mean log-marginal
(same scheme as the isotropic Phase 1 fit).

Outputs include the **anisotropy ellipse** (semi-axes 1/κ_x, 1/κ_y;
orientation θ; eccentricity √(1 − r_min²/r_max²)) and a plain-language
``dominant_direction_hint`` describing the major axis with uncertainty.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from sparc.causal.nuts import NUTSBlock, run_nuts, _split_r_hat, _ess_bulk
from sparc.run.correlogram_matern_fit import (
    _hdi,
    _heuristic_kappa_init,
    matern_correlation_np,
    matern_correlation_torch,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class AnisotropyResult:
    """Posterior summary for a single-variable anisotropic Matérn fit."""

    method: str                           # "bayes" | "lsq"
    nu: float
    # Posterior summaries (all means + 89% HDIs)
    kappa_x_mean: float
    kappa_x_hdi: tuple[float, float]
    kappa_y_mean: float
    kappa_y_hdi: tuple[float, float]
    theta_mean_rad: float
    theta_hdi_rad: tuple[float, float]
    sigma2_mean: float
    tau2_mean: float
    # Ellipse
    semi_major_m: float
    semi_minor_m: float
    eccentricity_mean: float
    eccentricity_hdi: tuple[float, float]
    dominant_direction_hint: str
    # Diagnostics
    fit_rmse: float
    r_hat: dict[str, float] = field(default_factory=dict)
    ess: dict[str, float] = field(default_factory=dict)
    converged: bool = True
    n_samples_used: int = 0
    nu_log_marginals: dict[float, float] = field(default_factory=dict)
    # Raw posterior arrays (kept for downstream consumers + ratio diagnostics)
    kappa_x_samples: np.ndarray = field(default_factory=lambda: np.array([]))
    kappa_y_samples: np.ndarray = field(default_factory=lambda: np.array([]))
    theta_samples_rad: np.ndarray = field(default_factory=lambda: np.array([]))
    eccentricity_samples: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_payload(self) -> dict:
        """JSON-friendly dict for ``ArtifactStore.write_struct``."""
        return {
            "method": self.method,
            "nu": float(self.nu),
            "nu_log_marginals": {str(k): float(v) for k, v in self.nu_log_marginals.items()},
            "kappa_x": {
                "mean": float(self.kappa_x_mean),
                "hdi89": [float(self.kappa_x_hdi[0]), float(self.kappa_x_hdi[1])],
                "samples": [float(x) for x in self.kappa_x_samples.tolist()],
            },
            "kappa_y": {
                "mean": float(self.kappa_y_mean),
                "hdi89": [float(self.kappa_y_hdi[0]), float(self.kappa_y_hdi[1])],
                "samples": [float(x) for x in self.kappa_y_samples.tolist()],
            },
            "theta_rad": {
                "mean": float(self.theta_mean_rad),
                "hdi89": [float(self.theta_hdi_rad[0]), float(self.theta_hdi_rad[1])],
                "mean_deg": float(math.degrees(self.theta_mean_rad)),
                "hdi89_deg": [
                    float(math.degrees(self.theta_hdi_rad[0])),
                    float(math.degrees(self.theta_hdi_rad[1])),
                ],
                "samples": [float(x) for x in self.theta_samples_rad.tolist()],
            },
            "ellipse": {
                "semi_major_m": float(self.semi_major_m),
                "semi_minor_m": float(self.semi_minor_m),
                "eccentricity": {
                    "mean": float(self.eccentricity_mean),
                    "hdi89": [
                        float(self.eccentricity_hdi[0]),
                        float(self.eccentricity_hdi[1]),
                    ],
                    "samples": [float(x) for x in self.eccentricity_samples.tolist()],
                },
            },
            "sigma2_mean": float(self.sigma2_mean),
            "tau2_mean": float(self.tau2_mean),
            "dominant_direction_hint": self.dominant_direction_hint,
            "fit_rmse": float(self.fit_rmse),
            "r_hat": {k: float(v) for k, v in self.r_hat.items()},
            "ess": {k: float(v) for k, v in self.ess.items()},
            "converged": bool(self.converged),
            "n_samples_used": int(self.n_samples_used),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_COMPASS_16 = [
    "E", "ENE", "NE", "NNE", "N", "NNW", "NW", "WNW",
    "W", "WSW", "SW", "SSW", "S", "SSE", "SE", "ESE",
]


def _theta_to_compass(theta_rad: float) -> str:
    """Map an angle in radians (math-convention, 0 = +x axis = East,
    increases CCW) to a 16-point compass label.  Director (mod π).
    """
    deg = (math.degrees(theta_rad) % 180.0)
    # 16-point compass spans 360°; we have a director (180°), so the
    # major axis label is e.g. "WSW–ENE".  Compute the principal
    # compass label of the +deg ray; the other end is implicit.
    idx = int(round((deg % 360.0) / 22.5)) % 16
    a = _COMPASS_16[idx]
    b = _COMPASS_16[(idx + 8) % 16]
    return f"{a}-{b}"


def _circular_mean_mod_pi(theta_samples: np.ndarray) -> float:
    """Circular mean for a director on [0, π).  We double the angle to map
    to [0, 2π), take the standard circular mean, then halve back.
    """
    s = np.sort(np.mod(theta_samples, np.pi))
    if len(s) == 0:
        return 0.0
    two = 2.0 * s
    sin_sum = float(np.mean(np.sin(two)))
    cos_sum = float(np.mean(np.cos(two)))
    mean2 = math.atan2(sin_sum, cos_sum) % (2.0 * math.pi)
    return float(0.5 * mean2)


def _circular_hdi_mod_pi(theta_samples: np.ndarray, prob: float = 0.89) -> tuple[float, float]:
    """89% HDI for a director angle on [0, π).  We rotate samples so the
    circular mean sits at π/2, take the standard interval HDI, then rotate
    back.  Returns angles in [0, π) (HDI may wrap; we report (lo, hi) in
    that case as lo > hi to make wrap explicit).
    """
    if len(theta_samples) < 2:
        v = float(theta_samples[0]) if len(theta_samples) == 1 else 0.0
        return (v, v)
    mu = _circular_mean_mod_pi(theta_samples)
    shifted = np.mod(theta_samples - mu + 0.5 * math.pi, math.pi)
    lo, hi = _hdi(shifted, prob=prob)
    lo_back = (lo - 0.5 * math.pi + mu) % math.pi
    hi_back = (hi - 0.5 * math.pi + mu) % math.pi
    return float(lo_back), float(hi_back)


def _make_dominant_direction_hint(
    theta_samples: np.ndarray,
    eccentricity_samples: np.ndarray,
) -> str:
    """Plain-language summary of the major axis with uncertainty."""
    if len(theta_samples) == 0:
        return "anisotropy posterior unavailable"
    th_mean = _circular_mean_mod_pi(theta_samples)
    th_lo, th_hi = _circular_hdi_mod_pi(theta_samples)
    # Width of HDI (handle wrap)
    width = (th_hi - th_lo) % math.pi
    half_width_deg = math.degrees(width) / 2.0
    ecc_mean = float(np.mean(eccentricity_samples)) if len(eccentricity_samples) else 0.0
    ecc_lo, ecc_hi = _hdi(eccentricity_samples) if len(eccentricity_samples) >= 2 else (ecc_mean, ecc_mean)
    compass = _theta_to_compass(th_mean)
    if ecc_mean < 0.15:
        flavour = "near-isotropic; no strong directional preference"
    elif ecc_mean < 0.4:
        flavour = "weak anisotropy"
    elif ecc_mean < 0.7:
        flavour = "moderate anisotropy, suggests directional process"
    else:
        flavour = "strong anisotropy, likely advective transport along this axis"
    return (
        f"{compass} +/-{half_width_deg:.0f}deg, "
        f"eccentricity {ecc_mean:.2f} [{ecc_lo:.2f}, {ecc_hi:.2f}]; {flavour}"
    )


# ---------------------------------------------------------------------------
# Bayesian fit
# ---------------------------------------------------------------------------


def fit_anisotropy_bayes(
    lags: np.ndarray,
    angles_rad: np.ndarray,
    morans: np.ndarray,
    n_pairs: Optional[np.ndarray] = None,
    *,
    n_samples: int = 400,
    n_warmup: int = 300,
    n_chains: int = 2,
    seed: int = 42,
    nu_grid: tuple[float, ...] = (0.5, 1.5, 2.5),
    kappa_init: Optional[float] = None,
) -> AnisotropyResult:
    """Posterior over (κ_x, κ_y, θ, σ², τ², obs_σ²) per ν; ν chosen by marginal.

    Parameters
    ----------
    lags : (L,) lag-bin centres (m).
    angles_rad : (K,) angle-bin centres (radians, [0, π)).
    morans : (L, K) Moran's I.
    n_pairs : (L, K) optional integer pair counts; bins with 0 pairs are masked.
    """
    lags = np.asarray(lags, dtype=np.float64)
    angles = np.asarray(angles_rad, dtype=np.float64)
    morans = np.asarray(morans, dtype=np.float64)
    if morans.shape != (len(lags), len(angles)):
        raise ValueError(
            f"morans shape {morans.shape} does not match (L,K)=({len(lags)},{len(angles)})"
        )

    if n_pairs is None:
        valid = np.ones_like(morans, dtype=bool)
    else:
        valid = np.asarray(n_pairs) > 0
    if valid.sum() < 6:
        raise ValueError(
            f"Need at least 6 valid (lag, angle) bins to fit anisotropy; got {int(valid.sum())}"
        )

    # Heuristic init: use isotropic Phase-1 estimator on the angle-pooled
    # mean curve.
    if kappa_init is None or not math.isfinite(kappa_init) or kappa_init <= 0:
        m_iso = np.array([
            np.nanmean(morans[li, valid[li]]) if valid[li].any() else 0.0
            for li in range(len(lags))
        ])
        kappa_init = _heuristic_kappa_init(lags, m_iso)
    log_k0 = math.log(max(float(kappa_init), 1e-9))
    sigma2_init = float(np.clip(np.nanmax(morans), 1e-3, 2.0))
    obs_var = float(np.nanvar(morans).clip(min=1e-6))

    # Tensor inputs ---------------------------------------------------------
    h_t = torch.tensor(lags, dtype=torch.float64)            # (L,)
    a_t = torch.tensor(angles, dtype=torch.float64)          # (K,)
    y_t = torch.tensor(morans, dtype=torch.float64)          # (L, K)
    valid_t = torch.tensor(valid, dtype=torch.float64)       # (L, K)
    # Per-bin variance weight: inflate variance for low-pair bins.
    if n_pairs is not None:
        nps = np.asarray(n_pairs, dtype=np.float64)
        wmax = float(nps.max()) if nps.max() > 0 else 1.0
        weight_np = wmax / np.clip(nps, 1.0, None)            # ~1 for full bins
    else:
        weight_np = np.ones_like(morans, dtype=np.float64)
    weight_t = torch.tensor(weight_np, dtype=torch.float64)
    h_zero = torch.tensor((lags == 0).astype(np.float64), dtype=torch.float64)  # (L,)

    # h broadcast to (L, K): same h_l for every angle.
    h_LK = h_t.unsqueeze(1).expand(-1, len(angles))

    def make_log_prob(nu_val: float):
        def _log_prob(p: dict[str, torch.Tensor]) -> torch.Tensor:
            kx = p["kappa_x"]                 # scalar > 0
            ky = p["kappa_y"]                 # scalar > 0
            theta = p["theta_unit"] * math.pi  # ∈ (0, π)
            sigma2 = p["sigma2"]
            tau2 = p["tau2"]
            obs_sigma2 = p["obs_sigma2"]
            # Per-angle effective κ: κ_eff(φ) = √((κx cos(φ-θ))² + (κy sin(φ-θ))²)
            dphi = a_t - theta                                  # (K,)
            c2 = torch.cos(dphi) ** 2
            s2 = torch.sin(dphi) ** 2
            kappa_eff = torch.sqrt((kx * kx) * c2 + (ky * ky) * s2)  # (K,)
            kappa_eff_LK = kappa_eff.unsqueeze(0).expand(len(lags), -1)  # (L,K)
            # Matérn ρ: closed-form for ν ∈ {0.5, 1.5, 2.5}; we call the
            # torch helper element-wise via z = κ·h.
            z = kappa_eff_LK * h_LK
            if math.isclose(nu_val, 0.5, abs_tol=1e-3):
                rho = torch.exp(-z)
            elif math.isclose(nu_val, 1.5, abs_tol=1e-3):
                rho = (1.0 + z) * torch.exp(-z)
            else:  # 2.5
                rho = (1.0 + z + z * z / 3.0) * torch.exp(-z)
            rho = torch.where(h_LK <= 0, torch.ones_like(rho), rho)
            mu = sigma2 * rho + tau2 * h_zero.unsqueeze(1)      # (L, K)
            sigma2_obs = obs_sigma2 * weight_t                  # (L, K)
            ll_per = -0.5 * (torch.log(sigma2_obs) + (y_t - mu) ** 2 / sigma2_obs)
            ll = torch.sum(ll_per * valid_t)
            # Priors
            lp_kx = -0.5 * ((torch.log(kx) - log_k0) ** 2)
            lp_ky = -0.5 * ((torch.log(ky) - log_k0) ** 2)
            lp_sigma2 = -0.5 * (sigma2 ** 2)
            lp_tau2 = -0.5 * (tau2 ** 2) / (0.25 ** 2)
            lp_obs = -2.0 * torch.log(obs_sigma2)
            # θ: uniform via logit transform's Jacobian — no extra prior term.
            return ll + lp_kx + lp_ky + lp_sigma2 + lp_tau2 + lp_obs
        return _log_prob

    nu_results: dict[float, dict] = {}
    for nu_val in nu_grid:
        blocks = [
            NUTSBlock(name="kappa_x", dim=1,
                      init=np.array([log_k0]), transform="log"),
            NUTSBlock(name="kappa_y", dim=1,
                      init=np.array([log_k0]), transform="log"),
            NUTSBlock(name="theta_unit", dim=1,
                      init=np.array([0.0]),  # logit(0.5) = 0 → θ = π/2
                      transform="logit"),
            NUTSBlock(name="sigma2", dim=1,
                      init=np.array([math.log(max(sigma2_init, 1e-9))]),
                      transform="log"),
            NUTSBlock(name="tau2", dim=1,
                      init=np.array([math.log(0.05)]),
                      transform="log"),
            NUTSBlock(name="obs_sigma2", dim=1,
                      init=np.array([math.log(max(obs_var, 1e-9))]),
                      transform="log"),
        ]
        chain_results = []
        try:
            for chain_idx in range(max(1, n_chains)):
                res = run_nuts(
                    log_prob_fn=make_log_prob(nu_val),
                    blocks=blocks,
                    n_samples=n_samples,
                    n_warmup=n_warmup,
                    max_depth=8,
                    target_accept=0.80,
                    seed=seed + chain_idx,
                    device="cpu",
                )
                chain_results.append(res)
        except Exception as exc:
            logger.warning("Anisotropy NUTS failed at ν=%.1f: %s", nu_val, exc)
            continue

        pooled = {k: np.concatenate([r.samples[k] for r in chain_results], axis=0)
                  for k in chain_results[0].samples}
        log_probs = np.concatenate([r.log_probs for r in chain_results], axis=0)
        lp_max = float(np.max(log_probs))
        log_marg = lp_max - math.log(np.mean(np.exp(lp_max - log_probs)))
        r_hat = {k: float(_split_r_hat(pooled[k][:, 0])) for k in pooled}
        ess = {k: float(_ess_bulk(pooled[k][:, 0])) for k in pooled}
        nu_results[nu_val] = dict(samples=pooled, log_marg=log_marg,
                                   r_hat=r_hat, ess=ess)

    if not nu_results:
        raise RuntimeError("All anisotropy NUTS chains failed across ν grid")

    nu_best = max(nu_results, key=lambda k: nu_results[k]["log_marg"])
    best = nu_results[nu_best]

    kx_s = best["samples"]["kappa_x"][:, 0]
    ky_s = best["samples"]["kappa_y"][:, 0]
    theta_s = best["samples"]["theta_unit"][:, 0] * math.pi
    sigma2_s = best["samples"]["sigma2"][:, 0]
    tau2_s = best["samples"]["tau2"][:, 0]

    # Ellipse (semi-axes are the ranges 1/κ; major is the larger one)
    rx_s = 1.0 / np.clip(kx_s, 1e-12, None)
    ry_s = 1.0 / np.clip(ky_s, 1e-12, None)
    semi_major_s = np.maximum(rx_s, ry_s)
    semi_minor_s = np.minimum(rx_s, ry_s)
    ecc_s = np.sqrt(np.clip(1.0 - (semi_minor_s / semi_major_s) ** 2, 0.0, 1.0))

    # κ_eff is largest along the θ direction (⇒ shortest range there) when
    # κ_x > κ_y.  In that case the *major* axis (longest range) is
    # perpendicular to θ.  Canonicalise the reported θ so it always points
    # along the major axis.
    swap = kx_s > ky_s
    theta_major = np.where(swap, np.mod(theta_s + 0.5 * math.pi, math.pi), theta_s)

    theta_mean = _circular_mean_mod_pi(theta_major)
    theta_hdi = _circular_hdi_mod_pi(theta_major)

    # Posterior-mean fit RMSE
    rho_hat = np.zeros_like(morans)
    kx_m, ky_m, th_m = float(np.mean(kx_s)), float(np.mean(ky_s)), float(np.mean(theta_s))
    for ki, phi in enumerate(angles):
        kappa_eff = math.sqrt((kx_m * math.cos(phi - th_m)) ** 2
                              + (ky_m * math.sin(phi - th_m)) ** 2)
        rho_hat[:, ki] = matern_correlation_np(lags, kappa_eff, nu_best)
    mu_hat = float(np.mean(sigma2_s)) * rho_hat
    mu_hat[lags == 0, :] += float(np.mean(tau2_s))
    fit_rmse = float(np.sqrt(np.mean(((morans - mu_hat) ** 2) * valid)))

    converged = (
        all(v < 1.05 for v in best["r_hat"].values())
        and all(v > 100 for v in best["ess"].values())
    )

    hint = _make_dominant_direction_hint(theta_major, ecc_s)

    return AnisotropyResult(
        method="bayes",
        nu=float(nu_best),
        kappa_x_mean=float(np.mean(kx_s)),
        kappa_x_hdi=_hdi(kx_s),
        kappa_y_mean=float(np.mean(ky_s)),
        kappa_y_hdi=_hdi(ky_s),
        theta_mean_rad=float(theta_mean),
        theta_hdi_rad=theta_hdi,
        sigma2_mean=float(np.mean(sigma2_s)),
        tau2_mean=float(np.mean(tau2_s)),
        semi_major_m=float(np.mean(semi_major_s)),
        semi_minor_m=float(np.mean(semi_minor_s)),
        eccentricity_mean=float(np.mean(ecc_s)),
        eccentricity_hdi=_hdi(ecc_s),
        dominant_direction_hint=hint,
        fit_rmse=fit_rmse,
        r_hat=best["r_hat"],
        ess=best["ess"],
        converged=converged,
        n_samples_used=int(len(kx_s)),
        nu_log_marginals={k: float(v["log_marg"]) for k, v in nu_results.items()},
        kappa_x_samples=kx_s,
        kappa_y_samples=ky_s,
        theta_samples_rad=theta_major,
        eccentricity_samples=ecc_s,
    )


# ---------------------------------------------------------------------------
# LSQ fallback
# ---------------------------------------------------------------------------


def fit_anisotropy_lsq(
    lags: np.ndarray,
    angles_rad: np.ndarray,
    morans: np.ndarray,
    n_pairs: Optional[np.ndarray] = None,
    *,
    nu: float = 1.5,
) -> AnisotropyResult:
    """Bounded LSQ fit at fixed ν.  Used as a fallback when NUTS fails."""
    from scipy import optimize

    lags = np.asarray(lags, dtype=np.float64)
    angles = np.asarray(angles_rad, dtype=np.float64)
    morans = np.asarray(morans, dtype=np.float64)
    if n_pairs is None:
        valid = np.ones_like(morans, dtype=bool)
    else:
        valid = np.asarray(n_pairs) > 0

    m_iso = np.array([
        np.nanmean(morans[li, valid[li]]) if valid[li].any() else 0.0
        for li in range(len(lags))
    ])
    k0 = _heuristic_kappa_init(lags, m_iso)
    sigma2_0 = float(np.clip(np.nanmax(morans), 1e-3, 2.0))

    def residual(theta):
        kx, ky, th, sigma2, tau2 = theta
        out = np.zeros_like(morans)
        for ki, phi in enumerate(angles):
            keff = math.sqrt((kx * math.cos(phi - th)) ** 2
                             + (ky * math.sin(phi - th)) ** 2)
            rho = matern_correlation_np(lags, keff, nu)
            out[:, ki] = sigma2 * rho
        out[lags == 0, :] += tau2
        return ((out - morans) * valid).ravel()

    res = optimize.least_squares(
        residual,
        x0=[k0, k0, math.pi / 2, sigma2_0, 0.05],
        bounds=([1e-6, 1e-6, 0.0, 1e-6, 0.0],
                [1e3, 1e3, math.pi, 5.0, 1.0]),
        max_nfev=400,
    )
    kx, ky, th, sigma2, tau2 = res.x
    rx, ry = 1.0 / max(kx, 1e-12), 1.0 / max(ky, 1e-12)
    semi_major = max(rx, ry)
    semi_minor = min(rx, ry)
    ecc = math.sqrt(max(0.0, 1.0 - (semi_minor / semi_major) ** 2))
    if kx > ky:
        th_major = (th + 0.5 * math.pi) % math.pi
    else:
        th_major = th % math.pi
    fit_rmse = float(np.sqrt(np.mean(res.fun ** 2)))
    hint = _make_dominant_direction_hint(
        np.array([th_major]), np.array([ecc])
    )
    return AnisotropyResult(
        method="lsq",
        nu=float(nu),
        kappa_x_mean=float(kx),
        kappa_x_hdi=(float(kx), float(kx)),
        kappa_y_mean=float(ky),
        kappa_y_hdi=(float(ky), float(ky)),
        theta_mean_rad=float(th_major),
        theta_hdi_rad=(float(th_major), float(th_major)),
        sigma2_mean=float(sigma2),
        tau2_mean=float(tau2),
        semi_major_m=float(semi_major),
        semi_minor_m=float(semi_minor),
        eccentricity_mean=float(ecc),
        eccentricity_hdi=(float(ecc), float(ecc)),
        dominant_direction_hint=hint,
        fit_rmse=fit_rmse,
        r_hat={},
        ess={},
        converged=bool(res.success),
        n_samples_used=1,
        nu_log_marginals={float(nu): 0.0},
        kappa_x_samples=np.array([kx]),
        kappa_y_samples=np.array([ky]),
        theta_samples_rad=np.array([th_major]),
        eccentricity_samples=np.array([ecc]),
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def fit_anisotropy(
    lags: np.ndarray,
    angles_rad: np.ndarray,
    morans: np.ndarray,
    n_pairs: Optional[np.ndarray] = None,
    *,
    method: str = "bayes",
    n_samples: int = 400,
    n_warmup: int = 300,
    n_chains: int = 2,
    seed: int = 42,
    kappa_init: Optional[float] = None,
) -> AnisotropyResult:
    """Dispatch Bayesian or LSQ anisotropy fit; auto-fallback on failure."""
    method = method.lower()
    if method == "lsq":
        return fit_anisotropy_lsq(lags, angles_rad, morans, n_pairs)
    if method != "bayes":
        raise ValueError(f"Unknown method {method!r}; expected 'bayes' or 'lsq'")
    try:
        return fit_anisotropy_bayes(
            lags, angles_rad, morans, n_pairs,
            n_samples=n_samples, n_warmup=n_warmup, n_chains=n_chains,
            seed=seed, kappa_init=kappa_init,
        )
    except Exception as exc:
        logger.warning("Bayesian anisotropy fit failed (%s); falling back to LSQ", exc)
        return fit_anisotropy_lsq(lags, angles_rad, morans, n_pairs)
