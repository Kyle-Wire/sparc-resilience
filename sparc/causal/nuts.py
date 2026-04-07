"""
No-U-Turn Sampler (NUTS) with PyTorch autograd for SPARC V2 Bayesian inference.

Blocked parameter sampling with 5 blocks:
  1. Treatment effect (β)
  2. Spatial random effects (φ)
  3. Process-rate latent (α)
  4. Observation noise (σ²)
  5. Spatial correlation range (ρ)

The likelihood is evaluated through the neural meta-learner via
``SPARCMetaLearner.predict_for_nuts()``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class NUTSBlock:
    """One parameter block for blocked NUTS."""
    name: str
    dim: int
    init: np.ndarray       # (dim,)
    transform: str = "none"  # "none" | "log" | "logit"


@dataclass
class NUTSResults:
    """Collected posterior samples and diagnostics."""
    samples: dict[str, np.ndarray]   # name → (n_samples, dim)
    log_probs: np.ndarray            # (n_samples,)
    acceptance_rate: float
    n_divergences: int
    r_hat: dict[str, np.ndarray]     # per-block convergence
    ess: dict[str, np.ndarray]       # effective sample size


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def _constrain(x: torch.Tensor, transform: str) -> torch.Tensor:
    if transform == "log":
        return x.exp()
    if transform == "logit":
        return torch.sigmoid(x)
    return x


def _log_det_jacobian(x: torch.Tensor, transform: str) -> torch.Tensor:
    if transform == "log":
        return x.sum()
    if transform == "logit":
        return (torch.log(torch.sigmoid(x)) + torch.log(1 - torch.sigmoid(x))).sum()
    return torch.tensor(0.0, device=x.device)


# ---------------------------------------------------------------------------
# Leapfrog integrator
# ---------------------------------------------------------------------------

def _leapfrog(
    theta: torch.Tensor,
    r: torch.Tensor,
    log_prob_fn: Callable[[torch.Tensor], torch.Tensor],
    step_size: float,
    n_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Leapfrog integrator (Hamiltonian dynamics)."""
    theta = theta.detach().requires_grad_(True)
    lp = log_prob_fn(theta)
    grad = torch.autograd.grad(lp, theta)[0]

    r = r + 0.5 * step_size * grad
    for _ in range(n_steps - 1):
        theta = (theta + step_size * r).detach().requires_grad_(True)
        lp = log_prob_fn(theta)
        grad = torch.autograd.grad(lp, theta)[0]
        r = r + step_size * grad
    theta = (theta + step_size * r).detach().requires_grad_(True)
    lp = log_prob_fn(theta)
    grad = torch.autograd.grad(lp, theta)[0]
    r = r + 0.5 * step_size * grad

    return theta.detach(), r.detach()


# ---------------------------------------------------------------------------
# NUTS tree building  (iterative doubling, Hoffman & Gelman 2014)
# ---------------------------------------------------------------------------

def _nuts_step(
    theta: torch.Tensor,
    log_prob_fn: Callable[[torch.Tensor], torch.Tensor],
    step_size: float,
    max_depth: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, bool]:
    """
    One NUTS transition via iterative doubling.

    Returns new theta and whether a divergence was detected.
    """
    d = theta.shape[0]
    r0 = torch.randn(d, dtype=theta.dtype, device=theta.device)

    theta_req = theta.detach().requires_grad_(True)
    lp0 = log_prob_fn(theta_req)
    H0 = float(lp0) - 0.5 * float(r0 @ r0)

    # Slice variable
    log_u = H0 + math.log(rng.random() + 1e-300)  # avoid log(0)

    theta_minus = theta.clone()
    theta_plus = theta.clone()
    r_minus = r0.clone()
    r_plus = r0.clone()

    candidate = theta.clone()
    n_valid = 1
    divergent = False

    for depth in range(max_depth):
        direction = 1 if rng.random() > 0.5 else -1
        eps = direction * step_size

        if direction == -1:
            theta_minus, r_minus = _leapfrog(theta_minus, r_minus, log_prob_fn, eps, 1)
            theta_prime = theta_minus
            r_prime = r_minus
        else:
            theta_plus, r_plus = _leapfrog(theta_plus, r_plus, log_prob_fn, eps, 1)
            theta_prime = theta_plus
            r_prime = r_plus

        theta_req2 = theta_prime.detach().requires_grad_(True)
        lp_prime = log_prob_fn(theta_req2)
        H_prime = float(lp_prime) - 0.5 * float(r_prime @ r_prime)

        # Divergence check
        if H_prime - H0 > 1000.0:
            divergent = True
            break

        # Accept into candidate set
        if H_prime > log_u:
            n_valid += 1
            if rng.random() < 1.0 / n_valid:
                candidate = theta_prime.clone()

        # U-turn check
        delta = theta_plus - theta_minus
        if float(delta @ r_minus) < 0 or float(delta @ r_plus) < 0:
            break

    return candidate, divergent


# ---------------------------------------------------------------------------
# Dual-averaging step-size adaptation (Hoffman & Gelman 2014 §3.2)
# ---------------------------------------------------------------------------

def _dual_average_step_size(
    init_step_size: float,
    log_prob_fn: Callable[[torch.Tensor], torch.Tensor],
    theta0: torch.Tensor,
    n_adapt: int,
    target_accept: float,
    rng: np.random.Generator,
) -> float:
    """Find a reasonable step size via dual averaging."""
    mu = math.log(10 * init_step_size)
    log_eps = math.log(init_step_size)
    log_eps_bar = 0.0
    H_bar = 0.0
    gamma = 0.05
    t0 = 10.0
    kappa = 0.75

    theta = theta0.clone()
    for m in range(1, n_adapt + 1):
        r = torch.randn_like(theta)
        theta_req = theta.detach().requires_grad_(True)
        lp0 = log_prob_fn(theta_req)
        H0 = float(lp0) - 0.5 * float(r @ r)

        theta_prime, r_prime = _leapfrog(theta, r, log_prob_fn, math.exp(log_eps), 1)
        theta_req2 = theta_prime.detach().requires_grad_(True)
        lp1 = log_prob_fn(theta_req2)
        H1 = float(lp1) - 0.5 * float(r_prime @ r_prime)

        delta_H = H1 - H0
        alpha = 1.0 if delta_H > 0 else math.exp(delta_H)
        w = 1.0 / (m + t0)
        H_bar = (1 - w) * H_bar + w * (target_accept - alpha)
        log_eps = mu - math.sqrt(m) / gamma * H_bar
        m_kappa = m ** (-kappa)
        log_eps_bar = m_kappa * log_eps + (1 - m_kappa) * log_eps_bar

    return math.exp(log_eps_bar)


# ---------------------------------------------------------------------------
# Convergence diagnostics
# ---------------------------------------------------------------------------

def _split_r_hat(chain: np.ndarray) -> float:
    """Split-R-hat for a single 1-D chain (n_samples,)."""
    n = len(chain)
    mid = n // 2
    halves = [chain[:mid], chain[mid:2 * mid]]
    if mid < 4:
        return float("nan")
    means = [h.mean() for h in halves]
    vars_ = [h.var(ddof=1) for h in halves]
    W = np.mean(vars_)
    B = mid * np.var(means, ddof=1)
    var_hat = (1 - 1 / mid) * W + B / mid
    return float(np.sqrt(var_hat / max(W, 1e-12)))


def _ess_bulk(chain: np.ndarray) -> float:
    """Bulk effective sample size via autocorrelation."""
    n = len(chain)
    if n < 10:
        return float(n)
    chain_centered = chain - chain.mean()
    fft = np.fft.fft(chain_centered, n=2 * n)
    acf = np.fft.ifft(fft * np.conj(fft)).real[:n]
    acf /= acf[0] + 1e-12
    # Sum pairs
    tau = 1.0
    for i in range(1, n - 1, 2):
        rho_pair = acf[i] + acf[i + 1]
        if rho_pair < 0:
            break
        tau += 2 * rho_pair
    return float(n / tau)


# ---------------------------------------------------------------------------
# Main sampler
# ---------------------------------------------------------------------------

def run_nuts(
    log_prob_fn: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    blocks: list[NUTSBlock],
    *,
    n_samples: int = 2000,
    n_warmup: int = 500,
    max_depth: int = 10,
    init_step_size: float = 0.1,
    target_accept: float = 0.80,
    seed: int = 42,
    device: str = "cpu",
) -> NUTSResults:
    """
    Run blocked NUTS sampling.

    Parameters
    ----------
    log_prob_fn : callable
        Takes ``dict[str, Tensor]`` (constrained params) → scalar log-prob.
    blocks : list[NUTSBlock]
        Defines the 5 parameter blocks with names, dims, initial values.
    n_samples : posterior draws to collect
    n_warmup : adaptation iterations
    max_depth : maximum NUTS tree depth
    init_step_size : initial leapfrog step size
    target_accept : target MH acceptance for step-size tuning
    seed : random seed
    device : torch device string
    """
    rng = np.random.default_rng(seed)
    dtype = torch.float64

    # Flatten blocks into a single vector
    total_dim = sum(b.dim for b in blocks)
    theta = torch.zeros(total_dim, dtype=dtype, device=device)
    slices: dict[str, slice] = {}
    offset = 0
    for b in blocks:
        s = slice(offset, offset + b.dim)
        slices[b.name] = s
        theta[s] = torch.tensor(b.init, dtype=dtype, device=device)
        offset += b.dim

    # Wrap log_prob_fn to operate on flat vector
    def flat_log_prob(flat_theta: torch.Tensor) -> torch.Tensor:
        params = {}
        lp_jac = torch.tensor(0.0, dtype=dtype, device=device)
        for b in blocks:
            raw = flat_theta[slices[b.name]]
            params[b.name] = _constrain(raw, b.transform)
            lp_jac = lp_jac + _log_det_jacobian(raw, b.transform)
        return log_prob_fn(params) + lp_jac

    # Step-size adaptation (use a fraction of warmup budget)
    n_adapt = max(50, n_warmup // 5)
    step_size = _dual_average_step_size(
        init_step_size, flat_log_prob, theta, n_adapt, target_accept, rng,
    )
    logger.info("NUTS adapted step size: %.6f  (n_adapt=%d)", step_size, n_adapt)

    # Warmup transitions: move theta to the typical set before sampling.
    # The dual-averaging above finds the step size at the initial point,
    # but doesn't move theta.  Without warmup transitions the sampler
    # starts far from the posterior mode and every sample diverges.
    n_warmup_trans = n_warmup - n_adapt
    if n_warmup_trans > 0:
        warmup_div = 0
        for _w in range(n_warmup_trans):
            theta, div = _nuts_step(theta, flat_log_prob, step_size, max_depth, rng)
            if div:
                warmup_div += 1
        logger.info(
            "NUTS warmup transitions: %d steps, %d divergences",
            n_warmup_trans, warmup_div,
        )

    # Sampling
    samples: dict[str, list[np.ndarray]] = {b.name: [] for b in blocks}
    log_probs: list[float] = []
    n_divergences = 0

    for i in range(n_samples):
        theta, div = _nuts_step(theta, flat_log_prob, step_size, max_depth, rng)
        if div:
            n_divergences += 1

        theta_req = theta.detach().requires_grad_(True)
        lp = flat_log_prob(theta_req)
        log_probs.append(float(lp))

        for b in blocks:
            constrained = _constrain(theta[slices[b.name]], b.transform)
            samples[b.name].append(constrained.detach().cpu().numpy())

        if (i + 1) % 500 == 0:
            logger.info("NUTS sample %d / %d  (divergences so far: %d)", i + 1, n_samples, n_divergences)

    # Stack
    samples_arr = {k: np.stack(v) for k, v in samples.items()}
    log_probs_arr = np.array(log_probs)

    # Diagnostics
    r_hat: dict[str, np.ndarray] = {}
    ess: dict[str, np.ndarray] = {}
    for b in blocks:
        chain = samples_arr[b.name]  # (n_samples, dim)
        r_hat[b.name] = np.array([_split_r_hat(chain[:, d]) for d in range(b.dim)])
        ess[b.name] = np.array([_ess_bulk(chain[:, d]) for d in range(b.dim)])

    acceptance_rate = 1.0 - n_divergences / max(n_samples, 1)

    logger.info(
        "NUTS finished: %d samples, %d divergences (%.1f%% accept)",
        n_samples, n_divergences, acceptance_rate * 100,
    )
    return NUTSResults(
        samples=samples_arr,
        log_probs=log_probs_arr,
        acceptance_rate=acceptance_rate,
        n_divergences=n_divergences,
        r_hat=r_hat,
        ess=ess,
    )
