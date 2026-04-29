"""
No-U-Turn Sampler (NUTS) with PyTorch autograd for SPARC V2 Bayesian inference.

Implements the efficient NUTS algorithm (Algorithm 3) with dual-averaging
step-size adaptation (Algorithm 6) from Hoffman & Gelman (2014).

Supports blocked parameter sampling with arbitrary parameter blocks
defined via ``NUTSBlock``.  Each block specifies a name, dimensionality,
initial value, and optional transform (``"none"`` | ``"log"`` | ``"logit"``).
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
    converged: dict[str, np.ndarray] = field(default_factory=dict)  # per-block bool


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
    inv_mass_diag: torch.Tensor | None = None,
    grad: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Leapfrog integrator with optional gradient caching.

    If *grad* is supplied, the gradient at the initial theta is reused
    instead of recomputed — this halves the per-leaf gradient cost in
    NUTS tree building where the initial gradient was already obtained
    from the previous leapfrog call.

    Returns (theta', r', grad_at_theta', log_prob_at_theta').
    """
    if grad is None:
        theta = theta.detach().requires_grad_(True)
        lp = log_prob_fn(theta)
        grad = torch.autograd.grad(lp, theta)[0]

    r = r + 0.5 * step_size * grad
    for _ in range(n_steps - 1):
        if inv_mass_diag is not None:
            theta = (theta + step_size * inv_mass_diag * r).detach().requires_grad_(True)
        else:
            theta = (theta + step_size * r).detach().requires_grad_(True)
        lp = log_prob_fn(theta)
        grad = torch.autograd.grad(lp, theta)[0]
        r = r + step_size * grad
    if inv_mass_diag is not None:
        theta = (theta + step_size * inv_mass_diag * r).detach().requires_grad_(True)
    else:
        theta = (theta + step_size * r).detach().requires_grad_(True)
    lp = log_prob_fn(theta)
    grad = torch.autograd.grad(lp, theta)[0]
    r = r + 0.5 * step_size * grad

    return theta.detach(), r.detach(), grad.detach(), float(lp.detach())


def _kinetic_energy(r: torch.Tensor, inv_mass_diag: torch.Tensor | None) -> float:
    """Kinetic energy: 0.5 * r^T M^{-1} r."""
    if inv_mass_diag is not None:
        return 0.5 * float((r * r * inv_mass_diag).sum())
    return 0.5 * float(r @ r)


# ---------------------------------------------------------------------------
# Algorithm 4: FindReasonableEpsilon  (Hoffman & Gelman 2014)
# ---------------------------------------------------------------------------

def _find_reasonable_epsilon(
    theta: torch.Tensor,
    log_prob_fn: Callable,
    rng: np.random.Generator,
    inv_mass_diag: torch.Tensor | None = None,
) -> float:
    """Algorithm 4 from Hoffman & Gelman 2014 — heuristic initial step size.

    Repeatedly doubles or halves ε until the acceptance probability of a
    single leapfrog step crosses 0.5.  The resulting ε is typically small
    enough for accurate integration but large enough to avoid wasting
    computation.
    """
    eps = 1.0

    # Sample momentum from mass matrix
    if inv_mass_diag is not None:
        mass_diag = 1.0 / inv_mass_diag
        r = torch.randn_like(theta) * mass_diag.sqrt()
    else:
        r = torch.randn_like(theta)

    # Initial Hamiltonian
    theta_req = theta.detach().requires_grad_(True)
    lp0 = log_prob_fn(theta_req)
    grad0 = torch.autograd.grad(lp0, theta_req)[0].detach()
    H0 = float(lp0.detach()) - _kinetic_energy(r, inv_mass_diag)

    # One leapfrog step at ε = 1
    theta_prime, r_prime, _, lp1 = _leapfrog(
        theta, r, log_prob_fn, eps, 1,
        inv_mass_diag=inv_mass_diag, grad=grad0,
    )
    H1 = lp1 - _kinetic_energy(r_prime, inv_mass_diag)

    # Direction: double (a=+1) if accept prob > 0.5, halve (a=−1) otherwise
    a = 1.0 if (H1 - H0) > math.log(0.5) else -1.0

    while True:
        theta_prime, r_prime, _, lp1 = _leapfrog(
            theta, r, log_prob_fn, eps, 1,
            inv_mass_diag=inv_mass_diag, grad=grad0,
        )
        H1 = lp1 - _kinetic_energy(r_prime, inv_mass_diag)
        if a * (H1 - H0) <= -a * math.log(2):
            break
        eps = (2.0 ** a) * eps
        if eps > 1e6 or eps < 1e-10:  # safety bounds
            break

    return eps


# ---------------------------------------------------------------------------
# NUTS tree building  (Algorithm 3 & 6, Hoffman & Gelman 2014)
# ---------------------------------------------------------------------------

_DELTA_MAX = 1000.0  # max allowed energy error before flagging divergence


def _compute_hamiltonian(
    theta: torch.Tensor,
    r: torch.Tensor,
    log_prob_fn: Callable[[torch.Tensor], torch.Tensor],
    inv_mass_diag: torch.Tensor | None,
) -> float:
    """Compute H = log p(θ) − 0.5 rᵀM⁻¹r (negative energy)."""
    with torch.no_grad():
        lp = log_prob_fn(theta.detach())
    return float(lp) - _kinetic_energy(r, inv_mass_diag)


def _build_tree(
    theta: torch.Tensor,
    r: torch.Tensor,
    grad: torch.Tensor,
    log_u: float,
    direction: int,
    depth: int,
    step_size: float,
    log_prob_fn: Callable[[torch.Tensor], torch.Tensor],
    H0: float,
    rng: np.random.Generator,
    inv_mass_diag: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,  # θ⁻, r⁻, grad⁻
    torch.Tensor, torch.Tensor, torch.Tensor,  # θ⁺, r⁺, grad⁺
    torch.Tensor,                  # θ' (candidate sample from this subtree)
    int,                           # n' (number of valid states in subtree)
    bool,                          # s' (no stopping criterion met)
    float, int,                    # α' (sum of accept probs), n_α' (leaf count)
]:
    """
    Recursive tree builder for efficient NUTS (Algorithm 3/6, Hoffman & Gelman 2014).

    Builds a balanced binary tree of depth *depth* by taking leapfrog steps
    in *direction* (±1).  At depth 0 a single leapfrog step is taken (base case).
    At depth j > 0, two subtrees of depth j−1 are built; their candidates are
    combined with probability n″/(n′+n″) (multinomial sampling within the
    subtree, eq. 12), and U-turn / divergence checks are applied to every
    subtree (eq. 9).

    *grad* is the gradient ∇log p at *theta*, cached from the previous leapfrog
    call to avoid redundant gradient evaluations.

    Returns
    -------
    θ⁻, r⁻, grad⁻ : leftmost (backward) leaf position/momentum/gradient
    θ⁺, r⁺, grad⁺ : rightmost (forward) leaf position/momentum/gradient
    θ'      : candidate position drawn from valid states in this subtree
    n'      : count of states satisfying the slice criterion (u ≤ exp H)
    s'      : True if no stopping criterion was triggered in this subtree
    α'      : sum of min(1, exp(H − H₀)) over leaf nodes (for dual averaging)
    n_α'    : number of leaf nodes evaluated (for dual averaging)
    """
    eps = direction * step_size

    if depth == 0:
        # ---- Base case: single leapfrog step ----
        theta_prime, r_prime, grad_prime, lp_prime = _leapfrog(
            theta, r, log_prob_fn, eps, 1,
            inv_mass_diag=inv_mass_diag,
            grad=grad,
        )
        H_prime = lp_prime - _kinetic_energy(r_prime, inv_mass_diag)

        # Slice membership: n' = I[u ≤ exp(H')]  i.e.  log_u ≤ H'
        n_prime = 1 if log_u <= H_prime else 0

        # Still-valid flag: stop if energy error too large
        s_prime = H_prime > log_u - _DELTA_MAX

        # Acceptance statistic for dual averaging (Algorithm 6)
        alpha_prime = min(1.0, math.exp(min(H_prime - H0, 0.0)))

        return (
            theta_prime, r_prime, grad_prime,  # θ⁻, r⁻, grad⁻
            theta_prime, r_prime, grad_prime,  # θ⁺, r⁺, grad⁺ (same for leaf)
            theta_prime,             # θ'
            n_prime,                 # n'
            s_prime,                 # s'
            alpha_prime, 1,          # α', n_α'
        )

    # ---- Recursion: build first (inner) subtree of depth j−1 ----
    (theta_minus, r_minus, grad_minus,
     theta_plus, r_plus, grad_plus,
     theta_prime, n_prime, s_prime,
     alpha_prime, n_alpha_prime) = _build_tree(
        theta, r, grad, log_u, direction, depth - 1, step_size,
        log_prob_fn, H0, rng, inv_mass_diag,
    )

    if s_prime:
        # ---- Build second (outer) subtree of depth j−1 ----
        if direction == -1:
            # Extend backward: start from current θ⁻
            (theta_minus, r_minus, grad_minus,
             _, _, _,
             theta_dblprime, n_dblprime, s_dblprime,
             alpha_dblprime, n_alpha_dblprime) = _build_tree(
                theta_minus, r_minus, grad_minus, log_u, direction, depth - 1,
                step_size, log_prob_fn, H0, rng, inv_mass_diag,
            )
        else:
            # Extend forward: start from current θ⁺
            (_, _, _,
             theta_plus, r_plus, grad_plus,
             theta_dblprime, n_dblprime, s_dblprime,
             alpha_dblprime, n_alpha_dblprime) = _build_tree(
                theta_plus, r_plus, grad_plus, log_u, direction, depth - 1,
                step_size, log_prob_fn, H0, rng, inv_mass_diag,
            )

        # Multinomial candidate selection within subtree (eq. 12):
        # pick new candidate with probability n″/(n′+n″)
        total_n = n_prime + n_dblprime
        if total_n > 0 and rng.random() < n_dblprime / max(total_n, 1):
            theta_prime = theta_dblprime

        # Accumulate acceptance statistics
        alpha_prime = alpha_prime + alpha_dblprime
        n_alpha_prime = n_alpha_prime + n_alpha_dblprime
        n_prime = total_n

        # U-turn check on the full subtree (eq. 9)
        delta = theta_plus - theta_minus
        s_prime = s_dblprime and (float(delta @ r_minus) >= 0) and (float(delta @ r_plus) >= 0)

    return (
        theta_minus, r_minus, grad_minus,
        theta_plus, r_plus, grad_plus,
        theta_prime,
        n_prime,
        s_prime,
        alpha_prime, n_alpha_prime,
    )


def _nuts_step(
    theta: torch.Tensor,
    log_prob_fn: Callable[[torch.Tensor], torch.Tensor],
    step_size: float,
    max_depth: int,
    rng: np.random.Generator,
    inv_mass_diag: torch.Tensor | None = None,
) -> tuple[torch.Tensor, bool, float]:
    """
    One NUTS transition via recursive tree doubling (Algorithm 3/6).

    Returns (new theta, divergent flag, mean acceptance probability).
    """
    d = theta.shape[0]

    # Compute initial gradient (used for first leapfrog in tree)
    theta_req = theta.detach().requires_grad_(True)
    lp0 = log_prob_fn(theta_req)
    grad0 = torch.autograd.grad(lp0, theta_req)[0].detach()
    lp0_val = float(lp0.detach())

    # Sample momentum from mass matrix
    if inv_mass_diag is not None:
        mass_diag = 1.0 / inv_mass_diag
        r0 = torch.randn(d, dtype=theta.dtype, device=theta.device) * mass_diag.sqrt()
    else:
        r0 = torch.randn(d, dtype=theta.dtype, device=theta.device)

    # Initial Hamiltonian (using pre-computed log prob)
    H0 = lp0_val - _kinetic_energy(r0, inv_mass_diag)

    # Slice variable: log(u) ~ log(Uniform(0, exp(H0)))
    log_u = H0 + math.log(rng.random() + 1e-300)

    # Initialise tree endpoints and candidate
    theta_minus = theta.clone()
    theta_plus = theta.clone()
    r_minus = r0.clone()
    r_plus = r0.clone()
    grad_minus = grad0.clone()
    grad_plus = grad0.clone()

    candidate = theta.clone()
    n_valid = 1        # total valid states across all doublings
    s = True           # no stopping criterion met
    divergent = False
    alpha_total = 0.0  # accumulated acceptance prob (for dual averaging)
    n_alpha_total = 0  # total leaf count

    for depth in range(max_depth):
        if not s:
            break

        # Choose a direction uniformly at random
        direction = 1 if rng.random() > 0.5 else -1

        # Build a new subtree of depth *depth* in the chosen direction
        if direction == -1:
            (theta_minus, r_minus, grad_minus,
             _, _, _,
             theta_prime, n_prime, s_prime,
             alpha_prime, n_alpha_prime) = _build_tree(
                theta_minus, r_minus, grad_minus, log_u, direction, depth,
                step_size, log_prob_fn, H0, rng, inv_mass_diag,
            )
        else:
            (_, _, _,
             theta_plus, r_plus, grad_plus,
             theta_prime, n_prime, s_prime,
             alpha_prime, n_alpha_prime) = _build_tree(
                theta_plus, r_plus, grad_plus, log_u, direction, depth,
                step_size, log_prob_fn, H0, rng, inv_mass_diag,
            )

        # Efficient candidate selection (Algorithm 3, eq. 10):
        # Accept new subtree's candidate with probability min(1, n'/n)
        if s_prime and n_prime > 0:
            accept_prob_candidate = min(1.0, n_prime / max(n_valid, 1))
            if rng.random() < accept_prob_candidate:
                candidate = theta_prime.clone()

        # Accumulate acceptance stats across all doublings
        alpha_total += alpha_prime
        n_alpha_total += n_alpha_prime

        # Update total valid count
        n_valid += n_prime

        # Check for divergence (any subtree flagged it via s'=False due to Δmax)
        if not s_prime and n_alpha_prime > 0 and (alpha_prime / n_alpha_prime) < 1e-10:
            divergent = True

        # U-turn check on the full trajectory
        delta = theta_plus - theta_minus
        s = s_prime and (float(delta @ r_minus) >= 0) and (float(delta @ r_plus) >= 0)

    # Mean acceptance probability averaged over all leaf nodes (Algorithm 6)
    mean_accept = alpha_total / max(n_alpha_total, 1)
    return candidate, divergent, mean_accept


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
    inv_mass_diag: torch.Tensor | None = None,
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
        if inv_mass_diag is not None:
            mass_diag = 1.0 / inv_mass_diag
            r = torch.randn_like(theta) * mass_diag.sqrt()
        else:
            r = torch.randn_like(theta)

        # Compute H0 using pre-computed lp
        theta_req = theta.detach().requires_grad_(True)
        lp0 = log_prob_fn(theta_req)
        grad0 = torch.autograd.grad(lp0, theta_req)[0].detach()
        H0 = float(lp0.detach()) - _kinetic_energy(r, inv_mass_diag)

        theta_prime, r_prime, _, lp_prime = _leapfrog(
            theta, r, log_prob_fn, math.exp(log_eps), 1,
            inv_mass_diag=inv_mass_diag,
            grad=grad0,
        )
        H1 = lp_prime - _kinetic_energy(r_prime, inv_mass_diag)

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
    init_step_size: float = 1.0,
    target_accept: float = 0.60,
    seed: int = 42,
    device: str = "cpu",
    param_scales: dict[str, np.ndarray] | None = None,
    thin: int = 1,
) -> NUTSResults:
    """
    Run blocked NUTS sampling.

    Parameters
    ----------
    log_prob_fn : callable
        Takes ``dict[str, Tensor]`` (constrained params) → scalar log-prob.
    blocks : list[NUTSBlock]
        Parameter blocks with names, dims, initial values, and transforms.
    n_samples : posterior draws to collect
    n_warmup : adaptation iterations
    max_depth : maximum NUTS tree depth
    init_step_size : initial leapfrog step size
    target_accept : target MH acceptance for step-size tuning
    seed : random seed
    device : torch device string
    param_scales : optional dict mapping block name → array of per-dim scales.
        NUTS samples in *scaled* space where θ_scaled = θ / scale, making all
        dimensions O(1).  Samples are back-transformed before storage.
    thin : int, default=1
        Keep every ``thin``-th post-warmup sample to reduce peak memory of the
        accumulated trace.  ``thin=2`` halves memory at the cost of slightly
        coarser posterior summaries (still unbiased).  Must be ≥1.
    """
    rng = np.random.default_rng(seed)
    dtype = torch.float64

    thin = max(1, int(thin))

    # Build per-dimension scale vector for param_scales rescaling
    total_dim = sum(b.dim for b in blocks)
    scale_vec: torch.Tensor | None = None
    if param_scales:
        _sv = np.ones(total_dim)
        offset_tmp = 0
        for b in blocks:
            if b.name in param_scales:
                s_arr = np.asarray(param_scales[b.name], dtype=np.float64)
                if s_arr.shape == ():
                    s_arr = np.full(b.dim, float(s_arr))
                _sv[offset_tmp:offset_tmp + b.dim] = s_arr
            offset_tmp += b.dim
        scale_vec = torch.tensor(_sv, dtype=dtype, device=device)
        logger.info("NUTS param_scales applied: %s", _sv.tolist())

    # Flatten blocks into a single vector
    theta = torch.zeros(total_dim, dtype=dtype, device=device)
    slices: dict[str, slice] = {}
    offset = 0
    for b in blocks:
        s = slice(offset, offset + b.dim)
        slices[b.name] = s
        init_val = torch.tensor(b.init, dtype=dtype, device=device)
        if scale_vec is not None:
            theta[s] = init_val / scale_vec[s]  # initialise in scaled space
        else:
            theta[s] = init_val
        offset += b.dim

    # Wrap log_prob_fn to operate on flat vector (in scaled space)
    def flat_log_prob(flat_theta: torch.Tensor) -> torch.Tensor:
        params = {}
        lp_jac = torch.tensor(0.0, dtype=dtype, device=device)
        for b in blocks:
            raw = flat_theta[slices[b.name]]
            if scale_vec is not None:
                raw = raw * scale_vec[slices[b.name]]  # back to original space
            params[b.name] = _constrain(raw, b.transform)
            lp_jac = lp_jac + _log_det_jacobian(raw, b.transform)
        return log_prob_fn(params) + lp_jac

    # FIX A4: Mass matrix adaptation during warmup
    # Phase 1: FindReasonableEpsilon (Algorithm 4) + dual averaging
    inv_mass_diag: torch.Tensor | None = None
    init_eps = _find_reasonable_epsilon(theta, flat_log_prob, rng, inv_mass_diag)
    logger.info("NUTS FindReasonableEpsilon: %.6f", init_eps)
    n_adapt = max(50, n_warmup // 5)
    step_size = _dual_average_step_size(
        init_eps, flat_log_prob, theta, n_adapt, target_accept, rng,
        inv_mass_diag=inv_mass_diag,
    )
    print(f"  NUTS: finding initial step size ({n_adapt} adaptation steps, ε₀={init_eps:.4f})...", flush=True)
    logger.info("NUTS initial step size: %.6f  (n_adapt=%d, eps0=%.6f)", step_size, n_adapt, init_eps)

    # Phase 2: warmup transitions to estimate mass matrix
    n_warmup_trans = n_warmup - n_adapt
    warmup_samples_for_mass: list[np.ndarray] = []
    if n_warmup_trans > 0:
        print(f"  NUTS: warmup phase ({n_warmup_trans} transitions)...", flush=True)
        warmup_div = 0
        for _w in range(n_warmup_trans):
            theta, div, _ = _nuts_step(
                theta, flat_log_prob, step_size, max_depth, rng,
                inv_mass_diag=inv_mass_diag,
            )
            if div:
                warmup_div += 1
            warmup_samples_for_mass.append(theta.detach().cpu().numpy())
            if (_w + 1) % 25 == 0:
                print(f"    warmup {_w + 1} / {n_warmup_trans}", flush=True)
        logger.info(
            "NUTS warmup transitions: %d steps, %d divergences",
            n_warmup_trans, warmup_div,
        )

        # Estimate diagonal mass matrix from warmup samples
        if len(warmup_samples_for_mass) >= 20:
            warmup_arr = np.stack(warmup_samples_for_mass)
            var_est = np.var(warmup_arr, axis=0)
            # Regularize: don't let any variance get too small or too large
            var_est = np.clip(var_est, 0.01, 1e6)
            inv_mass_diag = torch.tensor(
                1.0 / var_est, dtype=dtype, device=device,
            )
            logger.info(
                "NUTS mass matrix adapted: var range [%.4e, %.4e]",
                var_est.min(), var_est.max(),
            )

            # Re-adapt step size with FindReasonableEpsilon + dual averaging
            init_eps_mm = _find_reasonable_epsilon(theta, flat_log_prob, rng, inv_mass_diag)
            logger.info("NUTS FindReasonableEpsilon (mass matrix): %.6f", init_eps_mm)
            step_size = _dual_average_step_size(
                init_eps_mm, flat_log_prob, theta,
                min(n_adapt, 500), target_accept, rng,
                inv_mass_diag=inv_mass_diag,
            )
            logger.info("NUTS re-adapted step size with mass matrix: %.6f", step_size)

    # Sampling
    print(f"  NUTS: sampling phase ({n_samples} draws)...", flush=True)
    samples: dict[str, list[np.ndarray]] = {b.name: [] for b in blocks}
    log_probs: list[float] = []
    n_divergences = 0
    sum_accept_prob = 0.0

    for i in range(n_samples):
        theta, div, accept_prob = _nuts_step(
            theta, flat_log_prob, step_size, max_depth, rng,
            inv_mass_diag=inv_mass_diag,
        )
        if div:
            n_divergences += 1
        sum_accept_prob += accept_prob

        if thin > 1 and (i % thin) != 0:
            # Skip storing this draw to reduce peak memory of the trace.
            if (i + 1) % 100 == 0 or (i + 1) == n_samples:
                msg = f"  NUTS sample {i + 1} / {n_samples}  (divergences: {n_divergences}, accept: {sum_accept_prob / (i + 1):.1%})"
                print(msg, flush=True)
                logger.info("NUTS sample %d / %d  (divergences so far: %d)", i + 1, n_samples, n_divergences)
            continue

        theta_req = theta.detach().requires_grad_(True)
        lp = flat_log_prob(theta_req)
        log_probs.append(float(lp))

        for b in blocks:
            raw = theta[slices[b.name]]
            if scale_vec is not None:
                raw = raw * scale_vec[slices[b.name]]  # back-transform
            constrained = _constrain(raw, b.transform)
            samples[b.name].append(constrained.detach().cpu().numpy())

        if (i + 1) % 100 == 0 or (i + 1) == n_samples:
            msg = f"  NUTS sample {i + 1} / {n_samples}  (divergences: {n_divergences}, accept: {sum_accept_prob / (i + 1):.1%})"
            print(msg, flush=True)
            logger.info("NUTS sample %d / %d  (divergences so far: %d)", i + 1, n_samples, n_divergences)

    # Stack
    samples_arr = {k: np.stack(v) for k, v in samples.items()}
    log_probs_arr = np.array(log_probs)

    # Diagnostics
    r_hat: dict[str, np.ndarray] = {}
    ess: dict[str, np.ndarray] = {}
    converged: dict[str, np.ndarray] = {}
    for b in blocks:
        chain = samples_arr[b.name]  # (n_samples, dim)
        rh = np.array([_split_r_hat(chain[:, d]) for d in range(b.dim)])
        es = np.array([_ess_bulk(chain[:, d]) for d in range(b.dim)])
        r_hat[b.name] = rh
        ess[b.name] = es
        # Fix 3: convergence requires BOTH R̂ < 1.05 AND ESS > 400
        conv = (rh < 1.05) & (es > 400)
        converged[b.name] = conv
        for d in range(b.dim):
            if not conv[d]:
                reasons = []
                if rh[d] >= 1.05:
                    reasons.append(f"R̂={rh[d]:.3f}≥1.05")
                if es[d] <= 400:
                    reasons.append(f"ESS={es[d]:.0f}≤400")
                logger.warning(
                    "NUTS convergence FAILED for %s[%d]: %s",
                    b.name, d, ", ".join(reasons),
                )
            else:
                logger.info(
                    "NUTS converged for %s[%d]: R̂=%.3f, ESS=%.0f",
                    b.name, d, rh[d], es[d],
                )

    # FIX A3: use actual mean acceptance probability, not divergence proxy
    acceptance_rate = sum_accept_prob / max(n_samples, 1)

    logger.info(
        "NUTS finished: %d samples, %d divergences (%.1f%% mean accept)",
        n_samples, n_divergences, acceptance_rate * 100,
    )
    return NUTSResults(
        samples=samples_arr,
        log_probs=log_probs_arr,
        acceptance_rate=acceptance_rate,
        n_divergences=n_divergences,
        r_hat=r_hat,
        ess=ess,
        converged=converged,
    )
