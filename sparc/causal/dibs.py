"""DiBS — Differentiable Bayesian Structure Learning.

PyTorch/SVGD implementation of Lorch et al. (NeurIPS 2021).

Each of the K particles holds a latent matrix Z_k (p×p).
The soft adjacency is A_k = sigmoid(Z_k / τ) where τ is temperature.
Particles are jointly optimised via SVGD so the ensemble approximates the
DAG posterior p(G | D) ∝ BGe(G) × prior(G).

Temperature annealing (τ_start → τ_end, geometric schedule) drives the
soft adjacency toward near-binary values, concentrating particles on modes
of the posterior.  The final edge-inclusion probabilities are the particle-
mean soft adjacencies evaluated at τ_end.

References
----------
Lorch et al. 2021 — "DiBS: Differentiable Bayesian Structure Learning"
                    NeurIPS 2021, arXiv:2107.06958
Zheng et al. 2018 — "DAGs with NO TEARS"
Liu & Wang 2016   — "Stein Variational Gradient Descent"
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from .dag_warm_start import _rbf_kernel, _threshold_to_dag
from .mc3 import BGeSuffStats, DAGStructure, PhysicsInformedGraphPrior, notears_h

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dag_bge_score(suff: BGeSuffStats, adj: np.ndarray) -> float:
    """Full decomposed BGe score for a binary DAG adjacency matrix.

    log P(D | G) = Σ_j [ local(j ∪ Pa(j)) − local(Pa(j)) ]
    where local(S) = BGe marginal likelihood of column set S.
    """
    total = 0.0
    p = suff.p
    for j in range(p):
        parents = [i for i in range(p) if adj[i, j]]
        child_and_parents = sorted([j] + parents)
        node_score = suff.local_score(child_and_parents)
        if parents:
            node_score -= suff.local_score(sorted(parents))
        total += node_score
    return total


def _physics_prior_log_soft(
    A: torch.Tensor,
    log_p: torch.Tensor,
    log_1mp: torch.Tensor,
) -> torch.Tensor:
    """Differentiable physics-prior log-probability for soft adjacency A.

    log p_soft(A) = Σ_{i≠j} [A_ij · log(p_ij) + (1−A_ij) · log(1−p_ij)]

    Parameters
    ----------
    A        : (p, p) soft adjacency, values in (0, 1).
    log_p    : (p, p) log of edge-inclusion probabilities (diagonal ignored).
    log_1mp  : (p, p) log of (1 − edge-inclusion probabilities).
    """
    return (A * log_p + (1.0 - A) * log_1mp).sum()


# ---------------------------------------------------------------------------
# Result dataclass — field names match MC3Results for drop-in compatibility
# ---------------------------------------------------------------------------

@dataclass
class DiBSResult:
    """Output of :func:`run_dibs`.  Field names intentionally match
    :class:`~sparc.causal.mc3.MC3Results` for drop-in use in the pipeline.
    """
    best_dag: DAGStructure
    best_score: float
    edge_inclusion_probs: np.ndarray          # (p, p) posterior edge probabilities
    n_accepted: int                           # = n_steps (SVGD steps run)
    n_total: int                              # = n_steps
    trace: list[float] = field(default_factory=list)
    n_particles: int = 0
    physical_plausibility_score: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_dibs(
    data: pd.DataFrame,
    node_names: list[str],
    prior: PhysicsInformedGraphPrior,
    *,
    n_particles: int = 20,
    n_steps: int = 3_000,
    lambda_h: float = 10.0,
    lr: float = 0.005,
    tau_start: float = 1.0,
    tau_end: float = 0.05,
    device: str = "cpu",
    seed: int = 42,
    log_every: int = 500,
) -> DiBSResult:
    """Run DiBS structure learning.

    Parameters
    ----------
    data        : observational DataFrame; columns must match node_names.
    node_names  : ordered list of p variable names.
    prior       : physics-informed prior from MC³ pipeline.
    n_particles : number of SVGD particles (default 20 for good coverage).
    n_steps     : SVGD optimisation steps (default 3000).
    lambda_h    : acyclicity penalty weight (default 10.0, stronger than
                  warm-start to push particles toward valid DAGs).
    lr          : Adam learning rate for the inner update.
    tau_start   : initial temperature for sigmoid(Z/τ) annealing.
    tau_end     : final temperature (near-binary adjacency at convergence).
    device      : torch device string.
    seed        : random seed.
    log_every   : log progress every this many steps.

    Returns
    -------
    DiBSResult with MC3Results-compatible fields.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(int(rng.integers(0, 2**31)))

    dev = torch.device(device)

    # Fit sufficient statistics on data
    X = data[node_names].to_numpy(dtype=np.float64)
    suff = BGeSuffStats(X)
    p = suff.p

    logger.info(
        "DiBS: %d particles × %d steps, p=%d, τ: %.2f→%.2f, λ_h=%.1f",
        n_particles, n_steps, p, tau_start, tau_end, lambda_h,
    )

    # Pre-compute physics-prior log-probability tensors
    edge_p = np.clip(prior.edge_probs, 1e-6, 1.0 - 1e-6)
    log_p = torch.tensor(np.log(edge_p), dtype=torch.float64, device=dev)
    log_1mp = torch.tensor(np.log(1.0 - edge_p), dtype=torch.float64, device=dev)
    # Zero out diagonal (no self-loops)
    diag_mask = torch.eye(p, dtype=torch.float64, device=dev)
    log_p = log_p * (1.0 - diag_mask)
    log_1mp = log_1mp * (1.0 - diag_mask)

    # Initialise K particles as small-noise latent matrices
    Z_list: list[torch.Tensor] = [
        torch.tensor(
            rng.standard_normal((p, p)) * 0.1,
            dtype=torch.float64,
            device=dev,
            requires_grad=True,
        )
        for _ in range(n_particles)
    ]
    opt = torch.optim.Adam(Z_list, lr=lr)

    # Geometric temperature schedule
    tau_ratio = (tau_end / tau_start) ** (1.0 / max(n_steps - 1, 1))

    trace: list[float] = []
    best_particle_score = -math.inf
    best_Z: torch.Tensor | None = None

    for step in range(n_steps):
        tau = tau_start * (tau_ratio ** step)
        opt.zero_grad()

        A_list = [
            torch.sigmoid(Z / tau) * (1.0 - diag_mask)
            for Z in Z_list
        ]

        # Per-particle score = BGe(soft) + log_prior_soft − λ_h · h(A)
        obj_list = [
            suff.bge_score_soft(A)
            + _physics_prior_log_soft(A, log_p, log_1mp)
            - lambda_h * notears_h(A)
            for A in A_list
        ]

        # SVGD update
        K, dK = _rbf_kernel(Z_list)
        score_grads = [
            torch.autograd.grad(obj_list[j], Z_list[j], retain_graph=True)[0].flatten()
            for j in range(n_particles)
        ]
        for i in range(n_particles):
            phi_i = sum(
                K[j, i] * score_grads[j] + dK[i][j]
                for j in range(n_particles)
            ) / n_particles
            Z_list[i].grad = -phi_i.view(p, p)

        opt.step()

        # Track best particle (by soft score)
        best_step_score = max(float(o.item()) for o in obj_list)
        if step % log_every == 0 or step == n_steps - 1:
            trace.append(best_step_score)
            logger.info(
                "DiBS step %4d/%d  τ=%.4f  best_obj=%.2f",
                step, n_steps, tau, best_step_score,
            )
        if best_step_score > best_particle_score:
            best_particle_score = best_step_score
            best_idx = int(
                np.argmax([float(o.item()) for o in obj_list])
            )
            best_Z = Z_list[best_idx].detach().clone()

    # -----------------------------------------------------------------------
    # Extract posterior summaries from final particle ensemble
    # -----------------------------------------------------------------------
    tau_final = tau_end
    with torch.no_grad():
        # Edge inclusion probabilities = particle-mean soft adjacency at τ_end
        soft_mean = torch.zeros(p, p, dtype=torch.float64, device=dev)
        for Z in Z_list:
            soft_mean += torch.sigmoid(Z / tau_final) * (1.0 - diag_mask)
        soft_mean /= n_particles
        edge_probs = soft_mean.cpu().numpy()

    # MAP DAG from the best particle
    assert best_Z is not None
    best_soft = (
        torch.sigmoid(best_Z / tau_final) * (1.0 - diag_mask)
    ).cpu().numpy()
    map_adj = _threshold_to_dag(best_soft)

    # Evaluate true discrete BGe score on MAP DAG
    map_score = _dag_bge_score(suff, map_adj)

    map_dag = DAGStructure(
        adj=map_adj.astype(np.int8),
        node_names=node_names,
    )

    logger.info(
        "DiBS complete: MAP BGe score=%.2f, mean edge density=%.3f",
        map_score, float(edge_probs.mean()),
    )

    return DiBSResult(
        best_dag=map_dag,
        best_score=map_score,
        edge_inclusion_probs=edge_probs,
        n_accepted=n_steps,
        n_total=n_steps,
        trace=trace,
        n_particles=n_particles,
    )
