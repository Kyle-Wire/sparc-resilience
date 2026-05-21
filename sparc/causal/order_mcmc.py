"""Order-MCMC — MCMC over topological orderings for exact BGe structure learning.

MCMC whose state space is the set of all topological orderings (permutations)
of the p variables.  For each ordering σ the implied DAG score is:

    score(σ) = Σ_j  max_{Pa ⊆ pred(j,σ), |Pa| ≤ k_max}  [ local(j ∪ Pa) − local(Pa) ]

where local(S) is the BGe marginal likelihood of the column set S (cached).

Proposals swap two uniformly chosen positions in the ordering.  The Hastings
ratio is 1 (symmetric proposal), so acceptance probability is:

    α = min(1, exp(score(σ') − score(σ)))

Post-burnin: edge inclusion probabilities are accumulated from the current
(ordering, best-parent-sets) state at each accepted sample.

References
----------
Friedman & Koller 2003 — "Being Bayesian About Network Structure"
                          Machine Learning 50, 95–125
"""
from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .mc3 import BGeSuffStats, DAGStructure, PhysicsInformedGraphPrior

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _local_bge_score(
    suff: BGeSuffStats,
    child: int,
    parents: list[int],
) -> float:
    """BGe conditional score for one node:  local(child ∪ Pa) − local(Pa).

    Implements the standard BGe DAG-score decomposition.
    """
    all_cols = sorted([child] + parents)
    if not parents:
        return suff.local_score(all_cols)
    return suff.local_score(all_cols) - suff.local_score(sorted(parents))


def _best_parents_for_node(
    suff: BGeSuffStats,
    child: int,
    predecessors: list[int],
    k_max: int,
) -> tuple[list[int], float]:
    """Find the parent set Pa ⊆ predecessors (|Pa| ≤ k_max) maximising the
    BGe conditional score.  Returns (best_parents, best_score).
    """
    best_score = _local_bge_score(suff, child, [])
    best_pa: list[int] = []

    for k in range(1, min(k_max, len(predecessors)) + 1):
        for pa_tuple in itertools.combinations(predecessors, k):
            s = _local_bge_score(suff, child, list(pa_tuple))
            if s > best_score:
                best_score = s
                best_pa = list(pa_tuple)

    return best_pa, best_score


def _score_ordering(
    suff: BGeSuffStats,
    prior: PhysicsInformedGraphPrior,
    sigma: np.ndarray,
    k_max: int,
) -> tuple[float, list[list[int]]]:
    """Score an ordering and return (total_score, best_parents_per_node).

    total_score = Σ_j BGe_conditional(j, best_Pa_j) + log_prior(G)

    The physics prior is evaluated on the DAG implied by the best parent sets.
    """
    p = len(sigma)
    parents_per_node: list[list[int]] = [[] for _ in range(p)]
    bge_total = 0.0

    for pos, j in enumerate(sigma):
        predecessors = list(sigma[:pos])
        best_pa, best_s = _best_parents_for_node(suff, j, predecessors, k_max)
        parents_per_node[j] = best_pa
        bge_total += best_s

    # Physics prior: log p(G) using the DAG implied by parent sets
    adj = np.zeros((p, p), dtype=np.int8)
    for j in range(p):
        for par in parents_per_node[j]:
            adj[par, j] = 1

    dag = DAGStructure(adj=adj, node_names=list(suff.node_names) if hasattr(suff, "node_names") else [str(i) for i in range(p)])
    log_prior = prior.log_prior(dag, dag.node_names)

    return bge_total + log_prior, parents_per_node


# ---------------------------------------------------------------------------
# Result dataclass — field names match MC3Results for drop-in compatibility
# ---------------------------------------------------------------------------

@dataclass
class OrderMCMCResult:
    """Output of :func:`run_order_mcmc`.  Field names intentionally match
    :class:`~sparc.causal.mc3.MC3Results` for drop-in use in the pipeline.
    """
    best_dag: DAGStructure
    best_score: float
    edge_inclusion_probs: np.ndarray          # (p, p) posterior edge probabilities
    n_accepted: int
    n_total: int
    trace: list[float] = field(default_factory=list)
    physical_plausibility_score: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_order_mcmc(
    data: pd.DataFrame,
    node_names: list[str],
    prior: PhysicsInformedGraphPrior,
    *,
    n_iter: int = 50_000,
    burnin_frac: float = 0.25,
    k_max_parents: int = 3,
    seed: int = 42,
    log_every: int = 2_000,
) -> OrderMCMCResult:
    """Run Order-MCMC structure learning.

    Parameters
    ----------
    data           : observational DataFrame; columns must match node_names.
    node_names     : ordered list of p variable names.
    prior          : physics-informed prior from MC³ pipeline.
    n_iter         : total MCMC proposals (burnin + sampling).
    burnin_frac    : fraction of iterations discarded as burnin.
    k_max_parents  : maximum number of parents per node (controls compute).
                     C(p, k_max) parent-set candidates per node per evaluation.
    seed           : random seed.
    log_every      : log progress every this many iterations.

    Returns
    -------
    OrderMCMCResult with MC3Results-compatible fields.
    """
    rng = np.random.default_rng(seed)

    X = data[node_names].to_numpy(dtype=np.float64)
    suff = BGeSuffStats(X)
    p = suff.p

    burnin = int(n_iter * burnin_frac)
    logger.info(
        "Order-MCMC: p=%d, n_iter=%d, burnin=%d, k_max=%d",
        p, n_iter, burnin, k_max_parents,
    )

    # Attach node_names to suff for use in _score_ordering
    suff.node_names = node_names  # type: ignore[attr-defined]

    # Initialise with a random ordering
    sigma = rng.permutation(p).astype(np.intp)

    current_score, current_parents = _score_ordering(suff, prior, sigma, k_max_parents)

    # Track MAP (best ordering seen)
    best_score = current_score
    best_sigma = sigma.copy()
    best_parents = [list(pa) for pa in current_parents]

    # Edge frequency accumulation (post-burnin)
    edge_freq = np.zeros((p, p), dtype=np.float64)
    n_post_burnin = 0
    n_accepted = 0
    trace: list[float] = []

    for t in range(1, n_iter + 1):
        # Proposal: swap two randomly chosen positions
        i, j = rng.choice(p, size=2, replace=False)
        sigma_prop = sigma.copy()
        sigma_prop[i], sigma_prop[j] = sigma_prop[j], sigma_prop[i]

        prop_score, prop_parents = _score_ordering(suff, prior, sigma_prop, k_max_parents)

        # MH acceptance
        log_accept = prop_score - current_score
        if log_accept >= 0.0 or math.log(rng.random() + 1e-300) < log_accept:
            sigma = sigma_prop
            current_score = prop_score
            current_parents = prop_parents
            n_accepted += 1

            if current_score > best_score:
                best_score = current_score
                best_sigma = sigma.copy()
                best_parents = [list(pa) for pa in current_parents]

        # Post-burnin: accumulate edge indicators
        if t > burnin:
            n_post_burnin += 1
            for j_node in range(p):
                for par in current_parents[j_node]:
                    edge_freq[par, j_node] += 1.0

        if t % log_every == 0:
            accept_rate = n_accepted / t
            trace.append(current_score)
            logger.info(
                "Order-MCMC iter %6d/%d  score=%.2f  accept=%.3f",
                t, n_iter, current_score, accept_rate,
            )

    # Compute edge inclusion probabilities from post-burnin frequencies
    if n_post_burnin > 0:
        edge_probs = edge_freq / n_post_burnin
    else:
        edge_probs = np.zeros((p, p), dtype=np.float64)

    # Build MAP DAG from best ordering
    map_adj = np.zeros((p, p), dtype=np.int8)
    for j_node in range(p):
        for par in best_parents[j_node]:
            map_adj[par, j_node] = 1

    map_dag = DAGStructure(adj=map_adj, node_names=node_names)

    logger.info(
        "Order-MCMC complete: MAP score=%.2f, accept=%.3f, post-burnin samples=%d",
        best_score, n_accepted / max(n_iter, 1), n_post_burnin,
    )

    return OrderMCMCResult(
        best_dag=map_dag,
        best_score=best_score,
        edge_inclusion_probs=edge_probs,
        n_accepted=n_accepted,
        n_total=n_iter,
        trace=trace,
    )
