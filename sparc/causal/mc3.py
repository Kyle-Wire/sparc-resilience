"""
MC³ — Parallel-tempering Markov Chain Monte Carlo over DAG structures.

Uses a physics-informed graph prior (edge probability from spatial
autocorrelation, domain templates, and process-rate compatibility) combined
with the Bayesian Gaussian equivalent (BGe) marginal likelihood score.

Notation follows Madigan & York (1995) for MC-over-model-space,
extended to K parallel chains at different temperatures.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DAGStructure:
    """Lightweight adjacency representation for MC³ proposals."""
    adj: np.ndarray  # (p, p) binary adjacency
    node_names: list[str]

    @property
    def n_nodes(self) -> int:
        return len(self.node_names)

    def to_networkx(self) -> nx.DiGraph:
        G = nx.DiGraph()
        for i, name in enumerate(self.node_names):
            G.add_node(name)
        rows, cols = np.nonzero(self.adj)
        for r, c in zip(rows, cols):
            G.add_edge(self.node_names[r], self.node_names[c])
        return G

    @staticmethod
    def from_networkx(G: nx.DiGraph) -> "DAGStructure":
        names = list(G.nodes())
        idx = {n: i for i, n in enumerate(names)}
        adj = np.zeros((len(names), len(names)), dtype=np.int8)
        for u, v in G.edges():
            adj[idx[u], idx[v]] = 1
        return DAGStructure(adj=adj, node_names=names)

    def copy(self) -> "DAGStructure":
        return DAGStructure(adj=self.adj.copy(), node_names=list(self.node_names))


# ---------------------------------------------------------------------------
# BGe score
# ---------------------------------------------------------------------------

def _bge_local_score(
    data: np.ndarray,
    child_idx: int,
    parent_idxs: list[int],
    alpha_mu: float = 1.0,
    alpha_w: float | None = None,
) -> float:
    """
    Log marginal-likelihood contribution of *child* given *parents*
    under the BGe scoring criterion (Geiger & Heckerman 2002).

    Uses the closed-form Normal-Wishart posterior.
    """
    n, p = data.shape
    k = len(parent_idxs)
    alpha_w = alpha_w or (p + 2.0)

    # Select relevant columns
    cols = parent_idxs + [child_idx]
    X = data[:, cols]
    m = len(cols)

    # Sufficient stats
    X_bar = X.mean(axis=0)
    S = (X - X_bar).T @ (X - X_bar)

    # Prior scatter
    T0 = np.eye(m) * (alpha_mu * (alpha_w - m - 1.0) / (alpha_mu + 1.0))

    Tn = T0 + S + (alpha_mu * n / (alpha_mu + n)) * np.outer(X_bar, X_bar)

    # Log normalising constants
    alpha_n = alpha_w + n
    score = 0.0
    score += 0.5 * m * (math.lgamma((alpha_n - m + 1.0) / 2.0) -
                         math.lgamma((alpha_w - m + 1.0) / 2.0))
    score -= 0.5 * n * m * math.log(math.pi)
    score += 0.5 * m * math.log(alpha_mu / (alpha_mu + n))

    # Determinant ratio
    sign_0, logdet_0 = np.linalg.slogdet(T0)
    sign_n, logdet_n = np.linalg.slogdet(Tn)
    if sign_0 <= 0 or sign_n <= 0:
        return -1e12  # degenerate — assign very low score
    score += 0.5 * alpha_w * logdet_0 - 0.5 * alpha_n * logdet_n

    return float(score)


def bge_score(dag: DAGStructure, data: np.ndarray) -> float:
    """Full DAG BGe score (sum of local scores)."""
    total = 0.0
    for j in range(dag.n_nodes):
        parents = list(np.nonzero(dag.adj[:, j])[0])
        total += _bge_local_score(data, j, parents)
    return total


# ---------------------------------------------------------------------------
# Physics-informed graph prior
# ---------------------------------------------------------------------------

@dataclass
class PhysicsInformedGraphPrior:
    """
    Log-prior over DAG edges incorporating:
      * ``edge_probs``      (p, p) base probability of each edge (from spatial
                            autocorrelation / domain template)
      * ``penalty_acyclic`` penalty per edge to discourage dense graphs
    """
    edge_probs: np.ndarray  # (p, p) in (0, 1)
    penalty_acyclic: float = 1.0

    def log_prior(self, dag: DAGStructure) -> float:
        lp = 0.0
        for i in range(dag.n_nodes):
            for j in range(dag.n_nodes):
                if i == j:
                    continue
                prob = np.clip(self.edge_probs[i, j], 1e-6, 1 - 1e-6)
                if dag.adj[i, j]:
                    lp += math.log(prob) - self.penalty_acyclic
                else:
                    lp += math.log(1.0 - prob)
        return lp

    @classmethod
    def from_config(
        cls,
        node_names: list[str],
        dag_def: dict[str, Any],
        penalty: float = 1.0,
    ) -> "PhysicsInformedGraphPrior":
        """Build from a DAG definition dict (edges → higher prior weight)."""
        p = len(node_names)
        idx = {n: i for i, n in enumerate(node_names)}
        probs = np.full((p, p), 0.1)  # base prior
        for edge in dag_def.get("edges", []):
            i, j = idx.get(edge["parent"]), idx.get(edge["child"])
            if i is not None and j is not None:
                probs[i, j] = 0.8
        np.fill_diagonal(probs, 0.0)
        return cls(edge_probs=probs, penalty_acyclic=penalty)


# ---------------------------------------------------------------------------
# MC³ sampler
# ---------------------------------------------------------------------------

def _propose(dag: DAGStructure, rng: np.random.Generator) -> DAGStructure:
    """Single-edge add / remove / reverse proposal."""
    prop = dag.copy()
    p = dag.n_nodes
    move = rng.choice(["add", "remove", "reverse"])

    if move == "add":
        zeros = list(zip(*np.where(prop.adj == 0)))
        zeros = [(i, j) for i, j in zeros if i != j]
        if not zeros:
            return prop
        i, j = zeros[rng.integers(len(zeros))]
        prop.adj[i, j] = 1
    elif move == "remove":
        ones = list(zip(*np.where(prop.adj == 1)))
        if not ones:
            return prop
        i, j = ones[rng.integers(len(ones))]
        prop.adj[i, j] = 0
    else:  # reverse
        ones = list(zip(*np.where(prop.adj == 1)))
        if not ones:
            return prop
        i, j = ones[rng.integers(len(ones))]
        prop.adj[i, j] = 0
        prop.adj[j, i] = 1

    # Reject if cyclic
    G = nx.DiGraph(prop.adj)
    if not nx.is_directed_acyclic_graph(G):
        return dag.copy()  # stay in place
    return prop


@dataclass
class MC3Results:
    """Collected MC³ output."""
    best_dag: DAGStructure
    best_score: float
    edge_inclusion_probs: np.ndarray  # (p, p) marginal inclusion posterior
    n_accepted: int
    n_total: int
    trace: list[float] = field(default_factory=list)


def run_mc3(
    data: pd.DataFrame,
    node_names: list[str],
    prior: PhysicsInformedGraphPrior,
    *,
    init_dag: DAGStructure | None = None,
    n_iter: int = 10_000,
    n_chains: int = 4,
    temperatures: list[float] | None = None,
    swap_every: int = 10,
    burnin_frac: float = 0.25,
    seed: int = 42,
) -> MC3Results:
    """
    Run parallel-tempering MC³ over DAG space.

    Parameters
    ----------
    data : DataFrame — observational data (columns ⊇ node_names)
    node_names : list[str] — DAG node names
    prior : PhysicsInformedGraphPrior
    init_dag : optional — starting DAG (default: empty)
    n_iter : total MCMC iterations
    n_chains : parallel chains at different temperatures
    temperatures : list of inverse-temperatures (default: [1, 0.5, 0.25, 0.1])
    swap_every : attempt inter-chain swap every N steps
    burnin_frac : fraction of iterations to discard
    seed : random seed
    """
    rng = np.random.default_rng(seed)
    temperatures = temperatures or [1.0, 0.5, 0.25, 0.1]
    temperatures = temperatures[:n_chains]

    p = len(node_names)
    col_idx = [list(data.columns).index(n) for n in node_names]
    data_arr = data.values[:, col_idx].astype(np.float64)

    # Initialise chains
    if init_dag is None:
        init_dag = DAGStructure(adj=np.zeros((p, p), dtype=np.int8), node_names=node_names)

    chains: list[DAGStructure] = [init_dag.copy() for _ in range(n_chains)]
    scores: list[float] = [
        bge_score(chains[k], data_arr) + prior.log_prior(chains[k])
        for k in range(n_chains)
    ]

    # Tracking
    best_dag = chains[0].copy()
    best_score = scores[0]
    edge_counts = np.zeros((p, p), dtype=np.float64)
    n_post_burnin = 0
    n_accepted = 0
    trace: list[float] = []
    burnin_cutoff = int(n_iter * burnin_frac)

    for it in range(n_iter):
        for k in range(n_chains):
            prop = _propose(chains[k], rng)
            prop_score = bge_score(prop, data_arr) + prior.log_prior(prop)

            log_alpha = temperatures[k] * (prop_score - scores[k])
            if math.log(rng.random() + 1e-300) < log_alpha:
                chains[k] = prop
                scores[k] = prop_score
                if k == 0:
                    n_accepted += 1

        # Inter-chain swap
        if it % swap_every == 0 and n_chains > 1:
            k1 = rng.integers(n_chains - 1)
            k2 = k1 + 1
            log_swap = (temperatures[k1] - temperatures[k2]) * (scores[k2] - scores[k1])
            if math.log(rng.random() + 1e-300) < log_swap:
                chains[k1], chains[k2] = chains[k2], chains[k1]
                scores[k1], scores[k2] = scores[k2], scores[k1]

        # Track cold chain
        if scores[0] > best_score:
            best_score = scores[0]
            best_dag = chains[0].copy()

        trace.append(scores[0])

        if it >= burnin_cutoff:
            edge_counts += chains[0].adj.astype(np.float64)
            n_post_burnin += 1

    edge_probs = edge_counts / max(n_post_burnin, 1)

    logger.info(
        "MC³ finished: %d iterations, %d accepted (cold chain), best score=%.2f",
        n_iter, n_accepted, best_score,
    )
    return MC3Results(
        best_dag=best_dag,
        best_score=best_score,
        edge_inclusion_probs=edge_probs,
        n_accepted=n_accepted,
        n_total=n_iter,
        trace=trace,
    )
