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

import torch

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
# BGe score (with precomputed sufficient statistics for O(m³) scoring)
# ---------------------------------------------------------------------------

class BGeSuffStats:
    """Precomputed sufficient statistics for O(m³) BGe scoring.

    Instead of recomputing S = (X-X̄)ᵀ(X-X̄) from scratch (O(n·m²))
    every iteration, precompute the full p×p scatter matrix once and
    slice into it for each subset of columns (O(m²)).

    Two optimizations over the pure-NumPy original:

    1. **Score cache** — results are memoized by the sorted tuple of column
       indices.  For p=7 the cache has at most 127 unique keys; even for
       p=20 it stays below 1 M entries (~40 MB).  Eliminates redundant
       slogdet calls across the MCMC chain, where the same parent sets
       recur thousands of times.

    2. **Torch back-end** — ``self._scatter_t`` and ``self._col_means_t``
       are ``torch.float64`` tensors resident on *device* (default ``"cpu"``;
       pass ``device="cuda"`` on a GPU machine for p>15).  Matrix ops use
       ``torch.linalg.slogdet``, which is batched in ``local_score_batch``.
    """

    def __init__(self, data: np.ndarray, alpha_mu: float = 1.0,
                 alpha_w: float | None = None, device: str = "cpu"):
        self.n, self.p = data.shape
        self.alpha_mu = alpha_mu
        self.alpha_w = alpha_w or (self.p + 2.0)
        self.device = torch.device(device)

        # Precompute full sufficient stats (one-time O(n·p²))
        self.col_means = data.mean(axis=0)  # (p,) numpy — kept for backward compat
        centered = data - self.col_means
        self.scatter = centered.T @ centered  # (p, p) numpy — kept for backward compat

        # PyTorch tensors for fast GPU/CPU scoring via torch.linalg
        self._scatter_t = torch.tensor(
            self.scatter, dtype=torch.float64, device=self.device
        )
        self._col_means_t = torch.tensor(
            self.col_means, dtype=torch.float64, device=self.device
        )

        # Score cache: key = sorted tuple of column indices → log-marginal-likelihood.
        # BGe marginal likelihood is permutation-invariant in columns, so
        # tuple(sorted(cols)) is a valid key regardless of parent ordering.
        self._score_cache: dict[tuple[int, ...], float] = {}

    def local_score(self, cols: list[int]) -> float:
        """Log marginal likelihood for a subset of columns (cached)."""
        key = tuple(sorted(cols))
        cached = self._score_cache.get(key)
        if cached is not None:
            return cached

        m = len(cols)
        cols_t = torch.tensor(cols, dtype=torch.long, device=self.device)
        X_bar = self._col_means_t[cols_t]
        S = self._scatter_t[cols_t][:, cols_t]

        diag_val = self.alpha_mu * (self.alpha_w - m - 1.0) / (self.alpha_mu + 1.0)
        T0 = torch.eye(m, dtype=torch.float64, device=self.device) * diag_val
        Tn = T0 + S + (self.alpha_mu * self.n / (self.alpha_mu + self.n)) * torch.outer(X_bar, X_bar)

        alpha_n = self.alpha_w + self.n
        sign_0, logdet_0 = torch.linalg.slogdet(T0)
        sign_n, logdet_n = torch.linalg.slogdet(Tn)
        if sign_0.item() <= 0 or sign_n.item() <= 0:
            result = -1e12
        else:
            result = (
                0.5 * m * (math.lgamma((alpha_n - m + 1.0) / 2.0) -
                           math.lgamma((self.alpha_w - m + 1.0) / 2.0))
                - 0.5 * self.n * m * math.log(math.pi)
                + 0.5 * m * math.log(self.alpha_mu / (self.alpha_mu + self.n))
                + 0.5 * self.alpha_w * logdet_0.item()
                - 0.5 * alpha_n * logdet_n.item()
            )
        self._score_cache[key] = result
        return result

    def local_score_batch(self, cols_list: list[list[int]]) -> list[float]:
        """Score multiple column sets via cache + batched torch.linalg.slogdet.

        Uncached sets of the same size *m* are grouped and scored with a
        single batched slogdet call (shape B×m×m), eliminating per-call
        Python overhead for the dominant linear-algebra cost.  All results
        are written into the cache for subsequent ``local_score`` lookups.
        """
        from collections import defaultdict

        results: list[float | None] = [None] * len(cols_list)
        uncached: list[tuple[int, tuple[int, ...], list[int]]] = []

        for i, cols in enumerate(cols_list):
            key = tuple(sorted(cols))
            cached = self._score_cache.get(key)
            if cached is not None:
                results[i] = cached
            else:
                uncached.append((i, key, list(cols)))

        if not uncached:
            return results  # type: ignore[return-value]

        # Group uncached entries by subset size for batched slogdet
        by_size: dict[int, list[tuple[int, tuple[int, ...], list[int]]]] = defaultdict(list)
        for item in uncached:
            by_size[len(item[2])].append(item)

        for m, items in by_size.items():
            alpha_n = self.alpha_w + self.n
            diag_val = self.alpha_mu * (self.alpha_w - m - 1.0) / (self.alpha_mu + 1.0)
            scalar = (
                0.5 * m * (math.lgamma((alpha_n - m + 1.0) / 2.0) -
                           math.lgamma((self.alpha_w - m + 1.0) / 2.0))
                - 0.5 * self.n * m * math.log(math.pi)
                + 0.5 * m * math.log(self.alpha_mu / (self.alpha_mu + self.n))
            )
            B = len(items)
            T0_batch = torch.zeros(B, m, m, dtype=torch.float64, device=self.device)
            Tn_batch = torch.zeros(B, m, m, dtype=torch.float64, device=self.device)
            for b, (_idx, _key, cols) in enumerate(items):
                cols_t = torch.tensor(cols, dtype=torch.long, device=self.device)
                X_bar = self._col_means_t[cols_t]
                S = self._scatter_t[cols_t][:, cols_t]
                T0 = torch.eye(m, dtype=torch.float64, device=self.device) * diag_val
                Tn = T0 + S + (self.alpha_mu * self.n / (self.alpha_mu + self.n)) * torch.outer(X_bar, X_bar)
                T0_batch[b] = T0
                Tn_batch[b] = Tn

            signs_0, logdets_0 = torch.linalg.slogdet(T0_batch)
            signs_n, logdets_n = torch.linalg.slogdet(Tn_batch)

            for b, (idx, key, _cols) in enumerate(items):
                if signs_0[b].item() <= 0 or signs_n[b].item() <= 0:
                    val = -1e12
                else:
                    val = (
                        scalar
                        + 0.5 * self.alpha_w * logdets_0[b].item()
                        - 0.5 * alpha_n * logdets_n[b].item()
                    )
                self._score_cache[key] = val
                results[idx] = val

        return results  # type: ignore[return-value]

    def clear_cache(self) -> None:
        """Evict all memoized scores (e.g. after a data change)."""
        self._score_cache.clear()

    def bge_score_soft(self, A: torch.Tensor) -> torch.Tensor:
        """Differentiable BGe score for a soft adjacency matrix A ∈ [0,1]^(p×p).

        Uses the linear SEM parameterisation: residuals at node j are
        (I - A)[:, j] applied to the data, giving residual scatter
        R = (I-A)^T S (I-A).  The score is proportional to
        -n/2 * Σ_j log(R_jj).  Fully differentiable through torch ops
        so autograd can flow back to A during gradient warm-start.
        """
        p = A.shape[0]
        S = self._scatter_t.to(dtype=A.dtype, device=A.device)
        I = torch.eye(p, dtype=A.dtype, device=A.device)
        Im_A = I - A                        # (I - A)
        R = Im_A.T @ S @ Im_A               # (p, p) residual scatter
        diag_R = R.diagonal().clamp(min=1e-8)
        return -0.5 * self.n * diag_R.log().sum()


def notears_h(A: torch.Tensor) -> torch.Tensor:
    """NOTEARS acyclicity constraint: h(A) = tr(expm(A ⊙ A)) - p.

    h(A) == 0 iff A represents a DAG.  Used as a soft penalty during
    gradient warm-start so particles are initialised near acyclic solutions.

    Reference: Zheng et al. (2018) — DAGs with NO TEARS.
    """
    M = A * A                                # element-wise square
    E = torch.linalg.matrix_exp(M)          # expm(A ⊙ A)
    return E.trace() - A.shape[0]


def _bge_local_score(
    data: np.ndarray,
    child_idx: int,
    parent_idxs: list[int],
    alpha_mu: float = 1.0,
    alpha_w: float | None = None,
    suff: BGeSuffStats | None = None,
) -> float:
    """
    Log marginal-likelihood contribution of *child* given *parents*
    under the BGe scoring criterion (Geiger & Heckerman 2002).

    Uses the closed-form Normal-Wishart posterior.
    If *suff* is provided, uses precomputed sufficient statistics for
    O(m³) scoring instead of O(n·m²).
    """
    if suff is not None:
        # Fast path: slice precomputed scatter matrix
        joint_cols = parent_idxs + [child_idx]
        joint_score = suff.local_score(joint_cols)
        if parent_idxs:
            parent_score = suff.local_score(parent_idxs)
            return joint_score - parent_score
        return joint_score

    # Slow path: compute from raw data (fallback)
    n, p = data.shape
    alpha_w = alpha_w or (p + 2.0)

    cols = parent_idxs + [child_idx]
    X = data[:, cols]
    m = len(cols)

    X_bar = X.mean(axis=0)
    S = (X - X_bar).T @ (X - X_bar)

    T0 = np.eye(m) * (alpha_mu * (alpha_w - m - 1.0) / (alpha_mu + 1.0))
    Tn = T0 + S + (alpha_mu * n / (alpha_mu + n)) * np.outer(X_bar, X_bar)

    alpha_n = alpha_w + n
    score = 0.0
    score += 0.5 * m * (math.lgamma((alpha_n - m + 1.0) / 2.0) -
                         math.lgamma((alpha_w - m + 1.0) / 2.0))
    score -= 0.5 * n * m * math.log(math.pi)
    score += 0.5 * m * math.log(alpha_mu / (alpha_mu + n))

    sign_0, logdet_0 = np.linalg.slogdet(T0)
    sign_n, logdet_n = np.linalg.slogdet(Tn)
    if sign_0 <= 0 or sign_n <= 0:
        return -1e12
    score += 0.5 * alpha_w * logdet_0 - 0.5 * alpha_n * logdet_n

    # BGe *conditional* score: subtract the parent-only marginal
    if parent_idxs:
        Xp = data[:, parent_idxs]
        mp = len(parent_idxs)
        Xp_bar = Xp.mean(axis=0)
        Sp = (Xp - Xp_bar).T @ (Xp - Xp_bar)
        T0p = np.eye(mp) * (alpha_mu * (alpha_w - mp - 1.0) / (alpha_mu + 1.0))
        Tnp = T0p + Sp + (alpha_mu * n / (alpha_mu + n)) * np.outer(Xp_bar, Xp_bar)
        alpha_np = alpha_w + n

        parent_score = 0.0
        parent_score += 0.5 * mp * (math.lgamma((alpha_np - mp + 1.0) / 2.0) -
                                     math.lgamma((alpha_w - mp + 1.0) / 2.0))
        parent_score -= 0.5 * n * mp * math.log(math.pi)
        parent_score += 0.5 * mp * math.log(alpha_mu / (alpha_mu + n))

        s0p, ld0p = np.linalg.slogdet(T0p)
        snp, ldnp = np.linalg.slogdet(Tnp)
        if s0p <= 0 or snp <= 0:
            return -1e12
        parent_score += 0.5 * alpha_w * ld0p - 0.5 * alpha_np * ldnp

        score -= parent_score

    return float(score)


def bge_score(dag: DAGStructure, data: np.ndarray,
              suff: BGeSuffStats | None = None) -> float:
    """Full DAG BGe score (sum of local scores)."""
    total = 0.0
    for j in range(dag.n_nodes):
        parents = list(np.nonzero(dag.adj[:, j])[0])
        total += _bge_local_score(data, j, parents, suff=suff)
    return total


def _bge_node_local_scores(dag: DAGStructure, data: np.ndarray,
                           suff: BGeSuffStats) -> np.ndarray:
    """Compute per-node local BGe scores, returned as array of length p."""
    p = dag.n_nodes
    scores = np.zeros(p)
    for j in range(p):
        parents = list(np.nonzero(dag.adj[:, j])[0])
        scores[j] = _bge_local_score(data, j, parents, suff=suff)
    return scores


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
    penalty_acyclic: float = 0.5

    def log_prior(self, dag: DAGStructure) -> float:
        lp = 0.0
        for i in range(dag.n_nodes):
            for j in range(dag.n_nodes):
                if i == j:
                    continue
                prob = np.clip(self.edge_probs[i, j], 1e-6, 1 - 1e-6)
                if dag.adj[i, j]:
                    # Only penalise surprise inclusions (edges not supported by physics)
                    lp += math.log(prob) - (self.penalty_acyclic if prob < 0.5 else 0.0)
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

    @classmethod
    def from_pde_residuals(
        cls,
        data: np.ndarray,
        node_names: list[str],
        coords: np.ndarray,
        knn_k: int = 8,
        alpha_base: float = 0.1,
        penalty: float = 1.0,
    ) -> "PhysicsInformedGraphPrior":
        """Build a physics-informed prior from PDE-residual direction asymmetry.

        Calls :func:`pde_residual_edge_probs` to produce the ``(p, p)``
        edge-probability matrix and wraps it in a
        :class:`PhysicsInformedGraphPrior`.

        Parameters
        ----------
        data : (N, p) float array — observational data for each node.
        node_names : list of p node name strings.
        coords : (N, 2) float array — spatial coordinates.
        knn_k : k-nearest-neighbours used for the spatial Laplacian.
        alpha_base : uniform base probability blended into the final matrix
            (ensures no edge receives exactly 0 prior probability).
        penalty : ``penalty_acyclic`` passed to the prior.

        Returns
        -------
        PhysicsInformedGraphPrior
        """
        edge_probs = pde_residual_edge_probs(
            data=data,
            node_names=node_names,
            coords=coords,
            knn_k=knn_k,
            alpha_base=alpha_base,
        )
        return cls(edge_probs=edge_probs, penalty_acyclic=penalty)


# ---------------------------------------------------------------------------
# PDE-residual edge probability estimation
# ---------------------------------------------------------------------------

def pde_residual_edge_probs(
    data: np.ndarray,
    node_names: list[str],
    coords: np.ndarray,
    knn_k: int = 8,
    alpha_base: float = 0.1,
) -> np.ndarray:
    """Populate a ``(p, p)`` edge-probability matrix from spatial ANM direction asymmetry.

    For each ordered pair of nodes ``(i, j)``, fits the bivariate Additive
    Noise Model (ANM; Hoyer et al. 2009) in both causal directions using OLS,
    then scores each direction by the **spatial Laplacian energy** of its
    residuals:

    .. math::

        L(\\varepsilon) = \\frac{1}{N} \\sum_{n} \\bigl(\\varepsilon_n -
        \\overline{\\varepsilon}_{\\mathcal{N}(n)}\\bigr)^{2}

    where :math:`\\mathcal{N}(n)` is the k-nearest spatial neighbourhood.

    Physical interpretation: in the true causal direction ``i → j``, the
    noise term :math:`\\varepsilon = j - \\hat{j}(i)` should be
    spatially homogeneous (near-zero Laplacian energy) because the
    governing PDE constrains the residual spatial field to vary smoothly.
    The reverse direction ``j → i`` produces residuals whose spatial
    structure mirrors the confounder structure, giving higher Laplacian
    energy.  The relative energies are converted to edge probabilities via
    a softmax.

    The diagonal is set to zero (no self-loops).  All off-diagonal entries
    are blended with ``alpha_base`` so no edge receives exactly zero
    probability.

    Parameters
    ----------
    data : (N, p) float array
        Observational data; columns correspond to ``node_names``.
    node_names : list[str]
        Node names of length ``p``.
    coords : (N, 2) float array
        2-D spatial coordinates used to build the k-NN Laplacian.
    knn_k : int
        Number of spatial neighbours per observation.  Default 8.
    alpha_base : float
        Uniform prior blended into the final matrix in [0, 1).  Keeps the
        prior well-defined even for weakly informative data.

    Returns
    -------
    np.ndarray
        ``(p, p)`` matrix with entries in (0, 1) and zero diagonal.
        Ready for ``PhysicsInformedGraphPrior(edge_probs=...)``.
    """
    from sklearn.neighbors import NearestNeighbors

    data = np.asarray(data, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    N, p = data.shape
    if p != len(node_names):
        raise ValueError(
            f"data has {p} columns but node_names has {len(node_names)} entries."
        )
    if coords.shape[0] != N:
        raise ValueError(
            f"coords has {coords.shape[0]} rows but data has {N}."
        )

    # ---- Build k-NN spatial weight structure ----
    k = min(knn_k, N - 1)
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree")
    nbrs.fit(coords)
    _, knn_indices = nbrs.kneighbors(coords)
    # knn_indices[:, 0] is self; neighbors are knn_indices[:, 1:]
    neighbor_idx = knn_indices[:, 1:]  # (N, k)

    def _laplacian_energy(resid: np.ndarray) -> float:
        """Mean squared deviation of residual from its k-NN spatial mean."""
        resid = resid - resid.mean()
        nbr_mean = resid[neighbor_idx].mean(axis=1)  # (N,)
        return float(np.mean((resid - nbr_mean) ** 2))

    def _ols_residuals(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """OLS residuals of y regressed on x (with intercept)."""
        X = np.column_stack([np.ones(len(x)), x])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return y - X @ beta

    # ---- Per-pair direction test ----
    # energy_matrix[i, j] = Laplacian energy when we regress j on i (i → j)
    energy_matrix = np.full((p, p), np.inf)
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            resid = _ols_residuals(data[:, i], data[:, j])
            energy_matrix[i, j] = _laplacian_energy(resid)

    # ---- Convert to edge probabilities ----
    # For each ordered edge (i→j), its "plausibility score" is the
    # softmax of (−L(j|i)) over the two directions of the undirected pair.
    # edge_prob[i, j] ∝ exp(−L_ij) / (exp(−L_ij) + exp(−L_ji))
    # = sigmoid(L_ji − L_ij)
    edge_probs = np.zeros((p, p), dtype=np.float64)
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            L_ij = energy_matrix[i, j]   # i→j: j = f(i) + ε
            L_ji = energy_matrix[j, i]   # j→i: i = f(j) + ε'
            if not (np.isfinite(L_ij) and np.isfinite(L_ji)):
                edge_probs[i, j] = 0.5
                continue
            # Sigmoid of (opposite energy - own energy): high when i→j
            # residuals are spatially smoother than j→i residuals.
            delta = float(np.clip(L_ji - L_ij, -20.0, 20.0))
            edge_probs[i, j] = 1.0 / (1.0 + math.exp(-delta))

    # Blend with uniform base prior
    edge_probs = alpha_base + (1.0 - 2.0 * alpha_base) * edge_probs
    np.fill_diagonal(edge_probs, 0.0)
    return edge_probs


# ---------------------------------------------------------------------------
# MC³ sampler
# ---------------------------------------------------------------------------

def _propose(
    dag: DAGStructure, rng: np.random.Generator,
) -> tuple[DAGStructure, float, list[int]]:
    """Single-edge add / remove / reverse proposal.

    Returns
    -------
    proposed : DAGStructure
        The proposed DAG (equals *dag* on acyclicity rejection).
    log_hastings : float
        Log Metropolis-Hastings proposal ratio log[q(prop→curr)/q(curr→prop)].
        Per Green (1995) §3.2, add/remove have asymmetric pool sizes so the
        ratio is non-trivial.
    affected_nodes : list[int]
        Child node indices whose parent sets changed (for incremental scoring).
        Empty if the proposal was rejected for acyclicity.
    """
    prop = dag.copy()
    p = dag.n_nodes
    move = rng.choice(["add", "remove", "reverse"])

    n_edges = int(dag.adj.sum())
    n_possible = p * (p - 1)  # off-diagonal slots
    n_zeros = n_possible - n_edges

    if move == "add":
        zeros = list(zip(*np.where(prop.adj == 0)))
        zeros = [(i, j) for i, j in zeros if i != j]
        if not zeros:
            return prop, 0.0, []
        i, j = zeros[rng.integers(len(zeros))]
        prop.adj[i, j] = 1
        # Forward: pick "add" (1/3) then pick this zero-slot (1/n_zeros)
        # Reverse: pick "remove" (1/3) then pick this edge (1/(n_edges+1))
        log_hastings = math.log(n_zeros) - math.log(n_edges + 1)
        affected = [j]  # only child j's parent set changed
    elif move == "remove":
        ones = list(zip(*np.where(prop.adj == 1)))
        if not ones:
            return prop, 0.0, []
        i, j = ones[rng.integers(len(ones))]
        prop.adj[i, j] = 0
        # Forward: pick "remove" (1/3) then pick this edge (1/n_edges)
        # Reverse: pick "add" (1/3) then pick this zero-slot (1/(n_zeros+1))
        log_hastings = math.log(n_edges) - math.log(n_zeros + 1)
        affected = [j]
    else:  # reverse
        ones = list(zip(*np.where(prop.adj == 1)))
        if not ones:
            return prop, 0.0, []
        i, j = ones[rng.integers(len(ones))]
        prop.adj[i, j] = 0
        prop.adj[j, i] = 1
        # Edge count unchanged → symmetric within "reverse" move type
        log_hastings = 0.0
        affected = [i, j]  # both nodes' parent sets changed

    # Reject if cyclic
    G = nx.DiGraph(prop.adj)
    if not nx.is_directed_acyclic_graph(G):
        return dag.copy(), 0.0, []  # stay in place (MH rejection)
    return prop, log_hastings, affected


@dataclass
class MC3Results:
    """Collected MC³ output."""
    best_dag: DAGStructure
    best_score: float
    edge_inclusion_probs: np.ndarray  # (p, p) marginal inclusion posterior
    n_accepted: int
    n_total: int
    trace: list[float] = field(default_factory=list)
    n_accepted_per_chain: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64),
    )
    temperatures: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64),
    )
    physical_plausibility_score: dict[str, float] = field(default_factory=dict)


def run_mc3(
    data: pd.DataFrame,
    node_names: list[str],
    prior: PhysicsInformedGraphPrior,
    *,
    init_dag: DAGStructure | None = None,
    n_iter: int = 50_000,
    n_chains: int = 7,
    temperatures: list[float] | None = None,
    swap_every: int = 10,
    burnin_frac: float = 0.25,
    seed: int = 42,
    pde_coords: np.ndarray | None = None,
    pde_knn_k: int = 8,
    pde_scorer: Any | None = None,
    device: str = "cpu",
    min_iter: int = 10_000,
    converge_tol: float = 0.01,
    converge_window: int = 2_000,
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
    pde_coords : (N, 2) float array — spatial coordinates for computing
        physical plausibility scores on the winning DAG's edges via spatial
        Laplacian residual energy.  If ``None``, ``physical_plausibility_score``
        will be an empty dict.
    pde_knn_k : k-NN neighbours used for the spatial Laplacian energy when
        ``pde_coords`` is provided.
    """
    rng = np.random.default_rng(seed)
    temperatures = temperatures or [1.0, 0.75, 0.55, 0.4, 0.28, 0.2, 0.1]
    temperatures = temperatures[:n_chains]

    p = len(node_names)
    col_idx = [list(data.columns).index(n) for n in node_names]
    data_arr = data.values[:, col_idx].astype(np.float64)

    # Precompute sufficient statistics once (O(n·p²)) for fast O(m³) scoring.
    # Scores are memoized inside BGeSuffStats; repeated parent sets are O(1)
    # lookups across the full MCMC chain.
    suff = BGeSuffStats(data_arr, device=device)

    # Initialise chains
    if init_dag is None:
        init_dag = DAGStructure(adj=np.zeros((p, p), dtype=np.int8), node_names=node_names)

    chains: list[DAGStructure] = [init_dag.copy() for _ in range(n_chains)]

    # Per-node local BGe scores for incremental updates
    chain_node_scores: list[np.ndarray] = [
        _bge_node_local_scores(chains[k], data_arr, suff)
        for k in range(n_chains)
    ]
    chain_priors: list[float] = [
        prior.log_prior(chains[k]) for k in range(n_chains)
    ]
    scores: list[float] = [
        float(chain_node_scores[k].sum()) + chain_priors[k]
        for k in range(n_chains)
    ]

    # Tracking
    best_dag = chains[0].copy()
    best_score = scores[0]
    edge_counts = np.zeros((p, p), dtype=np.float64)
    n_post_burnin = 0
    n_accepted = 0
    n_accepted_per_chain = np.zeros(n_chains, dtype=np.int64)
    n_swap_attempts = np.zeros(max(n_chains - 1, 1), dtype=np.int64)
    n_swap_accepted = np.zeros(max(n_chains - 1, 1), dtype=np.int64)
    trace: list[float] = []
    burnin_cutoff = int(n_iter * burnin_frac)

    # Moving-window convergence tracking
    _cw_curr = np.zeros((p, p), dtype=np.float64)  # edge counts in current window
    _cw_prev = np.zeros((p, p), dtype=np.float64)  # edge counts in previous window
    _cw_n = 0  # samples accumulated in current window

    for it in range(n_iter):
        for k in range(n_chains):
            prop, log_hastings, affected = _propose(chains[k], rng)

            if not affected:
                # Proposal was rejected (acyclicity) or no-op — stay in place
                continue

            # Incremental scoring: only recompute affected nodes
            prop_node_scores = chain_node_scores[k].copy()
            for j in affected:
                parents = list(np.nonzero(prop.adj[:, j])[0])
                prop_node_scores[j] = _bge_local_score(
                    data_arr, j, parents, suff=suff,
                )
            prop_bge = float(prop_node_scores.sum())
            prop_prior = prior.log_prior(prop)
            prop_score = prop_bge + prop_prior

            # Optional PDE-identifiability penalty (injected by caller)
            if pde_scorer is not None:
                try:
                    prop_score = prop_score + pde_scorer(prop)
                except Exception as _e:  # noqa: BLE001
                    logger.debug("pde_scorer raised %s — skipping", _e)

            # Metropolis-Hastings acceptance with Hastings correction
            # (Green 1995 §3.2): log α = β·(score_prop - score_curr) + log_hastings
            # The Hastings ratio corrects for asymmetric proposal pool sizes
            # and is NOT tempered (it comes from the proposal, not the target).
            log_alpha = temperatures[k] * (prop_score - scores[k]) + log_hastings
            if math.log(rng.random() + 1e-300) < log_alpha:
                chains[k] = prop
                chain_node_scores[k] = prop_node_scores
                chain_priors[k] = prop_prior
                scores[k] = prop_score
                n_accepted_per_chain[k] += 1
                if k == 0:
                    n_accepted += 1

        # Inter-chain swap
        if it % swap_every == 0 and n_chains > 1:
            k1 = rng.integers(n_chains - 1)
            k2 = k1 + 1
            log_swap = (temperatures[k1] - temperatures[k2]) * (scores[k2] - scores[k1])
            n_swap_attempts[k1] += 1
            if math.log(rng.random() + 1e-300) < log_swap:
                n_swap_accepted[k1] += 1
                chains[k1], chains[k2] = chains[k2], chains[k1]
                scores[k1], scores[k2] = scores[k2], scores[k1]
                chain_node_scores[k1], chain_node_scores[k2] = (
                    chain_node_scores[k2], chain_node_scores[k1],
                )
                chain_priors[k1], chain_priors[k2] = (
                    chain_priors[k2], chain_priors[k1],
                )

        # Track cold chain
        if scores[0] > best_score:
            best_score = scores[0]
            best_dag = chains[0].copy()

        trace.append(scores[0])

        if it >= burnin_cutoff:
            _adj = chains[0].adj.astype(np.float64)
            edge_counts += _adj
            _cw_curr += _adj
            _cw_n += 1
            n_post_burnin += 1

            # Moving-window convergence check: compare successive windows of
            # converge_window post-burnin samples. Stop early once the max
            # absolute change in any edge inclusion probability drops below
            # converge_tol, but only after min_iter total iterations.
            if (it + 1) >= min_iter and _cw_n >= converge_window:
                if n_post_burnin > converge_window:
                    _p_curr = _cw_curr / converge_window
                    _p_prev = _cw_prev / converge_window
                    _max_delta = float(np.max(np.abs(_p_curr - _p_prev)))
                    if _max_delta < converge_tol:
                        print(
                            f"   MC³ converged at iter {it+1}  "
                            f"(max Δp={_max_delta:.4f} < tol={converge_tol})  "
                            f"post-burnin={n_post_burnin}",
                            flush=True,
                        )
                        break
                _cw_prev = _cw_curr.copy()
                _cw_curr = np.zeros((p, p), dtype=np.float64)
                _cw_n = 0

        # Progress logging
        if (it + 1) % 1000 == 0 or it == 0:
            n_edges = int(chains[0].adj.sum())
            per_chain_acc = "  ".join(
                f"c{k}(T={temperatures[k]:.2f}):{int(n_accepted_per_chain[k])}"
                for k in range(n_chains)
            )
            print(f"   MC³ iter {it+1}/{n_iter}  score={scores[0]:.2f}  "
                  f"edges={n_edges}  cold_acc={n_accepted}  [{per_chain_acc}]",
                  flush=True)

    edge_probs = edge_counts / max(n_post_burnin, 1)

    # ---- Physical plausibility score for winning-DAG edges ----
    physical_plausibility: dict[str, float] = {}
    if pde_coords is not None:
        from sklearn.neighbors import NearestNeighbors as _NNS

        data_arr_pps = data.values[:, col_idx].astype(np.float64)
        coords_arr = np.asarray(pde_coords, dtype=np.float64)
        k = min(pde_knn_k, len(coords_arr) - 1)
        nbrs_pps = _NNS(n_neighbors=k + 1, algorithm="ball_tree")
        nbrs_pps.fit(coords_arr)
        _, knn_idx_pps = nbrs_pps.kneighbors(coords_arr)
        neighbor_idx_pps = knn_idx_pps[:, 1:]

        def _lap_energy(resid: np.ndarray) -> float:
            resid = resid - resid.mean()
            nbr_mean = resid[neighbor_idx_pps].mean(axis=1)
            return float(np.mean((resid - nbr_mean) ** 2))

        def _ols_resid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            X = np.column_stack([np.ones(len(x)), x])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            return y - X @ beta

        for i in range(len(node_names)):
            for j in range(len(node_names)):
                if i == j:
                    continue
                if best_dag.adj[i, j]:
                    L_ij = _lap_energy(_ols_resid(data_arr_pps[:, i], data_arr_pps[:, j]))
                    L_ji = _lap_energy(_ols_resid(data_arr_pps[:, j], data_arr_pps[:, i]))
                    total = L_ij + L_ji
                    score_ij = 1.0 - (L_ij / total) if total > 0 else 0.5
                    key = f"{node_names[i]}->{node_names[j]}"
                    physical_plausibility[key] = float(score_ij)

    # Swap-rate diagnostic
    swap_rates = np.where(
        n_swap_attempts > 0,
        n_swap_accepted / n_swap_attempts.clip(1),
        0.0,
    )
    swap_rate_str = "  ".join(
        f"T{temperatures[k]:.2f}↔T{temperatures[k+1]:.2f}:{swap_rates[k]:.2f}"
        for k in range(n_chains - 1)
    )
    if (n_chains > 1) and float(swap_rates.min()) < 0.10:
        logger.warning(
            "MC³ swap rates below 0.10 for some adjacent pairs — consider a denser temperature ladder: %s",
            swap_rate_str,
        )
        print(f"   [WARN] MC³ swap rates (some < 0.10): {swap_rate_str}", flush=True)
    else:
        logger.info("MC³ swap rates: %s", swap_rate_str)
        print(f"   MC³ swap rates: {swap_rate_str}", flush=True)

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
        n_accepted_per_chain=n_accepted_per_chain.copy(),
        temperatures=np.asarray(temperatures, dtype=np.float64),
        physical_plausibility_score=physical_plausibility,
    )


def sample_dags_from_edge_probs(
    edge_probs: np.ndarray,
    node_names: list[str],
    n_samples: int = 50,
    max_attempts: int = 2000,
    seed: int = 42,
) -> list[np.ndarray]:
    """Sample valid DAG adjacency matrices from MC³ edge inclusion probabilities.

    Uses independent Bernoulli sampling per edge and Kahn's topological-sort
    cycle check.  Returns at most ``n_samples`` acyclic adjacency matrices
    (as boolean arrays of shape ``(p, p)``).
    """
    from collections import deque
    rng = np.random.default_rng(seed)
    p = edge_probs.shape[0]
    dags: list[np.ndarray] = []
    for _ in range(max_attempts):
        if len(dags) >= n_samples:
            break
        adj = rng.random((p, p)) < edge_probs
        np.fill_diagonal(adj, False)
        in_deg = adj.sum(axis=0).copy()
        q: deque = deque(i for i in range(p) if in_deg[i] == 0)
        visited = 0
        while q:
            u = q.popleft()
            visited += 1
            for v in np.where(adj[u])[0]:
                in_deg[v] -= 1
                if in_deg[v] == 0:
                    q.append(v)
        if visited == p:
            dags.append(adj.astype(bool))
    return dags
