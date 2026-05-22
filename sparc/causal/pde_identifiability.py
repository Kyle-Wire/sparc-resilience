"""
PDE identifiability — score a complete DAG by its PDE-residual plausibility.

This module provides :func:`compute_dag_pde_plausibility`, which scores a
*complete directed acyclic graph* (rather than individual edges) by evaluating
whether the implied causal ordering is consistent with the spatial smoothness
expected from the governing PDE.

Conceptually, it extends :func:`~sparc.causal.mc3.pde_residual_edge_probs`
from pairwise edge priors to a single composite score for a full DAG.  The
score is the log-product of per-edge probabilities along the topologically
ordered skeleton, minus a penalty for edges whose residuals are
spatially rough (high Laplacian energy relative to the parent set).

Usage in MC³
------------
Pass an instance of :class:`PDEDagScorer` as the optional ``pde_scorer``
argument to :func:`~sparc.causal.mc3.run_mc3`.  During the
Metropolis–Hastings accept/reject step the scorer adds
``lambda_pde * pde_scorer(proposed_dag)`` to the candidate log-score.

Example::

    scorer = PDEDagScorer.from_data(data, node_names, coords)
    results = run_mc3(data, node_names, pde_scorer=scorer)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (shared with mc3.pde_residual_edge_probs, kept local to avoid
# cross-import issues — these are tiny wrappers around numpy/sklearn)
# ---------------------------------------------------------------------------

def _build_knn_neighbors(coords: np.ndarray, k: int) -> np.ndarray:
    """Return ``(N, k)`` array of k-nearest neighbor indices (excluding self)."""
    from sklearn.neighbors import NearestNeighbors

    coords = np.asarray(coords, dtype=np.float64)
    N = coords.shape[0]
    k = min(k, N - 1)
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree")
    nbrs.fit(coords)
    _, knn_idx = nbrs.kneighbors(coords)
    return knn_idx[:, 1:]  # drop self (column 0)


def _laplacian_energy(resid: np.ndarray, neighbor_idx: np.ndarray) -> float:
    """Mean squared deviation of *resid* from its spatial k-NN mean.

    Lower energy → residuals are spatially smooth → PDE-consistent.
    """
    r = resid - resid.mean()
    nbr_mean = r[neighbor_idx].mean(axis=1)
    return float(np.mean((r - nbr_mean) ** 2))


def _ols_residuals_multivariate(
    X: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """OLS residuals of *y* regressed on columns of *X* (with intercept)."""
    Xb = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    return y - Xb @ beta


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_dag_pde_plausibility(
    adj: np.ndarray,
    node_names: List[str],
    data: np.ndarray,
    coords: np.ndarray,
    knn_k: int = 8,
    lambda_pde: float = 0.1,
) -> float:
    """Score a complete DAG by spatial-Laplacian PDE plausibility.

    For every node *j* that has at least one parent in ``adj``, this function:

    1. Fits OLS of *j* on all its parents (additive noise model).
    2. Computes the Laplacian energy of the residuals using the spatial
       k-NN graph derived from ``coords``.
    3. Accumulates a log-score term: ``−lambda_pde * L(ε_j)``.

    Nodes with no parents contribute zero to the score.

    A higher (less negative) score indicates the DAG's causal ordering
    produces spatially smooth residuals — consistent with a PDE constraint
    that requires the error field to be spatially homogeneous.

    Parameters
    ----------
    adj : (p, p) int array
        Binary adjacency matrix where ``adj[i, j] = 1`` means edge ``i → j``.
    node_names : list[str]
        Node names (length ``p``).
    data : (N, p) float array
        Observational data; columns correspond to ``node_names``.
    coords : (N, 2) float array
        2-D spatial coordinates for the Laplacian computation.
    knn_k : int
        k-nearest spatial neighbours per observation.
    lambda_pde : float
        Penalty weight per unit of Laplacian energy.  Higher values make
        the score more sensitive to spatial roughness.

    Returns
    -------
    float
        Log-plausibility score (always ≤ 0; higher = more plausible).
        Returns 0.0 for DAGs where no node has parents.
    """
    data = np.asarray(data, dtype=np.float64)
    adj = np.asarray(adj, dtype=np.int8)
    coords = np.asarray(coords, dtype=np.float64)
    p = len(node_names)

    if data.shape[1] != p:
        raise ValueError(
            f"data has {data.shape[1]} columns but node_names has {p} entries."
        )
    if coords.shape[0] != data.shape[0]:
        raise ValueError(
            f"coords has {coords.shape[0]} rows but data has {data.shape[0]}."
        )

    neighbor_idx = _build_knn_neighbors(coords, knn_k)
    log_score = 0.0

    for j in range(p):
        parent_indices = np.where(adj[:, j] == 1)[0]
        if len(parent_indices) == 0:
            continue
        X_pa = data[:, parent_indices]
        y_j = data[:, j]
        resid = _ols_residuals_multivariate(X_pa, y_j)
        energy = _laplacian_energy(resid, neighbor_idx)
        log_score -= lambda_pde * energy

    return float(log_score)


@dataclass
class PDEDagScorer:
    """Callable wrapper for :func:`compute_dag_pde_plausibility`.

    Designed to be passed as the ``pde_scorer`` argument to
    :func:`~sparc.causal.mc3.run_mc3`.  When called with a
    :class:`~sparc.causal.mc3.DAGStructure`, it returns a scalar float
    that is added to the log-score of each MC³ proposal.

    Attributes
    ----------
    node_names : list[str]
    data : (N, p) float array
    coords : (N, 2) float array
    knn_k : int
    lambda_pde : float
    """

    node_names: List[str]
    data: np.ndarray
    coords: np.ndarray
    knn_k: int = 8
    lambda_pde: float = 0.1

    def __call__(self, dag: Any) -> float:
        """Score *dag* (a :class:`~sparc.causal.mc3.DAGStructure`).

        Parameters
        ----------
        dag : DAGStructure
            The proposed DAG to score.

        Returns
        -------
        float
            Log-plausibility (≤ 0).
        """
        try:
            adj = dag.adj  # DAGStructure
        except AttributeError:
            # Fallback: assume dag is already an ndarray
            adj = np.asarray(dag, dtype=np.int8)

        return compute_dag_pde_plausibility(
            adj=adj,
            node_names=self.node_names,
            data=self.data,
            coords=self.coords,
            knn_k=self.knn_k,
            lambda_pde=self.lambda_pde,
        )

    @classmethod
    def from_data(
        cls,
        data: np.ndarray,
        node_names: List[str],
        coords: np.ndarray,
        knn_k: int = 8,
        lambda_pde: float = 0.1,
    ) -> "PDEDagScorer":
        """Convenience constructor."""
        return cls(
            node_names=node_names,
            data=np.asarray(data, dtype=np.float64),
            coords=np.asarray(coords, dtype=np.float64),
            knn_k=knn_k,
            lambda_pde=lambda_pde,
        )
