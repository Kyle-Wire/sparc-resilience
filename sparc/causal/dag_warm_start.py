"""Gradient warm-start for MC³ chain initialisation (S3-12b/c).

Provides independent Adam restarts (n_particles == 1) or SVGD joint
particle optimisation (n_particles > 1) over the NOTEARS continuous
relaxation of the DAG structure space.

References
----------
Zheng et al. 2018 — "DAGs with NO TEARS"
Liu & Wang 2016   — "Stein Variational Gradient Descent"
"""
from __future__ import annotations

import math

import networkx as nx
import numpy as np
import torch

from .mc3 import BGeSuffStats, notears_h


def _threshold_to_dag(A: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    """Threshold a soft adjacency matrix into a valid binary DAG adjacency.

    Greedily adds edges in descending soft-weight order, skipping any edge
    that would introduce a directed cycle.  Always returns a valid DAG.
    """
    p = A.shape[0]
    edges = [
        (float(A[i, j]), i, j)
        for i in range(p)
        for j in range(p)
        if i != j and A[i, j] > thresh
    ]
    edges.sort(reverse=True)
    G: nx.DiGraph = nx.DiGraph()
    G.add_nodes_from(range(p))
    for _, i, j in edges:
        G.add_edge(i, j)
        if not nx.is_directed_acyclic_graph(G):
            G.remove_edge(i, j)
    adj = np.zeros((p, p), dtype=int)
    for u, v in G.edges():
        adj[u, v] = 1
    return adj


def _rbf_kernel(
    W_list: list[torch.Tensor],
) -> tuple[torch.Tensor, list[list[torch.Tensor]]]:
    """Pairwise RBF kernel K[i,j] and gradient dK[i][j] = d/dw_i K(w_i, w_j).

    Bandwidth chosen via the median heuristic: h = median(||w_i - w_j||²) / (2 log n).
    """
    n = len(W_list)
    vecs = torch.stack([w.detach().flatten() for w in W_list])  # (n, d)
    dists_sq = torch.cdist(vecs, vecs).pow(2)                   # (n, n)
    h = float(dists_sq.median()) / (2.0 * math.log(n + 1)) + 1e-8
    K = torch.exp(-dists_sq / h)                                # (n, n)
    # d/dw_i K(w_i, w_j) = (2/h)(w_j - w_i) K[i,j]
    dK = [
        [(2.0 / h) * (vecs[j] - vecs[i]) * K[i, j] for j in range(n)]
        for i in range(n)
    ]
    return K, dK


def gradient_warm_start(
    suff: BGeSuffStats,
    n_particles: int = 4,
    n_steps: int = 500,
    lambda_h: float = 1.0,
    seed: int = 0,
) -> list[np.ndarray]:
    """Return n_particles initial DAG adjacency matrices via gradient warm-start.

    n_particles == 1 : independent Adam restart (S3-12b fallback).
    n_particles >  1 : SVGD joint optimisation with RBF repulsion (S3-12c).
                       Particles are pushed apart in weight-matrix space so
                       MC³ chains start from genuinely different modes.
    """
    rng = np.random.default_rng(seed)
    p = suff.p

    W_list = [
        torch.tensor(
            rng.standard_normal((p, p)) * 0.1,
            dtype=torch.float64,
            requires_grad=True,
        )
        for _ in range(n_particles)
    ]
    opt = torch.optim.Adam(W_list, lr=0.01)

    for _ in range(n_steps):
        opt.zero_grad()
        A_list = [torch.sigmoid(W) for W in W_list]
        scores = [
            suff.bge_score_soft(A) - lambda_h * notears_h(A)
            for A in A_list
        ]

        if n_particles == 1:
            # Independent Adam: minimise negative log-score
            (-scores[0]).backward()
        else:
            # SVGD: kernel-weighted score gradients + repulsion
            K, dK = _rbf_kernel(W_list)
            score_grads = [
                torch.autograd.grad(scores[j], W_list[j])[0].flatten()
                for j in range(n_particles)
            ]
            for i in range(n_particles):
                phi_i = sum(
                    K[j, i] * score_grads[j] + dK[i][j]
                    for j in range(n_particles)
                ) / n_particles
                W_list[i].grad = -phi_i.view(p, p)

        opt.step()

    return [
        _threshold_to_dag(torch.sigmoid(W).detach().numpy())
        for W in W_list
    ]
