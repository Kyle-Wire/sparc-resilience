"""
Spatial graph geometry utilities for SPARC.

These functions are shared across the training runner, the Bayesian causal
module, and the physics PDE operators.  They live here — not inside any
single pipeline stage — so callers import from a stable public seam.

Public API
----------
build_knn_index(coords, max_neighbors, return_dists=False)
    Build a KNN index (N, max_neighbors) from projected coordinates.

build_cardinal_neighbors(coords, resolution=None, tol_factor=1.5)
    Build N/S/E/W cardinal-neighbor indices for the physics Laplacian.

remap_indices_to_local(global_idx, batch_idx, neighbor_tensor)
    Remap global neighbor indices to batch-local indices (pure torch).
"""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# KNN index
# ---------------------------------------------------------------------------

def build_knn_index(
    coords: np.ndarray,
    max_neighbors: int,
    return_dists: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Build KNN index (N, max_neighbors) from projected coords.

    Parameters
    ----------
    coords : (N, 2) projected coordinate array
    max_neighbors : number of nearest neighbours (excluding self)
    return_dists : when True, also return (N, max_neighbors) distances

    Returns
    -------
    indices : (N, max_neighbors) int64 KNN index
    dists   : (N, max_neighbors) float64 distances — only when return_dists=True
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(coords)
    dists, indices = tree.query(coords, k=max_neighbors + 1)
    if return_dists:
        return indices[:, 1:], dists[:, 1:]
    return indices[:, 1:]  # exclude self


# ---------------------------------------------------------------------------
# Cardinal-neighbor index
# ---------------------------------------------------------------------------

def build_cardinal_neighbors(
    coords: np.ndarray,
    resolution: float | None = None,
    tol_factor: float = 1.5,
) -> tuple[np.ndarray, float]:
    """
    Build N/S/E/W cardinal neighbor indices for the physics Laplacian.

    Parameters
    ----------
    coords : (N, 2) — projected coordinates (x, y)
    resolution : grid spacing; auto-detected from nearest-neighbour
                 distances if None
    tol_factor : max distance tolerance as multiple of resolution

    Returns
    -------
    neighbor_idx : (N, 4) int64 — [North, South, East, West], -1 = missing
    resolution   : detected/used resolution
    """
    from scipy.spatial import cKDTree

    N = len(coords)
    tree = cKDTree(coords)

    if resolution is None:
        dists, _ = tree.query(coords, k=2)
        resolution = float(np.median(dists[:, 1]))

    tol = resolution * tol_factor

    # Offsets: North (+y), South (-y), East (+x), West (-x)
    offsets = np.array([
        [0, resolution],
        [0, -resolution],
        [resolution, 0],
        [-resolution, 0],
    ])

    neighbor_idx = np.full((N, 4), -1, dtype=np.int64)
    for k, offset in enumerate(offsets):
        targets = coords + offset
        dists, idxs = tree.query(targets, k=1)
        valid = dists < tol
        neighbor_idx[valid, k] = idxs[valid]

    # ------------------------------------------------------------------
    # Reject self-loops: edge points with no true neighbour in a direction
    # get assigned themselves (distance = resolution < tol).
    # ------------------------------------------------------------------
    arange_N = np.arange(N)
    for k in range(4):
        self_loop = neighbor_idx[:, k] == arange_N
        neighbor_idx[self_loop, k] = -1

    # ------------------------------------------------------------------
    # Reject wrong-direction assignments: ensure the found neighbour
    # actually lies in the expected cardinal direction.
    #   North: dy > 0,  South: dy < 0,  East: dx > 0,  West: dx < 0
    # ------------------------------------------------------------------
    _dir_axis = [1, 1, 0, 0]   # y, y, x, x
    _dir_sign = [1, -1, 1, -1]  # +, -, +, -
    for k in range(4):
        valid_mask = neighbor_idx[:, k] >= 0
        if not valid_mask.any():
            continue
        valid_idx = np.where(valid_mask)[0]
        delta = (coords[neighbor_idx[valid_idx, k], _dir_axis[k]]
                 - coords[valid_idx, _dir_axis[k]])
        wrong = (delta * _dir_sign[k]) <= 0
        neighbor_idx[valid_idx[wrong], k] = -1

    n_complete = int((neighbor_idx != -1).all(axis=1).sum())
    n_boundary = int((neighbor_idx == -1).any(axis=1).sum())
    logger.info(
        "Cardinal neighbors: %d/%d complete, %d boundary (res=%.2f)",
        n_complete, N, n_boundary, resolution,
    )
    return neighbor_idx, resolution


# ---------------------------------------------------------------------------
# Batch-local index remapping
# ---------------------------------------------------------------------------

def remap_indices_to_local(
    global_idx: np.ndarray,
    batch_idx: np.ndarray,
    neighbor_tensor: torch.Tensor,
) -> torch.Tensor:
    """Remap global neighbor indices to batch-local indices.

    Any global index not present in ``batch_idx`` is set to -1.

    Parameters
    ----------
    global_idx : (N,) int array of all point indices in the fold
    batch_idx  : (B,) int array of indices present in this batch
    neighbor_tensor : (B, K) long tensor of global neighbor indices

    Returns
    -------
    (B, K) long tensor with global indices replaced by batch-local
    positions; -1 where a neighbour is not in the batch.
    """
    map_size = int(global_idx.max()) + 1 if len(global_idx) else 0
    if map_size == 0:
        return torch.full_like(neighbor_tensor, -1)

    device = neighbor_tensor.device

    # Build global→local map in torch — no .cpu().numpy() round-trip on
    # neighbor_tensor, which may live on GPU/MPS.
    local_map = torch.full((map_size,), -1, dtype=torch.long)
    batch_t = torch.as_tensor(batch_idx, dtype=torch.long)
    local_map[batch_t] = torch.arange(len(batch_idx), dtype=torch.long)
    local_map = local_map.to(device)

    nb = neighbor_tensor                         # (B, K) — stays on device
    valid = (nb >= 0) & (nb < map_size)
    nb_safe = nb.clamp(0, map_size - 1)          # safe index for invalid entries
    return torch.where(valid, local_map[nb_safe], torch.full_like(nb, -1))
