"""Spatial block cross-validation used by every core stage.

Folds are built from square spatial blocks (side ``block_m``), assigned to
folds so fold sizes are balanced.  Training sets exclude a buffer: points of
other folds within ``buffer_m`` of any test point are dropped from training,
so a model is never trained on a test point's immediate neighbourhood.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SpatialFolds:
    fold_id: np.ndarray        # (n,) fold index per point
    block_id: np.ndarray       # (n,) spatial block index per point (for clustered SEs / bootstrap)
    n_folds: int
    block_m: float
    buffer_m: float
    train_masks: list          # list[np.ndarray bool] with buffer exclusion
    test_masks: list

    def split(self):
        for k in range(self.n_folds):
            yield np.flatnonzero(self.train_masks[k]), np.flatnonzero(self.test_masks[k])

    @property
    def n_blocks(self) -> int:
        return int(np.unique(self.block_id).size)


def block_ids(coords: np.ndarray, block_m: float) -> np.ndarray:
    """Integer id of the square block containing each point."""
    c = np.asarray(coords, dtype=np.float64)
    bx = np.floor((c[:, 0] - c[:, 0].min()) / block_m).astype(np.int64)
    by = np.floor((c[:, 1] - c[:, 1].min()) / block_m).astype(np.int64)
    _, inv = np.unique(by * (bx.max() + 1) + bx, return_inverse=True)
    return inv.astype(np.int64)


def make_spatial_folds(
    coords: np.ndarray,
    n_folds: int = 5,
    block_m: float = 500.0,
    buffer_m: float | None = None,
    seed: int = 42,
) -> SpatialFolds:
    """Balanced block-to-fold assignment with a training buffer.

    Blocks are shuffled then assigned greedily to the currently smallest
    fold, which keeps fold sizes within one block of each other.
    """
    from scipy.spatial import cKDTree

    coords = np.asarray(coords, dtype=np.float64)
    blk = block_ids(coords, block_m)
    n_blocks = int(blk.max()) + 1
    if n_blocks < n_folds:
        raise ValueError(f"only {n_blocks} spatial blocks for {n_folds} folds; reduce block_m")
    sizes = np.bincount(blk, minlength=n_blocks)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_blocks)
    order = order[np.argsort(-sizes[order], kind="stable")]  # big blocks first, random ties
    fold_of_block = np.empty(n_blocks, dtype=np.int64)
    load = np.zeros(n_folds)
    for b in order:
        k = int(np.argmin(load))
        fold_of_block[b] = k
        load[k] += sizes[b]
    fold = fold_of_block[blk]

    buffer_m = float(block_m / 3.0 if buffer_m is None else buffer_m)
    train_masks, test_masks = [], []
    for k in range(n_folds):
        test = fold == k
        train = ~test
        if buffer_m > 0 and test.any() and train.any():
            tree = cKDTree(coords[test])
            d, _ = tree.query(coords[train], k=1, distance_upper_bound=buffer_m)
            keep = ~np.isfinite(d)
            tr_idx = np.flatnonzero(train)
            train = np.zeros_like(train)
            train[tr_idx[keep]] = True
        train_masks.append(train)
        test_masks.append(test)
    return SpatialFolds(fold_id=fold, block_id=blk, n_folds=n_folds, block_m=float(block_m),
                        buffer_m=buffer_m, train_masks=train_masks, test_masks=test_masks)
