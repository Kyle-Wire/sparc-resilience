#!/usr/bin/env python3
"""
spatial_fold_factory.py — Spatial cross-validation fold generation.

Extracted from ``enhanced_spatial_cv.py`` to reduce that module's size.
All public symbols are re-exported from ``enhanced_spatial_cv`` for
backward compatibility.

Public API
----------
spatial_kfold_enhanced
latent_guided_spatial_kfold
_vsba_fold_quality_score
calculate_fold_spatial_autocorr
_spatial_smooth
score_model_spatial_consistency
estimate_spatial_autocorrelation_range
"""

from __future__ import annotations

import math
import numpy as np


# ---------------------------------------------------------------------------
# Spatial utility helpers
# ---------------------------------------------------------------------------

def calculate_fold_spatial_autocorr(residuals, coords, threshold=1000, _precomputed_indices=None):
    """
    Calculate Moran's I for a single fold's residuals using K-NN weights.
    Returns a simple spatial autocorrelation metric.
    """
    try:
        if len(residuals) < 10:  # Skip tiny folds
            return 0.0

        k_neighbors = min(8, len(residuals) - 1)

        if k_neighbors < 1:
            return 0.0

        if _precomputed_indices is not None:
            indices = _precomputed_indices
        else:
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(coords)
            _, indices = nbrs.kneighbors(coords)

        n = len(residuals)
        y_mean = np.mean(residuals)
        y_centered = residuals - y_mean

        # Vectorised Moran's I (row-standardised equal weights)
        yc_nb = y_centered[indices[:, 1:]]              # (N, K) — neighbour values
        cross = float((y_centered[:, None] * yc_nb).sum())
        denom = float(np.sum(y_centered ** 2))
        if denom > 0:
            morans_i = cross / (k_neighbors * denom)
            return morans_i
        else:
            return 0.0

    except Exception:
        return 0.0


def _spatial_smooth(predictions, coords, bandwidth=500, k=8, _precomputed=None):
    """Inverse-distance-weighted spatial smoothing of predictions.

    For each point computes the IDW mean of its k nearest neighbours'
    predictions using a Gaussian kernel with the given bandwidth (metres).
    Returns smoothed predictions as a float32 array of the same length.
    """
    n = len(predictions)
    k_use = min(k, n - 1)
    if k_use < 1:
        return predictions.copy()
    if _precomputed is not None:
        distances, indices = _precomputed
    else:
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=k_use + 1).fit(coords)
        distances, indices = nbrs.kneighbors(coords)
    # Skip self (index 0) — neighbours are columns 1..k_use
    dists = distances[:, 1:]          # (n, k_use)
    nbr_idx = indices[:, 1:]          # (n, k_use)
    weights = np.exp(-0.5 * (dists / max(bandwidth, 1e-3)) ** 2)  # Gaussian
    weight_sum = weights.sum(axis=1, keepdims=True)
    weight_sum[weight_sum < 1e-12] = 1.0
    weights = weights / weight_sum
    smoothed = (weights * predictions[nbr_idx]).sum(axis=1)
    return smoothed.astype(np.float32)


def score_model_spatial_consistency(predictions, coords, bandwidth=500):
    """Label-free spatial consistency score for a set of model predictions.

    Computes ``pseudo_residuals = predictions - _spatial_smooth(predictions)``
    and returns Moran's I on those pseudo-residuals.  Lower Moran's I =
    spatially consistent predictions = better model for label-free
    hyperparameter selection.

    Parameters
    ----------
    predictions : array-like, shape (n,)
    coords : array-like, shape (n, 2)
    bandwidth : float, metres — Gaussian kernel bandwidth for IDW smoothing

    Returns
    -------
    float : Moran's I on pseudo-residuals (lower is better).
            Returns 0.0 on any failure.
    """
    try:
        predictions = np.asarray(predictions, dtype=np.float32)
        coords = np.asarray(coords)
        if len(predictions) < 10:
            return 0.0
        from sklearn.neighbors import NearestNeighbors
        k = min(8, len(coords) - 1)
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        smoothed = _spatial_smooth(predictions, coords, bandwidth=bandwidth,
                                   _precomputed=(distances, indices))
        pseudo_residuals = predictions - smoothed
        return calculate_fold_spatial_autocorr(pseudo_residuals, coords,
                                               _precomputed_indices=indices)
    except Exception:
        return 0.0


def estimate_spatial_autocorrelation_range(coords, y, max_distance=None, n_bins=20, sample_size=5000):
    """
    Estimate spatial autocorrelation range using Moran's I over increasing distance bands.
    Uses downsampled spatial weights for memory efficiency.
    """
    from libpysal.weights import DistanceBand
    from esda.moran import Moran

    if max_distance is None:
        if len(coords) == 0:
            raise ValueError("coords is empty — cannot estimate spatial autocorrelation range")
        bounds = np.array([coords.min(axis=0), coords.max(axis=0)])
        max_distance = np.linalg.norm(bounds[1] - bounds[0]) * 0.3
    print("Estimating spatial autocorrelation range...")

    if len(coords) > sample_size:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(coords), sample_size, replace=False)
        coords = coords[idx]
        y = y[idx]

    bins = np.linspace(0, max_distance, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    moran_i_values = []

    for i in range(len(bins) - 1):
        try:
            w = DistanceBand(coords, threshold=bins[i + 1], binary=True, silence_warnings=True)
            if len(w.neighbors) < 10:
                moran_i_values.append(0)
                continue
            moran = Moran(y, w, two_tailed=False)
            moran_i_values.append(moran.I)
        except Exception:
            moran_i_values.append(0)

    threshold = 0.1
    for i, mi in enumerate(moran_i_values):
        if abs(mi) < threshold:
            print(f"Autocorrelation drops at ~{bin_centers[i]:.0f}m (Moran's I = {mi:.2f})")
            return bin_centers[i]

    print(f"No clear drop detected — defaulting to max: {max_distance}m")
    return max_distance


# ---------------------------------------------------------------------------
# Fold generation
# ---------------------------------------------------------------------------

def spatial_kfold_enhanced(X, y, coords, n_splits=5, block_size=None, buffer_size=0, method='block', stratify_y=True):
    """
    Generate stratified spatial cross-validation folds with optional buffer.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
    y : np.ndarray
    coords : np.ndarray, shape (n_samples, 2)
    n_splits : int
    block_size : float, optional
    buffer_size : float, optional
    method : str — 'block' | 'kmeans'
    stratify_y : bool

    Returns
    -------
    list of (train_idx, test_idx) tuples
    """
    n_samples = len(X)

    if block_size is None:
        if len(coords) == 0:
            raise ValueError("coords is empty — cannot determine block size")
        print("Determining optimal block size from target variable autocorrelation...")
        try:
            from sparc.run.spatial_autocorr_comprehensive import SpatialAutocorrelationAnalyzer
            bounds = np.array([coords.min(axis=0), coords.max(axis=0)])
            spatial_extent = np.linalg.norm(bounds[1] - bounds[0])
            max_dist = spatial_extent * 0.4
            analyzer = SpatialAutocorrelationAnalyzer(coords, max_distance=max_dist)
            correlogram_results = analyzer.compute_correlogram(
                y, plot=False, title="Block Size Selection Correlogram"
            )
            block_size = correlogram_results.get('first_zero_crossing',
                                                  correlogram_results['optimal_block_size'])
            print(f"Target variable zero-crossing: {block_size:.0f}m")
            max_viable = spatial_extent / (2.0 * n_splits)
            if block_size > max_viable:
                print(f"Capping block size: {block_size:.0f}m > extent/(2*{n_splits}) = {max_viable:.0f}m")
                block_size = max_viable
            block_size = max(block_size, 500)
        except Exception as e:
            print(f"Correlogram analysis failed: {e}")
            print("Falling back to traditional autocorrelation range estimation...")
            autocorr_range = estimate_spatial_autocorrelation_range(coords, y)
            block_size = max(autocorr_range * 2, 500)
        print(f"Final block size: {block_size:.0f}m")

    if method == 'block':
        if len(coords) == 0:
            raise ValueError("coords is empty — cannot create spatial folds")
        bounds = np.array([coords.min(axis=0), coords.max(axis=0)])
        n_blocks_x = int(np.ceil((bounds[1, 0] - bounds[0, 0]) / block_size))
        n_blocks_y = int(np.ceil((bounds[1, 1] - bounds[0, 1]) / block_size))

        block_assignments = np.zeros(n_samples, dtype=int)
        for i, (x, y_coord) in enumerate(coords):
            block_x = int((x - bounds[0, 0]) / block_size)
            block_y = int((y_coord - bounds[0, 1]) / block_size)
            block_assignments[i] = block_x * n_blocks_y + block_y

        unique_blocks = np.unique(block_assignments)
        rng = np.random.RandomState(42)
        rng.shuffle(unique_blocks)

        if stratify_y:
            block_means = [y[block_assignments == block].mean() for block in unique_blocks]
            sorted_blocks = [x for _, x in sorted(zip(block_means, unique_blocks))]
        else:
            sorted_blocks = unique_blocks

        folds = []
        for fold_idx in range(n_splits):
            fold_blocks = sorted_blocks[fold_idx::n_splits]
            test_mask = np.isin(block_assignments, fold_blocks)
            test_idx = np.where(test_mask)[0]

            if buffer_size > 0:
                from scipy.spatial import cKDTree
                print(f"Applying {buffer_size}m buffer to fold {fold_idx + 1}")
                tree = cKDTree(coords[test_idx])
                dist, _ = tree.query(coords, k=1)
                buffer_mask = dist <= buffer_size
                excluded_count = np.sum(buffer_mask & ~test_mask)
                print(f"  Excluded {excluded_count} training points within buffer zone")
            else:
                buffer_mask = np.zeros(len(coords), dtype=bool)

            train_idx = np.where(~(test_mask | buffer_mask))[0]
            folds.append((train_idx, test_idx))

    elif method == 'kmeans':
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_splits, random_state=42)
        cluster_labels = kmeans.fit_predict(coords)

        folds = []
        for fold_idx in range(n_splits):
            test_mask = cluster_labels == fold_idx
            test_idx = np.where(test_mask)[0]

            if buffer_size > 0:
                from scipy.spatial import cKDTree
                print(f"Applying {buffer_size}m buffer to fold {fold_idx + 1}")
                tree = cKDTree(coords[test_idx])
                dist, _ = tree.query(coords, k=1)
                buffer_mask = dist <= buffer_size
                excluded_count = np.sum(buffer_mask & ~test_mask)
                print(f"  Excluded {excluded_count} training points within buffer zone")
            else:
                buffer_mask = np.zeros(len(coords), dtype=bool)

            train_idx = np.where(~(test_mask | buffer_mask))[0]
            folds.append((train_idx, test_idx))

    else:
        raise ValueError("Method must be 'block' or 'kmeans'")

    for i, (train_idx, test_idx) in enumerate(folds):
        print(f"Fold {i+1}: Train={len(train_idx)}, Test={len(test_idx)}")
        if len(test_idx) < 50:
            print(f"Warning: Fold {i+1} has very small test set ({len(test_idx)} samples)")

    return folds


def latent_guided_spatial_kfold(
    X,
    y,
    coords,
    n_splits=5,
    block_size=None,
    buffer_size=0,
    method="block",
    stratify_y=True,
    prior_trunk=None,
    alpha_val: float = 1.0,
    vsba_scoring: bool = False,
    corr_payload: dict | None = None,
    vsba_n_epochs: int = 20,
):
    """
    Wrapper around ``spatial_kfold_enhanced`` that annotates folds with
    Area-of-Applicability (AoA) dissimilarity scores when a prior-city
    trunk is available (multi-city continual mode).

    When ``prior_trunk`` is ``None`` (single-city mode), this is a pure
    pass-through to ``spatial_kfold_enhanced``.
    """
    folds = spatial_kfold_enhanced(
        X=X, y=y, coords=coords, n_splits=n_splits,
        block_size=block_size, buffer_size=buffer_size,
        method=method, stratify_y=stratify_y,
    )

    if prior_trunk is None:
        return folds

    try:
        import torch
        from sklearn.metrics.pairwise import euclidean_distances
        from sklearn.neighbors import NearestNeighbors as _NNModel

        X_arr = np.asarray(X, dtype=np.float32)
        N = len(X_arr)
        physics_t = torch.from_numpy(X_arr)
        alpha_t = torch.full((N, 1), alpha_val, dtype=torch.float32)

        with torch.no_grad():
            embeddings = prior_trunk.encode(physics_t, alpha_t).cpu().numpy()

        rng_aoa = np.random.RandomState(42)
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            emb_train = embeddings[train_idx]
            emb_test = embeddings[test_idx]
            n_sample = min(500, len(emb_train))
            sample_idx = rng_aoa.choice(len(emb_train), n_sample, replace=False)
            emb_sample = emb_train[sample_idx]
            d_tt = euclidean_distances(emb_sample, emb_sample)
            off_diag = d_tt[np.triu_indices(n_sample, k=1)]
            mean_d_train = float(off_diag.mean()) if len(off_diag) > 0 else 1.0
            if mean_d_train == 0.0:
                mean_d_train = 1.0
            nn_model = _NNModel(n_neighbors=1, algorithm="ball_tree")
            nn_model.fit(emb_train)
            min_dists, _ = nn_model.kneighbors(emb_test)
            D_i = min_dists[:, 0] / mean_d_train
            ood_frac = float((D_i > 1.5).mean())
            if ood_frac > 0.30:
                print(
                    f"Fold {fold_idx + 1}: {ood_frac * 100:.0f}% OOD test points "
                    f"(AoA D > 1.5) — flagged as out-of-distribution"
                )
            else:
                print(f"Fold {fold_idx + 1} AoA: {ood_frac * 100:.0f}% OOD (D > 1.5, threshold 30%)")
    except Exception as exc:
        print(f"Latent AoA annotation failed (non-fatal): {exc}")

    if vsba_scoring:
        try:
            X_arr = np.asarray(X, dtype=np.float32)
            coords_arr = np.asarray(coords, dtype=np.float32)
            _vsba_fold_quality_score(
                X_arr, coords_arr, folds,
                corr_payload=corr_payload,
                n_epochs=vsba_n_epochs,
            )
        except Exception as exc:
            print(f"VSBA fold scoring failed (non-fatal): {exc}")

    return folds


def _vsba_fold_quality_score(
    X: np.ndarray,
    coords: np.ndarray,
    folds: list,
    corr_payload: dict | None = None,
    n_epochs: int = 20,
    latent_dim: int = 16,
    device: str = "cpu",
) -> list[dict]:
    """Train a VSBA on training-fold data and score each test fold by ELBO.

    Returns list of per-fold dicts with keys:
    ``{"fold": int, "elbo_score": float, "ood_label": bool, "elbo_percentile": float}``
    """
    try:
        from sparc.models.vsba import VariationalSpatialBlockAutoencoder
    except ImportError:
        print("VSBA: sparc.models.vsba not available — skipping fold scoring")
        return []

    n_features = X.shape[1]
    all_train_idx = np.unique(
        np.concatenate([tr for tr, _ in folds]) if folds else np.arange(len(X))
    )
    X_train = X[all_train_idx].astype(np.float32)
    coords_train = coords[all_train_idx].astype(np.float32)

    vsba = VariationalSpatialBlockAutoencoder.from_correlogram(
        corr_payload, n_features=n_features, latent_dim=latent_dim,
    )
    try:
        vsba.fit(X_train, coords_train, n_epochs=n_epochs, device=device)
    except Exception as exc:
        print(f"VSBA training failed (non-fatal): {exc}")
        return []

    scores = []
    for fold_idx, (_, test_idx) in enumerate(folds):
        if len(test_idx) == 0:
            scores.append({"fold": fold_idx + 1, "elbo_score": float("nan"),
                           "ood_label": False, "elbo_percentile": 50.0})
            continue
        try:
            score = vsba.elbo_score(
                X[test_idx].astype(np.float32),
                coords[test_idx].astype(np.float32),
                device=device,
            )
        except Exception:
            score = float("nan")
        scores.append({"fold": fold_idx + 1, "elbo_score": score,
                       "ood_label": False, "elbo_percentile": float("nan")})

    valid_scores = [s["elbo_score"] for s in scores if not math.isnan(s["elbo_score"])]
    if valid_scores:
        p10 = float(np.percentile(valid_scores, 10))
        for s in scores:
            if not math.isnan(s["elbo_score"]):
                pct = float(np.mean(np.array(valid_scores) <= s["elbo_score"]) * 100)
                s["elbo_percentile"] = pct
                s["ood_label"] = s["elbo_score"] <= p10
                label = "OOD" if s["ood_label"] else "in-dist"
                print(f"  Fold {s['fold']} VSBA ELBO: {s['elbo_score']:.4f} (p{pct:.0f}, {label})")

    return scores


# ---------------------------------------------------------------------------
# SpatialFoldPlan + SpatialFoldFactory
# ---------------------------------------------------------------------------

from dataclasses import dataclass as _dataclass
from typing import Any as _Any


@_dataclass
class SpatialFoldPlan:
    """Result of :meth:`SpatialFoldFactory.build`.

    Attributes
    ----------
    folds : list[tuple[np.ndarray, np.ndarray]]
        Each element is ``(train_indices, test_indices)`` for one CV fold.
    block_size_m : float
        Spatial block size used to generate the folds (metres).
    buffer_size_m : float
        Exclusion buffer between train and test blocks (metres).
    """

    folds: list
    block_size_m: float
    buffer_size_m: float


class SpatialFoldFactory:
    """Centralised factory for Stage 2 spatial CV fold generation.

    Eliminates the scattered block-size resolution, fold-size validation,
    and buffer calculation that previously lived inside
    ``EnhancedSpatialCV``.

    Usage
    -----
    ::

        plan = SpatialFoldFactory.build(config, X, y, coords)
        for train_idx, test_idx in plan.folds:
            ...

    Block-size priority
    -------------------
    1. ``config["block_size"]`` — explicit user override
    2. ``config["_correlogram_result"]["spatial_cv_configuration"]["optimal_block_size"]``
    3. :attr:`DEFAULT_BLOCK_SIZE_M` (500 m)

    Buffer-size priority
    --------------------
    1. ``config["buffer_size"]`` — explicit user override
    2. ``max(100, int(block_size / 3))``
    """

    MIN_FOLD_SAMPLES: int = 50
    DEFAULT_BLOCK_SIZE_M: float = 500.0

    @staticmethod
    def build(
        config: dict,
        X: "np.ndarray",
        y: "np.ndarray",
        coords: "np.ndarray",
    ) -> SpatialFoldPlan:
        """Generate spatial CV folds and return a :class:`SpatialFoldPlan`.

        Parameters
        ----------
        config : dict
            Pipeline configuration dict.  Recognised keys:
            ``block_size``, ``buffer_size``, ``n_splits``,
            ``_correlogram_result``.
        X : np.ndarray, shape (n, p)
        y : np.ndarray, shape (n,)
        coords : np.ndarray, shape (n, 2)

        Returns
        -------
        SpatialFoldPlan

        Raises
        ------
        ValueError
            When any test fold contains fewer than :attr:`MIN_FOLD_SAMPLES`
            samples.
        """
        block_size = SpatialFoldFactory._resolve_block_size(config)
        buffer_size = SpatialFoldFactory._resolve_buffer_size(config, block_size)
        n_splits = config.get("n_splits", 5)

        folds = spatial_kfold_enhanced(
            X, y, coords,
            n_splits=n_splits,
            block_size=block_size,
            buffer_size=buffer_size,
            method="block",
            stratify_y=True,
        )
        SpatialFoldFactory._validate_fold_sizes(folds)
        return SpatialFoldPlan(
            folds=folds,
            block_size_m=block_size,
            buffer_size_m=buffer_size,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_block_size(config: dict) -> float:
        # 1. Explicit user config
        if config.get("block_size") is not None:
            return float(config["block_size"])
        # 2. Correlogram recommendation
        corr = config.get("_correlogram_result") or {}
        opt = (corr.get("spatial_cv_configuration") or {}).get("optimal_block_size")
        if opt is not None:
            return float(opt)
        # 3. Hard default
        return SpatialFoldFactory.DEFAULT_BLOCK_SIZE_M

    @staticmethod
    def _resolve_buffer_size(config: dict, block_size: float) -> float:
        if config.get("buffer_size") is not None:
            return float(config["buffer_size"])
        return float(max(100, int(block_size / 3)))

    @staticmethod
    def _validate_fold_sizes(folds: list) -> None:
        for i, (train_idx, test_idx) in enumerate(folds):
            if len(test_idx) < SpatialFoldFactory.MIN_FOLD_SAMPLES:
                raise ValueError(
                    f"Fold {i} has only {len(test_idx)} test samples "
                    f"(minimum {SpatialFoldFactory.MIN_FOLD_SAMPLES}). "
                    "Increase block_size or reduce n_splits."
                )
