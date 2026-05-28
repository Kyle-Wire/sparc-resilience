#!/usr/bin/env python3
"""
DatasetProfiler — Automatic dataset characterisation and adaptive parameter recommendation.

Inspects the loaded data to derive:
- **n_samples**: total row count
- **mean_nn_distance**: average nearest-neighbour distance (metres, projected CRS)
- **spatial_extent**: bounding-box diagonal (metres)
- **point_density**: points per km²
- **available_ram_gb**: usable system RAM
- **n_features**: number of predictor columns
- **size_tier**: ``"small"`` / ``"medium"`` / ``"large"`` / ``"xlarge"``

Based on these metrics, ``recommend_parameters()`` returns a dict of per-model
hyperparameter overrides that are piped directly into the PipelineConfigurator
before Stage 2 begins.

Usage::

    from sparc.run.dataset_profiler import DatasetProfiler

    profiler = DatasetProfiler(data, feature_cols=predictors)
    profile = profiler.profile()        # dict of metrics
    recs = profiler.recommend_parameters()  # per-model hyperparams
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


# ── Tier boundaries (number of observations) ────────────────────────────
TIER_THRESHOLDS = {
    "small":  2_000,
    "medium": 20_000,
    "large":  100_000,
    # anything above 100 000 → "xlarge"
}


class DatasetProfiler:
    """
    Profiles a spatial dataset and recommends adaptive model parameters.

    Parameters
    ----------
    data : pd.DataFrame | gpd.GeoDataFrame
        The preprocessed dataset with projected coordinates.
    coord_cols : list[str], optional
        Two-element list ``[x_col, y_col]`` of projected coordinate columns.
        Defaults to ``["projected_X", "projected_Y"]``.
    feature_cols : list[str], optional
        Predictor column names (for ``n_features``).
    """

    def __init__(
        self,
        data: pd.DataFrame,
        coord_cols: Optional[List[str]] = None,
        feature_cols: Optional[List[str]] = None,
    ):
        self.data = data
        self.coord_cols = coord_cols or ["projected_X", "projected_Y"]
        self.feature_cols = feature_cols or []
        self._profile: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Profiling
    # ------------------------------------------------------------------

    def profile(self, max_nn_sample: int = 5000) -> Dict[str, Any]:
        """
        Compute all dataset metrics.  Result is cached.

        Parameters
        ----------
        max_nn_sample : int
            Max points to subsample when computing mean nearest-neighbour
            distance (expensive O(n²) operation).

        Returns
        -------
        dict with keys: n_samples, mean_nn_distance, spatial_extent,
            point_density, available_ram_gb, n_features, size_tier.
        """
        if self._profile is not None:
            return self._profile

        coords = self.data[self.coord_cols].values.astype(np.float64)
        n_samples = len(coords)

        if n_samples == 0:
            raise ValueError(
                "DatasetProfiler received an empty dataset. "
                "Ensure data loading and CRS projection succeeded before profiling."
            )

        # ── Spatial extent (bounding-box diagonal) ───────────────────
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        spatial_extent = math.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)

        # ── Nearest-neighbour distance ───────────────────────────────
        mean_nn = self._mean_nn_distance(coords, max_nn_sample)

        # ── Point density (per km²) ─────────────────────────────────
        area_km2 = ((x_max - x_min) / 1000.0) * ((y_max - y_min) / 1000.0)
        point_density = n_samples / max(area_km2, 1e-6)

        # ── Available RAM ────────────────────────────────────────────
        if PSUTIL_AVAILABLE:
            available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
        else:
            available_ram_gb = 8.0  # conservative assumption
            warnings.warn("psutil not installed — assuming 8 GB available RAM")

        # ── Size tier ────────────────────────────────────────────────
        if n_samples < TIER_THRESHOLDS["small"]:
            size_tier = "small"
        elif n_samples < TIER_THRESHOLDS["medium"]:
            size_tier = "medium"
        elif n_samples < TIER_THRESHOLDS["large"]:
            size_tier = "large"
        else:
            size_tier = "xlarge"

        self._profile = {
            "n_samples": n_samples,
            "mean_nn_distance": float(mean_nn),
            "spatial_extent": float(spatial_extent),
            "point_density": float(point_density),
            "available_ram_gb": round(available_ram_gb, 2),
            "n_features": len(self.feature_cols),
            "size_tier": size_tier,
        }
        return self._profile

    # ------------------------------------------------------------------
    # Parameter recommendations
    # ------------------------------------------------------------------

    def recommend_parameters(self) -> Dict[str, Dict[str, Any]]:
        """
        Return per-model hyperparameter recommendations.

        Keys: ``gwrf``, ``ggpgam``, ``meta_ensemble``,
        ``spatial_cv``, ``correlogram``.

        Recommendations adapt to:
        - **dataset size** (tier)
        - **spatial resolution** (mean_nn / extent)
        - **available RAM**
        - **feature count**
        """
        p = self.profile()
        tier = p["size_tier"]
        n = p["n_samples"]
        nn = p["mean_nn_distance"]
        extent = p["spatial_extent"]
        ram = p["available_ram_gb"]
        n_feat = p["n_features"]

        recs: Dict[str, Dict[str, Any]] = {}

        # ── GWRF ─────────────────────────────────────────────────────
        gwrf = self._recommend_gwrf(tier, n, ram)
        recs["gwrf"] = gwrf

        # ── GGPGAM ───────────────────────────────────────────────────
        ggpgam = self._recommend_ggpgam(tier, n, n_feat, ram)
        recs["ggpgam"] = ggpgam

        # ── Meta-Ensemble (Neural / Optuna) ──────────────────────
        meta = self._recommend_meta(tier, n, ram)
        recs["meta_ensemble"] = meta

        # ── Spatial CV ───────────────────────────────────────────────
        recs["spatial_cv"] = {
            "min_test_size": max(30, int(n * 0.03)),
        }

        # ── Correlogram ──────────────────────────────────────────────
        recs["correlogram"] = {
            "max_distance": self._recommend_correlogram_max_distance(extent, nn),
            "max_sample_size": self._recommend_correlogram_sample(tier, n),
            "n_lags": 20 if tier in ("large", "xlarge") else 15,
        }

        return recs

    # ------------------------------------------------------------------
    # Per-model recommendation helpers
    # ------------------------------------------------------------------

    def _recommend_gwrf(self, tier: str, n: int, ram: float) -> Dict[str, Any]:
        """GWRF hyperparameters scaled by tier, size, and RAM."""
        base: Dict[str, Any] = {}

        if tier == "small":
            base["n_estimators"] = 50
            base["k_neighbors"] = min(100, max(20, n // 5))
            base["subsample_n"] = None  # no subsampling needed
        elif tier == "medium":
            base["n_estimators"] = 100
            base["k_neighbors"] = min(150, max(50, n // 10))
            base["subsample_n"] = min(n, 3000) if n > 5000 else None
        elif tier == "large":
            base["n_estimators"] = 100
            base["k_neighbors"] = min(200, max(80, n // 20))
            base["subsample_n"] = min(n, 5000)
        else:  # xlarge
            base["n_estimators"] = 80  # fewer trees to save RAM
            base["k_neighbors"] = min(250, max(100, n // 50))
            base["subsample_n"] = min(n, 8000)

        # RAM guard: if < 8 GB, clip subsample_n
        if ram < 8 and base.get("subsample_n") is not None:
            base["subsample_n"] = min(base["subsample_n"], 2000)  # type: ignore[arg-type]

        base["min_samples_leaf"] = 5
        base["n_jobs"] = 1  # local RF always single-threaded

        return base

    def _recommend_ggpgam(
        self, tier: str, n: int, n_feat: int, ram: float
    ) -> Dict[str, Any]:
        """GGPGAM n_splines, n_spatial_bases, lam scaled by dataset."""
        base: Dict[str, Any] = {}

        if tier == "small":
            base["n_splines"] = max(3, min(5, n_feat))
            base["n_spatial_bases"] = max(5, min(8, n // 100))
            base["lam"] = 0.8   # stronger regularisation for fewer data
            base["max_iter"] = 100
        elif tier == "medium":
            base["n_splines"] = max(5, min(10, n_feat + 2))
            base["n_spatial_bases"] = max(10, min(20, n // 500))
            base["lam"] = 0.6
            base["max_iter"] = 100
        elif tier == "large":
            base["n_splines"] = max(8, min(15, n_feat + 3))
            base["n_spatial_bases"] = max(15, min(30, n // 1000))
            base["lam"] = 0.5
            base["max_iter"] = 150
        else:  # xlarge
            base["n_splines"] = max(10, min(20, n_feat + 4))
            base["n_spatial_bases"] = max(20, min(50, n // 2000))
            base["lam"] = 0.4
            base["max_iter"] = 200

        # RAM guard
        if ram < 8 and tier in ("large", "xlarge"):
            base["n_spatial_bases"] = min(base["n_spatial_bases"], 15)

        return base

    def _recommend_meta(
        self, tier: str, n: int, ram: float
    ) -> Dict[str, Any]:
        """Meta-Ensemble Optuna trial count."""
        base: Dict[str, Any] = {}

        if tier == "small":
            base["n_optuna_trials"] = 30
        elif tier == "medium":
            base["n_optuna_trials"] = 60
        elif tier == "large":
            base["n_optuna_trials"] = 100
        else:  # xlarge
            base["n_optuna_trials"] = 80 if ram < 16 else 120

        return base

    # ------------------------------------------------------------------
    # Correlogram helpers
    # ------------------------------------------------------------------

    def _recommend_correlogram_max_distance(
        self, extent: float, nn_dist: float
    ) -> float:
        """
        Data-driven max lag distance for the correlogram.

        Heuristic: 40 % of the spatial extent, but never less than
        20 × nearest-neighbour distance (to cover enough lags).
        """
        candidate = extent * 0.40
        floor = nn_dist * 20.0
        return max(candidate, floor, 500.0)  # absolute floor of 500 m

    def _recommend_correlogram_sample(self, tier: str, n: int) -> int:
        """Max sample size for correlogram analysis."""
        if tier == "small":
            return n  # use all
        elif tier == "medium":
            return min(n, 3000)
        elif tier == "large":
            return min(n, 5000)
        else:
            return min(n, 8000)

    # ------------------------------------------------------------------
    # Nearest-neighbour utility
    # ------------------------------------------------------------------

    @staticmethod
    def _mean_nn_distance(coords: np.ndarray, max_sample: int) -> float:
        """
        Mean nearest-neighbour distance from a point cloud.

        Uses a KD-tree for efficiency.  Sub-samples if n > *max_sample*.
        """
        from scipy.spatial import cKDTree

        n = len(coords)
        if n < 2:
            return 0.0
        if n > max_sample:
            rng = np.random.default_rng(42)
            idx = rng.choice(n, size=max_sample, replace=False)
            coords = coords[idx]

        tree = cKDTree(coords)
        dists, _ = tree.query(coords, k=2)  # k=2: closest after self
        nn_dists = dists[:, 1]
        return float(np.mean(nn_dists))

    # ------------------------------------------------------------------
    # Pretty-print
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a formatted summary string of the profile."""
        p = self.profile()
        lines = [
            "=" * 60,
            "  DATASET PROFILE",
            "=" * 60,
            f"  Observations      : {p['n_samples']:,}",
            f"  Features          : {p['n_features']}",
            f"  Size tier         : {p['size_tier'].upper()}",
            f"  Spatial extent    : {p['spatial_extent']:,.0f} m",
            f"  Mean NN distance  : {p['mean_nn_distance']:.1f} m",
            f"  Point density     : {p['point_density']:.1f} pts/km²",
            f"  Available RAM     : {p['available_ram_gb']:.1f} GB",
            "=" * 60,
        ]
        return "\n".join(lines)
