"""
sparc/run/cv_engine.py — SpatialCVEngine: deep module for spatial cross-validation.

Before this module, the CV sub-system leaked four separate modules across callers:

    from sparc.run.enhanced_spatial_cv import run_spatial_cv, build_oof_ensemble
    from sparc.run.spatial_fold_factory import spatial_kfold_enhanced
    from sparc.run.fold_trainer import FoldTrainer
    from sparc.run.fold_scaler import FoldScaler

Three other stage runners (``v2_neural_training``, ``gwen_variable_selection``,
``orchestrator``) imported these independently, meaning CV concerns spread across
the codebase.

Now one interface covers the whole sub-system::

    engine = SpatialCVEngine(ctx)
    folds  = engine.build_folds(coords, strategy="kmeans")
    result = engine.run(fast=False)

The fold factory, trainer, and scaler are implementation — they do not appear
in the external interface.

Interface
---------
::

    SpatialCVEngine(ctx: RunContext, *, fast: bool = False)
    engine.build_folds(coords, strategy="kmeans") → list[tuple[ndarray, ndarray]]
    engine.run(fast: bool | None = None) → CVResult
    engine.oof_predictions → np.ndarray   (after run())
    engine.feature_importance → pd.DataFrame | None
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CVResult — the output contract
# ---------------------------------------------------------------------------


@dataclass
class CVResult:
    """Output contract from :meth:`SpatialCVEngine.run`.

    Attributes
    ----------
    oof_predictions : np.ndarray
        Out-of-fold predictions assembled across all folds, shape ``(N,)``.
    oof_std : np.ndarray or None
        Per-prediction uncertainty, same shape as ``oof_predictions``, when
        the ensemble produces uncertainty estimates.
    fold_metrics : list[dict]
        Per-fold evaluation metrics (RMSE, Moran's I of residuals, etc.).
    selected_features : list[str] or None
        Features retained after GWEN selection (when selection was run).
    n_folds : int
        Number of CV folds actually executed.
    fast_mode : bool
        Whether fast mode was used (reduced folds / epochs).
    """

    oof_predictions: np.ndarray
    oof_std: Optional[np.ndarray] = None
    fold_metrics: List[Dict[str, Any]] = field(default_factory=list)
    selected_features: Optional[List[str]] = None
    n_folds: int = 0
    fast_mode: bool = False

    @property
    def rmse(self) -> float:
        """Mean RMSE across folds (returns 0.0 when fold_metrics is empty)."""
        rmses = [m["rmse"] for m in self.fold_metrics if "rmse" in m]
        return float(np.mean(rmses)) if rmses else 0.0


# ---------------------------------------------------------------------------
# SpatialCVEngine — the deep module
# ---------------------------------------------------------------------------


class SpatialCVEngine:
    """Single interface to the SPARC spatial cross-validation sub-system.

    Parameters
    ----------
    ctx : RunContext
        Pipeline context carrying config, paths, and registry.
    fast : bool
        When ``True``, use reduced folds and epochs for quick iteration.
    """

    def __init__(self, ctx: Any, *, fast: bool = False) -> None:
        self._ctx = ctx
        self._fast = fast
        self._result: Optional[CVResult] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build_folds(
        self,
        coords: np.ndarray,
        *,
        strategy: str = "kmeans",
        n_folds: Optional[int] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Construct spatial CV fold indices.

        Parameters
        ----------
        coords:
            Spatial coordinates, shape ``(N, 2)``.
        strategy:
            Fold construction strategy: ``"kmeans"``, ``"grid"``, or ``"knn"``.
        n_folds:
            Override the number of folds; defaults to
            ``config['cv_params']['n_spatial_folds']``.

        Returns
        -------
        list of (train_idx, test_idx) tuples
        """
        # Implementation lives in spatial_fold_factory — private to this module
        from sparc.run.spatial_fold_factory import spatial_kfold_enhanced

        cfg = self._ctx.config
        k = n_folds or cfg.get("cv_params", {}).get("n_spatial_folds", 5)
        if self._fast:
            k = min(k, 3)

        return spatial_kfold_enhanced(
            coords,
            n_splits=k,
            strategy=strategy,
        )

    def run(self, fast: Optional[bool] = None) -> CVResult:
        """Execute the full spatial CV pipeline for Stage 2.

        Parameters
        ----------
        fast:
            Override the instance-level ``fast`` setting for this run only.

        Returns
        -------
        CVResult
        """
        _fast = fast if fast is not None else self._fast

        # Delegate to the Stage 2 runner — the implementation
        from sparc.run.enhanced_spatial_cv import main as _cv_main

        try:
            _cv_main(self._ctx, fast_mode=_fast)
        except TypeError:
            # Older signature: main(fast_mode=bool)
            _cv_main(fast_mode=_fast)

        # Collect result from registry
        self._result = self._read_result_from_registry()
        return self._result

    @property
    def oof_predictions(self) -> Optional[np.ndarray]:
        """Out-of-fold predictions from the last :meth:`run` call."""
        if self._result is None:
            return None
        return self._result.oof_predictions

    @property
    def last_result(self) -> Optional[CVResult]:
        """The :class:`CVResult` from the last :meth:`run` call."""
        return self._result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _read_result_from_registry(self) -> CVResult:
        """Read OOF predictions and fold metrics from the artifact registry."""
        reg = getattr(self._ctx, "registry", None)
        if reg is None:
            return CVResult(oof_predictions=np.array([]))

        oof = np.array([])
        oof_std = None
        fold_metrics: List[Dict[str, Any]] = []

        try:
            from sparc.registry.store import get_active_store
            store = get_active_store()
            if store is not None and store.has("2", "oof_predictions"):
                oof = np.asarray(store.read_any("2", "oof_predictions"))
            if store is not None and store.has("2", "oof_std"):
                oof_std = np.asarray(store.read_any("2", "oof_std"))
            if store is not None and store.has("2", "fold_metrics"):
                raw = store.read_any("2", "fold_metrics")
                if isinstance(raw, list):
                    fold_metrics = raw
        except Exception:
            pass

        return CVResult(
            oof_predictions=oof,
            oof_std=oof_std,
            fold_metrics=fold_metrics,
            fast_mode=self._fast,
        )

    def __repr__(self) -> str:
        return f"SpatialCVEngine(fast={self._fast}, ran={self._result is not None})"
