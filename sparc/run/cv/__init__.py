"""
sparc/run/cv/ — Spatial cross-validation engine and fold utilities.

The primary interface for all CV work is :class:`SpatialCVEngine`.
The fold sub-modules (factory, trainer, scaler) are implementation;
callers should not import them directly.

Public interface::

    from sparc.run.cv import SpatialCVEngine, CVResult

    engine = SpatialCVEngine(ctx, fast=False)
    folds  = engine.build_folds(coords, strategy="kmeans")
    result = engine.run()

    # result.oof_predictions, result.fold_metrics, result.rmse ...
"""

from sparc.run.cv_engine import SpatialCVEngine, CVResult

__all__ = [
    "SpatialCVEngine",
    "CVResult",
]
