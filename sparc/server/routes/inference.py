"""sparc.server.routes.inference — /inference/* endpoints.

Migrated from app.py.  Importable independently of the full FastAPI app.

Currently migrated:
  - /inference/zero-shot
  - /inference/few-shot
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(tags=["inference"])


class _ZeroShotRequest(BaseModel):
    """Coordinates and optional band matrix for zero-shot inference.

    coords: list of [lat, lon] pairs, shape (N, 2)
    bands:  optional list of feature vectors, shape (N, F).  When omitted,
            only coordinate features are used.
    """

    coords: list[list[float]]
    bands: Optional[list[list[float]]] = None


class _FewShotRequest(_ZeroShotRequest):
    """Zero-shot request augmented with calibration observations.

    calibration_coords: shape (K, 2)
    calibration_X:      shape (K, F) — must match bands width if supplied
    calibration_y:      shape (K,)   — observed response values
    """

    calibration_coords: list[list[float]]
    calibration_X: list[list[float]]
    calibration_y: list[float]


@router.post("/inference/zero-shot")
async def inference_zero_shot(body: _ZeroShotRequest):
    """Run SpatialANP zero-shot inference over a target grid.

    Returns per-point ``mean`` and ``uncertainty`` estimates.
    No calibration data is required; the model uses its learned prior.
    """
    import numpy as _np
    from sparc.inference.zero_shot import zero_shot_predict
    from sparc.data.satellite_types import SatelliteFeatureSet

    coords = _np.asarray(body.coords, dtype=_np.float32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise HTTPException(400, "coords must be a list of [lat, lon] pairs")

    if body.bands is not None:
        bands = _np.asarray(body.bands, dtype=_np.float32)
        if bands.shape[0] != coords.shape[0]:
            raise HTTPException(400, "bands row count must match coords row count")

    features = SatelliteFeatureSet(coords=coords)

    try:
        result = zero_shot_predict(features=features)
    except Exception as exc:
        raise HTTPException(500, f"Zero-shot inference failed: {exc}") from exc

    return {
        "mean": result.y_pred.tolist(),
        "uncertainty": result.uncertainty.tolist(),
        "n_points": int(coords.shape[0]),
        "method": "SpatialANP-zero-shot",
    }


@router.post("/inference/few-shot")
async def inference_few_shot(body: _FewShotRequest):
    """Run SpatialANP few-shot inference calibrated on known observations.

    Returns per-point ``mean`` and ``uncertainty`` estimates together with
    the number of calibration points used.
    """
    import numpy as _np
    from sparc.inference.few_shot import few_shot_predict
    from sparc.data.satellite_types import SatelliteFeatureSet

    coords = _np.asarray(body.coords, dtype=_np.float32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise HTTPException(400, "coords must be a list of [lat, lon] pairs")

    if body.bands is not None:
        bands = _np.asarray(body.bands, dtype=_np.float32)
        if bands.shape[0] != coords.shape[0]:
            raise HTTPException(400, "bands row count must match coords row count")

    calib_coords = _np.asarray(body.calibration_coords, dtype=_np.float32)
    calib_X = _np.asarray(body.calibration_X, dtype=_np.float32)
    calib_y = _np.asarray(body.calibration_y, dtype=_np.float32)

    if calib_coords.ndim != 2 or calib_coords.shape[1] != 2:
        raise HTTPException(400, "calibration_coords must be a list of [lat, lon] pairs")
    if calib_X.shape[0] != calib_coords.shape[0] or calib_y.shape[0] != calib_coords.shape[0]:
        raise HTTPException(
            400,
            "calibration_coords, calibration_X, and calibration_y must have the same number of rows",
        )

    features = SatelliteFeatureSet(coords=coords)

    try:
        result = few_shot_predict(
            features=features,
            calibration_X=calib_X,
            calibration_y=calib_y,
            calibration_coords=calib_coords,
        )
    except Exception as exc:
        raise HTTPException(500, f"Few-shot inference failed: {exc}") from exc

    return {
        "mean": result.y_pred.tolist(),
        "uncertainty": result.uncertainty.tolist(),
        "n_calibration": result.n_calibration,
        "n_points": int(coords.shape[0]),
        "method": "SpatialANP-few-shot",
    }
