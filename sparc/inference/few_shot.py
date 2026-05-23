"""
Few-shot prediction stub for SPARC V4.

V4 will support few-shot calibration: given a small number of ground-truth
observations (e.g. 10–50 points), rapidly fine-tune the city head while
keeping the shared trunk frozen.  This module defines the interface;
implementation is deferred to V4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from sparc.data.satellite_types import SatelliteFeatureSet


@dataclass
class FewShotPrediction:
    """
    Container for few-shot inference results.

    Attributes
    ----------
    y_pred : (N,) predicted target values
    uncertainty : (N,) prediction uncertainty
    coords : (N, 2) coordinates
    n_calibration : number of ground-truth points used for calibration
    calibration_r2 : R² on calibration points (LOO-CV)
    metadata : additional info
    """
    y_pred: np.ndarray
    uncertainty: np.ndarray
    coords: np.ndarray
    n_calibration: int = 0
    calibration_r2: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def few_shot_predict(
    features: SatelliteFeatureSet,
    calibration_X: np.ndarray,
    calibration_y: np.ndarray,
    calibration_coords: np.ndarray,
    trunk_path: str | None = None,
    registry_path: str | None = None,
    n_finetune_epochs: int = 50,
) -> FewShotPrediction:
    """
    Few-shot inference with rapid calibration.

    Parameters
    ----------
    features : SatelliteFeatureSet for the target city
    calibration_X : (K, D) feature matrix for calibration points
    calibration_y : (K,) target values for calibration points
    calibration_coords : (K, 2) coordinates for calibration points
    trunk_path : path to a trained shared trunk checkpoint
    registry_path : path to city registry (loads global trunk)
    n_finetune_epochs : epochs for city head fine-tuning

    Returns
    -------
    FewShotPrediction
    """
    import torch
    import numpy as np
    from sparc.inference.anp import SpatialANP

    # Build target feature matrix
    coords = features.coords                  # (N, 2)
    band_mat = features.to_feature_matrix()   # (N, B)
    if band_mat.shape[1] > 0:
        feat_np = np.concatenate([coords, band_mat], axis=1).astype("float32")
    else:
        feat_np = coords.astype("float32")

    N = feat_np.shape[0]
    x_dim = feat_np.shape[1]

    # Build context feature matrix from calibration inputs
    # calibration_coords (K, 2) + calibration_X (K, D)
    calib_X_np = calibration_X.astype("float32")       # (K, D)
    calib_c_np = calibration_coords.astype("float32")  # (K, 2)
    ctx_feat = np.concatenate([calib_c_np, calib_X_np], axis=1)  # (K, 2+D)

    # Pad or truncate context features to match x_dim
    K, ctx_dim = ctx_feat.shape
    if ctx_dim < x_dim:
        ctx_feat = np.pad(ctx_feat, ((0, 0), (0, x_dim - ctx_dim)))
    elif ctx_dim > x_dim:
        ctx_feat = ctx_feat[:, :x_dim]

    target_t = torch.from_numpy(feat_np)
    ctx_x_t = torch.from_numpy(ctx_feat)
    ctx_y_t = torch.from_numpy(calibration_y.astype("float32")).unsqueeze(1)  # (K, 1)

    anp = SpatialANP(x_dim=x_dim)
    anp.eval()
    with torch.no_grad():
        mean, std = anp(ctx_x_t, ctx_y_t, target_t)

    return FewShotPrediction(
        y_pred=mean.squeeze(1).numpy(),
        uncertainty=std.squeeze(1).numpy(),
        coords=coords,
        n_calibration=K,
    )
