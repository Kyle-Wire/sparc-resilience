"""
Zero-shot prediction stub for SPARC V4.

V4 will use satellite-derived features + climate zone embedding
to predict spatial fields for unseen cities without any ground-truth
calibration data.  This module defines the interface; implementation
is deferred to V4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from sparc.data.satellite_types import SatelliteFeatureSet


@dataclass
class ZeroShotPrediction:
    """
    Container for zero-shot inference results.

    Attributes
    ----------
    y_pred : (N,) predicted target values
    uncertainty : (N,) prediction uncertainty (e.g. ensemble std)
    coords : (N, 2) coordinates
    climate_zone : Köppen zone used for conditioning
    metadata : additional info (model version, trunk hash, etc.)
    """
    y_pred: np.ndarray
    uncertainty: np.ndarray
    coords: np.ndarray
    climate_zone: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def zero_shot_predict(
    features: SatelliteFeatureSet,
    trunk_path: str | None = None,
    registry_path: str | None = None,
) -> ZeroShotPrediction:
    """
    Zero-shot inference for an unseen city.

    Parameters
    ----------
    features : SatelliteFeatureSet from remote sensing
    trunk_path : path to a trained shared trunk checkpoint
    registry_path : path to city registry (loads global trunk)

    Returns
    -------
    ZeroShotPrediction
    """
    import torch
    from sparc.inference.anp import SpatialANP

    # Build feature matrix from satellite features
    coords = features.coords  # (N, 2)
    band_mat = features.to_feature_matrix()  # (N, B) — may be (N, 0)
    if band_mat.shape[1] > 0:
        import numpy as np
        feat_np = np.concatenate([coords, band_mat], axis=1).astype("float32")
    else:
        feat_np = coords.astype("float32")

    N = feat_np.shape[0]
    x_dim = feat_np.shape[1]

    target_x = torch.from_numpy(feat_np)                 # (N, x_dim)
    ctx_x = torch.zeros(0, x_dim, dtype=torch.float32)  # empty context
    ctx_y = torch.zeros(0, 1, dtype=torch.float32)

    anp = SpatialANP(x_dim=x_dim)
    anp.eval()
    with torch.no_grad():
        mean, std = anp(ctx_x, ctx_y, target_x)

    return ZeroShotPrediction(
        y_pred=mean.squeeze(1).numpy(),
        uncertainty=std.squeeze(1).numpy(),
        coords=coords,
    )
