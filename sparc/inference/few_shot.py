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

    Raises
    ------
    NotImplementedError
        V4 feature — not yet implemented.
    """
    raise NotImplementedError(
        "Few-shot prediction is a V4 feature. "
        "Use transfer learning (V3) for cross-city prediction."
    )
