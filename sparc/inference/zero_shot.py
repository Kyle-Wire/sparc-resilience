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

    Raises
    ------
    NotImplementedError
        V4 feature — not yet implemented.
    """
    raise NotImplementedError(
        "Zero-shot prediction is a V4 feature. "
        "Use transfer learning (V3) for cross-city prediction."
    )
