"""InferenceDispatcher — single seam for zero-shot and few-shot prediction.

Callers never choose between zero_shot_predict / few_shot_predict.
Instead they supply features and optionally calibration data; the
dispatcher decides which path to take based on the number of calibration
points (K).

  K == 0  →  zero-shot path  (no context, learned prior)
  K >= 1  →  few-shot path   (calibration points as ANP context)

This concentrates the dispatch logic in one place (locality) and
prevents the HTTP layer from needing to know about inference modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sparc.data.satellite_types import SatelliteFeatureSet


@dataclass
class Prediction:
    """Unified prediction container for both inference modes.

    Attributes
    ----------
    y_pred       : (N,) predicted target values
    uncertainty  : (N,) prediction uncertainty (std)
    coords       : (N, 2) coordinates matching features.coords
    mode         : "zero_shot" | "few_shot"
    n_calibration: number of calibration points used (0 for zero-shot)
    metadata     : additional diagnostic info
    """
    y_pred: np.ndarray
    uncertainty: np.ndarray
    coords: np.ndarray
    mode: str
    n_calibration: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class InferenceDispatcher:
    """Owns the zero-vs-few-shot decision so callers don't have to.

    Usage
    -----
    >>> result = InferenceDispatcher().predict(features)
    >>> result = InferenceDispatcher().predict(features, calib_X, calib_y, calib_coords)
    """

    def predict(
        self,
        features: SatelliteFeatureSet,
        calibration_X: np.ndarray | None = None,
        calibration_y: np.ndarray | None = None,
        calibration_coords: np.ndarray | None = None,
    ) -> Prediction:
        """Predict at all locations in *features*.

        If calibration arrays are provided and non-empty (K > 0), the
        few-shot (ANP context) path is used.  Otherwise the zero-shot
        (learned prior) path is used.

        Parameters
        ----------
        features           : satellite feature set for target locations
        calibration_X      : (K, D) feature matrix for calibration points, or None
        calibration_y      : (K,) target values for calibration points, or None
        calibration_coords : (K, 2) coordinates for calibration points, or None

        Returns
        -------
        Prediction
        """
        # Determine K — number of calibration points
        K = 0
        if (
            calibration_X is not None
            and calibration_y is not None
            and calibration_coords is not None
        ):
            K = int(np.asarray(calibration_y).shape[0])

        if K == 0:
            return self._zero_shot(features)
        return self._few_shot(features, calibration_X, calibration_y, calibration_coords, K)

    # ------------------------------------------------------------------
    def _zero_shot(self, features: SatelliteFeatureSet) -> Prediction:
        from sparc.inference.zero_shot import zero_shot_predict

        raw = zero_shot_predict(features)
        return Prediction(
            y_pred=raw.y_pred,
            uncertainty=raw.uncertainty,
            coords=raw.coords,
            mode="zero_shot",
            n_calibration=0,
        )

    def _few_shot(
        self,
        features: SatelliteFeatureSet,
        calibration_X: np.ndarray,
        calibration_y: np.ndarray,
        calibration_coords: np.ndarray,
        K: int,
    ) -> Prediction:
        from sparc.inference.few_shot import few_shot_predict

        raw = few_shot_predict(
            features,
            calibration_X=calibration_X,
            calibration_y=calibration_y,
            calibration_coords=calibration_coords,
            n_finetune_epochs=0,  # baseline: context-only, no fine-tuning
        )
        return Prediction(
            y_pred=raw.y_pred,
            uncertainty=raw.uncertainty,
            coords=raw.coords,
            mode="few_shot",
            n_calibration=K,
        )
