"""
feature_matrix.py — Lightweight type wrappers that make the scaling contract
between callers and base-model training explicit at the seam boundary.

The Stage 2 base models (OLS, GWR, GWRF, GGPGAM) all apply their own
internal scalers.  ``generate_optimized_oof_predictions`` also applies a
*canonical* scaler as a pre-pass.  Callers must therefore always pass
**unscaled** feature data — passing already-scaled features would silently
double-scale and corrupt every base-model fit.

These two dataclasses encode that contract at the type level so that:

* ``generate_optimized_oof_predictions`` can assert it received
  ``RawFeatureMatrix`` (unscaled) rather than ``ScaledFeatureMatrix``.
* Readers of the call site immediately know what scaling state the array is in.

Usage
-----
::

    from sparc.run.feature_matrix import RawFeatureMatrix, ScaledFeatureMatrix
    from sklearn.preprocessing import StandardScaler

    raw = RawFeatureMatrix(data=X_unscaled)
    # pass raw to generate_optimized_oof_predictions — it will scale internally

    scaler = StandardScaler().fit(X_unscaled)
    scaled = ScaledFeatureMatrix(data=scaler.transform(X_unscaled), scaler=scaler)
    # pass scaled to meta-ensemble — it expects pre-scaled input
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RawFeatureMatrix:
    """A feature matrix that has **not** been scaled.

    Pass this to ``generate_optimized_oof_predictions``, which applies a
    canonical StandardScaler internally before distributing to each base model.

    Attributes
    ----------
    data : np.ndarray
        Shape ``(n_samples, n_features)``.  Values are in their original units.
    """

    data: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.data, np.ndarray):
            self.data = np.asarray(self.data, dtype=np.float64)

    def __len__(self) -> int:  # noqa: D105
        return len(self.data)

    @property
    def shape(self):  # noqa: D102
        return self.data.shape


@dataclass
class ScaledFeatureMatrix:
    """A feature matrix that **has** been scaled by ``scaler``.

    Pass this to components that consume pre-scaled input (e.g., the
    meta-ensemble, neural trunk) so they do not need to re-scale.

    Attributes
    ----------
    data : np.ndarray
        Shape ``(n_samples, n_features)``.  Values are z-scored / normalised.
    scaler : object
        The fitted sklearn-compatible scaler used to produce ``data``.
        Must expose ``.transform()`` for applying the same transform to new
        batches (e.g., at inference time).
    """

    data: np.ndarray
    scaler: Any = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.data, np.ndarray):
            self.data = np.asarray(self.data, dtype=np.float64)

    def __len__(self) -> int:  # noqa: D105
        return len(self.data)

    @property
    def shape(self):  # noqa: D102
        return self.data.shape
