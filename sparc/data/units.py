"""Consistent unit conversion for the SPARC pipeline.

All internal pipeline computations operate on z-scored (standardized)
values.  This module provides the single conversion point between
z-scores and original data units (e.g. °F, percentages).

Rule: No z-score should ever reach the UI layer.  All API responses
must call ``z_to_original()`` before returning values to the frontend.
"""

from __future__ import annotations

import numpy as np


def z_to_original(
    z_values: np.ndarray | float,
    mean: float,
    std: float,
) -> np.ndarray | float:
    """Convert standardised z-scores back to original data units.

    Parameters
    ----------
    z_values : array or scalar
        Values in z-score space.
    mean : float
        Mean of the original variable (before standardisation).
    std : float
        Standard deviation of the original variable.

    Returns
    -------
    Values in the original data scale.
    """
    if std == 0:
        return np.full_like(z_values, mean) if isinstance(z_values, np.ndarray) else mean
    return z_values * std + mean


def original_to_z(
    values: np.ndarray | float,
    mean: float,
    std: float,
) -> np.ndarray | float:
    """Convert original-scale values to z-scores.

    Parameters
    ----------
    values : array or scalar
        Values in original units.
    mean : float
        Mean of the original variable.
    std : float
        Standard deviation of the original variable.

    Returns
    -------
    Values in z-score space.
    """
    if std == 0:
        return np.zeros_like(values) if isinstance(values, np.ndarray) else 0.0
    return (values - mean) / std


def delta_z_to_original(
    delta_z: np.ndarray | float,
    std: float,
) -> np.ndarray | float:
    """Convert a z-score *difference* to original-scale units.

    For deltas (e.g. scenario cooling), the mean cancels out:
        Δ_original = Δ_z × std

    Parameters
    ----------
    delta_z : array or scalar
        Change expressed in z-score units.
    std : float
        Standard deviation of the target variable.

    Returns
    -------
    Change in original data units (e.g. °F).
    """
    if std == 0:
        return np.zeros_like(delta_z) if isinstance(delta_z, np.ndarray) else 0.0
    return delta_z * std
