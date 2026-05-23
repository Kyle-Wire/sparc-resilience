"""
sparc/models/base.py — SpatialModel ABC and OOFFoldResult contract.

Every Stage 2 base model must satisfy the ``SpatialModel`` interface so that
``train_single_model_fold_worker`` needs zero branching or introspection.

Key types
---------
OOFFoldResult
    Carries predictions, optional uncertainty, nan_fraction, and the
    execution mode so Stage 3 can audit quality without opening the CSV.

OOFAssembler
    Validates ``OOFFoldResult`` instances against a configurable nan
    threshold.  Raises ``ValueError`` instead of silently imputing.

SpatialModel
    ABC that every base model adapter must implement.  Enforces a
    consistent ``fit / predict / validate_for_fold`` surface.

InMemorySpatialModel
    Lightweight adapter backed by numpy mean — used in unit tests and
    as a reference implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np


# ---------------------------------------------------------------------------
# OOFFoldResult
# ---------------------------------------------------------------------------

@dataclass
class OOFFoldResult:
    """Result envelope returned by ``SpatialModel.predict``.

    Attributes
    ----------
    predictions : np.ndarray
        Shape ``(n_test,)``.  May contain NaN if the underlying model
        produced invalid output (see ``nan_fraction``).
    uncertainty : np.ndarray or None
        Per-prediction uncertainty estimates, same shape as ``predictions``.
        *None* when the model does not produce uncertainty.
    nan_fraction : float
        Fraction of ``predictions`` that are NaN.  Computed from
        ``predictions`` by :meth:`from_predictions`; callers may also
        set it directly when constructing manually.
    execution_mode : str
        One of ``"parallel"``, ``"sequential"``, or ``"fallback"``.
        Records how this fold was computed so the quality manifest has
        full provenance.
    """

    predictions: np.ndarray
    uncertainty: Optional[np.ndarray]
    nan_fraction: float
    execution_mode: Literal["parallel", "sequential", "fallback"]

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_predictions(
        cls,
        predictions: np.ndarray,
        uncertainty: Optional[np.ndarray] = None,
        execution_mode: Literal["parallel", "sequential", "fallback"] = "sequential",
    ) -> "OOFFoldResult":
        """Construct from a raw predictions array, computing nan_fraction automatically."""
        preds = np.asarray(predictions, dtype=np.float64)
        nan_fraction = float(np.isnan(preds).mean())
        return cls(
            predictions=preds,
            uncertainty=uncertainty,
            nan_fraction=nan_fraction,
            execution_mode=execution_mode,
        )


# ---------------------------------------------------------------------------
# OOFAssembler
# ---------------------------------------------------------------------------

class OOFAssembler:
    """Validates ``OOFFoldResult`` objects before Stage 3 consumption.

    Parameters
    ----------
    nan_threshold : float
        Inclusive upper bound on acceptable ``nan_fraction``.  Results
        with ``nan_fraction >= nan_threshold`` raise ``ValueError``.
        Default: 0.1 (10 % NaN triggers failure).
    """

    def __init__(self, nan_threshold: float = 0.1) -> None:
        self.nan_threshold = nan_threshold

    def validate(self, result: OOFFoldResult) -> None:
        """Raise ``ValueError`` if *result* exceeds the nan threshold.

        Parameters
        ----------
        result : OOFFoldResult

        Raises
        ------
        ValueError
            When ``result.nan_fraction >= self.nan_threshold``.
        """
        if result.nan_fraction >= self.nan_threshold:
            raise ValueError(
                f"OOFFoldResult nan_fraction={result.nan_fraction:.3f} exceeds "
                f"threshold={self.nan_threshold:.3f}.  Fold execution_mode was "
                f"'{result.execution_mode}'.  Raising instead of silently imputing "
                "to protect Stage 3 causal analysis from degraded signals."
            )


# ---------------------------------------------------------------------------
# SpatialModel ABC
# ---------------------------------------------------------------------------

class SpatialModel(ABC):
    """Abstract base for every Stage 2 spatial base model.

    Implementors must provide ``fit``, ``predict``, and
    ``validate_for_fold``.  The worker function
    ``train_single_model_fold_worker`` only ever calls these three
    methods — no introspection, no model-name gating.

    ``predict`` *must* return an :class:`OOFFoldResult` regardless of
    what the underlying implementation produces (tuple, ndarray, dict).
    Normalisation is the adapter's responsibility.
    """

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        coords: np.ndarray,
    ) -> "SpatialModel":
        """Fit on a training fold.

        Returns
        -------
        self
        """

    @abstractmethod
    def predict(
        self,
        X: np.ndarray,
        coords: np.ndarray,
    ) -> OOFFoldResult:
        """Return predictions for the test fold as an :class:`OOFFoldResult`."""

    @abstractmethod
    def validate_for_fold(self, n_train: int, n_features: int) -> None:
        """Assert or clamp hyperparameters to values safe for this fold size.

        Parameters
        ----------
        n_train : int
            Number of training samples in the fold.
        n_features : int
            Number of input features.

        Raises
        ------
        ValueError
            When the model cannot possibly train on this fold
            (e.g. ``n_train`` is smaller than a hard minimum).
        """


# ---------------------------------------------------------------------------
# InMemorySpatialModel — reference adapter + test double
# ---------------------------------------------------------------------------

class InMemorySpatialModel(SpatialModel):
    """Trivial adapter backed by a constant (training mean) predictor.

    Useful as:
    * A reference implementation that documents the contract.
    * A fast test double — no heavy dependencies required.

    Parameters
    ----------
    return_uncertainty : bool
        When *True*, ``predict`` returns a uniform uncertainty column
        equal to the training standard deviation.
    min_train : int
        ``validate_for_fold`` raises ``ValueError`` when ``n_train < min_train``.
    """

    def __init__(
        self,
        return_uncertainty: bool = False,
        min_train: int = 2,
    ) -> None:
        self.return_uncertainty = return_uncertainty
        self.min_train = min_train
        self._mean: Optional[float] = None
        self._std: Optional[float] = None

    def fit(self, X: np.ndarray, y: np.ndarray, coords: np.ndarray) -> "InMemorySpatialModel":
        self._mean = float(np.mean(y))
        self._std = float(np.std(y)) if self.return_uncertainty else None
        return self

    def predict(self, X: np.ndarray, coords: np.ndarray) -> OOFFoldResult:
        if self._mean is None:
            raise RuntimeError("Call fit() before predict().")
        n = len(X)
        preds = np.full(n, self._mean)
        unc = np.full(n, self._std) if self._std is not None else None
        return OOFFoldResult.from_predictions(preds, uncertainty=unc)

    def validate_for_fold(self, n_train: int, n_features: int) -> None:
        if n_train < self.min_train:
            raise ValueError(
                f"n_train={n_train} is smaller than min_train={self.min_train}. "
                "This fold is too small to train on."
            )
