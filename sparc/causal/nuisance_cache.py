"""NuisanceCache — typed loader for Stage 2 OOF outcome predictions.

Replaces the inline try/except block in ``CausalValidator.fit()`` (lines
408–431) with a named module that:

* Raises :class:`NuisanceUnavailable` (not bare ``Exception``) for every
  failure mode, so callers can distinguish "genuinely not available" from
  programming errors and choose to degrade gracefully with a logged INFO
  message rather than a swallowed debug log.
* Records *why* it was unavailable in the exception message — operators
  no longer need to hunt through ``--log-level=DEBUG`` output.
* Stores the source identifier in :class:`OOFNuisance` so downstream code
  (and the Coefficient metadata) can attest which path was taken.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


class NuisanceUnavailable(Exception):
    """Raised when Stage 2 OOF nuisance predictions cannot be loaded.

    Distinct from ``Exception`` so callers can catch it specifically and
    fall back gracefully without silencing unrelated bugs.
    """


@dataclass
class OOFNuisance:
    """Stage 2 out-of-fold outcome predictions ready for DML.

    Attributes
    ----------
    predictions : np.ndarray of shape ``(N,)``
        OOF predictions aligned to the training data rows.
    source : str
        Human-readable description of where these predictions came from,
        e.g. ``"stage2:oof_predictions"`` or ``"stage2:y_pred_oof"``.
    metadata : dict
        Any supplementary information (e.g. R² of the Stage 2 ensemble).
    """

    predictions: np.ndarray
    source: str
    metadata: dict = field(default_factory=dict)


class NuisanceCache:
    """Loads OOF nuisance predictions from an :class:`ArtifactStore`.

    Decouples the "try to find Stage 2 OOF predictions" concern from
    ``EdgeFitter`` / ``DMLEdgeEstimator`` so each can be tested in
    isolation with a stub store.

    Parameters
    ----------
    store : ArtifactStore-like or None
        Object with ``has(stage, artifact) -> bool`` and
        ``read_any(stage, artifact) -> pd.DataFrame | None``.
        Pass ``None`` to model a missing registry — raises
        :class:`NuisanceUnavailable` immediately.
    """

    def __init__(self, store: Any) -> None:
        self._store = store

    # ------------------------------------------------------------------

    def load(
        self,
        stage: str,
        artifact: str,
        expected_len: int,
    ) -> OOFNuisance:
        """Return OOF predictions from *store*.

        Parameters
        ----------
        stage : str
            Stage identifier, e.g. ``"2"``.
        artifact : str
            Artifact name, e.g. ``"oof_predictions"``.
        expected_len : int
            Number of rows the caller expects.  A mismatch raises
            :class:`NuisanceUnavailable` with the actual and expected
            lengths in the message so operators can diagnose immediately.

        Raises
        ------
        NuisanceUnavailable
            For every failure mode: None store, absent artifact, wrong
            shape, or missing numeric columns.
        """
        if self._store is None:
            raise NuisanceUnavailable(
                "NuisanceCache: store is None — "
                "no active ArtifactStore (registry not initialised?)."
            )

        if not self._store.has(stage, artifact):
            raise NuisanceUnavailable(
                f"NuisanceCache: artifact '{stage}/{artifact}' not found in store."
            )

        df = self._store.read_any(stage, artifact)

        if df is None or not hasattr(df, "columns"):
            raise NuisanceUnavailable(
                f"NuisanceCache: '{stage}/{artifact}' returned a non-DataFrame payload."
            )

        # Resolve prediction column
        if "y_pred_oof" in df.columns:
            col = "y_pred_oof"
        else:
            num_cols = [
                c for c in df.columns
                if np.issubdtype(df[c].dtype, np.number)
            ]
            if not num_cols:
                raise NuisanceUnavailable(
                    f"NuisanceCache: '{stage}/{artifact}' contains no numeric columns."
                )
            col = num_cols[0]

        preds = df[col].values.astype(float)

        if len(preds) != expected_len:
            raise NuisanceUnavailable(
                f"NuisanceCache: '{stage}/{artifact}' has {len(preds)} rows "
                f"but data has {expected_len} rows — cannot use as nuisance."
            )

        source = f"stage{stage}:{artifact}:{col}"
        return OOFNuisance(predictions=preds, source=source)
