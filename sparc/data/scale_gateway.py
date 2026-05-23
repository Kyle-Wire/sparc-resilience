"""ScaleGateway — enforces the 'no z-scores to UI layer' contract.

All user-facing numeric outputs from the SPARC pipeline must pass through
this module before reaching the server layer.  The rule is documented in
``sparc/data/units.py`` but was previously unenforced: callers manually
called ``z_to_original()`` on some routes and forgot on others.

This module provides **two** entry points:

``to_original_from_meta(values, meta)``
    Standalone function — useful when the caller already holds the scaling
    metadata dict (e.g., loaded from ``v2_neural_meta_info``).

``ScaleGateway(store).to_original(values, stage, artifact_id)``
    Store-backed class — used in API routes.  Loads ``y_mean/y_std`` from
    the artifact store automatically; callers only pass the values and the
    artifact key.

Seam contract
-------------
* Input:  values in **z-score** space
* Output: values in **original data units**
* No caller should call ``z_to_original`` directly in server routes;
  go through this gateway.
"""
from __future__ import annotations

import numpy as np

from sparc.data.units import z_to_original


def to_original_from_meta(
    values: "np.ndarray | float",
    meta: dict,
) -> "np.ndarray | float":
    """Convert z-scored *values* to original units using *meta*.

    Parameters
    ----------
    values:
        Scalar or array of z-scored values.
    meta:
        Dict containing at minimum ``"y_mean"`` and ``"y_std"`` keys.
        Typically the ``v2_neural_meta_info`` struct stored in the
        artifact registry.

    Returns
    -------
    Values in the original data scale.

    Raises
    ------
    KeyError
        If ``meta`` is missing ``"y_mean"`` or ``"y_std"``.
    """
    y_mean = meta["y_mean"]  # KeyError on missing key — intentional
    y_std = meta["y_std"]
    return z_to_original(values, y_mean, y_std)


class ScaleGateway:
    """Store-backed gateway that loads scaling metadata on demand.

    Parameters
    ----------
    store:
        Any object with a ``read_struct(stage, artifact_id) -> dict``
        method — typically an ``ArtifactStore`` instance.

    Usage
    -----
    ::

        gw = ScaleGateway(get_active_store())
        original_values = gw.to_original(z_predictions, stage="2")
    """

    _DEFAULT_ARTIFACT_ID = "v2_neural_meta_info"

    def __init__(self, store) -> None:
        self._store = store

    def to_original(
        self,
        values: "np.ndarray | float",
        stage: str,
        artifact_id: str = _DEFAULT_ARTIFACT_ID,
    ) -> "np.ndarray | float":
        """Load scaling params from the store and back-transform *values*.

        Parameters
        ----------
        values:
            Scalar or array of z-scored values.
        stage:
            Stage key (e.g. ``"2"``).
        artifact_id:
            Artifact id for the metadata struct.  Defaults to
            ``"v2_neural_meta_info"`` (Stage 2 neural model metadata).

        Raises
        ------
        RuntimeError
            Wraps any store read failure with a clear attribution message.
        """
        try:
            meta = self._store.read_struct(stage, artifact_id)
        except Exception as exc:
            raise RuntimeError(
                f"ScaleGateway: could not read scaling metadata "
                f"({stage!r}, {artifact_id!r}) — {exc}"
            ) from exc
        return to_original_from_meta(values, meta)
