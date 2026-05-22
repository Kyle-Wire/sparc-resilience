"""
MaternFitter — resilient Matérn fit with typed fallback reporting.

Wraps :func:`~sparc.run.correlogram_matern_fit.fit_matern_bayes` and
:func:`~sparc.run.correlogram_matern_fit.fit_matern_lsq` behind a single
``.fit()`` seam.  When the Bayesian path raises (e.g. GPU OOM, convergence
failure, numerical NaN) the result falls back to LSQ and the reason is
recorded in :attr:`MaternFitResult.fallback_reason` rather than being
silently discarded.

Dependency injection via ``bayes_fn`` / ``lsq_fn`` constructor params lets
tests simulate failure modes without real GPU hardware.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Callable, Optional

import numpy as np

from sparc.run.correlogram_matern_fit import (
    MaternFitResult,
    fit_matern_bayes,
    fit_matern_lsq,
)

logger = logging.getLogger(__name__)


class MaternFitter:
    """Resilient Matérn fitter with a typed seam and injectable dependencies.

    Parameters
    ----------
    bayes_fn:
        Callable with the same signature as :func:`fit_matern_bayes`.
        Defaults to the real NUTS-backed fitter.  Pass a raising callable in
        tests to exercise the fallback path without GPU hardware.
    lsq_fn:
        Callable with the same signature as :func:`fit_matern_lsq`.
        Defaults to the real least-squares fitter.
    """

    def __init__(
        self,
        *,
        bayes_fn: Optional[Callable] = None,
        lsq_fn: Optional[Callable] = None,
    ) -> None:
        self._bayes_fn = bayes_fn if bayes_fn is not None else fit_matern_bayes
        self._lsq_fn = lsq_fn if lsq_fn is not None else fit_matern_lsq

    def fit(
        self,
        lags: np.ndarray,
        morans: np.ndarray,
        *,
        prefer_bayes: bool = True,
        seed: int = 42,
    ) -> MaternFitResult:
        """Fit a Matérn correlation model.

        Parameters
        ----------
        lags:
            (L,) lag distances in metres (must be > 0 for non-trivial bins).
        morans:
            (L,) empirical Moran's I values at each lag.
        prefer_bayes:
            When ``True`` (default) attempt the Bayesian NUTS path first.
            Falls back to LSQ on any exception, recording the reason in
            :attr:`MaternFitResult.fallback_reason`.
            When ``False`` go directly to LSQ.
        seed:
            RNG seed forwarded to ``bayes_fn``.

        Returns
        -------
        MaternFitResult
            ``fallback_reason`` is ``None`` when Bayesian succeeded or when
            ``prefer_bayes=False``; a string description otherwise.
        """
        if not prefer_bayes:
            return self._lsq_fn(lags, morans)

        try:
            return self._bayes_fn(lags, morans, seed=seed)
        except Exception as exc:
            logger.warning("Bayesian Matérn fit failed (%s); falling back to LSQ", exc)
            result = self._lsq_fn(lags, morans)
            return dataclasses.replace(result, fallback_reason=str(exc))
