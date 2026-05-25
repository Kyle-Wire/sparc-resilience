"""
sparc/run/spatial_cv_stage.py — Typed stage interface for spatial cross-validation.

Provides a value-object input (CVSpec) and a stage class (SpatialCVStage)
that delegates to the existing SpatialCVEngine without exposing RunContext
internals to callers.

Usage::

    from sparc.run.spatial_cv_stage import CVSpec, SpatialCVStage

    spec = CVSpec(
        config=project_config.raw,
        data_path="data/brown4.csv",
        output_dir="output",
        fast=False,
    )
    result = SpatialCVStage.run(spec)
    print(result.rmse)

The :class:`CVResult` type is re-exported here for convenience.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from sparc.run.cv_engine import CVResult


@dataclass
class CVSpec:
    """Typed input specification for :class:`SpatialCVStage`.

    Parameters
    ----------
    config:
        Raw project configuration dict (same shape as ``load_config()`` returns).
    data_path:
        Path to the input CSV data file.
    output_dir:
        Directory for stage outputs.
    fast:
        When ``True``, use reduced folds / epochs for quick iteration.
    """

    config: Dict[str, Any]
    data_path: str
    output_dir: str
    fast: bool = False


class SpatialCVStage:
    """Stage adapter for the SPARC spatial cross-validation sub-system.

    Uses :class:`~sparc.run.cv_engine.SpatialCVEngine` internally.
    Callers provide a :class:`CVSpec`; the stage wires the ``RunContext``
    internally and returns a :class:`~sparc.run.cv_engine.CVResult`.
    """

    @staticmethod
    def run(spec: CVSpec) -> CVResult:
        """Execute spatial CV for *spec*.

        Parameters
        ----------
        spec:
            Typed input specification.

        Returns
        -------
        CVResult
            OOF predictions, fold metrics, and selected features.
        """
        # Build a minimal RunContext from the spec so that SpatialCVEngine
        # does not need to reach out to the file system or environment.
        ctx = _SpecContext(spec)
        from sparc.run.cv_engine import SpatialCVEngine
        engine = SpatialCVEngine(ctx, fast=spec.fast)
        return engine.run()


# ---------------------------------------------------------------------------
# Internal — minimal context adapter
# ---------------------------------------------------------------------------

class _SpecContext:
    """Adapts a CVSpec to the RunContext duck-type expected by SpatialCVEngine."""

    def __init__(self, spec: CVSpec) -> None:
        self._spec = spec
        self.config: Dict[str, Any] = spec.config
        self.registry = None  # no registry when running from a bare spec

    @property
    def fast_mode(self) -> bool:
        return self._spec.fast

    @property
    def output_dir(self) -> Path:
        return Path(self._spec.output_dir)

    @property
    def data_path(self) -> Path:
        return Path(self._spec.data_path)


__all__ = ["CVSpec", "SpatialCVStage", "CVResult"]
