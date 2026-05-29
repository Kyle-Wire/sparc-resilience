"""Stage output contracts for the SPARC pipeline.

Each class defines the expected schema for one stage's primary output.
Call ``validate()`` at stage-boundary seams to catch schema drift early,
before the next stage fails with an opaque AttributeError or KeyError.

Usage
-----
From the writer (Stage 0)::

    result = analyzer.analyze_variable(...)
    CorrelogramOutputs.validate(result)   # raises ContractViolation on bad output

From the reader (Stage 3, before using Stage 2 OOF data)::

    df = store.read_any("2", "oof_predictions")
    SpatialCVOutputs.validate(df)

From the writer (Stage 3)::

    output = validator.produce_scenario_coefficients(output_dir)
    CausalOutputs.validate(output)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ContractViolation(Exception):
    """Raised when a stage output fails its contract validation."""


# ---------------------------------------------------------------------------
# Typed result dataclasses — returned by the parse() classmethods
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CorrelogramResult:
    """Typed representation of a validated CorrelogramOutputs payload."""
    optimal_block_size: Any
    effective_range: Any
    correlogram_results: Any


@dataclass(frozen=True)
class SpatialCVResult:
    """Typed representation of a validated SpatialCVOutputs payload."""
    oof_df: Any


@dataclass(frozen=True)
class CausalResult:
    """Typed representation of a validated CausalOutputs payload."""
    metadata: dict
    direct_effects: dict
    mediator_propagation: dict
    all_structural_coefficients: dict


# ---------------------------------------------------------------------------
# Stage 0 → Stage 2 seam
# ---------------------------------------------------------------------------

class CorrelogramOutputs:
    """Validates the ``analysis_result`` dict produced by
    :meth:`CorrelogramAnalyzer.analyze_variable`.

    Required top-level keys in the dict:

    * ``optimal_block_size`` — float or None; used by Stage 2 CV config.
    * ``effective_range``    — float or None; bandwidth for GWR/GWRF.
    * ``correlogram_results`` — raw analyzer output containing ``lag_distances``.

    The ``correlogram_results`` sub-dict must itself be a dict with a
    ``lag_distances`` key (list).
    """

    REQUIRED_KEYS: frozenset[str] = frozenset({
        "optimal_block_size",
        "effective_range",
        "correlogram_results",
    })

    @staticmethod
    def validate(d: Any) -> None:
        """Raise :exc:`ContractViolation` if *d* does not satisfy the contract."""
        if not isinstance(d, dict):
            raise ContractViolation(
                f"CorrelogramOutputs expected dict, got {type(d).__name__}"
            )
        missing = CorrelogramOutputs.REQUIRED_KEYS - d.keys()
        if missing:
            raise ContractViolation(
                f"CorrelogramOutputs missing required keys: {sorted(missing)}"
            )
        corr = d.get("correlogram_results")
        if not isinstance(corr, (dict, list)):
            raise ContractViolation(
                "CorrelogramOutputs: 'correlogram_results' must be a dict or list, "
                f"got {type(corr).__name__}"
            )
        if isinstance(corr, dict):
            if "lag_distances" not in corr:
                raise ContractViolation(
                    "CorrelogramOutputs: 'correlogram_results' dict must contain "
                    "'lag_distances'"
                )
            if not isinstance(corr["lag_distances"], (list, tuple)):
                raise ContractViolation(
                    "CorrelogramOutputs: 'correlogram_results.lag_distances' must be a list"
                )

    @staticmethod
    def parse(d: Any) -> CorrelogramResult:
        """Validate *d* and return a typed :class:`CorrelogramResult`.

        Raises :exc:`ContractViolation` on invalid input — identical semantics
        to :meth:`validate` but returns a structured object instead of ``None``.
        """
        CorrelogramOutputs.validate(d)
        return CorrelogramResult(
            optimal_block_size=d["optimal_block_size"],
            effective_range=d["effective_range"],
            correlogram_results=d["correlogram_results"],
        )


# ---------------------------------------------------------------------------
# Stage 2 → Stage 3 seam
# ---------------------------------------------------------------------------

class SpatialCVOutputs:
    """Validates the OOF predictions DataFrame written by Stage 2.

    Stage 3 reads ``("2", "oof_predictions")`` and expects at least one
    numeric column (model predictions).  An empty or non-numeric DataFrame
    means Stage 2 failed silently, which would produce wrong causal estimates.
    """

    @staticmethod
    def validate(df: Any) -> None:
        """Raise :exc:`ContractViolation` if *df* is not a valid OOF DataFrame."""
        try:
            import pandas as pd
            import numpy as np
        except ImportError:
            return  # pandas/numpy not available — skip validation gracefully

        if not isinstance(df, pd.DataFrame):
            raise ContractViolation(
                f"SpatialCVOutputs expected pandas DataFrame, got {type(df).__name__}"
            )
        if df.empty:
            raise ContractViolation(
                "SpatialCVOutputs: OOF predictions DataFrame is empty"
            )
        numeric_cols = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c])
        ]
        if not numeric_cols:
            raise ContractViolation(
                "SpatialCVOutputs: OOF DataFrame has no numeric columns; "
                f"found columns: {list(df.columns)}"
            )

    @staticmethod
    def parse(df: Any) -> SpatialCVResult:
        """Validate *df* and return a typed :class:`SpatialCVResult`.

        Raises :exc:`ContractViolation` on invalid input.
        """
        SpatialCVOutputs.validate(df)
        return SpatialCVResult(oof_df=df)


# ---------------------------------------------------------------------------
# Stage 3 output seam
# ---------------------------------------------------------------------------

class CausalOutputs:
    """Validates ``scenario_coefficients.json`` produced by Stage 3.

    This dict is consumed by the ScenarioSimulator (Stage 4), the server
    ``/results/causal`` endpoint, and the desktop frontend.  All four
    top-level keys are required; each must be a dict.
    """

    REQUIRED_KEYS: frozenset[str] = frozenset({
        "metadata",
        "direct_effects",
        "mediator_propagation",
        "all_structural_coefficients",
    })

    @staticmethod
    def validate(d: Any) -> None:
        """Raise :exc:`ContractViolation` if *d* does not satisfy the contract."""
        if not isinstance(d, dict):
            raise ContractViolation(
                f"CausalOutputs expected dict, got {type(d).__name__}"
            )
        missing = CausalOutputs.REQUIRED_KEYS - d.keys()
        if missing:
            raise ContractViolation(
                f"CausalOutputs missing required keys: {sorted(missing)}"
            )
        for key in ("metadata", "direct_effects", "mediator_propagation",
                    "all_structural_coefficients"):
            val = d.get(key)
            if not isinstance(val, dict):
                raise ContractViolation(
                    f"CausalOutputs: '{key}' must be a dict, "
                    f"got {type(val).__name__}"
                )

    @staticmethod
    def parse(d: Any) -> CausalResult:
        """Validate *d* and return a typed :class:`CausalResult`.

        Raises :exc:`ContractViolation` on invalid input.
        """
        CausalOutputs.validate(d)
        return CausalResult(
            metadata=d["metadata"],
            direct_effects=d["direct_effects"],
            mediator_propagation=d["mediator_propagation"],
            all_structural_coefficients=d["all_structural_coefficients"],
        )
