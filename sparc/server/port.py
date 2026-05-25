"""sparc.server.port — PipelinePort seam.

Defines the single typed interface that HTTP routes cross to reach
pipeline functionality.  The HTTP layer never imports sparc.run,
sparc.data, or sparc.causal directly — everything goes through a
PipelinePort instance.

Implementations
---------------
InMemoryPipelineAdapter  — synchronous, no-I/O adapter used in tests and
                           for rapid response paths that don't need a full
                           pipeline run.
LivePipelineAdapter      — wires to the real sparc.run / sparc.data stack;
                           constructed once at server startup and attached
                           to ServerState.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from sparc.data.validation import validate_dataset, ValidationReport


# ---------------------------------------------------------------------------
# Port (abstract interface)
# ---------------------------------------------------------------------------


class PipelinePort(ABC):
    """The single seam between the HTTP layer and the pipeline internals.

    Routes construct nothing themselves — they call these three methods and
    return the results.  All pipeline coupling lives in concrete adapters.
    """

    @abstractmethod
    def validate(self, df: pd.DataFrame, config: dict) -> ValidationReport:
        """Run pre-flight data validation.

        Parameters
        ----------
        df:
            The dataset to validate.
        config:
            Project config dict (used to extract column names, CRS, etc.).

        Returns
        -------
        ValidationReport
            Structured result with ``passed``, ``n_critical``, and ``items``.
        """

    @abstractmethod
    def run_stage(self, stage_key: str, flags: dict) -> dict:
        """Execute one pipeline stage by its canonical key.

        Parameters
        ----------
        stage_key:
            One of ``"stage_0"``, ``"stage_1"``, ``"stage_2"``,
            ``"stage_3"``, ``"stage_4"``, ``"stage_5"``.
        flags:
            Optional overrides forwarded to the stage runner.

        Returns
        -------
        dict
            At minimum ``{"status": "ok" | "error", "message": str}``.
        """

    @abstractmethod
    def read_artifacts(self, stage: int, *, names: list[str] | None = None) -> list[dict]:
        """Read artifacts for the given stage from the current registry.

        Parameters
        ----------
        stage:
            Stage number (0–5).
        names:
            Optional filter — only return artifacts whose name is in this list.

        Returns
        -------
        list[dict]
            Each entry is a serialisable dict with at least ``name`` and
            ``stage`` keys.
        """


# ---------------------------------------------------------------------------
# InMemoryPipelineAdapter — test / stub implementation
# ---------------------------------------------------------------------------


class InMemoryPipelineAdapter(PipelinePort):
    """No-I/O adapter used in tests and for server startup health-checks.

    All methods return minimal valid responses without touching disk,
    network, or ML models.  Inject via ``ServerState`` or ``pytest`` fixtures.
    """

    def __init__(self) -> None:
        self._artifacts: dict[int, list[dict]] = {}
        self._stage_responses: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # PipelinePort implementation
    # ------------------------------------------------------------------

    def validate(self, df: pd.DataFrame, config: dict) -> ValidationReport:
        """Run validation using the real DataValidator — pure in-process."""
        target_col: str | None = None
        predictor_cols: list[str] | None = None
        coord_cols: list[str] | None = None
        expected_crs: str | None = None

        if config:
            data_cfg = config.get("data", {})
            target_col = data_cfg.get("target_column")
            predictor_cols = list(config.get("predictors", {}).get("base_model", []))
            coord_cols = list(data_cfg.get("coord_columns", []))
            expected_crs = config.get("crs", {}).get("target_projected")

        return validate_dataset(
            df,
            target_col=target_col,
            predictor_cols=predictor_cols or [],
            coord_cols=coord_cols or [],
            expected_crs=expected_crs,
        )

    def run_stage(self, stage_key: str, flags: dict) -> dict:
        """Return a canned success response (no actual execution)."""
        if stage_key in self._stage_responses:
            return self._stage_responses[stage_key]
        return {"status": "ok", "message": f"in-memory stub: {stage_key} skipped"}

    def read_artifacts(self, stage: int, *, names: list[str] | None = None) -> list[dict]:
        """Return pre-seeded artifacts or an empty list."""
        items = self._artifacts.get(stage, [])
        if names is not None:
            items = [a for a in items if a.get("name") in names]
        return list(items)

    # ------------------------------------------------------------------
    # Test helpers
    # ------------------------------------------------------------------

    def seed_artifacts(self, stage: int, artifacts: list[dict]) -> None:
        """Pre-seed artifact responses for test assertions."""
        self._artifacts[stage] = list(artifacts)

    def set_stage_response(self, stage_key: str, response: dict) -> None:
        """Override what run_stage returns for a specific stage key."""
        self._stage_responses[stage_key] = response
