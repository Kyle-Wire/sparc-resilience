"""Versioned decision protocol for SPARC pipeline stages.

Every pipeline stage emits a :class:`StageDecision` that records what happened,
why it happened, and what (if anything) the orchestrator should do next. These
decisions are persisted as struct artifacts in ``artifacts.db`` and form the
machine-readable reasoning trace that the desktop UI and downstream stages
consume.

See ``docs/prd/prd-reasoning-engine-spine.md`` for the full design rationale.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# Current schema version. Bump whenever a backwards-incompatible change is made
# to any of the models below. Older readers can dispatch on ``schema_version``.
STAGE_DECISION_SCHEMA_VERSION = 1


class MetricEvaluation(BaseModel):
    """A single scalar metric scored at a stage boundary."""

    name: str
    value: float
    threshold: Optional[float] = None
    passed: Optional[bool] = None
    direction: Literal["higher_is_better", "lower_is_better"]


class DecisionRecord(BaseModel):
    """A discrete reasoning step taken by a stage or the orchestrator.

    ``rule`` names the policy that fired (e.g. ``"improvement_gate"``);
    ``evidence`` carries the inputs that policy used; ``action`` is the
    outcome.
    """

    rule: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    action: Literal["accept", "reject", "fallback", "retry", "escalate"]


class NextAction(BaseModel):
    """A directive for the orchestrator to execute after this stage returns."""

    kind: Literal["none", "retry_stage", "rerun_stage", "escalate"] = "none"
    stage: Optional[int] = None
    reason: str = ""
    config_overrides: dict[str, Any] = Field(default_factory=dict)


class ProvenanceInfo(BaseModel):
    """Where the decisions in this stage came from.

    ``source`` is the high-level provenance label (``"correlogram"``,
    ``"surrogate_search"``, ``"user_override"``, ``"legacy_dispatch"``).
    ``search_log`` carries the BO trace when applicable; ``baseline_comparison``
    carries the improvement-gate evidence.
    """

    source: str
    search_log: Optional[dict[str, Any]] = None
    baseline_comparison: Optional[dict[str, Any]] = None


class StageDecision(BaseModel):
    """The full decision payload emitted by a pipeline stage."""

    schema_version: int = STAGE_DECISION_SCHEMA_VERSION
    stage: int
    status: Literal["passed", "failed", "needs_revision"]
    metrics: dict[str, MetricEvaluation] = Field(default_factory=dict)
    decisions: list[DecisionRecord] = Field(default_factory=list)
    next_actions: list[NextAction] = Field(default_factory=list)
    provenance: ProvenanceInfo

    # ------------------------------------------------------------------
    # Artifact-store round trip helpers
    # ------------------------------------------------------------------

    def to_artifact_struct(self) -> dict[str, Any]:
        """Return a JSON-serializable dict suitable for ``write_struct``."""
        return self.model_dump(mode="json")

    @classmethod
    def from_artifact_struct(cls, payload: dict[str, Any]) -> "StageDecision":
        """Re-hydrate a :class:`StageDecision` from a struct artifact payload."""
        return cls.model_validate(payload)


__all__ = [
    "STAGE_DECISION_SCHEMA_VERSION",
    "MetricEvaluation",
    "DecisionRecord",
    "NextAction",
    "ProvenanceInfo",
    "StageDecision",
]
