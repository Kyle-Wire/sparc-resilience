"""
ScenarioBundle: tolerant reader for stage-4 scenario artifacts.

Hides mode-variant precedence and missing-data branches from consumers.
Reads exclusively through :class:`ArtifactStore`; never opens the legacy
on-disk ``scenario_results*.gpkg`` files.

Typical use::

    bundle = ScenarioBundle.from_store(store)
    if "results" not in bundle.available:
        return  # no scenarios produced yet
    for sid in bundle.list_scenarios():
        sc = bundle.get_scenario(sid)
        ...

Every field on the bundle is optional; the ``available`` set advertises
what the underlying store actually contained when ``from_store`` ran.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from sparc.scenario.contract import (
    DELTA_SCENARIO_PREFIX,
    PRED_BASELINE_COL,
    PRED_SCENARIO_PREFIX,
    SCENARIO_ATTRIBUTION,
    SCENARIO_LIBRARY,
    SCENARIO_MC_CONSENSUS,
    SCENARIO_MC_CONSENSUS_SUMMARY,
    SCENARIO_MC_UNCERTAINTY,
    SCENARIO_METADATA,
    SCENARIO_RESULTS_VARIANTS,
    SCENARIO_STAGE,
    SCENARIO_SUMMARY_VARIANTS,
    SCENARIO_TRAJECTORY,
    UNCERTAINTY_COLS,
    delta_column,
    parse_scenario_index,
    pred_column,
)

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

    from sparc.registry.store import ArtifactStore


ScenarioId = int


@dataclass
class ScenarioRecord:
    """Per-scenario slice extracted from the active results table."""

    index: ScenarioId
    pred_column: str
    delta_column: Optional[str]


@dataclass
class ScenarioBundle:
    """All scenario artifacts a consumer typically needs.

    Construct via :meth:`from_store`; never instantiate directly.
    """

    results: Optional["pd.DataFrame"] = None
    results_artifact_id: Optional[str] = None
    summary: Optional["pd.DataFrame"] = None
    summary_artifact_id: Optional[str] = None
    attribution: Optional["pd.DataFrame"] = None
    trajectory: Optional["pd.DataFrame"] = None
    mc_uncertainty: Optional["pd.DataFrame"] = None
    mc_consensus: Optional["pd.DataFrame"] = None
    mc_consensus_summary: Optional["pd.DataFrame"] = None
    metadata: Optional[dict[str, Any]] = None
    library: Optional[dict[str, Any]] = None
    available: set[str] = field(default_factory=set)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_store(cls, store: "ArtifactStore") -> "ScenarioBundle":
        """Load scenario artifacts from ``store``, tolerating missing pieces.

        Mode-variant precedence is applied automatically: the highest-priority
        ``scenario_results*`` table that exists in the registry wins.
        """
        bundle = cls()

        # Tables — pick highest-priority variant that exists.
        for variant in SCENARIO_RESULTS_VARIANTS:
            if store.has(SCENARIO_STAGE, variant):
                bundle.results = store.read_table(SCENARIO_STAGE, variant)
                bundle.results_artifact_id = variant
                bundle.available.add("results")
                break

        for variant in SCENARIO_SUMMARY_VARIANTS:
            if store.has(SCENARIO_STAGE, variant):
                bundle.summary = store.read_any(SCENARIO_STAGE, variant)
                bundle.summary_artifact_id = variant
                bundle.available.add("summary")
                break

        # Optional tables.
        for attr, aid, kind in (
            ("attribution", SCENARIO_ATTRIBUTION, "table"),
            ("trajectory", SCENARIO_TRAJECTORY, "table"),
            ("mc_uncertainty", SCENARIO_MC_UNCERTAINTY, "table"),
            ("mc_consensus", SCENARIO_MC_CONSENSUS, "table"),
            ("mc_consensus_summary", SCENARIO_MC_CONSENSUS_SUMMARY, "table"),
        ):
            if store.has(SCENARIO_STAGE, aid):
                setattr(bundle, attr, store.read_any(SCENARIO_STAGE, aid))
                bundle.available.add(attr)

        # Structs.
        for attr, aid in (
            ("metadata", SCENARIO_METADATA),
            ("library", SCENARIO_LIBRARY),
        ):
            if store.has(SCENARIO_STAGE, aid):
                setattr(bundle, attr, store.read_struct(SCENARIO_STAGE, aid))
                bundle.available.add(attr)

        return bundle

    # ------------------------------------------------------------------
    # Per-scenario access
    # ------------------------------------------------------------------

    def list_scenarios(self) -> list[ScenarioId]:
        """Return the sorted set of scenario indices present in ``results``.

        Returns ``[]`` when no results table is loaded.
        """
        if self.results is None:
            return []
        ids: set[ScenarioId] = set()
        for col in self.results.columns:
            idx = parse_scenario_index(str(col))
            if idx is not None:
                ids.add(idx)
        return sorted(ids)

    def get_scenario(self, scenario_id: ScenarioId) -> Optional[ScenarioRecord]:
        """Return per-scenario column references, or ``None`` if missing."""
        if self.results is None:
            return None
        pcol = pred_column(scenario_id)
        if pcol not in self.results.columns:
            return None
        dcol = delta_column(scenario_id)
        return ScenarioRecord(
            index=scenario_id,
            pred_column=pcol,
            delta_column=dcol if dcol in self.results.columns else None,
        )

    def has_uncertainty(self) -> bool:
        """True if any of the optional uncertainty columns are present."""
        if self.results is None:
            return False
        return any(c in self.results.columns for c in UNCERTAINTY_COLS)

    def baseline_column(self) -> Optional[str]:
        """Return ``pred_baseline`` when present on the results table."""
        if self.results is None:
            return None
        return PRED_BASELINE_COL if PRED_BASELINE_COL in self.results.columns else None


__all__ = ["ScenarioBundle", "ScenarioRecord", "ScenarioId"]
