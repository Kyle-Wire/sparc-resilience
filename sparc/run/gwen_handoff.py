"""
VariableSelectionHandoff and GWENApprovalGate — Stage 1 → Stage 2 seam.

Problem solved
--------------
The GWEN approval gate was previously scattered across three independent
conditions:
  - ``gwen_approved.txt`` sentinel file in the run directory
  - ``pipeline_status.json`` in the artifact store
  - ``_stage_done('.gwen_complete')`` call in ``gwen_variable_selection.py``

None of these encode *which* feature set was approved, so Stage 2 always
trained on the original predictor columns regardless.

This module provides a single, typed seam::

    gate = GWENApprovalGate()
    gate.record_gwen_result(gwen_selected, original_predictors, threshold)

    # Option A — accept GWEN's suggestion
    handoff = gate.approve_gwen()

    # Option B — user keeps their own predictor set despite GWEN results
    handoff = gate.approve_original()

    handoff.active_features  # the list Stage 2 should train on
    handoff.decision          # SelectionDecision.GWEN_SELECTED | ORIGINAL_PREDICTOR_SET
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class SelectionDecision(str, Enum):
    """The outcome of the variable-selection approval step."""

    PENDING = "pending"
    GWEN_SELECTED = "gwen_selected"
    ORIGINAL_PREDICTOR_SET = "original_predictor_set"


@dataclass
class VariableSelectionHandoff:
    """Typed artifact representing the agreed-upon predictor set for Stage 2.

    Fields
    ------
    decision:
        Whether GWEN's recommendation or the original predictor set was chosen.
    active_features:
        The feature list Stage 2 should train on.
    gwen_selected_features:
        The features GWEN recommended, regardless of which path was chosen.
        Retained for audit and reporting.
    original_predictors:
        The predictor set from ``project.yml`` before GWEN screening.
    selection_threshold:
        The frequency threshold used to determine GWEN selections.
    selection_rate:
        Fraction of original features that GWEN retained
        (``len(gwen_selected) / len(original)``).
    """

    decision: SelectionDecision
    active_features: List[str]
    gwen_selected_features: List[str]
    original_predictors: List[str]
    selection_threshold: float
    selection_rate: float

    def to_dict(self) -> dict:
        """Return a JSON-safe representation for artifact persistence."""
        return {
            "decision": self.decision.value,
            "active_features": list(self.active_features),
            "gwen_selected_features": list(self.gwen_selected_features),
            "original_predictors": list(self.original_predictors),
            "selection_threshold": self.selection_threshold,
            "selection_rate": self.selection_rate,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VariableSelectionHandoff":
        return cls(
            decision=SelectionDecision(data["decision"]),
            active_features=list(data["active_features"]),
            gwen_selected_features=list(data["gwen_selected_features"]),
            original_predictors=list(data["original_predictors"]),
            selection_threshold=float(data["selection_threshold"]),
            selection_rate=float(data["selection_rate"]),
        )


class GWENApprovalGate:
    """Manage the user approval decision for GWEN variable selection.

    Usage
    -----
    ::

        gate = GWENApprovalGate(store=store)
        gate.record_gwen_result(selected, original, threshold)

        # --- approval prompt to user ---

        handoff = gate.approve_gwen()          # use GWEN list
        # or
        handoff = gate.approve_original()      # keep original despite GWEN

        stage2_features = gate.get_handoff().active_features
    """

    def __init__(self, store=None) -> None:
        self._store = store
        self._gwen_selected: Optional[List[str]] = None
        self._original_predictors: Optional[List[str]] = None
        self._selection_threshold: Optional[float] = None
        self._handoff: Optional[VariableSelectionHandoff] = None

    # ------------------------------------------------------------------
    # Record phase
    # ------------------------------------------------------------------

    def record_gwen_result(
        self,
        gwen_selected_features: List[str],
        original_predictors: List[str],
        selection_threshold: float,
    ) -> None:
        """Store GWEN's output prior to user approval.

        This does not constitute a decision — the gate remains PENDING
        until ``approve_gwen()`` or ``approve_original()`` is called.
        """
        self._gwen_selected = list(gwen_selected_features)
        self._original_predictors = list(original_predictors)
        self._selection_threshold = float(selection_threshold)
        self._handoff = None  # reset any prior decision

    # ------------------------------------------------------------------
    # Approval phase
    # ------------------------------------------------------------------

    def approve_gwen(self) -> VariableSelectionHandoff:
        """Approve GWEN's variable selection — Stage 2 trains on GWEN features."""
        self._require_result("approve_gwen")
        rate = self._selection_rate()
        self._handoff = VariableSelectionHandoff(
            decision=SelectionDecision.GWEN_SELECTED,
            active_features=list(self._gwen_selected),
            gwen_selected_features=list(self._gwen_selected),
            original_predictors=list(self._original_predictors),
            selection_threshold=self._selection_threshold,
            selection_rate=rate,
        )
        self._persist()
        return self._handoff

    def approve_original(self) -> VariableSelectionHandoff:
        """Override GWEN — Stage 2 trains on the original predictor set.

        This is the key user-override path: even if GWEN suggests a reduced
        feature list, the user can choose to keep their full predictor set.
        GWEN's suggestion is preserved in ``gwen_selected_features`` for audit.
        """
        self._require_result("approve_original")
        rate = self._selection_rate()
        self._handoff = VariableSelectionHandoff(
            decision=SelectionDecision.ORIGINAL_PREDICTOR_SET,
            active_features=list(self._original_predictors),
            gwen_selected_features=list(self._gwen_selected),
            original_predictors=list(self._original_predictors),
            selection_threshold=self._selection_threshold,
            selection_rate=rate,
        )
        self._persist()
        return self._handoff

    # ------------------------------------------------------------------
    # Query phase
    # ------------------------------------------------------------------

    def is_decided(self) -> bool:
        """Return True once an approval decision has been made."""
        return self._handoff is not None

    def get_handoff(self) -> VariableSelectionHandoff:
        """Return the decided handoff.

        Raises
        ------
        ValueError
            If called before ``approve_gwen()`` or ``approve_original()``.
        """
        if self._handoff is None:
            raise ValueError(
                "GWENApprovalGate.get_handoff() called before a decision was made. "
                "Call approve_gwen() or approve_original() first."
            )
        return self._handoff

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_result(self, method: str) -> None:
        if self._gwen_selected is None or self._original_predictors is None:
            raise ValueError(
                f"GWENApprovalGate.{method}() called before record_gwen_result()."
            )

    def _selection_rate(self) -> float:
        n_original = len(self._original_predictors)
        if n_original == 0:
            return 0.0
        return len(self._gwen_selected) / n_original

    def _persist(self) -> None:
        """Write the handoff to the artifact store when one is available."""
        if self._store is None or self._handoff is None:
            return
        try:
            self._store.write_struct(
                "1",
                "variable_selection_handoff",
                self._handoff.to_dict(),
                producer="GWENApprovalGate",
            )
        except Exception:
            pass  # Persistence failure must not break the approval gate
