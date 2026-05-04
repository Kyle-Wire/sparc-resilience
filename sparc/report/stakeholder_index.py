"""Stakeholder question → report-section index.

Emits a JSON map answering "where do I find the answer to X?" for each
of the seven Phase-B audiences. Used by the desktop app's "Find" panel
and by the Stage-5 report's table-of-contents page.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# question_id → list of (audience, section_title) pairs that address it
_INDEX: dict[str, list[tuple[str, str]]] = {
    "what_works":              [("planner", "Recommended interventions"),
                                ("council", "Recommended actions"),
                                ("public",  "What this means for you")],
    "how_much_does_it_cost":   [("planner", "Recommended interventions"),
                                ("council", "Budget allocation")],
    "how_confident":           [("technical", "Causal estimates"),
                                ("council", "How confident are we?"),
                                ("scientist", "Posterior diagnostics")],
    "who_benefits":            [("equity",   "Who benefits, and by how much?"),
                                ("council",  "Equity")],
    "spillover_effects":       [("equity",   "Spillover"),
                                ("scientist","Method stack")],
    "is_it_replicable":        [("auditor",  "Provenance"),
                                ("scientist","Provenance")],
    "what_methods_were_used":  [("scientist","Method stack"),
                                ("technical","Causal estimates")],
    "did_we_pass_audit":       [("auditor",  "Wager (2025) gap-completion table"),
                                ("scientist","Wager-2025 audit roll-up")],
    "where_to_target_first":   [("planner",  "Recommended interventions"),
                                ("council",  "Optimal targeted rollout (Wager-EWM)")],
    "what_could_go_wrong":     [("technical","Physics constraints"),
                                ("auditor",  "Refutation summary"),
                                ("scientist","Refutations")],
}


def stakeholder_index() -> dict[str, list[dict[str, str]]]:
    """Return the question→section index keyed for JSON serialisation."""
    return {
        q: [{"audience": a, "section": s} for (a, s) in pairs]
        for q, pairs in _INDEX.items()
    }


def write_stakeholder_index(dest: str | Path) -> Path:
    """Persist the index to *dest* as JSON. Returns the absolute path."""
    p = Path(dest)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(stakeholder_index(), indent=2), encoding="utf-8")
    return p


def lookup(question_id: str) -> list[dict[str, str]]:
    return stakeholder_index().get(question_id, [])
