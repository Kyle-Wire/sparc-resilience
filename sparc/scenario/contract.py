"""
Scenario artifact contract.

Single source of truth for the artifact IDs, column conventions, and
mode-variant precedence used by the stage-4 scenario simulator and every
downstream consumer (server endpoints, report figures, desktop UI).

The simulator may be rewritten / re-organized over time; consumers depend
on the names and conventions defined here, never on disk layout.

Conventions
-----------
- Stage id for all scenario artifacts: ``"4"`` (string).
- Per-row prediction columns on the ``scenario_results*`` tables follow
  the pattern ``pred_baseline``, ``pred_Scenario{N}`` (1-indexed).
- Per-row delta columns follow ``delta_Scenario{N}`` and are computed
  by the server when not present in the table.
- Optional uncertainty columns: ``pred_mean``, ``pred_std``, ``pred_p05``,
  ``pred_p95`` may appear on any results table.
- Mode-variant precedence (highest first):
  ``scenario_results_hybrid`` > ``scenario_results_reprediction``
  > ``scenario_results_dag`` > ``scenario_results``.

Consumers must read exclusively through :class:`ArtifactStore`.  The
legacy on-disk ``scenario_results*.gpkg`` files are slated for removal
and must be ignored.
"""

from __future__ import annotations

import re
from typing import Final


SCENARIO_STAGE: Final[str] = "4"

# Table artifact ids, ordered by precedence (most specific first).
SCENARIO_RESULTS_VARIANTS: Final[tuple[str, ...]] = (
    "scenario_results_hybrid",
    "scenario_results_reprediction",
    "scenario_results_dag",
    "scenario_results",
)

SCENARIO_SUMMARY_VARIANTS: Final[tuple[str, ...]] = (
    "scenario_summary_hybrid",
    "scenario_summary_reprediction",
    "scenario_summary_dag",
    "scenario_summary",
)

# Optional new tables (may or may not exist depending on simulator capabilities).
SCENARIO_ATTRIBUTION: Final[str] = "scenario_attribution"
SCENARIO_TRAJECTORY: Final[str] = "scenario_trajectory"
SCENARIO_MC_UNCERTAINTY: Final[str] = "scenario_mc_uncertainty"
SCENARIO_MC_CONSENSUS: Final[str] = "scenario_mc_consensus"
SCENARIO_MC_CONSENSUS_SUMMARY: Final[str] = "scenario_mc_consensus_summary"

# Struct artifacts.
SCENARIO_METADATA: Final[str] = "scenario_metadata"
SCENARIO_LIBRARY: Final[str] = "scenario_library"

# Column conventions.
PRED_BASELINE_COL: Final[str] = "pred_baseline"
PRED_SCENARIO_PREFIX: Final[str] = "pred_Scenario"
DELTA_SCENARIO_PREFIX: Final[str] = "delta_Scenario"
UNCERTAINTY_COLS: Final[tuple[str, ...]] = (
    "pred_mean",
    "pred_std",
    "pred_p05",
    "pred_p95",
)

# Compiled regex for parsing scenario index from a column name.
_SCENARIO_COL_RE: Final[re.Pattern[str]] = re.compile(
    rf"^{re.escape(PRED_SCENARIO_PREFIX)}(\d+)$"
)


def parse_scenario_index(column: str) -> int | None:
    """Return the 1-based scenario index encoded in a ``pred_Scenario{N}`` column.

    Returns ``None`` for columns that do not follow the convention.
    """
    m = _SCENARIO_COL_RE.match(column)
    if m is None:
        return None
    return int(m.group(1))


def pred_column(scenario_index: int) -> str:
    """Return the prediction column name for a 1-based scenario index."""
    return f"{PRED_SCENARIO_PREFIX}{scenario_index}"


def delta_column(scenario_index: int) -> str:
    """Return the delta column name for a 1-based scenario index."""
    return f"{DELTA_SCENARIO_PREFIX}{scenario_index}"


__all__ = [
    "SCENARIO_STAGE",
    "SCENARIO_RESULTS_VARIANTS",
    "SCENARIO_SUMMARY_VARIANTS",
    "SCENARIO_ATTRIBUTION",
    "SCENARIO_TRAJECTORY",
    "SCENARIO_MC_UNCERTAINTY",
    "SCENARIO_MC_CONSENSUS",
    "SCENARIO_MC_CONSENSUS_SUMMARY",
    "SCENARIO_METADATA",
    "SCENARIO_LIBRARY",
    "PRED_BASELINE_COL",
    "PRED_SCENARIO_PREFIX",
    "DELTA_SCENARIO_PREFIX",
    "UNCERTAINTY_COLS",
    "parse_scenario_index",
    "pred_column",
    "delta_column",
]
