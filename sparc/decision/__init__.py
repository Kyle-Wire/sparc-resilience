"""Decision-support layer.

Combines causal effects (CATE / dose-response / scenario deltas) with
operational constraints (budget, equity weights, robustness penalties)
to rank candidate interventions.

Pure-Python; consumes already-written stage outputs from
``Stage_3_Causal_Validation`` and ``Stage_4_Scenarios``.
"""

from .optimizer import (
    InterventionCandidate,
    OptimizerResult,
    rank_interventions,
    score_equity,
    propose_candidates_from_scenarios,
)
from .equity import (
    EquityScore,
    combine_equity_layers,
    disparity_index,
)
from .uncertainty import (
    UncertaintyResult,
    monte_carlo_decision,
)

__all__ = [
    "InterventionCandidate",
    "OptimizerResult",
    "rank_interventions",
    "score_equity",
    "propose_candidates_from_scenarios",
    "EquityScore",
    "combine_equity_layers",
    "disparity_index",
    "UncertaintyResult",
    "monte_carlo_decision",
]
