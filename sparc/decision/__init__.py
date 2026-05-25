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
    combine_equity_layers,
    disparity_index,
)
from .uncertainty import (
    monte_carlo_decision,
)

# CausalBandit is loaded lazily to avoid importing its heavy dependencies
# (scipy, numpy GP code) when the rest of sparc.decision is imported.
def __getattr__(name: str):
    if name == "CausalBandit":
        from .causal_bandit import CausalBandit
        # Cache in the module namespace so subsequent accesses don't call __getattr__
        globals()["CausalBandit"] = CausalBandit
        return CausalBandit
    raise AttributeError(f"module 'sparc.decision' has no attribute {name!r}")


__all__ = [
    # Optimizer — externally used
    "InterventionCandidate",
    "rank_interventions",
    "propose_candidates_from_scenarios",
    # Equity — externally used
    "combine_equity_layers",
    "disparity_index",
    # Uncertainty — externally used
    "monte_carlo_decision",
    # Lazy-loaded
    "CausalBandit",
]
