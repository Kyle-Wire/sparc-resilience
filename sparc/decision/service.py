"""Unified decision service facade.

``DecisionService`` wires together the decision sub-modules
(optimizer, uncertainty, equity, causal_bandit, policy_learning) behind
a single, stable interface so callers — server routes, tests, notebooks —
do not need to know the individual module signatures.

Feature flags let downstream code opt in to experimental sub-modules
(``causal_bandit``, ``policy_learning``) without conditional imports at
the call site.
"""
from __future__ import annotations

from typing import Any, Sequence


class DecisionService:
    """Facade over the SPARC decision sub-modules.

    Parameters
    ----------
    causal_bandit_enabled:
        When True, ``rank()`` will additionally consult the causal bandit
        module for exploration-aware scoring (not yet wired — reserved for
        future deepening).
    policy_learning_enabled:
        When True, ``rank()`` will consult the policy learning module for
        learned reward estimates (not yet wired — reserved for future
        deepening).
    """

    def __init__(
        self,
        *,
        causal_bandit_enabled: bool = False,
        policy_learning_enabled: bool = False,
    ) -> None:
        self.causal_bandit_enabled = causal_bandit_enabled
        self.policy_learning_enabled = policy_learning_enabled

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def rank(
        self,
        candidates: Sequence[Any],
        *,
        budget: float | None = None,
        robustness_lambda: float = 0.0,
        minimise: bool = False,
    ) -> list[dict]:
        """Rank *candidates* by causal benefit-per-cost.

        Delegates to ``sparc.decision.optimizer.rank_interventions``.

        Returns
        -------
        list[dict]
            Ranked candidate dicts, each containing at minimum ``name``,
            ``score``, and ``risk_adjusted_effect``.
        """
        from sparc.decision.optimizer import rank_interventions

        if not candidates:
            return []

        result = rank_interventions(
            list(candidates),
            budget=budget,
            robustness_lambda=robustness_lambda,
            minimise=minimise,
        )
        return result.ranked

    def assess_uncertainty(
        self,
        candidates: Sequence[Any],
        *,
        budget: float | None = None,
        robustness_lambda: float = 0.0,
        minimise: bool = False,
        n_draws: int = 500,
        seed: int | None = 42,
    ) -> list[Any]:
        """Run Monte Carlo uncertainty analysis over *candidates*.

        Delegates to ``sparc.decision.uncertainty.monte_carlo_decision``.

        Returns
        -------
        list[UncertaintyResult]
            One entry per candidate (same order as input).
        """
        from sparc.decision.uncertainty import monte_carlo_decision

        if not candidates:
            return []

        return monte_carlo_decision(
            list(candidates),
            budget=budget,
            robustness_lambda=robustness_lambda,
            minimise=minimise,
            n_draws=n_draws,
            seed=seed,
        )
