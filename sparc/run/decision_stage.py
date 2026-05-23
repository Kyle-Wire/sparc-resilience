"""Decision stage orchestrator.

Thin pipeline wrapper that:
  1. Applies equity layers to candidates (updates equity_weight in-place copy),
  2. Calls rank_interventions(),
  3. Calls monte_carlo_decision() for uncertainty-aware selection probabilities,
  4. Optionally persists decision_ranked.json to output_dir.

Called from sparc/__main__.py Stage 5 (batch CLI) and from
sparc/server/app.py for interactive use.
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Mapping, Optional, Sequence

from sparc.decision.equity import combine_equity_layers
from sparc.decision.optimizer import InterventionCandidate, rank_interventions
from sparc.decision.uncertainty import monte_carlo_decision


def run_decision_stage(
    candidates: Sequence[InterventionCandidate],
    equity_layers: Mapping[str, Sequence[float]] | Sequence[float] | None = None,
    *,
    budget: Optional[float] = None,
    robustness_lambda: float = 0.0,
    minimise: bool = False,
    n_mc_draws: int = 500,
    output_dir: Optional[str] = None,
) -> dict:
    """Rank interventions with equity weighting and MC uncertainty.

    Parameters
    ----------
    candidates:
        ``InterventionCandidate`` objects from causal/scenario outputs.
    equity_layers:
        One or more equity layers aligned to ``candidates``.
        Passed directly to :func:`combine_equity_layers`.
    budget:
        Optional total cost cap for greedy selection.
    robustness_lambda:
        Uncertainty penalty for risk-adjusted ranking.
    minimise:
        Set True when the target metric should be *reduced*.
    n_mc_draws:
        Number of Monte-Carlo draws for uncertainty estimation.
    output_dir:
        When provided, writes ``decision_ranked.json`` here.

    Returns
    -------
    dict with keys ``ranked`` (list[dict]) and ``uncertainty`` (list[dict]).
    """
    names = [c.name for c in candidates]

    # ── 1. Equity weighting ───────────────────────────────────────────────
    equity_scores = None
    equity_map: dict[str, float] = {}
    if equity_layers is not None and len(candidates) > 0:
        equity_scores = combine_equity_layers(equity_layers, names)
        equity_map = {es.candidate: es.weight for es in equity_scores}

    # Make fresh copies so we don't mutate the caller's objects.
    weighted: list[InterventionCandidate] = []
    for c in candidates:
        c_copy = dataclasses.replace(c, equity_weight=equity_map.get(c.name, 1.0))
        weighted.append(c_copy)

    # ── 2. Rank ───────────────────────────────────────────────────────────
    opt_result = rank_interventions(
        weighted,
        budget=budget,
        robustness_lambda=robustness_lambda,
        minimise=minimise,
    )

    # ── 3. MC uncertainty ─────────────────────────────────────────────────
    unc_results = monte_carlo_decision(
        weighted,
        budget=budget,
        robustness_lambda=robustness_lambda,
        minimise=minimise,
        n_draws=n_mc_draws,
    )

    result: dict = {
        "ranked": opt_result.ranked,
        "settings": opt_result.settings,
        "equity_summary": opt_result.equity_summary,
        "uncertainty": [dataclasses.asdict(u) for u in unc_results],
    }

    # ── 4. Persist artifact ───────────────────────────────────────────────
    if output_dir is not None:
        artifact_path = os.path.join(output_dir, "decision_ranked.json")
        with open(artifact_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, default=str)
        print(f"  [Decision] Ranked {len(weighted)} interventions → {artifact_path}")

    return result
