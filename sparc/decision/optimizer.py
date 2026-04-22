"""Intervention optimizer.

Ranks candidate interventions by expected causal benefit per unit cost,
with optional equity weighting and robustness (uncertainty) penalty.

Inputs are intentionally simple Python dicts so this module can be exercised
from ``server/app.py`` without pulling in the full pipeline state.

Mathematical model
------------------
For each candidate :math:`c` with mean causal effect :math:`\\mu_c`,
posterior std :math:`\\sigma_c`, cost :math:`k_c`, equity weight :math:`w_c`:

.. math::
    \\text{score}_c = w_c \\cdot \\frac{\\mu_c - \\lambda \\sigma_c}{k_c + \\epsilon}

where :math:`\\lambda` is the robustness penalty (defaults to 0 = risk-neutral)
and :math:`\\epsilon` is a small constant to avoid divide-by-zero.

The score is causal (uses :math:`\\mu` from the dose-response or scenario
posterior, not a correlation) and uncertainty-aware
(:math:`\\lambda > 0` penalises wide credible intervals).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional


_EPS = 1e-9


@dataclass
class InterventionCandidate:
    """A candidate intervention to be ranked."""

    name: str
    treatment: str
    magnitude: float
    mean_effect: float          # μ — expected change in target (causal)
    effect_std: float = 0.0     # σ — posterior std (0 ⇒ point estimate)
    cost: float = 1.0           # arbitrary units; only the ratio matters
    equity_weight: float = 1.0  # higher ⇒ population/area is prioritised
    notes: str = ""


@dataclass
class OptimizerResult:
    """Ranked optimizer output."""

    ranked: list[dict] = field(default_factory=list)
    settings: dict = field(default_factory=dict)
    equity_summary: dict = field(default_factory=dict)


def score_equity(equity_values: list[float], reference: Optional[float] = None) -> dict:
    """Return basic equity statistics for a vector of weights/exposures.

    ``reference`` lets callers pass an explicit population mean; otherwise
    the simple mean is used.  The output is a small dict suitable for
    serialising over the API.
    """
    if not equity_values:
        return {"n": 0, "mean": None, "min": None, "max": None, "gini": None}
    n = len(equity_values)
    s = float(sum(equity_values))
    mean = s / n if n else 0.0
    if reference is not None:
        mean = float(reference)
    sorted_v = sorted(float(v) for v in equity_values)
    # Gini coefficient (numerically stable for non-negative inputs).
    cum = 0.0
    weighted = 0.0
    for i, v in enumerate(sorted_v, start=1):
        cum += v
        weighted += i * v
    gini = (
        (2.0 * weighted) / (n * cum) - (n + 1.0) / n
        if cum > 0
        else 0.0
    )
    return {
        "n": n,
        "mean": mean,
        "min": sorted_v[0],
        "max": sorted_v[-1],
        "gini": gini,
    }


def rank_interventions(
    candidates: list[InterventionCandidate],
    *,
    budget: Optional[float] = None,
    robustness_lambda: float = 0.0,
    minimise: bool = False,
) -> OptimizerResult:
    """Rank candidates by causal benefit per cost (with optional uncertainty penalty).

    Parameters
    ----------
    candidates
        Candidate interventions.
    budget
        Optional total cost cap.  When provided the result also reports
        the greedy "selected" subset whose cumulative cost ≤ ``budget``.
    robustness_lambda
        Penalty applied to posterior std; 0 = risk-neutral, larger
        values prefer interventions with tighter posteriors.
    minimise
        If True the target is something we want to *reduce* (e.g.
        temperature, noise) and a more-negative ``mean_effect`` is better.
    """
    sign = -1.0 if minimise else 1.0
    scored: list[dict] = []
    for c in candidates:
        risk_adj = c.mean_effect - robustness_lambda * c.effect_std
        score = c.equity_weight * (sign * risk_adj) / (c.cost + _EPS)
        scored.append(
            {
                **asdict(c),
                "risk_adjusted_effect": risk_adj,
                "score": score,
            }
        )
    scored.sort(key=lambda r: r["score"], reverse=True)

    selected: list[str] = []
    if budget is not None and budget > 0:
        running = 0.0
        for row in scored:
            if running + row["cost"] <= budget:
                selected.append(row["name"])
                running += row["cost"]
        for row in scored:
            row["selected"] = row["name"] in selected

    equity_summary = score_equity([c.equity_weight for c in candidates])

    return OptimizerResult(
        ranked=scored,
        settings={
            "budget": budget,
            "robustness_lambda": robustness_lambda,
            "minimise": minimise,
            "n_candidates": len(candidates),
            "n_selected": len(selected),
        },
        equity_summary=equity_summary,
    )


def propose_candidates_from_scenarios(
    scenarios: list[dict],
    nuts_posteriors: Optional[dict] = None,
) -> list[InterventionCandidate]:
    """Convert a stage-4 scenarios dict into ``InterventionCandidate``s.

    ``scenarios`` is expected to be the list shape produced by
    ``/results/scenarios/detail`` (each row has ``name``, ``delta``,
    optional ``treatment`` / ``magnitude`` / ``cost``).  When NUTS
    posteriors are available the per-treatment posterior std is used
    for the uncertainty term; otherwise we fall back to 25 % CV.
    """
    out: list[InterventionCandidate] = []
    posteriors = (nuts_posteriors or {}).get("posteriors") if isinstance(nuts_posteriors, dict) else None
    posterior_lookup: dict[str, float] = {}
    if isinstance(posteriors, list):
        for entry in posteriors:
            if isinstance(entry, dict) and "treatment" in entry and "std" in entry:
                posterior_lookup[str(entry["treatment"]).lower()] = float(entry["std"])

    for row in scenarios or []:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or row.get("scenario") or "scenario")
        treatment = str(row.get("treatment") or row.get("variable") or name)
        magnitude = float(row.get("magnitude") or row.get("intervention_magnitude") or 0.0)
        mean_effect = float(row.get("delta") or row.get("mean_delta") or 0.0)
        cost = float(row.get("cost") or 1.0)
        equity = float(row.get("equity_weight") or 1.0)
        std_lookup = posterior_lookup.get(treatment.lower())
        if std_lookup is not None and magnitude:
            effect_std = abs(std_lookup * magnitude)
        else:
            effect_std = abs(mean_effect) * 0.25  # fallback CV
        out.append(
            InterventionCandidate(
                name=name,
                treatment=treatment,
                magnitude=magnitude,
                mean_effect=mean_effect,
                effect_std=effect_std,
                cost=cost,
                equity_weight=equity,
                notes=str(row.get("notes") or ""),
            )
        )
    return out
