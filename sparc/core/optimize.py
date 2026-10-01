"""S7 — budget-constrained allocation on validated benefit surfaces.

Each cell's cooling from adding dose d of a variable is modelled as the
footprint marginal b_i (total cooling across the neighbourhood per unit
dose, S4) decaying with the cell's saturation scale:

    marginal_i(d) = b_i·exp(−d/d_s,i)        (b_i constant if the cell's curve is linear/censored)

The dose axis is cut into segments; each (cell, segment) becomes an item for
:func:`sparc.scenario.budget.optimize` with benefit per unit equal to the
segment's average marginal.  Because the marginals are non-increasing, the
greedy ratio solver fills a cell's segments in order — the exact solution of
this concave continuous knapsack.  The chosen allocation is then re-run
through the scenario engine ("closed loop") and the realised cooling is
reported next to the planned one, since spillovers between treated cells are
not additive.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from sparc.core.response import VariableResponse
from sparc.core.scenarios import Intervention, ScenarioEngine, ScenarioSpec

log = logging.getLogger(__name__)


def build_segments(vr: VariableResponse, cost_per_unit, n_segments: int = 6, cap: np.ndarray | None = None,
                   benefit_weight: np.ndarray | None = None) -> pd.DataFrame:
    """Dose segments per cell.  ``cap`` limits the dose a cell can take (e.g.
    plantable space); ``benefit_weight`` rescales each cell's cooling (e.g.
    residents around it, mean 1)."""
    sign = -1.0 if vr.direction == "decrease" else 1.0
    maps = vr.maps
    b = -sign * maps["footprint_effect_per_unit"].to_numpy(float)       # cooling per unit dose
    if benefit_weight is not None:
        b = b * np.asarray(benefit_weight, float)
    ds = maps["saturation_scale_ds"].to_numpy(float)
    head = np.maximum(maps["headroom"].to_numpy(float), 0.0)
    if cap is not None:
        head = np.minimum(head, np.maximum(np.asarray(cap, float), 0.0))
    dmax = float(max(vr.doses))
    edges = np.linspace(0.0, dmax, n_segments + 1)
    cost = np.broadcast_to(np.asarray(cost_per_unit, dtype=float), b.shape)
    rows = []
    for k in range(n_segments):
        a, c = edges[k], edges[k + 1]
        width = np.clip(np.minimum(c, head) - a, 0.0, None)
        sat = np.isfinite(ds) & (ds > 0)
        avg = np.where(sat, b * ds * (np.exp(-a / np.where(sat, ds, 1.0)) - np.exp(-c / np.where(sat, ds, 1.0))) / (c - a), b)
        rows.append(pd.DataFrame({"cell": np.arange(len(b)), "segment": k, "x_max": width,
                                  "benefit_per_unit": avg, "cost_per_unit": cost}))
    seg = pd.concat(rows, ignore_index=True)
    return seg[(seg.x_max > 0) & (seg.benefit_per_unit > 0)].reset_index(drop=True)


def optimise_allocation(engine: ScenarioEngine, vr: VariableResponse, budget: float, cost_per_unit=1.0,
                        equity_scores: np.ndarray | None = None, equity_focus: float = 0.0,
                        multipliers=(0.25, 0.5, 1.0, 2.0), cap: np.ndarray | None = None,
                        benefit_weight: np.ndarray | None = None) -> dict:
    from sparc.scenario.budget import optimize, pareto_sweep

    seg = build_segments(vr, cost_per_unit, cap=cap, benefit_weight=benefit_weight)
    if seg.empty:
        return {"status": "no positive-benefit segments", "variable": vr.variable}
    eq = None
    if equity_scores is not None:
        eq = np.asarray(equity_scores, dtype=float)[seg["cell"].to_numpy()]
    benefits = seg["benefit_per_unit"].to_numpy(float)
    costs = seg["cost_per_unit"].to_numpy(float)
    xmax = seg["x_max"].to_numpy(float)
    res = optimize(benefits, budget, costs=costs, x_max=xmax, solver="greedy",
                   equity_scores=eq, equity_focus=equity_focus)
    alloc_items = np.asarray(res.allocation, dtype=float)
    n = engine.data.n
    dose = np.zeros(n)
    np.add.at(dose, seg["cell"].to_numpy(), alloc_items)
    sign = -1.0 if vr.direction == "decrease" else 1.0
    closed = engine.run(ScenarioSpec(name=f"optimised {vr.variable}",
                                     interventions=[Intervention(vr.variable, "add", 0.0, per_point=sign * dose)]))
    sweep = pareto_sweep(benefits, budget, costs=costs, x_max=xmax, multipliers=multipliers, solver="greedy",
                         equity_scores=eq, equity_focus=equity_focus)
    treated = dose > 0
    return {
        "variable": vr.variable,
        "budget": float(budget),
        "planned_total_cooling": float(res.total_benefit),
        "realized_total_cooling": float(-closed.delta.sum()),
        "realized_mean_cooling_treated": float(-closed.delta[treated].mean()) if treated.any() else 0.0,
        "realized_mean_cooling_all": float(-closed.delta.mean()),
        "n_cells_treated": int(treated.sum()),
        "mean_dose_treated": float(dose[treated].mean()) if treated.any() else 0.0,
        "total_cost": float(res.total_cost),
        "gini": float(res.gini),
        "pareto": sweep.to_dict(),
        "dose": dose,
        "closed_loop_delta": closed.delta,
    }
