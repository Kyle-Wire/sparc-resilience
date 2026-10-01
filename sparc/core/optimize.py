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

:func:`planned_allocation` is the planned part alone (no engine, well under
a second): Studio's budget planner calls it on the slider, and
:func:`optimise_allocation` is it plus the closed loop.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from sparc.core import progress
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


def _pareto_point(seg: pd.DataFrame, benefits: np.ndarray, costs: np.ndarray, alloc: np.ndarray,
                  budget: float, gini: float) -> dict:
    on = alloc > 1e-9
    return {"budget": float(budget), "total_benefit": float(np.sum(benefits * alloc)),
            "n_cells": int(np.unique(seg["cell"].to_numpy()[on]).size), "n_segments": int(on.sum()),
            "gini": float(gini)}


def planned_allocation(vr: VariableResponse, budget: float, cost_per_unit=1.0, equity_scores: np.ndarray | None = None,
                       equity_focus: float = 0.0, multipliers=(0.25, 0.5, 1.0, 2.0), cap: np.ndarray | None = None,
                       benefit_weight: np.ndarray | None = None, min_dose: float = 0.0) -> dict:
    """The greedy segment allocation of ``budget`` without the closed loop.

    Returns ``dose`` and ``planned_benefit`` per cell (in the order of
    ``vr.maps``), ``planned_total_cooling``, ``n_cells_treated`` (cells, not
    segments), ``mean_dose_treated``, ``total_cost``, ``gini`` (of the
    segment allocation), ``min_dose_dropped_cost`` and ``pareto`` (one point
    per budget multiplier: total benefit, cells, segments, Gini).  Cell
    allocations below ``min_dose`` are set to 0 *after* the greedy pass; the
    budget they free is reported, not re-spent.  ``equity_scores`` are per
    cell (0–1, aligned with the cells by id upstream).  Without any
    positive-benefit segment it returns ``{status, variable}``.
    """
    from sparc.scenario.budget import optimize

    with progress.task("segments"):
        seg = build_segments(vr, cost_per_unit, cap=cap, benefit_weight=benefit_weight)
    if seg.empty:
        return {"status": "no positive-benefit segments", "variable": vr.variable}
    cells = seg["cell"].to_numpy()
    eq = None
    if equity_scores is not None:
        eq = np.asarray(equity_scores, dtype=float)[cells]
    benefits = seg["benefit_per_unit"].to_numpy(float)
    costs = seg["cost_per_unit"].to_numpy(float)
    xmax = seg["x_max"].to_numpy(float)
    n = len(vr.maps)
    with progress.task("allocate") as sp:
        res = optimize(benefits, budget, costs=costs, x_max=xmax, solver="greedy",
                       equity_scores=eq, equity_focus=equity_focus)
        alloc = np.asarray(res.allocation, dtype=float)
        dose = np.zeros(n)
        np.add.at(dose, cells, alloc)
        dropped_cost, dropped = 0.0, False
        if min_dose and min_dose > 0:
            low = (dose > 0) & (dose < float(min_dose))
            drop = low[cells] & (alloc > 0)
            dropped = bool(drop.any())
            dropped_cost = float(np.sum(costs[drop] * alloc[drop]))
            alloc = np.where(drop, 0.0, alloc)
            dose = np.where(low, 0.0, dose)
        planned = np.zeros(n)
        np.add.at(planned, cells, benefits * alloc)
        treated = dose > 0
        total = float(np.sum(benefits * alloc)) if dropped else float(res.total_benefit)
        sp.metrics.update(planned_total=total, n_cells_treated=int(treated.sum()))
    progress.metric("planned_total", total, variable=vr.variable)
    with progress.task("pareto", unit="pareto"):
        points = []
        for m in multipliers:
            r = optimize(benefits, budget * float(m), costs=costs, x_max=xmax, solver="greedy",
                         equity_scores=eq, equity_focus=equity_focus)
            points.append(_pareto_point(seg, benefits, costs, np.asarray(r.allocation, dtype=float),
                                        budget * float(m), r.gini))
    return {
        "variable": vr.variable,
        "budget": float(budget),
        "dose": dose,
        "planned_benefit": planned,
        "planned_total_cooling": total,
        "n_cells_treated": int(treated.sum()),
        "mean_dose_treated": float(dose[treated].mean()) if treated.any() else 0.0,
        "total_cost": float(res.total_cost) - dropped_cost,
        "gini": float(res.gini),
        "min_dose": float(min_dose or 0.0),
        "min_dose_dropped_cost": dropped_cost,
        "pareto": {"points": points},
    }


def optimise_allocation(engine: ScenarioEngine, vr: VariableResponse, budget: float, cost_per_unit=1.0,
                        equity_scores: np.ndarray | None = None, equity_focus: float = 0.0,
                        multipliers=(0.25, 0.5, 1.0, 2.0), cap: np.ndarray | None = None,
                        benefit_weight: np.ndarray | None = None, min_dose: float = 0.0) -> dict:
    """:func:`planned_allocation` plus the closed loop: the allocation re-run through the scenario engine."""
    plan = planned_allocation(vr, budget, cost_per_unit=cost_per_unit, equity_scores=equity_scores,
                              equity_focus=equity_focus, multipliers=multipliers, cap=cap,
                              benefit_weight=benefit_weight, min_dose=min_dose)
    if "status" in plan:
        return plan
    dose = plan["dose"]
    sign = -1.0 if vr.direction == "decrease" else 1.0
    progress.check_cancel()
    with progress.task("closed_loop") as sp:
        closed = engine.run(ScenarioSpec(name=f"optimised {vr.variable}",
                                         interventions=[Intervention(vr.variable, "add", 0.0, per_point=sign * dose)]))
        realised = float(-closed.delta.sum())
        sp.metrics["realised_total"] = realised
    progress.metric("realised_total", realised, variable=vr.variable)
    treated = dose > 0
    return {
        "variable": vr.variable,
        "budget": float(budget),
        "planned_total_cooling": plan["planned_total_cooling"],
        "realized_total_cooling": realised,
        "realized_mean_cooling_treated": float(-closed.delta[treated].mean()) if treated.any() else 0.0,
        "realized_mean_cooling_all": float(-closed.delta.mean()),
        "n_cells_treated": plan["n_cells_treated"],
        "mean_dose_treated": plan["mean_dose_treated"],
        "total_cost": plan["total_cost"],
        "gini": plan["gini"],
        "min_dose_dropped_cost": plan["min_dose_dropped_cost"],
        "pareto": plan["pareto"],
        "dose": dose,
        "planned_benefit": plan["planned_benefit"],
        "closed_loop_delta": closed.delta,
    }
