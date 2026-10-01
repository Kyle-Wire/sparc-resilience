"""``optimize.planned_allocation`` (SPEC §11 item 15): the planned half of S7, without the scenario engine."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def _response(n: int = 400, seed: int = 0, direction: str = "increase"):
    from sparc.core.response import VariableResponse

    rng = np.random.default_rng(seed)
    foot = -np.abs(rng.normal(0.05, 0.03, n))                      # cooling per unit (ΔT < 0)
    foot[rng.random(n) < 0.1] *= -1.0                              # a few cells would warm: never treated
    ds = np.where(rng.random(n) < 0.7, rng.uniform(5.0, 30.0, n), np.nan)   # saturating or linear cells
    head = rng.uniform(0.0, 40.0, n)
    maps = pd.DataFrame({"footprint_effect_per_unit": foot if direction == "increase" else -foot,
                         "saturation_scale_ds": ds, "headroom": head})
    return VariableResponse(variable="canopy", direction=direction, doses=[0.0, 5.0, 10.0, 20.0, 40.0],
                            curve=pd.DataFrame({"dose": [0.0]}), maps=maps)


class _Engine:
    """Closed-loop stand-in: realised ΔT = −0.04 per unit of the per-point edit."""

    def __init__(self):
        self.specs = []

    def run(self, spec):
        self.specs.append(spec)
        dose = np.asarray(spec.interventions[0].per_point, float)
        return SimpleNamespace(delta=-0.04 * np.abs(dose))


def test_planned_totals_equal_optimise_allocation_and_the_greedy_solver():
    from sparc.core.optimize import build_segments, optimise_allocation, planned_allocation
    from sparc.scenario.budget import optimize

    vr = _response()
    plan = planned_allocation(vr, 1500.0, cost_per_unit=1.0)
    eng = _Engine()
    full = optimise_allocation(eng, vr, 1500.0, cost_per_unit=1.0)
    for k in ("planned_total_cooling", "n_cells_treated", "mean_dose_treated", "total_cost", "gini",
              "min_dose_dropped_cost"):
        assert full[k] == plan[k], k
    assert full["pareto"] == plan["pareto"]
    np.testing.assert_array_equal(full["dose"], plan["dose"])
    np.testing.assert_array_equal(eng.specs[0].interventions[0].per_point, plan["dose"])
    assert full["realized_total_cooling"] == pytest.approx(0.04 * plan["dose"].sum())

    seg = build_segments(vr, 1.0)
    ref = optimize(seg["benefit_per_unit"].to_numpy(float), 1500.0, costs=seg["cost_per_unit"].to_numpy(float),
                   x_max=seg["x_max"].to_numpy(float), solver="greedy")
    assert plan["planned_total_cooling"] == pytest.approx(ref.total_benefit, rel=1e-12)
    assert plan["total_cost"] == pytest.approx(ref.total_cost, rel=1e-12)
    assert plan["planned_benefit"].sum() == pytest.approx(ref.total_benefit, rel=1e-9)
    assert plan["total_cost"] <= 1500.0 * (1 + 1e-9)


def test_n_cells_treated_counts_cells_not_segments():
    from sparc.core.optimize import planned_allocation

    vr = _response()
    plan = planned_allocation(vr, 1500.0)
    dose = plan["dose"]
    treated = np.flatnonzero(dose > 0)
    assert plan["n_cells_treated"] == treated.size
    assert plan["mean_dose_treated"] == pytest.approx(dose[treated].mean())
    assert (plan["planned_benefit"][dose == 0] == 0).all() and (plan["planned_benefit"][treated] > 0).all()
    pts = plan["pareto"]["points"]
    assert [p["budget"] for p in pts] == [375.0, 750.0, 1500.0, 3000.0]
    one = next(p for p in pts if p["budget"] == 1500.0)
    assert one["n_cells"] == plan["n_cells_treated"]
    assert one["n_segments"] > one["n_cells"]                     # several dose segments per treated cell
    assert one["total_benefit"] == pytest.approx(plan["planned_total_cooling"], rel=1e-12)
    assert all(a["total_benefit"] <= b["total_benefit"] + 1e-12 for a, b in zip(pts, pts[1:]))
    assert (dose <= np.maximum(vr.maps["headroom"].to_numpy(), 0) + 1e-9).all()


def test_min_dose_is_a_post_filter_and_the_freed_budget_is_reported():
    from sparc.core.optimize import planned_allocation

    vr = _response(seed=2)
    base = planned_allocation(vr, 900.0)
    cut = planned_allocation(vr, 900.0, min_dose=8.0)
    low = (base["dose"] > 0) & (base["dose"] < 8.0)
    assert low.any()
    np.testing.assert_array_equal(cut["dose"], np.where(low, 0.0, base["dose"]))     # nothing re-spent
    assert cut["min_dose_dropped_cost"] == pytest.approx(base["dose"][low].sum())
    assert cut["total_cost"] == pytest.approx(base["total_cost"] - cut["min_dose_dropped_cost"])
    assert cut["n_cells_treated"] == base["n_cells_treated"] - int(low.sum())
    assert cut["planned_total_cooling"] == pytest.approx(cut["planned_benefit"].sum())
    assert cut["planned_total_cooling"] < base["planned_total_cooling"]
    assert base["min_dose_dropped_cost"] == 0.0


def test_equity_scores_are_per_cell_and_cap_limits_doses():
    from sparc.core.optimize import planned_allocation

    vr = _response(seed=3)
    n = len(vr.maps)
    eq = np.zeros(n)
    eq[: n // 2] = 1.0                                            # the first half of the cells is prioritised
    plain = planned_allocation(vr, 600.0)
    fair = planned_allocation(vr, 600.0, equity_scores=eq, equity_focus=1.0)
    assert fair["dose"][: n // 2].sum() > plain["dose"][: n // 2].sum()
    assert fair["dose"][n // 2:].sum() == 0.0                     # focus 1: zero-score cells get no benefit
    cap = np.full(n, 2.0)
    capped = planned_allocation(vr, 600.0, cap=cap)
    assert capped["dose"].max() <= 2.0 + 1e-9


def test_nothing_to_allocate_returns_status_and_decrease_direction_works():
    from sparc.core.optimize import optimise_allocation, planned_allocation

    vr = _response()
    vr.maps["footprint_effect_per_unit"] = np.abs(vr.maps["footprint_effect_per_unit"])   # every cell warms
    out = planned_allocation(vr, 100.0)
    assert out == {"status": "no positive-benefit segments", "variable": "canopy"}
    assert optimise_allocation(_Engine(), vr, 100.0) == out
    dec = planned_allocation(_response(direction="decrease"), 500.0)
    assert dec["n_cells_treated"] > 0 and dec["planned_total_cooling"] > 0
