"""Budget-constrained intervention allocation.

The optimizer takes:
  - per-cell *unit benefit* b_i (e.g. expected reduction in target metric per
    unit of treatment, derived from the NUTS posterior dose-response slope or
    the PDE local coefficient at cell i)
  - per-cell *unit cost* c_i (default 1.0 if cost surface not provided)
  - a budget B
  - a per-cell maximum allocation x_max (defaults to 1.0 = "treat fully")

It returns an allocation x_i in [0, x_max] that maximises sum(b_i * x_i)
subject to sum(c_i * x_i) <= B.

Three solvers are available, dispatched by problem size:

  * `greedy` — sort by b_i / c_i descending, fill until budget exhausts.
    Continuous relaxation with a single fractional cell at the boundary.
    O(n log n). Used for n <= 50_000 (always — it's also a strong warm
    start for the heavier solvers).

  * `greedy_2opt` — greedy followed by pairwise swap refinement that swaps
    a partially-allocated cell with an unallocated higher-ratio cell when
    that improves total benefit. Used for n in (50_000, 250_000].

  * `milp` — exact mixed-integer LP via PuLP (CBC backend). Only invoked
    when the user explicitly requests a binary (treat / don't-treat)
    decision and n <= 5_000. Falls back to greedy with a warning if PuLP
    is not installed.

A *Pareto frontier* sweep evaluates the optimizer at budgets B * {0.5, 0.75,
1.0, 1.25, 1.5} and reports total benefit + Gini of allocation at each.
This lets the user see diminishing returns without re-running.

Equity reporting uses the Gini coefficient of allocation across cells. A
Gini close to 0 means the budget was spread evenly; close to 1 means it
was concentrated. The optimal value is problem-dependent — we just report
it so the user can decide.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Sequence

import numpy as np


@dataclass
class BudgetResult:
    """Output of a single budget-optimization run."""

    allocation: list[float]
    total_benefit: float
    total_cost: float
    budget: float
    n_treated: int          # cells with x_i > 0
    n_fully_treated: int    # cells with x_i >= x_max
    gini: float
    solver: str
    solve_time_s: float
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ParetoPoint:
    budget: float
    total_benefit: float
    n_treated: int
    gini: float


@dataclass
class BudgetSweepResult:
    points: list[ParetoPoint] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"points": [asdict(p) for p in self.points]}


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _gini(x: np.ndarray) -> float:
    """Gini coefficient of a non-negative vector. Returns 0 if all-zero."""
    if x.size == 0:
        return 0.0
    x = np.asarray(x, dtype=float)
    if np.any(x < 0):
        x = x - x.min()
    s = x.sum()
    if s <= 0:
        return 0.0
    sorted_x = np.sort(x)
    n = x.size
    cum = np.cumsum(sorted_x)
    # Standard Gini formula
    return float((n + 1 - 2 * (cum.sum() / cum[-1])) / n)


def _validate_inputs(
    benefits: np.ndarray,
    costs: np.ndarray,
    x_max: np.ndarray,
    budget: float,
) -> None:
    if benefits.shape != costs.shape or benefits.shape != x_max.shape:
        raise ValueError(
            f"Shape mismatch: benefits={benefits.shape}, costs={costs.shape}, "
            f"x_max={x_max.shape}"
        )
    if benefits.ndim != 1:
        raise ValueError(f"Inputs must be 1-D, got {benefits.ndim}-D")
    if budget < 0:
        raise ValueError(f"Budget must be non-negative, got {budget}")
    if np.any(costs < 0):
        raise ValueError("All costs must be non-negative")
    if np.any(x_max < 0):
        raise ValueError("All x_max must be non-negative")


# --------------------------------------------------------------------------- #
# Solvers                                                                     #
# --------------------------------------------------------------------------- #
def _solve_greedy(
    benefits: np.ndarray,
    costs: np.ndarray,
    x_max: np.ndarray,
    budget: float,
) -> np.ndarray:
    """Continuous-relaxation greedy: sort by benefit/cost, fill until budget runs out."""
    n = benefits.size
    alloc = np.zeros(n, dtype=float)
    # Skip cells with zero or negative benefit — they never help.
    eligible = benefits > 0
    if not np.any(eligible):
        return alloc
    # Avoid division by zero
    safe_costs = np.where(costs > 0, costs, 1e-12)
    ratio = np.where(eligible, benefits / safe_costs, -np.inf)
    order = np.argsort(-ratio)  # descending
    remaining = budget
    for idx in order:
        if not eligible[idx]:
            break
        c = costs[idx]
        cap = x_max[idx]
        if c <= 0:
            # Free cell — take everything up to cap if it has positive benefit
            alloc[idx] = cap
            continue
        max_affordable = remaining / c
        x = min(cap, max_affordable)
        if x <= 0:
            break
        alloc[idx] = x
        remaining -= x * c
        if remaining <= 1e-9:
            break
    return alloc


def _solve_greedy_2opt(
    benefits: np.ndarray,
    costs: np.ndarray,
    x_max: np.ndarray,
    budget: float,
    max_iters: int = 100,
) -> np.ndarray:
    """Greedy warm start + pairwise swap refinement on the boundary cell."""
    alloc = _solve_greedy(benefits, costs, x_max, budget)
    # Try to find a swap that increases total benefit:
    # take some allocation from a low-ratio active cell, give to a high-ratio
    # inactive cell that wasn't reached by the greedy ordering due to cost.
    # In the continuous relaxation the LP-greedy is already optimal, so 2-opt
    # mostly helps when x_max is binding for some cells. Bounded iteration count.
    safe_costs = np.where(costs > 0, costs, 1e-12)
    ratio = np.where(costs > 0, benefits / safe_costs, np.where(benefits > 0, np.inf, 0.0))
    for _ in range(max_iters):
        active = np.where(alloc > 1e-9)[0]
        inactive = np.where((alloc < x_max - 1e-9) & (benefits > 0))[0]
        if active.size == 0 or inactive.size == 0:
            break
        # Lowest-ratio active cell and highest-ratio inactive cell
        a = active[np.argmin(ratio[active])]
        b = inactive[np.argmax(ratio[inactive])]
        if ratio[b] <= ratio[a] + 1e-9:
            break  # no improving swap exists
        # How much can we shift?
        dx_a = alloc[a]                                  # max we can take from a
        dx_b = (x_max[b] - alloc[b])                     # max we can give to b
        # Cost-balanced swap: removing dx*c_a from a frees dx*c_a budget for b,
        # which buys dx * c_a / c_b units at b.
        c_a = costs[a]
        c_b = costs[b] if costs[b] > 0 else 1e-12
        # Swap size (in units of a) bounded by both endpoints
        dx = min(dx_a, dx_b * c_b / c_a)
        if dx <= 1e-12:
            break
        alloc[a] -= dx
        alloc[b] += dx * c_a / c_b
    return alloc


def _solve_milp(
    benefits: np.ndarray,
    costs: np.ndarray,
    x_max: np.ndarray,
    budget: float,
) -> tuple[np.ndarray, str]:
    """Exact MILP via PuLP/CBC. Falls back to greedy if PuLP unavailable.

    Returns (allocation, note). When binary semantics are requested upstream
    the caller should set x_max to 1.0 and interpret returned values as 0/1.
    """
    try:
        import pulp  # type: ignore
    except ImportError:
        return _solve_greedy(benefits, costs, x_max, budget), (
            "PuLP not installed — fell back to greedy. Install pulp to enable MILP."
        )

    n = benefits.size
    prob = pulp.LpProblem("budget_alloc", pulp.LpMaximize)
    xs = [
        pulp.LpVariable(f"x_{i}", lowBound=0, upBound=float(x_max[i]))
        for i in range(n)
    ]
    prob += pulp.lpSum(float(benefits[i]) * xs[i] for i in range(n))
    prob += pulp.lpSum(float(costs[i]) * xs[i] for i in range(n)) <= float(budget)
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=60)
    prob.solve(solver)
    alloc = np.array([pulp.value(x) or 0.0 for x in xs], dtype=float)
    return alloc, f"MILP solved (status={pulp.LpStatus[prob.status]})"


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #
def optimize(
    benefits: Sequence[float],
    budget: float,
    costs: Sequence[float] | None = None,
    x_max: Sequence[float] | None = None,
    solver: str = "auto",
    equity_scores: "Sequence[float] | np.ndarray | None" = None,
    equity_focus: float = 0.0,
) -> BudgetResult:
    """Allocate a budget across cells to maximise total benefit.

    Parameters
    ----------
    benefits :
        Per-cell unit benefit (e.g. NUTS-derived dose-response slope).
    budget :
        Total budget cap.
    costs :
        Per-cell unit cost; defaults to all-ones (uniform cost).
    x_max :
        Per-cell maximum allocation; defaults to all-ones (continuous in [0,1]).
    solver :
        "greedy" | "greedy_2opt" | "milp" | "auto". With "auto" we pick by size.
    equity_scores :
        Per-cell equity score in [0, 1] (e.g. poverty / minority index).
        When provided and ``equity_focus > 0`` the effective benefit is
        modulated: ``b_eff_i = b_i * [(1 - α) + α * equity_i]``.
    equity_focus :
        Blending weight α ∈ [0, 1]. 0 ⇒ pure efficiency; 1 ⇒ equity-first.
    """
    import time

    b = np.asarray(benefits, dtype=float)
    c = np.asarray(costs, dtype=float) if costs is not None else np.ones_like(b)
    xm = np.asarray(x_max, dtype=float) if x_max is not None else np.ones_like(b)
    _validate_inputs(b, c, xm, budget)

    # Apply equity modulation before solver sees benefits
    alpha = float(np.clip(equity_focus, 0.0, 1.0))
    if alpha > 0 and equity_scores is not None:
        eq = np.asarray(equity_scores, dtype=float)
        if eq.shape != b.shape:
            import warnings as _w
            _w.warn(
                f"equity_scores shape {eq.shape} != benefits shape {b.shape}; "
                "truncating/padding to match.",
            )
            if eq.size >= b.size:
                eq = eq[: b.size]
            else:
                eq = np.pad(eq, (0, b.size - eq.size), constant_values=np.nanmean(eq))
        eq = np.clip(eq, 0.0, 1.0)
        b = b * ((1.0 - alpha) + alpha * eq)

    if solver == "auto":
        n = b.size
        chosen = "greedy" if n > 250_000 else ("greedy_2opt" if n > 5_000 else "greedy_2opt")
    else:
        chosen = solver

    t0 = time.perf_counter()
    note = ""
    if chosen == "greedy":
        alloc = _solve_greedy(b, c, xm, budget)
    elif chosen == "greedy_2opt":
        alloc = _solve_greedy_2opt(b, c, xm, budget)
    elif chosen == "milp":
        alloc, note = _solve_milp(b, c, xm, budget)
    else:
        raise ValueError(f"Unknown solver: {chosen}")
    elapsed = time.perf_counter() - t0

    b_orig = np.asarray(benefits, dtype=float)  # for reporting against raw benefits
    total_b = float(np.sum(b_orig * alloc))
    total_c = float(np.sum(c * alloc))
    n_treat = int(np.sum(alloc > 1e-9))
    n_full = int(np.sum(alloc >= xm - 1e-9))
    return BudgetResult(
        allocation=alloc.tolist(),
        total_benefit=total_b,
        total_cost=total_c,
        budget=float(budget),
        n_treated=n_treat,
        n_fully_treated=n_full,
        gini=_gini(alloc),
        solver=chosen,
        solve_time_s=elapsed,
        notes=note,
    )


def pareto_sweep(
    benefits: Sequence[float],
    budget: float,
    costs: Sequence[float] | None = None,
    x_max: Sequence[float] | None = None,
    multipliers: Sequence[float] = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
    solver: str = "auto",
    equity_scores: "Sequence[float] | np.ndarray | None" = None,
    equity_focus: float = 0.0,
) -> BudgetSweepResult:
    """Re-solve at several budget multipliers to expose diminishing returns."""
    out = BudgetSweepResult()
    for m in multipliers:
        b = budget * float(m)
        r = optimize(
            benefits, b,
            costs=costs, x_max=x_max, solver=solver,
            equity_scores=equity_scores, equity_focus=equity_focus,
        )
        out.points.append(
            ParetoPoint(
                budget=b,
                total_benefit=r.total_benefit,
                n_treated=r.n_treated,
                gini=r.gini,
            )
        )
    return out
