"""The kilometre scale: the same streets under different winds.

Comparing nearby readings isolates canopy's effect within a few hundred metres, and cannot see further:
a kilometre-scale cooling effect and a kilometre-scale factor that tracks canopy leave the same traces in
one run.  Heat Watch drives the same routes several times (morning, afternoon, evening).  When the wind
changes between runs, the canopy upwind of a street changes while everything fixed about the street
stays put, so comparing **each street with itself across runs** removes every time-invariant confounder,
whatever its scale.

The design (``wind_shift``):

* one value per cell and run (consecutive readings averaged), cells seen in at least two runs;
* cell fixed effects (each street compared with itself), run effects, warming drift and vehicle offsets
  per run;
* per-run coefficients on every isotropic layer (canopy and impervious rings, albedo, elevation, water)
  and on a smooth spatial basis (waves of 3 km and longer): what a confounder does may change through
  the day, and its time-varying part is absorbed as long as it does not line up with the wind;
* the treatment: canopy in the 1 km sector upwind minus the 1 km sector downwind of the cell, under each
  run's wind; the same contrasts of impervious surface and of near-water (a shore upwind cools too) are
  controls, with along-wind minus cross-wind contrasts of all three.

The estimand is **advected kilometre-scale cooling**: the change at a street when canopy rises by 10 pp
over the kilometre upwind rather than the kilometre downwind.  It is the part of canopy's
kilometre-scale effect that moves with the wind; a part that spreads the same way in every direction is
not identified by any wind change.

Inference is design-based: the estimate is recomputed with every run's wind rotated together by 10°,
20°, …, 350°.  Rotations keep the city's spatial structure and the runs' relative winds but break the
link between canopy and where the air came from, so they give the estimate's distribution when canopy
carries no cooling downwind.  The p-value is the share of rotations at least as extreme (cluster-robust
standard errors over 2 km blocks are also reported, but they understate the error of a kilometre-scale
contrast).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from sparc.core.identify import DOSE

REACH_M = 1000.0
STEP_DEG = 10
BASIS_M = 3000.0
BLOCK_M = 2000.0
VARIABLES = ("canopy", "impervious", "near_water")


def toward(from_deg: float) -> tuple[float, float]:
    """Unit vector the air moves toward (grid axes: +x east, +y north) for a wind blowing *from*
    ``from_deg`` (degrees clockwise from north)."""
    r = math.radians(from_deg)
    return (-math.sin(r), -math.cos(r))


def from_deg(u: float, v: float) -> float:
    """Meteorological direction of a wind vector that blows *toward* (u, v)."""
    return float((math.degrees(math.atan2(-u, -v)) + 360.0) % 360.0)


def sector_table(layout, reach_m: float = REACH_M, step_deg: int = STEP_DEG) -> dict:
    """For every wind direction (every ``step_deg``°) and variable: (upwind − downwind, along − cross)
    sector contrasts at every cell.  About five seconds for a city of 55k cells."""
    from sparc.core.identify.layers import sector_mean

    g = layout.grid
    w = layout.col("water_distance")
    vals = {"canopy": layout.col("canopy"), "impervious": layout.col("impervious")}
    if w is not None:
        vals["near_water"] = np.exp(-np.asarray(w, float) / 150.0)
    out = {}
    for d in range(0, 360, step_deg):
        u = toward(d)
        for name, v in vals.items():
            up = sector_mean(v, g, (-u[0], -u[1]), reach_m)
            dn = sector_mean(v, g, u, reach_m)
            left = sector_mean(v, g, (-u[1], u[0]), reach_m)
            right = sector_mean(v, g, (u[1], -u[0]), reach_m)
            out[(name, d)] = (up - dn, 0.5 * (up + dn) - 0.5 * (left + right))
    return {"step": step_deg, "reach_m": reach_m, "contrasts": out, "variables": sorted(vals)}


def _snap(d: float, step: int) -> int:
    return int(round(d / step) * step) % 360


def prepare(df: pd.DataFrame, L, run_order: list, grid, basis_m: float = BASIS_M) -> dict:
    """Everything that does not depend on the winds: the panel, the within-cell transformation and the
    controls projected out (QR), so each rotation only re-projects six columns."""
    from sparc.core.identify.layers import spatial_basis

    g2 = (df.groupby(["cell", "run"], sort=False)
            .agg(temp=("temp", "mean"), t=("t_s", "mean"), veh=("vehicle", "first")).reset_index())
    g2 = g2[g2["run"].isin(run_order)]
    g2 = g2[g2.groupby("cell")["run"].transform("nunique") >= 2].reset_index(drop=True)
    if g2.empty:
        raise ValueError("no cell was driven in two runs")
    cells = g2["cell"].to_numpy()
    rk = g2["run"].map({r: k for k, r in enumerate(run_order)}).to_numpy()
    basis = spatial_basis(grid, basis_m)
    iso = L.has(["cb0_0", "cb0_r100", "cb0_r300", "cb0_r1000", "impervious_0", "imp_r100", "imp_r300", "imp_r1000"]) \
        + L.geography()
    th = g2["t"].to_numpy(float) / 3600.0
    th = th - np.array([th[rk == k].mean() for k in range(len(run_order))])[rk]
    cols = []
    for k in range(len(run_order)):
        isk = (rk == k).astype(float)
        if k:
            cols.append(isk)
            cols += [isk * L.values[f][cells] for f in iso]
            cols += [isk * basis[cells, j] for j in range(basis.shape[1])]
        cols += [isk * th, isk * th ** 2]
    veh = g2["veh"].to_numpy()
    cols += [(veh == v).astype(float) for v in np.unique(veh)[1:]]
    _, inv = np.unique(cells, return_inverse=True)
    cnt = np.bincount(inv)

    def within(M: np.ndarray) -> np.ndarray:
        S = np.zeros((cnt.size, M.shape[1]))
        np.add.at(S, inv, M)
        return M - (S / cnt[:, None])[inv]

    X = within(np.column_stack(cols))
    X = X[:, X.std(axis=0) > 1e-9]
    Q, _ = np.linalg.qr(X)
    y = within(g2["temp"].to_numpy(float)[:, None])[:, 0]
    return {"cells": cells, "run": rk, "within": within, "Q": Q, "ry": y - Q @ (Q.T @ y),
            "n_cells": int(cnt.size), "n_obs": int(len(g2)), "runs": list(run_order)}


def _estimate(prep: dict, table: dict, dirs: list[int], blocks: np.ndarray) -> tuple[float, float]:
    from sparc.core.identify.estimators import cluster_ols

    cells, rk = prep["cells"], prep["run"]
    F = []
    for name in table["variables"]:
        A = np.empty(cells.size)
        B = np.empty(cells.size)
        for k, d in enumerate(dirs):
            s = rk == k
            a, b = table["contrasts"][(name, d)]
            A[s], B[s] = a[cells[s]], b[cells[s]]
        F += [A, B]
    F = prep["within"](np.column_stack(F))
    F = F - prep["Q"] @ (prep["Q"].T @ F)
    names = [f"{n}_{c}" for n in table["variables"] for c in ("updown", "alongcross")]
    b, V = cluster_ols(prep["ry"], F, blocks[cells])
    j = names.index("canopy_updown")
    return DOSE * float(b[j]), DOSE * math.sqrt(max(float(V[j, j]), 0.0))


def wind_shift(df: pd.DataFrame, L, table: dict, winds: dict, grid, rotate: bool = True,
               block_m: float = BLOCK_M) -> dict:
    """Advected kilometre-scale cooling from runs under different winds.  ``df``: cell, run, t_s, vehicle
    (unique across runs), temp; ``winds``: run → wind-from direction (degrees)."""
    from sparc.core.identify.layers import block_ids

    runs = [r for r in winds if r in set(df["run"])]
    if len(runs) < 2:
        raise ValueError("the wind-shift design needs at least two runs with a known wind")
    prep = prepare(df, L, runs, grid)
    step = int(table["step"])
    dirs = [_snap(winds[r], step) for r in runs]
    blocks = block_ids(grid, block_m)
    est, se = _estimate(prep, table, dirs, blocks)
    out = {"estimator": "wind_shift", "label": "Same streets, different winds: canopy upwind − downwind (1 km)",
           "estimand": "advected_1000", "kind": "effect", "level": "traverse", "where": "street",
           "estimate": est, "se": se, "lo": est - 1.959963984540054 * se, "hi": est + 1.959963984540054 * se,
           "n": prep["n_obs"], "n_cells": prep["n_cells"], "runs": runs, "wind_from_deg": dirs,
           "spread_deg": _spread(dirs)}
    if rotate:
        null = np.array([_estimate(prep, table, [(d + r) % 360 for d in dirs], blocks)[0]
                         for r in range(step, 360, step)])
        out.update(p_rotation=float((1 + np.sum(np.abs(null) >= abs(est))) / (1 + null.size)),
                   p_cooling=float((1 + np.sum(null <= est)) / (1 + null.size)),
                   null_sd=float(null.std()), n_rotations=int(null.size))
    return out


def _spread(dirs: list[int]) -> float:
    """Circular spread of the runs' wind directions (0 = all the same, 180 = opposite)."""
    if len(dirs) < 2:
        return 0.0
    best = 0.0
    for a in dirs:
        for b in dirs:
            best = max(best, abs((a - b + 180) % 360 - 180))
    return float(best)
