"""A simulated heat-watch campaign on the real layout: what vehicles would have measured, and the map
a random forest would have made from it.

Routes follow a street grid (every ``street_m`` along both axes) over paved cells.  A route line is
cut into **segments** at gaps (unpaved stretches); each segment is one continuous pass.  Segments are
dealt to ``n_vehicles`` vehicles that drive them one after another through the afternoon hour, one
sample per cell (about every ``step_s`` seconds).  A measurement is the planted temperature of the
cell plus the afternoon warming drift, the vehicle's sensor offset and sensor noise.

The **map** is made the way heat-watch products are: the traverse temperatures detrended in time (a
straight line through the hour, as the campaign protocol does), a random forest on land-cover
predictors fitted to them, used to fill every cell, then the real target's share of whole-degree
values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HOUR_S = 3600.0


@dataclass
class Campaign:
    samples: pd.DataFrame          # cell, seg, vehicle, order, t_s, temp (+ truth for diagnostics)
    product: np.ndarray            # the forest-made map at every cell
    drift_f_per_h: float


def route_segments(layout, street_m: float = 300.0, imp_min: float = 40.0, rng=None, min_len: int = 4
                   ) -> list[np.ndarray]:
    """Point indices of every route segment, in driving order (row-major lines of the street grid)."""
    g = layout.grid
    rng = rng or np.random.default_rng(0)
    k = max(int(round(street_m / g.dx)), 2)
    oy, ox = (int(v) for v in rng.integers(0, k, size=2))
    idx = np.full(g.shape, -1, dtype=np.int64)
    idx[g.iy, g.ix] = np.arange(g.iy.size)
    paved = np.zeros(g.shape, bool)
    paved[g.iy, g.ix] = layout.col("impervious") > imp_min
    segs: list[np.ndarray] = []

    def cut(line_idx: np.ndarray, line_ok: np.ndarray) -> None:
        run: list[int] = []
        for i, ok in zip(line_idx, line_ok):
            if ok and i >= 0:
                run.append(int(i))
            else:
                if len(run) >= min_len:
                    segs.append(np.asarray(run))
                run = []
        if len(run) >= min_len:
            segs.append(np.asarray(run))

    for row in range(oy, g.shape[0], k):                 # east-west streets
        cut(idx[row, :], paved[row, :])
    for col in range(ox, g.shape[1], k):                 # north-south streets
        cut(idx[:, col], paved[:, col])
    return segs


def drive(segs: list[np.ndarray], rng, n_vehicles: int = 10, step_s: float = 3.0) -> pd.DataFrame:
    """Deal the segments to vehicles and time every sample within the hour."""
    order = rng.permutation(len(segs))
    rows = []
    for v in range(n_vehicles):
        mine = order[v::n_vehicles]
        total = sum(len(segs[s]) for s in mine) * step_s
        speed = max(total / HOUR_S, 1.0)                 # a vehicle with more road drives faster
        t = float(rng.uniform(0, 60))
        for s in mine:
            cells = segs[s] if rng.random() < 0.5 else segs[s][::-1]
            dt = step_s / speed
            ts = t + dt * np.arange(cells.size)
            rows.append(pd.DataFrame({"cell": cells, "seg": int(s), "vehicle": v,
                                      "order": np.arange(cells.size), "t_s": ts}))
            t = float(ts[-1]) + dt + float(rng.uniform(5, 30))       # turning to the next street
    return pd.concat(rows, ignore_index=True)


def simulate(layout, T: np.ndarray, rng, features: np.ndarray, street_m: float = 300.0, n_vehicles: int = 10,
             drift_f_per_h: tuple[float, float] = (0.5, 1.5), vehicle_sd: float = 0.2, sensor_sd: float = 0.3,
             step_s: float = 3.0, product: bool = True) -> Campaign:
    """A campaign over the planted temperature field ``T`` (one value per cell)."""
    from sklearn.ensemble import RandomForestRegressor

    segs = route_segments(layout, street_m, rng=rng)
    df = drive(segs, rng, n_vehicles, step_s)
    slope = float(rng.uniform(*drift_f_per_h))
    off = rng.normal(0.0, vehicle_sd, n_vehicles)
    df["truth"] = T[df["cell"].to_numpy()]
    df["temp"] = (df["truth"] + slope * df["t_s"] / HOUR_S + off[df["vehicle"].to_numpy()]
                  + sensor_sd * rng.standard_normal(len(df)))
    prod = np.full(T.size, np.nan)
    if product:
        # the campaign protocol: detrend in time, then fill every cell with a forest on land cover
        t = df["t_s"].to_numpy() / HOUR_S
        b = np.polyfit(t, df["temp"].to_numpy(), 1)
        y = df["temp"].to_numpy() - b[0] * (t - t.mean())
        rf = RandomForestRegressor(n_estimators=120, min_samples_leaf=5, max_features=0.5, n_jobs=1,
                                   random_state=int(rng.integers(1 << 31)))
        cells = df["cell"].to_numpy()
        prod = rf.fit(features[cells], y).predict(features)
        share = layout.data.qa.get("target_fraction_integer_valued", 0.0)
        pick = rng.random(prod.size) < share
        prod[pick] = np.round(prod[pick])
    return Campaign(samples=df, product=prod, drift_f_per_h=slope)
