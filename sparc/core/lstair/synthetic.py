"""Planted worlds for the LST pilot: does the harness say what it should when the answer is known?

Each world is a set of cities in the dataset's cache format (:class:`~sparc.core.lstair.dataset.CityData`), so
:mod:`~sparc.core.lstair.evaluate` runs on them exactly as on real data.  A city (built from the blocks of
:mod:`sparc.core.synthetic`) has smooth random canopy, impervious, NDVI (a mediator of canopy), albedo, elevation
and a river; Heat Watch-like streets every 300 m over paved cells (:func:`sparc.core.identify.campaign.route_segments`)
are the driven cells.  Cities differ in their land-cover mix and seed.

**Afternoon air** (15–16 h) is physics-shaped: the normalised source q = (1 − albedo)(0.4 + 0.6·imp) − A_c·h(canopy)
− w_v·NDVI − w_w·water (+ the hidden source in the worlds that have one), centred, pushed through the steady
advection–diffusion–relaxation operator (:func:`sparc.core.operators.solve`, L = 150 m, no wind), times a gain,
plus a lapse term and sensor-scale noise; h(c) = 1 − e^{−c/15} is the saturating canopy shade of
:func:`sparc.core.synthetic.canopy_shape`.  **LST** (the ~10:00 Landsat overpass) is a linear function of the
cell's land cover plus pixel noise, plus the hidden source where the world says so.

Worlds (:data:`SCENARIOS`):

* ``informative`` — a hidden heat source H (patches of anthropogenic heat or hot roof material, independent of
  every land-cover layer) heats both the surface and the air: LST carries information beyond land cover.
  The harness must find B and C better than A (95% intervals below 0).
* ``redundant`` — no hidden source: LST is land cover plus noise.  The verdict must be "not shown".
* ``mismatch`` — the 10:00 surface pattern H_am and the 15:00 air pattern H_pm differ (correlation
  :data:`MISMATCH_RHO`): what LST shows is only partly what heats the afternoon air.  The harness reports how
  much of the ``informative`` gain is lost.

A known limit of the test design, stated rather than hidden: B − A tests information beyond *the land-cover
model A*, so an LST that is a function of land cover A cannot represent (e.g. a sharply nonlinear one) can still
look informative.  The ``redundant`` world uses an LST that A can represent.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from sparc.core import operators as ops
from sparc.core.lstair.dataset import CityData, CityGrid, derived_rasters
from sparc.core.synthetic import canopy_shape, gaussian_random_field

SCENARIOS = ("informative", "redundant", "mismatch")
MISMATCH_RHO = 0.3


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def make_city(k: int, scenario: str = "informative", n: int = 64, dx: float = 30.0, seed: int = 0,
              hidden_air: float = 1.0, hidden_lst: float = 4.0, noise_c: float = 0.15, lst_noise_c: float = 0.8,
              L: float = 150.0, gain: float = 2.5) -> CityData:
    """One synthetic city of ``scenario`` (see the module docstring); ``k`` and ``seed`` make it unique."""
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}")
    rng = np.random.default_rng([int(seed), int(k), SCENARIOS.index(scenario)])
    shape = (n, n)
    grid = CityGrid(crs="EPSG:32618", x0=500015.0 + 50000.0 * k, y0=4400015.0, dx=dx, nx=n, ny=n)
    mix = rng.normal(0.0, 0.5, 3)                                         # city-level land-cover mix
    U = gaussian_random_field(shape, 25.0, rng)
    canopy = 70.0 * _sig(1.2 * gaussian_random_field(shape, 5.0, rng) + 0.6 * U + mix[0])
    canopy[rng.random(shape) < 0.15] = 0.0
    imp = 100.0 * _sig(-0.8 * gaussian_random_field(shape, 7.0, rng) + 0.6 * gaussian_random_field(shape, 3.0, rng)
                       - 0.5 * U + mix[1] - 0.02 * (canopy - 30.0))
    albedo = np.clip(0.16 + 0.04 * gaussian_random_field(shape, 4.0, rng) - 0.02 * (imp / 100.0 - 0.5), 0.05, 0.5)
    ndvi = 0.1 + 0.004 * canopy - 0.001 * imp + 0.02 * rng.standard_normal(shape)
    elev = 30.0 + 12.0 * gaussian_random_field(shape, 30.0, rng)
    yy, xx = np.mgrid[0:n, 0:n]
    river = np.abs(yy - (n * (0.2 + 0.6 * rng.random()) + 5.0 * np.sin(xx / (6.0 + 4.0 * rng.random())))) < 1.2
    from scipy import ndimage

    water_dist = ndimage.distance_transform_edt(~river) * dx

    def hidden():
        z = gaussian_random_field(shape, 6.0, rng)
        return np.clip(z - 0.5, 0.0, None)                              # ~200 m patches: industrial sites, hot roofs
    H = hidden()
    if scenario == "informative":
        H_air, H_lst = H, H
    elif scenario == "redundant":
        H_air, H_lst = np.zeros(shape), np.zeros(shape)
    else:
        H2 = hidden()
        H_lst = H
        H_air = MISMATCH_RHO * (H - H.mean()) / (H.std() + 1e-12) + np.sqrt(1 - MISMATCH_RHO ** 2) * (H2 - H2.mean()) / (H2.std() + 1e-12)
        H_air = H_air * H.std() + H.mean()

    # afternoon air: physics-shaped source through the operator
    t = {"canopy_form": "saturating", "d_c": 15.0}
    q = (1.0 - albedo) * (0.4 + 0.6 * imp / 100.0) - 0.8 * canopy_shape(canopy, t) - 1.0 * ndvi - 0.6 * river
    q = q + 0.6 * hidden_air * H_air
    q = q - q.mean()
    phi = ops.solve(q, L, (0.0, 0.0), dx=dx)
    air = gain * phi - 0.006 * (elev - elev.mean())
    # surface at the morning overpass: linear in the cell's land cover (+ the hidden source)
    lst = (34.0 + 2.0 * mix[2] + 0.12 * imp - 0.06 * canopy - 20.0 * (albedo - 0.16) - 8.0 * ndvi
           + hidden_lst * H_lst + lst_noise_c * rng.standard_normal(shape))
    feats = {"canopy": canopy, "impervious": imp, "ndvi": ndvi, "albedo": albedo, "elevation": elev,
             "water_distance": water_dist}
    R = derived_rasters(feats, lst, dx)

    from sparc.core.identify.campaign import route_segments

    lay = SimpleNamespace(grid=grid.grid, col=lambda role: imp.ravel())
    segs = route_segments(lay, street_m=300.0, imp_min=25.0, rng=rng, min_len=4)
    cell = np.unique(np.concatenate(segs)) if segs else np.array([], dtype=np.int64)
    iy, ix = np.divmod(cell, n)
    Ta = 30.0 + air[iy, ix] + noise_c * rng.standard_normal(cell.size)
    slug = f"syn_{scenario}_{k}"
    tab = pd.DataFrame({"city": f"Synthetic {scenario} {k}", "slug": slug, "campaign_date": "2021-07-20", "run": "af",
                        "run_id": "2021-07-20 midday", "run_date": "2021-07-20", "cell": cell})
    x, y = grid.cell_xy(cell)
    tab["x"], tab["y"] = x, y
    tab["lat"], tab["lon"] = np.nan, np.nan
    tab["n_readings"], tab["n_visits"] = 3, 1
    tab["Ta"] = Ta
    tab["ta_background"] = float(np.median(Ta))
    tab["dTa"] = Ta - tab["ta_background"]
    tab["ta_era5"] = 29.0
    for name, r in R.items():
        tab[name] = r[iy, ix]
    tab["lst_scene_date"], tab["lst_offset_days"], tab["lst_clear_fraction"] = "2021-07-17", -3, 1.0
    driven = np.zeros(shape, bool)
    driven[iy, ix] = True
    meta = {"schema": 1, "status": "ok", "city": tab["city"].iloc[0], "slug": slug, "built_utc": f"synthetic-{seed}",
            "grid": grid.as_dict(), "synthetic": {"scenario": scenario, "seed": seed, "k": k},
            "runs": [{"run": "af", "run_id": "2021-07-20 midday", "era5": {"sw_down": 650.0, "lw_net": -90.0,
                                                                           "ta_era5": 29.0}}]}
    return CityData(slug=slug, meta=meta, table=tab, grid=grid, rasters={**R, "driven": driven})


def make_world(scenario: str, n_cities: int = 8, seed: int = 0, **kw) -> list[CityData]:
    return [make_city(k, scenario, seed=seed, **kw) for k in range(n_cities)]


def write_world(cities: list[CityData], root: str | Path) -> Path:
    """The cities in the dataset's cache layout under ``root/cities`` (for ``lstair evaluate --data root``)."""
    root = Path(root)
    for c in cities:
        d = root / "cities" / c.slug
        d.mkdir(parents=True, exist_ok=True)
        c.table.to_parquet(d / "table.parquet", index=False)
        np.savez_compressed(d / "rasters.npz", **c.rasters)
        (d / "meta.json").write_text(json.dumps(c.meta, indent=1), encoding="utf-8")
    return root


EXPECTED = {"informative": "B and C beat A (95% intervals below 0)", "redundant": "not shown (B − A and C − C0)",
            "mismatch": "degraded relative to informative (reported)"}


def validate(scenarios=SCENARIOS, n_cities: int = 8, seeds=(0,), settings=None, n: int = 64, out: str | Path | None = None
             ) -> dict:
    """Run the harness (:func:`sparc.core.lstair.evaluate.evaluate_window`) on every world; the verdicts against what
    each world requires, and the mismatch world's degradation (its mean B − A and C − A ΔMSE as a share of the
    informative world's, same seed)."""
    from sparc.core.lstair.evaluate import Settings, evaluate_window

    s = settings or Settings(physics_cell_m=30.0, physics_iter=40, within_city=False)
    rows = []
    t0 = time.perf_counter()
    for seed in seeds:
        for sc in scenarios:
            res, _ = evaluate_window(make_world(sc, n_cities, seed, n=n), "af", s)
            p = res.get("paired", {})
            row = {"scenario": sc, "seed": int(seed), "status": res.get("status"),
                   "B-A": {k: p.get("B-A", {}).get(k) for k in ("mean_diff_mse", "ci95", "verdict", "n_better")},
                   "C-A": {k: p.get("C-A", {}).get(k) for k in ("mean_diff_mse", "ci95", "verdict", "n_better")},
                   "C-C0": {k: p.get("C-C0", {}).get(k) for k in ("mean_diff_mse", "ci95", "verdict")},
                   "summary": res.get("summary"), "expected": EXPECTED[sc]}
            b, c = row["B-A"]["verdict"], row["C-A"]["verdict"]
            if sc == "informative":
                row["as_expected"] = b == "LST adds information beyond land cover" and c == "physics with LST beats land cover"
            elif sc == "redundant":
                row["as_expected"] = b == "not shown" and row["C-C0"]["verdict"] in ("not shown", None)
            rows.append(row)
    for seed in seeds:
        inf = next((r for r in rows if r["scenario"] == "informative" and r["seed"] == seed), None)
        mis = next((r for r in rows if r["scenario"] == "mismatch" and r["seed"] == seed), None)
        if inf and mis:
            # what LST adds: B over A, and C over the same physics without LST (C − A also holds the physics' own
            # advantage, which the overpass hour does not touch)
            deg = {}
            for k in ("B-A", "C-C0"):
                a, b = inf[k]["mean_diff_mse"], mis[k]["mean_diff_mse"]
                deg[k] = None if (a is None or b is None or a >= 0) else float(b / a)
            mis["share_of_informative_gain"] = deg
            mis["as_expected"] = all(v is not None and v < 1.0 for v in deg.values())
    out_d = {"scenarios": list(scenarios), "n_cities": n_cities, "seeds": list(seeds), "rows": rows,
             "all_as_expected": all(r.get("as_expected", True) for r in rows),
             "seconds": round(time.perf_counter() - t0, 1)}
    if out is not None:
        o = Path(out)
        o.mkdir(parents=True, exist_ok=True)
        (o / "validation.json").write_text(json.dumps(out_d, indent=1, default=str), encoding="utf-8")
        (o / "validation.md").write_text(validation_markdown(out_d), encoding="utf-8")
    return out_d


def validation_markdown(v: dict) -> str:
    L = ["# LST pilot: synthetic validation of the harness", "",
         f"{v['n_cities']} cities per world, seeds {v['seeds']}; leave-one-city-out, run af.", "",
         "| world | seed | B − A ΔMSE | 95% CI | B verdict | C − A ΔMSE | 95% CI | C verdict | C − C0 ΔMSE | 95% CI | "
         "expected | as expected |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def f(x):
        return "—" if x is None else f"{x:+.4f}"

    for r in v["rows"]:
        b, c = r["B-A"], r["C-A"]
        ci = (lambda p: "—" if not p.get("ci95") else f"{p['ci95'][0]:+.4f} to {p['ci95'][1]:+.4f}")
        extra = ""
        if "share_of_informative_gain" in r:
            sh = r["share_of_informative_gain"]
            extra = (f" (LST gain kept: B − A {sh['B-A']:.0%}, C − C0 {sh['C-C0']:.0%})" if None not in sh.values()
                     else "")
        c0 = r["C-C0"]
        L.append(f"| {r['scenario']} | {r['seed']} | {f(b['mean_diff_mse'])} | {ci(b)} | {b['verdict']} | "
                 f"{f(c['mean_diff_mse'])} | {ci(c)} | {c['verdict']} | {f(c0['mean_diff_mse'])} | {ci(c0)} | "
                 f"{r['expected']}{extra} | "
                 f"{'yes' if r.get('as_expected', True) else '**no**'} |")
    L += ["", f"All as expected: {'yes' if v['all_as_expected'] else '**no**'}.", ""]
    return "\n".join(L)


__all__ = ["SCENARIOS", "MISMATCH_RHO", "make_city", "make_world", "write_world", "validate", "validation_markdown"]
