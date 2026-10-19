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
* ``null`` — a true null: the air as in ``redundant``, and an "LST" field drawn independently of everything
  (smooth random fields with the LST's spatial scales and spread, plus pixel noise): a matched control column
  that carries no information, so adding it cannot truly help (E[ΔMSE] ≥ 0).  Neither B − A nor C − C0 may
  claim beyond the test's level.

**One draw proves little.**  :func:`validate` over many seeds reports, per world, the rate at which each
comparison makes its claim, with Clopper–Pearson 95% intervals: the false-claim rate in ``null`` and
``redundant`` (B − A) must be consistent with the one-sided level 0.025, and the power in ``informative`` is
reported (the harness's verdicts rest on these rates, not on a seed that happened to pass).

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

SCENARIOS = ("informative", "redundant", "mismatch", "null")
MISMATCH_RHO = 0.3
RATE_SEEDS = 10                # with this many seeds the verdict of validate() rests on rates, not single draws
MIN_POWER = 0.5                # floor on the informative world's claim rate in a rate run


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
    elif scenario in ("redundant", "null"):
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
    if scenario == "null":
        # a matched control: the spatial scales and spread of the LST, independent of land cover and air
        z = gaussian_random_field(shape, 5.0, rng) + 0.6 * gaussian_random_field(shape, 25.0, rng)
        z = (z - z.mean()) / (z.std() + 1e-12)
        lst = float(lst.mean()) + float(lst.std()) * z + lst_noise_c * rng.standard_normal(shape)
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


EXPECTED = {"informative": "B and C beat A (claims)", "redundant": "B − A not shown",
            "mismatch": "degraded relative to informative (reported)", "null": "not shown (B − A and C − C0)"}
CLAIMS = {"B-A": "LST adds information beyond land cover", "C-A": "physics with LST beats land cover",
          "C-C0": "LST adds information inside the physics"}


def _clopper_pearson(k: int, n: int, level: float = 0.95) -> list[float]:
    from scipy.stats import beta

    a = (1.0 - level) / 2.0
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - a, k + 1, n - k))
    return [lo, hi]


def claim_rates(rows: list[dict]) -> dict:
    """Per world and comparison: runs, claims, rate and its Clopper–Pearson 95% interval; for the null worlds
    (``null``; ``redundant`` for B − A) whether the claim count is consistent with the one-sided level
    (at most the binomial 99th percentile at rate 0.025)."""
    from scipy.stats import binom

    from sparc.core.lstair.evaluate import ALPHA

    out = {}
    for sc in dict.fromkeys(r["scenario"] for r in rows):
        rs = [r for r in rows if r["scenario"] == sc and r.get("status") == "done"]
        rec = {"n_runs": len(rs)}
        for k, claim in CLAIMS.items():
            got = [r[k]["verdict"] for r in rs if r[k].get("verdict") is not None]
            n, c = len(got), sum(v == claim for v in got)
            if not n:
                continue
            rec[k] = {"n": n, "claims": c, "rate": c / n, "ci95": _clopper_pearson(c, n)}
            if (sc == "null") or (sc == "redundant" and k == "B-A"):
                limit = int(binom.ppf(0.99, n, ALPHA))
                rec[k].update(level=ALPHA, max_claims_at_level=limit, calibrated=c <= limit)
        out[sc] = rec
    return out


def validate(scenarios=SCENARIOS, n_cities: int = 8, seeds=(0,), settings=None, n: int = 64, out: str | Path | None = None
             ) -> dict:
    """Run the harness (:func:`sparc.core.lstair.evaluate.evaluate_window`) on every world and seed; each row's
    verdicts against what its world requires, the mismatch world's degradation (its mean B − A and C − C0 ΔMSE as
    a share of the informative world's, same seed), and the claim rates over seeds (:func:`claim_rates`).

    With fewer than :data:`RATE_SEEDS` seeds ``all_as_expected`` is the rows' verdicts (single draws: an
    illustration, not a calibration).  With more it rests on the rates: the null worlds' false-claim counts
    consistent with the level, the informative world's B − A and C − A power at least :data:`MIN_POWER`, and the
    mismatch world keeping less than the informative world's gain on average."""
    from sparc.core.lstair.evaluate import Settings, evaluate_window

    s = settings or Settings(physics_cell_m=30.0, physics_iter=40, within_city=False)
    rows = []
    t0 = time.perf_counter()
    for seed in seeds:
        for sc in scenarios:
            res, _ = evaluate_window(make_world(sc, n_cities, seed, n=n), "af", s)
            p = res.get("paired", {})
            row = {"scenario": sc, "seed": int(seed), "status": res.get("status"),
                   **{k: {q: p.get(k, {}).get(q) for q in ("mean_diff_mse", "rel_change", "ci95", "p_sign_flip",
                                                            "verdict", "n_better")} for k in CLAIMS},
                   "summary": res.get("summary"), "expected": EXPECTED[sc]}
            b, c = row["B-A"]["verdict"], row["C-A"]["verdict"]
            if sc == "informative":
                row["as_expected"] = b == "LST adds information beyond land cover" and c == "physics with LST beats land cover"
            elif sc == "redundant":
                # C − C0 is descriptive here: an LST that is a function of land cover can stand in for what the
                # physics' fixed land-cover source form misses (evaluate's docstring); B − A is the claim's test
                row["as_expected"] = b == "not shown"
            elif sc == "null":
                row["as_expected"] = b != CLAIMS["B-A"] and row["C-C0"]["verdict"] != CLAIMS["C-C0"]
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
    rates = claim_rates(rows)
    rate_mode = len(list(seeds)) >= RATE_SEEDS
    checks = {}
    if rate_mode:
        for sc, rec in rates.items():
            for k, v in rec.items():
                if isinstance(v, dict) and "calibrated" in v:
                    checks[f"{sc} {k} false-claim rate ≤ level"] = bool(v["calibrated"])
        inf = rates.get("informative", {})
        for k in ("B-A", "C-A"):
            if k in inf:
                checks[f"informative {k} power ≥ {MIN_POWER}"] = inf[k]["rate"] >= MIN_POWER
        shares = [r["share_of_informative_gain"]["B-A"] for r in rows if r.get("share_of_informative_gain")
                  and r["share_of_informative_gain"].get("B-A") is not None]
        if shares:
            checks["mismatch keeps less of the B − A gain (mean share < 1)"] = float(np.mean(shares)) < 1.0
        ok = all(checks.values())
    else:
        ok = all(r.get("as_expected", True) for r in rows)
    out_d = {"scenarios": list(scenarios), "n_cities": n_cities, "seeds": list(seeds), "rows": rows, "rates": rates,
             "mode": "rates over seeds" if rate_mode else f"single draws (< {RATE_SEEDS} seeds: an illustration)",
             "checks": checks, "all_as_expected": ok, "seconds": round(time.perf_counter() - t0, 1)}
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

    def ci(p):
        return "—" if not p.get("ci95") else f"{p['ci95'][0]:+.4f} to {p['ci95'][1]:+.4f}"

    for r in v["rows"]:
        b, c = r["B-A"], r["C-A"]
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
    rates = v.get("rates") or {}
    if rates:
        L += ["", f"Claim rates over {len(v['seeds'])} seeds (Clopper–Pearson 95% intervals; mode: {v.get('mode')}):", "",
              "| world | comparison | runs | claims | rate | 95% CI | consistent with the 0.025 level |", "|---|---|---|---|---|---|---|"]
        for sc, rec in rates.items():
            for k in CLAIMS:
                if k in rec:
                    x = rec[k]
                    cal = "—" if "calibrated" not in x else ("yes" if x["calibrated"] else "**no**")
                    L.append(f"| {sc} | {k} | {x['n']} | {x['claims']} | {x['rate']:.2f} | {x['ci95'][0]:.2f}–"
                             f"{x['ci95'][1]:.2f} | {cal} |")
    for name, good in (v.get("checks") or {}).items():
        L.append(f"* {name}: {'yes' if good else '**no**'}")
    L += ["", f"All as expected: {'yes' if v['all_as_expected'] else '**no**'}.", ""]
    return "\n".join(L)


__all__ = ["SCENARIOS", "MISMATCH_RHO", "make_city", "make_world", "write_world", "validate", "validation_markdown",
           "claim_rates", "CLAIMS", "RATE_SEEDS"]
