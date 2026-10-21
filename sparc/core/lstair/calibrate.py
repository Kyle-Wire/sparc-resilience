"""Study S1 of ``docs/results/lst_air_pilot.md``: calibrating a held-out city's map with local stations.

**If** a city has a few point measurements of air temperature at the hour, **then** spreading their residuals
makes the held-out map better.  This is regression kriging (Hengl, Heuvelink & Rossiter 2007): a regression
(here a held-out model of :mod:`~sparc.core.lstair.evaluate`, base A by default) plus simple kriging of its
residuals.  The primary test showed LST is no substitute for air temperature in a new city (af B − A ΔSS
+0.006, 95% CI −0.006 to +0.018, Holm p 0.78; A's mean skill 0.150 over 27 units), so the question here is
whether a handful of local air measurements does what LST could not.

Input
-----
A finished ``lstair evaluate`` run: ``predictions.parquet`` holds each held-out city's scored cells (city, slug,
group = the held-out unit, run = the window, run_id, cell, x, y, dTa and ``pred_A`` … ``pred_C0``), predicted by
models that never saw that city's labels; ``dTa`` and every prediction are already centred to median 0 on the
run's scored cells.  Scoring uncalibrated on all cells reproduces the harness's per-unit skill exactly (checked
against ``results.json``; :func:`harness_check`).

Residual covariance
-------------------
For held-out unit u and window w, an **exponential semivariogram** γ(h) = c₀ + c₁·(1 − e^(−h/a)) (nugget c₀,
partial sill c₁, range a; practical range 3a) is fitted to the residuals dTa − pred of **every other unit** of
that window, never to u's (the registered rule: the covariance comes from the training cities).  Each run's
residuals are centred and divided by their SD, so that a city's variance does not set the pooled shape (Los
Angeles's afternoon dTa SD is 3.5 °C against 0.3–1.1 °C elsewhere; :func:`sparc.core.lstair.evaluate._weights`
makes the same choice for training).  Simple-kriging weights depend only on the nugget share c₀/(c₀ + c₁) and
the range, not on the sill's scale, so standardising changes nothing else.  Pairs are taken within a run only
(the run's cells, at most :data:`VARIO_CELLS` of them sampled per run, seeded), binned to 5 km
(:data:`BIN_EDGES_M`), and the model is fitted by Cressie's (1985) weighted least squares, Σ N_h (γ̂_h/γ(h) − 1)².
Per-run bin sums are computed once and the leave-one-unit-out pool sums the other units' runs only
(:func:`unit_variograms`).

Simulated stations (primary)
----------------------------
Per run of u, k ∈ {3, 10, 30} driven cells are taken as perfect stations: random sequential placement with at
least ``min_sep_m`` (1 km) between stations (:func:`draw_stations`), ``draws`` (20) draws seeded by (seed, k, run)
so that every base sees the same stations.  The test cells are the driven cells farther than ``buffer_m``
(500 m) from every station; a draw with fewer than :data:`~sparc.core.lstair.evaluate.MIN_CELLS` (30) is
skipped, as is one where k stations cannot be placed.  The station residuals (observed minus predicted
anomaly, minus their mean) are interpolated to the test cells by simple kriging with mean zero (Cressie 1993,
§3.4.5; the nugget is treated as noise, so the field is the smooth part); calibrated = pred + kriged residual.
Skill is scored exactly as the harness scores it, on the test cells of the draw: target and predictions centred
to median 0 on those cells (:func:`~sparc.core.lstair.evaluate.centre`), SS = 1 − MSE/MSE_N
(:func:`~sparc.core.lstair.evaluate.skill`), the uncalibrated base on the **same** cells.  Per run the mean over
valid draws, per unit the mean over its runs (as :func:`~sparc.core.lstair.evaluate.score_units`; a run with no
valid draw for k does not enter k).  ΔSS = SS(base calibrated) − SS(base) per unit, paired over units by
:func:`~sparc.core.lstair.evaluate.paired_skill` (one-sided sign-flip, 95% t-interval) and Holm's correction over
the k values within each (base, window) (:func:`~sparc.core.lstair.evaluate.holm`); the claim rule is the
harness's (:func:`~sparc.core.lstair.evaluate.skill_verdict`: Holm p ≤ 0.025, CI above 0, mean ΔSS ≥ 0.05, mean
SS of the calibrated map above 0, at least 8 units).  **Primary: base A, window af.**  Other bases and windows
are secondary.

Real stations (secondary)
-------------------------
NOAA Global Hourly (ISD) stations inside the city grid whose period of record (``isd-history.csv``, the station
list of the ``noaa-isd-pds`` bucket, else NCEI) covers the run's date; each one's reports in the run's hours
(±30 min: hourly METARs come at about :51) are read with :func:`sparc.core.forcing.station_obs` from
``s3://noaa-global-hourly-pds`` and the station value is their mean temperature.  A station is usable with a
finite temperature and a driven cell within ``near_m`` (1 km); stations within :data:`DUP_M` of each other are one
site (the station list repeats sites under several USAF/WBAN ids; the one with more reports is kept).  With at
least 2 usable stations the residuals (T_s − mean over stations) − (pred at the nearest driven cell − mean over
those) are kriged with the same leave-one-unit-out covariance and the run is scored on **all driven cells** (the
registered rule) and, as a variant, on the driven cells farther than ``buffer_m`` from every station.  Airport
sites stand for open ground, not streets, so this tests the network a city already has.  Per city the result
lists how many stations were found in the grid, had observations and were usable.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import logging
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from sparc.core import progress
from sparc.core.lstair import to_json
from sparc.core.lstair.evaluate import (ALPHA, MIN_CELLS, MIN_DSS, MIN_VERDICT_CITIES, PRIMARY_WINDOW, _ci, _f, _p,
                                        _testable, centre, holm, paired_skill, skill, skill_verdict)

log = logging.getLogger(__name__)

SCHEMA = 1
BASES = ("A", "B", "C")
KS = (3, 10, 30)
#: Semivariogram bins (m): fine where a 30 m grid's neighbours sit, coarser to 5 km.
BIN_EDGES_M = (0.0, 50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0, 700.0, 1000.0, 1400.0, 2000.0, 2800.0, 3800.0,
               5000.0)
VARIO_CELLS = 3000              # cells sampled per run for the variogram's pairs (all pairs within 5 km of them)
MIN_BIN_PAIRS = 30              # a bin with fewer pairs is not fitted
RANGE_BOUNDS_M = (10.0, 50000.0)
DUP_M = 100.0                   # real stations this close are one site
ISD_HISTORY = ("https://noaa-isd-pds.s3.amazonaws.com/isd-history.csv",
               "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv")
CLAIM = "local stations make the held-out map better"


@dataclass
class CalSettings:
    ks: tuple[int, ...] = KS
    draws: int = 20
    min_sep_m: float = 1000.0
    buffer_m: float = 500.0
    windows: tuple[str, ...] = ("af", "am", "pm")
    bases: tuple[str, ...] = BASES
    seed: int = 0
    boot: int = 4000                       # random sign vectors when the sign-flip test cannot be enumerated
    stations: bool = True                  # the real-station (ISD) path
    vario_max_m: float = 5000.0
    vario_cells: int = VARIO_CELLS
    near_m: float = 1000.0                 # a real station needs a driven cell this close
    min_stations: int = 2


# --------------------------------------------------------------------------- #
# Residual covariance                                                          #
# --------------------------------------------------------------------------- #
def _edges(max_m: float) -> np.ndarray:
    e = [x for x in BIN_EDGES_M if x < max_m]
    return np.asarray(e + [float(max_m)], float)


def standardised_residual(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """One run's residual y − p, centred and divided by its SD (all zeros for a perfect prediction)."""
    r = np.asarray(y, float) - np.asarray(p, float)
    r = r - r.mean()
    sd = r.std()
    return r / sd if sd > 0 else np.zeros_like(r)


def pair_stats(xy: np.ndarray, resid: dict[str, np.ndarray], max_m: float = 5000.0, max_cells: int = VARIO_CELLS,
               rng: np.random.Generator | None = None) -> dict:
    """Binned semivariogram sums of one run: per bin the pair count ``n``, the summed distance ``h`` and, per key of
    ``resid``, the summed squared difference.  Pairs are all pairs closer than ``max_m`` among at most
    ``max_cells`` cells sampled from the run."""
    from scipy.spatial import cKDTree

    xy = np.asarray(xy, float)
    edges = _edges(max_m)
    nb = edges.size - 1
    idx = np.arange(len(xy))
    if len(xy) > max_cells:
        idx = np.sort((rng or np.random.default_rng(0)).choice(len(xy), max_cells, replace=False))
    pairs = cKDTree(xy[idx]).query_pairs(max_m, output_type="ndarray") if idx.size > 1 else np.zeros((0, 2), int)
    i, j = idx[pairs[:, 0]], idx[pairs[:, 1]]
    h = np.hypot(*(xy[i] - xy[j]).T) if len(i) else np.zeros(0)
    b = np.searchsorted(edges, h, side="right") - 1
    keep = (h > 0) & (b >= 0) & (b < nb)
    b, i, j, h = b[keep], i[keep], j[keep], h[keep]
    out = {"n": np.bincount(b, minlength=nb).astype(float), "h": np.bincount(b, weights=h, minlength=nb)}
    for k, r in resid.items():
        out[k] = np.bincount(b, weights=(r[i] - r[j]) ** 2, minlength=nb)
    return out


def fit_variogram(n: np.ndarray, h_sum: np.ndarray, sq_sum: np.ndarray, min_pairs: int = MIN_BIN_PAIRS) -> dict:
    """Exponential semivariogram γ(h) = c₀ + c₁(1 − e^(−h/a)) fitted to binned sums by Cressie's (1985) weighted
    least squares, Σ N_h (γ̂_h/γ(h) − 1)², from three starting ranges (best cost kept)."""
    from scipy.optimize import least_squares

    n, h_sum, sq_sum = (np.asarray(v, float) for v in (n, h_sum, sq_sum))
    ok = n >= min_pairs
    base = {"n_pairs": int(n.sum()), "n_bins": int(ok.sum())}
    if ok.sum() < 3:
        return {**base, "status": "too few pairs", "nugget": 0.0, "psill": 0.0, "range_m": float("nan"),
                "practical_range_m": float("nan"), "nugget_share": float("nan")}
    h = h_sum[ok] / n[ok]
    g = sq_sum[ok] / (2.0 * n[ok])
    w = n[ok] / n[ok].sum()
    if g.max() <= 0:
        return {**base, "status": "no residual", "nugget": 0.0, "psill": 0.0, "range_m": float("nan"),
                "practical_range_m": float("nan"), "nugget_share": float("nan")}
    floor = 1e-9 * g.max()

    def resid(th):
        c0, c1, a = th
        return np.sqrt(w) * (g / np.maximum(c0 + c1 * (1.0 - np.exp(-h / a)), floor) - 1.0)

    lo, hi = [0.0, 0.0, RANGE_BOUNDS_M[0]], [np.inf, np.inf, RANGE_BOUNDS_M[1]]
    best = None
    for a0 in (200.0, 1000.0, 3000.0):
        x0 = [max(0.5 * g[0], floor), max(g.max() - 0.5 * g[0], floor), a0]
        r = least_squares(resid, x0, bounds=(lo, hi), x_scale=[g.max(), g.max(), a0])
        if best is None or r.cost < best.cost:
            best = r
    c0, c1, a = (float(v) for v in best.x)
    return {**base, "status": "ok", "nugget": c0, "psill": c1, "range_m": a, "practical_range_m": 3.0 * a,
            "nugget_share": c0 / (c0 + c1) if c0 + c1 > 0 else float("nan"),
            "bins": {"h_m": h.tolist(), "gamma": g.tolist(), "n": n[ok].astype(int).tolist()}}


def run_key(window: str, slug: str, run_id: str) -> int:
    return zlib.crc32(f"{window}|{slug}|{run_id}".encode("utf-8"))


def window_pair_stats(df: pd.DataFrame, bases, s: CalSettings) -> list[dict]:
    """Per run of one window's prediction table: its unit and its binned pair sums per base."""
    out = []
    for (sl, rid), part in df.groupby(["slug", "run_id"], sort=True):
        progress.check_cancel()
        y = part["dTa"].to_numpy(float)
        resid = {b: standardised_residual(y, part[f"pred_{b}"].to_numpy(float)) for b in bases}
        rng = np.random.default_rng([s.seed, run_key(str(part["run"].iat[0]), sl, rid)])
        st = pair_stats(part[["x", "y"]].to_numpy(float), resid, s.vario_max_m, s.vario_cells, rng)
        out.append({"slug": sl, "run_id": rid, "group": str(part["group"].iat[0]), "stats": st})
    return out


def unit_variograms(stats: list[dict], bases, units=None) -> dict:
    """base → unit → the variogram fitted to every run whose unit is not that unit (leave-one-unit-out), plus
    ``"_all"``: the pool of every unit (descriptive only, never used to krige)."""
    units = sorted({r["group"] for r in stats}) if units is None else list(units)
    nb = len(stats[0]["stats"]["n"]) if stats else 0

    def pool(keep) -> dict:                 # summed over the kept runs only: a left-out unit's sums never enter
        tot = {k: np.zeros(nb) for k in ["n", "h", *bases]}
        for r in stats:
            if keep(r["group"]):
                for k in tot:
                    tot[k] += r["stats"][k]
        return tot

    pools = {u: pool(lambda g, u=u: g != u) for u in units}
    pools["_all"] = pool(lambda g: True)
    return {b: {u: fit_variogram(p["n"], p["h"], p[b]) for u, p in pools.items()} for b in bases}


# --------------------------------------------------------------------------- #
# Kriging and simulated stations                                               #
# --------------------------------------------------------------------------- #
def exp_cov(h: np.ndarray, v: dict) -> np.ndarray:
    """The exponential covariance of the smooth part, c₁·e^(−h/a) (the nugget sits on the station diagonal)."""
    return v["psill"] * np.exp(-np.asarray(h, float) / v["range_m"])


def simple_krige(xy_s: np.ndarray, r_s: np.ndarray, xy_t: np.ndarray, v: dict) -> np.ndarray:
    """Simple kriging (known mean 0) of station residuals ``r_s`` to the points ``xy_t``: λ = (C + c₀I)⁻¹c,
    r̂ = λᵀr.  A variogram without a structured part (c₁ = 0, or not fitted) gives 0 everywhere."""
    from scipy.spatial.distance import cdist

    xy_s, xy_t, r_s = np.asarray(xy_s, float), np.asarray(xy_t, float), np.asarray(r_s, float)
    if not v or v.get("status") != "ok" or not v["psill"] > 0 or not np.isfinite(v["range_m"]):
        return np.zeros(len(xy_t))
    C = exp_cov(cdist(xy_s, xy_s), v) + (v["nugget"] + 1e-9 * v["psill"]) * np.eye(len(xy_s))
    lam = np.linalg.solve(C, exp_cov(cdist(xy_s, xy_t), v))
    return lam.T @ r_s


def draw_stations(xy: np.ndarray, k: int, min_sep_m: float, rng: np.random.Generator, tree=None) -> np.ndarray | None:
    """k cell indices, at least ``min_sep_m`` apart: cells in a random order, each kept if no kept one is closer
    (random sequential placement).  None when fewer than k fit.  ``tree``: a ``cKDTree`` of ``xy`` to reuse."""
    from scipy.spatial import cKDTree

    xy = np.asarray(xy, float)
    tree = cKDTree(xy) if tree is None else tree
    alive = np.ones(len(xy), bool)
    chosen = []
    for i in rng.permutation(len(xy)):
        if not alive[i]:
            continue
        chosen.append(int(i))
        if len(chosen) == k:
            return np.asarray(chosen)
        alive[tree.query_ball_point(xy[i], r=min_sep_m * (1 - 1e-12))] = False
    return None


def far_from(xy: np.ndarray, xy_s: np.ndarray, buffer_m: float) -> np.ndarray:
    """Cells farther than ``buffer_m`` from every station."""
    from scipy.spatial import cKDTree

    return cKDTree(np.asarray(xy_s, float)).query(np.asarray(xy, float))[0] > buffer_m


def _score(y: np.ndarray, p: np.ndarray, kr: np.ndarray) -> tuple[float, float]:
    """(SS of p, SS of p + kr) on one run's cells, target and predictions median-centred on them (the harness)."""
    g = np.zeros(len(y), int)
    yc = centre(y, g)
    return skill(yc, centre(p, g)), skill(yc, centre(p + kr, g))


def calibrate_run(xy: np.ndarray, y: np.ndarray, preds: dict[str, np.ndarray], vgs: dict[str, dict],
                  s: CalSettings, key: int) -> dict:
    """One run, every k: per valid draw the base's and the calibrated skill on the draw's test cells (per base),
    and the draws skipped (k stations do not fit; fewer than MIN_CELLS test cells)."""
    from scipy.spatial import cKDTree

    tree = cKDTree(np.asarray(xy, float))
    out = {}
    for k in s.ks:
        rng = np.random.default_rng([s.seed, int(k), key])
        rec = {"n_valid": 0, "skipped_no_room": 0, "skipped_few_cells": 0, "n_test": [],
               **{b: {"ss": [], "ss_cal": []} for b in preds}}
        for _ in range(s.draws):
            st = draw_stations(xy, k, s.min_sep_m, rng, tree)
            if st is None:
                rec["skipped_no_room"] += 1
                continue
            far = far_from(xy, xy[st], s.buffer_m)
            if far.sum() < MIN_CELLS:
                rec["skipped_few_cells"] += 1
                continue
            row = {}
            for b, p in preds.items():
                r = y[st] - p[st]
                kr = simple_krige(xy[st], r - r.mean(), xy[far], vgs[b])
                row[b] = _score(y[far], p[far], kr)
            if not all(np.isfinite(v).all() for v in row.values()):
                rec["skipped_few_cells"] += 1      # constant target on the test cells: no skill defined
                continue
            rec["n_valid"] += 1
            rec["n_test"].append(int(far.sum()))
            for b, (a, c) in row.items():
                rec[b]["ss"].append(a)
                rec[b]["ss_cal"].append(c)
        out[int(k)] = rec
    return out


def compare(units: list[str], ss_cal: np.ndarray, ss_base: np.ndarray, base: str, label: str, s: CalSettings) -> dict:
    """:func:`~sparc.core.lstair.evaluate.paired_skill` of the calibrated against the uncalibrated base, relabelled
    for this study (the statistics are the harness's; only the names and the claim change), with the unadjusted
    verdict."""
    rec = paired_skill(ss_cal, ss_base, units, boot=s.boot, seed=s.seed, key="B-A")
    x = f"{base}+cal"
    rec.update(comparison=f"{x} − {base} ({label})", key=label, claim=CLAIM, x=x, base=base,
               statistic=f"ΔSS = SS({base} calibrated) − SS({base}) per held-out unit, on the same test cells "
                         "(SS = 1 − MSE/MSE_N, mean over draws, then over the unit's runs)")
    rec.update(skill_verdict(rec, rec.get("p_sign_flip")))
    return rec


def _apply_holm(recs: dict) -> None:
    """Holm over the k values of one (base, window): a k not testable counts with p = 1."""
    fam = {k: float(r["p_sign_flip"]) if _testable(r) else 1.0 for k, r in recs.items()}
    adj = holm(fam)
    for k, r in recs.items():
        r["p_holm"] = adj[k]
        r["holm"] = {"m": len(fam), "family": [f"k={q}" for q in fam],
                     "untested_as_p1": [f"k={q}" for q in fam if not _testable(recs[q])]}
        if r.get("units"):
            r.update(skill_verdict(r, r["p_holm"], adjusted=True))


def simulate_window(df: pd.DataFrame, window: str, vgs: dict, s: CalSettings) -> dict:
    """Simulated stations on one window's prediction table: base → {"k": {k: paired record}}."""
    per_run = []
    for (sl, rid), part in df.groupby(["slug", "run_id"], sort=True):
        progress.check_cancel()
        g = str(part["group"].iat[0])
        xy = part[["x", "y"]].to_numpy(float)
        preds = {b: part[f"pred_{b}"].to_numpy(float) for b in s.bases}
        res = calibrate_run(xy, part["dTa"].to_numpy(float), preds, {b: vgs[b][g] for b in s.bases}, s,
                            run_key(window, sl, rid))
        per_run.append({"slug": sl, "run_id": rid, "group": g, "n": int(len(part)), "k": res})
    out = {}
    for b in s.bases:
        recs = {}
        for k in s.ks:
            units, ss0, ss1, rows = [], [], [], []
            for g in sorted({r["group"] for r in per_run}):
                rr = [r for r in per_run if r["group"] == g and r["k"][k]["n_valid"] > 0]
                if not rr:
                    continue
                a = float(np.mean([np.mean(r["k"][k][b]["ss"]) for r in rr]))
                c = float(np.mean([np.mean(r["k"][k][b]["ss_cal"]) for r in rr]))
                units.append(g)
                ss0.append(a)
                ss1.append(c)
                rows.append({"unit": g, "n_runs": len(rr), "n_draws": int(sum(r["k"][k]["n_valid"] for r in rr)),
                             "median_test_cells": float(np.median(sum((r["k"][k]["n_test"] for r in rr), [])))})
            rec = compare(units, np.asarray(ss1), np.asarray(ss0), b, f"k={k}", s)
            rec.update(k=int(k), window=window, n_runs_total=len(per_run),
                       n_runs_scored=int(sum(r["k"][k]["n_valid"] > 0 for r in per_run)),
                       n_draws_valid=int(sum(r["k"][k]["n_valid"] for r in per_run)),
                       n_draws_skipped_no_room=int(sum(r["k"][k]["skipped_no_room"] for r in per_run)),
                       n_draws_skipped_few_cells=int(sum(r["k"][k]["skipped_few_cells"] for r in per_run)),
                       unit_draws=rows)
            recs[int(k)] = rec
        _apply_holm(recs)
        out[b] = {"k": recs}
    return out


def harness_check(df: pd.DataFrame, per_unit: list[dict] | None, bases) -> dict:
    """The uncalibrated skill on all scored cells, per unit, against the harness's ``results.json`` per_unit:
    the largest absolute difference per base (0 up to rounding when the scoring is the harness's)."""
    out = {}
    if not per_unit:
        return {"status": "no per-unit results in results.json"}
    ref = {u["unit"]: u for u in per_unit}
    for b in bases:
        ss = {}
        for (sl, rid), part in df.groupby(["slug", "run_id"], sort=True):
            g = np.zeros(len(part), int)
            ss.setdefault(str(part["group"].iat[0]), []).append(
                skill(centre(part["dTa"].to_numpy(float), g), centre(part[f"pred_{b}"].to_numpy(float), g)))
        d = [abs(float(np.mean(v)) - ref[u][b]["ss"]) for u, v in ss.items() if u in ref and b in ref[u]]
        out[b] = {"n_units": len(d), "max_abs_diff": float(max(d)) if d else None}
    return out


# --------------------------------------------------------------------------- #
# Real stations (NOAA Global Hourly / ISD)                                     #
# --------------------------------------------------------------------------- #
def parse_isd_history(raw: bytes) -> pd.DataFrame:
    """``isd-history.csv`` → station (USAF+WBAN, the Global Hourly file name), name, icao, lat, lon, begin, end
    (yyyymmdd); rows without a position (or at 0, 0) dropped."""
    h = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
    out = pd.DataFrame({"station": h["USAF"].str.strip() + h["WBAN"].str.strip(), "name": h["STATION NAME"].str.strip(),
                        "icao": h["ICAO"].str.strip(), "lat": pd.to_numeric(h["LAT"], errors="coerce"),
                        "lon": pd.to_numeric(h["LON"], errors="coerce"),
                        "begin": pd.to_numeric(h["BEGIN"], errors="coerce"),
                        "end": pd.to_numeric(h["END"], errors="coerce")})
    out = out.dropna(subset=["lat", "lon", "begin", "end"])
    return out.loc[~((out["lat"] == 0) & (out["lon"] == 0))].reset_index(drop=True)


def isd_history(fetch: Callable[[str], bytes | None] | None = None, timeout: float = 30.0) -> tuple[pd.DataFrame, str]:
    """The ISD station list and the URL it came from (the noaa-isd-pds bucket, else NCEI).  Raises
    ``ConnectionError`` naming every failure when neither answers."""
    from sparc.core.climate import http_fetch

    errors = []
    for url in ISD_HISTORY:
        try:
            raw = fetch(url) if fetch is not None else http_fetch(url, timeout=timeout, retries=1)
        except Exception as exc:  # noqa: BLE001 - every source is tried, the failures are reported together
            errors.append(f"{url}: {exc!r}"[:300])
            continue
        if raw:
            return parse_isd_history(raw), url
        errors.append(f"{url}: not found")
    raise ConnectionError("; ".join(errors))


def grid_stations(history: pd.DataFrame, grid, date: dt.date) -> pd.DataFrame:
    """Stations of ``history`` inside the city grid (cell-edge rectangle, in the grid's CRS) whose period of
    record covers ``date``; with their grid coordinates x, y."""
    from pyproj import Transformer

    day = int(date.strftime("%Y%m%d"))
    lo0, la0, lo1, la1 = grid.bounds_lonlat(0.05)
    h = history.loc[(history["begin"] <= day) & (history["end"] >= day) & history["lon"].between(lo0, lo1)
                    & history["lat"].between(la0, la1)].copy()
    if h.empty:
        return h.assign(x=[], y=[])
    x, y = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True).transform(h["lon"].to_numpy(float),
                                                                                   h["lat"].to_numpy(float))
    h["x"], h["y"] = np.asarray(x, float), np.asarray(y, float)
    half = 0.5 * grid.dx
    inside = (h["x"].between(grid.x0 - half, grid.x0 + (grid.nx - 0.5) * grid.dx)
              & h["y"].between(grid.y0 - half, grid.y0 + (grid.ny - 0.5) * grid.dx))
    return h.loc[inside].reset_index(drop=True)


def run_hours_utc(meta: dict, run_id: str) -> tuple[dt.datetime, dt.datetime] | None:
    """A run's start and end (the dataset's local times in the city's time zone) in UTC."""
    for r in meta.get("runs") or []:
        if r.get("run_id") == run_id and r.get("start") and r.get("end"):
            tz = meta.get("tz") or "UTC"
            a, b = (pd.Timestamp(r[k]).tz_localize(tz).tz_convert("UTC").to_pydatetime() for k in ("start", "end"))
            return a, b
    return None


def default_reader(cache_dir: str | Path | None = None) -> Callable:
    """:func:`sparc.core.forcing.station_obs` with a per-process memo of the station-year files (one download
    serves a city's three windows)."""
    from sparc.core.climate import http_fetch
    from sparc.core.forcing import station_obs

    memo: dict[str, bytes | None] = {}

    def fetch(url: str) -> bytes | None:
        if url not in memo:
            if len(memo) > 40:                      # ~8 MB a file: bounded memory over a long run
                memo.clear()
            memo[url] = http_fetch(url, timeout=60.0, retries=2)
        return memo[url]

    def read(station: str, start_utc: dt.datetime, end_utc: dt.datetime) -> dict:
        return station_obs(station, start_utc, end_utc, fetch=fetch, cache_dir=cache_dir)

    return read


def station_readings(city, df_city: pd.DataFrame, history: pd.DataFrame, reader: Callable, s: CalSettings) -> dict:
    """One city: its stations in the grid and, per run of ``df_city`` (its rows of the prediction table), each
    station's mean temperature over the run's hours and whether it is usable."""
    from pyproj import Transformer
    from scipy.spatial import cKDTree

    meta, grid = city.meta, city.grid
    out = {"slug": city.slug, "city": meta.get("city"), "runs": [], "stations": {}}
    to_xy = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    by_date: dict[dt.date, pd.DataFrame] = {}
    for (w, rid), part in df_city.groupby(["run", "run_id"], sort=True):
        progress.check_cancel()
        hrs = run_hours_utc(meta, rid)
        rec = {"window": w, "run_id": rid, "group": str(part["group"].iat[0]), "readings": []}
        if hrs is None:
            rec["reason"] = "run hours unknown"
            out["runs"].append(rec)
            continue
        rec["hours_utc"] = [hrs[0].isoformat(), hrs[1].isoformat()]
        day = pd.Timestamp(hrs[0]).tz_convert(meta.get("tz") or "UTC").date()
        if day not in by_date:
            by_date[day] = grid_stations(history, grid, day)
        st = by_date[day]
        rec["n_in_grid"] = int(len(st))
        out["stations"].update({r.station: {"name": r.name, "icao": r.icao or None} for r in st.itertuples()})
        xy_c = part[["x", "y"]].to_numpy(float)
        cells = part["cell"].to_numpy()
        tree = cKDTree(xy_c)
        for r in st.itertuples():
            row = {"station": r.station, "x": float(r.x), "y": float(r.y)}
            try:
                o = reader(r.station, *hrs)
            except Exception as exc:  # noqa: BLE001 - a missing file or window is a count, not a failure
                row.update(status="no observations", error=repr(exc)[:200])
                rec["readings"].append(row)
                continue
            lat, lon = o.get("lat"), o.get("lon")
            if lat is not None and lon is not None and np.isfinite([lat, lon]).all():
                x, y = to_xy.transform(float(lon), float(lat))
                row.update(x=float(x), y=float(y))
            row.update(t_C=o.get("t_C"), n_reports=o.get("n_reports"))
            d, i = tree.query([row["x"], row["y"]])
            row.update(nearest_cell_m=float(d), nearest_cell=int(cells[i]))
            if row["t_C"] is None or not np.isfinite(row["t_C"]):
                row["status"] = "no temperature"
            elif d > s.near_m:
                row["status"] = f"no driven cell within {s.near_m:.0f} m"
            else:
                row["status"] = "usable"
            rec["readings"].append(row)
        use = sorted((q for q in rec["readings"] if q["status"] == "usable"), key=lambda q: -(q["n_reports"] or 0))
        kept = []
        for q in use:                                   # one site under several ids: the most reports kept
            if all(np.hypot(q["x"] - k["x"], q["y"] - k["y"]) > DUP_M for k in kept):
                kept.append(q)
            else:
                q["status"] = "duplicate site"
        rec.update(n_with_obs=int(sum(q["status"] not in ("no observations", "no temperature") for q in rec["readings"])),
                   n_usable=len(kept))
        out["runs"].append(rec)
    out["n_in_grid"] = len(out["stations"])
    return out


def calibrate_real(df: pd.DataFrame, readings: list[dict], vgs: dict, s: CalSettings) -> dict:
    """Score the real-station calibration: per base and window, units with at least one run of ≥ ``min_stations``
    usable stations, on all driven cells (registered) and on the cells farther than ``buffer_m`` (variant)."""
    runs = {(c["slug"], r["window"], r["run_id"]): r for c in readings for r in c["runs"]}
    per_run = []
    for (w, sl, rid), part in df.groupby(["run", "slug", "run_id"], sort=True):
        r = runs.get((sl, w, rid))
        if r is None:
            continue
        use = [q for q in r["readings"] if q["status"] == "usable"]
        r["scored"] = len(use) >= s.min_stations
        if not r["scored"]:
            r.setdefault("reason", f"{len(use)} usable station(s) (< {s.min_stations})")
            continue
        g = str(part["group"].iat[0])
        xy = part[["x", "y"]].to_numpy(float)
        y = part["dTa"].to_numpy(float)
        xy_s = np.array([[q["x"], q["y"]] for q in use])
        t = np.array([q["t_C"] for q in use], float)
        pos = pd.Series(np.arange(len(part)), index=part["cell"].to_numpy())
        near = pos.loc[[q["nearest_cell"] for q in use]].to_numpy()
        far = far_from(xy, xy_s, s.buffer_m)
        rec = {"window": w, "slug": sl, "run_id": rid, "group": g, "n_stations": len(use), "n": int(len(part)),
               "n_far": int(far.sum())}
        for b in s.bases:
            p = part[f"pred_{b}"].to_numpy(float)
            res = (t - t.mean()) - (p[near] - p[near].mean())
            kr = simple_krige(xy_s, res, xy, vgs[w][b][g])
            rec[b] = {"all": _score(y, p, kr), "far": _score(y[far], p[far], kr[far]) if far.sum() >= MIN_CELLS
                      else None, "residual_sd": float(res.std())}
        per_run.append(rec)
    out = {}
    for b in s.bases:
        out[b] = {}
        for w in [w for w in s.windows if w in vgs]:
            out[b][w] = {}
            for scope in ("all", "far"):
                units, ss0, ss1 = [], [], []
                for g in sorted({r["group"] for r in per_run if r["window"] == w}):
                    rr = [r for r in per_run if r["window"] == w and r["group"] == g and r[b][scope] is not None
                          and np.isfinite(r[b][scope]).all()]
                    if rr:
                        units.append(g)
                        ss0.append(float(np.mean([r[b][scope][0] for r in rr])))
                        ss1.append(float(np.mean([r[b][scope][1] for r in rr])))
                label = "real stations, all driven cells" if scope == "all" else \
                    f"real stations, cells > {s.buffer_m:.0f} m"
                out[b][w][scope] = compare(units, np.asarray(ss1), np.asarray(ss0), b, label, s)
    return {"paired": out, "per_run": [{k: v for k, v in r.items()} for r in per_run]}


# --------------------------------------------------------------------------- #
# The study                                                                    #
# --------------------------------------------------------------------------- #
def calibrate(pred: pd.DataFrame, s: CalSettings | None = None, cities: dict | None = None,
              history: pd.DataFrame | None = None, reader: Callable | None = None,
              per_unit_ref: dict | None = None) -> dict:
    """Study S1 on an evaluate run's prediction table.  ``cities`` (slug → :class:`~sparc.core.lstair.dataset.
    CityData`, rasters not needed), ``history`` (:func:`parse_isd_history`) and ``reader`` (station, start_utc,
    end_utc → :func:`sparc.core.forcing.station_obs`'s dict) feed the real-station path; ``per_unit_ref`` (window
    → the harness's per_unit) the harness check."""
    s = s or CalSettings()
    t0 = dt.datetime.now(dt.timezone.utc)
    missing = [f"pred_{b}" for b in s.bases if f"pred_{b}" not in pred.columns]
    if missing:
        raise ValueError(f"predictions lack {missing}")
    res = {"schema": SCHEMA, "study": "S1 calibrating with local stations", "settings": asdict(s),
           "primary": {"base": "A", "window": PRIMARY_WINDOW}, "rule": RULE, "windows": {}, "variograms": {},
           "harness_check": {}}
    vgs_all = {}
    for w in s.windows:
        df = pred.loc[pred["run"] == w]
        if df.empty:
            res["windows"][w] = {"status": "not run", "reason": "no predictions for this window"}
            continue
        log.info("S1 %s: variograms (%d runs)", w, df.groupby(["slug", "run_id"]).ngroups)
        vgs = unit_variograms(window_pair_stats(df, s.bases, s), s.bases)
        vgs_all[w] = vgs
        res["variograms"][w] = {b: variogram_summary(vgs[b]) for b in s.bases}
        log.info("S1 %s: simulated stations", w)
        res["windows"][w] = {"status": "done", "n_units": int(df["group"].nunique()), "bases": simulate_window(df, w, vgs, s)}
        if per_unit_ref and per_unit_ref.get(w):
            res["harness_check"][w] = harness_check(df, per_unit_ref[w], s.bases)
    res["real_stations"] = {"status": "skipped (--no-stations)"} if not s.stations else \
        real_stations(pred.loc[pred["run"].isin(list(vgs_all))], vgs_all, s, cities, history, reader)
    res["seconds"] = (dt.datetime.now(dt.timezone.utc) - t0).total_seconds()
    return res


def real_stations(pred: pd.DataFrame, vgs: dict, s: CalSettings, cities: dict | None, history: pd.DataFrame | None,
                  reader: Callable | None) -> dict:
    out = {"status": "done"}
    if history is None:
        try:
            history, out["history_source"] = isd_history()
        except ConnectionError as exc:
            return {"status": "ISD unreachable", "reason": str(exc)}
    reader = reader or default_reader()
    readings = []
    for sl, part in pred.groupby("slug", sort=True):
        c = (cities or {}).get(sl)
        if c is None:
            readings.append({"slug": sl, "city": str(part["city"].iat[0]), "n_in_grid": None, "runs": [],
                             "reason": "city not in the dataset"})
            continue
        log.info("S1 real stations: %s", sl)
        readings.append(station_readings(c, part, history, reader, s))
    out.update(calibrate_real(pred, readings, vgs, s))
    out["cities"] = readings
    return out


def variogram_summary(v: dict) -> dict:
    """Median over held-out units of the leave-one-unit-out fits, the fits themselves and the all-unit pool."""
    fits = {u: f for u, f in v.items() if u != "_all"}
    ok = [f for f in fits.values() if f.get("status") == "ok"]
    med = {k: float(np.median([f[k] for f in ok])) if ok else None
           for k in ("nugget", "psill", "range_m", "practical_range_m", "nugget_share")}
    return {"median": {**med, "n_units": len(ok), "n_not_fitted": len(fits) - len(ok)},
            "units": {u: {k: f.get(k) for k in ("status", "nugget", "psill", "range_m", "nugget_share", "n_pairs")}
                      for u, f in fits.items()},
            "pooled_all_units": v.get("_all")}


RULE = ("Per held-out unit and window, an exponential semivariogram (nugget, partial sill, range) is fitted by "
        "Cressie's weighted least squares to the standardised residuals of every OTHER unit's runs (pairs within a "
        "run, binned to 5 km).  Simulated stations: per run, k driven cells (≥ 1 km apart, random sequential "
        "placement, seeded draws) are perfect stations; their residuals minus their mean are spread by simple "
        "kriging; the calibrated and the uncalibrated map are scored on the same test cells (driven cells > 500 m "
        "from every station; ≥ 30 or the draw is skipped), target and predictions median-centred on them, "
        "SS = 1 − MSE/MSE_N; mean over draws, then over the unit's runs.  ΔSS = SS(calibrated) − SS(base) per unit, "
        "one-sided sign-flip, 95% t-interval, Holm over k within each (base, window); claim: Holm p ≤ 0.025, CI "
        "above 0, mean ΔSS ≥ 0.05, mean calibrated SS > 0, ≥ 8 units.  Primary: base A, af.")


# --------------------------------------------------------------------------- #
# Report                                                                       #
# --------------------------------------------------------------------------- #
def _row(r: dict, lead: list[str]) -> str:
    bw = f"{r['n_better']}/{r['n_worse']}" if "n_better" in r else "—"
    return "| " + " | ".join(lead + [str(r.get("n_units", 0)), _f(r.get("mean_ss_base"), 3),
                                     _f(r.get("mean_ss_x"), 3), _f(r.get("mean_dss"), 3, True), _ci(r.get("ci95")),
                                     _p(r.get("p_sign_flip")), _p(r.get("p_holm")), bw, r.get("verdict", "—")]) + " |"


HEAD = "| units | mean SS base | mean SS calibrated | mean ΔSS | 95% CI | sign-flip p | Holm p | better/worse | verdict |"
SEP = "|---|---|---|---|---|---|---|---|---|"


def calibrate_markdown(res: dict) -> str:
    s = res["settings"]
    L = ["# S1: calibrating held-out city maps with local stations", "",
         f"Predictions: `{res.get('eval', '—')}`; dataset: `{res.get('data', '—')}`.  k = "
         f"{', '.join(map(str, s['ks']))}; {s['draws']} draws; stations ≥ {s['min_sep_m']:.0f} m apart; test cells "
         f"> {s['buffer_m']:.0f} m from every station; seed {s['seed']}.", "", res["rule"], ""]
    hc = res.get("harness_check") or {}
    if hc:
        parts = [f"{w} {b} {v['max_abs_diff']:.1e}" for w, d in hc.items() for b, v in d.items()
                 if isinstance(v, dict) and v.get("max_abs_diff") is not None]
        L += [f"Harness check (uncalibrated SS on all cells vs results.json per unit, max |difference|): "
              f"{'; '.join(parts)}.", ""]
    pw = res["primary"]["window"]
    prim = ((res["windows"].get(pw) or {}).get("bases") or {}).get(res["primary"]["base"])
    L += [f"## Primary: A calibrated − A, {pw} (Holm over k)", ""]
    if prim:
        L += ["| k " + HEAD, "|---" + SEP]
        L += [_row(r, [str(k)]) for k, r in prim["k"].items()]
        L += [""] + [f"- k = {k}: {r['reason']} ({r['n_runs_scored']}/{r['n_runs_total']} runs, "
                     f"{r['n_draws_valid']} valid draws; skipped {r['n_draws_skipped_no_room']} with no room for k "
                     f"stations, {r['n_draws_skipped_few_cells']} with < {MIN_CELLS} test cells)"
                     for k, r in prim["k"].items()] + [""]
    else:
        L += ["not run", ""]
    L += ["## Every base and window (secondary except A af)", "", "| base | window | k " + HEAD, "|---|---|---" + SEP]
    for w, wr in res["windows"].items():
        for b, br in (wr.get("bases") or {}).items():
            L += [_row(r, [b, w, str(k)]) for k, r in br["k"].items()]
    L += ["", "## Residual variograms (leave-one-unit-out fits, median over held-out units; standardised residuals)",
          "", "| base | window | units | nugget | partial sill | nugget share | range a (m) | practical range 3a (m) |",
          "|---|---|---|---|---|---|---|---|"]
    for w, vb in res["variograms"].items():
        for b, v in vb.items():
            m = v["median"]
            L.append(f"| {b} | {w} | {m['n_units']} | {_f(m['nugget'], 3)} | {_f(m['psill'], 3)} | "
                     f"{_f(m['nugget_share'], 2)} | {_f(m['range_m'], 0)} | {_f(m['practical_range_m'], 0)} |")
    rs = res.get("real_stations") or {}
    L += ["", "## Real stations (NOAA Global Hourly / ISD; secondary)", ""]
    if rs.get("status") != "done":
        L += [f"{rs.get('status')}: {rs.get('reason', '')}".rstrip(": "), ""]
        return "\n".join(L) + "\n"
    L += [f"Station list: `{rs.get('history_source', 'given')}`.  Scored on all driven cells (registered) and, as a "
          f"variant, on the cells > {s['buffer_m']:.0f} m from every station; at least {s['min_stations']} usable "
          f"stations (finite temperature over the run's hours, a driven cell within {s['near_m']:.0f} m).", "",
          "| base | window | cells " + HEAD, "|---|---|---" + SEP]
    for b, bw in rs["paired"].items():
        for w, ws in bw.items():
            L += [_row(r, [b, w, "all" if sc == "all" else f"> {s['buffer_m']:.0f} m"]) for sc, r in ws.items()]
    L += ["", "| city | stations in grid | window | run | with observations | usable | scored |", "|---|---|---|---|---|---|---|"]
    for c in rs["cities"]:
        if not c["runs"]:
            L.append(f"| {c.get('city') or c['slug']} | {c.get('n_in_grid', '—')} | — | — | — | — | "
                     f"{c.get('reason', 'no run')} |")
        for r in c["runs"]:
            L.append(f"| {c.get('city') or c['slug']} | {r.get('n_in_grid', '—')} | {r['window']} | {r['run_id']} | "
                     f"{r.get('n_with_obs', '—')} | {r.get('n_usable', '—')} | "
                     f"{'yes' if r.get('scored') else r.get('reason', 'no')} |")
    return "\n".join(L) + "\n"


def run_calibration(data_dir: str | Path, eval_dir: str | Path, out_dir: str | Path, s: CalSettings | None = None,
                    history: pd.DataFrame | None = None, reader: Callable | None = None,
                    isd_cache: str | Path | None = None) -> dict:
    """``sparc core lstair calibrate``: read the evaluate run, calibrate, write calibrate.json and calibrate.md.
    ``isd_cache``: a folder that keeps the Global Hourly station-year files between runs (default: memory only)."""
    from sparc.core.lstair.dataset import load_city

    s = s or CalSettings()
    ev = Path(eval_dir)
    if not (ev / "predictions.parquet").exists():
        raise ValueError(f"{ev} has no predictions.parquet (the --out of a finished lstair evaluate)")
    pred = pd.read_parquet(ev / "predictions.parquet")
    ref = {}
    if (ev / "results.json").exists():
        r = json.loads((ev / "results.json").read_text(encoding="utf-8"))
        ref = {w: (r.get("windows", {}).get(w) or {}).get("per_unit") for w in s.windows}
    cities = {}
    if s.stations:
        root = Path(data_dir)
        root = root / "cities" if (root / "cities").is_dir() else root
        for sl in sorted(pred["slug"].unique()):
            if (root / sl / "meta.json").exists():
                c = load_city(root / sl, rasters=False)
                c.table = pd.DataFrame()                     # only the grid and the run times are needed
                cities[sl] = c
    if s.stations and reader is None:
        reader = default_reader(isd_cache)
    res = calibrate(pred, s, cities=cities, history=history, reader=reader, per_unit_ref=ref)
    res.update(eval=str(eval_dir), data=str(data_dir))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibrate.json").write_text(to_json(res), encoding="utf-8")
    (out / "calibrate.md").write_text(calibrate_markdown(res), encoding="utf-8")
    return res


__all__ = ["CalSettings", "pair_stats", "fit_variogram", "unit_variograms", "window_pair_stats", "simple_krige",
           "exp_cov", "draw_stations", "far_from", "calibrate_run", "simulate_window", "compare", "harness_check",
           "parse_isd_history", "isd_history", "grid_stations", "run_hours_utc", "station_readings", "calibrate_real",
           "calibrate", "calibrate_markdown", "run_calibration", "KS", "BASES", "CLAIM", "ALPHA", "MIN_DSS",
           "MIN_VERDICT_CITIES"]
