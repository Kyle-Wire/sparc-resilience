"""The kilometre lab: does the wind-shift design flag advected cooling when there is none, and how often
does it find it when there is?

One simulated campaign day = several runs over the same routes under different winds (by default a
sea-breeze day: light west-north-west in the morning, a southerly afternoon breeze, south-south-west in
the evening; pass the real winds once known).  Each run has its own warming drift, vehicle offsets and
sensor noise, its own weather-driven spatial pattern on top of a persistent one, and its own strength of
canopy's effect.  Worlds:

* ``null``, ``additive``, ``own_only``, ``coarse_scale``: no advected canopy cooling;
* ``confounded``: local cooling plus a hidden kilometre-scale factor that tracks canopy, **whose effect
  changes from run to run**;
* ``advected_water``: local canopy cooling plus shore cooling carried by each run's wind (a confounder
  that does move with the wind);
* ``physics``: canopy's cooling carried downwind (advection–diffusion with each run's wind; the
  advection length grows with wind speed, 300 m at 7.7 m/s).
"""

from __future__ import annotations

import copy
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

KM_WORLDS = ("null", "additive", "own_only", "coarse_scale", "confounded", "advected_water", "physics")
SEA_BREEZE_DAY = ((290.0, 2.0), (170.0, 7.7), (210.0, 4.0))      # (from °, m/s) morning, afternoon, evening
STRENGTH = (0.5, 1.0, 0.75)                                        # canopy's effect by run (time of day)
ADVECTION_S = 300.0 / 7.7                                          # advection length per m/s of wind


def _generator(kind: str, layout, real: dict, frm: float, spd: float):
    from sparc.core.identify.windshift import toward
    from sparc.core.simcheck import Generator

    base = "additive" if kind == "advected_water" else kind
    lay = copy.copy(layout)
    lay.cfg = copy.deepcopy(layout.cfg)
    u = toward(frm)
    lay.cfg.raw.setdefault("physics", {})["wind"] = [u[0] * spd, u[1] * spd]
    gen = Generator(base, lay, np.random.default_rng(123), target_signal_sd=math.sqrt(0.6) * real["sd"])
    if base == "physics":
        gen.v = (ADVECTION_S * spd * u[0], ADVECTION_S * spd * u[1])
        if hasattr(gen, "_q0_mean"):
            del gen._q0_mean
        gen._calibrate()
        gen._match_signal(math.sqrt(0.6) * real["sd"])
    return gen


def plant_runs(kind: str, layout, real: dict, rng, winds) -> list[np.ndarray]:
    """Temperature field of each run (one per wind)."""
    from sparc.core import operators as ops
    from sparc.core.identify.windshift import toward
    from sparc.core.simcheck import _grf_points

    g = layout.grid
    persist = _grf_points(g, real["residual_range_m"], rng)
    shared = None if kind == "physics" else _generator(kind, layout, real, 170.0, 7.7)
    wd = layout.col("water_distance")
    out = []
    for k, (frm, spd) in enumerate(winds):
        gen = _generator(kind, layout, real, frm, spd) if kind == "physics" else shared
        lead, rest, hidden = gen._parts(gen.C0, gen.I0)
        sig = STRENGTH[k % len(STRENGTH)] * lead + gen.rest_scale * rest + rng.uniform(0.4, 1.6) * hidden
        if kind == "advected_water" and wd is not None:
            u = toward(frm)
            src = np.zeros(g.shape)
            src[g.iy, g.ix] = np.exp(-np.asarray(wd, float) / 100.0)
            v = (ADVECTION_S * spd * u[0], ADVECTION_S * spd * u[1])
            sig = sig - 0.6 * ops.solve(src, 300.0, v, dx=g.dx)[g.iy, g.ix]
        noise = 0.8 * (0.75 * persist + 0.55 * _grf_points(g, real["residual_range_m"], rng)
                       + 0.35 * rng.standard_normal(g.iy.size))
        out.append(real["mean"] + sig + noise)
    return out


def simulate_runs(layout, fields: list[np.ndarray], rng, names: list[str]) -> pd.DataFrame:
    """The same routes driven once per run: one table with ``run`` (vehicles and passes unique per run)."""
    from sparc.core.identify.campaign import HOUR_S, drive, route_segments

    segs = route_segments(layout, 300.0, rng=rng)
    frames = []
    for k, (T, name) in enumerate(zip(fields, names)):
        df = drive(segs, rng, 10, 3.0)
        off = rng.normal(0.0, 0.2, 10)
        df["temp"] = (T[df["cell"].to_numpy()] + rng.uniform(-1.5, 1.5) * df["t_s"] / HOUR_S
                      + off[df["vehicle"].to_numpy()] + 0.3 * rng.standard_normal(len(df)))
        df["run"] = name
        df["vehicle"] += 100 * k
        df["seg"] += 1_000_000 * k
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def replicate(kind: str, rep: int, ctx: dict, winds, seed: int = 0) -> dict:
    from sparc.core.identify.windshift import wind_shift

    layout, L, real, table = ctx["layout"], ctx["L"], ctx["real"], ctx["table"]
    rng = np.random.default_rng([seed, rep, KM_WORLDS.index(kind) if kind in KM_WORLDS else 99, 7])
    t0 = time.time()
    names = [f"run{k}" for k in range(len(winds))]
    df = simulate_runs(layout, plant_runs(kind, layout, real, rng, winds), rng, names)
    r = wind_shift(df, L, table, {n: w[0] for n, w in zip(names, winds)}, layout.grid)
    return {"generator": kind, "rep": rep, "advects": kind == "physics", "estimate": r["estimate"], "se": r["se"],
            "p_rotation": r["p_rotation"], "p_cooling": r["p_cooling"], "null_sd": r["null_sd"],
            "n_cells": r["n_cells"], "winds": [list(w) for w in winds], "seconds": round(time.time() - t0, 1)}


_STATE: dict = {}


def _init(raw: dict, base_dir: str, threads: int = 1) -> None:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(threads)
    try:
        from threadpoolctl import threadpool_limits

        _STATE["limits"] = threadpool_limits(threads)
    except ImportError:  # pragma: no cover
        pass
    from sparc.core.config import CoreConfig

    _STATE["ctx"] = context(CoreConfig(raw=raw, base_dir=Path(base_dir)))


def context(cfg) -> dict:
    from sparc.core.identify.validate import prepare
    from sparc.core.identify.windshift import sector_table

    layout, L, _features, real = prepare(cfg)
    return {"layout": layout, "L": L, "real": real, "table": sector_table(layout)}


def _work(args) -> dict:
    kind, rep, winds, seed = args
    return replicate(kind, rep, _STATE["ctx"], winds, seed)


def run_km_lab(cfg, worlds=KM_WORLDS, reps: int = 12, winds=SEA_BREEZE_DAY, seed: int = 0, out_dir=None,
               workers: int = 1, log=print) -> list[dict]:
    from sparc.core import progress

    jobs = [(k, r, tuple(tuple(w) for w in winds), seed) for k in worlds for r in range(reps)]
    out = Path(out_dir) if out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    def done(row: dict, k: int) -> None:
        rows.append(row)
        if out:
            with open(out / "identify_kmlab.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        if log:
            log(f"[{k}/{len(jobs)}] {row['generator']:>14} #{row['rep'] + 1}: {row['estimate']:+.3f} "
                f"(rotation p {row['p_rotation']:.2f}, cooling p {row['p_cooling']:.2f})")

    with progress.stage("identify_km", label="Kilometre lab"):
        if workers > 1 and len(jobs) > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            with ProcessPoolExecutor(min(workers, len(jobs)), initializer=_init,
                                     initargs=(cfg.raw, str(cfg.base_dir))) as ex:
                futs = [ex.submit(_work, j) for j in jobs]
                for k, fut in enumerate(as_completed(futs), 1):
                    with progress.task("replicates", n=len(jobs), k=k, unit="replicates"):
                        done(fut.result(), k)
        else:
            ctx = context(cfg)
            for k, (kind, rep, w, s) in enumerate(jobs, 1):
                with progress.task(f"{kind} #{rep + 1}", n=len(jobs), k=k, unit="replicates"):
                    done(replicate(kind, rep, ctx, w, s), k)
    return rows


def summarize_km(rows: list[dict], alpha: float = 0.05) -> dict:
    """Per world: mean estimate, spread, how often the rotation test flags a directional effect (two-sided:
    false alarms in worlds without advected canopy cooling) and how often it finds cooling (one-sided:
    power in the advected world); the design's verdict."""
    from sparc.core.identify.validate import design_verdict

    per = {}
    for kind in [k for k in KM_WORLDS if any(r["generator"] == k for r in rows)]:
        rs = [r for r in rows if r["generator"] == kind]
        e = np.array([r["estimate"] for r in rs])
        flag = np.array([r["p_rotation"] < alpha for r in rs])
        cool = np.array([r["p_cooling"] < alpha for r in rs])
        per[kind] = {"n": len(rs), "mean": float(e.mean()), "sd": float(e.std(ddof=1)) if len(rs) > 1 else None,
                     "mean_se": float(np.mean([r["se"] for r in rs])),
                     "rotation_sd": float(np.mean([r["null_sd"] for r in rs])),
                     "excludes_zero": float(cool.mean() if kind == "physics" else flag.mean()),
                     "flags": float(flag.mean()), "finds_cooling": float(cool.mean()), "advects": kind == "physics"}
    verdict = design_verdict(per, "contrast")
    winds = rows[0]["winds"] if rows else []
    return {"designs": {"wind_shift": {
        "label": "Same streets, different winds: canopy upwind − downwind (1 km)", "kind": "contrast",
        "estimand": "advected_1000", "level": "traverse", "generators": per, "verdict": verdict, "winds": winds}},
        "n_rows": len(rows), "n_reps": {k: v["n"] for k, v in per.items()}}


def km_markdown(summ: dict) -> str:
    d = summ["designs"]["wind_shift"]
    w = ", ".join(f"from {a:.0f}° at {b:.1f} m/s" for a, b in d.get("winds", []))
    lines = ["# Kilometre lab: the same streets under different winds", "",
             f"One campaign day of {len(d.get('winds', []))} runs ({w}); each street compared with itself across runs; "
             "inference by rotating every run's wind together (35 rotations).", "",
             "| world | mean contrast (°F) | spread | flags a direction (p < 0.05) | finds cooling (one-sided) |",
             "|---|---|---|---|---|"]
    for k, v in d["generators"].items():
        lines.append(f"| {k} | {v['mean']:+.3f} | {v['sd'] or 0:.3f} | {v['flags']:.0%} | {v['finds_cooling']:.0%} |")
    lines += ["", f"**{d['verdict']['status'].capitalize()}**: " + "; ".join(d["verdict"]["reasons"]) + "."]
    return "\n".join(lines) + "\n"
