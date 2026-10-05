"""Which canopy designs recover a planted effect on this city's layout, and which report one when there is
none?

For every simulation-check generator and replicate: plant a temperature field on the real layout (the
generator's signal plus a spatially correlated residual, matched to the real target's spread), simulate
a traverse campaign over it and the forest-made map from that campaign, then run every design on the
traverses and on the map.

Every design is scored against **its own estimand**, computed exactly from the generator: the mean
change at a point when canopy rises +10 pp within 100 m, 300 m or 1 km of it (a disk edit, at street
points or anywhere in the city), or everywhere (``total``).  How far the cooling of a tree reaches
differs by generator (the advected ``physics`` world puts only about 40% of it within 300 m), and a
design is only expected to recover the part it can see.  Under ``null`` every estimand is 0.  The wind
signature test is scored on how often it flags an advected effect: never in worlds without advection,
as often as possible in ``physics``.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np

from sparc.core.identify import DOSE
from sparc.core.identify.estimators import ESTIMAND, ESTIMATORS, LABELS, run_all

GENERATORS = ("null", "additive", "own_only", "physics", "coarse_scale", "confounded")
REACHES = (100, 300, 1000)
LOCAL_WORLDS = ("additive", "own_only", "confounded")      # cooling within ~300 m
SPREAD_WORLDS = ("physics", "coarse_scale")                 # cooling spread over a kilometre


def prepare(cfg):
    """The real layout, its design layers, the forest's land-cover features and the real target's stats."""
    from sparc.core.identify.layers import build_layers
    from sparc.core.simcheck import _covariates, _product_features, load_layout, real_reference

    layout = load_layout(cfg)
    X = _covariates(layout)
    return layout, build_layers(layout), _product_features(layout), real_reference(layout, X)


def plant(kind: str, layout, real: dict, rng, signal_share: float = 0.6):
    """(temperature field, generator): the generator's signal plus a correlated residual (as the
    simulation check draws it), with the real target's mean and spread."""
    from sparc.core.simcheck import Generator, _grf_points

    gen = Generator(kind, layout, rng, target_signal_sd=math.sqrt(signal_share) * real["sd"])
    sig = gen.signal(gen.C0, gen.I0)
    resid_sd = math.sqrt(max(real["sd"] ** 2 - sig.var(), 0.2 * real["sd"] ** 2))
    noise = resid_sd * (0.85 * _grf_points(layout.grid, real["residual_range_m"], rng)
                        + math.sqrt(1 - 0.85 ** 2) * rng.standard_normal(sig.size))
    return real["mean"] + sig + noise, gen


def reach_profile(gen, layout, cells: np.ndarray, reaches=REACHES, dose: float = DOSE) -> dict[str, float]:
    """Exact truth: the mean change at ``cells`` when canopy rises by ``dose`` within each reach of the
    cell (disk edits, one cell at a time), and everywhere (``total``)."""
    g = layout.grid
    C0, I0 = gen.C0, gen.I0
    x, y = g.ix * g.dx, g.iy * g.dy
    base = gen.signal(C0, I0)
    up = np.clip(C0 + dose, 0.0, 100.0)
    out = {}
    for R in reaches:
        vals = [gen.signal(np.where(np.hypot(x - x[i], y - y[i]) <= R, up, C0), I0)[i] - base[i] for i in cells]
        out[f"within_{int(R)}"] = float(np.mean(vals))
    out["total"] = float(np.mean(gen.truth_delta(dose)[cells]))
    return out


_TRUTH: dict = {}


def truth_for(kind: str, gen, layout, n_cells: int = 80) -> dict:
    """Reach profiles at street points (paved cells, where vehicles drive) and anywhere in the city.  The
    canopy part of every generator is deterministic, so this is computed once per generator."""
    if kind == "null":
        z = {f"within_{r}": 0.0 for r in REACHES} | {"total": 0.0}
        return {"street": dict(z), "city": dict(z)}
    if kind not in _TRUTH:
        rng = np.random.default_rng(20201)
        paved = np.flatnonzero(layout.col("impervious") > 40.0)
        street = rng.choice(paved, min(n_cells, paved.size), replace=False)
        city = rng.choice(gen.C0.size, min(n_cells, gen.C0.size), replace=False)
        _TRUTH[kind] = {"street": reach_profile(gen, layout, street), "city": reach_profile(gen, layout, city)}
    return _TRUTH[kind]


def replicate(kind: str, rep: int, layout, L, features, real, seed: int = 0, which=ESTIMATORS) -> list[dict]:
    from sparc.core.identify.campaign import simulate

    rng = np.random.default_rng([seed, rep, GENERATORS.index(kind) if kind in GENERATORS else 99])
    t0 = time.time()
    T, gen = plant(kind, layout, real, rng)
    camp = simulate(layout, T, rng, features)
    truth = truth_for(kind, gen, layout)
    rows = []
    for r in run_all(camp.samples, camp.product, L, layout.grid, which):
        est = ESTIMAND[r["estimator"]]
        r.update(generator=kind, rep=rep, advects=kind == "physics", n_samples=int(len(camp.samples)),
                 drift_f_per_h=camp.drift_f_per_h, truth=None if est == "contrast" else truth[r["where"]][est],
                 truth_total=truth["city"]["total"], truth_street_total=truth["street"]["total"])
        if "city_estimate" in r:
            r["truth_city"] = truth["city"][est]
        r["seconds"] = round(time.time() - t0, 1)
        rows.append(r)
    return rows


# --------------------------------------------------------------------------- #
# Running the lab                                                              #
# --------------------------------------------------------------------------- #
_STATE: dict = {}


def _init_worker(raw: dict, base_dir: str, threads: int = 1) -> None:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(threads)
    try:                                   # a forked worker inherits BLAS pools sized for the whole machine
        from threadpoolctl import threadpool_limits

        _STATE["limits"] = threadpool_limits(threads)
    except ImportError:  # pragma: no cover - threadpoolctl ships with scikit-learn
        pass
    from sparc.core.config import CoreConfig

    _STATE["ctx"] = prepare(CoreConfig(raw=raw, base_dir=Path(base_dir)))


def _work(args) -> list[dict]:
    kind, rep, seed, which = args
    layout, L, features, real = _STATE["ctx"]
    return replicate(kind, rep, layout, L, features, real, seed, which)


def run_lab(cfg, generators=GENERATORS, reps: int = 8, seed: int = 0, which=ESTIMATORS, out_dir=None,
            workers: int = 1, log=print) -> list[dict]:
    """Every generator × replicate; rows are appended to ``out_dir/identify_lab.jsonl`` as they finish.
    ``workers`` > 1 runs replicates in parallel processes."""
    from sparc.core import progress

    jobs = [(k, r, seed, tuple(which)) for k in generators for r in range(reps)]
    out = Path(out_dir) if out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    def done(rr: list[dict], k: int) -> None:
        rows.extend(rr)
        if out:
            with open(out / "identify_lab.jsonl", "a", encoding="utf-8") as f:
                for r in rr:
                    f.write(json.dumps(r) + "\n")
        if log and rr:
            log(f"[{k}/{len(jobs)}] {rr[0]['generator']:>12} #{rr[0]['rep'] + 1}: "
                + "  ".join(f"{r['estimator']}={r['estimate']:+.3f}±{r['se']:.3f}" for r in rr))

    with progress.stage("identify", label="Canopy identification lab"):
        if workers > 1 and len(jobs) > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            with ProcessPoolExecutor(min(workers, len(jobs)), initializer=_init_worker,
                                     initargs=(cfg.raw, str(cfg.base_dir))) as ex:
                futs = [ex.submit(_work, j) for j in jobs]
                for k, fut in enumerate(as_completed(futs), 1):
                    with progress.task("replicates", n=len(jobs), k=k, unit="replicates"):
                        done(fut.result(), k)
        else:
            layout, L, features, real = prepare(cfg)
            for k, (kind, rep, s, w) in enumerate(jobs, 1):
                with progress.task(f"{kind} #{rep + 1}", n=len(jobs), k=k, unit="replicates"):
                    done(replicate(kind, rep, layout, L, features, real, s, w), k)
    return rows


# --------------------------------------------------------------------------- #
# Summaries and verdicts                                                       #
# --------------------------------------------------------------------------- #
def _order(kind: str) -> int:
    return GENERATORS.index(kind) if kind in GENERATORS else 99


Z_ONE = 1.6448536269514722          # one-sided 95%
FLOOR_DESIGNS = ("street_100", "street_300")


def _street_totals(rows: list[dict]) -> dict:
    """True city-wide effect at street points per (world, replicate): stored on every row since the floor was
    added, otherwise the levels design's truth (whose estimand it is)."""
    out = {(r["generator"], r["rep"]): r["truth_street_total"] for r in rows if r.get("truth_street_total") is not None}
    for r in rows:
        if r["estimator"] == "levels" and r.get("truth") is not None:
            out.setdefault((r["generator"], r["rep"]), r["truth"])
    return out


def floor_stats(rows: list[dict], name: str) -> dict | None:
    """Is the near-field estimate a valid floor on city-wide cooling?  Under "canopy never warms the air at a
    distance", +10 pp everywhere cools a street point at least as much as +10 pp within the design's reach,
    so the one-sided 95% bound ``estimate + 1.645 se`` should lie above the true city-wide change (at
    street points) in at least 95% of campaigns of every world."""
    tot = _street_totals(rows)
    per = {}
    for kind in sorted({r["generator"] for r in rows if r["estimator"] == name}, key=_order):
        rs = [r for r in rows if r["estimator"] == name and r["generator"] == kind and (kind, r["rep"]) in tot]
        if not rs:
            continue
        fl = np.array([r["estimate"] + Z_ONE * r["se"] for r in rs])
        t = np.array([tot[(kind, r["rep"])] for r in rs])
        per[kind] = {"n": len(rs), "holds": float(np.mean(t <= fl + 1e-12)), "mean_floor": float(fl.mean()),
                     "truth_total": float(t.mean()),
                     "tightness": float(np.mean([r["estimate"] for r in rs]) / t.mean()) if abs(t.mean()) > 1e-6 else None}
    if not per:
        return None
    worst = min(v["holds"] for v in per.values())
    need = min(0.9, 1.0 - 1.0 / max(min(v["n"] for v in per.values()), 1))
    return {"worlds": per, "valid": bool(worst >= need), "worst_hold": worst}


def summarize(rows: list[dict]) -> dict:
    """Per design and generator: mean estimate, its truth, bias, spread, RMSE, 95% coverage and how often
    the interval excludes zero (false positives under ``null``, power otherwise); for the street designs,
    whether they are a valid floor on city-wide cooling."""
    out: dict = {"designs": {}, "n_rows": len(rows),
                 "n_reps": {k: len({r["rep"] for r in rows if r["generator"] == k}) for k in GENERATORS
                            if any(r["generator"] == k for r in rows)}}
    for name in ESTIMATORS:
        mine = [r for r in rows if r["estimator"] == name]
        if not mine:
            continue
        per = {}
        for kind in sorted({r["generator"] for r in mine}, key=_order):
            rs = [r for r in mine if r["generator"] == kind]
            est = np.array([r["estimate"] for r in rs])
            se = np.array([r["se"] for r in rs])
            row = {"n": len(rs), "mean": float(est.mean()), "sd": float(est.std(ddof=1)) if len(rs) > 1 else None,
                   "mean_se": float(se.mean()), "excludes_zero": float(np.mean(np.abs(est) > 1.959964 * se))}
            if rs[0]["kind"] == "effect":
                tr = np.array([r["truth"] for r in rs], float)
                row.update(truth=float(tr.mean()), bias=float((est - tr).mean()),
                           rmse=float(np.sqrt(np.mean((est - tr) ** 2))),
                           coverage=float(np.mean(np.abs(est - tr) <= 1.959964 * se)),
                           share=float(est.mean() / tr.mean()) if abs(tr.mean()) > 1e-6 else None,
                           truth_total=float(np.mean([r["truth_total"] for r in rs])))
                if "city_estimate" in rs[0]:
                    ce = np.array([r["city_estimate"] for r in rs])
                    ct = np.array([r["truth_city"] for r in rs])
                    row.update(city_mean=float(ce.mean()), city_truth=float(ct.mean()), city_bias=float((ce - ct).mean()))
            else:
                row.update(advects=bool(rs[0].get("advects")))
            per[kind] = row
        out["designs"][name] = {"label": LABELS[name], "kind": mine[0]["kind"], "estimand": ESTIMAND[name],
                                "where": mine[0]["where"], "level": mine[0]["level"], "generators": per,
                                "verdict": design_verdict(per, mine[0]["kind"])}
        if name in FLOOR_DESIGNS:
            fs = floor_stats(rows, name)
            if fs:
                out["designs"][name]["floor"] = fs
    return out


def _recovers(v: dict, floor: float = 0.03) -> bool:
    """Bias within a third of the truth (at least ``floor`` °F) and 95% intervals covering it ≥ 80% of
    the time (allowing one miss in small labs)."""
    tol = max(abs(v["truth"]) / 3.0, floor)
    need = min(0.8, 1.0 - 1.0 / max(v["n"], 1))
    return abs(v["bias"]) <= tol and v["coverage"] >= need


def _understates(v: dict, floor: float = 0.03) -> bool:
    """Right sign and never stronger than the truth (beyond noise): a lower bound on the cooling."""
    t, m = v["truth"], v["mean"]
    return abs(t) > floor and m * t > 0 and abs(m) <= abs(t) + max(abs(t) / 3.0, floor) and abs(m) >= 0.25 * abs(t)


def _right_sign(v: dict, floor: float = 0.03) -> bool:
    t, m = v["truth"], v["mean"]
    return abs(t) <= floor or (m * t > 0 and abs(m) >= 0.25 * abs(t))


def design_verdict(per: dict, kind: str) -> dict:
    """Plain verdict on a design.

    * **trustworthy**: stays at zero under ``null`` (|mean| ≤ 0.05 °F, intervals excluding zero at most
      1 in 8) and recovers its own estimand in every world with an effect, the confounded one included;
    * **conservative**: clean under ``null``, recovers or understates the cooling in every world, never
      overstates it (a lower bound on how much its canopy cools);
    * **partial**: clean under ``null``, misses in one world;
    * **direction only**: clean under ``null`` and the right sign in every world (at least a quarter of
      the truth), but the size depends on the world;
    * **not trustworthy**: otherwise.

    The signature test is trustworthy when it flags advection in at most 1 in 8 worlds without it and in
    at least half of the advected ones; clean but blind to advection is **underpowered**."""
    reasons: list[str] = []
    if kind == "contrast":
        quiet = [v["excludes_zero"] for k, v in per.items() if not v.get("advects")]
        fp = max(quiet) if quiet else None
        power = per.get("physics", {}).get("excludes_zero")
        if fp is not None:
            reasons.append(f"flags advection in up to {fp:.0%} of worlds without it")
        if power is not None:
            reasons.append(f"detects it in {power:.0%} of advected worlds")
        ok = fp is not None and fp <= 0.125 and (power or 0.0) >= 0.5
        status = "trustworthy" if ok else "underpowered" if fp is not None and fp <= 0.125 else "not trustworthy"
        return {"trustworthy": bool(ok), "status": status, "reasons": reasons}
    null = per.get("null")
    clean = null is not None and abs(null["mean"]) <= 0.05 and null["excludes_zero"] <= 0.125
    if null is not None:
        reasons.append(f"null: {null['mean']:+.3f} °F, intervals exclude zero {null['excludes_zero']:.0%}")
    worlds = [k for k in per if k != "null"]
    good = [k for k in worlds if _recovers(per[k])]
    bad = [k for k in worlds if k not in good]
    under = [k for k in bad if _understates(per[k])]
    if worlds:
        reasons.append(f"recovers its estimand in {len(good)}/{len(worlds)} worlds"
                       + (f" (misses {', '.join(bad)})" if bad else ""))
    if under and len(under) == len(bad):
        reasons.append(f"understates the cooling in {', '.join(under)}"
                       + "".join(f" ({per[k]['mean'] / per[k]['truth']:.0%} of it)" for k in under[:1]))
    if clean and worlds and not bad:
        status = "trustworthy"
    elif clean and worlds and len(under) == len(bad):
        status = "conservative"
    elif clean and len(good) >= max(len(worlds) - 1, 1):
        status = "partial"
    elif clean and worlds and all(_right_sign(per[k]) for k in worlds):
        status = "direction only"
    else:
        status = "not trustworthy"
    return {"trustworthy": status == "trustworthy", "status": status, "clean_null": bool(clean),
            "recovers": f"{len(good)}/{len(worlds)}", "misses": bad, "understates": under, "reasons": reasons}


def lab_markdown(summ: dict) -> str:
    from sparc.core.identify.report import lab_markdown as md

    return md(summ)
