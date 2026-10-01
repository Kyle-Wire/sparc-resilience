"""Simulation check: does SPARC recover known effects *in this city*?

The synthetic-city benchmark (``sparc core benchmark``) uses an invented
layout.  Here the layout is the real one: the city's own canopy, impervious,
albedo, NDVI, elevation and water layers.  A generator plants a known
temperature response on them; the field then goes through an emulation of
how the target product was made; and the full pipeline (S0–S6, coarse
resolution) is asked to recover the planted canopy effect.

Generators (several, so the check is not circular — SPARC contains a physics
model, so a physics-only truth would flatter it):

* ``physics``      — a surface-energy source propagated by an
                     advection–diffusion–relaxation operator (L = 300 m,
                     downwind drift);
* ``additive``     — neighbourhood (150 m) saturating canopy, 400 m
                     impervious, own-cell albedo, elevation and water terms;
* ``own_only``     — canopy acts only on its own 30 m cell (finer than any
                     neighbourhood feature the models see);
* ``coarse_scale`` — canopy acts only through its 1 km neighbourhood mean;
* ``confounded``   — ``additive`` plus a hidden 2 km field correlated with
                     large-scale canopy (not given to the models; not part
                     of the canopy truth);
* ``null``         — ``additive`` without any canopy effect.

Each draw adds a spatially correlated residual, then emulates the product:
traverse-like sampling along a street grid on paved cells, sensor noise, a
random forest on land-cover predictors fitted to the samples to fill every
cell, and whole-degree classing of the same share of values as the real
target.  A **data-matching gate** compares the product with the real target
(spread, whole-degree share, residual correlation range); draws that fail
it are redrawn (up to three times) and reported.

Per replicate SPARC's uniform +10 pp canopy scenario is compared with the
truth: effect share (dose-normalised), rank correlation of the per-cell
changes, whether the jackknife 95% interval covers the truth, whether the
causal θ CI covers the true per-pp effect, interval coverage on the target,
and (null) false positives.  The summary proposes a city-wide bias
correction only if the share is stable across the non-null generators
(spread < 25%); otherwise it reports the range.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sparc.core import operators as ops

log = logging.getLogger(__name__)

GENERATORS = ("physics", "additive", "own_only", "coarse_scale", "confounded", "null")
TARGET = "T_sim"
DOSE = 10.0
TRUE_EFFECT_F = -0.25          # planted city-mean change for +10 pp canopy (°F), all non-null generators


# --------------------------------------------------------------------------- #
# Layout                                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class Layout:
    cfg: object
    df: pd.DataFrame            # clean input table (rows as prepare_frame keeps them)
    data: object                # fine CoreData of the real target
    roles: dict

    @property
    def grid(self):
        return self.data.grid

    def col(self, role: str) -> np.ndarray | None:
        c = self.roles.get(role)
        return self.data.frame[c].to_numpy(float) if c and c in self.data.frame else None


def load_layout(cfg) -> Layout:
    from sparc.core.data import prepare_frame
    from sparc.core.placebo import _clean_input

    df = _clean_input(cfg)
    fine = copy.deepcopy(cfg)
    fine.raw["data"]["coarse_m"] = None
    fine.raw["data"]["subsample"] = None
    data = prepare_frame(df, fine)
    roles = (cfg.raw.get("physics") or {}).get("roles") or {}
    if not roles.get("canopy") or not roles.get("impervious"):
        raise ValueError("simcheck needs physics roles canopy and impervious")
    return Layout(cfg=cfg, df=df, data=data, roles=roles)


def _focal(values: np.ndarray, grid, radius_m: float) -> np.ndarray:
    r = grid.rasterize(np.asarray(values, float))
    m = np.isfinite(r)
    return grid.sample(ops.masked_gaussian(r, m, max(radius_m / 2.0 / grid.dx, 0.5)))


def _z(v: np.ndarray) -> np.ndarray:
    return (v - v.mean()) / (v.std() + 1e-12)


def _grf_points(grid, range_m: float, rng) -> np.ndarray:
    from sparc.core.synthetic import gaussian_random_field

    return gaussian_random_field(grid.shape, range_m / grid.dx, rng)[grid.iy, grid.ix]


# --------------------------------------------------------------------------- #
# Generators                                                                   #
# --------------------------------------------------------------------------- #
class Generator:
    """Planted temperature T(canopy, impervious) on the real layout (°F, mean 0)."""

    def __init__(self, kind: str, layout: Layout, rng: np.random.Generator, target_signal_sd: float | None = None):
        if kind not in GENERATORS:
            raise ValueError(f"unknown generator {kind!r}")
        self.kind, self.L, self.g = kind, layout, layout.grid
        self.C0 = layout.col("canopy")
        self.I0 = layout.col("impervious")
        a = layout.col("albedo")
        e = layout.col("elevation")
        w = layout.col("water_distance")
        self.fixed = np.zeros(self.C0.size)
        if a is not None and kind != "physics":           # the physics source has its own albedo term
            self.fixed += -0.2 * _z(a)
        if e is not None:
            self.fixed += -0.01 * (e - e.mean()) * 1.8
        if w is not None:
            self.fixed += 0.6 * (1.0 - np.exp(-w / 300.0))
        self.albedo = a if a is not None else np.full(self.C0.size, 0.15)
        self.U = None
        if kind == "confounded":
            big = _focal(self.C0, self.g, 2000.0)
            self.U = 0.6 * _z(big) + 0.8 * _grf_points(self.g, 2000.0, rng)
        wind = (layout.cfg.raw.get("physics") or {}).get("wind") or [0.0, 0.0]
        nv = math.hypot(*wind[:2]) or 1.0
        self.v = (300.0 * wind[0] / nv, 300.0 * wind[1] / nv) if any(wind[:2]) else (0.0, 0.0)
        self.beta_c = 1.0
        self.beta_i = 1.0
        self.rest_scale = 1.0
        self._calibrate()
        if target_signal_sd:
            self._match_signal(float(target_signal_sd))

    # canopy and impervious terms ------------------------------------------------
    def _canopy_term(self, C: np.ndarray) -> np.ndarray:
        h = 1.0 - np.exp(-np.clip(C, 0, 100) / 30.0)
        if self.kind in ("additive", "confounded", "null"):
            return -_focal(h, self.g, 150.0)
        if self.kind == "own_only":
            return -h
        if self.kind == "coarse_scale":
            return -_focal(C / 100.0, self.g, 1000.0)
        raise AssertionError

    def _impervious_term(self, imp: np.ndarray) -> np.ndarray:
        return _focal(np.clip(imp, 0, 100) / 100.0, self.g, 400.0)

    def _source(self, C: np.ndarray, imp: np.ndarray) -> np.ndarray:
        h = 1.0 - np.exp(-np.clip(C, 0, 100) / 30.0)
        return (1.0 - self.albedo) * (0.4 + 0.6 * np.clip(imp, 0, 100) / 100.0) - 0.8 * h

    def _physics(self, C: np.ndarray, imp: np.ndarray) -> np.ndarray:
        g = self.g
        if not hasattr(self, "_q0_mean"):                 # centre on the *baseline* source only, so a
            self._q0_mean = float(self._source(self.C0, self.I0).mean())   # uniform edit is not removed
        r = np.zeros(g.shape)
        r[g.iy, g.ix] = self._source(C, imp) - self._q0_mean
        return ops.solve(r, 300.0, self.v, dx=g.dx)[g.iy, g.ix]

    def _parts(self, C: np.ndarray, imp: np.ndarray):
        """(canopy-bearing part, rescalable rest, hidden confounder part)."""
        if self.kind == "physics":
            lead = self.a_phys * self._physics(C, imp)
        else:
            lead = 0.0 if self.kind == "null" else self.beta_c * self._canopy_term(C)
        rest = self.beta_i * self._impervious_term(imp) + self.fixed
        hidden = 0.8 * self.U if self.U is not None else 0.0
        return lead, rest, hidden

    def signal(self, C: np.ndarray, imp: np.ndarray) -> np.ndarray:
        lead, rest, hidden = self._parts(C, imp)
        return lead + self.rest_scale * rest + hidden

    def _match_signal(self, target_sd: float) -> None:
        """Rescale the non-canopy terms so the signal's spread matches the real
        target's explained share (the canopy truth is untouched)."""
        lead, rest, hidden = self._parts(self.C0, self.I0)
        A = np.asarray(lead + hidden, float) * np.ones(self.C0.size)
        B = np.asarray(rest, float)
        va, vb, cab = A.var(), B.var(), float(np.cov(A, B)[0, 1])
        disc = cab ** 2 - vb * (va - target_sd ** 2)
        self.rest_scale = float(max((-cab + np.sqrt(disc)) / vb, 0.0)) if vb > 0 and disc >= 0 else 1.0

    def truth_delta(self, dose: float = DOSE) -> np.ndarray:
        """True per-cell change for a uniform canopy edit (clipped to [0, 100])."""
        C1 = np.clip(self.C0 + dose, 0.0, 100.0)
        return self.signal(C1, self.I0) - self.signal(self.C0, self.I0)

    def _calibrate(self) -> None:
        """Scale the canopy term so +10 pp gives TRUE_EFFECT_F city-wide, and the
        impervious term so −10 pp gives about −0.4 °F (as in the real run)."""
        imp1 = np.clip(self.I0 - DOSE, 0, 100)
        if self.kind == "physics":
            self.a_phys = 1.0
            d = float(np.mean(self.truth_delta()))
            self.a_phys = TRUE_EFFECT_F / d if d else 1.0
            return
        if self.kind != "null":
            self.beta_c = 1.0
            d = float(np.mean(self._canopy_term(np.clip(self.C0 + DOSE, 0, 100)) - self._canopy_term(self.C0)))
            self.beta_c = TRUE_EFFECT_F / d if d else 0.0
        di = float(np.mean(self._impervious_term(imp1) - self._impervious_term(self.I0)))
        self.beta_i = -0.4 / di if di else 1.0


# --------------------------------------------------------------------------- #
# Product emulation and the data-matching gate                                 #
# --------------------------------------------------------------------------- #
def _product_features(layout: Layout) -> np.ndarray:
    cols = []
    for role in ("canopy", "impervious", "albedo", "ndvi", "elevation", "water_distance"):
        v = layout.col(role)
        if v is None:
            continue
        cols.append(v)
        if role in ("canopy", "impervious", "albedo", "ndvi"):
            cols += [_focal(v, layout.grid, r) for r in (100.0, 300.0, 600.0)]
    return np.column_stack(cols)


def emulate_product(T: np.ndarray, layout: Layout, rng, features: np.ndarray, street_m: float = 300.0,
                    sample_frac: float = 0.08, sensor_sd: float = 0.3, int_share: float | None = None) -> np.ndarray:
    """Traverse sampling on a street grid of paved cells → random forest on
    land-cover predictors → whole-degree classing (share of the real target)."""
    from sklearn.ensemble import RandomForestRegressor

    g = layout.grid
    k = max(int(round(street_m / g.dx)), 2)
    oy, ox = rng.integers(0, k, size=2)
    imp = layout.col("impervious")
    route = (((g.iy % k) == oy) | ((g.ix % k) == ox)) & (imp > 40.0)
    idx = np.flatnonzero(route)
    n_s = min(idx.size, int(sample_frac * T.size))
    s = rng.choice(idx, n_s, replace=False) if n_s < idx.size else idx
    y = T[s] + sensor_sd * rng.standard_normal(s.size)
    rf = RandomForestRegressor(n_estimators=120, min_samples_leaf=5, max_features=0.5, n_jobs=1,
                               random_state=int(rng.integers(1 << 31)))
    prod = rf.fit(features[s], y).predict(features)
    share = layout.data.qa.get("target_fraction_integer_valued", 0.0) if int_share is None else int_share
    pick = rng.random(prod.size) < share
    prod[pick] = np.round(prod[pick])
    return prod


def residual_range(values: np.ndarray, X: np.ndarray, grid, max_m: float = 4000.0) -> float:
    """Distance at which the autocorrelation of the OLS residual drops below
    1/e (FFT on the raster, mask-normalised) — a cheap correlogram range."""
    Z = np.column_stack([np.ones(len(values)), X])
    beta, *_ = np.linalg.lstsq(Z, values, rcond=None)
    r = values - Z @ beta
    R = np.zeros(grid.shape)
    M = np.zeros(grid.shape)
    R[grid.iy, grid.ix] = r
    M[grid.iy, grid.ix] = 1.0
    py, px = 2 * grid.shape[0], 2 * grid.shape[1]
    fr, fm = np.fft.rfft2(R, (py, px)), np.fft.rfft2(M, (py, px))
    num = np.fft.irfft2(fr * np.conj(fr), (py, px))
    den = np.fft.irfft2(fm * np.conj(fm), (py, px))
    with np.errstate(invalid="ignore", divide="ignore"):
        ac = num / np.maximum(den, 1.0)
    ac = ac / ac[0, 0]
    yy = np.minimum(np.arange(py), py - np.arange(py))[:, None]
    xx = np.minimum(np.arange(px), px - np.arange(px))[None, :]
    d = np.hypot(yy, xx) * grid.dx
    for lag in np.arange(grid.dx, max_m, grid.dx):
        ring = (d >= lag - grid.dx / 2) & (d < lag + grid.dx / 2) & (den > 50)
        if ring.any() and float(np.mean(ac[ring])) < 1.0 / math.e:
            return float(lag)
    return float(max_m)


def gate_stats(product: np.ndarray, layout: Layout, X: np.ndarray, real: dict | None = None) -> dict:
    out = {"sd": float(product.std()), "integer_share": float(np.mean(np.isclose(product, np.round(product)))),
           "residual_range_m": residual_range(product, X, layout.grid)}
    if real:
        out["sd_ratio"] = out["sd"] / real["sd"]
        out["range_ratio"] = out["residual_range_m"] / real["residual_range_m"]
        out["pass"] = bool(0.8 <= out["sd_ratio"] <= 1.25 and abs(out["integer_share"] - real["integer_share"]) <= 0.05
                           and 0.5 <= out["range_ratio"] <= 2.0)
    return out


def draw(kind: str, layout: Layout, seed: int, real: dict, features: np.ndarray, X: np.ndarray,
         max_tries: int = 3, signal_share: float = 0.6) -> tuple[np.ndarray, Generator, dict]:
    """One gated simulated target: (product, generator, gate stats).  The
    planted signal explains ``signal_share`` of the real target's variance
    (Providence's held-out R² is 0.56; the product adds smoothing on top)."""
    noise_scale = 1.0
    for attempt in range(max_tries):
        rng = np.random.default_rng([seed, attempt, GENERATORS.index(kind)])
        gen = Generator(kind, layout, rng, target_signal_sd=math.sqrt(signal_share) * real["sd"])
        sig = gen.signal(gen.C0, gen.I0)
        resid_sd = math.sqrt(max(real["sd"] ** 2 - sig.var(), 0.2 * real["sd"] ** 2)) * noise_scale
        noise = resid_sd * (0.85 * _grf_points(layout.grid, real["residual_range_m"], rng)
                            + math.sqrt(1 - 0.85 ** 2) * rng.standard_normal(sig.size))
        T = real["mean"] + sig + noise
        prod = emulate_product(T, layout, rng, features)
        st = gate_stats(prod, layout, X, real)
        st["attempt"] = attempt
        st["signal_share"] = float(sig.var() / T.var())
        if st["pass"]:
            return prod, gen, st
        noise_scale *= float(np.clip(1.0 / max(st["sd_ratio"], 1e-3), 0.5, 2.0))
    return prod, gen, st


# --------------------------------------------------------------------------- #
# Replicates                                                                   #
# --------------------------------------------------------------------------- #
def sim_config(cfg, coarse: float | None, epochs: int = 200):
    c = copy.deepcopy(cfg)
    raw = c.raw
    can = (raw.get("physics") or {}).get("roles", {}).get("canopy")
    raw["name"] = f"{cfg.name}_simcheck"
    raw["data"]["target"] = TARGET
    raw["data"]["coarse_m"] = float(coarse) if coarse else None
    spec = dict((cfg.actionable or {}).get(can) or {"min": 0, "max": 100})
    spec["doses"] = [0, 5, 10, 20, 30]
    raw["actionable"] = {can: spec}
    raw["scenarios"] = [{"name": "Canopy", "variable": can, "direction": "increase", "increments": [DOSE]}]
    raw["joint_scenarios"] = []
    cz = raw.setdefault("causal", {})
    cz.update(enabled=True, treatments=[can], contrast={can: DOSE}, dag_audit=False,
              confounders={can: (cz.get("confounders") or {}).get(can, [])})
    raw["climate"] = {**(raw.get("climate") or {}), "enabled": False}
    raw["optimize"] = {**(raw.get("optimize") or {}), "enabled": False}
    raw["cv"]["distance_curve"] = {**(raw["cv"].get("distance_curve") or {}), "enabled": False}
    raw["cv"]["baselines"] = False
    raw["stacker"]["epochs"] = min(int(raw["stacker"].get("epochs", 400)), epochs)
    raw["stacker"]["tune_lambda"] = [0.0]
    return c


def _coarse_mean(values: np.ndarray, layout: Layout, coarse: float | None) -> np.ndarray:
    if not coarse:
        return values
    from sparc.core.data import _fine_to_coarse

    d = layout.data
    inv, members, _, _ = _fine_to_coarse(d.x, d.y_coord, d.grid.dx, float(coarse))
    return np.bincount(inv, weights=values, minlength=members.size) / members


def run_replicate(layout: Layout, kind: str, seed: int, real: dict, features: np.ndarray, X: np.ndarray,
                  coarse: float | None = 90.0, epochs: int = 200) -> dict:
    from scipy.stats import spearmanr

    from sparc.core.pipeline import run_core

    t0 = time.time()
    prod, gen, gate = draw(kind, layout, seed, real, features, X)
    df = layout.df.copy()
    df[TARGET] = prod
    cfg = sim_config(layout.cfg, coarse, epochs)
    cfg.raw["cv"]["seed"] = 42 + seed
    res = run_core(cfg, stages=("S0", "S1", "S2", "S3", "S4", "S5", "S6"), frame=df, write=False)
    can = layout.roles["canopy"]
    sc = res.scenarios[0]
    true_fine = gen.truth_delta()
    true_cells = _coarse_mean(true_fine, layout, coarse)
    real_fine = float(np.mean(np.clip(gen.C0 + DOSE, 0, 100) - gen.C0))
    model_dose = float(sc["mean_realized"][can])
    t_mean = float(np.mean(true_cells))
    m_mean, m_se = float(sc["mean_delta"]), float(sc.get("mean_delta_se") or np.nan)
    true_scaled = t_mean * model_dose / real_fine                 # truth at the model's realised dose
    c = (res.manifest.get("causal") or {}).get(can) or {}
    th, th_se = c.get("theta_sum"), c.get("se_sum")
    true_pp = t_mean / real_fine
    out = {
        "generator": kind, "seed": seed, "gate": gate, "seconds": round(time.time() - t0, 1),
        "true_mean_delta": t_mean, "true_realised_dose": real_fine,
        "model_mean_delta": m_mean, "model_se": m_se, "model_realised_dose": model_dose,
        "share": (m_mean / true_scaled) if abs(true_scaled) > 1e-9 else None,
        "ci_covers_truth": bool(abs(m_mean - true_scaled) <= 1.96 * m_se) if np.isfinite(m_se) else None,
        "significant": bool(abs(m_mean) > 1.96 * m_se) if np.isfinite(m_se) else None,
        "causal_theta_pp": th, "causal_se_pp": th_se, "true_pp": true_pp,
        "causal_covers_truth": bool(abs(th - true_pp) <= 1.96 * th_se) if th is not None and th_se else None,
        "causal_significant": bool(abs(th) > 1.96 * th_se) if th is not None and th_se else None,
        "interval_coverage": (res.manifest.get("metrics") or {}).get("stacker", {}).get("interval_coverage"),
        "oof_r2": (res.manifest.get("metrics") or {}).get("stacker", {}).get("r2"),
        "stacker_choice": res.manifest.get("stacker_choice"),
    }
    if res.ensemble is not None and kind != "null":
        delta = _scenario_delta(res)
        if delta is not None and delta.size == true_cells.size:
            out["rank_corr"] = float(spearmanr(delta, true_cells).correlation)
    return out


def _scenario_delta(res) -> np.ndarray | None:
    from sparc.core.mediators import MediatorChain
    from sparc.core.scenarios import ScenarioEngine, specs_from_config

    cfg = res.cfg
    med = MediatorChain(cfg.mediators).fit(res.data.frame) if cfg.mediators else None
    eng = ScenarioEngine(res.data, cfg, res.ensemble, res.influence.ranges_m, med)
    return eng.run(specs_from_config(cfg)[0]).delta


def real_reference(layout: Layout, X: np.ndarray) -> dict:
    t = layout.data.target_raw
    return {"mean": float(t.mean()), "sd": float(t.std()),
            "integer_share": float(layout.data.qa.get("target_fraction_integer_valued", 0.0)),
            "residual_range_m": residual_range(t, X, layout.grid)}


def _covariates(layout: Layout) -> np.ndarray:
    return layout.data.X.to_numpy(float)


def _worker(args):
    cfg_raw, base_dir, kind, seed, coarse, epochs, threads = args
    import torch

    from sparc.core.config import core_config_from_dict

    torch.set_num_threads(threads)
    cfg = core_config_from_dict(cfg_raw, base_dir=base_dir)
    layout = load_layout(cfg)
    X = _covariates(layout)
    real = real_reference(layout, X)
    feats = _product_features(layout)
    try:
        return run_replicate(layout, kind, seed, real, feats, X, coarse=coarse, epochs=epochs)
    except Exception as exc:                    # noqa: BLE001 - recorded, the study continues
        log.exception("simcheck %s/%d failed", kind, seed)
        return {"generator": kind, "seed": seed, "error": repr(exc)}


def run_simcheck(cfg, design: dict[str, int], out_dir: str | Path, coarse: float | None = 90.0, epochs: int = 200,
                 workers: int = 1, threads: int = 1) -> dict:
    """Run ``design`` = {generator: n replicates}, resuming from
    ``out_dir/simcheck.jsonl``; returns :func:`summarize`."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "simcheck.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if "error" not in r:
                done.add((r["generator"], r["seed"]))
    jobs = [(k, s) for k, n in design.items() for s in range(n) if (k, s) not in done]
    log.info("simcheck: %d replicates to run (%d done)", len(jobs), len(done))
    args = [(cfg.raw, str(cfg.base_dir), k, s, coarse, epochs, threads) for k, s in jobs]
    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(_worker, a) for a in args]
        for f in as_completed(futs):
            r = f.result()
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(r, default=float) + "\n")
            log.info("simcheck %s/%s: share %s, %ss", r.get("generator"), r.get("seed"), r.get("share"),
                     r.get("seconds"))
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    summ = summarize(rows)
    (out_dir / "simcheck_summary.json").write_text(json.dumps(summ, indent=1, default=float), encoding="utf-8")
    return summ


def _rate(xs) -> float | None:
    xs = [bool(x) for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def summarize(rows: list[dict], real_r2: float | None = None, r2_tol: float = 0.15) -> dict:
    """Per-generator recovery statistics.  With ``real_r2`` (the real run's
    held-out R²), replicates whose simulated held-out R² differs by more than
    ``r2_tol`` fail the data-matching gate too."""
    ok = [r for r in rows if "error" not in r]
    gens = {}
    for kind in GENERATORS:
        rs = [r for r in ok if r["generator"] == kind]
        if not rs:
            continue
        passed = [r for r in rs if (r.get("gate") or {}).get("pass")
                  and (real_r2 is None or r.get("oof_r2") is None or abs(r["oof_r2"] - real_r2) <= r2_tol)]
        use = passed or rs
        sh = [r["share"] for r in use if r.get("share") is not None]
        rc = [r["rank_corr"] for r in use if r.get("rank_corr") is not None]
        g = {"n": len(rs), "n_gate_pass": len(passed),
             "share_median": float(np.median(sh)) if sh else None,
             "share_iqr": [float(np.percentile(sh, 25)), float(np.percentile(sh, 75))] if sh else None,
             "rank_corr_mean": float(np.mean(rc)) if rc else None,
             "ci_coverage": _rate(r.get("ci_covers_truth") for r in use),
             "causal_ci_coverage": _rate(r.get("causal_covers_truth") for r in use),
             "interval_coverage_mean": float(np.mean([r["interval_coverage"] for r in use
                                                      if r.get("interval_coverage") is not None] or [np.nan])),
             "oof_r2_mean": float(np.mean([r["oof_r2"] for r in use if r.get("oof_r2") is not None] or [np.nan]))}
        if kind == "null":
            g["false_positive_rate"] = _rate(r.get("significant") for r in use)
            g["causal_false_positive_rate"] = _rate(r.get("causal_significant") for r in use)
        gens[kind] = g
    shares = [g["share_median"] for k, g in gens.items() if k != "null" and g.get("share_median") is not None]
    corr = {}
    if shares:
        med = float(np.median(shares))
        spread = (max(shares) - min(shares)) / abs(med) if med else float("inf")
        corr = {"share_median_across_generators": med, "share_range": [min(shares), max(shares)],
                "relative_spread": spread, "stable": bool(spread < 0.25),
                "correction_factor": (1.0 / med) if spread < 0.25 and med else None}
    return {"generators": gens, "bias_correction": corr, "n_rows": len(rows), "n_errors": len(rows) - len(ok)}


def simcheck_markdown(summ: dict) -> str:
    L = ["| generator | n (gate pass) | effect share (IQR) | rank corr | 95% CI covers truth | causal CI covers | "
         "interval coverage | false positives |", "|---|---|---|---|---|---|---|---|"]

    def f(v, nd=2):
        return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"

    for k, g in summ["generators"].items():
        iqr = g.get("share_iqr")
        L.append(f"| {k} | {g['n']} ({g['n_gate_pass']}) | {f(g['share_median'])}"
                 + (f" ({f(iqr[0])}–{f(iqr[1])})" if iqr else "") + f" | {f(g['rank_corr_mean'])} | "
                 f"{f(g['ci_coverage'])} | {f(g['causal_ci_coverage'])} | {f(g['interval_coverage_mean'])} | "
                 + (f"model {f(g.get('false_positive_rate'))}, causal {f(g.get('causal_false_positive_rate'))}"
                    if k == "null" else "") + " |")
    b = summ.get("bias_correction") or {}
    if b:
        L += ["", (f"Effect share is stable across generators (spread {b['relative_spread']:.0%}): city-wide canopy "
                   f"effects can be divided by {b['share_median_across_generators']:.2f}." if b["stable"] else
                   f"Effect share varies across generators ({b['share_range'][0]:.2f}–{b['share_range'][1]:.2f}): "
                   "no single correction; read city-wide canopy effects as a range.")]
    return "\n".join(L)


def merge_results(dirs) -> list[dict]:
    """Rows from several ``simcheck.jsonl`` files (latest result per generator/seed)."""
    rows = {}
    for d in dirs:
        path = Path(d) / "simcheck.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                key = (r.get("generator"), r.get("seed"))
                if "error" not in r or key not in rows:
                    rows[key] = r
    return list(rows.values())
