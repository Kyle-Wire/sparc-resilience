"""Negative controls on real data: does the pipeline find effects that cannot exist?

A model that attributes temperature to the right causes must give ~zero
effect to a layer that has the spatial structure of a real driver but no
physical link to the measured temperature.  Three placebos:

* ``shift``  — canopy and impervious replaced by toroidally shifted copies of
  themselves (half the raster in each direction, several km): same marginal
  distribution and autocorrelation, wrong place.
* ``rotate`` — the same layers rotated 180°.
* ``grf``    — the real data plus an extra covariate: a Gaussian random field
  rank-matched to canopy's distribution (no link to anything).

Each placebo is a full re-fit (S0–S6, coarse resolution by default): the
model's scenario Δ for +½, +1 and +2 sd of the placebo layer (fold-averaged,
jackknife SE) and the causal θ (DML, block-clustered SE) per sd should be
about zero.  The real canopy/impervious effects of the ``grf`` run (same
resolution, same pipeline) give the scale for "about zero": a placebo
*passes* when its 1-sd effect is within 2 SE of zero or smaller than 10% of
the real layer's 1-sd effect.  Spatially structured placebos are hard: a
smooth layer can pick up residual spatial trends, which is exactly what this
test is meant to expose.
"""

from __future__ import annotations

import copy
import logging

import numpy as np
import pandas as pd
from scipy import ndimage

log = logging.getLogger(__name__)

KINDS = ("grf", "shift", "rotate")
GRF = "placebo_grf"


# --------------------------------------------------------------------------- #
# Placebo layers                                                               #
# --------------------------------------------------------------------------- #
def _filled_raster(values: np.ndarray, grid) -> np.ndarray:
    """Raster of ``values`` with empty cells filled by the nearest data cell."""
    r = grid.rasterize(np.asarray(values, float))
    empty = ~np.isfinite(r)
    if empty.any():
        idx = ndimage.distance_transform_edt(empty, return_distances=False, return_indices=True)
        r = r[tuple(idx)]
    return r


def shift_layer(values: np.ndarray, grid, frac: tuple[float, float] = (0.5, 0.5)) -> np.ndarray:
    """Toroidal shift by ``frac`` of the raster in (y, x)."""
    r = _filled_raster(values, grid)
    sy, sx = int(round(frac[0] * r.shape[0])), int(round(frac[1] * r.shape[1]))
    return np.roll(r, (sy, sx), axis=(0, 1))[grid.iy, grid.ix]


def rotate_layer(values: np.ndarray, grid) -> np.ndarray:
    r = _filled_raster(values, grid)
    return r[::-1, ::-1][grid.iy, grid.ix]


def grf_layer(grid, like: np.ndarray, range_m: float, seed: int) -> np.ndarray:
    """Smooth random field at the points, rank-matched to ``like``'s distribution."""
    from sparc.core.synthetic import gaussian_random_field

    z = gaussian_random_field(grid.shape, range_m / grid.dx, np.random.default_rng(seed))[grid.iy, grid.ix]
    out = np.empty_like(z)
    out[np.argsort(z)] = np.sort(np.asarray(like, float))
    return out


def layer_correlation(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


# --------------------------------------------------------------------------- #
# Configs                                                                      #
# --------------------------------------------------------------------------- #
def placebo_config(cfg, kind: str, tested: list[str], sds: dict[str, float], doses_sd=(0.5, 1.0, 2.0)):
    """A copy of ``cfg`` for one placebo run: scenarios are dose ladders in sd
    units of the tested layers, causal treatments are the tested layers, and
    the climate, optimiser, CV-curve and baseline extras are off."""
    c = copy.deepcopy(cfg)
    raw = c.raw
    raw["name"] = f"{cfg.name}_placebo_{kind}"
    if kind == "grf" and GRF not in raw["predictors"]:
        raw["predictors"] = list(raw["predictors"]) + [GRF]
    preds = raw["predictors"]
    act = {}
    for v in tested:
        base = dict((cfg.actionable or {}).get(v) or {})
        lo, hi = base.get("min"), base.get("max")
        act[v] = {**base, "doses": [0.0] + [round(k * sds[v], 6) for k in doses_sd], "direction": "increase"}
        if v == GRF:
            act[v].update(min=None, max=None, unit="placebo units")
        elif lo is not None:
            act[v].update(min=lo, max=hi)
    raw["actionable"] = {k: {kk: vv for kk, vv in s.items() if vv is not None} for k, s in act.items()}
    raw["scenarios"] = [{"name": f"{v} (sd {sds[v]:.3g})", "variable": v, "direction": "increase",
                         "increments": [round(k * sds[v], 6) for k in doses_sd]} for v in tested]
    raw["joint_scenarios"] = []
    meds = set((raw.get("mediators") or {}).keys())
    cz = raw.setdefault("causal", {})
    conf = dict(cz.get("confounders") or {})
    for v in tested:
        if v not in conf:
            conf[v] = [p for p in preds if p != v and p not in meds and p not in tested]
    cz.update(enabled=True, treatments=list(tested), confounders={v: conf[v] for v in tested},
              contrast={v: float(sds[v]) for v in tested}, dag_audit=False)
    raw["climate"] = {**(raw.get("climate") or {}), "enabled": False}
    raw["optimize"] = {**(raw.get("optimize") or {}), "enabled": False}
    raw["cv"]["distance_curve"] = {**(raw["cv"].get("distance_curve") or {}), "enabled": False}
    raw["cv"]["baselines"] = False
    return c


def _clean_input(cfg, frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """The input table with the rows ``prepare_frame`` keeps, in its order."""
    from sparc.core.data import read_input

    df = read_input(cfg) if frame is None else frame
    d = cfg.data
    needed = [d["target"], d["x"], d["y"]] + cfg.predictors
    keep = np.isfinite(df[needed].to_numpy(dtype=float)).all(axis=1)
    return df.loc[keep].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Suite                                                                        #
# --------------------------------------------------------------------------- #
def _effects(res, tested: list[str]) -> dict:
    """Per tested layer: model scenario Δ per dose (sd units) and causal θ per sd."""
    out = {}
    for v in tested:
        ladder = [s for s in res.scenarios if s["name"].startswith(f"{v} (sd ")]
        doses = res.cfg.actionable[v]["doses"][1:]
        sd = float(res.cfg.raw["causal"]["contrast"][v])
        rows = []
        for s, d in zip(ladder, doses):
            rows.append({"dose_sd": float(d) / sd, "mean_delta": float(s["mean_delta"]),
                         "se": float(s.get("mean_delta_se") or np.nan), "frac_extrapolated": s.get("frac_extrapolated")})
        c = (res.manifest.get("causal") or {}).get(v) or {}
        resp = (res.responses or {}).get(v)
        out[v] = {"sd": sd, "model": rows,
                  "causal_theta_sum_per_sd": (c.get("theta_sum") or np.nan) * sd if c else None,
                  "causal_se_per_sd": (c.get("se_sum") or np.nan) * sd if c else None,
                  "causal_theta_own_per_sd": (c.get("theta_own") or np.nan) * sd if c else None,
                  "footprint_per_sd": float(resp.summary["mean_footprint_effect"]) * sd if resp is not None else None}
    return out


def _one_sd(e: dict) -> tuple[float, float]:
    r = min(e["model"], key=lambda r: abs(r["dose_sd"] - 1.0))
    return r["mean_delta"], r["se"]


def judge(placebo: dict, real: dict | None) -> dict:
    d, se = _one_sd(placebo)
    ratio = None
    if real is not None:
        rd, _ = _one_sd(real)
        ratio = abs(d) / abs(rd) if rd else None
    within = bool(np.isfinite(se) and abs(d) <= 2.0 * se)
    small = bool(ratio is not None and ratio < 0.10)
    c, cse = placebo.get("causal_theta_sum_per_sd"), placebo.get("causal_se_per_sd")
    causal_zero = bool(c is not None and cse is not None and np.isfinite(c) and abs(c) <= 1.96 * cse)
    return {"delta_1sd": d, "se_1sd": se, "ratio_to_real": ratio, "model_within_2se": within,
            "model_below_10pct_of_real": small, "model_pass": within or small, "causal_ci_covers_zero": causal_zero}


def run_placebo_suite(cfg, kinds=KINDS, coarse: float | None = 60.0, seed: int = 0,
                      grf_range_m: float = 600.0, write: bool = True, frame: pd.DataFrame | None = None) -> dict:
    """Run the placebo re-fits and tabulate placebo vs real effects."""
    from sparc.core.config import CoreConfig
    from sparc.core.data import prepare_frame
    from sparc.core.pipeline import run_core

    assert isinstance(cfg, CoreConfig)
    roles = (cfg.raw.get("physics") or {}).get("roles") or {}
    layers = [roles[r] for r in ("canopy", "impervious") if roles.get(r) in cfg.predictors]
    if not layers:
        raise ValueError("placebo suite needs physics roles canopy and/or impervious among the predictors")
    df = _clean_input(cfg, frame)
    fine_cfg = copy.deepcopy(cfg)
    fine_cfg.raw["data"]["coarse_m"] = None
    fine_cfg.raw["data"]["subsample"] = None
    fine = prepare_frame(df, fine_cfg)
    stages = ("S0", "S1", "S2", "S3", "S4", "S5", "S6")
    runs, layer_corr = {}, {}
    for kind in kinds:
        d = df.copy()
        if kind == "grf":
            d[GRF] = grf_layer(fine.grid, fine.frame[layers[0]].to_numpy(float), grf_range_m, seed)
            tested = layers + [GRF]
        else:
            for v in layers:
                orig = fine.frame[v].to_numpy(float)
                d[v] = shift_layer(orig, fine.grid) if kind == "shift" else rotate_layer(orig, fine.grid)
                layer_corr[f"{kind}:{v}"] = layer_correlation(orig, d[v].to_numpy(float))
            tested = list(layers)
        probe = copy.deepcopy(cfg)
        if kind == "grf":
            probe.raw["predictors"] = list(probe.raw["predictors"]) + [GRF]
        if coarse:
            probe.raw["data"]["coarse_m"] = float(coarse)
        pdat = prepare_frame(d, probe)
        sds = {v: float(pdat.frame[v].std()) for v in tested}
        pc = placebo_config(cfg, kind, tested, sds)
        if coarse:
            pc.raw["data"]["coarse_m"] = float(coarse)
        log.info("placebo %s: re-fitting (%s)", kind, ", ".join(tested))
        res = run_core(pc, stages=stages, frame=d, write=write)
        runs[kind] = {"effects": _effects(res, tested), "metrics": res.manifest.get("metrics", {}).get("stacker"),
                      "run_dir": str(res.run_dir) if res.run_dir else None}
    real = runs.get("grf", {}).get("effects", {})
    table = []
    for kind, r in runs.items():
        for v, e in r["effects"].items():
            is_placebo = kind != "grf" or v == GRF
            row = {"kind": kind, "variable": v, "placebo": is_placebo, "sd": e["sd"], "model": e["model"],
                   "causal_theta_sum_per_sd": e["causal_theta_sum_per_sd"], "causal_se_per_sd": e["causal_se_per_sd"],
                   "footprint_per_sd": e["footprint_per_sd"]}
            if is_placebo:
                ref = real.get(v) if v in real and kind != "grf" else real.get(layers[0])
                row["verdict"] = judge(e, ref)
                row["reference"] = v if v in real and kind != "grf" else layers[0]
            table.append(row)
    placebos = [r for r in table if r["placebo"]]
    return {"kinds": list(kinds), "coarse_m": coarse, "seed": seed, "layers": layers, "rows": table,
            "layer_correlation_with_original": layer_corr,
            "n_pass_model": int(sum(r["verdict"]["model_pass"] for r in placebos)),
            "n_pass_causal": int(sum(r["verdict"]["causal_ci_covers_zero"] for r in placebos)),
            "n_placebos": len(placebos),
            "runs": {k: {"metrics": v["metrics"], "run_dir": v["run_dir"]} for k, v in runs.items()}}


def placebo_markdown(res: dict, units: str = "") -> str:
    L = ["| run | layer | placebo? | model Δ at +1 sd (± SE) | ratio to real | causal θ per sd (± SE) | verdict |",
         "|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        d, se = _one_sd(r)
        v = r.get("verdict")
        c, cse = r["causal_theta_sum_per_sd"], r["causal_se_per_sd"]
        verdict = "real effect (reference)" if not r["placebo"] else (
            ("model ✓" if v["model_pass"] else "model ✗") + " · " + ("causal ✓" if v["causal_ci_covers_zero"] else "causal ✗"))
        ratio = "" if not v or v["ratio_to_real"] is None else f"{v['ratio_to_real']:.0%}"
        cs = "—" if c is None or not np.isfinite(c) else f"{c:+.3f} ± {cse:.3f}"
        L.append(f"| {r['kind']} | {r['variable']} | {'yes' if r['placebo'] else 'no'} | {d:+.3f} ± {se:.3f} {units} | "
                 f"{ratio} | {cs} | {verdict} |")
    return "\n".join(L)
