"""Multiverse analysis: do the conclusions survive reasonable analysis choices?

Every pipeline has forks a reviewer could have taken differently: CV block
size, the forcing, the mediator, whether the physics model is in the stack,
Spatial+, the neighbourhood scales, the correlogram horizon, the model set.
Each variant re-runs S0–S5 at coarse resolution with one choice changed;
the report then asks two questions a planner cares about:

* **Effect stability** — does every scenario keep its sign (and roughly its
  size) across variants?
* **Priority stability** — does the map of where to act stay the same?  For
  each lever, cells are ranked by fold-averaged marginal benefit per unit
  (S4); Kendall's τ against the baseline ranking and the Jaccard overlap of
  the top-decile cells measure agreement (τ ≥ 0.6 and Jaccard ≥ 0.6 =
  stable).
"""

from __future__ import annotations

import copy
import json
import logging
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

VARIANTS: dict[str, dict] = {
    "baseline": {},
    "blocks_1km": {"cv.block_m": 1000.0},
    "blocks_3km": {"cv.block_m": 3000.0},
    "generic_forcing": {"physics.sw_down": 800.0, "physics.lw_net": -100.0, "physics.wind": None,
                        "physics.forcing": None, "physics.forcing_info": None},
    "no_mediator": {"mediators": {}},
    "no_physics": {"models.physics": False},
    "spatial_plus": {"models.spatial_plus": ["mgwr"]},
    "focal_scale_1": {"influence.scales": [1.0]},
    "lag_1km": {"influence.max_lag_m": 1000.0},
    "no_gwrf": {"models.gwrf": False},
}
LABELS = {"baseline": "baseline", "blocks_1km": "1 km CV blocks", "blocks_3km": "3 km CV blocks",
          "generic_forcing": "generic forcing (no campaign day)", "no_mediator": "no NDVI mediator",
          "no_physics": "no physics model", "spatial_plus": "Spatial+ (MGWR)",
          "focal_scale_1": "one neighbourhood scale", "lag_1km": "1 km correlogram horizon",
          "no_gwrf": "no GW random forest"}


def apply_variant(cfg, changes: dict, name: str):
    c = copy.deepcopy(cfg)
    for key, val in changes.items():
        node = c.raw
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        if val is None and parts[-1] in ("forcing", "forcing_info"):
            node.pop(parts[-1], None)
        else:
            node[parts[-1]] = copy.deepcopy(val)
    c.raw["name"] = f"{cfg.name}_mv_{name}"
    c.raw["climate"] = {**(c.raw.get("climate") or {}), "enabled": False}
    c.raw["optimize"] = {**(c.raw.get("optimize") or {}), "enabled": False}
    c.raw["cv"]["distance_curve"] = {**(c.raw["cv"].get("distance_curve") or {}), "enabled": False}
    c.raw["cv"]["baselines"] = False
    return c


def run_variant(cfg_raw: dict, base_dir: str, name: str, coarse: float | None, out_dir: str, threads: int = 1) -> dict:
    import torch

    from sparc.core.config import core_config_from_dict
    from sparc.core.pipeline import run_core

    torch.set_num_threads(threads)
    t0 = time.time()
    cfg = core_config_from_dict(cfg_raw, base_dir=base_dir)
    vc = apply_variant(cfg, VARIANTS[name], name)
    if coarse:
        vc.raw["data"]["coarse_m"] = float(coarse)
    res = run_core(vc, stages=("S0", "S1", "S2", "S3", "S4", "S5"), write=True, resume=True)
    maps = {v: r.maps["marginal_benefit_per_unit"].to_numpy(float) for v, r in res.responses.items()}
    np.savez_compressed(Path(out_dir) / f"{name}_maps.npz", ids=res.data.ids, **maps)
    out = {"variant": name, "label": LABELS.get(name, name), "seconds": round(time.time() - t0, 1),
           "r2": res.manifest["metrics"]["stacker"]["r2"], "rmse": res.manifest["metrics"]["stacker"]["rmse"],
           "block_m": res.manifest["cv"]["block_m"], "stacker_choice": res.manifest.get("stacker_choice"),
           "scenarios": {s["name"]: {"mean_delta": s["mean_delta"], "se": s.get("mean_delta_se"),
                                     "frac_extrapolated": s.get("frac_extrapolated")} for s in res.scenarios},
           "run_dir": str(res.run_dir)}
    (Path(out_dir) / f"{name}.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    return out


def run_multiverse(cfg, out_dir, variants=None, coarse: float | None = 60.0, workers: int = 1, threads: int = 1) -> dict:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(variants or VARIANTS)
    todo = [n for n in names if not (out_dir / f"{n}.json").exists()]
    log.info("multiverse: %d variants to run (%d done)", len(todo), len(names) - len(todo))
    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(run_variant, cfg.raw, str(cfg.base_dir), n, coarse, str(out_dir), threads): n for n in todo}
        for f in as_completed(futs):
            try:
                r = f.result()
                log.info("multiverse %s: R² %.3f (%ss)", r["variant"], r["r2"], r["seconds"])
            except Exception:                   # noqa: BLE001 - recorded, others continue
                log.exception("multiverse variant %s failed", futs[f])
    summ = summarize(out_dir, names)
    (out_dir / "multiverse_summary.json").write_text(json.dumps(summ, indent=1, default=float), encoding="utf-8")
    return summ


def rank_agreement(a: np.ndarray, b: np.ndarray, top: float = 0.10) -> dict:
    from scipy.stats import kendalltau

    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    k = max(int(round(top * a.size)), 1)
    ta, tb = set(np.argsort(-np.abs(a))[:k]), set(np.argsort(-np.abs(b))[:k])
    return {"kendall_tau": float(kendalltau(np.abs(a), np.abs(b)).statistic),
            "top_decile_jaccard": len(ta & tb) / len(ta | tb)}


def summarize(out_dir, names=None) -> dict:
    out_dir = Path(out_dir)
    names = names or [p.stem for p in out_dir.glob("*.json") if not p.stem.startswith("multiverse")]
    runs = {}
    for n in names:
        p = out_dir / f"{n}.json"
        if p.exists():
            runs[n] = json.loads(p.read_text(encoding="utf-8"))
    if "baseline" not in runs:
        return {"runs": runs, "note": "baseline variant missing"}
    base = runs["baseline"]
    bmaps = np.load(out_dir / "baseline_maps.npz")
    # effect stability
    effects = {}
    for sname, b in base["scenarios"].items():
        vals = {n: r["scenarios"][sname]["mean_delta"] for n, r in runs.items() if sname in r["scenarios"]}
        signs = [np.sign(v) == np.sign(b["mean_delta"]) for v in vals.values()]
        effects[sname] = {"baseline": b["mean_delta"], "min": float(min(vals.values())), "max": float(max(vals.values())),
                          "sd_across": float(np.std(list(vals.values()))), "sign_stability": float(np.mean(signs)),
                          "values": vals}
    # priority stability
    priority = {}
    for n, r in runs.items():
        if n == "baseline" or not (out_dir / f"{n}_maps.npz").exists():
            continue
        vm = np.load(out_dir / f"{n}_maps.npz")
        if not np.array_equal(vm["ids"], bmaps["ids"]):
            continue
        priority[n] = {v: rank_agreement(bmaps[v], vm[v]) for v in bmaps.files if v != "ids" and v in vm.files}
    stab = [x for d in priority.values() for x in d.values()]
    return {
        "runs": {n: {k: r[k] for k in ("label", "r2", "rmse", "block_m", "stacker_choice", "seconds")} for n, r in runs.items()},
        "effects": effects, "priority": priority,
        "sign_stability_min": float(min(e["sign_stability"] for e in effects.values())) if effects else None,
        "median_kendall_tau": float(np.median([x["kendall_tau"] for x in stab])) if stab else None,
        "median_top_decile_jaccard": float(np.median([x["top_decile_jaccard"] for x in stab])) if stab else None,
    }


def multiverse_markdown(summ: dict, units: str = "") -> str:
    if "effects" not in summ:
        return summ.get("note", "")
    runs = summ["runs"]
    names = list(runs)
    L = ["| scenario | " + " | ".join(runs[n]["label"] for n in names) + " | sign stable |",
         "|---|" + "---|" * len(names) + "---|"]
    for s, e in summ["effects"].items():
        L.append(f"| {s} | " + " | ".join(f"{e['values'][n]:+.2f}" if n in e["values"] else "—" for n in names)
                 + f" | {e['sign_stability']:.0%} |")
    L += ["", "| variant | held-out R² | " + " | ".join(
        f"{v}: τ / top-10% overlap" for v in next(iter(summ["priority"].values()), {})) + " |",
        "|---|---|" + "---|" * len(next(iter(summ["priority"].values()), {}))]
    for n, d in summ["priority"].items():
        L.append(f"| {runs[n]['label']} | {runs[n]['r2']:.3f} | " + " | ".join(
            f"{x['kendall_tau']:.2f} / {x['top_decile_jaccard']:.2f}" for x in d.values()) + " |")
    L += ["", f"Baseline held-out R² {runs['baseline']['r2']:.3f}. Scenario values in {units}. Priority ranking = "
              "fold-averaged marginal benefit per unit (S4); τ = Kendall rank correlation with the baseline map; "
              "overlap = Jaccard index of the top-decile cells."]
    return "\n".join(L)
