"""Build the interactive results page for a finished core run.

    python scripts/results_page/build_page.py <run dir> <core config> [--out results.html]

Reads the run directory written by ``sparc core run`` (manifest.json,
predictions.parquet, response_*.parquet, scenario_deltas.parquet,
allocation.parquet, optimize.json, checkpoint.pkl for the CV folds) and writes
one self-contained HTML page: a map explorer (temperature, land cover, cooling
footprints, saturation, scenarios, budget plan, CV folds) plus charts and
tables for accuracy vs distance, area of influence, dose-response, scenarios,
the causal audit and the budget plan.

Page text comes from the config's optional ``report`` block::

    report:
      title: Providence Heat Model
      place: Providence, RI
      area: the Brown University / Providence study area
      caveats: ["..."]          # appended to the data-driven caveats
"""

from __future__ import annotations

import argparse
import base64
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def _enc(v, kind: str = "u16") -> dict:
    """Quantise a per-point layer (0 = missing) with its range and percentiles."""
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    lo, hi = (float(np.nanmin(v)), float(np.nanmax(v))) if ok.any() else (0.0, 1.0)
    if hi <= lo:
        hi = lo + 1e-9
    top = 254 if kind == "u8" else 65534
    q = np.zeros(v.size, np.uint8 if kind == "u8" else np.uint16)
    q[ok] = 1 + np.rint((v[ok] - lo) / (hi - lo) * top).astype(q.dtype)
    p = [float(x) for x in np.nanpercentile(v[ok], [1, 2, 50, 98, 99])] if ok.any() else [0.0] * 5
    return {"kind": kind, "lo": lo, "hi": hi, "b64": _b64(q), "p": p,
            "mean": float(np.nanmean(v)) if ok.any() else None}


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def collect(run: Path, cfg) -> dict:
    from sparc.core.data import load_core_data

    data = load_core_data(cfg)
    g = data.grid
    m = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    pred = pd.read_parquet(run / "predictions.parquet")
    if not np.array_equal(pred["id"].to_numpy(), data.ids):
        raise SystemExit("predictions.parquet does not match the config's data (different subsample or file?)")
    with open(run / "checkpoint.pkl", "rb") as fh:
        folds = pickle.load(fh)["folds"]

    L: dict = {
        "obs": _enc(pred["target"]),
        "pred": _enc(pred["pred"]),
        "err": _enc(pred["target"] - pred["pred"]),
        "hw": _enc(pred["pi_hi"] - pred["pred"]),
    }
    roles = (cfg.raw.get("physics") or {}).get("roles") or {}
    for role in ("canopy", "impervious", "ndvi", "albedo"):
        col = roles.get(role)
        if col and col in data.frame:
            L["in_" + col] = _enc(data.frame[col], "u8")
    fold = folds.fold_id.astype(np.uint8)
    excl = np.zeros(data.n, np.uint8)
    for k in range(folds.n_folds):
        excl |= ((~folds.train_masks[k] & ~folds.test_masks[k]).astype(np.uint8) << k)

    canopy = roles.get("canopy")
    for v in cfg.actionable:
        f = run / f"response_{v}.parquet"
        if not f.exists():
            continue
        r = pd.read_parquet(f)
        L[f"fp_{v}"] = _enc(r["footprint_effect_per_unit"])
        if v == canopy:
            L[f"own_{v}"] = _enc(r["own_effect_per_unit"])
        L[f"marg_{v}"] = _enc(r["marginal_benefit_per_unit"])
        L[f"A_{v}"] = _enc(r["max_cooling_A"])
        L[f"d90_{v}"] = _enc(r["d90"])
        cm = r["curve_model"].astype(str).to_numpy()
        cls = np.where(r["censored"].to_numpy(bool), 3,
                       np.where(cm == "saturating", 1, np.where(cm == "linear", 2, 0))).astype(np.uint8)
        L[f"cls_{v}"] = {"kind": "cat", "b64": _b64(cls), "labels": ["no fit", "saturating", "linear", "censored"]}
    scen_layers = []
    if (run / "scenario_deltas.parquet").exists():
        d = pd.read_parquet(run / "scenario_deltas.parquet")
        for c in d.columns:
            if c != "id":
                key = f"sc_{len(scen_layers)}"
                L[key] = _enc(d[c])
                scen_layers.append({"key": key, "name": c})
    if (run / "allocation.parquet").exists():
        a = pd.read_parquet(run / "allocation.parquet")
        L["alloc_dose"] = _enc(a["dose"])
        L["alloc_delta"] = _enc(a["closed_loop_delta"])

    corners = None
    crs = cfg.data.get("crs")
    if crs:
        from pyproj import Transformer

        tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        s = cfg.coord_scale
        x1, y1 = g.x0 + (g.nx - 1) * g.dx, g.y0 + (g.ny - 1) * g.dy

        def ll(xm, ym):
            lon, lat = tr.transform(xm / s, ym / s)
            return [lat, lon]

        corners = {"sw": ll(g.x0, g.y0), "se": ll(x1, g.y0), "nw": ll(g.x0, y1), "ne": ll(x1, y1)}

    def jl(name):
        p = run / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    rep = cfg.raw.get("report") or {}
    return _clean({
        "name": m["name"], "created": m["created_utc"], "commit": m.get("git_commit"), "n": m["n_points"],
        "timings": m.get("timings_s"), "qa": m.get("qa"), "influence": m.get("influence"), "cv": m.get("cv"),
        "metrics": m.get("metrics"), "lambda_pde": m.get("lambda_pde"), "lambda_scores": m.get("lambda_scores"),
        "stacker": m.get("stacker"), "stacker_choice": m.get("stacker_choice"), "spatial_plus": m.get("spatial_plus"),
        "physics": m.get("physics"), "cv_distance": m.get("cv_distance"),
        "response": m.get("response"), "curves": jl("response_curves.json"), "scenarios": m.get("scenarios"),
        "causal": m.get("causal"), "optimize": jl("optimize.json") or m.get("optimize"),
        "climate": m.get("climate"),
        "actionable": cfg.actionable, "units": data.target_units, "background": float(data.background),
        "geom": {"nx": g.nx, "ny": g.ny, "dx": g.dx, "n": int(data.n), "corners": corners,
                 "ix": _b64(g.ix.astype(np.uint16)), "iy": _b64(g.iy.astype(np.uint16))},
        "layers": L, "fold": _b64(fold), "excl": _b64(excl), "n_folds": int(folds.n_folds),
        "scen_layers": scen_layers, "caveats_extra": list(rep.get("caveats") or []),
    })


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir")
    ap.add_argument("config")
    ap.add_argument("--out", default=None, help="output HTML (default: <run dir>/results.html)")
    args = ap.parse_args(argv)

    from sparc.core.config import load_core_config

    cfg = load_core_config(args.config)
    run = Path(args.run_dir)
    blob = collect(run, cfg)
    rep = cfg.raw.get("report") or {}
    html = (HERE / "template.html").read_text(encoding="utf-8")
    html = (html.replace("{{TITLE}}", str(rep.get("title", "Urban Heat Model")))
                .replace("{{PLACE}}", str(rep.get("place", cfg.name)))
                .replace("{{AREA}}", str(rep.get("area", "the study area"))))
    payload = json.dumps(blob, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    html = html.replace("/*__DATA__*/", payload)
    out = Path(args.out) if args.out else run / "results.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
