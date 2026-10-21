"""Deterministic QA facts for each built lstair city: one JSON record per city (no network).

usage: python -I qa_city.py DATA_DIR OUT_JSON [slug ...]"""
import sys, json, math, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd

PROTOCOL = {"am": (5.5, 8.0), "af": (14.5, 17.0), "pm": (18.5, 21.0), "ni": (23.5, 26.0)}

def hour(ts):
    t = pd.Timestamp(ts)
    return t.hour + t.minute / 60 + t.second / 3600

def corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 30 and np.std(a[ok]) > 0 and np.std(b[ok]) > 0 else None

def r(x, nd=3):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), nd)

data, out_path, only = Path(sys.argv[1]), sys.argv[2], set(sys.argv[3:])
res = {}
for d in sorted((data / "cities").iterdir()):
    if only and d.name not in only:
        continue
    if not (d / "meta.json").exists():
        res[d.name] = {"status": "building"}; continue
    m = json.loads((d / "meta.json").read_text())
    if m.get("status") != "ok":
        res[d.name] = {"status": m.get("status"), "error": m.get("error")}
        continue
    t = pd.read_parquet(d / "table.parquet")
    g = m["grid"]
    rec = {"city": m["city"], "status": "ok", "campaign_date": m["campaign_date"],
           "campaign_date_source": m["campaign_date_source"], "tz": m["tz"], "n_rows": m["n_rows"],
           "centroid_lat": r(t["lat"].median(), 4), "centroid_lon": r(t["lon"].median(), 4),
           "grid_km": [r(g["nx"] * g["dx"] / 1000, 1), r(g["ny"] * g["dx"] / 1000, 1)],
           "warnings": m["warnings"], "n_duplicate_readings": m.get("n_duplicate_readings"),
           "files_read": [f["file"].rsplit("/", 1)[-1] for f in m.get("files", []) if f.get("rows")],
           "features": {k: (v if not isinstance(v, dict) else {kk: (vv if kk != "scenes" else len(vv)) for kk, vv in v.items()})
                        for k, v in (m.get("features") or {}).items()},
           "feature_nan_share": {c: r(t[c].isna().mean()) for c in ("canopy", "impervious", "ndvi", "albedo", "elevation", "water_distance", "lst")},
           "lst": None, "runs": []}
    L = m.get("lst") or {}
    sel = [s for s in (L.get("scenes") or []) if s.get("selected", True)]
    if sel:
        rec["lst"] = {"status": L.get("status"), "scene": sel[0].get("id"), "date": sel[0].get("date"),
                      "offset_days": sel[0].get("offset_days"), "clear_fraction": r(sel[0].get("clear_fraction_lst", sel[0].get("clear_fraction"))),
                      "cloud_cover": sel[0].get("cloud_cover"), "n_inspected": len(L.get("scenes") or []),
                      "lst_p5_p50_p95": [r(np.nanpercentile(t["lst"], q), 2) for q in (5, 50, 95)] if t["lst"].notna().any() else None}
    else:
        rec["lst"] = {"status": L.get("status"), "note": "no scene selected"}
    for run in m["runs"]:
        rr = {k: run.get(k) for k in ("run", "run_id", "window", "start", "end", "n_readings", "vehicles", "error")}
        if run.get("error"):
            rec["runs"].append(rr); continue
        sub = t[t["run_id"] == run["run_id"]]
        h0, h1 = hour(run["start"]), hour(run["end"])
        if run["run"] == "ni" and h0 < 12: h0 += 24
        if run["run"] == "ni" and h1 < 12: h1 += 24
        lo, hi = PROTOCOL.get(run["run"], (None, None))
        dr = run.get("drift") or {}
        e5 = run.get("era5") or {}
        y = sub["dTa"].to_numpy(float)
        rr.update({
            "start_h": r(h0, 2), "end_h": r(h1, 2), "duration_min": r((run.get("qa") or {}).get("duration_min"), 1),
            "inside_protocol": (lo is not None and h0 >= lo - 0.25 and h1 <= hi + 0.25),
            "n_cells": run.get("n_cells"), "share_on_grid": r((run.get("qa") or {}).get("share_on_grid")),
            "median_step_m": r((run.get("qa") or {}).get("median_step_m"), 1),
            "drift_method": dr.get("method"), "drift_per_h": r(dr.get("drift_per_h")), "drift_fallback": dr.get("fallback"),
            "vehicle_offsets_c_range": (r(max(run["vehicle_offsets_c"].values()) - min(run["vehicle_offsets_c"].values()))
                                        if run.get("vehicle_offsets_c") else None),
            "background_c": r(run.get("background_c"), 2), "ta_era5": r(e5.get("ta_era5"), 2),
            "bg_minus_era5": r(run.get("background_c") - e5["ta_era5"], 2) if e5.get("ta_era5") is not None else None,
            "sw_down": r(e5.get("sw_down"), 1), "era5_error": e5.get("error"),
            "dTa_sd": r(np.std(y)), "dTa_p1_p99": [r(np.percentile(y, 1), 2), r(np.percentile(y, 99), 2)],
            "share_abs_dTa_gt5": r(np.mean(np.abs(y) > 5)),
            "r_dTa_impervious": r(corr(y, sub["impervious"].to_numpy(float))),
            "r_dTa_canopy": r(corr(y, sub["canopy"].to_numpy(float))),
            "r_dTa_impervious_f300": r(corr(y, sub["impervious_f300"].to_numpy(float))),
            "r_dTa_lst": r(corr(y, sub["lst"].to_numpy(float))),
            "r_dTa_lst_f300": r(corr(y, sub["lst_f300"].to_numpy(float))),
            "r_lst_impervious": r(corr(sub["lst"].to_numpy(float), sub["impervious"].to_numpy(float))),
            "r_dTa_elevation": r(corr(y, sub["elevation"].to_numpy(float))),
            "lst_offset_days": r(sub["lst_offset_days"].iloc[0] if len(sub) else None, 0),
        })
        rec["runs"].append(rr)
    res[d.name] = rec
json.dump(res, open(out_path, "w"), indent=1)
for k, v in res.items():
    if v.get("status") != "ok":
        print(k, v.get("status"), (v.get("error") or "")[:120]); continue
    print(k, v["campaign_date"], v["centroid_lat"], v["centroid_lon"], "lst", (v["lst"] or {}).get("offset_days"), (v["lst"] or {}).get("clear_fraction"),
          " | ".join(f"{x['run']} {x.get('start_h')}-{x.get('end_h')} n={x.get('n_cells')} sd={x.get('dTa_sd')} rImp={x.get('r_dTa_impervious')} rLST={x.get('r_dTa_lst')}" for x in v["runs"]))
