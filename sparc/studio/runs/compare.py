"""Compare two runs (SPEC §6.8, api.md §6.5): config diff, provenance chips, metrics, timings, scenario effects,
climate and causal summaries, environment diff, outputs only in one run; and, for runs on the same grid
(same cell size and ids), a difference layer and the agreement of priority layers (Kendall τ and top-decile
Jaccard).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from sparc.studio.errors import ApiError
from sparc.studio.runs.common import clean, fnum, likely

__all__ = ["compare_runs", "same_grid", "layer_diff", "priority_agreement", "environment_packages", "env_diff",
           "flatten", "config_diff"]


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Dotted paths of a nested dict (lists are leaf values)."""
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict) and v:
                out.update(flatten(v, p))
            else:
                out[p] = v
    else:
        out[prefix] = obj
    return out


def config_diff(a: dict, b: dict, *, a_key: str = "a", b_key: str = "b") -> list[dict]:
    fa, fb = flatten(a or {}), flatten(b or {})
    out = []
    for p in sorted(set(fa) | set(fb)):
        va, vb = fa.get(p), fb.get(p)
        if json.dumps(va, sort_keys=True, default=str) != json.dumps(vb, sort_keys=True, default=str):
            out.append({"path": p, a_key: clean(va), b_key: clean(vb)})
    return out


def environment_packages(ctx) -> list[str]:
    p = ctx.run_dir / "environment.txt"
    if not p.exists():
        return []
    return [ln.strip() for ln in p.read_text("utf-8", errors="replace").splitlines()
            if ln.strip() and not ln.startswith("#")]


def _pkgs(lines: list[str]) -> dict[str, str]:
    out = {}
    for ln in lines:
        for sep in ("==", " @ ", "==="):
            if sep in ln:
                name, ver = ln.split(sep, 1)
                out[name.strip().lower()] = ver.strip()
                break
        else:
            out[ln.strip().lower()] = ""
    return out


def env_diff(a_lines: list[str], b_lines: list[str]) -> dict:
    a, b = _pkgs(a_lines), _pkgs(b_lines)
    return {"added": sorted(k for k in b if k not in a), "removed": sorted(k for k in a if k not in b),
            "changed": [{"name": k, "a": a[k], "b": b[k]} for k in sorted(a) if k in b and a[k] != b[k]]}


def same_grid(ca, cb) -> bool:
    ga, gb = ca.grid, cb.grid
    if ga is None or gb is None or ga.n != gb.n or abs(ga.dx - gb.dx) > 1e-6:
        return False
    return bool(np.array_equal(np.asarray(ga.ids).astype(str), np.asarray(gb.ids).astype(str)))


def _metrics(ctx) -> dict[str, float | None]:
    m = ctx.metrics or {}
    st = m.get("stacker") or {}
    out = {"r2": fnum(st.get("r2")), "rmse": fnum(st.get("rmse")), "mae": fnum(st.get("mae")),
           "bias": fnum(st.get("bias")), "coverage": fnum(st.get("interval_coverage")),
           "halfwidth": fnum(st.get("interval_mean_halfwidth")), "n_points": fnum(ctx.n)}
    for k, v in m.items():
        if k != "stacker" and isinstance(v, dict):
            out[f"r2:{k}"] = fnum(v.get("r2"))
    return out


def _scen(ctx) -> dict[str, dict]:
    sc = (ctx.manifest or {}).get("scenarios")
    if not isinstance(sc, list):
        return {}
    unit = ctx.units.get("target")
    return {str(s["name"]): likely(s.get("mean_delta"), s.get("mean_delta_se"), unit)
            for s in sc if isinstance(s, dict) and s.get("name") is not None and s.get("mean_delta") is not None}


def _climate(ca, cb) -> dict | None:
    def proj(ctx):
        c = (ctx.manifest or {}).get("climate") or {}
        return {f"{p.get('experiment')}:{p.get('period')}": fnum((p.get("warming") or {}).get("median"))
                for p in c.get("projections") or []}

    pa, pb = proj(ca), proj(cb)
    if not pa and not pb:
        return None
    return {"warming_median": [{"id": k, "a": pa.get(k), "b": pb.get(k)} for k in sorted(set(pa) | set(pb))]}


def _causal(ca, cb) -> dict | None:
    def tr(ctx):
        c = (ctx.manifest or {}).get("causal") or {}
        out = {}
        for t, d in (c.get("treatments") or {}).items():
            sp = (d or {}).get("spillover") or {}
            out[t] = {"theta_sum": fnum(sp.get("theta_sum")), "se_sum": fnum(sp.get("se_sum")),
                      "verdicts": {k: v.get("verdict") for k, v in ((d or {}).get("audit") or {}).items()
                                   if isinstance(v, dict)}}
        return out

    ta, tb = tr(ca), tr(cb)
    if not ta and not tb:
        return None
    return {"treatments": [{"treatment": t, "a": ta.get(t), "b": tb.get(t)} for t in sorted(set(ta) | set(tb))]}


def _present_outputs(ctx) -> set[str]:
    from sparc.studio.runs.outputs import output_entries

    return {e["id"] for e in output_entries(ctx) if e["state"] in ("present", "stale", "partial")}


def compare_runs(ca, cb, sa: dict, sb: dict) -> dict:
    """``GET /api/compare/runs`` (``sa``/``sb``: the runs' summaries)."""
    ma, mb = ca.manifest_raw or {}, cb.manifest_raw or {}
    pa, pb = ma.get("provenance") or {}, mb.get("provenance") or {}
    ka, kb = ca.checkpoint_json or {}, cb.checkpoint_json or {}

    def same(x, y):
        return None if not x or not y else x == y

    data = same(pa.get("input_sha256"), pb.get("input_sha256"))
    if data is None:
        data = same((ka.get("sections") or {}).get("data"), (kb.get("sections") or {}).get("data"))
    code = same(ka.get("code_sha256") or pa.get("code_sha256"), kb.get("code_sha256") or pb.get("code_sha256"))
    config = same(pa.get("config_sha256"), pb.get("config_sha256"))
    if config is None and ka.get("sections") and kb.get("sections"):
        secs = ("core", "s4", "s5", "climate", "s6", "s7")
        config = all(ka["sections"].get(s) == kb["sections"].get(s) for s in secs)
    met_a, met_b = _metrics(ca), _metrics(cb)
    metrics = [{"key": k, "a": met_a.get(k), "b": met_b.get(k),
                "delta": (met_b[k] - met_a[k]) if met_a.get(k) is not None and met_b.get(k) is not None else None}
               for k in list(dict.fromkeys(list(met_a) + list(met_b)))]
    ta, tb = ma.get("timings_s") or {}, mb.get("timings_s") or {}
    timings = [{"stage": s, "a": fnum(ta.get(s)), "b": fnum(tb.get(s))} for s in dict.fromkeys(list(ta) + list(tb))]
    sca, scb = _scen(ca), _scen(cb)
    scenarios = [{"name": n, "a": sca.get(n), "b": scb.get(n)} for n in dict.fromkeys(list(sca) + list(scb))]
    oa, ob = _present_outputs(ca), _present_outputs(cb)
    return clean({
        "a": sa, "b": sb, "same": {"data": data, "config": config, "code": code, "grid": same_grid(ca, cb)},
        "config_diff": config_diff(ca.cfg_raw, cb.cfg_raw), "metrics": metrics, "timings": timings,
        "scenarios": scenarios, "climate": _climate(ca, cb), "causal": _causal(ca, cb),
        "environment": env_diff(environment_packages(ca), environment_packages(cb)),
        "outputs": {"a_only": sorted(oa - ob), "b_only": sorted(ob - oa)},
    })


def _need_same(ca, cb) -> None:
    if not same_grid(ca, cb):
        raise ApiError("grid_mismatch", "the two runs are not on the same grid (cell size or ids differ)",
                       detail={"a": ca.run_id, "b": cb.run_id})


def layer_diff(ca, cb, key: str) -> np.ndarray:
    """``b − a`` of a layer both runs have (Float32), ``409 grid_mismatch`` on different grids."""
    from sparc.studio.runs.layers import layer_array

    _need_same(ca, cb)
    a = np.asarray(layer_array(ca, key), dtype=np.float32)
    b = np.asarray(layer_array(cb, key), dtype=np.float32)
    return (b - a).astype(np.float32)


def priority_agreement(ca, cb, layer: str) -> dict:
    """Kendall τ and top-decile Jaccard of a priority layer (``fp_<v>``: more negative = higher priority)."""
    from scipy.stats import kendalltau

    from sparc.studio.runs.layers import layer_array, layer_defs

    _need_same(ca, cb)
    a = np.asarray(layer_array(ca, layer), dtype=np.float64)
    b = np.asarray(layer_array(cb, layer), dtype=np.float64)
    d = layer_defs(ca).get(layer)
    sign = -1.0 if d is not None and (d.sign_note or "").startswith("negative = cooler") else 1.0
    ok = np.isfinite(a) & np.isfinite(b)
    n = int(ok.sum())
    if n < 3:
        raise ApiError("validation", f"layer {layer!r} has fewer than three cells in both runs",
                       detail={"errors": [{"path": "layer", "message": "too few cells", "code": "priority"}]})
    pa, pb = sign * a[ok], sign * b[ok]
    tau = kendalltau(pa, pb).statistic
    k = max(1, int(round(0.1 * n)))
    ta = set(np.argsort(-pa, kind="stable")[:k].tolist())
    tb = set(np.argsort(-pb, kind="stable")[:k].tolist())
    jac = len(ta & tb) / len(ta | tb)
    return {"kendall_tau": float(tau) if np.isfinite(tau) else 0.0, "top_decile_jaccard": float(jac), "n": n}
