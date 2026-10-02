"""Analysis tools of the run hub (SPEC §6.5, api.md §6.2, §6.4): region stats with a fold jackknife, breakdown,
the relationships hexbin, the correlogram and hexagon aggregates.

**Jackknife.**  A scenario's per-fold deltas ``Δ_k`` (K × n: ``scenario_detail.npz`` for configured
scenarios, ``results/<res_id>/folds.npy`` for exact results) give the SE of a region mean as
``std_k(mean_{i∈region} Δ_ki)·√(K−1)`` - exactly core's ``ScenarioResult.summary`` on the masked columns.
People-weighted means weight both the estimate and each fold mean by residents.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from sparc.studio.errors import ApiError
from sparc.studio.runs.common import clean, jackknife_se, likely

__all__ = ["scenario_arrays", "masked_likely", "region_stats", "breakdown", "hexbin", "acf", "hex_table",
           "weighted_mean"]


def weighted_mean(v: np.ndarray, w: np.ndarray | None) -> float | None:
    ok = np.isfinite(v) if w is None else (np.isfinite(v) & np.isfinite(w) & (w > 0))
    if not ok.any():
        return None
    if w is None:
        return float(v[ok].mean())
    return float(np.sum(v[ok] * w[ok]) / np.sum(w[ok]))


def _weighted_sd(v: np.ndarray, w: np.ndarray | None) -> float | None:
    m = weighted_mean(v, w)
    if m is None:
        return None
    ok = np.isfinite(v) if w is None else (np.isfinite(v) & np.isfinite(w) & (w > 0))
    if ok.sum() < 2:
        return 0.0
    if w is None:
        return float(np.std(v[ok]))
    return float(math.sqrt(np.sum(w[ok] * (v[ok] - m) ** 2) / np.sum(w[ok])))


def scenario_arrays(ctx, ref: str) -> tuple[np.ndarray, np.ndarray | None, str]:
    """``(delta[n], folds[K, n] | None, label)`` of a scenario reference (``configured:<slug>`` or ``res_…``)."""
    import pandas as pd

    from sparc.studio.runs import layers as L

    ref = str(ref)
    if ref.startswith("configured:"):
        slug = ref.split(":", 1)[1]
        match = next((s for s in ctx.configured_scenarios() if s["slug"] == slug), None)
        if match is None:
            raise ApiError("validation", f"no configured scenario {slug!r} in this run",
                           detail={"errors": [{"path": "scenarios", "message": f"unknown scenario {ref}",
                                               "code": "unknown_scenario"}]})
        delta = np.asarray(L.layer_array(ctx, f"sc:{slug}"), dtype=np.float64)
        det = ctx.scenario_detail()
        folds = None
        if det and match["name"] in det["folds"]:
            folds = np.asarray(det["folds"][match["name"]], dtype=np.float64)
        return delta, folds, match["name"]
    rid = ref[len("result:"):] if ref.startswith("result:") else ref
    rid = rid.split(":")[0]
    rdir = ctx.studio_dir / "results" / rid
    if not (rdir / "cells.parquet").exists():
        raise ApiError("validation", f"no exact result {rid!r} on this run",
                       detail={"errors": [{"path": "scenarios", "message": f"unknown result {ref}",
                                           "code": "unknown_result"}]})
    delta = np.asarray(L._aligned(ctx, pd.read_parquet(rdir / "cells.parquet"), "delta"), dtype=np.float64)
    folds = None
    if (rdir / "folds.npy").exists():
        f = np.load(rdir / "folds.npy", allow_pickle=False)
        if f.ndim == 2 and f.shape[1] == delta.size:
            folds = np.asarray(f, dtype=np.float64)
    from sparc.studio.runs.common import read_json_cached

    summ = read_json_cached(rdir / "summary.json") or {}
    label = ((summ.get("scenario") or {}).get("name")) or rid
    return delta, folds, label


def masked_likely(delta: np.ndarray, folds: np.ndarray | None, mask: np.ndarray, w: np.ndarray | None,
                  unit: str, what: str = "") -> dict | None:
    """Mean ΔT over ``mask`` with the fold jackknife SE (weights ``w``) as a ``Likely``."""
    if not mask.any():
        return None
    ww = None if w is None else w[mask]
    est = weighted_mean(delta[mask], ww)
    se = None
    if folds is not None and folds.shape[0] > 1:
        se = jackknife_se([weighted_mean(f[mask], ww) for f in folds])
    return likely(est, se, unit, what=what)


def _people(ctx) -> np.ndarray | None:
    from sparc.studio.runs import layers as L

    if "people" not in L.layer_defs(ctx):
        return None
    try:
        return np.asarray(L.layer_array(ctx, "people"), dtype=np.float64)
    except ApiError:
        return None


def region_stats(ctx, src, mask: np.ndarray, layers: list[str], weights: str | None,
                 scenarios: list[str] | None) -> dict:
    """``POST /stats/region`` (api.md §6.4)."""
    from sparc.studio.runs.selection import column_values

    g = src.grid
    people = _people(ctx)
    w = people if weights == "people" and people is not None else None
    if weights == "people" and people is None:
        raise ApiError("needs_layers", "people weights need the planner layers (planner.layers)",
                       action={"kind": "fetch_input", "label": "Fetch people and land cover", "method": "POST",
                               "path": f"/api/projects/{ctx.project_id}/inputs/layers"} if ctx.project_id else None)
    out: dict[str, Any] = {"n_cells": int(mask.sum()), "area_km2": float(mask.sum() * g.dx * g.dx / 1e6),
                           "people": float(np.nansum(people[mask])) if people is not None else None,
                           "layers": {}, "scenarios": {}}
    for key in layers or []:
        v = column_values(ctx, key)
        inside = v[mask]
        ok = np.isfinite(inside)
        q = np.percentile(inside[ok], [10, 50, 90]) if ok.any() else [None] * 3
        out["layers"][key] = {"mean": weighted_mean(inside, None if w is None else w[mask]),
                              "sd": _weighted_sd(inside, None if w is None else w[mask]),
                              "p10": q[0], "p50": q[1], "p90": q[2],
                              "mean_outside": weighted_mean(v[~mask], None if w is None else w[~mask])}
    unit = ctx.units.get("target", "°F")
    for ref in scenarios or []:
        delta, folds, _label = scenario_arrays(ctx, ref)
        inside = masked_likely(delta, folds, mask, w, unit, "the selection")
        outside = masked_likely(delta, folds, ~mask, w, unit, "outside the selection")
        empty = likely(0.0, None, unit)
        out["scenarios"][ref] = {"inside": inside or empty, "outside": outside or empty,
                                 "has_folds": folds is not None}
    return clean(out)


def _groups_for(ctx, src, by: dict) -> tuple[np.ndarray, list[str]]:
    """``(group index per row (-1 = none), labels)`` of a breakdown."""
    from sparc.core.planner import hex_ids
    from sparc.studio.runs import layers as L
    from sparc.studio.runs.selection import column_values

    g = src.grid
    kind = by.get("kind")
    if kind == "zone":
        if not g.zones:
            raise ApiError("validation", "this run has no zones",
                           detail={"errors": [{"path": "by.kind", "message": "no zones", "code": "no_zones"}]})
        return g.zone.astype(np.int64), [str(z) for z in g.zones]
    if kind == "fold":
        pred = ctx.predictions
        if pred is not None and "fold" in pred.columns:
            f = pred["fold"].to_numpy().astype(np.int64)
        elif ctx.folds is not None:
            f = ctx.folds.fold_id.astype(np.int64)
        else:
            raise ApiError("validation", "the CV design is not known yet",
                           detail={"errors": [{"path": "by.kind", "message": "no folds", "code": "no_folds"}]})
        k = int(f.max()) + 1 if f.size else 0
        return f, [f"Fold {i + 1}" for i in range(k)]
    if kind == "hex":
        size = float(by.get("size_m") or 250)
        keys = hex_ids(g.x, g.y, size)[0]
        uniq, inv = np.unique(keys, return_inverse=True)
        return inv.astype(np.int64), [str(int(k)) for k in uniq]
    if kind == "quantile":
        layer = by.get("layer")
        if not layer:
            raise ApiError("validation", "a quantile breakdown needs a layer",
                           detail={"errors": [{"path": "by.layer", "message": "required", "code": "missing"}]})
        q = int(by.get("q") or 5)
        q = max(2, min(q, 20))
        v = column_values(ctx, layer)
        ok = np.isfinite(v)
        if not ok.any():
            return np.full(v.size, -1, np.int64), []
        edges = np.unique(np.percentile(v[ok], np.linspace(0, 100, q + 1)))
        idx = np.full(v.size, -1, np.int64)
        idx[ok] = np.clip(np.searchsorted(edges, v[ok], side="right") - 1, 0, len(edges) - 2)
        labels = [f"Q{i + 1} ({edges[i]:.3g}–{edges[i + 1]:.3g})" for i in range(len(edges) - 1)]
        return idx, labels
    if kind == "category":
        layer = by.get("layer")
        if not layer:
            raise ApiError("validation", "a category breakdown needs a layer",
                           detail={"errors": [{"path": "by.layer", "message": "required", "code": "missing"}]})
        defs = L.layer_defs(ctx)
        v = column_values(ctx, layer)
        ok = np.isfinite(v) & (v != 255 if layer in defs and defs[layer].scale == "cat" else True)
        uniq = np.unique(v[ok])
        idx = np.full(v.size, -1, np.int64)
        idx[ok] = np.searchsorted(uniq, v[ok])
        labs = defs[layer].labels if layer in defs and defs[layer].labels else None
        labels = [labs[int(u)] if labs and 0 <= int(u) < len(labs) else f"{u:g}" for u in uniq]
        return idx, labels
    raise ApiError("validation", f"unknown breakdown kind {kind!r}",
                   detail={"errors": [{"path": "by.kind", "message": "unknown", "code": "kind"}]})


def breakdown(ctx, src, value: str, by: dict, weights: str | None, stat: str) -> dict:
    """``POST /stats/breakdown``: a layer grouped by zone, quantile, hexagon, category or fold."""
    from sparc.studio.runs.selection import column_values

    v = column_values(ctx, value)
    idx, labels = _groups_for(ctx, src, by)
    people = _people(ctx)
    w = people if weights == "people" else None
    if weights == "people" and people is None:
        raise ApiError("needs_layers", "people weights need the planner layers (planner.layers)")
    groups = []
    for i, lab in enumerate(labels):
        sel = idx == i
        if not sel.any():
            continue
        vals = v[sel]
        ok = np.isfinite(vals)
        q = None
        if stat == "box" and ok.any():
            q = [float(x) for x in np.percentile(vals[ok], [10, 25, 50, 75, 90])]
        groups.append({"label": lab, "n": int(sel.sum()),
                       "people": float(np.nansum(people[sel])) if people is not None else None,
                       "mean": weighted_mean(vals, None if w is None else w[sel]), "q": q})
    return clean({"groups": groups})


def hexbin(ctx, src, x: str, y: str, bins: int = 60, mask: np.ndarray | None = None) -> dict:
    """``POST /stats/hexbin``: 2-D counts (``counts[i][j]`` = x bin i, y bin j), Spearman ρ, binned means."""
    from scipy.stats import spearmanr

    from sparc.studio.runs.selection import column_values

    bins = max(2, min(int(bins or 60), 400))
    xv, yv = column_values(ctx, x), column_values(ctx, y)
    ok = np.isfinite(xv) & np.isfinite(yv)
    if ok.sum() < 2:
        return {"x_edges": [], "y_edges": [], "counts": [], "sel_counts": None, "spearman": None, "binned_mean": []}
    xr = (float(xv[ok].min()), float(xv[ok].max()))
    yr = (float(yv[ok].min()), float(yv[ok].max()))
    if xr[0] == xr[1]:
        xr = (xr[0] - 0.5, xr[1] + 0.5)
    if yr[0] == yr[1]:
        yr = (yr[0] - 0.5, yr[1] + 0.5)
    counts, xe, ye = np.histogram2d(xv[ok], yv[ok], bins=bins, range=[xr, yr])
    sel = None
    if mask is not None:
        m = ok & mask
        sel = np.histogram2d(xv[m], yv[m], bins=[xe, ye])[0].astype(int).tolist()
    rho = spearmanr(xv[ok], yv[ok]).statistic if ok.sum() > 2 else None
    xi = np.clip(np.searchsorted(xe, xv[ok], side="right") - 1, 0, bins - 1)
    sums = np.bincount(xi, weights=yv[ok], minlength=bins)
    cnt = np.bincount(xi, minlength=bins)
    centres = (xe[:-1] + xe[1:]) / 2
    binned = [{"x": float(centres[i]), "y": float(sums[i] / cnt[i])} for i in range(bins) if cnt[i] > 0]
    return clean({"x_edges": xe, "y_edges": ye, "counts": counts.astype(int).tolist(), "sel_counts": sel,
                  "spearman": float(rho) if rho is not None and np.isfinite(rho) else None, "binned_mean": binned})


def acf(ctx, src, layer: str, max_lag_m: float | None = None, n_perm: int = 19, n_bins: int = 20) -> dict:
    """``POST /stats/acf``: ``influence.fft_acf`` of a layer's raster with a permutation band."""
    from sparc.core.influence import fft_acf, permutation_band
    from sparc.studio.runs.selection import column_values

    g = src.grid
    v = column_values(ctx, layer)
    raster = g.raster(v)
    mask = g.mask_raster() & np.isfinite(raster)
    if mask.sum() < 2:
        raise ApiError("validation", f"layer {layer!r} has fewer than two valid cells",
                       detail={"errors": [{"path": "layer", "message": "too few cells", "code": "acf"}]})
    if max_lag_m is None:
        max_lag_m = float(((ctx.cfg_raw.get("influence") or {}).get("max_lag_m")) or 2000.0)
        extent = min(g.nx, g.ny) * g.dx
        max_lag_m = min(max_lag_m, extent / 2.0)
    max_lag_m = max(float(max_lag_m), 2.0 * g.dx)
    res = fft_acf(raster, mask, g.dx, max_lag_m, n_bins=n_bins)
    n_perm = max(1, min(int(n_perm or 19), 99))
    bm, bs = permutation_band(raster, mask, g.dx, max_lag_m, n_bins=n_bins, n_perm=n_perm)
    return clean({"lags_m": res["lags_m"], "acf": res["acf"], "band_mean": bm, "band_sd": bs})


def hex_table(ctx, src, size_m: float, layers: list[str], sums: list[str] | None = None):
    """Hexagon aggregates as a DataFrame: ``key, cx, cy, lon, lat, n_cells`` + one column per layer (mean, or
    sum for ``sums``)."""
    import pandas as pd

    from sparc.core.planner import hex_ids
    from sparc.studio.runs.grid import lonlat_transformer, transform_xy
    from sparc.studio.runs.selection import column_values

    g = src.grid
    keys, cx, cy = hex_ids(g.x, g.y, float(size_m))
    df = pd.DataFrame({"key": keys, "cx": cx, "cy": cy})
    sums = set(sums or [])
    for k in layers or []:
        df[k] = column_values(ctx, k)
    agg: dict[str, Any] = {"cx": "first", "cy": "first"}
    for k in layers or []:
        agg[k] = "sum" if k in sums else "mean"
    out = df.groupby("key").agg(agg)
    out["n_cells"] = df.groupby("key").size()
    out = out.reset_index()
    if g.crs:
        lon, lat = transform_xy(lonlat_transformer(g.crs), out["cx"].to_numpy() / g.coord_scale,
                                out["cy"].to_numpy() / g.coord_scale)
        out["lon"], out["lat"] = lon, lat
    else:
        out["lon"], out["lat"] = np.nan, np.nan
    cols = ["key", "cx", "cy", "lon", "lat", "n_cells"] + [k for k in layers or []]
    return out[cols]
