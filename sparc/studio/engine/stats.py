"""Statistics of exact results (SPEC §7.7–7.8): everything comes from ``delta`` (n) and ``folds`` (K × n).

* **Jackknife SE** of a mean over a mask: ``std_k(mean_{i∈mask} Δ_ki)·√(K−1)`` - on all cells exactly
  core's ``ScenarioResult.summary()``.
* **Regions**: ``all``, ``edited``, ``edited + ring`` (the edited cells buffered by the largest influence
  range of the edited levers), ``outside``, every named region of the scenario and every zone: n, mean ± SE,
  people-weighted mean, total Δ (target units × cells), share of cells cooled by ≥ 0.1 and ≥ 0.5.
* **Spill**: Δ summed inside and outside the edited cells, the share outside, and a ring profile (mean ± SE
  in rings of max(30 m, cell size) from the edited set out to 2 km).
* **Realised vs requested** per lever over the cells its edits touch, mediator moves, **cost** (Σ|realised|
  × per-unit cost) and cooling per cost unit.
* **Causal check** (``pipeline.causal_crosscheck``) and the **uncertainty envelope**
  (``uncertainty.scenario_uncertainty`` with the run's attached studies; specification from "check across
  runs" when available).
* **Paired difference** of two results on shared folds: ``SE(A−B) = std_k(mean_i(Δ_A,ki − Δ_B,ki))·√(K−1)``.
* The **plain-language card** (SPEC §7.7).

Negative Δ = cooler throughout.
"""

from __future__ import annotations

import math

import numpy as np

from sparc.studio.runs.common import clean, jackknife_se, likely

__all__ = ["masked_mean", "masked_se", "masked_likely", "paired_se", "pair_likely", "region_row", "auto_masks",
           "ring_profile", "spill", "realized_table", "cost_table", "causal_check", "uncertainty_block",
           "plain_card", "build_result", "result_summary_fields", "with_specification", "RING_MAX_M", "RING_MIN_W"]

RING_MAX_M = 2000.0
RING_MIN_W = 30.0


# ---------------------------------------------------------------------------
# means and standard errors
# ---------------------------------------------------------------------------

def _wmean(v: np.ndarray, w: np.ndarray | None) -> float | None:
    ok = np.isfinite(v) if w is None else (np.isfinite(v) & np.isfinite(w) & (w > 0))
    if not ok.any():
        return None
    if w is None:
        return float(v[ok].mean())
    return float(np.sum(v[ok] * w[ok]) / np.sum(w[ok]))


def masked_mean(delta: np.ndarray, mask: np.ndarray | None = None, w: np.ndarray | None = None) -> float | None:
    d = np.asarray(delta, dtype=np.float64)
    if mask is None:
        return _wmean(d, w)
    if not mask.any():
        return None
    return _wmean(d[mask], None if w is None else w[mask])


def masked_se(folds: np.ndarray | None, mask: np.ndarray | None = None, w: np.ndarray | None = None) -> float | None:
    """Fold jackknife SE of the (weighted) mean over ``mask``; None without folds (or K < 2)."""
    if folds is None or np.ndim(folds) != 2 or folds.shape[0] < 2:
        return None
    return jackknife_se([masked_mean(f, mask, w) for f in np.asarray(folds, dtype=np.float64)])


def masked_likely(delta, folds, mask, unit: str, what: str = "", w=None) -> dict | None:
    est = masked_mean(delta, mask, w)
    if est is None:
        return None
    return likely(est, masked_se(folds, mask, w), unit, what=what)


def paired_se(folds_a: np.ndarray | None, folds_b: np.ndarray | None, mask: np.ndarray | None = None) -> float | None:
    """``std_k(mean_i(Δ_A,ki − Δ_B,ki))·√(K−1)`` (shared fold models); None unless both have K ≥ 2 folds."""
    if folds_a is None or folds_b is None:
        return None
    a, b = np.asarray(folds_a, dtype=np.float64), np.asarray(folds_b, dtype=np.float64)
    if a.ndim != 2 or a.shape != b.shape or a.shape[0] < 2:
        return None
    return masked_se(a - b, mask)


def pair_likely(delta_a, folds_a, se_a, delta_b, folds_b, se_b, mask, unit: str, what: str = "") -> dict | None:
    """Likely of A − B over ``mask`` with ``paired`` true when the fold-paired SE applies; else the
    independent SE ``√(SE_A² + SE_B²)`` (paired false)."""
    da = masked_mean(delta_a, mask)
    db = masked_mean(delta_b, mask)
    if da is None or db is None:
        return None
    se = paired_se(folds_a, folds_b, mask)
    paired = se is not None
    if se is None and se_a is not None and se_b is not None:
        se = math.sqrt(float(se_a) ** 2 + float(se_b) ** 2)
    out = likely(da - db, se, unit, what=what) or likely(0.0, None, unit)
    out["paired"] = paired
    return out


# ---------------------------------------------------------------------------
# regions and spill
# ---------------------------------------------------------------------------

def region_row(name: str, auto: bool, mask: np.ndarray, delta: np.ndarray, folds, people, unit: str) -> dict | None:
    if not mask.any():
        return None
    d = np.asarray(delta, dtype=np.float64)[mask]
    return {"name": name, "auto": bool(auto), "n_cells": int(mask.sum()),
            "mean": masked_likely(delta, folds, mask, unit, what=name if not auto else ""),
            "people_weighted": masked_mean(delta, mask, people) if people is not None else None,
            "total": float(np.nansum(d)), "frac_cooled_01": float(np.mean(d <= -0.1)),
            "frac_cooled_05": float(np.mean(d <= -0.5))}


def distance_to(mask: np.ndarray, grid) -> np.ndarray:
    """Run-frame distance (m) of every cell to the nearest cell of ``mask`` (0 inside; inf when empty)."""
    from scipy.ndimage import distance_transform_edt

    if not mask.any():
        return np.full(mask.size, np.inf)
    ras = np.zeros((grid.ny, grid.nx), dtype=bool)
    ras[grid.iy[mask], grid.ix[mask]] = True
    dist = distance_transform_edt(~ras, sampling=float(grid.dx))
    return dist[grid.iy, grid.ix]


def auto_masks(edited: np.ndarray, grid, ring_m: float | None) -> dict[str, np.ndarray]:
    n = edited.size
    out = {"all": np.ones(n, dtype=bool), "edited": edited.copy()}
    if ring_m and edited.any() and not edited.all():
        out["edited + ring"] = distance_to(edited, grid) <= float(ring_m) + 1e-6
    out["outside"] = ~edited
    return out


def ring_profile(edited: np.ndarray, grid, delta: np.ndarray, folds, *, max_m: float = RING_MAX_M) -> list[dict]:
    """Mean Δ ± SE of the edited cells (``r_m`` 0) and of rings of width max(30 m, dx) out to ``max_m``."""
    if not edited.any():
        return []
    width = max(RING_MIN_W, float(grid.dx))
    dist = distance_to(edited, grid)
    n_rings = int(math.ceil(max_m / width))
    ring = np.where(edited, 0, np.ceil(dist / width - 1e-9)).astype(np.int64)
    ok = (ring <= n_rings) & np.isfinite(dist)
    idx = ring[ok]
    d = np.asarray(delta, dtype=np.float64)[ok]
    cnt = np.bincount(idx, minlength=n_rings + 1)
    sums = np.bincount(idx, weights=d, minlength=n_rings + 1)
    fold_means = None
    if folds is not None and np.ndim(folds) == 2 and folds.shape[0] >= 2:
        F = np.asarray(folds, dtype=np.float64)[:, ok]
        fold_means = np.vstack([np.bincount(idx, weights=f, minlength=n_rings + 1) for f in F])
        with np.errstate(invalid="ignore", divide="ignore"):
            fold_means = fold_means / np.maximum(cnt, 1)[None, :]
    rows = []
    for k in range(n_rings + 1):
        if cnt[k] == 0:
            continue
        se = jackknife_se(fold_means[:, k]) if fold_means is not None else None
        rows.append({"r_m": float(k * width), "mean": float(sums[k] / cnt[k]), "se": se, "n": int(cnt[k])})
    return rows


def spill(edited: np.ndarray, grid, delta: np.ndarray, folds, lever_ranges: dict[str, float]) -> dict:
    d = np.asarray(delta, dtype=np.float64)
    inside = float(np.nansum(d[edited]))
    outside = float(np.nansum(d[~edited]))
    total = inside + outside
    return {"inside": inside, "outside": outside, "outside_share": (outside / total) if abs(total) > 1e-12 else None,
            "rings": ring_profile(edited, grid, d, folds), "lever_ranges": {k: float(v) for k, v in lever_ranges.items()
                                                                          if v is not None}}


# ---------------------------------------------------------------------------
# realised, mediators, cost
# ---------------------------------------------------------------------------

def realized_table(realized: dict, requested: dict | None, lever_cells: dict | None) -> dict:
    out = {}
    for var, real in realized.items():
        real = np.asarray(real, dtype=np.float64)
        req = np.asarray((requested or {}).get(var, real), dtype=np.float64)
        cells = (lever_cells or {}).get(var)
        cells = (np.abs(req) > 0) | (np.abs(real) > 0) if cells is None else np.asarray(cells, dtype=bool)
        k = int(cells.sum())
        clipped = np.abs(real - req) > 1e-9 * np.maximum(1.0, np.abs(req))
        out[var] = {"requested_mean": float(req[cells].mean()) if k else 0.0,
                    "realized_mean": float(real[cells].mean()) if k else 0.0,
                    "requested_total": float(req[cells].sum()) if k else 0.0,
                    "realized_total": float(real[cells].sum()) if k else 0.0,
                    "clipped_share": float(clipped[cells].mean()) if k else 0.0}
    return out


def cost_table(realized: dict, per_unit: dict[str, float], delta: np.ndarray) -> dict:
    per_lever = {v: float(np.nansum(np.abs(np.asarray(r, dtype=np.float64))) * float(per_unit.get(v, 1.0)))
                 for v, r in realized.items()}
    total = float(sum(per_lever.values()))
    cooling = -float(np.nansum(delta))
    return {"total": total, "per_lever": per_lever, "cooling_per_cost": (cooling / total) if total > 0 else None}


# ---------------------------------------------------------------------------
# causal check and uncertainty
# ---------------------------------------------------------------------------

def core_summary(name: str, delta, folds, delta_sd, extrapolation, realized: dict) -> dict:
    """The ``ScenarioResult.summary()`` dict of a result (what core's cross-checks read)."""
    d = np.asarray(delta, dtype=np.float64)
    return {"name": name, "mean_delta": float(np.mean(d)), "mean_delta_se": masked_se(folds),
            "p10_delta": float(np.percentile(d, 10)), "p90_delta": float(np.percentile(d, 90)),
            "mean_delta_sd": float(np.mean(delta_sd)) if delta_sd is not None else None,
            "frac_extrapolated": float(np.mean(np.asarray(extrapolation) > 1.0)) if extrapolation is not None else None,
            "mean_realized": {k: float(np.mean(v)) for k, v in realized.items()}}


def causal_check(summ: dict, causal: dict | None) -> dict | None:
    from sparc.core.pipeline import causal_crosscheck

    if not causal:
        return None
    s = dict(summ)
    causal_crosscheck([s], causal)
    return s.get("causal_linear")


def uncertainty_block(summ: dict, cfg_raw: dict, manifest: dict | None, *, specification: list | None = None,
                      causal_linear: dict | None = None) -> dict | None:
    """``uncertainty.scenario_uncertainty`` of this result with the run's attached studies (multiverse and
    simcheck sections of the manifest) and the "check across runs" band as the specification."""
    from sparc.core.uncertainty import scenario_uncertainty

    m = manifest or {}
    s = dict(summ)
    if causal_linear:
        s["causal_linear"] = causal_linear
    row = (scenario_uncertainty({"config": cfg_raw, "scenarios": [s]}, m.get("multiverse"), m.get("simcheck"))
           .get("scenarios") or [None])[0]
    if row is None:
        return None
    return _finish_block(row, specification)


def _finish_block(row: dict, specification: list | None) -> dict:
    """The block from its bands: a multiverse ``specification`` wins over the "check across runs" band; the
    envelope spans the estimation, specification and attribution bands."""
    sources = []
    spec = row.get("specification")
    if row.get("estimation_95"):
        sources.append("estimation: fold-to-fold jackknife")
    if spec:
        sources.append("specification: multiverse")
    elif specification:
        spec = [float(min(specification)), float(max(specification))]
        sources.append("specification: check across runs")
    if row.get("attribution"):
        sources.append("attribution: simulation check")
    if row.get("causal_band"):
        sources.append("causal band: independent causal estimate")
    parts = [x for x in (row.get("estimation_95"), spec, row.get("attribution")) if x]
    env = [float(min(p[0] for p in parts)), float(max(p[1] for p in parts))] if parts else None
    return {"estimation_95": row.get("estimation_95"), "specification": spec,
            "attribution": row.get("attribution"), "causal_band": row.get("causal_band"), "envelope": env,
            "envelope_excludes_zero": (bool(env[1] < 0 or env[0] > 0) if env else None), "sources": sources}


def with_specification(unc: dict | None, specification: list | None) -> dict | None:
    """A stored result's uncertainty block with its "check across runs" band replaced by ``specification``
    (None removes it), the envelope and sources recomputed; a multiverse band is kept as it is."""
    if not isinstance(unc, dict) or "specification: multiverse" in (unc.get("sources") or []):
        return unc
    row = {k: unc.get(k) for k in ("estimation_95", "attribution", "causal_band")}
    return _finish_block({**row, "specification": None}, specification)


# ---------------------------------------------------------------------------
# the plain-language card
# ---------------------------------------------------------------------------

def _rng(lk: dict, unit: str, d: int = 2) -> str:
    lo, hi = lk.get("lo"), lk.get("hi")
    if lo is None or hi is None:
        return ""
    if lo * hi > 0:
        a, b = sorted((abs(lo), abs(hi)))
        return f" (likely range {a:.{d}f}–{b:.{d}f} {unit})"
    return f" (likely range {lo:+.{d}f} to {hi:+.{d}f} {unit})".replace("-", "−")


def plain_card(*, city: dict, edited: dict | None, edited_all: bool, unit: str, frac_extrapolated_edited: float | None,
               causal: dict | None, draft: bool = False, buys: list[str] | None = None) -> dict:
    """``{headline, confidence, qualifiers, buys}`` (SPEC §7.7)."""
    focus = city if (edited is None or edited_all) else edited
    est = float(focus["estimate"])
    verb = "Cools" if est < 0 else "Warms" if est > 0 else "Leaves unchanged"
    area = "the city" if focus is city else "the edited area"
    if est == 0:
        head = f"{verb} {area}."
    else:
        head = f"{verb} {area} by {abs(est):.2f} {unit}{_rng(focus, unit)}."
    if focus is not city:
        c = float(city["estimate"])
        head += f" City-wide: {abs(c):.3f} {unit} {'cooler' if c < 0 else 'warmer' if c > 0 else 'no change'}."
    conf = {"confident_cools": "Confident it cools", "confident_warms": "Confident it warms",
            "could_be_zero": "Could be zero", "unknown": "No uncertainty estimate"}[focus["confidence"]]
    quals = []
    if frac_extrapolated_edited is not None and frac_extrapolated_edited > 0.2:
        quals.append(f"partly outside observed conditions ({frac_extrapolated_edited:.0%} of edited cells)")
    if causal and causal.get("model_within") is False:
        quals.append("independent causal check disagrees")
    if draft:
        quals.append("preview only — not verified")
    return {"headline": head, "confidence": conf, "qualifiers": quals, "buys": list(buys or [])}


def buys_lines(*, delta: np.ndarray, obs: np.ndarray | None, people: np.ndarray | None, threshold: float | None,
               warming_mid: float | None, warming_label: str | None, cooling_per_cost: float | None,
               unit: str) -> list[str]:
    """"What it buys": residents moved below the threshold today and mid-century, cooling per 1,000 cost
    units, and the share of mid-century warming offset."""
    out: list[str] = []
    d = np.asarray(delta, dtype=np.float64)
    if obs is not None and people is not None and threshold is not None:
        p = np.nan_to_num(people)
        for label, w in (("today", 0.0), (f"mid-century ({warming_label})" if warming_label else None, warming_mid)):
            if label is None or w is None:
                continue
            before = float(p[obs + w >= threshold].sum())
            after = float(p[obs + w + d >= threshold].sum())
            moved = before - after
            if abs(moved) >= 0.5:
                out.append(f"{'Moves' if moved > 0 else 'Pushes'} {abs(moved):,.0f} residents "
                           f"{'below' if moved > 0 else 'above'} {threshold:g} {unit} {label}")
    if cooling_per_cost is not None:
        out.append(f"{1000.0 * cooling_per_cost:,.2f} {unit}·cells of cooling per 1,000 cost units")
    if warming_mid is not None and warming_mid > 0:
        share = -float(np.mean(d)) / float(warming_mid)
        out.append(f"Offsets {share:.0%} of the city's {warming_label or 'mid-century'} median warming")
    return out


# ---------------------------------------------------------------------------
# the whole result
# ---------------------------------------------------------------------------

def result_summary_fields(delta, folds, edited, extrapolation, unit: str) -> dict:
    """``{city, edited, frac_extrapolated_edited, has_folds}`` of a ResultSummary."""
    city = likely(float(np.mean(delta)), masked_se(folds), unit, what="the city") or likely(0.0, None, unit)
    ed = masked_likely(delta, folds, edited, unit, what="the edited area") if edited is not None and edited.any() \
        else None
    fx = float(np.mean(np.asarray(extrapolation)[edited] > 1.0)) if (extrapolation is not None and edited is not None
                                                                      and edited.any()) else None
    return {"city": city, "edited": ed, "frac_extrapolated_edited": fx,
            "has_folds": folds is not None and np.ndim(folds) == 2 and folds.shape[0] >= 2}


def build_result(*, result_id: str, kind: str, run_id: str, created_utc: str, job_id: str | None, delta, delta_sd,
                 extrapolation, folds, realized: dict, grid, unit: str, scenario: dict | None = None,
                 name: str = "scenario", requested: dict | None = None, lever_cells: dict | None = None,
                 mediators: dict | None = None, regions: dict | None = None, people=None, zones=None,
                 lever_ranges: dict | None = None, per_unit: dict | None = None, causal: dict | None = None,
                 cfg_raw: dict | None = None, manifest: dict | None = None, specification: list | None = None,
                 preview_delta=None, warnings: list | None = None, demo: bool = False, spec: dict | None = None,
                 obs=None, threshold: float | None = None, warming_mid: float | None = None,
                 warming_label: str | None = None, draft: bool = False) -> dict:
    """The ``Result`` of api.md §7.5 (without ``impacts``) as a JSON-safe dict."""
    delta = np.asarray(delta, dtype=np.float64)
    folds = None if folds is None else np.asarray(folds, dtype=np.float64)
    n = delta.size
    edited = np.zeros(n, dtype=bool)
    for v in realized.values():
        edited |= np.abs(np.asarray(v)) > 0
    sumf = result_summary_fields(delta, folds, edited, extrapolation, unit)
    ranges = {v: (lever_ranges or {}).get(v) for v in realized}
    ring_m = max([r for r in ranges.values() if r is not None] or [0.0])
    regions_out = []
    for nm, mk in auto_masks(edited, grid, ring_m).items():
        row = region_row(nm, True, mk, delta, folds, people, unit)
        if row is not None:
            regions_out.append(row)
    for nm, mk in (regions or {}).items():
        row = region_row(str(nm), False, np.asarray(mk, dtype=bool), delta, folds, people, unit)
        if row is not None:
            regions_out.append(row)
    if zones is not None:
        codes, labels = zones
        for k, z in enumerate(labels):
            row = region_row(f"zone {z}", True, codes == k, delta, folds, people, unit)
            if row is not None:
                regions_out.append(row)
    summ = core_summary(name, delta, folds, delta_sd, extrapolation, realized)
    cl = causal_check(summ, causal)
    unc = uncertainty_block(summ, cfg_raw or {}, manifest, specification=specification, causal_linear=cl) \
        if cfg_raw is not None else None
    cost = cost_table(realized, per_unit or {}, delta)
    med = {}
    for k, v in (mediators or {}).items():
        v = np.asarray(v, dtype=np.float64)
        if np.any(np.abs(v) > 1e-12):
            med[k] = {"mean_change": float(v[edited].mean()) if edited.any() else float(v.mean())}
    pve = None
    if preview_delta is not None:
        p = np.asarray(preview_delta, dtype=np.float64)
        md = float(np.mean(delta))
        pve = {"mean_abs_err": float(np.mean(np.abs(p - delta))),
               "rel_err": float(abs(float(np.mean(p)) - md) / max(abs(md), 1e-9))}
    buys = buys_lines(delta=delta, obs=obs, people=people, threshold=threshold, warming_mid=warming_mid,
                      warming_label=warming_label, cooling_per_cost=cost["cooling_per_cost"], unit=unit)
    plain = plain_card(city=sumf["city"], edited=sumf["edited"], edited_all=bool(edited.all()), unit=unit,
                       frac_extrapolated_edited=sumf["frac_extrapolated_edited"], causal=cl, draft=draft, buys=buys)
    summary = {"id": result_id, "scenario_id": (scenario or {}).get("id"), "run_id": run_id, "kind": kind,
               "created_utc": created_utc, "stale": False, "has_folds": sumf["has_folds"], "city": sumf["city"],
               "edited": sumf["edited"], "frac_extrapolated_edited": sumf["frac_extrapolated_edited"],
               "job_id": job_id}
    out = {"summary": summary, "spec": spec or {}, "scenario": scenario, "city": sumf["city"],
           "p10": float(np.percentile(delta, 10)), "p90": float(np.percentile(delta, 90)),
           "mean_delta_sd": float(np.mean(delta_sd)) if delta_sd is not None else None,
           "regions": regions_out, "spill": spill(edited, grid, delta, folds, ranges),
           # None when not computed (a preview has no extrapolation scores, or nothing was edited): never 0
           "extrapolated_edited": sumf["frac_extrapolated_edited"],
           "realized": realized_table(realized, requested, lever_cells), "mediators": med, "cost": cost,
           "causal_check": cl, "uncertainty": unc, "impacts": None, "preview_vs_exact": pve, "plain": plain,
           "warnings": [{"code": str(w.get("code")), "message": str(w.get("message"))} for w in warnings or []],
           "stale": False, "demo": bool(demo)}
    return clean(out)
