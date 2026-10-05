"""ViewModels of the run-hub tabs (SPEC §6.4, api.md §6.1 ``GET /runs/{rid}/views/{view}``).

Every view returns ``{view, availability, missing, units, caveats, demo, sections}``.  ``sections`` is
column-oriented and chart-ready; every key of api.md's table is present and ``null`` when the run lacks
the data (older code, a stage that did not run, a mid-run read).  The inner shapes are the ones the web
client renders (``studio-web/src/api/runs.ts``); each builder names the core output it summarises.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable

import numpy as np

from sparc.studio.errors import ApiError
from sparc.studio.runs.common import clean, fnum, likely

log = logging.getLogger("sparc.studio.runs")

__all__ = ["VIEWS", "SECTION_KEYS", "build_view"]

SECTION_KEYS: dict[str, tuple[str, ...]] = {
    "overview": ("kpis", "timings", "flags", "outputs_grid", "studies", "limitations", "findings"),
    "data": ("qa_tiles", "flags", "frac_hist", "dose_scale", "predictor_hists", "corr_matrix", "zone_counts",
             "coarse", "joins"),
    "accuracy": ("models", "obs_pred_bins", "resid_hist", "resid_by_zone", "resid_by_fold", "interval_honesty",
                 "stacker", "physics", "advection", "forcing", "cv_design", "live"),
    "distance": ("curve", "baselines", "verdict", "block_wins"),
    "influence": ("ranges", "correlogram", "rings", "anisotropy", "priors"),
    "response": ("levers", "literature"),
    "scenarios": ("rows", "ladders", "has_detail"),
    "climate": ("warming", "models_table", "exposure", "offset", "thresholds"),
    "heat": ("brief", "humidity", "categories", "kpis", "today", "futures", "hist", "verdicts"),
    "causal": ("treatments", "flags", "dag_audit"),
    "budget": ("kpis", "pareto", "caption", "top_cells", "status"),
    "planner": ("exposure", "person_mean", "hot_days", "equity", "plantable", "zones", "hex_files", "sites", "pairs",
                "gis"),
    "uncertainty": ("rows", "climate", "sources"),
    "provenance": ("hashes", "git", "platform", "launch"),
}
VIEWS = tuple(SECTION_KEYS)


def _safe(fn: Callable[[], Any], what: str, ctx) -> Any:
    """A section that fails to build is null (logged), never a failed view."""
    try:
        return fn()
    except ApiError:
        return None
    except Exception:
        log.exception("%s: section %s failed", ctx.run_id, what)
        return None


def _kpi(id_, label, value, *, unit=None, decimals=2, fmt="number", likely_=None, target=None, target_label=None,
         band=None, note=None, tone=None) -> dict:
    return {"id": id_, "label": label, "value": value, "unit": unit, "decimals": decimals, "format": fmt,
            "likely": likely_, "target": target, "target_label": target_label, "band": band, "note": note,
            "tone": tone}


def _flags(ctx) -> list[dict]:
    qa = (ctx.manifest or {}).get("qa") or {}
    return [{"code": str(f.get("code")), "severity": str(f.get("severity") or "info"), "message": str(f.get("message"))}
            for f in qa.get("flags") or [] if isinstance(f, dict)]


def _box(v: np.ndarray, label: str) -> dict:
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"label": label, "n": 0, "q": None, "mean": None}
    q = np.percentile(v, [10, 25, 50, 75, 90])
    return {"label": label, "n": int(v.size), "q": [float(x) for x in q], "mean": float(v.mean())}


def _hist(v: np.ndarray, bins: int = 30) -> dict | None:
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    lo, hi = float(v.min()), float(v.max())
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    counts, edges = np.histogram(v, bins=bins, range=(lo, hi))
    return {"edges": edges.tolist(), "counts": counts.astype(int).tolist()}


def _table(columns: list[tuple[str, str, str | None]], rows: list[list]) -> dict:
    return {"columns": [{"key": k, "label": lab, "unit": u} for k, lab, u in columns], "rows": rows}


# ---------------------------------------------------------------------------
# overview
# ---------------------------------------------------------------------------

def _overview(ctx, env) -> dict:
    m = ctx.manifest or {}
    tu = ctx.units.get("target")
    st = (ctx.metrics or {}).get("stacker") or {}
    qa = m.get("qa") or {}
    k = [
        _kpi("r2", "Held-out R²", fnum(st.get("r2")), decimals=2,
             note="computed from predictions.parquet while the run continues" if ctx.metrics_are_live else None),
        _kpi("rmse", "Held-out RMSE", fnum(st.get("rmse")), unit=tu, decimals=2),
    ]
    cov, tgt = fnum(st.get("interval_coverage")), fnum(st.get("interval_target")) or 0.9
    k.append(_kpi("coverage", "90% interval coverage", cov, fmt="percent", decimals=1, target=tgt,
                  target_label="target", tone=None if cov is None else ("good" if cov >= tgt - 0.02 else "warn")))
    hw = fnum(st.get("interval_mean_halfwidth"))
    k.append(_kpi("halfwidth", "Interval half-width", hw, unit=tu, decimals=2,
                  target=fnum(qa.get("target_rounding_noise_sd")), target_label="rounding-noise floor"))
    rows = _scenario_rows(ctx)
    head = env.get("headline_scenario")
    if head:
        r = next((r for r in rows if r["slug"] == head), None)
        if r is not None:
            k.append(_kpi("headline", r["name"], r["delta"]["estimate"], unit=tu, fmt="delta", likely_=r["delta"]))
    joint = [r for r in rows if r["kind"] == "joint"]
    if joint:
        r = joint[0]
        band = {"lo": r["causal"]["lo"], "hi": r["causal"]["hi"], "label": "causal check"} if r.get("causal") else None
        k.append(_kpi("package", r["name"], r["delta"]["estimate"], unit=tu, fmt="delta", likely_=r["delta"],
                      band=band))
    clim = m.get("climate") or {}
    mid = next((p for p in clim.get("projections") or [] if p.get("experiment") == "ssp245"
                and p.get("period") == "2041-2060"), None)
    if mid:
        k.append(_kpi("warming", "Mid-century warming (SSP2-4.5, 2041–2060)",
                      fnum((mid.get("warming") or {}).get("median")), unit=tu, fmt="signed", decimals=1))
    opt = m.get("optimize") or {}
    if fnum(opt.get("realized_total_cooling")) is not None:
        k.append(_kpi("budget", "Budget plan cooling (realised)", fnum(opt.get("realized_total_cooling")),
                      unit=f"{tu}·cells", decimals=0, note=f"{opt.get('n_cells_treated')} cells treated"))
    timings = [{"stage": r["id"], "label": env["labels"].get(r["id"], r["id"]), "seconds": r["seconds"],
                "state": r["state"], "reason": r["reason"]} for r in env["stage_rows"]]
    return {"kpis": k, "timings": timings, "flags": _flags(ctx) + env.get("run_flags", []),
            "outputs_grid": _outputs_grid(env["outputs"], env["stage_rows"]), "studies": _study_chips(ctx, env),
            "limitations": _limitations(ctx), "findings": env.get("findings") or []}


def _outputs_grid(entries: list[dict], stage_rows: list[dict]) -> list[dict]:
    """The outputs availability grid: every output the run has or should have, plus one row per post-run
    action that has not run and per stage the run skipped (its primary output, carrying the remedy) - not
    the dozen optional files a planner pack or a study would add."""
    skipped = {f"stage:{r['id']}" for r in stage_rows if r["state"] in ("skipped", "disabled", "not_requested")}
    out, offered = [], set()
    for e in entries:
        if e["state"] == "missing" and not e["_expected"]:
            pb = e["produced_by"]
            if not (pb.startswith("post:") or pb in skipped) or pb in offered:
                continue
            offered.add(pb)
        out.append({k2: e[k2] for k2 in ("id", "label", "group", "state", "produced_by", "view", "action")})
    return out


#: overview study chips: post-run actions and studies (Status Board columns, SPEC §5.12)
_CHIPS = ("baselines", "planner", "emulator", "uncertainty", "placebo", "multiverse", "simcheck", "reproduce")
_CHIP_STATE = {"done": "done", "cached": "done", "running": "running", "failed": "failed", "stale": "stale"}


def _study_chips(ctx, env) -> list[dict]:
    """``{kind, state, headline, study_id}`` per post-run action and study, from the run's Status Board cells
    (files, jobs, the studies index) with a one-line headline from the study summary or the manifest."""
    board = env.get("board") or {}
    rows = {r["kind"]: r for r in env.get("study_rows") or []}
    out = []
    for kind in _CHIPS:
        cell = board.get(kind) or {}
        state = _CHIP_STATE.get(cell.get("state"), "not_run")
        row = rows.get(kind) or {}
        headline = (row.get("summary") or {}).get("headline") if isinstance(row.get("summary"), dict) else None
        if headline is None and state in ("done", "stale"):
            headline = _study_headline(ctx, kind, row.get("summary") if isinstance(row.get("summary"), dict) else {})
        out.append({"kind": kind, "state": state, "headline": headline,
                    "study_id": cell.get("study_id") or row.get("id")})
    return out


def _study_headline(ctx, kind: str, summary: dict) -> str | None:
    m = ctx.manifest or {}
    if kind == "baselines":
        v = (m.get("baselines") or {}).get("verdict")
        return str(v) if v else None
    if kind == "planner":
        p = m.get("planner") or {}
        n = fnum(p.get("people_total"))
        return f"{n:,.0f} residents covered" if n is not None else None
    if kind == "emulator":
        levers = (m.get("emulator") or {}).get("levers") or {}
        rates = [fnum((v or {}).get("patch_pass_rate")) for v in levers.values() if isinstance(v, dict)]
        rates = [r for r in rates if r is not None]
        return f"patch pass rate {min(rates):.0%}–{max(rates):.0%}" if rates else None
    if kind == "uncertainty":
        n = len((m.get("uncertainty") or {}).get("scenarios") or [])
        return f"envelopes for {n} scenarios" if n else None
    if kind == "placebo":
        pz = m.get("placebo") or summary
        if pz and pz.get("n_placebos"):
            return (f"model {pz.get('n_pass_model') or 0}/{pz['n_placebos']} and causal "
                    f"{pz.get('n_pass_causal') or 0}/{pz['n_placebos']} placebos passed")
        return None
    if kind == "multiverse":
        mv = m.get("multiverse") or summary
        s = fnum((mv or {}).get("sign_stability_min"))
        return f"sign stable in at least {s:.0%} of variants" if s is not None else None
    if kind == "simcheck":
        bc = ((m.get("simcheck") or summary or {}).get("bias_correction") or {})
        r = bc.get("share_range")
        return f"effect share {r[0]:.2f}–{r[-1]:.2f} of the truth" if r else None
    if kind == "reproduce":
        ok = summary.get("pass")
        return None if ok is None else ("reproduced within tolerance" if ok else "did not reproduce")
    return None


def _limitations(ctx) -> list[str]:
    out = []
    p = ctx.run_dir / "model_card.md"
    if p.exists():
        text = p.read_text("utf-8", errors="replace")
        m = re.search(r"^##+\s*Limitations\s*$(.*?)(?=^##\s|\Z)", text, flags=re.M | re.S)
        if m:
            for line in m.group(1).splitlines():
                mm = re.match(r"^\s*[-*]\s+(.*)$", line)
                if mm:
                    out.append(mm.group(1).strip())
    for c in (ctx.cfg_raw.get("report") or {}).get("limitations") or []:
        out.append(str(c))
    return out


# ---------------------------------------------------------------------------
# data & QA
# ---------------------------------------------------------------------------

def _data(ctx, env) -> dict:
    m = ctx.manifest or {}
    qa = m.get("qa") or {}
    tu = ctx.units.get("target")
    levers = ctx.units.get("levers") or {}
    clipped = qa.get("clipped") or {}
    tiles = [
        _kpi("n_input", "Input points", qa.get("n_input"), fmt="int", decimals=0),
        _kpi("n_points", "Cells modelled", m.get("n_points") or ctx.meta.get("n_points"), fmt="int", decimals=0),
        _kpi("dropped", "Dropped (non-finite)", qa.get("n_dropped_nonfinite"), fmt="int", decimals=0),
        _kpi("clipped", "Clipped values", int(sum(clipped.values())) if clipped else 0, fmt="int", decimals=0,
             note=", ".join(f"{c}: {n}" for c, n in clipped.items()) or None),
        _kpi("fill", "Grid fill", fnum(qa.get("grid_fill_fraction")), fmt="percent", decimals=1),
        _kpi("collisions", "Cell collisions", fnum(qa.get("cell_collisions")), fmt="percent", decimals=2),
        _kpi("background", "Background", fnum(qa.get("background")), unit=tu, decimals=2,
             note=qa.get("background_source")),
        _kpi("noise_floor", "Rounding-noise floor", fnum(qa.get("target_rounding_noise_sd")), unit=tu, decimals=2),
        _kpi("cell_m", "Cell size", fnum(qa.get("cell_m")) or ctx.cell_m, unit="m", decimals=1),
    ]
    frac = None
    if qa.get("target_fractional_histogram"):
        h = qa["target_fractional_histogram"]
        frac = {"edges": [round(i / len(h), 4) for i in range(len(h) + 1)], "shares": h,
                "integer_share": fnum(qa.get("target_fraction_integer_valued")),
                "half_share": fnum(qa.get("target_fraction_half_valued")),
                "noise_sd": fnum(qa.get("target_rounding_noise_sd"))}
    dose = [{"lever": v, "unit": levers.get(v, ""), "sd": fnum(d.get("sd")), "doses": d.get("doses") or [],
             "doses_in_sd": d.get("doses_in_sd") or [], "percentile": d.get("median_cell_to_percentile") or []}
            for v, d in (qa.get("dose_scale") or {}).items()]
    data = ctx.data
    hists = corr = None
    if data is not None:
        hists = []
        for c in data.frame.columns:
            h = _hist(data.frame[c].to_numpy(float), 24)
            if h:
                hists.append({"name": c, "label": c.replace("_", " "), "unit": levers.get(c, ""), **h})
        cols = [c for c in data.frame.columns if np.issubdtype(data.frame[c].dtype, np.number)]
        mat = np.column_stack([data.target_raw] + [data.frame[c].to_numpy(float) for c in cols])
        with np.errstate(invalid="ignore", divide="ignore"):
            cm = np.corrcoef(mat, rowvar=False)
        corr = {"names": ["target"] + cols, "values": [[fnum(x) for x in row] for row in np.atleast_2d(cm)]}
    zc = None
    g = ctx.grid
    if g is not None and g.zones:
        counts = np.bincount(g.zone[g.zone >= 0].astype(np.int64), minlength=len(g.zones))
        zc = [{"zone": str(z), "n": int(counts[i])} for i, z in enumerate(g.zones)]
    elif g is not None:
        zc = []
    co = qa.get("coarse")
    coarse = [{"label": lab, "value": fnum(co.get(k2)) if k2 != "n_cells" else co.get(k2), "unit": u,
               "decimals": d} for k2, lab, u, d in (("cell_m", "Coarse cell", "m", 0),
                                                     ("fine_cell_m", "Input cell", "m", 1),
                                                     ("n_fine", "Input cells", None, 0),
                                                     ("n_cells", "Coarse cells", None, 0),
                                                     ("members_mean", "Input cells per coarse cell", None, 1),
                                                     ("frac_partial", "Partly covered coarse cells", None, 3))
              if k2 in co] if co else []
    prov = m.get("provenance") or {}
    joins = []
    for j in (ctx.cfg_raw.get("data") or {}).get("join") or []:
        key = j.get("key") or j.get("on") or (ctx.cfg_raw.get("data") or {}).get("id")
        joins.append({"path": str(j.get("path")), "key": key, "right_key": j.get("right_key") or j.get("right_on"),
                      "sha256": (prov.get("join_sha256") or {}).get(str(j.get("path"))) if isinstance(
                          prov.get("join_sha256"), dict) else None, "n_matched": None, "n_unmatched": None})
    return {"qa_tiles": tiles, "flags": _flags(ctx), "frac_hist": frac, "dose_scale": dose,
            "predictor_hists": hists, "corr_matrix": corr, "zone_counts": zc, "coarse": coarse, "joins": joins}


# ---------------------------------------------------------------------------
# accuracy
# ---------------------------------------------------------------------------

_MODEL_LABELS = {"ols": "Linear regression (OLS)", "mgwr": "Multiscale GWR", "gwrf": "Geographic random forest",
                 "gam": "GAM", "physics": "Physics model", "base_mean": "Equal-weight mean", "stacker": "Stack"}
_PHYS = [("L_m", "Relaxation length", "m"), ("a", "Physics amplitude", None), ("s", "Shade efficiency", None),
         ("a1", "Albedo coefficient", None), ("kappa_canopy", "Canopy shade rate", None), ("b", "Impervious term", None),
         ("gamma", "Elevation lapse", None), ("w", "Water influence scale", "m"), ("train_r2", "In-fold R²", None)]


def _accuracy(ctx, env) -> dict:
    m = ctx.manifest or {}
    metrics = ctx.metrics or {}
    pred = ctx.predictions
    stacks = m.get("stacker") if isinstance(m.get("stacker"), list) else []
    weights: dict[str, list[float]] = {}
    for s in stacks:
        for k2, w in ((s or {}).get("weights") or {}).items():
            weights.setdefault(k2, []).append(float(w))
    models = []
    for name, mm in metrics.items():
        if not isinstance(mm, dict):
            continue
        kind = "stack" if name == "stacker" else "mean" if name == "base_mean" else "base"
        w = float(np.mean(weights[name])) if name in weights else None
        models.append({"model": name, "label": _MODEL_LABELS.get(name, name), "kind": kind, "r2": fnum(mm.get("r2")),
                       "rmse": fnum(mm.get("rmse")), "mae": fnum(mm.get("mae")), "bias": fnum(mm.get("bias")),
                       "weight": w, "n": mm.get("n")})
    models.sort(key=lambda r: ({"stack": 0, "mean": 1, "base": 2}[r["kind"]], r["model"]))
    obs_pred = resid_hist = by_zone = by_fold = None
    if pred is not None and {"target", "pred"} <= set(pred.columns):
        from scipy.stats import spearmanr

        y, p = pred["target"].to_numpy(float), pred["pred"].to_numpy(float)
        ok = np.isfinite(y) & np.isfinite(p)
        lo, hi = float(min(y[ok].min(), p[ok].min())), float(max(y[ok].max(), p[ok].max()))
        # x-major: counts[i][j] = predicted bin i (x), observed bin j (y)
        counts, xe, ye = np.histogram2d(p[ok], y[ok], bins=40, range=[[lo, hi], [lo, hi]])
        rho = spearmanr(y[ok], p[ok]).statistic if ok.sum() > 2 else None
        obs_pred = {"x_edges": xe.tolist(), "y_edges": ye.tolist(), "counts": counts.astype(int).tolist(),
                    "spearman": fnum(rho)}
        r = y - p
        resid_hist = _hist(r, 40)
        g = ctx.grid
        if g is not None and g.zones:
            by_zone = [_box(r[g.zone == i], str(z)) for i, z in enumerate(g.zones)]
        elif g is not None:
            by_zone = []
        if "fold" in pred.columns:
            f = pred["fold"].to_numpy()
            by_fold = [_box(r[f == k2], f"Fold {int(k2) + 1}") for k2 in sorted(set(f.tolist()))]
    st = metrics.get("stacker") or {}
    honesty = None
    if st:
        diag = st.get("interval_diagnostics") or {}

        def rows(key, fmt):
            d = diag.get(key) or {}
            return [{"label": fmt(k2), "n": v.get("n"), "global": fnum(v.get("global")),
                     "adaptive": fnum(v.get("adaptive")), "halfwidth_global": fnum(v.get("halfwidth_global")),
                     "halfwidth_adaptive": fnum(v.get("halfwidth_adaptive"))} for k2, v in d.items()]

        by_fold_rows = rows("by_fold", lambda k2: f"Fold {int(k2) + 1}" if str(k2).isdigit() else str(k2))
        by_dist = rows("by_distance", str)
        by_zone_rows = rows("by_zone", str)
        if not diag and pred is not None and {"pi_lo", "pi_hi", "target"} <= set(pred.columns):
            by_fold_rows, by_dist = _live_honesty(pred)
        overall = diag.get("overall") or {}
        honesty = {"target": fnum(st.get("interval_target")) or 0.9,
                   "global": fnum(overall.get("global")) if overall else fnum(st.get("interval_coverage")),
                   "adaptive": fnum(overall.get("adaptive")) if overall else fnum(st.get("interval_coverage_adaptive")),
                   "halfwidth": fnum(st.get("interval_mean_halfwidth")), "by_fold": by_fold_rows,
                   "by_distance": by_dist, "by_zone": by_zone_rows}
    stacker = None
    if stacks or m.get("lambda_scores"):
        stacker = {"candidates": [{"name": str(k2), "rmse": fnum(v)} for k2, v in (m.get("lambda_scores") or {}).items()],
                   "chosen": m.get("stacker_choice"),
                   "models": sorted({k2 for s in stacks for k2 in ((s or {}).get("weights") or {})}),
                   "folds": [{"fold": i, "weights": (s or {}).get("weights") or {},
                              "residual_kept": None if (s or {}).get("residual_gated_off") is None
                              else not s.get("residual_gated_off"),
                              "val_mse_base": fnum((s or {}).get("val_mse_base")),
                              "val_mse_with_residual": fnum((s or {}).get("val_mse_with_residual")),
                              "best_epoch": (s or {}).get("best_epoch")} for i, s in enumerate(stacks)],
                   "spatial_plus": list(m.get("spatial_plus") or [])}
    phys = None
    pm = m.get("physics")
    pj = ctx.json("physics.json")
    if isinstance(pm, dict) or isinstance(pj, list):
        priors = ((pj or [{}])[0] or {}).get("priors") or {} if isinstance(pj, list) and pj else {}
        l_prior = (m.get("influence") or {}).get("L_prior_m")
        params = []
        for key, label, unit in _PHYS:
            v = (pm or {}).get(key)
            if not isinstance(v, dict):
                continue
            pr = priors.get(key)
            prior = f"{pr[0]:g} ± {pr[1]:g}" if isinstance(pr, list) and len(pr) == 2 else None
            if key == "L_m" and prior is None and fnum(l_prior) is not None:
                prior = f"starts at {float(l_prior):,.0f} m (S1 influence range)"
            params.append({"name": key, "label": label, "mean": fnum(v.get("mean")), "sd": fnum(v.get("sd")),
                           "prior": prior, "unit": unit})
        folds = []
        if isinstance(pj, list):
            for key, _l, _u in _PHYS:
                vals = [fnum((f or {}).get(key)) for f in pj]
                if any(v is not None for v in vals):
                    folds.append({"param": key, "values": vals})
        warns = [str(f.get("fit_warning")) for f in (pj or []) if isinstance(f, dict) and f.get("fit_warning")]
        phys = {"params": params, "folds": folds, "warnings": warns}
    adv = m.get("physics_advection")
    advection = None
    if isinstance(adv, dict):
        kept = adv.get("kept")
        verdict = ("advection kept: it lowers the held-out error" if kept else
                   "no advection: it does not beat v = 0 out of fold")
        advection = {"verdict": verdict, "selected": kept,
                     "fold_delta_rmse": [fnum(x) for x in adv.get("per_fold_delta_rmse") or []]}
    ph = (ctx.cfg_raw.get("physics") or {})
    fi = ph.get("forcing_info") or {}
    wind = ph.get("wind")
    ws = wd = None
    if isinstance(wind, (list, tuple)) and len(wind) == 2 and all(fnum(w) is not None for w in wind):
        u, v = float(wind[0]), float(wind[1])
        ws = float(np.hypot(u, v))
        wd = float((np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0)
    hours = fi.get("hours_local")
    forcing = {"date": fi.get("date"), "hours": f"{hours[0]}–{hours[1]} h" if isinstance(hours, list) and len(hours) == 2
               else ph.get("window"), "sw_down": fnum(ph.get("sw_down")), "lw_net": fnum(ph.get("lw_net")),
               "wind_speed": ws, "wind_dir_deg": wd, "station": fi.get("station"),
               "checks": [str(c) for c in fi.get("checks") or []]}
    cv = m.get("cv") or {}
    cvm = ctx.cv_meta
    cv_design = None
    if cvm:
        cv_design = {"n_folds": cvm["n_folds"], "block_m": cvm["block_m"], "buffer_m": cvm["buffer_m"],
                     "n_blocks": cv.get("n_blocks"), "test_sizes": list(cv.get("test_sizes") or [])}
        if not cv_design["test_sizes"] and pred is not None and "fold" in pred.columns:
            cv_design["test_sizes"] = np.bincount(pred["fold"].to_numpy().astype(int)).tolist()
    return {"models": models or None, "obs_pred_bins": obs_pred, "resid_hist": resid_hist, "resid_by_zone": by_zone,
            "resid_by_fold": by_fold, "interval_honesty": honesty, "stacker": stacker, "physics": phys,
            "advection": advection, "forcing": forcing, "cv_design": cv_design, "live": bool(ctx.metrics_are_live)}


def _live_honesty(pred) -> tuple[list[dict], list[dict]]:
    y = pred["target"].to_numpy(float)
    lo, hi = pred["pi_lo"].to_numpy(float), pred["pi_hi"].to_numpy(float)
    la = pred["pi_lo_adaptive"].to_numpy(float) if "pi_lo_adaptive" in pred else None
    ha = pred["pi_hi_adaptive"].to_numpy(float) if "pi_hi_adaptive" in pred else None

    def row(sel, label):
        cov = float(np.mean((y[sel] >= lo[sel]) & (y[sel] <= hi[sel]))) if sel.any() else None
        cova = float(np.mean((y[sel] >= la[sel]) & (y[sel] <= ha[sel]))) if la is not None and sel.any() else None
        return {"label": label, "n": int(sel.sum()), "global": cov, "adaptive": cova,
                "halfwidth_global": float(np.mean((hi[sel] - lo[sel]) / 2)) if sel.any() else None,
                "halfwidth_adaptive": float(np.mean((ha[sel] - la[sel]) / 2)) if la is not None and sel.any() else None}

    by_fold = []
    if "fold" in pred.columns:
        f = pred["fold"].to_numpy()
        by_fold = [row(f == k, f"Fold {int(k) + 1}") for k in sorted(set(f.tolist()))]
    by_dist = []
    if "dist_train_m" in pred.columns:
        d = pred["dist_train_m"].to_numpy(float)
        edges = np.unique(np.percentile(d[np.isfinite(d)], [0, 25, 50, 75, 100]))
        for i in range(len(edges) - 1):
            sel = (d >= edges[i]) & ((d < edges[i + 1]) if i < len(edges) - 2 else (d <= edges[i + 1]))
            by_dist.append(row(sel, f"{edges[i]:.0f}–{edges[i + 1]:.0f} m"))
    return by_fold, by_dist


# ---------------------------------------------------------------------------
# distance & baselines
# ---------------------------------------------------------------------------

def _distance(ctx, env) -> dict:
    m = ctx.manifest or {}
    cvd = m.get("cv_distance") or {}
    rows = [r for r in cvd.get("rows") or [] if isinstance(r, dict)]
    curve = None
    if rows:
        rand = [r for r in rows if str(r.get("label", "")).startswith("random")]
        blocks = [r for r in rows if r not in rand]
        blocks.sort(key=lambda r: float(r.get("block_m") or 0))
        series = [{"id": "stack", "label": "Stack", "kind": "stack",
                   "values": [fnum((r.get("stacker") or {}).get("r2")) for r in blocks],
                   "lo": [fnum(r.get("fold_r2_min")) for r in blocks], "hi": [fnum(r.get("fold_r2_max")) for r in blocks]}]
        names = []
        for r in blocks:
            for k2 in (r.get("models") or {}):
                if k2 not in names:
                    names.append(k2)
        for k2 in names:
            series.append({"id": k2, "label": _MODEL_LABELS.get(k2, k2), "kind": "base",
                           "values": [fnum(((r.get("models") or {}).get(k2) or {}).get("r2")) for r in blocks]})
        bl = []
        for r in blocks:
            for k2 in (r.get("baselines") or {}):
                if k2 not in bl:
                    bl.append(k2)
        for k2 in bl:
            series.append({"id": f"baseline:{k2}", "label": k2, "kind": "baseline",
                           "values": [fnum(((r.get("baselines") or {}).get(k2) or {}).get("r2")) for r in blocks]})
        main = next((r for r in rows if r.get("main")), None)
        curve = {"block_m": [float(r.get("block_m") or 0) for r in blocks], "metric": "r2", "series": series,
                 "main_block_m": fnum((main or {}).get("block_m")) if main else fnum((m.get("cv") or {}).get("block_m")),
                 "random": [{"id": "random", "label": str(r.get("label")), "value": fnum((r.get("stacker") or {}).get("r2"))}
                            for r in rand]}
    b = m.get("baselines") or {}
    brows = b.get("rows") or {}
    baselines = [{"id": k2, "label": str(v.get("label") or k2), "rmse": fnum(v.get("rmse")), "r2": fnum(v.get("r2")),
                  "delta_mse": fnum(v.get("delta_mse")), "delta_mse_se": fnum(v.get("delta_mse_se")),
                  "stack_better": bool(v.get("stack_better")), "baseline_better": bool(v.get("baseline_better"))}
                 for k2, v in brows.items()] if brows else None
    verdict = {"text": str(b.get("verdict")), "best_baseline": b.get("best_baseline"),
               "stack_wins": all(r["stack_better"] for r in baselines) if baselines else None} if b else None
    wins = [{"id": k2, "label": str(v.get("label") or k2), "frac": fnum(v.get("frac_blocks_stack_better")),
             "n_blocks": v.get("n_blocks")} for k2, v in brows.items()] if brows else None
    return {"curve": curve, "baselines": baselines, "verdict": verdict, "block_wins": wins}


# ---------------------------------------------------------------------------
# influence
# ---------------------------------------------------------------------------

def _influence(ctx, env) -> dict:
    inf = (ctx.manifest or {}).get("influence") or ctx.json("influence.json") or {}
    if not inf:
        return dict.fromkeys(SECTION_KEYS["influence"])
    full = ctx.json("influence.json") or inf
    fits = (full.get("diagnostics") or {}).get("ring_fits") or {}
    ranges = [{"predictor": p, "label": p.replace("_", " "), "range_m": fnum(v),
               "raw_range_m": fnum((fits.get(p) or {}).get("raw_range_m"))} for p, v in (inf.get("ranges_m") or {}).items()]
    acf = full.get("target_acf") or {}
    corr = {"lags_m": acf.get("lags") or acf.get("lags_m") or [], "acf": acf.get("acf") or [],
            "band_mean": acf.get("band_mean") or [], "band_sd": acf.get("band_sd") or [],
            "directional": acf.get("directional")} if acf else None
    edges = full.get("ring_edges_m") or []
    rings = [{"predictor": p, "label": p.replace("_", " "), "edges_m": edges, "betas": [fnum(x) for x in b],
              "family": (fits.get(p) or {}).get("family"), "significant": (fits.get(p) or {}).get("significant"),
              "scale_m": fnum((fits.get(p) or {}).get("scale_m"))} for p, b in (full.get("ring_betas") or {}).items()]
    an = [{"predictor": p, "label": p.replace("_", " "), "ratio": fnum(v.get("ratio")),
           "theta_deg": fnum(v.get("theta_deg")), "reliable": bool(v.get("reliable")),
           "ranges_m": {str(k2): fnum(x) for k2, x in (v.get("ranges_m") or {}).items()}}
          for p, v in (full.get("anisotropy") or {}).items() if isinstance(v, dict)]
    priors = {"L_prior_m": fnum(inf.get("L_prior_m")), "block_size_m": fnum(inf.get("block_size_m")),
              "resid_range_m": fnum(inf.get("target_resid_range_m"))}
    return {"ranges": ranges, "correlogram": corr, "rings": rings, "anisotropy": an, "priors": priors}


# ---------------------------------------------------------------------------
# response
# ---------------------------------------------------------------------------

def _response(ctx, env) -> dict:
    curves = ctx.json("response_curves.json") or {}
    resp = (ctx.manifest or {}).get("response") or {}
    levers_u = ctx.units.get("levers") or {}
    acts = ctx.cfg_raw.get("actionable") or {}
    out = {}
    for var in list(dict.fromkeys(list(curves) + list(resp))):
        c = (curves.get(var) or {}).get("curve") or {}
        s = (curves.get(var) or {}).get("summary") or resp.get(var) or {}
        curve = None
        if c:
            curve = {"dose": [float(x) for x in c.get("dose") or []],
                     "benefit": [fnum(x) for x in c.get("mean_benefit") or []],
                     "se": [fnum(x) for x in c.get("mean_se") or []],
                     "frac_extrapolated": [fnum(x) for x in c.get("frac_extrapolated") or []],
                     "realized_dose": [fnum(x) for x in c.get("mean_realized_dose") or []]}
        shapes = None
        if s:
            sat, cen = fnum(s.get("frac_saturating")), fnum(s.get("frac_censored"))
            lin, sig = fnum(s.get("frac_linear")), fnum(s.get("frac_sigmoid"))
            known = sum(x or 0 for x in (sat, lin, sig))
            shapes = {"saturating": sat, "linear": lin, "sigmoid": sig, "censored": cen,
                      "insufficient": max(0.0, 1.0 - known) if known else None}
        effects = {"own": fnum(s.get("mean_own_effect")), "footprint": fnum(s.get("mean_footprint_effect")),
                   "ratio": fnum(s.get("footprint_to_own_ratio")), "median_d90": fnum(s.get("median_d90")),
                   "median_max_cooling": fnum(s.get("median_max_cooling"))} if s else None
        fm = {"own": [float(x) for x in s.get("own_fold_means") or []],
              "footprint": [float(x) for x in s.get("footprint_fold_means") or []]} if s.get("own_fold_means") else None
        out[var] = {"label": (acts.get(var) or {}).get("label") or var.replace("_", " "), "unit": levers_u.get(var, ""),
                    "direction": str(s.get("direction") or (acts.get(var) or {}).get("direction") or "increase"),
                    "curve": curve, "shapes": shapes, "effects": effects, "fold_means": fm}
    lit = (ctx.manifest or {}).get("literature")
    literature = None
    if isinstance(lit, dict):
        rows = [{"key": r.get("key"), "quantity": r.get("quantity"), "per": r.get("per"), "low": fnum(r.get("low_C")),
                 "high": fnum(r.get("high_C")), "unit": "°C", "value": r.get("value"), "status": r.get("status"),
                 "citation": r.get("citation")} for r in lit.get("rows") or [] if isinstance(r, dict)]
        sparc = [{"quantity": q, "scenario": v.get("scenario"), "dose": fnum(v.get("realized_dose")),
                  "cooling": fnum(v.get("cooling_C")), "se": fnum(v.get("se_C")), "causal": fnum(v.get("causal_C")),
                  "frac_extrapolated": fnum(v.get("frac_extrapolated")), "unit": "°C"}
                 for q, v in (lit.get("sparc") or {}).items() if isinstance(v, dict)]
        literature = {"rows": rows, "sparc": sparc}
    return {"levers": out or None, "literature": literature}


# ---------------------------------------------------------------------------
# configured scenarios
# ---------------------------------------------------------------------------

def _tier(frac) -> str | None:
    if frac is None:
        return None
    return "in support" if frac <= 0.05 else "partly outside observed conditions" if frac <= 0.2 else "extrapolated"


def _scenario_rows(ctx) -> list[dict]:
    """Configured scenarios with their likely ranges (``scenarios.json`` / manifest + ``scenario_detail.npz``)."""
    from sparc.core.scenarios import specs_from_config

    sc = (ctx.manifest or {}).get("scenarios")
    if not isinstance(sc, list):
        sc = ctx.json("scenarios.json") or []
    tu = ctx.units.get("target")
    det = ctx.scenario_detail()
    specs = {}
    try:
        cfg = ctx.cfg
        if cfg is not None:
            specs = {s.name: s for s in specs_from_config(cfg)}
    except Exception:
        specs = {}
    slugs = {s["name"]: s["slug"] for s in ctx.configured_scenarios()}
    from sparc.studio.runs.heat import verdict_map

    verdicts = verdict_map(ctx)
    out = []
    for s in sc:
        if not isinstance(s, dict) or not s.get("name"):
            continue
        name = str(s["name"])
        spec = specs.get(name)
        single = spec is not None and len(spec.interventions) == 1 and spec.variable is not None
        cl = s.get("causal_linear")
        out.append({
            "slug": slugs.get(name) or name, "name": name, "kind": "single" if single else "joint",
            "lever": spec.variable if single else None, "dose": fnum(spec.dose) if single else None,
            "delta": likely(s.get("mean_delta"), s.get("mean_delta_se"), tu, what="the city") or
            likely(0.0, None, tu),
            "p10": fnum(s.get("p10_delta")), "p90": fnum(s.get("p90_delta")),
            "frac_extrapolated": fnum(s.get("frac_extrapolated")),
            "causal": {"delta": fnum(cl.get("delta")), "lo": fnum(cl.get("lo")), "hi": fnum(cl.get("hi")),
                       "model_within": cl.get("model_within")} if isinstance(cl, dict) else None,
            "tier": _tier(fnum(s.get("frac_extrapolated"))), "has_folds": bool(det and name in det["folds"]),
            "realized": {k2: fnum(v) for k2, v in (s.get("mean_realized") or {}).items()},
            "verdict": verdicts.get(name),
        })
    return out


def _scenarios(ctx, env) -> dict:
    rows = _scenario_rows(ctx)
    if not rows:
        return {"rows": None, "ladders": None, "has_detail": (ctx.run_dir / "scenario_detail.npz").exists()}
    levers_u = ctx.units.get("levers") or {}
    acts = ctx.cfg_raw.get("actionable") or {}
    ladders: dict[str, list] = {}
    for r in rows:
        if r["kind"] == "single" and r["lever"]:
            d = r["delta"]
            ladders.setdefault(r["lever"], []).append({"dose": r["dose"], "slug": r["slug"], "estimate": d["estimate"],
                                                       "lo": d["lo"], "hi": d["hi"],
                                                       "hollow": (r["frac_extrapolated"] or 0) > 0.2})
    lad = [{"lever": v, "label": (acts.get(v) or {}).get("label") or v.replace("_", " "), "unit": levers_u.get(v, ""),
            "points": sorted(p, key=lambda x: x["dose"] or 0)} for v, p in ladders.items()]
    return {"rows": rows, "ladders": lad, "has_detail": (ctx.run_dir / "scenario_detail.npz").exists()}


# ---------------------------------------------------------------------------
# climate
# ---------------------------------------------------------------------------

def _climate(ctx, env) -> dict:
    c = (ctx.manifest or {}).get("climate")
    if not isinstance(c, dict) or not c.get("projections"):
        return dict.fromkeys(SECTION_KEYS["climate"])
    projs = c["projections"]
    warming = []
    models: list[str] = []
    for p in projs:
        w = p.get("warming") or {}
        bm = w.get("by_model") or {}
        for k2 in bm:
            if k2 not in models:
                models.append(k2)
        warming.append({"id": f"{p.get('experiment')}:{p.get('period')}", "experiment": p.get("experiment"),
                        "label": p.get("label") or p.get("experiment"), "period": p.get("period"),
                        "n_models": p.get("n_models"), "median": fnum(w.get("median")), "p10": fnum(w.get("p10")),
                        "p90": fnum(w.get("p90")), "min": fnum(w.get("min")), "max": fnum(w.get("max")),
                        "by_model": {k2: fnum(v) for k2, v in bm.items()}})
    tu = ctx.units.get("target")
    table = _table([("model", "Model", None)] + [(w["id"], f"{w['label']} {w['period']}", tu) for w in warming],
                   [[mname] + [w["by_model"].get(mname) for w in warming] for mname in models])
    thresholds = [float(t) for t in c.get("thresholds") or []]
    tkeys = [_tkey(t) for t in thresholds]
    present = {_tkey(k2): fnum(v) for k2, v in ((c.get("present") or {}).get("share_at_or_above") or {}).items()}
    groups = []
    offset = []
    for p in projs:
        for v in p.get("variants") or []:
            sh = {}
            for k2, val in (v.get("share_at_or_above") or {}).items():
                if isinstance(val, dict):
                    sh[_tkey(k2)] = {"median": fnum(val.get("median")), "p10": fnum(val.get("p10")),
                                     "p90": fnum(val.get("p90"))}
                else:
                    sh[_tkey(k2)] = {"median": fnum(val), "p10": None, "p90": None}
            gid = f"{p.get('experiment')}:{p.get('period')}:{v.get('name')}"
            groups.append({"id": gid, "label": f"{p.get('label')} {p.get('period')} · {v.get('name')}",
                           "experiment": p.get("experiment"), "period": p.get("period"), "variant": v.get("name"),
                           "share": sh})
            if v.get("offset_share_of_median_warming") is not None:
                offset.append({"id": gid, "label": f"{p.get('label')} {p.get('period')}", "variant": v.get("name"),
                               "share": fnum(v.get("offset_share_of_median_warming"))})
    return {"warming": warming, "models_table": table,
            "exposure": {"thresholds": thresholds, "present": {k2: present.get(k2) for k2 in tkeys}, "groups": groups},
            "offset": offset, "thresholds": thresholds}


def _heat(ctx, env) -> dict:
    from sparc.studio.runs.heat import heat_view

    return heat_view(ctx, env.get("params") or {})


def _tkey(t) -> str:
    try:
        f = float(t)
        return str(int(f)) if f.is_integer() else f"{f:g}"
    except (TypeError, ValueError):
        return str(t)


# ---------------------------------------------------------------------------
# causal
# ---------------------------------------------------------------------------

def _full_section(ctx, key: str, rel: str) -> dict | None:
    """The stage file (complete) when present, else the manifest section (a summary in newer manifests)."""
    f = ctx.json(rel)
    if isinstance(f, dict) and f:
        return f
    m = (ctx.manifest or {}).get(key)
    return m if isinstance(m, dict) and m else None


def _causal_from_summary(m: dict) -> dict:
    """``causal.json``-shaped treatments from the manifest's per-treatment summary (older layout)."""
    out = {}
    for t, d in m.items():
        if t.startswith("_") or not isinstance(d, dict):
            continue
        out[t] = {"dml": {"theta": [d.get("dml_theta")], "se_cluster": [d.get("dml_se")]},
                  "spillover": {k: d.get(k) for k in ("theta_own", "se_own", "theta_nbr", "se_nbr", "theta_sum",
                                                      "se_sum")},
                  "audit": d.get("audit") if isinstance(d.get("audit"), dict) else {},
                  "sensitivity": d.get("sensitivity") if isinstance(d.get("sensitivity"), dict) else {}}
    return {"treatments": out, "flags": m.get("_flags") or []}


def _nbr_slope(me: dict) -> float | None:
    """The model's neighbourhood part of the adoption slope (adoption − own cell), beside θ_nbr."""
    a, o = fnum(me.get("adoption_slope")), fnum(me.get("own_slope"))
    return a - o if a is not None and o is not None else None


def _causal(ctx, env) -> dict:
    c = _full_section(ctx, "causal", "causal.json")
    if isinstance(c, dict) and not c.get("treatments") and c:
        c = _causal_from_summary(c)
    if not isinstance(c, dict) or not c.get("treatments"):
        return dict.fromkeys(SECTION_KEYS["causal"])
    tu = ctx.units.get("target")
    levers_u = ctx.units.get("levers") or {}
    cells_cols = set()
    try:
        import pyarrow.parquet as pq

        if (ctx.run_dir / "causal_cells.parquet").exists():
            cells_cols = set(pq.read_schema(ctx.run_dir / "causal_cells.parquet").names)
    except Exception:
        cells_cols = set()
    out = {}
    dag_rows = []
    for t, d in c["treatments"].items():
        if not isinstance(d, dict):
            continue
        sp = d.get("spillover") or {}
        dml = d.get("dml") or {}
        me = d.get("model_effects") or {}
        aud = d.get("audit") or {}

        def f(id_, label, est, se, model=None):
            est, se = fnum(est), fnum(se)
            return {"id": id_, "label": label, "est": est, "se": se,
                    "lo": est - 1.96 * se if est is not None and se is not None else None,
                    "hi": est + 1.96 * se if est is not None and se is not None else None, "model": fnum(model)}

        forest = []
        if dml.get("theta"):
            forest.append(f("dml", "DML (own cell)", (dml.get("theta") or [None])[0],
                            (dml.get("se_cluster") or [None])[0], me.get("own_slope")))
        if sp:
            forest += [f("theta_own", "Own cell", sp.get("theta_own"), sp.get("se_own"), me.get("own_slope")),
                       f("theta_nbr", "Neighbours", sp.get("theta_nbr"), sp.get("se_nbr"), _nbr_slope(me)),
                       f("theta_sum", "Own + neighbours", sp.get("theta_sum"), sp.get("se_sum"),
                         me.get("adoption_slope"))]
        audit = [{"check": k2, "label": k2.replace("_", " "), "verdict": str(v.get("verdict")),
                  "flag": bool(v.get("flag")), "model": fnum(v.get("model")), "causal": fnum(v.get("causal"))}
                 for k2, v in aud.items() if isinstance(v, dict) and "verdict" in v]
        dr = d.get("dr_curve") or {}
        dr_curve = {"t": [float(x) for x in dr.get("t_grid") or []], "theta": [fnum(x) for x in dr.get("theta") or []],
                    "lo": [fnum(x) for x in dr.get("lo") or []], "hi": [fnum(x) for x in dr.get("hi") or []],
                    "ess": fnum(dr.get("ess")), "clipped_frac": fnum(dr.get("clipped_frac")),
                    "note": dr.get("note")} if dr else None
        pd_ = me.get("own_pd_curve")
        model_pd = {"t": [float(x) for x in pd_.get("t") or []], "y": [fnum(x) for x in pd_.get("y") or []],
                    "se": [fnum(x) for x in pd_.get("se") or []]} if isinstance(pd_, dict) else None
        ct = d.get("cate") or {}
        q = ct.get("quantiles") or {}
        blp = ct.get("blp_self") or aud.get("blp_calibration") or {}
        cate = {"q": [fnum(q.get(k2)) for k2 in ("q05", "q25", "q50", "q75", "q95")], "mean": fnum(ct.get("mean")),
                "sd": fnum(ct.get("sd")), "blp": {"coef": fnum(blp.get("coef")), "se": fnum(blp.get("se")),
                                                  "p": fnum(blp.get("p"))} if blp else None} if q else None
        sens = d.get("sensitivity") or {}
        ev, rob = sens.get("e_value") or {}, sens.get("robustness") or {}
        sensitivity = {"e_value": fnum(ev.get("e_value")), "e_value_ci": fnum(ev.get("e_value_ci")),
                       "rv_q": fnum(rob.get("rv_q")), "rv_q_alpha": fnum(rob.get("rv_q_alpha")),
                       "design_effect": fnum(rob.get("design_effect"))} if sens else None
        hole = fnum(d.get("hole_scale_ratio"))
        controls = {"names": [str(x) for x in d.get("controls") or []], "basis_scale_m": fnum(d.get("basis_scale_m")),
                    "hole_scale_ratio": hole, "hole_warning": bool(hole is not None and hole > 1.0),
                    "nuisance_r2": {k2: fnum(v) for k2, v in (dml.get("nuisance_r2") or {}).items()}}
        out[t] = {"label": t.replace("_", " "), "unit": f"{tu} per {levers_u.get(t, 'unit')}", "forest": forest,
                  "audit": audit, "dr_curve": dr_curve, "model_pd_curve": model_pd, "cate": cate,
                  "cate_layer": f"cate_{t}" if f"cate:{t}" in cells_cols else None, "sensitivity": sensitivity,
                  "controls": controls}
        da = d.get("dag_audit")
        if isinstance(da, dict):
            for k2, v in da.items():
                dag_rows.append([t, str(k2), v if isinstance(v, (int, float, str, bool)) or v is None else str(v)])
        elif isinstance(da, list):
            for v in da:
                dag_rows.append([t, str(v.get("check") if isinstance(v, dict) else v),
                                 str(v.get("verdict")) if isinstance(v, dict) else None])
    flags = [{"treatment": str(f.get("treatment")), "check": str(f.get("check")), "verdict": str(f.get("verdict"))}
             for f in c.get("flags") or [] if isinstance(f, dict)]
    dag = _table([("treatment", "Treatment", None), ("check", "Check", None), ("result", "Result", None)],
                 dag_rows) if dag_rows else None
    return {"treatments": out, "flags": flags, "dag_audit": dag}


# ---------------------------------------------------------------------------
# budget
# ---------------------------------------------------------------------------

def _budget(ctx, env) -> dict:
    o = _full_section(ctx, "optimize", "optimize.json")
    if isinstance(o, dict) and isinstance((ctx.manifest or {}).get("optimize"), dict):
        o = {**ctx.manifest["optimize"], **o}
    if not isinstance(o, dict) or not o:
        return dict.fromkeys(SECTION_KEYS["budget"])
    tu = ctx.units.get("target")
    var = o.get("variable")
    lu = (ctx.units.get("levers") or {}).get(var, "")
    status = str(o.get("status") or "ok")
    if status != "ok" and o.get("planned_total_cooling") is None:
        return {"kpis": [], "pareto": None, "caption": "No cell has a positive cooling footprint for this lever, so "
                "the optimiser had nothing to allocate.", "top_cells": [], "status": status}
    planned, realised = fnum(o.get("planned_total_cooling")), fnum(o.get("realized_total_cooling"))
    kpis = [
        _kpi("cells", "Cells treated", o.get("n_cells_treated"), fmt="int", decimals=0),
        _kpi("mean_dose", "Mean dose (treated)", fnum(o.get("mean_dose_treated")), unit=lu, decimals=1),
        _kpi("planned", "Planned cooling", planned, unit=f"{tu}·cells", decimals=0),
        _kpi("realised", "Realised cooling (closed loop)", realised, unit=f"{tu}·cells", decimals=0,
             note="spillover non-additivity" if planned and realised and abs(realised - planned) > 0.05 * abs(planned)
             else None),
        _kpi("cost", "Cost", fnum(o.get("total_cost")), decimals=0),
        _kpi("gini", "Gini of the allocation", fnum(o.get("gini")), decimals=2),
        _kpi("constraint", "Constraint", o.get("constraint"), fmt="text"),
        _kpi("objective", "Objective", o.get("objective"), fmt="text"),
    ]
    pts = ((o.get("pareto") or {}).get("points") or [])
    points = [{"budget": float(p["budget"]), "total_benefit": fnum(p.get("total_benefit")),
               "n_segments": p.get("n_segments"), "gini": fnum(p.get("gini"))} for p in pts if p.get("budget") is not None]
    budget = fnum(o.get("budget"))
    pareto = {"points": points, "realised": [{"budget": budget, "total_benefit": realised}]
              if budget is not None and realised is not None else None, "budget": budget, "unit": f"{tu}·cells",
              "budget_unit": "cost units"}
    caption = _pareto_caption(points, budget, tu)
    top = []
    al = ctx.parquet("allocation.parquet")
    g = ctx.grid
    if al is not None and "dose" in al.columns:
        dose = al["dose"].to_numpy(float)
        cl = al["closed_loop_delta"].to_numpy(float) if "closed_loop_delta" in al.columns else None
        # the treated cells that cool most (closed loop, as the table shows), ties and cells without a
        # closed-loop value by dose, then by row - never the row order of the many cells sharing one dose
        treated = np.flatnonzero(np.isfinite(dose) & (dose > 0))
        benefit = -cl[treated] if cl is not None else np.full(treated.size, np.nan)
        order = treated[np.lexsort((treated, -dose[treated], -np.nan_to_num(benefit, nan=-np.inf)))][:25]
        for rank, i in enumerate(order, start=1):
            top.append({"rank": rank, "row": int(i), "id": clean(al["id"].iloc[i]) if "id" in al.columns else int(i),
                        "lon": fnum(g.lon[i]) if g is not None else None, "lat": fnum(g.lat[i]) if g is not None else None,
                        "dose": fnum(dose[i]), "benefit": fnum(-cl[i]) if cl is not None else None,
                        "zone": clean(g.zones[g.zone[i]]) if g is not None and g.zones and g.zone[i] >= 0 else None})
    return {"kpis": kpis, "pareto": pareto, "caption": caption, "top_cells": top, "status": status}


def _pareto_caption(points: list[dict], budget, tu) -> str:
    """'Doubling the budget …' from the Pareto points (computed, never hard-coded)."""
    by = {p["budget"]: p["total_benefit"] for p in points if p["total_benefit"] is not None}
    if budget is not None and budget in by and 2 * budget in by and by[budget]:
        gain = by[2 * budget] / by[budget]
        return (f"Doubling the budget from {budget:,.0f} to {2 * budget:,.0f} multiplies the planned cooling by "
                f"{gain:.2f}× (open loop) — {'diminishing' if gain < 2 else 'no diminishing'} returns. "
                "Points count segments, not cells.")
    if len(by) >= 2:
        b = sorted(by)
        return (f"Planned cooling rises from {by[b[0]]:,.0f} to {by[b[-1]]:,.0f} {tu}·cells as the budget goes from "
                f"{b[0]:,.0f} to {b[-1]:,.0f} (open loop; points count segments, not cells).")
    return "Open-loop planned cooling per budget level."


# ---------------------------------------------------------------------------
# planner
# ---------------------------------------------------------------------------

def _planner(ctx, env) -> dict:
    p = (ctx.manifest or {}).get("planner")
    if not isinstance(p, dict):
        p = ctx.json("planner/planner.json")
    if not isinstance(p, dict) or not p:
        return dict.fromkeys(SECTION_KEYS["planner"])
    tu = ctx.units.get("target")
    thresholds = [float(t) for t in p.get("thresholds") or []]
    tk = [_tkey(t) for t in thresholds]
    exp_groups, pm_rows = [], []
    for r in p.get("exposure") or []:
        label = f"{r.get('case')}{' (adapted)' if r.get('adapted') else ''}"
        exp_groups.append({"id": label, "label": label,
                           "share": [fnum(r.get(f"share_people_ge_{t}")) for t in tk]})
        pm_rows.append([str(r.get("case")), bool(r.get("adapted")), fnum(r.get("person_mean_temp"))])
    hd = p.get("hot_days") or {}
    panels: dict[str, dict] = {}
    for case in hd.get("cases") or []:
        cid = str(case.get("case"))
        pan = panels.setdefault(cid, {"id": cid, "label": cid, "thresholds": thresholds,
                                      "campaign": [None] * len(tk), "lower": [None] * len(tk)})
        which = "campaign" if case.get("label") == "campaign-like" else "lower"
        pan[which] = [fnum(case.get(f"days_ge_{t}")) for t in tk]
    eq = p.get("equity") or {}
    measures = list(eq)
    nq = max((len((eq[mm] or {}).get("quintiles") or []) for mm in measures), default=0)
    quint = [{"label": f"Q{i + 1}", "values": {mm: fnum((((eq[mm] or {}).get("quintiles") or [])[i] or {})
                                                     .get("mean_cooling")) if i < len((eq[mm] or {}).get("quintiles") or [])
                                              else None for mm in measures}} for i in range(nq)]
    conc = [{"label": mm, "value": fnum((eq[mm] or {}).get("concentration_index"))} for mm in measures]
    pl = p.get("plantable") or {}
    plantable = [_kpi("headroom", "Mean plantable headroom", fnum(pl.get("mean_headroom_pp")), unit="pp", decimals=1),
                 _kpi("total", "Total plantable headroom", fnum(pl.get("total_pp_cells")), unit="pp·cells", decimals=0),
                 _kpi("no_room", "Cells with no room", fnum(pl.get("share_cells_no_room")), fmt="percent", decimals=1),
                 _kpi("paved", "Paved share assumed plantable", fnum(pl.get("paved_share")), fmt="percent",
                      decimals=0)] if pl else []
    zrows = p.get("zones") or []
    zcols = list(zrows[0].keys()) if zrows else []
    zones = _table([(c, c.replace("_", " "), tu if c in ("temperature", "package_cooling") else None) for c in zcols],
                   [[clean(r.get(c)) for c in zcols] for r in zrows]) if zrows else None
    hex_files = []
    for f in p.get("hex_files") or []:
        m = re.search(r"(\d+)m", str(f))
        hex_files.append({"size_m": int(m.group(1)) if m else None, "relpath": f"planner/{f}", "format": "csv"})
    if p.get("geopackage"):
        hex_files.append({"size_m": None, "relpath": f"planner/{p['geopackage']}", "format": "gpkg"})
    g = ctx.grid
    sites = []
    if (ctx.run_dir / "planner" / "logger_sites.csv").exists() and g is not None:
        import pandas as pd

        ls = pd.read_csv(ctx.run_dir / "planner" / "logger_sites.csv")
        for _i, r in ls.iterrows():
            row = int(r["cell"])
            if 0 <= row < g.n:
                sites.append({"row": row, "id": clean(g.ids[row]), "lon": fnum(g.lon[row]), "lat": fnum(g.lat[row]),
                              "label": str(r.get("role")) if "role" in ls.columns else None})
    pairs = []
    if (ctx.run_dir / "planner" / "before_after_pairs.csv").exists():
        import pandas as pd

        try:
            bp = pd.read_csv(ctx.run_dir / "planner" / "before_after_pairs.csv")
            pairs = [{"treated": int(a), "control": int(b)} for a, b in zip(bp["treated"], bp["control"])]
        except Exception:
            pairs = []
    gis = []
    gdir = ctx.run_dir / "planner" / "geotiff"
    if gdir.is_dir():
        for f in sorted(gdir.glob("*.tif")):
            gis.append({"relpath": f"planner/geotiff/{f.name}", "kind": "geotiff", "bytes": f.stat().st_size,
                        "layer": f.stem})
    if (ctx.run_dir / "planner" / "hexagons.gpkg").exists():
        f = ctx.run_dir / "planner" / "hexagons.gpkg"
        gis.append({"relpath": "planner/hexagons.gpkg", "kind": "gpkg", "bytes": f.stat().st_size, "layer": None})
    return {"exposure": {"thresholds": thresholds, "groups": exp_groups},
            "person_mean": _table([("case", "Case", None), ("adapted", "Adapted", None),
                                   ("person_mean_temp", "Person-mean temperature", tu)], pm_rows),
            "hot_days": {"unit": "days/yr", "panels": list(panels.values())} if panels else None,
            "equity": {"measures": measures, "quintiles": quint, "concentration": conc} if measures else None,
            "plantable": plantable, "zones": zones, "hex_files": hex_files, "sites": sites, "pairs": pairs,
            "gis": gis}


# ---------------------------------------------------------------------------
# uncertainty
# ---------------------------------------------------------------------------

def _uncertainty(ctx, env) -> dict:
    u = (ctx.manifest or {}).get("uncertainty")
    if not isinstance(u, dict):
        u = ctx.json("uncertainty.json")
    if not isinstance(u, dict) or not u:
        return dict.fromkeys(SECTION_KEYS["uncertainty"])
    from sparc.core.catalog import scenario_slug
    from sparc.studio.runs.heat import verdict_map

    verdicts = verdict_map(ctx)
    rows = []
    taken: set[str] = set()
    for s in u.get("scenarios") or []:
        layers = []
        for key, label in (("estimation_95", "Estimation (95%)"), ("specification", "Specification"),
                           ("attribution", "Attribution"), ("causal_band", "Causal band"), ("envelope", "Envelope")):
            v = s.get(key)
            if isinstance(v, (list, tuple)) and len(v) == 2:
                layers.append({"id": key, "label": label, "lo": fnum(v[0]), "hi": fnum(v[1])})
        slug = scenario_slug(str(s.get("scenario")), taken)
        taken.add(slug)
        rows.append({"id": slug, "label": str(s.get("scenario")), "estimate": fnum(s.get("estimate")), "layers": layers,
                     "verdict": verdicts.get(str(s.get("scenario")))})
    tu = ctx.units.get("target")
    clim = u.get("climate") or []
    ctab = _table([("experiment", "Scenario", None), ("period", "Period", None), ("n_models", "Models", None),
                   ("median", "Median", tu), ("p10", "p10", tu), ("p90", "p90", tu), ("min", "Min", tu), ("max", "Max", tu)],
                  [[c.get("label") or c.get("experiment"), c.get("period"), c.get("n_models"), fnum(c.get("median")),
                    fnum(c.get("p10")), fnum(c.get("p90")), fnum(c.get("min")), fnum(c.get("max"))] for c in clim]) \
        if clim else None
    return {"rows": rows, "climate": ctab, "sources": _uncertainty_sources(ctx, u, env.get("studies_index") or [])}


_STUDY_STATE = {"succeeded": "done", "complete": "done", "done": "done", "running": "running", "starting": "running",
                "cancelling": "running", "queued": "queued", "blocked": "queued", "failed": "failed",
                "interrupted": "failed", "cancelled": "failed", "stale": "stale"}


def _uncertainty_sources(ctx, u: dict, index: list[dict]) -> list[dict]:
    """The studies behind the envelopes (``uncertainty.json`` ``sources``: folder paths) matched to the studies
    index, then the project's other multiverse / simcheck / placebo studies, so each can be attached or
    detached (``study_id``, ``attached``, ``state``)."""
    from pathlib import Path

    def key(p) -> str | None:
        try:
            return str(Path(p).expanduser().resolve())
        except (OSError, RuntimeError, TypeError, ValueError):
            return None

    kinds = ("multiverse", "simcheck", "placebo")
    by_dir = {key(r["out_dir"]): r for r in index if r.get("out_dir") and r["kind"] in kinds}
    out, used = [], set()

    def add(kind: str, label: str, row: dict | None, attached_default: bool | None) -> None:
        if row is not None:
            used.add(row["id"])
        st = (row or {}).get("status")
        out.append({"kind": kind, "label": label, "study_id": (row or {}).get("id"),
                    "attached": bool((row or {}).get("attached")) if row and row.get("attached") is not None
                    else attached_default,
                    "state": _STUDY_STATE.get(str(st), str(st)) if row and st else "done"})

    src = u.get("sources") or {}
    if src.get("multiverse"):
        row = by_dir.get(key(src["multiverse"]))
        add("multiverse", Path(str(src["multiverse"])).name or "multiverse", row, True)
    for d in src.get("simcheck") or []:
        row = by_dir.get(key(d))
        add("simcheck", Path(str(d)).name or "simcheck", row, True)
    if (ctx.manifest or {}).get("placebo"):
        row = next((r for r in index if r["kind"] == "placebo" and (r.get("attached") or
                                                                    r.get("target_run_id") == ctx.run_id)), None)
        add("placebo", "placebo.json (attached)", row, True)
    for r in index:
        if r["kind"] in kinds and r["id"] not in used:
            label = ((r.get("summary") or {}).get("label") if isinstance(r.get("summary"), dict) else None) or \
                Path(str(r.get("out_dir") or r["id"])).name
            used.add(r["id"])
            out.append({"kind": r["kind"], "label": str(label), "study_id": r["id"], "attached": bool(r.get("attached")),
                        "state": _STUDY_STATE.get(str(r.get("status")), str(r.get("status") or "not_run"))})
    return out


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------

def _provenance(ctx, env) -> dict:
    m = ctx.manifest or {}
    prov = m.get("provenance") or {}
    ck = ctx.checkpoint_json or {}
    hashes = {"input": prov.get("input_sha256"), "config": prov.get("config_sha256"),
              "code": prov.get("code_sha256") or ck.get("code_sha256"), "fingerprint": ck.get("fingerprint")
              or (ctx.run_state or {}).get("fingerprint")}
    for k2, v in (ck.get("sections") or {}).items():
        hashes[f"section:{k2}"] = v
    git = prov.get("git") or {}
    launch = dict(ctx.launch or {})
    if "config_raw" in launch:
        launch["config_raw"] = "(see GET /api/runs/{rid}/config)"
    return {"hashes": hashes, "git": {"commit": git.get("commit") or m.get("git_commit"),
                                      "dirty": git.get("core_dirty"), "branch": git.get("branch")},
            "platform": {**{k2: str(v) for k2, v in (prov.get("platform") or {}).items()},
                         **{f"version:{k2}": str(v) for k2, v in (m.get("versions") or {}).items()}},
            "launch": launch or None}


_BUILDERS: dict[str, Callable] = {"overview": _overview, "data": _data, "accuracy": _accuracy, "distance": _distance,
                                  "influence": _influence, "response": _response, "scenarios": _scenarios,
                                  "climate": _climate, "heat": _heat, "causal": _causal, "budget": _budget, "planner": _planner,
                                  "uncertainty": _uncertainty, "provenance": _provenance}


def build_view(ctx, view: str, env: dict) -> dict:
    """The ``ViewModel`` of ``view`` (``env``: outputs, tabs, stage rows, studies, findings, headline)."""
    if view not in _BUILDERS:
        raise ApiError("unknown_view", f"unknown view {view!r}", detail={"views": list(VIEWS)})
    from sparc.studio.runs.caveats import caveats_for

    tab = next((t for t in env["tabs"] if t["id"] == view), {"availability": "ready", "missing": []})
    sections = _safe(lambda: _BUILDERS[view](ctx, env), view, ctx) or {}
    full = {k2: sections.get(k2) for k2 in SECTION_KEYS[view]}
    return clean({"view": view, "availability": tab["availability"],
                  "missing": [{k2: x.get(k2) for k2 in ("output", "produced_by", "action")} for x in tab["missing"]],
                  "units": {"target": ctx.units.get("target"), "levers": ctx.units.get("levers") or {}},
                  "caveats": _safe(lambda: caveats_for(ctx), "caveats", ctx) or [], "demo": bool(env.get("demo")),
                  "sections": full})
