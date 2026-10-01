"""Manifest + markdown report for a core run."""

from __future__ import annotations

import platform
import subprocess
from datetime import datetime, timezone

import numpy as np


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return None


def _versions() -> dict:
    out = {"python": platform.python_version()}
    for mod in ("numpy", "pandas", "scipy", "sklearn", "torch"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    return out


def _physics_summary(ens) -> dict | None:
    if ens is None or not ens.has_physics:
        return None
    ps = [st.physics.params for st in ens.stacks]
    keys = [k for k in ps[0] if isinstance(ps[0][k], (int, float)) and not isinstance(ps[0][k], bool)]
    return {k: {"mean": float(np.mean([p[k] for p in ps])), "sd": float(np.std([p[k] for p in ps]))} for k in keys}


def _first(v):
    if isinstance(v, (list, tuple)) and v:
        return _first(v[0])
    return v


def _causal_summary(causal: dict) -> dict:
    out = {}
    for tr, r in ((causal or {}).get("treatments") or {}).items():
        if not isinstance(r, dict):
            continue
        dml = r.get("dml") or {}
        sp = r.get("spillover") or {}
        dr = r.get("dr_curve") or {}
        out[tr] = {
            "dml_theta": _first(dml.get("theta")), "dml_se": _first(dml.get("se_cluster")),
            "theta_own": sp.get("theta_own"), "se_own": sp.get("se_own"),
            "theta_nbr": sp.get("theta_nbr"), "se_nbr": sp.get("se_nbr"),
            "theta_sum": sp.get("theta_sum"), "se_sum": sp.get("se_sum"),
            "dr_ess": dr.get("ess"),
            "sensitivity": r.get("sensitivity"),
            "audit": r.get("audit"),
        }
    if causal and causal.get("flags"):
        out["_flags"] = causal["flags"]
    return out


def build_manifest(result, timings: dict, fast: bool, folds=None) -> dict:
    ens = result.ensemble
    inf = result.influence
    m = {
        "name": result.cfg.name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "versions": _versions(),
        "fast_mode": fast,
        "config": result.cfg.raw,
        "qa": result.data.qa,
        "n_points": result.data.n,
        "timings_s": {k: round(v, 2) for k, v in timings.items()},
    }
    if inf is not None:
        m["influence"] = {"ranges_m": inf.ranges_m, "target_resid_range_m": inf.target_resid_range_m,
                          "L_prior_m": inf.L_prior_m, "block_size_m": inf.block_size_m,
                          "anisotropy": inf.anisotropy}
    if folds is not None:
        m["cv"] = {"n_folds": folds.n_folds, "block_m": folds.block_m, "buffer_m": folds.buffer_m,
                   "test_sizes": [int(t.sum()) for t in folds.test_masks], "n_blocks": folds.n_blocks}
    if ens is not None:
        m["metrics"] = ens.metrics
        m["lambda_pde"] = ens.lambda_pde
        m["lambda_scores"] = ens.lambda_scores
        m["physics"] = _physics_summary(ens)
        m["physics_advection"] = ens.physics_selection
        m["stacker"] = ens.stacker_info
        m["stacker_choice"] = ens.stacker_choice
        m["spatial_plus"] = (result.cfg.raw.get("models") or {}).get("spatial_plus")
    if getattr(result, "baselines", None):
        m["baselines"] = result.baselines
    if getattr(result, "cv_distance", None):
        m["cv_distance"] = result.cv_distance
    if result.responses:
        m["response"] = {v: r.summary for v, r in result.responses.items()}
    if result.scenarios:
        m["scenarios"] = result.scenarios
    if getattr(result, "climate", None):
        m["climate"] = result.climate
    if result.causal:
        m["causal"] = _causal_summary(result.causal)
    if result.optimize:
        m["optimize"] = {k: v for k, v in result.optimize.items() if k not in ("dose", "closed_loop_delta", "pareto")}
    return m


def _f(v, nd=3):
    if v is None:
        return "—"
    try:
        if not np.isfinite(float(v)):
            return "—"
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def render_report(result) -> str:
    m = result.manifest
    u = result.data.target_units or ""
    L = [f"# SPARC core run — {m['name']}", "",
         f"*{m['created_utc']}* · commit `{m.get('git_commit')}` · {m['n_points']} points"
         + (" · **fast mode**" if m.get("fast_mode") else ""), ""]
    qa = m.get("qa", {})
    L += ["## S0 — Data", "",
          f"- grid {qa.get('grid_shape')} at {_f(qa.get('cell_m'), 2)} m, fill {_f(qa.get('grid_fill_fraction'), 2)}",
          f"- background (ΔT reference): {_f(qa.get('background'), 2)} {u} ({qa.get('background_source')})",
          f"- target values that are whole numbers: {_f(qa.get('target_fraction_integer_valued'), 2)} "
          f"(rounding noise ≈ {_f(qa.get('target_rounding_noise_sd'), 2)} {u})",
          f"- QA clips: {qa.get('clipped') or 'none'}"]
    if qa.get("coarse"):
        co = qa["coarse"]
        L.append(f"- coarse mode: {co['n_fine']:,} input cells → {co['n_cells']:,} cells of {co['cell_m']:g} m "
                 f"(mean {_f(co['members_mean'], 1)} inputs per cell; {co['frac_partial']:.0%} partial edge cells)")
    L.append("")
    if qa.get("flags"):
        L += ["**Data findings** (read before using the results):", ""]
        L += [f"- {'⚠' if f['severity'] == 'warn' else 'ℹ'} {f['message']}" for f in qa["flags"]]
        L.append("")
    if "influence" in m:
        inf = m["influence"]
        L += ["## S1 — Area of influence", "",
              f"Target residual range {_f(inf['target_resid_range_m'], 0)} m → physics length prior "
              f"L₀ = {_f(inf['L_prior_m'], 0)} m; CV block {_f(inf['block_size_m'], 0)} m.", "",
              "| variable | influence range (m) | anisotropy b/a | θ (°) | reliable |", "|---|---|---|---|---|"]
        for v, r in inf["ranges_m"].items():
            a = (inf.get("anisotropy") or {}).get(v, {})
            L.append(f"| {v} | {_f(r, 0)} | {_f(a.get('ratio'), 2)} | {_f(a.get('theta_deg'), 0)} | {a.get('reliable')} |")
        L.append("")
    if "metrics" in m:
        cv = m.get("cv") or {}
        L += ["## S2/S3 — Out-of-fold performance (spatial blocks, cross-fitted stacking)", "",
              f"{cv.get('n_folds')} folds of square blocks ({_f(cv.get('block_m'), 0)} m, {cv.get('n_blocks')} blocks "
              f"with data); training points within {_f(cv.get('buffer_m'), 0)} m of a test point are dropped.", "",
              f"| model | RMSE ({u}) | MAE | R² |", "|---|---|---|---|"]
        for k, v in m["metrics"].items():
            L.append(f"| {k} | {_f(v['rmse'])} | {_f(v['mae'])} | {_f(v['r2'])} |")
        s = m["metrics"].get("stacker", {})
        lam = m.get("lambda_pde")
        lam_txt = m.get("stacker_choice") or ("residual off — convex base only" if lam is None else f"λ_PDE = {lam}")
        L += ["", f"Stacker: {lam_txt} (out-of-fold RMSE by candidate: {m.get('lambda_scores')}); "
              f"{int(100 * s.get('interval_target', 0.9))}% cross-conformal interval coverage "
              f"{_f(s.get('interval_coverage'), 3)} (mean half-width {_f(s.get('interval_mean_halfwidth'), 3)} {u}).", ""]
        diag = s.get("interval_diagnostics")
        if diag:
            L += ["**Interval honesty** — pooled coverage is nearly guaranteed by construction, so coverage is also "
                  "shown per fold, by distance from training data and per zone (global vs distance-adaptive "
                  "cross-conformal intervals):", "", "| group | n | global coverage | adaptive coverage | "
                  f"global ± ({u}) | adaptive ± ({u}) |", "|---|---|---|---|---|---|"]
            for gname, groups in diag.items():
                rows = {"all": groups} if gname == "overall" else groups
                for k, v in rows.items():
                    label = "all" if gname == "overall" else f"{gname.replace('by_', '')} {k}"
                    L.append(f"| {label} | {v['n']} | {_f(v['global'], 3)} | {_f(v['adaptive'], 3)} | "
                             f"{_f(v['halfwidth_global'], 2)} | {_f(v['halfwidth_adaptive'], 2)} |")
            L.append("")
        if m.get("stacker"):
            st = m["stacker"]
            wts = [x.get("weights") or {} for x in st]
            names = list(wts[0]) if wts and wts[0] else []
            if names:
                L += ["**Stacker base weights (mean over folds):** " + ", ".join(
                    f"{n} {np.mean([w.get(n, 0.0) for w in wts]):.2f}" for n in names)
                      + f"; neural residual kept in {sum(not x['residual_gated_off'] for x in st)}/{len(st)} folds "
                      "(gated off where it did not beat the convex base on held-out inner blocks).", ""]
        if m.get("physics"):
            p = m["physics"]
            L += ["**Physics (mean ± sd over folds):** " + ", ".join(
                f"{k} = {_f(p[k]['mean'], 3)} ± {_f(p[k]['sd'], 3)}" for k in ("L_m", "vx_m", "vy_m", "a", "gamma") if k in p), ""]
        ph = (m.get("config") or {}).get("physics") or {}
        fi = ph.get("forcing_info")
        if fi:
            L += [f"**Forcing:** {fi.get('date')} {fi.get('hours_local')} local ({fi.get('tz')}) from `{fi.get('file')}` — "
                  f"SW↓ {_f(ph.get('sw_down'), 0)} W/m², net LW {_f(ph.get('lw_net'), 0)} W/m², wind {ph.get('wind')} m/s "
                  f"({fi.get('wind_source')}" + (f" {fi['station']}" if fi.get("station") else "") + "). "
                  + " · ".join(fi.get("checks") or []), ""]
        else:
            L += [f"**Forcing:** generic (SW↓ {_f(ph.get('sw_down'), 0)} W/m², net LW {_f(ph.get('lw_net'), 0)} W/m², "
                  f"wind {ph.get('wind')}); run `sparc core forcing` for the campaign day.", ""]
        adv = m.get("physics_advection")
        if adv:
            L += [f"**Advection:** {'kept' if adv['kept'] else 'dropped'} — held-out RMSE with − without advection "
                  f"= {_f(adv['delta_rmse_mean'], 4)} ± {_f(adv['delta_rmse_se'], 4)} {u} (kept only if lower by > 1 SE).", ""]
    if m.get("baselines"):
        b = m["baselines"]
        L += ["### Stack vs standard baselines (same folds, paired by CV block)", "",
              "| baseline | RMSE | R² | ΔRMSE vs stack | ΔMSE ± SE (block-clustered) | blocks where stack better |",
              "|---|---|---|---|---|---|"]
        for k, r in b["rows"].items():
            L.append(f"| {r['label']} | {_f(r['rmse'])} | {_f(r['r2'])} | {r['delta_rmse']:+.3f} | "
                     f"{r['delta_mse']:+.3f} ± {_f(r['delta_mse_se'])} | {r['frac_blocks_stack_better']:.0%} |")
        L += ["", f"Positive Δ = the stack is better. **Verdict:** {b['verdict']}.", ""]
    if m.get("cv_distance"):
        rows = m["cv_distance"]["rows"]
        names = list(rows[0]["models"]) if rows else []
        L += ["### Skill vs distance from training data (reporting only)", "",
              "| CV partition | block / buffer (m) | blocks | train kept | stacker R² | stacker RMSE | fold R² range | "
              + " | ".join(f"{n} R²" for n in names) + " |",
              "|---|---|---|---|---|---|---|" + "---|" * len(names)]
        for r in rows:
            L.append(f"| {r['label']} | {_f(r['block_m'], 0)} / {_f(r['buffer_m'], 0)} | {r['n_blocks']} | "
                     f"{_f(r['train_fraction_kept'], 2)} | {_f(r['stacker']['r2'])} | {_f(r['stacker']['rmse'])} | "
                     f"{_f(r['fold_r2_min'], 2)} – {_f(r['fold_r2_max'], 2)} | "
                     + " | ".join(_f(r["models"].get(n, {}).get("r2")) for n in names) + " |")
        bnames = sorted({k for r in rows for k in (r.get("baselines") or {})})
        if bnames:
            L += ["", "| CV partition | stacker R² | " + " | ".join(f"{n} R²" for n in bnames) + " |",
                  "|---|---|" + "---|" * len(bnames)]
            for r in rows:
                bl = r.get("baselines") or {}
                L.append(f"| {r['label']} | {_f(r['stacker']['r2'])} | " + " | ".join(
                    (_f(bl[n]["r2"]) + (" ▲" if bl[n]["baseline_better"] else " ▼" if bl[n]["stack_better"] else ""))
                    if n in bl else "—" for n in bnames) + " |")
            L += ["", "▼ stack better by > 2 block-clustered SE, ▲ baseline better.", ""]
        L += ["", "Random points leave every test point next to training data, so that row mostly measures "
              "interpolation and overstates skill for new areas. The main blocks (≥ the target's residual "
              "correlation range, buffer = block/3) measure prediction for an unseen neighbourhood; stacking, "
              "model selection and intervals use them.", ""]
    if "response" in m:
        L += ["## S4 — Saturation and marginal effects", "",
              "| variable | saturating | censored | linear | median d90 | median max cooling | own effect/unit | footprint/unit |",
              "|---|---|---|---|---|---|---|---|"]
        for v, s in m["response"].items():
            L.append(f"| {v} | {_f(s['frac_saturating'], 2)} | {_f(s['frac_censored'], 2)} | {_f(s['frac_linear'], 2)} | "
                     f"{_f(s['median_d90'], 2)} | {_f(s['median_max_cooling'], 3)} | {_f(s['mean_own_effect'], 4)} | "
                     f"{_f(s['mean_footprint_effect'], 4)} |")
        L += ["", "*Own effect*: only the cell itself changes. *Footprint*: total change summed over the "
              "neighbourhood (drives the optimiser). Saturation is fitted to neighbourhood-adoption sweeps "
              "against the realised neighbourhood dose; 'censored' = no knee within the tested doses/headroom.", ""]
    if "scenarios" in m:
        L += ["## S5 — Scenarios", "", f"| scenario | mean Δ ({u}) | p10 | p90 | SE of mean (jackknife) | extrapolated | "
              f"linear causal Δ (95%) |", "|---|---|---|---|---|---|---|"]
        for s in m["scenarios"]:
            c = s.get("causal_linear")
            ctext = (f"{_f(c['delta'])} ({_f(c['lo'])} to {_f(c['hi'])})" + ("" if c["model_within"] else " ⚑")) if c else "—"
            L.append(f"| {s['name']} | {_f(s['mean_delta'])} | {_f(s['p10_delta'])} | {_f(s['p90_delta'])} | "
                     f"{_f(s.get('mean_delta_se', s['mean_delta_sd']))} | {_f(s['frac_extrapolated'], 3)} | {ctext} |")
        L += ["", "Linear causal Δ: the S6 own + neighbour effect × the mean realised change (a local-slope "
              "extrapolation); ⚑ = the model's mean Δ lies outside its 95% band.", ""]
    if m.get("climate"):
        c = m["climate"]
        thr = c["thresholds"][-1]
        L += ["## Climate projections (CMIP6 delta method)", "",
              f"Change in {'-'.join(str(x) for x in c['months'])} mean daily {c['variable']} vs {c['baseline']}, "
              f"{c['n_models']} CMIP6 models at {c['site']['lat']:.3f}°, {c['site']['lon']:.3f}° "
              f"(land-weighted bilinear).  Future = observed + model warming (+ adaptation Δ).", "",
              f"| pathway | period | warming median (10–90%) | share ≥ {thr:g} {u}: no adaptation | " +
              " | ".join(f"with {a}" for a in c["adaptation"]) + " |",
              "|---|---|---|---|" + "---|" * len(c["adaptation"])]
        L.append(f"| today | — | — | {_f(100 * c['present']['share_at_or_above'][str(float(thr))], 1)}% |" +
                 " — |" * len(c["adaptation"]))
        for p in c["projections"]:
            w = p["warming"]
            cells = [f"{_f(100 * v['share_at_or_above'][str(float(thr))]['median'], 1)}%" for v in p["variants"]]
            L.append(f"| {p['label']} | {p['period']} | {_f(w['median'], 2)} ({_f(w['p10'], 2)} to {_f(w['p90'], 2)}) | "
                     + " | ".join(cells) + " |")
        L += ["", "Delta method: the adaptation effect is assumed not to change with background warming; daily "
              "extremes may warm more than the seasonal mean of daily maxima.", ""]
    if "causal" in m:
        L += ["## S6 — Causal validation", "",
              "| treatment | DML θ ± SE | θ_own | θ_nbr | θ_own+θ_nbr | audit |", "|---|---|---|---|---|---|"]
        for tr, c in m["causal"].items():
            if tr.startswith("_"):
                continue
            aud = c.get("audit") or {}
            verdicts = "; ".join(f"{k}: {v.get('verdict')}" for k, v in aud.items() if isinstance(v, dict) and "verdict" in v)
            L.append(f"| {tr} | {_f(c.get('dml_theta'), 4)} ± {_f(c.get('dml_se'), 4)} | {_f(c.get('theta_own'), 4)} | "
                     f"{_f(c.get('theta_nbr'), 4)} | {_f(c.get('theta_sum'), 4)} | {verdicts or '—'} |")
        L += ["", "The audit compares the model's implied effects with the causal estimates on matched estimands: "
              "neighbourhood-adoption slope ↔ θ_own + θ_nbr, own-only slope ↔ θ_own, own-only partial dependence ↔ "
              "doubly-robust dose-response.", ""]
    if "optimize" in m:
        o = m["optimize"]
        L += ["## S7 — Budget allocation", "",
              f"{o.get('variable')}: budget {_f(o.get('budget'), 0)} → {o.get('n_cells_treated')} cells treated "
              f"(mean dose {_f(o.get('mean_dose_treated'), 2)}); planned total cooling {_f(o.get('planned_total_cooling'), 2)}, "
              f"closed-loop realised {_f(o.get('realized_total_cooling'), 2)} ({u}·cells).", ""]
    L += ["## Timings (s)", "", ", ".join(f"{k}: {v}" for k, v in m.get("timings_s", {}).items()), ""]
    return "\n".join(L)
