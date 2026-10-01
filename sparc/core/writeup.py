"""Auto-written methods section and model card for a run (from its manifest).

``methods.md`` is a draft methods section with every number filled in from
the run; ``model_card.md`` states intended use, performance by distance from
training data, the effect checks, noise floor and limitations.  Both are
written next to ``report.md``; edit before publishing, but never by hand-
copying numbers: re-run instead.
"""

from __future__ import annotations

import numpy as np

MODEL_TEXT = {
    "ols": "ridge-regularised linear regression on the predictors and their neighbourhood means",
    "mgwr": "multiscale geographically weighted regression (Fotheringham et al. 2017)",
    "gwrf": "geographically weighted random forest",
    "gam": "generalised additive model with a spatial smooth (Wood 2017)",
    "physics": "a physics model: a surface-energy source term (shortwave absorption, canopy shading, "
               "impervious heat storage, evapotranspiration) propagated by an advection–diffusion–relaxation "
               "operator",
}


def _months(ms) -> str:
    ms = [int(x) for x in (ms or [])]
    if not ms:
        return "summer"
    names = "January February March April May June July August September October November December".split()
    return names[ms[0] - 1] if len(ms) == 1 else f"{names[ms[0] - 1]}–{names[ms[-1] - 1]}"


def _f(v, nd=2, unit=""):
    try:
        if v is None or not np.isfinite(float(v)):
            return "—"
    except (TypeError, ValueError):
        return str(v)
    return f"{float(v):.{nd}f}{unit}"


def _forcing(m: dict) -> tuple[str, dict]:
    ph = (m.get("config") or {}).get("physics") or {}
    fi = ph.get("forcing_info") or {}
    if fi:
        h = fi.get("hours_local") or ["?", "?"]
        return (f"{fi.get('date')} {h[0]}:00–{h[1]}:00 local ({fi.get('tz')}): SW↓ {_f(ph.get('sw_down'), 0)} W m⁻², "
                f"net LW {_f(ph.get('lw_net'), 0)} W m⁻², wind {ph.get('wind')} m s⁻¹ ({fi.get('wind_source')})"), fi
    return (f"generic daytime forcing (SW↓ {_f(ph.get('sw_down'), 0)} W m⁻², net LW {_f(ph.get('lw_net'), 0)} W m⁻²)",
            {})


def methods_markdown(m: dict) -> str:
    cfg = m.get("config") or {}
    qa = m.get("qa") or {}
    d = cfg.get("data") or {}
    u = d.get("target_units", "")
    inf = m.get("influence") or {}
    cv = m.get("cv") or {}
    met = (m.get("metrics") or {}).get("stacker") or {}
    forcing, _ = _forcing(m)
    co = qa.get("coarse")
    L = [f"# Methods — {m.get('name')}", "",
         "*Draft generated from the run manifest; numbers are this run's.*", "", "## Data", "",
         f"The response is `{d.get('target')}` ({u}) on {m.get('n_points', 0):,} cells of {_f(qa.get('cell_m'), 0)} m"
         + (f" (input cells of {_f(co['fine_cell_m'], 0)} m averaged onto {co['cell_m']:g} m)" if co else "")
         + f". Temperatures are analysed as differences from the "
           f"{str(qa.get('background_source', 'field_median')).replace('_', ' ')} "
           f"({_f(qa.get('background'))} {u}). Predictors: {', '.join(f'`{p}`' for p in cfg.get('predictors', []))}. "
           f"Physics forcing: {forcing}."]
    flags = qa.get("flags") or []
    if flags:
        L += ["", "Data checks raised: " + "; ".join(f["message"] for f in flags if f["severity"] == "warn") + "."]
    if inf:
        rng = inf.get("ranges_m") or {}
        L += ["", "## Area of influence", "",
              "Anisotropic empirical correlograms of the target's residual (after the predictors) and of each "
              "predictor give the distances over which they vary together. The target residual range "
              f"({_f(inf.get('target_resid_range_m'), 0)} m) sets the cross-validation block size and the physics "
              f"length prior (L₀ = {_f(inf.get('L_prior_m'), 0)} m). Predictor influence ranges: "
              + ", ".join(f"{k} {_f(v, 0)} m" for k, v in rng.items())
              + ". Neighbourhood (focal) means of each predictor at ½, 1 and 2 × its range enter the models."]
    models = [k for k, v in (cfg.get("models") or {}).items() if v]
    if models:
        sp = cfg.get("models", {}).get("spatial_plus") or []
        L += ["", "## Models", "",
              "Base models: " + "; ".join(MODEL_TEXT.get(k, k) + (" with Spatial+ (Dupont et al. 2022)" if k in sp else "")
                                          for k in models) + ". "
              "Their out-of-fold predictions are stacked: candidates are the equal-weight mean, a non-negative "
              "convex blend, and the blend plus a gated neural residual with a physics (PDE) penalty "
              f"(λ ∈ {cfg.get('stacker', {}).get('tune_lambda')}); the candidate with the lowest outer "
              f"out-of-fold RMSE is kept ({m.get('stacker_choice')})."]
        adv = m.get("physics_advection")
        if adv:
            L += ["", f"Advection by the campaign wind was {'kept' if adv['kept'] else 'dropped'} after an out-of-fold "
                      f"test (ΔRMSE with − without = {_f(adv['delta_rmse_mean'], 3)} ± {_f(adv['delta_rmse_se'], 3)} {u})."]
    if cv:
        L += ["", "## Validation", "",
              f"Spatial-block cross-validation (Roberts et al. 2017): {cv.get('n_blocks')} square blocks of "
              f"{_f(cv.get('block_m'), 0)} m in {cv.get('n_folds')} folds, with training points within "
              f"{_f(cv.get('buffer_m'), 0)} m of a test block removed. Every model, the stacker choice and the "
              "prediction intervals use these folds; base-model predictions are cross-fitted so the stacker never "
              "sees in-fold predictions. Prediction intervals are cross-conformal (nominal "
              f"{_f(100 * (met.get('interval_target') or 0.9), 0)}%), globally and scaled by distance to the "
              "nearest training data.",
              "", f"Held-out skill: RMSE {_f(met.get('rmse'))} {u}, R² {_f(met.get('r2'))}, interval coverage "
                  f"{_f(met.get('interval_coverage'))}."]
        rows = (m.get("cv_distance") or {}).get("rows") or []
        if rows:
            L += ["Skill against distance from training data: " + "; ".join(
                f"{r['label']} R² {_f(r['stacker']['r2'])}" for r in rows)
                  + ". Random-point CV leaves every test cell next to training cells and mostly measures "
                    "interpolation."]
        b = m.get("baselines")
        if b:
            L += ["", "Standard baselines were fitted on the same folds and compared with the stack by a block-"
                      "clustered paired difference in squared error: " + "; ".join(
                f"{r['label']} RMSE {_f(r['rmse'])} (ΔMSE {r['delta_mse']:+.3f} ± {_f(r['delta_mse_se'], 3)})"
                for r in b["rows"].values()) + f". Verdict: {b['verdict']}."]
    if m.get("response") or m.get("scenarios"):
        L += ["", "## Effects", "",
              "Decision quantities (scenario changes, own-cell and footprint sensitivities, saturation curves) "
              "use the mean over the fold models, so maps have no fold seams; their uncertainty is the "
              "delete-a-group jackknife over folds (SE = sd·√(K−1)). Scenario inputs are clipped to the observed "
              "support of each model input except in the physics term; the share of cells needing extrapolation "
              "is reported. Saturation is fitted per cell to neighbourhood-adoption dose sweeps (linear, "
              "saturating or sigmoid, chosen by AIC)."]
    if m.get("causal"):
        L += ["", "## Causal audit", "",
              "Each treatment's effect is re-estimated without the predictive models by double/debiased machine "
              "learning (Chernozhukov et al. 2018) with spatial-block cross-fitting, a spatial basis for "
              "unmeasured smooth confounding, own-cell and neighbourhood (spillover) exposures, a doubly robust "
              "dose-response curve, and sensitivity analysis (E-values, VanderWeele & Ding 2017; robustness "
              "values, Cinelli & Hazlett 2020). The model's effects are compared with these estimates."]
    if m.get("climate"):
        c = m["climate"]
        L += ["", "## Climate futures", "",
              f"Summer ({_months(c.get('months'))}) mean daily maximum temperature change from {c.get('n_models')} CMIP6 models "
              f"(Eyring et al. 2016) between {c.get('baseline')} and future periods at the site, land-weighted, "
              "added to the observed field (delta method), with and without the adaptation scenarios."]
    p = m.get("provenance") or {}
    L += ["", "## Reproducibility", "",
          f"Code commit `{((p.get('git') or {}).get('commit') or m.get('git_commit') or '?')[:12]}`, core code SHA-256 "
          f"`{(p.get('code_sha256') or '?')[:12]}`, input SHA-256 `{(p.get('input_sha256') or '?')[:12]}`, config "
          f"SHA-256 `{(p.get('config_sha256') or '?')[:12]}`; package versions in `environment.txt`. "
          "`sparc core reproduce <run dir>` re-runs and compares.", ""]
    return "\n".join(L)


def model_card_markdown(m: dict) -> str:
    cfg = m.get("config") or {}
    qa = m.get("qa") or {}
    u = (cfg.get("data") or {}).get("target_units", "")
    met = (m.get("metrics") or {}).get("stacker") or {}
    forcing, fi = _forcing(m)
    scen = m.get("scenarios") or []
    ses = [s.get("mean_delta_se") for s in scen if s.get("mean_delta_se") is not None]
    floor = 2.0 * float(np.median(ses)) if ses else None
    L = [f"# Model card — {m.get('name')}", "",
         f"*Generated {m.get('created_utc')} from the run manifest.*", "",
         "## Intended use", "",
         "- Screening and ranking where neighbourhood-scale cooling (tree canopy, less impervious cover, "
         "reflective surfaces) would lower afternoon air temperature in this study area, and by roughly how much.",
         f"- Conditions like the campaign day: {forcing}.",
         "- Comparing scenarios against each other; locating where effects saturate; framing climate futures.",
         "", "## Not intended for", "",
         "- Single-cell or site-design decisions: effects are learned from neighbourhood-scale variation; read "
         "cell values as part of a neighbourhood average.",
         "- Night-time temperatures, other weather, or other cities without refitting.",
         "- Health or mortality estimates, or regulatory compliance.",
         "", "## Performance (held-out spatial blocks)", "",
         f"- RMSE {_f(met.get('rmse'))} {u}, R² {_f(met.get('r2'))}, 90% interval coverage "
         f"{_f(met.get('interval_coverage'))}."]
    rows = (m.get("cv_distance") or {}).get("rows") or []
    if rows:
        L += ["", "| how far from training data | R² | RMSE |", "|---|---|---|"]
        L += [f"| {r['label']} | {_f(r['stacker']['r2'])} | {_f(r['stacker']['rmse'])} {u} |" for r in rows]
    b = m.get("baselines")
    if b:
        L += ["", f"- Against standard baselines on the same folds: {b['verdict']}."]
    L += ["", "## Effect checks", ""]
    flags = ((m.get("causal") or {}).get("_flags")) or []
    if m.get("causal"):
        verdicts = []
        for tr, r in m["causal"].items():
            if tr.startswith("_") or not isinstance(r, dict):
                continue
            a = (r.get("audit") or {})
            verdicts.append(f"{tr}: " + ", ".join(f"{k.replace('_', ' ')} {v.get('verdict')}" for k, v in a.items()
                                                 if isinstance(v, dict) and v.get("verdict")))
        L += ["- Causal audit (model vs DML/spillover/DR estimates): " + "; ".join(verdicts) + "."]
        if flags:
            L += [f"- Causal flags: {len(flags)} (see report)."]
    lit = m.get("literature")
    if lit and lit.get("sparc"):
        L += ["- Published effect sizes: " + "; ".join(
            f"{r['citation']} {r['quantity']} — SPARC {_f(r['sparc_C'])} °C per +0.10 vs "
            + (f"{r['low_C']:g}" if r['low_C'] == r['high_C'] else f"{r['low_C']:g}–{r['high_C']:g}") + " °C"
            for r in lit["rows"] if r.get("sparc_C") is not None) + "."]
    L += ["", "## Noise floor", "",
          f"- Target rounding noise ≈ {_f(qa.get('target_rounding_noise_sd'))} {u} "
          f"({_f(100 * (qa.get('target_fraction_integer_valued') or 0), 0)}% whole-degree values)."]
    if floor is not None:
        L += [f"- City-wide scenario differences smaller than ≈ {_f(floor, 2)} {u} (2 × median jackknife SE) are not "
              "distinguishable from fold-to-fold variation."]
    L += ["", "## Limitations", ""]
    lim = [f["message"] for f in qa.get("flags") or [] if f["severity"] == "warn"]
    lim += [str(x) for x in ((cfg.get("report") or {}).get("limitations") or [])]
    lim += ["Daytime (afternoon) model of one campaign day; it is never applied to night.",
            "Effects are associations adjusted for the measured predictors and smooth spatial confounding; "
            "unmeasured local factors (building shade, irrigation, traffic heat) can bias them.",
            "Scenario cells outside the observed combination of predictors are extrapolations (share reported)."]
    if not fi:
        lim.append("Physics forcing is generic, not the campaign day's.")
    L += [f"- {x}" for x in lim]
    p = m.get("provenance") or {}
    L += ["", "## Reproducibility", "",
          f"- commit `{((p.get('git') or {}).get('commit') or m.get('git_commit') or '?')[:12]}`, code "
          f"`{(p.get('code_sha256') or '?')[:12]}`, input `{(p.get('input_sha256') or '?')[:12]}`, config "
          f"`{(p.get('config_sha256') or '?')[:12]}`; `sparc core reproduce` re-runs and compares.", ""]
    return "\n".join(L)
