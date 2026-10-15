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


def _sig_ids(rows: list[dict]) -> str:
    """``M1(a), M4(a–c)``-style list of signature ids, grouped by mechanism."""
    by: dict[str, list[str]] = {}
    for r in rows:
        by.setdefault(r["mechanism"], []).append(r["sig"])
    out = []
    for mid, sigs in by.items():
        out.append(f"{mid}({sigs[0]})" if len(sigs) == 1 else f"{mid}({sigs[0]}–{sigs[-1]})"
                   if len(sigs) > 2 else f"{mid}({', '.join(sigs)})")
    return ", ".join(out)


def _by_reason(rows: list[dict]) -> str:
    """Not-testable signatures grouped by their own reason: ``M1(a), M2(a) (one time window …); M7(a) (night …)``."""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r.get("reason") or "reason not recorded", []).append(r)
    return "; ".join(f"{_sig_ids(rs)} ({why})" for why, rs in groups.items())


def _mechanism_facts(m: dict) -> dict | None:
    """What holds, what is contradicted and what is untested in the run's mechanism register."""
    reg = m.get("mechanisms")
    if not isinstance(reg, dict) or not reg.get("rows"):
        return None
    rows = reg["rows"]

    def claim(r):
        st = f": {r['statistic_text']}" if r.get("statistic_text") else ""
        rng = f" against {r['reference']['text']}" if r.get("reference") else ""
        return f"{r['mechanism']}({r['sig']}) {r['name'].lower()}{st}{rng}"

    nyt = [r for r in rows if r["status"] == "not_yet_tested"]
    return {"basis": reg.get("basis"),
            "supported": [claim(r) for r in rows if r["status"] == "supported"],
            "contradicted": [claim(r) for r in rows if r["status"] == "contradicted"],
            "gates": (reg.get("gates") or {}).get("plain") or (reg.get("gates") or {}).get("statements") or [],
            "not_testable": _by_reason([r for r in rows if r["status"] == "not_testable"]),
            "not_yet_tested": _sig_ids(nyt),
            "workstreams": sorted({w.strip() for r in nyt for w in str(r["workstream"]).split(",")}),
            "panel": {r["key"]: r for r in ((reg.get("panel") or {}).get("rows") or [])}}


def _sig(v, unit: str = "") -> str:
    from sparc.core.mechanisms import _sig as sig

    return sig(v, unit)


def _span(p: dict) -> str:
    ci = p.get("ci95") or [None, None]
    u = f" {p['unit']}" if p.get("unit") in ("m", "W/m²") else ""
    return f"{_sig(ci[0]) if ci[0] is not None else 'open'} to {_sig(ci[1]) if ci[1] is not None else 'open'}{u}"


def _identifiability_facts(m: dict) -> dict | None:
    """What the run's identifiability study (PHYSICS_PLAN W3) says, from the mechanism register's
    ``identifiability`` block, which uses the study only when it belongs to the run.  None without a study."""
    reg = m.get("mechanisms") if isinstance(m.get("mechanisms"), dict) else {}
    ib = reg.get("identifiability") if isinstance(reg.get("identifiability"), dict) else None
    if not ib or ib.get("status") == "absent":
        if isinstance(m.get("identifiability"), dict) and m["identifiability"]:
            return {"status": "unregistered"}
        return None
    if ib.get("status") != "used":
        return {"status": ib.get("status"), "note": ib.get("note")}
    from sparc.core.mechanisms import _SYM

    ps = [p for p in (ib.get("parameters") or {}).values() if p.get("in_model")]

    def sym(p):
        return _SYM.get(p["key"], p["key"])

    def item(p):
        return f"{_lc(p['label'])} ({_span(p)})"

    out = {"status": "used", "block": ib,
           "identified": [item(p) for p in ps if p["class"] == "identified" and not p.get("reaches_zero")],
           "zero": [item(p) for p in ps if p["class"] == "identified" and p.get("reaches_zero")],
           "not_identified": [f"{_lc(p['label'])} ({p['class_label']})" for p in ps
                              if p["class"] in ("practically_non_identifiable", "structurally_non_identifiable")],
           "penalty": [f"{sym(p)} = {_sig(p['estimate_run'])} (prior share {float(p['prior_share_run']):.2f})"
                       for p in ps if p.get("prior_share_run") is not None and float(p["prior_share_run"]) > 0.5
                       and p.get("estimate_run") is not None],
           "conflict": [f"{sym(p)} (p = {float(p['conflict']['p']):.2g})" for p in ps
                        if (p.get("conflict") or {}).get("flag") and p["conflict"].get("p") is not None],
           "psens": {d: [sym(p) for p in ps if (p.get("psens") or {}).get("diagnosis") == d]
                     for d in ("prior_data_conflict", "weak_likelihood")},
           "laplace_poor": ib.get("laplace_adequate") is False,
           "gated": [r["id"] for r in reg.get("rows") or [] if r.get("identifiability_gate")],
           "blocked": [f"{mm['id']} ({', '.join(_SYM.get(k, k) for k in mm['identifiability']['not_identified'])})"
                       for mm in reg.get("mechanisms") or []
                       if (mm.get("identifiability") or {}).get("not_identified")],
           "sandwich": [(r["label"], r["sandwich"], r) for r in ((reg.get("panel") or {}).get("rows") or [])
                        if r.get("sandwich")]}
    return out


def _lc(label: str) -> str:
    return label[:1].lower() + label[1:] if label else label


def _identifiability_text(f: dict) -> list[str]:
    """Sentences of the methods section (also the model card's bullets)."""
    ib = f["block"]
    bm = f" of {_sig(ib['block_m'])} m" if ib.get("block_m") else ""
    out = {"method": (f"Which physics parameters this map pins down was tested by profile likelihood (Raue et al. 2009) "
                      f"on all {int(ib['n']):,} labelled cells" if ib.get("n") else
                      "Which physics parameters this map pins down was tested by profile likelihood (Raue et al. 2009)")
           + (f" ({_sig(ib['coarse_m'])} m cells)" if ib.get("coarse_m") else "")
           + f" with every prior off, its curvature adjusted for spatial dependence with the run's {ib.get('n_clusters')} "
             f"cross-validation blocks{bm} as clusters (Chandler & Bate 2007); 95% intervals are where the adjusted "
             "profile statistic stays below 3.84. The priors' influence on the run is the penalty's share of the run "
             "objective's curvature in each parameter (prior share), power-scaling sensitivity of a Laplace "
             "approximation of the calibrated posterior (Kallioinen et al. 2024; Ribatet, Cooley & Davison 2012) "
             "and the Evans & Moshonov (2006) prior–data conflict check."}
    out["pinned"] = ("Identified by the data alone: " + "; ".join(f["identified"]) + "." if f["identified"] else
                     "No physics parameter is identified away from its limits by the data alone.")
    if f["zero"]:
        out["pinned"] += " Bounded by the data but not excluded from zero: " + "; ".join(f["zero"]) + "."
    if f["not_identified"]:
        out["pinned"] += " Not identified by the data alone: " + "; ".join(f["not_identified"]) + "."
    out["penalty"] = ("Set mostly by the penalty (prior share above one half): " + "; ".join(f["penalty"]) + "."
                      if f["penalty"] else "No parameter's value is set mostly by the penalty.")
    cf = []
    if f["conflict"]:
        cf.append("the Evans–Moshonov check flags a prior–data conflict for " + ", ".join(f["conflict"]))
    else:
        cf.append("the Evans–Moshonov check flags no prior–data conflict (it cannot test a parameter whose data-only "
                  "estimate sits on a fit bound)")
    if f["psens"]["prior_data_conflict"]:
        cf.append("power-scaling flags a potential prior–data conflict for " + ", ".join(f["psens"]["prior_data_conflict"]))
    if f["psens"]["weak_likelihood"]:
        cf.append("power-scaling flags a strong prior with a weak likelihood for " + ", ".join(f["psens"]["weak_likelihood"]))
    out["conflict"] = "Conflict: " + "; ".join(cf) + (
        " (power-scaling describes the Laplace approximation, which is poor for this run; the profile intervals do "
        "not rely on it)" if f["laplace_poor"] and (f["psens"]["prior_data_conflict"] or f["psens"]["weak_likelihood"])
        else "") + "."
    gate = ("A mechanism signature cannot be Supported while a key parameter is not identified by the data alone"
            + (f" (key parameters not identified: {', '.join(f['blocked'])})" if f["blocked"] else "")
            + (f"; downgraded to Not testable: {', '.join(f['gated'])}" if f["gated"] else "") + ".")
    out["gate"] = gate
    sw = []
    for label, s, r in f["sandwich"]:
        sw.append(f"{_lc(label)} {_sig(s['value'])} ± {_sig(s['se'])} {r['unit']} (block sandwich, full-data fit) "
                  f"against the fold jackknife {_sig(r.get('value'))} ± {_sig(r.get('se'))} {r['unit']}"
                  + (" — they differ by more than a factor of 2" if s.get("note") else ""))
    out["sandwich"] = ("Uncertainty of the physical quantities: " + "; ".join(sw) + "; the range tests use the "
                       "jackknife." if sw else "")
    return out


def _panel_value(row: dict | None, nd: int = 1) -> str | None:
    if not row or row.get("value") is None:
        return None
    se = f" ± {_f(row['se'], nd)}" if row.get("se") is not None else ""
    ab = row.get("at_bound")              # stopped by a fit bound: not an estimate (no '± 0')
    return (f"{_f(row['value'], nd)}{se} {row['unit']}"
            + (f" (at the {ab['side']} fit bound in {ab['k']} of {ab['n']} folds, so not identified by this run)"
               if ab else ""))


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
        pri = ((m.get("physics") or {}).get("priors") or {}) if isinstance(m.get("physics"), dict) else {}
        pri = pri or ((m.get("config") or {}).get("physics") or {}).get("priors") or {}
        L += ["", "The physics source terms carry weakly informative priors (mean, sd): canopy shading "
                  "s ~ N(0.6, 0.3), impervious storage a₁ ~ N(0.3, 0.3)"
                  + (f" (overridden: {pri})" if pri else "") + "; a tighter shading prior (sd 0.1) held s at its "
                  "prior even for a displaced placebo canopy layer."]
        adv = m.get("physics_advection")
        if adv:
            L += ["", f"Advection by the campaign wind was {'kept' if adv['kept'] else 'dropped'} after an out-of-fold "
                      f"test (ΔRMSE with − without = {_f(adv['delta_rmse_mean'], 3)} ± {_f(adv['delta_rmse_se'], 3)} {u})."]
    mf = _mechanism_facts(m)
    if mf:
        pn = mf["panel"]
        units = [f"{lab} {v}" for lab, v in (("aerodynamic resistance r_a =", _panel_value(pn.get("r_a"))),
                                             ("elevation coefficient γ =", _panel_value(pn.get("gamma"))),
                                             ("mixing length L =", _panel_value(pn.get("L"), 0))) if v]
        L += ["", "## Physics mechanisms", "",
              "Each physics term is written as an if–then claim (IF the mechanism holds, THEN a signature must "
              "appear that fitting this map did not force) and tested where this run allows (mechanism register, "
              "`mechanisms.json`; Oreskes et al. 1994). Fitted parameters are converted to physical units, with the "
              "delete-a-group jackknife SE over folds"
              + (": " + "; ".join(units) if units else "") + f". Basis: {mf['basis']}."
              + (" Supported: " + "; ".join(mf["supported"]) + "." if mf["supported"] else " No mechanism is "
                 "supported yet.")
              + (" Contradicted: " + "; ".join(mf["contradicted"]) + ". " + " ".join(mf["gates"])
                 if mf["contradicted"] else " No mechanism is contradicted.")
              + (f" Not testable with this data: {mf['not_testable']}." if mf["not_testable"] else "")
              + (f" Not yet tested (workstreams {', '.join(mf['workstreams'])} of the physics plan): "
                 f"{mf['not_yet_tested']}." if mf["not_yet_tested"] else "")]
    idf = _identifiability_facts(m)
    if idf:
        L += ["", "## Identifiability", ""]
        if idf["status"] == "used":
            t = _identifiability_text(idf)
            L += [" ".join(x for x in (t["method"], t["pinned"], t["penalty"], t["conflict"], t["gate"], t["sandwich"])
                           if x) + " Details: `identifiability.json`."]
        elif idf["status"] == "unregistered":
            L += ["An identifiability study (`identifiability.json`) exists, but the mechanism register predates it: "
                  "re-run `sparc core mechanisms` to use it."]
        else:
            L += [f"Not used: {idf.get('note')}."]
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
    studies = []
    pz = m.get("placebo")
    if pz and pz.get("n_placebos"):
        studies.append(f"placebo re-fits with displaced canopy/impervious layers and a random layer ({pz['n_pass_model']}/"
                       f"{pz['n_placebos']} pass for the model, {pz['n_pass_causal']}/{pz['n_placebos']} for the causal "
                       "check)")
    sc = m.get("simcheck")
    if sc and sc.get("generators"):
        b = sc.get("bias_correction") or {}
        studies.append("a simulation check planting −0.25 °F per +10 pp canopy on the real layout under "
                       f"{len(sc['generators'])} generators with product emulation"
                       + (f" (effect share {b['share_range'][0]:.2f}–{b['share_range'][1]:.2f})" if b.get("share_range") else ""))
    mv = m.get("multiverse")
    if mv and mv.get("effects"):
        studies.append(f"a multiverse of {len(mv.get('runs', {}))} analysis variants (sign stability ≥ "
                       f"{100 * (mv.get('sign_stability_min') or 0):.0f}%, priority-map Kendall τ "
                       f"{mv.get('median_kendall_tau') or float('nan'):.2f})")
    if studies:
        L += ["", "## Validation studies", "", "Beyond cross-validation: " + "; ".join(studies) + "."]
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
    pz = m.get("placebo")
    if pz and pz.get("n_placebos"):
        L += [f"- Placebo layers: the model passes {pz['n_pass_model']}/{pz['n_placebos']}, the causal check "
              f"{pz['n_pass_causal']}/{pz['n_placebos']} (displaced or random layers should show no effect)."]
    sc = m.get("simcheck")
    if sc and (sc.get("bias_correction") or {}).get("share_range"):
        b = sc["bias_correction"]
        L += [f"- Planted-effect recovery on this layout: {b['share_range'][0]:.2f}–{b['share_range'][1]:.2f} × the truth "
              f"across generators ({'stable' if b.get('stable') else 'not stable: read canopy effects as a range'})."]
    mv = m.get("multiverse")
    if mv and mv.get("effects"):
        L += [f"- Across {len(mv.get('runs', {}))} analysis variants: effect signs stable in ≥ "
              f"{100 * (mv.get('sign_stability_min') or 0):.0f}% of variants; priority maps τ "
              f"{mv.get('median_kendall_tau') or float('nan'):.2f}, top-10% overlap "
              f"{mv.get('median_top_decile_jaccard') or float('nan'):.2f}."]
    lit = m.get("literature")
    if lit and lit.get("sparc"):
        L += ["- Published effect sizes: " + "; ".join(
            f"{r['citation']} {r['quantity']} — SPARC {_f(r['sparc_C'])} °C per +0.10 vs "
            + (f"{r['low_C']:g}" if r['low_C'] == r['high_C'] else f"{r['low_C']:g}–{r['high_C']:g}") + " °C"
            for r in lit["rows"] if r.get("sparc_C") is not None) + "."]
    mf = _mechanism_facts(m)
    if mf:
        L += ["", "## Physics mechanisms", "",
              f"Each physics term is an if–then claim with a test (`mechanisms.json`). Basis: {mf['basis']}.", ""]
        L += [f"- What holds: {'; '.join(mf['supported'])}." if mf["supported"] else
              "- What holds: no mechanism is supported yet."]
        L += [f"- Contradicted: {'; '.join(mf['contradicted'])}. {' '.join(mf['gates'])}".rstrip()
              if mf["contradicted"] else "- Contradicted: none."]
        if mf["not_testable"]:
            L += [f"- Not testable with this data: {mf['not_testable']}."]
        if mf["not_yet_tested"]:
            L += [f"- Not yet tested: {mf['not_yet_tested']} (workstreams {', '.join(mf['workstreams'])})."]
    idf = _identifiability_facts(m)
    if idf:
        L += ["", "## Identifiability", ""]
        if idf["status"] == "used":
            t = _identifiability_text(idf)
            L += ["Profile likelihood on all labelled cells with the priors off, adjusted for spatial dependence with "
                  "the CV blocks as clusters (`identifiability.json`).", "",
                  f"- **What the data pin down.** {t['pinned']}", f"- **What the penalty sets.** {t['penalty']}",
                  f"- **{t['conflict'].split(':', 1)[0]}.**{t['conflict'].split(':', 1)[1]}", f"- {t['gate']}"]
            if t["sandwich"]:
                L += [f"- {t['sandwich']}"]
        elif idf["status"] == "unregistered":
            L += ["- An identifiability study exists, but the mechanism register predates it: re-run "
                  "`sparc core mechanisms`."]
        else:
            L += [f"- Not used: {idf.get('note')}."]
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
