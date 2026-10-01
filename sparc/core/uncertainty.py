"""Uncertainty report: the components of "how sure are we?", kept apart.

For every scenario of a finished run:

* **estimation** — fold-to-fold (delete-a-group jackknife) SE of the city-wide
  change: what re-fitting on different parts of the city does;
* **specification** — the spread of the same scenario across the multiverse
  variants (analysis choices), applied as ratios to the multiverse baseline
  (the multiverse runs on coarser cells, so its absolute changes differ from
  the run's; offsets are used only when the baseline is ~0 or of the other
  sign);
* **attribution** — for canopy scenarios, the simulation check's effect
  share on this city's layout: the planted effect is recovered times
  ``share``, so the true effect lies in estimate / [share range];
* **model vs causal** — whether the independent causal estimate's band
  contains the model's change (a flag, not added to any interval);
* **null artefact** — for canopy scenarios, the mean change the pipeline
  reports when the simulation plants *no* canopy effect (scaled to the
  scenario's dose), and whether the estimate is distinguishable from it
  (a flag, not added to any interval).

Climate futures carry their own spread (CMIP6 models within a pathway); the
pathway itself is a scenario choice, not uncertainty.

The *envelope* is the union of the estimation 95% interval, the
specification range and the attribution range — a plausible range for
reading, not a confidence interval.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

SIM_DOSE = 10.0          # simcheck's canopy edit (pp); sparc.core.simcheck.DOSE


def _canopy_var(m: dict) -> str | None:
    return (((m.get("config") or {}).get("physics") or {}).get("roles") or {}).get("canopy")


def scenario_uncertainty(m: dict, multiverse: dict | None = None, simcheck: dict | None = None) -> dict:
    can = _canopy_var(m)
    mv_eff = (multiverse or {}).get("effects") or {}
    bias = (simcheck or {}).get("bias_correction") or {}
    share_rng = bias.get("share_range")
    rows = []
    for s in m.get("scenarios") or []:
        est = float(s["mean_delta"])
        se = s.get("mean_delta_se")
        r = {"scenario": s["name"], "estimate": est, "se": se,
             "estimation_95": [est - 1.96 * se, est + 1.96 * se] if se is not None else None,
             "frac_extrapolated": s.get("frac_extrapolated")}
        e = mv_eff.get(s["name"])
        if e:
            base = float(e["baseline"])
            vals = [float(v) for v in e["values"].values()]
            if abs(base) > 1e-9 and base * est > 0:
                spec = [est * v / base for v in vals]
                r["specification_mode"] = "ratio"
            else:
                spec = [est + v - base for v in vals]
                r["specification_mode"] = "offset"
            r["specification"] = [float(min(spec)), float(max(spec))]
            r["sign_stability"] = e["sign_stability"]
        realised = set((s.get("mean_realized") or {}).keys())
        if share_rng and can and realised == {can}:
            lo, hi = float(min(share_rng)), float(max(share_rng))
            if lo > 0:
                cands = [est / lo, est / hi]
                r["attribution"] = [min(cands), max(cands)]
                r["attribution_note"] = (f"simcheck effect share {lo:.2f}–{hi:.2f}"
                                         + (f"; stable → corrected {est * bias['correction_factor']:+.3f}"
                                            if bias.get("stable") and bias.get("correction_factor") else ""))
        if can and realised == {can}:
            dose = float((s.get("mean_realized") or {}).get(can) or 0.0)
            for key, g in ((simcheck or {}).get("generators") or {}).items():
                if key.split("/")[0] != "null" or g.get("null_mean_delta") is None:
                    continue
                k = dose / SIM_DOSE
                nd, nse = g["null_mean_delta"] * k, (g.get("null_mean_delta_se") or 0.0) * abs(k)
                z = abs(est - nd) / math.sqrt((se or 0.0) ** 2 + nse ** 2) if (se or nse) else float("inf")
                r.setdefault("null_artifact", []).append(
                    {"product": key.split("/")[1] if "/" in key else "rf", "delta": nd, "se": nse,
                     "n": g.get("n"), "distinguishable": bool(z > 1.96)})
        cl = s.get("causal_linear")
        if cl:
            r["causal_band"] = [cl["lo"], cl["hi"]]
            r["model_within_causal"] = cl.get("model_within")
        parts = [x for x in (r.get("estimation_95"), r.get("specification"), r.get("attribution")) if x]
        if parts:
            r["envelope"] = [float(min(p[0] for p in parts)), float(max(p[1] for p in parts))]
            r["envelope_excludes_zero"] = bool(r["envelope"][1] < 0 or r["envelope"][0] > 0)
        rows.append(r)
    return {"scenarios": rows}


def climate_uncertainty(m: dict) -> list[dict]:
    c = m.get("climate") or {}
    out = []
    for p in c.get("projections") or []:
        w = p.get("warming") or {}
        out.append({"experiment": p.get("experiment"), "label": p.get("label"), "period": p.get("period"),
                    "n_models": p.get("n_models"), "median": w.get("median"),
                    "p10": w.get("p10"), "p90": w.get("p90"), "min": w.get("min"), "max": w.get("max")})
    return out


def uncertainty_report(run_dir, multiverse_dir=None, simcheck_dirs=(), placebo_path=None, real_r2_gate: bool = False) -> dict:
    """Write ``uncertainty.json``/``.md`` and record ``uncertainty`` (plus ``placebo``, ``simcheck`` and
    ``multiverse`` when given) in the manifest through :func:`sparc.core.runio.update_manifest`."""
    from sparc.core import progress, runio

    run_dir = Path(run_dir)
    m = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    sections: dict = {}
    if placebo_path and Path(placebo_path).exists():
        pz = json.loads(Path(placebo_path).read_text(encoding="utf-8"))
        m["placebo"] = sections["placebo"] = {k: pz.get(k) for k in ("rows", "n_pass_model", "n_pass_causal",
                                                                     "n_placebos", "coarse_m",
                                                                     "layer_correlation_with_original")}
        runio.write_json_atomic(run_dir / "placebo.json", pz, indent=1)
        with progress.run_dir_scope(run_dir):
            progress.artifact(run_dir / "placebo.json", role="placebo")
    mv = None
    if multiverse_dir and (Path(multiverse_dir) / "multiverse_summary.json").exists():
        mv = json.loads((Path(multiverse_dir) / "multiverse_summary.json").read_text(encoding="utf-8"))
    sc = None
    if simcheck_dirs:
        from sparc.core.simcheck import merge_results, summarize

        rows = merge_results(simcheck_dirs)
        real_r2 = ((m.get("metrics") or {}).get("stacker") or {}).get("r2") if real_r2_gate else None
        sc = summarize(rows, real_r2=real_r2) if rows else None
    out = {**scenario_uncertainty(m, mv, sc), "climate": climate_uncertainty(m),
           "sources": {"run": str(run_dir), "multiverse": str(multiverse_dir) if mv else None,
                       "simcheck": [str(d) for d in simcheck_dirs] if sc else []},
           "multiverse_stability": {k: (mv or {}).get(k) for k in ("sign_stability_min", "median_kendall_tau",
                                                                   "median_top_decile_jaccard")} if mv else None,
           "simcheck": sc}
    sections["uncertainty"] = {k: v for k, v in out.items() if k != "simcheck"}
    if sc:
        sections["simcheck"] = sc
    if mv:
        sections["multiverse"] = mv
    units = (m.get("config") or {}).get("data", {}).get("target_units", "")
    with progress.run_dir_scope(run_dir):
        runio.write_json_atomic(run_dir / "uncertainty.json", out, indent=1)
        progress.artifact(run_dir / "uncertainty.json", role="uncertainty")
        runio.update_manifest(run_dir, sections, source="uncertainty")
        progress.artifact(run_dir / "manifest.json", role="manifest")
        runio.write_text_atomic(run_dir / "uncertainty.md", uncertainty_markdown(out, units) + "\n")
        progress.artifact(run_dir / "uncertainty.md", role="uncertainty")
    return out


def uncertainty_markdown(u: dict, units: str = "") -> str:
    def rng(x):
        return "—" if not x else f"{x[0]:+.2f} … {x[1]:+.2f}"

    L = [f"| scenario | estimate ({units}) | estimation 95% | specification | attribution | envelope | causal band |",
         "|---|---|---|---|---|---|---|"]
    for r in u["scenarios"]:
        flag = "" if r.get("model_within_causal") in (None, True) else " ⚠"
        L.append(f"| {r['scenario']} | {r['estimate']:+.2f} | {rng(r.get('estimation_95'))} | "
                 f"{rng(r.get('specification'))} | {rng(r.get('attribution'))} | {rng(r.get('envelope'))} | "
                 f"{rng(r.get('causal_band'))}{flag} |")
    L += ["", "Estimation = fold-to-fold (jackknife) 95% interval. Specification = range across analysis choices "
              "(multiverse, as ratios to the multiverse baseline). Attribution = canopy effect corrected by the "
              "simulation check's recovery share range. Envelope = union of these (a plausible range, not a "
              "confidence interval). ⚠ = the model's change is outside the independent causal band."]
    nulls = [(r["scenario"], a) for r in u["scenarios"] for a in r.get("null_artifact") or []]
    if nulls:
        L += ["", f"| scenario | null simulation (target product) | spurious change ({units}) | estimate distinguishable? |",
              "|---|---|---|---|"]
        for name, a in nulls:
            prod = "route sampling + land-cover forest" if a["product"] == "rf" else "direct (no forest)"
            L.append(f"| {name} | {prod}, n={a.get('n')} | {a['delta']:+.2f} ± {a['se']:.2f} | "
                     f"{'yes' if a['distinguishable'] else 'no'} |")
        L += ["", "Null artefact = the mean change the pipeline reports when the simulation plants no canopy effect "
                  "(scaled to the scenario's dose). An estimate that is not distinguishable from it cannot be "
                  "attributed to canopy on this evidence."]
    if u.get("climate"):
        L += ["", f"| pathway | period | models | median warming ({units}) | 10th–90th percentile |", "|---|---|---|---|---|"]
        for c in u["climate"]:
            L.append(f"| {c.get('label') or c.get('experiment')} | {c['period']} | {c['n_models']} | "
                     f"{c['median']:+.1f} | {c['p10']:+.1f} … {c['p90']:+.1f} |" if c.get("median") is not None
                     else f"| {c.get('label')} | {c['period']} | {c['n_models']} | — | — |")
    return "\n".join(L)
