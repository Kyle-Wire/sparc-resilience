"""Sentences written from a run's numbers (SPEC §6.7 project report, §6.10 findings export).

Nothing here is canned: every sentence states numbers read from the manifest, the stage files, the study
summaries or the selected results, and its wording follows them (a scenario "cools with confidence" only
when its likely range excludes zero; returns "diminish" only when the frontier says so, through
:func:`sparc.core.results_page.pareto_caption`).  Functions return plain strings or lists of them; the
report module lays them out.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

__all__ = ["MODEL_LABELS", "fmt", "sfmt", "pct", "km", "likely", "summary_sentences", "accuracy_sentences", "validation_sentences",
           "scenario_sentences", "plan_sentences", "climate_sentences", "equity_sentences", "caveat_items",
           "limitation_items", "headline_scenario", "confidence_phrase"]


#: display names of the base models (as on the results page)
MODEL_LABELS = {"ols": "Ridge regression", "mgwr": "MGWR", "gwrf": "Geo. random forest", "gam": "GAM + spatial smooth",
                "physics": "Physics (heat transport)", "stacker": "Stacked model"}


def _num(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def fmt(v, d: int = 2) -> str:
    x = _num(v)
    return "—" if x is None else f"{x:,.{d}f}"


def sfmt(v, d: int = 2) -> str:
    x = _num(v)
    if x is None:
        return "—"
    return ("+" if x > 0 else "−" if x < 0 else "") + f"{abs(x):,.{d}f}"


def pct(v, d: int = 0) -> str:
    x = _num(v)
    return "—" if x is None else f"{100 * x:.{d}f}%"


def km(metres) -> str:
    """``2000`` → ``2``, ``390`` → ``0.39`` (kilometres, no trailing zeros)."""
    x = _num(metres)
    return "—" if x is None else f"{x / 1000:.3g}"


def likely(est, se) -> tuple[float | None, float | None]:
    """The 95% likely range ``est ± 1.96·se`` (``(None, None)`` without an SE)."""
    e, s = _num(est), _num(se)
    if e is None or s is None:
        return None, None
    return e - 1.96 * s, e + 1.96 * s


def confidence_phrase(est, lo, hi) -> str:
    """``cools with confidence`` / ``warms with confidence`` / ``could be zero`` from the likely range."""
    if lo is None or hi is None:
        return "has no uncertainty estimate"
    if hi < 0:
        return "cools with confidence"
    if lo > 0:
        return "warms with confidence"
    return "could be zero (its likely range includes no change)"


def headline_scenario(m: dict, preferred: str | None = None) -> dict | None:
    """The configured scenario the summary leads with: the project's headline, else the strongest cooling
    among those mostly within observed conditions, else the strongest cooling.

    ``preferred`` is the project's ``headline_scenario``: a configured-scenario slug (``scenario_slug`` of the
    manifest's names, in order, as ``RunContext.configured_scenarios`` and the Overview read it) or a name."""
    from sparc.core.catalog import scenario_slug

    rows = [s for s in m.get("scenarios") or [] if isinstance(s, dict) and _num(s.get("mean_delta")) is not None]
    if not rows:
        return None
    if preferred:
        slugs: dict[str, str] = {}
        for s in m.get("scenarios") or []:
            if isinstance(s, dict) and s.get("name"):
                slugs.setdefault(str(s["name"]), scenario_slug(str(s["name"]), slugs.values()))
        hit = next((s for s in rows if slugs.get(str(s.get("name"))) == preferred), None) or \
            next((s for s in rows if s.get("name") == preferred), None)
        if hit is not None:
            return hit
    for limit in (0.2, 0.5):                # mostly within observed conditions first
        safe = [s for s in rows if (_num(s.get("frac_extrapolated")) or 0) <= limit]
        if safe:
            return min(safe, key=lambda s: float(s["mean_delta"]))
    return min(rows, key=lambda s: float(s["mean_delta"]))


def summary_sentences(ctx, unit: str, headline: str | None = None) -> list[str]:
    m = ctx.manifest or {}
    st = (m.get("metrics") or {}).get("stacker") or {}
    cv = m.get("cv") or {}
    qa = m.get("qa") or {}
    out = []
    n = m.get("n_points") or ctx.row.get("n_points")
    cell = qa.get("cell_m")
    who = f"run {ctx.row.get('label') or ctx.run_id}"
    first = f"This report covers {who}"
    if n:
        first += f", {int(n):,} cells" + (f" of {fmt(cell, 0)} m" if cell else "")
    out.append(first + ".")
    r2, rmse, cov = _num(st.get("r2")), _num(st.get("rmse")), _num(st.get("interval_coverage"))
    if r2 is not None:
        block = f"{km(cv['block_m'])} km blocks" if cv.get("block_m") else "held-out areas"
        s = (f"On {block} the models never saw, the stacked model explains {pct(r2)} of the temperature variation "
             f"(RMSE {fmt(rmse)} {unit})")
        if cov is not None:
            target = _num(st.get("interval_target")) or 0.9
            s += (f", and its {pct(target)} prediction intervals cover {pct(cov, 1)} of held-out cells"
                  + (" — fewer than they aim for, so read them as somewhat narrow" if cov < target - 0.02 else ""))
        out.append(s + ".")
    hs = headline_scenario(m, headline)
    if hs is not None:
        lo, hi = likely(hs.get("mean_delta"), hs.get("mean_delta_se"))
        s = f"“{hs['name']}” changes the city-mean afternoon air temperature by {sfmt(hs['mean_delta'])} {unit}"
        if lo is not None:
            s += f" (likely {sfmt(lo)} to {sfmt(hi)} {unit}); it {confidence_phrase(hs['mean_delta'], lo, hi)}"
        fx = _num(hs.get("frac_extrapolated"))
        if fx is not None and fx > 0.2:
            s += f", but {pct(fx)} of the cells it edits lie beyond observed conditions"
        out.append(s + ".")
    o = ctx.json("optimize.json") or m.get("optimize") or {}
    if _num(o.get("planned_total_cooling")) is not None and _num(o.get("realized_total_cooling")) is not None:
        ratio = o["realized_total_cooling"] / o["planned_total_cooling"] if o["planned_total_cooling"] else None
        out.append(f"The budget plan treats {int(o.get('n_cells_treated') or 0):,} cells; re-predicted as a whole it "
                   f"cools the city by {fmt(o['realized_total_cooling'], 0)} {unit}·cells"
                   + (f", {pct(ratio)} of what the curves planned" if ratio is not None else "") + ".")
    cl = climate_mid(m)
    if cl is not None:
        w = cl["warming"]
        out.append(f"By {_period(cl)} under {cl.get('label') or cl.get('experiment')}, "
                   f"{cl.get('n_models')} climate models put summer warming at {sfmt(w.get('median'), 1)} {unit} "
                   f"(10–90% of models {sfmt(w.get('p10'), 1)} to {sfmt(w.get('p90'), 1)}).")
    return out


def _projections(m: dict) -> list[dict]:
    """The climate projections that carry a median warming (others - a partial section - are left out)."""
    cl = m.get("climate") or {}
    return [p for p in cl.get("projections") or [] if isinstance(p, dict) and isinstance(p.get("warming"), dict)
            and _num(p["warming"].get("median")) is not None]


def _period(p: dict) -> str:
    return str(p.get("period") or "").replace("-", "–")


def climate_mid(m: dict) -> dict | None:
    """The mid-century SSP2-4.5 projection (else the first one with a warming)."""
    P = _projections(m)
    if not P:
        return None
    return next((p for p in P if p.get("experiment") == "ssp245" and p.get("period") == "2041-2060"), P[0])


def accuracy_sentences(ctx, unit: str) -> list[str]:
    m = ctx.manifest or {}
    met = m.get("metrics") or {}
    out = []
    base = [(k, v) for k, v in met.items() if k not in ("stacker", "base_mean") and isinstance(v, dict)
            and _num(v.get("r2")) is not None]
    st = met.get("stacker") or {}
    if base and _num(st.get("r2")) is not None:
        best = max(base, key=lambda kv: kv[1]["r2"])
        gain = st["r2"] - best[1]["r2"]
        out.append(f"The best single model ({MODEL_LABELS.get(best[0], best[0])}) reaches R² {fmt(best[1]['r2'])}; the stacked model "
                   + (f"adds {fmt(gain)} to that." if gain > 0.005 else
                      "matches it (within 0.005)." if gain >= -0.005 else
                      f"scores {fmt(-gain)} lower: its blend weights are fitted out of fold, so on these blocks they "
                      "trail the best single model."))
    rows = (m.get("cv_distance") or ctx.json("cv_distance.json") or {}).get("rows") or []
    main = next((r for r in rows if isinstance(r, dict) and r.get("main")), None)
    rnd = next((r for r in rows if isinstance(r, dict) and r.get("buffer_m") == 0), None)
    if main is not None:
        out.append(f"Held-out fold R² ranges from {fmt(main.get('fold_r2_min'))} to {fmt(main.get('fold_r2_max'))}"
                   " across areas of the city.")
    if rnd is not None and main is not None and _num((rnd.get("stacker") or {}).get("r2")) is not None:
        out.append(f"Holding out random cells instead gives R² {fmt(rnd['stacker']['r2'])}: the gap to the "
                   "block score is interpolation from neighbouring training cells, not transferable skill.")
    b = m.get("baselines") or ctx.json("baselines.json") or {}
    if b.get("verdict"):
        out.append(f"Against standard baselines on the same held-out blocks: {b['verdict']}.")
    return out


def validation_sentences(ctx, studies: list[dict], unit: str) -> list[str]:
    """One sentence per study kind attached to (or targeting) the run, from the study summaries."""
    out = []
    by = {}
    for s in studies:
        by.setdefault(s["kind"], s)
    pz = (by.get("placebo") or {}).get("summary") or (ctx.manifest or {}).get("placebo") or {}
    if pz.get("n_placebos"):
        n = pz["n_placebos"]
        out.append(f"Placebo layers (real layers moved to the wrong place): the model shows no effect in "
                   f"{pz.get('n_pass_model')} of {n} tests and the causal check in {pz.get('n_pass_causal')} of {n}.")
    sc = (by.get("simcheck") or {}).get("summary") or {}
    bc = sc.get("bias_correction") or ((ctx.manifest or {}).get("simcheck") or {}).get("bias_correction") or {}
    if bc.get("share_range"):
        a, b = bc["share_range"][0], bc["share_range"][-1]
        out.append("When a known canopy effect is planted on this city's layout, the pipeline recovers "
                   + (f"about {fmt(a)} times the truth" if a == b else f"{fmt(a)}–{fmt(b)} times the truth")
                   + (" consistently across generators." if bc.get("stable") else
                      " (one generator so far)." if (bc.get("n_generators") or 2) < 2 else
                      ", depending on how the effect is generated."))
    mv = (by.get("multiverse") or {}).get("summary") or (ctx.manifest or {}).get("multiverse") or {}
    if _num(mv.get("sign_stability_min")) is not None:
        out.append(f"Across reasonable analysis choices (multiverse), every scenario keeps its sign in at least "
                   f"{pct(mv['sign_stability_min'])} of variants"
                   + (f"; priority maps agree with the baseline at median Kendall τ {fmt(mv['median_kendall_tau'])}"
                      if _num(mv.get("median_kendall_tau")) is not None else "") + ".")
    rp = (by.get("reproduce") or {}).get("summary") or {}
    if rp.get("pass") is not None:
        out.append("A re-run from the manifest " + ("reproduced the skill and effects within tolerance."
                                                    if rp["pass"] else
                                                    f"did not reproduce them ({rp.get('n_hard_fail')} hard checks failed)."))
    u = (ctx.manifest or {}).get("uncertainty") or ctx.json("uncertainty.json") or {}
    rows = [r for r in u.get("scenarios") or [] if isinstance(r, dict)]
    if rows:
        excl = sum(1 for r in rows if r.get("envelope_excludes_zero"))
        out.append(f"Combining estimation, specification and attribution uncertainty, {excl} of {len(rows)} "
                   "scenario envelopes exclude zero.")
    if not out:
        out.append("No validation study (placebo, simulation check, multiverse or reproduction) is attached to "
                   "this run yet.")
    return out


def scenario_sentences(rows: list[dict], unit: str) -> list[str]:
    """``rows``: ``{name, estimate, lo, hi, extrapolated}`` (configured and exact)."""
    vals = [r for r in rows if _num(r.get("estimate")) is not None]
    if not vals:
        return ["No scenario results are available for this run."]
    out = []
    cool = sorted(vals, key=lambda r: r["estimate"])
    best = cool[0]
    out.append(f"The strongest cooling is “{best['name']}”: {sfmt(best['estimate'])} {unit} city-wide"
               + (f" (likely {sfmt(best['lo'])} to {sfmt(best['hi'])})" if best.get("lo") is not None else "") + ".")
    conf = [r for r in vals if r.get("hi") is not None and r["hi"] < 0]
    zero = [r for r in vals if r.get("lo") is not None and r["lo"] <= 0 <= r["hi"]]
    if conf or zero:
        out.append(f"{len(conf)} of {len(vals)} scenarios cool with confidence"
                   + (f"; {len(zero)} could be zero" if zero else "") + ".")
    ext = [r["name"] for r in vals if (_num(r.get("extrapolated")) or 0) > 0.2]
    if ext:
        out.append(f"{', '.join(ext)} push{'es' if len(ext) == 1 else ''} more than 20% of edited cells beyond "
                   "observed conditions: read them as extrapolation.")
    return out


def plan_sentences(plan: dict, unit: str) -> str:
    p = plan.get("params") or {}
    pl = plan.get("planned") or {}
    s = (f"“{plan.get('name')}”: a budget of {fmt(p.get('budget'), 0)} on {p.get('lever')} treats "
         f"{int(pl.get('n_cells_treated') or 0):,} cells (mean dose {fmt(pl.get('mean_dose_treated'), 1)}) and plans "
         f"{fmt(pl.get('planned_total'), 0)} {unit}·cells of cooling")
    rz = plan.get("realised") or {}
    if _num(rz.get("total")) is not None and _num(pl.get("planned_total")):
        s += (f"; re-predicted as a whole it delivers {fmt(rz['total'], 0)} {unit}·cells "
              f"({pct(rz['total'] / pl['planned_total'])} of plan)")
    else:
        s += "; it has not been verified with the full model yet"
    g = _num(pl.get("gini"))
    if g is not None:
        s += f". Gini of the doses {fmt(g)}"
    return s + "."


def climate_sentences(m: dict, unit: str) -> list[str]:
    cl = m.get("climate") or {}
    if not cl.get("projections"):
        return ["This run has no climate projections."]
    P = _projections(m)
    if not P:
        return ["This run has no climate projections with warming numbers."]
    out = []
    mid = climate_mid(m)
    hi = max(P, key=lambda p: float(p["warming"]["median"]))
    out.append(f"{cl.get('n_models') or mid.get('n_models')} CMIP6 models: under {mid.get('label')} by "
               f"{_period(mid)} summer highs warm by {sfmt(mid['warming'].get('median'), 1)} {unit}; "
               f"the largest median warming is {sfmt(hi['warming'].get('median'), 1)} {unit} "
               f"({hi.get('label')}, {_period(hi)}).")
    pres = cl.get("present") or {}
    thr = (cl.get("thresholds") or [None])[0]
    share = (pres.get("share_at_or_above") or {}).get(f"{float(thr):.1f}") if thr is not None else None
    if share is not None:
        out.append(f"Today {pct(share, 1)} of cells reach {fmt(thr, 0)} {unit} on a campaign-like afternoon.")
    for v in mid.get("variants") or []:
        off = _num(v.get("offset_share_of_median_warming"))
        if off is not None and v.get("name") != "no adaptation":
            out.append(f"“{v['name']}” offsets {pct(off)} of that mid-century warming on average.")
    return out


def equity_sentences(m: dict, opt: dict) -> list[str]:
    out = []
    eq = (m.get("planner") or {}).get("equity") or {}
    for k, e in eq.items():
        ci = _num((e or {}).get("concentration_index"))
        if ci is None:
            continue
        where = ("concentrates in the higher quintiles" if ci > 0.05 else "concentrates in the lower quintiles"
                 if ci < -0.05 else "is shared about evenly")
        out.append(f"Ranked by {k}, the package's cooling {where} (concentration index {sfmt(ci, 3)}).")
    g = _num((opt or {}).get("gini"))
    if g is not None:
        out.append(f"The budget plan's doses have a Gini coefficient of {fmt(g)} "
                   f"({'concentrated in few cells' if g > 0.6 else 'spread over many cells'}).")
    if not out:
        out.append("No equity layers are available for this run (run the planner pack with population layers).")
    return out


def caveat_items(ctx, unit: str) -> list[str]:
    """The run's generated caveats (``caveats_for``), the budget frontier's caption written from its numbers, and
    flags the numbers raise (interval coverage, studies stale against a refitted run)."""
    from sparc.core.results_page import pareto_caption
    from sparc.studio.runs.caveats import caveats_for

    items = list(caveats_for(ctx))
    m = ctx.manifest or {}
    o = ctx.json("optimize.json") or m.get("optimize") or {}
    cap = pareto_caption(o, unit)
    if cap and "raises total cooling" in cap:
        items.append("Budget frontier: " + cap.split(" as the budget scales. ", 1)[-1])
    st = (m.get("metrics") or {}).get("stacker") or {}
    cov, target = _num(st.get("interval_coverage")), _num(st.get("interval_target")) or 0.9
    if cov is not None and cov < target - 0.02:
        items.append(f"The {pct(target)} prediction intervals cover only {pct(cov, 1)} of held-out cells: they are "
                     "too narrow on average.")
    return items


def limitation_items(ctx) -> list[str]:
    """The model card's "Limitations" section, else limitations stated from the run's numbers."""
    p = Path(ctx.run_dir) / "model_card.md"
    try:
        text = p.read_text("utf-8")
    except OSError:
        text = ""
    m = re.search(r"^## Limitations\s*\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    if m:
        items = [re.sub(r"^\s*[-*]\s+", "", x).strip() for x in m.group(1).splitlines() if x.strip().startswith(("-", "*"))]
        if items:
            return items
    man = ctx.manifest or {}
    cv = man.get("cv") or {}
    qa = man.get("qa") or {}
    out = []
    if cv.get("block_m"):
        out.append(f"Skill is measured on {km(cv['block_m'])} km blocks in {cv.get('n_folds')} folds; "
                   "areas farther from any observation than that are extrapolation.")
    if qa.get("cell_m"):
        out.append(f"Effects are learned at {fmt(qa['cell_m'], 0)} m from neighbourhood-scale variation; single-cell "
                   "designs are outside what the data can resolve.")
    out.append("The model describes afternoons like the campaign day; other weather, night-time temperatures and other "
               "cities need a refit.")
    return out
