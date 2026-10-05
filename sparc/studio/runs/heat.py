"""The Heat tab of a run: who is exposed to dangerous heat, what helps, how sure (``sparc.core.heat``).

Air temperature becomes the US National Weather Service heat index with the campaign's dewpoint (the airport
station's, else ERA5's, from the config's ``physics.forcing``), or a dewpoint the user picks in the tab
(``?dewpoint_C=``) to ask "what if the afternoon were more humid?".  Residents (``planner/planner_cells.parquet``
or the config's ``planner.layers``) are counted per NWS category today, with the first joint scenario (the
adaptation package) and in each CMIP6 future, where the range spans the climate models (10th to 90th
percentile warming) and the two humidity assumptions (constant dewpoint to constant relative humidity).  Without
population the counts are cells.

Every scenario also gets a plain verdict (Robust / Direction only / Not established) from its uncertainty row.
"""

from __future__ import annotations

import math

import numpy as np

from sparc.core import heat as heatmod
from sparc.studio.runs.common import fnum

__all__ = ["heat_view", "verdict_map", "campaign_humidity", "heat_layers_available", "today_heat_index",
           "adapted_heat_index", "package_name", "headline_future", "design_heat", "DEWPOINT_PRESETS", "HEAT_KEYS"]

HEAT_KEYS = ("brief", "humidity", "categories", "kpis", "today", "futures", "hist", "verdicts")
#: dewpoints a user can try when the campaign's humidity is unknown (or to ask "what if more humid")
DEWPOINT_PRESETS = [
    {"label": "Dry (10 °C)", "dewpoint_C": 10.0},
    {"label": "Typical summer (16 °C)", "dewpoint_C": 16.0},
    {"label": "Humid (21 °C)", "dewpoint_C": 21.0},
    {"label": "Oppressive (24 °C)", "dewpoint_C": 24.0},
]
_EC = 2          # category index of "Extreme caution" (people at it or worse are the headline count)


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def campaign_humidity(ctx) -> tuple[float | None, str | None]:
    """The campaign dewpoint (°C) and its source, from the config's forcing file (None when absent).

    The forcing path is relative to the config folder, which may have moved (an imported or copied run): the
    run's config, the manifest's and the launch snapshot's are each tried against each recorded config folder
    (then the working directory and its ``configs/``)."""
    def load():
        m = ctx.manifest_raw or {}
        launch = ctx.launch or {}
        raws = [ctx.cfg_raw, m.get("config"), launch.get("config_raw")]
        dirs = [ctx.config_dir, (m.get("provenance") or {}).get("config_dir"), launch.get("config_dir")]
        for raw in raws:
            if not isinstance(raw, dict):
                continue
            raw = raw.get("core", raw)
            if not ((raw.get("physics") or {}).get("forcing")):
                continue
            for d in dict.fromkeys(dirs):
                forcing = heatmod.load_forcing(raw, d)
                if forcing is not None:
                    return heatmod.campaign_dewpoint(forcing)
        return None, None

    return ctx._get("heat_campaign_dewpoint", load)


def _temps(ctx) -> np.ndarray | None:
    pred = ctx.predictions
    if pred is None or "target" not in pred.columns:
        return None
    t = pred["target"].to_numpy(float)
    return t if np.isfinite(t).any() else None


def _people(ctx, n: int) -> np.ndarray | None:
    """Residents per cell in run row order (None without a population layer)."""
    from sparc.studio.runs.layers import _aligned, people_layers

    pc = ctx.parquet("planner/planner_cells.parquet")
    if pc is not None and "people" in pc.columns:
        try:
            p = _aligned(ctx, pc, "people").astype(float)
            if len(p) == n and np.isfinite(p).any():
                return p
        except Exception:                                    # noqa: BLE001 - fall back to the layers table
            pass
    try:
        lay = people_layers(ctx)
    except Exception:                                        # noqa: BLE001
        lay = None
    if lay is not None and "people" in getattr(lay, "columns", []) and len(lay) == n:
        p = lay["people"].to_numpy(float)
        return p if np.isfinite(p).any() else None
    return None


def _package(ctx, n: int) -> tuple[str | None, np.ndarray | None]:
    """The adaptation package (first joint scenario) and its per-cell change, in the target's units."""
    deltas = ctx.parquet("scenario_deltas.parquet")
    if deltas is None or len(deltas) != n:
        return None, None
    joint = [j.get("name") for j in (ctx.cfg_raw.get("joint_scenarios") or []) if isinstance(j, dict)]
    for name in joint:
        if name and name in deltas.columns:
            return str(name), deltas[name].to_numpy(float)
    return None, None


def _futures(ctx) -> list[dict]:
    c = (ctx.manifest or {}).get("climate")
    if not isinstance(c, dict):
        c = ctx.json("climate.json") or {}
    out = []
    for p in c.get("projections") or []:
        w = p.get("warming") or {}
        if fnum(w.get("median")) is None:
            continue
        out.append({"id": f"{p.get('experiment')}:{p.get('period')}", "experiment": p.get("experiment"),
                    "label": p.get("label") or p.get("experiment"), "period": p.get("period"),
                    "median": float(w["median"]), "p10": fnum(w.get("p10")), "p90": fnum(w.get("p90"))})
    return out


def today_heat_index(ctx, dewpoint_c: float | None = None) -> np.ndarray | None:
    """Per-cell heat index (°F) on the campaign afternoon, run row order (None without humidity)."""
    td = dewpoint_c if dewpoint_c is not None else campaign_humidity(ctx)[0]
    t = _temps(ctx)
    if td is None or t is None:
        return None
    return heatmod.heat_index_today(t, ctx.target_units, td)


def package_name(ctx) -> str | None:
    """The adaptation package's name when its per-cell change is in ``scenario_deltas.parquet``."""
    t = _temps(ctx)
    return _package(ctx, len(t))[0] if t is not None else None


def adapted_heat_index(ctx) -> np.ndarray | None:
    """Per-cell heat index (°F) after the package's modelled cooling, campaign dewpoint."""
    td = campaign_humidity(ctx)[0]
    t = _temps(ctx)
    if td is None or t is None:
        return None
    _, adapt = _package(ctx, len(t))
    if adapt is None:
        return None
    return heatmod.heat_index_today(t + adapt, ctx.target_units, td)


def heat_layers_available(ctx) -> bool:
    return campaign_humidity(ctx)[0] is not None and (ctx.run_dir / "predictions.parquet").exists()


# ---------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------

def verdict_map(ctx) -> dict[str, dict]:
    """``{scenario name: verdict}`` from the run's uncertainty envelopes (empty when not computed)."""
    def load():
        u = (ctx.manifest or {}).get("uncertainty")
        if not isinstance(u, dict):
            u = ctx.json("uncertainty.json")
        if not isinstance(u, dict):
            return {}
        return {str(r["scenario"]): heatmod.effect_verdict(r) for r in u.get("scenarios") or []
                if isinstance(r, dict) and r.get("scenario")}

    return ctx._get("heat_verdicts", load)


def _verdict_rows(ctx) -> list[dict] | None:
    from sparc.core.catalog import scenario_slug

    u = (ctx.manifest or {}).get("uncertainty")
    if not isinstance(u, dict):
        u = ctx.json("uncertainty.json")
    if not isinstance(u, dict) or not u.get("scenarios"):
        return None
    vm = verdict_map(ctx)
    taken: set[str] = set()
    rows = []
    for r in u["scenarios"]:
        if not isinstance(r, dict) or not r.get("scenario"):
            continue
        name = str(r["scenario"])
        slug = scenario_slug(name, taken)
        taken.add(slug)
        v = vm.get(name) or {}
        env = r.get("envelope")
        rows.append({"scenario": name, "slug": slug, "verdict": v.get("verdict"), "label": v.get("label"),
                     "reasons": list(v.get("reasons") or []), "qualifiers": list(v.get("qualifiers") or []),
                     "estimate": fnum(r.get("estimate")),
                     "envelope": [fnum(env[0]), fnum(env[1])] if isinstance(env, (list, tuple)) and len(env) == 2
                     else None,
                     "frac_extrapolated": fnum(r.get("frac_extrapolated"))})
    return rows


# ---------------------------------------------------------------------------
# the view
# ---------------------------------------------------------------------------

def _case_row(case: dict, by_people: bool) -> dict:
    counts = case["people"] if by_people else case["cells"]
    return {"counts": {k: float(v) for k, v in counts.items()},
            "ec_or_worse": float(sum(v for i, (k, v) in enumerate(counts.items()) if i >= _EC)),
            "danger_or_worse": float(sum(v for i, (k, v) in enumerate(counts.items()) if i >= _EC + 1)),
            "mean_hi": float(case["person_mean_hi"] if by_people else case["cell_mean_hi"]),
            "max_hi": float(case["max_hi"])}


def _fmt_n(x: float, by_people: bool) -> str:
    x = float(x)
    if not by_people:
        return f"{x:,.0f} cell{'s' if round(x) != 1 else ''}"
    if x >= 10000:
        return f"{x / 1000:,.0f}k residents"
    if x >= 1000:
        return f"{x / 1000:,.1f}k residents"
    return f"{x:,.0f} resident{'s' if round(x) != 1 else ''}"


def _fmt_range(lo: float, hi: float, by_people: bool) -> str:
    if abs(hi - lo) < max(1.0, 0.005 * max(abs(hi), 1.0)):
        return _fmt_n(lo, by_people)
    if lo < 0.5:
        return "up to " + _fmt_n(hi, by_people)
    unit = "residents" if by_people else "cells"
    a, b = _fmt_n(lo, by_people).split(" ")[0], _fmt_n(hi, by_people).split(" ")[0]
    return f"{a}–{b} {unit}"


def heat_view(ctx, params: dict | None = None) -> dict:
    """The Heat tab's sections (``HEAT_KEYS``); ``params``: ``dewpoint_C`` (a user's what-if)."""
    params = params or {}
    camp_td, camp_src = campaign_humidity(ctx)
    user_td = fnum(params.get("dewpoint_C"))
    td = user_td if user_td is not None else camp_td
    src = "your setting" if user_td is not None else camp_src
    categories = [{"id": c[0], "label": c[1], "lo_F": None if math.isinf(c[2]) else c[2],
                   "hi_F": None if math.isinf(c[3]) else c[3], "note": heatmod.CATEGORY_NOTES[c[0]]}
                  for c in heatmod.CATEGORIES]
    humidity = {"dewpoint_C": td, "source": src, "campaign_dewpoint_C": camp_td, "campaign_source": camp_src,
                "user_set": user_td is not None, "needs_input": td is None, "rh_range": None,
                "presets": DEWPOINT_PRESETS, "assumptions": {
                    "constant_dewpoint": "moisture unchanged as it warms, so relative humidity falls (lower bound)",
                    "constant_rh": "relative humidity unchanged, so moisture rises with warming (upper bound)"},
                "method": heatmod.SOURCE}
    out: dict = dict.fromkeys(HEAT_KEYS)
    out.update({"humidity": humidity, "categories": categories, "verdicts": _verdict_rows(ctx)})
    t = _temps(ctx)
    if td is None or t is None:
        return out
    n = len(t)
    units = ctx.target_units
    people = _people(ctx, n)
    by_people = people is not None
    pkg, adapt = _package(ctx, n)
    futs = _futures(ctx)
    fut_map: dict[str, float] = {}
    for f in futs:
        fut_map[f["id"]] = f["median"]
        if f["p10"] is not None:
            fut_map[f["id"] + "@p10"] = f["p10"]
        if f["p90"] is not None:
            fut_map[f["id"] + "@p90"] = f["p90"]
    risk = heatmod.heat_risk(t, units, td, people=people, futures=fut_map, adaptation=adapt)
    humidity["rh_range"] = [round(v, 1) for v in risk["rh_today_range"]]
    cases = risk["cases"]

    def pick(label, adapted, hum):
        return next((c for c in cases if c["case"] == label and c["adapted"] == adapted and c["humidity"] == hum),
                    None)

    today = _case_row(pick("today", False, "observed"), by_people)
    today_pkg = _case_row(pick("today", True, "observed"), by_people) if adapt is not None else None
    total = float(np.nansum(people)) if by_people else float(n)
    out["today"] = {"measure": "people" if by_people else "cells", "total": total, "package": pkg,
                    "unadapted": today, "adapted": today_pkg}

    futures = []
    for f in futs:
        row = {k: f[k] for k in ("id", "experiment", "label", "period")}
        row.update({"warming_F": float(heatmod.delta_to_f(f["median"], units)),
                    "warming_lo_F": float(heatmod.delta_to_f(f["p10"], units)) if f["p10"] is not None else None,
                    "warming_hi_F": float(heatmod.delta_to_f(f["p90"], units)) if f["p90"] is not None else None})
        for key, adapted in (("unadapted", False), ("adapted", True)):
            if adapted and adapt is None:
                row[key] = None
                continue
            cd = pick(f["id"], adapted, "constant_dewpoint")
            crh = pick(f["id"], adapted, "constant_rh")
            lo_c = pick(f["id"] + "@p10", adapted, "constant_dewpoint") or cd
            hi_c = pick(f["id"] + "@p90", adapted, "constant_rh") or crh
            a, b = _case_row(cd, by_people), _case_row(crh, by_people)
            lo, hi = _case_row(lo_c, by_people), _case_row(hi_c, by_people)
            cell = {"constant_dewpoint": a, "constant_rh": b}
            for m in ("ec", "danger"):
                k = f"{m}_or_worse"
                cell[f"{m}_range_humidity"] = [min(a[k], b[k]), max(a[k], b[k])]
                cell[f"{m}_range_full"] = [min(lo[k], hi[k], a[k], b[k]), max(lo[k], hi[k], a[k], b[k])]
            row[key] = cell
        futures.append(row)
    out["futures"] = futures

    # today's heat-index distribution (residents or cells per 1 °F bin), with the package overlaid
    hi_now = heatmod.heat_index_today(t, units, td)
    w = np.nan_to_num(people) if by_people else None
    lo_e, hi_e = math.floor(np.nanmin(hi_now)), math.ceil(np.nanmax(hi_now)) + 1
    if adapt is not None:
        hi_pkg = heatmod.heat_index_today(t + adapt, units, td)
        lo_e = min(lo_e, math.floor(np.nanmin(hi_pkg)))
    step = max(1, int(math.ceil((hi_e - lo_e) / 40)))
    edges = np.arange(lo_e, hi_e + step, step, dtype=float)
    counts, _ = np.histogram(hi_now, bins=edges, weights=w)
    pkg_counts = np.histogram(hi_pkg, bins=edges, weights=w)[0] if adapt is not None else None
    out["hist"] = {"edges": edges.tolist(), "counts": counts.astype(float).tolist(),
                   "adapted_counts": pkg_counts.astype(float).tolist() if pkg_counts is not None else None,
                   "measure": "people" if by_people else "cells"}

    out["kpis"] = _kpis(today, today_pkg, futures, by_people, total, pkg)
    out["brief"] = _brief(ctx, humidity, today, today_pkg, futures, by_people, total, pkg, out["verdicts"])
    return out


def headline_future(futures: list[dict]) -> dict | None:
    """The future the brief leads with: the middle-of-the-road pathway (SSP2-4.5) at mid-century when present,
    else the future with the median warming (the hottest one saturates and the mildest understates)."""
    if not futures:
        return None
    mid = [f for f in futures if str(f.get("experiment") or "").lower().replace("-", "").replace(".", "")
           in ("ssp245", "rcp45")]
    if mid:
        by_period = [f for f in mid if str(f.get("period") or "").startswith(("2041", "2040", "2050"))]
        return (by_period or sorted(mid, key=lambda r: r["warming_F"]))[len(by_period or mid) // 2 if not by_period
                                                                          else 0]
    ranked = sorted(futures, key=lambda r: r["warming_F"])
    return ranked[len(ranked) // 2]


def _pkg_phrase(pkg: str) -> str:
    return pkg if pkg.lower().rstrip().endswith("package") else f"{pkg} package"


def design_heat(ctx, temps, people, delta, futures: dict[str, float]) -> dict | None:
    """Heat risk of a per-cell change ``delta`` (a Lab design), before and after, on the campaign afternoon and in
    each future (``{label: median warming in target units}``); None without campaign humidity.

    ``rows``: one per case and humidity assumption, with the residents (``measure``) at Extreme caution or worse
    and at Danger or worse before and after, and the number moved out of each."""
    td, src = campaign_humidity(ctx)
    if td is None:
        return None
    temps = np.asarray(temps, dtype=float)
    by_people = people is not None
    risk = heatmod.heat_risk(temps, ctx.target_units, td, people=people, futures=futures,
                             adaptation=np.asarray(delta, dtype=float))
    cases = risk["cases"]

    def pick(label, adapted, hum):
        return next(c for c in cases if c["case"] == label and c["adapted"] == adapted and c["humidity"] == hum)

    rows = []
    for label in ["today", *futures]:
        for hum in (("observed",) if label == "today" else heatmod.HUMIDITY_ASSUMPTIONS):
            b, a = _case_row(pick(label, False, hum), by_people), _case_row(pick(label, True, hum), by_people)
            rows.append({"case": label, "humidity": hum,
                         "warming_F": 0.0 if label == "today" else float(heatmod.delta_to_f(futures[label],
                                                                                            ctx.target_units)),
                         "before": b, "after": a, "ec_avoided": b["ec_or_worse"] - a["ec_or_worse"],
                         "danger_avoided": b["danger_or_worse"] - a["danger_or_worse"],
                         "mean_hi_change": a["mean_hi"] - b["mean_hi"]})
    today = rows[0]
    n = today["ec_avoided"]
    who = "residents" if by_people else "cells"
    felt = "typical resident" if by_people else "average cell"
    b_ec, a_ec = today["before"]["ec_or_worse"], today["after"]["ec_or_worse"]
    num = lambda x: _fmt_n(abs(x), by_people).split(" ")[0]  # noqa: E731
    dhi = today["mean_hi_change"]
    if n >= 0.5:
        headline = (f"On the campaign afternoon this design moves {num(n)} {who} out of Extreme caution or worse "
                    f"({num(b_ec)} → {num(a_ec)}) and lowers the heat index the {felt} feels by {-dhi:.1f} °F.")
    elif n <= -0.5:
        headline = (f"On the campaign afternoon this design puts {num(n)} more {who} at Extreme caution or worse "
                    f"({num(b_ec)} → {num(a_ec)}) and raises the heat index the {felt} feels by {dhi:.1f} °F.")
    else:
        headline = (f"On the campaign afternoon this design does not change how many {who} are at Extreme caution "
                    f"or worse ({num(b_ec)}); the heat index the {felt} feels changes by {dhi:+.2f} °F.")
    return {"dewpoint_C": td, "source": src, "measure": "people" if by_people else "cells", "headline": headline,
            "rows": rows, "method": heatmod.SOURCE}


def _kpis(today, today_pkg, futures, by_people, total, pkg) -> list[dict]:
    from sparc.studio.runs.views import _kpi

    who = "Residents" if by_people else "Cells"
    unit = "people" if by_people else "cells"
    k = [_kpi("ec_today", f"{who} at Extreme caution or worse", today["ec_or_worse"], unit=unit, decimals=0, fmt="int",
              note=f"{today['ec_or_worse'] / max(total, 1e-9):.0%} of {'residents' if by_people else 'cells'}",
              tone="warn" if today["ec_or_worse"] > 0 else None),
         _kpi("mean_hi", "Heat index felt by the typical resident" if by_people else "Mean heat index",
              today["mean_hi"], unit="°F", decimals=1),
         _kpi("max_hi", "Hottest cell's heat index", today["max_hi"], unit="°F", decimals=1)]
    if today_pkg is not None:
        k.append(_kpi("ec_pkg", f"With the {_pkg_phrase(pkg)}", today_pkg["ec_or_worse"], unit=unit, decimals=0,
                      fmt="int", note=f"{today_pkg['ec_or_worse'] - today['ec_or_worse']:+,.0f} vs today", tone="good"))
    f = headline_future(futures)
    if f is not None:
        r = f["unadapted"]["ec_range_full"]
        k.append(_kpi("ec_future", f"{f['label']} {f['period']}: at Extreme caution or worse", (r[0] + r[1]) / 2,
                      unit=unit, decimals=0, fmt="int", band={"lo": r[0], "hi": r[1], "label": "range"},
                      note="across climate models and humidity assumptions", tone="crit"))
    return k


def _brief(ctx, humidity, today, today_pkg, futures, by_people, total, pkg, verdicts) -> list[dict]:
    """Plain sentences: how hot, who is exposed, the future, what helps, how sure."""
    tu = "residents" if by_people else "cells"
    out = []
    td = humidity["dewpoint_C"]
    where = humidity["source"] or "assumed"
    out.append({"id": "how_hot", "title": "How hot it felt",
                "text": f"With a dewpoint of {td:.1f} °C ({where}), the heat index on the campaign afternoon "
                        f"reached {today['max_hi']:.0f} °F in the hottest cell; the "
                        f"{'typical resident' if by_people else 'average cell'} felt {today['mean_hi']:.0f} °F.",
                "tone": "neutral"})
    share = today["ec_or_worse"] / max(total, 1e-9)
    dz = today["danger_or_worse"]
    out.append({"id": "who", "title": "Who is exposed",
                "text": f"{_fmt_n(today['ec_or_worse'], by_people)} ({share:.0%} of {tu}) were at Extreme caution "
                        f"or worse (heat index 90 °F or more: heat cramps and exhaustion possible)"
                        + (f"; {_fmt_n(dz, by_people)} at Danger or worse." if dz >= 0.5 else "; none at Danger."),
                "tone": "warn" if today["ec_or_worse"] > 0 else "good"})
    f = headline_future(futures)
    if f is not None:
        u = f["unadapted"]
        ec, dg = u["ec_range_full"], u["danger_range_full"]
        txt = (f"Under {f['label']} by {f['period']} (+{f['warming_F']:.1f} °F, the median of the climate models), "
               f"{_fmt_range(ec[0], ec[1], by_people)} would be at Extreme caution or worse on a comparable "
               f"afternoon")
        txt += (f", and {_fmt_range(dg[0], dg[1], by_people)} at Danger or worse" if dg[1] >= 0.5
                else ", none yet at Danger")
        txt += ". Ranges span the climate models' 10th–90th percentile warming and the two humidity assumptions."
        hot = max(futures, key=lambda r: r["warming_F"])
        if hot is not f:
            d2 = hot["unadapted"]["danger_range_full"]
            txt += (f" In the hottest future ({hot['label']} {hot['period']}, +{hot['warming_F']:.1f} °F), "
                    f"{_fmt_range(d2[0], d2[1], by_people)} would be at Danger or worse.")
        out.append({"id": "future", "title": "As the climate warms", "text": txt, "tone": "crit"})
    if today_pkg is not None:
        cut = today["ec_or_worse"] - today_pkg["ec_or_worse"]
        txt = (f"The {_pkg_phrase(pkg)} would bring today's count at Extreme caution or worse from "
               f"{_fmt_n(today['ec_or_worse'], by_people).split(' ')[0]} to "
               f"{_fmt_n(today_pkg['ec_or_worse'], by_people)} ({_fmt_n(cut, by_people).split(' ')[0]} fewer)")
        if f is not None and f.get("adapted"):
            u, a = f["unadapted"], f["adapted"]
            m = "ec" if u["ec_range_full"][0] < 0.98 * total else "danger"
            lab = "Extreme caution or worse" if m == "ec" else "Danger or worse"
            ur, ar = u[f"{m}_range_full"], a[f"{m}_range_full"]
            txt += (f"; under {f['label']} {f['period']} it would cut those at {lab} from "
                    f"{_fmt_range(ur[0], ur[1], by_people)} to {_fmt_range(ar[0], ar[1], by_people)}")
        v = next((r for r in verdicts or [] if r["scenario"] == pkg), None)
        if v and v.get("label"):
            txt += f". Verdict on its cooling: {v['label']}"
        out.append({"id": "helps", "title": "What helps", "text": txt + ".", "tone": "good"})
    if verdicts:
        counts = {k: sum(1 for r in verdicts if r["verdict"] == k) for k in ("robust", "direction", "not_established")}
        out.append({"id": "sure", "title": "How sure",
                    "text": f"Of {len(verdicts)} adaptation scenarios, {counts['robust']} are Robust (the cooling "
                            f"holds across every uncertainty check), {counts['direction']} Direction only (cooling, "
                            f"but it could be close to none) and {counts['not_established']} Not established (it "
                            f"cannot be told apart from no effect). Heat-index counts use one dewpoint for the whole "
                            f"city; futures carry the humidity band.",
                    "tone": "neutral"})
    else:
        out.append({"id": "sure", "title": "How sure",
                    "text": "Uncertainty envelopes are not computed for this run yet, so scenarios have no verdict. "
                            "Compute them from the Uncertainty tab.", "tone": "neutral"})
    return out
