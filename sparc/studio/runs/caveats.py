"""Generated caveats of a run - the results page's ``caveats()`` logic (``sparc/core/results_page/template.html``)
in Python, item for item and in the same order (SPEC §6.4 Overview).

Cross-item contract (SPEC §10.2): ``caveats_for(run_ctx) -> list[str]``, used by the run-hub views and by
the engine and studies items (narratives, packs, reports).  ``run_ctx`` is a
:class:`~sparc.studio.runs.reader.RunContext` (or anything with ``manifest``, ``cfg_raw`` and ``run_dir``).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

__all__ = ["caveats_for"]


def _fmt(v, d: int = 2) -> str:
    """The template's ``fmt``: fixed decimals with thousands separators, "—" when not a finite number."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(f):
        return "—"
    return f"{f:,.{d}f}"


def _sfmt(v, d: int = 2) -> str:
    """The template's ``sfmt``: an explicit sign ("+" / "−") before ``fmt`` of the magnitude."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(f):
        return "—"
    return ("+" if f > 0 else "−" if f < 0 else "") + _fmt(abs(f), d)


def _placebo(ctx) -> dict | None:
    """The attached placebo summary: the manifest's section, else ``placebo.json`` in the run folder."""
    m = ctx.manifest or {}
    pz = m.get("placebo")
    if isinstance(pz, dict) and pz.get("n_placebos"):
        return pz
    p = Path(ctx.run_dir) / "placebo.json"
    if p.exists():
        try:
            return json.loads(p.read_text("utf-8"))
        except (OSError, ValueError):
            return None
    return None


def _null_artifact_caveat(m: dict, units: str) -> str | None:
    """Scenarios whose land-cover-filled null simulation (``product == "rf"``) is not distinguishable from the
    real estimate: the canopy effect is not established by this data."""
    rows = [r for r in ((m.get("uncertainty") or {}).get("scenarios") or []) if isinstance(r, dict)
            and any(isinstance(a, dict) and a.get("product") == "rf" and not a.get("distinguishable")
                    for a in r.get("null_artifact") or [])]
    if not rows:
        return None
    r0 = next((r for r in rows if re.search(r"\+10\b", str(r.get("scenario") or ""))), rows[0])
    arts = [a for a in r0.get("null_artifact") or [] if isinstance(a, dict)]
    a = next(x for x in arts if x.get("product") == "rf")
    d = next((x for x in arts if x.get("product") == "direct"), None)
    text = ("Canopy effects are not established by this data: when the simulation plants no canopy effect at all "
            "and imitates how the temperature product was made (a land-cover model filling unsampled streets), the "
            f"pipeline still reports {_sfmt(a.get('delta'), 2)} {units} for \"{r0.get('scenario')}\", as large as the "
            f"real estimate ({_sfmt(r0.get('estimate'), 2)} {units}).")
    if d is not None:
        text += (f" Without that land-cover fill the spurious change is {_sfmt(d.get('delta'), 2)} {units}, so most of "
                 "it is written into the target by the product itself.")
    return text + " Raw traverse readings, not the gridded product, are needed to settle canopy."


def caveats_for(ctx) -> list[str]:
    """Plain-language caveats derived from the run's numbers (plus ``report.caveats`` of its config)."""
    from sparc.core.catalog import unit_label

    m = ctx.manifest or {}
    raw = getattr(ctx, "cfg_raw", None) or m.get("config") or {}
    units = unit_label(((raw.get("data") or {}).get("target_units")) or "degF")
    qa = m.get("qa") or {}
    items: list[str] = []
    fi = qa.get("target_fraction_integer_valued") or 0
    if fi > 0.3:
        items.append(f"The temperature product is rounded: {_fmt(100 * fi, 0)}% of readings are whole degrees, "
                     f"so errors below about 0.3 {units} are noise.")
    rows = (m.get("cv_distance") or {}).get("rows") or []
    main = next((r for r in rows if isinstance(r, dict) and r.get("main")), None)
    if main:
        items.append(f"Accuracy varies by area: held-out fold R² ranges from {_fmt(main.get('fold_r2_min'), 2)} to "
                     f"{_fmt(main.get('fold_r2_max'), 2)}, so some neighbourhoods are predicted much better than "
                     "others.")
    sc = (m.get("simcheck") or {}).get("bias_correction")
    if isinstance(sc, dict) and sc.get("share_range"):
        a, b = sc["share_range"][0], sc["share_range"][-1]
        if (sc.get("n_generators") or 2) < 2 or a == b:
            share = f"about {_fmt(a, 2)} times the truth (one effect generator so far)"
        else:
            share = f"{_fmt(a, 2)}–{_fmt(b, 2)} times the truth depending on how the effect is generated"
        items.append("Effect sizes carry attribution uncertainty: when a known canopy effect is planted on this "
                     "city's layout and passed through the same kind of temperature product, the pipeline recovers "
                     f"{share} (see Methods & validation).")
    else:
        items.append("Effect sizes are not checked against known answers yet: how much of a planted canopy effect "
                     "the pipeline recovers on this city is not computed until a simulation check runs, so "
                     "scenario magnitudes may be too small or too large. Treat them as approximate.")
    null = _null_artifact_caveat(m, units)
    if null:
        items.append(null)
    pz = _placebo(ctx)
    if pz and pz.get("n_placebos") and (pz.get("n_pass_model") or 0) < pz["n_placebos"]:
        n, k = pz["n_placebos"], pz.get("n_pass_model") or 0
        items.append(f"Placebo layers (canopy and paving moved to the wrong place) still received some modelled "
                     f"effect in {n - k} of {n} tests: a smooth layer can pick up chance large-scale patterns. "
                     "Compare real effects with that floor, and lean on the causal check, which passed "
                     f"{pz.get('n_pass_causal') or 0} of {n}.")
    ext = [s.get("name") for s in (m.get("scenarios") or []) if isinstance(s, dict)
           and (s.get("frac_extrapolated") or 0) > 0.2]
    if ext:
        items.append("These scenarios push more than 20% of edited cells beyond observed conditions: "
                     f"{', '.join(str(e) for e in ext)}. Treat them as extrapolation, not predictions.")
    pv = (m.get("physics") or {}).get("v_norm_m")
    forcing = ((m.get("config") or raw).get("physics") or {}).get("forcing_info")
    if isinstance(pv, dict) and isinstance(pv.get("mean"), (int, float)) and pv["mean"] < 1 and not forcing:
        items.append("The physics model runs without advection (no wind record was supplied) and with a generic "
                     "energy balance for the configured time window.")
    if m.get("climate"):
        items.append("Climate futures use the delta method: today's measured pattern plus each model's change in "
                     "summer average daily highs. They assume land-cover effects stay the same as the climate warms, "
                     "and the hottest days may warm more than the seasonal average.")
    for c in (raw.get("report") or {}).get("caveats") or []:
        items.append(str(c))
    return items
