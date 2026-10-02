"""The project report (``export.report`` and ``POST /projects/{pid}/report/preview``, SPEC §6.7).

A report is built from one run, the on-disk result and plan formats of api.md §12.2
(``results/<id>/summary.json`` + ``cells.parquet``, ``plans/<id>/{params,planned,realised}.json`` +
``dose.npy``) and the project's findings, as a list of **blocks** (headings, paragraphs, tables, maps,
charts) that render to one self-contained HTML file (inline SVG charts and PNG maps; a print stylesheet
makes "Print → PDF" a clean document) or to Markdown (maps as data-URI images).  Every sentence comes from
:mod:`.narrative`, written from the numbers.

Sections: ``summary, accuracy, validation, scenarios, plans, climate, equity, caveats, limitations,
provenance, findings`` (api.md §10), in that order.
"""

from __future__ import annotations

import base64
import html
import math
import struct
import zlib
from pathlib import Path

import numpy as np

from sparc.studio import db as dbmod
from sparc.studio.exports import narrative as N
from sparc.studio.workspace import read_json, utc_now

__all__ = ["SECTIONS", "build_report", "render_html", "render_markdown", "png_bytes", "map_png", "bar_chart_svg",
           "md_to_html", "report_blocks"]

SECTIONS = ("summary", "accuracy", "validation", "scenarios", "plans", "climate", "equity", "caveats", "limitations",
            "provenance", "findings")
_TITLES = {"summary": "Summary", "accuracy": "How accurate is the model?", "validation": "Validation",
           "scenarios": "Scenarios", "plans": "Budget plans", "climate": "Climate futures", "equity": "Equity",
           "caveats": "Read these before using the numbers", "limitations": "Limitations",
           "provenance": "Provenance", "findings": "Findings"}
SEQ = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
DIV = ("#0d366b", "#256abf", "#6da7ec", "#f0efec", "#f19a8f", "#d6403f", "#8a1f22")
MAX_MAPS = 6


# ---------------------------------------------------------------------------
# images
# ---------------------------------------------------------------------------

def png_bytes(rgba: np.ndarray) -> bytes:
    """An 8-bit RGBA PNG of an ``(h, w, 4)`` uint8 array (no imaging library)."""
    rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
    h, w, _ = rgba.shape
    raw = b"".join(b"\x00" + rgba[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def _ramp(stops, t: np.ndarray) -> np.ndarray:
    cols = np.array([[int(s[i:i + 2], 16) for i in (1, 3, 5)] for s in stops], dtype=np.float64)
    t = np.clip(t, 0.0, 1.0) * (len(stops) - 1)
    k = np.minimum(np.floor(t).astype(int), len(stops) - 2)
    f = (t - k)[..., None]
    return cols[k] * (1 - f) + cols[k + 1] * f


def map_png(grid, values, *, diverging: bool = True, center: float = 0.0, max_px: int = 360) -> tuple[bytes, dict]:
    """``(PNG, {lo, mid, hi})`` of per-row ``values`` on the run grid, north up, cells without data clear.

    The colour range is the 2nd–98th percentile (symmetric around ``center`` when ``diverging``)."""
    v = np.asarray(values, dtype=np.float64)
    r = grid.raster(v)[::-1]                    # iy grows north: flip for an image
    ok = np.isfinite(r)
    fin = v[np.isfinite(v)]
    if fin.size == 0:
        lo, hi = 0.0, 1.0
    elif diverging:
        span = max(abs(np.percentile(fin, 2) - center), abs(np.percentile(fin, 98) - center)) or 1e-9
        lo, hi = center - span, center + span
    else:
        lo, hi = float(np.percentile(fin, 2)), float(np.percentile(fin, 98))
        hi = hi if hi > lo else lo + 1e-9
    t = np.where(ok, (r - lo) / (hi - lo), 0.0)
    rgb = _ramp(DIV if diverging else SEQ, t)
    img = np.zeros(r.shape + (4,), dtype=np.uint8)
    img[..., :3] = np.round(rgb).astype(np.uint8)
    img[..., 3] = np.where(ok, 255, 0)
    scale = max(1, math.floor(max_px / max(r.shape)))
    if scale > 1:
        img = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
    return png_bytes(img), {"lo": lo, "mid": center if diverging else None, "hi": hi}


def bar_chart_svg(rows: list[dict], unit: str, title: str) -> str:
    """Horizontal bars of ``estimate`` with ``lo``–``hi`` whiskers (negative = cooler, to the left)."""
    W, row_h, ml, mr, mt = 720, 26, 250, 90, 10
    H = mt + row_h * len(rows) + 30
    vals = [r["estimate"] for r in rows] + [r["lo"] for r in rows if r.get("lo") is not None] + \
        [r["hi"] for r in rows if r.get("hi") is not None]
    lo, hi = min(0.0, *vals), max(0.0, *vals)
    if hi - lo < 1e-9:
        hi = lo + 1.0
    iw = W - ml - mr

    def x(v):
        return ml + iw * (v - lo) / (hi - lo)

    e = html.escape
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-label="{e(title)}" '
             f'font-family="system-ui, sans-serif" font-size="12">',
             f'<line x1="{x(0):.1f}" x2="{x(0):.1f}" y1="{mt}" y2="{H - 24}" stroke="#7a7974"/>']
    for i, r in enumerate(rows):
        y = mt + i * row_h
        x0, x1 = x(0), x(r["estimate"])
        parts.append(f'<text x="{ml - 8}" y="{y + 16}" text-anchor="end" fill="#2c2c2a">{e(str(r["name"]))}</text>')
        fill = "#ffffff" if (r.get("extrapolated") or 0) > 0.2 else "#2a78d6"
        parts.append(f'<rect x="{min(x0, x1):.1f}" y="{y + 6}" width="{abs(x1 - x0):.1f}" height="13" fill="{fill}" '
                     f'stroke="#2a78d6" stroke-width="1.2"/>')
        if r.get("lo") is not None and r.get("hi") is not None:
            parts.append(f'<line x1="{x(r["lo"]):.1f}" x2="{x(r["hi"]):.1f}" y1="{y + 12.5}" y2="{y + 12.5}" '
                         'stroke="#0b0b0b" stroke-width="1.2"/>')
        parts.append(f'<text x="{W - mr + 6}" y="{y + 16}" fill="#0b0b0b">{e(N.sfmt(r["estimate"]))} {e(unit)}</text>')
    for v in np.linspace(lo, hi, 5):
        parts.append(f'<text x="{x(v):.1f}" y="{H - 8}" text-anchor="middle" fill="#7a7974">{e(N.sfmt(v))}</text>')
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# blocks of each section
# ---------------------------------------------------------------------------

def _p(text: str) -> dict:
    return {"t": "p", "text": text}


def _table(head, rows, caption=None) -> dict:
    return {"t": "table", "head": list(head), "rows": [list(r) for r in rows], "caption": caption}


def _img(png: bytes, alt: str, caption: str) -> dict:
    return {"t": "img", "mime": "image/png", "data": png, "alt": alt, "caption": caption}


def _legend(rng: dict, unit: str) -> str:
    if rng.get("mid") is not None:
        f = N.sfmt if rng["mid"] == 0 else N.fmt            # changes carry a sign, levels do not
        return f"blue = cooler, red = warmer; colours span {f(rng['lo'])} to {f(rng['hi'])} {unit}"
    return f"light to dark: {N.fmt(rng['lo'])} to {N.fmt(rng['hi'])} {unit}"


class _Env:
    def __init__(self, ctx, db, *, result_ids, plan_ids, finding_ids, project, maps):
        self.ctx, self.db, self.project, self.maps = ctx, db, project or {}, maps
        self.result_ids, self.plan_ids, self.finding_ids = list(result_ids or []), list(plan_ids or []), finding_ids
        self.unit = ctx.units.get("target") if hasattr(ctx, "units") else "°F"
        self.n_maps = 0

    def can_map(self) -> bool:
        return self.maps and self.ctx.grid is not None and self.n_maps < MAX_MAPS

    def map(self, values, *, diverging=True, center=0.0, alt="", caption="", unit=None):
        png, rng = map_png(self.ctx.grid, values, diverging=diverging, center=center)
        self.n_maps += 1
        return _img(png, alt, f"{caption} ({_legend(rng, unit or self.unit)})")


def _headline(env) -> str | None:
    meta = dbmod.loads(env.project.get("meta_json"), {}) or {}
    return meta.get("headline_scenario")


def _summary(env) -> list[dict]:
    ctx = env.ctx
    out = [_p(s) for s in N.summary_sentences(ctx, env.unit, _headline(env))]
    pred = ctx.predictions
    if env.can_map() and pred is not None and "pred" in pred.columns and len(pred) == (ctx.grid.n if ctx.grid else -1):
        bg = float(np.nanmedian(pred["pred"].to_numpy(np.float64)))
        out.append(env.map(pred["pred"].to_numpy(np.float64), center=bg, alt="Predicted air temperature",
                           caption=f"Held-out prediction of afternoon air temperature, centred on the city median "
                                   f"({N.fmt(bg, 1)} {env.unit})"))
    return out


def _accuracy(env) -> list[dict]:
    m = env.ctx.manifest or {}
    met = m.get("metrics") or {}
    out = [_p(s) for s in N.accuracy_sentences(env.ctx, env.unit)]
    rows = [[N.MODEL_LABELS.get(k, k), N.fmt(v.get("r2")), N.fmt(v.get("rmse"))] for k, v in met.items()
            if k != "base_mean" and isinstance(v, dict) and v.get("r2") is not None]
    if rows:
        out.append(_table(["Model", "Held-out R²", f"RMSE ({env.unit})"], rows, "Scored on spatial blocks the "
                                                                                 "models never saw."))
    st = met.get("stacker") or {}
    if st.get("interval_coverage") is not None:
        out.append(_p(f"Prediction intervals: target {N.pct(st.get('interval_target') or 0.9)}, coverage "
                      f"{N.pct(st['interval_coverage'], 1)}, mean half-width ±{N.fmt(st.get('interval_mean_halfwidth'))} "
                      f"{env.unit}."))
    return out


def _studies(env) -> list[dict]:
    rows = env.db.fetchall("SELECT DISTINCT s.* FROM studies s LEFT JOIN study_links l ON l.study_id = s.id "
                           "WHERE s.target_run_id = ? OR (l.run_id = ? AND l.attached = 1) "
                           "ORDER BY s.updated_utc DESC", (env.ctx.run_id, env.ctx.run_id))
    return [{"kind": r["kind"], "id": r["id"], "status": r.get("status"),
             "summary": dbmod.loads(r.get("summary_json")) or {}} for r in rows if r.get("status") in
            ("succeeded", "done")]


def _validation(env) -> list[dict]:
    studies = _studies(env)
    out = [_p(s) for s in N.validation_sentences(env.ctx, studies, env.unit)]
    u = (env.ctx.manifest or {}).get("uncertainty") or env.ctx.json("uncertainty.json") or {}
    rows = [r for r in u.get("scenarios") or [] if isinstance(r, dict)]

    def rng(x):
        return f"{N.sfmt(x[0])} … {N.sfmt(x[1])}" if isinstance(x, (list, tuple)) and len(x) == 2 else "—"

    if rows:
        out.append(_table(["Scenario", f"Estimate ({env.unit})", "Estimation 95%", "Specification", "Attribution",
                           "Envelope"],
                          [[r.get("scenario"), N.sfmt(r.get("estimate")), rng(r.get("estimation_95")),
                            rng(r.get("specification")), rng(r.get("attribution")), rng(r.get("envelope"))]
                           for r in rows], "Uncertainty envelopes (a plausible range, not a confidence interval)."))
    lit = (env.ctx.manifest or {}).get("literature") or {}
    lrows = [r for r in lit.get("rows") or [] if r.get("sparc_C") is not None]
    if lrows:
        out.append(_table(["Lever", "Published", "SPARC (°C per +0.10)", "Within ×2"],
                          [[r.get("quantity"), r.get("value"), N.fmt(r.get("sparc_C")),
                            "yes" if r.get("within_factor_2") else "no"] for r in lrows],
                          "Against published cooling rates (other cities and methods)."))
    return out


def _result(env, rid: str) -> tuple[dict, dict] | None:
    row = env.db.fetchone("SELECT * FROM results WHERE id = ?", (rid,))
    if row is None or not row.get("dir"):
        return None
    summ = read_json(Path(row["dir"]) / "summary.json")
    return (row, summ) if isinstance(summ, dict) else None


def _scenarios(env) -> list[dict]:
    m = env.ctx.manifest or {}
    rows = []
    for s in m.get("scenarios") or []:
        if not isinstance(s, dict):
            continue
        lo, hi = N.likely(s.get("mean_delta"), s.get("mean_delta_se"))
        rows.append({"name": s.get("name"), "estimate": s.get("mean_delta"), "lo": lo, "hi": hi,
                     "extrapolated": s.get("frac_extrapolated"), "kind": "configured"})
    exact = []
    for rid in env.result_ids:
        got = _result(env, rid)
        if got is None:
            continue
        row, summ = got
        city = summ.get("city") or {}
        name = (summ.get("scenario") or {}).get("name") or ((summ.get("summary") or {}).get("name")) or rid
        exact.append({"name": name, "estimate": city.get("estimate"), "lo": city.get("lo"), "hi": city.get("hi"),
                      "extrapolated": summ.get("extrapolated_edited"), "kind": "exact", "id": rid, "row": row,
                      "summ": summ})
    allrows = [r for r in rows + exact if r.get("estimate") is not None]
    out = [_p(s) for s in N.scenario_sentences(allrows, env.unit)]
    if allrows:
        out.append({"t": "svg", "svg": bar_chart_svg(allrows, env.unit, "Scenario effects"),
                    "caption": f"City-mean change in afternoon air temperature ({env.unit}); whiskers: likely range "
                               "(95%); hollow bars: more than 20% of edited cells beyond observed conditions."})
        out.append(_table(["Scenario", "Kind", f"Mean ΔT ({env.unit})", "Likely range", "Beyond observed"],
                          [[r["name"], r["kind"], N.sfmt(r["estimate"]),
                            f"{N.sfmt(r['lo'])} to {N.sfmt(r['hi'])}" if r.get("lo") is not None else "—",
                            N.pct(r.get("extrapolated"))] for r in allrows]))
    for r in exact:
        plain = (r["summ"].get("plain") or {}).get("headline")
        if plain:
            out.append(_p(f"{r['name']}: {plain}"))
        if env.can_map():
            cells = Path(r["row"]["dir"]) / "cells.parquet"
            if cells.exists():
                import pandas as pd

                d = pd.read_parquet(cells, columns=["delta"])["delta"].to_numpy(np.float64)
                if env.ctx.grid is not None and d.size == env.ctx.grid.n:
                    out.append(env.map(d, alt=f"ΔT of {r['name']}",
                                       caption=f"Change in afternoon air temperature: {r['name']}"))
    return out


def _plans(env) -> list[dict]:
    out = []
    for pid in env.plan_ids:
        row = env.db.fetchone("SELECT * FROM plans WHERE id = ?", (pid,))
        if row is None or not row.get("dir"):
            continue
        pdir = Path(row["dir"])
        planned = read_json(pdir / "planned.json") or dbmod.loads(row.get("summary_json"), {}) or {}
        realised = read_json(pdir / "realised.json")
        plan = {"name": row.get("name") or pid, "params": dbmod.loads(row.get("params_json"), {}) or {},
                "planned": planned, "realised": realised if isinstance(realised, dict) else None}
        out.append({"t": "h", "level": 3, "text": plan["name"]})
        out.append(_p(N.plan_sentences(plan, env.unit)))
        if env.can_map() and (pdir / "dose.npy").exists():
            dose = np.load(pdir / "dose.npy").astype(np.float64)
            if env.ctx.grid is not None and dose.size == env.ctx.grid.n:
                out.append(env.map(np.where(dose > 0, dose, np.nan), diverging=False, alt="Planned dose",
                                   caption=f"Where “{plan['name']}” acts (dose per cell; untreated cells blank)",
                                   unit=str(plan["params"].get("lever") or "")))
    if not out:
        o = env.ctx.json("optimize.json") or (env.ctx.manifest or {}).get("optimize") or {}
        if o.get("budget"):
            from sparc.core.results_page import pareto_caption

            out.append(_p(f"The configured budget optimisation spends {N.fmt(o.get('budget'), 0)} dose units of "
                          f"{o.get('variable')} on {int(o.get('n_cells_treated') or 0):,} cells."))
            cap = pareto_caption(o, env.unit)
            if cap:
                out.append(_p(cap))
        else:
            out.append(_p("No budget plan is selected for this report and the run has no budget optimisation."))
    return out


def _climate(env) -> list[dict]:
    m = env.ctx.manifest or {}
    out = [_p(s) for s in N.climate_sentences(m, env.unit)]
    P = [p for p in (m.get("climate") or {}).get("projections") or [] if isinstance(p, dict)]
    if P:
        out.append(_table(["Pathway", "Period", "Models", f"Median warming ({env.unit})", "10–90% of models"],
                          [[p.get("label"), p.get("period"), p.get("n_models"), N.sfmt((p.get("warming") or {}).get("median"), 1),
                            f"{N.sfmt((p.get('warming') or {}).get('p10'), 1)} to {N.sfmt((p.get('warming') or {}).get('p90'), 1)}"]
                           for p in P]))
    return out


def _equity(env) -> list[dict]:
    m = env.ctx.manifest or {}
    opt = env.ctx.json("optimize.json") or m.get("optimize") or {}
    out = [_p(s) for s in N.equity_sentences(m, opt)]
    eq = (m.get("planner") or {}).get("equity") or {}
    rows = []
    for k, e in eq.items():
        q = (e or {}).get("quintiles") or []
        rows.append([k, *[N.fmt(x.get("mean_cooling")) for x in q[:5]], N.sfmt(e.get("concentration_index"), 3)])
    if rows:
        out.append(_table(["Ranking", "Q1", "Q2", "Q3", "Q4", "Q5", "Concentration"], rows,
                          f"Mean cooling ({env.unit}) of the package by quintile (1 = lowest)."))
    return out


def _caveats(env) -> list[dict]:
    return [{"t": "list", "items": N.caveat_items(env.ctx, env.unit)}]


def _limitations(env) -> list[dict]:
    return [{"t": "list", "items": N.limitation_items(env.ctx)}]


def provenance_rows(ctx) -> list[tuple[str, str]]:
    m = ctx.manifest or {}
    prov = m.get("provenance") or {}
    git = prov.get("git") or {}
    ck = getattr(ctx, "checkpoint_json", None) or {}
    from sparc.studio import __version__

    return [("Run", f"{ctx.run_id} ({ctx.row.get('label') or m.get('name') or ''})"),
            ("Created", str(ctx.row.get("created_utc") or m.get("created_utc") or "—")),
            ("Code commit", str(git.get("commit") or m.get("git_commit") or "—")[:12]
             + (" (uncommitted changes)" if git.get("core_dirty") else "")),
            ("Core code SHA-256", str(prov.get("code_sha256") or ck.get("code_sha256") or "—")[:16]),
            ("Input data SHA-256", str(prov.get("input_sha256") or "—")[:16]),
            ("Config SHA-256", str(prov.get("config_sha256") or "—")[:16]),
            ("Checkpoint fingerprint", str(ck.get("fingerprint") or ctx.row.get("fingerprint") or "—")),
            ("Generated", f"{utc_now()} by SPARC Studio {__version__}")]


def _provenance(env) -> list[dict]:
    return [{"t": "kv", "rows": provenance_rows(env.ctx)}]


def _findings(env) -> list[dict]:
    from sparc.studio.exports import findings as F

    if env.finding_ids:
        rows = [r for r in (F.get_row_or_none(env.db, fid) for fid in env.finding_ids) if r is not None]
    else:
        rows = env.db.fetchall("SELECT * FROM findings WHERE run_id = ? ORDER BY position, created_utc",
                               (env.ctx.run_id,))
    if not rows:
        return [_p("No findings are pinned for this run.")]
    out = []
    for r in rows:
        out += F.finding_blocks(env.db, r)
    return out


_BUILDERS = {"summary": _summary, "accuracy": _accuracy, "validation": _validation, "scenarios": _scenarios,
             "plans": _plans, "climate": _climate, "equity": _equity, "caveats": _caveats,
             "limitations": _limitations, "provenance": _provenance, "findings": _findings}


def report_blocks(ctx, db, sections, *, result_ids=(), plan_ids=(), finding_ids=None, project=None,
                  maps: bool = True) -> list[dict]:
    env = _Env(ctx, db, result_ids=result_ids, plan_ids=plan_ids, finding_ids=finding_ids, project=project, maps=maps)
    want = [s for s in SECTIONS if s in set(sections)]
    blocks: list[dict] = []
    for s in want:
        blocks.append({"t": "h", "level": 2, "text": _TITLES[s], "id": s})
        blocks += _BUILDERS[s](env)
    return blocks


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

CSS = """
:root { --ink: #0b0b0b; --ink-2: #52514e; --muted: #7a7974; --line: #e1e0d9; --accent: #2a78d6; --page: #fcfcfb; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink); font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 860px; margin: 0 auto; padding: 32px 20px 64px; }
h1 { font-size: 1.9rem; line-height: 1.15; margin: 0 0 6px; }
h2 { font-size: 1.3rem; margin: 34px 0 10px; padding-top: 6px; border-top: 1px solid var(--line); }
h3 { font-size: 1.05rem; margin: 18px 0 6px; }
p, li { color: var(--ink-2); max-width: 70ch; }
.meta { color: var(--muted); font-size: 0.85rem; }
table { border-collapse: collapse; width: 100%; font-size: 0.86rem; margin: 10px 0; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { color: var(--ink-2); font-weight: 600; }
figure { margin: 14px 0; }
figure img, figure svg { max-width: 100%; height: auto; display: block; image-rendering: pixelated; }
figcaption, caption { color: var(--muted); font-size: 0.8rem; text-align: left; caption-side: bottom; padding-top: 4px; }
dl.kv { display: grid; grid-template-columns: max-content 1fr; gap: 3px 16px; font-size: 0.86rem; }
dl.kv dt { color: var(--muted); } dl.kv dd { margin: 0; font-family: ui-monospace, Menlo, monospace; word-break: break-all; }
.finding { border-left: 3px solid var(--accent); padding: 4px 0 4px 14px; margin: 16px 0; }
pre { white-space: pre-wrap; font-size: 0.8rem; }
@media print {
  @page { margin: 16mm 14mm; }
  body { background: #fff; font-size: 11pt; }
  main { max-width: none; padding: 0; }
  h2 { break-after: avoid; page-break-after: avoid; }
  table, figure, .finding, dl.kv { break-inside: avoid; page-break-inside: avoid; }
  a { color: inherit; text-decoration: none; }
}
"""


def md_to_html(text: str) -> str:
    """A small, safe Markdown subset (paragraphs, lists, **bold**, *italic*, `code`, [links](http…)); all text
    is escaped first."""
    import re

    def inline(s: str) -> str:
        s = html.escape(s, quote=False)
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"(?<![*\w])\*([^*]+)\*(?!\*)", r"<em>\1</em>", s)
        s = re.sub(r"\[([^\]]+)\]\((https?://[^)\s\"]+)\)", r'<a href="\2">\1</a>', s)
        return s

    out, para, items = [], [], []

    def flush():
        if para:
            out.append("<p>" + "<br>".join(inline(x) for x in para) + "</p>")
            para.clear()
        if items:
            out.append("<ul>" + "".join(f"<li>{inline(x)}</li>" for x in items) + "</ul>")
            items.clear()

    for line in (text or "").splitlines():
        s = line.strip()
        if not s:
            flush()
        elif s[:2] in ("- ", "* "):
            if para:
                flush()
            items.append(s[2:])
        else:
            if items:
                flush()
            para.append(s)
    flush()
    return "".join(out)


def _block_html(b: dict) -> str:
    e = html.escape
    t = b["t"]
    if t == "h":
        anchor = f' id="{e(b["id"])}"' if b.get("id") else ""
        return f"<h{b['level']}{anchor}>{e(b['text'])}</h{b['level']}>"
    if t == "p":
        return f"<p>{e(b['text'])}</p>"
    if t == "list":
        return "<ul>" + "".join(f"<li>{e(x)}</li>" for x in b["items"]) + "</ul>"
    if t == "table":
        cap = f"<caption>{e(b['caption'])}</caption>" if b.get("caption") else ""
        head = "".join(f"<th>{e(str(h))}</th>" for h in b["head"])
        rows = "".join("<tr>" + "".join(f"<td>{e('' if c is None else str(c))}</td>" for c in r) + "</tr>"
                       for r in b["rows"])
        return f"<table>{cap}<thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"
    if t == "kv":
        return '<dl class="kv">' + "".join(f"<dt>{e(k)}</dt><dd>{e(str(v))}</dd>" for k, v in b["rows"]) + "</dl>"
    if t == "img":
        src = f"data:{b['mime']};base64,{base64.b64encode(b['data']).decode()}"
        return (f'<figure><img src="{src}" alt="{e(b.get("alt") or "")}">'
                f'<figcaption>{e(b.get("caption") or "")}</figcaption></figure>')
    if t == "svg":
        return f'<figure>{b["svg"]}<figcaption>{e(b.get("caption") or "")}</figcaption></figure>'
    if t == "md":
        return md_to_html(b["text"])
    if t == "pre":
        return f"<pre>{e(b['text'])}</pre>"
    if t == "div":
        return f'<div class="{e(b["cls"])}">' + "".join(_block_html(x) for x in b["blocks"]) + "</div>"
    raise ValueError(f"unknown block {t!r}")


def render_html(title: str, subtitle: str, blocks: list[dict]) -> str:
    e = html.escape
    body = "\n".join(_block_html(b) for b in blocks)
    return ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            f"<title>{e(title)}</title>\n<style>{CSS}</style>\n</head>\n<body>\n<main>\n"
            f"<h1>{e(title)}</h1>\n<p class=\"meta\">{e(subtitle)}</p>\n{body}\n</main>\n</body>\n</html>\n")


def _md_cell(c) -> str:
    return ("" if c is None else str(c)).replace("|", "\\|").replace("\n", " ")


def _block_md(b: dict) -> str:
    t = b["t"]
    if t == "h":
        return "#" * b["level"] + " " + b["text"]
    if t == "p":
        return b["text"]
    if t == "list":
        return "\n".join(f"- {x}" for x in b["items"])
    if t == "table":
        lines = ["| " + " | ".join(_md_cell(h) for h in b["head"]) + " |",
                 "|" + "---|" * len(b["head"])]
        lines += ["| " + " | ".join(_md_cell(c) for c in r) + " |" for r in b["rows"]]
        if b.get("caption"):
            lines += ["", f"*{b['caption']}*"]
        return "\n".join(lines)
    if t == "kv":
        return "\n".join(f"- **{k}:** {v}" for k, v in b["rows"])
    if t == "img":
        src = f"data:{b['mime']};base64,{base64.b64encode(b['data']).decode()}"
        return f"![{b.get('alt') or ''}]({src})\n\n*{b.get('caption') or ''}*"
    if t == "svg":
        src = "data:image/svg+xml;base64," + base64.b64encode(b["svg"].encode("utf-8")).decode()
        return f"![chart]({src})\n\n*{b.get('caption') or ''}*"
    if t == "md":
        return b["text"]
    if t == "pre":
        return "```\n" + b["text"] + "\n```"
    if t == "div":
        return "\n\n".join(_block_md(x) for x in b["blocks"])
    raise ValueError(f"unknown block {t!r}")


def render_markdown(title: str, subtitle: str, blocks: list[dict]) -> str:
    return f"# {title}\n\n*{subtitle}*\n\n" + "\n\n".join(_block_md(b) for b in blocks) + "\n"


def report_title(ctx, project: dict | None) -> tuple[str, str]:
    raw = getattr(ctx, "cfg_raw", None) or {}
    rep = raw.get("report") or {}
    title = str(rep.get("title") or (project or {}).get("name") or "SPARC report")
    sub = f"Run {ctx.run_id}" + (f" · {rep['place']}" if rep.get("place") else "") + f" · generated {utc_now()}"
    return title, sub


def build_report(ctx, db, sections, *, result_ids=(), plan_ids=(), finding_ids=None, fmt: str = "html",
                 project: dict | None = None, maps: bool = True) -> str:
    """The report of ``ctx`` (a ``RunContext``) as one HTML document or Markdown text."""
    blocks = report_blocks(ctx, db, sections, result_ids=result_ids, plan_ids=plan_ids, finding_ids=finding_ids,
                           project=project, maps=maps)
    title, sub = report_title(ctx, project)
    return render_html(title, sub, blocks) if fmt == "html" else render_markdown(title, sub, blocks)
