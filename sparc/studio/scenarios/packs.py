"""Decision, plan and compare packs (SPEC §7.13): the ``export.decision_pack``, ``export.plan_pack`` and
``export.compare_pack`` job kinds.

Packs are built from **exact** results.  A pack of a scenario that has no exact result (``result_id`` is the
scenario id) is built from the emulator preview, stamped **DRAFT**, watermarked and carries no standard
errors.  Every pack is a zip under ``projects/<slug>/exports/<export_id>/``:

* ``brief.html`` - self-contained: title and place, a summary paragraph written from the numbers, ΔT and
  realised-dose maps rendered server-side with the shared OKLab LUTs (``matplotlib.image.imsave``), the KPI
  table with likely ranges, regions, equity, exposure, climate offset, the causal check, assumptions and
  caveats, the model card's limitations and provenance hashes;
* ``scenario.json``, ``summary.json``;
* ``cells.csv`` - id, lon, lat, zone, delta, delta_sd, extrapolation and the realised change per lever;
* ``delta.tif`` and ``realized_<var>.tif`` (float32 GeoTIFFs in the run's CRS);
* ``hex_250m.csv``, ``hex_500m.csv`` (+ ``hexagons.gpkg`` when geopandas is installed);
* ``README.txt`` - units and sign conventions.

The plan pack adds ``field_list.csv``, ``logger_sites.csv``, ``before_after_pairs.csv`` (ids and lon/lat) and
``pareto.csv``; the compare pack has per-item summaries, the paired differences, one difference GeoTIFF per
pair and its own brief.
"""

from __future__ import annotations

import base64
import html
import io
import json
import logging
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from sparc.studio.errors import ApiError

log = logging.getLogger("sparc.studio.scenarios")

__all__ = ["decision_pack", "plan_pack", "compare_pack", "write_geotiff", "map_png", "README_TEXT"]

README_TEXT = """SPARC decision pack
===================

Units and signs
---------------
* delta (ΔT): change of the afternoon land-surface/air temperature in {unit}; NEGATIVE = COOLER.
* delta_sd: fold-to-fold standard deviation of the per-cell change ({unit}).
* extrapolation: Mahalanobis extrapolation score of the edited inputs; > 1 = outside the 95th percentile of
  the conditions the model was trained on (treat as extrapolation).
* realized_<lever> / realised_<lever>: the change of the lever actually applied per cell after the physical
  bounds (lever units: {levers}).
* Likely ranges are 95% intervals (estimate ± 1.96 SE), SE = fold jackknife over the K cross-validation fold
  models.  "Confident it cools" means the whole interval is below zero.
* Totals (°·cells) are sums of per-cell changes over cells.

Files
-----
brief.html            the decision brief (open in a browser; print to PDF)
scenario.json         the scenario document and compiled edits
summary.json          every number in the brief
cells.csv             per cell: id, lon, lat, zone, delta, delta_sd, extrapolation, realised change per lever
delta.tif             ΔT GeoTIFF (float32, NaN = no data) in the run's CRS
realized_<lever>.tif  realised change per lever
hex_250m.csv, hex_500m.csv   hexagon summaries (cooling = −ΔT, people summed, temperature mean)
{extra}
{draft}"""

_DRAFT_NOTE = ("DRAFT: this pack was built from the linear emulator preview, not an exact engine run. It carries "
               "no standard errors; run the scenario exactly before using these numbers.")


# ---------------------------------------------------------------------------
# rendering helpers
# ---------------------------------------------------------------------------

def _lut_cmap(name: str):
    from matplotlib.colors import ListedColormap

    from sparc.studio.meta import RAMPS, lut

    return ListedColormap([tuple(c / 255.0 for c in rgb) for rgb in lut(RAMPS[name], 256)])


def map_png(grid, values: np.ndarray, *, diverging: bool = True, magnitude: bool = False) -> bytes:
    """A PNG of per-cell ``values`` on the run grid with the shared OKLab LUT (``divLight`` / ``seqLight``).

    ``magnitude`` maps ``|values|`` from 0 (unchanged cells, the lightest colour) to the 98th percentile of the
    changed cells, so an edit map reads "darker = larger change" whatever its sign.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    from matplotlib.image import imsave

    v = np.asarray(values, dtype=np.float64)
    if magnitude:
        v = np.abs(v)
    ras = grid.raster(v)[::-1]
    ok = np.isfinite(ras)
    if magnitude:
        moved = v[np.isfinite(v) & (v > 0)]
        vmin, vmax = 0.0, (float(np.percentile(moved, 98)) if moved.size else 1.0) or 1.0
        cmap = _lut_cmap("seqLight")
    elif diverging:
        m = float(np.nanpercentile(np.abs(v[np.isfinite(v)]), 98)) if np.isfinite(v).any() else 1.0
        m = m if m > 0 else 1.0
        vmin, vmax, cmap = -m, m, _lut_cmap("divLight")
    else:
        fin = v[np.isfinite(v)]
        vmin = float(np.nanpercentile(fin, 2)) if fin.size else 0.0
        vmax = float(np.nanpercentile(fin, 98)) if fin.size else 1.0
        if vmax <= vmin:
            vmax = vmin + 1.0
        cmap = _lut_cmap("seqLight")
    rgba = cmap(np.clip((np.nan_to_num(ras) - vmin) / (vmax - vmin), 0, 1))
    rgba[..., 3] = np.where(ok, 1.0, 0.0)
    scale = max(1, int(600 / max(grid.nx, grid.ny)))
    if scale > 1:
        rgba = np.repeat(np.repeat(rgba, scale, axis=0), scale, axis=1)
    buf = io.BytesIO()
    imsave(buf, rgba, format="png")
    return buf.getvalue()


def write_geotiff(grid, values: np.ndarray, path: Path) -> None:
    """A float32 GeoTIFF of per-cell values on the run grid, in the run's CRS (north up)."""
    import rasterio
    from rasterio.transform import from_origin

    s = float(grid.coord_scale or 1.0)
    tr = from_origin((grid.x0 - grid.dx / 2.0) / s, (grid.y0 + (grid.ny - 0.5) * grid.dx) / s, grid.dx / s,
                     grid.dx / s)
    ras = grid.raster(np.asarray(values, dtype=np.float64))[::-1].astype("float32")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with rasterio.open(tmp, "w", driver="GTiff", height=grid.ny, width=grid.nx, count=1, dtype="float32",
                           crs=grid.crs or None, transform=tr, nodata=np.nan, compress="deflate") as dst:
            dst.write(ras, 1)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _esc(v: Any) -> str:
    return html.escape(str(v))


def _fmt(v, d: int = 2, signed: bool = False) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(f):
        return "—"
    s = f"{f:+,.{d}f}" if signed else f"{f:,.{d}f}"
    return s.replace("-", "−")


def _likely_text(lk: dict | None, unit: str, draft: bool) -> str:
    if not lk:
        return "—"
    est = _fmt(lk.get("estimate"), 3, True)
    if draft or lk.get("lo") is None:
        return f"{est} {unit}"
    return f"{est} {unit} ({_fmt(lk.get('lo'), 3, True)} to {_fmt(lk.get('hi'), 3, True)})"


def _model_card_limitations(run_dir: Path) -> list[str]:
    p = Path(run_dir) / "model_card.md"
    try:
        text = p.read_text("utf-8")
    except OSError:
        return []
    out, on = [], False
    for line in text.splitlines():
        if line.startswith("#"):
            on = "limitation" in line.lower() or "caveat" in line.lower()
            continue
        if on and line.strip().startswith(("-", "*")):
            out.append(line.strip().lstrip("-* ").strip())
    return out[:12]


def narrative(name: str, res: dict, unit: str, n_cells: int, area_km2: float, draft: bool) -> str:
    """The summary paragraph, written from the numbers."""
    city = res.get("city") or {}
    regions = {r["name"]: r for r in res.get("regions") or []}
    ed = (regions.get("edited") or {}).get("mean")
    parts = [f"“{name}” changes {n_cells:,} cells ({area_km2:,.2f} km²)."]
    focus = ed or city
    if focus:
        est = float(focus.get("estimate") or 0.0)
        word = "cools" if est < 0 else "warms" if est > 0 else "does not change"
        near, far = sorted((abs(float(focus.get("lo") or 0.0)), abs(float(focus.get("hi") or 0.0))))
        rng = "" if draft or focus.get("lo") is None else \
            f" (likely range {_fmt(near, 2)}–{_fmt(far, 2)} {unit})" \
            if focus["lo"] * focus["hi"] > 0 else f" (likely range {_fmt(focus['lo'], 2, True)} to " \
                                                  f"{_fmt(focus['hi'], 2, True)} {unit})"
        parts.append(f"Where it acts it {word} afternoon temperatures by {_fmt(abs(est), 2)} {unit}{rng}.")
    if city:
        c = float(city.get("estimate") or 0.0)
        parts.append(f"City-wide the change is {_fmt(abs(c), 3)} {unit} {'cooler' if c < 0 else 'warmer'}.")
    sp = res.get("spill") or {}
    if sp.get("outside_share") is not None and abs(sp["outside_share"]) > 0.005:
        parts.append(f"{abs(sp['outside_share']):.0%} of the change lands outside the edited cells.")
    fx = res.get("extrapolated_edited")
    if fx:
        parts.append(f"{fx:.0%} of the edited cells are outside the conditions the model was trained on.")
    cc = res.get("causal_check")
    if cc:
        parts.append("The independent causal estimate " + ("agrees with" if cc.get("model_within") else
                                                            "disagrees with") + " the model's change.")
    cost = res.get("cost") or {}
    if cost.get("total"):
        parts.append(f"Its cost is {_fmt(cost['total'], 0)} units"
                     + (f", or {_fmt(1000 * cost['cooling_per_cost'], 2)} {unit}·cells of cooling per 1,000 units."
                        if cost.get("cooling_per_cost") is not None else "."))
    if draft:
        parts.append("These are preview numbers from the linear emulator, without uncertainty.")
    return " ".join(parts)


def _brief_html(*, title: str, place: str, paragraph: str, res: dict, impacts: dict | None, maps: dict[str, bytes],
                caveats: list[str], limitations: list[str], provenance: dict, unit: str, draft: bool,
                extra_sections: str = "") -> str:
    def img(name: str, cap: str) -> str:
        if name not in maps:
            return ""
        b64 = base64.b64encode(maps[name]).decode("ascii")
        return f'<figure><img alt="{_esc(cap)}" src="data:image/png;base64,{b64}"><figcaption>{_esc(cap)}</figcaption></figure>'

    rows = []
    for r in res.get("regions") or []:
        m = r.get("mean")
        rows.append(f"<tr><td>{_esc(r['name'])}</td><td>{r['n_cells']:,}</td><td>{_esc(_likely_text(m, unit, draft))}"
                    f"</td><td>{_fmt(r.get('total'), 1, True)}</td><td>{_fmt(100 * r.get('frac_cooled_01', 0), 0)}%"
                    f"</td></tr>")
    real = "".join(f"<tr><td>{_esc(v)}</td><td>{_fmt(d['requested_mean'], 3, True)}</td>"
                   f"<td>{_fmt(d['realized_mean'], 3, True)}</td><td>{_fmt(100 * d['clipped_share'], 0)}%</td></tr>"
                   for v, d in (res.get("realized") or {}).items())
    plain = res.get("plain") or {}
    eq_html = ""
    exp_html = ""
    off_html = ""
    if impacts:
        eq_rows = "".join(f"<tr><td>{_esc(k)}</td><td>{_fmt(v.get('concentration_index'), 3, True)}</td></tr>"
                          for k, v in (impacts.get("equity") or {}).items())
        if eq_rows:
            eq_html = ("<h2>Who benefits</h2><table><tr><th>ranking</th><th>concentration index (&gt; 0: benefit "
                       f"concentrates at the top)</th></tr>{eq_rows}</table>")
        thr = impacts.get("thresholds") or []
        ex_rows = "".join("<tr><td>{}</td><td>{}</td>{}</tr>".format(
            _esc(e["case"]), "with scenario" if e["adapted"] else "without",
            "".join(f"<td>{_fmt(e['people_ge'].get(f'{t:g}'), 0)}</td>" for t in thr))
            for e in impacts.get("exposure") or [])
        if ex_rows:
            exp_html = ("<h2>Residents at or above each threshold</h2><table><tr><th>case</th><th></th>"
                        + "".join(f"<th>≥ {t:g} {unit}</th>" for t in thr) + f"</tr>{ex_rows}</table>")
        off_rows = "".join(f"<tr><td>{_esc(o['label'])}</td><td>{_fmt(100 * (o['offset_share'] or 0), 1)}%</td></tr>"
                           for o in impacts.get("climate_offset") or [] if o.get("offset_share") is not None)
        if off_rows:
            off_html = f"<h2>Climate offset</h2><table><tr><th>future</th><th>share of median warming cancelled</th></tr>{off_rows}</table>"
    cc = res.get("causal_check")
    cc_html = ""
    if cc:
        cc_html = (f"<h2>Causal check</h2><p>Independent causal estimate {_fmt(cc['delta'], 3, True)} {unit} "
                   f"(95% band {_fmt(cc['lo'], 3, True)} to {_fmt(cc['hi'], 3, True)}); the model's city mean is "
                   f"{'inside' if cc['model_within'] else 'outside'} the band.</p>")
    prov = "".join(f"<tr><td>{_esc(k)}</td><td><code>{_esc(v)}</code></td></tr>" for k, v in provenance.items() if v)
    cav = "".join(f"<li>{_esc(c)}</li>" for c in caveats)
    lim = "".join(f"<li>{_esc(c)}</li>" for c in limitations)
    quals = "".join(f"<li>{_esc(q)}</li>" for q in plain.get("qualifiers") or [])
    buys = "".join(f"<li>{_esc(b)}</li>" for b in plain.get("buys") or [])
    city = res.get("city") or {}
    watermark = ('<div class="wm" aria-hidden="true">DRAFT</div><p class="draft">' + _esc(_DRAFT_NOTE) + "</p>") \
        if draft else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>
body{{font:15px/1.5 system-ui,-apple-system,sans-serif;max-width:60rem;margin:2rem auto;padding:0 1rem;color:#1b1b1a}}
h1{{font-size:1.6rem;margin:0}} h2{{font-size:1.15rem;margin-top:2rem;border-bottom:1px solid #ddd}}
table{{border-collapse:collapse;width:100%;margin:.5rem 0}} td,th{{border-bottom:1px solid #eee;padding:.25rem .5rem;text-align:left}}
figure{{display:inline-block;margin:.5rem 1rem .5rem 0;max-width:46%}} img{{max-width:100%;image-rendering:pixelated;border:1px solid #ddd}}
.lead{{font-size:1.1rem}} .kpi{{font-size:1.3rem;font-weight:600}} code{{font-size:.8rem}}
.draft{{background:#fff3cd;padding:.5rem;border:1px solid #e0c060}}
.wm{{position:fixed;top:40%;left:0;right:0;text-align:center;font-size:9rem;color:rgba(200,0,0,.12);transform:rotate(-20deg);pointer-events:none}}
@media print{{figure{{max-width:45%}}}}
</style></head><body>
{watermark}
<header><h1>{_esc(title)}</h1><p>{_esc(place)}</p></header>
<p class="lead">{_esc(paragraph)}</p>
<p class="kpi">{_esc(plain.get('headline') or '')}</p>
<p><strong>{_esc(plain.get('confidence') or '')}</strong></p>
{('<ul>' + quals + '</ul>') if quals else ''}
{('<h2>What it buys</h2><ul>' + buys + '</ul>') if buys else ''}
<h2>Maps</h2>
{img('delta', f'ΔT ({unit}; blue = cooler)')}{''.join(img(k, f'realised change of {k[9:]} (size of the change; darker = larger, lightest = unchanged)') for k in maps if k.startswith('realized_'))}
<h2>Key numbers</h2>
<table><tr><th>quantity</th><th>value</th></tr>
<tr><td>City mean ΔT</td><td>{_esc(_likely_text(city, unit, draft))}</td></tr>
<tr><td>10th–90th percentile of cell ΔT</td><td>{_fmt(res.get('p10'), 3, True)} to {_fmt(res.get('p90'), 3, True)} {unit}</td></tr>
<tr><td>Edited cells outside observed conditions</td><td>{_fmt(100 * (res.get('extrapolated_edited') or 0), 0)}%</td></tr>
<tr><td>Share of the change outside the edited cells</td><td>{_fmt(100 * ((res.get('spill') or {}).get('outside_share') or 0), 0)}%</td></tr>
<tr><td>Cost (units)</td><td>{_fmt((res.get('cost') or {}).get('total'), 0)}</td></tr></table>
<h2>Regions</h2><table><tr><th>region</th><th>cells</th><th>mean ΔT (likely range)</th><th>total ({unit}·cells)</th><th>cells cooled ≥ 0.1</th></tr>{''.join(rows)}</table>
<h2>Realised vs requested</h2><table><tr><th>lever</th><th>requested mean</th><th>realised mean</th><th>clipped</th></tr>{real}</table>
{eq_html}{exp_html}{off_html}{cc_html}
{extra_sections}
<h2>Assumptions and caveats</h2><ul>
<li>{'Preview (linear emulator): never saturates, never clips to the observed support; no uncertainty.' if draft else 'Exact: every fold model re-predicts the edited city; likely ranges are fold-jackknife 95% intervals.'}</li>
<li>Extrapolated cells (score &gt; 1) rely on conditions the model has not seen.</li>{cav}</ul>
{('<h2>Model limitations</h2><ul>' + lim + '</ul>') if lim else ''}
<h2>Provenance</h2><table>{prov}</table>
</body></html>
"""


def _write_zip(stage: Path, out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(stage).as_posix())
    os.replace(tmp, out)
    return int(out.stat().st_size)


# ---------------------------------------------------------------------------
# shared builders
# ---------------------------------------------------------------------------

def _run_ctx(db, run_id: str):
    from sparc.studio.runs.reader import RunContext

    row = db.fetchone("SELECT * FROM runs WHERE id = ?", (run_id,))
    if row is None:
        raise ApiError("not_found", f"no run {run_id!r}")
    import time

    return RunContext(row, ("pack", time.time()))


def _cells_frame(ctx, delta, delta_sd, extrapolation, realized: dict):
    import pandas as pd

    g = ctx.grid
    codes = g.zone_codes
    df = pd.DataFrame({"id": np.asarray(g.ids), "lon": g.lon.astype(np.float64), "lat": g.lat.astype(np.float64),
                       "zone": codes if codes is not None else None,
                       "delta": np.asarray(delta, dtype=np.float64)})
    df["delta_sd"] = np.asarray(delta_sd, dtype=np.float64) if delta_sd is not None else np.nan
    df["extrapolation"] = np.asarray(extrapolation, dtype=np.float64) if extrapolation is not None else np.nan
    for v, arr in realized.items():
        df[f"realised_{v}"] = np.asarray(arr, dtype=np.float64)
    return df


def _hex_csvs(ctx, delta, stage: Path) -> list[str]:
    from sparc.core.planner import export_hex_gpkg, summarize_hex
    from sparc.studio.runs import layers as L
    from sparc.studio.runs.grid import lonlat_transformer, transform_xy

    g = ctx.grid
    vals = {"cooling": -np.asarray(delta, dtype=np.float64)}
    lay = L.people_layers(ctx) if ctx.data is not None else None
    if lay is not None and "people" in lay:
        vals["people"] = np.nan_to_num(lay["people"].to_numpy(float))
    if ctx.data is not None:
        vals["temperature"] = np.asarray(ctx.data.target_raw, dtype=np.float64)
    files = []
    hexes = {}
    for size in (250, 500):
        h = summarize_hex(g.x, g.y, vals, float(size))
        if g.crs:
            lon, lat = transform_xy(lonlat_transformer(g.crs), h["cx"].to_numpy() / g.coord_scale,
                                    h["cy"].to_numpy() / g.coord_scale)
            h["lon"], h["lat"] = lon, lat
        h.to_csv(stage / f"hex_{size}m.csv", index=False)
        files.append(f"hex_{size}m.csv")
        hexes[size] = h
    if ctx.cfg is not None and g.crs:
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for size, h in hexes.items():
                    export_hex_gpkg(h.drop(columns=[c for c in ("lon", "lat") if c in h]), size, ctx.cfg,
                                    stage / "hexagons.gpkg")
            files.append("hexagons.gpkg")
        except Exception as exc:              # geopandas is optional
            log.info("hexagons.gpkg skipped: %s", exc)
    return files


def _provenance(ctx, spec: dict) -> dict:
    side = ctx.checkpoint_json or {}
    m = ctx.manifest_raw or {}
    return {"run": ctx.run_id, "run_dir": str(ctx.run_dir), "content_hash": spec.get("content_hash"),
            "checkpoint": spec.get("ckpt_key"), "checkpoint fingerprint": side.get("fingerprint"),
            "core code sha256": spec.get("code_sha"), "config sha256": ((m.get("provenance") or {}).get("config_sha256")),
            "git commit": m.get("git_commit"), "result": spec.get("id")}


def _decision_contents(ctx, stage: Path, *, name: str, scenario: dict | None, res: dict, spec: dict, delta,
                       delta_sd, extrapolation, realized: dict, draft: bool, thresholds=None, cache_dir=None,
                       extra_sections: str = "", extra_readme: str = "") -> None:
    from sparc.studio.runs.caveats import caveats_for
    from sparc.studio.runs.common import clean
    from sparc.studio.scenarios.impacts import compute_impacts

    unit = ctx.units.get("target", "°F")
    g = ctx.grid
    impacts = None
    try:
        impacts = compute_impacts(ctx, delta, thresholds=thresholds, cache_dir=cache_dir)
    except ApiError as exc:
        log.info("pack impacts skipped: %s", exc.message)
    if draft:
        for r in res.get("regions") or []:
            if r.get("mean"):
                r["mean"] = {**r["mean"], "se": None, "lo": None, "hi": None, "confidence": "unknown"}
        if res.get("city"):
            res["city"] = {**res["city"], "se": None, "lo": None, "hi": None, "confidence": "unknown"}
    (stage / "scenario.json").write_text(json.dumps(clean({"scenario": scenario, "spec": spec}), indent=1,
                                                    allow_nan=False), encoding="utf-8")
    (stage / "summary.json").write_text(json.dumps(clean({**res, "impacts": impacts, "draft": draft}), indent=1,
                                                   allow_nan=False), encoding="utf-8")
    _cells_frame(ctx, delta, delta_sd, extrapolation, realized).to_csv(stage / "cells.csv", index=False)
    write_geotiff(g, delta, stage / "delta.tif")
    maps = {"delta": map_png(g, delta)}
    for v, arr in realized.items():
        write_geotiff(g, arr, stage / f"realized_{v}.tif")
        maps[f"realized_{v}"] = map_png(g, arr, magnitude=True)
    hex_files = _hex_csvs(ctx, delta, stage)
    edited = np.zeros(g.n, dtype=bool)
    for arr in realized.values():
        edited |= np.abs(np.asarray(arr)) > 0
    para = narrative(name, res, unit, int(edited.sum()), float(edited.sum() * g.dx * g.dx / 1e6), draft)
    clim = (ctx.manifest.get("climate") or {}).get("site") or {}
    place = f"{ctx.name} · run {ctx.run_id}" + (f" · {clim['lat']:.3f}, {clim['lon']:.3f}" if clim.get("lat") is not None
                                                else "")
    brief = _brief_html(title=f"{'DRAFT — ' if draft else ''}{name}", place=place, paragraph=para, res=res,
                        impacts=impacts, maps=maps, caveats=caveats_for(ctx),
                        limitations=_model_card_limitations(ctx.run_dir), provenance=_provenance(ctx, spec),
                        unit=unit, draft=draft, extra_sections=extra_sections)
    (stage / "brief.html").write_text(brief, encoding="utf-8")
    levers = ", ".join(f"{v}: {u}" for v, u in (ctx.units.get("levers") or {}).items()) or "see the config"
    extra = ("hexagons.gpkg         hexagons as polygons\n" if "hexagons.gpkg" in hex_files else "") + extra_readme
    (stage / "README.txt").write_text(README_TEXT.format(unit=unit, levers=levers, extra=extra,
                                                         draft=("\n" + _DRAFT_NOTE + "\n") if draft else ""),
                                      encoding="utf-8")


def _export_dir(ctx_job, export_id: str) -> Path:
    if ctx_job.project_dir is None:
        raise ApiError("validation", "a pack export needs a project", detail={"errors": [
            {"path": "project_id", "message": "required", "code": "missing"}]})
    return Path(ctx_job.project_dir) / "exports" / export_id


def _stored_result(db, result_id: str) -> tuple[dict, dict, dict]:
    from sparc.studio.engine import store

    row = db.fetchone("SELECT * FROM results WHERE id = ?", (result_id,))
    if row is None:
        raise ApiError("not_found", f"no result {result_id!r}")
    rdir = Path(row["dir"])
    res = json.loads((rdir / "summary.json").read_text("utf-8"))
    spec = json.loads((rdir / "spec.json").read_text("utf-8"))
    cells = store.read_cells(rdir)
    arrays = {"delta": cells["delta"].to_numpy(np.float64),
              "delta_sd": cells["delta_sd"].to_numpy(np.float64) if "delta_sd" in cells else None,
              "extrapolation": cells["extrapolation"].to_numpy(np.float64) if "extrapolation" in cells else None,
              "realized": {c[len("realized_"):]: cells[c].to_numpy(np.float64) for c in cells.columns
                           if c.startswith("realized_")}}
    return row, res, {"spec": spec, **arrays}


def _draft_from_scenario(db, ctx, sid: str) -> tuple[dict, dict, dict]:
    """A preview-only result of scenario ``sid`` on the run of ``ctx`` (emulator; no SEs)."""
    from sparc.studio.engine import stats as S
    from sparc.studio.engine.compile import compile_scenario
    from sparc.studio.engine.preview import compute_delta, load_emulator
    from sparc.studio.workspace import utc_now

    srow = db.fetchone("SELECT * FROM scenarios WHERE id = ?", (sid,))
    if srow is None:
        raise ApiError("not_found", f"no scenario {sid!r}")
    doc = json.loads(srow["doc_json"])
    em = load_emulator(ctx)
    if em is None:
        raise ApiError("no_emulator", "a draft pack needs the run's emulator (or run the scenario exactly)")
    comp = compile_scenario(ctx, doc, db=db)
    delta = compute_delta(em, comp.dx(), comp.n)
    unit = ctx.units.get("target", "°F")
    spec = {"id": f"draft:{sid}", "kind": "preview", "scenario_id": sid, "content_hash": comp.content_hash,
            "compiled": comp.lever_summary(), "run_id": ctx.run_id, "created_utc": utc_now()}
    res = S.build_result(result_id=f"draft:{sid}", kind="exact", run_id=ctx.run_id, created_utc=utc_now(),
                         job_id=None, delta=delta, delta_sd=None, extrapolation=None, folds=None,
                         realized=comp.realized, grid=ctx.grid, unit=unit,
                         scenario={"id": sid, "revision": int(srow.get("revision") or 1), "name": doc.get("name")},
                         name=doc.get("name") or sid, requested=comp.requested, lever_cells=comp.lever_cells,
                         regions=comp.regions, per_unit=comp.costs, draft=True, spec=spec)
    return srow, res, {"spec": spec, "delta": delta, "delta_sd": None, "extrapolation": None,
                       "realized": comp.realized}


# ---------------------------------------------------------------------------
# the three packs
# ---------------------------------------------------------------------------

def decision_pack(jctx, export_id: str, result_id: str, thresholds=None) -> dict:
    """``export.decision_pack`` → ``{export_id, path, bytes, draft}``."""
    db = jctx.db
    draft = False
    if result_id.startswith("sc_"):
        srow = db.fetchone("SELECT * FROM scenarios WHERE id = ?", (result_id,))
        if srow is None:
            raise ApiError("not_found", f"no scenario {result_id!r}")
        exact = db.fetchone("SELECT id FROM results WHERE scenario_id = ? AND kind = 'exact' AND stale = 0 "
                            "ORDER BY created_utc DESC", (result_id,))
        if exact is not None:
            result_id = exact["id"]
        else:
            run_id = jctx.run_id or srow.get("anchor_run_id")
            if not run_id:
                raise ApiError("validation", "a draft pack needs the scenario's anchor run")
            ctx = _run_ctx(db, run_id)
            _row, res, arr = _draft_from_scenario(db, ctx, result_id)
            draft = True
    if not draft:
        row, res, arr = _stored_result(db, result_id)
        ctx = _run_ctx(db, row["run_id"])
        if row.get("kind") not in ("exact", "configured", "plan", "sweep_point"):
            draft = True
    name = ((res.get("scenario") or {}).get("name")) or arr["spec"].get("name") or result_id
    out_dir = _export_dir(jctx, export_id)
    with tempfile.TemporaryDirectory(prefix="pack-") as td:
        stage = Path(td) / "decision_pack"
        stage.mkdir()
        _decision_contents(ctx, stage, name=name, scenario=res.get("scenario"), res=res, spec=arr["spec"],
                           delta=arr["delta"], delta_sd=arr["delta_sd"], extrapolation=arr["extrapolation"],
                           realized=arr["realized"], draft=draft, thresholds=thresholds, cache_dir=jctx.cache_dir)
        zpath = out_dir / f"decision_pack_{_slug(name)}.zip"
        nbytes = _write_zip(stage, zpath)
    return {"export_id": export_id, "path": str(zpath), "bytes": nbytes, "draft": draft}


def _slug(s: str) -> str:
    from sparc.studio.workspace import slugify

    return slugify(str(s), max_len=40)


def plan_pack(jctx, export_id: str, plan_id: str) -> dict:
    """``export.plan_pack``: the decision-pack contents of the plan (its closed-loop result, else a DRAFT of
    the planned benefit) plus the field kit and the Pareto table."""
    import pandas as pd

    from sparc.studio import db as dbmod
    from sparc.studio.engine import stats as S
    from sparc.studio.scenarios.plans import field_kit, plan_out
    from sparc.studio.workspace import read_json, utc_now

    db = jctx.db
    prow = db.fetchone("SELECT * FROM plans WHERE id = ?", (plan_id,))
    if prow is None:
        raise ApiError("not_found", f"no plan {plan_id!r}")
    ctx = _run_ctx(db, prow["run_id"])
    plan = plan_out(prow)
    realised = read_json(Path(prow["dir"]) / "realised.json") or {}
    draft = not realised.get("result_id")
    params = dbmod.loads(prow.get("params_json"), {}) or {}
    lever = params.get("lever")
    if not draft:
        _row, res, arr = _stored_result(db, realised["result_id"])
    else:
        benefit = np.asarray(np.load(Path(prow["dir"]) / "planned_benefit.npy", allow_pickle=False), dtype=np.float64)
        dose = np.asarray(np.load(Path(prow["dir"]) / "dose.npy", allow_pickle=False), dtype=np.float64)
        sign = -1.0 if (read_json(Path(prow["dir"]) / "params.json") or {}).get("direction") == "decrease" else 1.0
        spec = {"id": f"draft:{plan_id}", "kind": "plan", "plan_id": plan_id, "run_id": ctx.run_id,
                "created_utc": utc_now()}
        res = S.build_result(result_id=f"draft:{plan_id}", kind="plan", run_id=ctx.run_id, created_utc=utc_now(),
                             job_id=None, delta=-benefit, delta_sd=None, extrapolation=None, folds=None,
                             realized={lever: sign * dose}, grid=ctx.grid, unit=ctx.units.get("target", "°F"),
                             name=plan["name"], draft=True, spec=spec)
        arr = {"spec": spec, "delta": -benefit, "delta_sd": None, "extrapolation": None,
               "realized": {lever: sign * dose}}
    kit = field_kit(db, ctx, prow)
    out_dir = _export_dir(jctx, export_id)
    planned = plan.get("planned") or {}
    extra = (f"<h2>Budget plan</h2><p>{_esc(planned.get('caption') or '')}</p>"
             f"<p>Planned cooling {_fmt(planned.get('planned_total'), 1)} vs realised "
             f"{_fmt((plan.get('realised') or {}).get('total'), 1)} (°·cells; the closed loop includes spillover "
             f"non-additivity).</p>")
    with tempfile.TemporaryDirectory(prefix="pack-") as td:
        stage = Path(td) / "plan_pack"
        stage.mkdir()
        _decision_contents(ctx, stage, name=f"Plan: {plan['name']}", scenario=None, res=res, spec=arr["spec"],
                           delta=arr["delta"], delta_sd=arr["delta_sd"], extrapolation=arr["extrapolation"],
                           realized=arr["realized"], draft=draft, cache_dir=jctx.cache_dir, extra_sections=extra,
                           extra_readme=("field_list.csv        ranked treated cells with ids and lon/lat\n"
                                         "logger_sites.csv      logger sites that sharpen the canopy effect\n"
                                         "before_after_pairs.csv  treated/control pairs for evaluation\n"
                                         "pareto.csv            planned benefit by budget\n"))
        pd.DataFrame(kit["cells"]).to_csv(stage / "field_list.csv", index=False)
        pd.DataFrame(kit["sites"], columns=["id", "lon", "lat", "role", "canopy", "impervious", "effect_sd"]).to_csv(
            stage / "logger_sites.csv", index=False)
        pd.DataFrame(kit["pairs"], columns=["treated_id", "control_id", "treated_lon", "treated_lat", "control_lon",
                                            "control_lat", "covariate_distance"]).to_csv(
            stage / "before_after_pairs.csv", index=False)
        fr = {f["budget"]: f["realised"] for f in plan.get("frontier") or []}
        pd.DataFrame([{**p, "realised": fr.get(p["budget"])} for p in planned.get("pareto") or []]).to_csv(
            stage / "pareto.csv", index=False)
        zpath = out_dir / f"plan_pack_{_slug(plan['name'])}.zip"
        nbytes = _write_zip(stage, zpath)
    return {"export_id": export_id, "path": str(zpath), "bytes": nbytes, "draft": draft}


def compare_pack(jctx, export_id: str, comparison_id: str) -> dict:
    """``export.compare_pack``: per-item summaries, paired differences, difference GeoTIFFs and a brief."""
    import pandas as pd

    db = jctx.db
    crow = db.fetchone("SELECT * FROM comparisons WHERE id = ?", (comparison_id,))
    if crow is None:
        raise ApiError("not_found", f"no comparison {comparison_id!r}")
    ctx = _run_ctx(db, crow["run_id"])
    summ = json.loads((Path(crow["dir"]) / "summary.json").read_text("utf-8"))
    unit = ctx.units.get("target", "°F")
    out_dir = _export_dir(jctx, export_id)
    draft = any(not it.get("has_folds") for it in summ.get("items") or [])
    with tempfile.TemporaryDirectory(prefix="pack-") as td:
        stage = Path(td) / "compare_pack"
        stage.mkdir()
        items = summ.get("items") or []
        pd.DataFrame([{"item": i, "label": it["label"], "city_mean": (it.get("city") or {}).get("estimate"),
                       "city_se": (it.get("city") or {}).get("se"), "cost": it.get("cost"),
                       "has_folds": it.get("has_folds")} for i, it in enumerate(items)]).to_csv(
            stage / "items.csv", index=False)
        rows = []
        maps = {}
        for p in summ.get("pairs") or []:
            a, b = p["a"], p["b"]
            c = p.get("city") or {}
            rows.append({"a": items[a]["label"], "b": items[b]["label"], "difference": c.get("estimate"),
                         "se": c.get("se"), "lo": c.get("lo"), "hi": c.get("hi"), "paired": c.get("paired")})
            f = Path(crow["dir"]) / f"diff_{a}__{b}.npy"
            if f.exists():
                d = np.load(f, allow_pickle=False).astype(np.float64)
                write_geotiff(ctx.grid, d, stage / f"diff_{a}__{b}.tif")
                maps[f"diff_{a}__{b}"] = map_png(ctx.grid, d)
        pd.DataFrame(rows).to_csv(stage / "pairs.csv", index=False)
        (stage / "summary.json").write_text(json.dumps(summ, indent=1, allow_nan=False), encoding="utf-8")
        figs = "".join(
            f'<figure><img alt="{k}" src="data:image/png;base64,{base64.b64encode(v).decode()}"><figcaption>'
            f'{_esc(items[int(k.split("_")[1])]["label"])} − {_esc(items[int(k.split("__")[1])]["label"])} '
            f'({unit}; blue = the first is cooler)</figcaption></figure>' for k, v in maps.items())
        prow = "".join(f"<tr><td>{_esc(r['a'])} − {_esc(r['b'])}</td><td>{_fmt(r['difference'], 3, True)}</td>"
                       f"<td>{_fmt(r['se'], 3)}</td><td>{'paired' if r['paired'] else 'independent'}</td></tr>"
                       for r in rows)
        irow = "".join(f"<tr><td>{_esc(it['label'])}</td><td>{_esc(_likely_text(it.get('city'), unit, False))}</td>"
                       f"<td>{_fmt(it.get('cost'), 0)}</td></tr>" for it in items)
        page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Comparison {comparison_id}</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;max-width:60rem;margin:2rem auto;padding:0 1rem}}
table{{border-collapse:collapse;width:100%}} td,th{{border-bottom:1px solid #eee;padding:.25rem .5rem;text-align:left}}
figure{{display:inline-block;max-width:46%;margin:.5rem}} img{{max-width:100%;image-rendering:pixelated}}</style></head>
<body><h1>Scenario comparison</h1><p>{_esc(ctx.name)} · run {_esc(ctx.run_id)}</p>
{'<p><strong>Some items have no fold-level results: their differences use independent SEs or none (re-run them exactly).</strong></p>' if draft else ''}
<h2>Items</h2><table><tr><th>item</th><th>city mean ΔT (likely range)</th><th>cost</th></tr>{irow}</table>
<h2>Paired differences</h2><table><tr><th>pair</th><th>difference ({unit})</th><th>SE</th><th>method</th></tr>{prow}</table>
<p>Paired SE uses the shared fold models: SE(A−B) = std over folds of the mean difference × √(K−1). Negative = the first item is cooler.</p>
<h2>Difference maps</h2>{figs}
</body></html>
"""
        (stage / "brief.html").write_text(page, encoding="utf-8")
        (stage / "README.txt").write_text(
            f"SPARC compare pack\n\nΔT in {unit}; negative = cooler. diff_<a>__<b>.tif = item a minus item b.\n"
            "items.csv: per-item city means; pairs.csv: paired differences (SE from shared folds when 'paired').\n",
            encoding="utf-8")
        zpath = out_dir / f"compare_pack_{comparison_id}.zip"
        nbytes = _write_zip(stage, zpath)
    return {"export_id": export_id, "path": str(zpath), "bytes": nbytes, "draft": draft}
