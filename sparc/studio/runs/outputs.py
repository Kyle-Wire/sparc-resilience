"""Output states, run-tab availability, documents, the files tree, conversions and the data dictionary
(SPEC §3.2, §6.1, §6.4 Files/Docs, api.md §6.1).

**States** (SPEC §6.1) of every catalog output of the run (``{var}`` expanded to the config's levers):

* ``present`` - the file exists;
* ``stale`` - more than 60 s older than ``manifest.created_utc`` while the manifest lacks its section;
* ``missing`` - with ``{produced_by, action}``;
* ``writing`` - the producing stage is running (live job) and no file has landed yet;
* ``partial`` - some files of a multi-file output exist.

**Tabs** (SPEC §3.2): one entry per run tab id, from the outputs whose ``view`` is that tab.  ``track`` is
running while a job is active, ``lab`` is missing without ``checkpoint.pkl``, ``map``, ``files``,
``provenance`` and ``validation`` are always ready.  Outputs of stages that did not run (disabled by config,
not requested) and optional outputs never degrade a tab.
"""

from __future__ import annotations

import csv
import html
import io
import json
import os
import re
from pathlib import Path

import numpy as np

from sparc.studio.errors import ApiError
from sparc.studio.runs.common import clean, mtime_iso
from sparc.studio.workspace import dir_size

__all__ = ["RUN_TAB_IDS", "FIXED_TABS", "stages_run", "output_entries", "tab_availability", "output_action",
           "output_content", "docs_list", "doc_content", "list_files", "file_table", "convert_file",
           "dictionary", "markdown_html", "DOCS"]

RUN_TAB_IDS = ("overview", "data", "accuracy", "distance", "influence", "response", "causal", "scenarios", "climate",
               "heat", "budget", "planner", "lab", "validation", "uncertainty", "provenance", "track", "map", "docs", "files")
FIXED_TABS = ("track", "map", "files", "provenance", "lab", "validation")
DOCS = [
    ("report", "report.md", "Run report", "run"),
    ("methods", "methods.md", "Methods", "run"),
    ("model_card", "model_card.md", "Model card", "run"),
    ("uncertainty", "uncertainty.md", "Uncertainty", "run"),
    ("placebo", "placebo.md", "Placebo tests", "study:placebo"),
    ("multiverse", "multiverse_summary.md", "Multiverse", "study:multiverse"),
    ("simcheck", "simcheck_summary.md", "Simulation check", "study:simcheck"),
    ("benchmark", "benchmark.md", "Effect benchmark", "study:benchmark"),
]
_STAGE_OF_MANIFEST = {"S0": "S0", "S1": "S1", "S2_S3": "S2_S3", "baselines": "baselines", "cv_curve": "cv_curve",
                      "S4": "S4", "S5": "S5", "climate": "climate", "S6": "S6", "S7": "S7"}


def stages_run(ctx, live: dict | None = None) -> set[str] | None:
    """Stage ids that ran (or are planned in a live job); None when unknown (every output is expected)."""
    if live:
        return {s for s, st in live.items() if st in ("planned", "running", "done", "cached")}
    m = ctx.manifest_raw or {}
    if isinstance(m.get("stages_run"), list):
        return set(m["stages_run"]) | {"finish"}
    if isinstance(m.get("timings_s"), dict) and m["timings_s"]:
        out = {k for k in m["timings_s"] if k in _STAGE_OF_MANIFEST}
        if "S5" in out and m.get("climate"):
            out.add("climate")
        return out | {"finish"}
    return None


def output_action(ctx, spec, *, resumable: bool) -> dict | None:
    """The one-click remedy of a missing output (api.md §0.3 ``Action``)."""
    from sparc.studio.runs.statusboard import relaunch_action

    pb = spec.produced_by
    rid = ctx.run_id
    kind, _, what = pb.partition(":")
    if kind == "stage":
        if resumable and what != "finish":
            return {"kind": "resume", "label": f"Resume to compute {what}", "method": "POST",
                    "path": f"/api/runs/{rid}/resume", "body": {}}
        if ctx.status in ("complete", "imported", "failed", "cancelled", "interrupted", "partial"):
            return relaunch_action(ctx, what)
        return None
    if kind == "post":
        if what == "emulator":
            return {"kind": "build_emulator", "label": "Build emulator", "method": "POST",
                    "path": f"/api/runs/{rid}/actions/emulator", "body": {}}
        labels = {"planner": "Run planner pack", "uncertainty": "Compute uncertainty envelopes",
                  "writeup": "Regenerate methods & model card", "baselines": "Score the baselines"}
        return {"kind": "run_job", "label": labels.get(what, f"Run {what}"), "method": "POST",
                "path": f"/api/runs/{rid}/actions/{what}", "body": {}}
    if kind == "study" and what in ("placebo", "simcheck", "multiverse", "reproduce"):
        return {"kind": "run_job", "label": f"Run the {what} study", "method": "POST",
                "path": f"/api/runs/{rid}/studies/{what}", "body": {}}
    return None


def _match_files(ctx, spec) -> list[str]:
    out = []
    for pat in spec.patterns():
        if any(ch in pat for ch in "*?["):
            out.extend(sorted(str(p.relative_to(ctx.run_dir)).replace(os.sep, "/") for p in ctx.run_dir.glob(pat)
                              if p.is_file()))
        elif (ctx.run_dir / pat).is_file():
            out.append(pat)
    return out


def output_entries(ctx, *, live: dict | None = None, resumable: bool = False) -> list[dict]:
    """Every catalog output of the run with its state (api.md ``OutputEntry``) plus ``_expected``."""
    from sparc.core.catalog import expand_outputs

    levers = list((ctx.cfg_raw.get("actionable") or {}))
    for v in ctx.response_vars():
        if v not in levers:
            levers.append(v)
    specs = expand_outputs(levers=levers, root="run")
    ran = stages_run(ctx, live)
    out = []
    for spec in specs:
        if spec.id == "planner_hex_other":
            continue
        files = _match_files(ctx, spec)
        fixed = [f for f in spec.files if not any(ch in f for ch in "*?[{")]
        stage = spec.produced_by.split(":", 1)[1] if spec.produced_by.startswith("stage:") else None
        live_state = (live or {}).get(stage) if stage else None
        if not files:
            state = "writing" if live_state == "running" else "missing"
        elif fixed and len([f for f in fixed if f in files]) < len(fixed):
            state = "partial"
        elif any(ctx.stale_file(f, spec.manifest_key) for f in files):
            state = "stale"
        else:
            state = "present"
        expected = not spec.optional
        if stage and ran is not None and stage not in ran:
            expected = False
        if spec.produced_by.startswith(("post:", "study:", "studio:", "progress:")):
            expected = False
        if spec.id == "scenario_detail" and ((ctx.manifest_raw or {}).get("schema_version") or 0) >= 2 and \
                (stage is None or ran is None or stage in ran):
            expected = True
        if spec.id in ("checkpoint", "checkpoint_json"):
            expected = False
        entry = {"id": spec.id, "label": spec.label, "group": spec.group, "state": state,
                 "produced_by": spec.produced_by, "view": spec.view, "formats": list(spec.formats),
                 "files": [{"relpath": f, "bytes": _size(ctx.run_dir / f), "mtime": mtime_iso(ctx.run_dir / f) or ""}
                           for f in files],
                 "action": output_action(ctx, spec, resumable=resumable) if state in ("missing", "partial") else None,
                 "_expected": expected, "_manifest_key": spec.manifest_key}
        out.append(entry)
    return out


def _size(p: Path) -> int:
    try:
        return int(p.stat().st_size)
    except OSError:
        return 0


def _makes(kind: str) -> tuple[str, ...]:
    """The ``produced_by`` values (``stage:`` / ``progress:`` as prefixes) of the outputs a job kind writes:
    ``run.core`` the stages, ``post.planner`` ``post:planner``, ``study.placebo`` ``study:placebo``, …"""
    if kind in ("run.core", "run.external"):
        return ("stage:", "progress:")
    group, _, name = kind.partition(".")
    return (f"{group}:{name}",) if name else ()


def tab_availability(ctx, entries: list[dict], *, job_active: bool, active_kinds=None) -> list[dict]:
    """One ``{id, availability, missing[]}`` per run tab id (SPEC §3.2).

    ``job_active``: a job runs on the run (the Track tab follows it).  ``active_kinds``: the kinds of the
    active jobs; a tab whose outputs are all missing is ``running`` ("being computed") only while one of them
    makes those outputs (``None``: any active job counts)."""
    made = None if active_kinds is None else {m for k in active_kinds for m in _makes(k)}

    def making(rows) -> bool:
        if made is None:
            return job_active
        return any(e["produced_by"] == m or (m.endswith(":") and e["produced_by"].startswith(m))
                   for e in rows for m in made)

    tabs = []
    for tab in RUN_TAB_IDS:
        if tab in FIXED_TABS:
            missing: list[dict] = []
            if tab == "track":
                avail = "running" if job_active else "ready"
            elif tab == "lab":
                if (ctx.run_dir / "checkpoint.pkl").exists():
                    avail = "ready"
                else:
                    avail = "missing"
                    missing = [{"output": "checkpoint", "produced_by": "stage:S2_S3", "action": None,
                                "reason": "needs a checkpoint"}]
            else:
                avail = "ready"
            tabs.append({"id": tab, "availability": avail, "missing": missing})
            continue
        if tab == "heat":
            # the Heat tab reads the held-out predictions (observed temperature) and whatever else is there
            pe = next((e for e in entries if e["id"] == "predictions"), None)
            st = (pe or {}).get("state", "missing")
            if st in ("present", "stale", "partial"):
                tabs.append({"id": tab, "availability": "stale" if st == "stale" else "ready", "missing": []})
            else:
                tabs.append({"id": tab, "availability": "running" if st == "writing" or making([pe or {
                    "produced_by": "stage:S2_S3"}]) else "missing", "missing": [
                    {"output": "predictions", "produced_by": "stage:S2_S3", "action": (pe or {}).get("action")}]})
            continue
        mine = [e for e in entries if e["view"] == tab]
        counted = [e for e in mine if e["_expected"] or e["state"] != "missing"]
        if tab == "data" and not counted:
            # the Data & QA tab reads the manifest's qa section, or S0's metadata while the run continues
            qa = bool(((ctx.manifest_raw or {}).get("qa"))) or bool(ctx.meta.get("n_points"))
            avail = "ready" if qa else ("running" if making([{"produced_by": "stage:S0"}]) else "missing")
            tabs.append({"id": tab, "availability": avail, "missing": [] if qa else [
                {"output": "manifest", "produced_by": "stage:S0", "action": None}]})
            continue
        if not counted:
            counted = mine
        states = [e["state"] for e in counted]
        miss = [{"output": e["id"], "produced_by": e["produced_by"], "action": e["action"]}
                for e in counted if e["state"] in ("missing", "partial")]
        if not counted:
            avail = "running" if job_active else "missing"
        elif all(s in ("missing", "writing") for s in states):
            avail = "running" if any(s == "writing" for s in states) or making(counted) else "missing"
        elif any(s == "stale" for s in states):
            avail = "stale"
        elif any(s in ("missing", "partial", "writing") for s in states):
            avail = "partial" if not any(s == "writing" for s in states) else "running"
        else:
            avail = "ready"
        tabs.append({"id": tab, "availability": avail, "missing": miss})
    return tabs


def _with_meta(content, meta: dict) -> dict:
    """An object's content with ``_meta`` added (non-object content is wrapped as ``{content: …}``)."""
    if isinstance(content, dict):
        return clean({**content, "_meta": meta})
    return clean({"content": content, "_meta": meta})


def output_content(ctx, oid: str) -> dict:
    """``GET /outputs/{oid}``: the normalised JSON content of one output plus ``_meta``."""
    from sparc.core.catalog import expand_outputs
    from sparc.studio.runs.statusboard import can_resume

    levers = list((ctx.cfg_raw.get("actionable") or {})) + ctx.response_vars()
    spec = next((s for s in expand_outputs(levers=levers, root="run") if s.id == oid), None)
    if spec is None:
        raise ApiError("not_found", f"no catalog output {oid!r}")
    files = _match_files(ctx, spec)
    key = spec.manifest_key
    sec = ctx.sections.get(key) if key else None
    meta = {"source": None, "stale": False, "older_code": bool(sec and sec.get("older_code"))}
    if key and key in ctx.manifest and spec.files[0].endswith(".json"):
        content = ctx.manifest[key]
        meta.update(source=(sec or {}).get("source") or "manifest", stale=bool((sec or {}).get("stale")))
        return _with_meta(content, meta)
    if not files:
        expected = spec.files[0]
        raise ApiError("output_missing", f"{spec.label} is not in this run",
                       detail={"output": oid, "produced_by": spec.produced_by, "expected_path": expected},
                       action=output_action(ctx, spec, resumable=can_resume(ctx, False)))
    rel = files[0]
    meta.update(source="file", stale=ctx.stale_file(rel, key))
    p = ctx.run_dir / rel
    if rel.endswith(".json"):
        return _with_meta(ctx.json(rel), meta)
    if rel.endswith(".parquet"):
        return _with_meta(file_table(ctx, rel, limit=200), meta)
    if rel.endswith((".md", ".txt", ".csv", ".jsonl")):
        return _with_meta({"text": p.read_text("utf-8", errors="replace")[:2_000_000], "files": files}, meta)
    if rel.endswith(".npz"):
        with np.load(p, allow_pickle=False) as z:
            arrays = {k: {"shape": list(z[k].shape), "dtype": str(z[k].dtype)} for k in z.files}
        return _with_meta({"arrays": arrays, "files": files}, meta)
    return _with_meta({"files": [{"relpath": f, "bytes": _size(ctx.run_dir / f)} for f in files]}, meta)


# ---------------------------------------------------------------------------
# docs
# ---------------------------------------------------------------------------

def _study_dirs(ctx, db, kind: str) -> list[Path]:
    rows = []
    try:
        rows = db.fetchall("SELECT s.out_dir FROM studies s LEFT JOIN study_links l ON l.study_id = s.id "
                           "WHERE s.kind = ? AND (s.target_run_id = ? OR l.run_id = ? "
                           "OR (? = 'benchmark' AND s.project_id = ?)) ORDER BY s.updated_utc DESC",
                           (kind, ctx.run_id, ctx.run_id, kind, ctx.project_id))
    except Exception:
        rows = []
    return [Path(r["out_dir"]) for r in rows if r.get("out_dir")]


def _doc_path(ctx, db, doc: str) -> Path | None:
    spec = next((d for d in DOCS if d[0] == doc), None)
    if spec is None:
        return None
    _id, fname, _title, where = spec
    if where == "run":
        return ctx.run_dir / fname
    kind = where.split(":", 1)[1]
    for d in _study_dirs(ctx, db, kind):
        if (d / fname).is_file():
            return d / fname
    if (ctx.run_dir / fname).is_file():
        return ctx.run_dir / fname
    return None


def docs_list(ctx, db) -> list[dict]:
    out = []
    for doc, fname, title, _where in DOCS:
        p = _doc_path(ctx, db, doc)
        present = bool(p and p.is_file())
        out.append({"id": doc, "file": fname, "title": title, "mtime": mtime_iso(p) if present else None,
                    "present": present, "regenerable": doc in ("methods", "model_card"), "frozen": doc == "report"})
    return out


def doc_content(ctx, db, doc: str) -> dict:
    if not any(d[0] == doc for d in DOCS):
        raise ApiError("not_found", f"unknown document {doc!r}")
    p = _doc_path(ctx, db, doc)
    if p is None or not p.is_file():
        spec = next(d for d in DOCS if d[0] == doc)
        produced = {"report": "stage:finish", "methods": "post:writeup", "model_card": "post:writeup",
                    "uncertainty": "post:uncertainty"}.get(doc, spec[3])
        raise ApiError("output_missing", f"{spec[2]} is not available for this run",
                       detail={"output": doc, "produced_by": produced, "expected_path": spec[1]})
    return {"markdown": p.read_text("utf-8", errors="replace"), "mtime": mtime_iso(p) or "",
            "frozen": doc == "report"}


# ---------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------

def _safe(ctx, rel: str | None) -> Path:
    from sparc.studio.security import safe_path

    if not rel or rel in (".", "/"):
        return ctx.run_dir.resolve()
    return safe_path(rel, ctx.run_dir)


def list_files(ctx, path: str | None = None) -> list[dict]:
    """The files tree of the run directory (one level): name, size, mtime, catalog id, state, in_manifest."""
    from sparc.core.catalog import is_ignored, match_output

    base = _safe(ctx, path)
    if not base.is_dir():
        raise ApiError("not_found", f"no folder {path!r} in this run")
    root = ctx.run_dir.resolve()
    manifest = ctx.manifest_raw or {}
    out = []
    for p in sorted(base.iterdir(), key=lambda q: (not q.is_dir(), q.name)):
        try:
            rp = p.resolve()
        except OSError:
            continue
        if rp != root and root not in rp.parents:
            continue                               # a symlink out of the run folder: never listed or sized
        rel = str(p.relative_to(root)).replace(os.sep, "/")
        if p.name == ".sparc.lock" or (p.is_file() and p.name.endswith(".tmp")):
            continue
        is_dir = p.is_dir()
        spec = None if is_dir or is_ignored(rel) else match_output(rel, "run")
        state = None
        if spec is not None:
            state = "stale" if ctx.stale_file(rel, spec.manifest_key) else "present"
        try:
            st = p.stat()
        except OSError:
            continue
        out.append({"name": p.name, "relpath": rel, "dir": is_dir,
                    "bytes": dir_size(p) if is_dir else int(st.st_size), "mtime": mtime_iso(p),
                    "output_id": spec.id if spec else None, "state": state,
                    "in_manifest": bool(spec and spec.manifest_key and spec.manifest_key in manifest)})
    return out


def file_table(ctx, path: str, limit: int = 200, columns: list[str] | None = None) -> dict:
    """A preview of a parquet / CSV / JSONL file: ``{columns: [{name, dtype}], rows, n_rows}``."""
    import pandas as pd

    p = _safe(ctx, path)
    if not p.is_file():
        raise ApiError("not_found", f"no file {path!r} in this run")
    limit = max(1, min(int(limit or 200), 5000))
    if p.suffix == ".parquet":
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(p)
        names = pf.schema_arrow.names
        cols = [c for c in (columns or names) if c in names]
        n_rows = pf.metadata.num_rows
        df = pf.read(columns=cols).slice(0, limit).to_pandas()
    elif p.suffix == ".csv":
        df = pd.read_csv(p, nrows=limit, usecols=lambda c: columns is None or c in columns)
        with open(p, "rb") as f:
            n_rows = max(0, sum(1 for _ in f) - 1)
    elif p.suffix == ".jsonl":
        rows = []
        n_rows = 0
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                n_rows += 1
                if len(rows) < limit:
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        continue
        df = pd.json_normalize(rows)
        if columns:
            df = df[[c for c in columns if c in df.columns]]
    else:
        raise ApiError("no_conversion", f"{p.suffix or 'this file'} is not tabular",
                       detail={"path": path})
    cols = [{"name": str(c), "dtype": str(df[c].dtype)} for c in df.columns]
    rows = [[_cell(v) for v in r] for r in df.itertuples(index=False, name=None)]
    return {"columns": cols, "rows": rows, "n_rows": int(n_rows)}


def _cell(v):
    if isinstance(v, (np.generic,)):
        v = v.item()
    if isinstance(v, float) and not np.isfinite(v):
        return None
    if isinstance(v, (list, dict, str, int, float, bool)) or v is None:
        return clean(v)
    return str(v)


def _lonlat_for(ctx, df) -> tuple[np.ndarray, np.ndarray] | None:
    """lon/lat per row of a per-cell table (by id, else by ``x_m``/``y_m``); None without a CRS."""
    g = ctx.grid
    if g is None or not g.crs:
        return None
    from sparc.studio.runs.grid import lonlat_transformer, transform_xy

    if {"x_m", "y_m"} <= set(df.columns):
        return transform_xy(lonlat_transformer(g.crs), df["x_m"].to_numpy(float) / g.coord_scale,
                            df["y_m"].to_numpy(float) / g.coord_scale)
    if "id" in df.columns:
        import pandas as pd

        look = pd.DataFrame({"lon": g.lon, "lat": g.lat}, index=np.asarray(g.ids).astype(str))
        m = look.reindex(df["id"].astype(str).to_numpy())
        return m["lon"].to_numpy(float), m["lat"].to_numpy(float)
    return None


def convert_file(ctx, path: str, as_: str) -> tuple[bytes, str, str]:
    """``(body, media type, file name)`` of a conversion (api.md §6.1 ``files/raw?as=``)."""
    import pandas as pd

    p = _safe(ctx, path)
    if not p.is_file():
        raise ApiError("not_found", f"no file {path!r} in this run")
    stem = p.stem
    if as_ == "html" and p.suffix == ".md":
        body = markdown_html(p.read_text("utf-8", errors="replace"), title=p.name)
        return body.encode("utf-8"), "text/html; charset=utf-8", f"{stem}.html"
    if p.suffix in (".parquet", ".csv"):
        df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
        if as_ == "csv":
            ll = _lonlat_for(ctx, df)
            if ll is not None and "lon" not in df.columns:
                df = df.copy()
                pos = 1 if "id" in df.columns else 0
                df.insert(pos, "lon", np.round(ll[0], 7))
                df.insert(pos + 1, "lat", np.round(ll[1], 7))
            return df.to_csv(index=False).encode("utf-8"), "text/csv; charset=utf-8", f"{stem}.csv"
        if as_ == "json":
            return df.to_json(orient="records").encode("utf-8"), "application/json", f"{stem}.json"
        if as_ == "geojson":
            ll = _lonlat_for(ctx, df)
            if ll is None:
                raise ApiError("needs_crs", "GeoJSON needs a run with a CRS (data.crs)")
            if len(df) > 60_000:
                raise ApiError("too_many_features", f"{len(df):,} points exceed the 60,000-feature GeoJSON limit; "
                               "download CSV with lon/lat instead", detail={"n": int(len(df)), "max": 60_000})
            return geojson_points(df, ll[0], ll[1]).encode("utf-8"), "application/geo+json", f"{stem}.geojson"
    if p.suffix == ".json" and as_ == "csv":
        rows = _json_rows(json.loads(p.read_text("utf-8")))
        if rows is None:
            raise ApiError("no_conversion", f"{p.name} is not tabular JSON")
        buf = io.StringIO()
        cols: list[str] = []
        for r in rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        w = csv.DictWriter(buf, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()})
        return buf.getvalue().encode("utf-8"), "text/csv; charset=utf-8", f"{stem}.csv"
    raise ApiError("no_conversion", f"cannot convert {p.name} to {as_}", detail={"path": path, "as": as_})


def _json_rows(obj) -> list[dict] | None:
    """Rows of a tabular JSON section: a list of objects, an object of equal-length lists or of objects."""
    if isinstance(obj, list) and obj and all(isinstance(r, dict) for r in obj):
        return obj
    if isinstance(obj, dict) and obj:
        vals = list(obj.values())
        if all(isinstance(v, list) for v in vals) and len({len(v) for v in vals}) == 1:
            return [dict(zip(obj, row)) for row in zip(*vals)]
        if all(isinstance(v, dict) for v in vals):
            return [{"key": k, **v} for k, v in obj.items()]
        for k in ("rows", "projections", "scenarios", "points"):
            if isinstance(obj.get(k), (list, dict)):
                r = _json_rows(obj[k])
                if r is not None:
                    return r
    return None


def geojson_points(df, lon, lat) -> str:
    feats = []
    cols = [c for c in df.columns if c not in ("lon", "lat")]
    recs = df[cols].to_dict(orient="records")
    for rec, x, y in zip(recs, lon, lat):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [round(float(x), 7),
                                                                                         round(float(y), 7)]},
                      "properties": clean(rec)})
    return json.dumps({"type": "FeatureCollection", "features": feats}, separators=(",", ":"))


def dictionary(ctx) -> list[dict]:
    """The data dictionary resolved for the run's config (rows of the outputs the run has)."""
    from sparc.core.catalog import dictionary_rows

    rows = dictionary_rows(ctx.cfg_raw)
    have = [r for r in rows if r["output"] == "manifest" or (ctx.run_dir / r["output"]).exists()]
    return have or rows


# ---------------------------------------------------------------------------
# markdown → html (escapes raw HTML; headings, lists, code, tables, emphasis, links)
# ---------------------------------------------------------------------------

def _inline(s: str) -> str:
    s = html.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", lambda m: f"<code>{m.group(1)}</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![*\w])\*([^*\n]+)\*(?![*\w])", r"<em>\1</em>", s)
    s = re.sub(r"\[([^\]]+)\]\(((?:https?://|#|/)[^)\s]*)\)",
               lambda m: f'<a href="{html.escape(m.group(2))}">{m.group(1)}</a>', s)
    return s


def markdown_html(md: str, title: str = "document") -> str:
    lines = md.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    i = 0
    para: list[str] = []

    def flush():
        if para:
            out.append("<p>" + _inline(" ".join(para)) + "</p>")
            para.clear()

    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            flush()
            j = i + 1
            code = []
            while j < len(lines) and not lines[j].startswith("```"):
                code.append(lines[j])
                j += 1
            out.append("<pre><code>" + html.escape("\n".join(code)) + "</code></pre>")
            i = j + 1
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            flush()
            lvl = len(m.group(1))
            out.append(f"<h{lvl}>{_inline(m.group(2).strip())}</h{lvl}>")
            i += 1
            continue
        if line.strip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|?[\s:|-]+\|?\s*$", lines[i + 1]):
            flush()
            head = [c.strip() for c in line.strip().strip("|").split("|")]
            rows = []
            j = i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                rows.append([c.strip() for c in lines[j].strip().strip("|").split("|")])
                j += 1
            out.append("<table><thead><tr>" + "".join(f"<th>{_inline(h)}</th>" for h in head) + "</tr></thead><tbody>"
                       + "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in rows)
                       + "</tbody></table>")
            i = j
            continue
        m = re.match(r"^\s*([-*+]|\d+[.)])\s+(.*)$", line)
        if m:
            flush()
            ordered = m.group(1)[0].isdigit()
            items = []
            j = i
            while j < len(lines):
                mm = re.match(r"^\s*([-*+]|\d+[.)])\s+(.*)$", lines[j])
                if not mm:
                    break
                items.append(mm.group(2))
                j += 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{_inline(t)}</li>" for t in items) + f"</{tag}>")
            i = j
            continue
        if not line.strip():
            flush()
        elif line.startswith(">"):
            flush()
            out.append("<blockquote>" + _inline(line.lstrip("> ")) + "</blockquote>")
        else:
            para.append(line.strip())
        i += 1
    flush()
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>{html.escape(title)}</title><style>body{{font:16px/1.55 system-ui,sans-serif;max-width:46rem;"
            "margin:2rem auto;padding:0 1rem}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;"
            "padding:.2rem .5rem}}code{{background:#f3f2ee;padding:0 .2rem}}</style></head><body>"
            + "\n".join(out) + "</body></html>")
