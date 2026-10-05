"""The scenario library (SPEC §7.4, api.md §7.4): scenarios, revisions and forks, statuses, ladders, promote to
config, design CSVs, the configured (S5) group and the project mirror files.

**Revisions.**  Each revision is its own scenario row (``sc_…``) with ``revision`` and ``parent_id``.  Patching
the ``edits``, ``regions``, ``costs`` or ``options`` of a scenario that already has an exact result is
``409 conflict_revision`` (the client forks instead), so results are never orphaned.

**Statuses** ``draft`` → ``previewed`` → ``exact`` → ``stale`` → ``archived``: ``exact`` while a current exact
result exists, ``stale`` when all of them were made from another checkpoint or core code.

**Mirror** (api.md §12.4): ``projects/<slug>/scenarios/<root>.json`` holds every revision descending from the
lineage root ``<root>`` (``{"schema": 1, "revisions": [ScenarioDoc + id/revision/parent_id/content_hash/
status/…], "archived": bool}``), rewritten atomically on every create, patch or fork.  ``sparc studio
--reindex`` rebuilds the ``scenarios`` table from these files (:func:`reindex_scenarios`).
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from sparc.studio import db as dbmod
from sparc.studio.db import reindex_hook
from sparc.studio.engine.compile import canonical_doc, content_hash, doc_dict
from sparc.studio.errors import ApiError
from sparc.studio.workspace import new_id, read_json, utc_now, write_json_atomic

log = logging.getLogger("sparc.studio.scenarios")

__all__ = ["get_row", "create", "scenario_out", "summary_out", "list_scenarios", "patch", "fork", "delete",
           "sync_status", "mark_previewed", "write_mirror", "root_of", "reindex_scenarios", "ladder_docs",
           "promote", "design_csv", "import_design", "configured_list", "configured_doc", "project_row",
           "STRUCTURAL"]

STRUCTURAL = ("edits", "regions", "costs", "options")


# ---------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------

def get_row(db, sid: str) -> dict:
    row = db.fetchone("SELECT * FROM scenarios WHERE id = ?", (sid,))
    if row is None:
        raise ApiError("not_found", f"no scenario {sid!r}")
    return row


def project_row(db, pid: str) -> dict:
    row = db.fetchone("SELECT * FROM projects WHERE id = ?", (pid,))
    if row is None:
        raise ApiError("not_found", f"no project {pid!r}")
    return row


def _doc(row: dict) -> dict:
    return dbmod.loads(row.get("doc_json"), {}) or {}


def _unit(db, row: dict) -> str:
    return "°F"


def _children(db, sid: str) -> list[str]:
    return [r["id"] for r in db.fetchall("SELECT id FROM scenarios WHERE parent_id = ? ORDER BY revision, created_utc",
                                         (sid,))]


def scenario_out(db, row: dict, unit: str = "°F") -> dict:
    """A ``scenarios`` row as the api.md ``Scenario``."""
    from sparc.studio.engine import store

    return {"id": row["id"], "project_id": row.get("project_id") or "", "revision": int(row.get("revision") or 1),
            "parent_id": row.get("parent_id"), "children": _children(db, row["id"]), "doc": _doc(row),
            "content_hash": row["content_hash"], "status": _status_of(row),
            "created_utc": row.get("created_utc") or "", "updated_utc": row.get("updated_utc") or "",
            "results": store.scenario_results(db, row["id"], unit)}


def _status_of(row: dict) -> str:
    if row.get("archived"):
        return "archived"
    st = row.get("status") or "draft"
    return st if st in ("draft", "previewed", "exact", "stale", "archived") else "draft"


def summary_out(db, row: dict, unit: str = "°F", run_id: str | None = None) -> dict:
    from sparc.studio.engine import store

    rows = db.fetchall("SELECT * FROM results WHERE scenario_id = ?" + (" AND run_id = ?" if run_id else "")
                       + " ORDER BY created_utc DESC", (row["id"], run_id) if run_id else (row["id"],))
    doc = _doc(row)
    return {"id": row["id"], "revision": int(row.get("revision") or 1), "parent_id": row.get("parent_id"),
            "status": _status_of(row), "created_utc": row.get("created_utc") or "",
            "updated_utc": row.get("updated_utc") or "", "name": doc.get("name") or row.get("name") or row["id"],
            "tags": dbmod.loads(row.get("tags_json"), []) or [], "latest": store.summary_out(rows[0], unit) if rows
            else None}


def list_scenarios(db, pid: str, *, run: str | None = None, tag: str | None = None, status: str | None = None,
                   q: str | None = None, archived: bool = False, unit: str = "°F") -> list[dict]:
    rows = db.fetchall("SELECT * FROM scenarios WHERE project_id = ? ORDER BY updated_utc DESC, created_utc DESC",
                       (pid,))
    out = []
    for r in rows:
        if not archived and r.get("archived"):
            continue
        tags = dbmod.loads(r.get("tags_json"), []) or []
        if tag and tag not in tags:
            continue
        if status and _status_of(r) != status:
            continue
        doc = _doc(r)
        if q and q.lower() not in f"{doc.get('name', '')} {doc.get('notes', '')} {' '.join(tags)}".lower():
            continue
        if run:
            has = db.fetchone("SELECT 1 AS x FROM results WHERE scenario_id = ? AND run_id = ?", (r["id"], run))
            if not has and r.get("anchor_run_id") != run:
                continue
        out.append(summary_out(db, r, unit, run))
    return out


def create(db, workspace, project_id: str, doc: Any, *, parent_id: str | None = None, revision: int = 1,
           status: str = "draft") -> dict:
    project_row(db, project_id)
    d = doc_dict(doc)
    now = utc_now()
    row = {"id": new_id("scenario"), "project_id": project_id, "revision": int(revision), "parent_id": parent_id,
           "name": d["name"], "doc_json": dbmod.dumps(d), "content_hash": content_hash(d),
           "tags_json": dbmod.dumps(list(d.get("tags") or [])), "status": status,
           "anchor_run_id": d.get("anchor_run_id"), "archived": 0, "created_utc": now, "updated_utc": now}
    db.insert("scenarios", row)
    write_mirror(db, workspace, row["id"])
    return row


def has_exact(db, sid: str) -> bool:
    return db.fetchone("SELECT 1 AS x FROM results WHERE scenario_id = ? AND kind = 'exact'", (sid,)) is not None


def patch(db, workspace, sid: str, body: dict) -> dict:
    """``PATCH /api/scenarios/{sid}``; ``409 conflict_revision`` when an exact result pins the structure."""
    row = get_row(db, sid)
    doc = _doc(row)
    updates: dict[str, Any] = {}
    if body.get("doc") is not None:
        new = doc_dict(body["doc"])
        changed = [k for k in STRUCTURAL if json.dumps(new.get(k), sort_keys=True) != json.dumps(doc.get(k),
                                                                                                 sort_keys=True)]
        if changed and content_hash(new) != row["content_hash"] and has_exact(db, sid):
            raise ApiError("conflict_revision", "this scenario already has an exact result: fork it to change "
                           f"its {', '.join(changed)}", detail={"scenario_id": sid, "changed": changed},
                           action={"kind": "open", "label": "Fork as a new revision", "method": "POST",
                                   "path": f"/api/scenarios/{sid}/fork", "body": {"doc": new}})
        doc = new
    for k in ("name", "notes"):
        if body.get(k) is not None:
            doc[k] = body[k]
    if body.get("tags") is not None:
        doc["tags"] = list(body["tags"])
    d = doc_dict(doc)
    updates.update(doc_json=dbmod.dumps(d), name=d["name"], content_hash=content_hash(d),
                   tags_json=dbmod.dumps(list(d.get("tags") or [])), anchor_run_id=d.get("anchor_run_id"),
                   updated_utc=utc_now())
    if body.get("archived") is not None:
        updates["archived"] = 1 if body["archived"] else 0
    if updates.get("content_hash") != row["content_hash"] and row.get("status") in ("draft", "previewed"):
        # The Lab previews an edit (120 ms) before it autosaves it (2 s): content its latest preview
        # showed stays "previewed"; any other new content is a draft again.
        updates["status"] = "previewed" if _PREVIEWED.get(sid) == preview_key(d) else "draft"
    db.update("scenarios", {"id": sid}, updates)
    sync_status(db, sid)
    write_mirror(db, workspace, sid)
    return get_row(db, sid)


def fork(db, workspace, sid: str, *, name: str | None = None, doc: Any = None) -> dict:
    """A new revision (revision + 1, ``parent_id`` = sid) from ``doc`` or the parent's document."""
    row = get_row(db, sid)
    d = doc_dict(doc) if doc is not None else _doc(row)
    if name:
        d["name"] = name
    return create(db, workspace, row["project_id"], d, parent_id=sid, revision=int(row.get("revision") or 1) + 1)


def delete(db, workspace, sid: str, *, results: bool = False, force: bool = False) -> None:
    from sparc.studio.engine import store
    from sparc.studio.schemas.common import ACTIVE_STATUSES

    row = get_row(db, sid)
    marks = ",".join("?" for _ in ACTIVE_STATUSES)
    live = db.fetchone(f"SELECT id FROM jobs WHERE scenario_id = ? AND status IN ({marks})", (sid, *ACTIVE_STATUSES))
    if live is not None:
        raise ApiError("active", "an exact run of this scenario is still running: cancel it first",
                       detail={"job_id": live["id"]})
    kids = _children(db, sid)
    if kids and not force:
        raise ApiError("has_children", f"scenario {sid} has {len(kids)} later revision(s): pass force=true",
                       detail={"children": kids})
    if results:
        for r in db.fetchall("SELECT id FROM results WHERE scenario_id = ?", (sid,)):
            store.delete_result(db, r["id"])
    root = root_of(db, sid)
    db.execute("UPDATE scenarios SET parent_id = ? WHERE parent_id = ?", (row.get("parent_id"), sid))
    db.execute("DELETE FROM scenarios WHERE id = ?", (sid,))
    if root == sid:
        rest = db.fetchall("SELECT id FROM scenarios WHERE project_id = ? AND parent_id IS NULL", (row["project_id"],))
        _remove_mirror(db, workspace, row["project_id"], sid)
        for r in rest:
            write_mirror(db, workspace, r["id"])
    else:
        write_mirror(db, workspace, root)


def sync_status(db, sid: str) -> str | None:
    """Recompute ``exact`` / ``stale`` from the scenario's exact results (draft/previewed otherwise kept)."""
    row = db.fetchone("SELECT * FROM scenarios WHERE id = ?", (sid,))
    if row is None:
        return None
    rows = db.fetchall("SELECT stale FROM results WHERE scenario_id = ? AND kind = 'exact'", (sid,))
    cur = row.get("status") or "draft"
    if rows:
        new = "exact" if any(not r.get("stale") for r in rows) else "stale"
    else:
        new = cur if cur in ("draft", "previewed") else "draft"
    if new != cur:
        db.update("scenarios", {"id": sid}, {"status": new})
    return new


# The evaluated content ({edits, options}) of each scenario's latest preview, so a save of exactly
# that content keeps "previewed". Process-local and bounded: a restart only loses this status hint.
_PREVIEWED: dict[str, str] = {}
_PREVIEWED_MAX = 1024


def preview_key(doc: Any) -> str:
    """Hash of the part of a doc a preview evaluates: its canonical edits and options."""
    c = canonical_doc(doc)
    text = json.dumps({"edits": c["edits"], "options": c["options"]}, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def mark_previewed(db, sid: str, doc: Any = None) -> None:
    """A preview of ``doc`` (``{edits, options}``) was shown for scenario ``sid``: a draft becomes ``previewed``."""
    if doc is not None:
        _PREVIEWED.pop(sid, None)
        _PREVIEWED[sid] = preview_key(doc)
        while len(_PREVIEWED) > _PREVIEWED_MAX:
            _PREVIEWED.pop(next(iter(_PREVIEWED)))
    row = db.fetchone("SELECT status FROM scenarios WHERE id = ?", (sid,))
    if row is not None and (row.get("status") or "draft") == "draft":
        db.update("scenarios", {"id": sid}, {"status": "previewed"})


# ---------------------------------------------------------------------------
# mirror files (api.md §12.4)
# ---------------------------------------------------------------------------

def root_of(db, sid: str) -> str:
    seen = set()
    cur = sid
    while True:
        row = db.fetchone("SELECT parent_id FROM scenarios WHERE id = ?", (cur,))
        if row is None or not row.get("parent_id") or row["parent_id"] in seen:
            return cur
        seen.add(cur)
        if db.fetchone("SELECT id FROM scenarios WHERE id = ?", (row["parent_id"],)) is None:
            return cur
        cur = row["parent_id"]


def _lineage(db, root: str) -> list[dict]:
    out, todo = [], [root]
    while todo:
        sid = todo.pop(0)
        row = db.fetchone("SELECT * FROM scenarios WHERE id = ?", (sid,))
        if row is None:
            continue
        out.append(row)
        todo.extend(_children(db, sid))
    return out


def _mirror_path(db, project_id: str, root: str) -> Path | None:
    p = db.fetchone("SELECT dir FROM projects WHERE id = ?", (project_id,))
    if p is None or not p.get("dir"):
        return None
    return Path(p["dir"]) / "scenarios" / f"{root}.json"


def write_mirror(db, workspace, sid: str) -> Path | None:
    """Rewrite the mirror file of ``sid``'s lineage (atomic)."""
    root = root_of(db, sid)
    rows = _lineage(db, root)
    if not rows:
        return None
    path = _mirror_path(db, rows[0]["project_id"], root)
    if path is None:
        return None
    revisions = []
    for r in rows:
        revisions.append({**_doc(r), "id": r["id"], "revision": int(r.get("revision") or 1),
                          "parent_id": r.get("parent_id"), "content_hash": r["content_hash"],
                          "status": r.get("status") or "draft", "archived": bool(r.get("archived")),
                          "project_id": r["project_id"], "created_utc": r.get("created_utc"),
                          "updated_utc": r.get("updated_utc")})
    try:
        write_json_atomic(path, {"schema": 1, "revisions": revisions, "archived": all(x["archived"] for x in revisions)})
    except OSError:
        log.exception("cannot write the scenario mirror %s", path)
        return None
    return path


def _remove_mirror(db, workspace, project_id: str, root: str) -> None:
    path = _mirror_path(db, project_id, root)
    if path is not None:
        try:
            path.unlink()
        except OSError:
            pass


@reindex_hook("scenarios", order=30)
def reindex_scenarios(db, workspace) -> dict:
    """Rebuild the ``scenarios`` table from ``projects/*/scenarios/*.json``."""
    n = 0
    for p in db.fetchall("SELECT id, dir FROM projects"):
        d = Path(p["dir"]) / "scenarios"
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.json")):
            obj = read_json(f)
            if not isinstance(obj, dict):
                continue
            for rev in obj.get("revisions") or []:
                if not isinstance(rev, dict) or not rev.get("id"):
                    continue
                try:
                    doc = doc_dict({k: v for k, v in rev.items() if k in ("name", "notes", "tags", "anchor_run_id",
                                                                         "edits", "regions", "costs", "options")})
                except Exception:
                    log.warning("reindex: skipping an unreadable revision in %s", f)
                    continue
                db.insert("scenarios", {
                    "id": rev["id"], "project_id": rev.get("project_id") or p["id"],
                    "revision": int(rev.get("revision") or 1), "parent_id": rev.get("parent_id"),
                    "name": doc["name"], "doc_json": dbmod.dumps(doc), "content_hash": content_hash(doc),
                    "tags_json": dbmod.dumps(list(doc.get("tags") or [])), "status": rev.get("status") or "draft",
                    "anchor_run_id": doc.get("anchor_run_id"), "archived": 1 if rev.get("archived") else 0,
                    "created_utc": rev.get("created_utc"), "updated_utc": rev.get("updated_utc")}, replace=True)
                n += 1
    return {"scenarios": n}


# ---------------------------------------------------------------------------
# ladders
# ---------------------------------------------------------------------------

def ladder_docs(row: dict, edit_index: int, amounts: list[float]) -> tuple[str, list[dict]]:
    """``(ladder id, docs)``: one copy of the scenario per amount with edit ``edit_index`` set to it."""
    doc = _doc(row)
    edits = doc.get("edits") or []
    if not 0 <= edit_index < len(edits):
        raise ApiError("validation", f"edit_index {edit_index} is out of range (the scenario has {len(edits)} edits)",
                       detail={"errors": [{"path": "edit_index", "message": "out of range", "code": "range"}]})
    lid = new_id("sc")[3:]
    out = []
    for a in amounts:
        d = json.loads(json.dumps(doc))
        d["edits"][edit_index]["amount"] = float(a)
        d["name"] = f"{doc.get('name')} · {edits[edit_index].get('lever')} {float(a):g}"
        d["tags"] = sorted(set((doc.get("tags") or []) + [f"ladder:{lid}"]))
        out.append(d)
    return lid, out


# ---------------------------------------------------------------------------
# promote to config
# ---------------------------------------------------------------------------

def _scenario_names(entry: dict, joint: bool) -> list[str]:
    """The names core gives the promoted entry (``specs_from_config``; U+2212 for decreases)."""
    from types import SimpleNamespace

    from sparc.core.scenarios import specs_from_config

    key = "joint_scenarios" if joint else "scenarios"
    return [s.name for s in specs_from_config(SimpleNamespace(raw={key: [entry]}))]


def promote(db, sid: str, *, apply: bool = False) -> dict:
    """``POST /api/scenarios/{sid}/promote``: a ``scenarios`` ladder (one lever) or a ``joint_scenarios`` package.

    Eligible only for whole-city ``add`` edits on actionable levers.  ``apply`` writes a new project config
    version (the run is never touched)."""
    import importlib

    row = get_row(db, sid)
    doc = _doc(row)
    project = project_row(db, row["project_id"])
    try:
        cs = importlib.import_module("sparc.studio.projects.config_service")
    except ModuleNotFoundError:
        return {"eligible": False, "reason": "the project config service is not installed", "yaml_diff": None,
                "names": [], "version": None}
    text = cs.read_text(project)
    try:
        parsed = cs.parse_yaml(text) if text.strip() else {}
    except ApiError as exc:
        return {"eligible": False, "reason": f"the project config does not parse: {exc.message}", "yaml_diff": None,
                "names": [], "version": None}
    raw = cs.core_block(parsed)
    actionable = raw.get("actionable") or {}
    edits = doc.get("edits") or []
    reason = None
    if not edits:
        reason = "the scenario has no edits"
    for i, e in enumerate(edits):
        w = e.get("where") or {"kind": "all"}
        if e.get("mode") != "add":
            reason = f"edit {i + 1} uses {e.get('mode')}; only add edits can be written to the config"
        elif w.get("kind") != "all":
            reason = f"edit {i + 1} is not city-wide; only whole-city edits can be written to the config"
        elif e.get("lever") not in actionable:
            reason = f"edit {i + 1} changes {e.get('lever')}, which is not an actionable lever of the project config"
        elif not e.get("amount"):
            reason = f"edit {i + 1} has no amount"
        if reason:
            break
    if reason is None and len({e["lever"] for e in edits}) != len(edits):
        reason = "the scenario edits one lever twice"
    if reason:
        return {"eligible": False, "reason": reason, "yaml_diff": None, "names": [], "version": None}
    new = json.loads(json.dumps(raw))
    if len(edits) == 1:
        e = edits[0]
        a = float(e["amount"])
        entry = {"name": doc["name"], "variable": e["lever"], "direction": "increase" if a > 0 else "decrease",
                 "increments": [abs(a)]}
        new.setdefault("scenarios", [])
        new["scenarios"] = list(new.get("scenarios") or []) + [entry]
        section, joint = "scenarios", False
    else:
        entry = {"name": doc["name"], "interventions": [
            {"variable": e["lever"], "direction": "increase" if float(e["amount"]) > 0 else "decrease",
             "increment": abs(float(e["amount"]))} for e in edits]}
        new["joint_scenarios"] = list(new.get("joint_scenarios") or []) + [entry]
        section, joint = "joint_scenarios", True
    names = _scenario_names(entry, joint)
    others = {k: v for k, v in parsed.items() if k != "core"} if isinstance(parsed, dict) and "core" in parsed else {}
    new_text = cs.dump_config(new, others)
    diff = cs.yaml_diff(text, new_text)
    version = None
    if apply:
        out = cs.patch_section(db, project, section, new[section], note=f"promoted scenario {doc['name']!r} ({sid})")
        version = out.get("version")
    return {"eligible": True, "reason": None, "yaml_diff": diff, "names": names, "version": version}


# ---------------------------------------------------------------------------
# design CSVs
# ---------------------------------------------------------------------------

def design_csv(db, ctx, sid: str, *, reader=None, project_dir=None) -> str:
    """``id,lever,change``: the realised per-cell edit of the scenario on the run of ``ctx`` (its newest exact
    result there, else the compiler's prediction)."""
    from sparc.studio.engine import store
    from sparc.studio.engine.compile import compile_scenario

    row = get_row(db, sid)
    res = db.fetchone("SELECT * FROM results WHERE scenario_id = ? AND run_id = ? AND kind = 'exact' "
                      "ORDER BY created_utc DESC", (sid, ctx.run_id))
    ids = np.asarray(ctx.grid.ids)
    changes: dict[str, np.ndarray] = {}
    if res is not None and res.get("dir") and (Path(res["dir"]) / "cells.parquet").exists():
        df = store.read_cells(Path(res["dir"]))
        for c in df.columns:
            if c.startswith("realized_"):
                changes[c[len("realized_"):]] = df[c].to_numpy(np.float64)
    else:
        comp = compile_scenario(ctx, _doc(row), db=db, reader=reader, project_dir=project_dir)
        changes = comp.realized
    buf = io.StringIO()
    buf.write("id,lever,change\n")
    for lever, v in changes.items():
        for i in np.flatnonzero(np.abs(v) > 0):
            buf.write(f"{ids[i]},{lever},{float(v[i]):.6g}\n")
    return buf.getvalue()


def import_design(db, ctx, raw: bytes) -> dict:
    """``POST /api/runs/{rid}/designs/import``: per-lever edit blobs (Int32 idx + Float32 change)."""
    import pandas as pd

    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise ApiError("validation", f"the design CSV does not parse: {exc}",
                       detail={"errors": [{"path": "body", "message": "not a CSV", "code": "csv"}]})
    cols = {c.strip().lower(): c for c in df.columns}
    if "id" not in cols or "lever" not in cols or not ({"change", "value"} & set(cols)):
        raise ApiError("validation", "a design CSV has columns id,lever,change (or id,lever,value)",
                       detail={"errors": [{"path": "body", "message": "missing columns", "code": "columns"}]})
    mode = "change" if "change" in cols else "value"
    data = ctx.data
    preds = list(ctx.cfg.predictors) if ctx.cfg is not None else []
    ids = np.asarray(ctx.grid.ids).astype(str)
    pos = {v: i for i, v in enumerate(ids)}
    unknown: list = []
    per_lever: dict[str, dict[int, float]] = {}
    raw_vals = df[cols[mode]]
    vals = pd.to_numeric(raw_vals, errors="coerce")
    bad = [i for i in np.flatnonzero(vals.isna().to_numpy() & raw_vals.notna().to_numpy())]
    if bad:                                  # text in a number column ("abc"); empty cells and NA stay skipped
        raise ApiError("validation", f"the {cols[mode]} column of the design CSV must be numeric: "
                       f"{len(bad)} row(s) are not numbers",
                       detail={"errors": [{"path": f"{cols[mode]}[{int(i) + 2}]",
                                           "message": f"row {int(i) + 2}: {str(raw_vals.iloc[i])[:40]!r} is not a "
                                                      f"number", "code": "number"} for i in bad[:20]]})
    for rid, lever, v in zip(df[cols["id"]], df[cols["lever"]].astype(str), vals.astype(float)):
        key = str(rid)
        r = pos.get(key)
        if r is None:
            try:
                r = pos.get(str(int(float(key))))
            except (ValueError, OverflowError):     # "ghost", "inf", "nan": an id this run does not have
                r = None
        if r is None:
            if len(unknown) < 1000:
                u = rid.item() if hasattr(rid, "item") else rid
                unknown.append(u if not isinstance(u, float) or np.isfinite(u) else str(u))   # "inf", "nan"
            continue
        if lever not in preds:
            raise ApiError("validation", f"{lever!r} is not a predictor of this run",
                           detail={"errors": [{"path": "lever", "message": f"unknown lever {lever}",
                                               "code": "unknown_lever"}]})
        if not np.isfinite(v):
            continue
        change = v
        if mode == "value":
            if data is None:
                raise ApiError("output_missing", "value designs need the run's data to compute the change",
                               detail={"output": "data", "produced_by": "stage:S0", "expected_path": None})
            change = v - float(data.frame[lever].to_numpy(float)[r])
        per_lever.setdefault(lever, {})[r] = per_lever.get(lever, {}).get(r, 0.0) + change
    blobs = {}
    bdir = Path(ctx.studio_dir) / "blobs"
    bdir.mkdir(parents=True, exist_ok=True)
    for lever, cells in per_lever.items():
        idx = np.fromiter(cells.keys(), dtype="<i4", count=len(cells))
        val = np.fromiter(cells.values(), dtype="<f4", count=len(cells))
        bid = new_id("blob")
        body = idx.tobytes() + val.tobytes()
        from sparc.core import runio

        runio.write_bytes_atomic(bdir / f"{bid}.bin", body)
        db.insert("blobs", {"id": bid, "run_id": ctx.run_id, "kind": "edit", "path": str(bdir / f"{bid}.bin"),
                            "bytes": len(body), "created_utc": utc_now()})
        blobs[lever] = bid
    return {"blobs": blobs, "n_rows": int(len(df)), "unknown_ids": unknown, "levers": sorted(per_lever), "mode": mode}


# ---------------------------------------------------------------------------
# configured scenarios (S5)
# ---------------------------------------------------------------------------

def configured_doc(cfg, name: str) -> dict | None:
    """The add-mode ScenarioDoc equivalent of configured scenario ``name`` ("Clone to edit")."""
    from sparc.core.scenarios import specs_from_config

    for spec in specs_from_config(cfg):
        if spec.name == name:
            return doc_dict({"name": name, "tags": ["configured"],
                             "edits": [{"lever": iv.variable, "mode": "add", "amount": float(iv.amount),
                                        "where": {"kind": "all"}} for iv in spec.interventions]})
    return None


def configured_list(ctx) -> list[dict]:
    """The run's configured scenarios (api.md §7.5 ``configured``)."""
    from sparc.studio.runs.common import fnum, likely

    unit = ctx.units.get("target", "°F")
    summaries = {str(s.get("name")): s for s in (ctx.manifest.get("scenarios") or []) if isinstance(s, dict)}
    det = ctx.scenario_detail()
    out = []
    for sc in ctx.configured_scenarios():
        s = summaries.get(sc["name"]) or {}
        doc = configured_doc(ctx.cfg, sc["name"]) if ctx.cfg is not None else None
        if doc is None:
            doc = doc_dict({"name": sc["name"], "edits": []})
        cl = s.get("causal_linear")
        est = fnum(s.get("mean_delta"))
        if est is None:
            df = ctx.parquet("scenario_deltas.parquet")
            est = float(df[sc["name"]].mean()) if df is not None and sc["name"] in df else 0.0
        out.append({"slug": sc["slug"], "name": sc["name"],
                    "city": likely(est, fnum(s.get("mean_delta_se")), unit, what="the city") or likely(0.0, None, unit),
                    "p10": fnum(s.get("p10_delta")), "p90": fnum(s.get("p90_delta")),
                    "frac_extrapolated": fnum(s.get("frac_extrapolated")),
                    "mean_realized": {k: float(v) for k, v in (s.get("mean_realized") or {}).items() if fnum(v) is not None},
                    "causal_linear": cl if isinstance(cl, dict) and {"delta", "lo", "hi", "model_within"} <= set(cl)
                    else None,
                    "has_folds": bool(det and sc["name"] in det["folds"]), "layer_key": f"sc:{sc['slug']}",
                    "doc": doc})
    return out
