"""The ``exports`` index (api.md §10, SPEC §6.7).

``POST /api/exports`` (:func:`create_export`) checks the params of ``export.<kind>`` with a fresh
``export_id`` injected (clients never send one), inserts the row (``status = running``) and its record
``projects/<slug>/exports/<export_id>/export.json``, then queues the job by name - the pack kinds of the
engine item included (SPEC §19 M3: the row exists before ``pack_on_finish`` completes it).  Each kind's
``on_finish`` completes the row; :func:`reconcile` repairs one whose hook was missed (a server restart while
the job ran) from the job's final status and result, and keeps ``export.json`` in step, so
``sparc studio --reindex`` rebuilds the table from those records (:func:`_reindex`).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from sparc.studio import db as dbmod
from sparc.studio.errors import ApiError, validation_error
from sparc.studio.workspace import new_id, read_json, utc_now, write_json_atomic

log = logging.getLogger("sparc.studio.exports")

__all__ = ["KINDS", "RUN_KINDS", "REF_PARAM", "export_out", "get_row", "list_exports", "create_export", "reconcile",
           "complete", "delete_export", "export_dir", "write_record", "unknown_ids"]

KINDS = ("bundle", "gis", "page", "report", "findings", "decision_pack", "plan_pack", "compare_pack")
#: kinds whose params name the run they export
RUN_KINDS = ("bundle", "gis", "page", "report")
#: pack kinds: the param naming the object packed (the export's ``ref``)
REF_PARAM = {"decision_pack": "result_id", "plan_pack": "plan_id", "compare_pack": "comparison_id"}
RECORD = "export.json"
_FINAL = ("succeeded", "failed", "cancelled", "interrupted")


def get_row(db, eid: str) -> dict:
    row = db.fetchone("SELECT * FROM exports WHERE id = ?", (eid,))
    if row is None:
        raise ApiError("not_found", f"no export {eid!r}")
    return row


def export_out(row: dict) -> dict:
    """An ``exports`` row as the api.md ``Export``."""
    opts = dbmod.loads(row.get("options_json"), {}) or {}
    return {"id": row["id"], "project_id": row.get("project_id") or "", "run_id": row.get("run_id"),
            "kind": row.get("kind") or "", "ref": row.get("ref"), "options": opts, "job_id": row.get("job_id") or "",
            "status": row.get("status") or "running", "path": row.get("path"), "bytes": row.get("bytes"),
            "draft": bool(opts.get("draft")), "created_utc": row.get("created_utc") or ""}


def export_dir(db, row: dict) -> Path | None:
    p = db.fetchone("SELECT dir FROM projects WHERE id = ?", (row.get("project_id"),))
    return Path(p["dir"]) / "exports" / row["id"] if p and p.get("dir") else None


def write_record(db, eid: str) -> None:
    """``exports/<eid>/export.json``: the row, for the reindex."""
    row = db.fetchone("SELECT * FROM exports WHERE id = ?", (eid,))
    if row is None:
        return
    d = export_dir(db, row)
    if d is None:
        return
    doc = {"schema": 1, **{k: row.get(k) for k in ("id", "project_id", "run_id", "kind", "ref", "job_id", "path",
                                                   "bytes", "status", "created_utc")},
           "options": dbmod.loads(row.get("options_json"), {}) or {}}
    try:
        write_json_atomic(d / RECORD, doc)
    except OSError:
        log.exception("could not write the record of export %s", eid)


def complete(db, eid: str, job: dict, result: dict | None) -> None:
    """Finish the row from a job's end: ``ready`` with the file, else ``failed``."""
    row = db.fetchone("SELECT options_json FROM exports WHERE id = ?", (eid,))
    if row is None:
        return
    if job.get("status") == "succeeded" and result and result.get("path"):
        opts = dbmod.loads(row.get("options_json"), {}) or {}
        if "draft" in result:
            opts["draft"] = bool(result.get("draft"))
        db.update("exports", {"id": eid}, {"status": "ready", "path": str(result["path"]),
                                           "bytes": int(result.get("bytes") or 0), "options_json": dbmod.dumps(opts)})
    else:
        db.update("exports", {"id": eid}, {"status": "failed"})
    write_record(db, eid)


def reconcile(db, row: dict) -> dict:
    """The row, repaired when its job ended without the row being completed (or the record lags behind)."""
    if row.get("status") == "running" and row.get("job_id"):
        j = db.fetchone("SELECT status, result_json FROM jobs WHERE id = ?", (row["job_id"],))
        if j is None or j["status"] in _FINAL:
            complete(db, row["id"], {"status": (j or {}).get("status") or "failed"},
                     dbmod.loads((j or {}).get("result_json")))
            return db.fetchone("SELECT * FROM exports WHERE id = ?", (row["id"],)) or row
    d = export_dir(db, row)
    if d is not None and d.is_dir():
        rec = read_json(d / RECORD) or {}
        if rec.get("status") != row.get("status") or rec.get("path") != row.get("path"):
            write_record(db, row["id"])
    return row


def list_exports(db, project_id: str) -> list[dict]:
    rows = db.fetchall("SELECT * FROM exports WHERE project_id = ? ORDER BY created_utc DESC, id DESC", (project_id,))
    return [export_out(reconcile(db, r)) for r in rows]


def _run_of(db, kind: str, params: dict) -> str | None:
    """The run an export belongs to (its params' ``run_id``; for packs, the packed object's run)."""
    if params.get("run_id"):
        return params["run_id"]
    if kind == "decision_pack":
        rid = params.get("result_id") or ""
        if rid.startswith("sc_"):
            r = db.fetchone("SELECT anchor_run_id AS run_id FROM scenarios WHERE id = ? ORDER BY revision DESC",
                            (rid,))
        else:
            r = db.fetchone("SELECT run_id FROM results WHERE id = ?", (rid,))
        if r is None:
            raise ApiError("not_found", f"no result or scenario {rid!r}")
        return r.get("run_id")
    if kind == "plan_pack":
        r = db.fetchone("SELECT run_id FROM plans WHERE id = ?", (params.get("plan_id"),))
        if r is None:
            raise ApiError("not_found", f"no plan {params.get('plan_id')!r}")
        return r.get("run_id")
    if kind == "compare_pack":
        r = db.fetchone("SELECT run_id FROM comparisons WHERE id = ?", (params.get("comparison_id"),))
        if r is None:
            raise ApiError("not_found", f"no comparison {params.get('comparison_id')!r}")
        return r.get("run_id")
    return None


async def create_export(sctx, kind: str, project_id: str, params: dict | None) -> dict:
    """``POST /api/exports``: the row first, then ``export.<kind>`` with the server-filled ``export_id``."""
    import asyncio

    from sparc.studio.jobs.kinds import get_kind

    if kind not in KINDS:
        raise ApiError("unknown_kind", f"unknown export kind {kind!r}", detail={"kinds": list(KINDS)})
    k = get_kind(f"export.{kind}")
    if k is None:
        raise ApiError("unknown_kind", f"export.{kind} is not available in this build", detail={"kind": kind})
    db = sctx.db
    project = db.fetchone("SELECT id, dir FROM projects WHERE id = ?", (project_id,))
    if project is None:
        raise ApiError("not_found", f"no project {project_id!r}")
    params = dict(params or {})
    if "export_id" in params:
        raise validation_error([{"path": "params.export_id", "code": "server_filled",
                                 "message": "export_id is filled by the server"}], "export_id is filled by the server")
    if kind == "findings":
        params.setdefault("project_id", project_id)
        if params["project_id"] != project_id:
            raise validation_error([{"path": "params.project_id", "code": "mismatch",
                                     "message": "the findings export belongs to the request's project"}],
                                   "project ids differ")
    eid = new_id("ex")
    full = {**params, "export_id": eid}
    p = k.validate_params(full)                       # 422 validation with params.<field> paths
    full = k.dump_params(p)
    run_id = _run_of(db, kind, full)
    if run_id:
        run = db.fetchone("SELECT id, project_id FROM runs WHERE id = ?", (run_id,))
        if run is None:
            raise ApiError("not_found", f"no run {run_id!r}")
        if run.get("project_id") and run["project_id"] != project_id:
            raise validation_error([{"path": "params.run_id", "code": "mismatch",
                                     "message": f"run {run_id} belongs to another project"}],
                                   "the run belongs to another project")
    await asyncio.to_thread(_precheck, sctx, kind, full, run_id, project_id)
    ref = full.get(REF_PARAM[kind]) if kind in REF_PARAM else None
    options = {key: v for key, v in full.items() if key not in ("export_id",)}
    row = {"id": eid, "project_id": project_id, "run_id": run_id, "kind": kind, "ref": ref,
           "options_json": dbmod.dumps(options), "job_id": None, "path": None, "bytes": None, "status": "running",
           "created_utc": utc_now()}
    out_dir = Path(project["dir"]) / "exports" / eid
    out_dir.mkdir(parents=True, exist_ok=True)
    await db.ainsert("exports", row)
    try:
        job = await sctx.jobs.submit(f"export.{kind}", full, project_id=project_id, run_id=run_id,
                                     label=f"{_LABELS.get(kind, kind)} export")
    except BaseException:
        await db.aexecute("DELETE FROM exports WHERE id = ?", (eid,))
        shutil.rmtree(out_dir, ignore_errors=True)
        raise
    await db.aupdate("exports", {"id": eid}, {"job_id": job["id"]})
    await asyncio.to_thread(write_record, db, eid)
    return {"export": export_out(get_row(db, eid)), "job": job}


_LABELS = {"bundle": "Run bundle", "gis": "GIS pack", "page": "Results page", "report": "Project report",
           "findings": "Findings", "decision_pack": "Decision pack", "plan_pack": "Plan pack",
           "compare_pack": "Compare pack"}


def unknown_ids(db, table: str, ids, project_id: str) -> list[str]:
    """The ids of ``ids`` that are not ``table`` rows of the project (results: of the project's runs)."""
    bad = []
    for i in ids or []:
        if table == "results":
            r = db.fetchone("SELECT r.id FROM results r JOIN runs u ON u.id = r.run_id WHERE r.id = ? AND "
                            "(u.project_id = ? OR u.project_id IS NULL)", (i, project_id))
        else:
            r = db.fetchone(f"SELECT id FROM {table} WHERE id = ? AND project_id = ?", (i, project_id))
        if r is None:
            bad.append(i)
    return bad


def _precheck(sctx, kind: str, params: dict, run_id: str | None, project_id: str) -> None:
    """What the job would only find out later: unknown outputs, layers, results, plans or findings, a GIS pack
    of a run without a CRS, a placebo study without ``placebo.json``."""
    db = sctx.db
    for key, table in (("result_ids", "results"), ("plan_ids", "plans"), ("finding_ids", "findings"),
                       ("ids", "findings")):
        bad = unknown_ids(db, table, params.get(key), project_id)
        if bad:
            raise validation_error([{"path": f"params.{key}", "code": "unknown_id",
                                     "message": f"not in this project: {bad}"}], f"unknown {table}")
    reader = sctx.services.get("reader")
    if reader is None or not run_id:
        return
    ctx = reader.get(run_id)
    if kind == "bundle" and params.get("outputs"):
        from sparc.studio.exports.bundle import check_outputs

        bad = check_outputs(ctx.cfg_raw, params["outputs"])
        if bad:
            raise validation_error([{"path": "params.outputs", "code": "unknown_output",
                                     "message": f"not catalogued run outputs: {bad}"}], "unknown outputs")
    if kind == "gis":
        g = ctx.grid
        if g is None or not g.crs:
            raise ApiError("needs_crs", "a GIS pack needs a run with a CRS (set data.crs in the config)",
                           detail={"errors": [{"path": "params.run_id", "message": "no CRS", "code": "needs_crs"}]})
        if params.get("layers"):
            from sparc.studio.runs.layers import layer_defs

            defs = layer_defs(ctx)
            bad = [k for k in params["layers"] if k not in defs]
            if bad:
                raise validation_error([{"path": "params.layers", "code": "unknown_layer",
                                         "message": f"unknown layers: {bad}"}], "unknown layers")
    if kind == "page" and params.get("placebo_study"):
        row = sctx.db.fetchone("SELECT out_dir FROM studies WHERE id = ?", (params["placebo_study"],))
        if row is None or not (Path(row["out_dir"]) / "placebo.json").is_file():
            raise validation_error([{"path": "params.placebo_study", "code": "study",
                                     "message": f"{params['placebo_study']} has no placebo.json"}],
                                   "not a finished placebo study")


def delete_export(db, eid: str) -> None:
    row = get_row(db, eid)
    if row.get("job_id"):
        j = db.fetchone("SELECT status FROM jobs WHERE id = ?", (row["job_id"],))
        if j is not None and j["status"] not in _FINAL:
            raise ApiError("active", "the export is still being built; cancel its job first",
                           detail={"job_id": row["job_id"]})
    d = export_dir(db, row)
    db.execute("DELETE FROM exports WHERE id = ?", (eid,))
    if d is not None:
        shutil.rmtree(d, ignore_errors=True)


def _reindex(db, workspace) -> dict:
    """Rebuild ``exports`` from ``projects/*/exports/*/export.json`` (finished files found without a record
    are left out: nothing says which project object they packed)."""
    n = 0
    root = Path(workspace.projects_dir)
    for rec in sorted(root.glob(f"*/exports/*/{RECORD}")) if root.is_dir() else []:
        doc = read_json(rec)
        if not isinstance(doc, dict) or not doc.get("id"):
            continue
        row = {k: doc.get(k) for k in ("id", "project_id", "run_id", "kind", "ref", "job_id", "path", "bytes",
                                       "status", "created_utc")}
        row["options_json"] = dbmod.dumps(doc.get("options") or {})
        if row.get("status") == "running":
            files = [p for p in rec.parent.iterdir() if p.is_file() and p.name != RECORD]
            if row.get("job_id"):
                j = db.fetchone("SELECT status, result_json FROM jobs WHERE id = ?", (row["job_id"],))
                res = dbmod.loads((j or {}).get("result_json")) or {}
                if j is not None and j["status"] == "succeeded" and res.get("path"):
                    row.update(status="ready", path=res["path"], bytes=res.get("bytes"))
                elif j is not None and j["status"] in _FINAL:
                    row["status"] = "failed"
            elif files:
                row.update(status="ready", path=str(files[0]), bytes=files[0].stat().st_size)
        db.insert("exports", row, replace=True)
        n += 1
    return {"exports": n}


try:
    dbmod.reindex_hook("exports", order=40)(_reindex)
except Exception:  # pragma: no cover
    log.exception("cannot register the exports reindex hook")
