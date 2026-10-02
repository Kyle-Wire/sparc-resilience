"""Project records: the ``projects`` row, the ``project.json`` mirror, the api.md ``Project`` shape, the reindex hook.

A project is a folder ``<ws>/projects/<slug>/`` (SPEC §4.2) holding
``project.json`` (the durable record), ``config.yml`` and its inputs.  The
``projects`` row is an index of ``project.json``: :func:`register_dir`
(re)builds it, at ``--reindex``, at server start (folders created while the
server was down, e.g. by a features job) and after jobs that create
projects.  ``project.json`` carries ``deleted_utc`` once a project is
deleted without its files, so it is never resurrected.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any

from sparc.studio import db as dbmod
from sparc.studio.errors import ApiError
from sparc.studio.projects.config_service import (
    CONFIG_NAME,
    core_block,
    parse_yaml,
    read_text,
    sync_versions,
)
from sparc.studio.schemas.common import ACTIVE_STATUSES
from sparc.studio.workspace import new_id, read_json, slugify, utc_now, write_json_atomic

log = logging.getLogger("sparc.studio.projects")

__all__ = ["PROJECT_JSON", "get_row", "rows", "claim_dir", "write_record", "read_record", "register_dir",
           "create_project", "project_out", "run_summary", "project_runs", "active_jobs", "patch_project",
           "touch", "delete_project", "scan_projects", "project_raw"]

PROJECT_JSON = "project.json"
RUN_MODES = ("fast", "coarse", "full", "custom")


# ---------------------------------------------------------------------------
# rows and records
# ---------------------------------------------------------------------------

def get_row(db, pid: str) -> dict:
    row = db.fetchone("SELECT * FROM projects WHERE id = ?", (pid,))
    if row is None:
        raise ApiError("not_found", f"no project {pid!r}")
    return row


def rows(db, *, archived: bool | None = False) -> list[dict]:
    if archived is None:
        return db.fetchall("SELECT * FROM projects ORDER BY updated_utc DESC, created_utc DESC")
    return db.fetchall("SELECT * FROM projects WHERE archived = ? ORDER BY updated_utc DESC, created_utc DESC",
                       (1 if archived else 0,))


def _meta(row: dict) -> dict:
    return dbmod.loads(row.get("meta_json"), {}) or {}


def read_record(project_dir: str | os.PathLike) -> dict | None:
    rec = read_json(Path(project_dir) / PROJECT_JSON)
    return rec if isinstance(rec, dict) and rec.get("id") else None


def write_record(project_dir: str | os.PathLike, rec: dict) -> None:
    write_json_atomic(Path(project_dir) / PROJECT_JSON, {"schema": 1, **rec})


def _record_of(row: dict) -> dict:
    m = _meta(row)
    return {"id": row["id"], "slug": row["slug"], "name": row["name"], "template": row.get("template"),
            "demo": bool(row.get("demo")), "created_utc": row.get("created_utc"), "updated_utc": row.get("updated_utc"),
            "archived": bool(row.get("archived")), "active_run_id": row.get("active_run_id"),
            "report": m.get("report") or {}, "headline_scenario": m.get("headline_scenario"),
            "cost_model": m.get("cost_model") or {}, "source": m.get("source")}


def claim_dir(workspace, name: str, taken: set[str] | None = None) -> tuple[str, Path]:
    """A fresh project folder for ``name``: ``(slug, dir)`` with a slug no project row or folder uses."""
    workspace.projects_dir.mkdir(parents=True, exist_ok=True)
    used = set(taken or ()) | {p.name for p in workspace.projects_dir.iterdir()}
    while True:
        slug = slugify(name, used)
        d = workspace.project_dir(slug)
        try:
            d.mkdir(parents=False, exist_ok=False)
            return slug, d
        except FileExistsError:
            used.add(slug)


def project_raw(project: dict) -> dict:
    """The project's raw ``core:`` block ({} when the config is missing or broken)."""
    text = read_text(project)
    try:
        return core_block(parse_yaml(text)) if text.strip() else {}
    except ApiError:
        return {}


# ---------------------------------------------------------------------------
# create / register
# ---------------------------------------------------------------------------

def create_project(db, workspace, *, name: str, template: str | None, demo: bool = False,
                   slug: str | None = None, project_dir: Path | None = None, meta: dict | None = None,
                   note: str = "created") -> dict:
    """Insert the row and write ``project.json`` for a folder whose ``config.yml`` already exists."""
    if project_dir is None:
        slug, project_dir = claim_dir(workspace, name, {r["slug"] for r in db.fetchall("SELECT slug FROM projects")})
    slug = slug or Path(project_dir).name
    if db.fetchone("SELECT id FROM projects WHERE slug = ?", (slug,)):
        raise ApiError("conflict", f"a project with slug {slug!r} exists", detail={"slug": slug})
    now = utc_now()
    row = {"id": new_id("project"), "slug": slug, "name": name, "dir": str(Path(project_dir).resolve()),
           "config_path": str((Path(project_dir) / CONFIG_NAME).resolve()), "template": template,
           "demo": 1 if demo else 0, "active_run_id": None, "archived": 0, "meta_json": dbmod.dumps(meta or {}),
           "created_utc": now, "updated_utc": now}
    write_record(project_dir, _record_of(row))
    db.insert("projects", row)
    sync_versions(db, row, note=note)
    return row


def register_dir(db, project_dir: str | os.PathLike) -> dict | None:
    """Insert the ``projects`` row of a folder from its ``project.json`` when it has none (an existing row is
    live state and keeps its values; only ``dir``/``config_path`` follow the folder).  Idempotent; None when
    there is no record or it was deleted."""
    pdir = Path(project_dir).resolve()
    rec = read_record(pdir)
    if rec is None or rec.get("deleted_utc"):
        return None
    meta = {"report": rec.get("report") or {}, "headline_scenario": rec.get("headline_scenario"),
            "cost_model": rec.get("cost_model") or {}}
    if rec.get("source"):
        meta["source"] = rec["source"]
    row = {"id": rec["id"], "slug": rec.get("slug") or pdir.name, "name": rec.get("name") or pdir.name,
           "dir": str(pdir), "config_path": str(pdir / CONFIG_NAME), "template": rec.get("template"),
           "demo": 1 if rec.get("demo") else 0, "active_run_id": rec.get("active_run_id"),
           "archived": 1 if rec.get("archived") else 0, "meta_json": dbmod.dumps(meta),
           "created_utc": rec.get("created_utc") or utc_now(), "updated_utc": rec.get("updated_utc") or utc_now()}
    existing = db.fetchone("SELECT id FROM projects WHERE id = ?", (row["id"],))
    clash = db.fetchone("SELECT id FROM projects WHERE slug = ? AND id != ?", (row["slug"], row["id"]))
    if clash is not None:
        log.warning("project folder %s: slug %s is taken by %s; not registered", pdir, row["slug"], clash["id"])
        return None
    if existing is None:
        db.insert("projects", row)
    else:                                      # the row is live state: only follow a moved folder
        db.update("projects", {"id": row["id"]}, {"dir": row["dir"], "config_path": row["config_path"]})
    sync_versions(db, row)
    return db.fetchone("SELECT * FROM projects WHERE id = ?", (row["id"],))


def scan_projects(db, workspace) -> dict:
    """Register every project folder of the workspace (``--reindex`` and server start)."""
    n = 0
    pdir = Path(workspace.projects_dir)
    if pdir.is_dir():
        for d in sorted(p for p in pdir.iterdir() if (p / PROJECT_JSON).is_file()):
            try:
                if register_dir(db, d) is not None:
                    n += 1
            except Exception:                      # one bad folder never stops the scan
                log.exception("project scan: skipping %s", d)
    return {"projects": n}


# ---------------------------------------------------------------------------
# wire shapes
# ---------------------------------------------------------------------------

def run_summary(row: dict, studies: list[str] | None = None) -> dict:
    """A ``runs`` row as api.md ``RunSummary`` (the same values ``GET /api/runs`` gives: the registry keeps
    ``duration_s`` and ``n_scenarios`` in ``stages_json``; the timestamps are the fallback)."""
    from sparc.studio.workspace import parse_utc

    info = dbmod.loads(row.get("stages_json"), {}) or {}
    info = info if isinstance(info, dict) else {}
    mode = str(row.get("mode") or "custom")
    mode = "coarse" if mode.startswith("coarse") else (mode if mode in RUN_MODES else "custom")
    duration = info.get("duration_s")
    if duration is None:
        t0, t1 = parse_utc(row.get("created_utc")), parse_utc(row.get("finished_utc"))
        duration = round(t1 - t0, 3) if t0 is not None and t1 is not None else None
    gd = row.get("git_dirty")
    return {"id": row["id"], "project_id": row.get("project_id"), "label": row.get("label"),
            "origin": row.get("origin") or "studio", "status": row.get("status") or "imported", "mode": mode,
            "coarse_m": row.get("coarse_m"), "created_utc": row.get("created_utc"),
            "finished_utc": row.get("finished_utc"), "duration_s": duration,
            "n_points": row.get("n_points"), "r2": row.get("r2"), "rmse": row.get("rmse"),
            "coverage": row.get("coverage"), "n_scenarios": info.get("n_scenarios", row.get("n_scenarios")),
            "checkpoint_bytes": row.get("checkpoint_bytes"), "has_emulator": bool(row.get("has_emulator")),
            "studies": list(studies or []), "git_commit": row.get("git_commit"),
            "git_dirty": None if gd is None else bool(gd), "demo": bool(row.get("demo")),
            "pinned": bool(row.get("pinned")), "parent_run_id": row.get("parent_run_id"),
            "study_id": row.get("study_id"), "last_job_id": row.get("last_job_id")}


def project_runs(db, pid: str) -> list[dict]:
    """The project's runs, newest first, as ``RunSummary``."""
    runs = db.fetchall("SELECT * FROM runs WHERE project_id = ? ORDER BY created_utc DESC", (pid,))
    links: dict[str, list[str]] = {}
    for r in db.fetchall("SELECT l.run_id, l.study_id FROM study_links l JOIN runs r ON r.id = l.run_id "
                         "WHERE r.project_id = ? AND l.attached = 1", (pid,)):
        links.setdefault(r["run_id"], []).append(r["study_id"])
    return [run_summary(r, links.get(r["id"])) for r in runs]


def active_jobs(db, pid: str) -> list[dict]:
    from sparc.studio.jobs.manager import job_out

    marks = ",".join("?" for _ in ACTIVE_STATUSES)
    return [job_out(r) for r in db.fetchall(f"SELECT * FROM jobs WHERE project_id = ? AND status IN ({marks}) "
                                            f"ORDER BY created_utc DESC", (pid, *ACTIVE_STATUSES))]


def project_out(db, row: dict, *, readiness_score: dict | None = None, raw: dict | None = None) -> dict:
    """A ``projects`` row as api.md ``Project``."""
    m = _meta(row)
    raw = project_raw(row) if raw is None else raw
    rep_cfg = raw.get("report") if isinstance(raw.get("report"), dict) else {}
    rep_meta = m.get("report") or {}
    report = {k: (rep_meta.get(k) if rep_meta.get(k) is not None else rep_cfg.get(k))
              for k in ("title", "place", "area")}
    cost = dict(m.get("cost_model") or {})
    for var, spec in (raw.get("actionable") or {}).items() if isinstance(raw.get("actionable"), dict) else []:
        if var not in cost:
            per = (spec or {}).get("cost_per_unit", 1.0) if isinstance(spec, dict) else 1.0
            try:
                cost[str(var)] = {"per_unit": float(per)}
            except (TypeError, ValueError):
                cost[str(var)] = {"per_unit": 1.0}
    n_runs = int(db.fetchval("SELECT COUNT(*) FROM runs WHERE project_id = ?", (row["id"],), default=0) or 0)
    last = db.fetchone("SELECT id, status, created_utc, r2 FROM runs WHERE project_id = ? "
                       "ORDER BY created_utc DESC LIMIT 1", (row["id"],))
    marks = ",".join("?" for _ in ACTIVE_STATUSES)
    n_active = int(db.fetchval(f"SELECT COUNT(*) FROM jobs WHERE project_id = ? AND status IN ({marks})",
                               (row["id"], *ACTIVE_STATUSES), default=0) or 0)
    return {"id": row["id"], "slug": row["slug"], "name": row["name"], "dir": row["dir"],
            "config_path": row["config_path"], "template": row.get("template"), "demo": bool(row.get("demo")),
            "active_run_id": row.get("active_run_id"), "archived": bool(row.get("archived")),
            "created_utc": row.get("created_utc") or "", "updated_utc": row.get("updated_utc") or "",
            "report": report, "headline_scenario": m.get("headline_scenario"), "cost_model": cost,
            "n_runs": n_runs,
            "last_run": ({"id": last["id"], "status": last["status"], "created_utc": last.get("created_utc") or "",
                          "r2": last.get("r2")} if last else None),
            "active_jobs": n_active, "readiness_score": readiness_score or {"done": 0, "total": 0}}


# ---------------------------------------------------------------------------
# patch / delete
# ---------------------------------------------------------------------------

def patch_project(db, pid: str, body: dict) -> dict:
    """``PATCH /api/projects/{pid}``: name, active run, headline scenario, cost model, report, archived."""
    row = get_row(db, pid)
    meta = _meta(row)
    values: dict[str, Any] = {}
    if body.get("name") is not None:
        name = str(body["name"]).strip()
        if not name:
            raise ApiError("validation", "name must not be empty",
                           detail={"errors": [{"path": "name", "message": "empty", "code": "empty"}]})
        values["name"] = name
    if "active_run_id" in body:
        rid = body["active_run_id"]
        if rid is not None:
            run = db.fetchone("SELECT id, project_id FROM runs WHERE id = ?", (rid,))
            if run is None:
                raise ApiError("not_found", f"no run {rid!r}")
            if run.get("project_id") not in (None, pid):
                raise ApiError("validation", f"run {rid} belongs to another project",
                               detail={"errors": [{"path": "active_run_id", "message": "other project",
                                                   "code": "other_project"}]})
        values["active_run_id"] = rid
    if "archived" in body and body["archived"] is not None:
        values["archived"] = 1 if body["archived"] else 0
    if "headline_scenario" in body:
        meta["headline_scenario"] = body["headline_scenario"]
    if body.get("cost_model") is not None:
        meta["cost_model"] = {str(k): {"per_unit": float((v or {}).get("per_unit", 1.0))}
                              for k, v in dict(body["cost_model"]).items()}
    if body.get("report") is not None:
        rep = dict(meta.get("report") or {})
        for k in ("title", "place", "area"):
            if k in body["report"]:
                rep[k] = body["report"][k]
        meta["report"] = rep
    values["meta_json"] = dbmod.dumps(meta)
    values["updated_utc"] = utc_now()
    db.update("projects", {"id": pid}, values)
    row = get_row(db, pid)
    write_record(row["dir"], {**(read_record(row["dir"]) or {}), **_record_of(row)})
    return row


def touch(db, pid: str) -> None:
    """Record that the project changed now (row and ``project.json``)."""
    row = get_row(db, pid)
    now = utc_now()
    db.update("projects", {"id": pid}, {"updated_utc": now})
    write_record(row["dir"], {**(read_record(row["dir"]) or _record_of(row)), "updated_utc": now})


def delete_project(db, workspace, pid: str, *, files: bool = False) -> None:
    """``DELETE /api/projects/{pid}``: refused while jobs run (``409 active``).  ``files=true`` removes the
    project folder (only inside the workspace) and the rows of the runs and studies that lived in it."""
    row = get_row(db, pid)
    marks = ",".join("?" for _ in ACTIVE_STATUSES)
    n = int(db.fetchval(f"SELECT COUNT(*) FROM jobs WHERE project_id = ? AND status IN ({marks})",
                        (pid, *ACTIVE_STATUSES), default=0) or 0)
    if n:
        raise ApiError("active", f"{n} job(s) of this project are still active", detail={"active_jobs": n})
    pdir = Path(row["dir"]).resolve()
    inside = Path(workspace.projects_dir).resolve() in pdir.parents

    def fn(conn):
        conn.execute("DELETE FROM projects WHERE id = ?", (pid,))
        conn.execute("DELETE FROM config_versions WHERE project_id = ?", (pid,))
        if files and inside:
            prefix = str(pdir) + os.sep
            for table, col in (("runs", "run_dir"), ("studies", "out_dir")):
                conn.execute(f"DELETE FROM {table} WHERE project_id = ? AND substr({col}, 1, ?) = ?",
                             (pid, len(prefix), prefix))
            conn.execute("DELETE FROM study_links WHERE run_id NOT IN (SELECT id FROM runs)")

    db.run(fn)
    if files and inside:
        shutil.rmtree(pdir, ignore_errors=True)
    elif pdir.is_dir():
        rec = read_record(pdir) or _record_of(row)
        write_record(pdir, {**rec, "deleted_utc": utc_now()})
