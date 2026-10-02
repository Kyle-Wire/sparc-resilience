"""The studies index: ``studies`` / ``study_links`` rows, their mirror files, attach / detach, staleness,
summaries, imports of existing study folders and the reindex hook (SPEC §8, §10.6, api.md §9).

**Rows.**  A study is one ``studies`` row: ``kind`` (``placebo``, ``simcheck``, ``multiverse``, ``reproduce``,
``benchmark``), ``target_run_id`` (the run it was launched from; ``None`` for the benchmark),
``out_dir`` (the folder its files go to: ``projects/<slug>/studies/<sid>/`` for Studio studies, the folder
itself for imported ones - core writes that same path into ``uncertainty.json`` ``sources``, which is how
the run hub matches its envelopes to studies), ``status`` (the job statuses, ``succeeded`` for imported
folders), ``summary`` (the numbers the overview chips and status rows read, plus ``headline`` and
``label``) and ``run_fingerprint`` (the target run's checkpoint fingerprint when the study was launched).
``study_links`` attaches a study to runs: an attached study feeds that run's uncertainty report.

**Mirror.**  Every change rewrites ``projects/<slug>/studies/<sid>/study.json`` (the row plus its links,
atomically), so ``sparc studio --reindex`` rebuilds both tables from disk (:func:`_reindex`).

**Stale.**  A study is stale vs a run (``Study.stale_vs``) when that run's checkpoint fingerprint differs
from the one recorded for the study (SPEC §8).

Cross-item contract (SPEC §10.2): :func:`import_study_dir` ``(dir, project_id, kind=None,
target_run_id=None) -> Study``, called by the projects item for the Providence example's study folders.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from sparc.studio import db as dbmod
from sparc.studio.errors import ApiError
from sparc.studio.workspace import new_id, read_json, utc_now, write_json_atomic

log = logging.getLogger("sparc.studio.studies")

__all__ = ["POST_KINDS", "RUN_STUDY_KINDS", "PROJECT_STUDY_KINDS", "STUDY_KINDS", "ATTACHABLE", "ROW_KINDS",
           "ACTIVE_JOB", "get_row", "study_out", "list_studies", "create_study", "update_study", "set_links",
           "attached_runs", "stale_vs", "write_mirror", "publish", "summary_for", "import_study_dir",
           "detect_kind", "delete_study", "effective_status", "project_dir_of", "attached_studies_of"]

POST_KINDS = ("baselines", "planner", "emulator", "uncertainty", "writeup")
RUN_STUDY_KINDS = ("placebo", "simcheck", "multiverse", "reproduce")
PROJECT_STUDY_KINDS = ("benchmark",)
STUDY_KINDS = RUN_STUDY_KINDS + PROJECT_STUDY_KINDS
#: kinds that feed a run's uncertainty report when attached to it
ATTACHABLE = ("placebo", "simcheck", "multiverse")
#: ``GET /runs/{rid}/studies`` rows, in this order
ROW_KINDS = ("baselines", "planner", "emulator", "uncertainty", "writeup", "placebo", "simcheck", "multiverse",
             "reproduce", "literature", "benchmark")
ACTIVE_JOB = ("queued", "blocked", "starting", "running", "cancelling")
MIRROR = "study.json"


# ---------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------

def get_row(db, sid: str) -> dict:
    row = db.fetchone("SELECT * FROM studies WHERE id = ?", (sid,))
    if row is None:
        raise ApiError("not_found", f"no study {sid!r}")
    return row


def project_dir_of(db, project_id: str | None) -> Path | None:
    if not project_id:
        return None
    p = db.fetchone("SELECT dir FROM projects WHERE id = ?", (project_id,))
    return Path(p["dir"]) if p and p.get("dir") else None


def attached_runs(db, sid: str) -> list[str]:
    return [r["run_id"] for r in db.fetchall("SELECT run_id FROM study_links WHERE study_id = ? AND attached = 1 "
                                             "ORDER BY rowid", (sid,))]


def attached_studies_of(db, run_id: str, kinds=ATTACHABLE) -> list[dict]:
    """Studies attached to ``run_id`` (newest first), of ``kinds``."""
    marks = ",".join("?" for _ in kinds)
    return db.fetchall(f"SELECT s.* FROM studies s JOIN study_links l ON l.study_id = s.id "
                       f"WHERE l.run_id = ? AND l.attached = 1 AND s.kind IN ({marks}) "
                       f"ORDER BY s.updated_utc DESC, s.created_utc DESC", (run_id, *kinds))


def effective_status(db, row: dict) -> str:
    """The row's status, or its job's while that job is queued or running (the row lags behind a little)."""
    st = row.get("status") or "queued"
    jid = row.get("job_id")
    if jid:
        j = db.fetchone("SELECT status FROM jobs WHERE id = ?", (jid,))
        if j is not None and (j["status"] in ACTIVE_JOB or st in ACTIVE_JOB):
            return "queued" if j["status"] == "blocked" else j["status"]
    return st


def stale_vs(db, row: dict) -> list[str]:
    """Runs (the target and the attached ones) whose checkpoint fingerprint differs from the recorded one."""
    fp = row.get("run_fingerprint")
    if not fp:
        return []
    ids = [i for i in [row.get("target_run_id"), *attached_runs(db, row["id"])] if i]
    out = []
    for rid in dict.fromkeys(ids):
        r = db.fetchone("SELECT fingerprint FROM runs WHERE id = ?", (rid,))
        if r is not None and r.get("fingerprint") and r["fingerprint"] != fp:
            out.append(rid)
    return out


def _children(db, sid: str) -> list[dict]:
    from sparc.studio.runs.registry import run_summary

    rows = db.fetchall("SELECT * FROM runs WHERE study_id = ? ORDER BY created_utc, id", (sid,))
    return [run_summary(r) for r in rows]


def study_out(db, row: dict) -> dict:
    """A ``studies`` row as the api.md ``Study``."""
    return {
        "id": row["id"], "project_id": row.get("project_id") or "", "kind": row.get("kind") or "",
        "target_run_id": row.get("target_run_id"), "job_id": row.get("job_id"), "out_dir": row.get("out_dir") or "",
        "status": effective_status(db, row), "params": dbmod.loads(row.get("params_json"), {}) or {},
        "summary": dbmod.loads(row.get("summary_json")), "origin": row.get("origin") or "studio",
        "created_utc": row.get("created_utc") or "", "updated_utc": row.get("updated_utc") or "",
        "children": _children(db, row["id"]), "attached_runs": attached_runs(db, row["id"]),
        "stale_vs": stale_vs(db, row),
    }


def list_studies(db, project_id: str, kind: str | None = None) -> list[dict]:
    if kind:
        rows = db.fetchall("SELECT * FROM studies WHERE project_id = ? AND kind = ? "
                           "ORDER BY created_utc DESC, id", (project_id, kind))
    else:
        rows = db.fetchall("SELECT * FROM studies WHERE project_id = ? ORDER BY created_utc DESC, id", (project_id,))
    return [study_out(db, r) for r in rows]


# ---------------------------------------------------------------------------
# mirror and events
# ---------------------------------------------------------------------------

def _mirror_path(db, row: dict) -> Path | None:
    pdir = project_dir_of(db, row.get("project_id"))
    return pdir / "studies" / row["id"] / MIRROR if pdir is not None else None


def write_mirror(db, sid: str) -> None:
    """Rewrite ``projects/<slug>/studies/<sid>/study.json`` from the row and its links (atomic)."""
    row = db.fetchone("SELECT * FROM studies WHERE id = ?", (sid,))
    if row is None:
        return
    path = _mirror_path(db, row)
    if path is None:
        return
    links = db.fetchall("SELECT run_id, attached FROM study_links WHERE study_id = ? ORDER BY rowid", (sid,))
    doc = {"schema": 1, **{k: row.get(k) for k in ("id", "project_id", "kind", "target_run_id", "job_id", "out_dir",
                                                   "status", "run_fingerprint", "origin", "created_utc",
                                                   "updated_utc")},
           "params": dbmod.loads(row.get("params_json"), {}) or {}, "summary": dbmod.loads(row.get("summary_json")),
           "links": [{"run_id": r["run_id"], "attached": bool(r["attached"])} for r in links]}
    try:
        write_json_atomic(path, doc)
    except OSError:
        log.exception("could not write the mirror of study %s", sid)


def publish(hub, row: dict, status: str | None = None) -> None:
    """``study.updated`` on the global stream (api.md §17.1)."""
    if hub is None or row is None:
        return
    try:
        hub.publish("study.updated", {"study_id": row["id"], "kind": row.get("kind"),
                                      "status": status or row.get("status"), "project_id": row.get("project_id"),
                                      "target_run_id": row.get("target_run_id")})
    except Exception:
        pass


# ---------------------------------------------------------------------------
# create / update / links
# ---------------------------------------------------------------------------

def create_study(db, *, project_id: str, kind: str, target_run_id: str | None, params: dict,
                 origin: str = "studio", out_dir: str | Path | None = None, attach: bool = True,
                 status: str = "queued", summary: dict | None = None, hub=None) -> dict:
    """Insert a study; Studio studies get ``projects/<slug>/studies/<sid>/`` (with ``children/``).

    ``attach``: link it (attached) to its target run, so it feeds that run's uncertainty (attachable kinds)."""
    pdir = project_dir_of(db, project_id)
    if pdir is None:
        raise ApiError("not_found", f"no project {project_id!r}")
    sid = new_id("st")
    if out_dir is None:
        out = pdir / "studies" / sid
        out.mkdir(parents=True, exist_ok=True)
        if kind in ("placebo", "multiverse", "reproduce"):
            (out / "children").mkdir(exist_ok=True)
    else:
        out = Path(out_dir)
    fp = None
    if target_run_id:
        r = db.fetchone("SELECT fingerprint FROM runs WHERE id = ?", (target_run_id,))
        fp = (r or {}).get("fingerprint")
    now = utc_now()
    row = {"id": sid, "project_id": project_id, "kind": kind, "target_run_id": target_run_id, "job_id": None,
           "out_dir": str(out), "status": status, "params_json": dbmod.dumps(params or {}),
           "summary_json": dbmod.dumps(summary) if summary is not None else None, "run_fingerprint": fp,
           "origin": origin, "created_utc": now, "updated_utc": now}

    def fn(conn):
        conn.execute("INSERT INTO studies (id, project_id, kind, target_run_id, job_id, out_dir, status, params_json, "
                     "summary_json, run_fingerprint, origin, created_utc, updated_utc) "
                     "VALUES (:id, :project_id, :kind, :target_run_id, :job_id, :out_dir, :status, :params_json, "
                     ":summary_json, :run_fingerprint, :origin, :created_utc, :updated_utc)", row)
        if attach and target_run_id and kind in ATTACHABLE:
            conn.execute("INSERT OR REPLACE INTO study_links (run_id, study_id, attached) VALUES (?, ?, 1)",
                         (target_run_id, sid))

    db.run(fn)
    write_mirror(db, sid)
    publish(hub, row)
    return db.fetchone("SELECT * FROM studies WHERE id = ?", (sid,))


def update_study(db, sid: str, hub=None, **values) -> dict | None:
    """Update columns (``params`` / ``summary`` as dicts), stamp ``updated_utc``, rewrite the mirror, publish."""
    vals: dict[str, Any] = {}
    for k, v in values.items():
        if k in ("params", "summary"):
            vals[f"{k}_json"] = dbmod.dumps(v) if v is not None else None
        else:
            vals[k] = v
    vals["updated_utc"] = utc_now()
    db.update("studies", {"id": sid}, vals)
    write_mirror(db, sid)
    row = db.fetchone("SELECT * FROM studies WHERE id = ?", (sid,))
    publish(hub, row)
    return row


def set_links(db, sid: str, run_id: str, attached: bool, hub=None) -> dict:
    """Attach (or detach) study ``sid`` to run ``run_id``; returns the study row."""
    row = get_row(db, sid)
    if row.get("kind") not in ATTACHABLE:
        raise ApiError("validation", f"a {row.get('kind')} study cannot be attached to a run",
                       detail={"errors": [{"path": "study_id", "message": "not attachable", "code": "kind"}]})
    if db.fetchone("SELECT id FROM runs WHERE id = ?", (run_id,)) is None:
        raise ApiError("not_found", f"no run {run_id!r}")
    db.execute("INSERT OR REPLACE INTO study_links (run_id, study_id, attached) VALUES (?, ?, ?)",
               (run_id, sid, 1 if attached else 0))
    if attached and not row.get("run_fingerprint") and not row.get("target_run_id"):
        # an imported study first attached to a run: what it is compared with from now on
        r = db.fetchone("SELECT fingerprint FROM runs WHERE id = ?", (run_id,))
        if r and r.get("fingerprint"):
            db.update("studies", {"id": sid}, {"run_fingerprint": r["fingerprint"]})
    return update_study(db, sid, hub=hub) or row


def delete_study(db, sid: str, *, files: bool = False) -> None:
    """Drop the study (``409 active`` while its job runs); ``files`` also removes a Studio study's folder and
    the child runs it holds (imported folders are never deleted)."""
    row = get_row(db, sid)
    if row.get("job_id"):
        j = db.fetchone("SELECT status FROM jobs WHERE id = ?", (row["job_id"],))
        if j is not None and j["status"] in ACTIVE_JOB:
            raise ApiError("active", "the study's job is still running; cancel it first",
                           detail={"job_id": row["job_id"]})
    mirror = _mirror_path(db, row)

    def fn(conn):
        conn.execute("DELETE FROM studies WHERE id = ?", (sid,))
        conn.execute("DELETE FROM study_links WHERE study_id = ?", (sid,))
        if files and row.get("origin") == "studio":
            conn.execute("DELETE FROM runs WHERE study_id = ?", (sid,))

    db.run(fn)
    if mirror is None:
        return
    if row.get("origin") != "studio" or files:
        # an imported study's folder in the project holds only its mirror; a Studio study's is its out_dir
        shutil.rmtree(mirror.parent, ignore_errors=True)
    else:
        _unlink(mirror)                     # its files stay; a reindex must not bring the study back


def _unlink(p: Path) -> None:
    try:
        p.unlink()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------

def _f(v, d: int = 2) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "—"
    return f"{x:.{d}f}" if x == x and abs(x) != float("inf") else "—"


def summary_for(kind: str, data: dict | None, *, label: str | None = None) -> dict | None:
    """The ``summary`` of a study from its core output (``placebo.json``, ``simcheck_summary``,
    ``multiverse_summary.json``, ``reproduce.json``, ``benchmark.json``): the fields the run overview reads
    (SPEC §19 M1–M2) plus ``headline`` and ``label``."""
    if not isinstance(data, dict):
        return None
    out: dict[str, Any] = {}
    if kind == "placebo":
        n, pm, pc = data.get("n_placebos"), data.get("n_pass_model"), data.get("n_pass_causal")
        out = {"n_placebos": n, "n_pass_model": pm, "n_pass_causal": pc, "kinds": data.get("kinds"),
               "coarse_m": data.get("coarse_m")}
        if n:
            out["headline"] = f"model passes {pm}/{n}, causal passes {pc}/{n}"
    elif kind == "simcheck":
        bc = data.get("bias_correction") or {}
        out = {"n_rows": data.get("n_rows"), "n_errors": data.get("n_errors"), "bias_correction": bc,
               "generators": sorted((data.get("generators") or {}).keys())}
        rng = bc.get("share_range")
        if rng:
            a, b = rng[0], rng[-1]
            out["headline"] = (f"effect share {_f(a)}" if a == b else f"effect share {_f(a)}–{_f(b)}") + \
                (" (stable)" if bc.get("stable") else "")
        elif data.get("n_rows"):
            out["headline"] = f"{data['n_rows']} replicates"
    elif kind == "multiverse":
        out = {k: data.get(k) for k in ("sign_stability_min", "median_kendall_tau", "median_top_decile_jaccard")}
        out["variants"] = sorted((data.get("runs") or {}).keys())
        if data.get("sign_stability_min") is not None:
            out["headline"] = (f"signs stable in ≥ {float(data['sign_stability_min']):.0%} of variants"
                               + (f", τ {_f(data.get('median_kendall_tau'))}"
                                  if data.get("median_kendall_tau") is not None else ""))
        elif data.get("note"):
            out["headline"] = str(data["note"])
    elif kind == "reproduce":
        checks = data.get("checks") or []
        out = {"pass": data.get("pass"), "n_hard_fail": sum(1 for c in checks if c.get("hard") and not c.get("ok")),
               "reproduction": data.get("reproduction"), "original": data.get("original")}
        if data.get("pass") is not None:
            out["headline"] = "reproduced" if data["pass"] else f"not reproduced ({out['n_hard_fail']} checks failed)"
    elif kind == "benchmark":
        runs = data.get("runs") or {}
        out = {"n": data.get("n"), "seed": data.get("seed"),
               "shares": {k: (r.get("stack") or {}).get("share") for k, r in runs.items()}}
        sh = [v for v in out["shares"].values() if v is not None]
        if sh:
            out["headline"] = "stack effect share " + " / ".join(_f(v) for v in sh)
    if label:
        out["label"] = label
    return out


# ---------------------------------------------------------------------------
# import of existing study folders (cross-item contract)
# ---------------------------------------------------------------------------

def detect_kind(d: Path) -> str | None:
    """The kind of a study folder from its files (``None`` when it is none of them)."""
    if (d / "placebo.json").is_file():
        return "placebo"
    if (d / "simcheck.jsonl").is_file() or (d / "simcheck_summary.json").is_file():
        return "simcheck"
    if (d / "multiverse_summary.json").is_file() or (d / "baseline.json").is_file():
        return "multiverse"
    if (d / "reproduce.json").is_file():
        return "reproduce"
    if (d / "benchmark.json").is_file():
        return "benchmark"
    return None


def _folder_summary(kind: str, d: Path) -> dict | None:
    if kind == "placebo":
        return read_json(d / "placebo.json")
    if kind == "simcheck":
        if (d / "simcheck.jsonl").is_file():
            from sparc.core.simcheck import merge_results, summarize

            rows = merge_results([d])
            return summarize(rows) if rows else read_json(d / "simcheck_summary.json")
        return read_json(d / "simcheck_summary.json")
    if kind == "multiverse":
        s = read_json(d / "multiverse_summary.json")
        if s is None and (d / "baseline.json").is_file():
            from sparc.core.multiverse import summarize

            s = summarize(d)
        return s
    if kind == "reproduce":
        return read_json(d / "reproduce.json")
    if kind == "benchmark":
        return read_json(d / "benchmark.json")
    return None


def _simcheck_parts(d: Path) -> list[Path]:
    """A simcheck folder, or - for a parent of several (``simcheck/{null,physics}``) - its sub-folders."""
    if (d / "simcheck.jsonl").is_file() or (d / "simcheck_summary.json").is_file():
        return [d]
    return sorted(p for p in d.iterdir() if p.is_dir() and ((p / "simcheck.jsonl").is_file()
                                                            or (p / "simcheck_summary.json").is_file()))


def import_study_dir(dir, project_id, kind=None, target_run_id=None, sctx=None) -> dict:
    """Register an existing study folder in place (``origin = imported``) and return its ``Study``.

    ``kind`` defaults to what the folder holds (:func:`detect_kind`).  A simcheck folder whose replicates sit
    in sub-folders (``simcheck/null``, ``simcheck/physics``) becomes one study per sub-folder, matching the
    folders an ``uncertainty.json`` lists as its sources; the first is returned.  An import of a folder
    already indexed returns the existing study.  The study is attached to ``target_run_id`` when given and
    its run's ``uncertainty.json`` (or ``placebo.json``) used it.  Errors: ``404 not_found`` (no folder or
    project), ``422 validation`` (not a study folder)."""
    db = sctx.db if sctx is not None else None
    if db is None:
        from sparc.studio.runs.registry import active_registry

        db = active_registry().db
    d = Path(str(dir)).expanduser().resolve()
    if not d.is_dir():
        raise ApiError("not_found", f"no study folder at {d}")
    if project_dir_of(db, project_id) is None:
        raise ApiError("not_found", f"no project {project_id!r}")
    k = kind or detect_kind(d)
    parts = _simcheck_parts(d) if k == "simcheck" else [d]
    if k not in STUDY_KINDS or not parts:
        raise ApiError("validation", f"{d} is not a study folder (placebo, simcheck, multiverse, reproduce or "
                       "benchmark files)", detail={"errors": [{"path": "dir", "message": "no study files",
                                                               "code": "study_dir"}]})
    hub = getattr(sctx, "hub", None) if sctx is not None else None
    if sctx is not None and getattr(sctx, "paths", None) is not None:
        try:
            sctx.paths.allow(d)
        except Exception:
            pass
    out = []
    for p in parts:
        existing = db.fetchone("SELECT * FROM studies WHERE out_dir = ? AND project_id = ?", (str(p), project_id))
        if existing is not None:
            out.append(existing)
            continue
        try:
            data = _folder_summary(k, p)
        except Exception:                   # an unreadable output: the study is still listed
            log.exception("could not summarise %s", p)
            data = None
        label = p.name if p == d else f"{d.name}/{p.name}"
        summary = summary_for(k, data, label=label) or {"label": label}
        attach = bool(target_run_id) and k in ATTACHABLE and _used_by(db, target_run_id, k, p)
        row = create_study(db, project_id=project_id, kind=k, target_run_id=target_run_id, params={},
                           origin="imported", out_dir=p, attach=attach, status="succeeded", summary=summary,
                           hub=hub)
        if not attach and target_run_id and k in ATTACHABLE:
            # known to the run but not one of its sources: listed detached, so it can be attached later
            db.execute("INSERT OR REPLACE INTO study_links (run_id, study_id, attached) VALUES (?, ?, 0)",
                       (target_run_id, row["id"]))
            write_mirror(db, row["id"])
        out.append(row)
    return study_out(db, out[0])


def _used_by(db, run_id: str, kind: str, d: Path) -> bool:
    """Whether the run's uncertainty report used folder ``d`` (``uncertainty.json`` ``sources``), or - for a
    placebo study - whether the run carries that placebo summary."""
    r = db.fetchone("SELECT run_dir FROM runs WHERE id = ?", (run_id,))
    if r is None:
        return False
    run_dir = Path(r["run_dir"])
    if kind == "placebo":
        mine = read_json(run_dir / "placebo.json")
        theirs = read_json(d / "placebo.json")
        return isinstance(mine, dict) and isinstance(theirs, dict) and mine.get("rows") == theirs.get("rows")
    u = read_json(run_dir / "uncertainty.json") or {}
    src = u.get("sources") or {}
    cands = [src.get("multiverse")] if kind == "multiverse" else list(src.get("simcheck") or [])
    for c in cands:
        if not c:
            continue
        p = Path(str(c)).expanduser()
        for base in ([p] if p.is_absolute() else [run_dir / p, run_dir.parent / p, Path.cwd() / p,
                                                 *(par / p for par in run_dir.parents)]):
            try:
                if base.resolve() == d:
                    return True
            except OSError:
                continue
    return False


# ---------------------------------------------------------------------------
# reindex (sparc studio --reindex)
# ---------------------------------------------------------------------------

def _reindex(db, workspace) -> dict:
    """Rebuild ``studies`` and ``study_links`` from ``projects/*/studies/*/study.json``."""
    n = links = 0
    root = Path(workspace.projects_dir)
    for mirror in sorted(root.glob(f"*/studies/*/{MIRROR}")) if root.is_dir() else []:
        doc = read_json(mirror)
        if not isinstance(doc, dict) or not doc.get("id"):
            continue
        row = {k: doc.get(k) for k in ("id", "project_id", "kind", "target_run_id", "job_id", "out_dir", "status",
                                       "run_fingerprint", "origin", "created_utc", "updated_utc")}
        row["params_json"] = dbmod.dumps(doc.get("params") or {})
        row["summary_json"] = dbmod.dumps(doc.get("summary")) if doc.get("summary") is not None else None
        if row.get("job_id"):
            j = db.fetchone("SELECT status FROM jobs WHERE id = ?", (row["job_id"],))
            if j is not None and j["status"] not in ACTIVE_JOB and row.get("status") in ACTIVE_JOB:
                row["status"] = j["status"]           # the job ended while the server was down
        db.insert("studies", row, replace=True)
        n += 1
        for link in doc.get("links") or []:
            if link.get("run_id"):
                db.execute("INSERT OR REPLACE INTO study_links (run_id, study_id, attached) VALUES (?, ?, ?)",
                           (link["run_id"], doc["id"], 1 if link.get("attached", True) else 0))
                links += 1
    return {"studies": n, "links": links}


try:
    dbmod.reindex_hook("studies", order=20)(_reindex)
except Exception:  # pragma: no cover - the foundation always provides the hook registry
    log.exception("cannot register the studies reindex hook")
