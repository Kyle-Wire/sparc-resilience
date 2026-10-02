"""The runs registry: index run directories in SQLite, compute their status, import runs in place and follow
live CLI runs (SPEC §4.3, §5.8, §5.11, §5.14, §10.6).

**Sources.**  Studio runs under ``projects/<slug>/runs/<run_id>/`` (with ``studio/launch.json``), study
children under ``projects/<slug>/studies/<sid>/children/``, runs imported in place (their Studio side folder
``<ws>/imports/<run_id>/studio/`` holds ``import.json``) and the user's **watch roots** (scanned every 10 s).
A run row is derived from ``launch.json``, ``run_state.json``, ``checkpoint.json``, the manifest and the
``jobs`` table, so ``sparc studio --reindex`` rebuilds it from disk (:func:`_reindex`).

**Status** (SPEC §5.8): ``queued`` / ``running`` while a ``run.core`` job is active; ``external_live`` for a
CLI run still going (its process alive on this host - core rewrites ``run_state.json`` only at stage
boundaries - or, from another host, ``run_state.json`` or its progress file written in the last 2 minutes);
a study child (``study_child`` / ``reproduction``, or a row or manifest naming a study) in the same state is
``running`` under its study's job and is never followed by a pseudo-job; else from ``run_state.status`` -
``complete``, and for cancelled / failed / interrupted runs ``partial`` when the checkpoint done set is not
empty; a manifest without ``run_state.json`` (older code) is ``complete``; an imported folder with neither
is ``imported``.

**Lineage.**  Children are linked through ``run_meta`` (``parent_run_id``, ``study_id``, ``role``), never
through directory names; legacy study folders without ``run_meta`` fall back to the name patterns
``<name>_placebo_<kind>``, ``<name>_mv_<variant>`` and ``<dir>_reproduce``, and the link is flagged
*inferred*.

**Run ids.**  Studio runs use the directory name (``YYYYMMDD-HHMMSS-<mode>-<4 hex>``); imported and
watched runs get the same shape from the manifest (or run_state) time, the mode and a hash of the path,
so a reindex finds the same id again.  Study children use the run_state start time first: they are indexed
while they run (no manifest yet), and the id must not change once the manifest exists.

Cross-item contract (SPEC §10.2): :func:`import_run` ``(dir, project_id, config_path=None,
trust_pickles=False) -> RunSummary``.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

from sparc.studio import db as dbmod
from sparc.studio.errors import ApiError
from sparc.studio.runs.common import fnum, read_json_cached
from sparc.studio.workspace import parse_utc, utc_iso, utc_now, write_json_atomic

log = logging.getLogger("sparc.studio.runs")

__all__ = ["Registry", "import_run", "active_registry", "run_summary", "pickle_trusted", "set_active_run",
           "RUN_FILES", "EXTERNAL_FRESH_S", "WATCH_INTERVAL_S"]

EXTERNAL_FRESH_S = 120.0
WATCH_INTERVAL_S = 10.0
#: files that make a directory a run directory
RUN_FILES = ("manifest.json", "run_state.json", "predictions.parquet", "studio/launch.json")
_SKIP_DIRS = {"studio", "planner", "geotiff", "cache", "__pycache__", "engine", "results", "plans", "sweeps",
              "comparisons", "blobs", "exports", "data", "inputs", "scenarios", "findings", ".git", "node_modules"}
_ACTIVE_JOB = ("queued", "blocked", "starting", "running", "cancelling")
#: origins of runs a study's job tracks (never CLI runs, SPEC §4.3)
_CHILD_ORIGINS = ("study_child", "reproduction")
_PATTERNS = [
    (re.compile(r"^(?P<parent>.+)_placebo_(?P<x>grf|shift|rotate)(_coarse[0-9.]+)?$"), "placebo:{x}"),
    (re.compile(r"^(?P<parent>.+)_mv_(?P<x>[A-Za-z0-9_]+)$"), "variant:{x}"),
    (re.compile(r"^(?P<parent>.+)_reproduce$"), "reproduction"),
]

_ACTIVE: "Registry | None" = None
#: a run id read from a folder's own ``launch.json`` names a side folder ``<ws>/imports/<run_id>``: one plain
#: path segment only (never absolute, never ``..``)
_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def _launch_run_id(launch: dict | None) -> str | None:
    """The ``run_id`` of a ``launch.json`` when it is a safe id, else ``None`` (the caller derives one).  An
    imported folder's ``launch.json`` is untrusted input: its id becomes a path under the workspace."""
    rid = (launch or {}).get("run_id")
    return rid if isinstance(rid, str) and _RUN_ID_RE.fullmatch(rid) and ".." not in rid else None


def active_registry(sctx=None) -> "Registry":
    """The registry of ``sctx`` (a server context), else of the running server."""
    reg = (getattr(sctx, "services", None) or {}).get("registry") if sctx is not None else None
    if reg is None:
        reg = _ACTIVE
    if reg is None:
        raise RuntimeError("the runs registry is not running (start the Studio server first)")
    return reg


def import_run(dir, project_id=None, config_path=None, trust_pickles=False, sctx=None) -> dict:
    """Cross-item contract (SPEC §10.2): register a run directory in place; returns its ``RunSummary``.

    Called as ``import_run(dir, project_id, config_path=…, trust_pickles=…)`` (sync; the caller passes its
    server context as ``sctx``).  Raises ``ApiError``: ``404 not_found``, ``422 needs_config``,
    ``422 mismatch``."""
    return active_registry(sctx).import_run(dir, project_id=project_id, config_path=config_path,
                                            trust_pickles=trust_pickles)


def pickle_trusted(run_id: str, sctx=None) -> bool:
    """Whether the engine may unpickle this run's checkpoint (SPEC §10.8): runs in the workspace always, runs
    imported in place only when imported with ``trust_pickles``."""
    return active_registry(sctx).trusted(run_id)


def set_active_run(db, project_id: str | None, run_id: str | None) -> None:
    """Set ``projects.active_run_id`` through the project service, which keeps ``project.json`` in sync (a
    rebuilt database restores from it); a direct row update when that service is not installed."""
    if not project_id:
        return
    import importlib

    try:
        svc = importlib.import_module("sparc.studio.projects.service")
    except ModuleNotFoundError:
        svc = None
    fn = getattr(svc, "patch_project", None) if svc is not None else None
    if fn is not None:
        fn(db, project_id, {"active_run_id": run_id})
    else:
        db.update("projects", {"id": project_id}, {"active_run_id": run_id})


def is_run_dir(path: Path) -> bool:
    return any((path / f).exists() for f in RUN_FILES)


def _json(path: Path) -> dict | None:
    obj = read_json_cached(path)
    return obj if isinstance(obj, dict) else None


def _stamp(ts: float | None) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(ts if ts is not None else time.time(), tz=timezone.utc).strftime("%Y%m%d-%H%M%S")


def _mode_tag(args: dict) -> str:
    if args.get("coarse"):
        return f"coarse{float(args['coarse']):g}".replace(".", "p")
    if args.get("fast"):
        return "fast"
    st = args.get("stages")
    if st and set(st) != {"S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"}:
        return "custom"
    return "full"


def _mode(args: dict) -> str:
    t = _mode_tag(args)
    return "coarse" if t.startswith("coarse") else t


class Registry:
    """Index of run directories (see the module docstring)."""

    def __init__(self, sctx=None, *, db=None, workspace=None, hub=None):
        self.sctx = sctx
        self.db = db if db is not None else sctx.db
        self.ws = workspace if workspace is not None else sctx.workspace
        self.hub = hub if hub is not None else (getattr(sctx, "hub", None) if sctx is not None else None)
        self._task: asyncio.Task | None = None
        self._ext_synth: dict[str, dict] = {}

    # ------------------------------------------------------------------ settings and helpers

    def _watch_roots(self) -> list[Path]:
        roots = []
        try:
            if self.sctx is not None:
                roots = list(self.sctx.settings().watch_roots)
            else:
                roots = list(self.db.get_setting("watch_roots", []) or [])
        except Exception:
            roots = []
        return [Path(os.path.expanduser(r)) for r in roots if r]

    def _publish(self, type_: str, data: dict) -> None:
        if self.hub is not None:
            try:
                self.hub.publish(type_, data)
            except Exception:
                pass

    def _allow(self, path: Path) -> None:
        if self.sctx is not None and getattr(self.sctx, "paths", None) is not None:
            try:
                self.sctx.paths.allow(path)
            except Exception:
                pass

    def _allow_imported(self, run_dir: Path, studio_dir: Path | None) -> None:
        """Register a run imported in place (and the config folder given at import) as a path root (SPEC §10.8)."""
        self._allow(run_dir)
        rec = _json(Path(studio_dir) / "import.json") if studio_dir is not None else None
        if rec and rec.get("config_path"):
            self._allow(Path(rec["config_path"]).parent)

    def project_for_dir(self, run_dir: Path) -> dict | None:
        """The project whose directory contains ``run_dir``."""
        rd = str(Path(run_dir).resolve())
        for p in self.db.fetchall("SELECT id, dir, demo FROM projects"):
            d = str(Path(p["dir"]).resolve())
            if rd == d or rd.startswith(d.rstrip(os.sep) + os.sep):
                return p
        return None

    def trusted(self, run_id: str) -> bool:
        row = self.db.fetchone("SELECT run_dir, studio_dir, origin FROM runs WHERE id = ?", (run_id,))
        if row is None:
            return False
        root = self.ws.root.resolve()
        rd = Path(row["run_dir"]).resolve()
        if rd == root or root in rd.parents:
            return True
        rec = _json(Path(row["studio_dir"]) / "import.json") or {}
        return bool(rec.get("trust_pickles"))

    # ------------------------------------------------------------------ scanning

    def scan(self) -> dict:
        """Index the workspace, the imported runs and the watch roots; returns counts."""
        out = {"workspace": 0, "imports": 0, "watch": 0, "refreshed": 0}
        projects = self.ws.projects_dir
        if projects.is_dir():
            for pdir in sorted(p for p in projects.iterdir() if p.is_dir()):
                for rd in sorted((pdir / "runs").glob("*")) if (pdir / "runs").is_dir() else []:
                    if rd.is_dir() and is_run_dir(rd):
                        if self._index_safe(rd, origin="studio", studio_dir=rd / "studio"):
                            out["workspace"] += 1
                studies = pdir / "studies"
                if studies.is_dir():
                    for sdir in sorted(s for s in studies.iterdir() if s.is_dir()):
                        for rd in self._walk_runs(sdir / "children", depth=2):
                            if self._index_safe(rd, origin="study_child", studio_dir=rd / "studio",
                                                study_id=sdir.name):
                                out["workspace"] += 1
        imports = self.ws.imports_dir
        if imports.is_dir():
            for idir in sorted(p for p in imports.iterdir() if p.is_dir()):
                rec = _json(idir / "studio" / "import.json")
                if not rec or not rec.get("dir"):
                    continue
                rd = Path(rec["dir"])
                if not rd.is_dir():
                    continue
                if self._index_safe(rd, origin=rec.get("origin") or "imported", studio_dir=idir / "studio",
                                    run_id=idir.name, project_id=rec.get("project_id")):
                    out["imports"] += 1
        for root in self._watch_roots():
            for rd in self._walk_runs(root, depth=4):
                if self.db.fetchone("SELECT id FROM runs WHERE run_dir = ?", (str(rd.resolve()),)):
                    continue
                if self._index_safe(rd, origin="imported", studio_dir=None, watched=True):
                    out["watch"] += 1
        for row in self.db.fetchall("SELECT id FROM runs"):
            if self.refresh(row["id"], publish=False) is not None:
                out["refreshed"] += 1
        return out

    def _walk_runs(self, root: Path, depth: int):
        if not root.is_dir():
            return
        stack = [(root, 0)]
        while stack:
            d, k = stack.pop()
            try:
                entries = sorted(p for p in d.iterdir() if p.is_dir() and p.name not in _SKIP_DIRS
                                 and not p.name.startswith("."))
            except OSError:
                continue
            for p in entries:
                if is_run_dir(p):
                    yield p
                elif k + 1 < depth:
                    stack.append((p, k + 1))

    def _index_safe(self, run_dir: Path, **kw) -> bool:
        try:
            self.index_run_dir(run_dir, **kw)
            return True
        except Exception:
            log.exception("could not index %s", run_dir)
            return False

    # ------------------------------------------------------------------ indexing

    def _imported_id(self, run_dir: Path, args: dict, manifest: dict | None, state: dict | None,
                     child: bool = False) -> str:
        m_ts = parse_utc((manifest or {}).get("created_utc"))
        s_ts = parse_utc((state or {}).get("started_utc"))
        ts = (s_ts if s_ts is not None else m_ts) if child else (m_ts if m_ts is not None else s_ts)
        if ts is None:
            try:
                ts = run_dir.stat().st_mtime
            except OSError:
                ts = time.time()
        h = hashlib.sha1(str(run_dir.resolve()).encode()).hexdigest()
        base = f"{_stamp(ts)}-{_mode_tag(args)}"
        for n in (4, 6, 8, 12):
            rid = f"{base}-{h[:n]}"
            row = self.db.fetchone("SELECT run_dir FROM runs WHERE id = ?", (rid,))
            if row is None or Path(row["run_dir"]).resolve() == run_dir.resolve():
                return rid
        return f"{base}-{h}"

    def index_run_dir(self, run_dir, *, origin: str, studio_dir=None, run_id: str | None = None,
                      project_id: str | None = None, study_id: str | None = None, watched: bool = False,
                      publish: bool = True) -> dict:
        """Insert or update the row of ``run_dir``; returns the row."""
        run_dir = Path(run_dir).resolve()
        existing = self.db.fetchone("SELECT * FROM runs WHERE run_dir = ?", (str(run_dir),))
        if existing is not None:
            if existing.get("origin") not in ("studio", "study_child"):
                self._allow_imported(run_dir, Path(existing["studio_dir"]))   # path roots after a restart
            return self.refresh(existing["id"], publish=publish) or existing
        launch = _json(Path(studio_dir or run_dir / "studio") / "launch.json")
        manifest = _json(run_dir / "manifest.json")
        state = _json(run_dir / "run_state.json")
        args = dict((launch or {}).get("args") or {})
        if not args and manifest:
            co = (manifest.get("qa") or {}).get("coarse")
            args = {"fast": bool(manifest.get("fast_mode")), "coarse": (co or {}).get("cell_m")}
        if run_id is None:
            run_id = _launch_run_id(launch) or (run_dir.name if origin in ("studio", "study_child") and launch
                                                else None)
        if run_id is None:
            child = origin in _CHILD_ORIGINS or bool(study_id) or _names_study(manifest)
            run_id = self._imported_id(run_dir, args, manifest, state, child=child)
        if studio_dir is None:
            studio_dir = self.ws.imports_dir / run_id / "studio"
            studio_dir.mkdir(parents=True, exist_ok=True)
            if not (studio_dir / "import.json").exists():
                write_json_atomic(studio_dir / "import.json", {
                    "schema": 1, "dir": str(run_dir), "project_id": project_id, "config_path": None,
                    "trust_pickles": False, "imported_utc": utc_now(), "origin": "imported", "watched": watched})
        if project_id is None:
            project_id = (launch or {}).get("project_id")
        if project_id is None:
            p = self.project_for_dir(run_dir)
            project_id = p["id"] if p else None
        base = {"id": run_id, "project_id": project_id, "run_dir": str(run_dir), "studio_dir": str(Path(studio_dir)),
                "origin": origin, "study_id": study_id, "status": "imported", "created_utc": None}
        self.db.insert("runs", base, replace=True)
        if origin not in ("studio", "study_child"):
            self._allow_imported(run_dir, Path(studio_dir))
        row = self.refresh(run_id, publish=False) or base
        if publish:
            self._publish("run.indexed", {"run_id": run_id, "project_id": row.get("project_id"),
                                          "origin": row.get("origin")})
        return row

    # ------------------------------------------------------------------ refresh (derive a row from disk)

    def refresh(self, run_id: str, *, publish: bool = True) -> dict | None:
        """Recompute the row of ``run_id`` from disk and the jobs table; publishes ``run.updated`` on change."""
        row = self.db.fetchone("SELECT * FROM runs WHERE id = ?", (run_id,))
        if row is None:
            return None
        values = self._derive(row)
        changed = [k for k, v in values.items() if row.get(k) != v]
        if changed:
            self.db.update("runs", {"id": run_id}, {k: values[k] for k in changed})
        new = {**row, **values}
        if changed and publish:
            self._publish("run.updated", {"run_id": run_id, "project_id": new.get("project_id"),
                                          "status": new.get("status"), "fields": sorted(changed)})
        self._stage_timings(new)
        return new

    def _jobs(self, run_id: str) -> tuple[dict | None, dict | None]:
        """``(active run job, latest run job)`` of kinds ``run.core`` / ``run.external``."""
        rows = self.db.fetchall("SELECT id, kind, status, created_utc, finished_utc, started_utc FROM jobs "
                                "WHERE run_id = ? AND kind IN ('run.core', 'run.external') "
                                "ORDER BY created_utc DESC, rowid DESC", (run_id,))
        active = next((r for r in rows if r["status"] in _ACTIVE_JOB), None)
        return active, (rows[0] if rows else None)

    def _derive(self, row: dict) -> dict:
        run_dir = Path(row["run_dir"])
        studio = Path(row["studio_dir"])
        launch = _json(studio / "launch.json")
        state = _json(run_dir / "run_state.json")
        ckpt = _json(run_dir / "checkpoint.json")
        manifest = _json(run_dir / "manifest.json")
        rec = _json(studio / "import.json") or {}
        args = dict((launch or {}).get("args") or {})
        if not args and manifest:
            co = (manifest.get("qa") or {}).get("coarse")
            args = {"fast": bool(manifest.get("fast_mode")), "coarse": (co or {}).get("cell_m")}
        active, last = self._jobs(row["id"])
        done = sorted(set((state or {}).get("done") or []) | set((ckpt or {}).get("done") or []))
        status, origin = self._status(row, state, manifest, active, last, done, rec, launch)
        prov = (manifest or {}).get("provenance") or {}
        st = ((manifest or {}).get("metrics") or {}).get("stacker") or {}
        timings = dict((manifest or {}).get("timings_s") or {})
        stage_s = float(sum(v for v in timings.values() if fnum(v) is not None)) if timings else None
        created = (launch or {}).get("created_utc") or (state or {}).get("started_utc")
        if not created and manifest and manifest.get("created_utc"):
            # core writes the manifest when the run ends: its start is that time less the stage timings
            t = parse_utc(manifest["created_utc"])
            created = utc_iso(t - stage_s) if t is not None and stage_s else manifest["created_utc"]
        if created and not str(created).endswith("Z"):
            created = utc_iso(parse_utc(created)) or created
        finished = None
        if status in ("complete", "partial", "failed", "cancelled", "interrupted", "imported"):
            if state and state.get("status") != "running":
                finished = state.get("updated_utc")
            elif manifest and manifest.get("created_utc"):
                finished = utc_iso(parse_utc(manifest["created_utc"]))
            elif last and last.get("finished_utc"):
                finished = last["finished_utc"]
        duration = None
        if timings:
            duration = stage_s
        elif created and finished:
            a, b = parse_utc(created), parse_utc(finished)
            duration = (b - a) if a and b and b >= a else None
        scen = (manifest or {}).get("scenarios")
        n_scen = len(scen) if isinstance(scen, list) else None
        if n_scen is None and (run_dir / "scenarios.json").exists():
            s2 = read_json_cached(run_dir / "scenarios.json")
            n_scen = len(s2) if isinstance(s2, list) else None
        meta = (manifest or {}).get("run_meta") or {}
        lineage = self._lineage(row, run_dir, manifest, meta)
        if origin != "external_live" and ("reproduction" in (lineage.get("role"), meta.get("origin"),
                                                               (launch or {}).get("origin"), (launch or {}).get("role"))):
            origin = "reproduction"              # SPEC §4.3 run origins: a reproduction child keeps its own
        old_stages = dbmod.loads(row.get("stages_json"), {}) or {}
        stages_info = {**(old_stages if isinstance(old_stages, dict) else {}),
                       "requested": args.get("stages"), "timings_s": timings or None, "n_scenarios": n_scen,
                       "duration_s": duration, "link": lineage.get("link"), "role": lineage.get("role")}
        pkl = run_dir / "checkpoint.pkl"
        git = prov.get("git") or {}
        project_id = row.get("project_id") or rec.get("project_id") or (launch or {}).get("project_id")
        demo = row.get("demo") or 0
        if project_id:
            p = self.db.fetchone("SELECT demo FROM projects WHERE id = ?", (project_id,))
            if p is not None:
                demo = int(bool(p.get("demo")))
        out = {
            "project_id": project_id, "origin": origin, "status": status, "mode": _mode(args),
            "coarse_m": fnum(args.get("coarse")), "stages_json": dbmod.dumps(stages_info),
            "created_utc": created, "finished_utc": finished,
            "n_points": ((state or {}).get("meta") or {}).get("n_points") or (manifest or {}).get("n_points")
            or row.get("n_points"),
            "r2": fnum(st.get("r2")), "rmse": fnum(st.get("rmse")), "coverage": fnum(st.get("interval_coverage")),
            "checkpoint_bytes": pkl.stat().st_size if pkl.exists() else None,
            "checkpoint_done_json": dbmod.dumps((ckpt or {}).get("done")) if ckpt else None,
            "fingerprint": (ckpt or {}).get("fingerprint") or (state or {}).get("fingerprint"),
            "code_sha": (ckpt or {}).get("code_sha256") or prov.get("code_sha256"),
            "config_sha": prov.get("config_sha256"), "input_sha": prov.get("input_sha256"),
            "git_commit": (manifest or {}).get("git_commit") or git.get("commit"),
            "git_dirty": (1 if git.get("core_dirty") else 0) if "core_dirty" in git else None,
            "has_emulator": int((run_dir / "emulator.npz").exists() and (run_dir / "emulator.json").exists()),
            "manifest_mtime": (run_dir / "manifest.json").stat().st_mtime if manifest else None,
            "last_job_id": (last or {}).get("id") or row.get("last_job_id"),
            "parent_run_id": lineage.get("parent_run_id") or row.get("parent_run_id"),
            "study_id": lineage.get("study_id") or row.get("study_id"), "demo": demo,
        }
        if not row.get("label"):
            name = (manifest or {}).get("name")
            if not name and launch:
                raw = launch.get("config_raw") or {}
                name = raw.get("core", raw).get("name")
            out["label"] = name or None
        return out

    def _status(self, row, state, manifest, active, last, done, rec, launch=None) -> tuple[str, str]:
        origin = row.get("origin") or "imported"
        child = self._study_child(row, manifest)
        if active is not None:
            if active["kind"] == "run.external" and not child:
                return "external_live", "external_live"
            if active["kind"] == "run.external":     # a pseudo-job an older server started for a study child
                return "running", ("study_child" if origin == "external_live" else origin)
            return ("queued" if active["status"] in ("queued", "blocked") else "running"), origin
        st = (state or {}).get("status")
        if origin == "external_live":
            origin = "study_child" if child else "imported"
        if st == "running":
            if self._externally_live(state):
                if child:                        # fitting under its study's job (SPEC §5.11), not a CLI run
                    return "running", origin
                if last is None or last["kind"] == "run.external":
                    return "external_live", "external_live"
            if last is not None and last["status"] in ("queued", "blocked", "starting", "running", "cancelling"):
                return "running", origin
            return ("partial" if done else "interrupted"), origin
        if st == "succeeded":
            return "complete", origin
        if st in ("cancelled", "failed"):
            return ("partial" if done else st), origin
        if state is None:
            if manifest:
                return "complete", origin
            if last is not None:
                ls = last["status"]
                if ls in ("failed", "cancelled", "interrupted"):
                    return ("partial" if done else ls), origin
                if ls == "succeeded":
                    return "complete", origin
            if origin in ("studio", "study_child", "reproduction") and last is None:
                if (launch or {}).get("job_id"):         # its job ran (and was deleted) without writing a thing
                    return "interrupted", origin
                return "queued", origin                  # launched, its job not created yet
            return "imported", origin
        return "imported", origin

    @staticmethod
    def _study_child(row: dict, manifest: dict | None) -> bool:
        """A study's child run (placebo, multiverse, reproduction): its study's job tracks it, so a live
        ``run_state.json`` is never read as a CLI run."""
        return row.get("origin") in _CHILD_ORIGINS or bool(row.get("study_id")) or _names_study(manifest)

    def _externally_live(self, state: dict | None) -> bool:
        """A CLI run still going (SPEC §5.14): ``run_state.status == running`` and its process alive on this
        host - core rewrites ``run_state.json`` only at stage boundaries, so a long stage (an hour of S2_S3) is
        live while its pid is - or, when the process cannot be checked (another host, no pid), a heartbeat
        under 2 minutes old (:func:`_fresh`)."""
        if not state or state.get("status") != "running":
            return False
        alive = _pid_alive(state)
        return alive is True or (alive is None and _fresh(state))

    def _gone(self, state: dict) -> bool:
        """A running CLI run that stopped: no heartbeat for 2 minutes and its process is gone (or cannot be
        checked from this host)."""
        return not _fresh(state) and _pid_alive(state) is not True

    def _lineage(self, row: dict, run_dir: Path, manifest: dict | None, meta: dict) -> dict:
        out: dict[str, Any] = {}
        if meta.get("parent_run_id") or meta.get("study_id") or meta.get("role"):
            out.update(parent_run_id=meta.get("parent_run_id"), study_id=meta.get("study_id"), role=meta.get("role"),
                       link="explicit")
            return out
        if row.get("parent_run_id"):                    # linked earlier (or by Studio): keep how
            old = dbmod.loads(row.get("stages_json"), {}) or {}
            old = old if isinstance(old, dict) else {}
            return {"parent_run_id": row["parent_run_id"], "role": old.get("role"), "link": old.get("link")}
        names = [str((manifest or {}).get("name") or ""), run_dir.name]
        for name in names:
            for rx, role in _PATTERNS:
                m = rx.match(name)
                if not m:
                    continue
                parent = self._find_parent(m.group("parent"), run_dir)
                if parent:
                    gd = m.groupdict()
                    return {"parent_run_id": parent, "role": role.format(x=gd.get("x") or ""), "link": "inferred"}
        return out

    def _find_parent(self, name: str, child_dir: Path) -> str | None:
        rows = self.db.fetchall("SELECT id, run_dir FROM runs")
        for r in rows:
            rd = Path(r["run_dir"])
            if rd == child_dir:
                continue
            m = _json(rd / "manifest.json") or {}
            if rd.name == name or m.get("name") == name:
                return r["id"]
        return None

    def _stage_timings(self, row: dict) -> None:
        """``stage_timings`` rows from the manifest (``timings_detail.stages`` or ``timings_s``) for runs whose
        stages no Studio job recorded (imported and CLI runs, SPEC §5.12)."""
        have = self.db.fetchval("SELECT COUNT(*) FROM stage_timings WHERE run_id = ? AND source = 'events'",
                                (row["id"],), default=0)
        if have:
            return
        m = _json(Path(row["run_dir"]) / "manifest.json")
        if not m:
            return
        timings = ((m.get("timings_detail") or {}).get("stages") or m.get("timings_s") or {})
        if not timings:
            return
        from sparc.studio.jobs.eta import host_id

        rows = [(row["id"], str(k), float(v), m.get("n_points"), row.get("mode"), None, host_id(),
                 m.get("git_commit"), "manifest") for k, v in timings.items() if fnum(v) is not None]
        if rows:
            self.db.executemany("INSERT OR REPLACE INTO stage_timings (run_id, stage, seconds, n_points, mode, "
                                "threads, host_id, git_commit, source) VALUES (?,?,?,?,?,?,?,?,?)", rows)

    # ------------------------------------------------------------------ summaries

    def studies_of(self, run_ids: list[str]) -> dict[str, list[str]]:
        """``{run_id: [study_id, …]}`` of the studies attached to each run (``study_links.attached = 1``) - the
        same ``RunSummary.studies`` the project endpoints build (``sparc.studio.projects.service``)."""
        if not run_ids:
            return {}
        out: dict[str, list[str]] = {}
        want = set(run_ids)
        try:
            rows = self.db.fetchall("SELECT run_id, study_id FROM study_links WHERE attached = 1 ORDER BY rowid")
        except Exception:
            rows = []
        for r in rows:
            if r["run_id"] in want and r.get("study_id"):
                lst = out.setdefault(r["run_id"], [])
                if r["study_id"] not in lst:
                    lst.append(r["study_id"])
        return out

    def summary(self, row: dict, studies: list[str] | None = None) -> dict:
        return run_summary(row, studies if studies is not None else self.studies_of([row["id"]]).get(row["id"], []))

    def summaries(self, rows: list[dict]) -> list[dict]:
        st = self.studies_of([r["id"] for r in rows])
        return [run_summary(r, st.get(r["id"], [])) for r in rows]

    # ------------------------------------------------------------------ import

    def import_run(self, dir, *, project_id=None, config_path=None, trust_pickles=False) -> dict:
        """Register ``dir`` in place after checking its ids and folds against its config (SPEC §9.1).

        Errors: ``404 not_found`` (no such run folder), ``422 needs_config`` (no config locates its data),
        ``422 mismatch`` (ids or folds differ from the config).
        """
        from sparc.studio.runs.reader import RunContext

        run_dir = Path(os.path.expanduser(str(dir))).resolve()
        if not run_dir.is_dir() or not is_run_dir(run_dir):
            raise ApiError("not_found", f"no run folder at {run_dir}")
        if config_path is not None and not Path(config_path).is_file():
            raise ApiError("not_found", f"no config file at {config_path}")
        if project_id is not None and self.db.fetchone("SELECT id FROM projects WHERE id = ?", (project_id,)) is None:
            raise ApiError("not_found", f"no project {project_id!r}")
        existing = self.db.fetchone("SELECT * FROM runs WHERE run_dir = ?", (str(run_dir),))
        root = self.ws.root.resolve()
        in_ws = run_dir == root or root in run_dir.parents
        if existing is not None:
            studio = Path(existing["studio_dir"])
            run_id = existing["id"]
        else:
            manifest = _json(run_dir / "manifest.json")
            state = _json(run_dir / "run_state.json")
            launch = _json(run_dir / "studio" / "launch.json")
            args = dict((launch or {}).get("args") or {})
            if not args and manifest:
                co = (manifest.get("qa") or {}).get("coarse")
                args = {"fast": bool(manifest.get("fast_mode")), "coarse": (co or {}).get("cell_m")}
            run_id = _launch_run_id(launch)
            if run_id is not None:                   # an id another folder holds is not taken over
                other = self.db.fetchone("SELECT run_dir FROM runs WHERE id = ?", (run_id,))
                if other is not None and Path(other["run_dir"]).resolve() != run_dir:
                    run_id = None
            run_id = run_id or self._imported_id(run_dir, args, manifest, state)
            studio = (run_dir / "studio") if (in_ws and launch) else self.ws.imports_dir / run_id / "studio"
        rec = {"schema": 1, "dir": str(run_dir), "project_id": project_id,
               "config_path": str(Path(config_path).resolve()) if config_path else None,
               "trust_pickles": bool(trust_pickles), "imported_utc": utc_now(), "origin": "imported"}
        probe_row = {"id": run_id, "run_dir": str(run_dir), "studio_dir": str(studio), "status": "imported"}
        tmp_rec = None
        created = None                               # the side folder this call made (removed if refused)
        if studio != run_dir / "studio":
            if not studio.parent.exists() and studio.parent.parent.resolve() == self.ws.imports_dir.resolve():
                created = studio.parent
            studio.mkdir(parents=True, exist_ok=True)
            tmp_rec = studio / "import.json"
            old = _json(tmp_rec)
            write_json_atomic(tmp_rec, rec)
        ctx = RunContext(probe_row, ("probe", time.time()))
        try:
            self._verify(ctx, config_path)
        except ApiError:
            if created is not None:
                shutil.rmtree(created, ignore_errors=True)
            elif tmp_rec is not None and old is not None:
                write_json_atomic(tmp_rec, old)
            elif tmp_rec is not None:
                tmp_rec.unlink(missing_ok=True)
            raise
        origin = "imported" if not (in_ws and (run_dir / "studio" / "launch.json").exists()) else "studio"
        row = self.index_run_dir(run_dir, origin=origin, studio_dir=studio, run_id=run_id, project_id=project_id)
        if project_id and row.get("project_id") != project_id:
            self.db.update("runs", {"id": run_id}, {"project_id": project_id})
            row = self.refresh(run_id) or row
        self._allow_imported(run_dir, studio)
        return self.summary(row)

    def _verify(self, ctx, config_path) -> None:
        if ctx.cfg is None:
            raise ApiError("needs_config", "this run has no usable config: pass config_path",
                           detail={"run_dir": str(ctx.run_dir)})
        pred = ctx.predictions
        data = ctx.data
        err = ctx.data_error or ""
        if data is None:
            if "ids differ" in err:
                raise ApiError("mismatch", "the run's predictions do not match the data its config points to",
                               detail={"reason": err})
            if pred is None:
                return                                   # mid-run folder: nothing to check yet
            raise ApiError("needs_config", "the run's data cannot be found from its config: pass config_path",
                           detail={"reason": err})
        if pred is not None and ctx.cv_meta is not None and ctx.folds is None:
            raise ApiError("mismatch", "the CV folds rebuilt from the config differ from the run's",
                           detail={"reason": ctx.folds_error})

    # ------------------------------------------------------------------ deletion helpers

    def forget(self, run_id: str) -> None:
        self.db.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        self.db.execute("DELETE FROM stage_timings WHERE run_id = ?", (run_id,))

    # ------------------------------------------------------------------ external (CLI) runs

    def watch_once(self) -> list[dict]:
        """Index new runs in watch roots; returns ``[{run_id, events_path, pid, project_id, label}]`` of the runs
        (in watch roots, or imported in place elsewhere) that are live now and have no pseudo-job yet."""
        live = []
        candidates: list[tuple[Path, dict]] = []
        seen: set[str] = set()
        for root in self._watch_roots():
            for rd in self._walk_runs(root, depth=4):
                rdr = str(rd.resolve())
                row = self.db.fetchone("SELECT * FROM runs WHERE run_dir = ?", (rdr,))
                if row is None:
                    if not self._index_safe(rd, origin="imported", studio_dir=None, watched=True):
                        continue
                    row = self.db.fetchone("SELECT * FROM runs WHERE run_dir = ?", (rdr,))
                    if row is None:
                        continue
                seen.add(rdr)
                candidates.append((rd, row))
        # runs imported in place outside the watch roots: a CLI run still going there is followed the same way
        for row in self.db.fetchall("SELECT * FROM runs WHERE origin IN ('imported', 'external_live')"):
            if row["run_dir"] not in seen:
                candidates.append((Path(row["run_dir"]), row))
        for rd, row in candidates:
            state = _json(rd / "run_state.json")
            # a study child is tracked by its study's job, never as a CLI run (the manifest is read only when live)
            if row.get("origin") == "studio" or not self._externally_live(state) \
                    or self._study_child(row, _json(rd / "manifest.json")):
                if row.get("status") == "external_live":
                    self.refresh(row["id"])          # it stopped reporting without a pseudo-job: re-derive
                continue
            active, _ = self._jobs(row["id"])
            if active is not None:
                continue
            live.append({"run_id": row["id"], "events_path": (state or {}).get("events_path"),
                         "pid": (state or {}).get("pid"), "project_id": row.get("project_id"),
                         "label": row.get("label") or rd.name})
        return live

    async def watch_tick(self) -> None:
        """One pass of the external-run watcher (every 10 s, SPEC §5.14)."""
        jobs = getattr(self.sctx, "jobs", None)
        live = await asyncio.to_thread(self.watch_once)
        if jobs is not None:
            for item in live:
                await self._start_external(jobs, item)
            await self._follow_external(jobs)

    async def _start_external(self, jobs, item: dict) -> None:
        ev = item.get("events_path")
        ev_ok = bool(ev) and Path(ev).is_file()
        job = await jobs.create_external("run.external", run_id=item["run_id"], events_path=ev if ev_ok else None,
                                         project_id=item.get("project_id"), label=f"CLI run {item['label']}",
                                         params={"run_id": item["run_id"]}, pid=item.get("pid"))
        if not ev_ok:
            synth = self.ws.job_dir(job["id"])
            synth.mkdir(parents=True, exist_ok=True)
            self.db.update("jobs", {"id": job["id"]}, {"job_dir": str(synth)})
            self._ext_synth[job["id"]] = {"path": synth / "events.jsonl", "stage": None, "done": set(), "seq": 0,
                                          "started": False}
            await asyncio.to_thread(self._synthesise, job["id"], item["run_id"])
            jobs.attach_external(job["id"], synth / "events.jsonl")
        self.refresh(item["run_id"])

    def _reattach_external(self, jobs, job: dict, state: dict) -> None:
        """Tail a running pseudo-job again after a server restart (the job manager leaves ``run.external`` rows
        to the registry): the run's events file, or the events Studio synthesises from ``run_state.json``, with
        the synthesis state (sequence, current stage, started/ended) recovered from what was already written."""
        if job["id"] in getattr(jobs, "tailers", {}):
            return
        ev = state.get("events_path")
        if ev and Path(ev).is_file():
            jobs.attach_external(job["id"], ev)
            return
        synth = Path(job.get("job_dir") or "")
        if not job.get("job_dir") or synth.resolve().parent != self.ws.jobs_dir.resolve():
            return                               # a CLI events file that went away: nothing to tail
        path = synth / "events.jsonl"
        rec = {"path": path, "stage": None, "done": set(), "seq": 0, "started": False}
        from sparc.studio.jobs.tailer import iter_lines, parse_line

        for _cursor, raw in iter_lines(path, 0):
            e = parse_line(raw)
            rec["seq"] = max(rec["seq"], int(e.get("seq") or 0))
            t = e.get("type")
            if t == "run.start":
                rec["started"] = True
            elif t == "stage.start":
                rec["stage"] = e.get("stage")
            elif t == "stage.end" and e.get("stage") == rec["stage"]:
                rec["stage"] = None
            elif t == "run.end":
                rec["ended"] = True
        self._ext_synth[job["id"]] = rec
        jobs.attach_external(job["id"], path)

    async def _follow_external(self, jobs) -> None:
        rows = self.db.fetchall("SELECT id, run_id, status, job_dir FROM jobs WHERE kind = 'run.external' "
                                "AND status = 'running'")
        for r in rows:
            run = self.db.fetchone("SELECT run_dir FROM runs WHERE id = ?", (r["run_id"],))
            if run is None:
                await jobs.detach_external(r["id"], "interrupted",
                                           error={"type": "Interrupted", "message": "the run was removed"})
                continue
            state = _json(Path(run["run_dir"]) / "run_state.json") or {}
            try:
                self._reattach_external(jobs, r, state)        # in the loop: tailers are asyncio tasks
            except Exception:
                log.exception("re-attaching the pseudo-job %s failed", r["id"])
            if r["id"] in self._ext_synth:
                await asyncio.to_thread(self._synthesise, r["id"], r["run_id"])
            st = state.get("status")
            if st in ("succeeded", "failed", "cancelled"):
                await jobs.detach_external(r["id"], st, error=state.get("error") if st == "failed" else None)
                self._ext_synth.pop(r["id"], None)
                self.refresh(r["run_id"])
            elif st == "running" and self._gone(state):
                await jobs.detach_external(r["id"], "interrupted",
                                           error={"type": "Interrupted",
                                                  "message": "the CLI run stopped reporting (no heartbeat for "
                                                             "2 minutes and its process is gone)"})
                self._ext_synth.pop(r["id"], None)
                self.refresh(r["run_id"])

    def _synthesise(self, job_id: str, run_id: str) -> None:
        """Append stage events derived from ``run_state.json`` (a CLI run without a progress file)."""
        from sparc.studio.events import encode_line

        s = self._ext_synth.get(job_id)
        run = self.db.fetchone("SELECT run_dir FROM runs WHERE id = ?", (run_id,))
        if s is None or run is None:
            return
        state = _json(Path(run["run_dir"]) / "run_state.json") or {}
        name = Path(run["run_dir"]).name
        lines = []

        def ev(type_, path, **fields):
            s["seq"] += 1
            base = {"v": 1, "type": type_, "seq": s["seq"], "ts": round(time.time(), 3), "t_rel": 0.0,
                    "pid": int(state.get("pid") or 0), "job": job_id, "lvl": "info", "span": None, "parent": None,
                    "path": path, "ctx": {}}
            lines.append(encode_line({**base, **fields}))

        run_path = [f"run:{name}"]
        if not s["started"]:
            ev("run.start", run_path, name=name, stages=[], fast=False, coarse=None, resume=False, cv_curve=None,
               config_sha256="", code_sha256="", run_meta={"origin": "external_live", "synthesised": True})
            s["started"] = True
        stage = state.get("stage")
        if stage != s["stage"]:
            if s["stage"]:
                ev("stage.end", run_path + [f"stage:{s['stage']}"], stage=s["stage"], status="ok", elapsed_s=0.0,
                   summary={})
            if stage and state.get("status") == "running":
                ev("stage.start", run_path + [f"stage:{stage}"], stage=stage, label=stage, est_s=None)
            s["stage"] = stage
        if state.get("status") in ("succeeded", "failed", "cancelled") and not s.get("ended"):
            ev("run.end", run_path, status=state["status"], elapsed_s=0.0, timings_s={},
               done=list(state.get("done") or []), error=state.get("error"))
            s["ended"] = True
        if lines:
            fd = os.open(s["path"], os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o644)
            try:
                for line in lines:
                    os.write(fd, line)
            finally:
                os.close(fd)

    # ------------------------------------------------------------------ server lifecycle

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name="runs-watch")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    async def _loop(self) -> None:
        while True:
            try:
                await self.watch_tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("external run watcher failed")
            await asyncio.sleep(WATCH_INTERVAL_S)


def _names_study(manifest: dict | None) -> bool:
    """The manifest's ``run_meta`` links the run to a study (a finished study child)."""
    return bool(((manifest or {}).get("run_meta") or {}).get("study_id"))


def _fresh(state: dict) -> bool:
    """A heartbeat under 2 minutes old: ``run_state.updated_utc``, or the run's progress file (its heartbeat
    event every 15 s)."""
    now = time.time()
    ts = parse_utc(state.get("updated_utc"))
    if ts is not None and now - ts <= EXTERNAL_FRESH_S:
        return True
    ev = state.get("events_path")
    if ev:
        try:
            return now - os.stat(ev).st_mtime <= EXTERNAL_FRESH_S
        except OSError:
            return False
    return False


def _pid_alive(state: dict) -> bool | None:
    """Whether the run's process is alive on this host: True, False (gone, or the pid now belongs to a process
    started after the run), None when it cannot be told (another host, no pid)."""
    import socket

    pid = state.get("pid")
    host = state.get("host")
    if (host and host != socket.gethostname()) or not pid:
        return None
    try:
        import psutil

        proc = psutil.Process(int(pid))
        if proc.status() == psutil.STATUS_ZOMBIE:
            return False
        started = parse_utc(state.get("started_utc"))
        return started is None or proc.create_time() <= started + 60.0
    except Exception as exc:
        if type(exc).__name__ in ("NoSuchProcess", "ZombieProcess"):
            return False
        return None


def run_summary(row: dict, studies: list[str] | None = None) -> dict:
    """A ``runs`` row as the api.md ``RunSummary`` (``studies``: the attached study ids).

    The project endpoints build the same object from the same row (``sparc.studio.projects.service.
    run_summary``); both read ``duration_s`` and ``n_scenarios`` from ``stages_json`` and fall back to the
    created/finished timestamps for the duration, so ``GET /api/runs`` and ``GET /api/projects/{pid}`` agree.
    """
    info = dbmod.loads(row.get("stages_json"), {}) or {}
    if not isinstance(info, dict):
        info = {}
    mode = str(row.get("mode") or "custom")
    mode = "coarse" if mode.startswith("coarse") else (mode if mode in ("fast", "coarse", "full", "custom")
                                                        else "custom")
    status = row.get("status") or "imported"
    origin = row.get("origin") or "imported"
    duration = info.get("duration_s")
    if duration is None:
        t0, t1 = parse_utc(row.get("created_utc")), parse_utc(row.get("finished_utc"))
        duration = round(t1 - t0, 3) if t0 is not None and t1 is not None else None
    return {
        "id": row["id"], "project_id": row.get("project_id"), "label": row.get("label"), "origin": origin,
        "status": status, "mode": mode, "coarse_m": row.get("coarse_m"), "created_utc": row.get("created_utc"),
        "finished_utc": row.get("finished_utc"), "duration_s": duration,
        "n_points": row.get("n_points"), "r2": row.get("r2"), "rmse": row.get("rmse"),
        "coverage": row.get("coverage"), "n_scenarios": info.get("n_scenarios"),
        "checkpoint_bytes": row.get("checkpoint_bytes"), "has_emulator": bool(row.get("has_emulator")),
        "studies": list(studies or []), "git_commit": row.get("git_commit"),
        "git_dirty": None if row.get("git_dirty") is None else bool(row.get("git_dirty")),
        "demo": bool(row.get("demo")), "pinned": bool(row.get("pinned")), "parent_run_id": row.get("parent_run_id"),
        "study_id": row.get("study_id"), "last_job_id": row.get("last_job_id"),
    }


# ---------------------------------------------------------------------------
# reindex (sparc studio --reindex)
# ---------------------------------------------------------------------------

def _reindex(db, workspace) -> dict:
    reg = Registry(db=db, workspace=workspace, hub=None)
    return reg.scan()


try:
    dbmod.reindex_hook("runs", order=10)(_reindex)
except Exception:  # pragma: no cover - the foundation always provides the hook registry
    log.exception("cannot register the runs reindex hook")
