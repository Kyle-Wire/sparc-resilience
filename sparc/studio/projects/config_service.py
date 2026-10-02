"""Project config: YAML round trip, versions, ``If-Match`` concurrency, sections, impact preview (SPEC §9.4).

The project's ``config.yml`` is the source of truth; ``config_versions``
(SQLite) is its history.  Every save writes the file atomically and records
a version row.  The file is always written with a top-level ``core:`` block
(other top-level keys are kept); comments survive a save of YAML text that
already has a ``core:`` block (it is written as given) and are lost by
section edits and ``{raw}`` saves (``comments_preserved: false``).

Writers of ``config.yml`` - the server and the ``input.*`` job workers that
link a fetched file - hold :func:`config_lock` around read-modify-write.  A
worker cannot write SQLite, so it leaves a note sidecar
(``.config_note.json``, keyed by the text's hash); :func:`sync_versions`
records any change of the file that has no version row yet (a worker link or
an edit outside Studio) before every read and write, so ``If-Match`` always
compares against the file on disk.
"""

from __future__ import annotations

import copy
import difflib
import hashlib
import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import yaml

from sparc.core import runio
from sparc.studio.errors import ApiError
from sparc.studio.workspace import read_json, utc_now

log = logging.getLogger("sparc.studio.projects")

__all__ = [
    "CONFIG_NAME", "NOTE_NAME", "PATH_KEYS", "parse_yaml", "core_block", "dump_config", "merged_raw",
    "build_core_config", "effective_config", "changed_from_defaults", "config_lock", "read_text", "sync_versions",
    "current_version", "get_config", "save_config", "patch_section", "history", "history_version",
    "edit_config_file", "parse_if_match", "yaml_diff", "absolutize_paths", "path_keys", "get_dotted", "set_dotted",
    "impact", "SECTION_STAGE",
]

CONFIG_NAME = "config.yml"
NOTE_NAME = ".config_note.json"
#: config keys holding file paths (resolved against the project folder)
PATH_KEYS = ("data.path", "physics.forcing", "climate.table", "planner.layers")
_STAGE_ORDER = ("S1", "S2_S3", "baselines", "cv_curve", "S4", "S5", "climate", "S6", "S7")
#: first stage a changed fingerprint section invalidates (S0 always runs)
SECTION_STAGE = {"data": "S1", "code": "S1", "core": "S1", "s4": "S4", "s5": "S5", "climate": "climate",
                 "s6": "S6", "s7": "S7"}
SECTION_LABELS = {"data": "input data", "code": "core code", "core": "model setup", "s4": "levers and responses",
                  "s5": "scenarios", "climate": "climate", "s6": "causal settings", "s7": "optimisation"}


# ---------------------------------------------------------------------------
# YAML
# ---------------------------------------------------------------------------

def parse_yaml(text: str) -> dict:
    """The YAML document as a mapping; ``422 yaml_error`` (``detail: {line, column}``, 1-based) otherwise."""
    try:
        doc = yaml.safe_load(text) if text and text.strip() else {}
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None) or getattr(exc, "context_mark", None)
        line = int(mark.line) + 1 if mark is not None else None
        col = int(mark.column) + 1 if mark is not None else None
        problem = getattr(exc, "problem", None) or str(exc)
        raise ApiError("yaml_error", f"YAML error{f' at line {line}' if line else ''}: {problem}",
                       detail={"line": line, "column": col})
    if doc is None:
        doc = {}
    if not isinstance(doc, dict):
        raise ApiError("yaml_error", "the config must be a YAML mapping (a 'core:' block)",
                       detail={"line": 1, "column": 1})
    block = doc.get("core", doc)
    if block is None:
        block = {}
    if not isinstance(block, dict):
        raise ApiError("yaml_error", "the 'core:' block must be a mapping", detail={"line": 1, "column": 1})
    return doc


def core_block(doc: dict) -> dict:
    """The ``core:`` block of a parsed config (or the document itself when it has none)."""
    block = doc.get("core", doc) if isinstance(doc, dict) else {}
    return dict(block or {})


def dump_config(raw: dict, others: dict | None = None) -> str:
    """YAML text with the ``core:`` block first, then any other top-level keys."""
    doc = {"core": raw}
    for k, v in (others or {}).items():
        if k != "core":
            doc[k] = v
    return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, default_flow_style=False, width=110)


def _others(doc: dict) -> dict:
    return {k: v for k, v in doc.items() if k != "core"} if "core" in doc else {}


def normalize_text(text: str) -> tuple[str, dict]:
    """Text to store for a YAML save: as given when it has a ``core:`` block, else wrapped into one."""
    doc = parse_yaml(text)
    if "core" in doc:
        return (text if text.endswith("\n") else text + "\n"), core_block(doc)
    return dump_config(dict(doc)), dict(doc)


def yaml_diff(old: str, new: str, old_label: str = "config.yml", new_label: str = "config.yml") -> str:
    return "".join(difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True),
                                        fromfile=old_label, tofile=new_label))


# ---------------------------------------------------------------------------
# dotted paths and the effective config
# ---------------------------------------------------------------------------

def get_dotted(raw: dict, path: str, default: Any = None) -> Any:
    cur: Any = raw
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return default
    return cur


def set_dotted(raw: dict, path: str, value: Any) -> dict:
    """Set ``a.b.c`` in place (intermediate mappings are created; a None section becomes a mapping).

    A digit part steps into an existing list (``data.join.0.path``) rather than
    replacing the list with a mapping; an index past the end raises ``IndexError``.
    """
    parts = path.split(".")
    cur: Any = raw
    for i, part in enumerate(parts[:-1]):
        keep_list = parts[i + 1].isdigit()
        if isinstance(cur, list):
            nxt = cur[int(part)]
        else:
            nxt = cur.get(part)
        if not (isinstance(nxt, dict) or (keep_list and isinstance(nxt, list))):
            nxt = {}
            if isinstance(cur, list):
                cur[int(part)] = nxt
            else:
                cur[part] = nxt
        cur = nxt
    if isinstance(cur, list):
        cur[int(parts[-1])] = value
    else:
        cur[parts[-1]] = value
    return raw


def path_keys(raw: dict) -> list[str]:
    """Dotted keys of ``raw`` that hold file paths (``data.join.<i>.path`` included), set or not."""
    keys = list(PATH_KEYS)
    data = raw.get("data") if isinstance(raw, dict) else None
    joins = data.get("join") if isinstance(data, dict) else None
    for i, _j in enumerate(joins if isinstance(joins, list) else []):
        keys.append(f"data.join.{i}.path")
    return keys


def absolutize_paths(raw: dict, base_dir: str | os.PathLike) -> dict:
    """A copy of ``raw`` whose path keys are absolute (resolved against ``base_dir``)."""
    out = copy.deepcopy(raw)
    base = Path(base_dir)
    for key in path_keys(out):
        v = get_dotted(out, key)
        if isinstance(v, str) and v.strip() and not Path(v).is_absolute():
            set_dotted(out, key, str((base / v).resolve()))
    return out


def merged_raw(raw: dict) -> dict:
    """DEFAULTS deep-merged with ``raw`` (core's own merge, keys stringified)."""
    from sparc.core.config import DEFAULTS, _deep_merge
    from sparc.core.provenance import _str_keys

    return _deep_merge(DEFAULTS, _str_keys(raw or {}))


def build_core_config(raw: dict, base_dir: str | os.PathLike, *, apply_forcing: bool = True):
    """A ``CoreConfig`` of ``raw`` without core's ``validate()`` (a config may still be incomplete).

    Like ``core_config_from_dict`` it applies ``physics.forcing`` - when the file
    is readable; a missing or broken file is left for ``validate_deep`` to report.
    """
    from sparc.core.config import CoreConfig

    cfg = CoreConfig(raw=merged_raw(raw), base_dir=Path(base_dir))
    forcing = cfg.raw["physics"].get("forcing") if isinstance(cfg.raw.get("physics"), dict) else None
    if apply_forcing and forcing:
        from sparc.core.forcing import apply_forcing_file

        try:
            cfg.raw["physics"] = apply_forcing_file(cfg.raw["physics"], cfg.resolve_path(forcing),
                                                    label=str(forcing))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            log.debug("forcing %s not applied: %s", forcing, exc)
    return cfg


def effective_config(raw: dict, base_dir: str | os.PathLike) -> dict:
    """What core would run with: DEFAULTS + ``raw`` + the forcing file's values."""
    return build_core_config(raw, base_dir).raw


def changed_from_defaults(raw: dict) -> list[dict]:
    """``[{path, value, default}]`` for every leaf of ``raw`` that differs from DEFAULTS (keys DEFAULTS lacks
    have ``default: null``).  Sub-mappings that exist in DEFAULTS are walked into."""
    from sparc.core.config import DEFAULTS
    from sparc.core.provenance import _str_keys

    out: list[dict] = []

    def same(a, b) -> bool:
        if isinstance(a, tuple):
            a = list(a)
        if isinstance(b, tuple):
            b = list(b)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) \
                and not isinstance(b, bool):
            return float(a) == float(b)
        return a == b

    def walk(r: dict, d: Any, prefix: str) -> None:
        for k, v in r.items():
            p = f"{prefix}{k}"
            dv = d.get(k) if isinstance(d, dict) else None
            if isinstance(v, dict) and isinstance(dv, dict) and dv:
                walk(v, dv, p + ".")
            elif not same(v, dv):
                out.append({"path": p, "value": v, "default": dv})

    walk(_str_keys(raw or {}), DEFAULTS, "")
    return out


def sections_hash(raw: dict) -> dict:
    """Per-top-level-key hashes of a raw config (``config_versions.sections_json``)."""
    from sparc.core.provenance import _str_keys

    return {str(k): hashlib.sha1(json.dumps(_str_keys(v), sort_keys=True, default=str).encode()).hexdigest()[:12]
            for k, v in (raw or {}).items()}


# ---------------------------------------------------------------------------
# file, lock and versions
# ---------------------------------------------------------------------------

@contextmanager
def config_lock(project_dir: str | os.PathLike, timeout: float = 60.0) -> Iterator[None]:
    """Exclusive lock on a project's config (threads and processes): ``<project>/.sparc.lock``."""
    with runio.run_lock(project_dir, timeout=timeout):
        yield


def config_path_of(project: dict) -> Path:
    return Path(project.get("config_path") or Path(project["dir"]) / CONFIG_NAME)


def read_text(project: dict) -> str:
    try:
        return config_path_of(project).read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_note(project_dir: str | os.PathLike, text: str, note: str) -> None:
    """Leave the note a later :func:`sync_versions` records for ``text`` (used by job workers)."""
    runio.write_json_atomic(Path(project_dir) / NOTE_NAME, {"sha256": _sha(text), "note": note,
                                                            "written_utc": utc_now()})


def _note_for(project_dir: Path, text: str) -> str | None:
    side = read_json(project_dir / NOTE_NAME)
    if isinstance(side, dict) and side.get("sha256") == _sha(text):
        return str(side.get("note") or "") or None
    return None


def _insert_version(conn, pid: str, version: int, text: str, note: str | None) -> None:
    try:
        raw = core_block(parse_yaml(text)) if text.strip() else {}
    except ApiError:
        raw = {}
    conn.execute("INSERT INTO config_versions (project_id, version, yaml, note, saved_utc, sections_json) "
                 "VALUES (?, ?, ?, ?, ?, ?)", (pid, int(version), text, note, utc_now(), json.dumps(sections_hash(raw))))


def sync_versions(db, project: dict, *, note: str | None = None) -> tuple[int, str]:
    """Record the file's current text as a new version when it differs from the latest one.

    Returns ``(current_version, text)``.  The first sync of a project records
    version 1.  The note is ``note``, else the worker's note sidecar for this
    text, else "edited outside Studio".  Runs under :func:`config_lock`, so it
    never sees a save half done (the file written, its row not yet).
    """
    pid = project["id"]
    pdir = Path(project["dir"])
    if not pdir.is_dir():                        # a deleted project: nothing to record (and no lock file to create)
        return current_version(db, pid), ""

    def fn(conn, text: str) -> int:
        row = conn.execute("SELECT version, yaml FROM config_versions WHERE project_id = ? "
                           "ORDER BY version DESC LIMIT 1", (pid,)).fetchone()
        if row is not None and row["yaml"] == text:
            return int(row["version"])
        version = int(row["version"]) + 1 if row is not None else 1
        why = note or _note_for(pdir, text) or ("created" if row is None else "edited outside Studio")
        _insert_version(conn, pid, version, text, why)
        return version

    with config_lock(pdir):
        text = read_text(project)
        return int(db.run(lambda conn: fn(conn, text))), text


def current_version(db, pid: str) -> int:
    return int(db.fetchval("SELECT MAX(version) FROM config_versions WHERE project_id = ?", (pid,), default=0) or 0)


def parse_if_match(value: str | None) -> int | None:
    """``If-Match: 3`` (also ``"3"`` and ``W/"3"``) → 3; None when absent; ``422 validation`` when malformed."""
    if value is None or not str(value).strip():
        return None
    v = str(value).strip()
    if v.startswith("W/"):
        v = v[2:]
    v = v.strip('"').strip()
    if v == "*":
        return None
    try:
        return int(v)
    except ValueError:
        raise ApiError("validation", "If-Match must be a config version number",
                       detail={"errors": [{"path": "If-Match", "message": "not a version", "code": "bad_if_match"}]})


def get_config(db, project: dict) -> dict:
    """``GET /api/projects/{pid}/config``."""
    version, text = sync_versions(db, project)
    try:
        raw = core_block(parse_yaml(text)) if text.strip() else {}
    except ApiError:                     # broken YAML written outside Studio: show the text so it can be fixed
        raw = {}
    return {"version": version, "yaml": text, "raw": raw,
            "effective": effective_config(raw, project["dir"]),
            "changed_from_defaults": changed_from_defaults(raw), "comments_preserved": False}


def _write(db, project: dict, new_text: str, *, if_match: int | None, note: str | None) -> dict:
    """Write ``new_text`` as the next version under the config lock; ``409 conflict`` on a stale ``if_match``."""
    pdir = Path(project["dir"])
    with config_lock(pdir):
        version, old_text = sync_versions(db, project)
        if if_match is not None and int(if_match) != version:
            raise ApiError("conflict", f"the config changed since version {if_match} (now version {version})",
                           detail={"current_version": version})
        if new_text == old_text:
            return {"version": version, "diff": "", "text": new_text}
        runio.write_text_atomic(config_path_of(project), new_text)

        def fn(conn) -> int:
            latest = conn.execute("SELECT MAX(version) AS v FROM config_versions WHERE project_id = ?",
                                  (project["id"],)).fetchone()["v"] or 0
            _insert_version(conn, project["id"], int(latest) + 1, new_text, note or "saved")
            return int(latest) + 1

        new_version = db.run(fn)
    return {"version": new_version, "diff": yaml_diff(old_text, new_text, f"config.yml@v{version}",
                                                       f"config.yml@v{new_version}"), "text": new_text}


def save_config(db, project: dict, *, text: str | None = None, raw: dict | None = None,
                if_match: int | None = None, note: str | None = None,
                validate: Callable[[dict], list] | None = None) -> dict:
    """``PUT /api/projects/{pid}/config``: ``{version, issues, diff}`` (validation never blocks a save)."""
    if text is None and raw is None:
        raise ApiError("validation", "send yaml or raw",
                       detail={"errors": [{"path": "yaml", "message": "yaml or raw is required", "code": "missing"}]})
    if text is not None:
        new_text, new_raw = normalize_text(text)
    else:
        if not isinstance(raw, dict):
            raise ApiError("validation", "raw must be an object",
                           detail={"errors": [{"path": "raw", "message": "not an object", "code": "type"}]})
        block = raw.get("core", raw) if isinstance(raw.get("core"), dict) else raw
        others = _others(parse_yaml(read_text(project)) if read_text(project).strip() else {})
        new_raw = dict(block)
        new_text = dump_config(new_raw, others)
    out = _write(db, project, new_text, if_match=if_match, note=note)
    issues = validate(new_raw) if validate is not None else []
    return {"version": out["version"], "issues": issues, "diff": out["diff"]}


def patch_section(db, project: dict, section: str, value: Any, *, if_match: int | None = None,
                  note: str | None = None, validate: Callable[[dict], list] | None = None) -> dict:
    """``PATCH /config/sections/{section}``: replace one top-level key (``null`` removes it → default)."""
    from sparc.studio.projects.config_schema import SECTIONS

    if section not in SECTIONS:
        raise ApiError("validation", f"unknown config section {section!r}",
                       detail={"errors": [{"path": "section", "message": f"one of {', '.join(SECTIONS)}",
                                           "code": "unknown_section"}]})
    pdir = Path(project["dir"])
    with config_lock(pdir):
        _version, text = sync_versions(db, project)
        doc = parse_yaml(text) if text.strip() else {}
        raw = core_block(doc)
        if value is None:
            raw.pop(section, None)
        else:
            raw[section] = value
        new_text = dump_config(raw, _others(doc))
        out = _write(db, project, new_text, if_match=if_match, note=note or f"section {section}")
    issues = validate(raw) if validate is not None else []
    return {"version": out["version"], "issues": issues, "diff": out["diff"]}


def history(db, pid: str) -> list[dict]:
    return db.fetchall("SELECT version, saved_utc, note FROM config_versions WHERE project_id = ? "
                       "ORDER BY version DESC", (pid,))


def history_version(db, pid: str, version: int) -> dict:
    row = db.fetchone("SELECT version, yaml FROM config_versions WHERE project_id = ? AND version = ?",
                      (pid, int(version)))
    if row is None:
        raise ApiError("not_found", f"no config version {version}")
    return row


def edit_config_file(project_dir: str | os.PathLike, mutate: Callable[[dict], Any], note: str,
                     config_path: str | os.PathLike | None = None) -> tuple[str, str]:
    """Read-modify-write ``config.yml`` under the lock without the database (job workers).

    ``mutate(raw)`` edits the ``core:`` block in place.  The new text gets a
    note sidecar so the server records it as a version with ``note``.
    Returns ``(old_text, new_text)``; unchanged text writes nothing.
    """
    pdir = Path(project_dir)
    path = Path(config_path) if config_path else pdir / CONFIG_NAME
    with config_lock(pdir):
        try:
            old = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            old = ""
        doc = parse_yaml(old) if old.strip() else {}
        raw = core_block(doc)
        mutate(raw)
        new = dump_config(raw, _others(doc))
        if new != old:
            write_note(pdir, new, note)
            runio.write_text_atomic(path, new)
    return old, new


# ---------------------------------------------------------------------------
# impact preview (SPEC §5.9)
# ---------------------------------------------------------------------------

def _run_args(run_dir: Path) -> dict:
    """Mode arguments of a run: Studio's ``launch.json`` args, else the manifest's fast flag."""
    launch = read_json(run_dir / "studio" / "launch.json") or {}
    args = dict(launch.get("args") or {})
    if "fast" not in args:
        m = read_json(run_dir / "manifest.json") or {}
        if "fast_mode" in m:
            args["fast"] = bool(m["fast_mode"])
        elif "fast" in m:
            args["fast"] = bool(m["fast"])
    return args


def _project_runs(db, project: dict) -> list[dict]:
    """Runs of the project that have a ``checkpoint.json``: indexed rows plus run folders under ``runs/``."""
    out: dict[str, dict] = {}
    try:
        rows = db.fetchall("SELECT id, run_dir, label FROM runs WHERE project_id = ?", (project["id"],))
    except Exception:
        rows = []
    for r in rows:
        d = Path(r["run_dir"])
        if (d / "checkpoint.json").is_file():
            out[str(d.resolve())] = {"run_id": r["id"], "label": r.get("label") or r["id"], "run_dir": d}
    runs_dir = Path(project["dir"]) / "runs"
    if runs_dir.is_dir():
        for d in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
            key = str(d.resolve())
            if key not in out and (d / "checkpoint.json").is_file():
                out[key] = {"run_id": d.name, "label": d.name, "run_dir": d}
    return list(out.values())


def refit_from(sections: list[str]) -> str | None:
    stages = [SECTION_STAGE[s] for s in sections if s in SECTION_STAGE]
    if not stages:
        return None
    return min(stages, key=_STAGE_ORDER.index)


def _phrase(changed: list[str], stage: str | None) -> str:
    if not changed:
        return "No fitted stage is affected: a re-run with this config would reuse this run's fits."
    what = ", ".join(SECTION_LABELS.get(s, s) for s in changed)
    if stage == "S7":
        return f"Only the budget optimisation would re-run ({what} changed)."
    return f"A re-run with this config would refit from {stage} ({what} changed)."


def _config_sections(cfg) -> dict[str, str] | None:
    """``fingerprint_sections`` of a config as a re-run without mode arguments would see it, or None when
    core cannot fingerprint it (a section of the wrong type: validation reports it).

    A config without ``data.path`` (every new blank project) has no input identity; its ``data`` section is
    the hash of an empty table, so the other sections still compare and adding the data file shows as a
    ``data`` change.
    """
    import pandas as pd

    from sparc.core.pipeline import fingerprint_sections

    try:
        no_data = not (isinstance(cfg.raw.get("data"), dict) and cfg.raw["data"].get("path"))
        return fingerprint_sections(cfg, False, pd.DataFrame() if no_data else None)
    except (ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
        log.info("impact: cannot fingerprint the config: %s", exc)
        return None


def launch_raw(raw: dict, project_dir: str | os.PathLike, workspace=None) -> dict:
    """``raw`` as a Studio launch would snapshot it (SPEC §4.3): path keys absolute, ``climate.cache`` the
    workspace cache, ``output.dir`` the project's ``runs`` folder."""
    out = absolutize_paths(raw, project_dir)
    if workspace is not None:                    # resolved, as the launch snapshot writes it
        set_dotted(out, "climate.cache", str(Path(workspace.cache_dir).resolve()))
    set_dotted(out, "output.dir", str((Path(project_dir) / "runs").resolve()))
    return out


def impact(db, project: dict, edited_raw: dict, workspace=None) -> dict:
    """``POST /config/impact``: which fingerprint sections the edit changes, and what each run would refit.

    Compares ``fingerprint_sections`` of the edited config with each project
    run's ``checkpoint.json`` sections (the pickle is never opened).  Runs
    launched by Studio are compared in their launch-snapshot form (absolute
    paths), others in the project's own form.
    """
    import pandas as pd

    from sparc.core.pipeline import FINGERPRINT_SECTIONS, fingerprint_sections

    pdir = Path(project["dir"])
    text = read_text(project)
    try:
        cur_raw = core_block(parse_yaml(text)) if text.strip() else {}
    except ApiError:                     # broken YAML written outside Studio: nothing to compare with
        cur_raw = None
    cur = _config_sections(build_core_config(cur_raw, pdir)) if cur_raw is not None else None
    new_cfg = build_core_config(edited_raw, pdir)
    new = _config_sections(new_cfg)
    if cur is None or new is None:       # a config core cannot fingerprint: every section may change
        changed = list(FINGERPRINT_SECTIONS)
    else:
        changed = [s for s in FINGERPRINT_SECTIONS if cur.get(s) != new.get(s)]
    launch_cfg = None
    runs = []
    for r in _project_runs(db, project):
        d = Path(r["run_dir"])
        side = read_json(d / "checkpoint.json") or {}
        old = side.get("sections") or {}
        args = _run_args(d)
        frame = None
        if (d / "input_frame.parquet").is_file():
            try:
                frame = pd.read_parquet(d / "input_frame.parquet")
            except Exception:
                frame = None
        try:
            if (d / "studio" / "launch.json").is_file():
                if launch_cfg is None:
                    launch_cfg = build_core_config(launch_raw(edited_raw, pdir, workspace), pdir)
                cfg = launch_cfg
            else:
                cfg = new_cfg
            sec = fingerprint_sections(cfg, bool(args.get("fast", False)), frame, coarse=args.get("coarse"),
                                       cv_curve=args.get("cv_curve"))
        except Exception as exc:                 # an unreadable config cannot be compared; say so
            log.info("impact: cannot fingerprint %s: %s", d, exc)
            runs.append({"run_id": r["run_id"], "label": r["label"], "checkpoint_done": list(side.get("done") or []),
                         "changed_sections": list(FINGERPRINT_SECTIONS), "refit_from": "S1",
                         "phrase": f"The edited config cannot be compared with this run ({exc})."})
            continue
        run_changed = [s for s in FINGERPRINT_SECTIONS if old.get(s) != sec.get(s)]
        stage = refit_from(run_changed)
        runs.append({"run_id": r["run_id"], "label": r["label"], "checkpoint_done": list(side.get("done") or []),
                     "changed_sections": run_changed, "refit_from": stage, "phrase": _phrase(run_changed, stage)})
    return {"changed_sections": changed, "runs": runs}
