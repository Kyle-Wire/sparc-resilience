"""Result, plan, sweep and comparison directories of a run (api.md §12.2) and their database rows.

``<studio_dir>/results/<res_id>/``:

* ``spec.json`` - ``{scenario_id, revision, content_hash, compiled: {levers, edits, …}, run_id, ckpt_key,
  code_sha, kind, …}``;
* ``summary.json`` - the ``Result`` of api.md §7.5 without ``impacts``;
* ``cells.parquet`` - ``id, delta, delta_sd, extrapolation, realized_<var>…`` (float32);
* ``folds.npy`` - float32 K × n; ``warnings.json``; ``impacts_<hash>.json`` (cached by params).

**Cache key** = ``(content_hash, run_id, checkpoint mtime+size, core code sha)``: an identical request returns
the stored result.  When the run's checkpoint or the core code changes, older results are marked ``stale``;
they stay readable with a banner.

The ``results``, ``plans``, ``sweeps`` and ``comparisons`` tables are an index of these directories:
``sparc studio --reindex`` rebuilds them through the :func:`reindex_results` hook.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Any

import numpy as np

from sparc.studio import db as dbmod
from sparc.studio.db import reindex_hook
from sparc.studio.errors import ApiError
from sparc.studio.workspace import read_json, utc_now, write_json_atomic

log = logging.getLogger("sparc.studio.engine")

__all__ = ["result_dir", "plan_dir", "sweep_dir", "comparison_dir", "checkpoint_key", "code_sha", "write_cells",
           "write_result", "insert_result_row", "find_cached", "summary_out", "result_row", "load_result",
           "delete_result", "refresh_stale", "scenario_results", "run_results", "read_cells", "read_folds",
           "reindex_results", "result_array", "RESULT_FIELDS"]

RESULT_FIELDS = ("delta", "delta_sd", "extrapolation", "abs")


# ---------------------------------------------------------------------------
# paths and keys
# ---------------------------------------------------------------------------

def result_dir(studio_dir, res_id: str) -> Path:
    return Path(studio_dir) / "results" / res_id


def plan_dir(studio_dir, plid: str) -> Path:
    return Path(studio_dir) / "plans" / plid


def sweep_dir(studio_dir, swid: str) -> Path:
    return Path(studio_dir) / "sweeps" / swid


def comparison_dir(studio_dir, cid: str) -> Path:
    return Path(studio_dir) / "comparisons" / cid


def checkpoint_key(run_dir) -> str | None:
    from sparc.core.session import checkpoint_key as _ck

    return _ck(run_dir)


_CODE: tuple[tuple, str] | None = None
_CODE_LOCK = threading.Lock()


def code_sha() -> str:
    """The core code digest (``checkpoint.json.code_sha256``), cached by the sources' stat."""
    global _CODE
    import sparc.core.pipeline as pipeline

    files = pipeline._code_files()
    key = tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in files)
    with _CODE_LOCK:
        if _CODE is not None and _CODE[0] == key:
            return _CODE[1]
    sha = pipeline._code_digest()
    with _CODE_LOCK:
        _CODE = (key, sha)
    return sha


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

def write_cells(rdir: Path, ids, delta, delta_sd, extrapolation, realized: dict) -> None:
    import pandas as pd

    from sparc.core import runio

    cols: dict[str, Any] = {"id": np.asarray(ids)}
    cols["delta"] = np.asarray(delta, dtype=np.float32)
    cols["delta_sd"] = np.asarray(delta_sd if delta_sd is not None else np.full(len(cols["delta"]), np.nan),
                                  dtype=np.float32)
    cols["extrapolation"] = np.asarray(extrapolation if extrapolation is not None else
                                       np.zeros(len(cols["delta"])), dtype=np.float32)
    for var, v in realized.items():
        cols[f"realized_{var}"] = np.asarray(v, dtype=np.float32)
    runio.write_parquet_atomic(pd.DataFrame(cols), rdir / "cells.parquet")


def write_result(rdir: Path, *, ids, delta, delta_sd, extrapolation, folds, realized: dict, spec: dict,
                 summary: dict, warnings: list | None = None) -> None:
    """Write a whole result directory (``summary.json`` last: its presence marks a complete result)."""
    from sparc.core import runio

    rdir.mkdir(parents=True, exist_ok=True)
    write_cells(rdir, ids, delta, delta_sd, extrapolation, realized)
    if folds is not None:
        with runio.atomic_open(rdir / "folds.npy", "wb") as fh:
            np.save(fh, np.asarray(folds, dtype=np.float32), allow_pickle=False)
    write_json_atomic(rdir / "spec.json", spec)
    write_json_atomic(rdir / "warnings.json", list(warnings or []))
    write_json_atomic(rdir / "summary.json", summary)


# ---------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------

def _row_from_dir(rdir: Path) -> dict | None:
    spec = read_json(rdir / "spec.json")
    summ = read_json(rdir / "summary.json")
    if not isinstance(spec, dict) or not isinstance(summ, dict):
        return None
    s = summ.get("summary") or {}
    small = {"city": s.get("city"), "edited": s.get("edited"),
             "frac_extrapolated_edited": s.get("frac_extrapolated_edited"),
             "name": (summ.get("scenario") or {}).get("name") or spec.get("name"),
             "revision": spec.get("revision")}
    return {"id": rdir.name, "scenario_id": spec.get("scenario_id"), "run_id": spec.get("run_id"),
            "kind": spec.get("kind") or s.get("kind") or "exact", "content_hash": spec.get("content_hash"),
            "code_sha": spec.get("code_sha"), "ckpt_key": spec.get("ckpt_key"), "dir": str(rdir),
            "summary_json": dbmod.dumps(small), "has_folds": int(bool(s.get("has_folds"))), "stale": 0,
            "job_id": spec.get("job_id") or s.get("job_id"), "created_utc": spec.get("created_utc") or s.get("created_utc")}


def insert_result_row(db, rdir: Path) -> dict | None:
    """Index a complete result directory (idempotent); returns the row."""
    row = _row_from_dir(Path(rdir))
    if row is None:
        return None
    db.insert("results", row, replace=True)
    return row


def result_row(db, res_id: str) -> dict:
    row = db.fetchone("SELECT * FROM results WHERE id = ?", (res_id,))
    if row is None:
        raise ApiError("not_found", f"no result {res_id!r}")
    return row


def find_cached(db, run_id: str, content_hash: str, ckpt: str | None, code: str | None,
                kind: str = "exact") -> dict | None:
    """The newest non-stale result with this cache key whose files are still there."""
    rows = db.fetchall("SELECT * FROM results WHERE run_id = ? AND content_hash = ? AND ckpt_key IS ? AND "
                       "code_sha IS ? AND kind = ? AND stale = 0 ORDER BY created_utc DESC",
                       (run_id, content_hash, ckpt, code, kind))
    for r in rows:
        if r.get("dir") and (Path(r["dir"]) / "summary.json").exists():
            return r
    return None


def summary_out(row: dict, unit: str = "°F") -> dict:
    """A ``results`` row as the api.md ``ResultSummary``."""
    from sparc.studio.runs.common import likely

    s = dbmod.loads(row.get("summary_json"), {}) or {}
    city = s.get("city") or likely(0.0, None, unit)
    return {"id": row["id"], "scenario_id": row.get("scenario_id"), "run_id": row["run_id"], "kind": row["kind"],
            "created_utc": row.get("created_utc") or "", "stale": bool(row.get("stale")),
            "has_folds": bool(row.get("has_folds")), "city": city, "edited": s.get("edited"),
            "frac_extrapolated_edited": s.get("frac_extrapolated_edited"), "job_id": row.get("job_id")}


def refresh_stale(db, run_id: str, run_dir) -> int:
    """Mark results of ``run_id`` made from another checkpoint or core code ``stale``; returns how many changed.

    Scenarios whose exact results all went stale become ``stale`` themselves."""
    ck = checkpoint_key(run_dir)
    code = code_sha()
    rows = db.fetchall("SELECT id, scenario_id, ckpt_key, code_sha, stale FROM results WHERE run_id = ?", (run_id,))
    changed = []
    for r in rows:
        stale = int(r.get("ckpt_key") != ck or r.get("code_sha") != code)
        if stale != int(r.get("stale") or 0):
            changed.append((stale, r["id"]))
    if changed:
        db.executemany("UPDATE results SET stale = ? WHERE id = ?", changed)
        sids = {r["scenario_id"] for r in rows if r.get("scenario_id")}
        try:
            from sparc.studio.scenarios import library

            for sid in sids:
                library.sync_status(db, sid)
        except Exception:
            log.exception("scenario status refresh failed")
    return len(changed)


def scenario_results(db, sid: str, unit: str = "°F") -> list[dict]:
    rows = db.fetchall("SELECT * FROM results WHERE scenario_id = ? ORDER BY created_utc DESC", (sid,))
    return [summary_out(r, unit) for r in rows]


def run_results(db, run_id: str, unit: str = "°F", kinds: tuple[str, ...] | None = None) -> list[dict]:
    rows = db.fetchall("SELECT * FROM results WHERE run_id = ? ORDER BY created_utc DESC", (run_id,))
    return [summary_out(r, unit) for r in rows if kinds is None or r["kind"] in kinds]


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def read_cells(rdir: Path):
    import pandas as pd

    p = Path(rdir) / "cells.parquet"
    if not p.exists():
        raise ApiError("not_found", f"result {Path(rdir).name} has no cells")
    return pd.read_parquet(p)


def read_folds(rdir: Path) -> np.ndarray | None:
    p = Path(rdir) / "folds.npy"
    if not p.exists():
        return None
    return np.load(p, allow_pickle=False)


def result_array(row: dict, field: str, obs: np.ndarray | None = None) -> np.ndarray:
    """``delta``, ``delta_sd``, ``extrapolation``, ``abs`` (obs + Δ) or ``realized_<var>`` as float32[n]."""
    df = read_cells(Path(row["dir"]))
    if field == "abs":
        if obs is None:
            raise ApiError("unknown_layer", "the absolute temperature needs the run's observations")
        return (np.asarray(obs, dtype=np.float64) + df["delta"].to_numpy(np.float64)).astype(np.float32)
    if field not in df.columns:
        raise ApiError("unknown_layer", f"result {row['id']} has no field {field!r}")
    return df[field].to_numpy(np.float32)


def load_result(db, row: dict) -> dict:
    """The stored ``Result`` (``summary.json``) with the current ``stale`` flag and cached default impacts."""
    rdir = Path(row["dir"])
    res = read_json(rdir / "summary.json")
    if not isinstance(res, dict):
        raise ApiError("not_found", f"result {row['id']} is incomplete (no summary.json)")
    stale = bool(row.get("stale"))
    res["stale"] = stale
    summ = dict(res.get("summary") or {})
    summ["stale"] = stale
    res["summary"] = summ
    res.setdefault("impacts", None)
    default = rdir / "impacts_default.json"
    if res.get("impacts") is None and default.exists():
        res["impacts"] = read_json(default)
    return res


def delete_result(db, res_id: str) -> None:
    row = result_row(db, res_id)
    if row.get("dir"):
        d = Path(row["dir"])
        if d.is_dir() and d.parent.name == "results":
            shutil.rmtree(d, ignore_errors=True)
    db.execute("DELETE FROM results WHERE id = ?", (res_id,))
    if row.get("scenario_id"):
        try:
            from sparc.studio.scenarios import library

            library.sync_status(db, row["scenario_id"])
        except Exception:
            log.exception("scenario status refresh failed")


# ---------------------------------------------------------------------------
# reindex (SPEC §10.6)
# ---------------------------------------------------------------------------

def _plan_row(pdir: Path, run: dict) -> dict | None:
    params = read_json(pdir / "params.json")
    if not isinstance(params, dict):
        return None
    planned = read_json(pdir / "planned.json") or {}
    realised = read_json(pdir / "realised.json") or {}
    return {"id": pdir.name, "project_id": params.get("project_id") or run.get("project_id"),
            "run_id": params.get("run_id") or run["id"], "name": params.get("name") or pdir.name,
            "params_json": dbmod.dumps(params.get("params") or {}), "summary_json": dbmod.dumps(planned),
            "dir": str(pdir), "verified_result_id": realised.get("result_id"),
            "created_utc": params.get("created_utc")}


def _sweep_row(sdir: Path, run: dict) -> dict | None:
    params = read_json(sdir / "params.json")
    if not isinstance(params, dict):
        return None
    curve = read_json(sdir / "curve.json")
    return {"id": sdir.name, "run_id": params.get("run_id") or run["id"], "params_json": dbmod.dumps(params),
            "dir": str(sdir), "job_id": params.get("job_id"), "summary_json": dbmod.dumps(curve),
            "created_utc": params.get("created_utc")}


def _comparison_row(cdir: Path, run: dict) -> dict | None:
    items = read_json(cdir / "items.json")
    if not isinstance(items, dict):
        return None
    summ = read_json(cdir / "summary.json")
    return {"id": cdir.name, "run_id": items.get("run_id") or run["id"], "items_json": dbmod.dumps(items.get("items")),
            "dir": str(cdir), "summary_json": dbmod.dumps(summ), "created_utc": items.get("created_utc")}


@reindex_hook("engine.results", order=40)
def reindex_results(db, workspace) -> dict:
    """Rebuild ``results``, ``plans``, ``sweeps`` and ``comparisons`` from every indexed run's ``studio/``."""
    counts = {"results": 0, "plans": 0, "sweeps": 0, "comparisons": 0}
    for run in db.fetchall("SELECT id, project_id, run_dir, studio_dir FROM runs"):
        sd = Path(run["studio_dir"]) if run.get("studio_dir") else Path(run["run_dir"]) / "studio"
        for sub, fn, table in (("results", None, "results"), ("plans", _plan_row, "plans"),
                               ("sweeps", _sweep_row, "sweeps"), ("comparisons", _comparison_row, "comparisons")):
            root = sd / sub
            if not root.is_dir():
                continue
            for d in sorted(p for p in root.iterdir() if p.is_dir()):
                try:
                    row = _row_from_dir(d) if fn is None else fn(d, run)
                    if row is None:
                        continue
                    if table == "results" and not row.get("run_id"):
                        row["run_id"] = run["id"]
                    db.insert(table, row, replace=True)
                    counts[table] += 1
                except Exception:
                    log.exception("reindex: skipping %s", d)
        try:
            refresh_stale(db, run["id"], run["run_dir"])
        except Exception:
            log.exception("reindex: stale check of %s failed", run["id"])
    return counts


def write_json(path: Path, obj) -> None:
    write_json_atomic(path, obj)


def now() -> str:
    return utc_now()


def rmtree_under(path: Path, parent_name: str) -> None:
    """Remove ``path`` when it is a child directory named under ``…/<parent_name>/``."""
    p = Path(path)
    if p.is_dir() and p.parent.name == parent_name:
        shutil.rmtree(p, ignore_errors=True)


def file_exists(path) -> bool:
    return os.path.exists(path)
