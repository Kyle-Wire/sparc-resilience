"""Dose sweeps (SPEC §7.9, api.md §7.8).

``POST /api/runs/{rid}/sweeps`` writes ``<studio_dir>/sweeps/<swid>/params.json`` (lever, doses, optional
selection) and enqueues ``engine.sweep``, which runs the engine once per dose on the selection
(:func:`sparc.studio.engine.ops.op_sweep`), stores each point as a ``sweep_point`` result, fits
``response.fit_saturation`` on the region-mean benefit against the neighbourhood dose and writes
``curve.json``.  The pipeline's own S4 curve (``response_curves.json``) is returned for overlay.
"""

from __future__ import annotations

from pathlib import Path

from sparc.studio import db as dbmod
from sparc.studio.errors import ApiError
from sparc.studio.workspace import new_id, read_json, utc_now, write_json_atomic

__all__ = ["create", "sweep_row", "sweep_out", "list_sweeps", "delete", "sweep_status"]


def create(db, ctx, lever: str, doses: list[float], selection: dict | None) -> dict:
    act = ctx.cfg_raw.get("actionable") or {}
    if lever not in act:
        raise ApiError("validation", f"{lever!r} is not an actionable lever of this run",
                       detail={"errors": [{"path": "lever", "message": "not actionable", "code": "lever"}]})
    doses = sorted({abs(float(d)) for d in doses if float(d) != 0})
    if not doses:
        raise ApiError("validation", "a sweep needs at least one non-zero dose",
                       detail={"errors": [{"path": "doses", "message": "no doses", "code": "doses"}]})
    if selection is not None:
        from sparc.studio.runs.selection import RunSource, resolve

        mask = resolve(RunSource(ctx, db), selection)
        if not mask.any():
            raise ApiError("validation", "the sweep selection is empty",
                           detail={"errors": [{"path": "selection", "message": "no cells", "code": "empty"}]})
    swid = new_id("sweep")
    sdir = Path(ctx.studio_dir) / "sweeps" / swid
    sdir.mkdir(parents=True, exist_ok=True)
    now = utc_now()
    params = {"schema": 1, "id": swid, "run_id": ctx.run_id, "lever": lever, "doses": doses, "selection": selection,
              "direction": str((act.get(lever) or {}).get("direction", "increase")).lower(), "created_utc": now,
              "job_id": None}
    write_json_atomic(sdir / "params.json", params)
    db.insert("sweeps", {"id": swid, "run_id": ctx.run_id, "params_json": dbmod.dumps(params), "dir": str(sdir),
                         "job_id": None, "summary_json": None, "created_utc": now})
    return params


def sweep_row(db, swid: str) -> dict:
    row = db.fetchone("SELECT * FROM sweeps WHERE id = ?", (swid,))
    if row is None:
        raise ApiError("not_found", f"no sweep {swid!r}")
    return row


def sweep_status(db, row: dict) -> str:
    if row.get("job_id"):
        j = db.fetchone("SELECT status FROM jobs WHERE id = ?", (row["job_id"],))
        if j is not None:
            return str(j["status"])
    curve = read_json(Path(row["dir"]) / "curve.json") if row.get("dir") else None
    return "succeeded" if curve else "queued"


def sweep_out(db, ctx, row: dict) -> dict:
    params = dbmod.loads(row.get("params_json"), {}) or {}
    curve = read_json(Path(row["dir"]) / "curve.json") or {}
    resp = (ctx.json("response_curves.json") or {}).get(params.get("lever")) if ctx is not None else None
    return {"params": params, "status": sweep_status(db, row),
            "curve": [{k: v for k, v in p.items() if k != "result_id"} for p in curve.get("curve") or []],
            "fit": curve.get("fit"), "pipeline_curve": (resp or {}).get("curve") if isinstance(resp, dict) else None,
            "points": list(curve.get("points") or [])}


def list_sweeps(db, run_id: str) -> list[dict]:
    out = []
    for r in db.fetchall("SELECT * FROM sweeps WHERE run_id = ? ORDER BY created_utc DESC", (run_id,)):
        p = dbmod.loads(r.get("params_json"), {}) or {}
        out.append({"id": r["id"], "lever": p.get("lever") or "", "doses": [float(d) for d in p.get("doses") or []],
                    "status": sweep_status(db, r), "created_utc": r.get("created_utc") or "", "job_id": r.get("job_id")})
    return out


def delete(db, swid: str) -> None:
    from sparc.studio.engine import store
    from sparc.studio.schemas.common import ACTIVE_STATUSES

    row = sweep_row(db, swid)
    if row.get("job_id"):
        j = db.fetchone("SELECT status FROM jobs WHERE id = ?", (row["job_id"],))
        if j is not None and j["status"] in ACTIVE_STATUSES:
            raise ApiError("active", "the sweep is still running: cancel its job first",
                           detail={"job_id": row["job_id"], "status": j["status"]})
    curve = read_json(Path(row["dir"]) / "curve.json") or {}
    ids = set(curve.get("points") or [])
    for r in db.fetchall("SELECT id, dir FROM results WHERE kind = 'sweep_point' AND run_id = ?", (row["run_id"],)):
        spec = read_json(Path(r["dir"]) / "spec.json") if r.get("dir") else None
        if r["id"] in ids or (isinstance(spec, dict) and spec.get("sweep_id") == swid):
            store.delete_result(db, r["id"])
    store.rmtree_under(Path(row["dir"]), "sweeps")
    db.execute("DELETE FROM sweeps WHERE id = ?", (swid,))
