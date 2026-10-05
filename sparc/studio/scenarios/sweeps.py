"""Dose sweeps (SPEC §7.9, api.md §7.8).

``POST /api/runs/{rid}/sweeps`` writes ``<studio_dir>/sweeps/<swid>/params.json`` (lever, doses, optional
selection) and enqueues ``engine.sweep``, which runs the engine once per dose on the selection
(:func:`sparc.studio.engine.ops.op_sweep`), stores each point as a ``sweep_point`` result, fits
``response.fit_saturation`` on the region-mean benefit and writes ``curve.json``.  The pipeline's own S4 curve
(``response_curves.json``) is returned for overlay.

Two fits, each on its own dose axis (:func:`fit_curve`): ``fit`` against the **requested dose** - the x axis
the curve, the overlay and the pipeline's curve are drawn on - and ``fit_neighbourhood`` against the mean
**neighbourhood dose** over the selection (the Gaussian-smoothed dose around each cell, SPEC §7.9), with each
point's ``neighbourhood_dose``.  For a regional sweep the neighbourhood dose is a fraction of the requested
one, so its ``d_s`` and ``d90`` are on a different scale and are never drawn on the requested-dose axis.
A ``curve.json`` written before the fits named their ``axis`` holds the neighbourhood fit as ``fit``:
:func:`sweep_out` reports it as ``fit_neighbourhood`` and refits ``fit`` on the requested doses of its points.
"""

from __future__ import annotations

from pathlib import Path

from sparc.studio import db as dbmod
from sparc.studio.errors import ApiError
from sparc.studio.workspace import new_id, read_json, utc_now, write_json_atomic

__all__ = ["create", "sweep_row", "sweep_out", "list_sweeps", "delete", "sweep_status", "fit_curve"]


def fit_curve(x, y, axis: str) -> dict | None:
    """``response.fit_saturation`` of benefit ``y`` (positive = cooler) on doses ``x`` (both starting at the
    untreated point 0, 0): ``{model, A, ds, d90, axis}`` with ``ds``/``d90`` in the units of ``axis``; None with
    fewer than three points."""
    import numpy as np

    from sparc.core.response import fit_saturation

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return None
    f = fit_saturation(x[:, None], y[:, None], np.ones((x.size, 1), dtype=bool), min_valid=min(4, x.size))

    def num(a):
        v = float(np.asarray(a).reshape(-1)[0])
        return v if np.isfinite(v) else None

    return {"model": str(np.asarray(f["model"]).reshape(-1)[0]), "A": num(f["A"]), "ds": num(f["ds"]),
            "d90": num(f["d90"]), "axis": axis}


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


def _fits(curve: dict) -> tuple[dict | None, dict | None]:
    """``(fit on the requested dose, fit on the neighbourhood dose)`` of a ``curve.json``."""
    fit, neigh = curve.get("fit"), curve.get("fit_neighbourhood")
    if isinstance(fit, dict) and "axis" not in fit:
        # written before the fits named their axis: that fit was on the neighbourhood dose
        neigh = {**fit, "axis": "neighbourhood_dose"}
        pts = sorted((p for p in curve.get("curve") or [] if isinstance(p, dict)), key=lambda p: float(p["dose"]))
        est = [((p.get("region") or p.get("city")) or {}).get("estimate") for p in pts]
        fit = fit_curve([0.0] + [float(p["dose"]) for p in pts],
                        [0.0] + [-float(e) if e is not None else float("nan") for e in est], "dose")
    return fit, neigh


def sweep_out(db, ctx, row: dict) -> dict:
    params = dbmod.loads(row.get("params_json"), {}) or {}
    curve = read_json(Path(row["dir"]) / "curve.json") or {}
    resp = (ctx.json("response_curves.json") or {}).get(params.get("lever")) if ctx is not None else None
    fit, neigh = _fits(curve)
    return {"params": params, "status": sweep_status(db, row),
            "curve": [{k: v for k, v in p.items() if k != "result_id"} for p in curve.get("curve") or []],
            "fit": fit, "fit_neighbourhood": neigh,
            "pipeline_curve": (resp or {}).get("curve") if isinstance(resp, dict) else None,
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
