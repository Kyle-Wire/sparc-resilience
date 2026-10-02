"""Budget plans (SPEC §7.10, api.md §7.9).

**Planned mode** (inline, well under a second, no engine): the lever's ``VariableResponse`` is rebuilt from
``response_<var>.parquet`` + ``response_curves.json`` + the config, then
``sparc.core.optimize.planned_allocation`` allocates the budget.  Inputs:

* cost: a scalar, or a per-cell column - any namespaced column (api.md §1) or
  ``csv:<project-relative path>:<column>`` joined by id;
* cap: plantable headroom (canopy role + planner layers, editable paved share) and/or a region (the cap is 0
  outside the selection);
* ``min_dose``: core's post-filter (the freed budget is reported as ``min_dose_dropped_cost``, not re-spent);
* objective: cooling, or people (HRSL smoothed at the lever's influence range, ``pipeline.optimizer_layers``);
* equity: share aged 60+, share under 5, density or a column, normalised to 0–1, with a focus 0–1;
* Pareto multipliers.

**Plan directory** ``<studio_dir>/plans/<plid>/``: ``params.json``, ``planned.json``, ``dose.npy``,
``planned_benefit.npy``; ``realised.json`` after **verify** (``engine.plan_verify``: the allocation as
``per_point`` through the engine) and ``frontier.json`` after **verify frontier**.

**Field kit**: the ranked treated cells with ids and lon/lat, ``planner.logger_sites`` and
``planner.matched_controls`` (treated = dose ≥ the median positive dose).
"""

from __future__ import annotations

import base64
import copy
import logging
from pathlib import Path
from typing import Any

import numpy as np

from sparc.studio import db as dbmod
from sparc.studio.errors import ApiError
from sparc.studio.workspace import new_id, read_json, utc_now, write_json_atomic

log = logging.getLogger("sparc.studio.scenarios")

__all__ = ["variable_response", "plan_inputs", "planned", "preview", "create", "plan_out", "plan_row", "list_plans",
           "field_kit", "frontier_points", "to_scenario_doc", "delete", "params_dict", "DEFAULT_MULTIPLIERS"]

DEFAULT_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0)


def params_dict(params: Any) -> dict:
    from sparc.studio.scenarios.schemas import PlanParams

    if hasattr(params, "model_dump"):
        return params.model_dump(mode="json", exclude_none=True)
    return PlanParams.model_validate(params).model_dump(mode="json", exclude_none=True)


# ---------------------------------------------------------------------------
# the response surface
# ---------------------------------------------------------------------------

def variable_response(ctx, var: str):
    """``VariableResponse`` of ``var`` rebuilt from the run's S4 files (``422 needs_responses`` without them)."""
    import pandas as pd

    from sparc.core.response import VariableResponse

    df = ctx.parquet(f"response_{var}.parquet")
    curves = ctx.json("response_curves.json") or {}
    if df is None or var not in curves:
        raise ApiError("needs_responses", f"budget plans need the S4 response surfaces of {var} "
                       f"(response_{var}.parquet); run S4 first", detail={"lever": var})

    def build():
        ids = np.asarray(ctx.grid.ids).astype(str)
        if "id" in df.columns and not (len(df) == len(ids) and np.array_equal(df["id"].to_numpy().astype(str), ids)):
            src = df.set_index(df["id"].astype(str)).reindex(ids).reset_index(drop=True)
        else:
            src = df.reset_index(drop=True)
        maps = pd.DataFrame({c: src[c].to_numpy(np.float64) for c in src.columns
                             if c not in ("id", "x_m", "y_m", "curve_model", "censored")})
        if "curve_model" in src.columns:
            maps["curve_model"] = src["curve_model"].astype(str).to_numpy()
        if "censored" in src.columns:
            maps["censored"] = src["censored"].fillna(False).astype(bool).to_numpy()
        spec = (ctx.cfg_raw.get("actionable") or {}).get(var) or {}
        cv = curves[var] or {}
        curve = pd.DataFrame(cv.get("curve") or {})
        doses = [float(d) for d in (curve["dose"] if "dose" in curve else spec.get("doses") or [0, 5, 10, 20, 30])]
        if not doses or doses[0] != 0.0:
            doses = [0.0] + doses
        return VariableResponse(variable=var, direction=str(spec.get("direction", "increase")).lower(), doses=doses,
                                curve=curve, maps=maps, summary=cv.get("summary") or {})

    return ctx._get(f"vr:{var}", build)


def _column(ctx, column: str, db=None, project_dir=None, *, what: str) -> np.ndarray:
    """A per-cell value from a namespaced column or ``csv:<path>:<column>`` (joined by id)."""
    if column.startswith("csv:"):
        import pandas as pd

        from sparc.studio.security import safe_path

        rest = column[4:]
        path, _, col = rest.rpartition(":")
        if not path or not col:
            raise ApiError("validation", f"{what} csv references are csv:<path>:<column>",
                           detail={"errors": [{"path": what, "message": "bad csv reference", "code": "csv"}]})
        if project_dir is None:
            raise ApiError("validation", f"{what}: csv references need the run's project",
                           detail={"errors": [{"path": what, "message": "no project", "code": "csv"}]})
        p = safe_path(path, project_dir)
        if not p.is_file():
            raise ApiError("validation", f"{what}: no file {path!r} in the project",
                           detail={"errors": [{"path": what, "message": "missing file", "code": "missing_file"}]})
        df = pd.read_csv(p)
        idcol = next((c for c in df.columns if c.lower() == "id"), None)
        if idcol is None or col not in df.columns:
            raise ApiError("validation", f"{what}: {path} needs columns id and {col}",
                           detail={"errors": [{"path": what, "message": "missing columns", "code": "columns"}]})
        s = df.set_index(df[idcol].astype(str))[col].astype(float)
        s = s[~s.index.duplicated()]
        return s.reindex(np.asarray(ctx.grid.ids).astype(str)).to_numpy(np.float64)
    from sparc.studio.runs.selection import column_values

    return np.asarray(column_values(ctx, column), dtype=np.float64)


def _norm01(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    ok = np.isfinite(v)
    if not ok.any():
        return np.zeros_like(v)
    lo, hi = float(v[ok].min()), float(v[ok].max())
    out = np.where(ok, (v - lo) / (hi - lo) if hi > lo else 0.0, np.nan)
    return np.where(np.isfinite(out), out, float(np.nanmean(out)) if np.isfinite(out).any() else 0.0)


def _needs_layers(ctx, why: str) -> ApiError:
    action = {"kind": "fetch_input", "label": "Fetch people and land cover", "method": "POST",
              "path": f"/api/projects/{ctx.project_id}/inputs/layers", "body": {}} if ctx.project_id else None
    return ApiError("needs_layers", f"{why} needs the planner layers (planner.layers)", action=action)


def plan_inputs(ctx, params: dict, *, db=None, project_dir=None) -> dict:
    """Everything ``planned_allocation`` takes, plus the constraint and objective labels."""
    from sparc.studio.runs import layers as L

    p = params_dict(params)
    var = p["lever"]
    if var not in (ctx.cfg_raw.get("actionable") or {}):
        raise ApiError("validation", f"{var!r} is not an actionable lever of this run",
                       detail={"errors": [{"path": "lever", "message": "not actionable", "code": "lever"}]})
    vr = variable_response(ctx, var)
    n = ctx.grid.n
    cost_spec = p.get("cost") or {"scalar": 1.0}
    if "column" in cost_spec:
        cost = _column(ctx, str(cost_spec["column"]), db, project_dir, what="cost.column")
        ok = np.isfinite(cost) & (cost > 0)
        if not ok.any():
            raise ApiError("validation", "the cost column has no positive values",
                           detail={"errors": [{"path": "cost.column", "message": "no positive values", "code": "cost"}]})
        cost = np.where(ok, cost, float(np.mean(cost[ok])))
    else:
        cost = float(cost_spec.get("scalar") or 1.0)
    cap = np.full(n, np.inf)
    constraint = []
    capd = p.get("cap") or {}
    canopy = ((ctx.cfg_raw.get("physics") or {}).get("roles") or {}).get("canopy")
    if capd.get("plantable"):
        if var != canopy:
            raise ApiError("validation", f"the plantable cap applies to the canopy lever ({canopy}), not {var}",
                           detail={"errors": [{"path": "cap.plantable", "message": "canopy only", "code": "cap"}]})
        lay = L.people_layers(ctx) if ctx.data is not None else None
        if lay is None:
            raise _needs_layers(ctx, "the plantable-space cap")
        from sparc.core.planner import plantable_headroom

        share = capd.get("paved_share")
        if share is None:
            share = float((ctx.cfg_raw.get("planner") or {}).get("paved_plantable_share", 0.2))
        cap = np.minimum(cap, plantable_headroom(ctx.data.frame[var].to_numpy(float), lay, float(share)))
        constraint.append(f"plantable space (open land + {float(share):.0%} of built-up area)")
    if capd.get("region"):
        from sparc.studio.runs.selection import RunSource, resolve

        mask = resolve(RunSource(ctx, db), capd["region"])
        cap = np.where(mask, cap, 0.0)
        constraint.append(f"within the selected region ({int(mask.sum()):,} cells)")
    weight = None
    objective = "total cooling"
    if p.get("objective") == "people":
        if ctx.data is None or L.people_layers(ctx) is None:
            raise _needs_layers(ctx, "the people objective")
        from sparc.core.pipeline import optimizer_layers

        cfg2 = copy.deepcopy(ctx.cfg)
        cfg2.raw["optimize"] = {**(cfg2.raw.get("optimize") or {}), "objective": "people", "plantable": False}
        ranges = ((ctx.manifest.get("influence") or {}).get("ranges_m")
                  or (ctx.json("influence.json") or {}).get("ranges_m") or {})
        _cap, weight, _c, objective = optimizer_layers(cfg2, ctx.data, var, ranges)
    eq = None
    focus = 0.0
    if p.get("equity"):
        e = p["equity"]
        focus = float(e.get("focus") or 0.0)
        src = e.get("source")
        if src == "column":
            if not e.get("column"):
                raise ApiError("validation", "equity source column needs equity.column",
                               detail={"errors": [{"path": "equity.column", "message": "required", "code": "missing"}]})
            eq = _norm01(_column(ctx, str(e["column"]), db, project_dir, what="equity.column"))
        else:
            lay = L.people_layers(ctx) if ctx.data is not None else None
            if lay is None or "people" not in lay:
                raise _needs_layers(ctx, f"the {src} equity score")
            people = np.nan_to_num(lay["people"].to_numpy(float))
            if src == "density":
                eq = _norm01(people)
            else:
                col = "people_60_plus" if src == "share_60_plus" else "people_under_5"
                if col not in lay:
                    raise _needs_layers(ctx, f"the {src} equity score ({col})")
                with np.errstate(invalid="ignore", divide="ignore"):
                    eq = _norm01(np.where(people > 0.5, np.nan_to_num(lay[col].to_numpy(float)) / people, np.nan))
    finite_cap = None if np.all(np.isinf(cap)) else np.where(np.isinf(cap), np.inf, np.maximum(cap, 0.0))
    return {"vr": vr, "budget": float(p["budget"]), "cost": cost, "cap": finite_cap, "weight": weight,
            "equity": eq, "focus": focus, "min_dose": float(p.get("min_dose") or 0.0),
            "multipliers": tuple(float(m) for m in (p.get("multipliers") or DEFAULT_MULTIPLIERS)),
            "constraint": "; ".join(constraint) or "unconstrained (bounds and headroom only)",
            "objective": objective, "lever": var}


def planned(ctx, params: dict, *, db=None, project_dir=None, budget: float | None = None) -> tuple[dict, dict]:
    """``(planned_allocation output, plan inputs)``."""
    from threadpoolctl import threadpool_limits

    from sparc.core.optimize import planned_allocation

    inp = plan_inputs(ctx, params, db=db, project_dir=project_dir)
    with threadpool_limits(1):
        out = planned_allocation(inp["vr"], float(budget if budget is not None else inp["budget"]),
                                 cost_per_unit=inp["cost"], equity_scores=inp["equity"], equity_focus=inp["focus"],
                                 multipliers=inp["multipliers"], cap=inp["cap"], benefit_weight=inp["weight"],
                                 min_dose=inp["min_dose"])
    return out, inp


def _summary(ctx, out: dict, inp: dict) -> dict:
    from sparc.core.catalog import unit_label

    unit = ctx.units.get("target", "°F")
    lever_unit = unit_label(((ctx.cfg_raw.get("actionable") or {}).get(inp["lever"]) or {}).get("unit")) or "units"
    if "status" in out:
        return {"planned_total": 0.0, "n_cells_treated": 0, "mean_dose_treated": 0.0, "total_cost": 0.0, "gini": 0.0,
                "min_dose_dropped_cost": 0.0, "pareto": [], "constraint": inp["constraint"],
                "objective": inp["objective"],
                "caption": f"Nothing to allocate: {inp['lever']} has no positive-benefit dose ({out['status']})."}
    pts = [{"budget": float(q["budget"]), "benefit": float(q["total_benefit"]), "n_cells": int(q["n_cells"]),
            "n_segments": int(q["n_segments"]), "gini": float(q["gini"])} for q in out["pareto"]["points"]]
    cap = (f"Budget {inp['budget']:,.4g} on {inp['lever']}: {out['n_cells_treated']:,} cells treated "
           f"(mean dose {out['mean_dose_treated']:.3g} {lever_unit}), planned cooling "
           f"{out['planned_total_cooling']:,.4g} {unit}·cells summed over the city; {inp['constraint']}; "
           f"objective {inp['objective']}.")
    if out.get("min_dose_dropped_cost"):
        cap += (f" Allocations below {inp['min_dose']:g} {lever_unit} were dropped, leaving "
                f"{out['min_dose_dropped_cost']:,.4g} of the budget unspent.")
    cap += " Planned totals add per-cell footprint effects; the closed loop (Verify) shows what spillover keeps."
    return {"planned_total": float(out["planned_total_cooling"]), "n_cells_treated": int(out["n_cells_treated"]),
            "mean_dose_treated": float(out["mean_dose_treated"]), "total_cost": float(out["total_cost"]),
            "gini": float(out["gini"]), "min_dose_dropped_cost": float(out.get("min_dose_dropped_cost") or 0.0),
            "pareto": pts, "constraint": inp["constraint"], "objective": inp["objective"], "caption": cap}


def _dose(out: dict, n: int) -> np.ndarray:
    return np.asarray(out.get("dose") if "dose" in out else np.zeros(n), dtype=np.float64)


def preview(ctx, params, *, db=None, project_dir=None) -> dict:
    """``POST /api/runs/{rid}/plans/preview`` → ``PlanPreview``."""
    out, inp = planned(ctx, params, db=db, project_dir=project_dir)
    s = _summary(ctx, out, inp)
    s["dose"] = base64.b64encode(_dose(out, ctx.grid.n).astype("<f4").tobytes()).decode("ascii")
    return s


# ---------------------------------------------------------------------------
# stored plans
# ---------------------------------------------------------------------------

def create(db, ctx, params, name: str, *, project_dir=None) -> dict:
    out, inp = planned(ctx, params, db=db, project_dir=project_dir)
    plid = new_id("plan")
    pdir = Path(ctx.studio_dir) / "plans" / plid
    pdir.mkdir(parents=True, exist_ok=True)
    from sparc.core import runio

    now = utc_now()
    p = params_dict(params)
    summary = _summary(ctx, out, inp)
    n = ctx.grid.n
    with runio.atomic_open(pdir / "dose.npy", "wb") as fh:
        np.save(fh, _dose(out, n).astype(np.float32), allow_pickle=False)
    with runio.atomic_open(pdir / "planned_benefit.npy", "wb") as fh:
        np.save(fh, np.asarray(out.get("planned_benefit", np.zeros(n)), dtype=np.float32), allow_pickle=False)
    write_json_atomic(pdir / "planned.json", summary)
    write_json_atomic(pdir / "params.json", {"schema": 1, "id": plid, "name": name, "run_id": ctx.run_id,
                                             "project_id": ctx.project_id, "params": p, "created_utc": now,
                                             "direction": inp["vr"].direction})
    row = {"id": plid, "project_id": ctx.project_id, "run_id": ctx.run_id, "name": name,
           "params_json": dbmod.dumps(p), "summary_json": dbmod.dumps(summary), "dir": str(pdir),
           "verified_result_id": None, "created_utc": now}
    db.insert("plans", row)
    return row


def plan_row(db, plid: str) -> dict:
    row = db.fetchone("SELECT * FROM plans WHERE id = ?", (plid,))
    if row is None:
        raise ApiError("not_found", f"no plan {plid!r}")
    return row


def plan_out(row: dict) -> dict:
    """A ``plans`` row as the api.md ``Plan``."""
    pdir = Path(row["dir"])
    planned_s = read_json(pdir / "planned.json") or dbmod.loads(row.get("summary_json"), {}) or {}
    planned_s = {k: v for k, v in planned_s.items() if k != "dose"}
    realised = read_json(pdir / "realised.json")
    frontier = read_json(pdir / "frontier.json")
    return {"id": row["id"], "name": row.get("name") or row["id"], "params": dbmod.loads(row.get("params_json"), {}),
            "planned": planned_s, "realised": realised if isinstance(realised, dict) and realised.get("result_id")
            else None, "frontier": frontier if isinstance(frontier, list) else None,
            "created_utc": row.get("created_utc") or ""}


def list_plans(db, run_id: str) -> list[dict]:
    return [plan_out(r) for r in db.fetchall("SELECT * FROM plans WHERE run_id = ? ORDER BY created_utc DESC",
                                             (run_id,))]


def delete(db, plid: str) -> None:
    from sparc.studio.engine import store

    row = plan_row(db, plid)
    rid = row.get("verified_result_id") or (read_json(Path(row["dir"]) / "realised.json") or {}).get("result_id")
    if rid and db.fetchone("SELECT id FROM results WHERE id = ?", (rid,)):
        store.delete_result(db, rid)
    store.rmtree_under(Path(row["dir"]), "plans")
    db.execute("DELETE FROM plans WHERE id = ?", (plid,))


def frontier_points(ctx, row: dict, *, db=None, project_dir=None) -> list[dict]:
    """``[{budget, planned, dose}]`` at every Pareto multiplier of the plan (for ``engine.plan_frontier``)."""
    params = dbmod.loads(row.get("params_json"), {}) or {}
    inp_mult = tuple(float(m) for m in (params.get("multipliers") or DEFAULT_MULTIPLIERS))
    pts = []
    for m in inp_mult:
        b = float(params["budget"]) * m
        out, _inp = planned(ctx, params, db=db, project_dir=project_dir, budget=b)
        pts.append({"budget": b, "planned": float(out.get("planned_total_cooling") or 0.0),
                    "dose": _dose(out, ctx.grid.n)})
    return pts


def to_scenario_doc(row: dict) -> dict:
    params = dbmod.loads(row.get("params_json"), {}) or {}
    return {"name": f"Plan: {row.get('name') or row['id']}", "tags": ["plan", f"plan:{row['id']}"],
            "anchor_run_id": row.get("run_id"),
            "edits": [{"lever": params.get("lever"), "mode": "per_cell", "per_cell_ref": f"plan:{row['id']}",
                       "where": {"kind": "all"}, "label": f"budget plan {row.get('name') or row['id']}"}]}


# ---------------------------------------------------------------------------
# field kit
# ---------------------------------------------------------------------------

def _lonlat(grid, idx: np.ndarray) -> tuple[list, list]:
    lon = [float(v) if np.isfinite(v) else None for v in np.asarray(grid.lon, dtype=np.float64)[idx]]
    lat = [float(v) if np.isfinite(v) else None for v in np.asarray(grid.lat, dtype=np.float64)[idx]]
    return lon, lat


def _id(v):
    return v.item() if hasattr(v, "item") else v


def field_kit(db, ctx, row: dict, *, n_sites: int = 30, min_spacing_m: float = 400.0, n_pairs: int = 30,
              min_distance_m: float = 1000.0) -> dict:
    """``POST /api/plans/{plid}/field-kit`` (api.md §7.9): ranked cells, logger sites, before/after pairs."""
    from sparc.core.planner import logger_sites, matched_controls
    from sparc.studio.engine import store
    from sparc.studio.runs import layers as L

    pdir = Path(row["dir"])
    grid = ctx.grid
    n = grid.n
    dose = np.asarray(np.load(pdir / "dose.npy", allow_pickle=False), dtype=np.float64)
    benefit = np.asarray(np.load(pdir / "planned_benefit.npy", allow_pickle=False), dtype=np.float64) \
        if (pdir / "planned_benefit.npy").exists() else np.zeros(n)
    realised = read_json(pdir / "realised.json") or {}
    closed = None
    if realised.get("result_id"):
        try:
            closed = store.result_array(store.result_row(db, realised["result_id"]), "delta").astype(np.float64)
        except ApiError:
            closed = None
    people = None
    try:
        if "people" in L.layer_defs(ctx):
            people = np.asarray(L.layer_array(ctx, "people"), dtype=np.float64)
    except ApiError:
        people = None
    plantable = None
    raw = ctx.cfg_raw
    roles = (raw.get("physics") or {}).get("roles") or {}
    can, imp = roles.get("canopy"), roles.get("impervious")
    if can and ctx.data is not None and L.people_layers(ctx) is not None:
        from sparc.core.planner import plantable_headroom

        share = float((raw.get("planner") or {}).get("paved_plantable_share", 0.2))
        plantable = plantable_headroom(ctx.data.frame[can].to_numpy(float), L.people_layers(ctx), share)
    codes = grid.zone_codes
    treated = np.flatnonzero(dose > 0)
    order = treated[np.argsort(-benefit[treated], kind="stable")]
    lon, lat = _lonlat(grid, order)
    cells = []
    for k, i in enumerate(order):
        cells.append({"rank": k + 1, "id": _id(grid.ids[i]), "lon": lon[k], "lat": lat[k],
                      "zone": _id(codes[i]) if codes is not None else None, "dose": float(dose[i]),
                      "planned_benefit": float(benefit[i]),
                      "closed_loop_delta": float(closed[i]) if closed is not None else None,
                      "people": float(people[i]) if people is not None and np.isfinite(people[i]) else None,
                      "plantable_pp": float(plantable[i]) if plantable is not None else None})
    x, y = np.asarray(grid.x, dtype=np.float64), np.asarray(grid.y, dtype=np.float64)
    sites: list[dict] = []
    data = ctx.data
    if can and imp and data is not None and ctx.exists(f"response_{can}.parquet"):
        r = ctx.parquet(f"response_{can}.parquet")
        col = "footprint_effect_sd" if "footprint_effect_sd" in r.columns else "footprint_effect_per_unit"
        eff = np.abs(L._aligned(ctx, r, col).astype(np.float64))
        try:
            s = logger_sites(data.frame[can].to_numpy(float), data.frame[imp].to_numpy(float), np.nan_to_num(eff),
                             x, y, n=int(n_sites), min_spacing_m=float(min_spacing_m))
            idx = s["cell"].to_numpy(np.int64)
            slon, slat = _lonlat(grid, idx)
            for k, i in enumerate(idx):
                sites.append({"id": _id(grid.ids[i]), "lon": slon[k], "lat": slat[k], "role": str(s["role"].iloc[k]),
                              "canopy": float(s["canopy"].iloc[k]), "impervious": float(s["impervious"].iloc[k]),
                              "effect_sd": float(s["effect_sd"].iloc[k])})
        except ValueError as exc:          # too few cells per stratum on a tiny grid
            log.info("logger sites skipped: %s", exc)
    pairs: list[dict] = []
    if treated.size and data is not None:
        pos = dose[dose > 0]
        tmask = dose >= float(np.median(pos))
        cols = [c for c in (can, imp, roles.get("elevation"), roles.get("water_distance")) if c and c in data.frame]
        if cols:
            cov = data.frame[cols].to_numpy(float)
            pr = matched_controls(tmask, cov, x, y, min_distance_m=float(min_distance_m), n_pairs=int(n_pairs))
            if len(pr):
                t = pr["treated"].to_numpy(np.int64)
                c = pr["control"].to_numpy(np.int64)
                tlon, tlat = _lonlat(grid, t)
                clon, clat = _lonlat(grid, c)
                for k in range(len(pr)):
                    pairs.append({"treated_id": _id(grid.ids[t[k]]), "control_id": _id(grid.ids[c[k]]),
                                  "treated_lon": tlon[k], "treated_lat": tlat[k], "control_lon": clon[k],
                                  "control_lat": clat[k], "covariate_distance": float(pr["covariate_distance"].iloc[k])})
    return {"cells": cells, "sites": sites, "pairs": pairs}
