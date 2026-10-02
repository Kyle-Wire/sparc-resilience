"""What the engine does per request (api.md §13): evaluate scenarios on a loaded :class:`RunSession`, compute
the result statistics and write the result directories (api.md §12.2).

Used by the engine host (one request at a time) and by the ``scenario.across_runs`` job, which loads one
engine at a time in its own process.  Payloads hold primitive values and numpy arrays only:

* an *intervention* is ``{variable, mode, amount, where: bool[n] | None, per_point: float64[n] | None}``
  (the compiler's output, :meth:`sparc.studio.engine.compile.Compiled.interventions`);
* ``options`` are the scenario options: ``clip_to_support`` and ``mediators`` are applied for the request
  and restored in a ``finally`` (SPEC §7.3);
* ``run_row`` is the run's ``runs`` row (``id, run_dir, studio_dir, project_id, status, origin``), from
  which a :class:`~sparc.studio.runs.reader.RunContext` gives the grid, zones, manifest and units.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from sparc.core import progress

log = logging.getLogger("sparc.studio.engine")

__all__ = ["scenario_options", "build_spec", "evaluate", "run_context", "store_result", "op_scenario", "op_batch",
           "op_rerun_configured", "op_sweep", "op_plan_verify", "op_plan_frontier", "configured_spec", "env_for"]


@contextmanager
def scenario_options(engine, options: dict | None):
    """Apply ``clip_to_support`` and ``mediators`` (``False`` → no mediator chain) for one request."""
    opts = options or {}
    saved = (engine.clip_support, engine.mediators)
    try:
        if "clip_to_support" in opts and opts["clip_to_support"] is not None:
            engine.clip_support = bool(opts["clip_to_support"])
        if opts.get("mediators") is False:
            engine.mediators = None
        yield engine
    finally:
        engine.clip_support, engine.mediators = saved


def build_spec(interventions: list[dict], name: str):
    from sparc.core.scenarios import Intervention, ScenarioSpec

    ivs = []
    for iv in interventions:
        where = iv.get("where")
        pp = iv.get("per_point")
        ivs.append(Intervention(str(iv["variable"]), str(iv.get("mode") or "add"), float(iv.get("amount") or 0.0),
                                where=None if where is None else np.asarray(where, dtype=bool),
                                per_point=None if pp is None else np.asarray(pp, dtype=np.float64)))
    return ScenarioSpec(name=name, interventions=ivs)


def evaluate(session, interventions: list[dict], options: dict | None, name: str):
    """``(ScenarioResult, mediator deltas)`` of one scenario on ``session`` (one engine pass, K fold ticks).

    An ``AttributeError`` / ``ImportError`` inside the engine pass is pickle drift (the fitted models no longer
    match the installed code, e.g. a scikit-learn upgrade after a cached baseline pass skipped the load-time
    evaluation): it is raised as :class:`~sparc.core.session.IncompatibleCheckpoint`."""
    from sparc.core.session import IncompatibleCheckpoint

    spec = build_spec(interventions, name)
    eng = session.engine
    with scenario_options(eng, options):
        try:
            res = eng.run(spec, keep_frame=True)
        except (AttributeError, ImportError) as exc:
            raise IncompatibleCheckpoint(f"{type(exc).__name__}: {exc}") from exc
        med_names = list(eng.mediators.models) if eng.mediators is not None else []
    base = session.data.frame
    mediators = {}
    if res.frame is not None:
        for m in med_names:
            if m in res.frame and m not in res.realized:
                mediators[m] = res.frame[m].to_numpy(float) - base[m].to_numpy(float)
    res.frame = None
    return res, mediators


def configured_spec(session, name: str):
    """The configured S5 scenario ``name`` of the run's config (``specs_from_config``)."""
    from sparc.core.scenarios import specs_from_config

    for spec in specs_from_config(session.cfg):
        if spec.name == name:
            return spec
    raise KeyError(f"no configured scenario {name!r} in this run's config")


# ---------------------------------------------------------------------------
# the run context of a request
# ---------------------------------------------------------------------------

def run_context(run_row: dict, session=None):
    """A :class:`RunContext` for ``run_row`` that reuses the session's data (no second data load)."""
    from sparc.studio.runs.reader import RunContext

    ctx = RunContext(dict(run_row), ("engine", time.time()))
    if session is not None:
        ctx._c["data"] = session.data
    return ctx


def _warming_mid(manifest: dict) -> tuple[float | None, str | None]:
    for p in ((manifest or {}).get("climate") or {}).get("projections") or []:
        if p.get("experiment") == "ssp245" and p.get("period") == "2041-2060":
            w = (p.get("warming") or {}).get("median")
            return (float(w) if w is not None else None), f"{p.get('label') or 'SSP2-4.5'} 2041–2060"
    return None, None


def env_for(ctx, session) -> dict:
    """Run-level inputs of the statistics: grid, zones, people, observations, units, ranges, studies."""
    raw = ctx.cfg_raw
    units = ctx.units.get("target", "°F")
    people = None
    lay = getattr(session, "layers", None) if session is not None else None
    if lay is not None and "people" in lay:
        people = np.nan_to_num(np.asarray(lay["people"], dtype=np.float64))
    data = session.data if session is not None else ctx.data
    obs = np.asarray(data.target_raw, dtype=np.float64) if data is not None and getattr(data, "target_raw", None) \
        is not None else None
    grid = ctx.grid
    zones = (grid.zone.astype(np.int64), list(grid.zones)) if grid is not None and grid.zones else None
    manifest = ctx.manifest or {}
    fahrenheit = units in ("°F",)
    thr = (raw.get("climate") or {}).get("thresholds") or ([90.0, 95.0] if fahrenheit else [32.0, 35.0])
    w_mid, w_label = _warming_mid(manifest)
    ranges = dict(session.ranges_m) if session is not None else dict(((manifest.get("influence") or {})
                                                                       .get("ranges_m")) or {})
    return {"units": units, "people": people, "obs": obs, "grid": grid, "zones": zones, "manifest": manifest,
            "cfg_raw": raw, "threshold": float(thr[0]) if thr else None, "warming_mid": w_mid,
            "warming_label": w_label, "ranges": ranges, "causal": getattr(session, "causal", None)}


def store_result(session, ctx, env: dict, res, mediators: dict, *, result: dict, kind: str, name: str,
                 scenario: dict | None = None, payload: dict | None = None, job_id: str | None = None,
                 extra_spec: dict | None = None) -> dict:
    """Compute the statistics of ``res`` and write ``results/<id>/``; returns the ``ResultSummary``."""
    from sparc.studio.engine import stats as S
    from sparc.studio.engine import store
    from sparc.studio.workspace import utc_now

    p = payload or {}
    rid = result["id"]
    rdir = Path(result["dir"])
    created = utc_now()
    realized = {k: np.asarray(v, dtype=np.float64) for k, v in res.realized.items()}
    folds = None if res.delta_folds is None else np.asarray(res.delta_folds, dtype=np.float64)
    spec = {"schema": 1, "id": rid, "kind": kind, "name": name, "scenario_id": (scenario or {}).get("id"),
            "revision": (scenario or {}).get("revision"), "content_hash": p.get("content_hash"),
            "compiled": p.get("compiled") or {}, "options": p.get("options") or {},
            "regions": sorted((p.get("regions") or {}).keys()), "run_id": ctx.run_id,
            "ckpt_key": session.checkpoint_key, "code_sha": session.code_sha, "created_utc": created,
            "job_id": job_id, **(extra_spec or {})}
    summary = S.build_result(
        result_id=rid, kind=kind, run_id=ctx.run_id, created_utc=created, job_id=job_id, delta=res.delta,
        delta_sd=res.delta_sd, extrapolation=res.extrapolation, folds=folds, realized=realized, grid=env["grid"],
        unit=env["units"], scenario=scenario, name=name, requested=p.get("requested"),
        lever_cells=p.get("lever_cells"), mediators=mediators,
        regions={k: np.asarray(v, dtype=bool) for k, v in (p.get("regions") or {}).items()},
        people=env["people"], zones=env["zones"], lever_ranges=env["ranges"], per_unit=p.get("per_unit") or {},
        causal=env["causal"], cfg_raw=env["cfg_raw"], manifest=env["manifest"],
        specification=p.get("specification"), preview_delta=p.get("preview_delta"),
        warnings=p.get("warnings"), demo=bool(p.get("demo")), spec=spec, obs=env["obs"],
        threshold=env["threshold"], warming_mid=env["warming_mid"], warming_label=env["warming_label"])
    store.write_result(rdir, ids=ctx.grid.ids if ctx.grid is not None else session.data.ids, delta=res.delta,
                       delta_sd=res.delta_sd, extrapolation=res.extrapolation, folds=folds, realized=realized,
                       spec=spec, summary=summary, warnings=summary.get("warnings"))
    progress.metric("scenario.mean_delta", float(np.mean(res.delta)), scenario=name)
    se = summary["summary"]["city"].get("se")
    progress.metric("scenario.se", se, scenario=name)
    progress.metric("scenario.frac_extrapolated", float(np.mean(np.asarray(res.extrapolation) > 1.0)), scenario=name)
    return summary["summary"]


# ---------------------------------------------------------------------------
# operations
# ---------------------------------------------------------------------------

def _one(session, ctx, env, item: dict, *, kind: str, job_id: str | None) -> dict:
    name = str(item.get("name") or "scenario")
    res, med = evaluate(session, item["interventions"], item.get("options"), name)
    return store_result(session, ctx, env, res, med, result=item["result"], kind=kind, name=name,
                        scenario=item.get("scenario"), payload=item, job_id=job_id,
                        extra_spec=item.get("extra_spec"))


def op_scenario(session, payload: dict, *, job_id: str | None = None) -> dict:
    ctx = run_context(payload["run_row"], session)
    env = env_for(ctx, session)
    with progress.task("scenario", key=str(payload.get("name") or "scenario")):
        summ = _one(session, ctx, env, payload, kind=str(payload.get("kind") or "exact"), job_id=job_id)
    return {"result_id": summ["id"], "results": [summ["id"]]}


def op_batch(session, payload: dict, *, job_id: str | None = None) -> dict:
    """One tick (unit ``scenario``) per scenario; a failing scenario is reported, the others still run."""
    ctx = run_context(payload["run_row"], session)
    env = env_for(ctx, session)
    items = list(payload.get("items") or [])
    ids, failed = [], []
    for k, item in enumerate(items, start=1):
        progress.check_cancel()
        name = str(item.get("name") or "scenario")
        try:
            with progress.task("scenario", k=k, n=len(items), key=name):
                summ = _one(session, ctx, env, item, kind="exact", job_id=job_id)
            ids.append(summ["id"])
        except progress.Cancelled:
            raise
        except Exception as exc:                 # noqa: BLE001 - one scenario never stops the batch
            log.exception("batch scenario %s failed", name)
            failed.append({"scenario_id": (item.get("scenario") or {}).get("id"),
                           "error": f"{type(exc).__name__}: {exc}"[:500]})
        progress.tick(k, len(items), unit="scenario", label=name)
    for f in payload.get("failed") or []:
        failed.append(f)
    return {"result_ids": ids, "failed": failed, "results": ids}


def op_rerun_configured(session, payload: dict, *, job_id: str | None = None) -> dict:
    ctx = run_context(payload["run_row"], session)
    env = env_for(ctx, session)
    name = str(payload["name"])
    spec = configured_spec(session, name)
    interventions = [{"variable": iv.variable, "mode": iv.mode, "amount": iv.amount, "where": iv.where,
                      "per_point": iv.per_point} for iv in spec.interventions]
    item = {**payload, "interventions": interventions, "name": name,
            "extra_spec": {"configured_slug": payload.get("slug"), "configured_name": name}}
    with progress.task("scenario", key=name):
        summ = _one(session, ctx, env, item, kind="configured", job_id=job_id)
    return {"result_id": summ["id"], "results": [summ["id"]]}


def op_sweep(session, payload: dict, *, job_id: str | None = None) -> dict:
    """``engine.sweep``: one exact run per dose with the selection mask, then ``response.fit_saturation`` on the
    region-mean benefit against the neighbourhood dose (SPEC §7.9)."""
    from sparc.core.response import fit_saturation
    from sparc.studio.engine import stats as S
    from sparc.studio.runs.common import likely
    from sparc.studio.workspace import utc_now, write_json_atomic

    ctx = run_context(payload["run_row"], session)
    env = env_for(ctx, session)
    lever = str(payload["lever"])
    sign = float(payload.get("sign") or 1.0)
    where = payload.get("where")
    mask = np.ones(session.data.n, dtype=bool) if where is None else np.asarray(where, dtype=bool)
    items = list(payload.get("items") or [])
    unit = env["units"]
    curve, ids, D, B = [], [], [0.0], [0.0]
    for k, item in enumerate(items, start=1):
        progress.check_cancel()
        dose = float(item["dose"])
        iv = [{"variable": lever, "mode": "add", "amount": sign * dose, "where": None if where is None else mask,
               "per_point": None}]
        name = f"{lever} {sign * dose:+g}".replace("-", "−")
        with progress.task("dose", k=k, n=len(items), key=f"{dose:g}") as sp:
            res, med = evaluate(session, iv, payload.get("options"), name)
            it = {**payload, "interventions": iv, "name": name, "result": item["result"],
                  "regions": {"sweep selection": mask} if where is not None else {},
                  "requested": None, "lever_cells": None,
                  "extra_spec": {"sweep_id": payload.get("sweep_id"), "dose": dose, "lever": lever}}
            summ = store_result(session, ctx, env, res, med, result=item["result"], kind="sweep_point", name=name,
                                payload=it, job_id=job_id, extra_spec=it["extra_spec"])
            real = np.abs(np.asarray(res.realized.get(lever, np.zeros(session.data.n)), dtype=np.float64))
            edited = real > 0
            reg = S.masked_likely(res.delta, res.delta_folds, mask, unit, what="the selection") \
                if where is not None else None
            neigh = session.resp._neigh_dose(real, lever)
            D.append(float(np.mean(neigh[mask])))
            B.append(-float(np.mean(res.delta[mask])))
            fx = float(np.mean(res.extrapolation[edited] > 1.0)) if edited.any() else 0.0
            curve.append({"dose": dose, "city": summ["city"] or likely(0.0, None, unit), "region": reg,
                          "realized": float(real[mask].mean()) if mask.any() else 0.0, "frac_extrapolated": fx,
                          "result_id": summ["id"]})
            sp.metrics.update(mean_benefit=-float(np.mean(res.delta)), frac_extrapolated=fx)
            ids.append(summ["id"])
    fit = None
    if len(D) >= 3:
        f = fit_saturation(np.asarray(D)[:, None], np.asarray(B)[:, None], np.ones((len(D), 1), dtype=bool),
                           min_valid=min(4, len(D)))

        def num(a):
            v = float(np.asarray(a).reshape(-1)[0])
            return v if np.isfinite(v) else None

        fit = {"model": str(np.asarray(f["model"]).reshape(-1)[0]), "A": num(f["A"]), "ds": num(f["ds"]),
               "d90": num(f["d90"])}
    out = {"curve": curve, "fit": fit, "points": ids, "finished_utc": utc_now()}
    sdir = Path(payload["sweep_dir"])
    write_json_atomic(sdir / "curve.json", out)
    return {"sweep_id": payload.get("sweep_id"), "points": ids, "results": ids}


def op_plan_verify(session, payload: dict, *, job_id: str | None = None) -> dict:
    """``engine.plan_verify``: the allocation as ``per_point`` through the engine (closed loop)."""
    from sparc.studio.workspace import write_json_atomic

    ctx = run_context(payload["run_row"], session)
    env = env_for(ctx, session)
    lever = str(payload["lever"])
    dose = np.asarray(payload["dose"], dtype=np.float64)
    sign = float(payload.get("sign") or 1.0)
    iv = [{"variable": lever, "mode": "add", "amount": 0.0, "where": None, "per_point": sign * dose}]
    name = str(payload.get("name") or "plan")
    with progress.task("closed_loop") as sp:
        res, med = evaluate(session, iv, payload.get("options"), name)
        item = {**payload, "interventions": iv,
                "extra_spec": {"plan_id": payload.get("plan_id")}}
        summ = store_result(session, ctx, env, res, med, result=payload["result"], kind="plan", name=name,
                            payload=item, job_id=job_id, extra_spec=item["extra_spec"])
        treated = dose > 0
        realised = {"total": -float(np.sum(res.delta)),
                    "mean_treated": -float(np.mean(res.delta[treated])) if treated.any() else 0.0,
                    "mean_all": -float(np.mean(res.delta)), "result_id": summ["id"]}
        sp.metrics["realised_total"] = realised["total"]
    progress.metric("realised_total", realised["total"], variable=lever)
    write_json_atomic(Path(payload["plan_dir"]) / "realised.json", realised)
    return {"result_id": summ["id"], "realised_total": realised["total"], "results": [summ["id"]]}


def op_plan_frontier(session, payload: dict, *, job_id: str | None = None) -> dict:
    """``engine.plan_frontier``: an exact closed loop at every budget multiplier."""
    from sparc.studio.workspace import write_json_atomic

    lever = str(payload["lever"])
    sign = float(payload.get("sign") or 1.0)
    pts = list(payload.get("points") or [])
    frontier = []
    for k, pt in enumerate(pts, start=1):
        progress.check_cancel()
        dose = np.asarray(pt["dose"], dtype=np.float64)
        iv = [{"variable": lever, "mode": "add", "amount": 0.0, "where": None, "per_point": sign * dose}]
        with progress.task("closed_loop", k=k, n=len(pts), key=f"{float(pt['budget']):g}"):
            res, _med = evaluate(session, iv, payload.get("options"), f"frontier {k}")
        frontier.append({"budget": float(pt["budget"]), "planned": float(pt["planned"]),
                         "realised": -float(np.sum(res.delta))})
    write_json_atomic(Path(payload["plan_dir"]) / "frontier.json", frontier)
    return {"frontier": frontier}


OPS = {"scenario": op_scenario, "batch": op_batch, "rerun_configured": op_rerun_configured, "sweep": op_sweep,
       "plan_verify": op_plan_verify, "plan_frontier": op_plan_frontier}


def run_op(op: str, session, payload: dict, *, job_id: str | None = None) -> dict:
    fn = OPS.get(op)
    if fn is None:
        raise ValueError(f"unknown engine op {op!r}")
    return fn(session, payload, job_id=job_id)


def describe(payload: Any) -> str:
    """A short text of a payload for logs (arrays as shapes)."""
    if isinstance(payload, dict):
        return "{" + ", ".join(f"{k}: {describe(v)}" for k, v in list(payload.items())[:12]) + "}"
    if isinstance(payload, np.ndarray):
        return f"array{payload.shape}"
    if isinstance(payload, (list, tuple)):
        return f"[{len(payload)} items]"
    return repr(payload)[:60]
