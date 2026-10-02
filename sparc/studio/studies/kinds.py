"""Job kinds of studies and post-run actions (SPEC §8, §10.3, api.md §8).

==================  ======  =====================================================================  =========
Kind                Lane    Library call (against the parent run's launch-snapshot config)         Writes
==================  ======  =====================================================================  =========
``post.baselines``  medium  ``baselines.baselines_for_run(run_dir, cfg, models)``                  run dir
``post.planner``    medium  ``planner.planner_pack(run_dir, cfg, …, cache_dir=<ws>/cache)``        run dir
``post.emulator``   heavy   ``emulator.emulator_for_run(run_dir, cfg, n_patches)``                 run dir
``post.uncertainty`` medium ``uncertainty.uncertainty_report(run_dir, mv, sc, placebo, gate)``     run dir
``post.writeup``    medium  ``writeup.methods_markdown`` / ``model_card_markdown``                 run dir
``study.placebo``   heavy   ``placebo.run_placebo_suite(cfg, …, children_dir, resume=True, meta)`` study dir
``study.simcheck``  heavy   ``simcheck.run_simcheck(cfg, design, out_dir, …)`` (resumable)         study dir
``study.multiverse`` heavy  ``multiverse.run_multiverse(cfg, out_dir, …, extra_variants, meta)``   study dir
``study.reproduce`` heavy   ``reproduce.reproduce(run_dir, …, config_dir, out_dir, meta)``         child run
``study.benchmark`` heavy   ``diagnostics.run_benchmark(seed, ab, epochs, n)``                     study dir
==================  ======  =====================================================================  =========

The config is the run's launch snapshot (``<studio_dir>/launch.json``), else its manifest config, else the
config given at import (:func:`run_config`), never the project's current config.  The kinds write the
``.md`` / ``.json`` files the CLI used to (``placebo.json/.md``, ``simcheck_summary.md``,
``multiverse_summary.md``, ``benchmark.json/.md``) with :mod:`sparc.core.runio`.

Server-side hooks: ``on_event`` registers each child run of a study as soon as its ``run.dir`` event
arrives (``origin = study_child``, ``study_id``, ``parent_run_id``) and follows the job's status;
``on_finish`` records the study's status and summary, rewrites the job result's ``children`` as run ids,
refreshes the parent run and - for attached studies, with the ``auto_uncertainty`` setting - enqueues
``post.uncertainty`` on every run the study is attached to.

Heavy libraries are imported inside the job functions: the server imports this module to list kinds.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from sparc.studio import db as dbmod
from sparc.studio.jobs.kinds import job_kind
from sparc.studio.studies import service
from sparc.studio.workspace import read_json

log = logging.getLogger("sparc.studio.studies")

__all__ = ["BaselinesParams", "PlannerParams", "EmulatorParams", "UncertaintyParams", "WriteupParams",
           "PlaceboParams", "SimcheckParams", "MultiverseParams", "ReproduceParams", "BenchmarkParams",
           "PARAMS", "run_config", "estimate_kind", "enqueue_uncertainty", "child_meta", "GHCN_HOST",
           "BASELINE_MODELS", "CORE_STAGES"]

GHCN_HOST = "noaa-ghcn-pds.s3.amazonaws.com"
#: ``sparc.core.baselines.BASELINES`` (kept here so validating params never imports the core models)
BASELINE_MODELS = ("regression_kriging", "hgb_xy", "hgb", "idw", "hgb_focal")
CORE_STAGES = ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7")
SIM_GENERATORS = ("physics", "additive", "own_only", "coarse_scale", "confounded", "null")


# ---------------------------------------------------------------------------
# params (api.md §8)
# ---------------------------------------------------------------------------

class BaselinesParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    models: list[str] | None = None

    @field_validator("models")
    @classmethod
    def _known(cls, v):
        if v is not None:
            bad = [m for m in v if m not in BASELINE_MODELS]
            if bad or not v:
                raise ValueError(f"models must be a non-empty subset of {list(BASELINE_MODELS)}"
                                 + (f" (unknown: {bad})" if bad else ""))
        return v


class PlannerParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    package: str | None = Field(None, description="configured scenario slug or exact name")
    thresholds: list[float] | None = None
    hex_sizes: list[float] | None = None
    export: bool = True

    @field_validator("hex_sizes")
    @classmethod
    def _pos(cls, v):
        if v is not None and any(x <= 0 for x in v):
            raise ValueError("hex sizes must be positive (metres)")
        return v


class EmulatorParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patches: int = Field(8, ge=1, le=64)


class UncertaintyParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    multiverse_study: str | None = None
    simcheck_studies: list[str] | None = None
    placebo_study: str | None = None
    real_r2_gate: bool = False


class WriteupParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PlaceboParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kinds: list[Literal["grf", "shift", "rotate"]] = Field(default_factory=lambda: ["grf", "shift", "rotate"],
                                                          min_length=1)
    coarse_m: float | None = Field(60.0, gt=0)
    seed: int = 0
    grf_range_m: float = Field(600.0, gt=0)

    @field_validator("kinds")
    @classmethod
    def _unique(cls, v):
        return list(dict.fromkeys(v))


class SimcheckParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    design: dict[Literal["physics", "additive", "own_only", "coarse_scale", "confounded", "null"], int]
    coarse_m: float | None = Field(90.0, gt=0)
    epochs: int = Field(200, ge=1)
    workers: int = Field(1, ge=1)
    threads: int = Field(1, ge=1)
    continue_study_id: str | None = None

    @field_validator("design")
    @classmethod
    def _counts(cls, v):
        if any(n < 0 for n in v.values()):
            raise ValueError("replicate counts must be ≥ 0")
        if not any(n > 0 for n in v.values()):
            raise ValueError("the design needs at least one replicate")
        return v


class MultiverseParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    variants: list[str] | None = None
    custom_variants: dict[str, dict[str, Any]] | None = None
    coarse_m: float | None = Field(60.0, gt=0)
    workers: int = Field(1, ge=1)
    threads: int = Field(1, ge=1)

    @field_validator("custom_variants")
    @classmethod
    def _dotted(cls, v):
        import re

        for name, changes in (v or {}).items():
            if not re.fullmatch(r"[A-Za-z0-9_]+", name):
                raise ValueError(f"variant name {name!r}: letters, digits and _ only")
            if not changes:
                raise ValueError(f"variant {name!r} changes nothing")
            for key in changes:
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*", str(key)):
                    raise ValueError(f"variant {name!r}: {key!r} is not a dotted config key")
        return v


class ReproduceParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stages: list[Literal["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"]] = Field(
        default_factory=lambda: ["S0", "S1", "S2", "S3"], min_length=1)
    tol_r2: float = Field(0.01, gt=0)
    tol_effect: float = Field(0.05, gt=0)


class BenchmarkParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int = 0
    ab: bool = True
    epochs: int = Field(150, ge=1)
    n: int = Field(96, ge=16, le=400)


PARAMS: dict[str, type[BaseModel]] = {
    "baselines": BaselinesParams, "planner": PlannerParams, "emulator": EmulatorParams,
    "uncertainty": UncertaintyParams, "writeup": WriteupParams, "placebo": PlaceboParams,
    "simcheck": SimcheckParams, "multiverse": MultiverseParams, "reproduce": ReproduceParams,
    "benchmark": BenchmarkParams,
}


# ---------------------------------------------------------------------------
# the run's config (worker side)
# ---------------------------------------------------------------------------

def run_config(ctx):
    """The parent run's ``CoreConfig``: launch snapshot → manifest config → the config given at import (its
    ``import.json``) → the job's config path.  The first whose data file exists wins (a manifest of a run
    made elsewhere can carry a relative data path); else the first that parses."""
    from sparc.core.config import core_config_from_dict, load_core_config

    cands = []
    launch = ctx.launch
    if launch and launch.get("config_raw") is not None:
        cands.append(lambda: core_config_from_dict(launch["config_raw"], base_dir=launch.get("config_dir")
                                                   or ctx.run_dir))
    m = read_json(Path(ctx.run_dir) / "manifest.json") if ctx.run_dir else None
    if isinstance(m, dict) and isinstance(m.get("config"), dict):
        base = (m.get("provenance") or {}).get("config_dir")
        if base:
            cands.append(lambda: core_config_from_dict(m["config"], base_dir=base))
    rec = read_json(Path(ctx.studio_dir) / "import.json") if ctx.studio_dir else None
    if isinstance(rec, dict) and rec.get("config_path") and Path(rec["config_path"]).is_file():
        cands.append(lambda: load_core_config(rec["config_path"]))
    if isinstance(m, dict) and isinstance(m.get("config"), dict):
        cands.append(lambda: core_config_from_dict(m["config"], base_dir=ctx.run_dir))
    if ctx.config_path and Path(ctx.config_path).is_file():
        cands.append(lambda: load_core_config(ctx.config_path))
    first = err = None
    for make in cands:
        try:
            cfg = make()
        except Exception as exc:            # an unusable candidate: try the next
            err = err or exc
            continue
        first = first or cfg
        try:
            if Path(cfg.data_path).is_file():
                return cfg
        except Exception:
            continue
    if first is not None:
        return first
    raise FileNotFoundError(f"no usable config for run {ctx.run_id!r} (launch.json, manifest or import config)"
                            + (f": {err}" if err else ""))


def child_meta(ctx) -> dict:
    """``run_meta`` of a study's child runs: the explicit link to the study and its parent run (SPEC §5.11)."""
    return {"study_id": ctx.study_id, "parent_run_id": ctx.run_id, "project_id": ctx.project_id,
            "origin": "study_child"}


def _units(cfg) -> str:
    return str((cfg.data or {}).get("target_units") or "")


def _study_dir(ctx) -> Path:
    if not ctx.study_dir:
        raise FileNotFoundError(f"job {ctx.job_id} has no study folder")
    d = Path(ctx.study_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# post-run actions
# ---------------------------------------------------------------------------

def _post_on_finish(sctx, job: dict, result) -> None:
    """The action wrote into the run folder: re-derive the run row and drop the cached run context."""
    rid = job.get("run_id")
    if not rid or sctx is None:
        return
    reg = sctx.services.get("registry")
    if reg is not None:
        try:
            reg.refresh(rid)
        except Exception:
            log.exception("refreshing run %s failed", rid)
    reader = sctx.services.get("reader")
    if reader is not None:
        reader.forget(rid)


def _post_estimate(kind: str):
    def est(sctx, job, params):
        row = sctx.db.fetchone("SELECT * FROM runs WHERE id = ?", (job.get("run_id"),)) if job.get("run_id") else None
        return estimate_kind(sctx.db, kind, row, params)
    return est


@job_kind("post.baselines", lane="medium", label="Reference baselines", params=BaselinesParams, needs_run=True,
          locks_run=True, on_finish=_post_on_finish, estimate=_post_estimate("baselines"))
def post_baselines(ctx, params: BaselinesParams) -> dict:
    from sparc.core.baselines import BASELINES, baselines_for_run

    res = baselines_for_run(Path(ctx.run_dir), run_config(ctx), tuple(params.models or BASELINES))
    return {"verdict": res.get("verdict"), "best_baseline": res.get("best_baseline"), "rows": res.get("rows") or {}}


def _package_name(run_dir: Path, package: str | None) -> str | None:
    """A configured scenario's exact name from its slug or name (``None``: the planner's default package)."""
    if not package:
        return None
    import pandas as pd

    from sparc.core.catalog import scenario_slug

    p = run_dir / "scenario_deltas.parquet"
    names = [c for c in pd.read_parquet(p).columns if c != "id"] if p.exists() else []
    if package in names:
        return package
    taken: set[str] = set()
    for name in names:
        slug = scenario_slug(name, taken)
        taken.add(slug)
        if slug == package:
            return name
    raise ValueError(f"no configured scenario {package!r} in this run (scenarios: {', '.join(names) or 'none'})")


@job_kind("post.planner", lane="medium", label="Planner pack", params=PlannerParams, needs_run=True, locks_run=True,
          network_hosts=(GHCN_HOST,), long=True, on_finish=_post_on_finish, estimate=_post_estimate("planner"))
def post_planner(ctx, params: PlannerParams) -> dict:
    from sparc.core.planner import planner_pack

    cfg = run_config(ctx)
    if not (cfg.raw.get("planner") or {}).get("layers"):
        raise ValueError("planner.layers is not set in this run's config (fetch the planner layers first)")
    run_dir = Path(ctx.run_dir)
    kw: dict[str, Any] = {"package": _package_name(run_dir, params.package), "thresholds": params.thresholds,
                          "export": params.export, "cache_dir": ctx.cache_dir}
    if params.hex_sizes:
        kw["hex_sizes"] = tuple(params.hex_sizes)
    out = planner_pack(run_dir, cfg, **kw)
    pdir = run_dir / "planner"
    files = sorted(str(p.relative_to(run_dir)) for p in pdir.rglob("*") if p.is_file()) if pdir.is_dir() else []
    return {"people_total": out.get("people_total"), "package": out.get("package"), "files": files,
            "hot_days": bool(out.get("hot_days"))}


def _emulator_preflight(sctx, job: dict, params) -> list[dict]:
    """The checkpoint is checked when the job starts, not when it is queued: a launch's "then" chain queues
    ``post.emulator`` before its run has written ``checkpoint.pkl``."""
    row = sctx.db.fetchone("SELECT run_dir FROM runs WHERE id = ?", (job.get("run_id"),))
    if row is None or not (Path(row["run_dir"]) / "checkpoint.pkl").is_file():
        return [{"reason": "the run has no checkpoint.pkl", "fatal": True, "code": "no_checkpoint"}]
    return []


@job_kind("post.emulator", lane="heavy", label="Scenario emulator", params=EmulatorParams, needs_run=True,
          locks_run=True, long=True, on_finish=_post_on_finish, preflight=_emulator_preflight,
          estimate=_post_estimate("emulator"))
def post_emulator(ctx, params: EmulatorParams) -> dict:
    from sparc.core.emulator import emulator_for_run

    meta = emulator_for_run(Path(ctx.run_dir), run_config(ctx), n_patches=int(params.patches))
    levers = {}
    for v, d in (meta.get("levers") or {}).items():
        val = d.get("validation") or {}
        levers[v] = {"patch_pass_rate": val.get("patch_pass_rate"),
                     "uniform_rel_err": (val.get("uniform") or {}).get("rel_err")}
    return {"levers": levers}


def _study_dirs(ctx, ids: list[str]) -> list[Path]:
    out = []
    for sid in ids:
        r = ctx.db.fetchone("SELECT out_dir FROM studies WHERE id = ?", (sid,))
        if r is None or not r.get("out_dir"):
            raise ValueError(f"no study {sid!r}")
        out.append(Path(r["out_dir"]))
    return out


def uncertainty_sources(ctx, params: UncertaintyParams) -> tuple[Path | None, list[Path], Path | None]:
    """``(multiverse dir, simcheck dirs, placebo.json)``: the params' studies, else the studies attached to the
    run (the newest multiverse, every simcheck, the newest placebo)."""
    explicit = params.multiverse_study or params.simcheck_studies or params.placebo_study
    if explicit:
        mv = _study_dirs(ctx, [params.multiverse_study])[0] if params.multiverse_study else None
        sc = _study_dirs(ctx, list(params.simcheck_studies or []))
        pz = _study_dirs(ctx, [params.placebo_study])[0] / "placebo.json" if params.placebo_study else None
        return mv, sc, pz
    rows = ctx.db.fetchall("SELECT s.kind, s.out_dir, s.status FROM studies s JOIN study_links l "
                           "ON l.study_id = s.id WHERE l.run_id = ? AND l.attached = 1 "
                           "ORDER BY s.updated_utc DESC, s.created_utc DESC", (ctx.run_id,))
    ok = [r for r in rows if r.get("out_dir") and r.get("status") in ("succeeded", "done", None)]
    mv = next((Path(r["out_dir"]) for r in ok if r["kind"] == "multiverse"
               and (Path(r["out_dir"]) / "multiverse_summary.json").is_file()), None)
    sc = [Path(r["out_dir"]) for r in ok if r["kind"] == "simcheck"
          and (Path(r["out_dir"]) / "simcheck.jsonl").is_file()]
    pz = next((Path(r["out_dir"]) / "placebo.json" for r in ok if r["kind"] == "placebo"
               and (Path(r["out_dir"]) / "placebo.json").is_file()), None)
    return mv, sc, pz


@job_kind("post.uncertainty", lane="medium", label="Uncertainty report", params=UncertaintyParams, needs_run=True,
          locks_run=True, on_finish=_post_on_finish, estimate=_post_estimate("uncertainty"))
def post_uncertainty(ctx, params: UncertaintyParams) -> dict:
    from sparc.core.uncertainty import uncertainty_report

    mv, sc, pz = uncertainty_sources(ctx, params)
    out = uncertainty_report(Path(ctx.run_dir), multiverse_dir=mv, simcheck_dirs=sc, placebo_path=pz,
                             real_r2_gate=bool(params.real_r2_gate))
    return {"n_scenarios": len(out.get("scenarios") or []), "sources": out.get("sources") or {}}


@job_kind("post.writeup", lane="medium", label="Methods & model card", params=WriteupParams, needs_run=True,
          locks_run=True, on_finish=_post_on_finish, estimate=_post_estimate("writeup"))
def post_writeup(ctx, params: WriteupParams) -> dict:
    from sparc.core import progress, runio
    from sparc.core.writeup import methods_markdown, model_card_markdown

    run_dir = Path(ctx.run_dir)
    m = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))     # the merged manifest
    files = []
    for name, text in (("methods.md", methods_markdown(m)), ("model_card.md", model_card_markdown(m))):
        runio.write_text_atomic(run_dir / name, text)
        progress.artifact(run_dir / name, role="docs")
        files.append(name)
    return {"files": files}


# ---------------------------------------------------------------------------
# server-side hooks
# ---------------------------------------------------------------------------

def _register_child(sctx, job: dict, run_dir: Path) -> str | None:
    """Index a study's child run (``origin = study_child``) and link it to the study and its parent run.

    While it runs, the child's own ``run_state.json`` names a live process and no ``run.core`` job of its own,
    which the registry reads as a CLI run (``external_live``) and would follow with a pseudo-job; the row is
    put back to ``study_child`` / ``running`` (the study's job tracks it).  A finished child keeps the origin
    the registry derives for it (``reproduction`` for a reproduction)."""
    reg = sctx.services.get("registry") if sctx is not None else None
    if reg is None or not run_dir:
        return None
    rd = Path(run_dir)
    if not rd.is_dir():
        return None
    try:
        row = reg.index_run_dir(rd, origin="study_child", studio_dir=rd / "studio", study_id=job.get("study_id"),
                                project_id=job.get("project_id"))
    except Exception:
        log.exception("could not register the study child %s", rd)
        return None
    fix = {}
    if row.get("origin") not in ("study_child", "reproduction"):
        fix["origin"] = "study_child"
    if row.get("status") == "external_live":
        fix["status"] = "running"
    if job.get("run_id") and not row.get("parent_run_id"):
        fix["parent_run_id"] = job["run_id"]
    if job.get("study_id") and row.get("study_id") != job["study_id"]:
        fix["study_id"] = job["study_id"]
    if fix:
        sctx.db.update("runs", {"id": row["id"]}, fix)
        if "origin" in fix or "status" in fix:
            try:
                sctx.hub.publish("run.updated", {"run_id": row["id"], "project_id": row.get("project_id"),
                                                 "status": fix.get("status", row.get("status")),
                                                 "fields": sorted(fix)})
            except Exception:
                pass
    return row["id"]


def _refresh_children(sctx, job: dict) -> None:
    """Re-derive the study's child rows (a nested run ended)."""
    for r in sctx.db.fetchall("SELECT run_dir FROM runs WHERE study_id = ?", (job.get("study_id"),)):
        _register_child(sctx, job, Path(r["run_dir"]))


def study_on_event(sctx, job: dict, ev: dict) -> None:
    sid = job.get("study_id")
    if not sid or sctx is None:
        return
    t = ev.get("type")
    if t == "run.dir" and ev.get("run_dir"):
        _register_child(sctx, job, Path(ev["run_dir"]))
    elif t == "run.end":
        _refresh_children(sctx, job)
    elif t == "job.status":
        st = ev.get("status")
        if st in ("starting", "running", "cancelling"):
            row = sctx.db.fetchone("SELECT status FROM studies WHERE id = ?", (sid,))
            if row is not None and row.get("status") != st:
                service.update_study(sctx.db, sid, hub=sctx.hub, status=st)


def _final_summary(kind: str, out_dir: Path, result: dict | None, child_dir: str | None) -> dict | None:
    if kind == "placebo":
        data = read_json(out_dir / "placebo.json")
    elif kind == "simcheck":
        data = read_json(out_dir / "simcheck_summary.json") or (result or {}).get("summary")
    elif kind == "multiverse":
        data = read_json(out_dir / "multiverse_summary.json")
    elif kind == "reproduce":
        data = read_json(Path(child_dir) / "reproduce.json") if child_dir else None
    elif kind == "benchmark":
        data = read_json(out_dir / "benchmark.json")
    else:
        data = None
    return service.summary_for(kind, data)


def study_on_finish(sctx, job: dict, result) -> None:
    """Record the study's end: status, summary, child run ids, the parent's refresh, auto uncertainty."""
    sid = job.get("study_id")
    if not sid or sctx is None:
        return
    db = sctx.db
    row = db.fetchone("SELECT * FROM studies WHERE id = ?", (sid,))
    if row is None:
        return
    kind = row["kind"]
    status = job.get("status") or "failed"
    result = dict(result or {})
    children = [c for c in (result.get("children") or []) if c]
    child_dir = result.get("child_run_dir") if kind == "reproduce" else None
    if child_dir:
        children = [child_dir]
    ids = [i for i in (_register_child(sctx, job, Path(c)) for c in children) if i]
    if kind == "multiverse" or kind == "placebo":   # children of a resumed study that this pass did not touch
        known = {r["id"] for r in db.fetchall("SELECT id FROM runs WHERE study_id = ?", (sid,))}
        ids = list(dict.fromkeys(ids + sorted(known)))
    if result and (children or kind == "reproduce"):
        if kind == "reproduce":
            result["child_run_id"] = ids[0] if ids else None
        else:
            result["children"] = ids
        db.update("jobs", {"id": job["id"]}, {"result_json": dbmod.dumps(result)})
    old = dbmod.loads(row.get("summary_json")) or {}
    summary = _final_summary(kind, Path(row["out_dir"]), result, child_dir) if status == "succeeded" else None
    if summary is None:
        summary = old or None
    elif old.get("label") and not summary.get("label"):
        summary["label"] = old["label"]
    service.update_study(db, sid, hub=sctx.hub, status=status, summary=summary)
    reg = sctx.services.get("registry")
    if reg is not None:
        for rid in [*ids, job.get("run_id")]:
            if rid:
                try:
                    reg.refresh(rid)
                except Exception:
                    log.exception("refreshing run %s failed", rid)
    if status == "succeeded" and kind in service.ATTACHABLE:
        try:
            auto = bool(sctx.settings().auto_uncertainty)
        except Exception:
            auto = True
        if auto:
            for rid in service.attached_runs(db, sid):
                enqueue_uncertainty(sctx, rid, wait=False)


def enqueue_uncertainty(sctx, run_id: str, *, wait: bool = True):
    """Queue ``post.uncertainty`` on ``run_id`` unless one is already waiting there.

    From a hook thread (``wait=False``) the submission is scheduled on the server loop; from a coroutine use
    ``await sctx.jobs.submit(...)`` through :func:`submit_uncertainty`."""
    if sctx is None or getattr(sctx, "jobs", None) is None:
        return None
    if sctx.db.fetchone("SELECT id FROM jobs WHERE run_id = ? AND kind = 'post.uncertainty' "
                        "AND status IN ('queued', 'blocked')", (run_id,)) is not None:
        return None
    coro = submit_uncertainty(sctx, run_id)
    loop = getattr(sctx.hub, "_loop", None)
    if loop is None or loop.is_closed():
        coro.close()
        return None
    fut = asyncio.run_coroutine_threadsafe(coro, loop)

    def done(f) -> None:
        if not f.cancelled() and f.exception() is not None:
            log.warning("auto uncertainty on %s failed: %s", run_id, f.exception())

    fut.add_done_callback(done)
    return fut.result(timeout=30) if wait else None


async def submit_uncertainty(sctx, run_id: str) -> dict | None:
    """``post.uncertainty`` on the attached studies of ``run_id`` (``None`` when one is already waiting)."""
    if sctx.db.fetchone("SELECT id FROM jobs WHERE run_id = ? AND kind = 'post.uncertainty' "
                        "AND status IN ('queued', 'blocked')", (run_id,)) is not None:
        return None
    run = sctx.db.fetchone("SELECT project_id, run_dir FROM runs WHERE id = ?", (run_id,))
    if run is None or not (Path(run["run_dir"]) / "manifest.json").is_file():
        return None
    return await sctx.jobs.submit("post.uncertainty", {}, run_id=run_id, project_id=run.get("project_id"),
                                  label="Uncertainty report (attached studies)")


# ---------------------------------------------------------------------------
# studies
# ---------------------------------------------------------------------------

def _study_estimate(kind: str):
    def est(sctx, job, params):
        row = sctx.db.fetchone("SELECT * FROM runs WHERE id = ?", (job.get("run_id"),)) if job.get("run_id") else None
        return estimate_kind(sctx.db, kind, row, params, study_id=job.get("study_id"))
    return est


def _pool_threads(settings, params) -> int:
    """A pool study holds ``workers × threads`` threads of the heavy slot."""
    w = int(params.get("workers") or 1)
    t = int(params.get("threads") or 1)
    return max(1, w * t)


@job_kind("study.placebo", lane="heavy", label="Placebo study", params=PlaceboParams, needs_run=True, long=True,
          estimate=_study_estimate("placebo"), on_event=study_on_event,
          on_event_types=("run.dir", "run.end", "job.status"), on_finish=study_on_finish)
def study_placebo(ctx, params: PlaceboParams) -> dict:
    from sparc.core import progress, runio
    from sparc.core.placebo import placebo_markdown, run_placebo_suite

    cfg = run_config(ctx)
    out = _study_dir(ctx)
    res = run_placebo_suite(cfg, kinds=tuple(params.kinds), coarse=params.coarse_m, seed=int(params.seed),
                            grf_range_m=float(params.grf_range_m), children_dir=out / "children", resume=True,
                            run_meta=child_meta(ctx))
    runio.write_json_atomic(out / "placebo.json", res, indent=1)
    runio.write_text_atomic(out / "placebo.md", placebo_markdown(res, _units(cfg)) + "\n")
    for name in ("placebo.json", "placebo.md"):
        progress.artifact(out / name, role="placebo")
    children = [r["run_dir"] for r in (res.get("runs") or {}).values() if r.get("run_dir")]
    return {"n_pass_model": res.get("n_pass_model"), "n_pass_causal": res.get("n_pass_causal"),
            "n_placebos": res.get("n_placebos"), "children": children}


@job_kind("study.simcheck", lane="heavy", label="Simulation check", params=SimcheckParams, needs_run=True, long=True,
          estimate=_study_estimate("simcheck"), threads=_pool_threads,
          on_event=study_on_event, on_event_types=("job.status",),
          on_finish=study_on_finish)
def study_simcheck(ctx, params: SimcheckParams) -> dict:
    from sparc.core import progress, runio
    from sparc.core.simcheck import run_simcheck, simcheck_markdown

    out = _study_dir(ctx)
    summ = run_simcheck(run_config(ctx), {k: int(v) for k, v in params.design.items() if v}, out,
                        coarse=params.coarse_m, epochs=int(params.epochs), workers=int(params.workers),
                        threads=int(params.threads))
    runio.write_text_atomic(out / "simcheck_summary.md", simcheck_markdown(summ) + "\n")
    for name in ("simcheck_summary.json", "simcheck_summary.md"):
        progress.artifact(out / name, role="simcheck")
    return {"n_rows": summ.get("n_rows"), "n_errors": summ.get("n_errors"), "summary": summ}


@job_kind("study.multiverse", lane="heavy", label="Multiverse study", params=MultiverseParams, needs_run=True,
          long=True, estimate=_study_estimate("multiverse"), threads=_pool_threads,
          on_event=study_on_event, on_event_types=("run.dir", "run.end", "job.status"),
          on_finish=study_on_finish)
def study_multiverse(ctx, params: MultiverseParams) -> dict:
    from sparc.core import progress, runio
    from sparc.core.multiverse import multiverse_markdown, run_multiverse

    cfg = run_config(ctx)
    out = _study_dir(ctx)
    cfg.raw.setdefault("output", {})["dir"] = str(out / "children")    # children follow cfg.output.dir
    summ = run_multiverse(cfg, out, variants=params.variants or None, coarse=params.coarse_m,
                          workers=int(params.workers), threads=int(params.threads),
                          extra_variants=params.custom_variants or None, run_meta=child_meta(ctx))
    runio.write_text_atomic(out / "multiverse_summary.md", multiverse_markdown(summ, _units(cfg)) + "\n")
    for name in ("multiverse_summary.json", "multiverse_summary.md"):
        progress.artifact(out / name, role="multiverse")
    children = []
    for p in sorted(out.glob("*.json")):
        doc = read_json(p)
        if isinstance(doc, dict) and doc.get("variant") and doc.get("run_dir"):
            children.append(doc["run_dir"])
    return {"sign_stability_min": summ.get("sign_stability_min"), "median_kendall_tau": summ.get("median_kendall_tau"),
            "children": children}


def reproduce_config_dir(ctx) -> str | None:
    """``launch.json`` ``config_dir``, else the folder of the config given at import, else the manifest's."""
    launch = ctx.launch or {}
    if launch.get("config_dir"):
        return str(launch["config_dir"])
    rec = read_json(Path(ctx.studio_dir) / "import.json") if ctx.studio_dir else None
    if isinstance(rec, dict) and rec.get("config_path"):
        return str(Path(rec["config_path"]).resolve().parent)
    m = read_json(Path(ctx.run_dir) / "manifest.json") or {}
    cd = (m.get("provenance") or {}).get("config_dir")
    return str(cd) if cd else None


@job_kind("study.reproduce", lane="heavy", label="Reproduction", params=ReproduceParams, needs_run=True, long=True,
          estimate=_study_estimate("reproduce"), on_event=study_on_event,
          on_event_types=("run.dir", "run.end", "job.status"), on_finish=study_on_finish)
def study_reproduce(ctx, params: ReproduceParams) -> dict:
    from sparc.core.reproduce import reproduce

    run_dir = Path(ctx.run_dir)
    out = _study_dir(ctx) / "children" / f"{run_dir.name}_reproduce"
    res = reproduce(run_dir, stages=tuple(params.stages), tol_r2=float(params.tol_r2),
                    tol_effect=float(params.tol_effect), config_dir=reproduce_config_dir(ctx), out_dir=out,
                    run_meta=child_meta(ctx))
    n_hard = sum(1 for c in res.get("checks") or [] if c.get("hard") and not c.get("ok"))
    return {"pass": bool(res.get("pass")), "child_run_id": None, "child_run_dir": res.get("reproduction"),
            "n_hard_fail": n_hard}


@job_kind("study.benchmark", lane="heavy", label="Effect benchmark", params=BenchmarkParams, long=True,
          estimate=_study_estimate("benchmark"), on_event=study_on_event,
          on_event_types=("job.status",), on_finish=study_on_finish)
def study_benchmark(ctx, params: BenchmarkParams) -> dict:
    from sparc.core import progress, runio
    from sparc.core.diagnostics import benchmark_markdown, run_benchmark

    out = _study_dir(ctx)
    bench = run_benchmark(seed=int(params.seed), spatial_plus_ab=bool(params.ab), epochs=int(params.epochs),
                          n=int(params.n))
    runio.write_json_atomic(out / "benchmark.json", bench, indent=2)
    runio.write_text_atomic(out / "benchmark.md", benchmark_markdown(bench) + "\n")
    for name in ("benchmark.json", "benchmark.md"):
        progress.artifact(out / name, role="benchmark")
    return {"runs": bench.get("runs") or {}}


# ---------------------------------------------------------------------------
# estimates (job ETAs and POST /api/studies/estimate)
# ---------------------------------------------------------------------------

REF_CELLS = 54_701                     # full Providence (the seed rates of SPEC §5.4)
REF_RUN_S = 6_096.6                    # its recorded full run
VARIANT_S = 1_200.0                    # a multiverse variant: S0–S5 at 60 m on Providence (13,675 cells)
VARIANT_CELLS = 13_675
REPLICATE_S = 254.0                    # a simcheck replicate: S0–S6 at 90 m on Providence (6,078 cells)
REPLICATE_CELLS = 6_078


def _p(params, key, default=None):
    if params is None:
        return default
    if isinstance(params, BaseModel):
        return getattr(params, key, default)
    return params.get(key, default)


def _fine_cells(run_row: dict | None) -> tuple[float, float]:
    """``(cells of the run at its own resolution, cell size in m)``."""
    if not run_row:
        return float(REF_CELLS), 30.0
    n = float(run_row.get("n_points") or 0) or float(REF_CELLS)
    st = read_json(Path(run_row["run_dir"]) / "run_state.json") or {}
    meta = st.get("meta") or {}
    cell = float(meta.get("cell_m") or 30.0)
    co = run_row.get("coarse_m") or meta.get("coarse_m")
    if co and float(co) > cell:
        n *= (float(co) / cell) ** 2            # the full-resolution cell count behind a coarse run
    return n, cell


def _coarse_cells(n_fine: float, cell: float, coarse) -> float:
    if not coarse or float(coarse) <= cell:
        return n_fine
    return max(1.0, n_fine * (cell / float(coarse)) ** 2)


def _run_seconds(run_row: dict | None) -> float:
    if not run_row:
        return REF_RUN_S
    info = dbmod.loads(run_row.get("stages_json"), {}) or {}
    d = info.get("duration_s") if isinstance(info, dict) else None
    if d:
        return float(d)
    return float(run_row.get("n_points") or REF_CELLS) * REF_RUN_S / REF_CELLS


def _timings(run_row: dict | None) -> dict:
    if not run_row:
        return {}
    m = read_json(Path(run_row["run_dir"]) / "manifest.json") or {}
    return dict(m.get("timings_s") or {})


def estimate_kind(db, kind: str, run_row: dict | None, params=None, *, study_id: str | None = None) -> dict:
    """``{est_s, est_lo, est_hi, peak_ram_gb, disk_bytes, n_children}`` of a study or post-run action.

    Pool studies divide by their workers; the coarse children scale with their cell count (power 1.15) from
    the Providence references above; post-run actions scale with the run's cells."""
    from sparc.studio.jobs.eta import checkpoint_bytes, peak_ram_gb

    n_fine, cell = _fine_cells(run_row)
    n_run = float((run_row or {}).get("n_points") or n_fine)
    T = _run_seconds(run_row)
    children = 0
    ram = 0.8
    disk = 0.0
    if kind == "placebo":
        nc = _coarse_cells(n_fine, cell, _p(params, "coarse_m", 60.0))
        children = len(_p(params, "kinds") or ["grf", "shift", "rotate"])
        est = children * 1.3 * VARIANT_S * (nc / VARIANT_CELLS) ** 1.15
        ram, disk = peak_ram_gb(nc, 5), children * checkpoint_bytes(nc)
    elif kind == "simcheck":
        design = dict(_p(params, "design") or {})
        n_rep = sum(int(v or 0) for v in design.values())
        done = 0
        sdir = None
        if study_id or _p(params, "continue_study_id"):
            r = db.fetchone("SELECT out_dir FROM studies WHERE id = ?", (study_id or _p(params, "continue_study_id"),))
            sdir = Path(r["out_dir"]) if r and r.get("out_dir") else None
        if sdir is not None and (sdir / "simcheck.jsonl").is_file():
            seen = set()
            for line in (sdir / "simcheck.jsonl").read_text("utf-8").splitlines():
                try:
                    rr = json.loads(line)
                except ValueError:
                    continue
                if "error" not in rr and rr.get("seed", 0) < int(design.get(rr.get("generator"), 0) or 0):
                    seen.add((rr.get("generator"), rr.get("seed")))
            done = len(seen)
        todo = max(0, n_rep - done)
        nc = _coarse_cells(n_fine, cell, _p(params, "coarse_m", 90.0))
        epochs = float(_p(params, "epochs", 200) or 200)
        workers = max(1, int(_p(params, "workers", 1) or 1))
        per = REPLICATE_S * (nc / REPLICATE_CELLS) ** 1.15 * (0.6 + 0.4 * epochs / 200.0)
        est = math.ceil(todo / workers) * per if todo else 0.0
        ram = peak_ram_gb(nc, 5) * min(workers, max(todo, 1))
    elif kind == "multiverse":
        from sparc.core.multiverse import VARIANTS

        names = list(_p(params, "variants") or VARIANTS) + [n for n in (_p(params, "custom_variants") or {})
                                                            if n not in (_p(params, "variants") or ())]
        children = len(names)
        nc = _coarse_cells(n_fine, cell, _p(params, "coarse_m", 60.0))
        workers = max(1, int(_p(params, "workers", 1) or 1))
        est = math.ceil(children / workers) * VARIANT_S * (nc / VARIANT_CELLS) ** 1.15
        ram, disk = peak_ram_gb(nc, 5) * min(workers, children), children * checkpoint_bytes(nc)
    elif kind == "reproduce":
        stages = set(_p(params, "stages") or ["S0", "S1", "S2", "S3"])
        tm = _timings(run_row)
        if tm:
            names = {"S2": "S2_S3", "S3": "S2_S3"}
            keys = {names.get(s, s) for s in stages}
            est = sum(float(v) for k, v in tm.items() if k in keys) or T * 0.6
        else:
            est = T * (0.6 if stages <= {"S0", "S1", "S2", "S3"} else 1.0)
        children = 1
        ram, disk = peak_ram_gb(n_run, 5), checkpoint_bytes(n_run)
    elif kind == "benchmark":
        n = int(_p(params, "n", 96) or 96)
        runs = 2 if _p(params, "ab", True) else 1
        est = runs * 120.0 * (n / 96.0) ** 2.3 * (0.5 + 0.5 * float(_p(params, "epochs", 150) or 150) / 150.0)
        ram = peak_ram_gb(n * n, 5)
    elif kind == "baselines":
        k = 5
        models = len(_p(params, "models") or BASELINE_MODELS)
        est = 10.0 + models * k * 12.0 * (n_run / REF_CELLS)
    elif kind == "planner":
        est = 20.0 + n_run * 1e-3
    elif kind == "emulator":
        try:
            m = read_json(Path(run_row["run_dir"]) / "manifest.json") if run_row else {}
            levers = len(((m or {}).get("config") or {}).get("actionable") or {}) or 3
        except Exception:
            levers = 3
        patches = int(_p(params, "patches", 8) or 8)
        est = 26.0 * n_run / REF_CELLS + levers * (2 + patches) * 13.0 * (n_run / REF_CELLS)
        ram = peak_ram_gb(n_run, 5)
    elif kind == "uncertainty":
        est = 5.0
    elif kind == "writeup":
        est = 2.0
    else:
        raise ValueError(f"no estimate for {kind!r}")
    est = max(float(est), 1.0)
    return {"est_s": round(est, 1), "est_lo": round(0.5 * est, 1), "est_hi": round(2.0 * est, 1),
            "peak_ram_gb": round(float(ram), 2), "disk_bytes": int(disk), "n_children": int(children)}
