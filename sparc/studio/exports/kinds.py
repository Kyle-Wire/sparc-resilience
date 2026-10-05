"""Export job kinds (SPEC §6.7, §10.3, api.md §8): ``export.bundle``, ``export.gis``, ``export.page``,
``export.report`` and ``export.findings`` (medium lane, process executor).

``export_id`` is filled by ``POST /api/exports`` (:func:`sparc.studio.exports.store.create_export`), which
created the ``exports`` row; each kind writes under ``projects/<slug>/exports/<export_id>/`` and its
``on_finish`` completes the row (status, path, bytes).  Heavy imports stay inside the job functions.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from sparc.studio.exports import store
from sparc.studio.jobs.kinds import job_kind

log = logging.getLogger("sparc.studio.exports")

__all__ = ["BundleParams", "GisParams", "PageParams", "ReportParams", "FindingsParams", "export_on_finish"]

ReportSection = Literal["summary", "accuracy", "validation", "scenarios", "plans", "climate", "heat", "equity",
                        "caveats", "limitations", "provenance", "findings"]


class BundleParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    export_id: str
    run_id: str
    outputs: list[str] | None = None
    include_checkpoint: bool = False


class GisParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    export_id: str
    run_id: str
    layers: list[str] | None = None


class PageParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    export_id: str
    run_id: str
    placebo_study: str | None = None


class ReportParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    export_id: str
    run_id: str
    sections: list[ReportSection] = Field(min_length=1)
    result_ids: list[str] | None = None
    plan_ids: list[str] | None = None
    finding_ids: list[str] | None = None
    format: Literal["html", "md"] = "html"

    @field_validator("sections")
    @classmethod
    def _unique(cls, v):
        return list(dict.fromkeys(v))


class FindingsParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    export_id: str
    project_id: str
    run_id: str | None = None
    ids: list[str] | None = None
    format: Literal["md", "html"] = "html"


def export_on_finish(sctx, job: dict, result) -> None:
    eid = (job.get("params") or {}).get("export_id")
    if eid and sctx is not None:
        store.complete(sctx.db, eid, job, result)


def _estimate(seconds: float):
    def est(sctx, job, params):
        return {"est_s": seconds, "est_lo": 0.5 * seconds, "est_hi": 2.0 * seconds}
    return est


def _export_dir(ctx, eid: str) -> Path:
    if ctx.project_dir is None:
        raise ValueError("an export needs a project")
    d = Path(ctx.project_dir) / "exports" / eid
    d.mkdir(parents=True, exist_ok=True)
    return d


def _run_ctx(ctx, run_id: str):
    """A ``RunContext`` of ``run_id`` from the read-only database (the worker has no server reader)."""
    from sparc.studio.runs.reader import RunContext

    row = ctx.db.fetchone("SELECT * FROM runs WHERE id = ?", (run_id,))
    if row is None:
        raise LookupError(f"no run {run_id!r}")
    return RunContext(row, ("export", time.time()))


def _slug(text: str) -> str:
    from sparc.studio.workspace import slugify

    return slugify(str(text), max_len=40)


def _ticker():
    from sparc.core import progress

    def cb(k: int, n: int, what: str) -> None:
        progress.tick(k, n, unit="files", label=what)
    return cb


@job_kind("export.bundle", lane="medium", label="Run bundle", params=BundleParams, on_finish=export_on_finish,
          estimate=_estimate(20.0))
def export_bundle(ctx, params: BundleParams) -> dict:
    from sparc.studio.exports.bundle import build_bundle

    rc = _run_ctx(ctx, params.run_id)
    out = _export_dir(ctx, params.export_id) / f"{_slug(rc.row.get('label') or rc.run_id)}_{rc.run_id}_bundle.zip"
    res = build_bundle(rc.row, rc.run_dir, rc.cfg_raw, out, outputs=params.outputs,
                       include_checkpoint=params.include_checkpoint, progress_cb=_ticker())
    return {"export_id": params.export_id, "path": res["path"], "bytes": res["bytes"]}


@job_kind("export.gis", lane="medium", label="GIS pack", params=GisParams, on_finish=export_on_finish,
          estimate=_estimate(30.0))
def export_gis(ctx, params: GisParams) -> dict:
    from sparc.studio.exports.gis import build_gis

    rc = _run_ctx(ctx, params.run_id)
    out = _export_dir(ctx, params.export_id) / f"{rc.run_id}_gis.zip"
    res = build_gis(rc, out, layers=params.layers, progress_cb=_ticker())
    return {"export_id": params.export_id, "path": res["path"], "bytes": res["bytes"], "files": res["files"]}


@job_kind("export.page", lane="medium", label="Standalone results page", params=PageParams,
          on_finish=export_on_finish, estimate=_estimate(30.0))
def export_page(ctx, params: PageParams) -> dict:
    from sparc.core.results_page import build_results_page
    from sparc.studio.studies.kinds import run_config

    rc = _run_ctx(ctx, params.run_id)
    placebo = None
    if params.placebo_study:
        row = ctx.db.fetchone("SELECT out_dir FROM studies WHERE id = ?", (params.placebo_study,))
        if row is None or not (Path(row["out_dir"]) / "placebo.json").is_file():
            raise ValueError(f"study {params.placebo_study!r} has no placebo.json")
        placebo = Path(row["out_dir"]) / "placebo.json"
    out = build_results_page(rc.run_dir, run_config(ctx), _export_dir(ctx, params.export_id) / "results.html",
                             placebo_path=placebo)
    return {"export_id": params.export_id, "path": str(out), "bytes": out.stat().st_size}


@job_kind("export.report", lane="medium", label="Project report", params=ReportParams, on_finish=export_on_finish,
          estimate=_estimate(15.0))
def export_report(ctx, params: ReportParams) -> dict:
    from sparc.core import runio
    from sparc.studio.exports.report import build_report

    rc = _run_ctx(ctx, params.run_id)
    project = ctx.db.fetchone("SELECT * FROM projects WHERE id = ?", (ctx.project_id,)) if ctx.project_id else None
    text = build_report(rc, ctx.db, params.sections, result_ids=params.result_ids or [], plan_ids=params.plan_ids or [],
                        finding_ids=params.finding_ids, fmt=params.format, project=project)
    out = _export_dir(ctx, params.export_id) / f"report.{params.format}"
    runio.write_text_atomic(out, text)
    return {"export_id": params.export_id, "path": str(out), "bytes": out.stat().st_size}


@job_kind("export.findings", lane="medium", label="Findings", params=FindingsParams, on_finish=export_on_finish,
          estimate=_estimate(5.0))
def export_findings_job(ctx, params: FindingsParams) -> dict:
    from sparc.studio.exports.findings import export_findings

    res = export_findings(ctx.db, params.project_id, _export_dir(ctx, params.export_id), run_id=params.run_id,
                          ids=params.ids, fmt=params.format)
    return {"export_id": params.export_id, "path": res["path"], "bytes": res["bytes"]}
