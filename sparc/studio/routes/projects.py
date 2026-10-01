"""Projects, files, data check and preview, config and link endpoints (api.md §5–5.3).

Everything that touches files or SQLite runs in a worker thread; the data
check additionally runs under ``threadpool_limits(1)`` (SPEC §10.5).  At
server start (and ``--reindex``) project folders are registered from their
``project.json``.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import re
from pathlib import Path

from fastapi import APIRouter, Body, Depends, Header, Query, Request
from fastapi.responses import Response

from sparc.studio.app import StudioContext, get_ctx, startup_hook
from sparc.studio.db import reindex_hook
from sparc.studio.errors import ApiError
from sparc.studio.projects import config_service as cs
from sparc.studio.projects import files as pfiles
from sparc.studio.projects import service, templates
from sparc.studio.projects.config_schema import config_json_schema
from sparc.studio.projects.datacheck import PreviewStore, data_check, dumps_header, preview_column
from sparc.studio.projects.readiness import readiness, readiness_score
from sparc.studio.projects.schemas import (
    ColumnSuggestion,
    ConfigCandidate,
    ConfigOut,
    ConfigPut,
    ConfigSaved,
    DataCheck,
    DataCheckRequest,
    FileInspect,
    FileRow,
    FileUploaded,
    HistoryRow,
    HistoryVersion,
    ImpactOut,
    LinkOut,
    LinkRequest,
    Project,
    ProjectCreate,
    ProjectCreated,
    ProjectDetail,
    ProjectImport,
    ProjectImported,
    ProjectPatch,
    SectionPatch,
    SuggestRequest,
    ValidateOut,
)
from sparc.studio.projects.validate import validate_deep, validate_report
from sparc.studio.schemas.common import Ok
from sparc.studio.security import stream_upload, upload_cap_bytes

router = APIRouter(tags=["projects"])

PREVIEWS = "projects.previews"


@startup_hook("projects.scan", phase="scan", order=5)
def _scan_on_start(sctx) -> None:
    """Register project folders that have no row yet (a deleted database, a features job's ``_open``)."""
    service.scan_projects(sctx.db, sctx.workspace)


@reindex_hook("projects", order=5)
def _reindex(db, workspace) -> dict:
    return service.scan_projects(db, workspace)


def _previews(sctx: StudioContext) -> PreviewStore:
    store = sctx.services.get(PREVIEWS)
    if store is None:
        store = sctx.services.setdefault(PREVIEWS, PreviewStore())
    return store


def _row(sctx: StudioContext, pid: str) -> dict:
    return service.get_row(sctx.db, pid)


def _out(sctx: StudioContext, row: dict) -> dict:
    raw = service.project_raw(row)
    rows = readiness(sctx.db, row, raw)
    return service.project_out(sctx.db, row, readiness_score=readiness_score(rows), raw=raw)


# ---------------------------------------------------------------------------
# projects
# ---------------------------------------------------------------------------

@router.get("/projects", response_model=list[Project])
async def list_projects(archived: bool = Query(False), sctx: StudioContext = Depends(get_ctx)):
    """Projects (active ones by default), most recently updated first."""
    def build():
        return [_out(sctx, r) for r in service.rows(sctx.db, archived=archived)]
    return await asyncio.to_thread(build)


@router.post("/projects", response_model=ProjectCreated, status_code=201)
async def create_project(body: ProjectCreate, sctx: StudioContext = Depends(get_ctx)):
    """Create a project from a template (blank, synthetic_demo, providence_example)."""
    opts = body.options.model_dump(exclude_unset=True) if body.options else {}
    res = await templates.create_from_template(sctx, body.name.strip(), body.template, opts)
    project = await asyncio.to_thread(_out, sctx, service.get_row(sctx.db, res["row"]["id"]))
    return {"project": project, "imported_runs": res["imported_runs"], "warnings": res["warnings"]}


@router.post("/projects/import", response_model=ProjectImported, status_code=201)
async def import_project(body: ProjectImport, sctx: StudioContext = Depends(get_ctx)):
    """Import an existing core config (paths made absolute, or the data copied in), plus run and study folders."""
    res = await templates.import_config(sctx, config_path=body.config_path, name=body.name,
                                        copy_data=body.copy_data, run_dirs=body.run_dirs,
                                        study_dirs=body.study_dirs, trust_pickles=body.trust_pickles)
    project = await asyncio.to_thread(_out, sctx, service.get_row(sctx.db, res["row"]["id"]))
    return {"project": project, "runs": res["runs"], "studies": res["studies"], "warnings": res["warnings"]}


@router.get("/projects/{pid}", response_model=ProjectDetail)
async def get_project(pid: str, sctx: StudioContext = Depends(get_ctx)):
    """The project with its readiness spine, runs, active jobs and current config version."""
    def build():
        row = _row(sctx, pid)
        version, _text = cs.sync_versions(sctx.db, row)
        raw = service.project_raw(row)
        spine = readiness(sctx.db, row, raw)
        return {"project": service.project_out(sctx.db, row, readiness_score=readiness_score(spine), raw=raw),
                "readiness": spine, "runs": service.project_runs(sctx.db, pid),
                "active_jobs": service.active_jobs(sctx.db, pid), "config_version": version}
    return await asyncio.to_thread(build)


@router.patch("/projects/{pid}", response_model=Project)
async def patch_project(pid: str, body: ProjectPatch, sctx: StudioContext = Depends(get_ctx)):
    """Rename, archive, set the active run, headline scenario, cost model or report text."""
    data = body.model_dump(exclude_unset=True)
    if "report" in data and data["report"] is not None:
        data["report"] = body.report.model_dump(exclude_unset=True)

    def run():
        row = service.patch_project(sctx.db, pid, data)
        return _out(sctx, row)
    return await asyncio.to_thread(run)


@router.delete("/projects/{pid}", response_model=Ok)
async def delete_project(pid: str, files: bool = Query(False), sctx: StudioContext = Depends(get_ctx)):
    """Remove the project (``files=true`` also deletes its folder).  ``409 active`` while jobs run."""
    await asyncio.to_thread(service.delete_project, sctx.db, sctx.workspace, pid, files=files)
    return {"ok": True}


# ---------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------

_NAME = re.compile(r"^[^/\\\x00]+$")


@router.put("/projects/{pid}/files/{kind}/{filename}", response_model=FileUploaded, status_code=201,
            openapi_extra={"requestBody": {"required": True, "content": {"application/octet-stream": {
                "schema": {"type": "string", "format": "binary"}}}}})
async def upload_file(pid: str, kind: str, filename: str, request: Request,
                      x_overwrite: str | None = Header(None, alias="X-Overwrite"),
                      sctx: StudioContext = Depends(get_ctx)):
    """Raw streamed upload (the body is the file).  ``X-Overwrite: 1`` replaces an existing file."""
    row = await asyncio.to_thread(_row, sctx, pid)
    d = pfiles.kind_dir(row["dir"], kind)
    if not _NAME.match(filename) or filename.startswith(".") or filename in ("..",):
        raise ApiError("validation", f"bad file name {filename!r}",
                       detail={"errors": [{"path": "filename", "message": "a plain file name", "code": "bad_name"}]})
    d.mkdir(parents=True, exist_ok=True)
    dest = pfiles.resolve_path(row["dir"], str(Path(pfiles.KIND_DIRS[kind]) / filename))
    if dest.exists() and str(x_overwrite or "").strip() not in ("1", "true", "yes"):
        raise ApiError("exists", f"{filename} already exists (send X-Overwrite: 1 to replace it)",
                       detail={"path": dest.relative_to(Path(row["dir"]).resolve()).as_posix()})
    n = await stream_upload(request, dest, max_bytes=upload_cap_bytes(sctx.settings()))
    inspect = None
    if dest.suffix.lower() in pfiles.TABLE_SUFFIXES:
        try:
            inspect = await asyncio.to_thread(pfiles.inspect_file, dest, 50000)
        except ApiError:
            inspect = None
    rel = dest.relative_to(Path(row["dir"]).resolve()).as_posix()
    return {"path": rel, "bytes": n, "kind": kind, "inspect": inspect}


@router.get("/projects/{pid}/files", response_model=list[FileRow])
async def list_files(pid: str, sctx: StudioContext = Depends(get_ctx)):
    """Files under data/, inputs/* and other/, with the config keys that use each."""
    def run():
        row = _row(sctx, pid)
        return pfiles.list_files(row["dir"], service.project_raw(row))
    return await asyncio.to_thread(run)


@router.get("/projects/{pid}/files/inspect", response_model=FileInspect)
async def inspect_file(pid: str, path: str = Query(...), rows: int = Query(50000, ge=1, le=2_000_000),
                       sctx: StudioContext = Depends(get_ctx)):
    """Column stats and a preview of a CSV or parquet table (``rows`` rows read)."""
    def run():
        row = _row(sctx, pid)
        return pfiles.inspect_file(pfiles.resolve_path(row["dir"], path, sctx.paths.roots()), rows)
    return await asyncio.to_thread(run)


@router.delete("/projects/{pid}/files", response_model=Ok)
async def delete_file(pid: str, path: str = Query(...), sctx: StudioContext = Depends(get_ctx)):
    """Delete a project file; ``409 in_use`` when the config references it."""
    def run():
        row = _row(sctx, pid)
        p = pfiles.resolve_path(row["dir"], path)
        if not p.is_file():
            raise ApiError("not_found", f"no such file: {path}")
        users = pfiles.used_by(service.project_raw(row), row["dir"], p)
        if users:
            raise ApiError("in_use", f"{path} is used by {', '.join(users)}", detail={"used_by": users})
        p.unlink()
    await asyncio.to_thread(run)
    return {"ok": True}


@router.post("/projects/{pid}/columns/suggest", response_model=ColumnSuggestion)
async def suggest_columns(pid: str, body: SuggestRequest, sctx: StudioContext = Depends(get_ctx)):
    """Suggested mapping (target, id, x/y, zone, coordinate unit, CRS hint, predictors, roles)."""
    def run():
        row = _row(sctx, pid)
        return pfiles.suggest_columns(pfiles.resolve_path(row["dir"], body.path, sctx.paths.roots()))
    return await asyncio.to_thread(run)


# ---------------------------------------------------------------------------
# data check and preview
# ---------------------------------------------------------------------------

@router.post("/projects/{pid}/data/check", response_model=DataCheck)
async def check_data(pid: str, body: DataCheckRequest | None = Body(None), sctx: StudioContext = Depends(get_ctx)):
    """Run S0 inline on the saved config deep-merged with ``config_patch``."""
    def run():
        row = _row(sctx, pid)
        resp, payload = data_check(service.project_raw(row), row["dir"], (body.config_patch if body else None))
        resp["preview_token"] = _previews(sctx).put(pid, payload)
        return resp
    return await asyncio.to_thread(run)


def _bin(body: bytes, headers: dict) -> Response:
    return Response(content=body, media_type="application/octet-stream",
                    headers={"Cache-Control": "no-store", **headers})


@router.get("/projects/{pid}/data/preview/{token}/grid.bin", response_class=Response,
            responses={200: {"content": {"application/octet-stream": {}},
                             "description": "Packed ix, iy (int32), lon, lat (float32); GridMeta in X-SPARC-Grid"}})
async def preview_grid(pid: str, token: str, sctx: StudioContext = Depends(get_ctx)):
    """The preview grid (api.md §0.4 packed arrays) of a data check."""
    p = _previews(sctx).get(pid, token)
    meta = p["grid_meta"]
    return _bin(p["grid_body"], {"X-SPARC-Offsets": dumps_header(p["grid_offsets"]),
                                 "X-SPARC-Grid": dumps_header(meta), "X-SPARC-Length": str(meta["n"]),
                                 "ETag": f'"{meta["etag"]}"'})


@router.get("/projects/{pid}/data/preview/{token}/{name}", response_class=Response,
            responses={200: {"content": {"application/octet-stream": {}},
                             "description": "Float32 column in row order"}})
async def preview_column_bin(pid: str, token: str, name: str, sctx: StudioContext = Depends(get_ctx)):
    """One column of a data check as little-endian Float32 (``<column>.bin``)."""
    if not name.endswith(".bin"):
        raise ApiError("not_found", f"no such preview resource: {name}")
    p = _previews(sctx).get(pid, token)
    col = name[:-4]
    a = preview_column(p, col)
    return _bin(a.astype("<f4", copy=False).tobytes(), {
        "X-SPARC-Dtype": "float32", "X-SPARC-Length": str(int(a.size)),
        "ETag": f'"{p["grid_meta"]["etag"]}-{hashlib.sha1(col.encode()).hexdigest()[:10]}"'})


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def _candidate(row: dict, body: ConfigCandidate | None) -> dict:
    if body is not None and body.yaml is not None:
        return cs.core_block(cs.parse_yaml(body.yaml))
    if body is not None and body.raw is not None:
        raw = body.raw
        return dict(raw["core"]) if isinstance(raw.get("core"), dict) else dict(raw)
    return service.project_raw(row)


@router.get("/config/schema", tags=["config"])
async def config_schema():
    """JSON schema of ``CoreConfigModel`` with ``x-ui`` hints per property."""
    return config_json_schema()


@router.get("/projects/{pid}/config", response_model=ConfigOut)
async def get_config(pid: str, sctx: StudioContext = Depends(get_ctx)):
    """The saved config: text, raw block, effective values and what differs from DEFAULTS."""
    def run():
        return cs.get_config(sctx.db, _row(sctx, pid))
    return await asyncio.to_thread(run)


@router.put("/projects/{pid}/config", response_model=ConfigSaved)
async def put_config(pid: str, body: ConfigPut, if_match: str | None = Header(None, alias="If-Match"),
                     sctx: StudioContext = Depends(get_ctx)):
    """Save the config (``If-Match: <version>``; ``409 conflict`` when stale).  Issues never block a save."""
    version = cs.parse_if_match(if_match)

    def run():
        row = _row(sctx, pid)
        out = cs.save_config(sctx.db, row, text=body.yaml, raw=body.raw, if_match=version, note=body.note,
                             validate=lambda raw: validate_deep(raw, row["dir"]))
        service.touch(sctx.db, pid)
        return out
    return await asyncio.to_thread(run)


@router.patch("/projects/{pid}/config/sections/{section}", response_model=ConfigSaved)
async def patch_section(pid: str, section: str, body: SectionPatch,
                        if_match: str | None = Header(None, alias="If-Match"), sctx: StudioContext = Depends(get_ctx)):
    """Replace one top-level section (``value: null`` restores its defaults)."""
    version = cs.parse_if_match(if_match)

    def run():
        row = _row(sctx, pid)
        out = cs.patch_section(sctx.db, row, section, body.value, if_match=version, note=body.note,
                               validate=lambda raw: validate_deep(raw, row["dir"]))
        service.touch(sctx.db, pid)
        return out
    return await asyncio.to_thread(run)


@router.post("/projects/{pid}/config/validate", response_model=ValidateOut)
async def validate_config(pid: str, body: ConfigCandidate | None = Body(None), sctx: StudioContext = Depends(get_ctx)):
    """Deep validation of a candidate (or the saved) config, plus the fast/coarse previews."""
    def run():
        row = _row(sctx, pid)
        return validate_report(_candidate(row, body), row["dir"])
    return await asyncio.to_thread(run)


@router.post("/projects/{pid}/config/impact", response_model=ImpactOut)
async def config_impact(pid: str, body: ConfigCandidate | None = Body(None), sctx: StudioContext = Depends(get_ctx)):
    """Which fingerprint sections an edit changes, and what each run with a checkpoint would refit."""
    def run():
        row = _row(sctx, pid)
        return cs.impact(sctx.db, row, _candidate(row, body), sctx.workspace)
    return await asyncio.to_thread(run)


@router.get("/projects/{pid}/config/history", response_model=list[HistoryRow])
async def config_history(pid: str, sctx: StudioContext = Depends(get_ctx)):
    def run():
        row = _row(sctx, pid)
        cs.sync_versions(sctx.db, row)
        return cs.history(sctx.db, pid)
    return await asyncio.to_thread(run)


@router.get("/projects/{pid}/config/history/{version}", response_model=HistoryVersion)
async def config_history_version(pid: str, version: int, sctx: StudioContext = Depends(get_ctx)):
    def run():
        _row(sctx, pid)
        return cs.history_version(sctx.db, pid, version)
    return await asyncio.to_thread(run)


# ---------------------------------------------------------------------------
# link into config
# ---------------------------------------------------------------------------

def _config_value(pdir: Path, p: Path) -> str:
    """A path as the config should hold it: project-relative inside the project, else absolute."""
    try:
        return p.resolve().relative_to(pdir.resolve()).as_posix()
    except ValueError:
        return str(p.resolve())


def _climate_meta(p: Path) -> tuple[dict | None, list[str] | None]:
    import pandas as pd

    try:
        df = pd.read_csv(p, usecols=lambda c: c in ("experiment", "period"))
    except Exception:                                # noqa: BLE001
        return None, None
    periods = {}
    for name in sorted(map(str, df.get("period", pd.Series(dtype=str)).unique())):
        m = re.fullmatch(r"(\d{4})-(\d{4})", name)
        if m:
            periods[name] = [int(m.group(1)), int(m.group(2))]
    exps = sorted(map(str, df["experiment"].unique())) if "experiment" in df else None
    return (periods or None), exps


@router.post("/projects/{pid}/link", response_model=LinkOut)
async def link_input(pid: str, body: LinkRequest, sctx: StudioContext = Depends(get_ctx)):
    """Preview (``apply: false``) or apply linking a fetched or uploaded input into the config."""
    from sparc.studio.projects import inputs as pin

    def run():
        row = _row(sctx, pid)
        pdir = Path(row["dir"])
        p = pfiles.resolve_path(pdir, body.path, sctx.paths.roots())
        if not p.is_file():
            raise ApiError("not_found", f"no such file: {body.path}")
        value = _config_value(pdir, p)
        _version, text = cs.sync_versions(sctx.db, row)
        doc = cs.parse_yaml(text) if text.strip() else {}
        raw = cs.core_block(doc)
        others = {k: v for k, v in doc.items() if k != "core"} if "core" in doc else None
        if body.kind == "features_new_project":
            from sparc.studio.projects.config_service import dump_config

            new_raw = pin.open_project_raw(raw, pdir, pin.FEATURES_FILE)
            new_text = dump_config(new_raw)
            out = {"yaml_diff": cs.yaml_diff("", new_text, "/dev/null", f"{row['slug']}_open/config.yml"),
                   "applied": False}
            if body.apply:
                out["new_project_id"] = _create_open_project(sctx, row, p, new_raw)
                out["applied"] = True
            return out
        if body.kind == "forcing":
            edit = pin.link_forcing(value)
        elif body.kind == "climate":
            periods, exps = _climate_meta(p)
            edit = pin.link_climate(value, periods=periods, experiments=exps)
        elif body.kind == "layers":
            edit = pin.link_layers(value)
        else:
            edit = pin.link_features(value)
        new_raw = copy.deepcopy(raw)
        edit(new_raw)
        new_text = cs.dump_config(new_raw, others)
        out = {"yaml_diff": cs.yaml_diff(text, new_text, "config.yml", "config.yml (linked)"), "applied": False}
        if body.apply:
            saved = cs.save_config(sctx.db, row, text=new_text, note=f"linked {body.kind} {value}")
            service.touch(sctx.db, pid)
            out.update(applied=True, version=saved["version"])
        return out
    return await asyncio.to_thread(run)


def _create_open_project(sctx: StudioContext, row: dict, features: Path, new_raw: dict) -> str:
    import shutil

    from sparc.core import runio
    from sparc.studio.projects.inputs import FEATURES_FILE

    name = f"{row['name']}_open"
    slug, odir = service.claim_dir(sctx.workspace, name,
                                   {r["slug"] for r in sctx.db.fetchall("SELECT slug FROM projects")})
    try:
        (odir / FEATURES_FILE).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(features, odir / FEATURES_FILE)
        side = features.with_suffix(".json")
        if side.is_file():
            shutil.copyfile(side, (odir / FEATURES_FILE).with_suffix(".json"))
        for sub in ("data", "inputs/forcing", "inputs/climate", "inputs/layers"):
            (odir / sub).mkdir(parents=True, exist_ok=True)
        runio.write_text_atomic(odir / cs.CONFIG_NAME, cs.dump_config(new_raw))
        new = service.create_project(sctx.db, sctx.workspace, name=name, template="features_open",
                                     demo=bool(row.get("demo")), slug=slug, project_dir=odir,
                                     meta={"source": {"template": "features_open", "from_project": row["id"]}},
                                     note=f"open predictors of {row['name']}")
    except BaseException:
        shutil.rmtree(odir, ignore_errors=True)
        raise
    return new["id"]
