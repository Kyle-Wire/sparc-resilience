"""Input endpoints (api.md §5.4): launch the ``input.*`` network jobs, input state and views, station lookup.

Each launch checks its preconditions in the request (``422 needs_crs``,
``422 requirements``) and returns ``202 Job``; the work runs in a job on the
network lane.  The station lookup reads only the cached ISD index and never
downloads.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, Body, Depends, Query

from sparc.studio.app import StudioContext, get_ctx
from sparc.studio.errors import ApiError
from sparc.studio.projects import inputs as pin
from sparc.studio.projects import service
from sparc.studio.projects.config_service import get_dotted
from sparc.studio.projects.kinds import (
    Cmip6Params,
    FeaturesParams,
    ForcingParams,
    GhcnParams,
    LayersParams,
    StationsParams,
)
from sparc.studio.projects.schemas import InputsState, StationRow
from sparc.studio.schemas.common import Job

router = APIRouter(tags=["inputs"])


def _manager(sctx: StudioContext):
    if sctx.jobs is None:
        raise ApiError("not_ready", "the server is still starting", status=503)
    return sctx.jobs


def _project(sctx: StudioContext, pid: str) -> tuple[dict, dict]:
    row = service.get_row(sctx.db, pid)
    return row, service.project_raw(row)


def _crs_action(pid: str) -> dict:
    return {"kind": "open", "label": "Set the CRS", "method": "GET", "path": f"/p/{pid}/setup/data"}


def _require_geometry(pid: str, raw: dict, *, need_id: bool = False) -> None:
    """Data path and target set (x/y default to ``x``/``y``), and the id for joins: ``422 requirements`` with
    ``detail.missing``."""
    data = raw.get("data") if isinstance(raw.get("data"), dict) else {}
    missing = [f"data.{k}" for k in ("path", "target") if not data.get(k)]
    if need_id and not data.get("id"):
        missing.append("data.id")
    if missing:
        raise ApiError("requirements", f"set {', '.join(missing)} first", detail={"missing": missing},
                       action={"kind": "open", "label": "Open the data step", "method": "GET",
                               "path": f"/p/{pid}/setup/data"})


def _require_crs(pid: str, raw: dict, what: str) -> None:
    data = raw.get("data") if isinstance(raw.get("data"), dict) else {}
    if not data.get("crs"):
        raise ApiError("needs_crs", f"{what} needs data.crs (the EPSG code of x/y)", detail={"missing": ["data.crs"]},
                       action=_crs_action(pid))


def _site(pid: str, raw: dict, pdir: str, lat, lon) -> None:
    try:
        pin.site_of(raw, pdir, lat, lon)
    except ApiError as exc:
        if exc.code == "needs_crs":
            exc.action = _crs_action(pid)
            exc.detail = {"missing": ["data.crs"]}
        raise


async def _submit(sctx: StudioContext, pid: str, kind: str, params: dict, label: str | None = None) -> dict:
    return await _manager(sctx).submit(kind, params, project_id=pid, label=label)


@router.post("/projects/{pid}/inputs/forcing", response_model=Job, status_code=202)
async def fetch_forcing(pid: str, body: ForcingParams, sctx: StudioContext = Depends(get_ctx)):
    """Campaign forcing from ERA5 and the nearest station (links ``physics.forcing``)."""
    row, raw = await asyncio.to_thread(_project, sctx, pid)
    if body.lat is None:
        await asyncio.to_thread(_site, pid, raw, row["dir"], None, None)
    return await _submit(sctx, pid, "input.forcing", body.model_dump(mode="json"),
                         f"Campaign forcing {body.date} {body.hours[0]:02d}–{body.hours[1]:02d} h")


@router.post("/projects/{pid}/inputs/layers", response_model=Job, status_code=202)
async def fetch_layers(pid: str, body: LayersParams | None = Body(None), sctx: StudioContext = Depends(get_ctx)):
    """People (HRSL) and land cover (WorldCover) on the study grid (links ``planner.layers``)."""
    _row, raw = await asyncio.to_thread(_project, sctx, pid)
    _require_geometry(pid, raw)
    _require_crs(pid, raw, "People & land cover")
    return await _submit(sctx, pid, "input.layers", (body or LayersParams()).model_dump(mode="json"))


@router.post("/projects/{pid}/inputs/features", response_model=Job, status_code=202)
async def fetch_features(pid: str, body: FeaturesParams | None = Body(None), sctx: StudioContext = Depends(get_ctx)):
    """Open predictors (bootstrap mode for a project without predictors)."""
    _row, raw = await asyncio.to_thread(_project, sctx, pid)
    _require_geometry(pid, raw, need_id=True)
    _require_crs(pid, raw, "Open predictors")
    return await _submit(sctx, pid, "input.features", (body or FeaturesParams()).model_dump(mode="json"))


@router.post("/projects/{pid}/inputs/cmip6", response_model=Job, status_code=202)
async def fetch_cmip6(pid: str, body: Cmip6Params | None = Body(None), sctx: StudioContext = Depends(get_ctx)):
    """CMIP6 change factors at the site (links ``climate.table``)."""
    body = body or Cmip6Params()
    row, raw = await asyncio.to_thread(_project, sctx, pid)
    if body.lat is None:
        await asyncio.to_thread(_site, pid, raw, row["dir"], None, None)
    return await _submit(sctx, pid, "input.cmip6", body.model_dump(mode="json"))


@router.post("/projects/{pid}/inputs/ghcn", response_model=Job, status_code=202)
async def fetch_ghcn(pid: str, body: GhcnParams | None = Body(None), sctx: StudioContext = Depends(get_ctx)):
    """GHCN-Daily Tmax of a station, cached under the workspace cache."""
    body = body or GhcnParams()
    _row, raw = await asyncio.to_thread(_project, sctx, pid)
    station = body.station or get_dotted(raw, "planner.ghcn_station")
    if not station:
        raise ApiError("validation", "give a station or set planner.ghcn_station",
                       detail={"errors": [{"path": "station", "message": "required", "code": "missing"}]})
    return await _submit(sctx, pid, "input.ghcn", {"station": str(station)}, f"GHCN-Daily {station}")


@router.post("/projects/{pid}/inputs/stations", response_model=Job, status_code=202)
async def fetch_stations(pid: str, body: StationsParams | None = Body(None), sctx: StudioContext = Depends(get_ctx)):
    """Download and cache the NOAA ISD station index (≈3 MB)."""
    await asyncio.to_thread(_project, sctx, pid)
    return await _submit(sctx, pid, "input.stations", {})


@router.get("/projects/{pid}/inputs", response_model=InputsState)
async def inputs_state(pid: str, sctx: StudioContext = Depends(get_ctx)):
    """What has been fetched, and whether the config uses it."""
    def run():
        row, raw = _project(sctx, pid)
        return pin.inputs_state(raw, row["dir"], sctx.workspace)
    return await asyncio.to_thread(run)


@router.get("/projects/{pid}/inputs/{kind}/view")
async def input_view(pid: str, kind: str, sctx: StudioContext = Depends(get_ctx)):
    """Chart-ready model of one input (forcing, climate, layers, features)."""
    def run():
        row, raw = _project(sctx, pid)
        return pin.input_view(kind, raw, row["dir"])
    return await asyncio.to_thread(run)


@router.get("/projects/{pid}/forcing/stations", response_model=list[StationRow])
async def forcing_stations(pid: str, lat: float | None = Query(None, ge=-90, le=90),
                           lon: float | None = Query(None, ge=-180, le=180), limit: int = Query(10, ge=1, le=100),
                           sctx: StudioContext = Depends(get_ctx)):
    """Nearest ISD stations from the cached index (default site: ``climate.site`` or the data centroid)."""
    def run():
        row, raw = _project(sctx, pid)
        path = Path(sctx.workspace.cache_dir) / pin.ISD_FILE
        if not path.is_file():
            raise ApiError("not_found", "the ISD station list is not cached yet", detail={"missing": pin.ISD_FILE},
                           action={"kind": "fetch_input", "label": "Fetch station list", "method": "POST",
                                   "path": f"/api/projects/{pid}/inputs/stations", "body": {}})
        la, lo = (lat, lon) if lat is not None and lon is not None else pin.site_of(raw, row["dir"])
        return pin.nearest_stations(sctx.workspace.cache_dir, la, lo, limit)
    return await asyncio.to_thread(run)
