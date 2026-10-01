"""Input job kinds (SPEC §9.3 step 4, §10.3; api.md §5.4, §8): network fetchers that write project inputs.

=================  ==========================================  =============================================
kind               library function                            writes / links
=================  ==========================================  =============================================
input.forcing      ``forcing.campaign_forcing``                ``inputs/forcing/*.json`` → physics.forcing
input.layers       ``opendata.fetch_layers``                   ``inputs/layers/layers.parquet`` → planner.layers
input.features     ``features_open.build_open_features``       ``inputs/features/open_features.parquet`` → a join
                                                               here, or the new project ``<name>_open``
input.cmip6        ``climate.cmip6_change_factors``            ``inputs/climate/cmip6_*.csv`` → climate.table
input.ghcn         ``planner.ghcn_tmax``                       ``<ws>/cache/ghcn_<station>.csv``
input.stations     (NOAA ISD index)                            ``<ws>/cache/isd-history.csv``
=================  ==========================================  =============================================

Every fetcher gets the workspace cache.  Workers never write SQLite: a link
edits ``config.yml`` under the config lock and leaves a note, and a new
``_open`` project is written as a folder with ``project.json``; the
server-side hook on the ``job.result`` event (before the job is reported
final) records the config version and registers the new project.

Library modules are imported inside the job functions (the server imports
this module only to list kinds).
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sparc.studio.jobs.kinds import job_kind

log = logging.getLogger("sparc.studio.projects")

__all__ = ["ForcingParams", "LayersParams", "FeaturesParams", "Cmip6Params", "GhcnParams", "StationsParams",
           "INPUT_KINDS", "STATION_ID"]

#: station ids (ISD USAF+WBAN, GHCN-Daily) - they name cache files, so nothing but letters and digits
STATION_ID = r"^[A-Za-z0-9]{1,16}$"
INPUT_KINDS = ("input.forcing", "input.layers", "input.features", "input.cmip6", "input.ghcn", "input.stations")
HOSTS = {
    "input.forcing": ("nsf-ncar-era5.s3.amazonaws.com", "noaa-global-hourly-pds.s3.amazonaws.com"),
    "input.layers": ("dataforgood-fb-data.s3.amazonaws.com", "esa-worldcover.s3.amazonaws.com"),
    "input.features": ("esa-worldcover.s3.amazonaws.com", "copernicus-dem-30m.s3.amazonaws.com",
                       "sentinel-cogs.s3.us-west-2.amazonaws.com"),
    "input.cmip6": ("cmip6-pds.s3.amazonaws.com",),
    "input.ghcn": ("noaa-ghcn-pds.s3.amazonaws.com",),
    "input.stations": ("noaa-isd-pds.s3.amazonaws.com",),
}


# ---------------------------------------------------------------------------
# params (api.md §5.4; additionalProperties: false)
# ---------------------------------------------------------------------------

class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ForcingParams(_Params):
    date: str = Field(..., description="campaign day, YYYY-MM-DD")
    hours: tuple[int, int] = Field(..., description="local hours [first, last], e.g. [15, 16]")
    tz: str = Field(..., description="IANA time zone, e.g. America/New_York")
    lat: float | None = Field(None, ge=-90, le=90)
    lon: float | None = Field(None, ge=-180, le=180)
    station: str | None = Field(None, pattern=STATION_ID, description="ISD station USAF+WBAN, e.g. 72507014765")
    wind_source: Literal["auto", "station", "era5"] = "auto"
    link: bool = True

    @field_validator("date")
    @classmethod
    def _date(cls, v: str) -> str:
        dt.date.fromisoformat(v)
        return v

    @field_validator("hours")
    @classmethod
    def _hours(cls, v: tuple[int, int]) -> tuple[int, int]:
        if not (0 <= v[0] < v[1] <= 23):
            raise ValueError("hours must be an increasing local range within 0–23, e.g. [15, 16]")
        return v

    @field_validator("tz")
    @classmethod
    def _tz(cls, v: str) -> str:
        from zoneinfo import ZoneInfo

        try:
            ZoneInfo(v)
        except Exception:                        # noqa: BLE001 - ZoneInfoNotFoundError is a KeyError
            raise ValueError(f"unknown time zone {v!r}") from None
        return v

    @model_validator(mode="after")
    def _site(self) -> "ForcingParams":
        if (self.lat is None) != (self.lon is None):
            raise ValueError("give both lat and lon, or neither")
        return self


class LayersParams(_Params):
    link: bool = True


class FeaturesParams(_Params):
    months: list[str] | None = Field(None, description="months of the Sentinel-2 composite, YYYY-MM")
    max_cloud: float = Field(20.0, ge=0, le=100)
    s2_tiles: list[Annotated[str, Field(pattern=r"^\d{1,2}[A-Za-z]{3}$")]] | None = Field(
        None, description="Sentinel-2 MGRS tiles, e.g. 19TCG")
    target: Literal["new_project", "this_project"] = "new_project"

    @field_validator("months")
    @classmethod
    def _months(cls, v: list[str] | None) -> list[str] | None:
        for m in v or []:
            y, mo = m.split("-")
            if not (len(y) == 4 and 1 <= int(mo) <= 12):
                raise ValueError(f"month {m!r} is not YYYY-MM")
        return v


class Cmip6Params(_Params):
    lat: float | None = Field(None, ge=-90, le=90)
    lon: float | None = Field(None, ge=-180, le=180)
    experiments: list[str] | None = None
    periods: dict[str, tuple[int, int]] | None = None
    baseline: tuple[int, int] = (1995, 2014)
    months: list[int] | None = None
    variable: Literal["tasmax", "tas"] | None = None
    models: list[str] | None = None
    workers: int = Field(4, ge=1, le=16)
    link: bool = True

    @model_validator(mode="after")
    def _check(self) -> "Cmip6Params":
        if (self.lat is None) != (self.lon is None):
            raise ValueError("give both lat and lon, or neither")
        if self.baseline[0] > self.baseline[1]:
            raise ValueError("baseline must be [first, last]")
        for name, (a, b) in (self.periods or {}).items():
            if a > b:
                raise ValueError(f"period {name} must be [first, last]")
        for m in self.months or []:
            if not 1 <= int(m) <= 12:
                raise ValueError("months are 1–12")
        return self


class GhcnParams(_Params):
    station: str | None = Field(None, pattern=STATION_ID, description="GHCN-Daily station id, e.g. USW00014765")


class StationsParams(_Params):
    pass


# ---------------------------------------------------------------------------
# worker helpers
# ---------------------------------------------------------------------------

def _project(ctx) -> tuple[Path, Path, dict]:
    from sparc.studio.projects.config_service import CONFIG_NAME, core_block, parse_yaml

    if not ctx.project_dir:
        raise RuntimeError("this job needs a project")
    pdir = Path(ctx.project_dir)
    cpath = Path(ctx.config_path) if ctx.config_path else pdir / CONFIG_NAME
    text = cpath.read_text(encoding="utf-8") if cpath.is_file() else ""
    return pdir, cpath, core_block(parse_yaml(text)) if text.strip() else {}


def _s0(raw: dict, pdir: Path, *, geometry_only: bool):
    """S0 at full resolution (no coarse cells, no window).  ``geometry_only`` (or a config without predictors,
    the features bootstrap mode) loads an in-memory copy whose predictors are just the target and whose roles
    are empty: enough for the grid and ids, never saved."""
    from sparc.core import progress
    from sparc.core.data import load_core_data
    from sparc.studio.projects.config_service import build_core_config

    cfg = build_core_config(raw, pdir)
    cfg.raw["data"]["coarse_m"] = None
    cfg.raw["data"]["subsample"] = None
    bootstrap = not cfg.raw.get("predictors")
    if geometry_only or bootstrap:
        cfg.raw["predictors"] = [cfg.raw["data"]["target"]]
        cfg.raw["physics"]["roles"] = {}
        cfg.raw["actionable"] = {}
        cfg.raw["mediators"] = {}
        cfg.raw["encodings"] = {"categorical": [], "circular_degrees": []}
    with progress.task("load_data"):
        data = load_core_data(cfg)
    return cfg, data, bootstrap


def _link(ctx, pdir: Path, cpath: Path, edit, what: str) -> bool:
    """Apply a link edit to ``config.yml`` (the server records the version); True once the config points at it."""
    from sparc.studio.projects.config_service import edit_config_file

    edit_config_file(pdir, edit, f"linked {what} ({ctx.kind} {ctx.job_id})", config_path=cpath)
    return True


def _artifact(path: Path, role: str) -> None:
    from sparc.core import progress

    progress.artifact(str(path), role=role)


# ---------------------------------------------------------------------------
# server-side hooks
# ---------------------------------------------------------------------------

def _record(sctx, job: dict, result: dict | None) -> None:
    """Record a linked config as a version and register a new ``_open`` project (idempotent, fast)."""
    from sparc.studio.projects import service
    from sparc.studio.projects.config_service import sync_versions

    pid = job.get("project_id")
    if pid:
        row = sctx.db.fetchone("SELECT * FROM projects WHERE id = ?", (pid,))
        if row is not None:
            sync_versions(sctx.db, row)
    open_dir = (result or {}).get("open_project_dir")
    if open_dir and Path(open_dir).is_dir():
        service.register_dir(sctx.db, open_dir)


def _on_event(sctx, job: dict, event: dict) -> None:
    if event.get("type") == "job.result":
        _record(sctx, job, event.get("result") or {})


def _on_finish(sctx, job: dict, result: dict | None) -> None:
    if job.get("status") == "succeeded":
        _record(sctx, job, result)


def _kind(kind: str, label: str, params, **kw):
    return job_kind(kind, lane="network", executor="process", label=label, params=params,
                    network_hosts=HOSTS[kind], on_event=_on_event, on_event_types=("job.result",),
                    on_finish=_on_finish, **kw)


# ---------------------------------------------------------------------------
# kinds
# ---------------------------------------------------------------------------

@_kind("input.forcing", "Campaign forcing (ERA5 + station)", ForcingParams)
def run_forcing(ctx, p: ForcingParams) -> dict:
    from sparc.core import runio
    from sparc.core.forcing import campaign_forcing
    from sparc.studio.projects.inputs import forcing_name, link_forcing, site_of

    pdir, cpath, raw = _project(ctx)
    lat, lon = site_of(raw, pdir, p.lat, p.lon)
    res = campaign_forcing(lat, lon, p.date, (int(p.hours[0]), int(p.hours[1])), p.tz, station=p.station,
                           wind_source=p.wind_source, cache_dir=ctx.cache_dir)
    rel = forcing_name(p.date, p.hours)
    out = pdir / rel
    runio.write_json_atomic(out, res, indent=1)
    _artifact(out, "input.forcing")
    linked = _link(ctx, pdir, cpath, link_forcing(rel), "physics.forcing") if p.link else False
    ph = res.get("physics") or {}
    return {"path": rel, "sw_down": ph.get("sw_down"), "lw_net": ph.get("lw_net"), "wind": list(ph.get("wind") or []),
            "checks": list(res.get("checks") or []), "linked": linked}


@_kind("input.layers", "People & land cover (HRSL + WorldCover)", LayersParams)
def run_layers(ctx, p: LayersParams) -> dict:
    from sparc.core import runio
    from sparc.core.opendata import fetch_layers
    from sparc.studio.projects.inputs import LAYERS_FILE, link_layers

    pdir, cpath, raw = _project(ctx)
    cfg, data, _ = _s0(raw, pdir, geometry_only=True)
    lay = fetch_layers(data, cfg)
    out = pdir / LAYERS_FILE
    runio.write_parquet_atomic(lay, out)
    _artifact(out, "input.layers")
    linked = _link(ctx, pdir, cpath, link_layers(LAYERS_FILE), "planner.layers") if p.link else False
    people = float(lay["people"].sum()) if "people" in lay else 0.0
    return {"path": LAYERS_FILE, "n_cells": int(len(lay)), "people_total": people, "linked": linked}


def _months(p: FeaturesParams, cfg) -> list[tuple[int, int]]:
    if p.months:
        return [tuple(int(x) for x in m.split("-")) for m in p.months]
    date = ((cfg.raw.get("physics") or {}).get("forcing_info") or {}).get("date")
    return [(int(date[:4]), int(date[5:7]))] if date else [(2020, 6), (2020, 7), (2020, 8)]


@_kind("input.features", "Open predictors (WorldCover + Sentinel-2 + DEM)", FeaturesParams)
def run_features(ctx, p: FeaturesParams) -> dict:
    import shutil

    from sparc.core import progress, runio
    from sparc.core.features_open import build_open_features, compare_features
    from sparc.studio.projects.config_service import dump_config
    from sparc.studio.projects.inputs import FEATURES_FILE, link_features, open_project_raw
    from sparc.studio.projects.service import claim_dir, read_record, write_record
    from sparc.studio.workspace import new_id, utc_now

    pdir, cpath, raw = _project(ctx)
    preds = list(raw.get("predictors") or [])
    try:
        cfg, data, bootstrap = _s0(raw, pdir, geometry_only=False)
    except (KeyError, ValueError):               # predictors absent or unusable: geometry only, no agreement
        cfg, data, bootstrap = _s0(raw, pdir, geometry_only=True)
        bootstrap = True
    df, prov = build_open_features(data, cfg, months=_months(p, cfg), max_cloud=p.max_cloud, s2_tiles=p.s2_tiles)
    roles = (cfg.raw.get("physics") or {}).get("roles") or {}
    agreement = [] if bootstrap or not preds else compare_features(df, data, roles)
    df = df.rename(columns={c: f"open_{c}" for c in df.columns if c != "id"})
    out = pdir / FEATURES_FILE
    runio.write_parquet_atomic(df, out)
    runio.write_json_atomic(out.with_suffix(".json"), {"provenance": prov, "agreement": agreement,
                                                       "bootstrap": bool(bootstrap), "target": p.target,
                                                       "months": [list(m) for m in _months(p, cfg)]}, indent=1)
    _artifact(out, "input.features")
    if p.target == "this_project":
        linked = _link(ctx, pdir, cpath, link_features(FEATURES_FILE), "the open features")
        return {"path": FEATURES_FILE, "agreement": agreement, "open_project_id": None, "linked": linked}
    # a new project <name>_open next to this one
    src = read_record(pdir) or {}
    name = f"{src.get('name') or pdir.name}_open"
    ws = ctx.workspace
    slug, odir = claim_dir(ws, name, {r["slug"] for r in ctx.db.fetchall("SELECT slug FROM projects")})
    try:
        (odir / "inputs" / "features").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out, odir / FEATURES_FILE)
        shutil.copyfile(out.with_suffix(".json"), (odir / FEATURES_FILE).with_suffix(".json"))
        for sub in ("data", "inputs/forcing", "inputs/climate", "inputs/layers"):
            (odir / sub).mkdir(parents=True, exist_ok=True)
        new_raw = open_project_raw(raw, pdir, FEATURES_FILE)
        runio.write_text_atomic(odir / "config.yml", dump_config(new_raw))
        now = utc_now()
        pid = new_id("project")
        write_record(odir, {"id": pid, "slug": slug, "name": name, "template": "features_open",
                            "demo": bool(src.get("demo")), "created_utc": now, "updated_utc": now, "archived": False,
                            "active_run_id": None, "report": src.get("report") or {}, "headline_scenario": None,
                            "cost_model": {}, "source": {"template": "features_open", "from_project": ctx.project_id,
                                                         "job_id": ctx.job_id}})
    except BaseException:
        shutil.rmtree(odir, ignore_errors=True)
        raise
    progress.emit("log", logger="sparc.studio.projects", level="INFO", msg=f"created project {name} ({pid})")
    return {"path": FEATURES_FILE, "agreement": agreement, "open_project_id": pid, "linked": True,
            "open_project_dir": str(odir), "bootstrap": bool(bootstrap)}


@_kind("input.cmip6", "CMIP6 change factors", Cmip6Params)
def run_cmip6(ctx, p: Cmip6Params) -> dict:
    from sparc.core import runio
    from sparc.core.climate import BASELINE, EXPERIMENTS, PERIODS, cmip6_change_factors
    from sparc.studio.projects.config_service import get_dotted
    from sparc.studio.projects.inputs import cmip6_name, link_climate, site_of

    pdir, cpath, raw = _project(ctx)
    lat, lon = site_of(raw, pdir, p.lat, p.lon)
    exps = tuple(p.experiments or get_dotted(raw, "climate.experiments") or EXPERIMENTS)
    periods = {str(k): (int(v[0]), int(v[1])) for k, v in
               (p.periods or get_dotted(raw, "climate.periods") or PERIODS).items()}
    months = tuple(int(m) for m in (p.months or get_dotted(raw, "climate.months") or (6, 7, 8)))
    variable = p.variable or get_dotted(raw, "climate.variable") or "tasmax"
    df = cmip6_change_factors(lat, lon, ctx.cache_dir, experiments=exps, periods=periods,
                              baseline=tuple(p.baseline or BASELINE), months=months, variable=variable,
                              models=p.models, max_workers=p.workers)
    if not len(df):
        raise RuntimeError("no CMIP6 model could be read for this site (see the warnings)")
    rel = cmip6_name(variable, months)
    out = pdir / rel
    runio.write_text_atomic(out, df.to_csv(index=False, lineterminator="\n"))
    _artifact(out, "input.cmip6")
    present = sorted(map(str, df["model"].unique()))
    skipped: list[str] = []
    cat = Path(ctx.cache_dir) / "pangeo-cmip6-tasmax-sftlf.csv"
    if cat.is_file():                            # the catalogue the fetch used: which models were dropped
        try:
            from sparc.core.climate import load_catalog, select_runs

            names = sorted(select_runs(load_catalog(Path(ctx.cache_dir)), exps, variable, p.models).source_id.unique())
            skipped = [m for m in names if m not in present]
        except Exception:                        # noqa: BLE001 - informative only
            skipped = []
    linked = False
    if p.link:
        linked = _link(ctx, pdir, cpath, link_climate(rel, periods={k: list(v) for k, v in periods.items()},
                                                      experiments=list(exps)), "climate.table")
    return {"path": rel, "n_models": len(present), "experiments": sorted(map(str, df["experiment"].unique())),
            "periods": sorted(map(str, df["period"].unique())), "skipped_models": skipped, "linked": linked,
            "site": [lat, lon]}


@_kind("input.ghcn", "GHCN-Daily station series", GhcnParams)
def run_ghcn(ctx, p: GhcnParams) -> dict:
    from sparc.core import progress, runio
    from sparc.core.planner import ghcn_tmax
    from sparc.studio.projects.config_service import get_dotted, set_dotted
    from sparc.studio.projects.inputs import GHCN_RECORD

    pdir, cpath, raw = _project(ctx)
    station = p.station or get_dotted(raw, "planner.ghcn_station")
    if not station:
        raise ValueError("no station: pass one or set planner.ghcn_station")
    if not re.fullmatch(STATION_ID, str(station)):
        raise ValueError(f"{station!r} is not a GHCN-Daily station id (letters and digits, e.g. USW00014765)")
    with progress.task("remote_object", key=f"ghcn:{station}", unit="remote_object"):
        s = ghcn_tmax(str(station), cache_dir=ctx.cache_dir)
    if not len(s):
        raise ValueError(f"GHCN station {station} has no TMAX records")
    years = [int(s.index.min().year), int(s.index.max().year)]
    path = Path(ctx.cache_dir) / f"ghcn_{station}.csv"
    runio.write_json_atomic(pdir / GHCN_RECORD, {"station": str(station), "years": years, "path": str(path),
                                                 "n_days": int(len(s))})
    if not get_dotted(raw, "planner.ghcn_station"):
        _link(ctx, pdir, cpath, lambda r: set_dotted(r, "planner.ghcn_station", str(station)), "planner.ghcn_station")
    return {"path": str(path), "years": years, "station": str(station)}


@_kind("input.stations", "NOAA ISD station index", StationsParams)
def run_stations(ctx, p: StationsParams) -> dict:
    import io

    import pandas as pd

    from sparc.core import progress, runio
    from sparc.core.climate import http_fetch
    from sparc.studio.projects.inputs import ISD_FILE, ISD_URLS

    raw = None
    for url in ISD_URLS:
        progress.check_cancel()
        with progress.task("remote_object", key=url.rsplit("/", 1)[-1], unit="remote_object"):
            try:
                raw = http_fetch(url)
            except Exception as exc:              # noqa: BLE001 - try the next mirror
                progress.warn("stations.mirror_failed", f"{url}: {exc}", url=url)
                raw = None
        if raw:
            break
    if not raw:
        raise FileNotFoundError("the NOAA ISD station index could not be downloaded")
    df = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
    cols = {c.strip().upper() for c in df.columns}
    if not {"USAF", "WBAN", "STATION NAME", "LAT", "LON"} <= cols:
        raise ValueError("the downloaded station index has an unexpected format")
    path = Path(ctx.cache_dir) / ISD_FILE
    runio.write_bytes_atomic(path, raw)
    return {"path": str(path), "n_stations": int(len(df))}
