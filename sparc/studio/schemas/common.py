"""Shared wire types (api.md §0–4): errors, pages, jobs, plans, selections, layers, grids and the
foundation endpoints' request/response bodies.

Field names follow api.md exactly; FastAPI turns these models into
``/openapi.json``, from which the web client's types are generated and
checked.  Feature items define their own models in their packages and reuse
these.
"""

from __future__ import annotations

from typing import Annotated, Any, Generic, Literal, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag

T = TypeVar("T")

# ---------------------------------------------------------------------------
# errors (§0.3, §16)
# ---------------------------------------------------------------------------

ActionKind = Literal["run_job", "open", "resume", "link_config", "build_emulator", "open_engine", "rerun_exact",
                     "fetch_input"]


class Action(BaseModel):
    """A one-click remedy the UI renders as a button."""
    kind: ActionKind
    label: str
    method: Literal["POST", "GET"] | None = None
    path: str | None = None
    body: Any = None


class ValidationErrorItem(BaseModel):
    path: str
    message: str
    code: str


class ErrorBody(BaseModel):
    code: str
    message: str
    detail: dict[str, Any] | None = None
    action: Action | None = None


class ErrorEnvelope(BaseModel):
    error: ErrorBody


class Page(BaseModel, Generic[T]):
    """``Page<T> = {items, next_cursor}`` (§0.1)."""
    items: list[T]
    next_cursor: str | None = None


class Ok(BaseModel):
    ok: bool = True


# ---------------------------------------------------------------------------
# jobs (§0.6)
# ---------------------------------------------------------------------------

JobStatus = Literal["queued", "blocked", "starting", "running", "cancelling", "succeeded", "failed", "cancelled",
                    "interrupted"]
JobLane = Literal["heavy", "medium", "network", "engine", "none"]
JobExecutor = Literal["process", "engine", "external"]

ACTIVE_STATUSES: tuple[str, ...] = ("queued", "blocked", "starting", "running", "cancelling")
LIVE_STATUSES: tuple[str, ...] = ("starting", "running", "cancelling")
FINAL_STATUSES: tuple[str, ...] = ("succeeded", "failed", "cancelled", "interrupted")


class JobError(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str
    message: str
    traceback_tail: str | None = None


class JobBlocked(BaseModel):
    reason: str
    actions: list[Action] = Field(default_factory=list)


class Job(BaseModel):
    id: str
    kind: str
    lane: JobLane
    executor: JobExecutor
    label: str
    status: JobStatus
    project_id: str | None = None
    run_id: str | None = None
    study_id: str | None = None
    scenario_id: str | None = None
    parent_job_id: str | None = None
    after_job_id: str | None = None
    priority: int = 0
    params: dict[str, Any] = Field(default_factory=dict)
    created_utc: str
    started_utc: str | None = None
    finished_utc: str | None = None
    progress: float | None = None
    eta_s: float | None = None
    eta_lo: float | None = None
    eta_hi: float | None = None
    stage: str | None = None
    current_path: list[str] | None = None
    exit_code: int | None = None
    error: JobError | None = None
    blocked: JobBlocked | None = None
    result: dict[str, Any] | None = None
    peak_rss_mb: float | None = None
    threads: int | None = None


class JobCreate(BaseModel):
    """``POST /api/jobs`` body."""
    model_config = ConfigDict(extra="forbid")
    kind: str
    params: dict[str, Any] = Field(default_factory=dict)
    project_id: str | None = None
    run_id: str | None = None
    study_id: str | None = None
    priority: int = 0
    after_job_id: str | None = None


class JobPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    priority: int


class KillRequest(BaseModel):
    force_now: bool = False


# ---------------------------------------------------------------------------
# shared types (§1)
# ---------------------------------------------------------------------------

StageId = Literal["S0", "S1", "S2_S3", "baselines", "cv_curve", "S4", "S5", "climate", "S6", "S7", "finish"]
STAGE_IDS: tuple[str, ...] = ("S0", "S1", "S2_S3", "baselines", "cv_curve", "S4", "S5", "climate", "S6", "S7",
                              "finish")


class PlanNode(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: StageId
    label: str
    state: Literal["will_run", "skipped", "cached"]
    reason: str | None = None
    units: dict[str, float] = Field(default_factory=dict)
    checkpoint_key: str | None = None
    est_s: float | None = None
    est_lo: float | None = None
    est_hi: float | None = None


class IssueFix(BaseModel):
    path: str
    value: Any = None


class Issue(BaseModel):
    level: Literal["error", "warn", "info"]
    path: str
    code: str
    message: str
    fix: IssueFix | None = None


SelectionCrs = Literal["EPSG:4326", "run_xy_m"]


class SelAll(BaseModel):
    kind: Literal["all"]


class SelZones(BaseModel):
    kind: Literal["zones"]
    values: list[float | int | str]


class SelPolygon(BaseModel):
    kind: Literal["polygon"]
    crs: SelectionCrs
    rings: list[list[tuple[float, float]]]


class SelCircle(BaseModel):
    kind: Literal["circle"]
    crs: SelectionCrs
    center: tuple[float, float]
    radius_m: float


class SelRect(BaseModel):
    kind: Literal["rect"]
    crs: SelectionCrs
    min: tuple[float, float]
    max: tuple[float, float]


class SelCells(BaseModel):
    kind: Literal["cells"]
    ids: list[int | str]


class SelBlob(BaseModel):
    kind: Literal["blob"]
    blob_id: str


class SelHex(BaseModel):
    kind: Literal["hex"]
    size_m: Literal[250, 500]
    keys: list[int]


class SelFilter(BaseModel):
    kind: Literal["filter"]
    column: str
    op: Literal["<", "<=", ">", ">=", "==", "between", "in"]
    value: float | int | str | tuple[float, float] | list[float | int | str]


class SelTop(BaseModel):
    kind: Literal["top"]
    column: str
    frac: float | None = None
    k: int | None = None
    direction: Literal["highest", "lowest"]
    within: SelectionSpec | None = None


class SelBuffer(BaseModel):
    kind: Literal["buffer"]
    of: SelectionSpec
    radius_m: float | None = None
    lever_range: str | None = None


class SelRegion(BaseModel):
    kind: Literal["region"]
    id: str


class SelOp(BaseModel):
    op: Literal["and", "or", "minus"]
    args: list[SelectionSpec]


class SelNot(BaseModel):
    op: Literal["not"]
    arg: SelectionSpec


def _selection_tag(v: Any) -> str | None:
    get = v.get if isinstance(v, dict) else (lambda k, d=None: getattr(v, k, d))
    kind = get("kind")
    if kind:
        return str(kind)
    op = get("op")
    if op == "not":
        return "not"
    return "op" if op else None


SelectionSpec = Annotated[
    Union[
        Annotated[SelAll, Tag("all")], Annotated[SelZones, Tag("zones")], Annotated[SelPolygon, Tag("polygon")],
        Annotated[SelCircle, Tag("circle")], Annotated[SelRect, Tag("rect")], Annotated[SelCells, Tag("cells")],
        Annotated[SelBlob, Tag("blob")], Annotated[SelHex, Tag("hex")], Annotated[SelFilter, Tag("filter")],
        Annotated[SelTop, Tag("top")], Annotated[SelBuffer, Tag("buffer")], Annotated[SelRegion, Tag("region")],
        Annotated[SelOp, Tag("op")], Annotated[SelNot, Tag("not")],
    ],
    Discriminator(_selection_tag),
]
"""A portable selection (api.md §1); column namespaces are documented there."""

for _m in (SelTop, SelBuffer, SelOp, SelNot):
    _m.model_rebuild()

SELECTION_KINDS: tuple[str, ...] = ("all", "zones", "polygon", "circle", "rect", "cells", "blob", "hex", "filter",
                                    "top", "buffer", "region", "and", "or", "minus", "not")


class LayerStats(BaseModel):
    n: int
    lo: float | None = None
    hi: float | None = None
    mean: float | None = None
    p1: float | None = None
    p2: float | None = None
    p50: float | None = None
    p98: float | None = None
    p99: float | None = None


class LayerSource(BaseModel):
    file: str
    column: str | None = None


class LayerMeta(BaseModel):
    key: str
    group: str
    label: str
    unit: str
    scale: Literal["seq", "div", "cat"]
    center: float | None = None
    decimals: int = 2
    mult: float = 1.0
    zero_blank: bool = False
    labels: list[str] | None = None
    desc: str = ""
    sign_note: str | None = None
    source: LayerSource | None = None
    dtype: Literal["float32", "uint8"] = "float32"
    stats: LayerStats


class GridCorners(BaseModel):
    sw: tuple[float, float]
    se: tuple[float, float]
    nw: tuple[float, float]
    ne: tuple[float, float]


class GridMeta(BaseModel):
    n: int
    nx: int
    ny: int
    dx_m: float
    x0_m: float
    y0_m: float
    crs: str | None = None
    coord_scale: float = 1.0
    has_lonlat: bool = False
    bounds_lonlat: tuple[float, float, float, float] | None = None
    corners: GridCorners | None = None
    ids_kind: Literal["int", "str"] = "int"
    zones: list[int | float | str] = Field(default_factory=list)
    n_folds: int | None = None
    units: dict[str, str] = Field(default_factory=dict)
    background: float | None = None
    etag: str


Availability = Literal["ready", "partial", "running", "missing", "stale"]


class OutputFile(BaseModel):
    relpath: str
    bytes: int
    mtime: str


class OutputEntry(BaseModel):
    id: str
    label: str
    group: str
    state: Literal["present", "stale", "missing", "writing", "partial"]
    produced_by: str
    view: str
    formats: list[str] = Field(default_factory=list)
    files: list[OutputFile] = Field(default_factory=list)
    action: Action | None = None


class Likely(BaseModel):
    """A plain-language estimate: ``lo``/``hi`` = estimate ± 1.96·se."""
    estimate: float
    se: float | None = None
    lo: float | None = None
    hi: float | None = None
    confidence: Literal["confident_cools", "confident_warms", "could_be_zero", "unknown"]
    phrase: str


# ---------------------------------------------------------------------------
# system (§2)
# ---------------------------------------------------------------------------

EngineHostState = Literal["absent", "starting", "ready", "busy", "recycling", "error"]


class EngineInfo(BaseModel):
    state: EngineHostState = "absent"


class Health(BaseModel):
    ok: bool
    version: str
    workspace: str
    pid: int
    started_utc: str
    active_jobs: int
    engine: EngineInfo


class MetaStage(BaseModel):
    id: StageId
    label: str
    desc: str
    checkpoint_key: str | None = None
    manifest_timing_key: str | None = None


class MetaJobKind(BaseModel):
    kind: str
    label: str
    lane: str
    executor: str
    needs_run: bool
    needs_checkpoint: bool
    locks_run: bool
    network_hosts: list[str]
    long: bool
    params_schema: dict[str, Any]


class MetaOutput(BaseModel):
    id: str
    label: str
    group: str
    files: list[str]
    produced_by: str
    view: str
    formats: list[str]
    manifest_key: str | None = None


class Palettes(BaseModel):
    seqLight: list[str]
    seqDark: list[str]
    divLight: list[str]
    divDark: list[str]
    cat: list[str]


class MetaMode(BaseModel):
    id: Literal["fast", "coarse", "full"]
    label: str
    desc: str
    default_coarse_m: float | None = None


class WarningCode(BaseModel):
    code: str
    label: str
    view: str | None = None


class Meta(BaseModel):
    version: str
    schema_version: Literal[1] = 1
    event_schema_version: Literal[1] = 1
    stages: list[MetaStage]
    job_kinds: list[MetaJobKind]
    output_catalog: list[MetaOutput]
    palettes: Palettes
    unit_costs: dict[str, float]
    modes: list[MetaMode]
    run_tab_outputs: dict[str, list[str]]
    edit_modes: list[str]
    selection_kinds: list[str]
    warning_codes: list[WarningCode]


class Versions(BaseModel):
    python: str
    sparc: str
    numpy: str | None = None
    pandas: str | None = None
    torch: str | None = None
    fastapi: str | None = None


class WebBuild(BaseModel):
    src_sha256: str
    vite: str | None = None
    react: str | None = None


class SystemInfo(BaseModel):
    cpu_count: int
    cpu_model: str
    mem_total_gb: float
    mem_available_gb: float
    disk_free_gb: float
    workspace: str
    workspace_bytes: int
    host_id: str
    versions: Versions
    web_build: WebBuild | None = None


class NetcheckRequest(BaseModel):
    hosts: list[str] | None = None


class NetcheckRow(BaseModel):
    host: str
    ok: bool
    ms: float | None = None
    error: str | None = None


class NetcheckResult(BaseModel):
    results: list[NetcheckRow]


class StorageCacheEntry(BaseModel):
    name: str
    bytes: int
    mtime: str | None = None


class StorageRun(BaseModel):
    run_id: str
    label: str | None = None
    project_id: str | None = None
    outputs_bytes: int
    checkpoint_bytes: int


class StorageStudy(BaseModel):
    study_id: str
    bytes: int


class StorageInfo(BaseModel):
    workspace_bytes: int
    free_bytes: int
    cache: list[StorageCacheEntry]
    runs: list[StorageRun]
    studies: list[StorageStudy]
    jobs_bytes: int


class FreedBytes(BaseModel):
    freed_bytes: int


class ShutdownRequest(BaseModel):
    stop_jobs: bool = False


# ---------------------------------------------------------------------------
# tracking (§3)
# ---------------------------------------------------------------------------

StageState = Literal["planned", "running", "done", "failed", "cancelled", "not_reached", "skipped", "cached",
                     "disabled", "not_requested"]


class StageStatus(BaseModel):
    state: StageState
    reason: str | None = None
    started_ts: float | None = None
    ended_ts: float | None = None
    elapsed_s: float | None = None
    est_s: float | None = None
    progress: float | None = None


class Span(BaseModel):
    span_id: str
    parent_id: str | None = None
    kind: Literal["run", "stage", "task"]
    name: str
    key: str | None = None
    k: int | None = None
    n: int | None = None
    unit: str | None = None
    status: Literal["running", "ok", "error", "cancelled"]
    started_ts: float
    ended_ts: float | None = None
    elapsed_s: float | None = None
    ctx: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)


class MetricLatest(BaseModel):
    value: float | int | str | bool | None = None
    unit: str | None = None
    tags: dict[str, Any] = Field(default_factory=dict)
    ts: float


class SeriesPoint(BaseModel):
    ts: float
    value: float


class WarningRow(BaseModel):
    code: str
    lvl: str
    message: str
    count: int
    stage: str | None = None
    first_cursor: int
    data: dict[str, Any] = Field(default_factory=dict)


class ArtifactRow(BaseModel):
    relpath: str
    role: str
    bytes: int
    stage: str | None = None
    ts: float


class CheckpointRow(BaseModel):
    action: str
    done: list[str]
    bytes: int | None = None
    ts: float


class ResourceRow(BaseModel):
    ts: float
    rss_mb: float
    cpu_pct: float
    n_procs: int
    threads: int | None = None


class HeartbeatGap(BaseModel):
    from_ts: float
    to_ts: float


class ChildRow(BaseModel):
    job_id: str | None = None
    run_id: str | None = None
    key: str
    label: str
    status: str
    progress: float | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


class TrackerSnapshot(BaseModel):
    job: Job
    cursor: int
    plan: list[PlanNode] | None = None
    stages: dict[str, StageStatus] | None = None
    spans: list[Span]
    metrics_latest: dict[str, MetricLatest]
    metric_series: dict[str, list[SeriesPoint]]
    warnings: list[WarningRow]
    artifacts: list[ArtifactRow]
    checkpoints: list[CheckpointRow]
    resources: list[ResourceRow]
    heartbeat_gaps: list[HeartbeatGap]
    children: list[ChildRow]


class EventsPage(BaseModel):
    """``GET /api/jobs/{jid}/events``: raw event lines (envelope + fields) each with its byte ``cursor``."""
    events: list[dict[str, Any]]
    next_cursor: int
    eof: bool


class LogLine(BaseModel):
    cursor: int
    ts: float
    level: str
    logger: str
    msg: str
    path: list[str]


class LogsPage(BaseModel):
    lines: list[LogLine]
    next_cursor: int


class MetricPoint(BaseModel):
    cursor: int
    ts: float
    value: float | None = None
    value_text: str | None = None
    tags: dict[str, Any] = Field(default_factory=dict)


class LaneState(BaseModel):
    lane: str
    slots: int
    running: list[str]
    queued: list[str]


class QueueState(BaseModel):
    paused: bool
    lanes: list[LaneState]


class UnitRate(BaseModel):
    unit: str
    median_s: float
    p25_s: float
    p75_s: float
    n: int


class StageHistoryRow(BaseModel):
    run_id: str
    label: str | None = None
    git_commit: str | None = None
    stage: str
    seconds: float
    n_points: int | None = None
    mode: str | None = None
    threads: int | None = None


class Timings(BaseModel):
    unit_rates: list[UnitRate]
    stage_history: list[StageHistoryRow]
