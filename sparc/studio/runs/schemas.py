"""Request and response models of the runs endpoints (api.md §6).  Field names follow api.md exactly."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from sparc.studio.schemas.common import Action, Issue, Job, Likely, OutputEntry, PlanNode, SelectionSpec

RunOrigin = Literal["studio", "imported", "study_child", "reproduction", "external_live"]
RunStatus = Literal["queued", "running", "complete", "partial", "failed", "cancelled", "interrupted", "external_live",
                    "imported"]
RunMode = Literal["fast", "coarse", "full", "custom"]
StageArg = Literal["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"]


class RunSummary(BaseModel):
    id: str
    project_id: str | None = None
    label: str | None = None
    origin: RunOrigin
    status: RunStatus
    mode: RunMode
    coarse_m: float | None = None
    created_utc: str | None = None
    finished_utc: str | None = None
    duration_s: float | None = None
    n_points: int | None = None
    r2: float | None = None
    rmse: float | None = None
    coverage: float | None = None
    n_scenarios: int | None = None
    checkpoint_bytes: int | None = None
    has_emulator: bool = False
    studies: list[str] = Field(default_factory=list)
    git_commit: str | None = None
    git_dirty: bool | None = None
    demo: bool = False
    pinned: bool = False
    parent_run_id: str | None = None
    study_id: str | None = None
    last_job_id: str | None = None


class RunPage(BaseModel):
    items: list[RunSummary]
    next_cursor: str | None = None


class SnapshotMatch(BaseModel):
    data: bool | None = None
    code: bool | None = None
    config: bool | None = None


class CheckpointInfo(BaseModel):
    present: bool
    bytes: int | None = None
    done: list[str] = Field(default_factory=list)
    saved_utc: str | None = None
    fingerprint: str | None = None
    matches_snapshot: SnapshotMatch | None = None
    changed_sections: list[str] = Field(default_factory=list)
    resumable: bool
    reuses: list[str] = Field(default_factory=list)
    saves_s: float | None = None
    reason: str | None = None


class PlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["fast", "coarse", "full"]
    coarse_m: float | None = Field(None, gt=0)
    stages: list[StageArg] | None = None
    cv_curve: bool | None = None
    threads: int | None = Field(None, ge=1)
    resume_run_id: str | None = None


class LaunchRequest(PlanRequest):
    label: str | None = None
    notes: str | None = None
    then: list[Literal["post.planner", "post.emulator", "post.uncertainty", "post.writeup",
                       "post.baselines"]] | None = None


class PreflightRow(BaseModel):
    check: str
    ok: bool
    severity: Literal["error", "warn", "info"]
    message: str
    action: Action | None = None


class RunPlan(BaseModel):
    nodes: list[PlanNode]
    total_est_s: float
    est_lo: float
    est_hi: float
    est_peak_rss_gb: float
    est_disk_gb: float
    network_hosts: list[str]
    preflight: list[PreflightRow]
    issues: list[Issue]
    resumable: CheckpointInfo | None = None
    threads: int


class LaunchResponse(BaseModel):
    run: RunSummary
    job: Job
    chain: list[Job]


class RerunResponse(BaseModel):
    run: RunSummary
    job: Job


class ImportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dir: str
    project_id: str | None = None
    config_path: str | None = None
    trust_pickles: bool = False


class RunPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str | None = None
    notes: str | None = None
    pinned: bool | None = None


class ResumeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    threads: int | None = Field(None, ge=1)
    use_current_config: bool = False


class RerunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    use_current_config: bool = True
    label: str | None = None


class FreedBytes(BaseModel):
    freed_bytes: int


class RunHeader(BaseModel):
    name: str
    created_utc: str | None = None
    git_commit: str | None = None
    git_dirty: bool | None = None
    versions: dict[str, Any] | None = None
    n_points: int | None = None
    grid_shape: tuple[int, int] | None = None
    cell_m: float | None = None
    fast: bool = False
    coarse_m: float | None = None
    run_dir: str
    demo: bool = False


class StageRow(BaseModel):
    id: str
    state: str
    seconds: float | None = None
    source: Literal["events", "manifest", "run_state"]
    reason: str | None = None


class OutputsSummary(BaseModel):
    present: int
    missing: int
    stale: int
    writing: int


class SectionTag(BaseModel):
    present: bool
    source: Literal["manifest", "file"] | None = None
    stale: bool = False
    older_code: bool = False


class RunFlag(BaseModel):
    code: str
    severity: str
    message: str


class RunDetail(BaseModel):
    run: RunSummary
    header: RunHeader
    state: dict[str, Any] | None = None
    launch: dict[str, Any] | None = None
    stages: list[StageRow]
    checkpoint: CheckpointInfo
    outputs_summary: OutputsSummary
    sections: dict[str, SectionTag]
    flags: list[RunFlag]
    children: list[RunSummary]
    jobs: list[Job]
    warnings_count: int


class BoardColumn(BaseModel):
    id: str
    label: str
    group: Literal["stage", "post", "study"]


class BoardCell(BaseModel):
    state: Literal["done", "cached", "running", "failed", "skipped", "stale", "not_run", "disabled"]
    seconds: float | None = None
    progress: float | None = None
    reason: str | None = None
    job_id: str | None = None
    study_id: str | None = None
    action: Action | None = None


class BoardRow(BaseModel):
    run: RunSummary
    cells: dict[str, BoardCell]


class StatusBoard(BaseModel):
    columns: list[BoardColumn]
    rows: list[BoardRow]


class TimelineStage(BaseModel):
    id: str
    start_ts: float | None = None
    end_ts: float | None = None
    seconds: float | None = None
    state: str


class TimelineJob(BaseModel):
    job_id: str
    kind: str
    start_ts: float | None = None
    end_ts: float | None = None
    status: str


class Timeline(BaseModel):
    source: Literal["events", "manifest"]
    stages: list[TimelineStage]
    jobs: list[TimelineJob]


class RunConfig(BaseModel):
    effective: dict[str, Any]
    raw: dict[str, Any]
    yaml: str
    source: Literal["launch", "manifest", "import"]
    config_dir: str
    vs_project_diff: list[dict[str, Any]]
    vs_defaults: list[dict[str, Any]]


class RunProvenance(BaseModel):
    provenance: dict[str, Any] | None = None
    git: dict[str, Any] | None = None
    platform: dict[str, Any] | None = None
    hashes: dict[str, str]
    environment: list[str]
    launch: dict[str, Any] | None = None


class EnvChanged(BaseModel):
    name: str
    a: str
    b: str


class EnvDiff(BaseModel):
    added: list[str]
    removed: list[str]
    changed: list[EnvChanged]


class RunEnvironment(BaseModel):
    packages: list[str]
    diff: EnvDiff | None = None


class TabMissing(BaseModel):
    output: str
    produced_by: str
    action: Action | None = None


class TabState(BaseModel):
    id: str
    availability: Literal["ready", "partial", "running", "missing", "stale"]
    missing: list[TabMissing]


class OutputsResponse(BaseModel):
    outputs: list[OutputEntry]
    tabs: list[TabState]


class BatchMissing(BaseModel):
    id: str
    produced_by: str | None = None
    action: Action | None = None


class OutputsBatch(BaseModel):
    results: dict[str, Any]
    missing: list[BatchMissing]


class ViewUnits(BaseModel):
    target: str
    levers: dict[str, str]


class ViewModel(BaseModel):
    view: str
    availability: Literal["ready", "partial", "running", "missing", "stale"]
    missing: list[TabMissing]
    units: ViewUnits
    caveats: list[str]
    demo: bool
    sections: dict[str, Any]


class DocEntry(BaseModel):
    id: Literal["report", "methods", "model_card", "uncertainty", "placebo", "multiverse", "simcheck", "benchmark"]
    file: str
    title: str
    mtime: str | None = None
    present: bool
    regenerable: bool
    frozen: bool


class DocContent(BaseModel):
    markdown: str
    mtime: str
    frozen: bool


class FileEntry(BaseModel):
    name: str
    relpath: str
    dir: bool
    bytes: int | None = None
    mtime: str | None = None
    output_id: str | None = None
    state: str | None = None
    in_manifest: bool


class TableColumn(BaseModel):
    name: str
    dtype: str


class FileTable(BaseModel):
    columns: list[TableColumn]
    rows: list[list[Any]]
    n_rows: int


class DictionaryRow(BaseModel):
    output: str
    column: str
    unit: str
    sign: str | None = None
    description: str


# ---------------------------------------------------------------------------
# grid, layers, cells, selections, tools (api.md §6.2–6.4)
# ---------------------------------------------------------------------------

class LayerGroup(BaseModel):
    id: str
    label: str
    layers: list[dict[str, Any]]


class LayersResponse(BaseModel):
    groups: list[LayerGroup]


class CellCurve(BaseModel):
    model: str
    A: float | None = None
    ds: float | None = None
    inflection: float | None = None
    d90: float | None = None
    dmax: float | None = None
    dose: list[float]
    benefit: list[float]


class CellInfo(BaseModel):
    index: int
    id: int | str
    lon: float | None = None
    lat: float | None = None
    zone: int | float | str | None = None
    values: dict[str, float | None]
    curves: dict[str, CellCurve]
    scenarios: dict[str, float | None]


class HexRow(BaseModel):
    key: int
    cx: float
    cy: float
    lon: float | None = None
    lat: float | None = None
    n_cells: int
    values: dict[str, float | None]


class HexResponse(BaseModel):
    hex: list[HexRow]


class ResolveRequest(BaseModel):
    selection: SelectionSpec


class ResolveResponse(BaseModel):
    n_cells: int
    area_km2: float
    people: float | None = None
    medians: dict[str, float | None]
    mask: str
    portable: bool
    warnings: list[str]


class RegionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=200)
    spec: SelectionSpec


class RegionOut(BaseModel):
    id: str
    name: str
    spec: dict[str, Any]
    n_cells: int
    created_utc: str | None = None
    portable: bool | None = None


class BlobOut(BaseModel):
    blob_id: str
    bytes: int


class RegionStatsRequest(BaseModel):
    selection: SelectionSpec
    layers: list[str] = Field(default_factory=list)
    weights: Literal["people"] | None = None
    scenarios: list[str] | None = None


class LayerRegionStats(BaseModel):
    mean: float | None = None
    sd: float | None = None
    p10: float | None = None
    p50: float | None = None
    p90: float | None = None
    mean_outside: float | None = None


class ScenarioRegionStats(BaseModel):
    inside: Likely
    outside: Likely
    has_folds: bool


class RegionStats(BaseModel):
    n_cells: int
    area_km2: float
    people: float | None = None
    layers: dict[str, LayerRegionStats]
    scenarios: dict[str, ScenarioRegionStats]


class BreakdownBy(BaseModel):
    kind: Literal["zone", "quantile", "hex", "category", "fold"]
    layer: str | None = None
    q: int | None = Field(None, ge=2, le=20)
    size_m: Literal[250, 500] | None = None


class BreakdownRequest(BaseModel):
    value: str
    by: BreakdownBy
    weights: Literal["people"] | None = None
    stat: Literal["box", "mean"] = "box"


class BreakdownGroup(BaseModel):
    label: str
    n: int
    people: float | None = None
    mean: float | None = None
    q: list[float] | None = None


class BreakdownResponse(BaseModel):
    groups: list[BreakdownGroup]


class HexbinRequest(BaseModel):
    x: str
    y: str
    bins: int = Field(60, ge=2, le=400)
    selection: SelectionSpec | None = None


class BinnedMean(BaseModel):
    x: float
    y: float


class HexbinResponse(BaseModel):
    x_edges: list[float]
    y_edges: list[float]
    counts: list[list[int]]
    sel_counts: list[list[int]] | None = None
    spearman: float | None = None
    binned_mean: list[BinnedMean]


class AcfRequest(BaseModel):
    layer: str
    max_lag_m: float | None = Field(None, gt=0)
    n_perm: int = Field(19, ge=1, le=99)


class AcfResponse(BaseModel):
    lags_m: list[float | None]
    acf: list[float | None]
    band_mean: list[float | None]
    band_sd: list[float | None]


# ---------------------------------------------------------------------------
# compare (api.md §6.5)
# ---------------------------------------------------------------------------

class SameChips(BaseModel):
    data: bool | None = None
    config: bool | None = None
    code: bool | None = None
    grid: bool


class MetricDiff(BaseModel):
    key: str
    a: float | None = None
    b: float | None = None
    delta: float | None = None


class TimingDiff(BaseModel):
    stage: str
    a: float | None = None
    b: float | None = None


class ScenarioDiff(BaseModel):
    name: str
    a: Likely | None = None
    b: Likely | None = None


class OutputsDiff(BaseModel):
    a_only: list[str]
    b_only: list[str]


class CompareRuns(BaseModel):
    a: RunSummary
    b: RunSummary
    same: SameChips
    config_diff: list[dict[str, Any]]
    metrics: list[MetricDiff]
    timings: list[TimingDiff]
    scenarios: list[ScenarioDiff]
    climate: dict[str, Any] | None = None
    causal: dict[str, Any] | None = None
    environment: EnvDiff
    outputs: OutputsDiff


class PriorityRequest(BaseModel):
    a: str
    b: str
    layer: str


class PriorityResponse(BaseModel):
    kendall_tau: float
    top_decile_jaccard: float
    n: int
