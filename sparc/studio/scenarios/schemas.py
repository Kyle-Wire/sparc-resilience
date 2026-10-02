"""Request and response models of the Scenario Lab (api.md §7).  Field names follow api.md exactly.

``ScenarioDoc`` is lenient about unknown keys (a stored doc carries ``id``, ``revision`` … next to the
editable fields, SPEC §7.2); request bodies that are not documents forbid them.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from sparc.studio.schemas.common import Action, Job, Likely, SelectionSpec

EditMode = Literal["add", "set", "scale", "floor", "ceiling", "fill_headroom", "to_percentile", "per_cell"]
ScenarioStatus = Literal["draft", "previewed", "exact", "stale", "archived"]
ResultKind = Literal["exact", "configured", "plan", "sweep_point"]
EngineState = Literal["no_checkpoint", "cold", "queued", "loading", "ready", "busy", "incompatible", "error"]
EngineHostState = Literal["absent", "starting", "ready", "busy", "recycling", "error"]


# ---------------------------------------------------------------------------
# documents
# ---------------------------------------------------------------------------

class Edit(BaseModel):
    model_config = ConfigDict(extra="ignore")
    lever: str
    mode: EditMode
    amount: float | None = None
    percentile: float | None = None
    paved_share: float | None = Field(None, ge=0, le=1)
    per_cell_ref: str | None = None
    where: SelectionSpec | None = None
    label: str | None = None


class CostSpec(BaseModel):
    model_config = ConfigDict(extra="ignore")
    per_unit: float = Field(ge=0)


class ScenarioOptions(BaseModel):
    model_config = ConfigDict(extra="ignore")
    clip_to_support: bool = True
    mediators: bool = True
    expert: bool = False


class ScenarioDoc(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str = Field(min_length=1, max_length=200)
    notes: str = ""
    tags: list[str] = Field(default_factory=list)
    anchor_run_id: str | None = None
    edits: list[Edit] = Field(default_factory=list)
    regions: dict[str, SelectionSpec] = Field(default_factory=dict)
    costs: dict[str, CostSpec] = Field(default_factory=dict)
    options: ScenarioOptions = Field(default_factory=ScenarioOptions)


class ResultSummary(BaseModel):
    id: str
    scenario_id: str | None = None
    run_id: str
    kind: ResultKind
    created_utc: str
    stale: bool
    has_folds: bool
    city: Likely
    edited: Likely | None = None
    frac_extrapolated_edited: float | None = None
    job_id: str | None = None


class Scenario(BaseModel):
    id: str
    project_id: str
    revision: int
    parent_id: str | None = None
    children: list[str] = Field(default_factory=list)
    doc: ScenarioDoc
    content_hash: str
    status: ScenarioStatus
    created_utc: str
    updated_utc: str
    results: list[ResultSummary] = Field(default_factory=list)


class ScenarioSummary(BaseModel):
    id: str
    revision: int
    parent_id: str | None = None
    status: ScenarioStatus
    created_utc: str
    updated_utc: str
    name: str
    tags: list[str] = Field(default_factory=list)
    latest: ResultSummary | None = None


class ScenarioCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    doc: ScenarioDoc


class ScenarioPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    doc: ScenarioDoc | None = None
    name: str | None = None
    tags: list[str] | None = None
    notes: str | None = None
    archived: bool | None = None


class ForkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str | None = None
    doc: ScenarioDoc | None = None


class FromTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    template: str
    params: dict[str, Any] = Field(default_factory=dict)
    run_id: str


class Template(BaseModel):
    id: str
    label: str
    desc: str
    params_schema: dict[str, Any]
    requires: list[Literal["layers", "canopy_role", "impervious_role", "albedo_role", "crs"]]


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------

class EngineRunRow(BaseModel):
    run_id: str
    est_rss_mb: float
    loaded_utc: str
    last_used_utc: str


class EngineHost(BaseModel):
    state: EngineHostState
    pid: int | None = None
    rss_mb: float | None = None
    budget_gb: float
    max_runs: int
    runs: list[EngineRunRow] = Field(default_factory=list)
    busy_job_id: str | None = None
    queue: list[str] = Field(default_factory=list)


class EngineError(BaseModel):
    type: str
    message: str


class RunEngine(BaseModel):
    state: EngineState
    progress: float | None = None
    step: str | None = None
    rss_mb: float | None = None
    est_rss_mb: float
    code_match: bool | None = None
    loaded_utc: str | None = None
    last_used_utc: str | None = None
    error: EngineError | None = None
    job_id: str | None = None
    action: Action | None = None


# ---------------------------------------------------------------------------
# levers, emulator, preview, compile
# ---------------------------------------------------------------------------

class LeverEmulator(BaseModel):
    available: bool
    trust: Literal["good", "rough", "none"]
    patch_pass_rate: float | None = None
    uniform_rel_err: float | None = None


class Lever(BaseModel):
    var: str
    label: str
    unit: str
    min: float | None = None
    max: float | None = None
    direction: Literal["increase", "decrease"]
    doses: list[float]
    cost_per_unit: float
    design_dose: float | None = None
    sd: float | None = None
    headroom_available: bool
    role: str | None = None
    mediator_children: list[str] = Field(default_factory=list)
    emulator: LeverEmulator


class EmulatorLever(BaseModel):
    design_dose: float | None = None
    bounds: list[float]
    direction: str
    trust: Literal["good", "rough", "none"]
    validation: dict[str, Any] = Field(default_factory=dict)


class EmulatorInfo(BaseModel):
    present: bool
    kernel_cells: int | None = None
    levers: dict[str, EmulatorLever] = Field(default_factory=dict)
    action: Action | None = None


class SparseEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    idx: str
    val: str


class PreviewRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    edits: list[Edit] = Field(default_factory=list)
    brush: dict[str, SparseEdit] | None = None
    options: ScenarioOptions | None = None
    request_seq: int
    scenario_id: str | None = Field(None, description="as built: marks a draft scenario 'previewed'")


class CompileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario: ScenarioDoc


class CompileLever(BaseModel):
    n_cells: int
    mean_requested: float
    total_requested: float
    predicted_mean_realised: float
    clipped_share: float
    est_cost: float


class CompileWarning(BaseModel):
    code: str
    message: str
    edit_index: int | None = None
    blocking: bool


class CompileEmulator(BaseModel):
    usable: bool
    hatched: bool
    reasons: list[str] = Field(default_factory=list)


class CompileResponse(BaseModel):
    content_hash: str
    portable: bool
    levers: dict[str, CompileLever]
    union_cells: int
    people: float | None = None
    warnings: list[CompileWarning]
    est_exact_s: float
    emulator: CompileEmulator


# ---------------------------------------------------------------------------
# runs of scenarios
# ---------------------------------------------------------------------------

class RunScenarioRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    force: bool = False


class RunScenarioResponse(BaseModel):
    cached: ResultSummary | None = None
    job: Job | None = None


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    scenario_ids: list[str] = Field(min_length=1)


class JobOut(BaseModel):
    job: Job


class LadderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    edit_index: int = Field(ge=0)
    amounts: list[float] = Field(min_length=1, max_length=20)
    run_id: str | None = None


class LadderResponse(BaseModel):
    scenarios: list[Scenario]
    job: Job | None = None


class AcrossRunsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_ids: list[str] = Field(min_length=1)


class AcrossRunsRow(BaseModel):
    run_id: str
    ok: bool
    reason: str | None = None
    load_s: float
    exact_s: float
    rss_gb: float


class AcrossRunsEstimate(BaseModel):
    runs: list[AcrossRunsRow]
    total_s: float
    peak_rss_gb: float


class PromoteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    apply: bool = False


class PromoteResponse(BaseModel):
    eligible: bool
    reason: str | None = None
    yaml_diff: str | None = None
    names: list[str] = Field(default_factory=list)
    version: int | None = None


class DesignImport(BaseModel):
    blobs: dict[str, str]
    n_rows: int
    unknown_ids: list[int | str]
    levers: list[str]
    mode: Literal["change", "value"]


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------

class CausalLinear(BaseModel):
    delta: float
    lo: float
    hi: float
    model_within: bool


class ConfiguredScenario(BaseModel):
    slug: str
    name: str
    city: Likely
    p10: float | None = None
    p90: float | None = None
    frac_extrapolated: float | None = None
    mean_realized: dict[str, float] = Field(default_factory=dict)
    causal_linear: CausalLinear | None = None
    has_folds: bool
    layer_key: str
    doc: ScenarioDoc


class RunScenarios(BaseModel):
    configured: list[ConfiguredScenario]
    results: list[ResultSummary]


class ImpactsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    thresholds: list[float] | None = None
    futures: list[dict[str, str]] | None = None


class Impacts(BaseModel):
    model_config = ConfigDict(extra="allow")
    thresholds: list[float]
    exposure: list[dict[str, Any]]
    equity: dict[str, Any]
    hot_days: dict[str, Any] | None = None
    hot_days_action: Action | None = None
    zones: list[dict[str, Any]]
    hexes: dict[str, list[dict[str, Any]]]
    climate_offset: list[dict[str, Any]]


class ResultScenarioRef(BaseModel):
    id: str
    revision: int
    name: str


class ResultRegion(BaseModel):
    name: str
    auto: bool
    n_cells: int
    mean: Likely | None = None
    people_weighted: float | None = None
    total: float
    frac_cooled_01: float
    frac_cooled_05: float


class Ring(BaseModel):
    r_m: float
    mean: float | None = None
    se: float | None = None
    n: int


class Spill(BaseModel):
    inside: float
    outside: float
    outside_share: float | None = None
    rings: list[Ring]
    lever_ranges: dict[str, float]


class Realized(BaseModel):
    requested_mean: float
    realized_mean: float
    requested_total: float
    realized_total: float
    clipped_share: float


class Cost(BaseModel):
    total: float
    per_lever: dict[str, float]
    cooling_per_cost: float | None = None


class Uncertainty(BaseModel):
    estimation_95: list[float] | None = None
    specification: list[float] | None = None
    attribution: list[float] | None = None
    causal_band: list[float] | None = None
    envelope: list[float] | None = None
    envelope_excludes_zero: bool | None = None
    sources: list[str] = Field(default_factory=list)


class PreviewVsExact(BaseModel):
    mean_abs_err: float
    rel_err: float


class Plain(BaseModel):
    headline: str
    confidence: str
    qualifiers: list[str]
    buys: list[str]


class ResultWarning(BaseModel):
    code: str
    message: str


class Result(BaseModel):
    summary: ResultSummary
    spec: dict[str, Any]
    scenario: ResultScenarioRef | None = None
    city: Likely
    p10: float | None = None
    p90: float | None = None
    mean_delta_sd: float | None = None
    regions: list[ResultRegion]
    spill: Spill
    extrapolated_edited: float
    realized: dict[str, Realized]
    mediators: dict[str, dict[str, float]]
    cost: Cost
    causal_check: CausalLinear | None = None
    uncertainty: Uncertainty | None = None
    impacts: Impacts | None = None
    preview_vs_exact: PreviewVsExact | None = None
    plain: Plain
    warnings: list[ResultWarning]
    stale: bool
    demo: bool


# ---------------------------------------------------------------------------
# compare, climate, sweeps
# ---------------------------------------------------------------------------

class ItemRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["result", "configured", "plan", "baseline"]
    id: str | None = None
    slug: str | None = None


class CompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[ItemRef] = Field(min_length=2, max_length=4)
    regions: list[str] | None = None
    thresholds: list[float] | None = None


class CompareItem(BaseModel):
    ref: dict[str, Any]
    label: str
    city: Likely
    edited: Likely | None = None
    cost: float | None = None
    has_folds: bool


class PairedLikely(Likely):
    paired: bool


class ComparePair(BaseModel):
    a: int
    b: int
    city: PairedLikely
    regions: dict[str, PairedLikely] = Field(default_factory=dict)
    layer_key: str


class Comparison(BaseModel):
    id: str
    items: list[CompareItem]
    pairs: list[ComparePair]
    equity: dict[str, dict[str, float | None]] = Field(default_factory=dict)
    exposure: list[dict[str, Any]] = Field(default_factory=list)
    cooling_per_cost: dict[str, float | None] = Field(default_factory=dict)
    needs_exact: list[dict[str, Any]] = Field(default_factory=list)


class ComparisonRow(BaseModel):
    id: str
    items: list[dict[str, Any]]
    created_utc: str


class ClimateExplore(BaseModel):
    model_config = ConfigDict(extra="forbid")
    adaptations: list[ItemRef] = Field(default_factory=list)
    thresholds: list[float] | None = None
    experiments: list[str] | None = None
    periods: list[str] | None = None
    statistic: Literal["median", "p10", "p90"] | dict[str, str] = "median"


class SweepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lever: str
    doses: list[float] = Field(min_length=1, max_length=40)
    selection: SelectionSpec | None = None


class SweepCreated(BaseModel):
    sweep_id: str
    job: Job


class SweepPoint(BaseModel):
    dose: float
    city: Likely
    region: Likely | None = None
    realized: float
    frac_extrapolated: float


class SweepFit(BaseModel):
    model: str
    A: float | None = None
    ds: float | None = None
    d90: float | None = None


class Sweep(BaseModel):
    params: dict[str, Any]
    status: str
    curve: list[SweepPoint]
    fit: SweepFit | None = None
    pipeline_curve: dict[str, Any] | None = None
    points: list[str]


class SweepRow(BaseModel):
    id: str
    lever: str
    doses: list[float]
    status: str
    created_utc: str
    job_id: str | None = None


# ---------------------------------------------------------------------------
# plans
# ---------------------------------------------------------------------------

class PlanCostScalar(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scalar: float = Field(gt=0)


class PlanCostColumn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    column: str


class PlanCap(BaseModel):
    model_config = ConfigDict(extra="forbid")
    plantable: bool = False
    paved_share: float | None = Field(None, ge=0, le=1)
    region: SelectionSpec | None = None


class PlanEquity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: Literal["share_60_plus", "share_under_5", "density", "column"]
    column: str | None = None
    focus: float = Field(0.0, ge=0, le=1)


class PlanParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lever: str
    budget: float = Field(gt=0)
    cost: PlanCostScalar | PlanCostColumn = Field(default_factory=lambda: PlanCostScalar(scalar=1.0))
    cap: PlanCap = Field(default_factory=PlanCap)
    min_dose: float = Field(0.0, ge=0)
    objective: Literal["cooling", "people"] = "cooling"
    equity: PlanEquity | None = None
    multipliers: list[float] | None = None


class ParetoPoint(BaseModel):
    budget: float
    benefit: float
    n_cells: int
    n_segments: int
    gini: float


class PlanPlanned(BaseModel):
    planned_total: float
    n_cells_treated: int
    mean_dose_treated: float
    total_cost: float
    gini: float
    min_dose_dropped_cost: float
    pareto: list[ParetoPoint]
    constraint: str
    objective: str
    caption: str


class PlanPreview(PlanPlanned):
    dose: str


class PlanCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    params: PlanParams
    name: str = Field(min_length=1, max_length=200)
    verify: bool = True


class PlanRealised(BaseModel):
    total: float
    mean_treated: float
    mean_all: float
    result_id: str


class FrontierPoint(BaseModel):
    budget: float
    planned: float
    realised: float


class Plan(BaseModel):
    id: str
    name: str
    params: PlanParams
    planned: PlanPlanned
    realised: PlanRealised | None = None
    frontier: list[FrontierPoint] | None = None
    created_utc: str


class PlanCreated(BaseModel):
    plan: Plan
    job: Job | None = None


class VerifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frontier: bool = False


class FieldKitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_sites: int = Field(30, ge=1, le=500)
    min_spacing_m: float = Field(400.0, ge=0)
    n_pairs: int = Field(30, ge=1, le=500)
    min_distance_m: float = Field(1000.0, ge=0)


class FieldKit(BaseModel):
    cells: list[dict[str, Any]]
    sites: list[dict[str, Any]]
    pairs: list[dict[str, Any]]
