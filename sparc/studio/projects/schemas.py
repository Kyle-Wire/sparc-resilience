"""Wire models of api.md §5 (projects, files, data check, config, inputs).

Field names follow api.md exactly.  ``RunSummary`` and ``Study`` belong to
the runs and studies items, so they appear here as plain objects.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from sparc.studio.schemas.common import Action, Issue, Job

ReadinessKey = Literal["data", "columns", "levers", "roles", "forcing", "climate_table", "people_layers",
                       "config_valid", "runs", "emulator", "studies"]


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# projects
# ---------------------------------------------------------------------------

class ProjectReport(BaseModel):
    title: str | None = None
    place: str | None = None
    area: str | None = None


class CostEntry(BaseModel):
    per_unit: float


class LastRun(BaseModel):
    id: str
    status: str
    created_utc: str
    r2: float | None = None


class ReadinessScore(BaseModel):
    done: int
    total: int


class Project(BaseModel):
    id: str
    slug: str
    name: str
    dir: str
    config_path: str
    template: str | None = None
    demo: bool
    active_run_id: str | None = None
    archived: bool
    created_utc: str
    updated_utc: str
    report: ProjectReport
    headline_scenario: str | None = None
    cost_model: dict[str, CostEntry] = Field(default_factory=dict)
    n_runs: int
    last_run: LastRun | None = None
    active_jobs: int
    readiness_score: ReadinessScore


class ReadinessRow(BaseModel):
    key: ReadinessKey
    label: str
    state: Literal["ok", "warn", "missing", "n/a"]
    detail: str
    action: Action | None = None


class ProjectOptions(_Body):
    seed: int | None = Field(None, ge=0, description="synthetic_demo: generator seed (default 0)")
    n: int | None = Field(None, ge=8, le=256, description="synthetic_demo: city side in cells (default 96)")
    import_existing_runs: bool | None = Field(None, description="providence_example: register output/core/"
                                              "providence runs and studies in place (default: when present)")
    trust_pickles: bool = Field(False, description="providence_example: mark imported checkpoints as trusted")


class ProjectCreate(_Body):
    name: str = Field(..., min_length=1, max_length=120)
    template: Literal["blank", "synthetic_demo", "providence_example"]
    options: ProjectOptions | None = None

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("the name must not be blank")
        return v


class ProjectCreated(BaseModel):
    project: Project
    imported_runs: list[str]
    warnings: list[str]


class ProjectImport(_Body):
    name: str | None = Field(None, max_length=120)
    config_path: str
    copy_data: bool = False
    run_dirs: list[str] = Field(default_factory=list)
    study_dirs: list[str] = Field(default_factory=list)
    trust_pickles: bool = False

    @field_validator("name")
    @classmethod
    def _name(cls, v: str | None) -> str | None:
        return (v.strip() or None) if isinstance(v, str) else v       # blank: the config's own name


class ProjectImported(BaseModel):
    project: Project
    runs: list[dict[str, Any]] = Field(description="RunSummary[] (api.md §6)")
    studies: list[dict[str, Any]] = Field(description="Study[] (api.md §9)")
    warnings: list[str]


class ProjectDetail(BaseModel):
    project: Project
    readiness: list[ReadinessRow]
    runs: list[dict[str, Any]] = Field(description="RunSummary[] (api.md §6), newest first")
    active_jobs: list[Job]
    config_version: int


class ReportPatch(_Body):
    title: str | None = None
    place: str | None = None
    area: str | None = None


class ProjectPatch(_Body):
    name: str | None = None
    active_run_id: str | None = None
    headline_scenario: str | None = None
    cost_model: dict[str, CostEntry] | None = None
    report: ReportPatch | None = None
    archived: bool | None = None


# ---------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------

class ColumnInfo(BaseModel):
    name: str
    dtype: str
    n_null: int
    min: float | None = None
    max: float | None = None
    n_unique: int | None = None
    sample: list[Any]


class FileInspect(BaseModel):
    n_rows: int
    n_rows_exact: bool
    columns: list[ColumnInfo]
    preview: list[list[Any]]


class FileUploaded(BaseModel):
    path: str
    bytes: int
    kind: str
    inspect: FileInspect | None = None


class FileRow(BaseModel):
    path: str
    kind: str
    bytes: int
    mtime: str
    used_by: list[str]


class SuggestRequest(_Body):
    path: str


class ColumnSuggestion(BaseModel):
    target: str | None = None
    id: str | None = None
    x: str | None = None
    y: str | None = None
    zone: str | None = None
    coord_unit: str
    crs_guess: str | None = None
    predictors: list[str]
    roles: dict[str, str]
    confidence: dict[str, float]


# ---------------------------------------------------------------------------
# data check
# ---------------------------------------------------------------------------

class DataCheckRequest(_Body):
    config_patch: dict[str, Any] | None = None


class CheckGrid(BaseModel):
    nx: int
    ny: int
    cell_m: float
    fill_fraction: float
    collisions: float


class CheckBackground(BaseModel):
    value: float
    source: str


class CheckFlag(BaseModel):
    code: str
    severity: Literal["warn", "info"]
    message: str


class DoseScale(BaseModel):
    sd: float | None = None
    doses: list[float]
    doses_in_sd: list[float | None]
    percentile_reached: list[float]


class DataCheck(BaseModel):
    n_points: int
    n_input: int
    n_dropped: int
    clipped: dict[str, int]
    grid: CheckGrid
    background: CheckBackground
    noise_floor: float | None = None
    flags: list[CheckFlag]
    dose_scale: dict[str, DoseScale]
    coarse: dict[str, Any] | None = None
    extent_m: tuple[float, float]
    columns_missing: list[str]
    preview_token: str
    preview_columns: list[str] = Field(default_factory=list, description="columns available as preview .bin")
    elapsed_s: float | None = None


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

class ChangedKey(BaseModel):
    path: str
    value: Any = None
    default: Any = None


class ConfigOut(BaseModel):
    version: int
    yaml: str
    raw: dict[str, Any]
    effective: dict[str, Any]
    changed_from_defaults: list[ChangedKey]
    comments_preserved: bool = False


class ConfigPut(_Body):
    yaml: str | None = None
    raw: dict[str, Any] | None = None
    note: str | None = None


class SectionPatch(_Body):
    value: Any = None
    note: str | None = None


class ConfigSaved(BaseModel):
    version: int
    issues: list[Issue]
    diff: str


class ConfigCandidate(_Body):
    yaml: str | None = None
    raw: dict[str, Any] | None = None


class ValidateOut(BaseModel):
    ok: bool
    issues: list[Issue]
    fast_overrides: dict[str, Any]
    coarse_preview: dict[str, Any] | None = None


class ImpactRun(BaseModel):
    run_id: str
    label: str
    checkpoint_done: list[str]
    changed_sections: list[str]
    refit_from: Literal["S0", "S1", "S2_S3", "baselines", "cv_curve", "S4", "S5", "climate", "S6", "S7",
                        "finish"] | None = None
    phrase: str


class ImpactOut(BaseModel):
    changed_sections: list[str]
    runs: list[ImpactRun]


class HistoryRow(BaseModel):
    version: int
    saved_utc: str | None = None
    note: str | None = None


class HistoryVersion(BaseModel):
    version: int
    yaml: str


class LinkRequest(_Body):
    kind: Literal["forcing", "climate", "layers", "features_join", "features_new_project"]
    path: str
    apply: bool = False


class LinkOut(BaseModel):
    yaml_diff: str
    applied: bool
    version: int | None = None
    new_project_id: str | None = None


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

class ForcingState(BaseModel):
    path: str
    date: str
    physics: dict[str, Any]
    checks: list[str]
    linked: bool


class ClimateState(BaseModel):
    path: str
    n_models: int
    experiments: list[str]
    periods: list[str]
    linked: bool


class LayersState(BaseModel):
    path: str
    n: int
    people_total: float
    linked: bool


class FeaturesState(BaseModel):
    path: str
    agreement: list[dict[str, Any]]
    linked: bool


class GhcnState(BaseModel):
    station: str
    years: tuple[int | None, int | None]


class InputsState(BaseModel):
    forcing: ForcingState | None = None
    climate: ClimateState | None = None
    layers: LayersState | None = None
    features: FeaturesState | None = None
    ghcn: GhcnState | None = None


class StationRow(BaseModel):
    usaf_wban: str
    name: str
    lat: float
    lon: float
    dist_km: float
    begin: str | None = None
    end: str | None = None
