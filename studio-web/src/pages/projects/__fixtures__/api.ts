// api.md fixtures for the projects pages: a project and its detail (§5), the synthetic demo
// config (raw core block and config.yml text), a header inspect and column suggestion
// (§5.1), a data check with a dose scale (§5.2), validation issues (§1 Issue), an impact
// preview (§5.3), and run plans (§6 `runs/plan`). Values follow the synthetic demo city.
import type {
  ColumnSuggestion,
  ConfigDoc,
  DataCheck,
  FileInspect,
  ImpactResult,
  ProjectDetail,
  ProjectFile,
  RunPlan,
} from "../../../api/projects";
import type { Issue, Job, PlanNode, Project, RunSummary } from "../../../api/types";

export const PID = "p_demo";

export function project(extra: Partial<Project> = {}): Project {
  return {
    id: PID,
    slug: "synthetic-city",
    name: "Synthetic city",
    dir: "/w/projects/synthetic-city",
    config_path: "/w/projects/synthetic-city/config.yml",
    template: "synthetic_demo",
    demo: true,
    active_run_id: null,
    archived: false,
    created_utc: "2026-10-01T20:00:00Z",
    updated_utc: "2026-10-01T20:05:00Z",
    report: { title: "Synthetic city (DEMO)", place: "a fictional city (synthetic data)", area: null },
    headline_scenario: null,
    cost_model: {},
    n_runs: 0,
    last_run: null,
    active_jobs: 0,
    readiness_score: { done: 8, total: 11 },
    ...extra,
  };
}

export function projectDetail(extra: Partial<Project> = {}, configVersion = 3): ProjectDetail {
  return {
    project: project(extra),
    readiness: [
      { key: "data", label: "Data file", state: "ok", detail: "data/city.csv", action: null },
      { key: "columns", label: "Columns mapped", state: "ok", detail: "target T, x/y", action: null },
      { key: "levers", label: "Levers", state: "ok", detail: "3 levers", action: null },
      { key: "roles", label: "Physics roles", state: "warn", detail: "5/6", action: null },
      { key: "forcing", label: "Campaign forcing", state: "missing", detail: "none", action: null },
      { key: "climate_table", label: "Climate table", state: "ok", detail: "demo_cmip6.csv", action: null },
      { key: "people_layers", label: "People layers", state: "ok", detail: "demo_layers.parquet", action: null },
      { key: "config_valid", label: "Config valid", state: "ok", detail: "no errors", action: null },
      { key: "runs", label: "Runs", state: "missing", detail: "none", action: null },
      { key: "emulator", label: "Emulator", state: "n/a", detail: "", action: null },
      { key: "studies", label: "Studies", state: "n/a", detail: "", action: null },
    ],
    runs: [],
    active_jobs: [],
    config_version: configVersion,
  };
}

/**
 * The synthetic demo's raw core block (sparc.core.synthetic.demo_config, trimmed). The
 * canopy role is left unmapped so the missing-role warning shows.
 */
export function demoRaw(): Record<string, unknown> {
  return {
    name: "synthetic_demo",
    data: { path: "data/city.csv", target: "T", x: "x", y: "y", coord_unit: "m", target_units: "degF", background: "median", crs: "EPSG:32619", id: "id" },
    predictors: ["canopy", "impervious", "albedo", "ndvi", "elevation", "water_dist"],
    qa: { clip: { canopy: [0, 100], impervious: [0, 100], albedo: [0.02, 0.9] } },
    actionable: {
      canopy: { min: 0, max: 100, doses: [0, 5, 10, 15, 20, 30, 40], cost_per_unit: 1.0, direction: "increase", unit: "pp" },
      impervious: { min: 0, max: 100, doses: [0, 5, 10, 20], direction: "decrease", unit: "pp" },
      albedo: { min: 0.02, max: 0.9, doses: [0, 0.05, 0.1, 0.15, 0.2], direction: "increase", unit: "albedo" },
    },
    mediators: { ndvi: { parents: ["canopy", "impervious"], context: ["elevation"], monotone: { canopy: 1, impervious: -1 } } },
    physics: { roles: { albedo: "albedo", impervious: "impervious", ndvi: "ndvi", elevation: "elevation", water_distance: "water_dist" }, tau_s: 1800 },
    cv: { n_folds: 3 },
    stacker: { epochs: 200, tune_lambda: [0, 1] },
    causal: { treatments: ["canopy"], confounders: { canopy: ["impervious", "albedo", "elevation", "water_dist"] }, contrast: { canopy: 10 } },
    optimize: { variable: "canopy", budget: 2000 },
    scenarios: [
      { name: "Canopy Increase", variable: "canopy", direction: "increase", increments: [5, 10, 20] },
      { name: "Impervious Decrease", variable: "impervious", direction: "decrease", increments: [10, 20] },
      { name: "Albedo Increase", variable: "albedo", direction: "increase", increments: [0.05, 0.1] },
    ],
    joint_scenarios: [
      {
        name: "Cooling package",
        interventions: [
          { variable: "canopy", direction: "increase", increment: 10 },
          { variable: "impervious", direction: "decrease", increment: 10 },
        ],
      },
    ],
    climate: { enabled: true, source: "table", table: "inputs/climate/demo_cmip6.csv" },
    planner: { layers: "inputs/layers/demo_layers.parquet" },
    report: { title: "Synthetic city (DEMO)", place: "a fictional city (synthetic data)" },
  };
}

/** A DEFAULTS-merged copy of a raw block, as `GET /config` `effective` would hold it. */
export function effectiveOf(raw: Record<string, unknown>): Record<string, unknown> {
  const physics = { enabled: true, window: "day", sw_down: 800, lw_net: -100, roles: {}, tau_s: 1800, L_max_m: 2000, v_max_m: 1000, fit_advection: "auto", select_advection: true, max_iter: 60 };
  return {
    ...raw,
    data: { x: "x", y: "y", coord_unit: "m", target_units: "degF", background: "median", join: [], ...(raw.data as object) },
    physics: { ...physics, ...((raw.physics as object) ?? {}) },
    optimize: { enabled: true, cost_per_unit: 1, plantable: true, objective: "cooling", equity_focus: 0, ...((raw.optimize as object) ?? {}) },
  };
}

export const DEMO_YAML = `core:
  name: synthetic_demo
  data:
    path: data/city.csv
    target: T
    x: x
    y: y
    coord_unit: m
    target_units: degF
    background: median
    crs: EPSG:32619
    id: id
  predictors:
  - canopy
  - impervious
  actionable:
    canopy:
      min: 0
      max: 100
      doses: [0, 5, 10]
  stacker:
    epochs: 200
    tune_lambda:
    - 0.0
    - 1.0
  scenarios:
  - name: Canopy Increase
    variable: canopy
    increments:
    - 5
    - 10
`;

export function configDoc(raw: Record<string, unknown> = demoRaw(), version = 3, yaml = DEMO_YAML): ConfigDoc {
  return {
    version,
    yaml,
    raw,
    effective: effectiveOf(raw),
    changed_from_defaults: [
      { path: "cv.n_folds", value: 3, default: 5 },
      { path: "stacker.epochs", value: 200, default: 400 },
    ],
    comments_preserved: false,
  };
}

export const FILES: ProjectFile[] = [
  { path: "data/city.csv", kind: "data", bytes: 412_000, mtime: "2026-10-01T20:00:00Z", used_by: ["data.path"] },
  { path: "inputs/climate/demo_cmip6.csv", kind: "climate", bytes: 9_000, mtime: "2026-10-01T20:00:00Z", used_by: ["climate.table"] },
  { path: "inputs/layers/demo_layers.parquet", kind: "layers", bytes: 120_000, mtime: "2026-10-01T20:00:00Z", used_by: ["planner.layers"] },
];

const col = (name: string, dtype: string, min: number | null, max: number | null, n_unique: number | null, sample: unknown[]) => ({ name, dtype, n_null: 0, min, max, n_unique, sample });

export const INSPECT: FileInspect = {
  n_rows: 9216,
  n_rows_exact: true,
  columns: [
    col("id", "int64", 0, 9215, 9216, [0, 1, 2]),
    col("x", "float64", 300000, 302850, 96, [300000, 300030, 300060]),
    col("y", "float64", 4630000, 4632850, 96, [4630000, 4630000, 4630000]),
    col("T", "float64", 81.2, 93.6, 4100, [88.1, 87.9, 88.4]),
    col("canopy", "float64", 0, 92, 820, [12.5, 40.1, 3.2]),
    col("impervious", "float64", 0, 100, 900, [70.2, 20.4, 88.8]),
    col("albedo", "float64", 0.05, 0.31, 260, [0.14, 0.18, 0.12]),
    col("ndvi", "float64", -0.1, 0.82, 700, [0.21, 0.55, 0.08]),
    col("elevation", "float64", 2, 61, 590, [12, 15, 9]),
    col("water_dist", "float64", 0, 2400, 900, [120, 860, 40]),
    col("district", "object", null, null, 6, ["north", "south", "east"]),
  ],
  preview: [
    [0, 300000, 4630000, 88.1, 12.5, 70.2, 0.14, 0.21, 12, 120, "north"],
    [1, 300030, 4630000, 87.9, 40.1, 20.4, 0.18, 0.55, 15, 860, "north"],
  ],
};

export const SUGGESTION: ColumnSuggestion = {
  target: "T",
  id: "id",
  x: "x",
  y: "y",
  zone: "district",
  coord_unit: "m",
  crs_guess: null,
  predictors: ["canopy", "impervious", "albedo", "ndvi", "elevation", "water_dist"],
  roles: { canopy: "canopy", impervious: "impervious", albedo: "albedo", ndvi: "ndvi", elevation: "elevation", water_distance: "water_dist" },
  confidence: { target: 0.8, id: 0.9, x: 0.9, y: 0.9, zone: 0.7, coord_unit: 0.8, crs: 0, "roles.canopy": 0.8 },
};

export const CHECK: DataCheck = {
  n_points: 9216,
  n_input: 9216,
  n_dropped: 0,
  clipped: { albedo: 0 },
  grid: { nx: 96, ny: 96, cell_m: 30, fill_fraction: 1, collisions: 0 },
  background: { value: 88.02, source: "median" },
  noise_floor: null,
  flags: [
    { code: "dose_scale_canopy", severity: "info", message: "canopy doses 40 exceed 3 sd of the layer (sd 11.8)" },
    { code: "target_rounded", severity: "warn", message: "72% of target values are whole degrees" },
  ],
  dose_scale: {
    canopy: { sd: 11.8, doses: [5, 10, 15, 20, 30, 40], doses_in_sd: [0.42, 0.85, 1.27, 1.69, 2.54, 3.39], percentile_reached: [63, 74, 82, 88, 95, 98] },
    impervious: { sd: 24.0, doses: [5, 10, 20], doses_in_sd: [0.21, 0.42, 0.83], percentile_reached: [41, 33, 20] },
  },
  coarse: null,
  extent_m: [2850, 2850],
  columns_missing: [],
  preview_token: "tok1",
  preview_columns: ["T", "canopy", "impervious"],
};

/** Validation issues at the dotted paths of several steps. */
export const ISSUES: Issue[] = [
  { level: "error", path: "data.target", code: "required", message: "data.target (the temperature column) is required" },
  { level: "error", path: "data.crs", code: "needs_crs", message: "objective 'people' needs data.crs" },
  { level: "warn", path: "actionable.canopy.doses.6", code: "dose_beyond", message: "dose 40 is beyond 3 sd" },
  { level: "warn", path: "physics.roles", code: "roles_missing", message: "canopy role not mapped" },
  { level: "error", path: "scenarios.1.variable", code: "scenario_not_actionable", message: "'impervious_x' is not a lever" },
  { level: "error", path: "climate.table", code: "missing_file", message: "climate table not found: x.csv" },
  { level: "error", path: "optimize.variable", code: "optimize_not_actionable", message: "optimize.variable 'ndvi' is not a lever" },
  { level: "info", path: "report.title", code: "no_title", message: "No report title: pages use the project name" },
];

export const IMPACT: ImpactResult = {
  changed_sections: ["stacker"],
  runs: [
    { run_id: "20261001-142233-fast-a1b2", label: "first fast run", checkpoint_done: ["S3", "S4"], changed_sections: ["core"], refit_from: "S2_S3", phrase: "a re-run with this config would refit from S2_S3" },
    { run_id: "20261001-152233-full-c3d4", label: "full run", checkpoint_done: ["S3"], changed_sections: ["core"], refit_from: "S2_S3", phrase: "a re-run with this config would refit from S2_S3" },
  ],
};

const node = (id: PlanNode["id"], label: string, state: PlanNode["state"], reason: string | null, units: Record<string, number> = {}, est: [number, number, number] | null = null): PlanNode => ({
  id,
  label,
  state,
  reason,
  units,
  checkpoint_key: null,
  est_s: est ? est[0] : null,
  est_lo: est ? est[1] : null,
  est_hi: est ? est[2] : null,
});

/** A plan for stages S0–S3 + S6 (S4 forced by S6), baselines off, CV curve off, S5 not requested. */
export function planNodes(): PlanNode[] {
  return [
    node("S0", "Data and QA", "will_run", null, { s0_load: 1 }, [0.3, 0.2, 0.4]),
    node("S1", "Area of influence", "cached", "checkpoint"),
    node("S2_S3", "Base models and stacker", "will_run", null, { "base_fit:mgwr": 3, "base_fit:gam": 3, "stacker_fit:mean": 3, checkpoint_save: 1 }, [180, 150, 220]),
    node("baselines", "Reference baselines", "skipped", "disabled_by_config:cv.baselines"),
    node("cv_curve", "Skill vs distance", "skipped", "disabled_by_config:cv.distance_curve.enabled"),
    node("S4", "Response surfaces", "will_run", "required_by:S6", { engine_pass: 22, checkpoint_save: 1 }, [290, 240, 360]),
    node("S5", "Scenarios", "skipped", "not_requested"),
    node("climate", "Climate and adaptation", "skipped", "requires_S5"),
    node("S6", "Causal validation", "will_run", null, { engine_pass: 8, "causal_step:dml": 1, "causal_step:dr": 1 }, [160, 130, 200]),
    node("S7", "Budget optimisation", "skipped", "no_budget"),
    node("finish", "Manifest and report", "will_run", null, {}, [2, 1, 3]),
  ];
}

export function runPlan(extra: Partial<RunPlan> = {}): RunPlan {
  return {
    nodes: planNodes(),
    total_est_s: 632,
    est_lo: 540,
    est_hi: 780,
    est_peak_rss_gb: 1.3,
    est_disk_gb: 0.09,
    network_hosts: [],
    preflight: [
      { check: "config_valid", ok: true, severity: "error", message: "The config is valid", action: null },
      { check: "disk", ok: true, severity: "warn", message: "54 GB free", action: null },
    ],
    issues: [],
    resumable: null,
    threads: 3,
    ...extra,
  };
}

export const PREFLIGHT_ERROR = {
  check: "data_file",
  ok: false,
  severity: "error" as const,
  message: "data/city.csv is missing",
  action: { kind: "open" as const, label: "Open the data step", method: "GET" as const, path: `/p/${PID}/setup/data` },
};

export function job(extra: Partial<Job> = {}): Job {
  return {
    id: "j_launch1",
    kind: "run.core",
    lane: "heavy",
    executor: "process",
    label: "Fast run",
    status: "queued",
    project_id: PID,
    run_id: "20261001-220000-fast-9f9f",
    study_id: null,
    scenario_id: null,
    parent_job_id: null,
    after_job_id: null,
    priority: 0,
    params: { run_id: "20261001-220000-fast-9f9f", resume: false },
    created_utc: "2026-10-01T22:00:00Z",
    started_utc: null,
    finished_utc: null,
    progress: null,
    eta_s: null,
    eta_lo: null,
    eta_hi: null,
    stage: null,
    current_path: null,
    exit_code: null,
    error: null,
    blocked: null,
    result: null,
    peak_rss_mb: null,
    threads: 3,
    ...extra,
  };
}

export function runSummary(extra: Partial<RunSummary> = {}): RunSummary {
  return {
    id: "20261001-220000-fast-9f9f",
    project_id: PID,
    label: null,
    origin: "studio",
    status: "queued",
    mode: "fast",
    coarse_m: null,
    created_utc: "2026-10-01T22:00:00Z",
    finished_utc: null,
    duration_s: null,
    n_points: null,
    r2: null,
    rmse: null,
    coverage: null,
    n_scenarios: null,
    checkpoint_bytes: null,
    has_emulator: false,
    studies: [],
    git_commit: null,
    git_dirty: null,
    demo: true,
    pinned: false,
    parent_run_id: null,
    study_id: null,
    last_job_id: "j_launch1",
    ...extra,
  };
}
