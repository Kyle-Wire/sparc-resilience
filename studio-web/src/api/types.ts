// Common wire types shared by the foundation and every feature item (api.md §0–1, §2, §17).
// Group-specific types (projects, runs, lab, studies, …) live in each group's api/<group>.ts.
// Field names and unions follow api.md exactly; api.md wins on any disagreement.

// ---------------------------------------------------------------- errors (§0.3, §16)

export type ErrorCode =
  | "bad_host" | "unauthorized" | "bad_origin"
  | "not_found" | "output_missing" | "unknown_kind" | "unknown_view" | "unknown_layer"
  | "no_emulator" | "no_climate_factors" | "example_unavailable"
  | "conflict" | "conflict_revision" | "active" | "not_cancellable" | "not_resumable" | "not_ready"
  | "no_checkpoint" | "engine_memory" | "untrusted_pickle" | "preflight_failed"
  | "grid_mismatch" | "string_ids" | "has_children" | "exists" | "in_use" | "imported_in_place" | "superseded"
  | "too_large" | "too_large_inline" | "too_many_features"
  | "bad_suffix" | "no_conversion"
  | "validation" | "yaml_error"
  | "needs_crs" | "needs_layers" | "needs_responses" | "needs_config" | "requirements" | "template_unavailable" | "mismatch"
  | "internal"
  // client-side codes (never sent by the server)
  | "network" | "aborted" | "bad_response" | `http_${number}`;

export type ActionKind =
  | "run_job" | "open" | "resume" | "link_config" | "build_emulator" | "open_engine" | "rerun_exact" | "fetch_input";

/** One-click remedy the UI renders as a button (api.md §0.3). */
export type Action = {
  kind: ActionKind;
  label: string;
  method?: "POST" | "GET";
  path?: string;
  body?: unknown;
};

export type ApiErrorBody = {
  code: ErrorCode | string;
  message: string;
  detail?: Record<string, unknown>;
  action?: Action;
};

export type ErrorEnvelope = { error: ApiErrorBody };

export type ValidationErrorItem = { path: string; message: string; code: string };

// ---------------------------------------------------------------- pagination (§0.1)

export type Page<T> = { items: T[]; next_cursor: string | null };

// ---------------------------------------------------------------- jobs (§0.6)

export type JobStatus =
  | "queued" | "blocked" | "starting" | "running" | "cancelling"
  | "succeeded" | "failed" | "cancelled" | "interrupted";

export const ACTIVE_JOB_STATUSES: readonly JobStatus[] = ["queued", "blocked", "starting", "running", "cancelling"];
export const FINAL_JOB_STATUSES: readonly JobStatus[] = ["succeeded", "failed", "cancelled", "interrupted"];

export function isActiveStatus(s: JobStatus): boolean {
  return ACTIVE_JOB_STATUSES.includes(s);
}

export type JobLane = "heavy" | "medium" | "network" | "engine" | "none";

export type Job = {
  id: string;
  kind: string;
  lane: JobLane;
  executor: "process" | "engine" | "external";
  label: string;
  status: JobStatus;
  project_id: string | null;
  run_id: string | null;
  study_id: string | null;
  scenario_id: string | null;
  parent_job_id: string | null;
  after_job_id: string | null;
  priority: number;
  params: Record<string, unknown>;
  created_utc: string;
  started_utc: string | null;
  finished_utc: string | null;
  progress: number | null;
  eta_s: number | null;
  eta_lo: number | null;
  eta_hi: number | null;
  stage: string | null;
  current_path: string[] | null;
  exit_code: number | null;
  error: { type: string; message: string; traceback_tail?: string } | null;
  blocked: { reason: string; actions: Action[] } | null;
  result: Record<string, unknown> | null;
  peak_rss_mb: number | null;
  threads: number | null;
};

// ---------------------------------------------------------------- shared types (§1)

export type StageId = "S0" | "S1" | "S2_S3" | "baselines" | "cv_curve" | "S4" | "S5" | "climate" | "S6" | "S7" | "finish";

export const STAGE_IDS: readonly StageId[] = ["S0", "S1", "S2_S3", "baselines", "cv_curve", "S4", "S5", "climate", "S6", "S7", "finish"];

export type PlanNode = {
  id: StageId;
  label: string;
  state: "will_run" | "skipped" | "cached";
  reason: string | null;
  units: Record<string, number>;
  checkpoint_key: string | null;
  est_s: number | null;
  est_lo: number | null;
  est_hi: number | null;
};

export type Issue = {
  level: "error" | "warn" | "info";
  path: string;
  code: string;
  message: string;
  fix?: { path: string; value: unknown };
};

export type SelectionCrs = "EPSG:4326" | "run_xy_m";

export type SelectionSpec =
  | { kind: "all" }
  | { kind: "zones"; values: (number | string)[] }
  | { kind: "polygon"; crs: SelectionCrs; rings: [number, number][][] }
  | { kind: "circle"; crs: SelectionCrs; center: [number, number]; radius_m: number }
  | { kind: "rect"; crs: SelectionCrs; min: [number, number]; max: [number, number] }
  | { kind: "cells"; ids: (number | string)[] }
  | { kind: "blob"; blob_id: string }
  | { kind: "hex"; size_m: 250 | 500; keys: number[] }
  | { kind: "filter"; column: string; op: "<" | "<=" | ">" | ">=" | "==" | "between" | "in"; value: number | string | [number, number] | (number | string)[] }
  // Stored docs echo an unset optional field as null (`k: null, within: null`): treat null as absent.
  | { kind: "top"; column: string; frac?: number | null; k?: number | null; direction: "highest" | "lowest"; within?: SelectionSpec | null }
  | { kind: "buffer"; of: SelectionSpec; radius_m?: number | null; lever_range?: string | null }
  | { kind: "region"; id: string }
  | { op: "and" | "or" | "minus"; args: SelectionSpec[] }
  | { op: "not"; arg: SelectionSpec };

/** Base64 of a little-endian, LSB-first bit array of length n (api.md §0.4). */
export type Bitset = string;

/** Brush edit: base64 Int32 LE row indices and base64 Float32 LE values (api.md §0.4). */
export type SparseEdit = { idx: string; val: string };

export type LayerStats = {
  n: number;
  lo: number | null;
  hi: number | null;
  mean: number | null;
  p1: number | null;
  p2: number | null;
  p50: number | null;
  p98: number | null;
  p99: number | null;
};

export type LayerMeta = {
  key: string;
  group: string;
  label: string;
  unit: string;
  scale: "seq" | "div" | "cat";
  center: number | null;
  decimals: number;
  mult: number;
  zero_blank: boolean;
  labels: string[] | null;
  desc: string;
  sign_note: string | null;
  source: { file: string; column: string | null } | null;
  dtype: "float32" | "uint8";
  stats: LayerStats;
  /** A named colour set for cat layers ("heat": the NWS heat-index categories). */
  palette?: "heat" | null;
};

export type LayerGroup = { id: string; label: string; layers: LayerMeta[] };

export type GridMeta = {
  n: number;
  nx: number;
  ny: number;
  dx_m: number;
  x0_m: number;
  y0_m: number;
  crs: string | null;
  coord_scale: number;
  has_lonlat: boolean;
  /** [west, south, east, north] */
  bounds_lonlat: [number, number, number, number] | null;
  /** Each corner is [lat, lon]. */
  corners: { sw: [number, number]; se: [number, number]; nw: [number, number]; ne: [number, number] } | null;
  ids_kind: "int" | "str";
  zones: (number | string)[];
  n_folds: number | null;
  units: { target: string };
  background: number | null;
  etag: string;
};

export type Availability = "ready" | "partial" | "running" | "missing" | "stale";

export type OutputState = "present" | "stale" | "missing" | "writing" | "partial";

export type OutputEntry = {
  id: string;
  label: string;
  group: string;
  state: OutputState;
  produced_by: string;
  view: string;
  formats: string[];
  files: { relpath: string; bytes: number; mtime: string }[];
  action: Action | null;
};

export type Likely = {
  estimate: number;
  se: number | null;
  lo: number | null;
  hi: number | null;
  confidence: "confident_cools" | "confident_warms" | "could_be_zero" | "unknown";
  phrase: string;
};

/** The fixed run-tab vocabulary shared with the server (SPEC §3.2). */
export type RunTabId =
  | "overview" | "data" | "accuracy" | "distance" | "influence" | "response" | "causal" | "scenarios" | "climate"
  | "heat" | "budget" | "planner" | "lab" | "validation" | "uncertainty" | "provenance" | "track" | "map" | "docs" | "files";

export const RUN_TAB_IDS: readonly RunTabId[] = [
  "overview", "data", "accuracy", "distance", "influence", "response", "causal", "scenarios", "climate",
  "heat", "budget", "planner", "lab", "validation", "uncertainty", "provenance", "track", "map", "docs", "files",
];

export type RunTabStatus = {
  id: RunTabId;
  availability: Availability;
  missing: { output: string; produced_by: string; action: Action | null }[];
};

/** `GET /api/runs/{rid}/outputs` (api.md §6.1). */
export type RunOutputs = { outputs: OutputEntry[]; tabs: RunTabStatus[] };

// ---------------------------------------------------------------- shell-level resources

/** Project (api.md §5). The foundation shell needs it for the switcher and projectNav. */
export type Project = {
  id: string;
  slug: string;
  name: string;
  dir: string;
  config_path: string;
  template: string | null;
  demo: boolean;
  active_run_id: string | null;
  archived: boolean;
  created_utc: string;
  updated_utc: string;
  report: { title: string | null; place: string | null; area: string | null };
  headline_scenario: string | null;
  cost_model: Record<string, { per_unit: number }>;
  n_runs: number;
  last_run: { id: string; status: string; created_utc: string; r2: number | null; has_checkpoint?: boolean } | null;
  active_jobs: number;
  readiness_score: { done: number; total: number };
};

/** Project readiness spine row (api.md §5). */
export type ReadinessRow = {
  key: "data" | "columns" | "levers" | "roles" | "forcing" | "climate_table" | "people_layers" | "config_valid" | "runs" | "emulator" | "studies";
  label: string;
  state: "ok" | "warn" | "missing" | "n/a";
  detail: string;
  action: Action | null;
};

export type RunStatus =
  | "queued" | "running" | "complete" | "partial" | "failed" | "cancelled" | "interrupted" | "external_live" | "imported";

/** RunSummary (api.md §6). The shell needs it for the active-run chip and breadcrumb. */
export type RunSummary = {
  id: string;
  project_id: string | null;
  label: string | null;
  origin: "studio" | "imported" | "study_child" | "reproduction" | "external_live";
  status: RunStatus;
  mode: "fast" | "coarse" | "full" | "custom";
  coarse_m: number | null;
  created_utc: string | null;
  finished_utc: string | null;
  duration_s: number | null;
  n_points: number | null;
  r2: number | null;
  rmse: number | null;
  coverage: number | null;
  n_scenarios: number | null;
  checkpoint_bytes: number | null;
  has_emulator: boolean;
  studies: string[];
  git_commit: string | null;
  git_dirty: boolean | null;
  demo: boolean;
  pinned: boolean;
  parent_run_id: string | null;
  study_id: string | null;
  last_job_id: string | null;
};

/** Run statuses whose resources never change again (SPEC §12.4: never refetched). */
export function isFinishedRun(status: RunStatus | null | undefined): boolean {
  return status === "complete" || status === "failed" || status === "cancelled" || status === "imported";
}

export type EngineHostState = "absent" | "starting" | "ready" | "busy" | "recycling" | "error";
export type EngineState = "no_checkpoint" | "cold" | "queued" | "loading" | "ready" | "busy" | "incompatible" | "error";

/** `GET /api/health` (api.md §2). */
export type Health = {
  ok: boolean;
  version: string;
  workspace: string;
  pid: number;
  started_utc: string;
  active_jobs: number;
  engine: { state: EngineHostState };
};

/** `GET /api/meta` (api.md §2). */
export type Meta = {
  version: string;
  schema_version: 1;
  event_schema_version: 1;
  stages: { id: StageId; label: string; desc: string; checkpoint_key: string | null; manifest_timing_key: string | null }[];
  job_kinds: {
    kind: string; label: string; lane: string; executor: string; needs_run: boolean; needs_checkpoint: boolean;
    locks_run: boolean; network_hosts: string[]; long: boolean; params_schema: object;
  }[];
  output_catalog: {
    id: string; label: string; group: string; files: string[]; produced_by: string; view: string;
    formats: string[]; manifest_key: string | null;
  }[];
  palettes: { seqLight: string[]; seqDark: string[]; divLight: string[]; divDark: string[]; cat: string[] };
  unit_costs: Record<string, number>;
  modes: { id: "fast" | "coarse" | "full"; label: string; desc: string; default_coarse_m?: number }[];
  run_tab_outputs: Record<RunTabId, string[]>;
  edit_modes: string[];
  selection_kinds: string[];
  warning_codes: { code: string; label: string; view: string | null }[];
};

/** `POST /api/findings` body (api.md §11); ChartFrame's "Pin to Findings" sends it. */
export type FindingCreate = {
  project_id: string;
  run_id?: string | null;
  view: string;
  url_state: string;
  title: string;
  note_md?: string;
  snapshot: Record<string, unknown>;
};

export type Finding = {
  id: string;
  project_id: string;
  run_id: string | null;
  view: string;
  url_state: string;
  title: string;
  note_md: string;
  snapshot: Record<string, unknown>;
  image_url: string | null;
  position: number;
  created_utc: string;
  updated_utc: string;
};

// ---------------------------------------------------------------- per-job events (§17.2)

export type EventLevel = "debug" | "info" | "warning" | "error";

/** Envelope present on every per-job event line (SPEC §5.3), plus the byte `cursor`. */
export type EventEnvelope = {
  v: 1;
  type: string;
  seq: number;
  ts: number;
  t_rel: number;
  pid: number;
  job: string;
  lvl: EventLevel;
  span: string | null;
  parent: string | null;
  path: string[];
  ctx: Record<string, string | number | boolean>;
  /** Byte offset of the line's first byte; absent on transient `resource` events. */
  cursor?: number;
};

type Ev<T extends string, F> = EventEnvelope & { type: T } & F;

export type RunStartEvent = Ev<"run.start", {
  name: string; stages: string[]; fast: boolean; coarse: number | null; resume: boolean; cv_curve: boolean | null;
  config_sha256: string; code_sha256: string; run_meta: Record<string, unknown>;
}>;
export type RunDirEvent = Ev<"run.dir", { run_dir: string; fingerprint: string }>;
export type RunPlanEvent = Ev<"run.plan", { nodes: PlanNode[]; total_units: Record<string, number>; n_points: number | null }>;
export type StageStartEvent = Ev<"stage.start", { stage: StageId; label: string; est_s: number | null }>;
export type StageEndEvent = Ev<"stage.end", { stage: StageId; status: "ok" | "error" | "cancelled"; elapsed_s: number; summary: Record<string, unknown> }>;
export type StageSkipEvent = Ev<"stage.skip", { stage: StageId; reason: string }>;
export type TaskStartEvent = Ev<"task.start", { name: string; key: string | null; k: number | null; n: number | null; unit: string | null }>;
export type TaskEndEvent = Ev<"task.end", {
  name: string; key: string | null; k: number | null; n: number | null; unit: string | null;
  status: "ok" | "error" | "cancelled"; elapsed_s: number; metrics: Record<string, unknown>; error?: { type: string; message: string };
}>;
export type TickEvent = Ev<"tick", { k: number; n: number; unit: string; frac: number; label: string; [metric: string]: unknown }>;
export type MetricEvent = Ev<"metric", { name: string; value: number | string | boolean | null; unit: string | null; tags: Record<string, unknown> }>;
export type ArtifactEvent = Ev<"artifact", { path: string; role: string; bytes: number; stage: string | null }>;
export type CheckpointEvent = Ev<"checkpoint", {
  action: "saved" | "loaded" | "mismatch"; done: string[]; bytes: number | null; elapsed_s: number | null;
  fingerprint: string | null; changed_sections: string[] | null;
}>;
export type WarningEvent = Ev<"warning", { code: string; message: string; data: Record<string, unknown> }>;
export type LogEvent = Ev<"log", { logger: string; level: string; msg: string }>;
export type HeartbeatEvent = Ev<"heartbeat", { rss_mb: number; cpu_s: number; threads: number }>;
export type CancelRequestedEvent = Ev<"cancel.requested", { by: "user" | "kill" | "shutdown" }>;
export type CancelAckEvent = Ev<"cancel.ack", { at_path: string[] }>;
export type RunEndEvent = Ev<"run.end", {
  status: "succeeded" | "failed" | "cancelled"; elapsed_s: number; timings_s: Record<string, number>; done: string[];
  error: { type: string; message: string; traceback_tail: string } | null;
}>;
export type JobStatusEvent = Ev<"job.status", { status: JobStatus; exit_code: number | null; error: Record<string, unknown> | null }>;
export type JobResultEvent = Ev<"job.result", { result: Record<string, unknown> }>;
export type ResourceEvent = Ev<"resource", { rss_mb: number; cpu_pct: number; n_procs: number; threads: number }>;

export type JobEvent =
  | RunStartEvent | RunDirEvent | RunPlanEvent | StageStartEvent | StageEndEvent | StageSkipEvent
  | TaskStartEvent | TaskEndEvent | TickEvent | MetricEvent | ArtifactEvent | CheckpointEvent
  | WarningEvent | LogEvent | HeartbeatEvent | CancelRequestedEvent | CancelAckEvent | RunEndEvent
  | JobStatusEvent | JobResultEvent | ResourceEvent;

/** Stream-control frames on the per-job stream (never written to events.jsonl). */
export type JobStreamControl =
  | { type: "resync"; after: number }
  | { type: "end"; status: JobStatus };

/** `GET /api/jobs/{jid}/events` (polling fallback). */
export type JobEventsPage = { events: (JobEvent & { cursor: number })[]; next_cursor: number; eof: boolean };

// ---------------------------------------------------------------- global events (§17.1)

type GEv<T extends string, F> = { type: T; gseq: number; ts: number } & F;

export type GlobalEvent =
  | GEv<"job.created", { job: Job }>
  | GEv<"job.status", {
      job_id: string; status: JobStatus; prev_status: JobStatus | null; exit_code: number | null;
      error: Job["error"]; run_id: string | null; project_id: string | null; kind: string; label: string;
    }>
  | GEv<"job.progress", { job_id: string; frac: number | null; eta_s: number | null; eta_lo: number | null; eta_hi: number | null; stage: string | null; path_tail: string[] }>
  | GEv<"run.updated", { run_id: string; project_id: string | null; status: RunStatus; fields: string[] }>
  | GEv<"run.indexed", { run_id: string; project_id: string | null; origin: string }>
  | GEv<"output.written", { run_id: string; relpath: string; role: string; output_id: string | null; stage: string | null; bytes: number }>
  | GEv<"engine.status", { run_id: string | null; state: EngineState | EngineHostState; progress: number | null; rss_mb: number | null }>
  | GEv<"scenario.result", { scenario_id: string | null; result_id: string; run_id: string; kind: string }>
  | GEv<"study.updated", { study_id: string; kind: string; status: string; project_id: string | null; target_run_id: string | null }>
  | GEv<"storage.low", { free_bytes: number; threshold_bytes: number }>
  | GEv<"resync", { reason: "expired" | "overflow" }>
  | GEv<"server_shutdown", { stop_jobs: boolean }>;

export type GlobalEventType = GlobalEvent["type"];

export const GLOBAL_EVENT_TYPES: readonly GlobalEventType[] = [
  "job.created", "job.status", "job.progress", "run.updated", "run.indexed", "output.written",
  "engine.status", "scenario.result", "study.updated", "storage.low", "resync", "server_shutdown",
];

/** Every per-job event `type`, used to register EventSource listeners (§17.2). */
export const JOB_EVENT_TYPES: readonly string[] = [
  "run.start", "run.dir", "run.plan", "stage.start", "stage.end", "stage.skip", "task.start", "task.end",
  "tick", "metric", "artifact", "checkpoint", "warning", "log", "heartbeat", "cancel.requested", "cancel.ack",
  "run.end", "job.status", "job.result", "resource",
];
