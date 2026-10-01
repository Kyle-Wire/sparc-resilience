// Typed endpoints and wire types of the tracking pages (api.md §2, §3, §5, §6, §9): jobs and
// their tracker, queue, timings, the project overview and Status Board, run lists, settings,
// storage and system. Field names follow api.md exactly; api.md wins on any disagreement.
//
// Run-group endpoints the tracking pages call (run lists, resume, storage deletes) are typed
// here as well: feature folders never import each other's modules.
import { api, apiUrl, type Query } from "./client";
import type {
  Action,
  Job,
  JobEvent,
  JobStatus,
  Page,
  PlanNode,
  RunSummary,
  StageId,
} from "./types";

const enc = encodeURIComponent;

// ---------------------------------------------------------------- tracker (§3)

export type StageUiState =
  | "planned" | "running" | "done" | "failed" | "cancelled" | "not_reached" | "skipped" | "cached" | "disabled" | "not_requested";

export type SnapshotStage = {
  state: StageUiState;
  reason: string | null;
  started_ts: number | null;
  ended_ts: number | null;
  elapsed_s: number | null;
  est_s: number | null;
  progress: number | null;
};

export type SpanKind = "run" | "stage" | "task";
export type SpanStatus = "running" | "ok" | "error" | "cancelled";

export type Span = {
  span_id: string;
  parent_id: string | null;
  kind: SpanKind;
  name: string;
  key: string | null;
  k: number | null;
  n: number | null;
  unit: string | null;
  status: SpanStatus;
  started_ts: number;
  ended_ts: number | null;
  elapsed_s: number | null;
  ctx: Record<string, unknown>;
  metrics: Record<string, unknown>;
};

export type MetricLatest = { value: number | string | boolean | null; unit: string | null; tags: Record<string, unknown>; ts: number | null };
export type SeriesPoint = { ts: number | null; value: number };

export type WarningRow = {
  code: string;
  lvl: string;
  message: string;
  count: number;
  stage: string | null;
  first_cursor: number;
  data: Record<string, unknown>;
};

export type ArtifactRow = { relpath: string; role: string; bytes: number; stage: string | null; ts: number | null };
export type CheckpointRow = { action: string; done: string[]; bytes: number | null; ts: number | null };
export type ResourceRow = { ts: number; rss_mb: number; cpu_pct: number; n_procs: number; threads?: number | null };
export type HeartbeatGap = { from_ts: number; to_ts: number };

export type ChildRow = {
  job_id: string | null;
  run_id: string | null;
  key: string;
  label: string;
  status: string;
  progress: number | null;
  metrics: Record<string, unknown>;
};

/**
 * `GET /api/jobs/{jid}/tracker` (api.md §3). `projection` is an optional additive field: the
 * server's raw projection state (`sparc/studio/jobs/tracker.py`), requested with
 * `?projection=1`. With it the client resumes exactly where the server stands; without it
 * the client rebuilds its counters from the fields above (stores/tracker.ts `fromSnapshot`).
 */
export type TrackerSnapshot = {
  job: Job;
  cursor: number;
  plan: PlanNode[] | null;
  stages: Partial<Record<StageId, SnapshotStage>> | null;
  spans: Span[];
  metrics_latest: Record<string, MetricLatest>;
  metric_series: Record<string, SeriesPoint[]>;
  warnings: WarningRow[];
  artifacts: ArtifactRow[];
  checkpoints: CheckpointRow[];
  resources: ResourceRow[];
  heartbeat_gaps: HeartbeatGap[];
  children: ChildRow[];
  /** events.jsonl passed 200 MB: debug lines stay on disk only (SPEC §5.6); the UI shows a note. */
  log_capped?: boolean;
  projection?: unknown;
};

export type LogLine = { cursor: number; ts: number; level: string; logger: string; msg: string; path: string[] };
export type LogsPage = { lines: LogLine[]; next_cursor: number };
export type MetricPoint = { cursor: number; ts: number; value: number | null; value_text: string | null; tags: Record<string, unknown> };
export type EventsPage = { events: (JobEvent & { cursor: number })[]; next_cursor: number; eof: boolean };

export type JobListQuery = {
  status?: string | string[];
  kind?: string | string[];
  project?: string;
  run?: string;
  study?: string;
  parent?: string;
  limit?: number;
  cursor?: string | null;
};

export function listJobs(q: JobListQuery = {}, signal?: AbortSignal): Promise<Page<Job>> {
  return api.get<Page<Job>>("/api/jobs", q as Query, signal);
}

export function getJob(jid: string, signal?: AbortSignal): Promise<Job> {
  return api.get<Job>(`/api/jobs/${enc(jid)}`, undefined, signal);
}

export function getTracker(jid: string, signal?: AbortSignal): Promise<TrackerSnapshot> {
  return api.get<TrackerSnapshot>(`/api/jobs/${enc(jid)}/tracker`, { projection: 1 }, signal);
}

export function getSpans(jid: string, q: { under?: string; max_depth?: number } = {}, signal?: AbortSignal): Promise<Span[]> {
  return api.get<Span[]>(`/api/jobs/${enc(jid)}/spans`, q, signal);
}

export function getEvents(jid: string, q: { after?: number | null; limit?: number; types?: string[]; min_lvl?: string } = {}, signal?: AbortSignal): Promise<EventsPage> {
  return api.get<EventsPage>(`/api/jobs/${enc(jid)}/events`, { after: q.after ?? undefined, limit: q.limit, types: q.types, min_lvl: q.min_lvl }, signal);
}

export type LogsQuery = { after?: number | null; level?: string; logger?: string; stage?: string; q?: string; limit?: number };

export function getLogs(jid: string, q: LogsQuery = {}, signal?: AbortSignal): Promise<LogsPage> {
  return api.get<LogsPage>(`/api/jobs/${enc(jid)}/logs`, { ...q, after: q.after ?? undefined }, signal);
}

export type RawStream = "events" | "stdout" | "stderr" | "text";

/** Download URL of `events.jsonl`, `stdout.log`, `stderr.log` or a rendered `log.txt`. */
export function logsRawUrl(jid: string, stream: RawStream): string {
  return apiUrl(`/api/jobs/${enc(jid)}/logs/raw`, { stream });
}

export function getMetrics(jid: string, names: string[], signal?: AbortSignal): Promise<Record<string, MetricPoint[]>> {
  return api.get<Record<string, MetricPoint[]>>(`/api/jobs/${enc(jid)}/metrics`, { names }, signal);
}

export function getWarnings(jid: string, signal?: AbortSignal): Promise<WarningRow[]> {
  return api.get<WarningRow[]>(`/api/jobs/${enc(jid)}/warnings`, undefined, signal);
}

export function getResources(jid: string, since_ts?: number, signal?: AbortSignal): Promise<ResourceRow[]> {
  return api.get<ResourceRow[]>(`/api/jobs/${enc(jid)}/resources`, { since_ts }, signal);
}

export function cancelJob(jid: string): Promise<Job> {
  return api.post<Job>(`/api/jobs/${enc(jid)}/cancel`);
}

/** Force stop (SIGKILL to the process group); refused before the grace period unless `force_now`. */
export function killJob(jid: string, force_now = false): Promise<Job> {
  return api.post<Job>(`/api/jobs/${enc(jid)}/kill`, force_now ? { force_now: true } : {});
}

export function retryJob(jid: string): Promise<Job> {
  return api.post<Job>(`/api/jobs/${enc(jid)}/retry`);
}

export function setJobPriority(jid: string, priority: number): Promise<Job> {
  return api.patch<Job>(`/api/jobs/${enc(jid)}`, { priority });
}

export function deleteJob(jid: string, files = false): Promise<{ ok: true }> {
  return api.del<{ ok: true }>(`/api/jobs/${enc(jid)}`, { files: files ? "true" : undefined });
}

// ---------------------------------------------------------------- queue and timings (§3)

export type LaneState = { lane: string; slots: number; running: string[]; queued: string[] };
export type QueueState = { paused: boolean; lanes: LaneState[] };

export function getQueue(signal?: AbortSignal): Promise<QueueState> {
  return api.get<QueueState>("/api/queue", undefined, signal);
}

export function pauseQueue(): Promise<QueueState> {
  return api.post<QueueState>("/api/queue/pause");
}

export function resumeQueue(): Promise<QueueState> {
  return api.post<QueueState>("/api/queue/resume");
}

export type UnitRate = { unit: string; median_s: number; p25_s: number; p75_s: number; n: number };
export type StageHistoryRow = {
  run_id: string;
  label: string | null;
  git_commit: string | null;
  stage: string;
  seconds: number;
  n_points: number | null;
  mode: string | null;
  threads: number | null;
};
export type Timings = { unit_rates: UnitRate[]; stage_history: StageHistoryRow[] };

export function getTimings(q: { unit?: string; stage?: string; project?: string; limit?: number } = {}, signal?: AbortSignal): Promise<Timings> {
  return api.get<Timings>("/api/timings", q, signal);
}

// ---------------------------------------------------------------- runs (§6)

export type RunListQuery = {
  project?: string;
  status?: string;
  mode?: string;
  origin?: string;
  q?: string;
  sort?: "created_desc" | "duration" | "r2";
  limit?: number;
  cursor?: string | null;
};

/** `GET /api/runs` or, with `pid`, `GET /api/projects/{pid}/runs`. */
export function listRuns(pid: string | null, q: RunListQuery = {}, signal?: AbortSignal): Promise<Page<RunSummary>> {
  const path = pid ? `/api/projects/${enc(pid)}/runs` : "/api/runs";
  return api.get<Page<RunSummary>>(path, q as Query, signal);
}

export function resumeRun(rid: string, body: { threads?: number; use_current_config?: boolean } = {}): Promise<Job> {
  return api.post<Job>(`/api/runs/${enc(rid)}/resume`, body);
}

export function rerunRun(rid: string, body: { use_current_config?: boolean; label?: string } = {}): Promise<{ run: RunSummary; job: Job }> {
  return api.post<{ run: RunSummary; job: Job }>(`/api/runs/${enc(rid)}/rerun`, body);
}

export function deleteRunData(rid: string, what: "checkpoint" | "outputs" | "all", force_files = false): Promise<{ freed_bytes: number }> {
  return api.del<{ freed_bytes: number }>(`/api/runs/${enc(rid)}`, { what, force_files: force_files ? "true" : undefined });
}

/** The slice of `GET /api/runs/{rid}` (RunDetail) Mission Control and the run tracker read. */
export type RunDetailLite = {
  run: RunSummary;
  header: { name: string; run_dir: string; demo: boolean; fast: boolean; coarse_m: number | null } & Record<string, unknown>;
  launch: Record<string, unknown> | null;
  checkpoint: { present: boolean; bytes: number | null; done: string[]; saved_utc: string | null; resumable: boolean; reuses: string[]; saves_s: number | null; reason: string | null } | null;
  jobs: Job[];
};

export function getRunDetail(rid: string, signal?: AbortSignal): Promise<RunDetailLite> {
  return api.get<RunDetailLite>(`/api/runs/${enc(rid)}`, undefined, signal);
}

export type RunConfig = {
  effective: Record<string, unknown>;
  raw: Record<string, unknown>;
  yaml: string;
  source: "launch" | "manifest" | "import";
  config_dir: string;
  vs_project_diff: { path: string; run: unknown; project: unknown }[];
  vs_defaults: { path: string; value: unknown; default: unknown }[];
};

export type RunProvenance = {
  provenance: Record<string, unknown> | null;
  git: Record<string, unknown> | null;
  platform: Record<string, unknown> | null;
  hashes: Record<string, string>;
  environment: string[];
  launch: Record<string, unknown> | null;
};

export function getRunConfig(rid: string, signal?: AbortSignal): Promise<RunConfig> {
  return api.get<RunConfig>(`/api/runs/${enc(rid)}/config`, undefined, signal);
}

export function getRunProvenance(rid: string, signal?: AbortSignal): Promise<RunProvenance> {
  return api.get<RunProvenance>(`/api/runs/${enc(rid)}/provenance`, undefined, signal);
}

// ---------------------------------------------------------------- project overview (§5, §6)

export type StatusCellState = "done" | "cached" | "running" | "failed" | "skipped" | "stale" | "not_run" | "disabled";

export type StatusCell = {
  state: StatusCellState;
  seconds: number | null;
  progress: number | null;
  reason: string | null;
  job_id: string | null;
  study_id: string | null;
  action: Action | null;
};

export type StatusBoardColumn = { id: string; label: string; group: "stage" | "post" | "study" };

/** `GET /api/projects/{pid}/status-board`. */
export type StatusBoard = {
  columns: StatusBoardColumn[];
  rows: { run: RunSummary; cells: Record<string, StatusCell> }[];
};

export function getStatusBoard(pid: string, signal?: AbortSignal): Promise<StatusBoard> {
  return api.get<StatusBoard>(`/api/projects/${enc(pid)}/status-board`, undefined, signal);
}

/** `GET /api/studies/{stid}/view` (api.md §9), read by the study child matrices. */
export type StudyView = {
  // placebo
  rows?: unknown;
  n_pass_model?: number;
  n_pass_causal?: number;
  children?: { kind: string; run_id: string | null; status: string; verdict: string | null }[];
  // simcheck
  design?: Record<string, number>;
  grid?: { generator: string; seed: number; status: "pending" | "running" | "done" | "gate_fail" | "error"; share: number | null; ci_covers: boolean | null; causal_covers: boolean | null; seconds: number | null; gate_attempt: number | null }[];
  eta_s?: number | null;
  // multiverse
  variants?: { name: string; label: string; status: string; r2: number | null; rmse: number | null; seconds: number | null; run_id: string | null }[];
  effects?: Record<string, unknown>;
  // reproduce
  pass?: boolean;
  checks?: { check: string; ok: boolean; hard: boolean; detail: string }[];
  [key: string]: unknown;
};

export function getStudyView(stid: string, signal?: AbortSignal): Promise<StudyView> {
  return api.get<StudyView>(`/api/studies/${enc(stid)}/view`, undefined, signal);
}

// ---------------------------------------------------------------- settings, system, storage (§2)

export type Settings = {
  thread_budget: number;
  threads_heavy: number;
  engine_threads: number;
  heavy_slots: number;
  medium_slots: number;
  network_slots: number;
  engine_max_runs: number;
  engine_mem_budget_gb: number;
  engine_idle_min: number;
  auto_uncertainty: boolean;
  watch_roots: string[];
  upload_max_gb: number;
  keep_job_logs_days: number | null;
  offline: boolean;
  notifications: boolean;
  basemap_url: string | null;
};

export function getSettings(signal?: AbortSignal): Promise<Settings> {
  return api.get<Settings>("/api/settings", undefined, signal);
}

/** `PUT /api/settings` takes a partial Settings and returns the full Settings. */
export function putSettings(patch: Partial<Settings>): Promise<Settings> {
  return api.put<Settings>("/api/settings", patch);
}

export type SystemInfo = {
  cpu_count: number;
  cpu_model: string;
  mem_total_gb: number;
  mem_available_gb: number;
  disk_free_gb: number;
  workspace: string;
  workspace_bytes: number;
  host_id: string;
  versions: { python: string; sparc: string; numpy: string; pandas: string; torch: string | null; fastapi: string };
  web_build: { src_sha256: string; vite: string; react: string } | null;
};

export function getSystem(signal?: AbortSignal): Promise<SystemInfo> {
  return api.get<SystemInfo>("/api/system", undefined, signal);
}

export type NetcheckResult = { results: { host: string; ok: boolean; ms: number | null; error: string | null }[] };

export function netcheck(hosts?: string[]): Promise<NetcheckResult> {
  return api.post<NetcheckResult>("/api/system/netcheck", hosts && hosts.length ? { hosts } : {});
}

export type StorageInfo = {
  workspace_bytes: number;
  free_bytes: number;
  cache: { name: string; bytes: number; mtime: string }[];
  runs: { run_id: string; label: string | null; project_id: string | null; outputs_bytes: number; checkpoint_bytes: number }[];
  studies: { study_id: string; bytes: number }[];
  jobs_bytes: number;
};

export function getStorage(signal?: AbortSignal): Promise<StorageInfo> {
  return api.get<StorageInfo>("/api/storage", undefined, signal);
}

export function deleteCache(name: string): Promise<{ freed_bytes: number }> {
  return api.del<{ freed_bytes: number }>(`/api/storage/cache/${enc(name)}`);
}

// ---------------------------------------------------------------- helpers

/** Statuses after which a job never changes again. */
export function isFinalStatus(s: JobStatus | null | undefined): boolean {
  return s === "succeeded" || s === "failed" || s === "cancelled" || s === "interrupted";
}
