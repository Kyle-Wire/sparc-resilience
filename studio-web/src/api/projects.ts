// Projects API (api.md §5, owner backend-projects) plus the launch endpoints the Launch page
// calls (api.md §6 `runs/plan`, `runs`, `runs/import`) and the few system reads the project
// pages need (meta, settings, storage). Types follow api.md exactly; api.md wins.
//
// Resource keys: `projects:list:<archived>`, `project:<pid>:config`, `project:<pid>:files`,
// `project:<pid>:history`, `meta`, `settings`, `storage`. The `project:<pid>…` keys are hit by
// `invalidate("project:<pid>")`, which the global stream fires on every job status change
// of the project (an input job that links into the config bumps its version).
import { api, getBin, putRaw, request } from "./client";
import type { OffsetEntry } from "./binary";
import { useResource } from "./resource";
import type { Action, GridMeta, Issue, Job, Meta, Page, PlanNode, Project, ReadinessRow, RunSummary, StageId } from "./types";

const enc = encodeURIComponent;
const base = (pid: string) => `/api/projects/${enc(pid)}`;

// ---------------------------------------------------------------- projects

export type ProjectTemplate = "blank" | "synthetic_demo" | "providence_example";

export type CreateProjectBody = {
  name: string;
  template: ProjectTemplate;
  options?: { seed?: number; n?: number; import_existing_runs?: boolean };
};

export type CreateProjectResult = { project: Project; imported_runs: string[]; warnings: string[] };

/** Study rows returned by an import (owner backend-studies-exports; only the id is relied on). */
export type ImportedStudy = { id: string; kind?: string; label?: string | null; [k: string]: unknown };

export type ImportProjectBody = {
  name?: string;
  config_path: string;
  copy_data?: boolean;
  run_dirs?: string[];
  study_dirs?: string[];
  trust_pickles?: boolean;
};

export type ImportProjectResult = { project: Project; runs: RunSummary[]; studies: ImportedStudy[]; warnings: string[] };

/** `GET /api/projects/{pid}` (same shape as the shell's ProjectDetail). */
export type ProjectDetail = { project: Project; readiness: ReadinessRow[]; runs: RunSummary[]; active_jobs: Job[]; config_version: number };

export type ProjectPatch = {
  name?: string;
  active_run_id?: string | null;
  headline_scenario?: string | null;
  cost_model?: Record<string, { per_unit: number }>;
  report?: Partial<Project["report"]>;
  archived?: boolean;
};

export function listProjects(archived = false, signal?: AbortSignal): Promise<Project[]> {
  return api.get<Project[]>("/api/projects", archived ? { archived: true } : undefined, signal);
}

export function createProject(body: CreateProjectBody): Promise<CreateProjectResult> {
  return api.post<CreateProjectResult>("/api/projects", body);
}

export function importProject(body: ImportProjectBody): Promise<ImportProjectResult> {
  return api.post<ImportProjectResult>("/api/projects/import", body);
}

export function getProject(pid: string, signal?: AbortSignal): Promise<ProjectDetail> {
  return api.get<ProjectDetail>(base(pid), undefined, signal);
}

export function patchProject(pid: string, patch: ProjectPatch): Promise<Project> {
  return api.patch<Project>(base(pid), patch);
}

export function deleteProject(pid: string, files = false): Promise<{ ok: true }> {
  return api.del<{ ok: true }>(base(pid), { files });
}

/** `POST /api/runs/import`: index a run folder in place (pickles need explicit trust). */
export type ImportRunBody = { dir: string; project_id?: string; config_path?: string; trust_pickles?: boolean };

export function importRun(body: ImportRunBody): Promise<RunSummary> {
  return api.post<RunSummary>("/api/runs/import", body);
}

/** Every project, with or without the archived ones (`?archived=true` lists them too). */
export function useProjectList(archived: boolean) {
  return useResource<Project[]>(`projects:list:${archived ? "all" : "active"}`, (s) => listProjects(archived, s), { tags: ["projects"] });
}

// ---------------------------------------------------------------- files (§5.1)

export type FileKind = "data" | "join" | "layers" | "features" | "forcing" | "climate" | "other";

export const FILE_KINDS: readonly FileKind[] = ["data", "join", "layers", "features", "forcing", "climate", "other"];

/** Upload suffix allowlist (SPEC §10.8). */
export const UPLOAD_SUFFIXES = [".csv", ".parquet", ".json", ".yml", ".yaml", ".tif", ".tiff", ".gpkg", ".geojson"] as const;

export type ProjectFile = { path: string; kind: FileKind | string; bytes: number; mtime: string; used_by: string[] };

export type InspectColumn = {
  name: string;
  dtype: string;
  n_null: number;
  min: number | null;
  max: number | null;
  n_unique: number | null;
  sample: unknown[];
};

export type FileInspect = { n_rows: number; n_rows_exact: boolean; columns: InspectColumn[]; preview: unknown[][] };

export type UploadResult = { path: string; bytes: number; kind: string; inspect: FileInspect | null };

export function uploadPath(pid: string, kind: FileKind, filename: string): string {
  return `${base(pid)}/files/${enc(kind)}/${enc(filename)}`;
}

/** Raw streamed PUT of one file; `overwrite` replaces a file of the same name (X-Overwrite: 1). */
export function uploadFile(
  pid: string,
  kind: FileKind,
  file: File,
  opts: { overwrite?: boolean; onProgress?: (sent: number, total: number) => void; signal?: AbortSignal } = {},
): Promise<UploadResult> {
  return putRaw<UploadResult>(uploadPath(pid, kind, file.name), file, {
    contentType: file.type || "application/octet-stream",
    headers: opts.overwrite ? { "X-Overwrite": "1" } : undefined,
    onProgress: opts.onProgress,
    signal: opts.signal,
  });
}

export function listFiles(pid: string, signal?: AbortSignal): Promise<ProjectFile[]> {
  return api.get<ProjectFile[]>(`${base(pid)}/files`, undefined, signal);
}

export function inspectFile(pid: string, path: string, rows?: number, signal?: AbortSignal): Promise<FileInspect> {
  return api.get<FileInspect>(`${base(pid)}/files/inspect`, { path, rows }, signal);
}

export function deleteFile(pid: string, path: string): Promise<{ ok: true }> {
  return api.del<{ ok: true }>(`${base(pid)}/files`, { path });
}

export function useProjectFiles(pid: string | null) {
  return useResource<ProjectFile[]>(pid ? `project:${pid}:files` : null, (s) => listFiles(pid!, s), { tags: pid ? [`project:${pid}`] : [] });
}

export function useFileInspect(pid: string | null, path: string | null) {
  return useResource<FileInspect>(pid && path ? `project:${pid}:inspect:${path}` : null, (s) => inspectFile(pid!, path!, undefined, s), {
    tags: pid ? [`project:${pid}:files`] : [],
  });
}

/** Whether an inspected column holds numbers (pandas/arrow dtype names). */
export function isNumericDtype(dtype: string): boolean {
  return /^(u?int|float|double|decimal|number|halffloat)/i.test(dtype.trim());
}

export type ColumnSuggestion = {
  target: string | null;
  id: string | null;
  x: string | null;
  y: string | null;
  zone: string | null;
  coord_unit: string | null;
  crs_guess: string | null;
  predictors: string[];
  roles: Record<string, string>;
  confidence: Record<string, number>;
};

export function suggestColumns(pid: string, path: string): Promise<ColumnSuggestion> {
  return api.post<ColumnSuggestion>(`${base(pid)}/columns/suggest`, { path });
}

// ---------------------------------------------------------------- data check (§5.2)

export type DataCheckFlag = { code: string; severity: "warn" | "info"; message: string };

export type DoseScaleEntry = { sd: number; doses: number[]; doses_in_sd: number[]; percentile_reached: number[] };

export type DataCheck = {
  n_points: number;
  n_input: number;
  n_dropped: number;
  clipped: Record<string, number>;
  grid: { nx: number; ny: number; cell_m: number; fill_fraction: number; collisions: number };
  background: { value: number; source: string };
  noise_floor: number | null;
  flags: DataCheckFlag[];
  dose_scale: Record<string, DoseScaleEntry>;
  coarse: Record<string, unknown> | null;
  extent_m: [number, number];
  columns_missing: string[];
  preview_token: string;
  /** Columns with a preview binary (additive field; absent on older servers). */
  preview_columns?: string[];
};

/** Run S0 inline on the saved config deep-merged with `config_patch` (unsaved form values). */
export function checkData(pid: string, config_patch?: Record<string, unknown>, signal?: AbortSignal): Promise<DataCheck> {
  return request<DataCheck>("POST", `${base(pid)}/data/check`, { body: config_patch ? { config_patch } : {}, signal });
}

export type PreviewGrid = { meta: GridMeta; buffer: ArrayBuffer; offsets: OffsetEntry[] };

/** Packed `ix, iy, lon, lat` of a data check, with its GridMeta in the X-SPARC-Grid header. */
export async function fetchPreviewGrid(pid: string, token: string, signal?: AbortSignal): Promise<PreviewGrid> {
  const r = await getBin(`${base(pid)}/data/preview/${enc(token)}/grid.bin`, "uint8", { signal });
  const head = r.headers.get("X-SPARC-Grid");
  if (!head) throw new Error("The preview grid came without its X-SPARC-Grid header");
  if (!r.offsets) throw new Error("The preview grid came without X-SPARC-Offsets");
  return { meta: JSON.parse(head) as GridMeta, buffer: r.buffer, offsets: r.offsets };
}

/** One Float32 column of a data check in row order. Tokens expire after 30 min. */
export async function fetchPreviewColumn(pid: string, token: string, column: string, signal?: AbortSignal): Promise<Float32Array> {
  const r = await getBin<Float32Array>(`${base(pid)}/data/preview/${enc(token)}/${enc(column)}.bin`, "float32", { signal });
  return r.data;
}

// ---------------------------------------------------------------- config (§5.3)

export type ConfigRaw = Record<string, unknown>;

export type ConfigDoc = {
  version: number;
  yaml: string;
  raw: ConfigRaw;
  effective: ConfigRaw;
  changed_from_defaults: { path: string; value: unknown; default: unknown }[];
  comments_preserved: false;
};

export type ConfigSaveResult = { version: number; issues: Issue[]; diff: string };

/** Top-level keys `PATCH …/config/sections/{section}` accepts. */
export const CONFIG_SECTIONS = [
  "name", "data", "predictors", "encodings", "qa", "actionable", "coupling", "mediators", "physics", "influence", "cv",
  "models", "stacker", "response", "scenarios", "joint_scenarios", "causal", "climate", "optimize", "planner", "report", "output",
] as const;
export type ConfigSection = (typeof CONFIG_SECTIONS)[number];

export type ValidateResult = { ok: boolean; issues: Issue[]; fast_overrides: Record<string, unknown>; coarse_preview: Record<string, unknown> | null };

export type ImpactRun = { run_id: string; label: string; checkpoint_done: string[]; changed_sections: string[]; refit_from: StageId | null; phrase: string };
export type ImpactResult = { changed_sections: string[]; runs: ImpactRun[] };

export type ConfigHistoryEntry = { version: number; saved_utc: string; note: string | null };

export type ConfigBody = { yaml: string; note?: string } | { raw: ConfigRaw; note?: string };

export function getConfig(pid: string, signal?: AbortSignal): Promise<ConfigDoc> {
  return api.get<ConfigDoc>(`${base(pid)}/config`, undefined, signal);
}

/** Save with optimistic concurrency: `If-Match: <version>`; a stale version is `409 conflict`. */
export function putConfig(pid: string, version: number, body: ConfigBody): Promise<ConfigSaveResult> {
  return api.put<ConfigSaveResult>(`${base(pid)}/config`, body, { headers: { "If-Match": String(version) } });
}

export function patchConfigSection(pid: string, section: ConfigSection, version: number, value: unknown, note?: string): Promise<ConfigSaveResult> {
  return api.patch<ConfigSaveResult>(`${base(pid)}/config/sections/${enc(section)}`, note ? { value, note } : { value }, {
    headers: { "If-Match": String(version) },
  });
}

/** Validate a YAML text, a raw object, or (no body) the saved config. */
export function validateConfig(pid: string, body: { yaml?: string; raw?: ConfigRaw } = {}, signal?: AbortSignal): Promise<ValidateResult> {
  return request<ValidateResult>("POST", `${base(pid)}/config/validate`, { body, signal });
}

export function configImpact(pid: string, body: { yaml?: string; raw?: ConfigRaw }): Promise<ImpactResult> {
  return api.post<ImpactResult>(`${base(pid)}/config/impact`, body);
}

export function configHistory(pid: string, signal?: AbortSignal): Promise<ConfigHistoryEntry[]> {
  return api.get<ConfigHistoryEntry[]>(`${base(pid)}/config/history`, undefined, signal);
}

export function configAtVersion(pid: string, version: number, signal?: AbortSignal): Promise<{ version: number; yaml: string }> {
  return api.get<{ version: number; yaml: string }>(`${base(pid)}/config/history/${version}`, undefined, signal);
}

export function useProjectConfig(pid: string | null) {
  return useResource<ConfigDoc>(pid ? `project:${pid}:config` : null, (s) => getConfig(pid!, s), { tags: pid ? [`project:${pid}`] : [] });
}

export function useConfigHistory(pid: string | null) {
  return useResource<ConfigHistoryEntry[]>(pid ? `project:${pid}:history` : null, (s) => configHistory(pid!, s), { tags: pid ? [`project:${pid}`] : [] });
}

/** The 409 conflict's server version (api.md §5.3 `detail.current_version`). */
export function conflictVersion(detail: Record<string, unknown> | null | undefined): number | null {
  const v = detail?.current_version;
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

// ---------------------------------------------------------------- link into config

export type LinkKind = "forcing" | "climate" | "layers" | "features_join" | "features_new_project";
export type LinkBody = { kind: LinkKind; path: string; apply?: boolean };
export type LinkResult = { yaml_diff: string; applied: boolean; version?: number; new_project_id?: string };

/** Preview (`apply: false`) or apply linking an input file into the config. */
export function linkInput(pid: string, body: LinkBody): Promise<LinkResult> {
  return api.post<LinkResult>(`${base(pid)}/link`, body);
}

// ---------------------------------------------------------------- launch (api.md §6)

export type RunMode = "fast" | "coarse" | "full";
export type RequestStage = "S0" | "S1" | "S2" | "S3" | "S4" | "S5" | "S6" | "S7";
export type ThenAction = "post.planner" | "post.emulator" | "post.uncertainty" | "post.writeup" | "post.baselines";

export const THEN_ACTIONS: readonly ThenAction[] = ["post.baselines", "post.planner", "post.emulator", "post.uncertainty", "post.writeup"];

export type RunPlanBody = {
  mode: RunMode;
  coarse_m?: number;
  stages?: RequestStage[];
  cv_curve?: boolean | null;
  threads?: number;
  resume_run_id?: string;
};

export type PreflightCheck = { check: string; ok: boolean; severity: "error" | "warn" | "info"; message: string; action: Action | null };

export type CheckpointInfo = {
  present: boolean;
  bytes: number | null;
  done: string[];
  saved_utc: string | null;
  fingerprint: string | null;
  matches_snapshot: { data: boolean; code: boolean; config: boolean } | null;
  changed_sections: string[];
  resumable: boolean;
  reuses: StageId[];
  saves_s: number | null;
  reason: string | null;
};

export type RunPlan = {
  nodes: PlanNode[];
  total_est_s: number;
  est_lo: number;
  est_hi: number;
  est_peak_rss_gb: number;
  est_disk_gb: number;
  network_hosts: string[];
  preflight: PreflightCheck[];
  issues: Issue[];
  resumable: CheckpointInfo | null;
  threads: number;
};

export type LaunchBody = RunPlanBody & { label?: string; notes?: string; then?: ThenAction[] };
export type LaunchResult = { run: RunSummary; job: Job; chain: Job[] };

export function planRun(pid: string, body: RunPlanBody, signal?: AbortSignal): Promise<RunPlan> {
  return request<RunPlan>("POST", `${base(pid)}/runs/plan`, { body, signal });
}

export function launchRun(pid: string, body: LaunchBody): Promise<LaunchResult> {
  return api.post<LaunchResult>(`${base(pid)}/runs`, body);
}

/** A preflight blocks Start when it failed with severity "error". */
export function blockingPreflight(plan: Pick<RunPlan, "preflight"> | null | undefined): PreflightCheck[] {
  return (plan?.preflight ?? []).filter((p) => !p.ok && p.severity === "error");
}

// ---------------------------------------------------------------- system reads used by these pages

/** `GET /api/settings` (the fields the project pages read). */
export type StudioSettings = {
  thread_budget: number;
  threads_heavy: number;
  engine_threads: number;
  offline: boolean;
  upload_max_gb: number;
  [k: string]: unknown;
};

export type StorageSummary = {
  workspace_bytes: number;
  free_bytes: number;
  cache: { name: string; bytes: number; mtime: string }[];
  runs: { run_id: string; label: string | null; project_id: string | null; outputs_bytes: number; checkpoint_bytes: number }[];
  jobs_bytes: number;
};

export function useMetaInfo() {
  return useResource<Meta>("meta", (s) => api.get<Meta>("/api/meta", undefined, s), { immutable: true });
}

export function useStudioSettings() {
  return useResource<StudioSettings>("settings", (s) => api.get<StudioSettings>("/api/settings", undefined, s), { tags: ["settings"] });
}

export function useStorageSummary() {
  return useResource<StorageSummary>("storage", (s) => api.get<StorageSummary>("/api/storage", undefined, s), { tags: ["storage"] });
}

/** Jobs of a project (newest first), e.g. the last input job of a kind. */
export function listProjectJobs(pid: string, kind?: string, limit = 5, signal?: AbortSignal): Promise<Page<Job>> {
  return api.get<Page<Job>>("/api/jobs", { project: pid, kind, limit }, signal);
}
