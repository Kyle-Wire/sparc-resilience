// Scenario Lab endpoints and types (api.md §7, §10 pack exports; SPEC §7). Owned by the Lab
// (frontend-lab); the server side is backend-engine. Field names follow api.md exactly.
//
// Resource tags: run-scoped Lab resources carry `run:<rid>:lab` (refreshed by any job status
// change on the run, since `run:<rid>` invalidates its sub-tags) and `run:<rid>:results`
// (refreshed by `scenario.result`); scenarios carry `scenario:<sid>`; the engine status
// carries `run:<rid>:engine` (refreshed by `engine.status`).
import { api, apiUrl, ensureDevAuth, errorFromResponse, getBin, putRaw, ApiError } from "./client";
import { decodeBitset, parseOffsets, unpack, unpackBits } from "./binary";
import { useResource } from "./resource";
import type { Action, Bitset, Job, Likely, Page, RunSummary, SelectionSpec, SparseEdit } from "./types";

const enc = encodeURIComponent;
const runBase = (rid: string) => `/api/runs/${enc(rid)}`;

// ---------------------------------------------------------------- scenario docs (§7.1)

export type EditMode = "add" | "set" | "scale" | "floor" | "ceiling" | "fill_headroom" | "to_percentile" | "per_cell";

export const EDIT_MODES: readonly EditMode[] = ["add", "set", "scale", "floor", "ceiling", "fill_headroom", "to_percentile", "per_cell"];

export type Edit = {
  lever: string;
  mode: EditMode;
  amount?: number;
  percentile?: number;
  paved_share?: number;
  /** "plan:<plid>" | "blob:<blob_id>" | "csv:<project-relative path>" */
  per_cell_ref?: string;
  /** Default {kind: "all"}. */
  where?: SelectionSpec;
  label?: string;
};

export type ScenarioOptions = { clip_to_support?: boolean; mediators?: boolean; expert?: boolean };

export type ScenarioDoc = {
  name: string;
  notes?: string;
  tags?: string[];
  anchor_run_id?: string | null;
  edits: Edit[];
  regions?: Record<string, SelectionSpec>;
  costs?: Record<string, { per_unit: number }>;
  options?: ScenarioOptions;
};

export type ScenarioStatus = "draft" | "previewed" | "exact" | "stale" | "archived";

export type ResultKind = "exact" | "configured" | "plan" | "sweep_point";

export type ResultSummary = {
  id: string;
  scenario_id: string | null;
  run_id: string;
  kind: ResultKind;
  created_utc: string;
  stale: boolean;
  has_folds: boolean;
  city: Likely;
  edited: Likely | null;
  frac_extrapolated_edited: number | null;
  job_id: string | null;
};

export type Scenario = {
  id: string;
  project_id: string;
  revision: number;
  parent_id: string | null;
  children: string[];
  doc: ScenarioDoc;
  content_hash: string;
  status: ScenarioStatus;
  created_utc: string;
  updated_utc: string;
  results: ResultSummary[];
};

export type ScenarioSummary = Pick<Scenario, "id" | "revision" | "parent_id" | "status" | "created_utc" | "updated_utc"> & {
  name: string;
  tags: string[];
  latest: ResultSummary | null;
};

export type ItemRef = { kind: "result"; id: string } | { kind: "configured"; slug: string } | { kind: "plan"; id: string } | { kind: "baseline" };

// ---------------------------------------------------------------- engine (§7.2)

export type EngineState = "no_checkpoint" | "cold" | "queued" | "loading" | "ready" | "busy" | "incompatible" | "error";

export type RunEngineStatus = {
  state: EngineState;
  progress: number | null;
  step: string | null;
  rss_mb: number | null;
  est_rss_mb: number;
  code_match: boolean | null;
  loaded_utc: string | null;
  last_used_utc: string | null;
  error: { type: string; message: string } | null;
  job_id: string | null;
  action: Action | null;
};

export type EngineHostStatus = {
  state: "absent" | "starting" | "ready" | "busy" | "recycling" | "error";
  pid: number | null;
  rss_mb: number | null;
  budget_gb: number;
  max_runs: number;
  runs: { run_id: string; est_rss_mb: number; loaded_utc: string; last_used_utc: string }[];
  busy_job_id: string | null;
  queue: string[];
};

export function getRunEngine(rid: string, signal?: AbortSignal): Promise<RunEngineStatus> {
  return api.get<RunEngineStatus>(`${runBase(rid)}/engine`, undefined, signal);
}

export function useRunEngine(rid: string | null) {
  return useResource<RunEngineStatus>(rid ? `run:${rid}:engine` : null, (s) => getRunEngine(rid!, s), { tags: rid ? [`run:${rid}:engine`, "engine"] : [] });
}

/** `POST /engine/open`: 202 Job (engine.open), or 200 with the status when already loaded. */
export async function openEngine(rid: string): Promise<{ job: Job | null; status: RunEngineStatus | null }> {
  const r = await api.post<Job | RunEngineStatus>(`${runBase(rid)}/engine/open`, {});
  if (r && typeof r === "object" && "kind" in r && "id" in r) return { job: r as Job, status: null };
  return { job: null, status: r as RunEngineStatus };
}

export function closeEngine(rid: string): Promise<RunEngineStatus> {
  return api.del<RunEngineStatus>(`${runBase(rid)}/engine`);
}

/**
 * Mark an imported run's checkpoint as trusted by re-registering its folder with
 * `trust_pickles` (api.md §6 `POST /api/runs/import`; SPEC §10.8 pickles).
 */
export function trustRunCheckpoint(dir: string, projectId: string | null, configPath?: string | null): Promise<RunSummary> {
  return api.post<RunSummary>("/api/runs/import", { dir, project_id: projectId, ...(configPath ? { config_path: configPath } : {}), trust_pickles: true });
}

// ---------------------------------------------------------------- levers, emulator, preview, compile (§7.3)

export type EmulatorTrust = "good" | "rough" | "none";

export type Lever = {
  var: string;
  label: string;
  unit: string;
  min: number | null;
  max: number | null;
  direction: "increase" | "decrease";
  doses: number[];
  cost_per_unit: number;
  design_dose: number | null;
  sd: number | null;
  headroom_available: boolean;
  role: string | null;
  mediator_children: string[];
  emulator: { available: boolean; trust: EmulatorTrust; patch_pass_rate: number | null; uniform_rel_err: number | null };
};

export function useLevers(rid: string | null) {
  return useResource<Lever[]>(rid ? `run:${rid}:lab:levers` : null, (s) => api.get<Lever[]>(`${runBase(rid!)}/levers`, undefined, s), { tags: rid ? [`run:${rid}:lab`] : [] });
}

export type EmulatorInfo = {
  present: boolean;
  kernel_cells: number | null;
  levers: Record<string, { design_dose: number | null; bounds: [number, number]; direction: string; trust: EmulatorTrust; validation: Record<string, unknown> }>;
  action: Action | null;
};

export function useEmulator(rid: string | null) {
  return useResource<EmulatorInfo>(rid ? `run:${rid}:lab:emulator` : null, (s) => api.get<EmulatorInfo>(`${runBase(rid!)}/emulator`, undefined, s), {
    tags: rid ? [`run:${rid}:lab`, `run:${rid}:outputs`] : [],
  });
}

export type PreviewSummary = {
  mean: number;
  edited_mean: number;
  n_edited: number;
  outside_share: number;
  trust: EmulatorTrust;
  hatched: boolean;
  reasons: string[];
  request_seq: number;
};

export type PreviewRequest = {
  edits: Edit[];
  brush?: Record<string, SparseEdit>;
  options?: ScenarioOptions;
  request_seq: number;
  /** As built (backend-engine): marks the draft scenario "previewed". */
  scenario_id?: string;
};

export type PreviewResult = {
  /** ΔT per row in target units (negative = cooler). */
  delta: Float32Array;
  /** 0/1 per row: cells the edits touch. */
  edited: Uint8Array;
  summary: PreviewSummary;
};

/** Decode a packed preview body (api.md §7.3): `delta` float32[n] and `edited` LSB-first bitset bytes. */
export function decodePreview(buffer: ArrayBuffer, offsetsHeader: string, summaryHeader: string | null, sentSeq: number): PreviewResult {
  const arrays = unpack(buffer, parseOffsets(offsetsHeader));
  const delta = arrays.delta as Float32Array | undefined;
  const editedBytes = arrays.edited as Uint8Array | undefined;
  if (!delta) throw new ApiError(200, "bad_response", "The preview response has no delta array");
  const n = delta.length;
  const edited = editedBytes ? unpackBits(editedBytes, n) : new Uint8Array(n);
  let summary: PreviewSummary = { mean: NaN, edited_mean: NaN, n_edited: 0, outside_share: 0, trust: "none", hatched: false, reasons: [], request_seq: sentSeq };
  if (summaryHeader) {
    try {
      summary = { ...summary, ...(JSON.parse(summaryHeader) as Partial<PreviewSummary>) };
    } catch {
      throw new ApiError(200, "bad_response", "X-SPARC-Summary is not valid JSON");
    }
  }
  if (typeof summary.request_seq !== "number") summary.request_seq = sentSeq;
  return { delta, edited, summary };
}

/**
 * `POST /api/runs/{rid}/preview` → packed binary (delta + edited bitset) with the summary in
 * X-SPARC-Summary. Errors (404 no_emulator, 409 superseded, 422) arrive as JSON envelopes.
 */
export async function postPreview(rid: string, body: PreviewRequest, signal?: AbortSignal): Promise<PreviewResult> {
  const url = `${runBase(rid)}/preview`;
  const send = () =>
    fetch(url, {
      method: "POST",
      headers: { Accept: "application/octet-stream", "Content-Type": "application/json" },
      body: JSON.stringify(body),
      credentials: "same-origin",
      signal,
    });
  let res: Response;
  try {
    res = await send();
    if (res.status === 401 && (await ensureDevAuth(true))) res = await send();
  } catch (e) {
    if ((e as { name?: unknown } | null)?.name === "AbortError") throw new ApiError(0, "aborted", "Request cancelled");
    throw new ApiError(0, "network", "Studio server is not reachable");
  }
  if (!res.ok) {
    const text = await res.text();
    let parsed: unknown = text;
    try {
      parsed = text ? JSON.parse(text) : null;
    } catch {
      /* keep the text */
    }
    throw errorFromResponse(res.status, parsed, res.statusText);
  }
  const offsets = res.headers.get("X-SPARC-Offsets");
  if (!offsets) throw new ApiError(res.status, "bad_response", "The preview response came without X-SPARC-Offsets");
  return decodePreview(await res.arrayBuffer(), offsets, res.headers.get("X-SPARC-Summary"), body.request_seq);
}

export type CompileWarning = { code: string; message: string; edit_index: number | null; blocking: boolean };

export type CompileLever = {
  n_cells: number;
  mean_requested: number;
  total_requested: number;
  predicted_mean_realised: number;
  clipped_share: number;
  est_cost: number;
};

export type CompileResult = {
  content_hash: string;
  portable: boolean;
  levers: Record<string, CompileLever>;
  union_cells: number;
  people: number | null;
  warnings: CompileWarning[];
  est_exact_s: number;
  emulator: { usable: boolean; hatched: boolean; reasons: string[] };
};

export function compileScenario(rid: string, scenario: ScenarioDoc, signal?: AbortSignal): Promise<CompileResult> {
  return api.post<CompileResult>(`${runBase(rid)}/compile`, { scenario }, { signal });
}

// ---------------------------------------------------------------- templates and library (§7.4)

export type TemplateRequirement = "layers" | "canopy_role" | "impervious_role" | "albedo_role" | "crs";

export type ScenarioTemplate = {
  id: string;
  label: string;
  desc: string;
  /** JSON schema of the template's parameters (`properties` with type, title, default, enum, minimum, maximum). */
  params_schema: { properties?: Record<string, TemplateParamSchema>; required?: string[] } & Record<string, unknown>;
  requires: TemplateRequirement[];
};

export type TemplateParamSchema = {
  type?: "number" | "integer" | "string" | "boolean" | "array";
  title?: string;
  description?: string;
  default?: unknown;
  enum?: (string | number)[];
  minimum?: number;
  maximum?: number;
};

export function useTemplates() {
  return useResource<ScenarioTemplate[]>("lab:templates", (s) => api.get<ScenarioTemplate[]>("/api/scenario-templates", undefined, s), { tags: ["lab:templates"] });
}

export function scenarioFromTemplate(pid: string, template: string, params: Record<string, unknown>, runId: string): Promise<Scenario> {
  return api.post<Scenario>(`/api/projects/${enc(pid)}/scenarios/from-template`, { template, params, run_id: runId });
}

export type ScenarioQuery = { run?: string; tag?: string; status?: string; q?: string; archived?: boolean };

export function listScenarios(pid: string, query: ScenarioQuery = {}, signal?: AbortSignal): Promise<ScenarioSummary[]> {
  return api.get<ScenarioSummary[]>(`/api/projects/${enc(pid)}/scenarios`, { ...query, archived: query.archived ? true : undefined }, signal);
}

export function useScenarios(pid: string | null, query: ScenarioQuery = {}, rid?: string) {
  const qk = JSON.stringify(query);
  return useResource<ScenarioSummary[]>(pid ? `project:${pid}:scenarios:${qk}` : null, (s) => listScenarios(pid!, query, s), {
    tags: pid ? [`project:${pid}:scenarios`, "scenarios", ...(rid ? [`run:${rid}:results`] : [])] : [],
    keepPrevious: true,
  });
}

export function createScenario(pid: string, doc: ScenarioDoc): Promise<Scenario> {
  return api.post<Scenario>(`/api/projects/${enc(pid)}/scenarios`, { doc });
}

export function getScenario(sid: string, signal?: AbortSignal): Promise<Scenario> {
  return api.get<Scenario>(`/api/scenarios/${enc(sid)}`, undefined, signal);
}

export function useScenario(sid: string | null) {
  return useResource<Scenario>(sid ? `scenario:${sid}` : null, (s) => getScenario(sid!, s), { tags: sid ? [`scenario:${sid}`, "scenarios"] : [] });
}

export type ScenarioPatch = { doc?: ScenarioDoc; name?: string; tags?: string[]; notes?: string; archived?: boolean };

/** `PATCH /api/scenarios/{sid}`; `409 conflict_revision` when an exact result exists (fork instead). */
export function patchScenario(sid: string, patch: ScenarioPatch): Promise<Scenario> {
  return api.patch<Scenario>(`/api/scenarios/${enc(sid)}`, patch);
}

export function forkScenario(sid: string, body: { name?: string; doc?: ScenarioDoc } = {}): Promise<Scenario> {
  return api.post<Scenario>(`/api/scenarios/${enc(sid)}/fork`, body);
}

export function deleteScenario(sid: string, opts: { results?: boolean; force?: boolean } = {}): Promise<{ ok: true }> {
  return api.del<{ ok: true }>(`/api/scenarios/${enc(sid)}`, { results: opts.results ? true : undefined, force: opts.force ? true : undefined });
}

/** `POST /api/scenarios/{sid}/run`: a cache hit returns the cached summary, else the engine.scenario job. */
export async function runScenarioExact(sid: string, runId: string, force = false): Promise<{ cached: ResultSummary | null; job: Job | null }> {
  const r = await api.post<{ cached?: ResultSummary; job?: Job }>(`/api/scenarios/${enc(sid)}/run`, { run_id: runId, force });
  return { cached: r?.cached ?? null, job: r?.job ?? null };
}

export async function runBatch(runId: string, scenarioIds: string[]): Promise<Job> {
  return (await api.post<{ job: Job }>("/api/scenarios/run-batch", { run_id: runId, scenario_ids: scenarioIds })).job;
}

export function makeLadder(sid: string, editIndex: number, amounts: number[], runId?: string): Promise<{ scenarios: Scenario[]; job: Job | null }> {
  return api.post<{ scenarios: Scenario[]; job: Job | null }>(`/api/scenarios/${enc(sid)}/ladder`, { edit_index: editIndex, amounts, ...(runId ? { run_id: runId } : {}) });
}

export type AcrossRunsEstimate = {
  runs: { run_id: string; ok: boolean; reason: string | null; load_s: number; exact_s: number; rss_gb: number }[];
  total_s: number;
  peak_rss_gb: number;
};

export function estimateAcrossRuns(sid: string, runIds: string[]): Promise<AcrossRunsEstimate> {
  return api.post<AcrossRunsEstimate>(`/api/scenarios/${enc(sid)}/across-runs/estimate`, { run_ids: runIds });
}

export function startAcrossRuns(sid: string, runIds: string[]): Promise<Job> {
  return api.post<Job>(`/api/scenarios/${enc(sid)}/across-runs`, { run_ids: runIds });
}

/** `scenario.across_runs` job result (api.md §8): one row per run, sign stability and spread. */
export type AcrossRunsResult = {
  rows: { run_id: string; city: Likely | null; ok: boolean; error: string | null }[];
  sign_stability: number | null;
  spread: number | null;
};

/**
 * The project's "check across runs" jobs (newest first, `GET /api/jobs?kind=…&project=`); the
 * caller picks a scenario's by `scenario_id`. Refreshed by every job event (`jobs` tag).
 */
export function useAcrossRunsJobs(pid: string | null) {
  return useResource<Page<Job>>(pid ? `project:${pid}:lab:across` : null, (s) => api.get<Page<Job>>("/api/jobs", { kind: "scenario.across_runs", project: pid!, limit: 100 }, s), {
    tags: pid ? ["jobs", `project:${pid}`] : [],
  });
}

export type PromoteResult = { eligible: boolean; reason: string | null; yaml_diff: string | null; names: string[]; version: number | null };

export function promoteScenario(sid: string, apply = false): Promise<PromoteResult> {
  return api.post<PromoteResult>(`/api/scenarios/${enc(sid)}/promote`, { apply });
}

export function designCsvUrl(sid: string, runId: string): string {
  return apiUrl(`/api/scenarios/${enc(sid)}/design.csv`, { run_id: runId });
}

export type DesignImport = { blobs: Record<string, string>; n_rows: number; unknown_ids: (number | string)[]; levers: string[]; mode: "change" | "value" };

/** `POST /api/runs/{rid}/designs/import` (raw CSV `id,lever,change` or `id,lever,value`). */
export function importDesignCsv(rid: string, csv: Blob | string): Promise<DesignImport> {
  const body = typeof csv === "string" ? new Blob([csv], { type: "text/csv" }) : csv;
  return putRaw<DesignImport>(`${runBase(rid)}/designs/import`, body, { method: "POST", contentType: "text/csv" });
}

/** `PUT /api/runs/{rid}/blobs?kind=edit` (Int32 idx[m] then Float32 val[m]). */
export async function uploadEditBlob(rid: string, body: Uint8Array, count: number): Promise<string> {
  const r = await putRaw<{ blob_id: string; bytes: number }>(`${runBase(rid)}/blobs`, body, { query: { kind: "edit" }, headers: { "X-SPARC-Count": String(count) } });
  return r.blob_id;
}

/** `PUT /api/runs/{rid}/blobs?kind=mask` (raw LSB-first bitset bytes) → a `{kind: "blob"}` selection. */
export async function uploadMaskBlob(rid: string, bits: Uint8Array): Promise<SelectionSpec> {
  const r = await putRaw<{ blob_id: string; bytes: number }>(`${runBase(rid)}/blobs`, bits, { query: { kind: "mask" } });
  return { kind: "blob", blob_id: r.blob_id };
}

// ---------------------------------------------------------------- selections and regions (§6.3, used by the Lab)

export type SelectionReply = {
  n_cells: number;
  area_km2: number;
  people: number | null;
  medians: Record<string, number | null>;
  mask: Bitset;
  portable: boolean;
  warnings: string[];
};

export function resolveSelection(rid: string, selection: SelectionSpec, signal?: AbortSignal): Promise<SelectionReply> {
  return api.post<SelectionReply>(`${runBase(rid)}/selection/resolve`, { selection }, { signal });
}

export function selectionMask(reply: SelectionReply, n: number): Uint8Array {
  return decodeBitset(reply.mask, n);
}

export type Region = { id: string; name: string; spec: SelectionSpec; n_cells: number; created_utc?: string; portable?: boolean };

export function useRegions(rid: string | null) {
  return useResource<Region[]>(rid ? `run:${rid}:regions` : null, (s) => api.get<Region[]>(`${runBase(rid!)}/regions`, undefined, s), { tags: rid ? [`run:${rid}:regions`] : [] });
}

export function createRegion(rid: string, name: string, spec: SelectionSpec): Promise<Region> {
  return api.post<Region>(`${runBase(rid)}/regions`, { name, spec });
}

// ---------------------------------------------------------------- results (§7.5)

export type ConfiguredScenario = {
  slug: string;
  name: string;
  city: Likely;
  p10: number | null;
  p90: number | null;
  frac_extrapolated: number | null;
  mean_realized: Record<string, number>;
  causal_linear: { delta: number; lo: number; hi: number; model_within: boolean } | null;
  has_folds: boolean;
  layer_key: string;
  doc: ScenarioDoc;
};

export type RunScenarios = { configured: ConfiguredScenario[]; results: ResultSummary[] };

export function useRunScenarios(rid: string | null) {
  return useResource<RunScenarios>(rid ? `run:${rid}:lab:scenarios` : null, (s) => api.get<RunScenarios>(`${runBase(rid!)}/scenarios`, undefined, s), {
    tags: rid ? [`run:${rid}:lab`, `run:${rid}:results`] : [],
  });
}

export function rerunConfigured(rid: string, slug: string): Promise<Job> {
  return api.post<Job>(`${runBase(rid)}/configured/${enc(slug)}/rerun-exact`, {});
}

export type ResultRegion = {
  name: string;
  auto: boolean;
  n_cells: number;
  /** Null for an empty region. */
  mean: Likely | null;
  people_weighted: number | null;
  total: number;
  frac_cooled_01: number;
  frac_cooled_05: number;
};

export type ExposureRow = { case: string; adapted: boolean; person_mean_temp: number; people_ge: Record<string, number>; share_people_ge: Record<string, number> };

export type Impacts = {
  thresholds: number[];
  exposure: ExposureRow[];
  equity: Record<string, { quintiles: { quintile: number; mean_cooling: number; people: number; value_range: [number, number] }[]; concentration_index: number }>;
  hot_days: { station: string; cases: Record<string, unknown>[] } | null;
  hot_days_action: Action | null;
  zones: Record<string, unknown>[];
  hexes: { "250": Record<string, unknown>[]; "500": Record<string, unknown>[] };
  climate_offset: { experiment: string; period: string; offset_share: number }[];
};

export type Result = {
  summary: ResultSummary;
  spec: Record<string, unknown>;
  scenario: { id: string; revision: number; name: string } | null;
  city: Likely;
  p10: number | null;
  p90: number | null;
  mean_delta_sd: number | null;
  regions: ResultRegion[];
  spill: {
    inside: number;
    outside: number;
    outside_share: number | null;
    rings: { r_m: number; mean: number | null; se: number | null; n: number }[];
    lever_ranges: Record<string, number>;
  };
  extrapolated_edited: number;
  realized: Record<string, { requested_mean: number; realized_mean: number; requested_total: number; realized_total: number; clipped_share: number }>;
  mediators: Record<string, { mean_change: number }>;
  cost: { total: number; per_lever: Record<string, number>; cooling_per_cost: number | null };
  causal_check: { delta: number; lo: number; hi: number; model_within: boolean } | null;
  uncertainty: {
    estimation_95: [number, number] | null;
    specification: [number, number] | null;
    attribution: [number, number] | null;
    causal_band: [number, number] | null;
    envelope: [number, number] | null;
    envelope_excludes_zero: boolean | null;
    sources: string[];
  } | null;
  impacts: Impacts | null;
  preview_vs_exact: { mean_abs_err: number; rel_err: number } | null;
  plain: { headline: string; confidence: string; qualifiers: string[]; buys: string[] };
  warnings: { code: string; message: string }[];
  stale: boolean;
  demo: boolean;
};

export function useResult(resId: string | null, rid?: string | null) {
  return useResource<Result>(resId ? `result:${resId}` : null, (s) => api.get<Result>(`/api/results/${enc(resId!)}`, undefined, s), {
    tags: resId ? [`result:${resId}`, ...(rid ? [`run:${rid}:results`] : [])] : [],
  });
}

export type ResultField = "delta" | "delta_sd" | "extrapolation" | "abs" | `realized_${string}`;

export async function getResultLayer(resId: string, field: ResultField, signal?: AbortSignal): Promise<Float32Array> {
  return (await getBin<Float32Array>(`/api/results/${enc(resId)}/layers/${enc(field)}.bin`, "float32", { signal })).data;
}

export type ImpactsRequest = { thresholds?: number[]; futures?: { experiment: string; period: string }[] };

export function postImpacts(resId: string, body: ImpactsRequest): Promise<Impacts> {
  return api.post<Impacts>(`/api/results/${enc(resId)}/impacts`, body);
}

export function deleteResult(resId: string): Promise<{ ok: true }> {
  return api.del<{ ok: true }>(`/api/results/${enc(resId)}`);
}

// ---------------------------------------------------------------- compare (§7.6)

export type ComparisonItem = { ref: ItemRef; label: string; city: Likely; edited: Likely | null; cost: number | null; has_folds: boolean };

export type PairedLikely = Likely & { paired: boolean };

export type ComparisonPair = { a: number; b: number; city: PairedLikely; regions: Record<string, PairedLikely>; layer_key: string };

export type Comparison = {
  id: string;
  items: ComparisonItem[];
  pairs: ComparisonPair[];
  equity: Record<string, Record<string, number>>;
  exposure: Record<string, unknown>[];
  cooling_per_cost: Record<string, number | null>;
  needs_exact: ItemRef[];
};

export function createComparison(rid: string, items: ItemRef[], opts: { regions?: string[]; thresholds?: number[] } = {}): Promise<Comparison> {
  return api.post<Comparison>(`${runBase(rid)}/compare`, { items, ...opts });
}

export function useComparison(cid: string | null) {
  return useResource<Comparison>(cid ? `comparison:${cid}` : null, (s) => api.get<Comparison>(`/api/comparisons/${enc(cid!)}`, undefined, s), { tags: cid ? [`comparison:${cid}`] : [] });
}

export type ComparisonListItem = { id: string; items: ItemRef[]; created_utc: string };

export function useComparisons(rid: string | null) {
  return useResource<ComparisonListItem[]>(rid ? `run:${rid}:lab:comparisons` : null, (s) => api.get<ComparisonListItem[]>(`${runBase(rid!)}/comparisons`, undefined, s), {
    tags: rid ? [`run:${rid}:lab`] : [],
  });
}

export function deleteComparison(cid: string): Promise<{ ok: true }> {
  return api.del<{ ok: true }>(`/api/comparisons/${enc(cid)}`);
}

// ---------------------------------------------------------------- climate × adaptation (§7.7)

export type WarmingRow = {
  experiment: string;
  label: string;
  period: string;
  n_models: number;
  median: number;
  p10: number;
  p90: number;
  min: number;
  max: number;
  by_model: Record<string, number>;
};

export type ClimateFactors = {
  present: boolean;
  path: string | null;
  experiments: string[];
  periods: string[];
  models: string[];
  warming: WarmingRow[];
  action: Action | null;
};

export function useClimateFactors(rid: string | null) {
  return useResource<ClimateFactors>(rid ? `run:${rid}:lab:climate` : null, (s) => api.get<ClimateFactors>(`${runBase(rid!)}/climate/factors`, undefined, s), {
    tags: rid ? [`run:${rid}:lab`] : [],
  });
}

export type ClimateStatistic = "median" | "p10" | "p90" | { model: string };

export type ShareStat = { median: number; p10: number; p90: number };

export type ClimateVariant = {
  name: string;
  mean: number;
  adaptation_mean_delta: number;
  offset_share_of_median_warming: number | null;
  share_at_or_above: Record<string, ShareStat>;
};

export type ClimateProjection = {
  experiment: string;
  label: string;
  period: string;
  n_models: number;
  warming: { median: number; p10: number; p90: number; min: number; max: number; by_model: Record<string, number> };
  variants: ClimateVariant[];
};

export type PresentClimate = { mean: number; share_at_or_above: Record<string, number> };

export type ClimateExplore = {
  /** api.md: summarize_projections' `present` (today); as built the server sends `true` here and today in `present_today`. */
  present: boolean | PresentClimate;
  present_today?: PresentClimate | null;
  statistic?: ClimateStatistic;
  thresholds: number[];
  projections: ClimateProjection[];
  adaptation: string[];
  people_exposure: Record<string, unknown>[] | null;
  units: string | Record<string, unknown>;
};

export type ClimateExploreRequest = {
  adaptations: ItemRef[];
  thresholds?: number[];
  experiments?: string[];
  periods?: string[];
  statistic?: ClimateStatistic;
};

export function exploreClimate(rid: string, body: ClimateExploreRequest, signal?: AbortSignal): Promise<ClimateExplore> {
  return api.post<ClimateExplore>(`${runBase(rid)}/climate/explore`, body, { signal });
}

// ---------------------------------------------------------------- sweeps (§7.8)

export type SweepPoint = { dose: number; city: Likely; region: Likely | null; realized: number; frac_extrapolated: number };

export type Sweep = {
  params: { lever: string; doses: number[]; selection?: SelectionSpec | null } & Record<string, unknown>;
  status: string;
  curve: SweepPoint[];
  fit: { model: string; A: number | null; ds: number | null; d90: number | null } | null;
  pipeline_curve: Record<string, unknown> | null;
  points: string[];
};

export type SweepListItem = { id: string; lever: string; doses: number[]; status: string; created_utc: string; job_id: string | null };

export function createSweep(rid: string, body: { lever: string; doses: number[]; selection?: SelectionSpec }): Promise<{ sweep_id: string; job: Job }> {
  return api.post<{ sweep_id: string; job: Job }>(`${runBase(rid)}/sweeps`, body);
}

export function useSweeps(rid: string | null) {
  return useResource<SweepListItem[]>(rid ? `run:${rid}:lab:sweeps` : null, (s) => api.get<SweepListItem[]>(`${runBase(rid!)}/sweeps`, undefined, s), {
    tags: rid ? [`run:${rid}:lab`, `run:${rid}:results`] : [],
  });
}

export function useSweep(swid: string | null, rid: string | null) {
  return useResource<Sweep>(swid ? `sweep:${swid}` : null, (s) => api.get<Sweep>(`/api/sweeps/${enc(swid!)}`, undefined, s), {
    tags: swid ? [`sweep:${swid}`, ...(rid ? [`run:${rid}:lab`, `run:${rid}:results`] : [])] : [],
  });
}

export function deleteSweep(swid: string): Promise<{ ok: true }> {
  return api.del<{ ok: true }>(`/api/sweeps/${enc(swid)}`);
}

// ---------------------------------------------------------------- plans (§7.9)

export type PlanCost = { scalar: number } | { column: string };

export type EquitySource = "share_60_plus" | "share_under_5" | "density" | "column";

export type PlanParams = {
  lever: string;
  budget: number;
  cost: PlanCost;
  cap: { plantable: boolean; paved_share?: number; region?: SelectionSpec };
  min_dose?: number;
  objective: "cooling" | "people";
  equity?: { source: EquitySource; column?: string; focus: number };
  multipliers?: number[];
};

export type ParetoRow = { budget: number; benefit: number; n_cells: number; n_segments: number; gini: number };

export type PlanPreview = {
  planned_total: number;
  n_cells_treated: number;
  mean_dose_treated: number;
  total_cost: number;
  gini: number;
  min_dose_dropped_cost: number;
  pareto: ParetoRow[];
  /** base64 Float32[n] dose per row. */
  dose: string;
  constraint: string;
  objective: string;
  caption: string;
};

export type Plan = {
  id: string;
  name: string;
  params: PlanParams;
  planned: Omit<PlanPreview, "dose">;
  realised: { total: number; mean_treated: number; mean_all: number; result_id: string } | null;
  frontier: { budget: number; planned: number; realised: number }[] | null;
  created_utc: string;
};

export function previewPlan(rid: string, params: PlanParams, signal?: AbortSignal): Promise<PlanPreview> {
  return api.post<PlanPreview>(`${runBase(rid)}/plans/preview`, params, { signal });
}

export function createPlan(rid: string, params: PlanParams, name: string, verify = true): Promise<{ plan: Plan; job: Job | null }> {
  return api.post<{ plan: Plan; job: Job | null }>(`${runBase(rid)}/plans`, { params, name, verify });
}

export function usePlans(rid: string | null) {
  return useResource<Plan[]>(rid ? `run:${rid}:lab:plans` : null, (s) => api.get<Plan[]>(`${runBase(rid!)}/plans`, undefined, s), { tags: rid ? [`run:${rid}:lab`, `run:${rid}:results`] : [] });
}

export function usePlan(plid: string | null, rid: string | null) {
  return useResource<Plan>(plid ? `plan:${plid}` : null, (s) => api.get<Plan>(`/api/plans/${enc(plid!)}`, undefined, s), {
    tags: plid ? [`plan:${plid}`, ...(rid ? [`run:${rid}:lab`, `run:${rid}:results`] : [])] : [],
  });
}

export function verifyPlan(plid: string, frontier = false): Promise<Job> {
  return api.post<Job>(`/api/plans/${enc(plid)}/verify`, { frontier });
}

export type PlanField = "dose" | "planned_benefit" | "closed_loop_delta";

export async function getPlanLayer(plid: string, field: PlanField, signal?: AbortSignal): Promise<Float32Array> {
  return (await getBin<Float32Array>(`/api/plans/${enc(plid)}/layers/${field}.bin`, "float32", { signal })).data;
}

export function planToScenario(plid: string): Promise<Scenario> {
  return api.post<Scenario>(`/api/plans/${enc(plid)}/to-scenario`, {});
}

export type FieldKitCell = {
  rank: number;
  id: number | string;
  lon: number | null;
  lat: number | null;
  zone: number | string | null;
  dose: number;
  planned_benefit: number | null;
  closed_loop_delta: number | null;
  people: number | null;
  plantable_pp: number | null;
};

export type FieldKitSite = { id: number | string; lon: number | null; lat: number | null; role: string; canopy: number | null; impervious: number | null; effect_sd: number | null };

export type FieldKitPair = {
  treated_id: number | string;
  control_id: number | string;
  treated_lon: number | null;
  treated_lat: number | null;
  control_lon: number | null;
  control_lat: number | null;
  covariate_distance: number | null;
};

export type FieldKit = { cells: FieldKitCell[]; sites: FieldKitSite[]; pairs: FieldKitPair[] };

export function getFieldKit(plid: string, body: { n_sites?: number; min_spacing_m?: number; n_pairs?: number; min_distance_m?: number } = {}): Promise<FieldKit> {
  return api.post<FieldKit>(`/api/plans/${enc(plid)}/field-kit`, body);
}

export function deletePlan(plid: string): Promise<{ ok: true }> {
  return api.del<{ ok: true }>(`/api/plans/${enc(plid)}`);
}

// ---------------------------------------------------------------- packs (api.md §10; exact only, SPEC §7.13)

export type PackKind = "decision_pack" | "plan_pack" | "compare_pack";

export type ExportRecord = {
  id: string;
  project_id: string;
  run_id: string | null;
  kind: string;
  ref: string | null;
  options: Record<string, unknown>;
  job_id: string;
  status: "running" | "ready" | "failed";
  path: string | null;
  bytes: number | null;
  draft: boolean;
  created_utc: string;
};

export function exportPack(pid: string, kind: PackKind, params: Record<string, unknown>): Promise<{ export: ExportRecord; job: Job }> {
  return api.post<{ export: ExportRecord; job: Job }>("/api/exports", { kind, project_id: pid, params });
}

// ---------------------------------------------------------------- project runs (across-runs dialog)

export function useProjectRunList(pid: string | null) {
  return useResource<Page<RunSummary>>(pid ? `project:${pid}:lab:runs` : null, (s) => api.get<Page<RunSummary>>(`/api/projects/${enc(pid!)}/runs`, { limit: 200 }, s), {
    tags: pid ? [`project:${pid}`, "runs"] : [],
  });
}
