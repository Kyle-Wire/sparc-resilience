/**
 * Typed fetch wrappers for the SPARC FastAPI server at localhost:8008.
 */
import type {
  HealthResponse,
  ProjectLoadResponse,
  ValidationResponse,
  InitResponse,
  TemplateInfo,
  DataSummary,
  DataPreview,
  DagDefinition,
  DagValidation,
  MC3Result,
  ProjectConfig,
  CorrelogramData,
  CausalResults,
  PdpCurves,
  DoseResponseData,
  CausalDiagnostics,
  ScenarioDetail,
  ReportPayload,
  GeoJsonData,
  PipelineEvent,
  ValidationReport,
  DataVersion,
  RunManifest,
  StageManifest,
  MissingArtifactDetail,
} from "./types";

const BASE = "http://127.0.0.1:8008";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status}: ${body}`);
  }
  return res.json() as Promise<T>;
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

// ------------------------------------------------------------------
// Health
// ------------------------------------------------------------------
export const health = () => get<HealthResponse>("/health");

// ------------------------------------------------------------------
// Pipeline run events (reconnection support)
// ------------------------------------------------------------------
export interface RunEventsResponse {
  is_running: boolean;
  current_stage: number | null;
  events: PipelineEvent[];
}
export const getRunEvents = () => get<RunEventsResponse>("/run/events");

// ------------------------------------------------------------------
// Project
// ------------------------------------------------------------------
export const loadProject = (path: string) =>
  post<ProjectLoadResponse>(`/project/load?path=${encodeURIComponent(path)}`);

export const validateProject = (path: string) =>
  post<ValidationResponse>(`/project/validate?path=${encodeURIComponent(path)}`);

export const initProject = (template: string, output: string) =>
  post<InitResponse>(`/project/init?template=${encodeURIComponent(template)}&output=${encodeURIComponent(output)}`);

/**
 * Create a project from the wizard payload — combines `init` (scaffold from
 * template) + structured edits to `project.yml` (identity + CRS) in one call.
 * The server handles template hygiene: only fields the wizard captures are
 * written, plus the template's seeded physics/DAG defaults (which are visible
 * and editable on their pages).
 */
export interface CreateProjectPayload {
  template: string;
  output: string;
  identity: {
    name: string;
    description?: string;
    author?: string;
    response_units?: string;
  };
  crs: {
    input: string;
    projected: string;
  };
}

export const createProject = (payload: CreateProjectPayload) =>
  post<InitResponse>("/project/create", payload);

export const listTemplates = () =>
  get<{ templates: TemplateInfo[] }>("/project/templates");

// ------------------------------------------------------------------
// Data
// ------------------------------------------------------------------
export const dataSummary = () => get<DataSummary>("/data/summary");

export const checkCrsDistortion = (inputEpsg: string, projectedEpsg: string) =>
  get<{
    center_lon: number;
    center_lat: number;
    k_x: number;
    k_y: number;
    k_mean: number;
    area_distortion_pct: number;
    input_crs_name: string;
    projected_crs_name: string;
    assessment: string;
  }>(`/crs/distortion?input_epsg=${encodeURIComponent(inputEpsg)}&projected_epsg=${encodeURIComponent(projectedEpsg)}`);

export const dataPreview = (n = 50) => get<DataPreview>(`/data/preview?n=${n}`);

export interface DataHistogram {
  variable: string;
  bins: number[];
  edges: number[];
  n_total: number;
  n_finite: number;
  n_missing: number;
  min: number | null;
  max: number | null;
  mean: number | null;
  std: number | null;
}

/** Server-side histogram — single pass on backend, returns ~bins ints. */
export const dataHistogram = (variable: string, bins = 40) =>
  get<DataHistogram>(
    `/data/histogram?variable=${encodeURIComponent(variable)}&bins=${bins}`,
  );

export const dataGeoJson = (variable?: string) =>
  get<GeoJsonData>(variable ? `/data/geojson?variable=${encodeURIComponent(variable)}` : "/data/geojson");

export async function uploadData(file: File): Promise<ProjectLoadResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/data/upload`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export const listDataFiles = () =>
  get<{ project_dir: string; files: { name: string; path: string; relative: string; size: number }[] }>("/data/files");

export const selectDataFile = (path: string) =>
  post<ProjectLoadResponse>(`/data/select?path=${encodeURIComponent(path)}`);

// ------------------------------------------------------------------
// Results
// ------------------------------------------------------------------
export const getResults = (stage: number, format: "json" | "geojson" = "json") =>
  get<unknown>(`/results/${stage}?format=${format}`);

export const getPredictions = (stage: number) =>
  get<unknown>(`/results/${stage}/predictions?format=geojson`);

export const getStagePlots = (stage: number) =>
  get<{ plots: { name: string; filename: string; path: string }[] }>(`/results/${stage}/plots`);

/** Build a full URL for a stage plot image. */
export const stagePlotUrl = (stage: number, path: string) =>
  `${BASE}/results/${stage}/plots/${encodeURIComponent(path)}`;

// ------------------------------------------------------------------
// Run registry (artifacts manifest)
// ------------------------------------------------------------------

/** Fetch the full run manifest. Use `refresh` to reload from disk, `rescan`
 *  to additionally walk known artifact patterns. */
export const getManifest = (opts: { refresh?: boolean; rescan?: boolean } = {}) => {
  const params = new URLSearchParams();
  if (opts.refresh) params.set("refresh", "true");
  if (opts.rescan) params.set("rescan", "true");
  const q = params.toString();
  return get<RunManifest>(`/results/manifest${q ? `?${q}` : ""}`);
};

/** Fetch a single stage manifest. Returns a `pending` skeleton if the stage
 *  has not been registered yet — callers can render an empty-state without
 *  distinguishing missing-vs-empty. */
export const getStageManifest = (stage: string | number) =>
  get<StageManifest>(`/results/manifest/${stage}`);

/** Force a disk walk to import any newly-written artifacts. */
export const rescanManifest = () =>
  post<{ newly_registered: number; total_artifacts: number }>(
    "/results/manifest/rescan",
  );

/**
 * Type guard: extract a `MissingArtifactDetail` body from a thrown error
 * message of the form `"404: { ...json... }"` produced by `get`/`post`.
 * Returns `null` when the error isn't a structured 404.
 */
export function parseMissingArtifact(err: unknown): MissingArtifactDetail | null {
  if (!(err instanceof Error)) return null;
  const m = err.message.match(/^404:\s*(\{.*\})$/s);
  if (!m) return null;
  try {
    const parsed = JSON.parse(m[1]);
    const detail = parsed?.detail;
    if (detail && typeof detail === "object" && detail.error === "missing_artifact") {
      return detail as MissingArtifactDetail;
    }
  } catch {
    /* fall through */
  }
  return null;
}

// ------------------------------------------------------------------
// Scenarios
// ------------------------------------------------------------------
export const runScenarios = () => post<{ status: string; n_scenarios: number; summary_rows: number }>("/scenarios/run");

export const scenarioResults = (format: "json" | "geojson" = "geojson") =>
  get<unknown>(`/scenarios/results?format=${format}`);

export const getNutsSummary = () =>
  get<{
    acceptance_rate?: number;
    n_divergences?: number;
    posteriors?: Record<string, unknown>[];
    convergence?: Record<string, unknown>[];
    bma?: Record<string, unknown>[];
  }>("/results/scenarios/nuts_summary");

// ------------------------------------------------------------------
// DAG
// ------------------------------------------------------------------
export const getDag = () => get<DagDefinition>("/dag");

export const validateDag = (dag: DagDefinition) =>
  post<DagValidation>("/dag/validate", dag);

export const getMc3Result = () => get<MC3Result>("/dag/mc3_result");

export const approveDag = () =>
  post<{ status: string }>("/dag/approve", {});

export const rejectDag = () =>
  post<{ status: string }>("/dag/reject", {});

// ------------------------------------------------------------------
// Config
// ------------------------------------------------------------------
export const getConfig = () => get<ProjectConfig>("/project/config");

export const updateConfig = (config: Partial<ProjectConfig>) =>
  post<{ status: string }>("/project/config", config);

// For PUT we need a separate helper
async function put<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<T>;
}

export const saveConfig = (config: Partial<ProjectConfig>) =>
  put<{ status: string }>("/project/config", config);

// ------------------------------------------------------------------
// Report
// ------------------------------------------------------------------
export const generateReport = (format: "markdown" | "json" = "markdown") =>
  post<{ markdown: string; report?: unknown }>(`/report/generate?format=${format}`);

/** Request a PDF report — returns binary blob or HTML fallback JSON. */
export async function generatePdfReport(): Promise<{ blob: Blob | null; htmlFallback?: string }> {
  const res = await fetch(`${BASE}/report/pdf`, { method: "POST" });
  if (!res.ok) throw new Error(await res.text());

  const ct = res.headers.get("content-type") ?? "";
  if (ct.includes("application/pdf")) {
    return { blob: await res.blob() };
  }
  // HTML fallback (weasyprint not installed)
  const json = await res.json();
  return { blob: null, htmlFallback: json.html_path };
}

// ------------------------------------------------------------------
// Structured results
// ------------------------------------------------------------------
export const getCorrelogramData = () =>
  get<CorrelogramData>("/results/correlogram");

export const getModelPerformance = () =>
  get<{ models: { name: string; r2: number; rmse?: number }[] }>("/results/model_performance");

export const getGwenData = () =>
  get<{ rows: Record<string, unknown>[] }>("/results/gwen");

export const getSpatialCvPredictions = () =>
  get<GeoJsonData>("/results/spatial_cv/predictions");

export const getCausalResults = () =>
  get<CausalResults>("/results/causal");

export const getDoseResponseCurves = () =>
  get<DoseResponseData>("/results/causal/dose_response");

export const getCausalDiagnostics = () =>
  get<CausalDiagnostics>("/results/causal/diagnostics");

export interface SensitivityRecord {
  effect_label: string;
  point_estimate: number;
  ci_lower: number | null;
  ci_upper: number | null;
  rr_equivalent: number;
  e_value_point: number;
  e_value_ci: number | null;
  interpretation: string;
}

export interface CausalSensitivity {
  method: string;
  results: SensitivityRecord[];
}

export const getCausalSensitivity = () =>
  get<CausalSensitivity>("/results/causal/sensitivity");

export interface NegativeControlResponse {
  variable: string;
  n: number;
  mean_observed: number;
  mean_null: number;
  std_null: number;
  p_value: number;
  z_score: number;
  n_permutations: number;
  passed: boolean;
  interpretation: string;
}

export const getCausalNegativeControl = (variable: string, nPermutations = 1000) =>
  get<NegativeControlResponse>(
    `/results/causal/negative_control?variable=${encodeURIComponent(variable)}&n_permutations=${nPermutations}`,
  );

// ------------------------------------------------------------------
// Run comparison (Phase 15)
// ------------------------------------------------------------------
export interface RunCandidate {
  path: string;
  name: string;
  mtime: number;
}

export interface RunDiscoveryResponse {
  search_root: string;
  runs: RunCandidate[];
}

export interface ConfigDiffRow {
  path: string;
  a: unknown;
  b: unknown;
  kind: "added" | "removed" | "changed";
}

export interface ModelDiffRow {
  model: string;
  r2_a: number | null;
  r2_b: number | null;
  rmse_a: number | null;
  rmse_b: number | null;
  mae_a: number | null;
  mae_b: number | null;
  r2_delta: number | null;
  rmse_delta: number | null;
  mae_delta: number | null;
}

export interface CausalDiffRow {
  effect: string;
  estimate_a: number | null;
  estimate_b: number | null;
  ci_a: [number | null, number | null];
  ci_b: [number | null, number | null];
  delta: number | null;
}

export interface DagEdgeDiff {
  from: string;
  to: string;
  prob_a?: number | null;
  prob_b?: number | null;
  delta?: number | null;
  probability?: number | null;
}

export interface DagDiff {
  added: DagEdgeDiff[];
  removed: DagEdgeDiff[];
  changed: DagEdgeDiff[];
  n_a: number;
  n_b: number;
  found_a: boolean;
  found_b: boolean;
}

export interface RunDiffResponse {
  run_a: string;
  run_b: string;
  config: { found_a: boolean; found_b: boolean; changes: ConfigDiffRow[] };
  causal: { found_a: boolean; found_b: boolean; effects: CausalDiffRow[] };
  models: { found_a: boolean; found_b: boolean; rows: ModelDiffRow[] };
  dag: DagDiff;
}

export const discoverRuns = (searchRoot?: string) =>
  get<RunDiscoveryResponse>(
    `/runs/discover${searchRoot ? `?search_root=${encodeURIComponent(searchRoot)}` : ""}`,
  );

export const diffRuns = (runA: string, runB: string) =>
  get<RunDiffResponse>(
    `/runs/diff?run_a=${encodeURIComponent(runA)}&run_b=${encodeURIComponent(runB)}`,
  );

export const getPdpCurves = () =>
  get<PdpCurves>("/results/pdp_curves");

/** PINN / PDE-constrained neural-network PDP curves (preferred over GWRF). */
export const getNeuralPdp = () =>
  get<PdpCurves>("/results/neural_pdp");

/** Per-endpoint availability map: tells the UI which result tabs have data. */
export const getResultsAvailability = () =>
  get<Record<string, { available: boolean; source: string; output_dir?: string }>>(
    "/results/availability",
  );

export const getScenarioDetail = () =>
  get<ScenarioDetail>("/results/scenarios/detail");

export const getReportData = () =>
  get<ReportPayload>("/results/report");

// ------------------------------------------------------------------
// CATE map, local coefficients, scenario increments
// ------------------------------------------------------------------
export const getCateMapVariables = () =>
  get<{ variables: string[] }>("/results/causal/cate_map/variables");

export const getCateMap = (variable: string) =>
  get<GeoJsonData>(`/results/causal/cate_map?variable=${encodeURIComponent(variable)}`);

export const getLocalCoefVariables = () =>
  get<{ variables: string[] }>("/results/local_coefficients/variables");

export const getLocalCoefficients = (variable: string) =>
  get<GeoJsonData>(`/results/local_coefficients?variable=${encodeURIComponent(variable)}`);

export const getScenarioIncrement = (variable: string, increment: number) =>
  get<ScenarioDetail>(`/results/scenarios/increment?variable=${encodeURIComponent(variable)}&increment=${increment}`);

export interface ScenarioVariableInfo {
  increments: number[];
  sign: "plus" | "minus";
}

export const getScenarioVariables = () =>
  get<{ variables: Record<string, ScenarioVariableInfo> }>("/results/scenarios/variables");

// ------------------------------------------------------------------
// Phase 5 — scenario bundle endpoints (attribution, trajectory, uncertainty)
// ------------------------------------------------------------------
export interface AttributionRecord {
  scenario_id: number | string;
  variable: string;
  contribution: number;
  [key: string]: unknown;
}

export const getScenarioAttribution = () =>
  get<{ records: AttributionRecord[]; columns: string[] }>("/results/scenarios/attribution");

export interface TrajectoryRecord {
  scenario_id: number | string;
  t: number | string;
  geometry_id?: string | number;
  value: number;
  std?: number;
  [key: string]: unknown;
}

export const getScenarioTrajectory = (scenarioId?: number | string) =>
  get<{ records: TrajectoryRecord[]; columns: string[] }>(
    scenarioId !== undefined && scenarioId !== ""
      ? `/results/scenarios/trajectory?scenario_id=${encodeURIComponent(String(scenarioId))}`
      : "/results/scenarios/trajectory",
  );

export const getScenarioUncertainty = (scenarioId?: number | string) =>
  get<GeoJsonData>(
    scenarioId !== undefined && scenarioId !== ""
      ? `/results/scenarios/uncertainty?scenario_id=${encodeURIComponent(String(scenarioId))}`
      : "/results/scenarios/uncertainty",
  );

/**
 * Download the full results bundle (ZIP of all artifacts + manifest).
 * Returns the streaming Response so the caller can surface it to the user.
 */
export const downloadResultsBundle = async (): Promise<Blob> => {
  const res = await fetch(`${BASE}/results/bundle`);
  if (!res.ok) throw new Error(`bundle download failed: ${res.status}`);
  return await res.blob();
};

// ------------------------------------------------------------------
// Decision-support (Phase 9): intervention candidates + optimizer
// ------------------------------------------------------------------
export interface InterventionCandidate {
  name: string;
  treatment: string;
  magnitude: number;
  mean_effect: number;
  effect_std: number;
  cost: number;
  equity_weight: number;
  notes?: string;
}

export interface RankedIntervention extends InterventionCandidate {
  risk_adjusted_effect: number;
  score: number;
  selected?: boolean;
}

export interface OptimizerResponse {
  ranked: RankedIntervention[];
  settings: {
    budget: number | null;
    robustness_lambda: number;
    minimise: boolean;
    n_candidates: number;
    n_selected: number;
  };
  equity_summary: {
    n: number;
    mean: number | null;
    min: number | null;
    max: number | null;
    gini: number | null;
  };
}

export const getDecisionCandidates = () =>
  get<{ candidates: InterventionCandidate[] }>("/decision/candidates");

export const optimizeDecisions = (body: {
  candidates?: InterventionCandidate[];
  budget?: number | null;
  robustness_lambda?: number;
  minimise?: boolean;
}) => post<OptimizerResponse>("/decision/optimize", body);

export interface EquityScore {
  candidate: string;
  weight: number;
  layer_breakdown: Record<string, number>;
}

export interface EquityResponse {
  scores: EquityScore[];
  disparity_index: number;
}

export const computeEquity = (body: {
  candidate_names: string[];
  layers: Record<string, number[]>;
  weights?: Record<string, number>;
  invert?: Record<string, boolean>;
}) => post<EquityResponse>("/decision/equity", body);

// Census ACS auto-fetch (TIGER/Geocoder + ACS 5-year)
export interface CensusContext {
  counties: { state_fips: string; county_fips: string; name: string }[];
  n_counties: number;
  population: number | null;
  median_income: number | null;
  poverty_rate: number | null;
  mobile_home_share: number | null;
  vintage: number;
  source: string;
  notes: string;
}

export interface CensusEquityResponse {
  bbox: { minlon: number; minlat: number; maxlon: number; maxlat: number };
  layers: Record<string, number>;
  invert: Record<string, boolean>;
  context: CensusContext;
}

export const getCensusEquity = (bbox?: {
  minlon: number; minlat: number; maxlon: number; maxlat: number;
}) => {
  const qs = bbox
    ? `?minlon=${bbox.minlon}&minlat=${bbox.minlat}&maxlon=${bbox.maxlon}&maxlat=${bbox.maxlat}`
    : "";
  return get<CensusEquityResponse>(`/equity/census${qs}`);
};

export interface TargetingResponse extends GeoJsonData {
  summary: {
    variable: string;
    n_features: number;
    top_k: number;
    max_priority: number;
    min_priority: number;
  };
}

export const getTargeting = (variable: string, topK = 50) =>
  get<TargetingResponse>(
    `/decision/targeting?variable=${encodeURIComponent(variable)}&top_k=${topK}`,
  );

export interface UncertaintyRecord {
  candidate: string;
  selection_probability: number;
  rank_mean: number;
  rank_p10: number;
  rank_p90: number;
  effect_p10: number;
  effect_p50: number;
  effect_p90: number;
}

export interface UncertaintyResponse {
  uncertainty: UncertaintyRecord[];
  settings: {
    n_draws: number;
    robustness_lambda: number;
    budget: number | null;
    minimise: boolean;
  };
}

export const decisionUncertainty = (body: {
  candidates?: InterventionCandidate[];
  budget?: number | null;
  robustness_lambda?: number;
  minimise?: boolean;
  n_draws?: number;
  seed?: number | null;
}) => post<UncertaintyResponse>("/decision/uncertainty", body);

// ------------------------------------------------------------------
// Budget-constrained intervention allocation (greedy / 2-opt / MILP)
// ------------------------------------------------------------------
export interface BudgetOptimizeRequest {
  budget: number;
  benefit_source: "cate" | "local_coef" | "uniform";
  variable?: string;
  cost_column?: string;
  x_max_column?: string;
  solver?: "greedy" | "greedy_2opt" | "milp" | "auto";
  pareto_sweep?: boolean;
}

export interface BudgetResult {
  allocation: number[];
  total_benefit: number;
  total_cost: number;
  budget: number;
  n_treated: number;
  n_fully_treated: number;
  gini: number;
  solver: string;
  solve_time_s: number;
  notes?: string;
}

export interface BudgetParetoPoint {
  budget: number;
  total_benefit: number;
  n_treated: number;
  gini: number;
}

export interface BudgetOptimizeResponse {
  result: BudgetResult;
  n_cells: number;
  benefit_source: string;
  benefit_description: string;
  pareto?: { points: BudgetParetoPoint[] };
}

export const optimizeBudget = (req: BudgetOptimizeRequest) =>
  post<BudgetOptimizeResponse>("/scenarios/budget/optimize", req);

// ------------------------------------------------------------------
// Data preparation (raster + fishnet + zonal stats)
// ------------------------------------------------------------------
export interface PrepareDataPayload {
  boundary_path?: string;
  raster_paths: string[];
  resolution?: number;
  crs?: string;
  stats?: string;
  set_as_data?: boolean;
}

export interface PrepareDataResult {
  status: string;
  n_cells: number;
  columns: string[];
  csv_path: string;
  gpkg_path: string;
  set_as_data: boolean;
}

export const prepareData = (payload: PrepareDataPayload) =>
  post<PrepareDataResult>("/data/prepare", payload);

export interface FishnetPayload {
  bounds: [number, number, number, number];
  resolution: number;
  crs?: string;
  boundary_path?: string;
}

export const createFishnet = (payload: FishnetPayload) =>
  post<{ n_cells: number; columns: string[] }>("/data/fishnet", payload);

export interface ZonalStatsPayload {
  fishnet_path: string;
  raster_paths: string[];
  stats?: string;
}

export const runZonalStats = (payload: ZonalStatsPayload) =>
  post<{ n_cells: number; columns: string[]; csv_path: string }>("/data/zonal_stats", payload);

// ------------------------------------------------------------------
// Data validation
// ------------------------------------------------------------------
export const validateData = () =>
  post<ValidationReport>("/data/validate");

// ------------------------------------------------------------------
// Data versioning
// ------------------------------------------------------------------
export const getDataVersions = () =>
  get<{ versions: DataVersion[] }>("/data/versions");

export const selectDataVersion = (version: number) =>
  post<ProjectLoadResponse>(`/data/select_version?version=${version}`);

// ------------------------------------------------------------------
// Session log
// ------------------------------------------------------------------
export interface SessionLogEntry {
  timestamp: string;
  type: string;
  stage?: number;
  message?: string;
  [key: string]: unknown;
}

export const getRunLog = () =>
  get<{ entries: SessionLogEntry[]; path: string }>("/run/log");

// ------------------------------------------------------------------
// Artifact downloads
// ------------------------------------------------------------------
export interface ArtifactInfo {
  stage: number;
  stage_label: string;
  filename: string;
  relative_path: string;
  absolute_path: string;
  size_bytes: number;
  extension: string;
}

export const listArtifacts = () =>
  get<{ artifacts: ArtifactInfo[]; total: number }>("/results/artifacts");

export function downloadArtifact(stage: number, relativePath: string, filename: string): void {
  const url = `${BASE}/results/download/${stage}/${encodeURIComponent(relativePath)}`;
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

export function downloadGeoPackage(): void {
  const url = `${BASE}/results/geopackage`;
  const a = document.createElement("a");
  a.href = url;
  a.download = "results.gpkg";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

// ------------------------------------------------------------------
// Artifact export-on-demand (db-only architecture)
// ------------------------------------------------------------------
// Mirrors the FastAPI endpoints introduced in Phase 5:
//   GET  /artifacts/{stage}                       -- list
//   GET  /artifacts/{stage}/{artifact_id}/download?fmt=  -- download single
//   POST /artifacts/{stage}/export                -- bundle stage as .zip

export interface DbArtifactInfo {
  artifact_id: string;
  format: string;
  storage_kind: string;
  partial: boolean;
  producer: string | null;
}

export async function listStageArtifacts(stage: string | number): Promise<{
  stage: string;
  count: number;
  artifacts: DbArtifactInfo[];
}> {
  return get(`/artifacts/${encodeURIComponent(String(stage))}`);
}

/**
 * Trigger a browser download of a single artifact from artifacts.db.
 * The server materialises the file on demand (CSV / parquet / JSON /
 * GPKG / GeoJSON / joblib / pkl / npy).
 */
export function downloadDbArtifact(
  stage: string | number,
  artifactId: string,
  fmt?: "csv" | "parquet" | "json" | "gpkg" | "geojson" | "joblib" | "pkl" | "npy",
): void {
  const qs = fmt ? `?fmt=${encodeURIComponent(fmt)}` : "";
  const url =
    `${BASE}/artifacts/${encodeURIComponent(String(stage))}` +
    `/${encodeURIComponent(artifactId)}/download${qs}`;
  const a = document.createElement("a");
  a.href = url;
  // Server sets Content-Disposition with filename; leaving `download`
  // as a hint lets the browser respect the suggested name.
  a.download = `${artifactId}${fmt ? "." + fmt : ""}`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

/**
 * Bundle every artifact for a stage into a downloadable .zip and trigger
 * a browser download.  Uses POST so the server can lazily materialise.
 */
export async function downloadStageZip(stage: string | number): Promise<void> {
  const url = `${BASE}/artifacts/${encodeURIComponent(String(stage))}/export`;
  const res = await fetch(url, { method: "POST" });
  if (!res.ok) {
    throw new Error(`Stage export failed (${res.status}): ${await res.text()}`);
  }
  const blob = await res.blob();
  const objUrl = URL.createObjectURL(blob);
  try {
    const a = document.createElement("a");
    a.href = objUrl;
    a.download = `stage_${stage}_artifacts.zip`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  } finally {
    // Defer revocation so the browser has time to start the download.
    setTimeout(() => URL.revokeObjectURL(objUrl), 5000);
  }
}

// ------------------------------------------------------------------
// WebSocket helper
// ------------------------------------------------------------------
export function createPipelineSocket(): WebSocket {
  return new WebSocket("ws://127.0.0.1:8008/run/stream");
}

// ------------------------------------------------------------------
// Reproducibility (Phase 17)
// ------------------------------------------------------------------
export interface ProvenanceData {
  schema_version: number;
  timestamp_utc: string;
  config_hash: string;
  data: { path: string | null; sha256: string | null; size_bytes: number | null };
  git: { commit: string | null; dirty: boolean | null };
  env: Record<string, string>;
  extras?: Record<string, unknown>;
}

export interface ProvenanceResponse {
  provenance: ProvenanceData;
  frozen_config_present: boolean;
}

export const getProvenance = (runDir: string) =>
  get<ProvenanceResponse>(`/reproduce/provenance?run_dir=${encodeURIComponent(runDir)}`);

export const freezeCurrentRun = () =>
  post<{ path: string; output_dir: string }>("/reproduce/freeze");

export const loadFrozenRun = (runDir: string) =>
  post<{ loaded: boolean; config_hash: string | null; data: ProvenanceData["data"]; warnings: string[] }>(
    "/reproduce/load",
    { run_dir: runDir },
  );

export interface ReproduceVerifyResponse {
  config_hash_match: boolean;
  data_hash_match: boolean;
  git_commit_match: boolean;
  env_diffs: Record<string, { a: string | null; b: string | null }>;
  a: ProvenanceData;
  b: ProvenanceData;
  sidecars_b: {
    checked: { path: string; expected: string | null; actual: string | null; match: boolean }[];
    missing: string[];
    n_match: number;
    n_mismatch: number;
  };
}

export const verifyReproduction = (runA: string, runB: string) =>
  get<ReproduceVerifyResponse>(
    `/reproduce/verify?run_a=${encodeURIComponent(runA)}&run_b=${encodeURIComponent(runB)}`,
  );

// ------------------------------------------------------------------
// Scenario library (Phase 17)
// ------------------------------------------------------------------
export interface ScenarioLibraryEntry {
  id: string;
  parent_id: string | null;
  author: string;
  comment: string;
  created_utc: string;
  config_hash: string | null;
  scenario: Record<string, unknown>;
}

export interface ScenarioTimeline {
  entries: ScenarioLibraryEntry[];
  children: Record<string, string[]>;
  roots: string[];
  count: number;
}

export const getScenarioLibrary = () => get<ScenarioTimeline>("/scenarios/library");

// ---------------------------------------------------------------------------
// Block export (Phase C) — capture a UI block as PNG and persist it via the
// run registry so it shows up in the Report.
// ---------------------------------------------------------------------------

export interface BlockExportResponse {
  saved_to: string;
  size_bytes: number;
  registered: boolean;
  artifact?: Record<string, unknown>;
  registry_error?: string;
}

export function exportBlock(opts: {
  artifact_id: string;
  stage?: string;
  label?: string;
  png_b64: string;
}): Promise<BlockExportResponse> {
  return post<BlockExportResponse>("/results/export", opts);
}

export const appendScenarioToLibrary = (
  scenario: Record<string, unknown>,
  opts: { author?: string; comment?: string; parent_id?: string | null } = {},
) => post<ScenarioLibraryEntry>("/scenarios/library", { scenario, ...opts });

// ------------------------------------------------------------------
// Standalone HTML snapshot (Phase 17)
// ------------------------------------------------------------------
export async function downloadStandaloneSnapshot(opts: {
  runDir?: string;
  projectName?: string;
  chatHistory?: { role: string; content: string }[];
}): Promise<{ ok: boolean; filename?: string; error?: string }> {
  const body: Record<string, unknown> = {};
  if (opts.runDir) body.run_dir = opts.runDir;
  if (opts.projectName) body.project_name = opts.projectName;
  if (opts.chatHistory) body.chat_history = opts.chatHistory;
  const res = await fetch(`${BASE}/report/standalone`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    return { ok: false, error: txt || `HTTP ${res.status}` };
  }
  const blob = await res.blob();
  const cd = res.headers.get("Content-Disposition") || "";
  const m = /filename="([^"]+)"/.exec(cd);
  const filename = m ? m[1] : "sparc_snapshot.html";
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  return { ok: true, filename };
}

// ------------------------------------------------------------------
// Context layers (Phase 18)
// ------------------------------------------------------------------
export interface ContextLayer {
  id: string;
  name: string;
  kind: "basemap" | "overlay";
  url_template: string | null;
  attribution: string;
  description: string;
  domains: string[];
  default_for: string[];
  opacity_default?: number;
  tile_size?: number;
  max_zoom?: number;
  available: boolean;
  recommended?: boolean;
}

export interface ContextLayerCatalog {
  domain: string | null;
  layers: ContextLayer[];
  roadmap: ContextLayer[];
  defaults: string[];
  count: number;
}

export const getContextLayers = (domain?: string) =>
  get<ContextLayerCatalog>(`/context/layers${domain ? `?domain=${encodeURIComponent(domain)}` : ""}`);

// ------------------------------------------------------------------
// Audience-specific reports (Phase 19)
// ------------------------------------------------------------------
export type ReportAudience = "technical" | "planner" | "public";
export type ReportFormat = "md" | "html" | "pdf";

/** Download an audience-specific report. Returns the markdown/html text
 *  (so the caller can render an in-app preview) and triggers the file
 *  download for non-text formats. For "html" we also trigger the
 *  download but return the body text for inline preview. */
export async function downloadAudienceReport(opts: {
  audience: ReportAudience;
  fmt: ReportFormat;
  runDir?: string;
  preview?: boolean;  // when true, do not auto-download (used for inline preview)
}): Promise<{ ok: boolean; filename?: string; text?: string; error?: string }> {
  const body: Record<string, unknown> = {};
  if (opts.runDir) body.run_dir = opts.runDir;
  const url = new URL(`${BASE}/report/audience`);
  url.searchParams.set("audience", opts.audience);
  url.searchParams.set("fmt", opts.fmt);
  const res = await fetch(url.toString(), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    return { ok: false, error: txt || `HTTP ${res.status}` };
  }
  const cd = res.headers.get("Content-Disposition") || "";
  const m = /filename=([^;]+)/.exec(cd);
  const filename = m ? m[1].replace(/"/g, "").trim() : `sparc_${opts.audience}.${opts.fmt}`;
  if (opts.fmt === "pdf") {
    const blob = await res.blob();
    if (!opts.preview) {
      const u = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = u;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(u);
    }
    return { ok: true, filename };
  }
  const text = await res.text();
  if (!opts.preview) {
    const blob = new Blob([text], { type: opts.fmt === "html" ? "text/html" : "text/markdown" });
    const u = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = u;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(u);
  }
  return { ok: true, filename, text };
}
