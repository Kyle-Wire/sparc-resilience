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

export const listTemplates = () =>
  get<{ templates: TemplateInfo[] }>("/project/templates");

// ------------------------------------------------------------------
// Data
// ------------------------------------------------------------------
export const dataSummary = () => get<DataSummary>("/data/summary");

export const dataPreview = (n = 50) => get<DataPreview>(`/data/preview?n=${n}`);

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
// Scenarios
// ------------------------------------------------------------------
export const runScenarios = () => post<{ status: string; n_scenarios: number; summary_rows: number }>("/scenarios/run");

export const scenarioResults = (format: "json" | "geojson" = "geojson") =>
  get<unknown>(`/scenarios/results?format=${format}`);

// ------------------------------------------------------------------
// DAG
// ------------------------------------------------------------------
export const getDag = () => get<DagDefinition>("/dag");

export const validateDag = (dag: DagDefinition) =>
  post<DagValidation>("/dag/validate", dag);

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

export const getPdpCurves = () =>
  get<PdpCurves>("/results/pdp_curves");

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
// WebSocket helper
// ------------------------------------------------------------------
export function createPipelineSocket(): WebSocket {
  return new WebSocket("ws://127.0.0.1:8008/run/stream");
}
