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

// ------------------------------------------------------------------
// WebSocket helper
// ------------------------------------------------------------------
export function createPipelineSocket(): WebSocket {
  return new WebSocket("ws://127.0.0.1:8008/run/stream");
}
