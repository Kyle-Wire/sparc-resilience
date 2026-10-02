// Exports and the report builder (api.md §8 export params, §10; SPEC §6.7): `POST /api/exports`
// creates the exports row and its `export.<kind>` job (the server fills `export_id`), the
// project's export history, downloads, deletes and the live report preview.
import { api } from "./client";
import { useResource } from "./resource";
import type { Job } from "./types";

const enc = encodeURIComponent;

export type ExportKind = "bundle" | "gis" | "page" | "report" | "findings" | "decision_pack" | "plan_pack" | "compare_pack";

export const EXPORT_LABELS: Record<ExportKind, string> = {
  bundle: "Run bundle (ZIP)",
  gis: "GIS pack",
  page: "Standalone results page",
  report: "Project report",
  findings: "Findings",
  decision_pack: "Decision pack",
  plan_pack: "Plan pack",
  compare_pack: "Compare pack",
};

/** `Export` (api.md §10). `draft` comes from the pack's options (preview-only packs). */
export type Export = {
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

export type ReportSection =
  | "summary" | "accuracy" | "validation" | "scenarios" | "plans" | "climate"
  | "equity" | "caveats" | "limitations" | "provenance" | "findings";

export const REPORT_SECTIONS: readonly { id: ReportSection; label: string; hint: string }[] = [
  { id: "summary", label: "Summary", hint: "Headline numbers written from the run" },
  { id: "accuracy", label: "Accuracy", hint: "Held-out skill, interval coverage, baselines" },
  { id: "validation", label: "Validation", hint: "Placebo, simulation check, multiverse, reproduction" },
  { id: "scenarios", label: "Scenarios", hint: "Configured scenarios and the selected exact results" },
  { id: "plans", label: "Plans", hint: "The selected budget plans" },
  { id: "climate", label: "Climate", hint: "Mid-century warming and exposure" },
  { id: "equity", label: "Equity", hint: "Who the cooling reaches" },
  { id: "caveats", label: "Caveats", hint: "Generated from the numbers" },
  { id: "limitations", label: "Limitations", hint: "From the model card" },
  { id: "provenance", label: "Provenance", hint: "Run, commit and hashes" },
  { id: "findings", label: "Findings", hint: "The selected pinned findings" },
];

// -- params of each export kind, without the server-filled export_id (api.md §8)

export type BundleParams = { run_id: string; outputs?: string[]; include_checkpoint?: boolean };
export type GisParams = { run_id: string; layers?: string[] };
export type PageParams = { run_id: string; placebo_study?: string };
export type ReportParams = { run_id: string; sections: ReportSection[]; result_ids?: string[]; plan_ids?: string[]; finding_ids?: string[]; format: "html" | "md" };
export type FindingsExportParams = { project_id: string; run_id?: string; ids?: string[]; format: "md" | "html" };

export type ExportParams = {
  bundle: BundleParams;
  gis: GisParams;
  page: PageParams;
  report: ReportParams;
  findings: FindingsExportParams;
  decision_pack: { result_id: string; thresholds?: number[] };
  plan_pack: { plan_id: string };
  compare_pack: { comparison_id: string };
};

export type ReportPreviewBody = { run_id: string; sections: ReportSection[]; result_ids?: string[]; plan_ids?: string[]; finding_ids?: string[] };

// ---------------------------------------------------------------- endpoints

export function createExport<K extends ExportKind>(pid: string, kind: K, params: ExportParams[K]): Promise<{ export: Export; job: Job }> {
  return api.post<{ export: Export; job: Job }>("/api/exports", { kind, project_id: pid, params });
}

export function listExports(pid: string, signal?: AbortSignal): Promise<Export[]> {
  return api.get<Export[]>(`/api/projects/${enc(pid)}/exports`, undefined, signal);
}

export function getExport(eid: string, signal?: AbortSignal): Promise<Export> {
  return api.get<Export>(`/api/exports/${enc(eid)}`, undefined, signal);
}

/** The streamed download (`409 not_ready` until the job finishes). */
export function exportDownloadUrl(eid: string): string {
  return `/api/exports/${enc(eid)}/download`;
}

export function deleteExport(eid: string): Promise<{ ok: true }> {
  return api.del<{ ok: true }>(`/api/exports/${enc(eid)}`);
}

export function previewReport(pid: string, body: ReportPreviewBody, signal?: AbortSignal): Promise<{ html: string }> {
  return api.post<{ html: string }>(`/api/projects/${enc(pid)}/report/preview`, body, { signal });
}

// ---------------------------------------------------------------- hooks

/**
 * The project's export history. Tagged under `project:<pid>`, so the `job.status` events of
 * an export job (they carry the project id) refresh it when the file is ready.
 */
export function useExports(pid: string | null) {
  return useResource<Export[]>(pid ? `project:${pid}:exports` : null, (s) => listExports(pid!, s), {
    tags: pid ? [`project:${pid}:exports`, "exports"] : [],
    keepPrevious: true,
  });
}

/** One export, refreshed when its job changes state (`job:<jid>`). */
export function useExport(eid: string | null, jobId: string | null) {
  return useResource<Export>(eid ? `export:${eid}` : null, (s) => getExport(eid!, s), {
    tags: eid ? [`export:${eid}`, "exports", ...(jobId ? [`job:${jobId}`] : [])] : [],
  });
}
