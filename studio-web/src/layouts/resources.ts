// Shell-level resources shared by the layouts and feature pages (same cache keys everywhere,
// so a project or run is fetched once per change).
import { api } from "../api/client";
import { useResource } from "../api/resource";
import type { Job, Page, Project, ReadinessRow, RunOutputs, RunStatus, RunSummary } from "../api/types";

/** `GET /api/projects/{pid}` (api.md §5). */
export type ProjectDetail = {
  project: Project;
  readiness: ReadinessRow[];
  runs: RunSummary[];
  active_jobs: Job[];
  config_version: number;
};

/** The parts of `GET /api/runs/{rid}` (api.md §6 RunDetail) the shell uses. */
export type RunDetailHead = {
  run: RunSummary;
  header: {
    name: string;
    created_utc: string | null;
    git_commit: string | null;
    git_dirty: boolean | null;
    n_points: number | null;
    grid_shape: [number, number] | null;
    cell_m: number | null;
    fast: boolean;
    coarse_m: number | null;
    run_dir: string;
    demo: boolean;
  };
  checkpoint?: { present: boolean } | null;
};

export function useProject(pid: string | null) {
  return useResource<ProjectDetail>(pid ? `project:${pid}` : null, (s) => api.get<ProjectDetail>(`/api/projects/${encodeURIComponent(pid!)}`, undefined, s), {
    tags: pid ? [`project:${pid}`, "projects"] : [],
  });
}

export function useProjects() {
  return useResource<Project[]>("projects", (s) => api.get<Project[]>("/api/projects", undefined, s), { tags: ["projects"] });
}

export function useRunDetail(rid: string | null) {
  return useResource<RunDetailHead>(rid ? `run:${rid}` : null, (s) => api.get<RunDetailHead>(`/api/runs/${encodeURIComponent(rid!)}`, undefined, s), {
    tags: rid ? [`run:${rid}`] : [],
  });
}

/** `GET /api/runs/{rid}/outputs`: catalog states and the per-tab status dots. */
export function useRunOutputs(rid: string | null) {
  return useResource<RunOutputs>(rid ? `run:${rid}:outputs` : null, (s) => api.get<RunOutputs>(`/api/runs/${encodeURIComponent(rid!)}/outputs`, undefined, s), {
    tags: rid ? [`run:${rid}`, `run:${rid}:outputs`] : [],
  });
}

export function useRecentRuns(enabled: boolean) {
  return useResource<Page<RunSummary>>(enabled ? "runs:recent" : null, (s) => api.get<Page<RunSummary>>("/api/runs", { limit: 50, sort: "created_desc" }, s), { tags: ["runs"] });
}

/** Run statuses that can still change (a job is or may be writing). */
export function runIsLive(status: RunStatus | null | undefined): boolean {
  return status === "running" || status === "queued" || status === "external_live";
}
