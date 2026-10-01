// Global job list fed by the global SSE stream (SPEC §5.7, §12.4).
//
// `connectJobs()` (called once by the shell) opens the global stream, loads the active
// jobs, applies `job.*` and `engine.status` events, invalidates resource tags for run,
// scenario and study events, refetches everything on `resync`, and raises a toast (plus an
// opt-in browser notification) when a job ends. While the stream is down and the manager
// polls the active list, a job that leaves the list is fetched once so its end is still
// toasted and its run, project and job resources are refreshed. The tracker of one job (Mission Control)
// lives in stores/tracker.ts (frontend-tracking); this store only holds the 1 Hz summary.
import { create } from "zustand";
import { api, ApiError } from "../api/client";
import { invalidate, invalidateAll } from "../api/resource";
import { getStreams, type StreamManager } from "../api/sse";
import {
  FINAL_JOB_STATUSES,
  isActiveStatus,
  type EngineHostState,
  type EngineState,
  type GlobalEvent,
  type Job,
  type JobStatus,
  type Page,
} from "../api/types";
import { fmtDuration, fmtNum, fmtPct } from "../theme/format";
import { useUi } from "./ui";

export type JobProgress = {
  frac: number | null;
  eta_s: number | null;
  eta_lo: number | null;
  eta_hi: number | null;
  stage: string | null;
  path_tail: string[];
  ts: number;
};

export type EngineInfo = { state: EngineState | EngineHostState | null; progress: number | null; rss_mb: number | null };

type JobsState = {
  jobs: Record<string, Job>;
  progress: Record<string, JobProgress>;
  host: EngineInfo;
  engineByRun: Record<string, EngineInfo>;
  storageLow: { free_bytes: number; threshold_bytes: number } | null;
  serverShutdown: boolean;
  loaded: boolean;
  setActive: (jobs: Job[]) => void;
  upsert: (job: Job) => void;
  remove: (id: string) => void;
  applyGlobal: (events: GlobalEvent[]) => void;
  setHost: (state: EngineHostState) => void;
  reset: () => void;
};

const initial = {
  jobs: {} as Record<string, Job>,
  progress: {} as Record<string, JobProgress>,
  host: { state: null, progress: null, rss_mb: null } as EngineInfo,
  engineByRun: {} as Record<string, EngineInfo>,
  storageLow: null,
  serverShutdown: false,
  loaded: false,
};

/** Pure reducer for one batch of global events (exported for tests). */
export function reduceGlobal(s: Pick<JobsState, "jobs" | "progress" | "host" | "engineByRun" | "storageLow" | "serverShutdown">, events: GlobalEvent[]) {
  let jobs = s.jobs;
  let progress = s.progress;
  let host = s.host;
  let engineByRun = s.engineByRun;
  let storageLow = s.storageLow;
  let serverShutdown = s.serverShutdown;
  for (const ev of events) {
    switch (ev.type) {
      case "job.created":
        jobs = { ...jobs, [ev.job.id]: ev.job };
        break;
      case "job.status": {
        const prev = jobs[ev.job_id];
        const base: Job = prev ?? {
          id: ev.job_id, kind: ev.kind, lane: "none", executor: "process", label: ev.label, status: ev.status,
          project_id: ev.project_id, run_id: ev.run_id, study_id: null, scenario_id: null, parent_job_id: null,
          after_job_id: null, priority: 0, params: {}, created_utc: new Date(ev.ts * 1000).toISOString(),
          started_utc: null, finished_utc: null, progress: null, eta_s: null, eta_lo: null, eta_hi: null, stage: null,
          current_path: null, exit_code: null, error: null, blocked: null, result: null, peak_rss_mb: null, threads: null,
        };
        const final = FINAL_JOB_STATUSES.includes(ev.status);
        jobs = {
          ...jobs,
          [ev.job_id]: {
            ...base,
            status: ev.status,
            exit_code: ev.exit_code ?? base.exit_code,
            error: ev.error ?? base.error,
            finished_utc: final ? base.finished_utc ?? new Date(ev.ts * 1000).toISOString() : base.finished_utc,
            started_utc: ev.status === "running" && !base.started_utc ? new Date(ev.ts * 1000).toISOString() : base.started_utc,
            progress: ev.status === "succeeded" ? 1 : base.progress,
          },
        };
        break;
      }
      case "job.progress": {
        progress = {
          ...progress,
          [ev.job_id]: { frac: ev.frac, eta_s: ev.eta_s, eta_lo: ev.eta_lo, eta_hi: ev.eta_hi, stage: ev.stage, path_tail: ev.path_tail ?? [], ts: ev.ts },
        };
        const j = jobs[ev.job_id];
        if (j) jobs = { ...jobs, [ev.job_id]: { ...j, progress: ev.frac, eta_s: ev.eta_s, eta_lo: ev.eta_lo, eta_hi: ev.eta_hi, stage: ev.stage } };
        break;
      }
      case "engine.status": {
        const info: EngineInfo = { state: ev.state, progress: ev.progress, rss_mb: ev.rss_mb };
        if (ev.run_id) engineByRun = { ...engineByRun, [ev.run_id]: info };
        else host = info;
        break;
      }
      case "storage.low":
        storageLow = { free_bytes: ev.free_bytes, threshold_bytes: ev.threshold_bytes };
        break;
      case "server_shutdown":
        serverShutdown = true;
        break;
      default:
        break;
    }
  }
  return { jobs, progress, host, engineByRun, storageLow, serverShutdown };
}

/** Resource tags a global event makes stale (SPEC §12.4). */
export function tagsForGlobalEvent(ev: GlobalEvent): string[] {
  switch (ev.type) {
    case "job.created":
      return ["jobs", ...(ev.job.run_id ? [`run:${ev.job.run_id}:jobs`] : []), ...(ev.job.project_id ? [`project:${ev.job.project_id}`] : [])];
    case "job.status": {
      const t = ["jobs", `job:${ev.job_id}`];
      if (ev.run_id) t.push(`run:${ev.run_id}`);
      if (ev.project_id) t.push(`project:${ev.project_id}`);
      return t;
    }
    case "run.updated":
      return [`run:${ev.run_id}`, "runs", ...(ev.project_id ? [`project:${ev.project_id}`] : [])];
    case "run.indexed":
      return ["runs", `run:${ev.run_id}`, ...(ev.project_id ? [`project:${ev.project_id}`] : []), "projects"];
    case "output.written":
      return [`run:${ev.run_id}:outputs`, `run:${ev.run_id}:views`, `run:${ev.run_id}:layers`, `run:${ev.run_id}:files`];
    case "engine.status":
      return ev.run_id ? [`run:${ev.run_id}:engine`, "engine"] : ["engine"];
    case "scenario.result":
      return [...(ev.scenario_id ? [`scenario:${ev.scenario_id}`] : []), `run:${ev.run_id}:results`, `run:${ev.run_id}:layers`];
    case "study.updated":
      return [`study:${ev.study_id}`, "studies", ...(ev.target_run_id ? [`run:${ev.target_run_id}:studies`] : []), ...(ev.project_id ? [`project:${ev.project_id}`] : [])];
    case "storage.low":
      return ["storage"];
    default:
      return [];
  }
}

export const useJobs = create<JobsState>((set, get) => ({
  ...initial,
  setActive: (list) => {
    // Replace the active set: drop jobs no longer active unless we already know they ended.
    const jobs: Record<string, Job> = {};
    for (const [id, j] of Object.entries(get().jobs)) if (!isActiveStatus(j.status)) jobs[id] = j;
    for (const j of list) jobs[j.id] = j;
    set({ jobs, loaded: true });
  },
  upsert: (job) => set({ jobs: { ...get().jobs, [job.id]: job } }),
  remove: (id) => {
    const jobs = { ...get().jobs };
    delete jobs[id];
    set({ jobs });
  },
  applyGlobal: (events) => set(reduceGlobal(get(), events)),
  setHost: (state) => set({ host: { ...get().host, state } }),
  reset: () => set({ ...initial }),
}));

const LANE_RANK: Record<string, number> = { heavy: 0, engine: 1, medium: 2, network: 3, none: 4 };

/** Active jobs, running first (heavy lane first), then queued, oldest first within a group. */
export function activeJobs(jobs: Record<string, Job>): Job[] {
  const rank = (s: JobStatus) => (s === "running" || s === "cancelling" ? 0 : s === "starting" ? 1 : s === "blocked" ? 3 : 2);
  return Object.values(jobs)
    .filter((j) => isActiveStatus(j.status))
    .sort((a, b) => rank(a.status) - rank(b.status) || (LANE_RANK[a.lane] ?? 5) - (LANE_RANK[b.lane] ?? 5) || a.created_utc.localeCompare(b.created_utc));
}

/** The job the browser tab title mirrors: the first running job in `activeJobs` order. */
export function mostImportantJob(jobs: Record<string, Job>): Job | null {
  return activeJobs(jobs).find((j) => j.status === "running" || j.status === "cancelling" || j.status === "starting") ?? null;
}

/** "▶ 42% S2_S3 · SPARC Studio" (plus "(2 running)" when several run). */
export function tabTitle(jobs: Record<string, Job>, progress: Record<string, JobProgress>, pageTitle: string | null): string {
  const top = mostImportantJob(jobs);
  if (!top) return pageTitle ? `${pageTitle} · SPARC Studio` : "SPARC Studio";
  const p = progress[top.id];
  const frac = p?.frac ?? top.progress;
  const stage = p?.stage ?? top.stage;
  const running = activeJobs(jobs).filter((j) => j.status === "running" || j.status === "cancelling").length;
  const parts = ["▶", frac !== null && frac !== undefined ? fmtPct(frac) : top.status, stage ?? top.label];
  return `${running > 1 ? `(${running} running) ` : ""}${parts.join(" ")} · SPARC Studio`;
}

/** Active (queued or running) jobs for one run. */
export function jobsForRun(jobs: Record<string, Job>, rid: string): Job[] {
  return activeJobs(jobs).filter((j) => j.run_id === rid);
}

function headline(job: Job): string {
  const r = job.result ?? {};
  const bits: string[] = [];
  const num = (k: string) => (typeof r[k] === "number" ? (r[k] as number) : null);
  const r2 = num("r2");
  if (r2 !== null) bits.push(`R² ${fmtNum(r2, 3)}`);
  const rmse = num("rmse");
  if (rmse !== null) bits.push(`RMSE ${fmtNum(rmse, 2)}`);
  const el = num("elapsed_s");
  if (el !== null) bits.push(fmtDuration(el));
  return bits.join(" · ");
}

function notifyEnd(job: Job): void {
  const ui = useUi.getState();
  const kind = job.status === "succeeded" ? "success" : job.status === "failed" ? "error" : "warning";
  const verb = { succeeded: "finished", failed: "failed", cancelled: "was cancelled", interrupted: "was interrupted" }[job.status as string] ?? job.status;
  const body = job.status === "failed" ? job.error?.message ?? "" : headline(job);
  ui.pushToast({ kind, title: `${job.label} ${verb}`, body: body || undefined, href: `/jobs/${job.id}`, linkLabel: "Open" });
  ui.announce(`${job.label} ${verb}`);
  if (ui.notifications && typeof Notification !== "undefined" && Notification.permission === "granted" && typeof document !== "undefined" && document.hidden) {
    try {
      new Notification(`SPARC Studio: ${job.label} ${verb}`, { body: body || undefined, tag: job.id });
    } catch {
      /* notifications unavailable */
    }
  }
}

/** Resource tags a job status change makes stale (the same as a `job.status` event). */
function tagsForJob(job: Pick<Job, "id" | "run_id" | "project_id">): string[] {
  const t = ["jobs", `job:${job.id}`];
  if (job.run_id) t.push(`run:${job.run_id}`);
  if (job.project_id) t.push(`project:${job.project_id}`);
  return t;
}

/**
 * Apply one polled active-job list (global stream in polling fallback). Jobs that were
 * active and are missing from the list have ended (or were deleted): each is fetched once,
 * stored with its final status, toasted and its tags invalidated; status changes of jobs
 * still listed invalidate their tags as a `job.status` event would. Exported for tests.
 */
export async function reconcilePolledJobs(list: Job[], fetchJob: (id: string) => Promise<Job> = (id) => api.get<Job>(`/api/jobs/${encodeURIComponent(id)}`)): Promise<void> {
  const before = useJobs.getState().jobs;
  useJobs.getState().setActive(list);
  const listed = new Set(list.map((j) => j.id));
  const tags = new Set<string>();
  for (const j of list) {
    const prev = before[j.id];
    if (!prev || prev.status !== j.status) for (const t of tagsForJob(j)) tags.add(t);
  }
  const gone = Object.values(before).filter((j) => isActiveStatus(j.status) && !listed.has(j.id));
  for (const t of tags) invalidate(t);
  await Promise.all(
    gone.map(async (old) => {
      let j: Job;
      try {
        j = await fetchJob(old.id);
      } catch (e) {
        // Deleted, or the server is unreachable: leave it out of the active set either way.
        if (e instanceof ApiError && e.status === 404) for (const t of tagsForJob(old)) invalidate(t);
        return;
      }
      useJobs.getState().upsert(j);
      for (const t of tagsForJob(j)) invalidate(t);
      if (FINAL_JOB_STATUSES.includes(j.status)) notifyEnd(j);
    }),
  );
}

/** Reload the active jobs (start-up and after a `resync`, when events may have been lost). */
async function refreshActive(): Promise<void> {
  let page: Page<Job>;
  try {
    page = await api.get<Page<Job>>("/api/jobs", { status: "active", limit: 500 });
  } catch {
    return; // the connection pill reports the outage
  }
  await reconcilePolledJobs(page.items ?? []);
}

/**
 * Wire the global stream into this store (idempotent per manager). Returns a disconnect
 * function. The shell calls it once on mount.
 */
export function connectJobs(manager: StreamManager = getStreams()): () => void {
  const offs: (() => void)[] = [];
  offs.push(
    manager.onGlobal((events) => {
      const before = useJobs.getState().jobs;
      useJobs.getState().applyGlobal(events);
      const tags = new Set<string>();
      for (const ev of events) for (const t of tagsForGlobalEvent(ev)) tags.add(t);
      for (const t of tags) invalidate(t);
      for (const ev of events) {
        if (ev.type === "job.status" && FINAL_JOB_STATUSES.includes(ev.status) && before[ev.job_id]?.status !== ev.status) {
          // Fetch the full job for its result summary, then toast.
          api
            .get<Job>(`/api/jobs/${encodeURIComponent(ev.job_id)}`)
            .then((j) => {
              useJobs.getState().upsert(j);
              notifyEnd(j);
            })
            .catch(() => {
              const j = useJobs.getState().jobs[ev.job_id];
              if (j) notifyEnd(j);
            });
        } else if (ev.type === "job.status" && ev.status === "running" && before[ev.job_id]?.status !== "running") {
          useUi.getState().announce(`${ev.label} started`);
        } else if (ev.type === "storage.low") {
          useUi.getState().pushToast({ kind: "warning", title: "Disk space is low", body: "Free space in Settings › Storage.", href: "/settings" });
        } else if (ev.type === "server_shutdown") {
          useUi.getState().pushToast({ kind: "warning", title: "The Studio server is shutting down", body: ev.stop_jobs ? "Running jobs are being stopped." : "Running jobs keep going and are reattached on the next start.", timeout: 0 });
        }
      }
    }),
  );
  offs.push(
    manager.onResync(() => {
      void refreshActive();
      invalidateAll();
    }),
  );
  offs.push(manager.onPollJobs((list) => void reconcilePolledJobs(list)));
  manager.startGlobal();
  void refreshActive();
  api
    .get<{ engine?: { state: EngineHostState } }>("/api/health")
    .then((h) => h.engine && useJobs.getState().setHost(h.engine.state))
    .catch(() => {});
  return () => {
    for (const off of offs) off();
  };
}
