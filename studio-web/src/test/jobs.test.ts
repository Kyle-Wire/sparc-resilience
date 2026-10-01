import { describe, expect, it } from "vitest";
import type { GlobalEvent, Job } from "../api/types";
import { ApiError } from "../api/client";
import { clearResources, mutate, peekResource } from "../api/resource";
import { activeJobs, mostImportantJob, reconcilePolledJobs, reduceGlobal, tabTitle, tagsForGlobalEvent, useJobs } from "../stores/jobs";
import { useUi } from "../stores/ui";

function job(p: Partial<Job>): Job {
  return {
    id: "j_1", kind: "run.core", lane: "heavy", executor: "process", label: "Fast run", status: "running", project_id: "p_1", run_id: "r1",
    study_id: null, scenario_id: null, parent_job_id: null, after_job_id: null, priority: 0, params: {}, created_utc: "2026-10-01T10:00:00Z",
    started_utc: null, finished_utc: null, progress: null, eta_s: null, eta_lo: null, eta_hi: null, stage: null, current_path: null, exit_code: null,
    error: null, blocked: null, result: null, peak_rss_mb: null, threads: null, ...p,
  };
}

const empty = { jobs: {}, progress: {}, host: { state: null, progress: null, rss_mb: null }, engineByRun: {}, storageLow: null, serverShutdown: false };

describe("jobs store (global SSE)", () => {
  it("applies job.created, job.progress and job.status", () => {
    const evs: GlobalEvent[] = [
      { type: "job.created", gseq: 1, ts: 100, job: job({ status: "queued" }) },
      { type: "job.status", gseq: 2, ts: 101, job_id: "j_1", status: "running", prev_status: "queued", exit_code: null, error: null, run_id: "r1", project_id: "p_1", kind: "run.core", label: "Fast run" },
      { type: "job.progress", gseq: 3, ts: 102, job_id: "j_1", frac: 0.42, eta_s: 300, eta_lo: 240, eta_hi: 360, stage: "S2_S3", path_tail: ["fold 4/5", "mgwr"] },
    ];
    const s = reduceGlobal(empty, evs);
    expect(s.jobs.j_1.status).toBe("running");
    expect(s.jobs.j_1.progress).toBe(0.42);
    expect(s.jobs.j_1.started_utc).not.toBeNull();
    expect(s.progress.j_1).toMatchObject({ frac: 0.42, stage: "S2_S3", path_tail: ["fold 4/5", "mgwr"] });
    const done = reduceGlobal(s, [{ type: "job.status", gseq: 4, ts: 200, job_id: "j_1", status: "succeeded", prev_status: "running", exit_code: 0, error: null, run_id: "r1", project_id: "p_1", kind: "run.core", label: "Fast run" }]);
    expect(done.jobs.j_1.status).toBe("succeeded");
    expect(done.jobs.j_1.progress).toBe(1);
    expect(done.jobs.j_1.finished_utc).toBe(new Date(200000).toISOString());
  });

  it("creates a job from job.status when job.created was missed", () => {
    const s = reduceGlobal(empty, [{ type: "job.status", gseq: 9, ts: 5, job_id: "j_x", status: "running", prev_status: "starting", exit_code: null, error: null, run_id: null, project_id: null, kind: "input.forcing", label: "Forcing" }]);
    expect(s.jobs.j_x).toMatchObject({ kind: "input.forcing", label: "Forcing", status: "running" });
  });

  it("tracks engine state per run and for the host", () => {
    const s = reduceGlobal(empty, [
      { type: "engine.status", gseq: 1, ts: 1, run_id: null, state: "ready", progress: null, rss_mb: 900 },
      { type: "engine.status", gseq: 2, ts: 1, run_id: "r1", state: "loading", progress: 0.6, rss_mb: null },
    ]);
    expect(s.host.state).toBe("ready");
    expect(s.engineByRun.r1).toEqual({ state: "loading", progress: 0.6, rss_mb: null });
  });

  it("maps events to resource tags", () => {
    expect(tagsForGlobalEvent({ type: "output.written", gseq: 1, ts: 1, run_id: "r1", relpath: "predictions.parquet", role: "predictions", output_id: "predictions", stage: "S2_S3", bytes: 10 })).toContain("run:r1:outputs");
    expect(tagsForGlobalEvent({ type: "scenario.result", gseq: 1, ts: 1, scenario_id: "sc_1", result_id: "res_1", run_id: "r1", kind: "exact" })).toContain("scenario:sc_1");
    expect(tagsForGlobalEvent({ type: "job.status", gseq: 1, ts: 1, job_id: "j_1", status: "failed", prev_status: "running", exit_code: 1, error: null, run_id: "r1", project_id: "p_1", kind: "run.core", label: "x" })).toEqual(["jobs", "job:j_1", "run:r1", "project:p_1"]);
  });

  it("orders active jobs and mirrors the top one in the tab title", () => {
    const jobs = {
      a: job({ id: "a", label: "Forcing", lane: "network", status: "running", created_utc: "2026-10-01T09:00:00Z" }),
      b: job({ id: "b", label: "Full run", lane: "heavy", status: "running", created_utc: "2026-10-01T10:00:00Z", stage: "S2_S3" }),
      c: job({ id: "c", label: "Queued", status: "queued" }),
      d: job({ id: "d", label: "Done", status: "succeeded" }),
    };
    expect(activeJobs(jobs).map((j) => j.id)).toEqual(["b", "a", "c"]);
    expect(mostImportantJob(jobs)?.id).toBe("b");
    const progress = { b: { frac: 0.42, eta_s: 1, eta_lo: 1, eta_hi: 1, stage: "S2_S3", path_tail: [], ts: 1 } };
    expect(tabTitle(jobs, progress, "Accuracy")).toBe("(2 running) ▶ 42% S2_S3 · SPARC Studio");
    expect(tabTitle({ b: jobs.b }, progress, "Accuracy")).toBe("▶ 42% S2_S3 · SPARC Studio");
    expect(tabTitle({ d: jobs.d }, {}, "Accuracy")).toBe("Accuracy · SPARC Studio");
  });
});

describe("jobs store while polling", () => {
  it("a job that leaves the active list is fetched, stored final, toasted and its resources refreshed", async () => {
    useJobs.getState().reset();
    useUi.setState({ toasts: [] });
    useJobs.getState().setActive([job({ id: "j_1", status: "running" }), job({ id: "j_2", status: "queued", run_id: "r2", label: "Planner pack" })]);
    mutate("run:r1", "old run detail"); // unmounted cached entries are dropped on invalidation
    const fetched: string[] = [];
    await reconcilePolledJobs([job({ id: "j_2", status: "running", run_id: "r2", label: "Planner pack" })], async (id) => {
      fetched.push(id);
      return job({ id, status: "succeeded", result: { r2: 0.81 } });
    });
    expect(fetched).toEqual(["j_1"]);
    const s = useJobs.getState();
    expect(s.jobs.j_1.status).toBe("succeeded");
    expect(s.jobs.j_2.status).toBe("running");
    expect(activeJobs(s.jobs).map((j) => j.id)).toEqual(["j_2"]);
    expect(useUi.getState().toasts.map((t) => t.title)).toEqual(["Fast run finished"]);
    expect(peekResource("run:r1")).toBeUndefined();
    clearResources();
  });

  it("a job that disappeared (deleted) is dropped quietly", async () => {
    useJobs.getState().reset();
    useUi.setState({ toasts: [] });
    useJobs.getState().setActive([job({ id: "j_1", status: "running" })]);
    await reconcilePolledJobs([], async () => {
      throw new ApiError(404, "not_found", "no job");
    });
    expect(useJobs.getState().jobs.j_1).toBeUndefined();
    expect(useUi.getState().toasts).toEqual([]);
  });
});
