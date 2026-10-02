// Wire fixtures for the studies, exports and findings pages, written from api.md §8–11 (study
// status rows, Study, per-kind study views, Export, Finding) and the core outputs the views
// summarise (simcheck_summary.json, multiverse_summary.json, placebo.json).
import type { Export } from "../../../api/exports";
import type { SimcheckView, Study, StudyKind, StudyStatusRow } from "../../../api/studies";
import type { Finding, Job, OutputEntry, Project, RunSummary } from "../../../api/types";

export const PID = "p_st";
export const RID = "20261001-142233-fast-ab12";

export function job(id: string, kind: string, extra: Partial<Job> = {}): Job {
  return {
    id,
    kind,
    lane: "heavy",
    executor: "process",
    label: kind,
    status: "queued",
    project_id: PID,
    run_id: RID,
    study_id: null,
    scenario_id: null,
    parent_job_id: null,
    after_job_id: null,
    priority: 0,
    params: {},
    created_utc: "2026-10-02T10:00:00Z",
    started_utc: null,
    finished_utc: null,
    progress: null,
    eta_s: null,
    eta_lo: null,
    eta_hi: null,
    stage: null,
    current_path: null,
    exit_code: null,
    error: null,
    blocked: null,
    result: null,
    peak_rss_mb: null,
    threads: null,
    ...extra,
  };
}

export function run(id = RID, extra: Partial<RunSummary> = {}): RunSummary {
  return {
    id,
    project_id: PID,
    label: "Fast demo run",
    origin: "studio",
    status: "complete",
    mode: "fast",
    coarse_m: null,
    created_utc: "2026-10-01T14:22:33Z",
    finished_utc: "2026-10-01T14:25:01Z",
    duration_s: 148,
    n_points: 2400,
    r2: 0.81,
    rmse: 0.34,
    coverage: 0.9,
    n_scenarios: 8,
    checkpoint_bytes: 350 * 1024 * 1024,
    has_emulator: false,
    studies: [],
    git_commit: "5a04f44",
    git_dirty: false,
    demo: true,
    pinned: false,
    parent_run_id: null,
    study_id: null,
    last_job_id: null,
    ...extra,
  };
}

export function project(extra: Partial<Project> = {}): Project {
  return {
    id: PID,
    slug: "demo",
    name: "Demo city",
    dir: "/w/projects/demo",
    config_path: "/w/projects/demo/config.yml",
    template: "synthetic",
    demo: true,
    active_run_id: RID,
    archived: false,
    created_utc: "2026-10-01T10:00:00Z",
    updated_utc: "2026-10-01T10:00:00Z",
    report: { title: null, place: null, area: null },
    headline_scenario: null,
    cost_model: {},
    n_runs: 1,
    last_run: { id: RID, status: "complete", created_utc: "2026-10-01T14:22:33Z", r2: 0.81, has_checkpoint: true },
    active_jobs: 0,
    readiness_score: { done: 8, total: 10 },
    ...extra,
  };
}

export function projectDetail(runs: RunSummary[] = [run()]) {
  return { project: project(), readiness: [], runs, active_jobs: [], config_version: 3 };
}

export function runDetail(rid = RID, extra: Partial<RunSummary> = {}) {
  return {
    run: run(rid, extra),
    header: {
      name: "synthetic_demo",
      created_utc: "2026-10-01T14:22:33Z",
      git_commit: "5a04f44",
      git_dirty: false,
      n_points: 2400,
      grid_shape: [48, 50],
      cell_m: 30,
      fast: true,
      coarse_m: null,
      run_dir: `/w/projects/demo/runs/${rid}`,
      demo: extra.demo ?? true,
    },
    checkpoint: { present: true },
  };
}

export function statusRow(kind: StudyKind, extra: Partial<StudyStatusRow> = {}): StudyStatusRow {
  return {
    kind,
    state: "not_run",
    study_id: null,
    job_id: null,
    updated_utc: null,
    headline: null,
    estimate: { est_s: 300, est_lo: 240, est_hi: 420 },
    attached: null,
    action: null,
    requirements: { ok: true, missing: [] },
    ...extra,
  };
}

export function study(id: string, kind: string, extra: Partial<Study> = {}): Study {
  return {
    id,
    project_id: PID,
    kind,
    target_run_id: RID,
    job_id: null,
    out_dir: `/w/projects/demo/studies/${id}`,
    status: "succeeded",
    params: {},
    summary: null,
    origin: "studio",
    created_utc: "2026-10-01T15:00:00Z",
    updated_utc: "2026-10-01T16:00:00Z",
    children: [],
    attached_runs: [],
    stale_vs: [],
    ...extra,
  };
}

/**
 * A simcheck view mid-study: physics has a share of exactly 1, a 1.6 replicate that needed a
 * gate redraw, and a running replicate; additive has 0.5 and one pending; the null generator
 * has a finished replicate (no share) and an error.
 */
export function simcheckView(): SimcheckView {
  return {
    design: { physics: 3, additive: 2, own_only: 0, coarse_scale: 0, confounded: 0, null: 2 },
    grid: [
      { generator: "physics", seed: 0, status: "done", share: 1.0, ci_covers: true, causal_covers: true, seconds: 40, gate_attempt: 0 },
      { generator: "physics", seed: 1, status: "done", share: 1.6, ci_covers: false, causal_covers: true, seconds: 44, gate_attempt: 1 },
      { generator: "physics", seed: 2, status: "running", share: null, ci_covers: null, causal_covers: null, seconds: null, gate_attempt: null },
      { generator: "additive", seed: 0, status: "done", share: 0.5, ci_covers: true, causal_covers: false, seconds: 38, gate_attempt: 0 },
      { generator: "null", seed: 0, status: "done", share: null, ci_covers: true, causal_covers: true, seconds: 35, gate_attempt: 0 },
      { generator: "null", seed: 1, status: "error", share: null, ci_covers: null, causal_covers: null, seconds: 3, gate_attempt: 0 },
    ],
    generators: {
      physics: { n: 2, n_gate_pass: 2, share_median: 1.3, share_iqr: [1.15, 1.45], rank_corr_mean: 0.8, ci_coverage: 0.5, causal_ci_coverage: 1, interval_coverage_mean: 0.9, oof_r2_mean: 0.7 },
      additive: { n: 1, n_gate_pass: 1, share_median: 0.5, share_iqr: [0.5, 0.5], rank_corr_mean: 0.6, ci_coverage: 1, causal_ci_coverage: 0, interval_coverage_mean: 0.88, oof_r2_mean: 0.66 },
      null: {
        n: 1,
        n_gate_pass: 1,
        share_median: null,
        share_iqr: null,
        rank_corr_mean: null,
        ci_coverage: 1,
        causal_ci_coverage: 1,
        interval_coverage_mean: 0.9,
        oof_r2_mean: 0.7,
        false_positive_rate: 0,
        causal_false_positive_rate: 0,
        null_mean_delta: -0.01,
        null_mean_delta_se: null,
        null_causal_pp: 0.0,
      },
    },
    bias_correction: { share_median_across_generators: 0.9, share_range: [0.5, 1.3], relative_spread: 0.89, stable: false, n_generators: 2, correction_factor: null },
    eta_s: 120,
  };
}

export function exportRow(id: string, kind: string, extra: Partial<Export> = {}): Export {
  return {
    id,
    project_id: PID,
    run_id: RID,
    kind,
    ref: null,
    options: {},
    job_id: `j_${id}`,
    status: "ready",
    path: `/w/projects/demo/exports/${id}/out.zip`,
    bytes: 2_400_000,
    draft: false,
    created_utc: "2026-10-02T09:00:00Z",
    ...extra,
  };
}

export function finding(id: string, position: number, extra: Partial<Finding> = {}): Finding {
  return {
    id,
    project_id: PID,
    run_id: RID,
    view: "/r/:rid/accuracy",
    url_state: `/r/${RID}/accuracy?model=stack`,
    title: `Finding ${id}`,
    note_md: "",
    snapshot: { kind: "chart", title: `Finding ${id}`, table: { columns: [{ key: "m", label: "Model" }, { key: "r2", label: "R²" }], rows: [["stack", 0.81]] } },
    image_url: null,
    position,
    created_utc: `2026-10-01T1${position}:00:00Z`,
    updated_utc: `2026-10-01T1${position}:00:00Z`,
    ...extra,
  };
}

export function output(id: string, label: string, bytes: number, state: OutputEntry["state"] = "present"): OutputEntry {
  return { id, label, group: "model", state, produced_by: "stage:S2_S3", view: "accuracy", formats: ["json"], files: [{ relpath: `${id}.json`, bytes, mtime: "2026-10-01T14:25:00Z" }], action: null };
}
