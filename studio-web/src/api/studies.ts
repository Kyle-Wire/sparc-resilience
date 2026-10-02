// Studies and post-run actions (api.md §8 params, §9; SPEC §8): typed endpoint functions,
// resource hooks, the launch param shapes of every study and action kind, and the per-kind
// study view shapes the Validation tab, the Studies hub and the study page render.
//
// View shapes follow the core outputs they summarise (placebo.json rows, simcheck.jsonl and
// simcheck_summary.json, multiverse_summary.json, reproduce.json, benchmark.json). Every field
// a study in progress may not have yet is optional or nullable; the pages render what exists.
import { api } from "./client";
import { useResource } from "./resource";
import type { Action, Job, RunSummary } from "./types";

const enc = encodeURIComponent;

// ---------------------------------------------------------------- kinds

export type StudyKind =
  | "baselines" | "planner" | "emulator" | "uncertainty" | "writeup"
  | "placebo" | "simcheck" | "multiverse" | "reproduce" | "literature" | "benchmark";

/** Display order of the Validation tab cards (one per kind). */
export const STUDY_KINDS: readonly StudyKind[] = [
  "baselines", "placebo", "simcheck", "multiverse", "uncertainty", "reproduce",
  "emulator", "planner", "writeup", "literature", "benchmark",
];

/** Post-run actions: `POST /api/runs/{rid}/actions/{kind}` → 202 Job. */
export type ActionKind = "baselines" | "planner" | "emulator" | "uncertainty" | "writeup";
export const ACTION_KINDS: readonly ActionKind[] = ["baselines", "planner", "emulator", "uncertainty", "writeup"];

/** Run studies: `POST /api/runs/{rid}/studies/{kind}` → 202 {study, job}. */
export type RunStudyKind = "placebo" | "simcheck" | "multiverse" | "reproduce";
export const RUN_STUDY_KINDS: readonly RunStudyKind[] = ["placebo", "simcheck", "multiverse", "reproduce"];

/** Study kinds that can be attached to a run to feed its uncertainty report. */
export const ATTACHABLE_KINDS: readonly StudyKind[] = ["placebo", "simcheck", "multiverse"];

export function isActionKind(k: string): k is ActionKind {
  return (ACTION_KINDS as readonly string[]).includes(k);
}

export function isRunStudyKind(k: string): k is RunStudyKind {
  return (RUN_STUDY_KINDS as readonly string[]).includes(k);
}

export const KIND_LABELS: Record<StudyKind, string> = {
  baselines: "Reference baselines",
  planner: "Planner pack",
  emulator: "Scenario emulator",
  uncertainty: "Uncertainty report",
  writeup: "Methods & model card",
  placebo: "Placebo tests",
  simcheck: "Simulation check",
  multiverse: "Multiverse",
  reproduce: "Reproduction",
  literature: "Literature check",
  benchmark: "Effect benchmark",
};

// ---------------------------------------------------------------- wire types (§9)

export type StudyState = "not_run" | "queued" | "running" | "done" | "stale" | "failed";

export type StudyEstimate = { est_s: number; est_lo: number; est_hi: number };

/** `GET /api/runs/{rid}/studies` row. */
export type StudyStatusRow = {
  kind: StudyKind;
  state: StudyState;
  study_id: string | null;
  job_id: string | null;
  updated_utc: string | null;
  headline: string | null;
  estimate: StudyEstimate | null;
  attached: boolean | null;
  action: Action | null;
  requirements: { ok: boolean; missing: string[] };
};

export type Study = {
  id: string;
  project_id: string;
  kind: string;
  target_run_id: string | null;
  job_id: string | null;
  out_dir: string;
  status: string;
  params: Record<string, unknown>;
  summary: Record<string, unknown> | null;
  origin: "studio" | "imported";
  created_utc: string;
  updated_utc: string;
  children: RunSummary[];
  attached_runs: string[];
  stale_vs: string[];
};

/** `POST /api/studies/estimate` reply. */
export type StudyCost = { est_s: number; est_lo: number; est_hi: number; est_peak_rss_gb: number; est_disk_gb: number; n_children: number };

// ---------------------------------------------------------------- launch params (§8)

export type BaselinesParams = { models?: string[] };
export type PlannerParams = { package?: string; thresholds?: number[]; hex_sizes?: number[]; export?: boolean };
export type EmulatorParams = { patches?: number };
export type UncertaintyParams = { multiverse_study?: string; simcheck_studies?: string[]; placebo_study?: string; real_r2_gate?: boolean };
export type WriteupParams = Record<string, never>;

export type PlaceboKind = "grf" | "shift" | "rotate";
export const PLACEBO_KINDS: readonly PlaceboKind[] = ["grf", "shift", "rotate"];
export type PlaceboParams = { kinds: PlaceboKind[]; coarse_m: number | null; seed: number; grf_range_m: number };

export type SimGenerator = "physics" | "additive" | "own_only" | "coarse_scale" | "confounded" | "null";
export const SIM_GENERATORS: readonly SimGenerator[] = ["physics", "additive", "own_only", "coarse_scale", "confounded", "null"];
export type SimcheckParams = {
  design: Record<SimGenerator, number>;
  coarse_m: number | null;
  epochs: number;
  workers: number;
  threads: number;
  continue_study_id?: string;
};

export type MultiverseParams = {
  variants?: string[];
  custom_variants?: Record<string, Record<string, unknown>>;
  coarse_m: number | null;
  workers: number;
  threads: number;
};

export type ReproduceParams = { stages?: string[]; tol_r2?: number; tol_effect?: number };
export type BenchmarkParams = { seed?: number; ab?: boolean; epochs?: number; n?: number };

export type StudyParams = {
  baselines: BaselinesParams;
  planner: PlannerParams;
  emulator: EmulatorParams;
  uncertainty: UncertaintyParams;
  writeup: WriteupParams;
  placebo: PlaceboParams;
  simcheck: SimcheckParams;
  multiverse: MultiverseParams;
  reproduce: ReproduceParams;
  benchmark: BenchmarkParams;
};

export type LaunchableKind = keyof StudyParams;

// ---------------------------------------------------------------- study views (§9)

export type PlaceboVerdict = {
  delta_1sd: number | null;
  se_1sd: number | null;
  ratio_to_real: number | null;
  model_within_2se: boolean;
  model_below_10pct_of_real: boolean;
  model_pass: boolean;
  causal_ci_covers_zero: boolean;
};

/** One placebo.json row: a tested layer of one re-fit (the `grf` run's real layers are the reference). */
export type PlaceboRow = {
  kind: string;
  variable: string;
  placebo: boolean;
  sd: number | null;
  model: { dose_sd: number; mean_delta: number | null; se: number | null; frac_extrapolated?: number | null }[];
  causal_theta_sum_per_sd: number | null;
  causal_se_per_sd: number | null;
  footprint_per_sd?: number | null;
  verdict?: PlaceboVerdict | null;
  reference?: string | null;
};

export type PlaceboView = {
  rows?: PlaceboRow[];
  layer_correlation?: Record<string, number | null>;
  n_pass_model?: number;
  n_pass_causal?: number;
  n_placebos?: number;
  children?: { kind: string; run_id: string | null; status: string; verdict: string | null }[];
};

export type SimCellStatus = "pending" | "running" | "done" | "gate_fail" | "error";

export type SimcheckCell = {
  generator: string;
  seed: number;
  status: SimCellStatus;
  share: number | null;
  ci_covers: boolean | null;
  causal_covers: boolean | null;
  seconds: number | null;
  /** 0 = the first draw passed the data-matching gate; > 0 = redrawn. */
  gate_attempt: number | null;
};

/** simcheck.summarize() per generator (the null generator adds the false-positive fields). */
export type SimcheckGeneratorSummary = {
  n: number;
  n_gate_pass: number;
  share_median: number | null;
  share_iqr: [number, number] | null;
  rank_corr_mean: number | null;
  ci_coverage: number | null;
  causal_ci_coverage: number | null;
  interval_coverage_mean: number | null;
  oof_r2_mean: number | null;
  false_positive_rate?: number | null;
  causal_false_positive_rate?: number | null;
  null_mean_delta?: number | null;
  null_mean_delta_se?: number | null;
  null_causal_pp?: number | null;
};

export type BiasCorrection = {
  share_median_across_generators?: number | null;
  share_range?: [number, number] | null;
  relative_spread?: number | null;
  stable?: boolean;
  n_generators?: number;
  correction_factor?: number | null;
};

export type SimcheckView = {
  design?: Record<string, number>;
  grid?: SimcheckCell[];
  generators?: Record<string, SimcheckGeneratorSummary>;
  bias_correction?: BiasCorrection | null;
  eta_s?: number | null;
};

export type MultiverseVariant = { name: string; label: string; status: string; r2: number | null; rmse: number | null; seconds: number | null; run_id: string | null };

export type MultiverseEffect = {
  baseline: number | null;
  min: number | null;
  max: number | null;
  sd_across: number | null;
  sign_stability: number | null;
  values: Record<string, number | null>;
};

export type MultiverseView = {
  variants?: MultiverseVariant[];
  effects?: Record<string, MultiverseEffect>;
  /** variant → lever → agreement of its priority map with the baseline's. */
  priority?: Record<string, Record<string, { kendall_tau: number | null; top_decile_jaccard: number | null }>>;
  stability?: { sign_stability_min?: number | null; median_kendall_tau?: number | null; median_top_decile_jaccard?: number | null } | null;
};

export type ReproduceCheck = { check: string; ok: boolean; hard: boolean; detail: string };
export type ReproduceView = { pass?: boolean | null; checks?: ReproduceCheck[]; original?: string | null; reproduction?: string | null };

export type BenchmarkShare = { share: number | null; corr: number | null; mean_delta?: number | null };
export type BenchmarkRun = {
  variable?: string;
  dose?: number;
  true_mean_delta?: number | null;
  models: Record<string, BenchmarkShare>;
  stack: BenchmarkShare;
  footprint?: { share: number | null; corr: number | null } | null;
  oof?: Record<string, { rmse: number | null; r2: number | null }>;
};
export type BenchmarkView = { runs?: Record<string, BenchmarkRun> };

export type StudyViews = {
  placebo: PlaceboView;
  simcheck: SimcheckView;
  multiverse: MultiverseView;
  reproduce: ReproduceView;
  benchmark: BenchmarkView;
};
export type ViewKind = keyof StudyViews;
export const VIEW_KINDS: readonly ViewKind[] = ["placebo", "simcheck", "multiverse", "reproduce", "benchmark"];

export function hasStudyView(kind: string): kind is ViewKind {
  return (VIEW_KINDS as readonly string[]).includes(kind);
}

// ---------------------------------------------------------------- truth vs recovered (§9)

export type TruthRow = {
  quantity: "canopy_scenario" | "footprint_mean" | "L_m" | "influence_radius_m" | "noise_sd";
  label: string;
  truth: number;
  recovered: number | null;
  se: number | null;
  share: number | null;
  unit: string;
  scenario: string | null;
};

// ---------------------------------------------------------------- endpoints

export function getRunStudies(rid: string, signal?: AbortSignal): Promise<StudyStatusRow[]> {
  return api.get<StudyStatusRow[]>(`/api/runs/${enc(rid)}/studies`, undefined, signal);
}

export type LaunchResult = { job: Job; study: Study | null };

/**
 * Launch a study or post-run action with its params (api.md §8–9): actions go to
 * `/runs/{rid}/actions/{kind}` (→ Job), run studies to `/runs/{rid}/studies/{kind}` and the
 * benchmark to `/projects/{pid}/studies/benchmark` (→ {study, job}).
 */
export async function launchStudy<K extends LaunchableKind>(kind: K, target: { rid?: string | null; pid?: string | null }, params: StudyParams[K]): Promise<LaunchResult> {
  if (isActionKind(kind)) {
    if (!target.rid) throw new Error(`${KIND_LABELS[kind]} needs a run`);
    const job = await api.post<Job>(`/api/runs/${enc(target.rid)}/actions/${kind}`, params);
    return { job, study: null };
  }
  if (kind === "benchmark") {
    if (!target.pid) throw new Error("The benchmark needs a project");
    return api.post<LaunchResult>(`/api/projects/${enc(target.pid)}/studies/benchmark`, params);
  }
  if (!target.rid) throw new Error(`${KIND_LABELS[kind]} needs a run`);
  return api.post<LaunchResult>(`/api/runs/${enc(target.rid)}/studies/${kind}`, params);
}

export function estimateStudy(kind: string, runId: string | null, params: object, signal?: AbortSignal): Promise<StudyCost> {
  return api.post<StudyCost>("/api/studies/estimate", { kind, ...(runId ? { run_id: runId } : {}), params }, { signal });
}

export function getProjectStudies(pid: string, kind?: string | null, signal?: AbortSignal): Promise<Study[]> {
  return api.get<Study[]>(`/api/projects/${enc(pid)}/studies`, kind ? { kind } : undefined, signal);
}

export function getStudy(stid: string, signal?: AbortSignal): Promise<Study> {
  return api.get<Study>(`/api/studies/${enc(stid)}`, undefined, signal);
}

export function getStudyView<K extends ViewKind>(stid: string, signal?: AbortSignal): Promise<StudyViews[K]> {
  return api.get<StudyViews[K]>(`/api/studies/${enc(stid)}/view`, undefined, signal);
}

export function resumeStudy(stid: string): Promise<Job> {
  return api.post<Job>(`/api/studies/${enc(stid)}/resume`, {});
}

export function attachStudy(stid: string, runId: string, attach: boolean): Promise<{ study: Study; job: Job | null }> {
  return api.post<{ study: Study; job: Job | null }>(`/api/studies/${enc(stid)}/${attach ? "attach" : "detach"}`, { run_id: runId });
}

export type SimcheckMerge = { summary: { generators?: Record<string, SimcheckGeneratorSummary>; bias_correction?: BiasCorrection | null; n_rows?: number; n_errors?: number }; markdown: string };

export function mergeSimcheck(studyIds: string[]): Promise<SimcheckMerge> {
  return api.post<SimcheckMerge>("/api/studies/simcheck/merge", { study_ids: studyIds });
}

export function deleteStudy(stid: string, files: boolean): Promise<{ ok: true }> {
  return api.del<{ ok: true }>(`/api/studies/${enc(stid)}`, { files });
}

export function getTruth(rid: string, signal?: AbortSignal): Promise<{ rows: TruthRow[] }> {
  return api.get<{ rows: TruthRow[] }>(`/api/runs/${enc(rid)}/truth`, undefined, signal);
}

// ---------------------------------------------------------------- hooks

/** Study status rows of a run; refetched on `study.updated` and the run's job events. */
export function useRunStudies(rid: string | null) {
  return useResource<StudyStatusRow[]>(rid ? `run:${rid}:studies` : null, (s) => getRunStudies(rid!, s), {
    tags: rid ? [`run:${rid}:studies`, "studies"] : [],
    keepPrevious: true,
  });
}

export function useProjectStudies(pid: string | null) {
  return useResource<Study[]>(pid ? `project:${pid}:studies` : null, (s) => getProjectStudies(pid!, null, s), {
    tags: pid ? [`project:${pid}:studies`, "studies"] : [],
    keepPrevious: true,
  });
}

/** One study; also refetched on run events (`runs`), so child runs show up as they are indexed. */
export function useStudy(stid: string | null) {
  return useResource<Study>(stid ? `study:${stid}` : null, (s) => getStudy(stid!, s), { tags: stid ? [`study:${stid}`, "studies", "runs"] : [] });
}

/** Same cache key as the tracker's child matrix, so both read one entry. */
export function useStudyView<K extends ViewKind>(stid: string | null, _kind?: K) {
  return useResource<StudyViews[K]>(stid ? `study:${stid}:view` : null, (s) => getStudyView<K>(stid!, s), {
    tags: stid ? [`study:${stid}`] : [],
    keepPrevious: true,
  });
}

export function useTruth(rid: string | null) {
  return useResource<{ rows: TruthRow[] }>(rid ? `run:${rid}:truth` : null, (s) => getTruth(rid!, s), { tags: rid ? [`run:${rid}:outputs`] : [] });
}

/** `threads_heavy` from Settings: simcheck and multiverse need workers × threads within it. */
export function useThreadsHeavy(): number | null {
  const res = useResource<{ threads_heavy?: number }>("settings", (s) => api.get<{ threads_heavy?: number }>("/api/settings", undefined, s), { tags: ["settings"] });
  const v = res.data?.threads_heavy;
  return typeof v === "number" && v > 0 ? v : null;
}

/** `GET /api/projects/{pid}/status-board` (api.md §6): the study columns feed the hub matrix. */
export type BoardCell = {
  state: "done" | "cached" | "running" | "failed" | "skipped" | "stale" | "not_run" | "disabled";
  seconds: number | null;
  progress: number | null;
  reason: string | null;
  job_id: string | null;
  study_id: string | null;
  action: Action | null;
};
export type StatusBoard = {
  columns: { id: string; label: string; group: "stage" | "post" | "study" }[];
  rows: { run: RunSummary; cells: Record<string, BoardCell> }[];
};

export function useStatusBoard(pid: string | null) {
  return useResource<StatusBoard>(pid ? `project:${pid}:status-board` : null, (s) => api.get<StatusBoard>(`/api/projects/${enc(pid!)}/status-board`, undefined, s), {
    tags: pid ? [`project:${pid}`, "runs", "jobs", "studies"] : [],
    keepPrevious: true,
  });
}
