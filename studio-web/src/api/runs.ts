// Runs group (api.md §6, §6.1, §6.2, §6.5, §9 reproduce, §9 writeup): typed endpoint
// functions, resource hooks and the ViewModel section shapes the run hub renders.
//
// ViewModels (`GET /api/runs/{rid}/views/{view}`) are column-oriented and chart-ready. api.md
// fixes the envelope and the section keys per view; the section types below fix the inner
// shapes the client renders (each derived from the core output it summarises, see the
// comments). Every section may be null (older-code runs, or not computed); the hub renders
// "not in this run (older code)" for those instead of failing.
import { api, apiUrl, errorFromResponse, getBin } from "./client";
import { useResource } from "./resource";
import type { Action, Availability, GridMeta, Job, Likely, OutputState, Page, RunStatus, RunSummary, StageId } from "./types";
import type { CellInfo } from "../map/Inspector";

export type { CellInfo, CellCurve } from "../map/Inspector";

const enc = encodeURIComponent;
export const runBase = (rid: string) => `/api/runs/${enc(rid)}`;

// ---------------------------------------------------------------- RunDetail (api.md §6)

export type CheckpointInfo = {
  present: boolean;
  bytes: number | null;
  done: string[];
  saved_utc: string | null;
  fingerprint: string | null;
  matches_snapshot: { data: boolean; code: boolean; config: boolean } | null;
  changed_sections: string[];
  resumable: boolean;
  reuses: StageId[];
  saves_s: number | null;
  reason: string | null;
};

export type RunStageRow = {
  id: StageId;
  state: string;
  seconds: number | null;
  source: "events" | "manifest" | "run_state";
  reason: string | null;
};

export type SectionTag = { present: boolean; source: "manifest" | "file" | null; stale: boolean; older_code: boolean };

export type RunFlag = { code: string; severity: string; message: string };

export type RunHeader = {
  name: string;
  created_utc: string | null;
  git_commit: string | null;
  git_dirty: boolean | null;
  versions: Record<string, string> | null;
  n_points: number | null;
  grid_shape: [number, number] | null;
  cell_m: number | null;
  fast: boolean;
  coarse_m: number | null;
  run_dir: string;
  demo: boolean;
};

/** `GET /api/runs/{rid}`. */
export type RunDetail = {
  run: RunSummary;
  header: RunHeader;
  state: Record<string, unknown> | null;
  launch: Record<string, unknown> | null;
  stages: RunStageRow[];
  checkpoint: CheckpointInfo;
  outputs_summary: { present: number; missing: number; stale: number; writing: number };
  sections: Record<string, SectionTag>;
  flags: RunFlag[];
  children: RunSummary[];
  jobs: Job[];
  warnings_count: number;
};

// ---------------------------------------------------------------- ViewModels (api.md §6.1)

export type ViewName =
  | "overview"
  | "data"
  | "accuracy"
  | "distance"
  | "influence"
  | "response"
  | "scenarios"
  | "climate"
  | "causal"
  | "budget"
  | "planner"
  | "uncertainty"
  | "provenance";

export const VIEW_NAMES: readonly ViewName[] = [
  "overview",
  "data",
  "accuracy",
  "distance",
  "influence",
  "response",
  "scenarios",
  "climate",
  "causal",
  "budget",
  "planner",
  "uncertainty",
  "provenance",
];

export type MissingOutput = { output: string; produced_by: string; action: Action | null };

export type ViewUnits = { target: string; levers: Record<string, string> };

/** Every section key is present but may be null (older-code runs). */
export type Sections<S> = { [K in keyof S]: S[K] | null };

export type ViewModel<S = Record<string, unknown>> = {
  view: string;
  availability: Availability;
  missing: MissingOutput[];
  units: ViewUnits;
  caveats: string[];
  demo: boolean;
  sections: Sections<S>;
};

// -- building blocks

/**
 * A key number. `format` picks the wording: "delta" = a temperature change (cooler/warmer,
 * with `likely` for its range), "percent" = a 0..1 fraction, "signed" = explicit sign.
 */
export type ViewKpi = {
  id: string;
  label: string;
  value: number | string | null;
  unit?: string | null;
  decimals?: number | null;
  format?: "number" | "int" | "percent" | "delta" | "signed" | "text";
  likely?: Likely | null;
  /** Comparison value (coverage target, noise floor) shown as "vs …". */
  target?: number | null;
  target_label?: string | null;
  /** An independent band (the causal check on a package Δ). */
  band?: { lo: number | null; hi: number | null; label: string } | null;
  note?: string | null;
  tone?: "good" | "warn" | "crit" | null;
};

export type KeyValue = { label: string; value: number | string | boolean | null; unit?: string | null; decimals?: number | null };

export type TableCell = string | number | boolean | null;

export type GenericTable = { columns: { key: string; label: string; unit?: string | null }[]; rows: TableCell[][] };

export type BinsData = { edges: number[]; counts: number[] };

/** Box of a group: q = [p10, q1, median, q3, p90]. */
export type BoxRow = { label: string; n: number | null; q: [number, number, number, number, number] | null; mean: number | null };

export type ViewFlag = { code: string; severity: string; message: string };

// -- overview (manifest, timings_s/events, catalog states, studies, model card, findings)

export type OverviewSections = {
  kpis: ViewKpi[];
  timings: { stage: string; label: string; seconds: number | null; state: string; reason?: string | null }[];
  flags: ViewFlag[];
  outputs_grid: { id: string; label: string; group: string; state: OutputState; produced_by: string; view: string; action: Action | null }[];
  studies: { kind: string; state: string; headline: string | null; study_id: string | null }[];
  limitations: string[];
  findings: { id: string; title: string; note_md: string; view: string; url_state: string; created_utc: string }[];
};

// -- data & QA (manifest.qa, input frame)

export type DataSections = {
  qa_tiles: ViewKpi[];
  flags: ViewFlag[];
  /** qa.target_fractional_histogram: share of targets per tenth of a unit. */
  frac_hist: { edges: number[]; shares: number[]; integer_share: number | null; half_share: number | null; noise_sd: number | null };
  /** qa.dose_scale, one row per lever. */
  dose_scale: { lever: string; unit: string; sd: number | null; doses: number[]; doses_in_sd: (number | null)[]; percentile: (number | null)[] }[];
  predictor_hists: { name: string; label: string; unit: string; edges: number[]; counts: number[] }[];
  corr_matrix: { names: string[]; values: (number | null)[][] };
  zone_counts: { zone: string; n: number }[];
  coarse: KeyValue[];
  joins: { path: string; key: string | null; right_key: string | null; sha256: string | null; n_matched: number | null; n_unmatched: number | null }[];
};

// -- accuracy (manifest.metrics, stacker, physics, predictions.parquet)

export type ModelRow = {
  model: string;
  label: string;
  kind?: "base" | "stack" | "mean" | null;
  r2: number | null;
  rmse: number | null;
  mae: number | null;
  bias: number | null;
  /** Mean NNLS blend weight across folds (base models only). */
  weight: number | null;
  n: number | null;
};

export type CoverageRow = {
  label: string;
  n: number | null;
  global: number | null;
  adaptive: number | null;
  halfwidth_global?: number | null;
  halfwidth_adaptive?: number | null;
};

export type AccuracySections = {
  models: ModelRow[];
  obs_pred_bins: { x_edges: number[]; y_edges: number[]; counts: number[][]; spearman?: number | null };
  resid_hist: BinsData;
  resid_by_zone: BoxRow[];
  resid_by_fold: BoxRow[];
  interval_honesty: {
    target: number;
    global: number | null;
    adaptive: number | null;
    halfwidth: number | null;
    by_fold: CoverageRow[];
    by_distance: CoverageRow[];
    by_zone: CoverageRow[];
  };
  stacker: {
    /** lambda_scores: candidate → held-out RMSE. */
    candidates: { name: string; rmse: number | null }[];
    chosen: string | null;
    models: string[];
    folds: {
      fold: number;
      weights: Record<string, number>;
      residual_kept: boolean | null;
      val_mse_base: number | null;
      val_mse_with_residual: number | null;
      best_epoch: number | null;
    }[];
    spatial_plus: string[];
  };
  physics: {
    params: { name: string; label: string; mean: number | null; sd: number | null; prior: string | null; unit: string | null }[];
    folds: { param: string; values: (number | null)[] }[];
    warnings: string[];
  };
  advection: { verdict: string; selected: boolean | null; fold_delta_rmse: (number | null)[] };
  forcing: {
    date: string | null;
    hours: string | null;
    sw_down: number | null;
    lw_net: number | null;
    wind_speed: number | null;
    /** Direction the wind blows from, degrees clockwise from north. */
    wind_dir_deg: number | null;
    station: string | null;
    checks: string[];
  };
  cv_design: { n_folds: number; block_m: number | null; buffer_m: number | null; n_blocks: number | null; test_sizes: number[] };
  /** True when the metrics were computed from predictions.parquet while the run continues. */
  live: boolean;
};

// -- distance & baselines (cv_distance.json, baselines.json)

export type DistanceSections = {
  curve: {
    block_m: number[];
    metric: "r2" | "rmse";
    series: { id: string; label: string; kind: "stack" | "base" | "baseline"; values: (number | null)[]; lo?: (number | null)[] | null; hi?: (number | null)[] | null }[];
    main_block_m: number | null;
    /** The random-points ("leaky") reference row. */
    random: { id: string; label: string; value: number | null }[];
  };
  baselines: {
    id: string;
    label: string;
    rmse: number | null;
    r2: number | null;
    delta_mse: number | null;
    delta_mse_se: number | null;
    stack_better: boolean;
    baseline_better: boolean;
  }[];
  verdict: { text: string; best_baseline: string | null; stack_wins: boolean | null };
  block_wins: { id: string; label: string; frac: number | null; n_blocks: number | null }[];
};

// -- influence (influence.json)

export type InfluenceSections = {
  ranges: { predictor: string; label: string; range_m: number | null; raw_range_m?: number | null }[];
  correlogram: {
    lags_m: number[];
    acf: (number | null)[];
    band_mean: (number | null)[];
    band_sd: (number | null)[];
    /** Direction (degrees, "0" "45" "90" "135") → ACF at the same lags. */
    directional: Record<string, (number | null)[]> | null;
  };
  rings: { predictor: string; label: string; edges_m: number[]; betas: (number | null)[]; family: string | null; significant: boolean | null; scale_m: number | null }[];
  anisotropy: { predictor: string; label: string; ratio: number | null; theta_deg: number | null; reliable: boolean; ranges_m: Record<string, number | null> }[];
  priors: { L_prior_m: number | null; block_size_m: number | null; resid_range_m: number | null };
};

// -- response (response_curves.json, response_<var>.parquet, literature)

export type ResponseLever = {
  label: string;
  unit: string;
  direction: "increase" | "decrease";
  /** City-mean cooling per dose (positive = cooler), ± SE, extrapolated share, realised dose. */
  curve: { dose: number[]; benefit: (number | null)[]; se: (number | null)[]; frac_extrapolated: (number | null)[]; realized_dose: (number | null)[] } | null;
  /** Share of cells per fitted curve shape. */
  shapes: { saturating: number | null; linear: number | null; sigmoid: number | null; censored: number | null; insufficient?: number | null } | null;
  /** Mean ΔT per +1 lever unit: own cell, footprint (target·cells), their ratio. */
  effects: { own: number | null; footprint: number | null; ratio: number | null; median_d90: number | null; median_max_cooling: number | null } | null;
  fold_means: { own: number[]; footprint: number[] } | null;
};

export type ResponseSections = {
  levers: Record<string, ResponseLever>;
  literature: {
    rows: { key: string; quantity: string; per: string; low: number | null; high: number | null; unit: string; value: string; status: string | null; citation: string | null }[];
    sparc: { quantity: string; scenario: string; dose: number | null; cooling: number | null; se: number | null; causal: number | null; unit: string }[];
  };
};

// -- configured scenarios (scenarios.json, scenario_detail.npz)

export type ScenarioRow = {
  slug: string;
  name: string;
  kind: "single" | "joint";
  lever: string | null;
  dose: number | null;
  /** City-mean ΔT (negative = cooler) with its likely range. */
  delta: Likely;
  p10: number | null;
  p90: number | null;
  frac_extrapolated: number | null;
  causal: { delta: number; lo: number | null; hi: number | null; model_within: boolean | null } | null;
  tier: string | null;
  has_folds: boolean;
  realized: Record<string, number | null>;
};

export type ScenariosSections = {
  rows: ScenarioRow[];
  ladders: { lever: string; label: string; unit: string; points: { dose: number; slug: string; estimate: number; lo: number | null; hi: number | null; hollow: boolean }[] }[];
  has_detail: boolean;
};

// -- climate (climate.json)

export type ClimateProjection = {
  id: string;
  experiment: string;
  label: string;
  period: string;
  n_models: number | null;
  median: number | null;
  p10: number | null;
  p90: number | null;
  min: number | null;
  max: number | null;
  by_model: Record<string, number | null>;
};

export type ShareRange = { median: number | null; p10: number | null; p90: number | null };

export type ClimateSections = {
  warming: ClimateProjection[];
  models_table: GenericTable;
  exposure: {
    thresholds: number[];
    /** Threshold (as a string key, e.g. "90") → share of cells at or above it today. */
    present: Record<string, number | null>;
    groups: { id: string; label: string; experiment: string; period: string; variant: string; share: Record<string, ShareRange> }[];
  };
  /** Share of the median warming an adaptation variant offsets. */
  offset: { id: string; label: string; variant: string; share: number | null }[];
  thresholds: number[];
};

// -- causal audit (causal.json, causal_cells.parquet)

export type CausalTreatment = {
  label: string;
  unit: string;
  forest: { id: string; label: string; est: number | null; se: number | null; lo: number | null; hi: number | null; model: number | null }[];
  audit: { check: string; label: string; verdict: string; flag: boolean; model: number | null; causal: number | null }[];
  dr_curve: { t: number[]; theta: (number | null)[]; lo: (number | null)[]; hi: (number | null)[]; ess: number | null; clipped_frac: number | null; note: string | null } | null;
  model_pd_curve: { t: number[]; y: (number | null)[]; se: (number | null)[] } | null;
  /** q = [q05, q25, q50, q75, q95] of the per-cell CATE. */
  cate: { q: [number, number, number, number, number]; mean: number | null; sd: number | null; blp: { coef: number | null; se: number | null; p: number | null } | null } | null;
  cate_layer: string | null;
  sensitivity: { e_value: number | null; e_value_ci: number | null; rv_q: number | null; rv_q_alpha: number | null; design_effect: number | null } | null;
  controls: { names: string[]; basis_scale_m: number | null; hole_scale_ratio: number | null; hole_warning: boolean; nuisance_r2: Record<string, number | null> } | null;
};

export type CausalSections = {
  treatments: Record<string, Sections<CausalTreatment> & { label: string; unit: string }>;
  flags: { treatment: string; check: string; verdict: string }[];
  dag_audit: GenericTable;
};

// -- budget (optimize.json, allocation.parquet)

export type BudgetSections = {
  kpis: ViewKpi[];
  pareto: {
    points: { budget: number; total_benefit: number; n_segments: number | null; gini: number | null }[];
    realised: { budget: number; total_benefit: number }[] | null;
    budget: number | null;
    unit: string;
    budget_unit: string;
  };
  /** Computed by the server from the numbers (e.g. the 2×/1× budget ratio). */
  caption: string;
  top_cells: {
    rank: number;
    row: number;
    id: string | number;
    lon: number | null;
    lat: number | null;
    dose: number | null;
    benefit: number | null;
    zone: string | number | null;
  }[];
  /** "ok", or core's "no positive-benefit segments". */
  status: string;
};

// -- planner pack (planner/planner.json and files)

export type PlannerSections = {
  /** Share of residents at or above each threshold, per group (today and futures). */
  exposure: { thresholds: number[]; groups: { id: string; label: string; share: (number | null)[]; lo?: (number | null)[] | null; hi?: (number | null)[] | null }[] };
  person_mean: GenericTable;
  hot_days: { unit: string; panels: { id: string; label: string; thresholds: number[]; campaign: (number | null)[]; lower: (number | null)[] }[] };
  equity: { measures: string[]; quintiles: { label: string; values: Record<string, number | null> }[]; concentration: { label: string; value: number | null }[] };
  plantable: ViewKpi[];
  zones: GenericTable;
  hex_files: { size_m: number; relpath: string; format: string }[];
  /** Rows are run row indices. */
  sites: { row: number; id: string | number; lon: number | null; lat: number | null; label: string | null }[];
  pairs: { treated: number; control: number }[];
  gis: { relpath: string; kind: string; bytes: number | null; layer: string | null }[];
};

// -- uncertainty (uncertainty.json)

export type UncertaintySections = {
  rows: { id: string; label: string; estimate: number | null; layers: { id: string; label: string; lo: number | null; hi: number | null }[] }[];
  climate: GenericTable;
  sources: { kind: string; label: string; study_id: string | null; attached: boolean; state: string | null }[];
};

// -- provenance (manifest.provenance, launch.json)

export type ProvenanceSections = {
  hashes: Record<string, string | null>;
  git: { commit: string | null; dirty: boolean | null; branch?: string | null };
  platform: Record<string, string | null>;
  launch: Record<string, unknown>;
};

export type ViewSections = {
  overview: OverviewSections;
  data: DataSections;
  accuracy: AccuracySections;
  distance: DistanceSections;
  influence: InfluenceSections;
  response: ResponseSections;
  scenarios: ScenariosSections;
  climate: ClimateSections;
  causal: CausalSections;
  budget: BudgetSections;
  planner: PlannerSections;
  uncertainty: UncertaintySections;
  provenance: ProvenanceSections;
};

export type ViewModelOf<V extends ViewName> = ViewModel<ViewSections[V]>;

export function getView<V extends ViewName>(rid: string, view: V, signal?: AbortSignal): Promise<ViewModelOf<V>> {
  return api.get<ViewModelOf<V>>(`${runBase(rid)}/views/${view}`, undefined, signal);
}

/** A run ViewModel. Refetched when the run writes outputs (`output.written` → `run:<rid>:views`). */
export function useView<V extends ViewName>(rid: string | null, view: V | null) {
  return useResource<ViewModelOf<V>>(rid && view ? `run:${rid}:views:${view}` : null, (s) => getView(rid!, view!, s), {
    tags: rid ? [`run:${rid}`, `run:${rid}:views`] : [],
    keepPrevious: true,
  });
}

/** The full RunDetail (same cache entry as the shell's run header). */
export function useRunDetailFull(rid: string | null) {
  return useResource<RunDetail>(rid ? `run:${rid}` : null, (s) => api.get<RunDetail>(runBase(rid!), undefined, s), { tags: rid ? [`run:${rid}`] : [] });
}

// ---------------------------------------------------------------- docs (api.md §6.1)

export type DocId = "report" | "methods" | "model_card" | "uncertainty" | "placebo" | "multiverse" | "simcheck" | "benchmark";

export type DocEntry = { id: DocId; file: string; title: string; mtime: string | null; present: boolean; regenerable: boolean; frozen: boolean };

export type DocContent = { markdown: string; mtime: string; frozen: boolean };

export function useRunDocs(rid: string | null) {
  return useResource<DocEntry[]>(rid ? `run:${rid}:files:docs` : null, (s) => api.get<DocEntry[]>(`${runBase(rid!)}/docs`, undefined, s), {
    tags: rid ? [`run:${rid}`, `run:${rid}:files`] : [],
  });
}

export function useRunDoc(rid: string | null, doc: string | null) {
  return useResource<DocContent>(rid && doc ? `run:${rid}:files:doc:${doc}` : null, (s) => api.get<DocContent>(`${runBase(rid!)}/docs/${enc(doc!)}`, undefined, s), {
    tags: rid ? [`run:${rid}`, `run:${rid}:files`] : [],
  });
}

// ---------------------------------------------------------------- post-run actions and studies (api.md §9)

/** `POST /api/runs/{rid}/actions/writeup` → 202 Job (Regenerate methods & model card). */
export function regenerateWriteup(rid: string): Promise<Job> {
  return api.post<Job>(`${runBase(rid)}/actions/writeup`, {});
}

/** `POST /api/runs/{rid}/studies/reproduce` → 202 {study, job}. */
export function reproduceRun(
  rid: string,
  params: { stages?: string[]; tol_r2?: number; tol_effect?: number } = {},
): Promise<{ study: { id: string; kind: string; project_id: string }; job: Job }> {
  return api.post(`${runBase(rid)}/studies/reproduce`, params);
}

/**
 * "Clone to edit" a configured scenario (api.md §7.4–7.5): its equivalent add-mode ScenarioDoc
 * comes from `GET /api/runs/{rid}/scenarios`, and is saved as a new Lab scenario of the
 * project (`POST /api/projects/{pid}/scenarios`). Returns the new scenario's id.
 */
export async function cloneConfiguredScenario(rid: string, pid: string, slug: string): Promise<string> {
  const list = await api.get<{ configured: { slug: string; name: string; doc: Record<string, unknown> }[] }>(`${runBase(rid)}/scenarios`);
  const c = list.configured.find((x) => x.slug === slug);
  if (!c) throw new Error(`Configured scenario "${slug}" is not in this run`);
  const doc = { ...c.doc, name: `${c.name} (copy)`, anchor_run_id: rid };
  const created = await api.post<{ id: string }>(`/api/projects/${enc(pid)}/scenarios`, { doc });
  return created.id;
}

// ---------------------------------------------------------------- files (api.md §6.1)

export type FileEntry = {
  name: string;
  relpath: string;
  dir: boolean;
  bytes: number | null;
  mtime: string | null;
  output_id: string | null;
  state: string | null;
  in_manifest: boolean;
};

export type FileTable = { columns: { name: string; dtype: string }[]; rows: unknown[][]; n_rows: number };

export type DictionaryRow = { output: string; column: string; unit: string; sign: string | null; description: string };

export function useRunFiles(rid: string | null, path: string | null) {
  const p = path ?? "";
  return useResource<FileEntry[]>(rid ? `run:${rid}:files:dir:${p}` : null, (s) => api.get<FileEntry[]>(`${runBase(rid!)}/files`, p ? { path: p } : undefined, s), {
    tags: rid ? [`run:${rid}`, `run:${rid}:files`] : [],
  });
}

export function useFileTable(rid: string | null, path: string | null, limit = 200) {
  return useResource<FileTable>(
    rid && path ? `run:${rid}:files:table:${path}:${limit}` : null,
    (s) => api.get<FileTable>(`${runBase(rid!)}/files/table`, { path: path!, limit }, s),
    {
      tags: rid ? [`run:${rid}`, `run:${rid}:files`] : [],
    },
  );
}

export function useRunDictionary(rid: string | null) {
  return useResource<DictionaryRow[]>(rid ? `run:${rid}:dictionary` : null, (s) => api.get<DictionaryRow[]>(`${runBase(rid!)}/dictionary`, undefined, s), {
    tags: rid ? [`run:${rid}`] : [],
  });
}

export type Conversion = "csv" | "json" | "html" | "geojson";

/** Download URL of a run file (native, or converted with `as`). */
export function fileRawUrl(rid: string, path: string, as?: Conversion | null): string {
  return apiUrl(`${runBase(rid)}/files/raw`, { path, as: as ?? undefined });
}

/** `GET /api/runs/{rid}/files/raw` as text (small JSON and markdown previews). */
export async function fetchFileText(rid: string, path: string, signal?: AbortSignal): Promise<string> {
  const res = await fetch(fileRawUrl(rid, path), { credentials: "same-origin", signal });
  if (!res.ok) {
    let body: unknown = null;
    try {
      body = await res.json();
    } catch {
      /* not JSON */
    }
    throw errorFromResponse(res.status, body, res.statusText);
  }
  return res.text();
}

/** `DELETE /api/runs/{rid}/checkpoint` (also evicts the run from the engine host). */
export function deleteCheckpoint(rid: string): Promise<{ freed_bytes: number }> {
  return api.del<{ freed_bytes: number }>(`${runBase(rid)}/checkpoint`);
}

// ---------------------------------------------------------------- config, provenance, environment

export type RunConfig = {
  effective: Record<string, unknown>;
  raw: Record<string, unknown>;
  yaml: string;
  source: "launch" | "manifest" | "import";
  config_dir: string;
  vs_project_diff: { path: string; run: unknown; project: unknown }[];
  vs_defaults: { path: string; value: unknown; default: unknown }[];
};

export type RunEnvironment = {
  packages: string[];
  diff?: { added: string[]; removed: string[]; changed: { name: string; a: string; b: string }[] };
};

export function useRunConfig(rid: string | null) {
  return useResource<RunConfig>(rid ? `run:${rid}:config` : null, (s) => api.get<RunConfig>(`${runBase(rid!)}/config`, undefined, s), { tags: rid ? [`run:${rid}`] : [] });
}

export function useRunEnvironment(rid: string | null, diffWith: string | null) {
  return useResource<RunEnvironment>(
    rid ? `run:${rid}:environment:${diffWith ?? ""}` : null,
    (s) => api.get<RunEnvironment>(`${runBase(rid!)}/environment`, diffWith ? { diff_with: diffWith } : undefined, s),
    { tags: rid ? [`run:${rid}`] : [], keepPrevious: true },
  );
}

/** Runs of a project (pickers for environment diff and compare). */
export function useProjectRuns(pid: string | null) {
  return useResource<Page<RunSummary>>(
    pid ? `project:${pid}:runs` : null,
    (s) => api.get<Page<RunSummary>>(`/api/projects/${enc(pid!)}/runs`, { limit: 200, sort: "created_desc" }, s),
    {
      tags: pid ? [`project:${pid}`, "runs"] : ["runs"],
    },
  );
}

// ---------------------------------------------------------------- cells and layers (api.md §6.2)

export function getCell(rid: string, index: number, scenarios: string[] = [], signal?: AbortSignal): Promise<CellInfo> {
  return api.get<CellInfo>(`${runBase(rid)}/cells/${index}`, scenarios.length ? { scenarios } : undefined, signal);
}

/** One cell for the inspector, with the deltas of the given scenarios (`configured:<slug>`, `res_…`). */
export function useCell(rid: string | null, index: number | null, scenarios: string[] = []) {
  const refs = scenarios.join(",");
  return useResource<CellInfo>(rid !== null && index !== null ? `run:${rid}:cell:${index}:${refs}` : null, (s) => getCell(rid!, index!, refs ? refs.split(",") : [], s), {
    tags: rid ? [`run:${rid}:layers`] : [],
  });
}

/** `GET /api/runs/{rid}/grid` alone (units and CRS without the packed geometry). */
export function useGridMeta(rid: string | null) {
  return useResource<GridMeta>(rid ? `run:${rid}:gridmeta` : null, (s) => api.get<GridMeta>(`${runBase(rid!)}/grid`, undefined, s), { tags: rid ? [`run:${rid}`] : [] });
}

export type LayerExportFormat = "tif" | "csv" | "geojson" | "parquet";

export function layerExportUrl(rid: string, key: string, fmt: LayerExportFormat): string {
  return apiUrl(`${runBase(rid)}/export/layer/${enc(key)}`, { fmt });
}

/** Folds k class array: 0 train, 1 test, 2 buffer, 255 outside the design. */
export async function getFoldClasses(rid: string, k: number, signal?: AbortSignal): Promise<Uint8Array> {
  const r = await getBin<Uint8Array>(`${runBase(rid)}/folds/${k}.bin`, "uint8", { signal });
  return r.data;
}

// ---------------------------------------------------------------- compare runs (api.md §6.5)

export type CompareRuns = {
  a: RunSummary;
  b: RunSummary;
  same: { data: boolean | null; config: boolean | null; code: boolean | null; grid: boolean };
  config_diff: { path: string; a: unknown; b: unknown }[];
  metrics: { key: string; a: number | null; b: number | null; delta: number | null }[];
  timings: { stage: string; a: number | null; b: number | null }[];
  scenarios: { name: string; a: Likely | null; b: Likely | null }[];
  climate: Record<string, unknown> | null;
  causal: Record<string, unknown> | null;
  environment: { added: string[]; removed: string[]; changed: { name: string; a: string; b: string }[] };
  outputs: { a_only: string[]; b_only: string[] };
};

export type PriorityAgreement = { kendall_tau: number; top_decile_jaccard: number; n: number };

export function useCompareRuns(a: string | null, b: string | null) {
  return useResource<CompareRuns>(a && b ? `compare:${a}:${b}` : null, (s) => api.get<CompareRuns>("/api/compare/runs", { a: a!, b: b! }, s), {
    tags: a && b ? [`run:${a}`, `run:${b}`] : [],
  });
}

/** `GET /api/compare/layer.bin` → Float32 (b − a); 409 grid_mismatch when the grids differ. */
export async function getCompareLayer(a: string, b: string, key: string, signal?: AbortSignal): Promise<Float32Array> {
  const r = await getBin<Float32Array>("/api/compare/layer.bin", "float32", { query: { a, b, key }, signal });
  return r.data;
}

export function postPriority(a: string, b: string, layer: string): Promise<PriorityAgreement> {
  return api.post<PriorityAgreement>("/api/compare/priority", { a, b, layer });
}

// ---------------------------------------------------------------- small helpers

/** Whether a run's outputs can still change (a job is or may be writing). */
export function runStatusIsLive(status: RunStatus | null | undefined): boolean {
  return status === "running" || status === "queued" || status === "external_live";
}
