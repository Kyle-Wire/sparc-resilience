// Launch-form state and the exact param shapes each study and post-run action posts (api.md §8).
// Forms keep editing state (text fields, "fine grid" toggles); `paramsFor` turns it into the
// wire params, and `paramProblems` lists what must be fixed before Launch is enabled. The
// defaults mirror the core library and CLI defaults.
import {
  PLACEBO_KINDS,
  SIM_GENERATORS,
  type BaselinesParams,
  type BenchmarkParams,
  type EmulatorParams,
  type LaunchableKind,
  type MultiverseParams,
  type PlaceboKind,
  type PlaceboParams,
  type PlannerParams,
  type ReproduceParams,
  type SimGenerator,
  type SimcheckParams,
  type StudyParams,
  type UncertaintyParams,
} from "../../../api/studies";
import { BASELINE_VARIANT, BUILTIN_NAMES, customVariantsJson, draftsFromJson, type CustomVariantDraft } from "./multiverse";

/** sparc.core.baselines.BASELINES with their labels. */
export const BASELINE_MODELS: readonly { id: string; label: string }[] = [
  { id: "regression_kriging", label: "Regression kriging (ridge + Matérn GP on residuals)" },
  { id: "hgb_xy", label: "Gradient boosting, covariates + x, y" },
  { id: "hgb", label: "Gradient boosting, covariates only" },
  { id: "idw", label: "Inverse-distance interpolation (no covariates)" },
  { id: "hgb_focal", label: "Gradient boosting on the stack's inputs (ablation)" },
];

export const PLACEBO_LABELS: Record<PlaceboKind, string> = {
  grf: "Random field (a fake layer with real spatial texture)",
  shift: "Shifted layers (half the city along each axis)",
  rotate: "Rotated layers (180°)",
};

export const GENERATOR_LABELS: Record<SimGenerator, string> = {
  physics: "Physics (energy balance with spill-over)",
  additive: "Additive (local, no spill-over)",
  own_only: "Own cell only",
  coarse_scale: "Coarse-scale effect",
  confounded: "Confounded by an unobserved layer",
  null: "Null (no effect: false positives)",
};

/** `sparc core simcheck --design` default. */
export const DEFAULT_DESIGN: Record<SimGenerator, number> = { physics: 20, additive: 20, own_only: 8, coarse_scale: 8, confounded: 8, null: 20 };

/** Stages a reproduction can re-run (`S0` always does: it reloads the data). */
export const REPRODUCE_STAGES = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"] as const;

export type BaselinesForm = { models: string[] };
export type PlannerForm = { package: string; thresholds: string; hex_sizes: string; export: boolean };
export type EmulatorForm = { patches: number };
export type UncertaintyForm = { mode: "attached" | "choose"; multiverse_study: string; simcheck_studies: string[]; placebo_study: string; real_r2_gate: boolean };
export type WriteupForm = Record<string, never>;
export type PlaceboForm = { kinds: PlaceboKind[]; fine: boolean; coarse_m: number; seed: number; grf_range_m: number };
export type SimcheckForm = { design: Record<SimGenerator, number>; fine: boolean; coarse_m: number; epochs: number; workers: number; threads: number; continue_study_id: string };
export type MultiverseForm = { variants: string[]; custom: CustomVariantDraft[]; fine: boolean; coarse_m: number; workers: number; threads: number };
export type ReproduceForm = { stages: string[]; tol_r2: number; tol_effect: number };
export type BenchmarkForm = { seed: number; ab: boolean; epochs: number; n: number };

export type Forms = {
  baselines: BaselinesForm;
  planner: PlannerForm;
  emulator: EmulatorForm;
  uncertainty: UncertaintyForm;
  writeup: WriteupForm;
  placebo: PlaceboForm;
  simcheck: SimcheckForm;
  multiverse: MultiverseForm;
  reproduce: ReproduceForm;
  benchmark: BenchmarkForm;
};

export function defaultForm<K extends LaunchableKind>(kind: K): Forms[K] {
  const d: Forms = {
    baselines: { models: BASELINE_MODELS.map((m) => m.id) },
    planner: { package: "", thresholds: "", hex_sizes: "250, 500", export: true },
    emulator: { patches: 8 },
    uncertainty: { mode: "attached", multiverse_study: "", simcheck_studies: [], placebo_study: "", real_r2_gate: false },
    writeup: {},
    placebo: { kinds: [...PLACEBO_KINDS], fine: false, coarse_m: 60, seed: 0, grf_range_m: 600 },
    simcheck: { design: { ...DEFAULT_DESIGN }, fine: false, coarse_m: 90, epochs: 200, workers: 1, threads: 1, continue_study_id: "" },
    // Every built-in variant, as `sparc core multiverse` runs by default (baseline alone compares nothing).
    multiverse: { variants: [...BUILTIN_NAMES], custom: [], fine: false, coarse_m: 60, workers: 1, threads: 1 },
    reproduce: { stages: ["S0", "S1", "S2", "S3"], tol_r2: 0.01, tol_effect: 0.05 },
    benchmark: { seed: 0, ab: true, epochs: 150, n: 96 },
  };
  return structuredClone(d[kind]);
}

/** "90, 95 °F" → [90, 95]; null when any item is not a finite number. Empty → []. */
export function parseNumberList(text: string): number[] | null {
  const parts = text
    .replace(/−/g, "-")
    .split(/[\s,;]+/)
    .map((s) => s.trim())
    .filter(Boolean);
  const out: number[] = [];
  for (const p of parts) {
    const v = Number(p);
    if (!Number.isFinite(v)) return null;
    out.push(v);
  }
  return out;
}

function baselinesParams(f: BaselinesForm): BaselinesParams {
  // Keep the core order whatever order the boxes were ticked in.
  return { models: BASELINE_MODELS.map((m) => m.id).filter((id) => f.models.includes(id)) };
}

function plannerParams(f: PlannerForm): PlannerParams {
  const p: PlannerParams = {};
  if (f.package.trim()) p.package = f.package.trim();
  const th = parseNumberList(f.thresholds);
  if (th && th.length) p.thresholds = th;
  const hex = parseNumberList(f.hex_sizes);
  if (hex && hex.length) p.hex_sizes = hex;
  p.export = f.export;
  return p;
}

function emulatorParams(f: EmulatorForm): EmulatorParams {
  return { patches: Math.round(f.patches) };
}

function uncertaintyParams(f: UncertaintyForm): UncertaintyParams {
  if (f.mode === "attached") return { real_r2_gate: f.real_r2_gate };
  const p: UncertaintyParams = {};
  if (f.multiverse_study) p.multiverse_study = f.multiverse_study;
  if (f.simcheck_studies.length) p.simcheck_studies = [...f.simcheck_studies];
  if (f.placebo_study) p.placebo_study = f.placebo_study;
  p.real_r2_gate = f.real_r2_gate;
  return p;
}

function placeboParams(f: PlaceboForm): PlaceboParams {
  return { kinds: PLACEBO_KINDS.filter((k) => f.kinds.includes(k)), coarse_m: f.fine ? null : f.coarse_m, seed: Math.round(f.seed), grf_range_m: f.grf_range_m };
}

function simcheckParams(f: SimcheckForm): SimcheckParams {
  const design = Object.fromEntries(SIM_GENERATORS.map((g) => [g, Math.max(0, Math.round(f.design[g] ?? 0))])) as Record<SimGenerator, number>;
  const p: SimcheckParams = { design, coarse_m: f.fine ? null : f.coarse_m, epochs: Math.round(f.epochs), workers: Math.round(f.workers), threads: Math.round(f.threads) };
  if (f.continue_study_id) p.continue_study_id = f.continue_study_id;
  return p;
}

function multiverseParams(f: MultiverseForm): MultiverseParams {
  const variants = [BASELINE_VARIANT, ...f.variants.filter((v) => v !== BASELINE_VARIANT)];
  const p: MultiverseParams = { variants, coarse_m: f.fine ? null : f.coarse_m, workers: Math.round(f.workers), threads: Math.round(f.threads) };
  const custom = customVariantsJson(f.custom);
  if (Object.keys(custom.value).length) p.custom_variants = custom.value;
  return p;
}

function reproduceParams(f: ReproduceForm): ReproduceParams {
  const stages = REPRODUCE_STAGES.filter((s) => s === "S0" || f.stages.includes(s));
  return { stages: [...stages], tol_r2: f.tol_r2, tol_effect: f.tol_effect };
}

function benchmarkParams(f: BenchmarkForm): BenchmarkParams {
  return { seed: Math.round(f.seed), ab: f.ab, epochs: Math.round(f.epochs), n: Math.round(f.n) };
}

/** The wire params of a kind's launch (api.md §8). */
export function paramsFor<K extends LaunchableKind>(kind: K, form: Forms[K]): StudyParams[K] {
  const f = form as unknown;
  switch (kind) {
    case "baselines":
      return baselinesParams(f as BaselinesForm) as StudyParams[K];
    case "planner":
      return plannerParams(f as PlannerForm) as StudyParams[K];
    case "emulator":
      return emulatorParams(f as EmulatorForm) as StudyParams[K];
    case "uncertainty":
      return uncertaintyParams(f as UncertaintyForm) as StudyParams[K];
    case "writeup":
      return {} as StudyParams[K];
    case "placebo":
      return placeboParams(f as PlaceboForm) as StudyParams[K];
    case "simcheck":
      return simcheckParams(f as SimcheckForm) as StudyParams[K];
    case "multiverse":
      return multiverseParams(f as MultiverseForm) as StudyParams[K];
    case "reproduce":
      return reproduceParams(f as ReproduceForm) as StudyParams[K];
    case "benchmark":
      return benchmarkParams(f as BenchmarkForm) as StudyParams[K];
  }
  throw new Error(`unknown kind ${String(kind)}`);
}

const num = (v: unknown, fallback: number): number => (typeof v === "number" && Number.isFinite(v) ? v : fallback);

/** The cell-size fields of a stored `coarse_m` (null = the fine grid). */
function cellOf(v: unknown, fallback: number): { fine: boolean; coarse_m: number } {
  return v === null ? { fine: true, coarse_m: fallback } : { fine: false, coarse_m: num(v, fallback) };
}

/**
 * A launch form back from a study's stored params (`Study.params`), so "Run again" starts from
 * the last study's settings; anything missing or unknown keeps the default. A simulation check
 * starts a new study (`continue_study_id` is never carried over).
 */
export function formFromParams<K extends LaunchableKind>(kind: K, params: Record<string, unknown> | null | undefined): Forms[K] {
  const base = defaultForm(kind);
  if (!params) return base;
  const list = (v: unknown): unknown[] | null => (Array.isArray(v) ? v : null);
  switch (kind) {
    case "placebo": {
      const f = base as PlaceboForm;
      const kinds = PLACEBO_KINDS.filter((k) => (list(params.kinds) ?? f.kinds).includes(k));
      const out: PlaceboForm = { ...f, ...cellOf(params.coarse_m, f.coarse_m), kinds: kinds.length ? kinds : f.kinds, seed: num(params.seed, f.seed), grf_range_m: num(params.grf_range_m, f.grf_range_m) };
      return out as Forms[K];
    }
    case "simcheck": {
      const f = base as SimcheckForm;
      const d = params.design && typeof params.design === "object" ? (params.design as Record<string, unknown>) : {};
      const design = Object.fromEntries(SIM_GENERATORS.map((g) => [g, Math.max(0, Math.round(num(d[g], params.design ? 0 : f.design[g])))])) as Record<SimGenerator, number>;
      const out: SimcheckForm = { ...f, ...cellOf(params.coarse_m, f.coarse_m), design, epochs: num(params.epochs, f.epochs), workers: num(params.workers, f.workers), threads: num(params.threads, f.threads), continue_study_id: "" };
      return out as Forms[K];
    }
    case "multiverse": {
      const f = base as MultiverseForm;
      const v = list(params.variants);
      const custom = params.custom_variants && typeof params.custom_variants === "object" ? (params.custom_variants as Record<string, Record<string, unknown>>) : null;
      const out: MultiverseForm = {
        ...f,
        ...cellOf(params.coarse_m, f.coarse_m),
        // no list = every built-in variant (core's default)
        variants: v ? BUILTIN_NAMES.filter((n) => n === BASELINE_VARIANT || v.includes(n)) : [...BUILTIN_NAMES],
        custom: draftsFromJson(custom),
        workers: num(params.workers, f.workers),
        threads: num(params.threads, f.threads),
      };
      return out as Forms[K];
    }
    case "reproduce": {
      const f = base as ReproduceForm;
      const st = list(params.stages);
      const out: ReproduceForm = { stages: st ? REPRODUCE_STAGES.filter((x) => st.includes(x)) : f.stages, tol_r2: num(params.tol_r2, f.tol_r2), tol_effect: num(params.tol_effect, f.tol_effect) };
      return out as Forms[K];
    }
    case "benchmark": {
      const f = base as BenchmarkForm;
      const out: BenchmarkForm = { seed: num(params.seed, f.seed), ab: typeof params.ab === "boolean" ? params.ab : f.ab, epochs: num(params.epochs, f.epochs), n: num(params.n, f.n) };
      return out as Forms[K];
    }
    default:
      return base;
  }
}

/** Why a form cannot be launched yet (empty = ready). `threadsHeavy` bounds workers × threads. */
export function paramProblems<K extends LaunchableKind>(kind: K, form: Forms[K], threadsHeavy: number | null): string[] {
  const out: string[] = [];
  const pool = (workers: number, threads: number) => {
    if (!(workers >= 1) || !(threads >= 1)) out.push("Workers and threads must be at least 1.");
    else if (threadsHeavy !== null && workers * threads > threadsHeavy)
      out.push(`Workers × threads is ${workers * threads}, above the heavy-job thread budget of ${threadsHeavy} (Settings).`);
  };
  const cell = (fine: boolean, m: number) => {
    if (!fine && !(m > 0)) out.push("The coarse cell size must be positive (or tick “fine grid”).");
  };
  switch (kind) {
    case "baselines":
      if (!(form as BaselinesForm).models.length) out.push("Pick at least one baseline model.");
      break;
    case "planner": {
      const f = form as PlannerForm;
      if (parseNumberList(f.thresholds) === null) out.push("Thresholds must be numbers separated by commas.");
      const hex = parseNumberList(f.hex_sizes);
      if (hex === null || hex.some((h) => !(h > 0))) out.push("Hexagon sizes must be positive numbers (metres).");
      break;
    }
    case "emulator": {
      const p = (form as EmulatorForm).patches;
      if (!(p >= 1 && p <= 64)) out.push("Validation patches must be between 1 and 64.");
      break;
    }
    case "uncertainty": {
      const f = form as UncertaintyForm;
      if (f.mode === "choose" && !f.multiverse_study && !f.simcheck_studies.length && !f.placebo_study) out.push("Choose at least one study, or use the attached studies.");
      break;
    }
    case "placebo": {
      const f = form as PlaceboForm;
      if (!f.kinds.length) out.push("Pick at least one placebo kind.");
      cell(f.fine, f.coarse_m);
      if (f.kinds.includes("grf") && !(f.grf_range_m > 0)) out.push("The random-field range must be positive.");
      break;
    }
    case "simcheck": {
      const f = form as SimcheckForm;
      const total = SIM_GENERATORS.reduce((a, g) => a + Math.max(0, Math.round(f.design[g] ?? 0)), 0);
      if (!total && !f.continue_study_id) out.push("The design has no replicates: give at least one generator a count.");
      cell(f.fine, f.coarse_m);
      if (!(f.epochs >= 1)) out.push("Epochs must be at least 1.");
      pool(f.workers, f.threads);
      break;
    }
    case "multiverse": {
      const f = form as MultiverseForm;
      out.push(...customVariantsJson(f.custom).errors);
      cell(f.fine, f.coarse_m);
      pool(f.workers, f.threads);
      break;
    }
    case "reproduce": {
      const f = form as ReproduceForm;
      // A zero tolerance would fail on float noise alone; the server refuses it too.
      if (!(f.tol_r2 > 0) || !(f.tol_effect > 0)) out.push("Tolerances must be greater than zero.");
      break;
    }
    case "benchmark": {
      const f = form as BenchmarkForm;
      if (!(f.n >= 16)) out.push("The synthetic city needs at least 16 cells per side.");
      if (!(f.epochs >= 1)) out.push("Epochs must be at least 1.");
      break;
    }
  }
  return out;
}

/** Total replicates of a simcheck design. */
export function designTotal(design: Record<string, number>): number {
  return Object.values(design).reduce((a, n) => a + Math.max(0, Math.round(Number(n) || 0)), 0);
}
