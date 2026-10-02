// Pure view models of the tracking pages (no React): stage rail items, fold × model grids,
// the stacker leaderboard, advection decision, per-stage panel data, Gantt rows, log lines,
// warning and output routing, child matrices and the Status Board layout. Everything here
// reads the reducer state of stores/tracker.ts (or wire rows) and is unit tested.
import type { ArtifactRow, LogLine, StatusBoard, StatusCell } from "../../api/tracking";
import type { Job, PlanNode, RunTabId, StageId } from "../../api/types";
import type { GanttRow } from "../../charts";
import { STAGE_IDS, stageOf, type SpanRow, type TrackerState } from "../../stores/tracker";

// ---------------------------------------------------------------- stages

export const STAGE_LABELS: Record<StageId, string> = {
  S0: "Data and QA",
  S1: "Area of influence",
  S2_S3: "Base models and stacker",
  baselines: "Reference baselines",
  cv_curve: "Skill vs distance",
  S4: "Response surfaces",
  S5: "Scenarios",
  climate: "Climate and adaptation",
  S6: "Causal validation",
  S7: "Budget optimisation",
  finish: "Manifest and report",
};

/** Short chip names (S0 … finish) as the rail shows them. */
export const STAGE_SHORT: Record<StageId, string> = {
  S0: "S0", S1: "S1", S2_S3: "S2–S3", baselines: "Baselines", cv_curve: "CV curve", S4: "S4", S5: "S5", climate: "Climate", S6: "S6", S7: "S7", finish: "Finish",
};

/** Checkpoint key of each stage (the done-set entry a checkpoint save adds; SPEC §5.3). */
const DEFAULT_CK: Partial<Record<StageId, string>> = { S1: "S3", S2_S3: "S3", baselines: "baselines", cv_curve: "cv_curve", S4: "S4", S5: "S5", climate: "climate", S6: "S6" };

/** Plain words for a skip / cached / plan reason (SPEC §5.3, §5.4). */
export function reasonText(reason: string | null | undefined): string {
  if (!reason) return "";
  if (reason === "checkpoint") return "loaded from the checkpoint";
  if (reason === "not_requested") return "not requested for this run";
  if (reason.startsWith("disabled_by_config:")) return `disabled in the config (${reason.slice("disabled_by_config:".length)})`;
  if (reason.startsWith("required_by:")) return `needed by ${reason.slice("required_by:".length)}`;
  const words: Record<string, string> = {
    no_budget: "no budget configured",
    no_treatments: "no causal treatments configured",
    no_responses: "no response surfaces to use",
    requires_S5: "needs the S5 scenarios",
    no_partitions: "no CV partitions left after the skip rules",
  };
  return words[reason] ?? reason.replace(/_/g, " ");
}

export type RailItem = {
  id: StageId;
  short: string;
  label: string;
  state: string;
  reason: string | null;
  reasonText: string;
  elapsed_s: number | null;
  est_s: number | null;
  progress: number | null;
  checkpoints: { action: string; done: string[]; bytes: number | null; ts: number | null }[];
};

/**
 * The eleven rail chips, S0 … finish: state, reason, duration (or estimate) and the
 * checkpoint saves that include the stage's checkpoint key.
 */
export function railItems(s: TrackerState, etaStages: Record<string, number> = {}): RailItem[] {
  const plan = new Map<string, PlanNode>((s.plan ?? []).map((n) => [n.id, n]));
  // Without a plan (self-tests, synthesised external runs) only the stages seen so far are known.
  const ids = s.plan ? STAGE_IDS : STAGE_IDS.filter((id) => s.stages?.[id]);
  return ids.map((id) => {
    const st = s.stages?.[id];
    const node = plan.get(id);
    const ck = node?.checkpoint_key ?? DEFAULT_CK[id] ?? null;
    const reason = st?.reason ?? null;
    const state = st?.state ?? (s.plan ? "not_requested" : "planned");
    return {
      id,
      short: STAGE_SHORT[id],
      label: node?.label || STAGE_LABELS[id],
      state,
      reason,
      reasonText: reasonText(reason),
      elapsed_s: st?.elapsed_s ?? null,
      est_s: etaStages[id] ?? st?.est_s ?? node?.est_s ?? null,
      progress: st?.progress ?? null,
      checkpoints: ck ? s.checkpoints.filter((c) => c.action === "saved" && c.done.includes(ck)).slice(0, 1) : [],
    };
  });
}

/**
 * What a stage panel says when it has nothing to show: why the stage did not (or will not)
 * produce data in this job, else `waiting` (data arrives while the stage runs).
 */
export function idleNote(s: TrackerState, stage: StageId, waiting: string): string {
  const st = s.stages?.[stage];
  switch (st?.state) {
    case "cached":
      return `${STAGE_SHORT[stage]} was loaded from the checkpoint; this job did not recompute it.`;
    case "skipped":
    case "disabled":
    case "not_requested":
      return `Not run: ${reasonText(st.reason) || st.state.replace(/_/g, " ")}.`;
    case "not_reached":
      return "The job ended before this stage.";
    case "cancelled":
      return "Cancelled before this stage finished.";
    case "failed":
      return "This stage failed; the logs and warnings below say why.";
    default:
      return waiting;
  }
}

/** The stage whose panel opens by default: the running one, else the latest that ran. */
export function defaultStage(s: TrackerState): StageId {
  if (s.stage && (STAGE_IDS as readonly string[]).includes(s.stage)) return s.stage as StageId;
  let best: StageId = "S0";
  let bestTs = -Infinity;
  for (const id of STAGE_IDS) {
    const st = s.stages?.[id];
    const t = st?.started_ts ?? null;
    if (t !== null && t >= bestTs) {
      best = id;
      bestTs = t;
    }
  }
  return best;
}

// ---------------------------------------------------------------- duplicate with changes

const CHECKLIST: readonly string[] = ["S0", "S1", "S2_S3", "S4", "S5", "S6", "S7"];

/**
 * "Duplicate with changes": the Launch page of the job's project prefilled with this run's
 * mode, coarse cell size, requested stages and CV-curve choice (its `mode`, `coarse`, `stages`
 * and `cv` query parameters), or a study's Validation tab. Null for other kinds.
 */
export function duplicateHref(job: Pick<Job, "kind" | "project_id" | "run_id">, run: { mode: string; coarse_m: number | null } | null, s: TrackerState): string | null {
  if (job.kind.startsWith("study.") && job.run_id) return `/r/${encodeURIComponent(job.run_id)}/validation`;
  if (job.kind !== "run.core" || !job.project_id) return null;
  const r = s.run;
  const q = new URLSearchParams();
  const mode = run && run.mode !== "custom" ? run.mode : r && typeof r.fast === "boolean" ? (r.coarse ? "coarse" : r.fast ? "fast" : "full") : null;
  if (mode === "fast" || mode === "coarse" || mode === "full") q.set("mode", mode);
  const coarse = run?.coarse_m ?? r?.coarse ?? null;
  if (mode === "coarse" && typeof coarse === "number" && coarse > 0) q.set("coarse", String(coarse));
  const asked = new Set((r?.stages ?? []).map((x) => (x === "S2" || x === "S3" ? "S2_S3" : x)));
  const stages = CHECKLIST.filter((x) => asked.has(x));
  if (stages.length && stages.length < CHECKLIST.length) q.set("stages", stages.join(","));
  if (r?.cv_curve === true || r?.cv_curve === false) q.set("cv", r.cv_curve ? "on" : "off");
  const qs = q.toString();
  return `/p/${encodeURIComponent(job.project_id)}/launch${qs ? `?${qs}` : ""}`;
}

// ---------------------------------------------------------------- span helpers

/** `name[k/n]` / `name[key]` of a path element → {name, k, n, key}. */
export function parseElement(el: string): { kind: string; name: string; k: number | null; n: number | null; key: string | null } {
  const i = el.indexOf(":");
  const kind = i >= 0 ? el.slice(0, i) : "";
  const rest = i >= 0 ? el.slice(i + 1) : el;
  const m = /^([^[]*)(?:\[(.*)\])?$/.exec(rest);
  const name = m?.[1] ?? rest;
  const inner = m?.[2];
  if (inner === undefined) return { kind, name, k: null, n: null, key: null };
  const kn = /^(\d+)\/(\d+)$/.exec(inner);
  if (kn) return { kind, name, k: Number(kn[1]), n: Number(kn[2]), key: null };
  return { kind, name, k: null, n: null, key: inner };
}

/** The `[k/n]` of the nearest ancestor element named `name` in a path, or null. */
export function kOf(path: readonly string[], name: string): { k: number; n: number } | null {
  for (let i = path.length - 1; i >= 0; i--) {
    const e = parseElement(path[i]);
    if (e.name === name && e.k !== null && e.n !== null) return { k: e.k, n: e.n };
  }
  return null;
}

/** True for spans of the job's own run (not of a nested study child run). */
export function isOwnSpan(sp: SpanRow): boolean {
  return !sp.path.slice(1).some((e) => e.startsWith("run:"));
}

/** The job's own spans matching `pred`, in start order. */
function spansOf(s: TrackerState, pred: (sp: SpanRow) => boolean): SpanRow[] {
  return Object.values(s.spans)
    .filter((sp) => isOwnSpan(sp) && pred(sp))
    .sort((a, b) => (a.started_ts ?? 0) - (b.started_ts ?? 0));
}

const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);

/** Human breadcrumb of a span path: `cv_curve › 1000 m blocks › fold 4/5 › mgwr`. */
export function pathCrumbs(s: TrackerState, path: readonly string[] | null): string[] {
  if (!path) return [];
  const byPath = new Map<string, SpanRow>();
  for (const sp of Object.values(s.spans)) byPath.set(sp.path.join("\u0001"), sp);
  const out: string[] = [];
  path.forEach((el, i) => {
    const e = parseElement(el);
    if (e.kind === "run" && i === 0) return;
    if (e.kind === "stage") {
      out.push(e.name);
      return;
    }
    const sp = byPath.get(path.slice(0, i + 1).join("\u0001"));
    const key = sp?.key ?? e.key;
    const kn = e.k !== null && e.n !== null ? `${e.k}/${e.n}` : null;
    if (e.name === "base_model" || e.name === "baseline_model" || e.name === "remote_object") out.push(key ?? e.name);
    else if (e.name === "cv_partition" || e.name === "placebo_kind" || e.name === "variant") out.push(key ?? (kn ? `${e.name.replace(/_/g, " ")} ${kn}` : e.name));
    else if (key && kn) out.push(`${e.name.replace(/_/g, " ")} ${kn} (${key})`);
    else if (kn) out.push(`${e.name.replace(/_/g, " ")} ${kn}`);
    else if (key) out.push(`${e.name.replace(/_/g, " ")} ${key}`);
    else out.push(e.kind === "run" ? `run ${e.name}` : e.name.replace(/_/g, " "));
  });
  return out;
}

/** The pid of the job's worker (from its root span id `<pid>:<n>`). */
export function workerPid(s: TrackerState): number | null {
  const root = Object.values(s.spans).find((sp) => sp.kind === "run" && sp.depth === 1) ?? Object.values(s.spans)[0];
  const pid = root ? Number(root.span_id.split(":")[0]) : NaN;
  if (Number.isFinite(pid)) return pid;
  const hb = Object.keys(s.hb_last)[0];
  return hb && Number.isFinite(Number(hb)) ? Number(hb) : null;
}

// ---------------------------------------------------------------- fold × model

export type FoldCell = { seconds: number | null; rmse: number | null; r2: number | null; status: "running" | "ok" | "error" | "cancelled" | "pending"; spanId: string | null };

export type FoldGrid = { folds: string[]; models: string[]; cells: FoldCell[][]; running: [number, number] | null };

/**
 * The fold × model grid of `base_model` tasks under `stage` (S2_S3, or cv_curve restricted
 * to one partition): seconds, held-out RMSE and R² from each `task.end`'s metrics.
 */
export function foldModelGrid(s: TrackerState, stage: "S2_S3" | "cv_curve", partition?: string | null): FoldGrid {
  const tasks = spansOf(s, (sp) => sp.kind === "task" && sp.name === "base_model" && stageOf(sp.path) === stage && (!partition || sp.ctx?.partition === partition));
  const models: string[] = [];
  let K = 0;
  for (const sp of tasks) {
    const m = sp.key ?? parseElement(sp.path[sp.path.length - 1]).key ?? "model";
    if (!models.includes(m)) models.push(m);
    const f = kOf(sp.path, "fold");
    if (f) K = Math.max(K, f.n);
  }
  const folds = Array.from({ length: K }, (_, i) => `Fold ${i + 1}`);
  const cells: FoldCell[][] = folds.map(() => models.map(() => ({ seconds: null, rmse: null, r2: null, status: "pending" as const, spanId: null })));
  let running: [number, number] | null = null;
  for (const sp of tasks) {
    const f = kOf(sp.path, "fold");
    if (!f) continue;
    const m = models.indexOf(sp.key ?? parseElement(sp.path[sp.path.length - 1]).key ?? "model");
    const met = sp.metrics ?? {};
    cells[f.k - 1][m] = {
      seconds: num(met.fit_s) ?? sp.elapsed_s,
      rmse: num(met.heldout_rmse),
      r2: num(met.heldout_r2),
      status: sp.status,
      spanId: sp.span_id,
    };
    if (sp.status === "running") running = [f.k - 1, m];
  }
  return { folds, models, cells, running };
}

/** CV-curve partitions in order, with the one running (or the latest) marked active. */
export function cvPartitions(s: TrackerState): { label: string; status: string; active: boolean }[] {
  const parts = spansOf(s, (sp) => sp.kind === "task" && sp.name === "cv_partition");
  const out = parts.map((sp) => ({ label: (sp.key ?? (sp.ctx?.partition as string | undefined) ?? `partition ${sp.k ?? "?"}`) as string, status: sp.status, active: false }));
  const run = out.findIndex((p) => p.status === "running");
  if (run >= 0) out[run].active = true;
  else if (out.length) out[out.length - 1].active = true;
  return out;
}

// ---------------------------------------------------------------- stacker leaderboard

export type LeaderRow = { candidate: string; rmse: number; winner: boolean; relToBest: number };

/**
 * Stacker candidates from the `candidate_rmse` metrics emitted inside `stage` (S2_S3 by default;
 * cv_curve re-fits them per partition): OOF RMSE per candidate, sorted, with the winner by the
 * core rule (simplest first; a later candidate wins only when it is more than 0.1% better).
 */
export function stackerLeaderboard(s: TrackerState, stage: "S2_S3" | "cv_curve" = "S2_S3"): LeaderRow[] {
  const st = s.stages?.[stage];
  const lo = st?.started_ts ?? null;
  const hi = st?.ended_ts ?? null;
  const order: string[] = [];
  const latest = new Map<string, number>();
  const points: { cand: string; ts: number | null; value: number }[] = [];
  for (const [key, series] of Object.entries(s.metric_series)) {
    const m = /^candidate_rmse\{candidate=(.*)\}$/.exec(key);
    if (!m) continue;
    for (const p of series) points.push({ cand: m[1], ts: p.ts, value: p.value });
  }
  points.sort((a, b) => (a.ts ?? 0) - (b.ts ?? 0));
  for (const p of points) {
    if (lo !== null && p.ts !== null && p.ts < lo - 1e-6) continue;
    if (hi !== null && p.ts !== null && p.ts > hi + 1e-6) continue;
    if (!order.includes(p.cand)) order.push(p.cand);
    latest.set(p.cand, p.value);
  }
  if (!order.length) {
    // no series (older snapshot): the latest values
    for (const [key, v] of Object.entries(s.metrics_latest)) {
      const m = /^candidate_rmse\{candidate=(.*)\}$/.exec(key);
      if (m && typeof v.value === "number") {
        order.push(m[1]);
        latest.set(m[1], v.value);
      }
    }
  }
  let best: { cand: string; rmse: number } | null = null;
  for (const c of order) {
    const r = latest.get(c)!;
    if (best === null || r < best.rmse * (1 - 1e-3)) best = { cand: c, rmse: r };
  }
  const min = Math.min(...order.map((c) => latest.get(c)!));
  return order
    .map((c) => ({ candidate: c, rmse: latest.get(c)!, winner: best?.cand === c, relToBest: latest.get(c)! / min - 1 }))
    .sort((a, b) => a.rmse - b.rmse || order.indexOf(a.candidate) - order.indexOf(b.candidate));
}

/** Plain name of a stacker candidate key (`mean`, `nnls`, `residual:<λ>`). */
export function candidateLabel(c: string): string {
  if (c === "mean") return "equal-weight mean";
  if (c === "nnls") return "convex (NNLS) blend";
  if (c.startsWith("residual:")) return `blend + neural residual (λ_PDE = ${c.slice(9)})`;
  return c;
}

// ---------------------------------------------------------------- advection decision

export type Advection =
  | { state: "none" }
  | { state: "checking"; done: number; total: number | null }
  | { state: "decided"; kept: boolean; mean: number; se: number; folds: number };

/** The physics advection check of S2_S3: kept only if refits with v = 0 are worse by more than one SE. */
export function advectionDecision(s: TrackerState): Advection {
  const check = spansOf(s, (sp) => sp.name === "advection_check" && stageOf(sp.path) === "S2_S3").pop();
  if (!check) return { state: "none" };
  const refits = spansOf(s, (sp) => sp.name === "adv_refit" && sp.status === "ok" && stageOf(sp.path) === "S2_S3");
  const d = refits
    .map((sp) => {
      const a = num(sp.metrics?.heldout_rmse_adv);
      const b = num(sp.metrics?.heldout_rmse_no_adv);
      return a !== null && b !== null ? a - b : null;
    })
    .filter((v): v is number => v !== null);
  if (check.status === "running" || !d.length) {
    const total = refits[0]?.n ?? spansOf(s, (sp) => sp.name === "adv_refit")[0]?.n ?? null;
    return { state: "checking", done: refits.length, total };
  }
  const mean = d.reduce((a, b) => a + b, 0) / d.length;
  const se = d.length > 1 ? Math.sqrt(d.reduce((a, b) => a + (b - mean) ** 2, 0) / (d.length - 1)) / Math.sqrt(d.length) : 0;
  return { state: "decided", kept: mean < -se, mean, se, folds: d.length };
}

/** Per-fold metrics of the physics base model (fit time, held-out skill and any extra params). */
export function physicsPerFold(s: TrackerState): { fold: number; metrics: Record<string, number> }[] {
  return spansOf(s, (sp) => sp.name === "base_model" && sp.key === "physics" && stageOf(sp.path) === "S2_S3" && sp.status === "ok")
    .map((sp) => {
      const f = kOf(sp.path, "fold");
      const metrics: Record<string, number> = {};
      for (const [k, v] of Object.entries(sp.metrics ?? {})) if (typeof v === "number") metrics[k] = v;
      return { fold: f?.k ?? 0, metrics };
    })
    .sort((a, b) => a.fold - b.fold);
}

// ---------------------------------------------------------------- other stage panels

export function metricValue(s: TrackerState, key: string): number | null {
  const v = s.metrics_latest[key]?.value;
  return typeof v === "number" ? v : null;
}

/** S0 summary (the stage's `stage.end` summary) and its QA flags (`qa.*` warnings). */
export function s0Summary(s: TrackerState): { summary: Record<string, unknown> | null; flags: { code: string; message: string; count: number }[] } {
  const sp = spansOf(s, (x) => x.kind === "stage" && x.name === "S0").pop();
  const flags = Object.values(s.warnings)
    .filter((w) => w.code.startsWith("qa."))
    .map((w) => ({ code: w.code, message: w.message, count: w.count }));
  const summary = sp && sp.status !== "running" && Object.keys(sp.metrics ?? {}).length ? sp.metrics : null;
  return { summary, flags };
}

/** S1 influence ranges and anisotropy per predictor, in emission order. */
export function influenceRows(s: TrackerState): { predictor: string; range_m: number | null; anisotropy: number | null }[] {
  const rows = new Map<string, { predictor: string; range_m: number | null; anisotropy: number | null }>();
  for (const [key, v] of Object.entries(s.metrics_latest)) {
    const m = /^influence\.(range_m|anisotropy_ratio)\{predictor=(.*)\}$/.exec(key);
    if (!m) continue;
    const row = rows.get(m[2]) ?? { predictor: m[2], range_m: null, anisotropy: null };
    if (m[1] === "range_m") row.range_m = typeof v.value === "number" ? v.value : null;
    else row.anisotropy = typeof v.value === "number" ? v.value : null;
    rows.set(m[2], row);
  }
  return [...rows.values()];
}

/** Baseline held-out RMSE per model and fold (`baseline_model` tasks; fold order = emission order). */
export function baselineRows(s: TrackerState): { model: string; rmse: number[] }[] {
  const out = new Map<string, number[]>();
  for (const sp of spansOf(s, (x) => x.name === "baseline_model" && stageOf(x.path) === "baselines" && x.status === "ok")) {
    const r = num(sp.metrics?.heldout_rmse);
    if (r === null) continue;
    const m = sp.key ?? "baseline";
    out.set(m, [...(out.get(m) ?? []), r]);
  }
  return [...out.entries()].map(([model, rmse]) => ({ model, rmse }));
}

/** Mean and standard error of a list (SE = sd / √n, 0 for one value). */
export function meanSe(xs: number[]): { mean: number; se: number } {
  const n = xs.length;
  const mean = xs.reduce((a, b) => a + b, 0) / Math.max(n, 1);
  const se = n > 1 ? Math.sqrt(xs.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1)) / Math.sqrt(n) : 0;
  return { mean, se };
}

/** CV curve points: block size (m, parsed from the partition label), R² and RMSE. */
export function cvCurvePoints(s: TrackerState): { label: string; block_m: number | null; r2: number | null; rmse: number | null }[] {
  const rows = new Map<string, { label: string; block_m: number | null; r2: number | null; rmse: number | null; ts: number }>();
  for (const [key, v] of Object.entries(s.metrics_latest)) {
    const m = /^cv_row\.(r2|rmse)\{partition=(.*)\}$/.exec(key);
    if (!m) continue;
    const label = m[2];
    const bm = /([\d.]+)\s*m\b/.exec(label);
    const row = rows.get(label) ?? { label, block_m: bm ? Number(bm[1]) : null, r2: null, rmse: null, ts: v.ts ?? 0 };
    if (m[1] === "r2") row.r2 = typeof v.value === "number" ? v.value : null;
    else row.rmse = typeof v.value === "number" ? v.value : null;
    rows.set(label, row);
  }
  return [...rows.values()].sort((a, b) => (a.block_m ?? 0) - (b.block_m ?? 0) || a.ts - b.ts).map(({ ts: _t, ...r }) => r);
}

export type DosePoint = { dose: number; key: string; mean: number | null; se: number | null; frac_extrapolated: number | null; status: string };

/** S4 dose–response points per lever (`variable[v/V]` › `dose[d/D]` tasks). */
export function doseCurves(s: TrackerState): { lever: string; points: DosePoint[]; running: boolean }[] {
  const levers = new Map<string, { lever: string; points: DosePoint[]; running: boolean }>();
  const varByPath = new Map<string, string>();
  for (const sp of spansOf(s, (x) => x.name === "variable" && stageOf(x.path) === "S4")) varByPath.set(sp.path.join("\u0001"), sp.key ?? parseElement(sp.path[sp.path.length - 1]).key ?? "lever");
  for (const sp of spansOf(s, (x) => x.name === "dose" && stageOf(x.path) === "S4")) {
    const parent = sp.path.slice(0, -1).join("\u0001");
    const lever = varByPath.get(parent) ?? "lever";
    const entry = levers.get(lever) ?? { lever, points: [], running: false };
    const met = sp.metrics ?? {};
    const dose = Number(sp.key);
    entry.points.push({ dose: Number.isFinite(dose) ? dose : sp.k ?? entry.points.length + 1, key: sp.key ?? String(sp.k ?? ""), mean: num(met.mean_benefit), se: num(met.mean_se), frac_extrapolated: num(met.frac_extrapolated), status: sp.status });
    if (sp.status === "running") entry.running = true;
    levers.set(lever, entry);
  }
  for (const l of levers.values()) l.points.sort((a, b) => a.dose - b.dose);
  return [...levers.values()];
}

/** S5 scenario feed: one row per scenario from the `scenario.*` metrics, in emission order. */
export function scenarioFeed(s: TrackerState): { name: string; mean_delta: number | null; se: number | null; frac_extrapolated: number | null; ts: number }[] {
  const rows = new Map<string, { name: string; mean_delta: number | null; se: number | null; frac_extrapolated: number | null; ts: number }>();
  for (const [key, v] of Object.entries(s.metrics_latest)) {
    const m = /^scenario\.(mean_delta|se|frac_extrapolated)\{scenario=(.*)\}$/.exec(key);
    if (!m) continue;
    const row = rows.get(m[2]) ?? { name: m[2], mean_delta: null, se: null, frac_extrapolated: null, ts: v.ts ?? 0 };
    const val = typeof v.value === "number" ? v.value : null;
    if (m[1] === "mean_delta") row.mean_delta = val;
    else if (m[1] === "se") row.se = val;
    else row.frac_extrapolated = val;
    row.ts = Math.min(row.ts, v.ts ?? row.ts);
    rows.set(m[2], row);
  }
  return [...rows.values()].sort((a, b) => a.ts - b.ts);
}

/** CMIP6 model fetches (`remote_object` tasks with unit `climate_model`): done, running, total if planned. */
export function climateTicker(s: TrackerState): { done: string[]; running: string[]; planned: number | null } {
  const tasks = spansOf(s, (x) => x.name === "remote_object" && x.unit === "climate_model");
  return {
    done: tasks.filter((t) => t.status === "ok").map((t) => t.key ?? "model"),
    running: tasks.filter((t) => t.status === "running").map((t) => t.key ?? "model"),
    planned: s.planned_units.climate_model ?? null,
  };
}

export const CAUSAL_STEPS: { name: string; label: string }[] = [
  { name: "model_effects", label: "Model effects" },
  { name: "dml", label: "DML" },
  { name: "spillover", label: "Spillover" },
  { name: "cate", label: "CATE" },
  { name: "dr_curve", label: "DR curve" },
  { name: "sensitivity", label: "Sensitivity" },
  { name: "audit", label: "Audit" },
];

export type TreatmentRow = {
  treatment: string;
  steps: Record<string, { status: string; frac: number | null }>;
  theta: number | null;
  theta_se: number | null;
  verdicts: { check: string; verdict: string }[];
};

/** S6 treatment × step checklist with θ ± SE and the audit verdicts (`causal.audit_flag` warnings). */
export function causalChecklist(s: TrackerState): TreatmentRow[] {
  const rows = new Map<string, TreatmentRow>();
  const row = (t: string) => {
    let r = rows.get(t);
    if (!r) {
      r = { treatment: t, steps: {}, theta: null, theta_se: null, verdicts: [] };
      rows.set(t, r);
    }
    return r;
  };
  for (const sp of spansOf(s, (x) => x.name === "model_effects" && stageOf(x.path) === "S6")) {
    const r = row(sp.key ?? "treatment");
    r.steps.model_effects = { status: sp.status, frac: sp.status === "running" ? s.partial[sp.span_id]?.[2] ?? null : null };
  }
  const treatByPath = new Map<string, string>();
  for (const sp of spansOf(s, (x) => x.name === "treatment" && stageOf(x.path) === "S6")) {
    const t = sp.key ?? "treatment";
    treatByPath.set(sp.path.join("\u0001"), t);
    const r = row(t);
    r.theta = num(sp.metrics?.theta) ?? r.theta;
    r.theta_se = num(sp.metrics?.theta_se) ?? r.theta_se;
  }
  for (const sp of spansOf(s, (x) => stageOf(x.path) === "S6" && CAUSAL_STEPS.some((c) => c.name === x.name) && x.name !== "model_effects")) {
    const t = treatByPath.get(sp.path.slice(0, -1).join("\u0001"));
    if (!t) continue;
    row(t).steps[sp.name] = { status: sp.status, frac: sp.status === "running" ? s.partial[sp.span_id]?.[2] ?? null : null };
  }
  for (const [key, v] of Object.entries(s.metrics_latest)) {
    const m = /^(theta|theta_se)\{treatment=(.*)\}$/.exec(key);
    if (m && typeof v.value === "number") {
      const r = row(m[2]);
      if (m[1] === "theta") r.theta = v.value;
      else r.theta_se = v.value;
    }
  }
  for (const w of Object.values(s.warnings)) {
    if (w.code !== "causal.audit_flag") continue;
    const t = String(w.data?.treatment ?? "");
    if (!t) continue;
    row(t).verdicts.push({ check: String(w.data?.check ?? ""), verdict: String(w.data?.verdict ?? "") });
  }
  return [...rows.values()];
}

/** Chip status word of an audit verdict ("consistent" / "magnitude differs" / "sign conflict"). */
export function verdictStatus(verdict: string): { status: string; text: string } {
  const v = verdict.toLowerCase();
  if (v.includes("sign")) return { status: "failed", text: verdict || "sign conflict" };
  if (v.includes("magnitude")) return { status: "stale", text: verdict || "magnitude differs" };
  return { status: "done", text: verdict || "consistent" };
}

/** S7 planned vs realised totals per variable. */
export function budgetTotals(s: TrackerState): { variable: string; planned: number | null; realised: number | null }[] {
  const rows = new Map<string, { variable: string; planned: number | null; realised: number | null }>();
  for (const [key, v] of Object.entries(s.metrics_latest)) {
    const m = /^(planned_total|realised_total)(?:\{variable=(.*)\})?$/.exec(key);
    if (!m) continue;
    const name = m[2] ?? "all";
    const r = rows.get(name) ?? { variable: name, planned: null, realised: null };
    if (m[1] === "planned_total") r.planned = typeof v.value === "number" ? v.value : null;
    else r.realised = typeof v.value === "number" ? v.value : null;
    rows.set(name, r);
  }
  return [...rows.values()];
}

// ---------------------------------------------------------------- gantt

function spanLabel(sp: { kind: string; name: string; key: string | null; k: number | null; n: number | null }): string {
  if (sp.kind === "run") return `run ${sp.name}`;
  if (sp.kind === "stage") return STAGE_LABELS[sp.name as StageId] ? `${sp.name} · ${STAGE_LABELS[sp.name as StageId]}` : sp.name;
  const base = sp.name.replace(/_/g, " ");
  if (sp.k !== null && sp.n !== null) return `${base} ${sp.k}/${sp.n}${sp.key ? ` (${sp.key})` : ""}`;
  if (sp.key) return `${base} ${sp.key}`;
  return base;
}

/** Gantt rows (span tree) from span rows, optionally only those overlapping a time window. */
export function ganttRows(spans: { span_id: string; parent_id: string | null; kind: string; name: string; key: string | null; k: number | null; n: number | null; status: string; started_ts: number | null; ended_ts: number | null }[], window?: { from: number; to: number } | null): GanttRow[] {
  return spans
    .filter((sp) => sp.started_ts !== null)
    .filter((sp) => !window || ((sp.ended_ts ?? Infinity) >= window.from && (sp.started_ts as number) <= window.to))
    .map((sp) => ({ id: sp.span_id, parentId: sp.parent_id, label: spanLabel(sp), start: sp.started_ts as number, end: sp.ended_ts, status: sp.status }));
}

// ---------------------------------------------------------------- logs

const LEVELS: Record<string, number> = { debug: 10, info: 20, warning: 30, error: 40, critical: 50 };

export type LogFilter = { level: string; logger: string; stage: string; q: string };

/** The same filter rules as `GET /api/jobs/{jid}/logs` (min level, logger prefix, stage, substring). */
export function matchesLog(line: LogLine, f: LogFilter): boolean {
  if ((LEVELS[line.level] ?? 20) < (LEVELS[f.level || "debug"] ?? 10)) return false;
  if (f.logger && !line.logger.startsWith(f.logger)) return false;
  if (f.stage && stageOf(line.path) !== f.stage) return false;
  if (f.q && !line.msg.toLowerCase().includes(f.q.toLowerCase())) return false;
  return true;
}

// ---------------------------------------------------------------- routing to run views

const STAGE_TAB: Record<string, RunTabId> = {
  S0: "data", S1: "influence", S2_S3: "accuracy", baselines: "distance", cv_curve: "distance", S4: "response", S5: "scenarios",
  climate: "climate", S6: "causal", S7: "budget", finish: "docs",
};

const WARNING_TAB: [string, RunTabId][] = [
  ["qa.", "data"], ["forcing.", "data"], ["cv.", "distance"], ["baselines.", "distance"], ["influence.", "influence"], ["physics.", "accuracy"],
  ["interval.", "uncertainty"], ["causal.", "causal"], ["checkpoint.", "files"], ["reproduce.", "validation"], ["planner.", "planner"],
  ["layers.", "map"], ["climate.", "climate"],
];

/** The run tab a warning code points at (meta's `warning_codes[].view` wins when given). */
export function warningTab(code: string, metaView?: string | null): RunTabId | null {
  if (metaView) return metaView as RunTabId;
  for (const [prefix, tab] of WARNING_TAB) if (code.startsWith(prefix)) return tab;
  return null;
}

/** The run tab that shows an artifact: the catalog's view for its file, else its stage's tab. */
export function artifactTab(a: ArtifactRow, catalog?: { files: string[]; view: string }[] | null): RunTabId | null {
  for (const o of catalog ?? []) {
    if (o.files.some((g) => globMatch(g, a.relpath))) return o.view as RunTabId;
  }
  return a.stage ? STAGE_TAB[a.stage] ?? null : null;
}

/** Run tab of a stage (Status Board and rail links). */
export function stageTab(stage: string): RunTabId | null {
  return STAGE_TAB[stage] ?? null;
}

/** `*` (no slash) and `**` glob match against a run-relative path. */
export function globMatch(glob: string, path: string): boolean {
  const re = glob
    .split("**")
    .map((part) => part.split("*").map((p) => p.replace(/[.+?^${}()|[\]\\]/g, "\\$&")).join("[^/]*"))
    .join(".*");
  return new RegExp(`^${re}$`).test(path);
}

// ---------------------------------------------------------------- study children

export type ChildCell = { key: string; label: string; status: string; progress: number | null; run_id: string | null; stages: Record<string, { state: string; reason: string | null }> | null; metrics: Record<string, unknown> };

/** Children of a study job keyed by kind (`placebo:<kind>`, `variant:<name>`, …) with their mini rails. */
export function childCells(s: TrackerState): ChildCell[] {
  return Object.values(s.children).map((c) => {
    const st = c.state;
    const status = st.run_status === "succeeded" || st.run_status === "failed" || st.run_status === "cancelled" ? st.run_status : st.run_status === "running" ? "running" : "planned";
    const metrics: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(st.metrics_latest)) metrics[k] = v.value;
    return {
      key: c.key, label: c.label, status, progress: st.progress, run_id: c.run_id,
      stages: st.stages ? Object.fromEntries(Object.entries(st.stages).map(([k, v]) => [k, { state: v.state, reason: v.reason }])) : null, metrics,
    };
  });
}

export type PriorityRow = { variant: string; tau: number | null; jaccard: number | null; levers: number };

function median(xs: number[]): number | null {
  if (!xs.length) return null;
  const v = [...xs].sort((a, b) => a - b);
  const m = v.length >> 1;
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}

/**
 * Priority stability of a multiverse (`view.priority`, as `multiverse.summarize` writes it:
 * variant → lever → {kendall_tau, top_decile_jaccard} against the baseline priority map):
 * per variant, the median over levers of Kendall's τ and of the top-decile Jaccard overlap.
 */
export function priorityStability(priority: unknown): PriorityRow[] {
  if (!priority || typeof priority !== "object") return [];
  const out: PriorityRow[] = [];
  for (const [variant, levers] of Object.entries(priority as Record<string, unknown>)) {
    if (!levers || typeof levers !== "object") continue;
    const taus: number[] = [];
    const jacs: number[] = [];
    for (const v of Object.values(levers as Record<string, { kendall_tau?: unknown; top_decile_jaccard?: unknown } | null>)) {
      const t = num(v?.kendall_tau);
      const j = num(v?.top_decile_jaccard);
      if (t !== null) taus.push(t);
      if (j !== null) jacs.push(j);
    }
    out.push({ variant, tau: median(taus), jaccard: median(jacs), levers: Object.keys(levers as object).length });
  }
  return out;
}

export type SimCell = { generator: string; seed: number; status: "pending" | "running" | "done" | "error" | "gate_fail"; share: number | null; seconds: number | null; redraw: boolean };

/**
 * The simcheck generator × seed grid: the design from the job params, replicate tasks
 * (`replicate[<generator>/<seed>]`) and `share` metrics; a study view (when available) adds
 * gate redraws and statuses the events do not carry.
 */
export function simcheckGrid(s: TrackerState, job: Pick<Job, "params">, view?: { grid?: { generator: string; seed: number; status: string; share: number | null; seconds: number | null; gate_attempt: number | null }[] } | null): { generators: string[]; seeds: number[]; cells: Map<string, SimCell> } {
  const design = (job.params?.design ?? {}) as Record<string, number>;
  const generators = Object.keys(design).filter((g) => Number(design[g]) > 0);
  let maxSeed = Math.max(0, ...generators.map((g) => Number(design[g])));
  const cells = new Map<string, SimCell>();
  const put = (c: SimCell) => {
    if (!generators.includes(c.generator)) generators.push(c.generator);
    maxSeed = Math.max(maxSeed, c.seed + 1);
    cells.set(`${c.generator}/${c.seed}`, c);
  };
  for (const g of view?.grid ?? []) {
    put({ generator: g.generator, seed: g.seed, status: (["pending", "running", "done", "error", "gate_fail"].includes(g.status) ? g.status : "pending") as SimCell["status"], share: g.share, seconds: g.seconds, redraw: (g.gate_attempt ?? 0) > 0 });
  }
  for (const sp of spansOf(s, (x) => x.name === "replicate")) {
    const key = sp.key ?? parseElement(sp.path[sp.path.length - 1]).key ?? "";
    const [gen, seedText] = key.split("/");
    const seed = Number(seedText);
    if (!gen || !Number.isFinite(seed)) continue;
    const prev = cells.get(`${gen}/${seed}`);
    const status: SimCell["status"] = sp.status === "ok" ? "done" : sp.status === "running" ? "running" : "error";
    put({ generator: gen, seed, status: prev && prev.status === "gate_fail" ? prev.status : status, share: prev?.share ?? null, seconds: sp.elapsed_s ?? prev?.seconds ?? null, redraw: prev?.redraw ?? false });
  }
  for (const [key, v] of Object.entries(s.metrics_latest)) {
    const m = /^share\{generator=([^,}]*),seed=([^,}]*)\}$/.exec(key);
    if (!m || typeof v.value !== "number") continue;
    const seed = Number(m[2]);
    const prev = cells.get(`${m[1]}/${seed}`);
    put({ generator: m[1], seed, status: prev?.status === "error" ? "error" : "done", share: v.value, seconds: prev?.seconds ?? null, redraw: prev?.redraw ?? false });
  }
  return { generators, seeds: Array.from({ length: maxSeed }, (_, i) => i), cells };
}

// ---------------------------------------------------------------- queue order

type QueuedJob = Pick<Job, "id" | "priority" | "created_utc">;

/** True when the scheduler visits `a` before `b` (priority descending, then oldest first). */
function visitsBefore(a: QueuedJob, pa: number, b: QueuedJob, pb: number): boolean {
  return pa > pb || (pa === pb && a.created_utc < b.created_utc);
}

/**
 * Priority changes that swap the queued job at `i` with its neighbour above (`dir` −1) or below
 * (+1) and move nothing else. The scheduler orders a lane by priority, then age, so one new
 * priority can jump a job past several neighbours of equal priority; both ways of restoring the
 * order (raising jobs from the bottom up, lowering them from the top down) are tried and the
 * one that patches fewer jobs is returned.
 */
export function queueMovePatches(lane: readonly QueuedJob[], i: number, dir: -1 | 1): { id: string; priority: number }[] {
  const j = i + dir;
  if (i < 0 || j < 0 || i >= lane.length || j >= lane.length) return [];
  const order = [...lane];
  [order[i], order[j]] = [order[j], order[i]];
  const n = order.length;
  const raise = order.map((x) => x.priority);
  for (let k = n - 2; k >= 0; k--) {
    if (!visitsBefore(order[k], raise[k], order[k + 1], raise[k + 1])) raise[k] = raise[k + 1] + (order[k].created_utc < order[k + 1].created_utc ? 0 : 1);
  }
  const lower = order.map((x) => x.priority);
  for (let k = 1; k < n; k++) {
    if (!visitsBefore(order[k - 1], lower[k - 1], order[k], lower[k])) lower[k] = lower[k - 1] - (order[k - 1].created_utc < order[k].created_utc ? 0 : 1);
  }
  const diff = (p: number[]) => order.flatMap((x, k) => (p[k] !== x.priority ? [{ id: x.id, priority: p[k] }] : []));
  const up = diff(raise);
  const down = diff(lower);
  return down.length < up.length ? down : up;
}

// ---------------------------------------------------------------- status board

export type BoardGroup = { group: "stage" | "post" | "study"; label: string; columns: StatusBoard["columns"] };

/** Columns grouped as the board header shows them (stages | post-run | studies), in server order. */
export function boardGroups(board: StatusBoard): BoardGroup[] {
  const labels = { stage: "Pipeline stages", post: "Post-run actions", study: "Studies" } as const;
  const out: BoardGroup[] = [];
  for (const c of board.columns) {
    const last = out[out.length - 1];
    if (last && last.group === c.group) last.columns.push(c);
    else out.push({ group: c.group, label: labels[c.group] ?? c.group, columns: [c] });
  }
  return out;
}

/** Chip status word and meta text of a board cell. */
export function boardCellChip(cell: StatusCell | undefined): { status: string; text: string; meta: string | null } {
  if (!cell) return { status: "not_run", text: "not run", meta: null };
  switch (cell.state) {
    case "done":
      return { status: "done", text: "done", meta: cell.seconds !== null ? secondsText(cell.seconds) : null };
    case "running":
      // a job still waiting (a launch's "then" chain, the queue) is a running cell with reason queued/blocked
      if (cell.reason === "queued" || cell.reason === "blocked") return { status: cell.reason, text: cell.reason, meta: null };
      return { status: "running", text: "running", meta: cell.progress !== null ? `${Math.round(cell.progress * 100)}%` : null };
    case "skipped":
      return { status: "skipped", text: "skipped", meta: cell.reason ? reasonText(cell.reason) : null };
    case "not_run":
      return { status: "not_run", text: "not run", meta: null };
    default:
      return { status: cell.state, text: cell.state.replace(/_/g, " "), meta: cell.reason ? reasonText(cell.reason) : null };
  }
}

function secondsText(s: number): string {
  if (s < 10) return `${s.toFixed(1)} s`;
  if (s < 60) return `${Math.round(s)} s`;
  if (s < 3600) return `${Math.round(s / 60)} min`;
  const h = Math.floor(s / 3600);
  const m = Math.round((s - h * 3600) / 60);
  return m ? `${h} h ${m} m` : `${h} h`;
}

/** Where a board cell click goes: its job's tracker (stage columns), its study, or the analysis view. */
export function boardCellHref(rid: string, column: string, group: string, cell: StatusCell | undefined): string | null {
  if (!cell || cell.state === "not_run" || cell.state === "disabled") return null;
  if (group === "stage") {
    if (cell.job_id) return `/jobs/${encodeURIComponent(cell.job_id)}?stage=${encodeURIComponent(column)}`;
    const tab = stageTab(column);
    return tab ? `/r/${encodeURIComponent(rid)}/${tab}` : null;
  }
  if (group === "study") {
    if (cell.study_id) return `/studies/${encodeURIComponent(cell.study_id)}`;
    return `/r/${encodeURIComponent(rid)}/validation`;
  }
  const postTab: Record<string, string> = { planner: "planner", emulator: "lab", uncertainty: "uncertainty", writeup: "docs", baselines: "distance" };
  if (cell.state === "running" && cell.job_id) return `/jobs/${encodeURIComponent(cell.job_id)}`;
  return postTab[column] ? `/r/${encodeURIComponent(rid)}/${postTab[column]}` : null;
}
