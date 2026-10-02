// Setup wizard steps (SPEC §9.3): ids, labels, the config sections each one edits (for its
// issue summary) and the completion dot. Dots come from the server readiness spine
// (`GET /api/projects/{pid}` readiness[]); scenarios, analysis and about have no spine row,
// so their dot is derived from the draft and its validation issues.
import type { Issue, ReadinessRow } from "../../../api/types";
import { getList, getString } from "./dotted";

export type StepId = "data" | "levers" | "physics" | "inputs" | "scenarios" | "analysis" | "about";

export const STEPS: readonly { id: StepId; label: string }[] = [
  { id: "data", label: "Data" },
  { id: "levers", label: "Levers" },
  { id: "physics", label: "Physics" },
  { id: "inputs", label: "Inputs" },
  { id: "scenarios", label: "Scenarios" },
  { id: "analysis", label: "Analysis" },
  { id: "about", label: "About" },
];

export function isStepId(v: string | null | undefined): v is StepId {
  return STEPS.some((s) => s.id === v);
}

/** Config sections (dotted prefixes) whose issues belong to each step. */
export const STEP_SECTIONS: Record<StepId, readonly string[]> = {
  data: ["data"],
  levers: ["predictors", "encodings", "qa", "actionable", "coupling", "mediators"],
  physics: ["physics"],
  inputs: ["physics.forcing", "climate.table", "planner.layers"],
  scenarios: ["scenarios", "joint_scenarios"],
  analysis: ["causal", "climate", "optimize", "planner", "influence", "cv", "models", "stacker", "response"],
  about: ["report", "name"],
};

/** Readiness spine rows behind each step's dot. */
export const STEP_READINESS: Record<StepId, readonly ReadinessRow["key"][]> = {
  data: ["data", "columns"],
  levers: ["levers"],
  physics: ["roles"],
  inputs: ["forcing", "climate_table", "people_layers"],
  scenarios: [],
  analysis: [],
  about: [],
};

export type DotState = "ok" | "warn" | "missing" | "n/a";

const RANK: Record<DotState, number> = { "n/a": 0, ok: 1, warn: 2, missing: 3 };

function worst(states: DotState[]): DotState {
  return states.reduce<DotState>((a, b) => (RANK[b] > RANK[a] ? b : a), "n/a");
}

function under(path: string, prefix: string): boolean {
  return path === prefix || path.startsWith(prefix + ".") || path.startsWith(prefix + "[");
}

/** Issues of the step's sections (inputs paths are excluded from physics/analysis duplicates). */
export function stepIssues(step: StepId, issues: readonly Issue[]): Issue[] {
  return issues.filter((i) => STEP_SECTIONS[step].some((p) => under(i.path, p)));
}

/** Completion dot of a step. */
export function stepDot(step: StepId, readiness: readonly ReadinessRow[] | undefined, raw: unknown, issues: readonly Issue[]): DotState {
  const keys = STEP_READINESS[step];
  if (keys.length) {
    const rows = (readiness ?? []).filter((r) => keys.includes(r.key));
    if (rows.length) return worst(rows.map((r) => r.state));
  }
  const own = stepIssues(step, issues);
  if (own.some((i) => i.level === "error")) return "missing";
  if (step === "scenarios") {
    const n = getList(raw, "scenarios").length + getList(raw, "joint_scenarios").length;
    return n ? (own.some((i) => i.level === "warn") ? "warn" : "ok") : "warn";
  }
  if (step === "about") return getString(raw, "report.title") ? "ok" : "warn";
  if (own.some((i) => i.level === "warn")) return "warn";
  return keys.length ? "n/a" : "ok";
}

export const DOT_TEXT: Record<DotState, string> = { ok: "complete", warn: "needs attention", missing: "missing", "n/a": "not applicable" };
