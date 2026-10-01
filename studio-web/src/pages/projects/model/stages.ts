// Launch stage checklist (SPEC §5.4, §3.2 Launch): the user picks stages; the rules core's
// gating predicates apply are shown up front (sparc.core.pipeline `_plan`, `_wants_s2`,
// `_wants_s4`, `_climate_runs`):
// - S0 (data) and S1 (influence) always run;
// - S2 and S3 run together, and whenever any later stage (S4…S7) is requested;
// - S4 runs when requested and whenever S6 or S7 needs its responses;
// - the climate stage rides on S5; baselines and the CV curve hang off S2–S3.
// The plan request sends only the explicit choices, so the server marks implied stages
// `will_run` with `reason: required_by:<stage>` (the plan graph shows "required by S6").
import type { RequestStage } from "../../../api/projects";

export type ChecklistId = "S0" | "S1" | "S2_S3" | "S4" | "S5" | "S6" | "S7";

export type ChecklistRow = {
  id: ChecklistId;
  label: string;
  desc: string;
  checked: boolean;
  /** The checkbox cannot be changed (always runs, or forced by a later stage). */
  locked: boolean;
  /** Why it is locked on: "always runs", "required by S6". */
  reason: string | null;
};

export const CHECKLIST: readonly { id: ChecklistId; label: string; desc: string }[] = [
  { id: "S0", label: "S0 Data and QA", desc: "Load, clean and grid the table" },
  { id: "S1", label: "S1 Area of influence", desc: "Correlograms and influence ranges" },
  { id: "S2_S3", label: "S2–S3 Models and stacker", desc: "Cross-fitted base models, stacker, baselines, CV curve" },
  { id: "S4", label: "S4 Response surfaces", desc: "Dose sweeps and footprints of every lever" },
  { id: "S5", label: "S5 Scenarios", desc: "Configured scenarios (and climate × adaptation)" },
  { id: "S6", label: "S6 Causal validation", desc: "DML, spillover, CATE, dose-response, audit" },
  { id: "S7", label: "S7 Budget optimisation", desc: "Allocation of the budget and its closed-loop check" },
];

/** The default selection: a full run. */
export const ALL_CHOICES: readonly ChecklistId[] = ["S0", "S1", "S2_S3", "S4", "S5", "S6", "S7"];

const ORDER: readonly ChecklistId[] = ALL_CHOICES;

/** The first chosen stage after `id` among `candidates` (core's `_first_requested`). */
function firstChosen(chosen: ReadonlySet<ChecklistId>, candidates: readonly ChecklistId[]): ChecklistId | null {
  for (const c of candidates) if (chosen.has(c)) return c;
  return null;
}

function stageName(id: ChecklistId): string {
  return id === "S2_S3" ? "S2–S3" : id;
}

/** Checklist rows for a selection, with implied stages locked on and the reason shown. */
export function checklistRows(selection: Iterable<ChecklistId>): ChecklistRow[] {
  const chosen = new Set(selection);
  return CHECKLIST.map(({ id, label, desc }) => {
    let checked = chosen.has(id);
    let locked = false;
    let reason: string | null = null;
    if (id === "S0" || id === "S1") {
      checked = true;
      locked = true;
      reason = "always runs";
    } else if (id === "S2_S3") {
      const by = firstChosen(chosen, ["S4", "S5", "S6", "S7"]);
      if (by) {
        checked = true;
        locked = true;
        reason = `required by ${stageName(by)}`;
      }
    } else if (id === "S4") {
      const by = firstChosen(chosen, ["S6", "S7"]);
      if (by) {
        checked = true;
        locked = true;
        reason = `required by ${by}`;
      }
    }
    return { id, label, desc, checked, locked, reason };
  });
}

/** Toggle one stage; locked rows do not change. Returns the new explicit selection (ordered). */
export function toggleStage(selection: Iterable<ChecklistId>, id: ChecklistId, on: boolean): ChecklistId[] {
  const row = checklistRows(selection).find((r) => r.id === id);
  if (!row || row.locked) return ORDER.filter((s) => new Set(selection).has(s));
  const next = new Set(selection);
  if (on) next.add(id);
  else next.delete(id);
  return ORDER.filter((s) => next.has(s));
}

/** The `stages` body of `runs/plan` and `runs` for an explicit selection. */
export function requestStages(selection: Iterable<ChecklistId>): RequestStage[] {
  const out: RequestStage[] = [];
  const chosen = new Set(selection);
  for (const id of ORDER) {
    if (!chosen.has(id)) continue;
    if (id === "S2_S3") out.push("S2", "S3");
    else out.push(id);
  }
  return out;
}

/** Parse the `stages` query value back into a selection (unknown ids dropped). */
export function parseSelection(list: readonly string[]): ChecklistId[] {
  const set = new Set(list);
  return ORDER.filter((s) => set.has(s));
}

/** Whether a selection is the full run (every stage chosen). */
export function isFullSelection(selection: Iterable<ChecklistId>): boolean {
  const chosen = new Set(selection);
  return ORDER.every((s) => chosen.has(s));
}
