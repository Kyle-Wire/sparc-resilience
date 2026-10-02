// Plan graph view model (SPEC §5.4): node states, plain-language reasons, unit summaries and
// estimate ranges of a `RunPlan`. PlanNode reasons (api.md §1): not_requested |
// disabled_by_config:<key> | checkpoint | no_budget | no_treatments | no_responses |
// requires_S5 | no_partitions | required_by:<stage>.
import type { PlanNode, StageId } from "../../../api/types";
import { fmtDuration, fmtDurationRange, fmtInt } from "../../../theme/format";

export const STAGE_ORDER: readonly StageId[] = ["S0", "S1", "S2_S3", "baselines", "cv_curve", "S4", "S5", "climate", "S6", "S7", "finish"];

const STAGE_SHORT: Record<string, string> = { S2_S3: "S2–S3", cv_curve: "CV curve" };

export function stageShort(id: string): string {
  return STAGE_SHORT[id] ?? id;
}

/** Plain text for a plan node reason. */
export function planReasonText(reason: string | null | undefined): string | null {
  if (!reason) return null;
  if (reason.startsWith("required_by:")) return `required by ${stageShort(reason.slice("required_by:".length))}`;
  if (reason.startsWith("disabled_by_config:")) return `disabled in the config (${reason.slice("disabled_by_config:".length)})`;
  switch (reason) {
    case "not_requested":
      return "not requested";
    case "checkpoint":
      return "from the checkpoint";
    case "no_budget":
      return "no budget set (optimize.budget)";
    case "no_treatments":
      return "no causal treatments configured";
    case "no_responses":
      return "the optimised variable has no response (not a lever)";
    case "requires_S5":
      return "needs S5 (scenarios)";
    case "no_partitions":
      return "no distance-curve partitions left";
    default:
      return reason.replace(/_/g, " ");
  }
}

const UNIT_WORDS: Record<string, [string, string]> = {
  s0_load: ["data load", "data loads"],
  s1_influence: ["influence pass", "influence passes"],
  engine_pass: ["engine pass", "engine passes"],
  checkpoint_save: ["checkpoint save", "checkpoint saves"],
  adv_refit: ["advection refit", "advection refits"],
  climate_model: ["climate model", "climate models"],
  pareto: ["Pareto pass", "Pareto passes"],
};

/**
 * Units grouped by family: `base_fit:mgwr` + `base_fit:gam` → "10 base fits",
 * `stacker_fit:*` → "stacker fits", `causal_step:*` → "causal steps".
 */
export function unitSummary(units: Record<string, number>): string {
  const groups = new Map<string, number>();
  for (const [k, n] of Object.entries(units)) {
    if (!n) continue;
    const fam = k.includes(":") ? k.split(":")[0] : k;
    groups.set(fam, (groups.get(fam) ?? 0) + n);
  }
  const words = (fam: string, n: number) => {
    const w = UNIT_WORDS[fam];
    if (w) return `${fmtInt(n)} ${n === 1 ? w[0] : w[1]}`;
    const base = fam.replace(/_/g, " ");
    return `${fmtInt(n)} ${base}${n === 1 ? "" : "s"}`;
  };
  return [...groups.entries()].map(([fam, n]) => words(fam, n)).join(" · ");
}

export type PlanNodeView = {
  id: StageId;
  label: string;
  state: PlanNode["state"];
  reason: string | null;
  estimate: string | null;
  units: string;
};

/** Nodes in stage order with text ready for display. */
export function planNodeViews(nodes: readonly PlanNode[]): PlanNodeView[] {
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const ordered = [...STAGE_ORDER.filter((id) => byId.has(id)).map((id) => byId.get(id)!), ...nodes.filter((n) => !STAGE_ORDER.includes(n.id))];
  return ordered.map((n) => {
    let estimate: string | null = null;
    if (n.state === "will_run") {
      // Below 10 s a range reads as noise ("0–0 s"): show the point estimate; below 1 s
      // (data load, finish) "≈0.0 s" says less than "< 1 s".
      if (n.est_lo !== null && n.est_hi !== null && n.est_hi > n.est_lo && n.est_hi >= 10) estimate = `≈${fmtDurationRange(n.est_lo, n.est_hi)}`;
      else if (n.est_s !== null) estimate = n.est_s < 1 ? "< 1 s" : `≈${fmtDuration(n.est_s)}`;
    }
    return { id: n.id, label: n.label, state: n.state, reason: planReasonText(n.reason), estimate, units: n.state === "will_run" ? unitSummary(n.units) : "" };
  });
}

/** Counts by state ("9 will run · 2 skipped · 0 cached"). */
export function planCounts(nodes: readonly PlanNode[]): Record<PlanNode["state"], number> {
  const c = { will_run: 0, skipped: 0, cached: 0 };
  for (const n of nodes) c[n.state] += 1;
  return c;
}
