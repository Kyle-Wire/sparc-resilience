// The quick estimate (the emulator preview) and the full-model run, in plain words for a planner
// (phase 3, Design page right pane). Pure: summary and compile in, text out; PreviewPane and
// RunFullModel only lay it out, and DesignMap's hatch key reads HATCH_KEY.
// - The big number: "About −0.42 °F" where you made changes, "−0.05 °F" citywide.
// - Three short facts: area changed (km² or hectares when the cell size is known, else places;
//   share of the city; residents), estimated cost (only with a cost model), heat-risk shift (the
//   heat preview's Extreme-caution move, worded as the old HeatPreviewCard words it).
// - The trust line: reliable / rough / rough in the hatched areas, with exactAdvice() reasons in plain
//   words (words.plainAdvice).
// - The run button's label "Run full model (≈ 2 min)", why it cannot run yet, and whether the shown
//   full-model result is up to date (then there is nothing new to run).
// The pane's own words live in QUICK_WORDS; repeated concepts (the hatch key, the heat-risk level, the
// quick estimate and the full model's names) come from the shared glossary.
import type { CompileResult, CompileWarning, EngineState, Lever, PreviewSummary, Result, ScenarioDoc } from "../../../api/lab";
import { GLOSSARY, heatCategory, term } from "../../../content/glossary";
import type { HeatShift } from "../../../theme/heat";
import { fmtInt, fmtNum, fmtPct, fmtSignedValue, unitLabel } from "../../../theme/format";
import type { ExactAdvice } from "../components/CompilePanel";
import { LAB_WORDS, PREVIEW_TRUST_LINE, plainAdvice } from "./words";

/** The NWS level the heat-risk lines count from ("Extreme caution", glossary heat categories). */
const EC = heatCategory("extreme_caution").label;

/** Words of the quick-estimate pane and the run block. */
export const QUICK_WORDS = {
  area: "Area changed",
  cost: "Estimated cost",
  costTip: "In the cost units set for this run (per unit of change in each place). Set them in Advanced options.",
  heat: "Heat risk",
  /** Tooltip of the heat-risk line: what the level means. */
  heatTip: `Residents (or places, without resident data) at ${EC} or worse on a hot afternoon today: a feels-like temperature of 90 °F or more.`,
  heatUnproven: "unproven",
  noChangeYet: "No places changed yet",
  noChangeYetHint: "None of the places you chose can take this change. Try another place or amount.",
  noNoticeable: "No noticeable change",
  wholeCity: "the whole city",
  unavailable: `The ${term("emulator_preview").label.toLowerCase()} is not available right now.`,
  building: `Building the ${term("emulator_preview").label.toLowerCase()}…`,
  needsChange: "Make a change first — pick what to change on the left.",
  unfinished: (n: number) => (n === 1 ? "1 change is not finished yet, so it is left out." : `${fmtInt(n)} changes are not finished yet, so they are left out.`),
  blocked: (changes: number[]) =>
    changes.length
      ? `${changes.length === 1 ? `Change ${changes[0]}` : `Changes ${listAnd(changes.map(String))}`} cannot run as set — see ${LAB_WORDS.advanced} for why.`
      : `Some changes cannot run as set — see ${LAB_WORDS.advanced} for why.`,
} as const;

/** The map's hatch key: what the hatched places mean on the quick estimate and on a full-model layer (glossary). */
export const HATCH_KEY: Readonly<Record<"preview" | "result", { label: string; note: string }>> = GLOSSARY.hatch;

/** "a, b and c". */
function listAnd(xs: string[]): string {
  return xs.length <= 1 ? (xs[0] ?? "") : `${xs.slice(0, -1).join(", ")} and ${xs[xs.length - 1]}`;
}

const finite = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

/** Rounded to `sig` significant digits (1234 → 1230 at 3). */
export function roundSig(v: number, sig = 3): number {
  if (!finite(v) || v === 0) return v;
  const p = Math.pow(10, sig - 1 - Math.floor(Math.log10(Math.abs(v))));
  return Math.round(v * p) / p;
}

// ---------------------------------------------------------------- the big number

export type ChangeTone = "cool" | "warm" | "neutral";

/** Below this (target units) a change reads as "no noticeable change". */
export const NOTICEABLE = 0.005;

/** Cool for a cooling, warm for a warming, neutral when it rounds to nothing (never crimson for cooling). */
export function changeTone(v: number | null | undefined): ChangeTone {
  if (!finite(v) || Math.abs(v) < NOTICEABLE) return "neutral";
  return v < 0 ? "cool" : "warm";
}

/** Decimals a planner needs: 2 below 1 degree, 1 below 10, none above. */
function tempDecimals(v: number): number {
  const a = Math.abs(v);
  return a >= 10 ? 0 : a >= 1 ? 1 : 2;
}

/** A signed temperature change, rounded for reading: "−0.42 °F", "+1.3 °F", "less than 0.01 °F". */
export function approxTemp(v: number | null | undefined, unit: string): string {
  if (!finite(v)) return "—";
  if (Math.abs(v) < NOTICEABLE) return `less than ${fmtNum(0.01, 2)} ${unitLabel(unit)}`.trimEnd();
  return fmtSignedValue(v, unit, tempDecimals(v));
}

export type BigNumber = { text: string; tone: ChangeTone; /** false: words instead of a number. */ numeric: boolean };

/** The headline number: "About −0.42 °F" (where you made changes), or why there is none. */
export function bigNumber(editedMean: number | null | undefined, unit: string, nEdited: number): BigNumber {
  if (!(nEdited > 0) || !finite(editedMean)) return { text: QUICK_WORDS.noChangeYet, tone: "neutral", numeric: false };
  if (Math.abs(editedMean) < NOTICEABLE) return { text: QUICK_WORDS.noNoticeable, tone: "neutral", numeric: false };
  return { text: `About ${approxTemp(editedMean, unit)}`, tone: changeTone(editedMean), numeric: true };
}

/** The citywide figure: "−0.05 °F" (shown with "citywide"). */
export function cityText(mean: number | null | undefined, unit: string): string {
  if (!finite(mean)) return "—";
  return approxTemp(mean, unit).replace(/^less/, "Less");
}

// ---------------------------------------------------------------- the three facts

/** An area for a planner: km² from 1 km² up, hectares below (one 30 m cell is 0.09 ha). */
export function areaWords(km2: number): string {
  if (!finite(km2) || km2 < 0) return "—";
  if (km2 >= 1) return `${fmtNum(km2, km2 >= 10 ? 0 : 1)} km²`;
  const ha = km2 * 100;
  if (ha === 0) return "0 ha";
  return `${fmtNum(ha, ha >= 10 ? 0 : ha >= 1 ? 1 : 2)} ha`;
}

/** Share of the city: "4% of the city", "under 1% of the city", "the whole city". */
export function cityShare(nEdited: number, nTotal: number | null | undefined): string | null {
  if (!finite(nTotal) || nTotal <= 0 || !(nEdited > 0)) return null;
  const s = nEdited / nTotal;
  if (s >= 0.995) return QUICK_WORDS.wholeCity;
  if (s < 0.01) return "under 1% of the city";
  return `${fmtPct(s)} of the city`;
}

/** "about 12,300 residents" (three significant digits from 1,000 up), null without a residents count. */
export function residentsText(people: number | null | undefined): string | null {
  if (!finite(people) || people < 0) return null;
  const n = Math.round(people);
  if (n < 1000) return `${fmtInt(n)} ${n === 1 ? "resident" : "residents"}`;
  return `about ${fmtInt(roundSig(n, 3))} residents`;
}

/**
 * Area changed, in one line: "0.27 ha (43% of the city) · 35 residents". With the cell size unknown
 * the area is a count of places ("3 places").
 */
export function areaLine(nEdited: number, cellAreaKm2: number | null, nTotal: number | null, people: number | null): string {
  const n = Math.max(0, Math.round(nEdited));
  const area = cellAreaKm2 !== null && finite(cellAreaKm2) && cellAreaKm2 > 0 ? areaWords(n * cellAreaKm2) : `${fmtInt(n)} ${n === 1 ? "place" : "places"}`;
  const share = cityShare(n, nTotal);
  const who = residentsText(people);
  return [share ? `${area} (${share})` : area, who].filter(Boolean).join(" · ");
}

/** Total estimated cost of the compiled changes. */
export function totalCost(compile: Pick<CompileResult, "levers"> | null): number {
  if (!compile) return 0;
  return Object.values(compile.levers).reduce((a, x) => a + (finite(x.est_cost) ? x.est_cost : 0), 0);
}

/**
 * A cost model is set when a changed lever has a price of its own: a scenario cost (Advanced options)
 * or a catalogue cost other than the default 1 per unit (which only counts units of change).
 */
export function hasCostModel(compile: Pick<CompileResult, "levers"> | null, levers: readonly Pick<Lever, "var" | "cost_per_unit">[], costs: ScenarioDoc["costs"] | undefined): boolean {
  if (!compile) return false;
  return Object.keys(compile.levers).some((v) => {
    if (costs?.[v] !== undefined) return true;
    const c = levers.find((l) => l.var === v)?.cost_per_unit;
    return finite(c) && c !== 1;
  });
}

/** "about 27,600", "about 1.2 million". */
export function costWords(total: number): string {
  if (!finite(total)) return "—";
  const a = Math.abs(total);
  if (a >= 1e9) return `about ${fmtNum(total / 1e9, 1)} billion`;
  if (a >= 1e6) return `about ${fmtNum(total / 1e6, 1)} million`;
  if (a >= 100) return `about ${fmtInt(roundSig(total, 3))}`;
  return `about ${fmtNum(total, a >= 10 ? 0 : 1)}`;
}

/** The cost fact, or null without a cost model or a cost. */
export function costLine(compile: Pick<CompileResult, "levers"> | null, levers: readonly Pick<Lever, "var" | "cost_per_unit">[], costs: ScenarioDoc["costs"] | undefined): string | null {
  const total = totalCost(compile);
  if (!(total > 0) || !hasCostModel(compile, levers, costs)) return null;
  return costWords(total);
}

/**
 * The heat-risk shift in one line, worded as the old HeatPreviewCard words it: residents (or places,
 * without a residents layer) moved out of, or into, NWS Extreme caution or worse.
 */
export function heatLine(shift: Pick<HeatShift, "ecBefore" | "ecAfter" | "measure"> | null | undefined): string | null {
  if (!shift || !finite(shift.ecBefore) || !finite(shift.ecAfter)) return null;
  const who = shift.measure === "people" ? "residents" : "places";
  const moved = Math.round(shift.ecBefore - shift.ecAfter);
  const n = (v: number) => fmtInt(v >= 1000 ? roundSig(v, 3) : v);
  if (moved > 0) return `Moves about ${n(moved)} ${who} out of ${EC} or worse`;
  if (moved < 0) return `Puts about ${n(-moved)} more ${who} at ${EC} or worse`;
  return `No change in ${who} at ${EC} or worse`;
}

// ---------------------------------------------------------------- trust

export type TrustTone = "good" | "rough" | "hatched";
export type TrustView = { tone: TrustTone; line: string; reasons: string[] };

/**
 * The plain trust line. "hatched" only when the map hatches the changed places (summary.hatched);
 * "rough" when exactAdvice() recommends the full model or the preview itself is not good; else good.
 */
export function trustView(
  summary: Pick<PreviewSummary, "hatched" | "reasons" | "trust">,
  advice: ExactAdvice,
  levers: readonly Pick<Lever, "var" | "label" | "role" | "direction">[] = [],
): TrustView {
  const tone: TrustTone = summary.hatched ? "hatched" : advice.recommend || summary.trust !== "good" ? "rough" : "good";
  if (tone === "good") return { tone, line: PREVIEW_TRUST_LINE.good, reasons: [] };
  // exactAdvice's generic "<lever>: preview is rough" only restates the rough / hatched line above the
  // reasons (the lever cards' trust dots say which lever): the reasons keep what the line does not say.
  const raw = (advice.reasons.length ? advice.reasons : summary.reasons).filter((r) => !ROUGH_LEVER.test(r));
  const reasons = plainAdvice(raw, levers);
  return { tone, line: tone === "hatched" ? PREVIEW_TRUST_LINE.hatched : PREVIEW_TRUST_LINE.rough, reasons };
}

/** exactAdvice's per-lever "Albedo: preview is rough" reason. */
const ROUGH_LEVER = /^.+: preview is rough$/i;

// ---------------------------------------------------------------- the whole pane

export type QuickInput = {
  summary: PreviewSummary;
  unit: string;
  compile: CompileResult | null;
  levers: readonly Lever[];
  costs?: ScenarioDoc["costs"];
  cellAreaKm2: number | null;
  /** Cells in the run's grid (the share of the city). */
  nTotal: number | null;
  heat: Pick<HeatShift, "ecBefore" | "ecAfter" | "measure"> | null;
  advice: ExactAdvice;
};

export type QuickView = {
  big: BigNumber;
  city: string;
  area: string;
  cost: string | null;
  heat: string | null;
  trust: TrustView;
};

/** Everything the quick-estimate pane says, from the preview summary and the compile. */
export function quickView(q: QuickInput): QuickView {
  const s = q.summary;
  return {
    big: bigNumber(s.edited_mean, q.unit, s.n_edited),
    city: cityText(s.mean, q.unit),
    area: areaLine(s.n_edited, q.cellAreaKm2, q.nTotal, q.compile?.people ?? null),
    cost: costLine(q.compile, q.levers, q.costs),
    heat: heatLine(q.heat),
    trust: trustView(s, q.advice, q.levers),
  };
}

// ---------------------------------------------------------------- run the full model

/** "≈ 2 min", "under 1 min", "≈ 1.5 h" (null without an estimate). */
export function approxDuration(seconds: number | null | undefined): string | null {
  if (!finite(seconds) || seconds < 0) return null;
  if (seconds < 45) return "under 1 min";
  if (seconds < 3570) return `≈ ${Math.max(1, Math.round(seconds / 60))} min`;
  return `≈ ${fmtNum(seconds / 3600, seconds < 36000 ? 1 : 0)} h`;
}

/** "Run full model (≈ 2 min)". */
export function runLabel(estSeconds: number | null | undefined): string {
  const d = approxDuration(estSeconds);
  return d ? `${LAB_WORDS.runFullModel} (${d})` : LAB_WORDS.runFullModel;
}

/**
 * "Build quick estimate (≈1 min)". The server builds a fast run's emulator in about a minute and a
 * full run's in about 4–5 minutes per lever.
 */
export function buildLabel(mode: string | null | undefined, nLevers: number): string {
  const per = mode && mode !== "fast" ? Math.max(1, nLevers) : 0;
  return `${LAB_WORDS.buildQuickEstimate} (${per ? `≈${4 * per}–${5 * per} min` : "≈1 min"})`;
}

/**
 * The shown full-model result answers the changes as they stand: it is not stale (the model has not
 * changed since), it belongs to the open scenario revision (`sid` / `revision`), nothing is waiting
 * to be saved (`dirty`: an edit or a brush stroke since the last save; a saved change to a scenario
 * with a result forks a new revision) and no full-model run is under way. Then "Run full model" has
 * nothing new to run, and the run block points to the answer instead.
 */
export function fullModelUpToDate(
  result: Pick<Result, "stale" | "summary" | "scenario"> | null | undefined,
  s: { sid: string | null; revision?: number | null; dirty: boolean; running?: boolean },
): boolean {
  if (!result || result.stale || result.summary.stale || s.dirty || s.running || !s.sid) return false;
  if (result.summary.kind !== "exact") return false;
  if (result.scenario) {
    if (result.scenario.id !== s.sid) return false;
    if (finite(s.revision) && finite(result.scenario.revision) && result.scenario.revision !== s.revision) return false;
  } else if (result.summary.scenario_id !== s.sid) return false;
  return true;
}

export type RunState = {
  /** Usable edits or brush strokes exist. */
  hasChanges: boolean;
  /** Edits in the draft that are not complete enough to evaluate. */
  unfinished: number;
  /** Why the run cannot run exactly at all (no checkpoint, incompatible): the engine line says it. */
  runReason: string | null;
  blocking: readonly Pick<CompileWarning, "edit_index">[];
};

/**
 * Why "Run full model" is disabled, in plain words (null when it is enabled, or when the engine line
 * under the button already says why).
 */
export function runDisabledReason(r: RunState): string | null {
  if (r.runReason) return null;
  if (!r.hasChanges) return QUICK_WORDS.needsChange;
  if (r.blocking.length) {
    const changes = [...new Set(r.blocking.map((w) => w.edit_index).filter((i): i is number => i !== null && i !== undefined))].sort((a, b) => a - b).map((i) => i + 1);
    return QUICK_WORDS.blocked(changes);
  }
  return null;
}

/** A note under the run button: changes left out because they are unfinished (null when none). */
export function unfinishedNote(r: Pick<RunState, "unfinished" | "hasChanges">): string | null {
  return r.hasChanges && r.unfinished > 0 ? QUICK_WORDS.unfinished(r.unfinished) : null;
}

/** Tone of the engine line's dot (text always carries the state). */
export function engineTone(state: EngineState | null): "good" | "busy" | "idle" | "off" | "crit" {
  switch (state) {
    case "ready":
      return "good";
    case "loading":
    case "queued":
    case "busy":
      return "busy";
    case "cold":
    case null:
      return "idle";
    case "error":
      return "crit";
    default:
      return "off";
  }
}
