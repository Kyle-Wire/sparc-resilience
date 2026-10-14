// Pure helpers of the Design step flow (phase 3): the planner modes and amount scales of step 03,
// the where-presets of step 02 (preset → selection tree and back), and the plain sentences of the
// flow: a change in one line ("Add 10 pts tree canopy in Downtown") and the live "how much fits"
// sentence from the compile ("You asked for +20 pts tree canopy; about 14 pts fits — the rest is
// beyond what the space allows."). No React, no store: the step components (components/flow/*)
// read these and write edits through WorkbenchModel.setEdit / addEdit only.
//
// Every selection a preset builds is a shape the server already accepts (the same nodes the
// SelectionBuilder and the scenario templates write: all, zones, region, top by "pred:target" or
// "layer:people", or of zones/regions, and the map tools' shapes).
import type { CompileResult, Edit, EditMode, Lever, Region } from "../../../api/lab";
import type { SelectionSpec } from "../../../api/types";
import { fmtInt, fmtPct, fmtSig, unitLabel } from "../../../theme/format";
import type { ColumnOption, MapPick } from "../components/SelectionBuilder";
import { describeSelection, editProblems, withMode } from "./doc";
import { MODE_WORDS, WHERE_PRESET_ORDER, leverUnitWord, modeFits, titleCase, type WherePreset } from "./words";

// ---------------------------------------------------------------- words of the flow's sentences

/**
 * The step flow's own sentence pieces and control labels (planner words). They belong with
 * words.ts; phase 4 folds both into the shared glossary.
 */
export const FLOW_WORDS = {
  pickFirst: "Pick what to change first.",
  noLevers: "This run has no levers to change.",
  notALever: "This change edits a model input directly (expert mode).",
  // step 03 modes for levers that cool by going down (less paving)
  takeAway: "Take away",
  takeAwayHint: "Takes this much away everywhere you chose.",
  bringDown: "Bring down to at most",
  bringDownHint: "Lowers places above this level down to it; places already below stay as they are.",
  howToChange: "How to change it",
  howMuchAdd: "How much to add",
  howMuchTakeAway: "How much to take away",
  atLeast: "Lowest level allowed",
  atMost: "Highest level allowed",
  plantingShare: "Share of the planting space",
  typical: "Typical amounts",
  otherMode: "This change uses “{mode}”, set in Advanced options.",
  backToSimple: "Switch to a simple amount",
  checking: "Checking how much fits…",
  blocked: "This change cannot run as it is:",
  // the "how much fits" sentence
  askedFor: "You asked for",
  askedTogether: "Together with your other {what} changes, you asked for",
  onAverage: "on average",
  allFits: "all of it fits.",
  aboutFits: "about {amount} fits",
  restBeyond: "the rest is beyond what the space allows.",
  restBeyondSome: "in {share} of the places the rest is beyond what the space allows.",
  nothingFloor: "Every place you chose is already at or above this level, so nothing changes.",
  nothingCeiling: "Every place you chose is already at or below this level, so nothing changes.",
  nothingPlanting: "There is no free planting space in the places you chose, so nothing changes.",
  nothingAsked: "This asks for no change yet.",
  // step 02
  share: "Share of the city",
  hottestShare: "The hottest",
  peopleShare: "The most-lived-in",
  ofTheCity: "of the city",
  neighbourhoods: "Neighbourhoods",
  usePick: "Use {label}",
  brushHint: "Or paint the change place by place with the map’s brush.",
  drawPending: "Until you draw, this change still applies {where}.",
  applies: "Applies {where}.",
  customClose: "Hide custom filter",
  // the change sentence
  notFinished: "not finished — left out of the estimate",
} as const;

/** "{x} and {y}" placeholders → values. */
export function fill(template: string, values: Record<string, string>): string {
  return template.replace(/\{(\w+)\}/g, (m, k: string) => (k in values ? values[k] : m));
}

/** What a lever changes, as a noun for sentences ("tree canopy"), by lever role. */
export const LEVER_MEASURE: Record<"canopy" | "impervious" | "albedo", string> = {
  canopy: "tree canopy",
  impervious: "paving",
  albedo: "surface reflectivity",
};

type LeverLike = Pick<Lever, "var" | "label" | "role" | "direction" | "unit"> & Partial<Pick<Lever, "min" | "max" | "doses" | "sd" | "design_dose" | "headroom_available">>;

export function leverMeasure(lever: Pick<Lever, "var" | "label" | "role"> | undefined, varName?: string): string {
  const role = lever?.role?.toLowerCase() ?? null;
  if (role && role in LEVER_MEASURE) return LEVER_MEASURE[role as keyof typeof LEVER_MEASURE];
  return titleCase(lever?.label || lever?.var || varName || "the change").toLowerCase();
}

/** The lever's unit as a sentence word: "pts" for percentage points, "" for unitless ("units"). */
export function amountUnit(lever: Pick<Lever, "var" | "label" | "role" | "direction" | "unit"> | undefined): string {
  const u = leverUnitWord(lever);
  if (!u || u === "units" || u === "unit") return "";
  return unitLabel(u);
}

/** "10 pts", "0.1", "15%" (magnitude, up to `sig` significant digits). */
export function fmtAmount(v: number, unit: string, sig = 3): string {
  const n = fmtSig(Math.abs(v), sig);
  if (!unit) return n;
  return unit === "%" ? `${n}%` : `${n} ${unit}`;
}

/** "+10 pts", "−0.1". */
export function fmtSignedAmount(v: number, unit: string, sig = 3): string {
  const body = fmtAmount(v, unit, sig);
  if (Number(fmtSig(Math.abs(v), sig).replace(/,/g, "")) === 0) return body;
  return `${v < 0 ? "−" : "+"}${body}`;
}

/** "a", "a and b", "a, b and c". */
export function listJoin(parts: readonly string[]): string {
  if (parts.length <= 1) return parts[0] ?? "";
  return `${parts.slice(0, -1).join(", ")} and ${parts[parts.length - 1]}`;
}

/** 1 → "1st", 22 → "22nd", 75 → "75th", 13 → "13th". */
export function ordinal(n: number): string {
  const r = Math.round(n);
  const t = r % 100;
  const s = t >= 11 && t <= 13 ? "th" : r % 10 === 1 ? "st" : r % 10 === 2 ? "nd" : r % 10 === 3 ? "rd" : "th";
  return `${r}${s}`;
}

// ---------------------------------------------------------------- step 03: modes and amounts

/** +1 for levers that cool by going up (canopy, albedo), −1 for those that cool by going down (paving). */
export function directionSign(lever: Pick<Lever, "direction"> | undefined): 1 | -1 {
  return lever?.direction === "decrease" ? -1 : 1;
}

/**
 * The planner modes of a lever, in order. Levers that cool by going up: Add · Bring up to at least
 * · Use available planting space (canopy with the planner layers only). Levers that cool by going
 * down: Take away (an "add" of a negative amount) · Bring down to at most (ceiling). Every other
 * mode lives in Advanced options.
 */
export function plannerModes(lever: (Pick<Lever, "direction" | "role" | "headroom_available">) | undefined): EditMode[] {
  if (!lever) return ["add"];
  if (lever.direction === "decrease") return ["add", "ceiling"];
  return (["add", "floor", "fill_headroom"] as EditMode[]).filter((m) => modeFits(m, lever));
}

export function isPlannerMode(mode: EditMode, lever: Parameters<typeof plannerModes>[0]): boolean {
  return plannerModes(lever).includes(mode);
}

/** The label of a planner mode for a lever ("Take away" for an add on a lever that cools by going down). */
export function modeLabel(mode: EditMode, lever: Pick<Lever, "direction"> | undefined): string {
  if (lever?.direction === "decrease" && mode === "add") return FLOW_WORDS.takeAway;
  if (lever?.direction === "decrease" && mode === "ceiling") return FLOW_WORDS.bringDown;
  return MODE_WORDS[mode]?.label ?? mode;
}

export function modeHint(mode: EditMode, lever: Pick<Lever, "direction"> | undefined): string {
  if (lever?.direction === "decrease" && mode === "add") return FLOW_WORDS.takeAwayHint;
  if (lever?.direction === "decrease" && mode === "ceiling") return FLOW_WORDS.bringDownHint;
  return MODE_WORDS[mode]?.hint ?? "";
}

/** The label of the amount control of a planner mode. */
export function amountLabel(mode: EditMode, lever: Pick<Lever, "direction"> | undefined): string {
  if (mode === "fill_headroom") return FLOW_WORDS.plantingShare;
  if (mode === "floor") return FLOW_WORDS.atLeast;
  if (mode === "ceiling") return FLOW_WORDS.atMost;
  return directionSign(lever) < 0 ? FLOW_WORDS.howMuchTakeAway : FLOW_WORDS.howMuchAdd;
}

/** Width of a lever's range: max − min, else from its doses, sd or design dose. */
export function leverSpan(lever: Partial<Pick<Lever, "min" | "max" | "doses" | "sd" | "design_dose">> | undefined): number {
  const lo = lever?.min ?? null;
  const hi = lever?.max ?? null;
  if (lo !== null && hi !== null && hi > lo) return hi - lo;
  const dose = Math.max(0, ...(lever?.doses ?? []).map((d) => Math.abs(d)).filter(Number.isFinite));
  if (dose > 0) return dose * 2;
  if (lever?.sd && lever.sd > 0) return lever.sd * 3;
  return Math.max(1, Math.abs(lever?.design_dose ?? 1) * 5);
}

const decimalsOf = (step: number) => Math.max(0, Math.min(10, Math.ceil(-Math.log10(step) - 1e-9)));

/** Round to a step without float noise (0.1 + 0.2 → 0.3). */
export function roundToStep(v: number, step: number): number {
  if (!(step > 0) || !Number.isFinite(v)) return v;
  return Number((Math.round(v / step) * step).toFixed(decimalsOf(step)));
}

/** A slider step for a span (1 / 0.1 / 0.01 / 0.001), refined until every dose sits on it. */
export function niceStep(span: number, doses: readonly number[] = []): number {
  let step = span > 20 ? 1 : span > 2 ? 0.1 : span > 0.2 ? 0.01 : 0.001;
  const off = (d: number) => Math.abs(d / step - Math.round(d / step)) > 1e-6;
  while (step > 1e-6 && doses.some((d) => Number.isFinite(d) && off(d))) step /= 10;
  return Number(step.toPrecision(1));
}

/**
 * How step 03 shows a mode's amount: display = amount × factor, inside [min, max] by `step`, in
 * `unit`. Add shows the size of the change (for levers that cool by going down, the amount taken
 * away, stored negative); floor / ceiling show the level in the lever's units; planting space shows
 * a percentage of the free planting space (stored as a 0–1 share).
 */
export type AmountScale = { factor: number; min: number; max: number; step: number; unit: string };

export function amountScale(mode: EditMode, lever: LeverLike | undefined): AmountScale {
  if (mode === "fill_headroom") return { factor: 100, min: 0, max: 100, step: 5, unit: "%" };
  const span = leverSpan(lever);
  const doses = (lever?.doses ?? []).map((d) => Math.abs(d));
  const step = niceStep(span, doses);
  const unit = amountUnit(lever);
  if (mode === "floor" || mode === "ceiling" || mode === "set") {
    const lo = lever?.min ?? 0;
    return { factor: 1, min: lo, max: lever?.max ?? lo + span, step, unit };
  }
  if (mode === "scale") return { factor: 1, min: 0, max: 3, step: 0.05, unit: "×" };
  if (mode === "to_percentile") return { factor: 1, min: 0, max: 100, step: 5, unit: "" };
  return { factor: directionSign(lever), min: 0, max: span, step, unit };
}

/** The amount as step 03 shows it (null when not set). */
export function toDisplay(edit: Pick<Edit, "amount" | "mode">, lever: LeverLike | undefined): number | null {
  if (!Number.isFinite(edit.amount)) return null;
  const s = amountScale(edit.mode, lever);
  return Number(((edit.amount as number) * s.factor).toPrecision(12));
}

/** A shown amount back to the stored one. */
export function fromDisplay(v: number, mode: EditMode, lever: LeverLike | undefined): number {
  const s = amountScale(mode, lever);
  return Number((v / s.factor).toPrecision(12));
}

/** Typical amounts of a lever for "add" (its configured doses, as shown: sizes of the change). */
export function typicalAmounts(lever: LeverLike | undefined): number[] {
  const span = leverSpan(lever);
  const out = [...new Set((lever?.doses ?? []).map((d) => Math.abs(d)).filter((d) => d > 0 && d <= span))].sort((a, b) => a - b);
  return out.slice(0, 5);
}

/** A starting amount for a mode (stored units), or undefined for withMode's own default. */
export function defaultAmount(mode: EditMode, lever: LeverLike | undefined): number | undefined {
  if (!lever) return undefined;
  const dose = Math.abs(lever.design_dose ?? typicalAmounts(lever)[0] ?? lever.sd ?? 0);
  const { min, max, step } = amountScale(mode, lever);
  const clamp = (v: number) => roundToStep(Math.min(max, Math.max(min, v)), step);
  switch (mode) {
    case "add":
      return clamp(dose) * directionSign(lever);
    case "floor":
      return clamp((lever.min ?? 0) + 2 * dose);
    case "ceiling":
      return clamp((lever.max ?? (lever.min ?? 0) + leverSpan(lever)) - 2 * dose);
    case "fill_headroom":
      return 0.5;
    default:
      return undefined;
  }
}

/** The init of a change created from a lever card (a planner "add" of the design dose, in the lever's direction). */
export function newChangeInit(lever: LeverLike): Partial<Edit> {
  const amount = defaultAmount("add", lever);
  return amount === undefined ? {} : { amount };
}

/** Switch a change's mode, starting from a sensible amount for it. */
export function switchMode(edit: Edit, mode: EditMode, lever: LeverLike | undefined): Edit {
  return withMode(edit, mode, defaultAmount(mode, lever));
}

/**
 * Swap a change's lever (a lever card pick): keeps where; keeps the mode when it is a planner mode
 * of the new lever, else "add"; the amount restarts from the new lever's default (an amount in one
 * lever's units means nothing in another's) and the old lever's label is dropped.
 */
export function switchLever(edit: Edit, lever: LeverLike): Edit {
  if (edit.lever === lever.var) return edit;
  const mode = isPlannerMode(edit.mode, lever as Parameters<typeof plannerModes>[0]) ? edit.mode : "add";
  const next = withMode({ ...edit, lever: lever.var }, mode, defaultAmount(mode, lever));
  delete next.label;
  return next;
}

// ---------------------------------------------------------------- step 02: where presets

/** What the where presets and sentences need to know about the run. */
export type WhereContext = {
  zones: readonly (number | string)[];
  regions: readonly Pick<Region, "id" | "name">[];
  /** Filter column of the observed temperature ("pred:target"), null when the run has none. */
  temperatureColumn: string | null;
  /** Filter column of the residents layer ("layer:people"), null when the run has none. */
  populationColumn: string | null;
  /** The latest shape drawn or areas picked on the map. */
  mapPick?: MapPick | null;
  columns?: readonly ColumnOption[];
};

/** Default share of the "top X%" presets. */
export const TOP_SHARE = 0.1;

const DRAWN_KINDS = new Set(["circle", "rect", "polygon", "hex", "blob", "cells"]);
const isAll = (s: SelectionSpec | undefined) => !s || ("kind" in s && s.kind === "all");

/** The presets this run can offer, in order (a preset whose data is missing is left out). */
export function wherePresets(ctx: WhereContext): WherePreset[] {
  return WHERE_PRESET_ORDER.filter((p) => {
    if (p === "neighbourhood") return ctx.zones.length > 0 || ctx.regions.length > 0;
    if (p === "hottest") return !!ctx.temperatureColumn;
    if (p === "people") return !!ctx.populationColumn;
    return true;
  });
}

/** The zones and saved regions of a neighbourhood selection (zones, a region, or an "or" of them), else null. */
export type Neighbourhood = { zones: (number | string)[]; regions: string[] };

export function neighbourhoodOf(where: SelectionSpec | undefined): Neighbourhood | null {
  if (!where) return null;
  const parts = "kind" in where ? [where] : where.op === "or" ? where.args : null;
  if (!parts || !parts.length) return null;
  const out: Neighbourhood = { zones: [], regions: [] };
  for (const p of parts) {
    if (!("kind" in p)) return null;
    if (p.kind === "zones") out.zones.push(...p.values);
    else if (p.kind === "region") out.regions.push(p.id);
    else return null;
  }
  return out.zones.length || out.regions.length ? out : null;
}

/** A neighbourhood as a selection: zones → zones; one region → region; several parts → or. Null when empty. */
export function neighbourhoodSelection(n: Neighbourhood): SelectionSpec | null {
  const nodes: SelectionSpec[] = [];
  if (n.zones.length) nodes.push({ kind: "zones", values: [...n.zones] });
  for (const id of n.regions) nodes.push({ kind: "region", id });
  if (!nodes.length) return null;
  return nodes.length === 1 ? nodes[0] : { op: "or", args: nodes };
}

/** Add or remove one zone or saved region from a neighbourhood (null when that would leave it empty). */
export function toggleNeighbourhood(where: SelectionSpec | undefined, item: { zone: number | string } | { region: string }): SelectionSpec | null {
  const n = neighbourhoodOf(where) ?? { zones: [], regions: [] };
  if ("zone" in item) {
    const on = n.zones.some((z) => String(z) === String(item.zone));
    n.zones = on ? n.zones.filter((z) => String(z) !== String(item.zone)) : [...n.zones, item.zone];
  } else {
    const on = n.regions.includes(item.region);
    n.regions = on ? n.regions.filter((r) => r !== item.region) : [...n.regions, item.region];
  }
  return neighbourhoodSelection(n);
}

/** The top-share selection of a column (the "Hottest places" / "most people" presets). */
export function topSelection(column: string, share: number = TOP_SHARE): SelectionSpec {
  return { kind: "top", column, frac: share, direction: "highest" };
}

/** The share of a top-share selection, else null. */
export function topShareOf(where: SelectionSpec | undefined): number | null {
  return where && "kind" in where && where.kind === "top" && where.frac != null ? where.frac : null;
}

/** A top selection with a new share (clamped to 0.1%–100%). */
export function withTopShare(where: SelectionSpec, share: number): SelectionSpec {
  if (!("kind" in where) || where.kind !== "top") return where;
  const frac = Math.min(1, Math.max(0.001, share));
  return { ...where, frac, k: undefined };
}

/**
 * Which preset a selection reads as: all → city; zones / saved regions (or an "or" of them) →
 * neighbourhood; the highest share (no "within") by the temperature column → hottest, by the
 * residents column → people; drawn or picked shapes → draw; anything else → custom.
 */
export function wherePresetOf(where: SelectionSpec | undefined, ctx: Pick<WhereContext, "temperatureColumn" | "populationColumn">): WherePreset {
  if (isAll(where)) return "city";
  if (neighbourhoodOf(where)) return "neighbourhood";
  if (where && "kind" in where) {
    if (where.kind === "top" && where.direction === "highest" && where.frac != null && where.k == null && !where.within) {
      if (ctx.temperatureColumn && where.column === ctx.temperatureColumn) return "hottest";
      if (ctx.populationColumn && where.column === ctx.populationColumn) return "people";
    }
    if (DRAWN_KINDS.has(where.kind)) return "draw";
  }
  return "custom";
}

/**
 * The selection a preset starts from, or null when it needs a pick first (drawing before anything
 * is drawn) or its data is missing. A neighbourhood starts from the first saved region (named
 * areas read best), else the first zone; the top presets use `share`.
 */
export function presetSelection(preset: WherePreset, ctx: WhereContext, share: number = TOP_SHARE): SelectionSpec | null {
  switch (preset) {
    case "city":
      return { kind: "all" };
    case "neighbourhood":
      if (ctx.regions.length) return { kind: "region", id: ctx.regions[0].id };
      return ctx.zones.length ? { kind: "zones", values: [ctx.zones[0]] } : null;
    case "hottest":
      return ctx.temperatureColumn ? topSelection(ctx.temperatureColumn, share) : null;
    case "people":
      return ctx.populationColumn ? topSelection(ctx.populationColumn, share) : null;
    case "draw":
      return ctx.mapPick?.spec ?? null;
    case "custom":
      return null;
  }
}

function columnLabel(ctx: WhereContext, c: string): string {
  const o = ctx.columns?.find((x) => x.value === c || x.value.endsWith(":" + c));
  return (o?.label ?? c.replace(/^[a-z_]+:/, "")).replace(/\s*\([^)]*\)$/, "").toLowerCase();
}

function regionName(ctx: WhereContext, id: string): string {
  return ctx.regions.find((r) => r.id === id)?.name ?? "a saved area";
}

/**
 * Where a change applies, as the end of a sentence: "across the whole city", "in Downtown and
 * zone 2", "in the hottest 10% of places", "in the area drawn on the map", …
 */
export function wherePhrase(where: SelectionSpec | undefined, ctx: WhereContext): string {
  if (isAll(where)) return "across the whole city";
  const n = neighbourhoodOf(where);
  if (n) return `in ${listJoin([...n.regions.map((id) => regionName(ctx, id)), ...(n.zones.length ? [n.zones.length === 1 ? `zone ${n.zones[0]}` : `zones ${listJoin(n.zones.map(String))}`] : [])])}`;
  const w = where!;
  if ("kind" in w) {
    switch (w.kind) {
      case "top": {
        const preset = wherePresetOf(w, ctx);
        const share = w.frac != null ? fmtPct(w.frac, w.frac * 100 < 1 ? 1 : 0) : null;
        if (preset === "hottest") return `in the hottest ${share} of places`;
        if (preset === "people") return `in the ${share} of places where the most people live`;
        const how = share ? `the ${share} of places` : `the ${fmtInt(w.k ?? 0)} places`;
        const within = w.within ? ` within ${describeSelection(w.within, { regionName: (id) => regionName(ctx, id), columnLabel: (c) => columnLabel(ctx, c) }).replace(/\bcells\b/g, "places")}` : "";
        return `in ${how} with the ${w.direction} ${columnLabel(ctx, w.column)}${within}`;
      }
      case "circle":
      case "rect":
      case "polygon":
        return "in the area drawn on the map";
      case "hex":
        return `in ${fmtInt(w.keys.length)} hexagon${w.keys.length === 1 ? "" : "s"} picked on the map`;
      case "blob":
        return "in the places picked on the map";
      case "cells":
        return `in ${fmtInt(w.ids.length)} listed place${w.ids.length === 1 ? "" : "s"}`;
      default:
        break;
    }
  }
  return `in ${describeSelection(w, { regionName: (id) => regionName(ctx, id), columnLabel: (c) => columnLabel(ctx, c) }).replace(/\bcells\b/g, "places")}`;
}

// ---------------------------------------------------------------- the change in one sentence

export type SentenceContext = WhereContext & {
  levers: readonly Lever[];
  /** Saved plans (per-place changes from a plan): "plan:<id>" → its name. */
  planRefs?: readonly { ref: string; label: string }[];
};

function perCellSource(ref: string | undefined, ctx: SentenceContext): string {
  if (!ref) return "(source not chosen yet)";
  if (ref.startsWith("plan:")) {
    const name = ctx.planRefs?.find((p) => p.ref === ref)?.label;
    return name ? `from the plan “${name}”` : "from a saved plan";
  }
  if (ref.startsWith("blob:")) return "as painted on the map";
  if (ref.startsWith("csv:")) return `from the file ${ref.slice(4).split("/").pop()}`;
  return `from ${ref}`;
}

/**
 * A change in one plain sentence: "Add 10 pts tree canopy in Downtown", "Take away 10 pts paving
 * across the whole city", "Bring tree canopy up to at least 30 pts in the hottest 10% of places",
 * "Plant trees on 50% of the available planting space in zone 1". Missing amounts read "…".
 */
export function editSentence(edit: Edit, ctx: SentenceContext): string {
  const lever = ctx.levers.find((l) => l.var === edit.lever);
  const what = leverMeasure(lever, edit.lever);
  const unit = amountUnit(lever);
  const where = wherePhrase(edit.where, ctx);
  const a = edit.amount;
  const has = Number.isFinite(a);
  const amt = has ? fmtAmount(a as number, unit) : "…";
  let s: string;
  switch (edit.mode) {
    case "add":
      s = has && (a as number) < 0 ? `${FLOW_WORDS.takeAway} ${amt} ${what} ${where}` : `${MODE_WORDS.add.verb} ${amt} ${what} ${where}`;
      break;
    case "floor":
      s = `Bring ${what} up to at least ${amt} ${where}`;
      break;
    case "ceiling":
      s = `Bring ${what} down to at most ${amt} ${where}`;
      break;
    case "set":
      s = `Set ${what} to ${amt} ${where}`;
      break;
    case "scale":
      s = `Multiply ${what} by ${has ? fmtSig(a as number) : "…"} ${where}`;
      break;
    case "fill_headroom": {
      const share = has ? fmtPct(a as number) : "…";
      s = lever?.role === "canopy" || !lever ? `${MODE_WORDS.fill_headroom.verb} trees on ${share} of the available planting space ${where}` : `Use ${share} of the available space for ${what} ${where}`;
      break;
    }
    case "to_percentile":
      s = `Bring ${what} to the city’s ${Number.isFinite(edit.percentile) ? ordinal(edit.percentile as number) : "…"} percentile ${where}`;
      break;
    case "per_cell":
      s = `Change ${what} place by place, ${perCellSource(edit.per_cell_ref, ctx)}${isAll(edit.where) ? "" : ` ${where}`}`;
      break;
    default:
      s = `${MODE_WORDS[edit.mode as EditMode]?.verb ?? String(edit.mode)} ${amt} ${what} ${where}`;
  }
  return s.replace(/\s+/g, " ").trim();
}

// ---------------------------------------------------------------- the live "how much fits" sentence

export type FitTone = "fits" | "clipped" | "nothing";
export type FitSentence = { text: string; tone: FitTone };

/**
 * The step-03 sentence for a change from the compile: requested vs predicted realised change and
 * the share of places clipped. "You asked for +20 pts tree canopy; about 14 pts fits — the rest is
 * beyond what the space allows." Compile numbers are per lever, so a lever used by several changes
 * says so ("Together with your other tree canopy changes, …"). Null before the first compile or
 * while the change is incomplete (it is not compiled then).
 */
export function compileSentence(edit: Edit, ctx: { compile: CompileResult | null; levers: readonly Lever[]; edits: readonly Edit[] }): FitSentence | null {
  if (!ctx.compile || editProblems(edit).length) return null;
  const c = ctx.compile.levers[edit.lever];
  if (!c) return null;
  const lever = ctx.levers.find((l) => l.var === edit.lever);
  const what = leverMeasure(lever, edit.lever);
  const unit = amountUnit(lever);
  const shared = ctx.edits.filter((e) => e.lever === edit.lever && editProblems(e).length === 0).length > 1;
  const req = c.mean_requested;
  if (!c.n_cells || !Number.isFinite(req) || Number(fmtSig(Math.abs(req), 3).replace(/,/g, "")) === 0) {
    if (shared) return null;
    const text = edit.mode === "floor" ? FLOW_WORDS.nothingFloor : edit.mode === "ceiling" ? FLOW_WORDS.nothingCeiling : edit.mode === "fill_headroom" ? FLOW_WORDS.nothingPlanting : FLOW_WORDS.nothingAsked;
    return { text, tone: "nothing" };
  }
  const lead = shared ? fill(FLOW_WORDS.askedTogether, { what }) : FLOW_WORDS.askedFor;
  const avg = edit.mode === "add" && !shared ? "" : ` ${FLOW_WORDS.onAverage}`;
  const asked = `${lead} ${fmtSignedAmount(req, unit)} ${what}${avg}`;
  const real = Number.isFinite(c.predicted_mean_realised) ? c.predicted_mean_realised : req;
  const ratio = real / req;
  const fits = ratio >= 0.98 || fmtAmount(real, unit, 2) === fmtAmount(req, unit, 2);
  if (fits) return { text: `${asked}; ${FLOW_WORDS.allFits}`, tone: "fits" };
  const about = fill(FLOW_WORDS.aboutFits, { amount: fmtAmount(Math.max(0, ratio) * Math.abs(req), unit, 2) });
  const share = c.clipped_share;
  const rest = Number.isFinite(share) && share > 0.005 && share < 0.995 ? fill(FLOW_WORDS.restBeyondSome, { share: fmtPct(share) }) : FLOW_WORDS.restBeyond;
  return { text: `${asked}; ${about} — ${rest}`, tone: "clipped" };
}
