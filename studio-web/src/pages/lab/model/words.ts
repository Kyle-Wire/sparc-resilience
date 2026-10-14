// Plain words of the Scenario Lab Design flow (phase 3), for a city planner rather than a modeller.
// Every planner-facing label in the Lab step flow, preview pane, run button and answer card comes
// from here, so phase 4 can fold it into the shared glossary (src/content/glossary.ts) in one move.
// Precise meanings stay in the tooltips (`tip`); the technical terms stay in Advanced options and
// Model details.
import type { EditMode, EmulatorTrust, EngineState, Lever } from "../../../api/lab";
import type { Likely } from "../../../api/types";
import { fmtPct } from "../../../theme/format";

// ---------------------------------------------------------------- the flow's own words

export const LAB_WORDS = {
  stepWhat: "What to change",
  stepWhere: "Where",
  stepHowMuch: "How much",
  addChange: "Add another change",
  editChange: "Edit",
  removeChange: "Remove",
  advanced: "Advanced options",
  quickEstimate: "Quick estimate",
  quickEstimateKicker: "QUICK ESTIMATE",
  whereChanged: "where you made changes",
  citywide: "citywide",
  runFullModel: "Run full model",
  fullModelKicker: "FULL MODEL RESULT",
  buildQuickEstimate: "Build quick estimate",
  noQuickEstimateTitle: "No quick estimate yet",
  /** The body under "No quick estimate yet" when the full model can run (the build button gives the time). */
  noQuickEstimate: "Build it once to see the effect of each change as you make it. The full model works either way.",
  /** …when the run has no usable full model either (no checkpoint, incompatible): the estimate is the way in. */
  noQuickEstimateNoModel: "Build it once to try changes on this run.",
  /** …while the model's state is unknown or it failed to load (no promise about the full model). */
  noQuickEstimatePlain: "Build it once to see the effect of each change as you make it.",
  startFromExample: "Start from an example",
  moreExamples: (n: number) => `More examples (${n})`,
  fewerExamples: "Fewer examples",
  emptyChanges: "Pick what to change to start. The quick estimate updates as you go.",
  /** The intro with no quick estimate and no usable full model: the estimate is the first step. */
  emptyChangesBuildFirst: "First build the quick estimate (right), then pick what to change.",
  /** The intro with no quick estimate while the full model can run (or its state is not known yet). */
  emptyChangesNoEstimate: "Pick what to change to start. Build the quick estimate (right) to see the effect as you go.",
  previewIdle: "Make a change to see the quick estimate.",
  previewBusy: "Working out the quick estimate…",
  pinToCompare: "Pin to compare",
  exportPack: "Export decision pack",
  showOnMap: "Show on map",
  /** The run block once the full model has a current result: a secondary button to the answer card… */
  seeFullResult: "See full model result",
  /** …and one line under it (no number: the answer card carries it). */
  fullModelUpToDate: "Full model is up to date for these changes.",
  /** A result exists but is out of date (the model changed, or the changes are not saved yet). */
  fullModelOutdated: "The full model result below is out of date — run it again to refresh it.",
  /** The quick-estimate kicker once the full model has the firm answer. */
  quickEstimateBefore: "Quick estimate (before the full model)",
} as const;

/**
 * The body of the "No quick estimate yet" box, by the engine state (so it never contradicts the
 * engine line under Run full model): without a usable model the estimate is the way in.
 */
export function noQuickEstimateText(state: EngineState | null): string {
  if (state === "no_checkpoint" || state === "incompatible") return LAB_WORDS.noQuickEstimateNoModel;
  if (state === null || state === "error") return LAB_WORDS.noQuickEstimatePlain;
  return LAB_WORDS.noQuickEstimate;
}

/**
 * The steps rail's intro before the first change. Without a quick estimate it never promises one:
 * with no usable full model either, building the estimate (right pane) comes first.
 */
export function emptyChangesText(noEmulator: boolean, state: EngineState | null): string {
  if (!noEmulator) return LAB_WORDS.emptyChanges;
  if (state === "no_checkpoint" || state === "incompatible") return LAB_WORDS.emptyChangesBuildFirst;
  return LAB_WORDS.emptyChangesNoEstimate;
}

/** Advanced options (one disclosure under step 3). */
export const ADVANCED_WORDS = {
  clip: "Keep changes within realistic limits",
  clipTip: "Never push a place beyond the range seen anywhere in the city today (the model is only trustworthy inside it).",
  mediators: "Include knock-on effects",
  mediatorsTip: "Let related measures follow the change, e.g. greenness follows tree cover.",
  expert: "Expert mode",
  expertTip: "Edit any model input, not only the planning levers.",
  portable: "Reusable in other runs",
  portableYes: "Yes — the places are drawn on the map, so other runs can use this scenario.",
  portableNo: "This run only — the places come from this run's own grid.",
  details: "Change details",
  detailsCaption: "Per change: what you asked for and what the space allows",
  fullModelTime: "Full model run",
  cellsChanged: "Places changed",
  residents: "Residents there",
  cost: "Estimated cost",
  editor: "Scenario details, all modes, costs and files",
  model: "Model",
} as const;

// ---------------------------------------------------------------- levers ("What to change")

/** Icon key of a lever card (the card component maps it to a glyph). */
export type LeverIcon = "trees" | "paving" | "roofs" | "lever";

export type LeverWords = {
  /** Plain name: "Trees & canopy". */
  name: string;
  /** One-line effect: what the change does on the ground. */
  oneLine: string;
  icon: LeverIcon;
  /** Plain unit word for sentences ("pts" for percentage points), when the lever's unit is not plain already. */
  unitWord?: string;
};

/** Plain words by lever role (Lever.role). */
export const LEVER_WORDS: Record<"canopy" | "impervious" | "albedo", LeverWords> = {
  canopy: { name: "Trees & canopy", oneLine: "More tree cover shades streets and cools the air around it.", icon: "trees", unitWord: "pts" },
  impervious: { name: "Less paving", oneLine: "Swapping pavement for soil and plants keeps the ground cooler.", icon: "paving", unitWord: "pts" },
  albedo: { name: "Cool roofs & surfaces", oneLine: "Lighter roofs and pavements reflect sunlight instead of soaking it up.", icon: "roofs" },
};

/** "pct_canopy" / "Canopy cover" → "Canopy Cover" (the fallback name of an unknown lever). */
export function titleCase(s: string): string {
  return s
    .replace(/[_-]+/g, " ")
    .trim()
    .split(/\s+/)
    .map((w) => (w ? w[0].toUpperCase() + w.slice(1) : w))
    .join(" ");
}

/** Plain words of a lever: by role, else the catalogue label in title case. */
export function leverWords(lever: Pick<Lever, "var" | "label" | "role" | "direction"> | undefined, varName?: string): LeverWords {
  const role = lever?.role?.toLowerCase() ?? null;
  if (role && role in LEVER_WORDS) return LEVER_WORDS[role as keyof typeof LEVER_WORDS];
  const name = titleCase(lever?.label || lever?.var || varName || "Change");
  const oneLine = lever ? (lever.direction === "increase" ? `More ${lever.label.toLowerCase()} cools the city.` : `Less ${lever.label.toLowerCase()} cools the city.`) : "A change to one of the model's inputs.";
  return { name, oneLine, icon: "lever" };
}

/** Plain unit for sentences: "pp"/"%" → "pts", the lever's own unit otherwise ("" for unitless). */
export function leverUnitWord(lever: Pick<Lever, "unit" | "role" | "var" | "label" | "direction"> | undefined): string {
  const w = leverWords(lever);
  if (w.unitWord && (lever?.unit === "pp" || lever?.unit === "%" || lever?.unit === "pct")) return w.unitWord;
  if (lever?.unit === "pp") return "pts";
  return lever?.unit ?? "";
}

// ---------------------------------------------------------------- how-much modes

export type ModeWords = {
  /** Button / option label. */
  label: string;
  /** One line under the control. */
  hint: string;
  /** Verb phrase for the collapsed sentence: "Add 10 pts", "Bring up to at least 30 pts". */
  verb: string;
  /** Shown as one of the three planner modes (the rest live in Advanced options). */
  planner: boolean;
};

export const MODE_WORDS: Record<EditMode, ModeWords> = {
  add: { label: "Add", hint: "Adds this much everywhere you chose.", verb: "Add", planner: true },
  floor: { label: "Bring up to at least", hint: "Raises places below this level up to it; places already above stay as they are.", verb: "Bring up to at least", planner: true },
  fill_headroom: {
    label: "Use available planting space",
    hint: "Plants trees on this share of the open ground that can take them (not buildings or roads).",
    verb: "Plant",
    planner: true,
  },
  set: { label: "Set to exactly", hint: "Every chosen place gets exactly this value.", verb: "Set to", planner: false },
  scale: { label: "Multiply by", hint: "Multiplies today's value in every chosen place.", verb: "Multiply by", planner: false },
  ceiling: { label: "Cap at", hint: "Lowers places above this level down to it; places below stay as they are.", verb: "Cap at", planner: false },
  to_percentile: { label: "Bring up to a citywide level", hint: "Brings each chosen place up to the level that this share of the city already reaches.", verb: "Bring up to the city's", planner: false },
  per_cell: { label: "Place by place (plan, brush or file)", hint: "A different change for each place, from a saved plan, brush strokes or an imported file.", verb: "Apply", planner: false },
};

/** The three planner modes, in order. */
export const PLANNER_MODES: readonly EditMode[] = ["add", "floor", "fill_headroom"];
/** The other modes (Advanced options). */
export const ADVANCED_MODES: readonly EditMode[] = ["set", "scale", "ceiling", "to_percentile", "per_cell"];

/** "Use available planting space" needs a canopy lever with the planner layers (plantable headroom). */
export function modeFits(mode: EditMode, lever: Pick<Lever, "role" | "headroom_available"> | undefined): boolean {
  return mode !== "fill_headroom" || (lever?.role === "canopy" && lever.headroom_available);
}

// ---------------------------------------------------------------- where presets

export type WherePreset = "city" | "neighbourhood" | "hottest" | "people" | "draw" | "custom";

export const WHERE_PRESETS: Record<WherePreset, { label: string; hint: string }> = {
  city: { label: "Whole city", hint: "Every place in the study area." },
  neighbourhood: { label: "A neighbourhood", hint: "One or more zones or saved areas." },
  hottest: { label: "Hottest places", hint: "The hottest share of the city, by measured temperature." },
  people: { label: "Where most people are exposed", hint: "The places where the most residents live." },
  draw: { label: "Draw on the map", hint: "Draw a shape or pick areas on the map, then use it here." },
  custom: { label: "Custom filter…", hint: "Combine conditions (zones, layers, distances) in the full selection builder." },
};

/** Preset order of the segmented control ("custom" is a link, not a segment). */
export const WHERE_PRESET_ORDER: readonly WherePreset[] = ["city", "neighbourhood", "hottest", "people", "draw"];

// ---------------------------------------------------------------- engine (full model)

/** One muted line under "Run full model". `{pct}` is replaced by the loading progress. */
export const ENGINE_WORDS: Record<EngineState, string> = {
  no_checkpoint: "Model not available for this run — quick estimate only",
  cold: "Model loads when you run it",
  queued: "Model waiting to load…",
  loading: "Loading model… {pct}",
  ready: "Model ready",
  busy: "Model busy — your run will wait its turn",
  incompatible: "Model not available for this run — quick estimate only",
  error: "Model could not load — see Advanced options",
};

/** The engine line when the run has neither a full model nor a quick estimate yet (no dead end: build the estimate). */
export const ENGINE_NO_MODEL_NO_ESTIMATE = "Full model not available for this run — build the quick estimate above to explore changes";

/**
 * The engine line with the loading share filled in ("Loading model… 40%"). With `noEmulator` (no quick
 * estimate yet either), a run without a usable model says to build the quick estimate instead of
 * "quick estimate only".
 */
export function engineLine(state: EngineState | null, progress: number | null, opts: { noEmulator?: boolean } = {}): string {
  if (!state) return "Checking the model…";
  if (opts.noEmulator && (state === "no_checkpoint" || state === "incompatible")) return ENGINE_NO_MODEL_NO_ESTIMATE;
  const text = ENGINE_WORDS[state];
  return text.replace("{pct}", progress !== null && Number.isFinite(progress) ? fmtPct(progress) : "").trimEnd();
}

// ---------------------------------------------------------------- future climates

/** Emissions pathways in plain words, by normalised code ("ssp245", "rcp45"). */
const EMISSIONS: Record<string, string> = {
  ssp119: "very low emissions",
  ssp126: "low emissions",
  ssp245: "moderate emissions",
  ssp370: "high emissions",
  ssp434: "moderate emissions",
  ssp460: "moderate-high emissions",
  ssp534: "high, then falling emissions",
  ssp585: "very high emissions",
  rcp26: "low emissions",
  rcp45: "moderate emissions",
  rcp60: "moderate-high emissions",
  rcp85: "very high emissions",
};

/** "SSP2-4.5" / "ssp245" / "RCP 4.5" → "ssp245" / "rcp45" (lower case, letters and digits only). */
export function pathwayCode(experiment: string): string {
  return experiment.toLowerCase().replace(/[^a-z0-9]/g, "");
}

/** The emissions pathway in plain words ("moderate emissions"); null when it is not a known one. */
export function emissionsWord(experiment: string): string | null {
  return EMISSIONS[pathwayCode(experiment)] ?? null;
}

/** "2041-2060" → "2041–2060" (an en dash between the years). */
export function periodWord(period: string): string {
  return period.trim().replace(/^(\d{4})\s*[-–]\s*(\d{4})$/, "$1–$2");
}

/** The era of a period by its first year: near-term (to 2040), mid-century (to 2070), end of century. */
function eraWord(period: string): string | null {
  const y = Number(/(\d{4})/.exec(period)?.[1]);
  if (!Number.isFinite(y) || y <= 0) return null;
  return y <= 2040 ? "Near-term" : y <= 2070 ? "Mid-century" : "End of century";
}

/**
 * A future climate in plain words: ("ssp245", "2041-2060") → "Mid-century, moderate emissions
 * (2041–2060)". The pathway code (SSP2-4.5…) stays out of the label: callers put it in a tooltip or
 * in Technical details. An unknown pathway keeps its code, upper case.
 */
export function futureLabel(experiment: string, period: string): string {
  const era = eraWord(period);
  const emissions = emissionsWord(experiment) ?? experiment.toUpperCase();
  const years = periodWord(period);
  return `${era ? `${era}, ` : ""}${emissions}${years ? ` (${years})` : ""}`;
}

/**
 * A future-climate case of the impacts tables ("ssp245 2041-2060", "SSP2-4.5 2041-2060") in plain
 * words; null when the case names no pathway and period (today, or an unknown case).
 */
export function futureCaseLabel(c: string): string | null {
  const years = /(\d{4})\s*[-–]\s*(\d{4})/.exec(c);
  const pathway = casePathway(c);
  if (!years || !pathway) return null;
  return futureLabel(pathway, `${years[1]}-${years[2]}`);
}

/** The emissions pathway a case names ("SSP2-4.5 2041-2060" → "SSP2-4.5"), or null. */
export function casePathway(c: string): string | null {
  return /\b(ssp\s*\d[\d.\-]*\d|rcp\s*\d[\d.]*)/i.exec(c)?.[1] ?? null;
}

/** Where the humidity of the heat-risk figures comes from, in plain words. */
export function humiditySourceWord(source: string | null | undefined): string {
  switch ((source ?? "").toLowerCase()) {
    case "campaign":
      return "in the field survey";
    case "station":
      return "at the weather station";
    default:
      return "";
  }
}

// ---------------------------------------------------------------- trust (quick estimate)

/** The trust dot of a lever card. */
export const TRUST_WORDS: Record<EmulatorTrust, { label: string; tone: "good" | "warn" | "neutral" }> = {
  good: { label: "Quick estimate reliable", tone: "good" },
  rough: { label: "Quick estimate rough", tone: "warn" },
  none: { label: "No quick estimate", tone: "neutral" },
};

/** Tooltip of the trust dot. */
export const TRUST_TIP = "How reliable the quick estimate is for this change. The full model is always the firm answer.";

/** The plain trust line of the preview pane. */
export const PREVIEW_TRUST_LINE = {
  good: "The quick estimate is reliable for this change.",
  hatched: "Rough in the hatched areas — run the full model for a firm answer.",
  rough: "The quick estimate is rough for this change — run the full model for a firm answer.",
} as const;

// ---------------------------------------------------------------- confidence (full model)

export type ConfidenceTone = "cool" | "warm" | "neutral";

/** The answer card's confidence chip by Likely.confidence. Cooling is never shown in crimson. */
export const CONFIDENCE_WORDS: Record<Likely["confidence"], { label: string; tone: ConfidenceTone; tip: string }> = {
  confident_cools: { label: "Confident it cools", tone: "cool", tip: "The whole likely range is below zero: the change cools." },
  confident_warms: { label: "Confident it warms", tone: "warm", tip: "The whole likely range is above zero: the change warms." },
  could_be_zero: { label: "Might be no real effect", tone: "neutral", tip: "The likely range includes zero: the effect could be too small to tell apart from no change." },
  unknown: { label: "No range available", tone: "neutral", tip: "This result has no uncertainty estimate, so there is no likely range." },
};

// ---------------------------------------------------------------- "why run the full model" in plain words

const SWAPS: [RegExp, string][] = [
  [/\bthe linear preview\b/gi, "the quick estimate"],
  [/\bpreviews?\b/gi, "quick estimate"],
  [/\bemulator\b/gi, "quick estimate"],
  [/\brun exact\b/gi, "run the full model"],
  [/\bexact (run|result|engine)s?\b/gi, "full model"],
  [/\bedits\b/gi, "changes"],
  [/\bedited cells\b/gi, "changed places"],
  [/\bcells\b/gi, "places"],
  [/\bextrapolates\b/gi, "goes beyond measured conditions"],
  [/\bextrapolate\b/gi, "go beyond measured conditions"],
  [/\bextrapolat(ed|ion|ing)\b/gi, "beyond measured conditions"],
];

function sentence(s: string): string {
  const t = s.trim().replace(/\s+/g, " ");
  if (!t) return t;
  const cap = t[0].toUpperCase() + t.slice(1);
  return /[.!?]$/.test(cap) ? cap : `${cap}.`;
}

/**
 * exactAdvice() reasons rewritten for a planner. Known reasons get their own sentence (lever labels
 * become plain lever names); server-supplied reasons get the term swaps above. Duplicates are dropped.
 */
export function plainAdvice(reasons: readonly string[], levers: readonly Pick<Lever, "var" | "label" | "role" | "direction">[] = []): string[] {
  const nameOf = (label: string) => {
    const l = levers.find((x) => x.label === label || x.var === label);
    return l ? leverWords(l).name : label;
  };
  const out: string[] = [];
  for (const r of reasons) {
    let m: RegExpExecArray | null;
    let text: string;
    if (/^no usable preview/i.test(r)) text = "There is no quick estimate for this kind of change.";
    // the server's hatching reasons (sparc/studio/engine/preview.py hatch() and preview_status())
    else if ((m = /unreliable at this scale \(([\d,]+) edited cells/i.exec(r))) text = `The change covers a large area (${m[1]} places), where the quick estimate is less reliable.`;
    else if ((m = /unreliable: (\S+) has a rough emulator and edits ([\d,]+) cells/i.exec(r))) text = `The quick estimate is rough for ${nameOf(m[1]).toLowerCase()} over this many places (${m[2]}).`;
    else if ((m = /unreliable at this scale \(uniform (\S+) rel\. error ([\d.]+%)\)/i.exec(r))) text = `Across the whole city the quick estimate for ${nameOf(m[1]).toLowerCase()} can be off by about ${m[2]}.`;
    else if ((m = /^the emulator has no (.+) lever$/i.exec(r))) text = `There is no quick estimate for ${m[1].split(/,\s*/).map((v) => nameOf(v).toLowerCase()).join(", ")}.`;
    else if (/emulator was built on other cells/i.test(r)) text = "The quick estimate was built for a different map of this run; rebuild it.";
    else if (/preview is unreliable at this scale/i.test(r)) text = "The quick estimate is less reliable for a change of this size.";
    else if ((m = /^(.+): preview is rough$/i.exec(r))) text = `The quick estimate is rough for ${nameOf(m[1]).toLowerCase()}.`;
    else if ((m = /^(.+): preview is unavailable$/i.exec(r))) text = `There is no quick estimate for ${nameOf(m[1]).toLowerCase()}.`;
    else if (/requested change is clipped/i.test(r)) text = "More than a fifth of what you asked for does not fit the space available, and the quick estimate does not account for that.";
    else text = sentence(SWAPS.reduce((s, [re, w]) => s.replace(re, w), r));
    if (!out.includes(text)) out.push(text);
  }
  return out;
}
