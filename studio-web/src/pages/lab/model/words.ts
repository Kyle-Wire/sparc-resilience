// Plain words of the Scenario Lab Design flow, for a city planner rather than a modeller. Every
// planner-facing label in the Lab step flow, preview pane, run button and answer card comes from
// here. Repeated concepts (levers, modes, engine lines, quick-estimate trust, confidence, emissions
// futures and periods, the plain names of terms) read the shared glossary (src/content/glossary.ts →
// sparc/core/glossary.json), so the Lab, the run pages and the results page say the same thing;
// the Lab's own one-off sentences stay here. Precise meanings stay in the tooltips (`tip`, InfoTip);
// the technical terms stay in Advanced options and Model details.
import type { EditMode, EmulatorTrust, EngineState, Lever } from "../../../api/lab";
import type { Likely } from "../../../api/types";
import { GLOSSARY, confidence, lever as glossaryLever, mode as glossaryMode, periodLabel, ssp, term, type LeverEntry, type LeverIcon as GlossaryLeverIcon } from "../../../content/glossary";
import { fmtPct } from "../../../theme/format";

const lower = (s: string) => s.charAt(0).toLowerCase() + s.slice(1);
const cap = (s: string) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);

// ---------------------------------------------------------------- the flow's own words

export const LAB_WORDS = {
  stepWhat: "What to change",
  stepWhere: "Where",
  stepHowMuch: "How much",
  addChange: "Add another change",
  editChange: "Edit",
  removeChange: "Remove",
  advanced: "Advanced options",
  quickEstimate: term("emulator_preview").label,
  quickEstimateKicker: term("emulator_preview").label.toUpperCase(),
  whereChanged: lower(term("where_changed").label),
  citywide: lower(term("citywide").label),
  runFullModel: term("run_exact").label,
  fullModelKicker: "FULL MODEL RESULT",
  buildQuickEstimate: "Build quick estimate",
  noQuickEstimateTitle: "No quick estimate yet",
  /** The body under "No quick estimate yet" when the full model can run (the build button gives the time). */
  noQuickEstimate: "Build it once to see the effect of each change as you make it. The full model works either way.",
  /** …when the run saved no trained models (no checkpoint): the estimate cannot be built either, a new run is the way in. */
  noQuickEstimateNoModel: "Launch a new run (Fast takes a few minutes) to try changes.",
  /** …when the full model cannot load (incompatible) but the estimate can still be built: it is the way in. */
  noQuickEstimateBuildOnly: "Build it once to try changes on this run.",
  /** The link replacing the build button on a run without saved models (to /p/{pid}/launch). */
  launchRun: "Launch a run",
  /** …while the model's state is unknown or it failed to load (no promise about the full model). */
  noQuickEstimatePlain: "Build it once to see the effect of each change as you make it.",
  startFromExample: "Start from an example",
  moreExamples: (n: number) => `More examples (${n})`,
  fewerExamples: "Fewer examples",
  emptyChanges: "Pick what to change to start. The quick estimate updates as you go.",
  /** The intro on a run that saved no trained models (no checkpoint) and has no quick estimate: a new run is the way in. */
  emptyChangesBuildFirst: "This run did not save its trained models, so changes can't be tried on it. Launch a new run to design scenarios.",
  /** The intro with no quick estimate while the full model cannot load (incompatible): the estimate is the first step. */
  emptyChangesEstimateFirst: "First build the quick estimate (right), then pick what to change.",
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
  quickEstimateBefore: `${term("emulator_preview").label} (before the ${lower(term("exact_engine").label)})`,
} as const;

/**
 * The body of the "No quick estimate yet" box, by the engine state (so it never contradicts the
 * engine line under Run full model): a run that saved no trained models cannot get an estimate
 * either (the build unpickles checkpoint.pkl), so a new run is the way in; a model that cannot load
 * still leaves the estimate.
 */
export function noQuickEstimateText(state: EngineState | null): string {
  if (state === "no_checkpoint") return LAB_WORDS.noQuickEstimateNoModel;
  if (state === "incompatible") return LAB_WORDS.noQuickEstimateBuildOnly;
  if (state === null || state === "error") return LAB_WORDS.noQuickEstimatePlain;
  return LAB_WORDS.noQuickEstimate;
}

/**
 * The steps rail's intro before the first change. Without a quick estimate it never promises one:
 * a run that saved no trained models sends to a new run; with a model that cannot load, building
 * the estimate (right pane) comes first.
 */
export function emptyChangesText(noEmulator: boolean, state: EngineState | null): string {
  if (!noEmulator) return LAB_WORDS.emptyChanges;
  if (state === "no_checkpoint") return LAB_WORDS.emptyChangesBuildFirst;
  if (state === "incompatible") return LAB_WORDS.emptyChangesEstimateFirst;
  return LAB_WORDS.emptyChangesNoEstimate;
}

/** Advanced options (one disclosure under step 3). */
export const ADVANCED_WORDS = {
  clip: term("realistic_limits").label,
  clipTip: term("realistic_limits").tip,
  mediators: `Include ${lower(term("mediators").label)}`,
  mediatorsTip: term("mediators").tip,
  expert: term("expert_mode").label,
  expertTip: term("expert_mode").tip,
  portable: term("portable").label,
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
export type LeverIcon = GlossaryLeverIcon;

/** A lever's plain words (the glossary's lever entry: name, one line, icon, unit word, measure). */
export type LeverWords = LeverEntry;

/** Plain words by lever role (Lever.role), from the glossary. */
export const LEVER_WORDS: Record<"canopy" | "impervious" | "albedo", LeverWords> = {
  canopy: glossaryLever("canopy")!,
  impervious: glossaryLever("impervious")!,
  albedo: glossaryLever("albedo")!,
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

/** A mode's words from the glossary (its `help` is the hint under the control). */
function modeWords(m: EditMode): ModeWords {
  const g = glossaryMode(m);
  return { label: g.label, hint: g.help, verb: g.verb, planner: g.planner };
}

/** Plain words of every edit mode, from the glossary. */
export const MODE_WORDS: Record<EditMode, ModeWords> = {
  add: modeWords("add"),
  floor: modeWords("floor"),
  fill_headroom: modeWords("fill_headroom"),
  set: modeWords("set"),
  scale: modeWords("scale"),
  ceiling: modeWords("ceiling"),
  to_percentile: modeWords("to_percentile"),
  per_cell: modeWords("per_cell"),
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

/** One muted line under "Run full model" (glossary engine states). `{pct}` is replaced by the loading progress. */
export const ENGINE_WORDS: Record<EngineState, string> = GLOSSARY.engine.states;

/** The engine line when the run has neither a full model nor a quick estimate yet (no dead end: build the estimate). */
export const ENGINE_NO_MODEL_NO_ESTIMATE = GLOSSARY.engine.noModelNoEstimate;

/** The engine line when the run saved no trained models and has no quick estimate (the estimate cannot be built either). */
export const ENGINE_NO_CHECKPOINT = GLOSSARY.engine.noCheckpoint;

/**
 * The engine line with the loading share filled in ("Loading model… 40%"). With `noEmulator` (no quick
 * estimate yet either), a run whose model cannot load says to build the quick estimate instead of
 * "quick estimate only", and a run that saved no trained models says so (no estimate to build).
 */
export function engineLine(state: EngineState | null, progress: number | null, opts: { noEmulator?: boolean } = {}): string {
  if (!state) return GLOSSARY.engine.checking;
  if (opts.noEmulator && state === "no_checkpoint") return ENGINE_NO_CHECKPOINT;
  if (opts.noEmulator && state === "incompatible") return ENGINE_NO_MODEL_NO_ESTIMATE;
  const text = ENGINE_WORDS[state];
  return text.replace("{pct}", progress !== null && Number.isFinite(progress) ? fmtPct(progress) : "").trimEnd();
}

// ---------------------------------------------------------------- future climates

/** "SSP2-4.5" / "ssp245" / "RCP 4.5" → "ssp245" / "rcp45" (lower case, letters and digits only). */
export function pathwayCode(experiment: string): string {
  return experiment.toLowerCase().replace(/[^a-z0-9]/g, "");
}

/** The emissions pathway in plain words ("moderate emissions", from the glossary); null when it is not a known one. */
export function emissionsWord(experiment: string): string | null {
  return ssp(experiment)?.label ?? null;
}

/** "2041-2060" → "2041–2060" (an en dash between the years). */
export function periodWord(period: string): string {
  return period.trim().replace(/^(\d{4})\s*[-–]\s*(\d{4})$/, "$1–$2");
}

/**
 * A future climate in plain words, as the run pages word it (glossary emissions and periods):
 * ("ssp245", "2041-2060") → "Moderate emissions, mid-century (2041–2060)"; a period the glossary
 * does not name keeps its years ("Moderate emissions, 2061–2080"). The pathway code (SSP2-4.5…)
 * stays out of the label: callers put it in a tooltip or in Technical details. An unknown pathway
 * keeps its code, upper case.
 */
export function futureLabel(experiment: string, period: string): string {
  const emissions = emissionsWord(experiment) ?? experiment.toUpperCase();
  const when = periodLabel(periodWord(period));
  return `${cap(emissions)}${when ? `, ${when}` : ""}`;
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

/** The trust dot of a lever card (glossary quick-estimate trust). */
export const TRUST_WORDS: Record<EmulatorTrust, { label: string; tone: "good" | "warn" | "neutral" }> = {
  good: { label: GLOSSARY.quickTrust.dot.good, tone: "good" },
  rough: { label: GLOSSARY.quickTrust.dot.rough, tone: "warn" },
  none: { label: GLOSSARY.quickTrust.dot.none, tone: "neutral" },
};

/** Tooltip of the trust dot. */
export const TRUST_TIP = GLOSSARY.quickTrust.tip;

/** The plain trust line of the preview pane. */
export const PREVIEW_TRUST_LINE: Readonly<Record<"good" | "hatched" | "rough", string>> = GLOSSARY.quickTrust.line;

// ---------------------------------------------------------------- confidence (full model)

export type ConfidenceTone = "cool" | "warm" | "neutral";

const confidenceWords = (c: Likely["confidence"], tone: ConfidenceTone) => ({ label: confidence(c).chip, tone, tip: confidence(c).sentence });

/** The answer card's confidence chip by Likely.confidence (glossary words). Cooling is never shown in crimson. */
export const CONFIDENCE_WORDS: Record<Likely["confidence"], { label: string; tone: ConfidenceTone; tip: string }> = {
  confident_cools: confidenceWords("confident_cools", "cool"),
  confident_warms: confidenceWords("confident_warms", "warm"),
  could_be_zero: confidenceWords("could_be_zero", "neutral"),
  unknown: confidenceWords("unknown", "neutral"),
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
