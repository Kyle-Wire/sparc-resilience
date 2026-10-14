// The shared glossary: plain words for SPARC's technical terms, for a city planner. The data lives
// in sparc/core/glossary.json (also read by the standalone results page) and is imported directly
// (vite.config.ts allows the file for the dev server; tsconfig includes it). Planner pages show the
// plain `label`; the `tip` and the `technical` name go in an InfoTip (ⓘ) or a <Term> tooltip.
//
// Helpers:
// - term(id): a term's label/tip/technical/more. Unknown ids throw in dev and tests (a typo should
//   fail loudly); in a production build they fall back to the id itself.
// - lever(role), mode(m), confidence(c), shape(s), heatCategory(k), trustWord(g): the fixed sections.
// - sspLabel(code), periodLabel(period): future climates in plain words.
import data from "../../../sparc/core/glossary.json";

export type TermEntry = {
  /** Plain label shown on the page ("Range reliability"). */
  label: string;
  /** One or two plain sentences (≤ 240 characters) for the tooltip. */
  tip: string;
  /** The technical name ("90% interval coverage"), shown in mono under the tip. */
  technical?: string;
  /** A longer explanation, for help pages and Model details. */
  more?: string;
};

/** Icon key of a lever card (the card component maps it to a glyph). */
export type LeverIcon = "trees" | "paving" | "roofs" | "lever";

export type LeverEntry = {
  /** Plain name: "Trees & canopy". */
  name: string;
  /** One-line effect: what the change does on the ground. */
  oneLine: string;
  icon: LeverIcon;
  /** Plain unit word for sentences ("pts" for percentage points), when the lever's unit is not plain already. */
  unitWord?: string;
  /** What the lever changes, as a noun for sentences: "Add 10 pts tree canopy in Downtown". */
  measure?: string;
};

export type ModeEntry = {
  /** Button / option label. */
  label: string;
  /** One line under the control. */
  help: string;
  /** Verb phrase for the collapsed sentence: "Add 10 pts", "Bring up to at least 30 pts". */
  verb: string;
  /** Shown as one of the planner modes (the rest live in Advanced options). */
  planner: boolean;
};

export type ConfidenceEntry = { chip: string; sentence: string };
export type ShapeEntry = { label: string; tip: string; technical?: string };
export type HeatEntry = { label: string; tip: string };
export type TrustEntry = { label: string; sentence: string };
export type EmissionsEntry = { label: string; technical: string; tip?: string };

type Raw = typeof data;
export type TermId = keyof Raw["terms"];
export type LeverRole = keyof Raw["levers"];
export type ModeKey = keyof Raw["modes"];
export type ConfidenceKey = keyof Raw["confidence"];
export type ShapeKey = keyof Raw["shapes"];
export type HeatKey = keyof Raw["heat"];
export type TrustKey = keyof Raw["trust"];
export type EngineStateKey = keyof Raw["engine"]["states"];

export type Glossary = {
  version: number;
  about: string;
  terms: Record<TermId, TermEntry>;
  levers: Record<LeverRole, Omit<LeverEntry, "icon"> & { icon: string }>;
  modes: Record<ModeKey, ModeEntry>;
  confidence: Record<ConfidenceKey, ConfidenceEntry>;
  shapes: Record<ShapeKey, ShapeEntry>;
  heat: Record<HeatKey, HeatEntry>;
  trust: Record<TrustKey, TrustEntry>;
  trustRanges: Record<"ok" | "narrow" | "wide", string>;
  quickTrust: { tip: string; dot: Record<"good" | "rough" | "none", string>; line: Record<"good" | "hatched" | "rough", string> };
  hatch: Record<"preview" | "result", { label: string; note: string }>;
  engine: { states: Record<EngineStateKey, string>; noModelNoEstimate: string; noCheckpoint: string; checking: string };
  emissions: Record<string, EmissionsEntry>;
  periods: Record<string, { label: string }>;
};

/** The whole glossary (typed view of sparc/core/glossary.json). */
export const GLOSSARY: Glossary = data;

const has = <T extends object>(o: T, k: PropertyKey): k is keyof T => typeof k === "string" && Object.prototype.hasOwnProperty.call(o, k);

/** Unknown keys fail loudly in dev and tests; a production page degrades to the key itself. */
const STRICT = import.meta.env.DEV || import.meta.env.MODE === "test";

/** True when `id` is a glossary term (for data-driven ids, e.g. from the server). */
export function isTermId(id: string): id is TermId {
  return has(GLOSSARY.terms, id);
}

/** A glossary term. Throws for an unknown id in dev and tests. */
export function term(id: TermId): TermEntry {
  if (isTermId(id)) return GLOSSARY.terms[id];
  if (STRICT) throw new Error(`glossary: unknown term "${String(id)}"`);
  return { label: String(id), tip: "" };
}

const LEVER_ICONS: readonly LeverIcon[] = ["trees", "paving", "roofs", "lever"];

/**
 * Plain words of a lever by its role ("canopy", "impervious", "albedo", as Lever.role), or a lever
 * object (its role, else its var when that names a role). Null for a lever without plain words:
 * callers fall back to the catalogue label.
 */
export function lever(l: string | { role?: string | null; var?: string | null } | null | undefined): LeverEntry | null {
  const keys = typeof l === "string" ? [l] : l ? [l.role, l.var] : [];
  for (const k of keys) {
    const key = (k ?? "").toLowerCase();
    if (has(GLOSSARY.levers, key)) {
      const e = GLOSSARY.levers[key];
      return { ...e, icon: (LEVER_ICONS as readonly string[]).includes(e.icon) ? (e.icon as LeverIcon) : "lever" };
    }
  }
  return null;
}

/** Plain words of an edit mode ("add", "floor", …, plus "take_away"/"bring_down" for levers that cool by going down). */
export function mode(m: ModeKey): ModeEntry;
export function mode(m: string): ModeEntry | null;
export function mode(m: string): ModeEntry | null {
  return has(GLOSSARY.modes, m) ? GLOSSARY.modes[m] : null;
}

/** The confidence chip and sentence of a result (Likely.confidence). */
export function confidence(c: ConfidenceKey): ConfidenceEntry;
export function confidence(c: string): ConfidenceEntry | null;
export function confidence(c: string): ConfidenceEntry | null {
  return has(GLOSSARY.confidence, c) ? GLOSSARY.confidence[c] : null;
}

/** A response-curve shape in plain words: saturating → "Diminishing", linear → "Steady", … */
export function shape(s: ShapeKey): ShapeEntry;
export function shape(s: string): ShapeEntry | null;
export function shape(s: string): ShapeEntry | null {
  return has(GLOSSARY.shapes, s) ? GLOSSARY.shapes[s] : null;
}

const HEAT_ORDER: readonly HeatKey[] = ["below", "caution", "extreme_caution", "danger", "extreme_danger"];

/**
 * An NWS heat-index category by key ("extreme_caution") or by index (0 = below caution … 4 =
 * extreme danger, as theme/heat.ts heatCategory() returns).
 */
export function heatCategory(k: HeatKey | 0 | 1 | 2 | 3 | 4): HeatEntry;
export function heatCategory(k: string | number): HeatEntry | null;
export function heatCategory(k: string | number): HeatEntry | null {
  const key = typeof k === "number" ? HEAT_ORDER[k] : k;
  return key !== undefined && has(GLOSSARY.heat, key) ? GLOSSARY.heat[key] : null;
}

/** The run's trust grade in words: label ("Good") and one plain sentence. */
export function trustWord(g: TrustKey): TrustEntry {
  return GLOSSARY.trust[g];
}

/** "SSP2-4.5" / "ssp245" / "RCP 4.5" → "ssp245" / "rcp45" (lower case, letters and digits only). */
function pathwayKey(code: string): string {
  return code.toLowerCase().replace(/[^a-z0-9]/g, "");
}

/** An emissions pathway's entry (plain label, technical code, optional tip), or null when unknown. */
export function ssp(code: string): EmissionsEntry | null {
  const k = pathwayKey(code);
  return has(GLOSSARY.emissions, k) ? GLOSSARY.emissions[k] : null;
}

/** An emissions pathway in plain words ("ssp245" → "moderate emissions"); an unknown code comes back as given. */
export function sspLabel(code: string): string {
  return ssp(code)?.label ?? code;
}

/** A climate period in plain words ("2041-2060" → "mid-century (2041–2060)"); others get an en dash only. */
export function periodLabel(period: string): string {
  const m = /^\s*(\d{4})\s*[-–]\s*(\d{4})\s*$/.exec(period);
  if (!m) return period;
  const key = `${m[1]}-${m[2]}`;
  return has(GLOSSARY.periods, key) ? GLOSSARY.periods[key].label : `${m[1]}–${m[2]}`;
}
