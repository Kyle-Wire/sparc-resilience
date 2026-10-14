// Plain-language result wording (SPEC §7.7): the headline with its likely range, the
// confidence line from the 95% interval (est ± 1.96·SE), the qualifiers and "what it buys".
// - "Confident it cools" iff hi < 0; "Confident it warms" iff lo > 0; "Could be zero" when
//   the interval spans 0.
// - "partly beyond the conditions measured in this city (23% of the places you changed)" when more
//   than 20% of the EDITED cells are extrapolated ("mostly beyond…" above half); "A separate cross-check finds less cooling — see
//   How sure are we" (the independent causal check disagrees: less / more cooling or warming, or the
//   opposite effect, by its estimate against the model's citywide one); "preview only — not verified".
// - The answer card's headline has no numbers (answerHeadlineText): its big number and likely-range
//   strip carry them, so "Cools the places you changed" says what happens and the figures show once.
import type { Impacts, Result } from "../../../api/lab";
import type { Likely } from "../../../api/types";
import { fmtInt, fmtNum, fmtPct, fmtSigned, fmtValue, unitLabel } from "../../../theme/format";
import { casePathway, emissionsWord, pathwayCode, periodWord } from "./words";

export type ConfidenceWord = "Confident it cools" | "Confident it warms" | "Could be zero" | "No uncertainty estimate";

/** 95% interval of a Likely: its own lo/hi, else est ± 1.96·SE. */
export function interval95(l: Pick<Likely, "estimate" | "se" | "lo" | "hi">): [number, number] | null {
  if (l.lo !== null && l.hi !== null && Number.isFinite(l.lo) && Number.isFinite(l.hi)) return [l.lo, l.hi];
  if (l.se !== null && Number.isFinite(l.se) && Number.isFinite(l.estimate)) return [l.estimate - 1.96 * l.se, l.estimate + 1.96 * l.se];
  return null;
}

/**
 * The Likely with its 95% range filled in from ±1.96·SE when the server sent only the SE, so
 * every card (headline range and confidence line) words it exactly as `confidenceWord` does.
 */
export function withRange(l: Likely): Likely {
  if (l.lo !== null && l.hi !== null && Number.isFinite(l.lo) && Number.isFinite(l.hi)) return l;
  const iv = interval95(l);
  return iv ? { ...l, lo: iv[0], hi: iv[1] } : l;
}

export function confidenceWord(l: Pick<Likely, "estimate" | "se" | "lo" | "hi">): ConfidenceWord {
  const iv = interval95(l);
  if (!iv) return "No uncertainty estimate";
  if (iv[1] < 0) return "Confident it cools";
  if (iv[0] > 0) return "Confident it warms";
  return "Could be zero";
}

/** Share above which the extrapolation qualifier is added. */
export const EXTRAPOLATION_QUALIFIER_AT = 0.2;

/** Share above which the qualifier says "mostly" rather than "partly". */
export const EXTRAPOLATION_MOSTLY_AT = 0.5;

export function extrapolationQualifier(fracEdited: number | null | undefined): string | null {
  if (fracEdited === null || fracEdited === undefined || !Number.isFinite(fracEdited) || fracEdited <= EXTRAPOLATION_QUALIFIER_AT) return null;
  return `${fracEdited > EXTRAPOLATION_MOSTLY_AT ? "mostly" : "partly"} beyond the conditions measured in this city (${fmtPct(fracEdited)} of the places you changed)`;
}

/** "Cools the edited area by 0.62 °F (likely range 0.44–0.80 °F)." */
export function headline(l: Likely, unit: string, subject: string, decimals = 2): string {
  const est = l.estimate;
  if (!Number.isFinite(est)) return `No estimate for ${subject}.`;
  if (est === 0) return `Leaves ${subject} unchanged.`;
  const verb = est < 0 ? "Cools" : "Warms";
  let s = `${verb} ${subject} by ${fmtValue(Math.abs(est), unit, decimals)}`;
  const iv = interval95(l);
  if (iv) {
    // In "by how much" terms: for cooling the range of the magnitude is [−hi, −lo].
    const lo = est < 0 ? -iv[1] : iv[0];
    const hi = est < 0 ? -iv[0] : iv[1];
    s += ` (likely range ${fmtNum(lo, decimals)}–${fmtValue(hi, unit, decimals)})`;
  }
  return s + ".";
}

/** "City-wide: 0.021 °F cooler." */
export function cityLine(city: Likely, unit: string, decimals = 3): string {
  const e = city.estimate;
  if (!Number.isFinite(e)) return "";
  if (e === 0) return "City-wide: no change.";
  return `City-wide: ${fmtValue(Math.abs(e), unit, decimals)} ${e < 0 ? "cooler" : "warmer"}.`;
}

/** The section name the causal qualifier points to (the answer card turns it into a link). */
export const CAUSAL_QUALIFIER_LINK = "How sure are we";
/** The opening words every causal qualifier shares (the card finds the line by them). */
const CAUSAL_LEAD = "A separate cross-check";
/** "Keep in mind" line when the independent causal check disagrees and its direction is unknown. */
export const CAUSAL_QUALIFIER = `${CAUSAL_LEAD} gives a different answer — see ${CAUSAL_QUALIFIER_LINK}`;

/**
 * "Keep in mind" line when the independent causal check disagrees, worded by what it finds against
 * the model's estimate of the same quantity (both citywide, negative = cooler): "A separate
 * cross-check finds less cooling — see How sure are we", "…more cooling…", "…less warming…", "…more
 * warming…", or "…the opposite effect…". Without both figures: the generic CAUSAL_QUALIFIER.
 */
export function causalQualifier(checkDelta: number | null | undefined, modelEstimate: number | null | undefined): string {
  if (typeof checkDelta !== "number" || typeof modelEstimate !== "number" || !Number.isFinite(checkDelta) || !Number.isFinite(modelEstimate) || modelEstimate === 0) return CAUSAL_QUALIFIER;
  const what =
    checkDelta !== 0 && Math.sign(checkDelta) !== Math.sign(modelEstimate)
      ? "the opposite effect"
      : `${Math.abs(checkDelta) < Math.abs(modelEstimate) ? "less" : "more"} ${modelEstimate < 0 ? "cooling" : "warming"}`;
  return `${CAUSAL_LEAD} finds ${what} — see ${CAUSAL_QUALIFIER_LINK}`;
}

/** True for a causal qualifier line (any wording of causalQualifier). */
export function isCausalQualifier(q: string): boolean {
  return q.startsWith(CAUSAL_LEAD) && q.endsWith(`— see ${CAUSAL_QUALIFIER_LINK}`);
}

/**
 * The answer card's headline, without numbers (the big number and the likely-range strip carry them):
 * "Cools the places you changed", "Warms the whole city", "May cool the places you changed" (the likely
 * range crosses zero), "Leaves the city unchanged".
 */
export function answerHeadlineText(l: Pick<Likely, "estimate" | "se" | "lo" | "hi">, subject: string): string {
  const est = l.estimate;
  if (!Number.isFinite(est)) return `No estimate for ${subject}`;
  if (est === 0) return `Leaves ${subject} unchanged`;
  const word = confidenceWord(l);
  if (word === "Could be zero") return `${est < 0 ? "May cool" : "May warm"} ${subject}`;
  return `${est < 0 ? "Cools" : "Warms"} ${subject}`;
}

export type PlainInput = {
  edited: Likely | null;
  city: Likely;
  unit: string;
  fracExtrapolatedEdited: number | null;
  causalDisagrees?: boolean;
  /** The causal check's estimate and the model's, same quantity (words the disagreement's direction). */
  causal?: { delta: number; estimate: number } | null;
  previewOnly?: boolean;
  buys?: string[];
};

export type PlainWording = { headline: string; city: string; confidence: ConfidenceWord; qualifiers: string[]; buys: string[] };

/** The plain-language card's text for a result. */
export function plainWording(p: PlainInput): PlainWording {
  const main = p.edited ?? p.city;
  const subject = p.edited ? "the edited area" : "the city";
  const qualifiers: string[] = [];
  const ex = extrapolationQualifier(p.fracExtrapolatedEdited);
  if (ex) qualifiers.push(ex);
  if (p.causalDisagrees) qualifiers.push(p.causal ? causalQualifier(p.causal.delta, p.causal.estimate) : CAUSAL_QUALIFIER);
  if (p.previewOnly) qualifiers.push("preview only — not verified");
  return {
    headline: headline(main, p.unit, subject),
    city: p.edited ? cityLine(p.city, p.unit) : "",
    confidence: confidenceWord(main),
    qualifiers,
    buys: p.buys ?? [],
  };
}

/**
 * Cooling per 1,000 cost units in words. `cooling_per_cost` follows the server (stats.cost_table:
 * −Σ delta / total cost, positive = cooler), so a negative value is warming bought per cost.
 */
export function perCostText(coolingPerCost: number, unit: string, per = "1,000 cost units"): string {
  const v = coolingPerCost * 1000;
  const warming = v < 0 && Number(Math.abs(v).toFixed(3)) !== 0;
  return `${fmtNum(Math.abs(v), 3)} ${unitLabel(unit)}·cells of ${warming ? "warming" : "cooling"} per ${per}`;
}

/** "1,200 fewer" / "300 more" (rounded residents). */
const fewerMore = (n: number) => `${fmtInt(Math.abs(Math.round(n)))} ${n < 0 ? "more" : "fewer"}`;

/**
 * "What it buys" lines from a result's impacts and cost (SPEC §7.7), in plain words:
 * - residents (unless `residents: false`): "2,880 fewer residents feel an air temperature of 90 °F or
 *   more today; 2,880 fewer by mid-century (moderate emissions)." The measure is the air temperature
 *   where residents live (impacts.exposure), never the heat index of the answer card's "Extreme
 *   caution" fact: the card leaves this line out when it shows that fact, so one claim has one number.
 * - the climate offset: "Cancels out about 31% of the warming expected by 2041–2060 (moderate
 *   emissions)." (the pathway code stays out: words.futureLabel's plain words);
 * - the cooling per 1,000 cost units.
 */
export function buysLines(r: Pick<Result, "cost" | "impacts">, unit: string, opts: { residents?: boolean } = {}): string[] {
  const u = unitLabel(unit);
  const out: string[] = [];
  const imp: Impacts | null = r.impacts;
  if (imp) {
    const t = imp.thresholds.find((x) => Math.abs(x - 90) < 1e-9) ?? imp.thresholds[0];
    if (t !== undefined && opts.residents !== false) {
      const key = Object.keys(imp.exposure[0]?.people_ge ?? {}).find((k) => Number(k) === t) ?? String(t);
      const moved = (caseName: (c: string) => boolean) => {
        const base = imp.exposure.find((e) => caseName(e.case) && !e.adapted);
        const adapted = imp.exposure.find((e) => caseName(e.case) && e.adapted);
        if (!base || !adapted) return null;
        const a = base.people_ge[key];
        const b = adapted.people_ge[key];
        return Number.isFinite(a) && Number.isFinite(b) ? a - b : null;
      };
      const today = moved((c) => /today|present|obs/i.test(c));
      const midCase = imp.exposure.find((e) => /2041|mid/i.test(e.case) && /245|2-?4\.?5/.test(e.case))?.case ?? null;
      const mid = midCase ? moved((c) => c === midCase) : null;
      const pathway = midCase ? casePathway(midCase) : null;
      const midWho = pathway ? emissionsWord(pathway) : null;
      if (today !== null) {
        out.push(
          `${fewerMore(today)} residents feel an air temperature of ${fmtNum(t, 0)} ${u} or more today${mid !== null ? `; ${fewerMore(mid)} by mid-century${midWho ? ` (${midWho})` : ""}` : ""}.`,
        );
      }
    }
    const off = imp.climate_offset.find((o) => /245/.test(pathwayCode(o.experiment)) && /2041/.test(o.period)) ?? imp.climate_offset[0];
    if (off && Number.isFinite(off.offset_share)) {
      const who = emissionsWord(off.experiment);
      const share = fmtPct(Math.abs(off.offset_share));
      out.push(
        off.offset_share >= 0
          ? `Cancels out about ${share} of the warming expected by ${periodWord(off.period)}${who ? ` (${who})` : ""}.`
          : `Adds about ${share} to the warming expected by ${periodWord(off.period)}${who ? ` (${who})` : ""}.`,
      );
    }
  }
  if (r.cost.cooling_per_cost !== null && Number.isFinite(r.cost.cooling_per_cost)) {
    out.push(`${perCostText(r.cost.cooling_per_cost, unit)}.`);
  }
  return out;
}

/**
 * The server's own "what it buys" lines (shown until the impacts are in) in plain words: the climate
 * offset loses its pathway code ("Offsets 31% of the city's SSP2-4.5 2041-2060 median warming" →
 * "Cancels out about 31% of the warming expected by 2041–2060 (moderate emissions)."); the cost and
 * residents lines are left out (the card's facts carry the cost, and counts the residents itself).
 */
export function plainServerBuys(lines: readonly string[]): string[] {
  const out: string[] = [];
  for (const b of lines) {
    if (/cost|residents/i.test(b)) continue;
    const m = /^offsets (-?[\d.]+%) of the city's (.+?) median warming\.?$/i.exec(b.trim());
    if (m) {
      const years = /(\d{4})\s*[-–]\s*(\d{4})/.exec(m[2]);
      const pathway = casePathway(m[2]);
      const who = pathway ? emissionsWord(pathway) : null;
      out.push(`Cancels out about ${m[1].replace(/^-/, "")} of the warming expected ${years ? `by ${years[1]}–${years[2]}` : "by mid-century"}${who ? ` (${who})` : ""}.`);
      continue;
    }
    if (/ssp|rcp/i.test(b)) continue;
    out.push(b);
  }
  return out;
}

/**
 * The caption of an equity group's concentration index.  The index divides by the mean cooling, so a cooling
 * and a warming of the same shape have the same index: for a scenario that warms on net, positive means the
 * warming (not the benefit) concentrates in the higher quintiles.  `residentMeanCooling` (positive = cooler)
 * names the direction; without it (impacts cached before it) the caption speaks of "the change".
 */
export function concentrationCaption(ci: number | null | undefined, residentMeanCooling: number | null | undefined, group: string): string {
  const g = group.replace(/_/g, " ");
  if (ci === null || ci === undefined || !Number.isFinite(ci)) return `Concentration index not computed for ${g}.`;
  const mu = residentMeanCooling;
  const head = `Concentration index ${fmtSigned(ci, 3)}`;
  if (mu === 0) return `${head}: the scenario makes no net change to share out across the quintiles of ${g}.`;
  const what = mu === null || mu === undefined || !Number.isFinite(mu) ? "the change (cooling or warming)" : mu > 0 ? "the cooling" : "the warming";
  return `${head} (positive: ${what} concentrates in the higher quintiles of ${g}; negative: in the lower ones).`;
}
