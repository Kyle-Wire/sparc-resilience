// Plain-language result wording (SPEC §7.7): the headline with its likely range, the
// confidence line from the 95% interval (est ± 1.96·SE), the qualifiers and "what it buys".
// - "Confident it cools" iff hi < 0; "Confident it warms" iff lo > 0; "Could be zero" when
//   the interval spans 0.
// - "partly outside observed conditions (23% of edited cells)" when more than 20% of the
//   EDITED cells are extrapolated; "independent causal check disagrees"; "preview only — not
//   verified".
import type { Impacts, Result } from "../../../api/lab";
import type { Likely } from "../../../api/types";
import { fmtInt, fmtNum, fmtPct, fmtSigned, fmtValue, unitLabel } from "../../../theme/format";

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

export function extrapolationQualifier(fracEdited: number | null | undefined): string | null {
  if (fracEdited === null || fracEdited === undefined || !Number.isFinite(fracEdited) || fracEdited <= EXTRAPOLATION_QUALIFIER_AT) return null;
  return `partly outside observed conditions (${fmtPct(fracEdited)} of edited cells)`;
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

export type PlainInput = {
  edited: Likely | null;
  city: Likely;
  unit: string;
  fracExtrapolatedEdited: number | null;
  causalDisagrees?: boolean;
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
  if (p.causalDisagrees) qualifiers.push("independent causal check disagrees");
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

/** "What it buys" lines from a result's impacts and cost (SPEC §7.7). */
export function buysLines(r: Pick<Result, "cost" | "impacts">, unit: string): string[] {
  const u = unitLabel(unit);
  const out: string[] = [];
  const imp: Impacts | null = r.impacts;
  if (imp) {
    const t = imp.thresholds.find((x) => Math.abs(x - 90) < 1e-9) ?? imp.thresholds[0];
    if (t !== undefined) {
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
      const mid = moved((c) => /2041|mid/i.test(c) && /245/.test(c));
      if (today !== null) out.push(`${fmtInt(Math.round(today))} residents moved below ${fmtNum(t, 0)} ${u} today${mid !== null ? `, ${fmtInt(Math.round(mid))} by mid-century (SSP2-4.5)` : ""}.`);
    }
    const off = imp.climate_offset.find((o) => /245/.test(o.experiment) && /2041/.test(o.period)) ?? imp.climate_offset[0];
    if (off && Number.isFinite(off.offset_share)) out.push(`Offsets ${fmtPct(off.offset_share)} of ${off.experiment.toUpperCase()} ${off.period} median warming.`);
  }
  if (r.cost.cooling_per_cost !== null && Number.isFinite(r.cost.cooling_per_cost)) {
    out.push(`${perCostText(r.cost.cooling_per_cost, unit)}.`);
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
