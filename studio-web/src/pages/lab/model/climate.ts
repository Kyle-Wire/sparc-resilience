// Client-side climate maps (SPEC §7.11): future temperature = observed + the selected warming
// statistic + the adaptation's ΔT, and the ≥ T exceedance map, computed in the browser so
// switching period, statistic, threshold or adaptation is instant. Warming values are in the
// target unit (core summarize_projections multiplies delta_K by to_units).
import type { ClimateExplore, ClimateStatistic, PresentClimate, WarmingRow } from "../../../api/lab";

/**
 * Today's observed exposure of an explore reply: `present` (api.md, the summarize_projections
 * dict) or, as built, `present_today` next to `present: true`.
 */
export function presentToday(data: Pick<ClimateExplore, "present" | "present_today"> | null | undefined): PresentClimate | null {
  if (!data) return null;
  if (data.present && typeof data.present === "object") return data.present;
  return data.present_today ?? null;
}

/** The share of a `{"90.0": 0.3}`-style record at threshold T (keys are numbers as text). */
export function shareAt(rec: Record<string, number> | null | undefined, t: number): number | null {
  if (!rec) return null;
  const k = Object.keys(rec).find((x) => Number(x) === t);
  const v = k === undefined ? undefined : rec[k];
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

/** The warming of one experiment × period under a statistic (median / p10 / p90 / one model). */
export function warmingValue(row: Pick<WarmingRow, "median" | "p10" | "p90" | "by_model">, stat: ClimateStatistic): number | null {
  if (typeof stat === "object") {
    const v = row.by_model[stat.model];
    return typeof v === "number" && Number.isFinite(v) ? v : null;
  }
  const v = row[stat];
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

/**
 * Future temperature per row: obs + warming + Δ (Δ in target units, negative = cooler; NaN
 * where the observation is missing). A missing adaptation Δ for a row counts as no change.
 */
export function futureTemperature(obs: ArrayLike<number>, warming: number, adaptation?: ArrayLike<number> | null): Float32Array {
  const out = new Float32Array(obs.length);
  for (let i = 0; i < obs.length; i++) {
    const d = adaptation ? adaptation[i] : 0;
    out[i] = obs[i] + warming + (Number.isFinite(d) ? d : 0);
  }
  return out;
}

/** 1 where the temperature is at or above T, 0 below, 255 (no data) where it is missing. */
export function exceedance(temps: ArrayLike<number>, threshold: number): Uint8Array {
  const out = new Uint8Array(temps.length);
  for (let i = 0; i < temps.length; i++) {
    const t = temps[i];
    out[i] = Number.isFinite(t) ? (t >= threshold ? 1 : 0) : 255;
  }
  return out;
}

/** Share of rows (with data) at or above T. */
export function shareAtOrAbove(temps: ArrayLike<number>, threshold: number): number | null {
  let n = 0;
  let k = 0;
  for (let i = 0; i < temps.length; i++) {
    const t = temps[i];
    if (!Number.isFinite(t)) continue;
    n++;
    if (t >= threshold) k++;
  }
  return n ? k / n : null;
}

export function statisticLabel(stat: ClimateStatistic): string {
  if (typeof stat === "object") return `model ${stat.model}`;
  return stat === "median" ? "median of models" : stat === "p10" ? "10th percentile of models" : "90th percentile of models";
}

/** URL form of a statistic: "median" | "p10" | "p90" | "model:<name>". */
export function encodeStatistic(stat: ClimateStatistic): string {
  return typeof stat === "object" ? `model:${stat.model}` : stat;
}

export function decodeStatistic(raw: string | null): ClimateStatistic {
  if (raw === "p10" || raw === "p90" || raw === "median") return raw;
  if (raw && raw.startsWith("model:") && raw.length > 6) return { model: raw.slice(6) };
  return "median";
}
