// US National Weather Service heat index (Rothfusz regression with the NWS low- and high-humidity
// adjustments), the same formula as sparc.core.heat.heat_index_f, for live previews in the browser:
// the Lab counts residents a design moves out of the NWS categories while the user edits.

/** Category lower bounds (°F): Caution, Extreme caution, Danger, Extreme danger. */
export const HEAT_EDGES_F = [80, 90, 103, 125] as const;

export function isFahrenheit(units: string | null | undefined): boolean {
  const u = (units ?? "").toLowerCase().replace("°", "").replace("deg", "").trim();
  return u === "f" || u === "fahrenheit" || u === "degf";
}

/** Relative humidity (%) from air temperature and dewpoint (°C), Magnus formula. */
export function rhFromDewpoint(tC: number, tdC: number): number {
  const es = (t: number) => 6.112 * Math.exp((17.62 * t) / (243.12 + t));
  return Math.min(100, Math.max(0, (100 * es(tdC)) / es(tC)));
}

/** NWS heat index (°F) for air temperature (°F) and relative humidity (%). */
export function heatIndexF(t: number, r: number): number {
  const simple = 0.5 * (t + 61 + (t - 68) * 1.2 + r * 0.094);
  if ((simple + t) / 2 < 80) return simple;
  let hi =
    -42.379 + 2.04901523 * t + 10.14333127 * r - 0.22475541 * t * r - 6.83783e-3 * t * t - 5.481717e-2 * r * r + 1.22874e-3 * t * t * r + 8.5282e-4 * t * r * r - 1.99e-6 * t * t * r * r;
  if (r < 13 && t >= 80 && t <= 112) hi -= ((13 - r) / 4) * Math.sqrt(Math.max(0, (17 - Math.abs(t - 95)) / 17));
  else if (r > 85 && t >= 80 && t <= 87) hi += ((r - 85) / 10) * ((87 - t) / 5);
  return hi;
}

/** 0 = below caution … 4 = extreme danger. */
export function heatCategory(hiF: number): number {
  let k = 0;
  for (const e of HEAT_EDGES_F) if (hiF >= e) k++;
  return k;
}

/** Heat index (°F) of a temperature in the target's units at a given dewpoint (°C). */
export function heatIndexAt(temp: number, units: string, dewpointC: number): number {
  const tF = isFahrenheit(units) ? temp : (temp * 9) / 5 + 32;
  const tC = ((tF - 32) * 5) / 9;
  return heatIndexF(tF, rhFromDewpoint(tC, dewpointC));
}

export type HeatShift = {
  /** Residents (or cells, without people) at Extreme caution or worse, before and after. */
  ecBefore: number;
  ecAfter: number;
  dangerBefore: number;
  dangerAfter: number;
  /** Change of the person-weighted (or cell-mean) heat index, °F. */
  meanHiChange: number;
  total: number;
  measure: "people" | "cells";
};

/** Residents moved across the NWS categories by a per-cell change `delta` (target units, negative = cooler). */
export function heatShift(obs: ArrayLike<number>, delta: ArrayLike<number>, people: ArrayLike<number> | null, units: string, dewpointC: number): HeatShift {
  const n = Math.min(obs.length, delta.length);
  let ecB = 0;
  let ecA = 0;
  let dgB = 0;
  let dgA = 0;
  let w = 0;
  let sB = 0;
  let sA = 0;
  for (let i = 0; i < n; i++) {
    const t = obs[i];
    if (!Number.isFinite(t)) continue;
    const d = Number.isFinite(delta[i]) ? delta[i] : 0;
    const p = people ? (Number.isFinite(people[i]) ? people[i] : 0) : 1;
    if (p <= 0) continue;
    const b = heatIndexAt(t, units, dewpointC);
    const a = d === 0 ? b : heatIndexAt(t + d, units, dewpointC);
    if (b >= HEAT_EDGES_F[1]) ecB += p;
    if (a >= HEAT_EDGES_F[1]) ecA += p;
    if (b >= HEAT_EDGES_F[2]) dgB += p;
    if (a >= HEAT_EDGES_F[2]) dgA += p;
    w += p;
    sB += p * b;
    sA += p * a;
  }
  return { ecBefore: ecB, ecAfter: ecA, dangerBefore: dgB, dangerAfter: dgA, meanHiChange: w > 0 ? (sA - sB) / w : 0, total: w, measure: people ? "people" : "cells" };
}
