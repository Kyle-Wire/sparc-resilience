// Colour domains for map layers (SPEC §6.3 "Domain rules"):
// - clip to the 2nd–98th percentiles;
// - diverging scales are symmetric about the centre (city median for temperatures, 0 for Δ);
// - `zero_blank` paints values ≤ 0 as --nodata, and its domain starts at 0;
// - `mult` applies display scaling (domains are in display units: value × mult);
// - the user can lock the scale across layers, or type a manual range.
import type { LayerMeta } from "../api/types";

export type Domain = {
  kind: "seq" | "div" | "cat";
  /** Display units (raw × mult). */
  lo: number;
  hi: number;
  /** Diverging centre in display units; null for seq/cat. */
  center: number | null;
  mult: number;
  zeroBlank: boolean;
  /** Category count for cat layers. */
  nCat: number;
  /** A named colour set for cat layers ("heat": the NWS heat-index categories). */
  palette?: "heat" | null;
};

/** Linear-interpolated quantiles of the finite values (optionally only v > 0). */
export function quantiles(values: ArrayLike<number>, qs: number[], positiveOnly = false): (number | null)[] {
  const arr: number[] = [];
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (Number.isFinite(v) && (!positiveOnly || v > 0)) arr.push(v);
  }
  if (!arr.length) return qs.map(() => null);
  const sorted = Float64Array.from(arr).sort();
  return qs.map((q) => {
    const pos = Math.min(1, Math.max(0, q)) * (sorted.length - 1);
    const i = Math.floor(pos);
    const f = pos - i;
    return i + 1 < sorted.length ? sorted[i] + (sorted[i + 1] - sorted[i]) * f : sorted[i];
  });
}

export type DomainOptions = {
  /** Manual [lo, hi] in display units. */
  manual?: [number, number] | null;
  /** A locked domain to reuse (same kind only). */
  lock?: Domain | null;
};

/** Domain of a layer from its stats (or, when stats lack percentiles, from its values). */
export function computeDomain(meta: LayerMeta, values?: ArrayLike<number> | null, opts: DomainOptions = {}): Domain {
  const mult = meta.mult || 1;
  const zeroBlank = !!meta.zero_blank;
  if (meta.scale === "cat") {
    const nCat = Math.max(2, meta.labels?.length ?? Math.round((meta.stats.hi ?? 3) + 1));
    return { kind: "cat", lo: 0, hi: nCat - 1, center: null, mult: 1, zeroBlank: false, nCat, palette: meta.palette ?? null };
  }
  if (opts.lock && opts.lock.kind === meta.scale) return { ...opts.lock, mult, zeroBlank };

  let p2 = meta.stats.p2;
  let p98 = meta.stats.p98;
  if (zeroBlank && values) {
    // The stats include the zeros of untreated cells; clip on the painted (positive) cells.
    [p2, p98] = quantiles(values, [0.02, 0.98], true);
  } else if ((p2 === null || p98 === null) && values) {
    [p2, p98] = quantiles(values, [0.02, 0.98]);
  }
  if (p2 === null || p98 === null) {
    p2 = meta.stats.lo ?? 0;
    p98 = meta.stats.hi ?? 1;
  }
  let lo = p2 * mult;
  let hi = p98 * mult;
  if (lo > hi) [lo, hi] = [hi, lo];

  if (meta.scale === "div") {
    const c = (meta.center ?? 0) * mult;
    if (opts.manual) return { kind: "div", lo: Math.min(opts.manual[0], c), hi: Math.max(opts.manual[1], c), center: c, mult, zeroBlank, nCat: 0 };
    const span = Math.max(Math.abs(lo - c), Math.abs(hi - c)) || 1e-9;
    return { kind: "div", lo: c - span, hi: c + span, center: c, mult, zeroBlank, nCat: 0 };
  }
  if (opts.manual) {
    const [a, b] = opts.manual;
    return { kind: "seq", lo: Math.min(a, b), hi: Math.max(a, b) > Math.min(a, b) ? Math.max(a, b) : Math.min(a, b) + 1e-9, center: null, mult, zeroBlank, nCat: 0 };
  }
  if (zeroBlank) lo = 0;
  if (!(hi > lo)) hi = lo + 1e-9;
  return { kind: "seq", lo, hi, center: null, mult, zeroBlank, nCat: 0 };
}

/** Position on the ramp in [0, 1] for a display-unit value (diverging: centre → 0.5). */
export function valueToT(v: number, d: Domain): number {
  let t: number;
  if (d.kind === "div" && d.center !== null) {
    const c = d.center;
    t = v < c ? (0.5 * (v - d.lo)) / (c - d.lo || 1e-12) : 0.5 + (0.5 * (v - c)) / (d.hi - c || 1e-12);
  } else t = (v - d.lo) / (d.hi - d.lo || 1e-12);
  return t < 0 ? 0 : t > 1 ? 1 : t;
}

/** Inverse of valueToT (legend ticks, histogram colouring). */
export function tToValue(t: number, d: Domain): number {
  if (d.kind === "div" && d.center !== null) return t < 0.5 ? d.lo + (t / 0.5) * (d.center - d.lo) : d.center + ((t - 0.5) / 0.5) * (d.hi - d.center);
  return d.lo + t * (d.hi - d.lo);
}

/** Layer summary for the legend: n, mean, p10/median/p90 in display units (zero_blank skips ≤ 0). */
export function layerSummary(values: ArrayLike<number>, d: Domain): { n: number; mean: number | null; p10: number | null; p50: number | null; p90: number | null } {
  const arr: number[] = [];
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!Number.isFinite(v) || (d.zeroBlank && v <= 0)) continue;
    const x = v * d.mult;
    arr.push(x);
    sum += x;
  }
  if (!arr.length) return { n: 0, mean: null, p10: null, p50: null, p90: null };
  const [p10, p50, p90] = quantiles(arr, [0.1, 0.5, 0.9]);
  return { n: arr.length, mean: sum / arr.length, p10, p50, p90 };
}

/** Category shares for cat layers: count per class index (values ≥ nCat ignored). */
export function categoryCounts(values: ArrayLike<number>, nCat: number): number[] {
  const c = new Array<number>(nCat).fill(0);
  for (let i = 0; i < values.length; i++) {
    const k = values[i];
    if (Number.isInteger(k) && k >= 0 && k < nCat) c[k]++;
  }
  return c;
}
