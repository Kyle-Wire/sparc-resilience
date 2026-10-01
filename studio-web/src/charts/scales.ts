// Scales and ticks for the SVG chart kit (no d3). Linear and log scales map a domain to a
// pixel range; ticks are "nice" 1/2/5 × 10^k steps; band scales place categories.
import { fmtNum, MINUS } from "../theme/format";

export type Scale = {
  (v: number): number;
  kind: "linear" | "log";
  domain: [number, number];
  range: [number, number];
  invert: (px: number) => number;
  ticks: (count?: number) => number[];
};

const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

/** The 1/2/5 × 10^k step giving about `count` intervals over [lo, hi]. */
export function tickStep(lo: number, hi: number, count = 5): number {
  const span = Math.abs(hi - lo);
  if (!(span > 0) || !Number.isFinite(span)) return 1;
  const raw = span / Math.max(1, count);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const err = raw / mag;
  const mult = err >= 7.07 ? 10 : err >= 3.16 ? 5 : err >= 1.41 ? 2 : 1;
  return mult * mag;
}

/** Decimals needed to print multiples of `step` exactly. */
export function stepDecimals(step: number): number {
  if (!(step > 0)) return 0;
  return Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
}

function roundTo(v: number, step: number): number {
  const d = stepDecimals(step);
  return Number(v.toFixed(Math.min(20, d + 2)));
}

/** Nice ticks covering [lo, hi] (inclusive, within the domain). */
export function niceTicks(lo: number, hi: number, count = 5): number[] {
  if (!isNum(lo) || !isNum(hi)) return [];
  if (lo === hi) return [lo];
  const [a, b] = lo < hi ? [lo, hi] : [hi, lo];
  const step = tickStep(a, b, count);
  const start = Math.ceil(a / step - 1e-9);
  const end = Math.floor(b / step + 1e-9);
  const out: number[] = [];
  for (let i = start; i <= end; i++) out.push(roundTo(i * step, step) || 0);
  return lo < hi ? out : out.reverse();
}

/** Extend [lo, hi] outward to tick multiples. */
export function niceDomain(lo: number, hi: number, count = 5): [number, number] {
  if (!isNum(lo) || !isNum(hi)) return [0, 1];
  if (lo === hi) {
    const d = lo === 0 ? 1 : Math.abs(lo) * 0.1;
    return [lo - d, hi + d];
  }
  const step = tickStep(lo, hi, count);
  return [roundTo(Math.floor(lo / step + 1e-9) * step, step) || 0, roundTo(Math.ceil(hi / step - 1e-9) * step, step) || 0];
}

/** Powers of ten in [lo, hi]; with fewer than three decades, 2× and 5× too. */
export function logTicks(lo: number, hi: number): number[] {
  if (!(lo > 0) || !(hi > 0)) return [];
  const [a, b] = lo < hi ? [lo, hi] : [hi, lo];
  const k0 = Math.floor(Math.log10(a));
  const k1 = Math.ceil(Math.log10(b));
  const mults = k1 - k0 <= 2 ? [1, 2, 5] : [1];
  const out: number[] = [];
  for (let k = k0; k <= k1; k++)
    for (const m of mults) {
      const v = Number((m * Math.pow(10, k)).toPrecision(12));
      if (v >= a * (1 - 1e-9) && v <= b * (1 + 1e-9)) out.push(v);
    }
  return out;
}

export function linearScale(domain: [number, number], range: [number, number], opts: { clamp?: boolean } = {}): Scale {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const k = d1 === d0 ? 0 : (r1 - r0) / (d1 - d0);
  const f = ((v: number) => {
    let x = r0 + (v - d0) * k;
    if (opts.clamp) x = Math.min(Math.max(x, Math.min(r0, r1)), Math.max(r0, r1));
    return x;
  }) as Scale;
  f.kind = "linear";
  f.domain = [d0, d1];
  f.range = [r0, r1];
  f.invert = (px: number) => (k === 0 ? d0 : d0 + (px - r0) / k);
  f.ticks = (count = 5) => niceTicks(d0, d1, count);
  return f;
}

export function logScale(domain: [number, number], range: [number, number]): Scale {
  const [d0, d1] = domain.map((d) => Math.max(d, 1e-12)) as [number, number];
  const l0 = Math.log10(d0);
  const l1 = Math.log10(d1);
  const [r0, r1] = range;
  const k = l1 === l0 ? 0 : (r1 - r0) / (l1 - l0);
  const f = ((v: number) => r0 + (Math.log10(Math.max(v, 1e-12)) - l0) * k) as Scale;
  f.kind = "log";
  f.domain = [d0, d1];
  f.range = [r0, r1];
  f.invert = (px: number) => Math.pow(10, k === 0 ? l0 : l0 + (px - r0) / k);
  f.ticks = () => logTicks(d0, d1);
  return f;
}

export type BandScale = {
  (key: string): number;
  bandwidth: number;
  step: number;
  keys: string[];
  /** Category under a pixel, or null. */
  at: (px: number) => string | null;
};

/** Evenly spaced bands; `padding` is the inner gap as a fraction of the step. */
export function bandScale(keys: string[], range: [number, number], padding = 0.2, outer = 0.1): BandScale {
  const [r0, r1] = range;
  const n = Math.max(1, keys.length);
  const step = (r1 - r0) / (n + 2 * outer - padding);
  const bw = step * (1 - padding);
  const start = r0 + step * outer;
  const index = new Map(keys.map((k, i) => [k, i]));
  const f = ((key: string) => start + (index.get(key) ?? 0) * step) as BandScale;
  f.bandwidth = bw;
  f.step = step;
  f.keys = keys;
  f.at = (px: number) => {
    const i = Math.floor((px - start) / step);
    if (i < 0 || i >= keys.length) return null;
    return px - (start + i * step) <= bw ? keys[i] : null;
  };
  return f;
}

/** [min, max] of the finite values, or null. */
export function extent(values: Iterable<number | null | undefined>): [number, number] | null {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of values) {
    if (!isNum(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return lo <= hi ? [lo, hi] : null;
}

/** Pad a domain by a fraction of its span (and include `include` values, e.g. 0). */
export function padDomain(d: [number, number] | null, frac = 0.05, include: number[] = []): [number, number] {
  let [lo, hi] = d ?? [0, 1];
  for (const v of include) {
    lo = Math.min(lo, v);
    hi = Math.max(hi, v);
  }
  if (lo === hi) return [lo - (Math.abs(lo) || 1) * 0.1, hi + (Math.abs(hi) || 1) * 0.1];
  const p = (hi - lo) * frac;
  return [lo - (include.includes(lo) ? 0 : p), hi + (include.includes(hi) ? 0 : p)];
}

/** Tick label with just enough decimals for the step and a Unicode minus. */
export function formatTick(v: number, step?: number, signed = false): string {
  const d = step ? stepDecimals(step) : Math.abs(v) >= 100 ? 0 : Math.abs(v) >= 1 ? 1 : 2;
  const s = fmtNum(v, Math.min(d, 6));
  if (signed && v > 0) return "+" + s;
  return s;
}

/** Tick label for log axes: 100, 1,000, 2,000, 0.5. */
export function formatLogTick(v: number): string {
  if (v >= 1) return fmtNum(v, 0);
  return fmtNum(v, Math.min(6, Math.ceil(-Math.log10(v))));
}

const DURATION_STEPS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 10800, 21600, 43200, 86400];

/** Ticks for an elapsed-time axis in seconds (1 s … 1 d steps). */
export function durationTicks(lo: number, hi: number, count = 6): { ticks: number[]; step: number } {
  const span = Math.max(1e-9, hi - lo);
  const step = DURATION_STEPS.find((s) => span / s <= count) ?? Math.ceil(span / count / 86400) * 86400;
  const out: number[] = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) out.push(t);
  return { ticks: out, step };
}

/** "0", "30 s", "5 min", "1 h 30", "2 h" for elapsed-time ticks. */
export function formatDurationTick(t: number, step: number): string {
  const neg = t < 0;
  const s = Math.abs(t);
  let out: string;
  if (s === 0) out = "0";
  else if (step < 60) out = `${Math.round(s)} s`;
  else if (step < 3600) out = `${Math.round(s / 60)} min`;
  else {
    const h = Math.floor(s / 3600);
    const m = Math.round((s - h * 3600) / 60);
    out = m ? `${h} h ${m}` : `${h} h`;
  }
  return neg ? MINUS + out : out;
}
