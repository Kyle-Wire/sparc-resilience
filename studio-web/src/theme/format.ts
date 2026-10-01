// Number, unit, duration and size formatting (SPEC §1.2 principle 9, §12.6).
// Negative numbers always use the Unicode minus (U+2212), never "-". Missing values are "—".

export const MINUS = "−";
export const DASH = "–"; // en dash for ranges
export const EMPTY = "—";

const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

/** Replace ASCII hyphen-minus signs in a formatted number with U+2212. */
export function unicodeMinus(s: string): string {
  return s.replace(/^-|(?<=[\s(])-(?=\d)/g, MINUS);
}

const nfCache = new Map<string, Intl.NumberFormat>();
function nf(min: number, max: number, grouping = true): Intl.NumberFormat {
  const k = `${min}:${max}:${grouping}`;
  let f = nfCache.get(k);
  if (!f) {
    f = new Intl.NumberFormat("en-US", { minimumFractionDigits: min, maximumFractionDigits: max, useGrouping: grouping });
    nfCache.set(k, f);
  }
  return f;
}

/** Fixed decimals with thousands separators and U+2212: fmtNum(-1234.5, 1) → "−1,234.5". */
export function fmtNum(v: number | null | undefined, decimals = 2): string {
  if (!isNum(v)) return EMPTY;
  const d = Math.max(0, Math.min(20, decimals));
  const s = nf(d, d).format(v);
  // "-0.00" (a tiny negative rounded to zero) prints as zero.
  if (/^-0(\.0+)?$/.test(s)) return s.slice(1);
  return unicodeMinus(s);
}

/** Integer with separators: 54701 → "54,701". */
export function fmtInt(v: number | null | undefined): string {
  return fmtNum(isNum(v) ? Math.round(v) : v, 0);
}

/** Explicit sign: +0.41, −0.41, 0.00. */
export function fmtSigned(v: number | null | undefined, decimals = 2): string {
  if (!isNum(v)) return EMPTY;
  const body = fmtNum(Math.abs(v), decimals);
  const zero = Number(Math.abs(v).toFixed(decimals)) === 0;
  return zero ? body : (v > 0 ? "+" : MINUS) + body;
}

/** Up to `sig` significant digits, no trailing zeros (axis ticks, compact numbers). */
export function fmtSig(v: number | null | undefined, sig = 3): string {
  if (!isNum(v)) return EMPTY;
  if (v === 0) return "0";
  const mag = Math.floor(Math.log10(Math.abs(v)));
  const decimals = Math.max(0, sig - 1 - mag);
  return fmtNum(v, Math.min(decimals, 10)).replace(/(\.\d*?)0+$/, "$1").replace(/\.$/, "");
}

const UNIT_LABELS: Record<string, string> = {
  degF: "°F",
  degC: "°C",
  K: "K",
  pct: "%",
  percent: "%",
  pp: "pp",
  km2: "km²",
  m2: "m²",
  "m^2": "m²",
  "km^2": "km²",
  "degF*cells": "°F·cells",
  "degF·cells": "°F·cells",
};

/** Display form of a unit string ("degF" → "°F"); unknown units pass through. */
export function unitLabel(unit: string | null | undefined): string {
  if (!unit) return "";
  return UNIT_LABELS[unit] ?? unit;
}

/** A number with its unit: "0.41 °F", "12%", "3 pp". */
export function fmtValue(v: number | null | undefined, unit?: string | null, decimals = 2): string {
  if (!isNum(v)) return EMPTY;
  const u = unitLabel(unit);
  const n = fmtNum(v, decimals);
  if (!u) return n;
  return u === "%" ? n + "%" : `${n} ${u}`;
}

/** Signed number with its unit: "−0.41 °F". */
export function fmtSignedValue(v: number | null | undefined, unit?: string | null, decimals = 2): string {
  if (!isNum(v)) return EMPTY;
  const u = unitLabel(unit);
  const n = fmtSigned(v, decimals);
  if (!u) return n;
  return u === "%" ? n + "%" : `${n} ${u}`;
}

/** "cooler" for negative temperature changes, "warmer" for positive (SPEC §6.1 sign rules). */
export function coolerWarmer(delta: number | null | undefined, decimals = 2): "cooler" | "warmer" | "no change" | "" {
  if (!isNum(delta)) return "";
  if (Number(Math.abs(delta).toFixed(decimals)) === 0) return "no change";
  return delta < 0 ? "cooler" : "warmer";
}

/** A temperature change in words: −0.41 → "0.41 °F cooler", 0.2 → "0.20 °F warmer". */
export function fmtTempChange(delta: number | null | undefined, unit?: string | null, decimals = 2): string {
  if (!isNum(delta)) return EMPTY;
  const w = coolerWarmer(delta, decimals);
  if (w === "no change") return "no change";
  return `${fmtValue(Math.abs(delta), unit, decimals)} ${w}`;
}

/** Fraction as a percentage: 0.123 → "12%"; decimals apply to the percentage. */
export function fmtPct(frac: number | null | undefined, decimals = 0): string {
  if (!isNum(frac)) return EMPTY;
  return fmtNum(frac * 100, decimals) + "%";
}

/** A range with an en dash, or "to" when a bound is negative: "0.30–0.52 °F", "−0.52 to −0.30 °F". */
export function fmtRange(lo: number | null | undefined, hi: number | null | undefined, unit?: string | null, decimals = 2): string {
  if (!isNum(lo) || !isNum(hi)) return EMPTY;
  const u = unitLabel(unit);
  const sep = lo < 0 || hi < 0 ? " to " : DASH;
  const body = `${fmtNum(lo, decimals)}${sep}${fmtNum(hi, decimals)}`;
  if (!u) return body;
  return u === "%" ? body + "%" : `${body} ${u}`;
}

// ---------------------------------------------------------------- durations

/**
 * Human duration: 0.3 s, 12 s, 4 min 12 s, 26 min, 1 h 40 m, 2 d 3 h.
 * Below 10 min the seconds are kept; from 10 min they are rounded away.
 */
export function fmtDuration(seconds: number | null | undefined): string {
  if (!isNum(seconds)) return EMPTY;
  const neg = seconds < 0;
  let s = Math.abs(seconds);
  let out: string;
  if (s < 10) out = `${Number.isInteger(s) ? s : s.toFixed(1)} s`;
  else if (s < 59.5) out = `${Math.round(s)} s`;
  else if (s < 600) {
    s = Math.round(s);
    const m = Math.floor(s / 60);
    const r = s - m * 60;
    out = r ? `${m} min ${r} s` : `${m} min`;
  } else if (s < 3570) out = `${Math.round(s / 60)} min`;
  else if (s < 48 * 3600) {
    let h = Math.floor(s / 3600);
    let m = Math.round((s - h * 3600) / 60);
    if (m === 60) {
      h += 1;
      m = 0;
    }
    out = m ? `${h} h ${m} m` : `${h} h`;
  } else {
    let d = Math.floor(s / 86400);
    let h = Math.round((s - d * 86400) / 3600);
    if (h === 24) {
      d += 1;
      h = 0;
    }
    out = h ? `${d} d ${h} h` : `${d} d`;
  }
  return neg ? MINUS + out : out;
}

/** "4–6 min" when both ends are whole minutes under an hour, else "1 h 40 m–1 h 55 m". */
export function fmtDurationRange(lo: number | null | undefined, hi: number | null | undefined): string {
  if (!isNum(lo) || !isNum(hi)) return EMPTY;
  if (lo >= 59.5 && hi < 3570 && (lo >= 600 || Math.round(lo) % 60 === 0) && (hi >= 600 || Math.round(hi) % 60 === 0)) {
    return `${Math.round(lo / 60)}${DASH}${Math.round(hi / 60)} min`;
  }
  if (lo < 59.5 && hi < 59.5) return `${Math.round(lo)}${DASH}${Math.round(hi)} s`;
  return `${fmtDuration(lo)}${DASH}${fmtDuration(hi)}`;
}

/** ETA text for job chips: "≈12 min (10–15 min)". */
export function fmtEta(eta: number | null | undefined, lo?: number | null, hi?: number | null): string {
  if (!isNum(eta)) return isNum(lo) && isNum(hi) ? `≈${fmtDurationRange(lo, hi)}` : EMPTY;
  const head = `≈${fmtDuration(eta)}`;
  return isNum(lo) && isNum(hi) && hi > lo ? `${head} (${fmtDurationRange(lo, hi)})` : head;
}

/** Clock time for a duration since start: 75 → "1:15", 3725 → "1:02:05". */
export function fmtClock(seconds: number | null | undefined): string {
  if (!isNum(seconds)) return EMPTY;
  const s = Math.max(0, Math.floor(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  const pad = (x: number) => String(x).padStart(2, "0");
  return h ? `${h}:${pad(m)}:${pad(r)}` : `${m}:${pad(r)}`;
}

// ---------------------------------------------------------------- sizes

/** Decimal byte sizes (kB = 1000 B): 218804 → "219 kB", 525e6 → "525 MB", 1.2e9 → "1.2 GB". */
export function fmtBytes(bytes: number | null | undefined): string {
  if (!isNum(bytes)) return EMPTY;
  const neg = bytes < 0;
  let b = Math.abs(bytes);
  const units = ["B", "kB", "MB", "GB", "TB", "PB"];
  let i = 0;
  while (b >= 999.5 && i < units.length - 1) {
    b /= 1000;
    i++;
  }
  const txt = i === 0 ? `${Math.round(b)} B` : `${b < 9.95 ? b.toFixed(1) : Math.round(b).toString()} ${units[i]}`;
  return neg ? MINUS + txt : txt;
}

/** Progress of a transfer: "312/525 MB" (both in the larger value's unit). */
export function fmtBytesOf(done: number | null | undefined, total: number | null | undefined): string {
  if (!isNum(done) || !isNum(total)) return fmtBytes(done ?? total);
  const t = fmtBytes(total);
  const unit = t.split(" ")[1] ?? "B";
  const scale = { B: 1, kB: 1e3, MB: 1e6, GB: 1e9, TB: 1e12, PB: 1e15 }[unit] ?? 1;
  const d = done / scale;
  const dTxt = unit === "B" ? String(Math.round(d)) : d < 9.95 && total / scale < 9.95 ? d.toFixed(1) : String(Math.round(d));
  return `${dTxt}/${t}`;
}

// ---------------------------------------------------------------- dates

function toDate(v: string | number | Date | null | undefined): Date | null {
  if (v === null || v === undefined || v === "") return null;
  const d = v instanceof Date ? v : typeof v === "number" ? new Date(v * 1000) : new Date(v);
  return Number.isNaN(d.getTime()) ? null : d;
}

/** "2026-10-01 14:22" in local time (ISO strings or unix seconds). */
export function fmtDateTime(v: string | number | Date | null | undefined): string {
  const d = toDate(v);
  if (!d) return EMPTY;
  const p = (x: number) => String(x).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}

export function fmtDate(v: string | number | Date | null | undefined): string {
  const d = toDate(v);
  if (!d) return EMPTY;
  const p = (x: number) => String(x).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`;
}

/** "just now", "3 min ago", "2 h ago", "5 d ago" (falls back to the date after 30 days). */
export function fmtRelative(v: string | number | Date | null | undefined, now: number = Date.now()): string {
  const d = toDate(v);
  if (!d) return EMPTY;
  const s = (now - d.getTime()) / 1000;
  if (s < 0) return fmtDateTime(d);
  if (s < 45) return "just now";
  if (s < 3600) return `${Math.max(1, Math.round(s / 60))} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  if (s < 30 * 86400) return `${Math.round(s / 86400)} d ago`;
  return fmtDate(d);
}

/** "1 cell", "1,284 cells". */
export function fmtCount(n: number | null | undefined, noun: string, plural?: string): string {
  if (!isNum(n)) return EMPTY;
  return `${fmtInt(n)} ${n === 1 ? noun : plural ?? noun + "s"}`;
}
