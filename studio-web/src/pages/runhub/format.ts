// View-model formatting for the run hub (SPEC §1.2 principle 9, §6.4): every number carries
// its unit and temperature changes read "cooler"/"warmer". Pure functions, unit-tested.
import type { CellCurve, GenericTable, MissingOutput, ViewKpi, ViewUnits } from "../../api/runs";
import type { Likely, OutputState } from "../../api/types";
import { hexKey, type GridData } from "../../map/grid";
import { EMPTY, fmtInt, fmtNum, fmtPct, fmtRange, fmtSigned, fmtSignedValue, fmtTempChange, fmtValue, unitLabel } from "../../theme/format";

const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

const TEMP_UNITS = new Set(["°F", "°C", "K"]);

/** Whether a unit is a temperature (changes in it are worded cooler/warmer). */
export function isTempUnit(unit: string | null | undefined): boolean {
  return TEMP_UNITS.has(unitLabel(unit));
}

/** Display unit of the target ("°F"). */
export function targetUnit(units: ViewUnits | null | undefined): string {
  return unitLabel(units?.target ?? "");
}

/**
 * A temperature change and its likely range in words:
 * "0.62 °F cooler (likely range 0.30–0.94 °F cooler)"; ranges spanning zero read
 * "likely range 0.10 °F cooler to 0.05 °F warmer".
 */
export function likelyText(l: Pick<Likely, "estimate" | "lo" | "hi"> | null | undefined, unit: string, decimals = 2): string {
  if (!l || !isNum(l.estimate)) return EMPTY;
  const head = isTempUnit(unit) ? fmtTempChange(l.estimate, unit, decimals) : fmtSignedValue(l.estimate, unit, decimals);
  if (!isNum(l.lo) || !isNum(l.hi)) return head;
  return `${head} (likely range ${rangeWords(l.lo, l.hi, unit, decimals)})`;
}

/** A range of changes in cooler/warmer words for temperatures, signed numbers otherwise. */
export function rangeWords(lo: number, hi: number, unit: string, decimals = 2): string {
  if (!isTempUnit(unit)) return fmtRange(lo, hi, unit, decimals);
  if (hi < 0) return `${fmtRange(-hi, -lo, unit, decimals)} cooler`;
  if (lo > 0) return `${fmtRange(lo, hi, unit, decimals)} warmer`;
  return `${fmtTempChange(lo, unit, decimals)} to ${fmtTempChange(hi, unit, decimals)}`;
}

/** Plain-language confidence of a likely range. */
export function confidenceWords(l: Pick<Likely, "lo" | "hi" | "confidence"> | null | undefined): string {
  if (!l) return "";
  if (l.confidence === "confident_cools") return "Confident it cools.";
  if (l.confidence === "confident_warms") return "Confident it warms.";
  if (l.confidence === "could_be_zero") return "Could be zero.";
  if (isNum(l.lo) && isNum(l.hi)) return l.hi < 0 ? "Confident it cools." : l.lo > 0 ? "Confident it warms." : "Could be zero.";
  return "No uncertainty estimate.";
}

export type KpiDisplay = { value: string; unit: string; note: string | null };

/** The tile text for a ViewModel KPI. */
export function kpiDisplay(k: ViewKpi, units?: ViewUnits | null): KpiDisplay {
  const unit = unitLabel(k.unit ?? (k.format === "delta" ? (units?.target ?? "") : ""));
  const d = k.decimals ?? 2;
  const notes: string[] = [];
  let value: string;
  let shownUnit = unit;
  if (typeof k.value === "string") value = k.value;
  else if (k.format === "percent") {
    value = fmtPct(k.value, k.decimals ?? 0);
    shownUnit = "";
  } else if (k.format === "int") value = fmtInt(k.value);
  else if (k.format === "delta") {
    if (isTempUnit(unit) && isNum(k.value)) {
      value = fmtTempChange(k.value, unit, d);
      shownUnit = "";
    } else value = fmtSigned(k.value, d);
  } else if (k.format === "signed") value = fmtSigned(k.value, d);
  else value = fmtNum(k.value, d);
  if (k.likely && isNum(k.likely.lo) && isNum(k.likely.hi)) notes.push(`likely range ${rangeWords(k.likely.lo, k.likely.hi, unit, d)}`);
  if (isNum(k.target ?? null)) {
    const t = k.target as number;
    const tv = k.format === "percent" ? fmtPct(t, k.decimals ?? 0) : fmtValue(t, unit, d);
    notes.push(`${k.target_label ?? "target"} ${tv}`);
  }
  if (k.band && isNum(k.band.lo) && isNum(k.band.hi)) notes.push(`${k.band.label} ${rangeWords(k.band.lo, k.band.hi, unit, d)}`);
  if (k.note) notes.push(k.note);
  return { value, unit: shownUnit, note: notes.length ? notes.join(" · ") : null };
}

/** "stage:S6" → "stage S6"; "post:planner" → "the planner post-run action"; "study:placebo" → "the placebo study". */
export function producedByText(p: string | null | undefined): string {
  if (!p) return "an unknown step";
  const [kind, name] = p.split(":");
  if (kind === "stage") return `stage ${name}`;
  if (kind === "post") return `the ${name} post-run action`;
  if (kind === "study") return `the ${name} study`;
  if (kind === "studio") return `Studio (${name})`;
  return p;
}

/** One sentence per missing output ("causal: produced by stage S6"). */
export function missingText(m: MissingOutput): string {
  return `${m.output} — produced by ${producedByText(m.produced_by)}`;
}

/** Status tone (StatusChip) for an output state. */
export function outputStateStatus(state: OutputState | string | null | undefined): string {
  switch (state) {
    case "present":
      return "done";
    case "writing":
      return "running";
    case "partial":
      return "partial";
    case "stale":
      return "stale";
    case "missing":
      return "missing";
    default:
      return "queued";
  }
}

export const OUTPUT_STATE_TEXT: Record<string, string> = {
  present: "present",
  stale: "stale",
  missing: "missing",
  writing: "being written",
  partial: "partly written",
};

/** A GenericTable as kit `ChartTable`/`Table` input (header labels with units). */
export function tableColumns(t: GenericTable): { key: string; label: string; unit?: string }[] {
  return t.columns.map((c) => ({ key: c.key, label: c.label, unit: c.unit ? unitLabel(c.unit) : undefined }));
}

/** A cell of a generic table: numbers keep 3 significant decimals and the Unicode minus. */
export function cellText(v: unknown, decimals = 3): string {
  if (v === null || v === undefined || v === "") return EMPTY;
  if (typeof v === "number") return Number.isInteger(v) ? fmtInt(v) : fmtNum(v, decimals);
  if (typeof v === "boolean") return v ? "yes" : "no";
  return String(v);
}

/** Pretty label of a snake_case or dotted key ("lambda_scores" → "Lambda scores"). */
export function humanize(key: string): string {
  const s = key.replace(/[_.]+/g, " ").trim();
  return s ? s[0].toUpperCase() + s.slice(1) : key;
}

/** Metres in the nicest unit: 315 m, 1.2 km. */
export function fmtDistance(m: number | null | undefined): string {
  if (!isNum(m)) return EMPTY;
  return Math.abs(m) >= 1000 ? `${fmtNum(m / 1000, 1)} km` : `${fmtInt(m)} m`;
}

// ---------------------------------------------------------------- response curves

/**
 * A cell's fitted dose–benefit curve. The server sends sampled `dose`/`benefit`; when it sends
 * only the parameters, the saturating form B(D) = A·(1 − exp(−D/d_s)) is rebuilt here over
 * [0, dmax] (core `response.fit_saturation`). Linear and sigmoid fits need their sampled
 * points (their slope and width are not part of the parameters).
 */
export function curvePoints(c: CellCurve, n = 25): { dose: number[]; benefit: number[] } {
  if (c.dose.length && c.dose.length === c.benefit.length) return { dose: c.dose, benefit: c.benefit };
  if (c.model === "saturating" && isNum(c.A) && isNum(c.ds) && c.ds > 0) {
    const top = isNum(c.dmax) && c.dmax > 0 ? c.dmax : isNum(c.d90) ? c.d90 * 1.5 : c.ds * 3;
    const dose = Array.from({ length: n }, (_, i) => (top * i) / (n - 1));
    return { dose, benefit: dose.map((d) => (c.A as number) * (1 - Math.exp(-d / (c.ds as number)))) };
  }
  return { dose: [], benefit: [] };
}

// ---------------------------------------------------------------- hex mode

/**
 * Hex choropleth: every row takes the mean of the finite values in its hexagon (pointy-top
 * hexagons of `size_m` across flats in the run frame, the planner's hex keys). Rows with no
 * value stay NaN.
 */
export function hexMeans(grid: GridData, values: ArrayLike<number>, size_m: number): Float32Array {
  const n = Math.min(grid.n, values.length);
  const keys = new Float64Array(n);
  const sum = new Map<number, number>();
  const cnt = new Map<number, number>();
  const { x0_m, y0_m, dx_m } = grid.meta;
  for (let r = 0; r < n; r++) {
    const k = hexKey(x0_m + grid.ix[r] * dx_m, y0_m + grid.iy[r] * dx_m, size_m);
    keys[r] = k;
    const v = values[r];
    if (!Number.isFinite(v)) continue;
    sum.set(k, (sum.get(k) ?? 0) + v);
    cnt.set(k, (cnt.get(k) ?? 0) + 1);
  }
  const out = new Float32Array(n);
  for (let r = 0; r < n; r++) {
    const c = cnt.get(keys[r]) ?? 0;
    out[r] = c ? (sum.get(keys[r]) as number) / c : NaN;
  }
  return out;
}

/** Rows whose (x, y) values fall inside a box (Relationships brush → selection mask). */
export function maskFromBox(x: ArrayLike<number>, y: ArrayLike<number>, box: { x: [number, number]; y: [number, number] }): Uint8Array {
  const n = Math.min(x.length, y.length);
  const out = new Uint8Array(n);
  const [x0, x1] = [Math.min(...box.x), Math.max(...box.x)];
  const [y0, y1] = [Math.min(...box.y), Math.max(...box.y)];
  for (let i = 0; i < n; i++) {
    const a = x[i];
    const b = y[i];
    if (a >= x0 && a <= x1 && b >= y0 && b <= y1) out[i] = 1;
  }
  return out;
}

/** Rank agreement chip text for a priority comparison. */
export function agreementWords(tau: number | null | undefined): string {
  if (!isNum(tau)) return EMPTY;
  const a = Math.abs(tau);
  const w = a >= 0.7 ? "strong" : a >= 0.4 ? "moderate" : a >= 0.2 ? "weak" : "little";
  return `${w} agreement (τ ${fmtSigned(tau, 2)})`;
}
