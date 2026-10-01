// Brush (SPEC §12.5): paint per-lever edits with a radius of 60/120/250/500 m. Edits are
// Float32 arrays aligned with run rows, clamped against the exact Float32 inputs, and a stroke
// OVERWRITES the cells it touches (edit = clamp(base + amount) − base), so repainting the
// same cells never accumulates. Erase sets the touched cells back to 0.
import { fmtInt } from "../../theme/format";
import { rowsInRadius, type GridData } from "../grid";
import type { MapTool, RasterPoint, ToolPreview, ToolResult } from "./types";

export const BRUSH_RADII_M = [60, 120, 250, 500] as const;

export type BrushSettings = {
  /** Lever change per painted cell, in lever units. */
  amount: number;
  mode: "add" | "erase";
  radius_m: number;
  /** Exact current lever values per row (Float32 from the layer endpoint). */
  base: Float32Array;
  /** Allowed absolute range of the lever, e.g. [0, 100] for canopy %. */
  bounds: [number, number];
};

/**
 * Apply one dab to `edit` in place: for each row, edit = fround(clamp(base + amount) − base)
 * (or 0 when erasing). Rows with a non-finite base are left unedited. Returns rows changed.
 */
export function applyBrush(edit: Float32Array, rows: number[], s: BrushSettings): number[] {
  const changed: number[] = [];
  const [lo, hi] = s.bounds;
  for (const r of rows) {
    const b = s.base[r];
    let next: number;
    if (s.mode === "erase" || !Number.isFinite(b)) next = 0;
    else {
      const target = Math.min(hi, Math.max(lo, b + s.amount));
      next = Math.fround(Math.fround(target) - b);
      // Float32 rounding must never push base + edit outside the bounds.
      if (Math.fround(b + next) > hi) next = Math.fround(next - (Math.fround(b + next) - hi));
      if (Math.fround(b + next) < lo) next = Math.fround(next + (lo - Math.fround(b + next)));
    }
    if (edit[r] !== next) {
      edit[r] = next;
      changed.push(r);
    }
  }
  return changed;
}

/** Rows with a non-zero edit. */
export function editedRows(edit: Float32Array): number {
  let c = 0;
  for (let i = 0; i < edit.length; i++) if (edit[i] !== 0) c++;
  return c;
}

export function brushTool(g: GridData, edit: Float32Array, settings: () => BrushSettings, onChange?: (rows: number[]) => void): MapTool {
  let painting = false;
  let touched = new Set<number>();
  let last: RasterPoint | null = null;
  const dab = (p: RasterPoint) => {
    const s = settings();
    const rows = rowsInRadius(g, p.px, p.py, s.radius_m);
    const changed = applyBrush(edit, rows, s);
    for (const r of rows) touched.add(r);
    if (changed.length) onChange?.(changed);
  };
  const stroke = (from: RasterPoint, to: RasterPoint) => {
    // Interpolate dabs so fast strokes leave no gaps (step = half the radius).
    const step = Math.max(0.5, settings().radius_m / g.meta.dx_m / 2);
    const d = Math.hypot(to.px - from.px, to.py - from.py);
    const n = Math.max(1, Math.ceil(d / step));
    for (let i = 1; i <= n; i++) dab({ px: from.px + ((to.px - from.px) * i) / n, py: from.py + ((to.py - from.py) * i) / n });
  };
  return {
    id: "brush",
    pans: false,
    down(p) {
      painting = true;
      touched = new Set();
      last = p;
      dab(p);
      return null;
    },
    move(p, buttons) {
      if (!painting || !(buttons & 1)) return null;
      if (last) stroke(last, p);
      last = p;
      return null;
    },
    up(): ToolResult | null {
      if (!painting) return null;
      painting = false;
      last = null;
      const rows = [...touched];
      const s = settings();
      return { kind: "edit", rows, label: `${s.mode === "erase" ? "Erased" : "Painted"} ${fmtInt(rows.length)} cells · ${fmtInt(editedRows(edit))} edited` };
    },
    finish: () => null,
    cancel() {
      painting = false;
      last = null;
    },
    preview(hover): ToolPreview | null {
      if (!hover) return null;
      const s = settings();
      return { kind: "brush", cx: hover.px, cy: hover.py, r: s.radius_m / g.meta.dx_m, erase: s.mode === "erase" };
    },
  };
}
