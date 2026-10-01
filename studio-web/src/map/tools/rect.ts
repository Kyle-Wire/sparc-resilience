// Rectangle selection: drag a box; produces {kind: "rect"} in lon/lat (or run metres).
import { fmtInt } from "../../theme/format";
import type { GridData } from "../grid";
import { countMask, crsOf, maskWhere, toCrs } from "./shapes";
import type { MapTool, RasterPoint, ToolPreview, ToolResult } from "./types";

export function rectSelection(g: GridData, a: RasterPoint, b: RasterPoint): ToolResult {
  const x0 = Math.min(a.px, b.px);
  const x1 = Math.max(a.px, b.px);
  const y0 = Math.min(a.py, b.py);
  const y1 = Math.max(a.py, b.py);
  const corners = [toCrs(g, { px: x0, py: y1 }), toCrs(g, { px: x1, py: y0 }), toCrs(g, { px: x0, py: y0 }), toCrs(g, { px: x1, py: y1 })];
  const min: [number, number] = [Math.min(...corners.map((c) => c[0])), Math.min(...corners.map((c) => c[1]))];
  const max: [number, number] = [Math.max(...corners.map((c) => c[0])), Math.max(...corners.map((c) => c[1]))];
  const mask = maskWhere(g, (px, py) => px >= x0 && px <= x1 && py >= y0 && py <= y1);
  const crs = crsOf(g);
  return { kind: "selection", spec: { kind: "rect", crs, min, max }, mask, label: `Rectangle · ${fmtInt(countMask(mask))} cells`, portable: crs === "EPSG:4326" };
}

export function rectTool(g: GridData): MapTool {
  let start: RasterPoint | null = null;
  let cur: RasterPoint | null = null;
  return {
    id: "rect",
    pans: false,
    down(p) {
      start = p;
      cur = p;
      return null;
    },
    move(p) {
      if (start) cur = p;
      return null;
    },
    up(p) {
      if (!start) return null;
      const a = start;
      start = null;
      cur = null;
      if (Math.abs(p.px - a.px) < 0.5 && Math.abs(p.py - a.py) < 0.5) return null;
      return rectSelection(g, a, p);
    },
    finish: () => null,
    cancel() {
      start = null;
      cur = null;
    },
    preview(): ToolPreview | null {
      return start && cur ? { kind: "rect", x0: start.px, y0: start.py, x1: cur.px, y1: cur.py } : null;
    },
  };
}
