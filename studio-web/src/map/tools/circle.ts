// Circle selection: press at the centre, drag out the radius; produces {kind: "circle"}.
import { fmtInt } from "../../theme/format";
import type { GridData } from "../grid";
import { countMask, crsOf, maskWhere, toCrs } from "./shapes";
import type { MapTool, RasterPoint, ToolPreview, ToolResult } from "./types";

export function circleSelection(g: GridData, c: RasterPoint, rRaster: number): ToolResult {
  const radius_m = Number((rRaster * g.meta.dx_m).toFixed(1));
  const mask = maskWhere(g, (px, py) => (px - c.px) ** 2 + (py - c.py) ** 2 <= rRaster * rRaster);
  const crs = crsOf(g);
  return {
    kind: "selection",
    spec: { kind: "circle", crs, center: toCrs(g, c), radius_m },
    mask,
    label: `Circle ${Math.round(radius_m)} m · ${fmtInt(countMask(mask))} cells`,
    portable: crs === "EPSG:4326",
  };
}

export function circleTool(g: GridData): MapTool {
  let center: RasterPoint | null = null;
  let r = 0;
  return {
    id: "circle",
    pans: false,
    down(p) {
      center = p;
      r = 0;
      return null;
    },
    move(p) {
      if (center) r = Math.hypot(p.px - center.px, p.py - center.py);
      return null;
    },
    up(p) {
      if (!center) return null;
      const c = center;
      const rr = Math.hypot(p.px - c.px, p.py - c.py);
      center = null;
      r = 0;
      return rr < 0.5 ? null : circleSelection(g, c, rr);
    },
    finish: () => null,
    cancel() {
      center = null;
      r = 0;
    },
    preview(): ToolPreview | null {
      return center ? { kind: "circle", cx: center.px, cy: center.py, r } : null;
    },
  };
}
