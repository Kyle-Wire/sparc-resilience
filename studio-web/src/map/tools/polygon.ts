// Polygon selection: click to add vertices; double-click, Enter or clicking the first vertex
// closes it. Produces {kind: "polygon"} with one closed ring.
import { fmtInt } from "../../theme/format";
import type { GridData } from "../grid";
import { countMask, crsOf, maskWhere, pointInPolygon, toCrs } from "./shapes";
import type { MapTool, RasterPoint, ToolPreview, ToolResult } from "./types";

export function polygonSelection(g: GridData, pts: RasterPoint[]): ToolResult | null {
  if (pts.length < 3) return null;
  const ring = pts.map((p) => toCrs(g, p));
  ring.push(ring[0]);
  const mask = maskWhere(g, (px, py) => pointInPolygon(px, py, pts));
  const crs = crsOf(g);
  return { kind: "selection", spec: { kind: "polygon", crs, rings: [ring] }, mask, label: `Polygon · ${fmtInt(countMask(mask))} cells`, portable: crs === "EPSG:4326" };
}

export function polygonTool(g: GridData): MapTool {
  let pts: RasterPoint[] = [];
  const close = () => {
    const r = polygonSelection(g, pts);
    pts = [];
    return r;
  };
  return {
    id: "polygon",
    pans: false,
    down(p) {
      if (pts.length >= 3 && Math.hypot(p.px - pts[0].px, p.py - pts[0].py) < 1) return close();
      const last = pts[pts.length - 1];
      if (!last || Math.hypot(p.px - last.px, p.py - last.py) > 0.25) pts.push(p);
      return null;
    },
    move: () => null,
    up: () => null,
    finish: close,
    cancel() {
      pts = [];
    },
    preview(hover): ToolPreview | null {
      if (!pts.length) return null;
      return { kind: "polygon", points: hover ? [...pts, hover] : pts, closed: false };
    },
  };
}
