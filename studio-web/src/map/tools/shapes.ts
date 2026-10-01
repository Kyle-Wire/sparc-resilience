// Shared geometry for the selection tools: masks of rows inside a raster-space shape and the
// conversion of raster shapes to SelectionSpec coordinates (lon/lat when the grid has a CRS,
// else run metres, flagged non-portable).
import type { SelectionCrs } from "../../api/types";
import { rasterToLonLat, rasterToXY, type GridData } from "../grid";
import type { RasterPoint } from "./types";

export function crsOf(g: GridData): SelectionCrs {
  return g.meta.has_lonlat && g.meta.corners ? "EPSG:4326" : "run_xy_m";
}

/** A raster point in the selection CRS: [lon, lat] or [x_m, y_m]. */
export function toCrs(g: GridData, p: RasterPoint): [number, number] {
  if (crsOf(g) === "EPSG:4326") {
    const ll = rasterToLonLat(g, p.px, p.py);
    if (ll) return [Number(ll[0].toFixed(7)), Number(ll[1].toFixed(7))];
  }
  const [x, y] = rasterToXY(g, p.px, p.py);
  return [Number(x.toFixed(3)), Number(y.toFixed(3))];
}

/** Mask (0/1 per row) of cells whose centre satisfies `inside(px, py)`. */
export function maskWhere(g: GridData, inside: (px: number, py: number) => boolean): Uint8Array {
  const m = new Uint8Array(g.n);
  for (let r = 0; r < g.n; r++) {
    const px = g.ix[r] + 0.5;
    const py = g.ny - 1 - g.iy[r] + 0.5;
    if (inside(px, py)) m[r] = 1;
  }
  return m;
}

/** Even–odd point-in-polygon test (raster units). */
export function pointInPolygon(px: number, py: number, poly: RasterPoint[]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const a = poly[i];
    const b = poly[j];
    if (a.py > py !== b.py > py && px < ((b.px - a.px) * (py - a.py)) / (b.py - a.py || 1e-12) + a.px) inside = !inside;
  }
  return inside;
}

export function countMask(m: Uint8Array): number {
  let c = 0;
  for (let i = 0; i < m.length; i++) c += m[i];
  return c;
}
