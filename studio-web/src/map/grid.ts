// Grid geometry on the client (SPEC §6.3, §12.5). Coordinate systems:
// - row r: run row order (predictions.parquet / data.ids);
// - cell (ix, iy): lattice indices, iy grows north; cell centre in metres is
//   (x0_m + ix·dx_m, y0_m + iy·dx_m) (sparc.core.grid.Grid);
// - raster (px, py): continuous, 1 unit = 1 cell, north up; cell (ix, iy) covers
//   px ∈ [ix, ix+1), py ∈ [ny−1−iy, ny−iy);
// - screen: CSS px in the map viewport = (tx + px·scale, ty + py·scale).
import type { GridMeta } from "../api/types";
import { unpack, type OffsetEntry } from "../api/binary";
import { buildCellToPt, buildRowToPix } from "./colour";

export type GridData = {
  meta: GridMeta;
  n: number;
  nx: number;
  ny: number;
  ix: Int32Array;
  iy: Int32Array;
  lon: Float32Array;
  lat: Float32Array;
  zone: Int16Array;
  cellToPt: Int32Array;
  rowToPix: Int32Array;
};

/** Build GridData from the packed grid.bin body and its X-SPARC-Offsets. */
export function parseGridBin(meta: GridMeta, buffer: ArrayBuffer, offsets: OffsetEntry[]): GridData {
  const a = unpack(buffer, offsets);
  const ix = a.ix as Int32Array;
  const iy = a.iy as Int32Array;
  if (!ix || !iy) throw new Error("grid.bin is missing ix/iy");
  const n = ix.length;
  const lon = (a.lon as Float32Array) ?? new Float32Array(n).fill(NaN);
  const lat = (a.lat as Float32Array) ?? new Float32Array(n).fill(NaN);
  const zone = (a.zone as Int16Array) ?? new Int16Array(n);
  return makeGrid(meta, ix, iy, lon, lat, zone);
}

export function makeGrid(meta: GridMeta, ix: Int32Array, iy: Int32Array, lon?: Float32Array, lat?: Float32Array, zone?: Int16Array): GridData {
  const n = ix.length;
  return {
    meta,
    n,
    nx: meta.nx,
    ny: meta.ny,
    ix,
    iy,
    lon: lon ?? new Float32Array(n).fill(NaN),
    lat: lat ?? new Float32Array(n).fill(NaN),
    zone: zone ?? new Int16Array(n),
    cellToPt: buildCellToPt(ix, iy, meta.nx, meta.ny),
    rowToPix: buildRowToPix(ix, iy, meta.nx, meta.ny),
  };
}

/** Row under a raster point, or −1. */
export function rowAt(g: GridData, px: number, py: number): number {
  const x = Math.floor(px);
  const y = Math.floor(py);
  if (x < 0 || y < 0 || x >= g.nx || y >= g.ny) return -1;
  return g.cellToPt[y * g.nx + x];
}

/** Raster centre of a row's cell. */
export function rowCenter(g: GridData, r: number): { px: number; py: number } {
  return { px: g.ix[r] + 0.5, py: g.ny - 1 - g.iy[r] + 0.5 };
}

/** Raster point → run frame metres. */
export function rasterToXY(g: GridData, px: number, py: number): [number, number] {
  const dx = g.meta.dx_m;
  return [g.meta.x0_m + (px - 0.5) * dx, g.meta.y0_m + (g.ny - 0.5 - py) * dx];
}

/** Run frame metres → raster point. */
export function xyToRaster(g: GridData, x: number, y: number): { px: number; py: number } {
  const dx = g.meta.dx_m;
  return { px: (x - g.meta.x0_m) / dx + 0.5, py: g.ny - 0.5 - (y - g.meta.y0_m) / dx };
}

/**
 * Raster point → [lon, lat] by bilinear interpolation between the grid's corner cell centres
 * (GridMeta.corners, [lat, lon] each). Null without a CRS.
 */
export function rasterToLonLat(g: GridData, px: number, py: number): [number, number] | null {
  const c = g.meta.corners;
  if (!c || !g.meta.has_lonlat) return null;
  const fx = g.nx > 1 ? (px - 0.5) / (g.nx - 1) : 0;
  const fy = g.ny > 1 ? 1 - (py - 0.5) / (g.ny - 1) : 0;
  const lat = (1 - fx) * (1 - fy) * c.sw[0] + fx * (1 - fy) * c.se[0] + (1 - fx) * fy * c.nw[0] + fx * fy * c.ne[0];
  const lon = (1 - fx) * (1 - fy) * c.sw[1] + fx * (1 - fy) * c.se[1] + (1 - fx) * fy * c.nw[1] + fx * fy * c.ne[1];
  return [lon, lat];
}

/** "41.8240°N, 71.4128°W" (hemisphere-aware). */
export function fmtLonLat(ll: [number, number] | null): string {
  if (!ll) return "";
  const [lon, lat] = ll;
  return `${Math.abs(lat).toFixed(4)}°${lat >= 0 ? "N" : "S"}, ${Math.abs(lon).toFixed(4)}°${lon >= 0 ? "E" : "W"}`;
}

/** Rows whose cell centre lies within `radius_m` of a raster point. */
export function rowsInRadius(g: GridData, px: number, py: number, radius_m: number): number[] {
  const rc = radius_m / g.meta.dx_m;
  const out: number[] = [];
  const x0 = Math.max(0, Math.floor(px - rc - 1));
  const x1 = Math.min(g.nx - 1, Math.ceil(px + rc + 1));
  const y0 = Math.max(0, Math.floor(py - rc - 1));
  const y1 = Math.min(g.ny - 1, Math.ceil(py + rc + 1));
  const r2 = rc * rc;
  for (let y = y0; y <= y1; y++)
    for (let x = x0; x <= x1; x++) {
      const r = g.cellToPt[y * g.nx + x];
      if (r < 0) continue;
      const ddx = x + 0.5 - px;
      const ddy = y + 0.5 - py;
      if (ddx * ddx + ddy * ddy <= r2) out.push(r);
    }
  return out;
}

/** Nice scale-bar length (1/2/5 × 10^k m) not longer than `maxM`. */
export function niceLength(maxM: number): number {
  if (!(maxM > 0)) return 0;
  const k = Math.pow(10, Math.floor(Math.log10(maxM)));
  for (const m of [5, 2, 1]) if (m * k <= maxM) return m * k;
  return k;
}

export function fmtMeters(m: number): string {
  return m >= 1000 ? `${Number((m / 1000).toPrecision(3))} km` : `${Math.round(m)} m`;
}

/** Python-compatible round-half-to-even (numpy.round). */
function roundHalfEven(v: number): number {
  const r = Math.round(v);
  return Math.abs(v % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/**
 * Pointy-top hex key of a point in run metres, identical to sparc.core.planner.hex_ids
 * (size = centre-to-centre spacing; key = (q + 100000)·1e6 + (r + 100000)).
 */
export function hexKey(x: number, y: number, size_m: number): number {
  const R = size_m / Math.sqrt(3);
  const q = ((Math.sqrt(3) / 3) * x - y / 3) / R;
  const r = ((2 / 3) * y) / R;
  const cx = q;
  const cz = r;
  const cy = -cx - cz;
  let rx = roundHalfEven(cx);
  let ry = roundHalfEven(cy);
  let rz = roundHalfEven(cz);
  const dx = Math.abs(rx - cx);
  const dy = Math.abs(ry - cy);
  const dz = Math.abs(rz - cz);
  const fixX = dx > dy && dx > dz;
  const fixY = !fixX && dy > dz;
  if (fixX) rx = -ry - rz;
  if (fixY) ry = -rx - rz;
  if (!fixX && !fixY) rz = -rx - ry;
  return (rx + 100000) * 1000000 + (rz + 100000);
}

/** Centre (metres) of a hex key, the inverse of hexKey. */
export function hexCenter(key: number, size_m: number): [number, number] {
  const R = size_m / Math.sqrt(3);
  const hq = Math.floor(key / 1000000) - 100000;
  const hr = (key % 1000000) - 100000;
  return [R * Math.sqrt(3) * (hq + hr / 2), R * 1.5 * hr];
}
