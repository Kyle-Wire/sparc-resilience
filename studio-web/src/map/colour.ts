// Raster colouring for GridCanvas (SPEC §12.5). One pixel per 30 m cell, north up:
// cellToPt[(ny−1−iy)·nx + ix] = row (−1 = empty). colorize() writes packed RGBA words into a
// Uint32Array view of an nx×ny ImageData in about a millisecond for 54,701 cells.
import { getLuts, heatColors, hexToRgb255, packRgba, themeColors } from "../theme/palette";
import type { Domain } from "./domain";

/** Raster index → run row (−1 where there is no observation). */
export function buildCellToPt(ix: ArrayLike<number>, iy: ArrayLike<number>, nx: number, ny: number): Int32Array {
  const out = new Int32Array(nx * ny).fill(-1);
  for (let r = 0; r < ix.length; r++) {
    const x = ix[r];
    const y = iy[r];
    if (x < 0 || y < 0 || x >= nx || y >= ny) continue;
    out[(ny - 1 - y) * nx + x] = r;
  }
  return out;
}

/** Run row → raster index (the inverse map, used by the colouring loop). */
export function buildRowToPix(ix: ArrayLike<number>, iy: ArrayLike<number>, nx: number, ny: number): Int32Array {
  const out = new Int32Array(ix.length);
  for (let r = 0; r < ix.length; r++) {
    const x = ix[r];
    const y = iy[r];
    out[r] = x < 0 || y < 0 || x >= nx || y >= ny ? -1 : (ny - 1 - y) * nx + x;
  }
  return out;
}

export type Palette = {
  /** 256 packed RGBA words for the ramp in use. */
  seq32: Uint32Array;
  div32: Uint32Array;
  /** Categorical colours for classes 1..3 (class 0 = grey). */
  cat: [number, number, number];
  gray: number;
  nodata: number;
  /** The five NWS heat-index category colours (cat layers with `palette: "heat"`). */
  heat: number[];
};

const paletteCache = new Map<boolean, Palette>();

/** Packed colours for a theme (LUTs rebuilt per theme, SPEC §12.7). */
export function mapPalette(dark: boolean): Palette {
  let p = paletteCache.get(dark);
  if (!p) {
    const l = getLuts(dark);
    const c = themeColors(dark);
    const pk = (hex: string) => {
      const [r, g, b] = hexToRgb255(hex);
      return packRgba(r, g, b);
    };
    p = { seq32: l.seq32, div32: l.div32, cat: [pk(c.s1), pk(c.s2), pk(c.s3)], gray: pk(c.gray), nodata: pk(c.nodata), heat: heatColors(dark).map(pk) };
    paletteCache.set(dark, p);
  }
  return p;
}

export type ColorizeOptions = {
  /** 0/1 per row: unselected rows are dimmed to `dimAlpha`. */
  selection?: Uint8Array | null;
  dimAlpha?: number;
};

/**
 * Colour every row into `out` (nx·ny packed RGBA, little-endian 0xAABBGGRR):
 * NaN → transparent; zero_blank values ≤ 0 → --nodata; numeric layers → ramp index
 * round(t·255); categorical → grey for 0, s1..s3 for 1..3 (more classes spread over the
 * sequential ramp; the "heat" palette uses the NWS category colours), out-of-range classes (e.g. 255 "not in the fold design") → --nodata.
 */
export function colorize(out: Uint32Array, rowToPix: Int32Array, values: ArrayLike<number>, d: Domain, pal: Palette, opts: ColorizeOptions = {}): void {
  out.fill(0);
  const n = Math.min(rowToPix.length, values.length);
  const sel = opts.selection ?? null;
  const dim = (opts.dimAlpha ?? 70) << 24;
  if (d.kind === "cat") {
    const nCat = d.nCat;
    const heat = d.palette === "heat" && nCat <= pal.heat.length;
    const many = nCat > 4 && !heat;
    for (let r = 0; r < n; r++) {
      const p = rowToPix[r];
      if (p < 0) continue;
      const k = values[r];
      let c: number;
      if (!(k >= 0) || k >= nCat || k !== Math.floor(k)) c = Number.isNaN(k) ? 0 : pal.nodata;
      else if (heat) c = pal.heat[k];
      else if (many) c = pal.seq32[Math.round((k / (nCat - 1)) * 255)];
      else c = k === 0 ? pal.gray : pal.cat[k - 1];
      if (c && sel && !sel[r]) c = ((c & 0x00ffffff) | dim) >>> 0;
      out[p] = c;
    }
    return;
  }
  const mult = d.mult;
  const lo = d.lo;
  const hi = d.hi;
  const zb = d.zeroBlank;
  if (d.kind === "div" && d.center !== null) {
    const c0 = d.center;
    const kLo = 127.5 / (c0 - lo || 1e-12);
    const kHi = 127.5 / (hi - c0 || 1e-12);
    const lut = pal.div32;
    for (let r = 0; r < n; r++) {
      const p = rowToPix[r];
      if (p < 0) continue;
      const raw = values[r];
      if (raw !== raw) continue; // NaN: transparent
      let c: number;
      if (zb && raw <= 0) c = pal.nodata;
      else {
        const v = raw * mult;
        let i = v < c0 ? (v - lo) * kLo : 127.5 + (v - c0) * kHi;
        i = i < 0 ? 0 : i > 255 ? 255 : i;
        c = lut[Math.round(i)];
      }
      if (sel && !sel[r]) c = ((c & 0x00ffffff) | dim) >>> 0;
      out[p] = c;
    }
    return;
  }
  const k = 255 / (hi - lo || 1e-12);
  const lut = pal.seq32;
  for (let r = 0; r < n; r++) {
    const p = rowToPix[r];
    if (p < 0) continue;
    const raw = values[r];
    if (raw !== raw) continue;
    let c: number;
    if (zb && raw <= 0) c = pal.nodata;
    else {
      let i = (raw * mult - lo) * k;
      i = i < 0 ? 0 : i > 255 ? 255 : i;
      c = lut[Math.round(i)];
    }
    if (sel && !sel[r]) c = ((c & 0x00ffffff) | dim) >>> 0;
    out[p] = c;
  }
}

/** 0/255 alpha raster of the rows flagged in `mask` (hatching, selection outlines). */
export function maskRaster(out: Uint8ClampedArray, rowToPix: Int32Array, mask: ArrayLike<number>): void {
  out.fill(0);
  for (let r = 0; r < rowToPix.length && r < mask.length; r++) {
    const p = rowToPix[r];
    if (p >= 0 && mask[r]) out[p * 4 + 3] = 255;
  }
}

/**
 * Raster-space boundary segments between cells whose class differs (zone outlines, the
 * selection outline). Returns an SVG path in raster units (1 unit = 1 cell).
 */
export function outlinePath(cellToPt: Int32Array, nx: number, ny: number, classOf: (row: number) => number | string | null): string {
  const parts: string[] = [];
  const cls = (px: number, py: number) => {
    if (px < 0 || py < 0 || px >= nx || py >= ny) return null;
    const r = cellToPt[py * nx + px];
    return r < 0 ? null : classOf(r);
  };
  for (let py = 0; py < ny; py++)
    for (let px = 0; px < nx; px++) {
      const c = cls(px, py);
      if (c === null) continue;
      if (cls(px + 1, py) !== c) parts.push(`M${px + 1},${py}v1`);
      if (cls(px - 1, py) !== c) parts.push(`M${px},${py}v1`);
      if (cls(px, py + 1) !== c) parts.push(`M${px},${py + 1}h1`);
      if (cls(px, py - 1) !== c) parts.push(`M${px},${py}h1`);
    }
  return parts.join("");
}
