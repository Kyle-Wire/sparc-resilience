// Test grids: a hand-made 3×3 and a 54,701-cell synthetic city.
import type { GridMeta } from "../api/types";
import { makeGrid, type GridData } from "../map/grid";

export function gridMeta(nx: number, ny: number, n: number, extra: Partial<GridMeta> = {}): GridMeta {
  return {
    n, nx, ny, dx_m: 30, x0_m: 1000, y0_m: 2000, crs: "EPSG:32619", coord_scale: 1, has_lonlat: true,
    bounds_lonlat: [-71.5, 41.7, -71.3, 41.9],
    corners: { sw: [41.7, -71.5], se: [41.7, -71.3], nw: [41.9, -71.5], ne: [41.9, -71.3] },
    ids_kind: "int", zones: [], n_folds: 5, units: { target: "degF" }, background: 88, etag: "e1", ...extra,
  };
}

/**
 * 3×3 lattice with 7 observed cells (rows 0..6); (ix=1, iy=1) and (ix=2, iy=2) are empty.
 *   iy=2 (north row): r0 r1 .
 *   iy=1:             r2 .  r3
 *   iy=0 (south row): r4 r5 r6
 */
export function grid3(): GridData {
  const ix = Int32Array.from([0, 1, 0, 2, 0, 1, 2]);
  const iy = Int32Array.from([2, 2, 1, 1, 0, 0, 0]);
  // zone holds indices into meta.zones: rows 0..2 are zone 1, rows 3..6 zone 2
  return makeGrid(gridMeta(3, 3, 7, { zones: [1, 2] }), ix, iy, undefined, undefined, Int16Array.from([0, 0, 0, 1, 1, 1, 1]));
}

/** A blobby synthetic city with exactly n observed cells on an nx×ny lattice. */
export function bigGrid(n = 54701, nx = 320, ny = 240): GridData {
  const ix = new Int32Array(n);
  const iy = new Int32Array(n);
  let r = 0;
  const cx = nx / 2;
  const cy = ny / 2;
  const cells: [number, number, number][] = [];
  for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) cells.push([x, y, ((x - cx) / cx) ** 2 + ((y - cy) / cy) ** 2 + 0.15 * Math.sin(x * 0.3) * Math.cos(y * 0.2)]);
  cells.sort((a, b) => a[2] - b[2]);
  for (; r < n; r++) {
    ix[r] = cells[r][0];
    iy[r] = cells[r][1];
  }
  return makeGrid(gridMeta(nx, ny, n), ix, iy);
}
