// Test helpers for the run hub: a RunDetail, a small grid with its packed grid.bin, a layer
// catalogue, and fetch-mock routes for a run (built on the foundation test kit).
import { pack } from "../../../api/binary";
import type { RunDetail } from "../../../api/runs";
import type { GridMeta, LayerGroup, LayerMeta, RunSummary } from "../../../api/types";
import type { MockHandler } from "../../../test/render";

export function runSummary(rid: string, extra: Partial<RunSummary> = {}): RunSummary {
  return {
    id: rid,
    project_id: "p_1",
    label: "Fast demo run",
    origin: "studio",
    status: "complete",
    mode: "fast",
    coarse_m: null,
    created_utc: "2026-10-01T21:21:18Z",
    finished_utc: "2026-10-01T21:21:49Z",
    duration_s: 31,
    n_points: 7,
    r2: 0.81,
    rmse: 0.34,
    coverage: 0.89,
    n_scenarios: 8,
    checkpoint_bytes: 1_200_000,
    has_emulator: false,
    studies: [],
    git_commit: "5a04f44",
    git_dirty: true,
    demo: true,
    pinned: false,
    parent_run_id: null,
    study_id: null,
    last_job_id: null,
    ...extra,
  };
}

export function runDetail(rid: string, extra: Partial<RunSummary> = {}): RunDetail {
  return {
    run: runSummary(rid, extra),
    header: {
      name: "synthetic_demo",
      created_utc: "2026-10-01T21:21:18Z",
      git_commit: "5a04f44",
      git_dirty: true,
      versions: { python: "3.11.15", numpy: "1.26.4" },
      n_points: 7,
      grid_shape: [3, 3],
      cell_m: 30,
      fast: true,
      coarse_m: null,
      run_dir: `/w/projects/demo/runs/${rid}`,
      demo: true,
    },
    state: null,
    launch: null,
    stages: [],
    checkpoint: {
      present: true,
      bytes: 1_200_000,
      done: ["S3", "S4", "S5"],
      saved_utc: "2026-10-01T21:21:40Z",
      fingerprint: "68c17c7b",
      matches_snapshot: { data: true, code: true, config: true },
      changed_sections: [],
      resumable: true,
      reuses: ["S1", "S2_S3"],
      saves_s: 18,
      reason: null,
    },
    outputs_summary: { present: 10, missing: 2, stale: 0, writing: 0 },
    sections: {},
    flags: [],
    children: [],
    jobs: [],
    warnings_count: 0,
  };
}

/** The 3×3 test lattice of src/test/grid.ts (7 observed cells, two zones). */
export function gridMeta(extra: Partial<GridMeta> = {}): GridMeta {
  return {
    n: 7,
    nx: 3,
    ny: 3,
    dx_m: 30,
    x0_m: 1000,
    y0_m: 2000,
    crs: "EPSG:32619",
    coord_scale: 1,
    has_lonlat: true,
    bounds_lonlat: [-71.5, 41.7, -71.3, 41.9],
    corners: { sw: [41.7, -71.5], se: [41.7, -71.3], nw: [41.9, -71.5], ne: [41.9, -71.3] },
    ids_kind: "int",
    zones: [1, 2],
    n_folds: 3,
    units: { target: "degF" },
    background: 88,
    etag: "g1",
    ...extra,
  };
}

export function gridBin(): { raw: ArrayBuffer; headers: Record<string, string> } {
  const { buffer, offsets } = pack([
    { name: "ix", dtype: "int32", data: Int32Array.from([0, 1, 0, 2, 0, 1, 2]) },
    { name: "iy", dtype: "int32", data: Int32Array.from([2, 2, 1, 1, 0, 0, 0]) },
    { name: "lon", dtype: "float32", data: Float32Array.from([-71.5, -71.4, -71.5, -71.3, -71.5, -71.4, -71.3]) },
    { name: "lat", dtype: "float32", data: Float32Array.from([41.9, 41.9, 41.8, 41.8, 41.7, 41.7, 41.7]) },
    { name: "zone", dtype: "int16", data: Int16Array.from([0, 0, 0, 1, 1, 1, 1]) }, // indices into GridMeta.zones
  ]);
  return { raw: buffer, headers: { "content-type": "application/octet-stream", "X-SPARC-Offsets": JSON.stringify(offsets) } };
}

const stats = (lo: number, hi: number) => ({ n: 7, lo, hi, mean: (lo + hi) / 2, p1: lo, p2: lo, p50: (lo + hi) / 2, p98: hi, p99: hi });

export function layerMeta(p: Partial<LayerMeta> & { key: string }): LayerMeta {
  return {
    group: "temperature",
    label: p.key,
    unit: "degF",
    scale: "seq",
    center: null,
    decimals: 1,
    mult: 1,
    zero_blank: false,
    labels: null,
    desc: "",
    sign_note: null,
    source: { file: "predictions.parquet", column: p.key },
    dtype: "float32",
    stats: stats(80, 86),
    ...p,
  };
}

export const LAYER_GROUPS: LayerGroup[] = [
  {
    id: "temperature",
    label: "Temperature",
    layers: [
      layerMeta({ key: "obs", label: "Observed", scale: "div", center: 83 }),
      layerMeta({ key: "resid", label: "Residual", scale: "div", center: 0, stats: stats(-1, 1), sign_note: "positive = model too cool" }),
      layerMeta({ key: "dist_train_m", label: "Distance to training data", unit: "m", stats: stats(0, 300) }),
    ],
  },
  { id: "scenarios", label: "Scenarios", layers: [layerMeta({ key: "sc:canopy-increase-plus-10", label: "Canopy Increase +10", scale: "div", center: 0, stats: stats(-1, 0) })] },
  {
    id: "effects",
    label: "Effects",
    layers: [layerMeta({ key: "fp_canopy", label: "Canopy footprint per pp", unit: "degF·cells", scale: "div", center: 0, stats: stats(-0.2, 0) })],
  },
];

export const LAYER_VALUES: Record<string, Float32Array> = {
  obs: Float32Array.from([80, 81, 82, 83, 84, 85, 86]),
  resid: Float32Array.from([-1, -0.5, 0, 0.2, 0.5, 1, NaN]),
  dist_train_m: Float32Array.from([0, 50, 100, 150, 200, 250, 300]),
  "sc:canopy-increase-plus-10": Float32Array.from([-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7]),
  fp_canopy: Float32Array.from([-0.01, -0.02, -0.05, -0.08, -0.1, -0.15, -0.2]),
};

export function binResponse(a: Float32Array | Uint8Array): { raw: ArrayBuffer; headers: Record<string, string> } {
  const copy = a.slice();
  return {
    raw: copy.buffer as ArrayBuffer,
    headers: { "content-type": "application/octet-stream", "X-SPARC-Dtype": a instanceof Uint8Array ? "uint8" : "float32", "X-SPARC-Length": String(a.length) },
  };
}

/** Fetch-mock routes for a run's grid, layer catalogue and layer binaries. */
export function mapRoutes(rid: string, extra: Partial<GridMeta> = {}): Record<string, MockHandler> {
  const base = `/api/runs/${rid}`;
  const routes: Record<string, MockHandler> = {
    [`GET ${base}`]: { body: runDetail(rid) },
    [`GET ${base}/grid`]: { body: gridMeta(extra) },
    [`GET ${base}/grid.bin`]: () => gridBin(),
    [`GET ${base}/layers`]: { body: { groups: LAYER_GROUPS } },
  };
  for (const [k, v] of Object.entries(LAYER_VALUES)) routes[`GET ${base}/layers/${encodeURIComponent(k)}.bin`] = () => binResponse(v);
  return routes;
}

/**
 * Chart SVGs that are not inside a ChartFrame figure. The map's own side column (the legend's
 * brushable histogram and the pinned-cell inspector of the foundation map kit) is excluded.
 */
export function unframedCharts(root: HTMLElement): string[] {
  const out: string[] = [];
  for (const svg of root.querySelectorAll("svg.chart")) {
    if (svg.closest(".map-side")) continue;
    if (!svg.closest("figure.chart-frame")) out.push(svg.getAttribute("aria-label") ?? "(unnamed)");
  }
  return out;
}
