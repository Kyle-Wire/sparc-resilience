// Lab test fixtures: the 3×3 test lattice (7 observed cells, zones 1 and 2) served as a run,
// its layer catalogue and binaries, levers, scenarios, an exact result, and fetch-mock routes.
import { pack, packBits } from "../../../api/binary";
import type { Lever, Result, Scenario, ScenarioDoc } from "../../../api/lab";
import type { GridMeta, LayerGroup, LayerMeta, Likely } from "../../../api/types";
import type { MockHandler, MockRoute } from "../../../test/render";

export const RID = "r_lab";
export const PID = "p_1";

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

export function gridBin(): MockRoute {
  const { buffer, offsets } = pack([
    { name: "ix", dtype: "int32", data: Int32Array.from([0, 1, 0, 2, 0, 1, 2]) },
    { name: "iy", dtype: "int32", data: Int32Array.from([2, 2, 1, 1, 0, 0, 0]) },
    { name: "lon", dtype: "float32", data: Float32Array.from([-71.5, -71.4, -71.5, -71.3, -71.5, -71.4, -71.3]) },
    { name: "lat", dtype: "float32", data: Float32Array.from([41.9, 41.9, 41.8, 41.8, 41.7, 41.7, 41.7]) },
    { name: "zone", dtype: "int16", data: Int16Array.from([0, 0, 0, 1, 1, 1, 1]) },
  ]);
  return { raw: buffer, headers: { "content-type": "application/octet-stream", "X-SPARC-Offsets": JSON.stringify(offsets) } };
}

const stats = (lo: number, hi: number) => ({ n: 7, lo, hi, mean: (lo + hi) / 2, p1: lo, p2: lo, p50: (lo + hi) / 2, p98: hi, p99: hi });

export function layer(p: Partial<LayerMeta> & { key: string }): LayerMeta {
  return {
    group: "inputs",
    label: p.key,
    unit: "",
    scale: "seq",
    center: null,
    decimals: 1,
    mult: 1,
    zero_blank: false,
    labels: null,
    desc: "",
    sign_note: null,
    source: null,
    dtype: "float32",
    stats: stats(0, 100),
    ...p,
  };
}

export const GROUPS: LayerGroup[] = [
  { id: "temperature", label: "Temperature", layers: [layer({ key: "obs", group: "temperature", label: "Observed temperature", unit: "degF", scale: "div", center: 88, stats: stats(85, 91) })] },
  {
    id: "inputs",
    label: "Inputs",
    layers: [layer({ key: "Pct_Canopy", label: "Canopy cover", unit: "pp" }), layer({ key: "Albedo", label: "Albedo", unit: "", stats: stats(0.1, 0.4) })],
  },
  { id: "planner", label: "Planner & people", layers: [layer({ key: "people", group: "planner", label: "Residents", unit: "people", stats: stats(0, 50) })] },
];

export const VALUES: Record<string, Float32Array> = {
  obs: Float32Array.from([86, 88, 89.5, 90, 91, 85, 87]),
  Pct_Canopy: Float32Array.from([12.3, 97.7, 0.1, 55.55, 99.99, 3.3, 40]),
  Albedo: Float32Array.from([0.12, 0.2, 0.31, 0.15, 0.18, 0.22, 0.1]),
  people: Float32Array.from([10, 0, 25, 5, 40, 12, 3]),
};

export function bin(a: Float32Array): MockRoute {
  const copy = a.slice();
  return { raw: copy.buffer as ArrayBuffer, headers: { "content-type": "application/octet-stream", "X-SPARC-Dtype": "float32", "X-SPARC-Length": String(a.length) } };
}

export function lever(v: string, extra: Partial<Lever> = {}): Lever {
  return {
    var: v,
    label: v === "Pct_Canopy" ? "Canopy cover" : v,
    unit: v === "Pct_Canopy" ? "pp" : "",
    min: 0,
    max: v === "Pct_Canopy" ? 100 : 1,
    direction: "increase",
    doses: v === "Pct_Canopy" ? [5, 10, 20, 30] : [0.05, 0.1],
    cost_per_unit: v === "Pct_Canopy" ? 1 : 100,
    design_dose: v === "Pct_Canopy" ? 10 : 0.1,
    sd: v === "Pct_Canopy" ? 12 : 0.05,
    headroom_available: v === "Pct_Canopy",
    role: v === "Pct_Canopy" ? "canopy" : "albedo",
    mediator_children: [],
    emulator: { available: true, trust: v === "Pct_Canopy" ? "good" : "rough", patch_pass_rate: 0.95, uniform_rel_err: v === "Pct_Canopy" ? 0.2 : 2.46 },
    ...extra,
  };
}

export const LEVERS: Lever[] = [lever("Pct_Canopy"), lever("Albedo")];

export const L = (estimate: number, lo: number | null, hi: number | null, se: number | null = null): Likely => ({
  estimate,
  lo,
  hi,
  se,
  confidence: hi !== null && hi < 0 ? "confident_cools" : lo !== null && lo > 0 ? "confident_warms" : "could_be_zero",
  phrase: "",
});

export function doc(extra: Partial<ScenarioDoc> = {}): ScenarioDoc {
  return {
    name: "Downtown cool corridor",
    notes: "",
    tags: ["district"],
    anchor_run_id: RID,
    edits: [{ lever: "Pct_Canopy", mode: "add", amount: 10, where: { kind: "zones", values: [1] }, label: "street trees" }],
    regions: {},
    costs: {},
    options: { clip_to_support: true, mediators: true, expert: false },
    ...extra,
  };
}

export function scenario(id: string, extra: Partial<Scenario> = {}): Scenario {
  return {
    id,
    project_id: PID,
    revision: 1,
    parent_id: null,
    children: [],
    doc: doc(),
    content_hash: "h1",
    status: "draft",
    created_utc: "2026-10-01T10:00:00Z",
    updated_utc: "2026-10-01T10:00:00Z",
    results: [],
    ...extra,
  };
}

export function result(id: string, extra: Partial<Result> = {}): Result {
  const edited = L(-0.62, -0.8, -0.44, 0.092);
  const city = L(-0.021, -0.03, -0.012, 0.0046);
  return {
    summary: { id, scenario_id: "sc_a", run_id: RID, kind: "exact", created_utc: "2026-10-01T11:00:00Z", stale: false, has_folds: true, city, edited, frac_extrapolated_edited: 0.23, job_id: "j_1" },
    spec: {},
    scenario: { id: "sc_a", revision: 1, name: "Downtown cool corridor" },
    city,
    p10: -0.3,
    p90: 0,
    mean_delta_sd: 0.05,
    regions: [
      { name: "edited cells", auto: true, n_cells: 3, mean: edited, people_weighted: -0.6, total: -1.86, frac_cooled_01: 1, frac_cooled_05: 0.67 },
      { name: "outside", auto: true, n_cells: 4, mean: L(-0.05, -0.08, -0.02, 0.015), people_weighted: -0.04, total: -0.2, frac_cooled_01: 0.25, frac_cooled_05: 0 },
    ],
    spill: {
      inside: -1.86,
      outside: -0.2,
      outside_share: 0.12,
      rings: [
        { r_m: 0, mean: -0.62, se: 0.09, n: 3 },
        { r_m: 30, mean: -0.1, se: 0.02, n: 2 },
        { r_m: 60, mean: -0.03, se: 0.01, n: 2 },
      ],
      lever_ranges: { Pct_Canopy: 120 },
    },
    extrapolated_edited: 0.23,
    realized: { Pct_Canopy: { requested_mean: 10, realized_mean: 9.2, requested_total: 30, realized_total: 27.6, clipped_share: 0.08 } },
    mediators: { NDVI: { mean_change: 0.04 } },
    cost: { total: 27.6, per_lever: { Pct_Canopy: 27.6 }, cooling_per_cost: 0.074 }, // positive = cooler (stats.cost_table)
    causal_check: { delta: -0.03, lo: -0.05, hi: -0.01, model_within: true },
    uncertainty: { estimation_95: [-0.03, -0.012], specification: null, attribution: null, causal_band: [-0.05, -0.01], envelope: [-0.05, -0.005], envelope_excludes_zero: true, sources: ["estimation"] },
    impacts: null,
    preview_vs_exact: { mean_abs_err: 0.004, rel_err: 0.15 },
    plain: { headline: "", confidence: "", qualifiers: [], buys: [] },
    warnings: [],
    stale: false,
    demo: true,
    ...extra,
  };
}

export function previewBody(seq: number, delta: number[], edited: number[]): MockRoute {
  const { buffer, offsets } = pack([
    { name: "delta", dtype: "float32", data: Float32Array.from(delta) },
    { name: "edited", dtype: "uint8", data: packBits(edited) },
  ]);
  const n = edited.filter(Boolean).length;
  return {
    raw: buffer,
    headers: {
      "content-type": "application/octet-stream",
      "X-SPARC-Offsets": JSON.stringify(offsets),
      "X-SPARC-Summary": JSON.stringify({ mean: delta.reduce((a, b) => a + b, 0) / delta.length, edited_mean: -0.5, n_edited: n, outside_share: 0.1, trust: "good", hatched: false, reasons: [], request_seq: seq }),
    },
  };
}

/** Routes every Lab page needs for the run: detail, grid, layers, levers, engine, emulator, lists. */
export function runRoutes(rid = RID): Record<string, MockHandler> {
  const base = `/api/runs/${rid}`;
  const r: Record<string, MockHandler> = {
    [`GET ${base}`]: {
      body: {
        run: { id: rid, project_id: PID, label: "Fast demo run", status: "complete", mode: "fast", coarse_m: null, demo: true, checkpoint_bytes: 1_200_000 },
        header: { name: "demo", run_dir: `/w/projects/demo/runs/${rid}`, created_utc: null, git_commit: null, git_dirty: null, n_points: 7, grid_shape: [3, 3], cell_m: 30, fast: true, coarse_m: null, demo: true },
      },
    },
    [`GET ${base}/grid`]: { body: gridMeta() },
    [`GET ${base}/grid.bin`]: () => gridBin(),
    [`GET ${base}/layers`]: { body: { groups: GROUPS } },
    [`GET ${base}/levers`]: { body: LEVERS },
    [`GET ${base}/emulator`]: { body: { present: true, kernel_cells: 41, levers: {}, action: null } },
    [`GET ${base}/engine`]: { body: { state: "cold", progress: null, step: null, rss_mb: null, est_rss_mb: 300, code_match: true, loaded_utc: null, last_used_utc: null, error: null, job_id: null, action: null } },
    [`GET ${base}/regions`]: { body: [] },
    [`GET ${base}/plans`]: { body: [] },
    [`GET ${base}/sweeps`]: { body: [] },
    [`GET ${base}/scenarios`]: { body: { configured: [], results: [] } },
    "GET /api/scenario-templates": { body: [] },
    [`GET /api/projects/${PID}/scenarios`]: { body: [] },
  };
  for (const [k, v] of Object.entries(VALUES)) r[`GET ${base}/layers/${encodeURIComponent(k)}.bin`] = () => bin(v);
  return r;
}
