// Library and plan helpers: lineage order, ladder groups and doses, number lists, field-kit
// CSV/GeoJSON, sweep overlays, rate limiting, spill rings, the locked compare scale.
import { afterEach, describe, expect, it, vi } from "vitest";
import type { FieldKit, ScenarioSummary } from "../../../api/lab";
import { lockedDomain } from "../Compare";
import { budgetRange, decodeDose, defaultParams } from "../Plans";
import { spillRings, uncertaintyLayers } from "../components/ResultInspector";
import { exactAdvice } from "../components/CompilePanel";
import { newEdit } from "../model/doc";
import { fieldKitCsv, fieldKitGeoJson, ladderDoses, ladderGroups, lineage, parseNumberList, parseSites } from "../model/library";
import { firstLadderEdit } from "../Library";
import { missingParams } from "../components/TemplateGallery";
import { dominantSign, fitOverlay, overlayXs, pipelineCurve } from "../model/sweep";
import { debounce, rateLimit } from "../model/timing";
import { bytesToBase64 } from "../../../api/binary";

const S = (id: string, parent: string | null, extra: Partial<ScenarioSummary> = {}): ScenarioSummary => ({
  id,
  revision: 1,
  parent_id: parent,
  status: "draft",
  created_utc: "2026-10-01T10:00:00Z",
  updated_utc: "2026-10-01T10:00:00Z",
  name: id,
  tags: [],
  latest: null,
  ...extra,
});

describe("lineage", () => {
  it("lists roots by recency, each followed by its revisions depth-first", () => {
    const list = [
      S("a", null, { updated_utc: "2026-10-01T09:00:00Z" }),
      S("b", null, { updated_utc: "2026-10-02T09:00:00Z" }),
      S("a2", "a", { revision: 2 }),
      S("a3", "a2", { revision: 3 }),
      S("a2b", "a", { revision: 2, created_utc: "2026-10-01T11:00:00Z" }),
      S("orphan", "gone"),
    ];
    const t = lineage(list);
    // "orphan" (its parent is not listed) is a root, updated after "a"
    expect(t.map((n) => `${"-".repeat(n.depth)}${n.s.id}`)).toEqual(["b", "orphan", "a", "-a2", "--a3", "-a2b"]);
    expect(t.find((n) => n.s.id === "a")!.children).toBe(2);
  });

  it("survives a parent cycle", () => {
    const t = lineage([S("x", "y"), S("y", "x")]);
    expect(t.map((n) => n.s.id).sort()).toEqual(["x", "y"]);
  });
});

describe("ladders", () => {
  it("groups ladder:<id> tags and finds the dose that varies", () => {
    const g = ladderGroups([S("r1", null, { tags: ["ladder:L1"] }), S("r2", null, { tags: ["ladder:L1", "x"] }), S("solo", null, { tags: ["ladder:L2"] })]);
    expect(g).toEqual([{ id: "L1", members: [expect.objectContaining({ id: "r1" }), expect.objectContaining({ id: "r2" })] }]);
    const base = { name: "n", edits: [newEdit("Albedo", "set", 0.35), newEdit("Pct_Canopy", "add", 5)] };
    const d = ladderDoses([base, { ...base, edits: [base.edits[0], { ...base.edits[1], amount: 10 }] }, { ...base, edits: [base.edits[0], { ...base.edits[1], amount: 20 }] }]);
    expect(d).toEqual({ editIndex: 1, lever: "Pct_Canopy", doses: [5, 10, 20] });
    expect(ladderDoses([base, base])).toBeNull();
  });

  it("parses number lists", () => {
    expect(parseNumberList("5, 10;20  30")).toEqual({ values: [5, 10, 20, 30], bad: [] });
    expect(parseNumberList("−2, x")).toEqual({ values: [-2], bad: ["x"] });
  });

  it("starts a ladder on the first edit that has an amount", () => {
    const sc = (edits: ReturnType<typeof newEdit>[]) => ({ doc: { name: "n", edits } }) as Parameters<typeof firstLadderEdit>[0];
    expect(firstLadderEdit(sc([{ lever: "Pct_Canopy", mode: "per_cell", per_cell_ref: "blob:bl_1" }, newEdit("Albedo", "set", 0.35)]))).toBe(1);
    expect(firstLadderEdit(sc([newEdit("Albedo", "set", 0.35)]))).toBe(0);
  });
});

describe("Around sites template", () => {
  it("reads typed lon, lat lines and reports what is not a pair", () => {
    expect(parseSites("-71.41, 41.82\n−71.40 41.83\n\n-71.39;41.80")).toEqual({ sites: [[-71.41, 41.82], [-71.4, 41.83], [-71.39, 41.8]], bad: [] });
    expect(parseSites("-71.41, 41.82, 5\nnorth gate\n200, 41")).toEqual({ sites: [], bad: ["-71.41, 41.82, 5", "north gate", "200, 41"] });
    expect(parseSites("  ")).toEqual({ sites: [], bad: [] });
  });

  it("reads an uploaded points CSV by its lon/lat header in any column order", () => {
    const csv = "site_id,latitude,longitude,notes\n1,41.82,-71.41,school\r\n2,41.83,-71.40,park\n3,,-71.39,no lat\n4,41.84,-71.38,";
    expect(parseSites(csv)).toEqual({ sites: [[-71.41, 41.82], [-71.4, 41.83], [-71.38, 41.84]], bad: ["3,,-71.39,no lat"] });
    expect(parseSites("lon\tlat\n-71.41\t41.82").sites).toEqual([[-71.41, 41.82]]);
  });

  it("keeps Create disabled until the required sites are given", () => {
    const t = {
      id: "around_sites",
      label: "Around sites",
      desc: "",
      requires: [],
      params_schema: { properties: { sites: { type: "array" as const, title: "sites [lon, lat]" }, radius_m: { type: "number" as const, default: 200 } }, required: ["sites"] },
    };
    expect(missingParams(t, { radius_m: 200 })).toEqual(["sites"]);
    expect(missingParams(t, { sites: [] })).toEqual(["sites"]);
    expect(missingParams(t, { sites: [[-71.4, 41.8]] })).toEqual([]);
  });
});

const KIT: FieldKit = {
  cells: [
    { rank: 1, id: 101, lon: -71.41, lat: 41.82, zone: 3, dose: 20, planned_benefit: 0.8, closed_loop_delta: -0.6, people: 12, plantable_pp: 25 },
    { rank: 2, id: 102, lon: null, lat: null, zone: 3, dose: 10, planned_benefit: 0.4, closed_loop_delta: null, people: 3, plantable_pp: 12 },
  ],
  sites: [{ id: 7, lon: -71.4, lat: 41.8, role: "treated", canopy: 12, impervious: 70, effect_sd: 0.1 }],
  pairs: [{ treated_id: 101, control_id: 205, treated_lon: -71.41, treated_lat: 41.82, control_lon: -71.43, control_lat: 41.84, covariate_distance: 0.3 }],
};

describe("field kit downloads", () => {
  it("renders CSV with ids and lon/lat", () => {
    const csv = fieldKitCsv(KIT, "cells").split(/\r?\n/);
    expect(csv[0]).toBe("rank,id,lon,lat,zone,dose,planned_benefit,closed_loop_delta,people,plantable_pp");
    expect(csv[1]).toBe("1,101,-71.41,41.82,3,20,0.8,-0.6,12,25");
    expect(fieldKitCsv(KIT, "pairs").split(/\r?\n/)[1]).toBe("101,205,-71.41,41.82,-71.43,41.84,0.3");
  });

  it("renders GeoJSON points and pair lines, skipping rows without lon/lat", () => {
    const { geojson, skipped } = fieldKitGeoJson(KIT);
    const fc = JSON.parse(geojson) as { features: { geometry: { type: string; coordinates: unknown }; properties: { layer: string } }[] };
    expect(skipped).toBe(1);
    expect(fc.features.map((f) => `${f.properties.layer}:${f.geometry.type}`)).toEqual(["cell:Point", "logger_site:Point", "before_after_pair:LineString"]);
    expect(fc.features[0].geometry.coordinates).toEqual([-71.41, 41.82]);
  });
});

describe("sweeps", () => {
  it("draws the saturating fit on the points' sign", () => {
    const pts = [
      { x: 5, y: -0.2 },
      { x: 10, y: -0.35 },
    ];
    const xs = overlayXs([5, 10], 2);
    expect(xs).toEqual([0, 5, 10]);
    const y = fitOverlay({ model: "saturating", A: 0.5, ds: 8, d90: 18 }, xs, pts)!;
    expect(y[0]).toBeCloseTo(0);
    expect(y[2]).toBeCloseTo(-0.5 * (1 - Math.exp(-10 / 8)));
    const lin = fitOverlay({ model: "linear", A: null, ds: null, d90: null }, [0, 10], [{ x: 10, y: -0.3 }])!;
    expect(lin[1]).toBeCloseTo(-0.3);
    expect(fitOverlay({ model: "sigmoid", A: 1, ds: null, d90: 5 }, xs, pts)).toBeNull();
    expect(dominantSign([-1, -2, 3])).toBe(-1);
    expect(pipelineCurve({ dose: [0, 10], benefit: [0, 0.4] }, -1)).toEqual({ x: [0, 10], y: [-0, -0.4] });
    expect(pipelineCurve({ nope: 1 }, 1)).toBeNull();
  });
});

describe("timing", () => {
  afterEach(() => vi.useRealTimers());

  it("rateLimit fires at most once per window and always with the last value", () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    const calls: number[] = [];
    const f = rateLimit((v: number) => calls.push(v), 150);
    for (let i = 0; i < 100; i++) {
      f(i);
      vi.advanceTimersByTime(10);
    }
    vi.advanceTimersByTime(200);
    expect(calls.length).toBeLessThanOrEqual(Math.ceil(1000 / 150) + 1);
    expect(calls.at(-1)).toBe(99);
    const d = debounce((v: number) => calls.push(v), 100);
    const before = calls.length;
    d(1);
    vi.advanceTimersByTime(50);
    d(2);
    vi.advanceTimersByTime(99);
    expect(calls.length).toBe(before);
    vi.advanceTimersByTime(1);
    expect(calls.at(-1)).toBe(2);
  });
});

describe("result and compare helpers", () => {
  it("turns spill rings into chart rings with the edited set first", () => {
    expect(
      spillRings([
        { r_m: 60, mean: -0.1, se: 0.01, n: 50 },
        { r_m: 0, mean: -0.5, se: 0.05, n: 20 },
        { r_m: 30, mean: -0.2, se: 0.02, n: 40 },
      ]),
    ).toEqual([
      { r0_m: 0, r1_m: 0, mean: -0.5, se: 0.05, n: 20 },
      { r0_m: 0, r1_m: 30, mean: -0.2, se: 0.02, n: 40 },
      { r0_m: 30, r1_m: 60, mean: -0.1, se: 0.01, n: 50 },
    ]);
  });

  it("lists uncertainty layers that exist", () => {
    const l = uncertaintyLayers({ estimation_95: [-0.5, -0.3], specification: null, attribution: [-0.6, -0.2], causal_band: null, envelope: [-0.7, -0.1], envelope_excludes_zero: true, sources: [] });
    expect(l.map((x) => x.id)).toEqual(["estimation", "attribution", "envelope"]);
  });

  it("locks one symmetric scale across items", () => {
    const d = lockedDomain([Float32Array.from([-1, -0.5, 0]), null, Float32Array.from([0, 0.2, 2])]);
    expect(d.kind).toBe("div");
    expect(d.center).toBe(0);
    expect(d.lo).toBe(-d.hi);
    expect(d.hi).toBeGreaterThan(1.5);
  });

  it("recommends an exact run when the preview cannot be trusted", () => {
    const lever = (trust: "good" | "rough" | "none") => ({
      var: "Albedo", label: "Albedo", unit: "", min: 0, max: 1, direction: "increase" as const, doses: [], cost_per_unit: 1, design_dose: 0.1, sd: 0.05,
      headroom_available: false, role: "albedo", mediator_children: [], emulator: { available: trust !== "none", trust, patch_pass_rate: 0.9, uniform_rel_err: 0.2 },
    });
    const compile = { content_hash: "h", portable: true, levers: { Albedo: { n_cells: 10, mean_requested: 0.1, total_requested: 1, predicted_mean_realised: 0.1, clipped_share: 0, est_cost: 1 } }, union_cells: 10, people: null, warnings: [], est_exact_s: 13, emulator: { usable: true, hatched: false, reasons: [] } };
    expect(exactAdvice(compile, null, [lever("good")]).recommend).toBe(false);
    expect(exactAdvice(compile, null, [lever("rough")]).reasons).toEqual(["Albedo: preview is rough"]);
    expect(exactAdvice({ ...compile, emulator: { usable: true, hatched: true, reasons: ["Preview unreliable at this scale (uniform albedo rel. error 246%)"] } }, null, [lever("good")]).reasons).toEqual(["Preview unreliable at this scale (uniform albedo rel. error 246%)"]);
    expect(exactAdvice(null, null, []).recommend).toBe(false);
  });

  it("plans: default params, budget range, dose decoding", () => {
    const lever = { var: "Pct_Canopy", label: "Canopy", unit: "pp", min: 0, max: 100, direction: "increase" as const, doses: [5, 10, 20, 30], cost_per_unit: 1, design_dose: 10, sd: 12, headroom_available: true, role: "canopy", mediator_children: [], emulator: { available: true, trust: "good" as const, patch_pass_rate: 1, uniform_rel_err: 0.1 } };
    expect(budgetRange(lever, 1000)).toEqual([30, 30000]);
    expect(defaultParams(lever, 1000)).toMatchObject({ lever: "Pct_Canopy", budget: 20000, cost: { scalar: 1 }, cap: { plantable: true }, objective: "cooling" });
    const f = Float32Array.from([0, 2.5, 0, 10]);
    expect(Array.from(decodeDose(bytesToBase64(new Uint8Array(f.buffer))))).toEqual([0, 2.5, 0, 10]);
  });
});
