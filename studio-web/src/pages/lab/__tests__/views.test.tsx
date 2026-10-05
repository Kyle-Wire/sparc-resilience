// The other Lab pages against a mocked server: a saved plan (planned vs realised with the
// spillover label, field kit, Plan → scenario), a sweep curve, Climate × adaptation (charts,
// explore request, the client-side future map) and Compare (paired SE, needs-exact chips).
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { clearResources } from "../../../api/resource";
import type { Comparison, Plan, Sweep } from "../../../api/lab";
import { navigate } from "../../../router";
import { useJobs } from "../../../stores/jobs";
import { byText, click, flush, mockFetch, render, waitFor } from "../../../test/render";
import Climate from "../Climate";
import Compare from "../Compare";
import Plans from "../Plans";
import Sweeps from "../Sweeps";
import { forgetDrafts } from "../model/draft";
import { L, PID, RID, bin, result, runRoutes, scenario } from "../__fixtures__/lab";

let fetchMock: ReturnType<typeof mockFetch> | null = null;

beforeEach(() => {
  clearResources();
  forgetDrafts();
  useJobs.getState().reset();
});

afterEach(() => {
  fetchMock?.restore();
  fetchMock = null;
});

const PLAN: Plan = {
  id: "pl_1",
  name: "Canopy 20k",
  params: { lever: "Pct_Canopy", budget: 20000, cost: { scalar: 1 }, cap: { plantable: true }, objective: "people", equity: { source: "share_60_plus", focus: 0.3 }, multipliers: [0.5, 1, 2] },
  planned: {
    planned_total: 1704,
    n_cells_treated: 5,
    mean_dose_treated: 12,
    total_cost: 20000,
    gini: 0.31,
    min_dose_dropped_cost: 0,
    pareto: [
      { budget: 10000, benefit: 1000, n_cells: 3, n_segments: 3, gini: 0.2 },
      { budget: 20000, benefit: 1704, n_cells: 5, n_segments: 6, gini: 0.31 },
    ],
    constraint: "plantable headroom",
    objective: "people cooled",
    caption: "Planned allocation",
  },
  realised: { total: 1037, mean_treated: -0.21, mean_all: -0.02, result_id: "res_plan" },
  frontier: [{ budget: 10000, planned: 1000, realised: 640 }],
  created_utc: "2026-10-01T12:00:00Z",
};

describe("Plans", () => {
  it("shows planned vs realised with the spillover label, builds the field kit and turns the plan into a scenario", async () => {
    fetchMock = mockFetch({
      ...runRoutes(),
      "GET /api/plans/pl_1": { body: PLAN },
      "GET /api/plans/pl_1/layers/dose.bin": () => bin(Float32Array.from([20, 0, 10, 0, 5, 0, 0])),
      "GET /api/plans/pl_1/layers/planned_benefit.bin": () => bin(Float32Array.from([1, 0, 0.5, 0, 0.2, 0, 0])),
      "GET /api/plans/pl_1/layers/closed_loop_delta.bin": () => bin(Float32Array.from([-0.3, -0.1, -0.2, 0, -0.1, 0, 0])),
      "POST /api/plans/pl_1/field-kit": {
        body: {
          cells: [{ rank: 1, id: 101, lon: -71.41, lat: 41.82, zone: 1, dose: 20, planned_benefit: 1, closed_loop_delta: -0.3, people: 10, plantable_pp: 25 }],
          sites: [{ id: 101, lon: -71.41, lat: 41.82, role: "treated", canopy: 12, impervious: 60, effect_sd: 0.1 }],
          pairs: [],
        },
      },
      "POST /api/plans/pl_1/to-scenario": { status: 201, body: scenario("sc_plan", { doc: { name: "Canopy 20k", edits: [{ lever: "Pct_Canopy", mode: "per_cell", per_cell_ref: "plan:pl_1" }] } }) },
    });
    navigate(`/r/${RID}/lab/plans/pl_1`, { replace: true });
    const { container } = render(<Plans />);
    await waitFor(() => container.querySelector("[data-spillover]"), 8000, "plan detail");
    expect(container.querySelector("[data-spillover]")!.textContent).toMatch(/Planned 1,704 vs realised 1,037 °F·cells/);
    expect(container.querySelector("[data-spillover]")!.textContent).toMatch(/spillover non-additivity/);
    expect(byText(container, ".kpi", "Realised (exact)")!.textContent).toMatch(/61% of planned/);
    click(byText(container, "button", "Build field kit"));
    await waitFor(() => byText(container, "td", "101"), 5000, "field kit");
    expect(byText(container, "button", "GeoJSON")).not.toBeNull();
    click(byText(container, "button", "Plan → scenario"));
    await waitFor(() => window.location.pathname === `/r/${RID}/lab/s/sc_plan`, 5000, "to scenario");
  });
});

const SWEEP: Sweep = {
  params: { lever: "Pct_Canopy", doses: [5, 10, 20] },
  status: "succeeded",
  curve: [
    { dose: 5, city: L(-0.01, -0.015, -0.005, 0.0025), region: L(-0.2, -0.25, -0.15, 0.025), realized: 4.8, frac_extrapolated: 0 },
    { dose: 10, city: L(-0.018, -0.024, -0.012, 0.003), region: L(-0.34, -0.4, -0.28, 0.03), realized: 9.5, frac_extrapolated: 0.1 },
    { dose: 20, city: L(-0.026, -0.034, -0.018, 0.004), region: L(-0.48, -0.58, -0.38, 0.05), realized: 18, frac_extrapolated: 0.35 },
  ],
  fit: { model: "saturating", A: 0.55, ds: 9, d90: 20.7 },
  pipeline_curve: { dose: [0, 10, 20], benefit: [0, 0.3, 0.45] },
  points: ["res_s1", "res_s2", "res_s3"],
};

describe("Sweeps", () => {
  it("draws the sweep curve with its fit and lists the points", async () => {
    fetchMock = mockFetch({
      ...runRoutes(),
      [`GET /api/runs/${RID}/sweeps`]: { body: [{ id: "sw_1", lever: "Pct_Canopy", doses: [5, 10, 20], status: "succeeded", created_utc: "2026-10-01T12:00:00Z", job_id: null }] },
      "GET /api/sweeps/sw_1": { body: SWEEP },
    });
    navigate(`/r/${RID}/lab/sweeps/sw_1`, { replace: true });
    const { container } = render(<Sweeps />);
    await waitFor(() => container.querySelector('svg.chart[aria-label="Response to the swept lever"]'), 8000, "sweep chart");
    expect(byText(container, "figure", "Fit: saturating, A = 0.550, d_s = 9.00")).not.toBeNull();
    const points = [...container.querySelectorAll("table.tbl")].find((t) => t.querySelector("caption")?.textContent === "Sweep points")!;
    expect(points.querySelectorAll("tbody tr").length).toBe(3);
  });
});

describe("Climate × adaptation", () => {
  it("charts warming and exposure, explores with the chosen adaptation and maps the future client-side", async () => {
    const explored: unknown[] = [];
    fetchMock = mockFetch({
      ...runRoutes(),
      [`GET /api/runs/${RID}/scenarios`]: { body: { configured: [], results: [result("res_1").summary] } },
      [`GET /api/projects/${PID}/scenarios`]: { body: [{ id: "sc_a", revision: 1, parent_id: null, status: "exact", created_utc: "", updated_utc: "", name: "Downtown cool corridor", tags: [], latest: result("res_1").summary }] },
      [`GET /api/runs/${RID}/climate/factors`]: {
        body: {
          present: true,
          path: "cmip6.csv",
          experiments: ["ssp245", "ssp585"],
          periods: ["2041-2060"],
          models: ["A", "B"],
          warming: [
            { experiment: "ssp245", label: "SSP2-4.5", period: "2041-2060", n_models: 2, median: 3.6, p10: 2.7, p90: 4.5, min: 2.5, max: 4.9, by_model: { A: 3.2, B: 4 } },
            { experiment: "ssp585", label: "SSP5-8.5", period: "2041-2060", n_models: 2, median: 4.5, p10: 3.4, p90: 5.6, min: 3, max: 6, by_model: { A: 4.1, B: 4.9 } },
          ],
          action: null,
        },
      },
      [`POST /api/runs/${RID}/climate/explore`]: (_u, init) => {
        explored.push(JSON.parse(String(init.body)));
        return {
          body: {
            present: { mean: 88.1, share_at_or_above: { "90": 0.29 } },
            thresholds: [90],
            projections: [
              {
                experiment: "ssp245",
                label: "SSP2-4.5",
                period: "2041-2060",
                n_models: 2,
                warming: { median: 3.6, p10: 2.7, p90: 4.5, min: 2.5, max: 4.9, by_model: {} },
                variants: [
                  { name: "no adaptation", mean: 91.7, adaptation_mean_delta: 0, offset_share_of_median_warming: null, share_at_or_above: { "90": { median: 0.86, p10: 0.7, p90: 0.95 } } },
                  { name: "Downtown cool corridor", mean: 91.6, adaptation_mean_delta: -0.021, offset_share_of_median_warming: 0.41, share_at_or_above: { "90": { median: 0.84, p10: 0.68, p90: 0.94 } } },
                ],
              },
            ],
            adaptation: ["Downtown cool corridor"],
            people_exposure: null,
            units: "degF",
          },
        };
      },
      "GET /api/results/res_1/layers/delta.bin": () => bin(Float32Array.from([-0.5, 0, -1.2, -0.3, 0, -0.4, 0])),
    });
    navigate(`/r/${RID}/lab/climate`, { replace: true });
    const { container } = render(<Climate />);
    await waitFor(() => container.querySelector('svg.chart[aria-label="Warming by scenario and period"]'), 8000, "warming chart");
    await waitFor(() => explored.length > 0, 5000, "explore");
    expect(explored[0]).toMatchObject({ adaptations: [], statistic: "median" });
    const chip = byText(container, "button.chip", "Downtown cool corridor")!;
    click(chip);
    await waitFor(() => explored.length > 1, 5000, "explore with the adaptation");
    expect(explored.at(-1)).toMatchObject({ adaptations: [{ kind: "result", id: "res_1" }] });
    await waitFor(() => container.querySelector('svg.chart[aria-label="Share of cells at or above 90.0 °F"]'), 5000, "exposure bars");
    expect(byText(container, "figure", "Cancels 41% of SSP2-4.5 2041–2060 median warming")).not.toBeNull();
    // the future map: obs [86, 88, 89.5, 90, 91, 85, 87] + median warming 3.6 → 5 of 7 cells ≥ 90
    await waitFor(() => byText(container, ".cap", "Warming +3.60 °F (median of models)"), 5000, "map caption");
    expect(byText(container, ".cap", "Warming +3.60 °F")!.textContent).toMatch(/71% of cells at or above 90/);
    await flush(2);
  });
});

const CMP: Comparison = {
  id: "cmp_1",
  items: [
    { ref: { kind: "result", id: "res_1" }, label: "Downtown cool corridor", city: L(-0.021, -0.03, -0.012, 0.0046), edited: L(-0.62, -0.8, -0.44, 0.09), cost: 27.6, has_folds: true, per_cell: true },
    { ref: { kind: "configured", slug: "cool-roofs" }, label: "Cool roofs", city: L(-0.01, -0.03, 0.01, 0.01), edited: null, cost: 400, has_folds: false, per_cell: true },
  ],
  pairs: [{ a: 0, b: 1, city: { ...L(0.011, -0.005, 0.027, 0.008), paired: false }, regions: {}, layer_key: "cmp:cmp_1:a__b" }],
  equity: {},
  exposure: [],
  cooling_per_cost: {},
  needs_exact: [{ kind: "configured", slug: "cool-roofs" }],
};

describe("Compare", () => {
  it("creates the comparison, shows paired SE or the needs-exact chip, and renders small multiples", async () => {
    fetchMock = mockFetch({
      ...runRoutes(),
      [`POST /api/runs/${RID}/compare`]: { status: 201, body: CMP },
      "GET /api/comparisons/cmp_1": { body: CMP },
      "GET /api/results/res_1/layers/delta.bin": () => bin(Float32Array.from([-0.5, 0, -1.2, -0.3, 0, -0.4, 0])),
      [`GET /api/runs/${RID}/layers/sc%3Acool-roofs.bin`]: () => bin(Float32Array.from([-0.1, -0.1, 0, 0, -0.2, 0, 0])),
      [`GET /api/runs/${RID}/layers/cmp%3Acmp_1%3Aa__b.bin`]: () => bin(Float32Array.from([0.4, -0.1, 1.2, 0.3, -0.2, 0.4, 0])),
    });
    navigate(`/r/${RID}/lab/compare?items=${encodeURIComponent("res:res_1,configured:cool-roofs")}`, { replace: true });
    const { container } = render(<Compare />);
    await waitFor(() => new URLSearchParams(window.location.search).get("cid") === "cmp_1", 8000, "comparison created");
    await waitFor(() => byText(container, ".badge", "needs exact re-run"), 8000, "pairs table");
    expect(fetchMock.calls.find((c) => c.method === "POST")!.body).toEqual({ items: [{ kind: "result", id: "res_1" }, { kind: "configured", slug: "cool-roofs" }] });
    expect(byText(container, ".badge", "needs exact: Cool roofs")).not.toBeNull();
    expect(byText(container, "button", "Re-run exactly")).not.toBeNull();
    await waitFor(() => container.querySelectorAll(".mini-maps figure").length === 2, 5000, "small multiples");
    expect(container.querySelector(".mini-maps")!.textContent).toContain("Cool roofs");
    // pairs read A − B, as the server computes them (diff = A − B, SE(A − B))
    expect(byText(container, "caption", "Pairwise differences (A − B)")).not.toBeNull();
    expect(byText(container, "td", "Downtown cool corridor − Cool roofs")).not.toBeNull();
    expect(byText(container, "h3", "Difference map: Downtown cool corridor − Cool roofs")).not.toBeNull();
    expect(container.textContent).not.toContain("Cool roofs − Downtown cool corridor");
    // only one comparison was created
    expect(fetchMock.calls.filter((c) => c.method === "POST").length).toBe(1);
  });

  it("says an unverified plan has only city totals and draws no difference map for it", async () => {
    const plan: Comparison = {
      ...CMP,
      id: "cmp_2",
      items: [{ ref: { kind: "plan", id: "plan_1" }, label: "Plan: Trees (planned)", city: { ...L(-0.54, -0.54, -0.54, 0), se: null, lo: null, hi: null }, edited: null, cost: 2000, has_folds: false, per_cell: false }, CMP.items[0]],
      pairs: [{ a: 0, b: 1, city: { ...L(-0.52, -0.52, -0.52, 0), paired: false }, regions: {}, layer_key: null }],
      needs_exact: [{ kind: "plan", id: "plan_1" }],
    };
    fetchMock = mockFetch({
      ...runRoutes(),
      [`POST /api/runs/${RID}/compare`]: { status: 201, body: plan },
      "GET /api/comparisons/cmp_2": { body: plan },
      "GET /api/results/res_1/layers/delta.bin": () => bin(Float32Array.from([-0.5, 0, -1.2, -0.3, 0, -0.4, 0])),
    });
    navigate(`/r/${RID}/lab/compare?items=${encodeURIComponent("plan:plan_1,res:res_1")}`, { replace: true });
    const { container } = render(<Compare />);
    await waitFor(() => byText(container, "h3", "Difference map: Plan: Trees (planned) − Downtown cool corridor"), 8000, "pair");
    expect(byText(container, ".cap", "No difference map: an unverified plan has no per-cell ΔT")).not.toBeNull();
    const note = byText(container, ".cap", "Plan: Trees (planned): not verified.");
    expect(note).not.toBeNull();
    expect(note!.textContent).toContain("°F·cells");
    expect(fetchMock.calls.some((c) => c.url.includes("/layers/cmp"))).toBe(false);
    await flush(2);
  });
});
