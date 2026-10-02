// Every ViewModel-driven run tab against the synth_run-derived fixtures (SPEC §6.4, api.md
// §6.1): full fixtures render without a section failing; with every section null (an
// older-code run) the view still renders and says "not in this run (older code)"; the live
// banner, the server's missing-output action and the server-computed Budget caption; and
// every chart sits inside a ChartFrame.
import { afterEach, describe, expect, it } from "vitest";
import type { ComponentType } from "react";
import { clearResources } from "../../../api/resource";
import type { ViewModelOf, ViewName } from "../../../api/runs";
import { navigate } from "../../../router";
import { flush, mockFetch, render, waitFor, type MockHandler } from "../../../test/render";
import { olderCodeFixture, sectionKeys, viewFixture } from "../__fixtures__/views";
import Accuracy from "../Accuracy";
import Budget from "../Budget";
import Causal from "../Causal";
import Climate from "../Climate";
import DataQa from "../DataQa";
import Distance from "../Distance";
import Influence from "../Influence";
import Overview from "../Overview";
import Planner from "../Planner";
import Provenance from "../Provenance";
import Response from "../Response";
import Scenarios from "../Scenarios";
import Uncertainty from "../Uncertainty";
import { binResponse, LAYER_GROUPS, layerMeta, mapRoutes, runDetail, unframedCharts } from "./helpers";

const PAGES: { view: ViewName; tab: string; Page: ComponentType }[] = [
  { view: "overview", tab: "", Page: Overview },
  { view: "data", tab: "data", Page: DataQa },
  { view: "accuracy", tab: "accuracy", Page: Accuracy },
  { view: "distance", tab: "distance", Page: Distance },
  { view: "influence", tab: "influence", Page: Influence },
  { view: "response", tab: "response", Page: Response },
  { view: "scenarios", tab: "scenarios", Page: Scenarios },
  { view: "climate", tab: "climate", Page: Climate },
  { view: "causal", tab: "causal", Page: Causal },
  { view: "budget", tab: "budget", Page: Budget },
  { view: "planner", tab: "planner", Page: Planner },
  { view: "uncertainty", tab: "uncertainty", Page: Uncertainty },
  { view: "provenance", tab: "provenance", Page: Provenance },
];

let seq = 0;

/** Render a tab of a fresh run whose /views/{view} returns `vm`. */
async function renderView<V extends ViewName>(p: { view: V; tab: string; Page: ComponentType }, vm: ViewModelOf<V>, extra: Record<string, MockHandler> = {}) {
  const rid = `r_view${++seq}`;
  const m = mockFetch({
    ...mapRoutes(rid),
    [`GET /api/runs/${rid}/views/scenarios`]: { body: viewFixture("scenarios") },
    [`GET /api/runs/${rid}/views/${p.view}`]: { body: vm },
    [`GET /api/runs/${rid}/environment`]: { body: { packages: ["numpy==1.26.4", "scipy==1.16.3"] } },
    [`GET /api/runs/${rid}/config`]: {
      body: {
        effective: {},
        raw: {},
        yaml: "name: synthetic_demo\n",
        source: "launch",
        config_dir: "/w/p",
        vs_project_diff: [],
        vs_defaults: [{ path: "cv.n_folds", value: 3, default: 5 }],
      },
    },
    "GET /api/projects/p_1/runs": { body: { items: [], next_cursor: null } },
    ...extra,
  });
  navigate(`/r/${rid}${p.tab ? "/" + p.tab : ""}`, { replace: true });
  const r = render(<p.Page />);
  await waitFor(() => r.container.querySelector(`[data-view="${p.view}"]`) && !r.container.textContent?.includes(`Loading ${p.view}`), 5000, `${p.view} view`);
  await flush(6);
  return { ...r, rid, calls: m.calls, restore: m.restore };
}

describe("run-hub views with full fixtures", () => {
  afterEach(() => {
    clearResources();
  });

  for (const p of PAGES) {
    it(`${p.view}: renders every section, frames every chart, no section fails`, async () => {
      const r = await renderView(p, viewFixture(p.view));
      try {
        expect(r.container.querySelector(`[data-view="${p.view}"]`)).not.toBeNull();
        // SectionBoundary would print "could not be displayed" for a malformed section.
        expect(r.container.textContent).not.toContain("could not be displayed");
        expect(r.container.querySelectorAll("[data-older-code]").length).toBe(0);
        expect(unframedCharts(r.container)).toEqual([]);
        if (!["provenance"].includes(p.view)) expect(r.container.querySelectorAll("figure.chart-frame").length + r.container.querySelectorAll(".kpis").length).toBeGreaterThan(0);
      } finally {
        r.restore();
      }
    });
  }

  it("charts carry units and cooler/warmer wording", async () => {
    const r = await renderView(PAGES.find((p) => p.view === "scenarios")!, viewFixture("scenarios"));
    try {
      const text = r.container.textContent ?? "";
      expect(text).toContain("°F");
      expect(text).toMatch(/cooler/);
      expect(text).toContain("−"); // Unicode minus, never "-" before a number
      expect(r.container.querySelector("figure.chart-frame details.table-view summary")!.textContent).toBe("Table view");
    } finally {
      r.restore();
    }
  });
});

describe("older-code runs (every section null)", () => {
  afterEach(() => clearResources());

  for (const p of PAGES) {
    it(`${p.view}: renders and says "not in this run (older code)" for each section`, async () => {
      const r = await renderView(p, olderCodeFixture(p.view));
      try {
        const marks = r.container.querySelectorAll("[data-older-code]");
        expect(marks.length).toBeGreaterThan(0);
        expect(r.container.textContent).toContain("not in this run (older code)");
        expect(r.container.textContent).not.toContain("could not be displayed");
        // No view drops a section silently: one placeholder per section key at least, except
        // keys that are flags or parts of another section (live, has_detail, caption, status,
        // thresholds), the planner's field-kit lists (shown together on one map) and the
        // optional DAG audit (off unless configured, so null is not "older code").
        const folded = new Set(["live", "has_detail", "caption", "status", "thresholds", "hex_files", "sites", "pairs", "dag_audit"]);
        const shown = sectionKeys(p.view).filter((k) => !folded.has(k));
        expect(marks.length).toBeGreaterThanOrEqual(shown.length);
      } finally {
        r.restore();
      }
    });
  }
});

describe("view behaviour", () => {
  afterEach(() => clearResources());

  it("Accuracy shows a live banner when sections.live is true, and none otherwise", async () => {
    const page = PAGES.find((p) => p.view === "accuracy")!;
    const live = viewFixture("accuracy");
    live.sections.live = true;
    const a = await renderView(page, live);
    try {
      const banner = a.container.querySelector('[data-live="true"]');
      expect(banner).not.toBeNull();
      expect(banner!.textContent).toMatch(/Live/);
      expect(banner!.textContent).toContain("predictions.parquet");
    } finally {
      a.restore();
      a.unmount();
    }
    const b = await renderView(page, viewFixture("accuracy"));
    try {
      expect(b.container.querySelector('[data-live="true"]')).toBeNull();
    } finally {
      b.restore();
    }
  });

  it("shows the live banner on any tab while the run is running", async () => {
    const page = PAGES.find((p) => p.view === "influence")!;
    const rid = `r_view${seq + 1}`;
    const r = await renderView(page, viewFixture("influence"), { [`GET /api/runs/${rid}`]: { body: runDetail(rid, { status: "running" }) } });
    try {
      expect(r.container.querySelector('[data-live="true"]')).not.toBeNull();
    } finally {
      r.restore();
    }
  });

  it("Causal maps the CATE and model slopes when causal_cells.parquet exists", async () => {
    const page = PAGES.find((p) => p.view === "causal")!;
    const rid = `r_view${seq + 1}`;
    const groups = [
      ...LAYER_GROUPS,
      { id: "causal", label: "Causal", layers: [layerMeta({ key: "cate_canopy", label: "CATE canopy", scale: "div", center: 0, stats: { n: 7, lo: -0.1, hi: 0, mean: -0.05, p1: -0.1, p2: -0.1, p50: -0.05, p98: 0, p99: 0 } })] },
    ];
    const r = await renderView(page, viewFixture("causal"), {
      [`GET /api/runs/${rid}/layers`]: { body: { groups } },
      [`GET /api/runs/${rid}/layers/cate_canopy.bin`]: () => binResponse(Float32Array.from([-0.1, -0.08, -0.06, -0.05, -0.03, -0.01, 0])),
    });
    try {
      await waitFor(() => r.container.querySelector('[role="application"]'), 3000, "CATE map");
      expect(r.container.querySelector(".map-side h3")!.textContent).toBe("CATE canopy");
      expect([...r.container.querySelectorAll("a")].some((a) => a.getAttribute("href") === `/r/${rid}/map?layer=cate_canopy`)).toBe(true);
    } finally {
      r.restore();
    }
  });

  it("Causal shows the server's effect unit once and the dose-response x axis in the lever unit", async () => {
    const page = PAGES.find((p) => p.view === "causal")!;
    const vm = viewFixture("causal");
    // The server sends the treatment unit as the effect unit (views._causal: "<target> per <lever unit>").
    vm.units = { target: "°F", levers: { canopy: "pp" } };
    vm.sections.treatments!.canopy.unit = "°F per pp";
    const r = await renderView(page, vm);
    try {
      const text = r.container.textContent ?? "";
      expect(text).toContain("Units: °F per pp");
      expect(text).toContain("Effect per unit (°F per pp)");
      expect(text).toContain("CATE (°F per pp)");
      expect(text).not.toContain("per °F per");
      const dr = [...r.container.querySelectorAll("figure.chart-frame")].find((f) => f.getAttribute("data-chart")?.includes("dose–response"))!;
      expect(dr).toBeDefined();
      expect(dr.textContent).toContain("canopy (pp)");
      expect(dr.textContent).not.toContain("canopy (°F per pp)");
    } finally {
      r.restore();
    }
  });

  it("Response's literature panel gives SPARC's number as a rate per +0.10, scaled from the nearest scenario", async () => {
    const page = PAGES.find((p) => p.view === "response")!;
    const vm = viewFixture("response");
    // Only +20 configured: core rescales its city-mean cooling (0.833 °C) by 10/20 to a per-+0.10 rate.
    const lit = vm.sections.literature!;
    lit.sparc = lit.sparc.map((m) => (m.quantity === "canopy" ? { ...m, scenario: "Canopy Increase +20", dose: 20, cooling: 0.4166, se: 0.127, frac_extrapolated: 1 } : m));
    const r = await renderView(page, vm);
    try {
      const fig = [...r.container.querySelectorAll("figure.chart-frame")].find((f) => f.getAttribute("data-chart") === "Literature check")!;
      expect(fig).toBeDefined();
      const cap = fig.querySelector("figcaption")!.textContent!;
      expect(cap).toContain("SPARC: 0.42 °C of cooling per +0.10 canopy cover, scaled from Canopy Increase +20 (realised dose 20 pp)");
      expect(cap).toContain("100% of cells beyond the observed range");
      expect(cap).not.toContain("cools by");
      expect(fig.textContent).toContain("Units: cooling per +0.10 canopy cover, °C");
      const dot = [...fig.querySelectorAll("[aria-label]")].map((e) => e.getAttribute("aria-label") ?? "").find((a) => a.startsWith("SPARC per +0.10 canopy cover · from Canopy Increase +20"));
      expect(dot).toContain("extrapolated");
    } finally {
      r.restore();
    }
  });

  it("the Budget caption comes from the ViewModel", async () => {
    const page = PAGES.find((p) => p.view === "budget")!;
    for (const caption of ["Doubling the budget buys 1.6× the cooling.", "Tripling the budget buys 2.1× the cooling."]) {
      const vm = viewFixture("budget");
      vm.sections.caption = caption;
      const r = await renderView(page, vm);
      try {
        const fig = [...r.container.querySelectorAll("figure.chart-frame")].find((f) => f.getAttribute("data-chart")?.startsWith("Cooling bought"));
        expect(fig).toBeDefined();
        expect(fig!.querySelector("figcaption")!.textContent).toContain(caption);
      } finally {
        r.restore();
        r.unmount();
      }
    }
  });

  it("Budget explains core's 'no positive-benefit segments' instead of reporting it missing", async () => {
    const vm = viewFixture("budget");
    vm.sections.status = "no positive-benefit segments";
    const r = await renderView(PAGES.find((p) => p.view === "budget")!, vm);
    try {
      expect(r.container.querySelector('[data-status="no positive-benefit segments"]')!.textContent).toContain("No cell has a positive cooling footprint");
    } finally {
      r.restore();
    }
  });

  it("Distance says why a stage the config disabled has no section, with the Overview's remedy, not 'older code'", async () => {
    const page = PAGES.find((p) => p.view === "distance")!;
    const rid = `r_view${seq + 1}`;
    const vm = viewFixture("distance");
    vm.sections.curve = null;
    const detail = runDetail(rid);
    detail.stages = [
      { id: "baselines", state: "done", seconds: 2, source: "manifest", reason: null },
      { id: "cv_curve", state: "skipped", seconds: null, source: "manifest", reason: "disabled_by_config:cv.distance_curve.enabled" },
    ];
    const overview = viewFixture("overview");
    const rerun = { kind: "open" as const, label: "Re-run with the CV curve", method: "GET" as const, path: `/p/p_1/launch?from=${rid}` };
    overview.sections.outputs_grid = [
      { id: "cv_distance", label: "Skill vs distance", group: "trust", state: "missing", produced_by: "stage:cv_curve", view: "distance", action: rerun },
    ];
    const r = await renderView(page, vm, {
      [`GET /api/runs/${rid}`]: { body: detail },
      [`GET /api/runs/${rid}/views/overview`]: { body: overview },
    });
    try {
      await waitFor(() => r.container.querySelector("[data-absent]")?.querySelector("button"), 3000, "the re-run action");
      expect(r.container.querySelectorAll("[data-older-code]").length).toBe(0);
      const note = r.container.querySelector("[data-absent]")!;
      expect(note.textContent).toContain("Skill vs block size");
      expect(note.textContent).toContain("disabled in the config (cv.distance_curve.enabled)");
      expect(note.querySelector("button")!.textContent).toBe("Re-run with the CV curve");
    } finally {
      r.restore();
    }
  });

  it("Distance keeps 'older code' when the run's stage rows cannot explain a null section", async () => {
    const vm = viewFixture("distance");
    vm.sections.curve = null;
    const r = await renderView(PAGES.find((p) => p.view === "distance")!, vm);
    try {
      expect(r.container.querySelectorAll("[data-older-code]").length).toBe(1);
      expect(r.container.querySelector("[data-absent]")).toBeNull();
      expect(r.calls.some((c) => c.url.includes("/views/overview"))).toBe(false);
    } finally {
      r.restore();
    }
  });

  it("Planner says why hot days, equity and zones are absent from a pack it read, not 'older code'", async () => {
    const vm = viewFixture("planner");
    vm.sections.hot_days = null;
    vm.sections.equity = null;
    vm.sections.zones = null;
    const r = await renderView(PAGES.find((p) => p.view === "planner")!, vm);
    try {
      expect(r.container.querySelectorAll("[data-older-code]").length).toBe(0);
      const notes = [...r.container.querySelectorAll("[data-absent]")].map((n) => n.textContent ?? "");
      expect(notes).toHaveLength(3);
      expect(notes.find((t) => t.startsWith("Hot days"))).toContain("planner.ghcn_station");
      expect(notes.find((t) => t.startsWith("Equity"))).toContain("package");
      expect(notes.find((t) => t.startsWith("Zones"))).toContain("zone column");
    } finally {
      r.restore();
    }
  });

  it("Planner equity bars are labelled as mean cooling in the target unit, by quintile of the measure", async () => {
    const vm = viewFixture("planner");
    vm.units = { ...vm.units, target: "°F" };
    vm.sections.equity = {
      measures: ["population density", "share aged 60+"],
      quintiles: [1, 2, 3, 4, 5].map((q) => ({ label: `Q${q}`, values: { "population density": 0.1 * q, "share aged 60+": 0.05 * q } })),
      concentration: [
        { label: "population density", value: 0.02 },
        { label: "share aged 60+", value: -0.05 },
      ],
    };
    const r = await renderView(PAGES.find((p) => p.view === "planner")!, vm);
    try {
      const fig = [...r.container.querySelectorAll("figure.chart-frame")].find((f) => (f.getAttribute("data-chart") ?? "").toLowerCase().includes("population density"))!;
      expect(fig).toBeDefined();
      const head = [...fig.querySelectorAll("details.table-view th")].map((th) => th.textContent ?? "");
      expect(head.some((h) => h.startsWith("Mean cooling") && h.includes("°F"))).toBe(true);
      expect(head.some((h) => h.startsWith("population density"))).toBe(false);
      expect(fig.textContent).toContain("Units: °F (positive = cooler)");
      expect(fig.querySelector("svg.chart")!.textContent).toContain("Mean cooling (°F)");
    } finally {
      r.restore();
    }
  });

  it("a missing view shows the server's action as a button", async () => {
    const vm = olderCodeFixture("planner");
    vm.availability = "missing";
    vm.missing = [
      { output: "planner", produced_by: "post:planner", action: { kind: "run_job", label: "Run planner pack", method: "POST", path: "/api/runs/x/actions/planner", body: {} } },
    ];
    const r = await renderView(PAGES.find((p) => p.view === "planner")!, vm);
    try {
      expect(r.container.textContent).toContain("produced by the planner post-run action");
      expect([...r.container.querySelectorAll("button")].some((b) => b.textContent === "Run planner pack")).toBe(true);
    } finally {
      r.restore();
    }
  });

  it("Accuracy residual-histogram brush becomes a pred:resid selection in the URL", async () => {
    const r = await renderView(PAGES.find((p) => p.view === "accuracy")!, viewFixture("accuracy"));
    try {
      const fig = [...r.container.querySelectorAll("figure.chart-frame")].find((f) => f.getAttribute("data-chart")?.startsWith("Residuals"))!;
      const bars = fig.querySelectorAll('svg.chart rect[role="img"]');
      bars[bars.length - 1].dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
      await flush(3);
      const sel = new URLSearchParams(window.location.search).get("sel");
      expect(sel).toMatch(/^s\./);
      const { decodeSelection } = await import("../../../stores/selection");
      expect(decodeSelection(sel)).toMatchObject({ kind: "filter", column: "pred:resid", op: "between" });
      expect(fig.parentElement!.textContent).toContain("Show on the map");
    } finally {
      r.restore();
    }
  });
});
