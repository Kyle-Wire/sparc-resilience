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
