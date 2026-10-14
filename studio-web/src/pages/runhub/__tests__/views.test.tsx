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
import { byText, click, flush, mockFetch, render, waitFor, type MockHandler } from "../../../test/render";
import { olderCodeFixture, sectionKeys, viewFixture } from "../__fixtures__/views";
import Accuracy from "../Accuracy";
import Budget from "../Budget";
import Causal from "../Causal";
import Climate from "../Climate";
import DataQa from "../DataQa";
import Heat from "../Heat";
import Distance from "../Distance";
import Influence from "../Influence";
import Overview from "../Overview";
import Planner, { PlannerGisGallery, PlannerWhereToAct } from "../Planner";
import Provenance from "../Provenance";
import Response from "../Response";
import Scenarios from "../Scenarios";
import Uncertainty from "../Uncertainty";
import { binResponse, LAYER_GROUPS, layerMeta, mapRoutes, runDetail, unframedCharts } from "./helpers";

/** The planner pack's parts together: Who is exposed (the page), Where to act (What cools) and the GIS gallery (Reports). */
function PlannerPack() {
  return (
    <>
      <Planner />
      <PlannerWhereToAct />
      <PlannerGisGallery />
    </>
  );
}

const PAGES: { view: ViewName; tab: string; Page: ComponentType }[] = [
  { view: "overview", tab: "", Page: Overview },
  { view: "data", tab: "data", Page: DataQa },
  { view: "accuracy", tab: "accuracy", Page: Accuracy },
  { view: "distance", tab: "distance", Page: Distance },
  { view: "influence", tab: "influence", Page: Influence },
  { view: "response", tab: "response", Page: Response },
  { view: "scenarios", tab: "scenarios", Page: Scenarios },
  { view: "climate", tab: "climate", Page: Climate },
  { view: "heat", tab: "heat", Page: Heat },
  { view: "causal", tab: "causal", Page: Causal },
  { view: "budget", tab: "budget", Page: Budget },
  { view: "planner", tab: "planner", Page: PlannerPack },
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

  // The Heat view is computed by Studio from any finished run (never an older-code output).
  for (const p of PAGES.filter((x) => x.view !== "heat")) {
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
        // optional DAG audit (off unless configured, so null is not "older code"). The accuracy
        // view's forcing card is shown on Data & checks (input QA), not on Accuracy.
        const folded = new Set(["live", "has_detail", "caption", "status", "thresholds", "hex_files", "sites", "pairs", "dag_audit", ...(p.view === "accuracy" ? ["forcing"] : [])]);
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

  it("Heat › Today shows the brief and today's residents by category, not the futures or verdicts, and re-asks the server for a what-if dewpoint", async () => {
    const page = PAGES.find((p) => p.view === "heat")!;
    const rid = `r_view${seq + 1}`;
    const humid = viewFixture("heat");
    humid.sections.humidity!.dewpoint_C = 21;
    humid.sections.humidity!.user_set = true;
    humid.sections.humidity!.source = "your setting";
    const r = await renderView(page, viewFixture("heat"), { [`GET /api/runs/${rid}/views/heat?dewpoint_C=21`]: { body: humid } });
    try {
      const text = r.container.textContent ?? "";
      const brief = r.container.querySelector('[aria-label="Heat brief"]')!.textContent ?? "";
      expect(brief).toContain("Extreme caution or worse");
      // the futures live under Future climate, the verdicts on How much to trust this
      expect(brief).not.toContain("As the climate warms");
      expect(brief).not.toContain("How sure");
      expect(r.container.querySelector('[aria-label="Heat key numbers"]')!.textContent).not.toContain("SSP");
      expect(text).toContain("Residents by heat-risk category");
      expect(text).not.toContain("Not established");
      expect(r.container.querySelector('[aria-label="Exposure in each climate future"]')).toBeNull();
      click(byText(r.container, "button", "Humid (dew point 70 °F)"));
      await waitFor(() => r.calls.some((c) => c.url.endsWith("/views/heat?dewpoint_C=21")), 3000, "what-if request");
      await waitFor(() => r.container.textContent?.includes("your setting"), 3000, "what-if view");
      // the what-if dewpoint is page state, so Future climate uses it too
      expect(new URLSearchParams(window.location.search).get("dewpoint")).toBe("21");
      expect(r.container.querySelector('[aria-label="Heat brief"]')).not.toBeNull();
    } finally {
      r.restore();
    }
  });

  it("Heat asks for a dewpoint when the run has no campaign humidity", async () => {
    const page = PAGES.find((p) => p.view === "heat")!;
    const vm = viewFixture("heat");
    const h = vm.sections.humidity!;
    Object.assign(h, { dewpoint_C: null, source: null, campaign_dewpoint_C: null, campaign_source: null, needs_input: true, rh_range: null });
    for (const k of ["brief", "kpis", "today", "futures", "hist"] as const) vm.sections[k] = null;
    const r = await renderView(page, vm);
    try {
      expect(r.container.textContent).toContain("no humidity reading for the measurement day");
      expect(byText(r.container, "button", "Typical summer (dew point 61 °F)")).not.toBeNull();
      expect(r.container.textContent).not.toContain("could not be displayed");
    } finally {
      r.restore();
    }
  });

  it("Scenarios and Uncertainty show each scenario's verdict", async () => {
    const sc = await renderView(PAGES.find((p) => p.view === "scenarios")!, viewFixture("scenarios"));
    try {
      expect(sc.container.textContent).toContain("Direction only");
    } finally {
      sc.restore();
    }
    const un = await renderView(PAGES.find((p) => p.view === "uncertainty")!, viewFixture("uncertainty"));
    try {
      expect(un.container.querySelector('[aria-label="Verdict per scenario"]')!.textContent).toContain("Robust");
    } finally {
      un.restore();
    }
  });

  it("Model details keeps the technical names and explains each jargon heading with a glossary ⓘ or Term (no native titles)", async () => {
    const terms = (root: Element) => new Set([...root.querySelectorAll<HTMLElement>("button.infotip[data-term], button.term[data-term]")].map((b) => b.dataset.term));
    const expected: [ViewName, string[], string[]][] = [
      ["accuracy", ["Stacker", "Physics card", "CV design"], ["stacker", "blend_weights", "physics_card", "advection", "cv_design", "residual", "interval_honesty"]],
      ["distance", ["Stack vs baselines"], ["distance_skill", "baselines"]],
      ["causal", ["Audit verdicts", "Sensitivity to hidden confounding"], ["causal_audit", "cate", "e_value", "dose_response", "partial_dependence"]],
      ["uncertainty", ["What the evidence supports"], ["envelope", "estimation", "specification", "attribution", "causal_band", "verdict_robust"]],
    ];
    for (const [view, headings, ids] of expected) {
      const r = await renderView(PAGES.find((p) => p.view === view)!, viewFixture(view) as never);
      try {
        expect(r.container.querySelector("[title]"), view).toBeNull();
        const text = r.container.textContent ?? "";
        for (const h of headings) expect(text, `${view}: ${h}`).toContain(h);
        const found = terms(r.container);
        for (const id of ids) expect(found.has(id), `${view}: ${id}`).toBe(true);
        // an ⓘ sits beside a heading, never inside it
        expect(r.container.querySelector("h2 button.infotip, h3 button.infotip"), view).toBeNull();
      } finally {
        r.restore();
        r.unmount();
      }
    }
  });

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

  it("Data & checks holds the input QA: the forcing card beside the QA flags, the cell size in the grid-fill tile, the histogram full width without zones", async () => {
    const page = PAGES.find((p) => p.view === "data")!;
    const data = viewFixture("data");
    data.sections.qa_tiles = [...(data.sections.qa_tiles ?? []), { id: "cell_m", label: "Cell size", value: 30, unit: "m", decimals: 1 } as NonNullable<typeof data.sections.qa_tiles>[number]];
    data.sections.zone_counts = [];
    const rid = `r_view${seq + 1}`;
    const r = await renderView(page, data, { [`GET /api/runs/${rid}/views/accuracy`]: { body: viewFixture("accuracy") } });
    try {
      const forcing = await waitFor(() => r.container.querySelector('[data-forcing] [aria-label="Forcing"]'), 5000, "the forcing card");
      expect(forcing.closest("[data-forcing]")!.textContent).toContain("QA flags");
      const tiles = [...r.container.querySelectorAll('[aria-label="Data QA"] .kpi')];
      expect(tiles.map((t) => t.querySelector(".label")!.textContent)).not.toContain("Cell size");
      const fill = tiles.find((t) => t.querySelector(".label")!.textContent === "Grid fill")!;
      expect(fill.querySelector(".note")!.textContent).toBe("of 30 m cells");
      // 7 tiles in a row of at most 4: no lone tile on the last line
      expect(r.container.querySelector('[aria-label="Data QA"]')!.getAttribute("data-cols")).toBe("4");
      const noZones = r.container.querySelector("[data-no-zones]")!;
      expect(noZones.querySelector('figure[data-chart="Fractional part of the target"]')).not.toBeNull();
      expect(noZones.querySelector(":scope > p.cap")!.textContent).toBe("This run has no zone column.");
      expect(r.container.querySelector(".grid2 figure[data-chart=\"Fractional part of the target\"]")).toBeNull();
    } finally {
      r.restore();
    }
  });

  it("Accuracy leaves the forcing card to Data & checks", async () => {
    const r = await renderView(PAGES.find((p) => p.view === "accuracy")!, viewFixture("accuracy"));
    try {
      expect(r.container.querySelector('[aria-label="Forcing"]')).toBeNull();
      expect(r.container.querySelector('[aria-label="CV design"]')).not.toBeNull();
    } finally {
      r.restore();
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
        const fig = [...r.container.querySelectorAll("figure.chart-frame")].find((f) => f.getAttribute("data-chart")?.startsWith("How much cooling each budget buys"));
        expect(fig).toBeDefined();
        expect(fig!.querySelector("figcaption")!.textContent).toContain(caption);
      } finally {
        r.restore();
        r.unmount();
      }
    }
  });

  it("views leave the run-wide caveats to the Summary (only the full Overview lists them)", async () => {
    for (const view of ["response", "climate", "distance"] as const) {
      const vm = viewFixture(view);
      vm.caveats = ["Generic physics forcing: the run did not use the campaign day's radiation and wind."];
      const r = await renderView(PAGES.find((p) => p.view === view)!, vm);
      try {
        expect(r.container.textContent, view).not.toContain("Generic physics forcing");
        expect(r.container.textContent, view).not.toMatch(/caveats? for this/);
      } finally {
        r.restore();
        r.unmount();
        clearResources();
      }
    }
  });

  it("Budget's worded key numbers (constraint, objective) read as one caption line below the number tiles", async () => {
    const r = await renderView(PAGES.find((p) => p.view === "budget")!, viewFixture("budget"));
    try {
      const tiles = r.container.querySelector('[aria-label="Budget"]')!;
      const labels = [...tiles.querySelectorAll(".kpi .label")].map((l) => l.textContent);
      expect(labels).toEqual(["Cells treated", "Mean dose", "Planned cooling", "Realised cooling", "Cost", "Gini"]);
      expect(r.container.querySelector("[data-kpi-text]")!.textContent).toBe(
        "Constraint: plantable space (WorldCover open land + 20% of built-up area) · Objective: total cooling",
      );
    } finally {
      r.restore();
    }
  });

  it("Budget explains core's 'no positive-benefit segments' instead of reporting it missing", async () => {
    const vm = viewFixture("budget");
    vm.sections.status = "no positive-benefit segments";
    const r = await renderView(PAGES.find((p) => p.view === "budget")!, vm);
    try {
      expect(r.container.querySelector('[data-status="no positive-benefit segments"]')!.textContent).toContain("No place cools anything, on the spot or nearby, with this lever");
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
    const rerun = { kind: "open" as const, label: "Re-run with the distance test", method: "GET" as const, path: `/p/p_1/launch?from=${rid}` };
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
      expect(note.textContent).toContain("turned off in the project settings");
      expect(note.querySelector("button")!.textContent).toBe("Re-run with the distance test");
    } finally {
      r.restore();
    }
  });

  it("Distance under Model details leaves the skipped-curve notice (and its re-run) and the baseline verdict to Trust, one line each", async () => {
    const rid = `r_view${seq + 1}`;
    const vm = viewFixture("distance");
    vm.sections.curve = null;
    const detail = runDetail(rid);
    detail.stages = [
      { id: "baselines", state: "done", seconds: 2, source: "manifest", reason: null },
      { id: "cv_curve", state: "skipped", seconds: null, source: "manifest", reason: "disabled_by_config:cv.distance_curve.enabled" },
    ];
    const overview = viewFixture("overview");
    overview.sections.outputs_grid = [
      { id: "cv_distance", label: "Skill vs distance", group: "trust", state: "missing", produced_by: "stage:cv_curve", view: "distance", action: { kind: "open", label: "Re-run with the distance test", method: "GET", path: `/p/p_1/launch?from=${rid}` } },
    ];
    const r = await renderView({ view: "distance", tab: "details/reach", Page: () => <Distance detailsOnly /> }, vm, {
      [`GET /api/runs/${rid}`]: { body: detail },
      [`GET /api/runs/${rid}/views/overview`]: { body: overview },
    });
    try {
      await waitFor(() => r.container.querySelector('[data-on-trust="distance"]'), 3000, "the line to Trust");
      expect(r.container.querySelector('[data-on-trust="distance"] a')!.getAttribute("href")).toBe(`/r/${rid}/trust#distance`);
      expect(r.container.querySelector('[data-on-trust="validation"] a')!.getAttribute("href")).toBe(`/r/${rid}/trust#validation`);
      expect(r.container.textContent).not.toContain("Re-run with the distance test");
      expect(r.container.textContent).not.toContain("cv.distance_curve.enabled");
      expect(r.container.textContent).not.toContain("stack not distinguishable");
      expect(r.container.querySelectorAll("[data-absent], [data-older-code]").length).toBe(0);
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

  it("Planner equity bars are labelled as average cooling in the target unit, by fifth of places ranked by the measure", async () => {
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
      expect(head.some((h) => h.startsWith("Average cooling") && h.includes("°F"))).toBe(true);
      expect(head.some((h) => h.startsWith("population density"))).toBe(false);
      expect(fig.textContent).toContain("Units: °F (positive = cooler)");
      expect(fig.querySelector("svg.chart")!.textContent).toContain("Average cooling (°F)");
      // quintiles as fifths of places, named at the ends by the measure
      const rows = [...fig.querySelectorAll("details.table-view tbody tr")].map((tr) => tr.querySelector("td")?.textContent ?? "");
      expect(rows).toEqual(["Lowest density", "2nd fifth", "Middle fifth", "4th fifth", "Highest density"]);
      // the concentration index explains itself in a tooltip chip, not a native title
      const block = fig.closest("section.card")!;
      expect(block.querySelector("[title]")).toBeNull();
      expect(block.querySelectorAll("[data-tip-chip]").length).toBe(2);
      expect(block.textContent).toContain("Who gets the cooling:");
      expect(block.textContent).toContain("Share of residents aged 60+: −0.05");
    } finally {
      r.restore();
    }
  });

  it("a missing view shows one plain sentence and the server's action as a button, never the list of output keys", async () => {
    const vm = olderCodeFixture("planner");
    vm.availability = "missing";
    const planner = { kind: "run_job" as const, label: "Run planner pack", method: "POST" as const, path: "/api/runs/x/actions/planner", body: {} };
    vm.missing = ["planner", "planner_cells", "planner_hex:250", "logger_sites"].map((output) => ({ output, produced_by: "post:planner", action: planner }));
    const r = await renderView(PAGES.find((p) => p.view === "planner")!, vm);
    try {
      expect(r.container.textContent).toContain("Run the planner pack to see where to act, who is exposed, and the GIS files.");
      expect(r.container.textContent).not.toContain("produced by");
      expect(r.container.textContent).not.toContain("planner_cells");
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
