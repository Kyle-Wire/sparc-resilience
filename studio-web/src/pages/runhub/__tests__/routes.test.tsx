// The run-hub route module (SPEC §3.2, §12.3; phase 2 information architecture): the planner
// tabs Summary · Map · What cools · Heat risk · How much to trust this and the advanced Model
// details tab, each with the server views its status dot covers; the old run tab URLs are not
// declared here any more (they redirect, ../../legacy/routes.ts); the composers render their
// section navigation and the pages of the current section.
import { afterEach, describe, expect, it } from "vitest";
import { clearResources } from "../../../api/resource";
import { RunLayout, RunTabs } from "../../../layouts/RunLayout";
import { buildRegistry, fillPath, matchRoute, navigate, registry, RegistryProvider, type RouteModule } from "../../../router";
import { routes as coreRoutes } from "../../core/routes";
import { routes as labRoutes } from "../../lab/routes";
import { flush, mockFetch, render, waitFor } from "../../../test/render";
import Details, { DETAIL_SECTION_LABELS, detailSectionOf } from "../Details";
import HeatRisk, { heatSectionOf } from "../HeatRisk";
import { RUN_HUB_TABS, routes } from "../routes";
import { viewFixture } from "../__fixtures__/views";
import { runDetail } from "./helpers";

const charts = (root: ParentNode) => [...root.querySelectorAll("figure.chart-frame")].map((f) => f.getAttribute("data-chart") ?? "");

const TABS: Record<string, { label: string; path: string; views: string[]; advanced?: boolean }> = {
  summary: { label: "Summary", path: "/r/:rid", views: ["overview", "heat", "climate"] },
  map: { label: "Map", path: "/r/:rid/map", views: ["map"] },
  effects: { label: "What cools", path: "/r/:rid/effects", views: ["response", "influence", "causal"] },
  heat: { label: "Heat risk", path: "/r/:rid/heat/:section?", views: ["heat", "climate", "planner"] },
  trust: { label: "How much to trust this", path: "/r/:rid/trust", views: ["accuracy", "validation", "uncertainty", "distance"] },
  details: {
    label: "Model details",
    path: "/r/:rid/details/:section?",
    views: ["data", "accuracy", "distance", "influence", "response", "causal", "uncertainty", "track", "provenance", "docs", "files"],
    advanced: true,
  },
};

const OLD = ["data", "accuracy", "distance", "influence", "response", "causal", "scenarios", "climate", "budget", "planner", "uncertainty", "provenance", "docs/:doc?", "files"];

const modules: Record<string, RouteModule> = {
  "./pages/core/routes.ts": { routes: coreRoutes },
  "./pages/lab/routes.ts": { routes: labRoutes },
  "./pages/runhub/routes.ts": { routes },
};

describe("run-hub routes", () => {
  afterEach(() => {
    clearResources();
    navigate("/", { replace: true });
  });

  it("declares the planner tabs and Model details with their views; the old tab URLs are gone", () => {
    expect(RUN_HUB_TABS).toEqual(Object.keys(TABS));
    for (const [id, t] of Object.entries(TABS)) {
      const r = routes.find((x) => x.runTab?.id === id)!;
      expect(r, id).toBeDefined();
      expect(r.path, id).toBe(t.path);
      expect(r.runTab!.label, id).toBe(t.label);
      expect(r.runTab!.views, id).toEqual(t.views);
      expect(!!r.runTab!.advanced, id).toBe(!!t.advanced);
      expect(r.component, id).toBeDefined();
    }
    const paths = routes.map((r) => r.path);
    for (const old of OLD) expect(paths, old).not.toContain(`/r/:rid/${old}`);
    expect(paths).toEqual(expect.arrayContaining(["/r/:rid/details/docs/:doc?", "/p/:pid/compare"]));
  });

  it("matches deep links: Heat risk and Model details sections, a document", () => {
    const reg = buildRegistry(modules);
    expect(reg.problems).toEqual([]);
    expect(matchRoute(reg.compiled, "/r/r1")!.route.title).toBe("Summary");
    expect(matchRoute(reg.compiled, "/r/r1/effects")!.route.title).toBe("What cools");
    expect(matchRoute(reg.compiled, "/r/r1/heat")!.params).toEqual({ rid: "r1" });
    expect(matchRoute(reg.compiled, "/r/r1/heat/future")!.params).toEqual({ rid: "r1", section: "future" });
    expect(matchRoute(reg.compiled, "/r/r1/details")!.route.title).toBe("Model details");
    expect(matchRoute(reg.compiled, "/r/r1/details/reach")!.params).toEqual({ rid: "r1", section: "reach" });
    const doc = matchRoute(reg.compiled, "/r/r1/details/docs/methods")!;
    expect(doc.route.path).toBe("/r/:rid/details/docs/:doc?");
    expect(doc.params).toEqual({ rid: "r1", doc: "methods" });
    expect(matchRoute(reg.compiled, "/r/r1/details/docs")!.route.path).toBe("/r/:rid/details/docs/:doc?");
    expect(matchRoute(reg.compiled, "/p/p_1/compare")!.route.title).toBe("Compare runs");
    expect(fillPath("/r/:rid/heat/:section?", { rid: "r1" })).toBe("/r/r1/heat");
    // unknown or absent sections fall back to the defaults
    expect(heatSectionOf(undefined)).toBe("today");
    expect(heatSectionOf("people")).toBe("people");
    expect(heatSectionOf("bogus")).toBe("today");
    expect(detailSectionOf("/r/:rid/details/:section?", undefined)).toBe("data");
    expect(detailSectionOf("/r/:rid/details/:section?", "runlog")).toBe("runlog");
    expect(detailSectionOf("/r/:rid/details/:section?", "bogus")).toBe("data");
    expect(detailSectionOf("/r/:rid/details/docs/:doc?", undefined)).toBe("docs");
  });

  it("RunTabs shows the seven tabs flat, the Scenario Lab between Heat risk and Trust, Model details last", () => {
    const reg = buildRegistry(modules);
    const { container } = render(<RunTabs rid="r1" tabs={reg.runTabs} outputs={undefined} activeId="heat" />);
    const links = [...container.querySelectorAll("a.run-tab")];
    expect(links.map((a) => a.textContent)).toEqual(["Summary", "Map", "What cools", "Heat risk", "Scenario Lab", "How much to trust this", "Model details"]);
    expect(links.map((a) => a.getAttribute("href"))).toEqual(["/r/r1", "/r/r1/map", "/r/r1/effects", "/r/r1/heat", "/r/r1/lab", "/r/r1/trust", "/r/r1/details"]);
    expect(container.querySelector('a[data-tab="heat"]')!.getAttribute("aria-current")).toBe("page");
    expect(container.querySelector('a[data-tab="details"]')!.classList.contains("advanced")).toBe(true);
  });

  it("the app registry picks the module up and RunLayout highlights the tab of a section URL", async () => {
    for (const id of RUN_HUB_TABS) expect(registry.runTabs.some((t) => t.id === id && routes.includes(t.route)), id).toBe(true);
    const reg = buildRegistry(modules);
    const m = mockFetch({ "GET /api/runs/r_layout": { body: runDetail("r_layout") }, "GET /api/runs/r_layout/outputs": { body: { outputs: [], tabs: [] } } });
    navigate("/r/r_layout/details/docs/methods", { replace: true });
    const { container } = render(
      <RegistryProvider registry={reg}>
        <RunLayout rid="r_layout">
          <p>page</p>
        </RunLayout>
      </RegistryProvider>,
    );
    await flush(4);
    expect(container.querySelector('a[data-tab="details"]')!.getAttribute("aria-current")).toBe("page");
    navigate("/r/r_layout/heat/future", { replace: true });
    await flush(2);
    expect(container.querySelector('a[data-tab="heat"]')!.getAttribute("aria-current")).toBe("page");
    m.restore();
  });

  it("Model details renders its section rail and the current section's page", async () => {
    const m = mockFetch({ "GET /api/runs/r1": { body: runDetail("r1") } });
    try {
      navigate("/r/r1/details/provenance", { replace: true });
      const { container } = render(<Details />);
      const rail = container.querySelector('nav.section-rail[aria-label="Model details"]')!;
      expect([...rail.querySelectorAll("a")].map((a) => [a.textContent, a.getAttribute("href")])).toEqual(
        Object.entries(DETAIL_SECTION_LABELS).map(([id, label]) => [label, `/r/r1/details/${id}`]),
      );
      expect([...rail.querySelectorAll("a")].map((a) => a.textContent)).toEqual([
        "Data & checks",
        "Accuracy",
        "Reach & spatial structure",
        "Response curves",
        "Cause-and-effect audit",
        "Uncertainty sources",
        "Run log",
        "Provenance",
        "Documents",
        "Files",
      ]);
      expect(rail.querySelector('a[aria-current="page"]')!.getAttribute("data-section")).toBe("provenance");
      expect(container.querySelector(".detail-body")!.getAttribute("data-section")).toBe("provenance");
      await waitFor(() => container.querySelector('.detail-body [data-view="provenance"]'), 10000, "the provenance page");
    } finally {
      m.restore();
    }
  });

  it("Model details titles a section's page with its rail label under one intro, and leaves What cools' charts there", async () => {
    const m = mockFetch({ "GET /api/runs/r1": { body: runDetail("r1") }, "GET /api/runs/r1/views/response": { body: viewFixture("response") } });
    try {
      navigate("/r/r1/details/response", { replace: true });
      const { container } = render(<Details />);
      await waitFor(() => container.querySelector('.detail-body [data-view="response"] figure.chart-frame'), 10000, "the response page");
      await flush(4);
      const body = container.querySelector<HTMLElement>(".detail-body")!;
      expect(body.querySelector('[data-view="response"] h2')!.textContent).toBe("Response curves");
      // one intro: the section's line (with its plain-summary link) under the page's title, not the page's own
      expect(body.querySelectorAll(".detail-intro")).toHaveLength(1);
      expect(body.querySelectorAll(".runhub-view > p.cap.prose")).toHaveLength(1);
      const intro = body.querySelector('[data-view="response"] > p.cap.prose')!;
      expect(intro.querySelector('.detail-intro[data-intro="response"]')).not.toBeNull();
      expect(intro.previousElementSibling!.querySelector("h2")!.textContent).toBe("Response curves");
      // nothing above the title: the section opens with its h2, the line follows it
      expect(body.querySelector("h2")!.compareDocumentPosition(body.querySelector(".detail-intro")!) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
      expect(body.querySelector('.detail-intro a[data-plain-link="effects"]')!.getAttribute("href")).toBe("/r/r1/effects");
      // the city-mean curve and the curve shapes are What cools'; the technical extras stay
      const titles = charts(body);
      expect(titles.filter((t) => /dose–response$/.test(t))).toEqual([]);
      expect(titles).not.toContain("Curve shapes across cells");
      expect(titles.some((t) => /requested vs realised dose$/.test(t))).toBe(true);
      expect(titles).toEqual(expect.arrayContaining(["Fold means", "Literature check"]));
    } finally {
      m.restore();
    }
  });

  it("Model details › Reach: one title, the levers' reach and the baseline verdict left to their tabs", async () => {
    const m = mockFetch({
      "GET /api/runs/r1": { body: runDetail("r1") },
      "GET /api/runs/r1/views/distance": { body: viewFixture("distance") },
      "GET /api/runs/r1/views/influence": { body: viewFixture("influence") },
    });
    try {
      navigate("/r/r1/details/reach", { replace: true });
      const { container } = render(<Details />);
      await waitFor(() => container.querySelector('.detail-body [data-view="influence"] .small-multiples'), 10000, "the reach pages");
      await flush(4);
      const body = container.querySelector<HTMLElement>(".detail-body")!;
      expect(body.querySelector('[data-view="distance"] h2')!.textContent).toBe("Reach & spatial structure");
      expect(body.querySelector('[data-view="influence"] h3#view-influence')!.textContent).toBe("How far each predictor reaches");
      expect([...body.querySelectorAll(".detail-intro a[data-plain-link]")].map((a) => a.getAttribute("data-plain-link"))).toEqual(["trust", "effects"]);
      // the intro is the first page's, under its title; the second page has none
      expect(body.querySelector('[data-view="distance"] > p.cap.prose .detail-intro[data-intro="reach"]')).not.toBeNull();
      expect(body.querySelector('[data-view="influence"] > p.cap.prose')).toBeNull();
      // the verdict against the baselines is on Trust (Independent checks): one line here
      expect(body.textContent).not.toContain("stack not distinguishable from hgb_focal");
      expect(body.querySelector('[data-on-trust="validation"] a')!.getAttribute("href")).toBe("/r/r1/trust#validation");
      // the levers' ring kernels are on What cools; the other predictors' stay
      const ring = [...body.querySelectorAll("figure.chart-frame")].find((f) => f.getAttribute("data-chart") === "Ring-kernel β by distance")!;
      const panels = [...ring.querySelectorAll(".small-multiples .panel h4")].map((h) => h.textContent ?? "");
      expect(panels.length).toBeGreaterThan(0);
      for (const lever of ["canopy", "impervious", "albedo"]) expect(panels.some((t) => t.startsWith(lever))).toBe(false);
      expect(ring.querySelector("figcaption a")!.getAttribute("href")).toBe("/r/r1/effects");
    } finally {
      m.restore();
    }
  });

  it("Model details › Run log: a compact record (no outputs grid, commit or run directory) and the job log without the tabs Files and Provenance hold", async () => {
    const detail = runDetail("r1");
    detail.run.last_job_id = "j_1";
    const job = {
      id: "j_1", kind: "run.launch", lane: "heavy", executor: "process", label: "Launch", status: "succeeded", project_id: "p_1", run_id: "r1", study_id: null, scenario_id: null,
      parent_job_id: null, after_job_id: null, priority: 0, params: {}, created_utc: "2026-10-01T21:00:00Z", started_utc: "2026-10-01T21:00:01Z", finished_utc: "2026-10-01T21:20:00Z",
      progress: 1, eta_s: null, eta_lo: null, eta_hi: null, stage: null, current_path: null, exit_code: 0, error: null, blocked: null, result: null, peak_rss_mb: null, threads: null,
    };
    const m = mockFetch({
      "GET /api/runs/r1": { body: detail },
      "GET /api/runs/r1/views/overview": { body: viewFixture("overview") },
      "GET /api/jobs": { body: { items: [job], next_cursor: null } },
    });
    try {
      navigate("/r/r1/details/runlog?tab=outputs", { replace: true });
      const { container } = render(<Details />);
      const body = container.querySelector<HTMLElement>(".detail-body")!;
      const record = await waitFor(() => body.querySelector<HTMLElement>('[data-run-record][data-compact] [aria-label="Run header"]'), 10000, "the run record");
      await flush(6);
      // the composer titles the section, its line under the title
      expect(body.querySelector(".detail-header h2")!.textContent).toBe("Run log");
      expect(body.querySelector(".detail-header p .detail-intro")).not.toBeNull();
      // Provenance holds the commit, versions and run directory; Files holds the outputs
      const header = record.textContent!;
      expect(header).not.toContain("5a04f44");
      expect(header).not.toContain("/w/projects/demo/runs/r1");
      expect(header).toContain("3 × 3 cells at 30 m");
      expect(body.querySelector('[aria-label="Outputs availability"]')).toBeNull();
      // the job log shows the stage times: the run record's timing bar is left out
      expect(charts(body)).not.toContain("Stage timings");
    } finally {
      m.restore();
    }
  });

  it("Heat risk renders its sub-tabs (Today · Who is exposed · Future climate), Today by default", async () => {
    const m = mockFetch({ "GET /api/runs/r1": { body: runDetail("r1") } });
    try {
      navigate("/r/r1/heat", { replace: true });
      const { container } = render(<HeatRisk />);
      const nav = container.querySelector('nav.section-nav[aria-label="Heat risk"]')!;
      expect([...nav.querySelectorAll("a")].map((a) => [a.textContent, a.getAttribute("href")])).toEqual([
        ["Today", "/r/r1/heat/today"],
        ["Who is exposed", "/r/r1/heat/people"],
        ["Future climate", "/r/r1/heat/future"],
      ]);
      expect(nav.querySelector('a[aria-current="page"]')!.textContent).toBe("Today");
      expect(container.querySelector("#adapt")).toBeNull();
      navigate("/r/r1/heat/future", { replace: true });
      await flush(2);
      expect(nav.querySelector('a[aria-current="page"]')!.textContent).toBe("Future climate");
      // "Can adaptation keep up?" (the Lab's Climate × adaptation) follows on the Future climate page
      const adapt = await waitFor(() => container.querySelector("#adapt h2"), 5000, "Can adaptation keep up?");
      expect(adapt.textContent).toBe("Can adaptation keep up?");
    } finally {
      m.restore();
    }
  });
});
