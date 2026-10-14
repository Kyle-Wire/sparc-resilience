import { afterEach, describe, expect, it, vi } from "vitest";
import { clearResources } from "../api/resource";
import { RUN_TAB_IDS, type Availability, type Project, type RunOutputs, type RunTabId } from "../api/types";
import { ADVANCED_NAV_KEY, currentProjectNav, ProjectNav } from "../layouts/AppShell";
import { RunLayout, RunTabs, tabStatus } from "../layouts/RunLayout";
import { buildRegistry, fillPath, matchRoute, navigate, registry as appRegistry, RegistryProvider, type RouteModule } from "../router";
import { App } from "../App";
import { StreamManager, setStreams, type EventSourceLike } from "../api/sse";
import { click, flush, mockFetch, render, waitFor } from "./render";

/** An EventSource that never connects (the shell opens the global stream). */
class SilentES implements EventSourceLike {
  onopen: ((ev: Event) => void) | null = null;
  onerror: ((ev: Event) => void) | null = null;
  onmessage: ((ev: MessageEvent) => void) | null = null;
  addEventListener() {}
  close() {}
}

// The fixture route modules live under a test glob, exactly like src/pages/*/routes.ts.
const fixtureModules = import.meta.glob<RouteModule>("./fixtures/pages/*/routes.ts", { eager: true });

const project: Project = {
  id: "p_1", slug: "demo", name: "Demo city", dir: "/w/projects/demo", config_path: "/w/projects/demo/config.yml", template: "synthetic_demo",
  demo: true, active_run_id: null, archived: false, created_utc: "2026-10-01T00:00:00Z", updated_utc: "2026-10-01T00:00:00Z",
  report: { title: null, place: null, area: null }, headline_scenario: null, cost_model: {}, n_runs: 0, last_run: null, active_jobs: 0,
  readiness_score: { done: 3, total: 10 },
};

describe("route registry", () => {
  afterEach(() => {
    clearResources();
    navigate("/", { replace: true });
  });

  it("picks up fixture routes.ts modules from a glob", () => {
    expect(Object.keys(fixtureModules).sort()).toEqual(["./fixtures/pages/alpha/routes.ts", "./fixtures/pages/beta/routes.ts"]);
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const reg = buildRegistry(fixtureModules);
    warn.mockRestore();
    const paths = reg.routes.map((r) => r.path);
    expect(paths).toContain("/r/:rid/accuracy");
    expect(paths).toContain("/r/:rid/lab/library");
    expect(paths).toContain("/r/:rid/old");
    // One owner per tab id: the duplicate "files" from beta is reported and ignored.
    expect(reg.problems.some((p) => p.includes('run tab "files" already declared'))).toBe(true);
    expect(reg.runTabs.filter((t) => t.id === "files")).toHaveLength(1);
    // A tab listing an unknown server view is reported and ignored.
    expect(reg.problems.some((p) => p.includes('run tab "bogus" lists unknown view "nope"'))).toBe(true);
    expect(reg.runTabs.some((t) => t.id === "bogus")).toBe(false);
  });

  it("orders run tabs flat by order, the advanced ones (Model details) last", () => {
    const reg = buildRegistry(fixtureModules);
    expect(reg.runTabs.map((t) => `${t.advanced ? "adv:" : ""}${t.id}`)).toEqual(["overview", "causal", "accuracy", "trust", "lab", "adv:docs", "adv:files"]);
    expect(reg.projectNav.map((n) => n.id)).toEqual(["runs", "lab", "tools"]);
  });

  it("the app registry includes the foundation 404 route", () => {
    expect(appRegistry.routes.some((r) => r.path === "*")).toBe(true);
  });

  it("the app registry declares the seven run tabs in order, Model details last and advanced, with no problems", () => {
    expect(appRegistry.problems).toEqual([]);
    expect(appRegistry.runTabs.map((t) => [t.id, t.label, fillPath(t.route.path, { rid: "r1" })])).toEqual([
      ["summary", "Summary", "/r/r1"],
      ["map", "Map", "/r/r1/map"],
      ["effects", "What cools", "/r/r1/effects"],
      ["heat", "Heat risk", "/r/r1/heat"],
      ["lab", "Scenario Lab", "/r/r1/lab"],
      ["trust", "How much to trust this", "/r/r1/trust"],
      ["details", "Model details", "/r/r1/details"],
    ]);
    expect(appRegistry.runTabs.filter((t) => t.advanced).map((t) => t.id)).toEqual(["details"]);
    const views = Object.fromEntries(appRegistry.runTabs.map((t) => [t.id, t.views]));
    expect(views).toEqual({
      summary: ["overview", "heat", "climate"],
      map: ["map"],
      effects: ["response", "influence", "causal"],
      heat: ["heat", "climate", "planner"],
      lab: ["lab", "scenarios", "budget"],
      trust: ["accuracy", "validation", "uncertainty", "distance"],
      details: ["data", "accuracy", "distance", "influence", "response", "causal", "uncertainty", "track", "provenance", "docs", "files"],
    });
    // every server view id has a home tab
    for (const v of RUN_TAB_IDS) expect(appRegistry.runTabs.some((t) => t.views.includes(v)), v).toBe(true);
  });

  it("tabStatus aggregates a tab's views: all ready, worst when none is available, partial when mixed", () => {
    const out = (rows: [RunTabId, Availability][]): RunOutputs => ({ outputs: [], tabs: rows.map(([id, availability]) => ({ id, availability, missing: [] })) });
    expect(tabStatus(out([]), ["overview"])).toBeNull();
    expect(tabStatus(out([["overview", "ready"], ["heat", "ready"]]), ["overview", "heat", "uncertainty"])!.availability).toBe("ready");
    expect(tabStatus(out([["overview", "running"], ["heat", "missing"]]), ["overview", "heat"])!.availability).toBe("missing");
    expect(tabStatus(out([["overview", "running"], ["heat", "stale"]]), ["overview", "heat"])!.availability).toBe("stale");
    expect(tabStatus(out([["overview", "running"], ["heat", "running"]]), ["overview", "heat"])!.availability).toBe("running");
    expect(tabStatus(out([["overview", "ready"], ["heat", "missing"]]), ["overview", "heat"])).toEqual({ availability: "partial", reason: "heat: not available yet" });
    expect(tabStatus(out([["overview", "partial"]]), "overview")!.availability).toBe("partial");
    expect(tabStatus(out([["overview", "ready"], ["heat", "running"]]), ["overview", "heat"])!.availability).toBe("partial");
  });

  it("RunTabs renders one flat row with aggregated status dots, Model details after a divider", () => {
    const reg = buildRegistry(fixtureModules);
    const outputs: RunOutputs = {
      outputs: [],
      tabs: [
        { id: "overview", availability: "ready", missing: [] },
        { id: "accuracy", availability: "running", missing: [] },
        { id: "causal", availability: "missing", missing: [{ output: "causal", produced_by: "stage:S6", action: null }] },
        { id: "validation", availability: "ready", missing: [] },
        { id: "uncertainty", availability: "missing", missing: [] },
        { id: "lab", availability: "running", missing: [] },
        { id: "scenarios", availability: "running", missing: [] },
        { id: "files", availability: "ready", missing: [] },
      ],
    };
    const { container } = render(<RunTabs rid="r1" tabs={reg.runTabs} outputs={outputs} activeId="accuracy" />);
    const nav = container.querySelector('nav[aria-label="Run views"]')!;
    expect(nav.querySelectorAll('[role="group"]')).toHaveLength(0);
    const links = [...nav.querySelectorAll("a.run-tab")];
    expect(links.map((a) => a.getAttribute("data-tab"))).toEqual(["overview", "causal", "accuracy", "trust", "lab", "docs", "files"]);
    expect(links.map((a) => a.getAttribute("href"))).toEqual(["/r/r1", "/r/r1/causal", "/r/r1/accuracy", "/r/r1/trust", "/r/r1/lab", "/r/r1/docs", "/r/r1/files"]);
    // the divider sits right before the first advanced tab, which is styled quieter
    const divider = nav.querySelector(".run-tab-divider")!;
    expect(divider.getAttribute("aria-hidden")).toBe("true");
    expect(divider.nextElementSibling!.getAttribute("data-tab")).toBe("docs");
    expect([...nav.querySelectorAll("a.run-tab.advanced")].map((a) => a.getAttribute("data-tab"))).toEqual(["docs", "files"]);
    const acc = container.querySelector('a[data-tab="accuracy"]')!;
    expect(acc.getAttribute("aria-current")).toBe("page");
    expect(acc.querySelector(".tab-dot")!.getAttribute("data-a")).toBe("running");
    expect(acc.textContent).toContain("being computed");
    const causal = container.querySelector('a[data-tab="causal"]')!;
    expect(causal.getAttribute("title")).toContain("produced by stage:S6");
    // mixed views → partial; all running → running
    expect(container.querySelector('a[data-tab="trust"] .tab-dot')!.getAttribute("data-a")).toBe("partial");
    expect(container.querySelector('a[data-tab="lab"] .tab-dot')!.getAttribute("data-a")).toBe("running");
    // Tabs without a server entry show no dot.
    expect(container.querySelector('a[data-tab="docs"] .tab-dot')).toBeNull();
  });

  it("RunLayout fetches the run and its outputs and builds the tab bar from the registry", async () => {
    const reg = buildRegistry(fixtureModules);
    const fetch = mockFetch({
      "GET /api/runs/r1": { body: { run: { id: "r1", project_id: "p_1", label: "Fast run", status: "complete", mode: "fast", coarse_m: null, demo: true }, header: { name: "demo" } } },
      "GET /api/runs/r1/outputs": { body: { outputs: [], tabs: [{ id: "overview", availability: "ready", missing: [] }, { id: "docs", availability: "stale", missing: [] }] } },
    });
    navigate("/r/r1/lab/library");
    const { container } = render(
      <RegistryProvider registry={reg}>
        <RunLayout rid="r1">
          <p>child</p>
        </RunLayout>
      </RegistryProvider>,
    );
    await flush(6);
    fetch.restore();
    expect(container.querySelector("h1")!.textContent).toBe("Fast run");
    expect(container.textContent).toContain("DEMO");
    expect(container.textContent).toContain("FAST");
    const links = [...container.querySelectorAll("a.run-tab")].map((a) => a.getAttribute("data-tab"));
    expect(links).toEqual(["overview", "causal", "accuracy", "trust", "lab", "docs", "files"]);
    // A sub-route of the Lab highlights the Lab tab.
    expect(container.querySelector('a[data-tab="lab"]')!.getAttribute("aria-current")).toBe("page");
    expect(container.querySelector('a[data-tab="docs"] .tab-dot')!.getAttribute("data-a")).toBe("stale");
    expect(fetch.calls.map((c) => c.url).sort()).toEqual(["/api/runs/r1", "/api/runs/r1/outputs"]);
  });

  it("App keeps the run layout mounted across tab navigation, so the followed tab keeps focus", async () => {
    const reg = buildRegistry(fixtureModules);
    const fetch = mockFetch({
      "GET /api/runs/r1": { body: { run: { id: "r1", project_id: null, label: "Fast run", status: "complete", mode: "fast", coarse_m: null, demo: false }, header: { name: "demo" } } },
      "GET /api/runs/r1/outputs": { body: { outputs: [], tabs: [] } },
      "GET /api/jobs": { body: { items: [], next_cursor: null } },
      "GET /api/health": { body: { ok: true, engine: { state: "absent" } } },
      "GET /api/projects": { body: [] },
    });
    setStreams(new StreamManager({ EventSource: SilentES, frame: (cb) => cb() }));
    try {
      navigate("/r/r1/accuracy");
      const { container } = render(
        <RegistryProvider registry={reg}>
          <App />
        </RegistryProvider>,
      );
      await waitFor(() => container.querySelector('[data-testid="fixture-page"]')?.textContent?.startsWith("Accuracy"), 10000, "the accuracy page");
      const nav = container.querySelector("nav.run-tabs")!;
      const causal = container.querySelector<HTMLAnchorElement>('a[data-tab="causal"]')!;
      causal.focus();
      click(causal);
      await waitFor(() => container.querySelector('[data-testid="fixture-page"]')?.textContent?.startsWith("Causal"), 10000, "the causal page");
      expect(window.location.pathname).toBe("/r/r1/causal");
      expect(container.querySelector("nav.run-tabs")).toBe(nav); // not remounted
      expect(document.activeElement).toBe(causal);
      expect(causal.getAttribute("aria-current")).toBe("page");
    } finally {
      fetch.restore();
      setStreams(null);
    }
  });

  it("App follows a redirect route: replaces the history entry, keeps the query and the hash", async () => {
    const reg = buildRegistry(fixtureModules);
    const fetch = mockFetch({
      "GET /api/runs/r1": { body: { run: { id: "r1", project_id: null, label: "Fast run", status: "complete", mode: "fast", coarse_m: null, demo: false }, header: { name: "demo" } } },
      "GET /api/runs/r1/outputs": { body: { outputs: [], tabs: [] } },
      "GET /api/jobs": { body: { items: [], next_cursor: null } },
      "GET /api/health": { body: { ok: true, engine: { state: "absent" } } },
      "GET /api/projects": { body: [] },
    });
    setStreams(new StreamManager({ EventSource: SilentES, frame: (cb) => cb() }));
    try {
      navigate("/r/r1/old?x=1#sec");
      const depth = window.history.length;
      const { container } = render(
        <RegistryProvider registry={reg}>
          <App />
        </RegistryProvider>,
      );
      await waitFor(() => container.querySelector('[data-testid="fixture-page"]')?.textContent?.startsWith("Accuracy"), 10000, "the redirect target");
      expect(window.location.pathname + window.location.search + window.location.hash).toBe("/r/r1/accuracy?x=1#sec");
      expect(window.history.length).toBe(depth); // replaced, not pushed
    } finally {
      fetch.restore();
      setStreams(null);
    }
  });

  it("marks one project nav entry current: its own or claimed route, else the longest target prefix; the root only on itself", () => {
    const reg = appRegistry;
    const entries = reg.projectNav.flatMap((n) => {
      const to = n.to("p_1", project);
      return to === null ? [] : [{ id: n.id, to, route: n.route, claims: n.claims }];
    });
    const at = (path: string) => currentProjectNav(entries, path, matchRoute(reg.compiled, path)?.route ?? null);
    expect(at("/p/p_1")).toBe("overview");
    expect(at("/p/p_1/setup")).toBe("setup");
    expect(at("/p/p_1/setup/levers")).toBe("setup");
    expect(at("/p/p_1/setup/inputs")).toBe("setup");
    expect(at("/p/p_1/launch")).toBe("setup");
    expect(at("/p/p_1/runs")).toBe("runs");
    expect(at("/p/p_1/exports")).toBe("reports");
    expect(at("/p/p_1/config")).toBe("config");
    expect(at("/p/p_1/studies")).toBe("studies");
    expect(at("/jobs")).toBe("activity");
    expect(at("/jobs/j_1")).toBe("activity");
    expect(at("/settings")).toBeNull();
    // run pages: the Lab's pages highlight Scenario Lab, every other run page Runs
    expect(at("/r/r1/lab")).toBe("lab");
    expect(at("/r/r1/lab/library")).toBe("lab");
    expect(at("/r/r1/lab/s/sc_1")).toBe("lab");
    expect(at("/r/r1")).toBe("runs");
    expect(at("/r/r1/map")).toBe("runs");
    expect(at("/r/r1/details/runlog")).toBe("runs");
    expect(at("/r/r1/heat/future")).toBe("runs");
  });

  it("ProjectNav renders declared entries, hides null targets and greys disabled ones", () => {
    const reg = buildRegistry(fixtureModules);
    const { container } = render(<ProjectNav registry={reg} project={project} />);
    expect([...container.querySelectorAll("ul.shell-nav > li > a")].map((a) => a.getAttribute("href"))).toEqual(["/p/p_1/runs"]);
    const disabled = container.querySelector('[aria-disabled="true"]')!;
    expect(disabled.textContent).toBe("Scenario Lab");
    expect(container.querySelector('[role="tooltip"]')!.textContent).toBe("Needs a run with a checkpoint");
  });

  it("ProjectNav keeps the Advanced group collapsed by default, remembers it open, and opens it on its own pages", async () => {
    const reg = buildRegistry(fixtureModules);
    try {
      window.localStorage.removeItem(ADVANCED_NAV_KEY);
    } catch {
      /* ignore */
    }
    navigate("/p/p_1/runs", { replace: true });
    const first = render(<ProjectNav registry={reg} project={project} />);
    const toggle = first.container.querySelector<HTMLButtonElement>("button.shell-nav-toggle")!;
    expect(toggle.textContent).toContain("Advanced");
    expect(toggle.getAttribute("aria-expanded")).toBe("false");
    expect(first.container.querySelector("#shell-nav-advanced")).toBeNull();
    click(toggle);
    expect(toggle.getAttribute("aria-expanded")).toBe("true");
    expect(first.container.querySelector('#shell-nav-advanced a')!.getAttribute("href")).toBe("/p/p_1/tools");
    expect(window.localStorage.getItem(ADVANCED_NAV_KEY)).toBe("1");
    first.unmount();
    // remembered across mounts
    const again = render(<ProjectNav registry={reg} project={project} />);
    expect(again.container.querySelector("button.shell-nav-toggle")!.getAttribute("aria-expanded")).toBe("true");
    click(again.container.querySelector("button.shell-nav-toggle"));
    expect(window.localStorage.getItem(ADVANCED_NAV_KEY)).toBe("0");
    again.unmount();
    // a page of the group (here a claimed one) opens it and marks its entry current
    navigate("/p/p_1/tools/x", { replace: true });
    const onPage = render(<ProjectNav registry={reg} project={project} />);
    await flush(1);
    expect(onPage.container.querySelector("button.shell-nav-toggle")!.getAttribute("aria-expanded")).toBe("true");
    expect(onPage.container.querySelector('#shell-nav-advanced a[aria-current="page"]')!.textContent).toBe("Tools");
    // and it still collapses on a click there
    click(onPage.container.querySelector("button.shell-nav-toggle"));
    expect(onPage.container.querySelector("button.shell-nav-toggle")!.getAttribute("aria-expanded")).toBe("false");
    expect(onPage.container.querySelector("#shell-nav-advanced")).toBeNull();
  });

  it("the app's project nav: Overview · Set up · Runs · Scenario Lab · Reports, then Advanced: Studies · Config editor · Activity", () => {
    const main = appRegistry.projectNav.filter((n) => n.group !== "advanced").map((n) => n.label);
    const adv = appRegistry.projectNav.filter((n) => n.group === "advanced").map((n) => n.label);
    expect(main).toEqual(["Overview", "Set up", "Runs", "Scenario Lab", "Reports"]);
    expect(adv).toEqual(["Studies", "Config editor", "Activity"]);
    const to = Object.fromEntries(appRegistry.projectNav.map((n) => [n.id, n.to("p_1", { ...project, active_run_id: "r9" })]));
    expect(to).toEqual({
      overview: "/p/p_1",
      setup: "/p/p_1/setup/data",
      runs: "/p/p_1/runs",
      lab: "/r/r9/lab",
      reports: "/p/p_1/exports",
      studies: "/p/p_1/studies",
      config: "/p/p_1/config",
      activity: "/jobs",
    });
  });
});
