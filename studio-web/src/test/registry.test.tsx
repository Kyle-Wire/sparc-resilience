import { afterEach, describe, expect, it, vi } from "vitest";
import { clearResources } from "../api/resource";
import type { Project, RunOutputs } from "../api/types";
import { currentProjectNav, ProjectNav } from "../layouts/AppShell";
import { RunLayout, RunTabs } from "../layouts/RunLayout";
import { buildRegistry, matchRoute, navigate, registry as appRegistry, RegistryProvider, type RouteModule } from "../router";
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
    // One owner per tab id: the duplicate "files" from beta is reported and ignored.
    expect(reg.problems.some((p) => p.includes('run tab "files" already declared'))).toBe(true);
    expect(reg.runTabs.filter((t) => t.id === "files")).toHaveLength(1);
  });

  it("orders run tabs by group (Model, Effects, Decisions, Trust, Run) then order", () => {
    const reg = buildRegistry(fixtureModules);
    expect(reg.runTabs.map((t) => `${t.group}:${t.id}`)).toEqual([
      "Model:overview",
      "Model:accuracy",
      "Effects:causal",
      "Decisions:lab",
      "Trust:validation",
      "Run:docs",
      "Run:files",
    ]);
    expect(reg.projectNav.map((n) => n.id)).toEqual(["runs", "lab"]);
  });

  it("the app registry includes the foundation 404 route", () => {
    expect(appRegistry.routes.some((r) => r.path === "*")).toBe(true);
  });

  it("RunTabs renders grouped tabs in order with status dots from /outputs", () => {
    const reg = buildRegistry(fixtureModules);
    const outputs: RunOutputs = {
      outputs: [],
      tabs: [
        { id: "overview", availability: "ready", missing: [] },
        { id: "accuracy", availability: "running", missing: [] },
        { id: "causal", availability: "missing", missing: [{ output: "causal", produced_by: "stage:S6", action: null }] },
        { id: "files", availability: "ready", missing: [] },
      ],
    };
    const { container } = render(<RunTabs rid="r1" tabs={reg.runTabs} outputs={outputs} activeId="accuracy" />);
    const groups = [...container.querySelectorAll('[role="group"]')].map((g) => g.getAttribute("aria-label"));
    expect(groups).toEqual(["Model", "Effects", "Decisions", "Trust", "Run"]);
    const links = [...container.querySelectorAll("a.run-tab")];
    expect(links.map((a) => a.getAttribute("data-tab"))).toEqual(["overview", "accuracy", "causal", "lab", "validation", "docs", "files"]);
    expect(links.map((a) => a.getAttribute("href"))).toEqual(["/r/r1", "/r/r1/accuracy", "/r/r1/causal", "/r/r1/lab", "/r/r1/validation", "/r/r1/docs", "/r/r1/files"]);
    const acc = container.querySelector('a[data-tab="accuracy"]')!;
    expect(acc.getAttribute("aria-current")).toBe("page");
    expect(acc.querySelector(".tab-dot")!.getAttribute("data-a")).toBe("running");
    expect(acc.textContent).toContain("being computed");
    const causal = container.querySelector('a[data-tab="causal"]')!;
    expect(causal.getAttribute("title")).toContain("produced by stage:S6");
    // Tabs without a server entry show no dot.
    expect(container.querySelector('a[data-tab="lab"] .tab-dot')).toBeNull();
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
    expect(links).toEqual(["overview", "accuracy", "causal", "lab", "validation", "docs", "files"]);
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

  it("marks one project nav entry current: its own route, else the longest target prefix; the root only on itself", () => {
    const reg = appRegistry;
    const entries = reg.projectNav.flatMap((n) => {
      const to = n.to("p_1", project);
      return to === null ? [] : [{ id: n.id, to, route: n.route }];
    });
    const at = (path: string) => currentProjectNav(entries, path, matchRoute(reg.compiled, path)?.route ?? null);
    expect(at("/p/p_1")).toBe("overview");
    expect(at("/p/p_1/setup/levers")).toBe("setup");
    expect(at("/p/p_1/setup/inputs")).toBe("inputs");
    expect(at("/p/p_1/launch")).toBe("launch");
    expect(at("/p/p_1/runs")).toBe("runs");
    expect(at("/p/p_1/config")).toBeNull();
  });

  it("ProjectNav renders declared entries, hides null targets and greys disabled ones", () => {
    const reg = buildRegistry(fixtureModules);
    const { container } = render(<ProjectNav registry={reg} project={project} />);
    expect([...container.querySelectorAll("a")].map((a) => a.getAttribute("href"))).toEqual(["/p/p_1/runs"]);
    const disabled = container.querySelector('[aria-disabled="true"]')!;
    expect(disabled.textContent).toBe("Scenario Lab");
    expect(container.querySelector('[role="tooltip"]')!.textContent).toBe("Needs a run with a checkpoint");
  });
});
