// The run-hub route module (SPEC §3.2, §12.3): every run tab of the work item is declared with
// runTab metadata in its SPEC group, and the foundation RunLayout shows them grouped
// (Model · Effects · Decisions · Trust · Run) and ordered.
import { afterEach, describe, expect, it } from "vitest";
import { clearResources } from "../../../api/resource";
import type { RunOutputs } from "../../../api/types";
import { RunLayout, RunTabs } from "../../../layouts/RunLayout";
import { buildRegistry, fillPath, matchRoute, navigate, registry, RegistryProvider, type RouteModule } from "../../../router";
import { routes as coreRoutes } from "../../core/routes";
import { flush, mockFetch, render } from "../../../test/render";
import { RUN_HUB_TABS, routes } from "../routes";
import { runDetail } from "./helpers";

const GROUPS: Record<string, string[]> = {
  Model: ["overview", "data", "accuracy", "distance", "influence"],
  Effects: ["response", "causal"],
  Decisions: ["heat", "scenarios", "climate", "budget", "planner"],
  Trust: ["uncertainty", "provenance"],
  Run: ["map", "docs", "files"],
};

const modules: Record<string, RouteModule> = { "./pages/core/routes.ts": { routes: coreRoutes }, "./pages/runhub/routes.ts": { routes } };

describe("run-hub routes", () => {
  afterEach(() => {
    clearResources();
    navigate("/", { replace: true });
  });

  it("declares every run tab of the item with runTab metadata in its SPEC group", () => {
    expect([...RUN_HUB_TABS].sort()).toEqual(Object.values(GROUPS).flat().sort());
    for (const [group, ids] of Object.entries(GROUPS))
      for (const id of ids) {
        const r = routes.find((x) => x.runTab?.id === id)!;
        expect(r, id).toBeDefined();
        expect(r.runTab!.group, id).toBe(group);
        expect(r.runTab!.label.length).toBeGreaterThan(0);
      }
    // every /r/ route of the module is a tab; Compare is a project page
    expect(routes.filter((r) => r.path.startsWith("/r/") && !r.runTab)).toEqual([]);
    expect(routes.map((r) => r.path)).toEqual(expect.arrayContaining(["/r/:rid", "/r/:rid/map", "/r/:rid/docs/:doc?", "/r/:rid/files", "/r/:rid/provenance", "/p/:pid/compare"]));
  });

  it("matches deep links, including an optional document", () => {
    const reg = buildRegistry(modules);
    expect(reg.problems).toEqual([]);
    expect(matchRoute(reg.compiled, "/r/r1")!.route.title).toBe("Run overview");
    expect(matchRoute(reg.compiled, "/r/r1/docs")!.params).toEqual({ rid: "r1" });
    expect(matchRoute(reg.compiled, "/r/r1/docs/methods")!.params).toEqual({ rid: "r1", doc: "methods" });
    expect(matchRoute(reg.compiled, "/p/p_1/compare")!.route.title).toBe("Compare runs");
    expect(fillPath("/r/:rid/docs/:doc?", { rid: "r1" })).toBe("/r/r1/docs");
  });

  it("RunTabs shows the tabs grouped Model · Effects · Decisions · Trust · Run and ordered", () => {
    const reg = buildRegistry(modules);
    const outputs: RunOutputs = { outputs: [], tabs: [{ id: "causal", availability: "missing", missing: [{ output: "causal", produced_by: "stage:S6", action: null }] }] };
    const { container } = render(<RunTabs rid="r1" tabs={reg.runTabs} outputs={outputs} activeId="accuracy" />);
    const groups = [...container.querySelectorAll('[role="group"]')];
    expect(groups.map((g) => g.getAttribute("aria-label"))).toEqual(Object.keys(GROUPS));
    groups.forEach((g) => {
      const ids = [...g.querySelectorAll("a.run-tab")].map((a) => a.getAttribute("data-tab"));
      expect(ids).toEqual(GROUPS[g.getAttribute("aria-label")!]);
    });
    const hrefs = [...container.querySelectorAll("a.run-tab")].map((a) => a.getAttribute("href"));
    expect(hrefs).toContain("/r/r1");
    expect(hrefs).toContain("/r/r1/docs");
    expect(hrefs).toContain("/r/r1/map");
    expect(container.querySelector('a[data-tab="accuracy"]')!.getAttribute("aria-current")).toBe("page");
    expect(container.querySelector('a[data-tab="causal"] .tab-dot')!.getAttribute("data-a")).toBe("missing");
  });

  it("the app registry picks the module up and RunLayout renders its tab bar", async () => {
    for (const id of RUN_HUB_TABS)
      expect(
        registry.runTabs.some((t) => t.id === id && t.route.path.startsWith("/r/:rid")),
        id,
      ).toBe(true);
    const reg = buildRegistry(modules);
    const m = mockFetch({ "GET /api/runs/r_layout": { body: runDetail("r_layout") }, "GET /api/runs/r_layout/outputs": { body: { outputs: [], tabs: [] } } });
    navigate("/r/r_layout/influence", { replace: true });
    const { container } = render(
      <RegistryProvider registry={reg}>
        <RunLayout rid="r_layout">
          <p>page</p>
        </RunLayout>
      </RegistryProvider>,
    );
    await flush(4);
    expect([...container.querySelectorAll("a.run-tab")].map((a) => a.getAttribute("data-tab"))).toEqual(Object.values(GROUPS).flat());
    expect(container.querySelector('a[data-tab="influence"]')!.getAttribute("aria-current")).toBe("page");
    m.restore();
  });
});
