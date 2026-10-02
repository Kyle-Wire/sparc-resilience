// The studies route module (SPEC §3.1, §3.2, §12.3): the Validation run tab in group Trust,
// the five other routes, and the Studies, Exports and Findings project nav entries.
import { afterEach, describe, expect, it } from "vitest";
import { clearResources } from "../../../api/resource";
import { ProjectNav } from "../../../layouts/AppShell";
import { buildRegistry, matchRoute, navigate, registry, type RouteModule } from "../../../router";
import { render } from "../../../test/render";
import { routes as coreRoutes } from "../../core/routes";
import { STUDIES_NAV_IDS, STUDIES_RUN_TABS, routes } from "../routes";
import { project } from "../__fixtures__/api";

const MODULES: Record<string, RouteModule> = { "./pages/core/routes.ts": { routes: coreRoutes }, "./pages/studies/routes.ts": { routes } };

afterEach(() => {
  clearResources();
  navigate("/", { replace: true });
});

describe("studies routes", () => {
  it("registers the Validation run tab in group Trust and the five other routes", () => {
    const v = routes.find((r) => r.path === "/r/:rid/validation")!;
    expect(v).toBeDefined();
    expect(v.runTab).toEqual({ id: "validation", label: "Validation", group: "Trust", order: expect.any(Number) });
    expect(STUDIES_RUN_TABS).toEqual(["validation"]);
    expect(routes.map((r) => r.path).sort()).toEqual(["/findings", "/p/:pid/exports", "/p/:pid/findings", "/p/:pid/studies", "/r/:rid/validation", "/studies/:stid"].sort());
  });

  it("declares the Studies, Exports and Findings nav entries after the Lab", () => {
    expect(STUDIES_NAV_IDS).toEqual(["studies", "exports", "findings"]);
    const p = project({ id: "p_9" });
    const nav = Object.fromEntries(routes.filter((r) => r.projectNav).map((r) => [r.projectNav!.id, r.projectNav!]));
    expect(nav.studies.to("p_9", p)).toBe("/p/p_9/studies");
    expect(nav.exports.to("p_9", p)).toBe("/p/p_9/exports");
    expect(nav.findings.to("p_9", p)).toBe("/p/p_9/findings");
    expect(nav.studies.order).toBeGreaterThan(60); // Scenario Lab is 60 (SPEC §3.1 order)
    expect(nav.studies.order).toBeLessThan(nav.exports.order);
    expect(nav.exports.order).toBeLessThan(nav.findings.order);
  });

  it("matches deep links", () => {
    const reg = buildRegistry(MODULES);
    expect(reg.problems).toEqual([]);
    expect(matchRoute(reg.compiled, "/r/r1/validation")!.params).toEqual({ rid: "r1" });
    expect(matchRoute(reg.compiled, "/studies/st_1")!.params).toEqual({ stid: "st_1" });
    expect(matchRoute(reg.compiled, "/p/p_1/exports")!.route.title).toBe("Exports & reports");
    expect(matchRoute(reg.compiled, "/p/p_1/findings")!.route.title).toBe("Findings");
    expect(matchRoute(reg.compiled, "/findings")!.route.title).toBe("Findings");
    expect(matchRoute(reg.compiled, "/p/p_1/studies")!.route.title).toBe("Studies");
  });

  it("is picked up by the app registry, with the tab in the Trust group before Uncertainty", () => {
    const tab = registry.runTabs.find((t) => t.id === "validation")!;
    expect(tab.group).toBe("Trust");
    expect(routes).toContain(tab.route);
    const trust = registry.runTabs.filter((t) => t.group === "Trust").map((t) => t.id);
    expect(trust[0]).toBe("validation");
    for (const id of STUDIES_NAV_IDS) expect(registry.projectNav.some((n) => n.id === id && routes.includes(n.route)), id).toBe(true);
    expect(registry.problems.filter((x) => x.includes("studies"))).toEqual([]);
  });

  it("ProjectNav renders the three entries as links", () => {
    const reg = buildRegistry(MODULES);
    const { container } = render(<ProjectNav registry={reg} project={project({ id: "p_9" })} />);
    const links = [...container.querySelectorAll("a")].map((a) => [a.textContent, a.getAttribute("href")]);
    expect(links).toEqual([
      ["Studies", "/p/p_9/studies"],
      ["Exports", "/p/p_9/exports"],
      ["Findings", "/p/p_9/findings"],
    ]);
  });
});
