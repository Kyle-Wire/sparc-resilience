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
  it("registers Reports, Studies, the study page and the workspace Findings; no run tab any more", () => {
    expect(STUDIES_RUN_TABS).toEqual([]);
    expect(routes.map((r) => r.path).sort()).toEqual(["/findings", "/p/:pid/exports", "/p/:pid/studies", "/studies/:stid"].sort());
    // the run's Validation page lives in How much to trust this; the old URLs redirect
    expect(routes.some((r) => r.path === "/r/:rid/validation" || r.path === "/p/:pid/findings")).toBe(false);
  });

  it("declares Reports in the main nav (after the Lab) and Studies in the Advanced group", () => {
    expect(STUDIES_NAV_IDS).toEqual(["reports", "studies"]);
    const p = project({ id: "p_9" });
    const nav = Object.fromEntries(routes.filter((r) => r.projectNav).map((r) => [r.projectNav!.id, r.projectNav!]));
    expect(nav.reports.label).toBe("Reports");
    expect(nav.reports.to("p_9", p)).toBe("/p/p_9/exports");
    expect(nav.reports.group).toBeUndefined();
    expect(nav.reports.order).toBeGreaterThan(40); // Scenario Lab is 40
    expect(nav.studies.to("p_9", p)).toBe("/p/p_9/studies");
    expect(nav.studies.group).toBe("advanced");
  });

  it("matches deep links", () => {
    const reg = buildRegistry(MODULES);
    expect(reg.problems).toEqual([]);
    expect(matchRoute(reg.compiled, "/studies/st_1")!.params).toEqual({ stid: "st_1" });
    expect(matchRoute(reg.compiled, "/p/p_1/exports")!.route.title).toBe("Reports");
    expect(matchRoute(reg.compiled, "/findings")!.route.title).toBe("Findings");
    expect(matchRoute(reg.compiled, "/p/p_1/studies")!.route.title).toBe("Studies");
  });

  it("is picked up by the app registry; the Trust tab covers the validation view", () => {
    expect(registry.runTabs.find((t) => t.id === "trust")!.views).toContain("validation");
    for (const id of STUDIES_NAV_IDS) expect(registry.projectNav.some((n) => n.id === id && routes.includes(n.route)), id).toBe(true);
    expect(registry.problems.filter((x) => x.includes("studies"))).toEqual([]);
  });

  it("ProjectNav renders Reports as a link and Studies under Advanced", () => {
    const reg = buildRegistry(MODULES);
    const { container } = render(<ProjectNav registry={reg} project={project({ id: "p_9" })} />);
    expect([...container.querySelectorAll("ul.shell-nav > li > a")].map((a) => [a.textContent, a.getAttribute("href")]).slice(0, 1)).toEqual([["Reports", "/p/p_9/exports"]]);
    expect(container.querySelector("button.shell-nav-toggle")!.textContent).toContain("Advanced");
  });
});
