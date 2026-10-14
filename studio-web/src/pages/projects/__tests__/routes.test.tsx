// The projects route module (SPEC §3.2, §3.1, §12.3; phase 2): the five routes are registered,
// the nav entries "Set up" (current on every setup step and on Launch) and "Config editor"
// (Advanced group) are declared (Overview and Runs belong to tracking), the foundation registry
// picks the module up, and the shell's ProjectNav links them.
import { afterEach, describe, expect, it } from "vitest";
import { ProjectNav } from "../../../layouts/AppShell";
import { buildRegistry, matchRoute, registry } from "../../../router";
import { render } from "../../../test/render";
import { PROJECT_NAV_IDS, routes } from "../routes";
import { project } from "../__fixtures__/api";
import { MODULES, resetAll } from "./helpers";

afterEach(() => resetAll());

const FIVE = ["/", "/projects", "/p/:pid/setup/:step", "/p/:pid/config", "/p/:pid/launch"];

describe("projects routes", () => {
  it("registers the five routes of the work item", () => {
    const paths = routes.map((r) => r.path);
    for (const p of FIVE) expect(paths, p).toContain(p);
    expect(routes.every((r) => !r.runTab)).toBe(true);
  });

  it("declares the nav entries Set up (claiming the steps and Launch) and Config editor (Advanced)", () => {
    expect(PROJECT_NAV_IDS).toEqual(["setup", "config"]);
    const nav = Object.fromEntries(routes.filter((r) => r.projectNav).map((r) => [r.projectNav!.id, r.projectNav!]));
    const p = project();
    expect(nav.setup.label).toBe("Set up");
    expect(nav.setup.to("p_1", p)).toBe("/p/p_1/setup/data");
    expect(nav.setup.claims).toEqual(expect.arrayContaining(["/p/:pid/setup", "/p/:pid/setup/inputs", "/p/:pid/launch"]));
    expect(nav.setup.group).toBeUndefined();
    expect(nav.config.label).toBe("Config editor");
    expect(nav.config.to("p_1", p)).toBe("/p/p_1/config");
    expect(nav.config.group).toBe("advanced");
    expect(PROJECT_NAV_IDS).not.toContain("overview");
    expect(PROJECT_NAV_IDS).not.toContain("runs");
  });

  it("matches deep links; the static inputs path wins over :step", () => {
    const reg = buildRegistry(MODULES);
    expect(reg.problems).toEqual([]);
    expect(matchRoute(reg.compiled, "/")!.route.title).toBe("Home");
    expect(matchRoute(reg.compiled, "/projects")!.route.title).toBe("Projects");
    expect(matchRoute(reg.compiled, "/p/p_1/setup/levers")!.params).toEqual({ pid: "p_1", step: "levers" });
    expect(matchRoute(reg.compiled, "/p/p_1/setup/inputs")!.route.title).toBe("Inputs");
    expect(matchRoute(reg.compiled, "/p/p_1/setup")!.route.title).toBe("Setup");
    expect(matchRoute(reg.compiled, "/p/p_1/config")!.route.title).toBe("Config");
    expect(matchRoute(reg.compiled, "/p/p_1/launch")!.route.title).toBe("Launch");
  });

  it("is picked up by the foundation router's registry", () => {
    for (const p of FIVE) expect(registry.routes.some((r) => r.path === p && routes.includes(r)), p).toBe(true);
    for (const id of PROJECT_NAV_IDS) expect(registry.projectNav.some((n) => n.id === id && routes.includes(n.route)), id).toBe(true);
    expect(registry.problems.filter((x) => x.includes("projects"))).toEqual([]);
  });

  it("ProjectNav renders Set up as a link and Config editor under Advanced", () => {
    const reg = buildRegistry(MODULES);
    const { container } = render(<ProjectNav registry={reg} project={project({ id: "p_9" })} />);
    const links = [...container.querySelectorAll("ul.shell-nav > li > a")].map((a) => [a.textContent, a.getAttribute("href")]);
    expect(links[0]).toEqual(["Set up", "/p/p_9/setup/data"]);
    expect(container.querySelector("button.shell-nav-toggle")!.textContent).toContain("Advanced");
  });
});
