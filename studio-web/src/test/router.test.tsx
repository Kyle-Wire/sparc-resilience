import { afterEach, describe, expect, it } from "vitest";
import { lazy } from "react";
import { Link, buildRegistry, codecs, compileRoutes, fillPath, matchRoute, navigate, RegistryProvider, runTabForRoute, setQuery, useRoute, useUrlState, type RouteDef } from "../router";
import { click, flush, render } from "./render";

const P = lazy(() => Promise.resolve({ default: () => null }));
const def = (path: string, title = path): RouteDef => ({ path, component: P, title });

const routes = [
  def("/"),
  def("/projects"),
  def("/p/:pid"),
  def("/p/:pid/setup/:step"),
  def("/p/:pid/setup/data", "setup data (static)"),
  def("/r/:rid"),
  def("/r/:rid/docs/:doc?"),
  def("/r/:rid/lab/sweeps/:swid?"),
  def("/files/*"),
  def("*"),
];
const compiled = compileRoutes(routes);
const m = (p: string) => matchRoute(compiled, p);

describe("route matching", () => {
  it("matches static, params and the catch-all", () => {
    expect(m("/")?.route.path).toBe("/");
    expect(m("/projects")?.route.path).toBe("/projects");
    expect(m("/p/p_1")).toMatchObject({ route: { path: "/p/:pid" }, params: { pid: "p_1" } });
    expect(m("/nope/at/all")?.route.path).toBe("*");
  });
  it("prefers static segments over params", () => {
    expect(m("/p/p_1/setup/data")?.route.title).toBe("setup data (static)");
    expect(m("/p/p_1/setup/levers")).toMatchObject({ route: { path: "/p/:pid/setup/:step" }, params: { step: "levers" } });
  });
  it("supports optional params", () => {
    expect(m("/r/20261001-142233-full-a1b2/docs")).toMatchObject({ route: { path: "/r/:rid/docs/:doc?" }, params: { rid: "20261001-142233-full-a1b2" } });
    expect(m("/r/x/docs/methods")?.params).toEqual({ rid: "x", doc: "methods" });
    expect(m("/r/x/lab/sweeps")?.route.path).toBe("/r/:rid/lab/sweeps/:swid?");
  });
  it("decodes params and captures wildcards", () => {
    expect(m("/p/a%20b")?.params.pid).toBe("a b");
    expect(m("/files/a/b%2Fc")?.params["*"]).toBe("a/b/c");
  });
  it("ignores trailing slashes and extra segments fall to *", () => {
    expect(m("/projects/")?.route.path).toBe("/projects");
    expect(m("/r/x/docs/a/b")?.route.path).toBe("*");
  });
  it("fills patterns", () => {
    expect(fillPath("/r/:rid/docs/:doc?", { rid: "r 1" })).toBe("/r/r%201/docs");
    expect(fillPath("/r/:rid/docs/:doc?", { rid: "x", doc: "methods" })).toBe("/r/x/docs/methods");
    expect(() => fillPath("/p/:pid", {})).toThrow();
  });
});

describe("query codecs", () => {
  it("round-trip and omit defaults", () => {
    const i = codecs.int(0);
    expect(i.parse("42")).toBe(42);
    expect(i.parse("x")).toBe(0);
    expect(i.format(0)).toBeNull();
    const b = codecs.bool(false);
    expect(b.parse("1")).toBe(true);
    expect(b.format(true)).toBe("1");
    const e = codecs.enum(["swipe", "side", "diff"] as const, "swipe");
    expect(e.parse("diff")).toBe("diff");
    expect(e.parse("bogus")).toBe("swipe");
    const l = codecs.list();
    expect(l.parse("res_1,configured:green")).toEqual(["res_1", "configured:green"]);
    expect(l.format([])).toBeNull();
    const f = codecs.float(null);
    expect(f.parse("-1.5")).toBe(-1.5);
    expect(f.parse("")).toBeNull();
  });
  it("json codec survives unicode and is URL-safe", () => {
    const j = codecs.json<{ a: number; s: string } | null>(null);
    const v = { a: 1, s: "Zone 3 — ΔT ≤ −0.5" };
    const enc = j.format(v)!;
    expect(enc).toMatch(/^[A-Za-z0-9_-]+$/);
    expect(j.parse(enc)).toEqual(v);
    expect(j.parse("%%%")).toBeNull();
  });
});

describe("history router", () => {
  afterEach(() => navigate("/", { replace: true }));

  function Probe() {
    const r = useRoute();
    const [layer, setLayer] = useUrlState("layer", codecs.string("obs"));
    return (
      <div>
        <span data-testid="path">{r.pathname}</span>
        <span data-testid="title">{r.route?.title}</span>
        <span data-testid="params">{JSON.stringify(r.params)}</span>
        <span data-testid="layer">{layer}</span>
        <button type="button" onClick={() => setLayer("resid")}>
          set
        </button>
        <Link to="/r/run1/docs/methods">docs</Link>
      </div>
    );
  }

  it("navigates, matches and keeps state in the URL", async () => {
    const reg = buildRegistry({ "x/routes.ts": { routes } });
    const { container } = render(
      <RegistryProvider registry={reg}>
        <Probe />
      </RegistryProvider>,
    );
    const q = (id: string) => container.querySelector(`[data-testid=${id}]`)!.textContent;
    expect(q("layer")).toBe("obs");
    click(container.querySelector("a"));
    await flush();
    expect(q("path")).toBe("/r/run1/docs/methods");
    expect(JSON.parse(q("params")!)).toEqual({ rid: "run1", doc: "methods" });
    click(container.querySelector("button"));
    await flush();
    expect(window.location.search).toBe("?layer=resid");
    expect(q("layer")).toBe("resid");
    setQuery({ layer: null, sel: "rg_1" });
    await flush();
    expect(window.location.search).toBe("?sel=rg_1");
    expect(q("layer")).toBe("obs");
    window.history.back();
    await flush(5);
  });

  it("Link leaves modified clicks to the browser", () => {
    const { container } = render(<Link to="/x">x</Link>);
    const a = container.querySelector("a")!;
    let prevented: boolean | null = null;
    // Runs after React's handler: record its decision, then stop jsdom from navigating.
    const after = (e: Event) => {
      prevented = e.defaultPrevented;
      e.preventDefault();
    };
    document.addEventListener("click", after);
    a.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true, button: 0, ctrlKey: true }));
    document.removeEventListener("click", after);
    expect(prevented).toBe(false);
    expect(window.location.pathname).toBe("/");
  });
});

describe("run tab of a route", () => {
  it("uses the route's own tab, else the longest tab-path prefix", () => {
    const reg = buildRegistry({
      "a/routes.ts": {
        routes: [
          { ...def("/r/:rid"), runTab: { id: "overview", label: "Overview", group: "Model", order: 1 } },
          { ...def("/r/:rid/lab"), runTab: { id: "lab", label: "Lab", group: "Decisions", order: 1 } },
          def("/r/:rid/lab/library"),
          def("/r/:rid/unknown"),
        ],
      },
    });
    const byPath = (p: string) => reg.routes.find((r) => r.path === p)!;
    expect(runTabForRoute(reg, byPath("/r/:rid/lab/library"))?.id).toBe("lab");
    expect(runTabForRoute(reg, byPath("/r/:rid"))?.id).toBe("overview");
    expect(runTabForRoute(reg, byPath("/r/:rid/unknown"))).toBeNull();
  });
});
