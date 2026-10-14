import { afterEach, describe, expect, it } from "vitest";
import { lazy } from "react";
import { RUN_TAB_IDS } from "../api/types";
import {
  Link,
  buildRegistry,
  codecs,
  compileRoutes,
  fillPath,
  matchRoute,
  navigate,
  redirectTarget,
  RegistryProvider,
  runHref,
  runTabForRoute,
  runViewHref,
  setQuery,
  useRoute,
  useUrlState,
  VIEW_HREF,
  withSearch,
  type RouteDef,
} from "../router";
import { click, flush, render, waitFor } from "./render";

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
    for (const items of [["a,b", "c"], ["50%", "x%2Cy", "%41"], ["configured:canopy_+10%"]]) expect(l.parse(l.format(items))).toEqual(items);
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
    window.history.back(); // popstate: back to the path before the setQuery replace chain
    await waitFor(() => q("path") === "/", 5000, "popstate");
    expect(q("layer")).toBe("obs");
  });

  it("restores the scroll position of each history entry on back and forward", async () => {
    let y = 0;
    const scrolled: number[] = [];
    const orig = Object.getOwnPropertyDescriptor(window, "scrollY");
    const origTo = window.scrollTo;
    Object.defineProperty(window, "scrollY", { configurable: true, get: () => y });
    window.scrollTo = ((_x: number, top: number) => {
      y = top;
      scrolled.push(top);
    }) as typeof window.scrollTo;
    try {
      navigate("/long-page");
      y = 300;
      navigate("/other"); // remembers 300 for /long-page, scrolls to the top
      expect(y).toBe(0);
      y = 120;
      window.history.back();
      await waitFor(() => window.location.pathname === "/long-page" && y === 300, 5000, "restore on back");
      window.history.forward();
      await waitFor(() => window.location.pathname === "/other" && y === 120, 5000, "restore on forward");
      expect(scrolled.slice(-2)).toEqual([300, 120]);
    } finally {
      if (orig) Object.defineProperty(window, "scrollY", orig);
      else delete (window as { scrollY?: number }).scrollY;
      window.scrollTo = origTo;
    }
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
          { ...def("/r/:rid"), runTab: { id: "summary", label: "Summary", order: 1, views: ["overview"] } },
          { ...def("/r/:rid/lab"), runTab: { id: "lab", label: "Lab", order: 2, views: ["lab"] } },
          { ...def("/r/:rid/heat/:section?"), runTab: { id: "heat", label: "Heat risk", order: 3, views: ["heat"] } },
          def("/r/:rid/lab/library"),
          def("/r/:rid/unknown"),
        ],
      },
    });
    const byPath = (p: string) => reg.routes.find((r) => r.path === p)!;
    expect(runTabForRoute(reg, byPath("/r/:rid/lab/library"))?.id).toBe("lab");
    expect(runTabForRoute(reg, byPath("/r/:rid"))?.id).toBe("summary");
    expect(runTabForRoute(reg, byPath("/r/:rid/heat/:section?"))?.id).toBe("heat");
    expect(runTabForRoute(reg, byPath("/r/:rid/unknown"))).toBeNull();
  });
});

describe("run URLs", () => {
  it("runHref builds the run tab and section URLs, encoding each segment", () => {
    expect(runHref("r1")).toBe("/r/r1");
    expect(runHref("r 1", "summary", "ignored")).toBe("/r/r%201");
    expect(runHref("r1", "map")).toBe("/r/r1/map");
    expect(runHref("r1", "effects")).toBe("/r/r1/effects");
    expect(runHref("r1", "heat", "future")).toBe("/r/r1/heat/future");
    expect(runHref("r1", "lab", "library")).toBe("/r/r1/lab/library");
    expect(runHref("r1", "trust")).toBe("/r/r1/trust");
    expect(runHref("r1", "details")).toBe("/r/r1/details");
    expect(runHref("r1", "details", "docs/model card")).toBe("/r/r1/details/docs/model%20card");
    expect(runHref("r1", "details", null)).toBe("/r/r1/details");
  });

  it("VIEW_HREF maps every server view id to its new page; runViewHref knows nothing else", () => {
    expect(Object.keys(VIEW_HREF).sort()).toEqual([...RUN_TAB_IDS].sort());
    expect(Object.fromEntries(RUN_TAB_IDS.map((v) => [v, VIEW_HREF[v]("r1")]))).toEqual({
      overview: "/r/r1",
      data: "/r/r1/details/data",
      accuracy: "/r/r1/trust",
      distance: "/r/r1/details/reach",
      influence: "/r/r1/effects",
      response: "/r/r1/effects",
      causal: "/r/r1/details/causal",
      scenarios: "/r/r1/lab/library",
      climate: "/r/r1/heat/future",
      heat: "/r/r1/heat",
      budget: "/r/r1/lab/plans",
      planner: "/r/r1/heat/people",
      lab: "/r/r1/lab",
      validation: "/r/r1/trust",
      uncertainty: "/r/r1/trust",
      provenance: "/r/r1/details/provenance",
      track: "/r/r1/details/runlog",
      map: "/r/r1/map",
      docs: "/r/r1/details/docs",
      files: "/r/r1/details/files",
    });
    expect(runViewHref("r1", "planner")).toBe("/r/r1/heat/people");
    expect(runViewHref("r1", "table")).toBeNull();
    expect(runViewHref("r1", "toString")).toBeNull();
    expect(runViewHref("r1", null)).toBeNull();
  });

  it("withSearch merges the current query into a target (the target's keys win) and redirectTarget keeps the hash", () => {
    expect(withSearch("/r/r1/trust", "")).toBe("/r/r1/trust");
    expect(withSearch("/r/r1/trust", "?model=stack")).toBe("/r/r1/trust?model=stack");
    expect(withSearch("/p/p1/exports?tab=notes", "?run=r1&tab=x")).toBe("/p/p1/exports?tab=notes&run=r1");
    expect(withSearch("/a#h", "?q=1")).toBe("/a?q=1#h");
    const route: RouteDef = { path: "/r/:rid/old", title: "Old", redirect: (p, search) => withSearch(`/r/${p.rid}/new`, search) };
    expect(redirectTarget(route, { rid: "r1" }, { search: "?a=1", hash: "#x" })).toBe("/r/r1/new?a=1#x");
    expect(redirectTarget({ ...route, redirect: () => "/r/r1/new#own" }, { rid: "r1" }, { search: "", hash: "#x" })).toBe("/r/r1/new#own");
    expect(redirectTarget(def("/x"), {}, { search: "", hash: "" })).toBeNull();
  });
});
