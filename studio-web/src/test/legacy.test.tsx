// Legacy URLs (phase 2 information architecture): every old run tab and moved page resolves,
// through the app registry, to its new home with the query and hash kept; the app replaces the
// history entry and renders the new page.
import { afterEach, describe, expect, it } from "vitest";
import { App } from "../App";
import { clearResources } from "../api/resource";
import { StreamManager, setStreams, type EventSourceLike } from "../api/sse";
import { RUN_TAB_IDS } from "../api/types";
import { routes as legacyRoutes, LEGACY_PATHS } from "../pages/legacy/routes";
import { matchRoute, navigate, redirectTarget, registry, runViewHref } from "../router";
import { mockFetch, render, waitFor } from "./render";

class SilentES implements EventSourceLike {
  onopen: ((ev: Event) => void) | null = null;
  onerror: ((ev: Event) => void) | null = null;
  onmessage: ((ev: MessageEvent) => void) | null = null;
  addEventListener() {}
  close() {}
}

/** Old URL → new URL (the spec's table). */
const CASES: [string, string][] = [
  ["/r/r1/data", "/r/r1/details/data"],
  ["/r/r1/accuracy", "/r/r1/trust"],
  ["/r/r1/distance", "/r/r1/details/reach"],
  ["/r/r1/influence", "/r/r1/effects"],
  ["/r/r1/response", "/r/r1/effects"],
  ["/r/r1/causal", "/r/r1/details/causal"],
  ["/r/r1/scenarios", "/r/r1/lab/library"],
  ["/r/r1/climate", "/r/r1/heat/future"],
  ["/r/r1/budget", "/r/r1/lab/plans"],
  ["/r/r1/planner", "/r/r1/heat/people"],
  ["/r/r1/validation", "/r/r1/trust"],
  ["/r/r1/uncertainty", "/r/r1/trust"],
  ["/r/r1/provenance", "/r/r1/details/provenance"],
  ["/r/r1/docs", "/r/r1/details/docs"],
  ["/r/r1/docs/methods", "/r/r1/details/docs/methods"],
  ["/r/r1/files", "/r/r1/details/files"],
  ["/r/r1/track", "/r/r1/details/runlog"],
  ["/r/r1/lab/climate", "/r/r1/heat/future"],
  ["/p/p1/findings", "/p/p1/exports?tab=notes"],
];

function resolve(url: string): string | null {
  const u = new URL(url, "http://localhost");
  const m = matchRoute(registry.compiled, u.pathname);
  if (!m) return null;
  return redirectTarget(m.route, m.params, { search: u.search, hash: u.hash });
}

afterEach(() => {
  clearResources();
  navigate("/", { replace: true });
});

describe("legacy redirects", () => {
  it("every legacy route is in the app registry, redirect-only, with no problems", () => {
    expect(registry.problems).toEqual([]);
    for (const r of legacyRoutes) {
      expect(registry.routes, r.path).toContain(r);
      expect(r.redirect, r.path).toBeTypeOf("function");
      expect(r.component, r.path).toBeUndefined();
      expect(r.runTab, r.path).toBeUndefined();
    }
    // every pattern of the table is covered, and nothing else
    const covered = new Set(CASES.map(([old]) => matchRoute(registry.compiled, old)!.route.path));
    expect([...covered].sort()).toEqual([...LEGACY_PATHS].sort());
  });

  it.each(CASES)("%s → %s", (old, to) => {
    expect(resolve(old)).toBe(to);
  });

  it("keeps the query and the hash", () => {
    expect(resolve("/r/r1/accuracy?model=stack#cv")).toBe("/r/r1/trust?model=stack#cv");
    expect(resolve("/r/r1/files?f=checkpoint.pkl#checkpoint")).toBe("/r/r1/details/files?f=checkpoint.pkl#checkpoint");
    expect(resolve("/r/r1/docs/model_card#limits")).toBe("/r/r1/details/docs/model_card#limits");
    expect(resolve("/r/r1/scenarios?configured=cool-roofs")).toBe("/r/r1/lab/library?configured=cool-roofs");
    expect(resolve("/p/p1/findings?run=r1")).toBe("/p/p1/exports?tab=notes&run=r1");
    expect(resolve("/r/r%201/data")).toBe("/r/r%201/details/data");
  });

  it("redirect targets agree with VIEW_HREF for the views they replace", () => {
    for (const v of RUN_TAB_IDS) {
      if (v === "overview" || v === "heat" || v === "lab" || v === "map") continue; // URLs that did not move
      const old = v === "docs" ? "/r/r1/docs" : `/r/r1/${v}`;
      expect(resolve(old), v).toBe(runViewHref("r1", v));
    }
  });

  it("URLs that did not move are pages, not redirects", () => {
    for (const p of ["/r/r1", "/r/r1/map", "/r/r1/heat", "/r/r1/lab", "/r/r1/lab/library", "/r/r1/lab/plans", "/p/p1/exports", "/findings"]) {
      const m = matchRoute(registry.compiled, p)!;
      expect(m.route.redirect, p).toBeUndefined();
      expect(m.route.component, p).toBeDefined();
    }
  });

  it("the app replaces a legacy run URL with its new page", async () => {
    const fetch = mockFetch({
      "GET /api/runs/r1": { body: { run: { id: "r1", project_id: null, label: "Fast run", status: "complete", mode: "fast", coarse_m: null, demo: false }, header: { name: "demo" } } },
      "GET /api/runs/r1/outputs": { body: { outputs: [], tabs: [] } },
      "GET /api/runs/r1/docs": { body: [] },
      "GET /api/jobs": { body: { items: [], next_cursor: null } },
      "GET /api/health": { body: { ok: true, engine: { state: "absent" } } },
      "GET /api/projects": { body: [] },
    });
    setStreams(new StreamManager({ EventSource: SilentES, frame: (cb) => cb() }));
    try {
      navigate("/r/r1/docs/methods?x=1#top");
      const depth = window.history.length;
      const { container } = render(<App />);
      await waitFor(() => window.location.pathname === "/r/r1/details/docs/methods", 10000, "the redirect");
      expect(window.location.search).toBe("?x=1");
      expect(window.location.hash).toBe("#top");
      expect(window.history.length).toBe(depth);
      await waitFor(() => container.querySelector('nav.section-rail a[aria-current="page"]'), 10000, "the Model details rail");
      expect(container.querySelector('nav.section-rail a[aria-current="page"]')!.textContent).toBe("Documents");
      expect(container.querySelector('a.run-tab[data-tab="details"]')!.getAttribute("aria-current")).toBe("page");
    } finally {
      fetch.restore();
      setStreams(null);
    }
  });
});
