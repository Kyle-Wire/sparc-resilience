// What cools (`/r/:rid/effects`): one section per lever with its dose–response curve, curve
// shapes and how far its cooling reaches (from the Influence view), Where to act, and the link
// to Scenario Lab sweeps. The technical parts stay under Model details, not here.
import { afterEach, describe, expect, it, vi } from "vitest";
import { clearResources } from "../../../api/resource";
import type { ViewModelOf } from "../../../api/runs";
import { navigate } from "../../../router";
import { flush, mockFetch, render, waitFor, type MockHandler } from "../../../test/render";
import { olderCodeFixture, viewFixture } from "../__fixtures__/views";
import Effects, { leverAnchor } from "../Effects";
import { mapRoutes, unframedCharts } from "./helpers";

// Where to act is the Planner pack's part (owned by Heat risk); a stub keeps this test on What cools.
vi.mock("../Planner", async (orig) => ({ ...(await orig<typeof import("../Planner")>()), PlannerWhereToAct: () => "Where-to-act stub" }));

let seq = 0;

async function renderEffects(opts: { response?: ViewModelOf<"response">; influence?: MockHandler; planner?: MockHandler; search?: string } = {}) {
  const rid = `r_eff${++seq}`;
  const m = mockFetch({
    ...mapRoutes(rid),
    ...(opts.planner ? { [`GET /api/runs/${rid}/views/planner`]: opts.planner } : {}),
    [`GET /api/runs/${rid}/views/response`]: { body: opts.response ?? viewFixture("response") },
    [`GET /api/runs/${rid}/views/influence`]: opts.influence ?? { body: viewFixture("influence") },
  });
  navigate(`/r/${rid}/effects${opts.search ?? ""}`, { replace: true });
  const r = render(<Effects />);
  await waitFor(() => r.container.querySelector('[data-view="response"]') && !r.container.textContent?.includes("Loading what cools"), 5000, "What cools");
  await flush(6);
  return { ...r, rid, restore: m.restore };
}

const charts = (root: HTMLElement) => [...root.querySelectorAll("figure.chart-frame")].map((f) => f.getAttribute("data-chart") ?? "");

describe("What cools", () => {
  afterEach(() => clearResources());

  it("shows one section per lever: curve with band, curve shapes and how far cooling reaches", async () => {
    const r = await renderEffects();
    try {
      await waitFor(() => charts(r.container).some((t) => t.endsWith("how far cooling reaches")), 3000, "reach charts");
      const levers = Object.entries(viewFixture("response").sections.levers!);
      for (const [key, L] of levers) {
        const sec = r.container.querySelector<HTMLElement>(`#${leverAnchor(key)}`)!;
        expect(sec, key).not.toBeNull();
        expect(sec.querySelector("h3")!.textContent).toBe(L.label);
        expect(charts(sec)).toEqual([`${L.label}: dose–response`, `${L.label}: how far cooling reaches`, "Curve shapes across cells"]);
      }
      expect(unframedCharts(r.container)).toEqual([]);
      expect(r.container.textContent).not.toContain("could not be displayed");
      expect(r.container.querySelectorAll("[data-older-code]").length).toBe(0);
    } finally {
      r.restore();
    }
  });

  it("marks the lever's influence range and links to its influence circle on the map", async () => {
    const r = await renderEffects();
    try {
      const sec = await waitFor(() => r.container.querySelector<HTMLElement>(`#${leverAnchor("canopy")} figure[data-chart="canopy: how far cooling reaches"]`), 3000, "canopy reach");
      expect(sec.querySelector("figcaption")!.textContent).toContain("Influence range 315 m");
      const a = [...sec.querySelectorAll("a")].find((x) => x.textContent === "Influence circle on the map")!;
      expect(a.getAttribute("href")).toBe(`/r/${r.rid}/map?layer=canopy&ov=influence&infl=canopy`);
    } finally {
      r.restore();
    }
  });

  it("leaves the technical parts to Model details", async () => {
    const r = await renderEffects();
    try {
      await waitFor(() => charts(r.container).some((t) => t.endsWith("how far cooling reaches")), 3000, "reach charts");
      const titles = charts(r.container);
      for (const gone of ["requested vs realised dose", "Fold means", "Literature check", "Target correlogram", "Anisotropy roses", "Ring-kernel β by distance", "Influence range per predictor"])
        expect(titles.some((t) => t.includes(gone)), gone).toBe(false);
      const text = r.container.textContent ?? "";
      for (const gone of ["Own vs footprint", "Published values", "Length scales", "Maps for this lever"]) expect(text, gone).not.toContain(gone);
    } finally {
      r.restore();
    }
  });

  it("ends with Where to act and a link to try other amounts in the Scenario Lab", async () => {
    const r = await renderEffects();
    try {
      const where = r.container.querySelector("#where")!;
      expect(where.querySelector("h2")!.textContent).toBe("Where to act");
      expect(where.textContent).toContain("Where-to-act stub");
      const lab = [...r.container.querySelectorAll("a")].find((a) => a.textContent === "Try other amounts in the Scenario Lab →")!;
      expect(lab.getAttribute("href")).toBe(`/r/${r.rid}/lab/sweeps`);
    } finally {
      r.restore();
    }
  });

  it("without a planner pack, Where to act is its own titled empty state under no second heading", async () => {
    const planner = { ...viewFixture("planner"), availability: "missing" as const };
    const r = await renderEffects({ planner: { body: planner } });
    try {
      const where = r.container.querySelector<HTMLElement>("#where")!;
      expect(where.querySelector("h2")).toBeNull();
      expect(where.getAttribute("aria-label")).toBe("Where to act");
      expect(where.textContent).toContain("Where-to-act stub");
    } finally {
      r.restore();
    }
  });

  it("keeps the levers when the run has no Influence sections, saying the reach is not in this run", async () => {
    const r = await renderEffects({ influence: { body: olderCodeFixture("influence") } });
    try {
      await waitFor(() => r.container.querySelector("[data-older-code]"), 3000, "older-code reach");
      expect(r.container.querySelectorAll(`[id^="lever-"] figure[data-chart$="dose–response"]`).length).toBe(3);
      expect(r.container.textContent).toContain("canopy: how far cooling reaches: not in this run (older code)");
    } finally {
      r.restore();
    }
  });

  it("says when the reach could not be loaded and still shows the curves", async () => {
    const r = await renderEffects({ influence: { status: 500, body: { error: { code: "internal", message: "boom" } } } });
    try {
      await waitFor(() => r.container.textContent?.includes("how far cooling reaches could not be loaded"), 3000, "reach error");
      expect(r.container.querySelectorAll(`[id^="lever-"] figure[data-chart$="dose–response"]`).length).toBe(3);
    } finally {
      r.restore();
    }
  });

  it("says when no lever was swept", async () => {
    const vm = viewFixture("response");
    vm.sections.levers = {};
    const r = await renderEffects({ response: vm });
    try {
      expect(r.container.textContent).toContain("No actionable levers were swept in this run.");
    } finally {
      r.restore();
    }
  });
});
