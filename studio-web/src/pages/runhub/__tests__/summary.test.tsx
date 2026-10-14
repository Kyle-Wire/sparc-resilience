// Summary (the run's planner landing page): four headline cards that link to their tabs, no
// run metadata / timings / outputs grid / accuracy numbers, caveats behind one closed
// disclosure, next steps, the stage rail while the run is still going, and the cards'
// fallbacks (configured scenarios, the overview's warming KPI, missing views).
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { clearResources } from "../../../api/resource";
import type { RunStageRow } from "../../../api/runs";
import { navigate } from "../../../router";
import { useJobs } from "../../../stores/jobs";
import { flush, mockFetch, render, waitFor, type MockHandler } from "../../../test/render";
import { viewFixture } from "../__fixtures__/views";
import Summary, { coolingPick, keepInMind, trustInputs, warmingPick } from "../Summary";
import { runDetail } from "./helpers";

let seq = 0;
const nextRid = () => `r_sum${++seq}`;

async function mount(rid: string, routes: Record<string, MockHandler>, hash = "") {
  const m = mockFetch(routes);
  navigate(`/r/${rid}${hash}`, { replace: true });
  const r = render(<Summary />);
  await waitFor(() => r.container.querySelectorAll('[data-card][data-state="loading"]').length === 0, 5000, "summary");
  await flush(3);
  return { ...r, m };
}

const card = (c: HTMLElement, id: string) => c.querySelector<HTMLElement>(`[data-card="${id}"]`)!;

function views(rid: string, extra: Record<string, MockHandler> = {}): Record<string, MockHandler> {
  return {
    [`GET /api/runs/${rid}`]: { body: runDetail(rid) },
    [`GET /api/runs/${rid}/views/overview`]: { body: viewFixture("overview") },
    [`GET /api/runs/${rid}/views/heat`]: { body: viewFixture("heat") },
    [`GET /api/runs/${rid}/views/climate`]: { body: viewFixture("climate") },
    [`GET /api/runs/${rid}/views/scenarios`]: { body: viewFixture("scenarios") },
    ...extra,
  };
}

beforeEach(() => useJobs.setState({ jobs: {} }));
afterEach(() => clearResources());

describe("Summary", () => {
  it("shows exactly four headline cards, each linking to its tab, under no second run header", async () => {
    const rid = nextRid();
    const { container, m } = await mount(rid, views(rid));
    // the run header above the tabs (RunLayout) names the run, its mode and state: not repeated here
    expect(container.querySelector(".summary-head")).toBeNull();
    expect(container.querySelector("h1")).toBeNull();
    // the only h2s are the four headline cards' titles (the run header is RunLayout's h1)
    expect([...container.querySelectorAll("h2")].every((h) => h.matches(".summary-card > .label"))).toBe(true);
    expect(container.textContent).not.toContain("Fast demo run");

    const cards = [...container.querySelectorAll<HTMLElement>("[data-card]")];
    expect(cards.map((c) => c.dataset.card)).toEqual(["cooling", "heat", "trust", "climate"]);
    expect(cards.every((c) => c.dataset.state === "ready")).toBe(true);

    const cooling = card(container, "cooling");
    expect(cooling.querySelector(".label")!.textContent).toBe("Cooling potential");
    expect(cooling.querySelector(".value")!.textContent).toBe("1.31°F cooler");
    expect(cooling.textContent).toContain("Cooling package, city-wide average");
    expect(cooling.textContent).toContain("likely 0.56–2.05 °F cooler");
    expect(cooling.querySelector("a")!.getAttribute("href")).toBe(`/r/${rid}/lab/library`);

    const heat = card(container, "heat");
    expect(heat.querySelector(".label")!.textContent).toBe("Residents at heat risk");
    expect(heat.querySelector(".value")!.textContent).toBe("36,669residents");
    expect(heat.textContent).toContain("Extreme caution or worse");
    expect(heat.textContent).toContain("21% of 173,661 residents");
    expect(heat.querySelector("a")!.getAttribute("href")).toBe(`/r/${rid}/heat`);

    const trust = card(container, "trust");
    expect(trust.querySelector(".label")!.textContent).toBe("How much to trust this");
    expect(trust.querySelector(".value")!.textContent).toBe("Good");
    expect(trust.dataset.grade).toBe("good");
    expect(trust.querySelector("a")!.getAttribute("href")).toBe(`/r/${rid}/trust`);

    const climate = card(container, "climate");
    expect(climate.querySelector(".value")!.textContent).toBe("3.2°F warmer");
    expect(climate.textContent).toContain("by 2041–2060 (SSP2-4.5)");
    expect(climate.textContent).toContain("2.5–4.1 °F across 6 climate models");
    expect(climate.querySelector("a")!.getAttribute("href")).toBe(`/r/${rid}/heat/future`);

    // the package KPI is in the overview: the configured scenarios are not fetched
    expect(m.calls.some((c) => c.url.includes("/views/scenarios"))).toBe(false);
    m.restore();
  });

  it("leaves out run metadata, timings, the outputs grid and accuracy numbers", async () => {
    const rid = nextRid();
    const { container, m } = await mount(rid, views(rid));
    const text = container.textContent!;
    for (const gone of ["5a04f44", "Run directory", "Checkpoint", "Stage timings", "Outputs", "R²", "RMSE", "coverage", "0.815", "89.0%"]) expect(text).not.toContain(gone);
    expect(container.querySelector('[aria-label="Run header"]')).toBeNull();
    expect(container.querySelector('[aria-label="Run key numbers"]')).toBeNull();
    expect(container.querySelector("[data-output]")).toBeNull();
    expect(container.querySelector('[aria-label="Run progress"]')).toBeNull();
    m.restore();
  });

  it("collapses the caveats into one closed disclosure (limitations linked to Trust), and offers next steps", async () => {
    const rid = nextRid();
    const { container, m } = await mount(rid, views(rid));
    const details = container.querySelectorAll("details");
    expect(details).toHaveLength(1);
    expect(details[0].open).toBe(false);
    expect(details[0].querySelector("summary")!.textContent).toBe("Things to keep in mind (1)");
    expect(details[0].textContent).toContain("Generic physics forcing");
    // the model-card limitations live on How much to trust this, once
    expect(details[0].textContent).not.toContain("Daytime (afternoon) model");
    expect(details[0].querySelector("a")!.getAttribute("href")).toBe(`/r/${rid}/trust#limitations`);
    // the one home of the run's caveats: How much to trust this links here
    expect(details[0].id).toBe("keep-in-mind");
    const next = container.querySelector('nav[aria-label="Next steps"]')!;
    expect([...next.querySelectorAll("a")].map((a) => [a.textContent, a.getAttribute("href")])).toEqual([
      ["Explore the map", `/r/${rid}/map`],
      ["Try a scenario", `/r/${rid}/lab`],
      ["See what cools", `/r/${rid}/effects`],
    ]);
    m.restore();
  });

  it("opens Things to keep in mind for a #keep-in-mind link", async () => {
    const rid = nextRid();
    const { container, m } = await mount(rid, views(rid), "#keep-in-mind");
    expect(container.querySelector<HTMLDetailsElement>("details#keep-in-mind")!.open).toBe(true);
    m.restore();
  });

  it("while the run is running shows the stage rail with a link to the run log", async () => {
    const rid = nextRid();
    const stages: RunStageRow[] = [
      { id: "S0", state: "done", seconds: 2, source: "events", reason: null },
      { id: "S1", state: "done", seconds: 30, source: "events", reason: null },
      { id: "S2_S3", state: "running", seconds: null, source: "events", reason: null },
      { id: "S4", state: "pending", seconds: null, source: "events", reason: null },
    ];
    const detail = { ...runDetail(rid, { status: "running", r2: null, coverage: null }), stages };
    const missing = (view: "heat" | "climate" | "overview") => ({ ...viewFixture(view), availability: "missing" as const });
    const { container, m } = await mount(
      rid,
      views(rid, {
        [`GET /api/runs/${rid}`]: { body: detail },
        [`GET /api/runs/${rid}/views/overview`]: { body: { ...viewFixture("overview"), availability: "running", sections: { ...viewFixture("overview").sections, kpis: [] } } },
        [`GET /api/runs/${rid}/views/heat`]: { body: missing("heat") },
        [`GET /api/runs/${rid}/views/climate`]: { body: missing("climate") },
        [`GET /api/runs/${rid}/views/scenarios`]: { body: { ...viewFixture("scenarios"), availability: "missing" } },
      }),
    );
    const progress = container.querySelector<HTMLElement>('[aria-label="Run progress"]')!;
    expect(progress.dataset.progress).toBe("running");
    expect(progress.querySelector('a[href$="/details/runlog"]')!.getAttribute("href")).toBe(`/r/${rid}/details/runlog`);
    const rail = progress.querySelector('ol[aria-label="Pipeline stages"]')!;
    expect([...rail.querySelectorAll("li")].map((li) => li.getAttribute("data-status"))).toEqual(["done", "done", "running", "todo"]);
    expect(rail.querySelector('[aria-current="step"]')!.getAttribute("data-stage")).toBe("S2_S3");
    // stage labels come from the overview's timings when it has them
    expect(rail.querySelector('[data-stage="S2_S3"] [title]')!.getAttribute("title")).toBe("Models + stacker");

    expect(card(container, "cooling").dataset.state).toBe("absent");
    expect(card(container, "cooling").textContent).toContain("Appears once the scenarios stage finishes");
    expect(card(container, "heat").textContent).toContain("Appears when the run finishes");
    expect(card(container, "trust").textContent).toContain("Graded once the model has been tested");
    expect(card(container, "climate").textContent).toContain("Appears once the climate stage finishes");
    m.restore();
  });

  it("falls back to the configured scenario that cools most and the overview's warming KPI", async () => {
    const rid = nextRid();
    const overview = viewFixture("overview");
    overview.sections.kpis = (overview.sections.kpis ?? []).filter((k) => k.id !== "package" && k.id !== "headline");
    const scen = viewFixture("scenarios");
    scen.sections.rows = (scen.sections.rows ?? []).filter((r) => r.slug !== "cooling-package");
    const heat = viewFixture("heat");
    heat.sections.today = null;
    heat.sections.humidity = { ...heat.sections.humidity!, needs_input: true };
    const { container, m } = await mount(
      rid,
      views(rid, {
        [`GET /api/runs/${rid}/views/overview`]: { body: overview },
        [`GET /api/runs/${rid}/views/scenarios`]: { body: scen },
        [`GET /api/runs/${rid}/views/heat`]: { body: heat },
        [`GET /api/runs/${rid}/views/climate`]: { status: 404, body: { error: { code: "not_found", message: "no climate" } } },
      }),
    );
    await waitFor(() => card(container, "cooling").dataset.state === "ready", 5000, "scenarios fallback");
    expect(card(container, "cooling").textContent).toContain("Canopy Increase +20, city-wide average");
    expect(card(container, "cooling").querySelector(".value")!.textContent).toBe("1.50°F cooler");
    expect(card(container, "climate").querySelector(".value")!.textContent).toBe("3.2°F warmer");
    expect(card(container, "climate").textContent).toContain("Mid-century warming (SSP2-4.5)");
    expect(card(container, "heat").dataset.state).toBe("absent");
    expect(card(container, "heat").textContent).toContain("Needs a humidity setting");
    m.restore();
  });
});

describe("Summary card builders", () => {
  it("coolingPick prefers the package, then the headline scenario, then the coolest configured scenario", () => {
    const kpis = viewFixture("overview").sections.kpis!;
    const rows = viewFixture("scenarios").sections.rows!;
    expect(coolingPick(kpis, rows)!.name).toBe("Cooling package");
    expect(coolingPick(kpis.filter((k) => k.id !== "package"), rows)!.name).toBe("Canopy Increase +10");
    expect(coolingPick([], rows)!.name).toBe("Canopy Increase +20");
    expect(coolingPick([], rows.slice(0, 3))!.likely.estimate).toBeCloseTo(-1.4998, 3);
    expect(coolingPick([], [])).toBeNull();
    expect(coolingPick(null, null)).toBeNull();
  });

  it("warmingPick takes SSP2-4.5 for 2041–2060, else another mid-century scenario, never another period", () => {
    const warming = viewFixture("climate").sections.warming!;
    expect(warmingPick(warming)!.id).toBe("ssp245:2041-2060");
    expect(warmingPick(warming.filter((p) => p.experiment !== "ssp245"))!.period).toBe("2041-2060");
    expect(warmingPick(warming.filter((p) => p.period !== "2041-2060"))).toBeNull();
    expect(warmingPick(null)).toBeNull();
  });

  it("trustInputs reads the overview KPIs, else the run summary", () => {
    const kpis = viewFixture("overview").sections.kpis!;
    expect(trustInputs(kpis, null)).toEqual({ r2: 0.815, coverage: 0.8902, coverageTarget: 0.9 });
    expect(trustInputs([], { r2: 0.5, coverage: 0.8 })).toEqual({ r2: 0.5, coverage: 0.8, coverageTarget: null });
  });

  it("keepInMind lists the caveats once each, not the limitations", () => {
    const o = viewFixture("overview");
    o.caveats = ["A", "B", " A "];
    o.sections.limitations = ["B", "C"];
    expect(keepInMind(o)).toEqual(["A", "B"]);
    expect(keepInMind(undefined)).toEqual([]);
  });
});
