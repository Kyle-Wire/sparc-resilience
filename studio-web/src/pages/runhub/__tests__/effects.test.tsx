// What cools (`/r/:rid/effects`): one section per lever with a plain sentence on diminishing
// returns, its dose–response curve (likely range, "most of the benefit" marker), how far its
// cooling spreads (from the Influence view) and the curve shapes in the glossary's words, Where to
// act, and the link to Scenario Lab sweeps. The technical parts stay under Model details, not here.
import { afterEach, describe, expect, it, vi } from "vitest";
import { clearResources } from "../../../api/resource";
import type { ResponseLever, ViewModelOf } from "../../../api/runs";
import { navigate } from "../../../router";
import { flush, mockFetch, render, waitFor, type MockHandler } from "../../../test/render";
import { olderCodeFixture, viewFixture } from "../__fixtures__/views";
import Effects, { leverAnchor } from "../Effects";
import Influence, { REACH_TITLE } from "../Influence";
import Response, { MOST_OF_THE_BENEFIT, leverName, saturationText, shapeLabel, shapeShares } from "../Response";
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
const CURVE = "Cooling by amount";
const SHAPES = "How the cooling builds up, place by place";
const hasReach = (root: HTMLElement) => charts(root).includes(REACH_TITLE);
const lever = (key: string) => viewFixture("response").sections.levers![key];

describe("What cools", () => {
  afterEach(() => clearResources());

  it("shows one section per lever, named in plain words: curve with its likely range, how far the cooling spreads, and the curve shapes", async () => {
    const r = await renderEffects();
    try {
      await waitFor(() => hasReach(r.container), 3000, "reach charts");
      const levers = Object.entries(viewFixture("response").sections.levers!);
      for (const [key, L] of levers) {
        const sec = r.container.querySelector<HTMLElement>(`#${leverAnchor(key)}`)!;
        expect(sec, key).not.toBeNull();
        expect(sec.querySelector("h3")!.textContent).toBe(leverName(key, L));
        expect(charts(sec)).toEqual([CURVE, REACH_TITLE, SHAPES]);
        // each chart explains itself in an ⓘ beside its title
        expect([...sec.querySelectorAll("figure.chart-frame .chart-note button.infotip")].map((b) => b.getAttribute("data-term"))).toEqual(["dose_response", "ring_kernel", "saturation"]);
      }
      expect(leverName("canopy", lever("canopy"))).toBe("Trees & canopy");
      expect(leverName("impervious", lever("impervious"))).toBe("Less paving");
      expect(unframedCharts(r.container)).toEqual([]);
      expect(r.container.textContent).not.toContain("could not be displayed");
      expect(r.container.querySelectorAll("[data-older-code]").length).toBe(0);
      // tooltips are InfoTips and Terms, never a native title
      expect(r.container.querySelector("[title]")).toBeNull();
    } finally {
      r.restore();
    }
  });

  it("says plainly where each extra step pays off less, and by how much most of the cooling comes", async () => {
    const r = await renderEffects();
    try {
      const canopy = r.container.querySelector<HTMLElement>(`#${leverAnchor("canopy")} p[data-saturation="canopy"]`)!;
      expect(canopy.textContent).toBe("In every place, each extra step of tree canopy cools less than the last (diminishing returns). About 90% of the possible cooling comes by +32 pts.");
      expect([...canopy.querySelectorAll("button.infotip")].map((b) => b.getAttribute("data-term"))).toEqual(["saturation", "d90"]);
      // no median d90 for this lever: the clause is left out
      const paving = r.container.querySelector<HTMLElement>(`#${leverAnchor("impervious")} p[data-saturation="impervious"]`)!;
      expect(paving.textContent).toBe("In 24% of places, each extra cut in paving cools less than the last (diminishing returns). In 48%, each extra cut cools about as much as the last (steady returns).");
      expect([...paving.querySelectorAll("button.infotip")].map((b) => b.getAttribute("data-term"))).toEqual(["saturation"]);
    } finally {
      r.restore();
    }
  });

  it("stacks the curve shapes in the glossary's words, each place counted once", async () => {
    // the server adds the share of places with too few doses to fit (views._response: 1 − the fitted shapes)
    const vm = viewFixture("response");
    vm.sections.levers!.impervious.shapes!.insufficient = 0.1508;
    const r = await renderEffects({ response: vm });
    try {
      const fig = r.container.querySelector<HTMLElement>(`#${leverAnchor("impervious")} figure[data-chart="${SHAPES}"]`)!;
      expect([...fig.querySelectorAll(".legend-row > span")].map((x) => x.textContent)).toEqual([
        "Diminishing returns",
        "Steady returns",
        "Slow start, then strong",
        "Hit the limit",
        "Too few amounts to tell",
      ]);
      expect(fig.querySelector("figcaption")!.textContent).toBe(
        "Less paving, share of places: diminishing returns 24%; steady returns 48%; slow start, then strong 0%; hit the limit 13%; too few amounts to tell 15%.",
      );
    } finally {
      r.restore();
    }
  });

  it("marks where a typical place gets most of the benefit on the curve and gives the top amount with its likely range", async () => {
    const r = await renderEffects();
    try {
      const canopy = r.container.querySelector<HTMLElement>(`#${leverAnchor("canopy")} figure[data-chart="${CURVE}"]`)!;
      expect([...canopy.querySelectorAll("svg text")].map((t) => t.textContent)).toContain(MOST_OF_THE_BENEFIT);
      const cap = canopy.querySelector("figcaption")!.textContent!;
      expect(cap).toContain("If every place adds 40 pts of tree canopy (where there is room), the city cools by 1.85 °F on average (likely range 0.63–3.07 °F).");
      expect(cap).toContain("The “most of the benefit” line: by +32 pts, a typical place gets about 90% of the cooling it can get.");
      expect(canopy.querySelector(".chart-note button.term")!.getAttribute("data-term")).toBe("interval_95");
      // no median d90: no marker
      const paving = r.container.querySelector<HTMLElement>(`#${leverAnchor("impervious")} figure[data-chart="${CURVE}"]`)!;
      expect([...paving.querySelectorAll("svg text")].map((t) => t.textContent)).not.toContain(MOST_OF_THE_BENEFIT);
      // a likely range that crosses zero: "might not cool at all", each end as cooler or warmer, never a negative cooling
      expect(paving.querySelector("figcaption")!.textContent).toContain(
        "If every place removes 20 pts of paving (where there is room), the city cools by 0.35 °F on average, but it might not cool at all (likely range: 0.25 °F warmer to 0.95 °F cooler).",
      );
    } finally {
      r.restore();
    }
  });

  it("titles the reach chart How far the cooling spreads, marks the reach and links to it on the map", async () => {
    const r = await renderEffects();
    try {
      const sec = await waitFor(() => r.container.querySelector<HTMLElement>(`#${leverAnchor("canopy")} figure[data-chart="${REACH_TITLE}"]`), 3000, "canopy reach");
      expect(sec.querySelector("figcaption")!.textContent).toContain("Trees & canopy: most of its effect on temperature (90%) is felt within 315 m.");
      const tip = sec.querySelector<HTMLElement>(".chart-note button.infotip")!;
      expect(tip.getAttribute("data-term")).toBe("ring_kernel");
      expect(tip.getAttribute("aria-label")).toBe(`What is ${REACH_TITLE}?`);
      const a = [...sec.querySelectorAll("a")].find((x) => x.textContent === "Show the reach on the map")!;
      expect(a.getAttribute("href")).toBe(`/r/${r.rid}/map?layer=canopy&ov=influence&infl=canopy`);
      expect(sec.querySelector("figcaption")!.textContent).toContain("Below the zero line, more tree canopy nearby makes a place cooler.");
      // a lever that cools by going down: the zero line says what removing it does
      const paving = await waitFor(() => r.container.querySelector<HTMLElement>(`#${leverAnchor("impervious")} figure[data-chart="${REACH_TITLE}"]`), 3000, "paving reach");
      expect(paving.querySelector("figcaption")!.textContent).toContain("Above the zero line: paving nearby makes a place warmer, so removing it cools.");
      expect(paving.querySelector(".chart-units")!.textContent).toBe("Units: % of the strongest effect of paving nearby (negative = cooler)");
    } finally {
      r.restore();
    }
  });

  it("leaves the technical parts to Model details", async () => {
    const r = await renderEffects();
    try {
      await waitFor(() => hasReach(r.container), 3000, "reach charts");
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
      expect(r.container.querySelectorAll(`[id^="lever-"] figure[data-chart="${CURVE}"]`).length).toBe(3);
      expect(r.container.textContent).toContain(`Trees & canopy: ${REACH_TITLE.toLowerCase()}: not in this run (older code)`);
    } finally {
      r.restore();
    }
  });

  it("says when the reach could not be loaded and still shows the curves", async () => {
    const r = await renderEffects({ influence: { status: 500, body: { error: { code: "internal", message: "boom" } } } });
    try {
      await waitFor(() => r.container.textContent?.includes("Trees & canopy: how far the cooling spreads could not be loaded."), 3000, "reach error");
      expect(r.container.querySelectorAll(`[id^="lever-"] figure[data-chart="${CURVE}"]`).length).toBe(3);
    } finally {
      r.restore();
    }
  });

  it("says when no lever was swept", async () => {
    const vm = viewFixture("response");
    vm.sections.levers = {};
    const r = await renderEffects({ response: vm });
    try {
      expect(r.container.textContent).toContain("This run did not test any of the changes a plan can make (no levers were swept).");
    } finally {
      r.restore();
    }
  });
});

describe("diminishing returns in plain words", () => {
  const shapes = (x: Partial<NonNullable<ResponseLever["shapes"]>>): NonNullable<ResponseLever["shapes"]> => ({ saturating: null, linear: null, sigmoid: null, censored: null, ...x });
  const withShapes = (L: ResponseLever, s: NonNullable<ResponseLever["shapes"]> | null, d90: number | null = null): ResponseLever => ({
    ...L,
    shapes: s,
    effects: L.effects ? { ...L.effects, median_d90: d90 } : null,
  });

  it("counts each place in one shape: a place still cooling at its limit leaves the saturating (then S-shaped) share", () => {
    expect(shapeShares(shapes({ saturating: 0.2562, linear: 0.7438, sigmoid: 0, censored: 0.2036 }))).toEqual({
      saturating: expect.closeTo(0.0526, 6),
      linear: 0.7438,
      sigmoid: 0,
      censored: 0.2036,
      insufficient: null,
    });
    // more censored places than saturating ones: the rest come out of the S-shaped share
    const s = shapeShares(shapes({ saturating: 0.1, linear: 0.5, sigmoid: 0.3, censored: 0.25, insufficient: 0.1 }));
    expect(s.saturating).toBe(0);
    expect(s.sigmoid).toBeCloseTo(0.15, 6);
    expect((s.saturating ?? 0) + (s.linear ?? 0) + (s.sigmoid ?? 0) + (s.censored ?? 0) + (s.insufficient ?? 0)).toBeCloseTo(1, 6);
    expect(shapeLabel("saturating")).toBe("Diminishing returns");
    expect(shapeLabel("linear")).toBe("Steady returns");
    expect(shapeLabel("sigmoid")).toBe("Slow start, then strong");
    expect(shapeLabel("censored")).toBe("Hit the limit");
  });

  it("leads with diminishing returns, adds the biggest other shape and the 90% amount, and leaves out what the run lacks", () => {
    const canopy = lever("canopy");
    expect(saturationText("canopy", withShapes(canopy, shapes({ saturating: 0.72, linear: 0.28, sigmoid: 0, censored: 0 }), 18))).toEqual({
      shapes: "In 72% of places, each extra step of tree canopy cools less than the last (diminishing returns).",
      d90: "About 90% of the possible cooling comes by +18 pts.",
    });
    // no saturating places: the biggest shape leads, in full
    expect(saturationText("albedo", withShapes(lever("albedo"), shapes({ saturating: 0, linear: 0.4, sigmoid: 0, censored: 0.6 })))).toEqual({
      shapes: "In 60% of places, it is still cooling at the most the space allows (hit the limit).",
      d90: null,
    });
    // a lever that cools when it goes down reads as a cut
    expect(saturationText("impervious", withShapes(lever("impervious"), shapes({ saturating: 0.6, linear: 0.4 }), 12))!.d90).toBe("About 90% of the possible cooling comes by a cut of 12 pts.");
    expect(saturationText("canopy", withShapes(canopy, null, 18))).toBeNull();
  });
});

describe("Model details keeps the technical names, each with an ⓘ", () => {
  afterEach(() => clearResources());

  /** The ⓘ (or Term) ids in a chart's header note, by chart title. */
  const noteTips = (root: HTMLElement, title: string) =>
    [...root.querySelectorAll(`figure.chart-frame[data-chart="${title}"] .chart-note button`)].map((b) => b.getAttribute("data-term") ?? b.getAttribute("aria-label"));

  it("Response: realised dose, own vs footprint, fold means and the literature check", async () => {
    const rid = `r_eff${++seq}`;
    const m = mockFetch({ ...mapRoutes(rid), [`GET /api/runs/${rid}/views/response`]: { body: viewFixture("response") } });
    navigate(`/r/${rid}/details/response`, { replace: true });
    const r = render(<Response detailsOnly />);
    try {
      await waitFor(() => r.container.querySelector('figure.chart-frame[data-chart="Fold means"]'), 5000, "response details");
      await flush(4);
      expect(noteTips(r.container, "canopy: requested vs realised dose")).toEqual(["realised"]);
      expect(noteTips(r.container, "Fold means")).toEqual(["held_out_fold"]);
      expect(noteTips(r.container, "Literature check")).toEqual(["What is Literature check?"]);
      const terms = r.container.querySelector('[data-effect-terms="true"]')!;
      expect([...terms.querySelectorAll("button.term")].map((b) => b.getAttribute("data-term"))).toEqual(["own_cell", "footprint", "d90"]);
      expect(terms.textContent).toContain("Footprint / own-cell ratio 29.2");
      expect(r.container.querySelector("[title]")).toBeNull();
    } finally {
      m.restore();
    }
  });

  it("Influence: influence ranges, correlogram, ring kernels, anisotropy and the length-scale priors", async () => {
    const rid = `r_eff${++seq}`;
    const m = mockFetch({ ...mapRoutes(rid), [`GET /api/runs/${rid}/views/influence`]: { body: viewFixture("influence") } });
    navigate(`/r/${rid}/details/reach`, { replace: true });
    const r = render(<Influence detailsOnly />);
    try {
      await waitFor(() => r.container.querySelector('[aria-label="Length scales"]'), 5000, "influence details");
      await flush(4);
      expect(noteTips(r.container, "Influence range per predictor")).toEqual(["length_scale"]);
      expect(noteTips(r.container, "Target correlogram")).toEqual(["What is Target correlogram?"]);
      expect(noteTips(r.container, "Ring-kernel β by distance")).toEqual(["ring_kernel"]);
      expect(noteTips(r.container, "Anisotropy roses")).toEqual(["anisotropy"]);
      const priors = r.container.querySelector<HTMLElement>('[aria-label="Length scales"]')!.closest("section.card")!;
      expect(priors.querySelector("header button.infotip")!.getAttribute("data-term")).toBe("priors");
      expect([...priors.querySelectorAll("dt button.infotip")].map((b) => b.getAttribute("aria-label"))).toEqual([
        "What is Relaxation length prior L?",
        "What is CV block size?",
        "What is Residual range?",
      ]);
      expect(r.container.querySelector("[title]")).toBeNull();
    } finally {
      m.restore();
    }
  });
});
