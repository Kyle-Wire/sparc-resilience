// The full-model result (phase 3): the answer card (plain headline, big number on a likely-range strip
// with a zero line, the confidence chip in words and never crimson for cooling, up to four facts,
// actions) and the accordion under it, where nothing repeats: the cost shows once (in the card), the
// cooling outside your changes once (Where it cools), realised vs requested only in "What was actually
// changed". Impacts are worked out once, on their own, as a result without them arrives; the card and
// the two sections share that request (a button only to try again after an error).
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Result } from "../../../api/lab";
import { clearResources } from "../../../api/resource";
import type { Likely } from "../../../api/types";
import { byText, click, mockFetch, render, waitFor, type MockHandler } from "../../../test/render";
import { AnswerCard, RangeStrip, answerHeadline, citywideFact, rangeStripText, resultHasCost } from "../components/result/AnswerCard";
import { causalQualifier } from "../model/plain";
import { resultLayerMetas } from "../model/layers";
import { ResultView } from "../components/result/ResultView";
import { L, LEVERS, PID, RID, result } from "../__fixtures__/lab";

let fetchMock: ReturnType<typeof mockFetch> | null = null;

beforeEach(() => clearResources());

afterEach(() => {
  fetchMock?.restore();
  fetchMock = null;
});

const SECTIONS = ["where", "heat", "future", "changed", "sure", "tech"] as const;

/** The server's answer for a run without resident data (no impacts to work out). */
const NEEDS_LAYERS: MockHandler = { status: 422, body: { error: { code: "needs_layers", message: "impacts need the planner layers (planner.layers: residents and land cover)" } } };

function routes(res: Result, extra: Record<string, MockHandler> = {}): Record<string, MockHandler> {
  return {
    [`GET /api/results/${res.summary.id}`]: { body: res },
    [`GET /api/runs/${RID}/levers`]: { body: LEVERS },
    "GET /api/jobs": { body: { items: [], next_cursor: null } },
    [`POST /api/results/${res.summary.id}/impacts`]: NEEDS_LAYERS,
    ...extra,
  };
}

/** Impacts with one heat-risk row today (120 residents out of Extreme caution) and exposure today and mid-century. */
const IMPACTS = {
  thresholds: [90],
  exposure: [
    { case: "today", adapted: false, person_mean_temp: 91, people_ge: { "90": 400 }, share_people_ge: { "90": 0.4 } },
    { case: "today", adapted: true, person_mean_temp: 90.5, people_ge: { "90": 300 }, share_people_ge: { "90": 0.3 } },
    { case: "ssp245 2041-2060", adapted: false, person_mean_temp: 93, people_ge: { "90": 900 }, share_people_ge: { "90": 0.9 } },
    { case: "ssp245 2041-2060", adapted: true, person_mean_temp: 92.6, people_ge: { "90": 850 }, share_people_ge: { "90": 0.85 } },
  ],
  equity: {}, hot_days: null, hot_days_action: null, zones: [], hexes: { "250": [], "500": [] },
  climate_offset: [{ experiment: "ssp245", period: "2041-2060", offset_share: 0.07 }],
  heat: {
    dewpoint_C: 16, source: null, measure: "people", method: "NWS", headline: "",
    rows: [{ case: "today", humidity: "observed", warming_F: 0, before: { counts: {}, ec_or_worse: 500, danger_or_worse: 0, mean_hi: 90, max_hi: 95 }, after: { counts: {}, ec_or_worse: 380, danger_or_worse: 0, mean_hi: 89, max_hi: 94 }, ec_avoided: 120, danger_avoided: 0, mean_hi_change: -0.4 }],
  },
};

/** A result whose edited-area Likely is `edited` (the card's main number). */
function withEdited(id: string, edited: Likely, extra: Partial<Result> = {}): Result {
  const base = result(id, extra);
  return { ...base, summary: { ...base.summary, edited } };
}

/** A result with a real cost model (the scenario prices canopy), total 27,600. */
function costed(id: string): Result {
  return result(id, {
    spec: { costs: { Pct_Canopy: { per_unit: 1000 } } },
    cost: { total: 27600, per_lever: { Pct_Canopy: 27600 }, cooling_per_cost: 0.0000747 },
  });
}

function openSection(container: HTMLElement, id: string): HTMLElement {
  const details = container.querySelector<HTMLDetailsElement>(`details.rv-section[data-section="${id}"]`);
  if (!details) throw new Error(`no result section ${id}`);
  if (!details.open) click(details.querySelector("summary"));
  return details;
}

const count = (text: string, needle: string) => text.split(needle).length - 1;

describe("Answer card", () => {
  it("leads with a numberless headline: the big number and the likely-range strip carry the figures, once", async () => {
    // the server's headline repeats the figures: the card does not use it
    const plain = { headline: "Cools the edited area by 0.62 °F (likely range 0.44–0.80 °F). City-wide: 0.021 °F cooler.", confidence: "", qualifiers: [], buys: [] };
    fetchMock = mockFetch(routes(result("res_1", { plain })));
    const a = render(<ResultView rid={RID} resId="res_1" pid={PID} unit="degF" nFolds={3} />);
    const h = await waitFor(() => a.container.querySelector(".result-inspector .plain-result .headline"), 5000, "headline");
    expect(h.textContent).toBe("Cools the places you changed");
    const card = a.container.querySelector(".answer-card")!;
    expect(card.querySelector(".kicker")!.textContent).toBe("FULL MODEL RESULT");
    expect(card.querySelector(".answer-big")!.textContent).toBe("−0.62 °F");
    expect(card.textContent).toContain("where you made changes");
    expect(card.querySelector(".badge")!.textContent).toBe("DEMO");
    expect(count(card.textContent ?? "", "0.62")).toBe(1); // the big number only
  });

  it("words the headline by what happens and where: the whole city, the city, or a range that crosses zero", () => {
    expect(answerHeadline(result("r"))).toBe("Cools the places you changed");
    // the changes cover every place of the run (7 cells): one figure, for the whole city
    const all = result("r_all", { regions: [{ ...result("x").regions[0], n_cells: 7 }] });
    expect(answerHeadline(all, 7)).toBe("Cools the whole city");
    expect(answerHeadline(all, 8)).toBe("Cools the places you changed");
    const cityOnly = { ...result("r2"), summary: { ...result("r2").summary, edited: null } };
    expect(answerHeadline(cityOnly)).toBe("Cools the city");
    expect(answerHeadline(withEdited("r3", L(-0.1, -0.3, 0.1, 0.1)))).toBe("May cool the places you changed");
    expect(answerHeadline(withEdited("r4", L(0.5, 0.3, 0.7, 0.1)))).toBe("Warms the places you changed");
  });

  it("gives the citywide fact at the big number's precision, and leaves it out when it would repeat it", () => {
    expect(citywideFact(result("r"), "degF")).toBe("−0.02 °F");
    // the changes cover the whole city: the big number is the citywide figure
    const all = result("r_all", { regions: [{ ...result("x").regions[0], n_cells: 7 }] });
    expect(citywideFact(all, "degF", 7)).toBeNull();
    // the two round to the same figure
    const same = withEdited("r_same", L(-1.031, -1.5, -0.5, 0.28), { city: L(-1.029, -1.5, -0.5, 0.28) });
    expect(citywideFact(same, "degF")).toBeNull();
    // a citywide-only result has no separate citywide fact
    expect(citywideFact({ ...result("r2"), summary: { ...result("r2").summary, edited: null } }, "degF")).toBeNull();
    // on the card: a change of the whole city reads as one figure, citywide
    fetchMock = mockFetch({ [`GET /api/runs/${RID}/levers`]: { body: LEVERS } });
    const { container } = render(<AnswerCard rid={RID} resId="r_all" pid={PID} result={all} unit="degF" nCells={7} />);
    expect(container.querySelector(".answer-headline")!.textContent).toBe("Cools the whole city");
    expect(container.querySelector(".answer-where")!.textContent).toBe("citywide");
    expect(container.querySelector('.answer-fact[data-fact="city"]')).toBeNull();
  });

  it("links the cross-check qualifier to How sure are we, worded by what the check finds, opening that section", async () => {
    // the model says −0.021 °F citywide; the cross-check finds −0.4 °F: more cooling
    fetchMock = mockFetch(routes(result("res_q", { causal_check: { delta: -0.4, lo: -0.55, hi: -0.27, model_within: false } })));
    const { container } = render(<ResultView rid={RID} resId="res_q" pid={PID} unit="degF" nFolds={3} />);
    const line = await waitFor(() => container.querySelector('.answer-card [data-qualifier="causal"]'), 5000, "causal qualifier");
    expect(line.textContent).toBe("A separate cross-check finds more cooling — see How sure are we");
    expect(line.textContent).toBe(causalQualifier(-0.4, -0.021));
    expect(container.textContent).not.toContain("independent causal check disagrees");
    expect(container.textContent).not.toContain("gives a different answer");
    const sure = container.querySelector<HTMLDetailsElement>('details.rv-section[data-section="sure"]')!;
    expect(sure.open).toBe(false);
    click(line.querySelector("button"));
    await waitFor(() => sure.open && sure.querySelector(".rv-section-body"), 5000, "the sure section");
    const causal = sure.querySelector(".rv-causal")!.textContent ?? "";
    expect(causal).toContain("Cause-and-effect check disagrees");
    // plain range words at the card's precision, no "between … to …", no 3 decimals
    expect(causal).toContain("between −0.55 and −0.27 °F");
    expect(causal).not.toMatch(/\d\.\d{3}/);
    // the uncertainty chart's caption names no internal source ids
    expect(sure.textContent).not.toContain("Sources:");
  });

  it("words the cross-check by direction: less or more cooling or warming, or the opposite effect", () => {
    expect(causalQualifier(-0.41, -1.03)).toBe("A separate cross-check finds less cooling — see How sure are we");
    expect(causalQualifier(-1.5, -1.03)).toBe("A separate cross-check finds more cooling — see How sure are we");
    expect(causalQualifier(0.2, -1.03)).toBe("A separate cross-check finds the opposite effect — see How sure are we");
    expect(causalQualifier(0.1, 0.5)).toBe("A separate cross-check finds less warming — see How sure are we");
    expect(causalQualifier(null, -1)).toBe("A separate cross-check gives a different answer — see How sure are we");
  });

  it("words the confidence chip per confidence and never paints cooling crimson", () => {
    fetchMock = mockFetch({ [`GET /api/runs/${RID}/levers`]: { body: LEVERS } });
    const cases: [Likely, string, string][] = [
      [L(-0.62, -0.8, -0.44, 0.09), "Confident it cools", "cool"],
      [L(0.5, 0.3, 0.7, 0.1), "Confident it warms", "warm"],
      [L(-0.1, -0.3, 0.1, 0.1), "Might be no real effect", "neutral"],
      [L(-0.1, null, null, null), "No range available", "neutral"],
    ];
    for (const [edited, chip, tone] of cases) {
      const { container, unmount } = render(<AnswerCard rid={RID} resId="res_c" pid={PID} result={withEdited("res_c", edited)} unit="degF" />);
      const card = container.querySelector<HTMLElement>(".answer-card")!;
      expect(card.querySelector(".answer-chip")!.textContent).toBe(chip);
      expect(card.getAttribute("data-tone")).toBe(tone);
      expect(card.querySelector(".answer-chip")!.getAttribute("data-conf")).toBe(tone);
      // nothing in the card uses the crimson accent or the critical tone
      expect(card.querySelector('[data-tone="accent"], [data-tone="crit"]')).toBeNull();
      expect(card.style.getPropertyValue("--vcard-accent")).not.toContain("accent");
      unmount();
    }
  });

  it("puts the estimate on a zero line: a cooling range sits left of zero in the cool tint", () => {
    const { container } = render(<RangeStrip estimate={-0.62} lo={-0.8} hi={-0.44} unit="degF" confidence="confident_cools" />);
    const svg = container.querySelector("svg.answer-strip")!;
    expect(svg.getAttribute("data-tone")).toBe("cool");
    expect(svg.getAttribute("data-straddles")).toBeNull();
    const bars = [...svg.querySelectorAll(".answer-strip-bar")];
    expect(bars.map((b) => b.getAttribute("data-side"))).toEqual(["cool"]);
    const zero = Number(svg.querySelector(".answer-strip-zero")!.getAttribute("x1"));
    const dot = Number(svg.querySelector(".answer-strip-dot")!.getAttribute("cx"));
    const bar = bars[0];
    expect(dot).toBeLessThan(zero);
    expect(Number(bar.getAttribute("x")) + Number(bar.getAttribute("width"))).toBeLessThan(zero);
    expect(svg.querySelector("title")!.textContent).toBe("Likely range −0.80 to −0.44 °F; the whole range is below zero. Confident it cools.");
  });

  it("styles a result that might be no real effect as neutral, its range visibly straddling zero", async () => {
    fetchMock = mockFetch(routes(withEdited("res_z", L(-0.1, -0.3, 0.1, 0.1))));
    const { container } = render(<ResultView rid={RID} resId="res_z" pid={PID} unit="degF" nFolds={3} />);
    const card = await waitFor(() => container.querySelector<HTMLElement>(".answer-card"), 5000, "card");
    expect(card.getAttribute("data-tone")).toBe("neutral");
    expect(card.getAttribute("data-confidence")).toBe("could_be_zero");
    expect(card.style.getPropertyValue("--vcard-accent")).toBe("var(--line-strong)");
    const svg = card.querySelector("svg.answer-strip")!;
    expect(svg.getAttribute("data-tone")).toBe("neutral");
    expect(svg.getAttribute("data-straddles")).toBe("true");
    const bars = [...svg.querySelectorAll(".answer-strip-bar")];
    expect(bars.map((b) => b.getAttribute("data-side"))).toEqual(["cool", "warm"]);
    // the two halves meet at the zero line
    const zero = Number(svg.querySelector(".answer-strip-zero")!.getAttribute("x1"));
    expect(Number(bars[0].getAttribute("x")) + Number(bars[0].getAttribute("width"))).toBeCloseTo(zero, 6);
    expect(Number(bars[1].getAttribute("x"))).toBeCloseTo(zero, 6);
    expect(rangeStripText({ estimate: -0.1, lo: -0.3, hi: 0.1, unit: "degF", confidence: "could_be_zero" })).toContain("the range crosses zero. Could be zero.");
    // the cooling result is the cool tone
    expect(card.querySelector(".answer-chip")!.textContent).toBe("Might be no real effect");
  });

  it("shows up to four facts, the cost only with a real cost model, and counts the residents on its own", async () => {
    // the impacts request is held until the card has shown that it is counting
    let release: () => void = () => {};
    const held = new Promise<void>((r) => (release = r));
    fetchMock = mockFetch(
      routes(costed("res_f"), {
        "POST /api/results/res_f/impacts": async () => {
          await held;
          return { body: IMPACTS };
        },
      }),
    );
    const { container } = render(<ResultView rid={RID} resId="res_f" pid={PID} unit="degF" nFolds={3} />);
    await waitFor(() => container.querySelector('.answer-fact[data-fact="cost"]'), 5000, "cost fact");
    const facts = () => [...container.querySelectorAll(".answer-fact")].map((f) => f.getAttribute("data-fact"));
    expect(facts()).toEqual(["city", "heat", "cost", "per"]);
    expect(container.querySelector('.answer-fact[data-fact="city"]')!.textContent).toContain("−0.02 °F");
    expect(container.querySelector('.answer-fact[data-fact="cost"]')!.textContent).toContain("about 27,600");
    expect(container.querySelector('.answer-fact[data-fact="per"]')!.textContent).toContain("Cost per °F");
    // residents: worked out without asking, "Counting residents…" meanwhile, no button
    const heat = () => container.querySelector('.answer-fact[data-fact="heat"]')!;
    expect(heat().querySelector(".answer-fact-value")!.textContent).toBe("Counting residents…");
    expect(heat().querySelector("button")).toBeNull();
    release();
    await waitFor(() => heat().textContent?.includes("120"), 5000, "residents fact");
    expect(heat().querySelector("dt")!.textContent).toBe("Residents out of Extreme caution");
    expect(fetchMock.calls.filter((c) => c.method === "POST" && c.url === "/api/results/res_f/impacts")).toHaveLength(1);
    expect(heat().querySelector("button")).toBeNull();
    // and what it buys, by default: with the residents fact on the card, no second residents figure
    const notes = container.querySelector(".answer-notes")!.textContent ?? "";
    expect(notes).toContain("Cancels out about 7% of the warming expected by 2041–2060 (moderate emissions).");
    expect(notes).not.toMatch(/residents|SSP|ssp/);
  });

  it("labels the likely range on the strip itself", () => {
    const { container } = render(<RangeStrip estimate={-1.03} lo={-1.58} hi={-0.48} unit="degF" confidence="confident_cools" />);
    expect(container.querySelector(".answer-strip-label")!.textContent).toBe("likely range");
    // no range: no label
    const none = render(<RangeStrip estimate={-0.1} lo={null} hi={null} unit="degF" confidence="unknown" />);
    expect(none.container.querySelector(".answer-strip-label")).toBeNull();
  });

  it("leaves the cost out without a cost model (every lever at the default price of 1)", () => {
    expect(resultHasCost(result("r"), LEVERS)).toBe(false);
    expect(resultHasCost(costed("r"), LEVERS)).toBe(true);
    expect(resultHasCost(result("r", { cost: { total: 50, per_lever: { Albedo: 50 }, cooling_per_cost: 0.1 } }), LEVERS)).toBe(true);
    expect(resultHasCost(result("r", { cost: { total: 0, per_lever: {}, cooling_per_cost: null } }), LEVERS)).toBe(false);
  });

  it("pins, exports the decision pack and shows the result on the map", async () => {
    const onShowLayer = vi.fn();
    fetchMock = mockFetch(routes(result("res_a"), { [`POST /api/projects/${PID}/exports`]: { status: 202, body: { job: { id: "j_pack", kind: "export.decision_pack", status: "queued" } } } }));
    const { container } = render(<ResultView rid={RID} resId="res_a" pid={PID} unit="degF" nFolds={3} onShowLayer={onShowLayer} />);
    const actions = await waitFor(() => container.querySelector('[aria-label="Result actions"]'), 5000, "actions");
    const labels = [...actions.querySelectorAll("button")].map((b) => b.textContent);
    expect(labels).toEqual(["Pin to compare", "Export decision pack", "Show on map"]);
    click([...actions.querySelectorAll("button")].find((b) => b.textContent === "Show on map")!);
    expect(onShowLayer).toHaveBeenCalledWith("res:res_a:delta");
  });
});

describe("Result accordion", () => {
  it("builds each section on first open and says each thing once: cost, cooling outside your changes, realised vs requested", async () => {
    fetchMock = mockFetch(routes(costed("res_o")));
    const onShowLayer = vi.fn();
    const { container } = render(<ResultView rid={RID} resId="res_o" pid={PID} unit="degF" nFolds={3} onShowLayer={onShowLayer} />);
    await waitFor(() => container.querySelector(".answer-card"), 5000, "card");
    const sections = [...container.querySelectorAll("details.rv-section")].map((d) => d.getAttribute("data-section"));
    expect(sections).toEqual([...SECTIONS]);
    // closed sections have no body yet
    expect(container.querySelector(".rv-section-body")).toBeNull();
    for (const id of SECTIONS) openSection(container, id);
    await waitFor(() => container.querySelectorAll(".rv-section-body").length === SECTIONS.length, 5000, "every section");
    const text = container.textContent ?? "";
    // the cost: once, in the card
    expect(count(text, "27,600")).toBe(1);
    expect(count(text, "Estimated cost")).toBe(1);
    expect(container.querySelector(".answer-card")!.textContent).toContain("about 27,600");
    // cooling outside your changes: once, in Where it cools
    expect(count(text, "outside your changes")).toBe(1);
    const where = container.querySelector('details[data-section="where"]')!;
    expect(where.textContent).toContain("About 12% of the cooling lands outside your changes");
    // realised vs requested: only in What was actually changed
    const changed = container.querySelector('details[data-section="changed"]')!;
    expect(changed.textContent).toContain("You asked for (average)");
    expect(changed.textContent).toContain("+10.00 pts");
    expect(changed.textContent).toContain("+9.20 pts");
    expect(changed.textContent).toContain("Trees & canopy");
    expect(count(text, "You asked for")).toBe(1);
    expect(count(text, "What fits")).toBe(1);
    // the extrapolation share shows in the card's "Keep in mind" (> 20%), not again in How sure are we
    expect(count(text, "23%")).toBe(1);
    // the quick estimate vs full model gap, and the causal badge linking to the trust page
    const sure = container.querySelector('details[data-section="sure"]')!;
    expect(sure.querySelector("[data-preview-vs-exact]")!.textContent).toContain("off by about 0.004 °F per place");
    const badge = sure.querySelector<HTMLAnchorElement>(".rv-causal a.pill")!;
    expect(badge.textContent).toBe("Cause-and-effect check agrees");
    expect(badge.getAttribute("href")).toBe(`/r/${RID}/trust`);
    // technical details: SE, folds, method, knock-on effects
    const tech = container.querySelector('details[data-section="tech"]')!;
    expect(tech.textContent).toContain("Folds (jackknife)");
    expect(tech.textContent).toContain("NDVI follows +0.040");
    // no KPI row anywhere in the result
    expect(container.querySelector(".kpi")).toBeNull();
    // map layer toggles live in Where it cools
    click([...where.querySelectorAll("button")].find((b) => b.textContent === "Spread between model runs")!);
    expect(onShowLayer).toHaveBeenCalledWith("res:res_o:delta_sd");
    // closing keeps the built body
    openSection(container, "where");
    click(where.querySelector("summary"));
    expect((where as HTMLDetailsElement).open).toBe(false);
    expect(where.querySelector(".rv-section-body")).not.toBeNull();
  });

  it("says a whole-city change has nowhere outside to cool, gives each source of doubt its own row, and keeps the details to what is new", async () => {
    const x = result("x");
    const whole = result("res_w", {
      regions: [{ ...x.regions[0], n_cells: 7 }],
      spill: { ...x.spill, outside_share: 0, rings: [{ r_m: 0, mean: -0.62, se: 0.09, n: 7 }] },
      uncertainty: { estimation_95: [-1.58, -0.48], specification: null, attribution: null, causal_band: [-0.55, -0.27], envelope: [-1.58, -0.27], envelope_excludes_zero: true, sources: ["estimation"] },
    });
    fetchMock = mockFetch(routes(whole));
    const { container } = render(<ResultView rid={RID} resId="res_w" pid={PID} unit="degF" nFolds={3} nCells={7} onShowLayer={() => {}} />);
    await waitFor(() => container.querySelector(".answer-card"), 5000, "card");
    for (const id of SECTIONS) openSection(container, id);
    await waitFor(() => container.querySelectorAll(".rv-section-body").length === SECTIONS.length, 5000, "every section");
    // Where it cools: one plain line instead of a one-bar chart; the map layers stay
    const where = container.querySelector('details[data-section="where"]')!;
    expect(where.querySelector('[data-reach="whole-city"]')!.textContent).toBe("Your changes cover the whole city, so there is nowhere outside them for the cooling to reach.");
    expect(where.querySelector(".rv-section-body")!.textContent).not.toContain("How far the cooling reaches");
    expect(where.querySelector(".rv-section-body svg.chart")).toBeNull();
    expect(where.querySelector('[aria-label="Show result layers on the map"]')).not.toBeNull();
    // How sure are we: one row per source, so no range hides inside a wider one
    const sure = container.querySelector('details[data-section="sure"]')!;
    expect([...sure.querySelectorAll('svg text.t-label[aria-hidden="true"]')].map((t) => t.textContent)).toEqual(["Model fit", "Independent check", "All sources together"]);
    expect(sure.textContent).toContain("tick = estimate");
    expect(sure.textContent).not.toContain("black tick");
    // Technical details: a teaser without jargon; no citywide SE or edited-cells row repeating the card
    const tech = container.querySelector('details[data-section="tech"]')!;
    expect(tech.querySelector("summary")!.textContent).toContain("For analysts: the statistics behind these numbers");
    expect(tech.querySelector("summary")!.textContent).not.toMatch(/standard error|folds/i);
    expect(tech.textContent).toContain("Standard error");
    expect(tech.textContent).not.toContain("Citywide standard error");
    expect(tech.textContent).not.toContain("Every area (cells)");
  });

  it("describes the result's map layers in plain words", () => {
    const descs = resultLayerMetas("r", "degF", ["Pct_Canopy"]).map((m) => m.desc);
    expect(descs.slice(0, 3)).toEqual([
      "The full model's estimate of the temperature change in each place.",
      "How much the model's repeat runs disagree in each place; more spread means less sure.",
      "Places where the change goes beyond anything measured in this city (hatched).",
    ]);
    expect(descs.join(" ")).not.toMatch(/fold|ΔT|standard deviation|score/i);
  });

  it("works out heat risk & people once, for the card and both sections, without a button", async () => {
    fetchMock = mockFetch(routes(result("res_i"), { "POST /api/results/res_i/impacts": { body: { ...IMPACTS, heat: null } } }));
    const { container } = render(<ResultView rid={RID} resId="res_i" pid={PID} unit="degF" nFolds={3} />);
    await waitFor(() => container.querySelector(".answer-card"), 5000, "card");
    const heat = openSection(container, "heat");
    await waitFor(() => heat.querySelector("table"), 5000, "exposure table");
    expect([...heat.querySelectorAll("button")].some((b) => /work out/i.test(b.textContent ?? ""))).toBe(false);
    expect(fetchMock.calls.filter((c) => c.method === "POST" && c.url === "/api/results/res_i/impacts")).toHaveLength(1);
    // today in Heat risk & people, the future in Against future warming
    expect(heat.textContent).toContain("Residents at or above each temperature today");
    expect(heat.textContent).not.toContain("2041-2060");
    const future = openSection(container, "future");
    await waitFor(() => future.querySelector(".rv-section-body"), 5000, "future body");
    // the futures in plain words (gauge title and table rows); the pathway code only in a tooltip
    expect(count(future.textContent ?? "", "Mid-century, moderate emissions (2041–2060)")).toBeGreaterThanOrEqual(2);
    expect(future.textContent).not.toMatch(/ssp|SSP|median warming/);
    expect(future.querySelector('td [title="ssp245 2041-2060"]')).not.toBeNull();
    // the card's "What it buys" picks the impacts up: without a heat-risk fact, the residents line
    // stays, named by its own measure (air temperature, not the heat index)
    expect(container.querySelector(".answer-notes")!.textContent).toContain("100 fewer residents feel an air temperature of 90 °F or more today");
    // still one request after both sections opened
    expect(fetchMock.calls.filter((c) => c.method === "POST" && c.url === "/api/results/res_i/impacts")).toHaveLength(1);
  });

  it("offers one retry after an error, and nothing to retry on a run without resident data", async () => {
    let calls = 0;
    fetchMock = mockFetch(
      routes(result("res_e"), {
        "POST /api/results/res_e/impacts": () => (++calls === 1 ? { status: 500, body: { error: { code: "internal", message: "worker died" } } } : { body: IMPACTS }),
      }),
    );
    const { container } = render(<ResultView rid={RID} resId="res_e" pid={PID} unit="degF" nFolds={3} />);
    const fact = await waitFor(() => container.querySelector('.answer-fact[data-fact="heat"] button'), 5000, "retry");
    expect(fact.textContent).toBe("Try again");
    expect(container.querySelector('.answer-fact[data-fact="heat"]')!.textContent).toContain("worker died");
    const heat = openSection(container, "heat");
    await waitFor(() => heat.querySelector('[data-impacts="error"]'), 5000, "section error");
    expect(calls).toBe(1); // no retry on its own
    click(byText(heat, "button", "Try again"));
    await waitFor(() => container.querySelector('.answer-fact[data-fact="heat"]')?.textContent?.includes("120"), 5000, "residents after retry");
    expect(calls).toBe(2);

    clearResources();
    fetchMock.restore();
    fetchMock = mockFetch(routes(result("res_n")));
    const n = render(<ResultView rid={RID} resId="res_n" pid={PID} unit="degF" nFolds={3} />);
    await waitFor(() => fetchMock!.calls.some((c) => c.method === "POST" && c.url === "/api/results/res_n/impacts"), 5000, "impacts request");
    const nHeat = openSection(n.container, "heat");
    await waitFor(() => nHeat.querySelector('[data-impacts="no-data"]'), 5000, "no resident data");
    expect(nHeat.textContent).toContain("This run has no resident data");
    expect(n.container.querySelector('.answer-fact[data-fact="heat"]')).toBeNull();
    expect([...n.container.querySelectorAll("button")].some((b) => b.textContent === "Try again")).toBe(false);
  });

  it("words the reach chart for a planner and leaves out an 'outside your changes' share for a whole-city change", async () => {
    fetchMock = mockFetch(routes(result("res_w")));
    const { container } = render(<ResultView rid={RID} resId="res_w" pid={PID} unit="degF" nFolds={3} />);
    await waitFor(() => container.querySelector(".answer-card"), 5000, "card");
    const where = openSection(container, "where");
    await waitFor(() => where.querySelector(".rv-section-body"), 5000, "where body");
    const text = where.textContent ?? "";
    expect(text).toContain("Distance from your changes");
    expect(text).toContain("dark bar = places you changed · lighter bars = rings around them · lines = likely range");
    expect(text).not.toMatch(/edited|whiskers|1\.96/);
    expect(text).toContain("outside your changes");
    // the same result, its changes covering the whole run (3 cells): no share, nothing to split
    const all = render(<ResultView rid={RID} resId="res_w" pid={PID} unit="degF" nFolds={3} nCells={3} />);
    await waitFor(() => all.container.querySelector(".answer-card"), 5000, "card");
    const allWhere = openSection(all.container, "where");
    await waitFor(() => allWhere.querySelector(".rv-section-body"), 5000, "where body");
    expect(allWhere.querySelector("[data-outside-share]")).toBeNull();
    expect(allWhere.textContent).not.toContain("outside your changes");
  });
});
