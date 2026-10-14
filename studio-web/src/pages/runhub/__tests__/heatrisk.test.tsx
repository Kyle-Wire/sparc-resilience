// Heat risk (`/r/:rid/heat/:section?`, phase 2): Future climate stacks the Climate view, heat
// risk in each future and "Can adaptation keep up?" (the Lab's Climate × adaptation, embedded)
// with each climate number shown once; the what-if dewpoint is page state shared with Today; Who
// is exposed is the planner pack's exposure part, and its "where to act" part is a named export
// What cools renders.
import { afterEach, describe, expect, it } from "vitest";
import { clearResources } from "../../../api/resource";
import { navigate } from "../../../router";
import { byText, click, flush, mockFetch, render, waitFor, type MockHandler } from "../../../test/render";
import { PID, bin, result, runRoutes } from "../../lab/__fixtures__/lab";
import { viewFixture } from "../__fixtures__/views";
import HeatRisk from "../HeatRisk";
import { PlannerWhereToAct } from "../Planner";
import { unframedCharts } from "./helpers";

let seq = 0;

const charts = (root: ParentNode) => [...root.querySelectorAll("figure.chart-frame")].map((f) => f.getAttribute("data-chart") ?? "");

function climateRoutes(rid: string, explored: unknown[]): Record<string, MockHandler> {
  return {
    [`GET /api/runs/${rid}/scenarios`]: { body: { configured: [], results: [result("res_1").summary] } },
    [`GET /api/projects/${PID}/scenarios`]: {
      body: [{ id: "sc_a", revision: 1, parent_id: null, status: "exact", created_utc: "", updated_utc: "", name: "Downtown cool corridor", tags: [], latest: result("res_1").summary }],
    },
    [`GET /api/runs/${rid}/climate/factors`]: {
      body: {
        present: true,
        path: "cmip6.csv",
        experiments: ["ssp245"],
        periods: ["2041-2060"],
        models: ["A", "B"],
        warming: [{ experiment: "ssp245", label: "SSP2-4.5", period: "2041-2060", n_models: 2, median: 3.6, p10: 2.7, p90: 4.5, min: 2.5, max: 4.9, by_model: { A: 3.2, B: 4 } }],
        action: null,
      },
    },
    [`POST /api/runs/${rid}/climate/explore`]: (_u, init) => {
      explored.push(JSON.parse(String(init.body)));
      return {
        body: {
          present: { mean: 88.1, share_at_or_above: { "90": 0.29 } },
          thresholds: [90],
          projections: [
            {
              experiment: "ssp245",
              label: "SSP2-4.5",
              period: "2041-2060",
              n_models: 2,
              warming: { median: 3.6, p10: 2.7, p90: 4.5, min: 2.5, max: 4.9, by_model: {} },
              variants: [
                { name: "no adaptation", mean: 91.7, adaptation_mean_delta: 0, offset_share_of_median_warming: null, share_at_or_above: { "90": { median: 0.86, p10: 0.7, p90: 0.95 } } },
                { name: "Downtown cool corridor", mean: 91.6, adaptation_mean_delta: -0.021, offset_share_of_median_warming: 0.41, share_at_or_above: { "90": { median: 0.84, p10: 0.68, p90: 0.94 } } },
              ],
            },
          ],
          adaptation: ["Downtown cool corridor"],
          people_exposure: null,
          units: "degF",
        },
      };
    },
    "GET /api/results/res_1/layers/delta.bin": () => bin(Float32Array.from([-0.5, 0, -1.2, -0.3, 0, -0.4, 0])),
  };
}

describe("Heat risk", () => {
  afterEach(() => clearResources());

  it("Future climate: the warming once, heat risk in each future, and Can adaptation keep up? once a scenario is picked", async () => {
    const rid = `r_heat${++seq}`;
    const explored: unknown[] = [];
    const humid = viewFixture("heat");
    Object.assign(humid.sections.humidity!, { dewpoint_C: 21, user_set: true, source: "your setting" });
    // even with adaptation variants, the offset's one home here is "Can adaptation keep up?"
    const climate = viewFixture("climate");
    climate.sections.offset = [{ id: "ssp245_2041", label: "SSP2-4.5 2041–2060", variant: "package", share: 0.4 }];
    const m = mockFetch({
      ...runRoutes(rid),
      ...climateRoutes(rid, explored),
      [`GET /api/runs/${rid}/views/climate`]: { body: climate },
      [`GET /api/runs/${rid}/views/heat?dewpoint_C=21`]: { body: humid },
    });
    try {
      navigate(`/r/${rid}/heat/future?dewpoint=21`, { replace: true });
      const { container } = render(<HeatRisk />);
      // the what-if dewpoint set under Today travels with the section links (not to Who is exposed)
      const nav = container.querySelector('nav.section-nav[aria-label="Heat risk"]')!;
      expect([...nav.querySelectorAll("a")].map((a) => a.getAttribute("href"))).toEqual([`/r/${rid}/heat/today?dewpoint=21`, `/r/${rid}/heat/people`, `/r/${rid}/heat/future?dewpoint=21`]);
      await waitFor(() => container.querySelector('[aria-label="Exposure in each climate future"]'), 8000, "heat risk in each future");
      await waitFor(() => container.querySelector('svg.chart[aria-label="Warming by scenario and period"]'), 8000, "warming chart");
      const adapt = await waitFor(() => container.querySelector<HTMLElement>("#adapt [data-climate-adaptation]"), 8000, "Can adaptation keep up?");
      await flush(4);

      // the futures table: no Today row (Today has it) and no warming column (the Warming chart has it)
      const table = container.querySelector('[aria-label="Exposure in each climate future"]')!;
      expect([...table.querySelectorAll("thead th")].map((th) => th.textContent)).not.toContain("Warming (°F)");
      expect(table.textContent).not.toContain("Today");
      expect(container.querySelector('[data-view="heat"]')!.textContent).toContain("dewpoint of 21.0 °C (your setting)");
      expect(charts(container)).toContain("Residents by heat-risk category in each future");
      expect(charts(container)).not.toContain("Residents by heat-risk category");
      // the embedded Climate view: no offset gauges and no button down to the section below; the exposure spans the width
      const view = container.querySelector<HTMLElement>('[data-view="climate"]')!;
      expect(charts(view)).not.toContain("Share of median warming offset");
      expect(view.textContent).not.toContain("Warming offset");
      expect(view.querySelector("header a")).toBeNull();
      const exposure = view.querySelector('figure.chart-frame[data-chart^="Share of cells at or above"]')!;
      expect(exposure.parentElement!.classList.contains("grid2")).toBe(false);
      // the Lab tool adds no second warming chart and asks the server nothing until a scenario is picked
      expect(container.querySelectorAll('svg.chart[aria-label="Warming by scenario and period"]').length).toBe(1);
      expect(adapt.textContent).toContain("Pick one or more adaptations");
      expect(explored).toEqual([]);
      expect(adapt.querySelector('select[aria-label="Adaptation for the map"]')).toBeNull();

      click(byText(adapt, "button.chip", "Downtown cool corridor")!);
      await waitFor(() => explored.length > 0, 5000, "explore with the adaptation");
      expect(explored.at(-1)).toMatchObject({ adaptations: [{ kind: "result", id: "res_1" }] });
      await waitFor(() => adapt.querySelector('svg.chart[aria-label="Share of cells at or above 90.0 °F"]'), 5000, "the scenario × projection matrix");
      expect(byText(adapt, "figure", "Cancels 41% of SSP2-4.5 2041–2060 median warming")).not.toBeNull();
      // the adapted-future map offers the picked adaptations only (the unadapted map is above)
      const pick = await waitFor(() => adapt.querySelector<HTMLSelectElement>('select[aria-label="Adaptation for the map"]'), 5000, "adapted-future map");
      expect([...pick.options].map((o) => o.textContent)).toEqual(["Downtown cool corridor"]);
      expect(container.querySelectorAll('svg.chart[aria-label="Warming by scenario and period"]').length).toBe(1);
      expect(unframedCharts(container)).toEqual([]);
      expect(container.textContent).not.toContain("could not be displayed");
    } finally {
      m.restore();
    }
  });

  it("Who is exposed: exposure, person-mean temperature, hot days and equity; where to act and the GIS files are elsewhere", async () => {
    const rid = `r_heat${++seq}`;
    const m = mockFetch({ ...runRoutes(rid), [`GET /api/runs/${rid}/views/planner`]: { body: viewFixture("planner") } });
    try {
      navigate(`/r/${rid}/heat/people`, { replace: true });
      const { container } = render(<HeatRisk />);
      await waitFor(() => container.querySelector('[data-view="planner"] figure.chart-frame'), 8000, "Who is exposed");
      await flush(4);
      const page = container.querySelector('[data-view="planner"]')!;
      expect(page.querySelector("h2")!.textContent).toBe("Who is exposed");
      expect(charts(page)).toEqual(expect.arrayContaining(["Residents exposed to heat", "Hot days per year"]));
      expect(page.textContent).toContain("Person-mean temperature");
      const titles = [...page.querySelectorAll("h3")].map((h) => h.textContent);
      for (const gone of ["Plantable space", "Zones", "Logger sites and before/after pairs", "GIS gallery"]) expect(titles).not.toContain(gone);
      expect(page.querySelector('[aria-label="GIS files"]')).toBeNull();
      expect(byText(page, "a", "What cools › Where to act")!.getAttribute("href")).toBe(`/r/${rid}/effects#where`);
      expect(unframedCharts(container)).toEqual([]);
    } finally {
      m.restore();
    }
  });

  it("PlannerWhereToAct (What cools): plantable space, zones and logger sites, for the run it is given", async () => {
    const rid = `r_heat${++seq}`;
    const m = mockFetch({ ...runRoutes(rid), [`GET /api/runs/${rid}/views/planner`]: { body: viewFixture("planner") } });
    try {
      navigate("/", { replace: true });
      const { container } = render(<PlannerWhereToAct rid={rid} />);
      await waitFor(() => container.querySelector('[aria-label="Plantable space"]'), 8000, "where to act");
      await flush(4);
      const titles = [...container.querySelectorAll("h3")].map((h) => h.textContent);
      expect(titles).toEqual(expect.arrayContaining(["Plantable space", "Zones", "Logger sites"]));
      expect(titles).not.toContain("GIS gallery");
      expect(container.querySelector("h2")).toBeNull();
      expect(byText(container, "a", "Hex map 250 m")!.getAttribute("href")).toBe(`/r/${rid}/map?hex=250`);
      expect(container.textContent).toMatch(/\d+ logger sites/);
      expect(container.textContent).not.toContain("before/after pairs");
    } finally {
      m.restore();
    }
  });
});
