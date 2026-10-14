// The Design page's planner chrome (phase 3 review): the map shows the planner's layer groups, plain
// tool labels, a ramp-only legend and no compare / opacity / lock-scale controls unless expert mode is
// on; the Brush and Pick settings show only while drawing; the empty compare tray takes no room; the
// examples say what they need only when the run lacks it; the engine line never sends a run without a
// model or a quick estimate to a dead end; the quick estimate never repeats its big number citywide.
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { clearResources } from "../../../api/resource";
import type { ScenarioTemplate } from "../../../api/lab";
import { makeGrid } from "../../../map/grid";
import { byText, click, mockFetch, render, waitFor } from "../../../test/render";
import { CompareTray } from "../components/CompareTray";
import { DesignMap, PICK_WORDS, PREVIEW_KEY, plannerGroups, type DesignMapProps } from "../components/DesignMap";
import { PreviewPane } from "../components/flow/PreviewPane";
import { REQ_TEXT, TemplateGallery, knownMissing } from "../components/TemplateGallery";
import { defaultLayerKey } from "../Design";
import { routes as labRoutes } from "../routes";
import { forgetDrafts, getDraft } from "../model/draft";
import { useTray } from "../model/tray";
import type { WorkbenchModel } from "../model/useWorkbench";
import { ENGINE_NO_MODEL_NO_ESTIMATE, engineLine } from "../model/words";
import { GROUPS, LEVERS, PID, RID, gridMeta, layer, runRoutes } from "../__fixtures__/lab";

let fetchMock: ReturnType<typeof mockFetch> | null = null;

beforeEach(() => {
  clearResources();
  forgetDrafts();
  useTray.setState({ byRun: {} });
});

afterEach(() => {
  fetchMock?.restore();
  fetchMock = null;
});

const grid = () => makeGrid(gridMeta(), Int32Array.from([0, 1, 0, 2, 0, 1, 2]), Int32Array.from([2, 2, 1, 1, 0, 0, 0]));
const CV = { id: "cv", label: "CV design", layers: [layer({ key: "fold", group: "cv", label: "CV fold" })] };

function mapProps(over: Partial<DesignMapProps> = {}): DesignMapProps {
  return {
    rid: RID,
    grid: grid(),
    groups: [...GROUPS, CV],
    levers: LEVERS,
    store: getDraft(RID, null, 7),
    brushTick: 0,
    brushState: null,
    preview: null,
    previewBusy: false,
    previewError: null,
    resultId: null,
    realizedVars: [],
    tint: null,
    layerKey: "obs",
    onLayerChange: () => {},
    onBrushChange: () => {},
    onSelection: () => {},
    showPreviewCard: false,
    drawing: false,
    ...over,
  };
}

const themes = (root: ParentNode) => [...root.querySelectorAll('[aria-label="Map theme"] button')].map((b) => b.textContent);

describe("Design map, planner chrome", () => {
  it("shows the planner's layer groups, plain tool labels and a ramp-only legend, without expert controls", async () => {
    fetchMock = mockFetch(runRoutes());
    const { container } = render(<DesignMap {...mapProps()} />);
    await waitFor(() => container.querySelector(".map-side .card h3")?.textContent === "Observed temperature", 5000, "legend");
    expect(themes(container)).toEqual(["Your changes", "Temperature", "Planner & people"]); // not "Design": the page's sub-tab
    const tools = [...container.querySelectorAll('[aria-label="Map tool"] button')];
    expect(tools.map((b) => b.textContent)).toEqual(["Move", "Paint", "Box", "Circle", "Shape", "Pick area"]);
    expect(tools.every((b) => b.querySelector("svg") && b.getAttribute("aria-label")!.startsWith(b.textContent!))).toBe(true);
    expect(container.querySelector('[aria-label="Layer opacity"]')).toBeNull();
    expect(byText(container, "button", "Lock scale")).toBeNull();
    expect(byText(container, "button", "Side by side")).toBeNull();
    // the legend: the colour ramp only
    const legend = container.querySelector(".map-side .card[data-compact-legend]")!;
    expect(legend).not.toBeNull();
    expect(legend.querySelector('[aria-label="Layer summary"]')).toBeNull();
    expect(legend.querySelector('input[aria-label="Range minimum"]')).toBeNull();
    expect(legend.querySelector(".legend-ramp")).not.toBeNull();
    // not drawing: no brush or pick settings
    expect(container.querySelector('[aria-label="Brush"]')).toBeNull();
    expect(container.querySelector('[aria-label="Pick"]')).toBeNull();
  });

  it("shows the Brush and Pick settings in plain words while drawing, or while the Paint tool is in use", async () => {
    fetchMock = mockFetch(runRoutes());
    const r = render(<DesignMap {...mapProps({ drawing: true })} />);
    const brush = await waitFor(() => r.container.querySelector('[aria-label="Brush"]'), 5000, "brush card");
    expect([...brush.querySelectorAll("select option")].map((o) => o.textContent)).toEqual(["Trees & canopy", "Cool roofs & surfaces"]);
    expect(brush.querySelector(".unit")!.textContent).toBe("pts");
    const pick = r.container.querySelector('[aria-label="Pick"]')!;
    expect([...pick.querySelectorAll("button")].map((b) => b.textContent)).toEqual([PICK_WORDS.zone, PICK_WORDS.hex250, PICK_WORDS.hex500]);
    expect(pick.textContent).not.toMatch(/Hex|Zone/);
    r.unmount();

    const p = render(<DesignMap {...mapProps({ drawing: false })} />);
    expect(p.container.querySelector('[aria-label="Brush"]')).toBeNull();
    // Paint is enabled once the lever's values have loaded
    const paint = await waitFor(() => {
      const b = byText(p.container, '[aria-label="Map tool"] button', "Paint") as HTMLButtonElement | null;
      return b && !b.disabled ? b : null;
    }, 5000, "Paint enabled");
    click(paint);
    await waitFor(() => p.container.querySelector('[aria-label="Brush"]'), 5000, "brush card while painting");
    expect(p.container.querySelector('[aria-label="Pick"]')).toBeNull();
  });

  it("shows the whole catalogue and every control in expert mode", async () => {
    fetchMock = mockFetch(runRoutes());
    const store = getDraft(RID, null, 7);
    store.getState().change((d) => ({ ...d, options: { ...(d.options ?? {}), expert: true } }));
    const { container } = render(<DesignMap {...mapProps({ store })} />);
    await waitFor(() => container.querySelector('[aria-label="Layer summary"]'), 5000, "full legend");
    expect(themes(container)).toContain("CV design");
    expect(themes(container)).toContain("Inputs");
    expect(byText(container, "button", "Lock scale")).not.toBeNull();
    expect(container.querySelector('[aria-label="Layer opacity"]')).not.toBeNull();
    expect(container.querySelector('[aria-label="Brush"]')).not.toBeNull();
  });

  it("keeps the group of a layer named by a link, and opens on the observed temperature until the first change", () => {
    expect(plannerGroups([...GROUPS, CV], null).map((g) => g.id)).toEqual(["temperature", "planner"]);
    expect(plannerGroups([...GROUPS, CV], "fold").map((g) => g.id)).toEqual(["temperature", "planner", "cv"]);
    expect(defaultLayerKey(GROUPS, null)).toBe("obs");
    expect(defaultLayerKey(GROUPS, 0)).toBe("obs");
    expect(defaultLayerKey(GROUPS, 3)).toBe(PREVIEW_KEY);
    expect(defaultLayerKey([CV], null)).toBe(PREVIEW_KEY); // no observed temperature: the quick estimate layer
    // an up-to-date full-model result: the map shows its temperature change, beside the answer card
    expect(defaultLayerKey(GROUPS, 3, "res_1")).toBe("res:res_1:delta");
    expect(defaultLayerKey(GROUPS, 3, null)).toBe(PREVIEW_KEY);
  });

  it("keeps the layer select and its label together when the toolbar wraps", async () => {
    const { container } = render(<DesignMap {...mapProps()} />);
    const pick = await waitFor(() => container.querySelector(".map-layer-pick"), 5000, "layer pick");
    expect([...pick.children].map((c) => c.tagName)).toEqual(["LABEL", "SELECT"]);
    expect(pick.querySelector("label")!.getAttribute("for")).toBe(pick.querySelector("select")!.id);
  });
});

describe("Compare tray", () => {
  it("takes no room until something is pinned", () => {
    const empty = render(<CompareTray rid={RID} />);
    expect(empty.container.innerHTML).toBe("");
    useTray.getState().pin(RID, { kind: "result", id: "res_1" }, "Corridor");
    const one = render(<CompareTray rid={RID} />);
    const tray = one.container.querySelector('[aria-label="Compare tray"]')!;
    expect(tray.textContent).toContain("Corridor");
    expect(tray.textContent).toContain("Pin one more");
  });
});

describe("Examples (templates)", () => {
  const T = (id: string, requires: ScenarioTemplate["requires"]): ScenarioTemplate => ({ id, label: id, desc: "Server words (S4 footprint effect).", requires, params_schema: { properties: {}, required: [] } });

  it("says what an example needs only when the run lacks it, in plain words", async () => {
    expect(knownMissing(T("a", ["canopy_role"]), LEVERS)).toEqual([]);
    expect(knownMissing(T("b", ["impervious_role", "crs"]), LEVERS, { hasCrs: true })).toEqual(["impervious_role"]);
    expect(knownMissing(T("c", ["layers", "crs"]), LEVERS)).toEqual([]); // unknown: say nothing
    expect(knownMissing(T("c", ["layers"]), LEVERS, { hasLayers: false })).toEqual(["layers"]);
    expect(Object.values(REQ_TEXT).join(" ")).not.toMatch(/role|CRS|layers/);
    fetchMock = mockFetch({ "GET /api/scenario-templates": { body: [T("footprint_priority", ["canopy_role"]), T("depave", ["impervious_role"])] } });
    const { container } = render(<TemplateGallery pid={PID} rid={RID} levers={LEVERS} context={{ hasCrs: true, hasLayers: true }} onCreated={() => {}} />);
    const cards = await waitFor(() => (container.querySelectorAll(".template").length === 2 ? [...container.querySelectorAll(".template")] : null), 5000, "cards");
    expect(cards[0].querySelector("[data-unavailable]")).toBeNull();
    expect(cards[0].textContent).toContain("where each new tree cools the surrounding neighbourhood the most");
    expect(cards[0].textContent).not.toContain("S4");
    expect(cards[1].querySelector("[data-unavailable]")!.textContent).toBe("Not available for this run: it needs paving data.");
    expect(container.textContent).not.toMatch(/planning action/i);
  });

  it("shows three compact examples, the usable ones first, and the rest behind More examples", async () => {
    const all = [T("depave", ["impervious_role"]), T("shade_hottest", ["canopy_role"]), T("cool_roofs", ["albedo_role"]), T("fill_plantable", ["canopy_role"]), T("around_sites", ["crs"])];
    fetchMock = mockFetch({ "GET /api/scenario-templates": { body: all } });
    const { container } = render(<TemplateGallery pid={PID} rid={RID} levers={LEVERS} context={{ hasCrs: true, hasLayers: true }} onCreated={() => {}} />);
    const ids = () => [...container.querySelectorAll(".template")].map((c) => c.getAttribute("data-template"));
    await waitFor(() => (ids().length ? true : null), 5000, "cards");
    // depave needs paving data this run lacks: it goes after the usable ones
    expect(ids()).toHaveLength(3);
    expect(ids()).not.toContain("depave");
    // compact: a title and one line (the whole sentence in the tooltip)
    const first = container.querySelector(".template")!;
    expect(first.querySelector(".template-desc")!.getAttribute("title")).toBe(first.querySelector(".template-desc")!.textContent);
    const more = byText(container, "button", "More examples (2)")!;
    expect(more.getAttribute("aria-expanded")).toBe("false");
    click(more);
    expect(ids()).toHaveLength(5);
    expect(ids()[4]).toBe("depave");
    const fewer = byText(container, "button", "Fewer examples")!;
    expect(fewer.getAttribute("aria-expanded")).toBe("true");
    click(fewer);
    expect(ids()).toHaveLength(3);
  });
});

describe("Engine line and the quick estimate's citywide row", () => {
  it("hides the top bar's engine dot on the Design page, which says the engine state under Run full model", () => {
    expect(labRoutes.filter((r) => r.hideEngine).map((r) => r.path)).toEqual(["/r/:rid/lab", "/r/:rid/lab/s/:sid"]);
  });

  it("points a run with neither a model nor a quick estimate to building the estimate", () => {
    expect(engineLine("no_checkpoint", null, { noEmulator: true })).toBe(ENGINE_NO_MODEL_NO_ESTIMATE);
    expect(engineLine("incompatible", null, { noEmulator: true })).toBe(ENGINE_NO_MODEL_NO_ESTIMATE);
    expect(ENGINE_NO_MODEL_NO_ESTIMATE).not.toContain("quick estimate only");
    expect(engineLine("no_checkpoint", null)).toBe("Model not available for this run — quick estimate only");
    expect(engineLine("ready", null, { noEmulator: true })).toBe("Model ready");
  });

  it("leaves the citywide row out when the changes cover the whole city or it rounds to the big number", () => {
    const model = (n_edited: number, mean: number, edited_mean: number) =>
      ({
        rid: RID,
        grid: { n: 7 },
        unit: "degF",
        levers: LEVERS,
        doc: { costs: {} },
        preview: { summary: { mean, edited_mean, n_edited, outside_share: 0, trust: "good", hatched: false, reasons: [], request_seq: 1 } },
        previewBusy: false,
        previewError: null,
        noEmulator: false,
        emulator: { present: true },
        compile: null,
        cellAreaKm2: 0.0009,
        heatPreview: null,
        advice: { recommend: false, reasons: [] },
      }) as unknown as WorkbenchModel;
    const ddText = (root: ParentNode, term: string) => [...root.querySelectorAll("dt")].find((d) => d.textContent === term)?.nextElementSibling?.textContent ?? null;
    const whole = render(<PreviewPane model={model(7, -1.03, -1.03)} />);
    expect(ddText(whole.container, "citywide")).toBeNull();
    expect(whole.container.querySelector("[data-city-mean]")!.getAttribute("data-city-mean")).toBe("-1.03");
    expect(ddText(whole.container, "Area changed")).toContain("the whole city");
    const same = render(<PreviewPane model={model(6, -1.031, -1.034)} />);
    expect(ddText(same.container, "citywide")).toBeNull();
    const part = render(<PreviewPane model={model(3, -0.05, -0.42)} />);
    expect(ddText(part.container, "citywide")).toBe("−0.05 °F");
  });
});
