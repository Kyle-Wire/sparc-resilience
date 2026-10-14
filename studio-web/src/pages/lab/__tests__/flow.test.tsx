// The Design step flow (phase 3): pure helpers of model/flow.ts (planner modes and amount scales,
// where-presets → selection trees the server accepts, the change sentence and the "how much fits"
// sentence) and the three steps + the collapsed sentence against a stub workbench model.
import { describe, expect, it, vi } from "vitest";
import type { CompileLever, CompileResult, Edit, Lever } from "../../../api/lab";
import type { SelectionSpec } from "../../../api/types";
import type { GridData } from "../../../map/grid";
import { act } from "react";
import { byText, click, key, render, typeInto, waitFor } from "../../../test/render";
import { RID, doc, gridMeta, lever } from "../__fixtures__/lab";
import { EditSentence, editSentence as editSentenceOf } from "../components/flow/EditSentence";
import { StepHowMuch } from "../components/flow/StepHowMuch";
import { StepWhat } from "../components/flow/StepWhat";
import { StepWhere } from "../components/flow/StepWhere";
import { validateSelection } from "../model/doc";
import {
  amountScale,
  compileSentence,
  defaultAmount,
  editSentence,
  fromDisplay,
  modeLabel,
  neighbourhoodOf,
  newChangeInit,
  niceStep,
  plannerModes,
  presetSelection,
  roundToStep,
  switchLever,
  switchMode,
  toDisplay,
  toggleNeighbourhood,
  wherePresetOf,
  wherePresets,
  withTopShare,
  type SentenceContext,
  type WhereContext,
} from "../model/flow";
import type { WorkbenchModel } from "../model/useWorkbench";
import { TRUST_TIP, WHERE_PRESET_ORDER } from "../model/words";

const CANOPY = lever("Pct_Canopy");
const ALBEDO = lever("Albedo");
const PAVING = lever("Pct_Impervious", { label: "Impervious", unit: "pp", min: 0, max: 100, direction: "decrease", doses: [10, 20], design_dose: 10, sd: 15, role: "impervious", headroom_available: false, cost_per_unit: 2 });
const LEVERS3: Lever[] = [CANOPY, ALBEDO, PAVING];

const CTX: WhereContext = {
  zones: [1, 2],
  regions: [{ id: "rg_1", name: "Downtown" }],
  temperatureColumn: "pred:target",
  populationColumn: "layer:people",
  mapPick: null,
};
const SCTX: SentenceContext = {
  ...CTX,
  levers: LEVERS3,
  planRefs: [{ ref: "plan:pl_1", label: "Budget plan" }],
  columns: [{ value: "predictor:Pct_Canopy", label: "Canopy cover (pp)", group: "Inputs" }],
};

const e = (x: Partial<Edit> = {}): Edit => ({ lever: "Pct_Canopy", mode: "add", amount: 10, where: { kind: "all" }, ...x });

function compile(levers: Record<string, Partial<CompileLever>>, extra: Partial<CompileResult> = {}): CompileResult {
  return {
    content_hash: "h",
    portable: true,
    levers: Object.fromEntries(Object.entries(levers).map(([k, v]) => [k, { n_cells: 3, mean_requested: 10, total_requested: 30, predicted_mean_realised: 10, clipped_share: 0, est_cost: 30, ...v }])),
    union_cells: 3,
    people: 35,
    warnings: [],
    est_exact_s: 120,
    emulator: { usable: true, hatched: false, reasons: [] },
    ...extra,
  };
}

// ---------------------------------------------------------------- step 03: modes and amounts

describe("planner modes and amounts", () => {
  it("offers the planner modes in the lever's cooling direction", () => {
    expect(plannerModes(CANOPY)).toEqual(["add", "floor", "fill_headroom"]);
    expect(plannerModes(ALBEDO)).toEqual(["add", "floor"]); // planting space is canopy-only
    expect(plannerModes(PAVING)).toEqual(["add", "ceiling"]);
    expect(modeLabel("add", PAVING)).toBe("Take away");
    expect(modeLabel("ceiling", PAVING)).toBe("Bring down to at most");
    expect(modeLabel("floor", CANOPY)).toBe("Bring up to at least");
    expect(plannerModes(undefined)).toEqual(["add"]); // an expert-mode predictor
  });

  it("scales amounts in natural units: sizes of the change, levels, or a share of the planting space", () => {
    expect(amountScale("add", CANOPY)).toEqual({ factor: 1, min: 0, max: 100, step: 1, unit: "pts" });
    expect(amountScale("add", ALBEDO)).toMatchObject({ factor: 1, min: 0, max: 1, step: 0.01, unit: "" });
    expect(amountScale("floor", CANOPY)).toMatchObject({ min: 0, max: 100, unit: "pts" });
    expect(amountScale("fill_headroom", CANOPY)).toEqual({ factor: 100, min: 0, max: 100, step: 5, unit: "%" });
    // paving cools by going down: the control shows how much is taken away, the edit stores it negative
    expect(amountScale("add", PAVING).factor).toBe(-1);
    expect(toDisplay({ mode: "add", amount: -10 }, PAVING)).toBe(10);
    expect(fromDisplay(15, "add", PAVING)).toBe(-15);
    expect(toDisplay({ mode: "fill_headroom", amount: 0.55 }, CANOPY)).toBe(55);
    expect(fromDisplay(55, "fill_headroom", CANOPY)).toBe(0.55);
    expect(toDisplay({ mode: "add", amount: undefined }, CANOPY)).toBeNull();
  });

  it("picks a slider step every dose sits on, without float noise", () => {
    expect(niceStep(100)).toBe(1);
    expect(niceStep(100, [2.5, 10])).toBe(0.1);
    expect(niceStep(1, [0.05, 0.1])).toBe(0.01);
    expect(roundToStep(0.1 + 0.2, 0.01)).toBe(0.3);
    expect(roundToStep(24.6, 1)).toBe(25);
  });

  it("starts changes from the design dose in the cooling direction and sensible levels", () => {
    expect(newChangeInit(CANOPY)).toEqual({ amount: 10 });
    expect(newChangeInit(PAVING)).toEqual({ amount: -10 });
    expect(defaultAmount("floor", CANOPY)).toBe(20);
    expect(defaultAmount("ceiling", PAVING)).toBe(80);
    expect(defaultAmount("fill_headroom", CANOPY)).toBe(0.5);
    expect(switchMode(e({ where: { kind: "zones", values: [1] } }), "floor", CANOPY)).toEqual({ lever: "Pct_Canopy", mode: "floor", amount: 20, where: { kind: "zones", values: [1] } });
  });

  it("swaps a change's lever keeping where, a planner mode the new lever has, and dropping the old label", () => {
    const where: SelectionSpec = { kind: "zones", values: [1] };
    expect(switchLever(e({ mode: "floor", amount: 30, where, label: "street trees" }), ALBEDO)).toEqual({ lever: "Albedo", mode: "floor", amount: 0.2, where });
    expect(switchLever(e({ mode: "fill_headroom", amount: 0.5, where }), ALBEDO)).toEqual({ lever: "Albedo", mode: "add", amount: 0.1, where });
    expect(switchLever(e({ mode: "set", amount: 40, where }), PAVING)).toEqual({ lever: "Pct_Impervious", mode: "add", amount: -10, where });
    const same = e();
    expect(switchLever(same, CANOPY)).toBe(same);
  });
});

// ---------------------------------------------------------------- step 02: where presets

describe("where presets", () => {
  it("offers only the presets whose data the run has", () => {
    expect(wherePresets(CTX)).toEqual([...WHERE_PRESET_ORDER]);
    expect(wherePresets({ ...CTX, populationColumn: null })).not.toContain("people");
    expect(wherePresets({ ...CTX, temperatureColumn: null })).not.toContain("hottest");
    expect(wherePresets({ ...CTX, zones: [], regions: [] })).toEqual(["city", "hottest", "people", "draw"]);
  });

  it("builds selection trees the server accepts, and reads each back as its preset", () => {
    const pick = { spec: { kind: "circle", crs: "EPSG:4326", center: [-71.4, 41.8], radius_m: 300 } as SelectionSpec, label: "Circle 300 m" };
    const ctx = { ...CTX, mapPick: pick };
    expect(presetSelection("city", ctx)).toEqual({ kind: "all" });
    expect(presetSelection("neighbourhood", ctx)).toEqual({ kind: "region", id: "rg_1" });
    expect(presetSelection("neighbourhood", { ...ctx, regions: [] })).toEqual({ kind: "zones", values: [1] });
    expect(presetSelection("hottest", ctx)).toEqual({ kind: "top", column: "pred:target", frac: 0.1, direction: "highest" });
    expect(presetSelection("people", ctx, 0.25)).toEqual({ kind: "top", column: "layer:people", frac: 0.25, direction: "highest" });
    expect(presetSelection("draw", ctx)).toEqual(pick.spec);
    expect(presetSelection("draw", CTX)).toBeNull(); // nothing drawn yet
    expect(presetSelection("people", { ...CTX, populationColumn: null })).toBeNull();
    for (const p of ["city", "neighbourhood", "hottest", "people", "draw"] as const) {
      const s = presetSelection(p, ctx)!;
      expect(validateSelection(s, "where")).toEqual([]);
      expect(wherePresetOf(s, ctx)).toBe(p);
    }
    expect(wherePresetOf(undefined, CTX)).toBe("city");
  });

  it("combines zones and saved areas of a neighbourhood with 'or'", () => {
    const both = toggleNeighbourhood({ kind: "zones", values: [1] }, { region: "rg_1" })!;
    expect(both).toEqual({ op: "or", args: [{ kind: "zones", values: [1] }, { kind: "region", id: "rg_1" }] });
    expect(validateSelection(both)).toEqual([]);
    expect(wherePresetOf(both, CTX)).toBe("neighbourhood");
    expect(neighbourhoodOf(both)).toEqual({ zones: [1], regions: ["rg_1"] });
    expect(toggleNeighbourhood(both, { zone: 2 })).toEqual({ op: "or", args: [{ kind: "zones", values: [1, 2] }, { kind: "region", id: "rg_1" }] });
    expect(toggleNeighbourhood(both, { zone: 1 })).toEqual({ kind: "region", id: "rg_1" });
    expect(toggleNeighbourhood({ kind: "region", id: "rg_1" }, { region: "rg_1" })).toBeNull(); // never empty
  });

  it("reads anything the presets cannot show as a custom filter", () => {
    expect(wherePresetOf({ kind: "filter", column: "predictor:Pct_Canopy", op: ">=", value: 30 }, CTX)).toBe("custom");
    expect(wherePresetOf({ kind: "top", column: "pred:target", frac: 0.1, direction: "highest", within: { kind: "zones", values: [1] } }, CTX)).toBe("custom");
    expect(wherePresetOf({ kind: "top", column: "pred:target", frac: 0.1, direction: "lowest" }, CTX)).toBe("custom");
    expect(wherePresetOf({ kind: "top", column: "pred:target", frac: 0.1, direction: "highest", k: null, within: null }, CTX)).toBe("hottest"); // stored docs echo nulls
    expect(wherePresetOf({ op: "and", args: [{ kind: "zones", values: [1] }, { kind: "region", id: "rg_1" }] }, CTX)).toBe("custom");
    expect(wherePresetOf({ kind: "hex", size_m: 250, keys: [1, 2] }, CTX)).toBe("draw");
    expect(withTopShare({ kind: "top", column: "pred:target", frac: 0.1, direction: "highest" }, 0.2)).toMatchObject({ frac: 0.2 });
  });
});

// ---------------------------------------------------------------- sentences

describe("the change in one sentence", () => {
  it("says what, how much and where in plain words", () => {
    expect(editSentence(e({ where: { kind: "region", id: "rg_1" } }), SCTX)).toBe("Add 10 pts tree canopy in Downtown");
    expect(editSentence(e({ where: { kind: "zones", values: [1] } }), SCTX)).toBe("Add 10 pts tree canopy in zone 1");
    expect(editSentence(e({ where: { kind: "zones", values: [1, 2] } }), SCTX)).toBe("Add 10 pts tree canopy in zones 1 and 2");
    expect(editSentence(e({ where: toggleNeighbourhood({ kind: "region", id: "rg_1" }, { zone: 2 })! }), SCTX)).toBe("Add 10 pts tree canopy in Downtown and zone 2");
    expect(editSentence(e({ where: presetSelection("hottest", CTX)! }), SCTX)).toBe("Add 10 pts tree canopy in the hottest 10% of places");
    expect(editSentence(e({ where: presetSelection("people", CTX, 0.2)! }), SCTX)).toBe("Add 10 pts tree canopy in the 20% of places where the most people live");
    expect(editSentence(e({ lever: "Pct_Impervious", amount: -10 }), SCTX)).toBe("Take away 10 pts paving across the whole city");
    expect(editSentence(e({ mode: "floor", amount: 30, where: { kind: "zones", values: [1] } }), SCTX)).toBe("Bring tree canopy up to at least 30 pts in zone 1");
    expect(editSentence(e({ lever: "Pct_Impervious", mode: "ceiling", amount: 60 }), SCTX)).toBe("Bring paving down to at most 60 pts across the whole city");
    expect(editSentence(e({ mode: "fill_headroom", amount: 0.5 }), SCTX)).toBe("Plant trees on 50% of the available planting space across the whole city");
    expect(editSentence(e({ lever: "Albedo", mode: "set", amount: 0.35, where: { kind: "filter", column: "predictor:Pct_Canopy", op: ">=", value: 30 } }), SCTX)).toBe("Set surface reflectivity to 0.35 in canopy cover ≥ 30");
    expect(editSentence(e({ mode: "to_percentile", amount: undefined, percentile: 75 }), SCTX)).toBe("Bring tree canopy to the city’s 75th percentile across the whole city");
    expect(editSentence(e({ mode: "per_cell", amount: undefined, per_cell_ref: "plan:pl_1" }), SCTX)).toBe("Change tree canopy place by place, from the plan “Budget plan”");
    expect(editSentence(e({ where: { kind: "circle", crs: "EPSG:4326", center: [-71.4, 41.8], radius_m: 300 } }), SCTX)).toBe("Add 10 pts tree canopy in the area drawn on the map");
    expect(editSentence(e({ amount: undefined }), SCTX)).toBe("Add … tree canopy across the whole city");
  });

  it("keeps the shell's editSentence(model, edit) signature", () => {
    expect(editSentenceOf({ levers: LEVERS3, regions: [{ id: "rg_1", name: "Downtown", spec: { kind: "all" }, n_cells: 7 }] }, e({ where: { kind: "region", id: "rg_1" } }))).toBe("Add 10 pts tree canopy in Downtown");
  });
});

describe("the 'how much fits' sentence", () => {
  const ctx = (edits: Edit[], c: CompileResult | null) => ({ compile: c, levers: LEVERS3, edits });

  it("says when all of it fits", () => {
    const edit = e();
    expect(compileSentence(edit, ctx([edit], compile({ Pct_Canopy: {} })))).toEqual({ text: "You asked for +10 pts tree canopy; all of it fits.", tone: "fits" });
    // a realised change that rounds to the request still fits
    expect(compileSentence(edit, ctx([edit], compile({ Pct_Canopy: { predicted_mean_realised: 9.85, clipped_share: 0.3 } })))?.tone).toBe("fits");
  });

  it("says how much fits when the space clips the request, and where", () => {
    const edit = e({ amount: 20 });
    expect(compileSentence(edit, ctx([edit], compile({ Pct_Canopy: { mean_requested: 20, predicted_mean_realised: 14, clipped_share: 1 } })))).toEqual({
      text: "You asked for +20 pts tree canopy; about 14 pts fits — the rest is beyond what the space allows.",
      tone: "clipped",
    });
    const ten = e();
    expect(compileSentence(ten, ctx([ten], compile({ Pct_Canopy: { predicted_mean_realised: 9.2, clipped_share: 0.08 } })))?.text).toBe(
      "You asked for +10 pts tree canopy; about 9.2 pts fits — in 8% of the places the rest is beyond what the space allows.",
    );
    const paving = e({ lever: "Pct_Impervious", amount: -10 });
    expect(compileSentence(paving, ctx([paving], compile({ Pct_Impervious: { mean_requested: -10, predicted_mean_realised: -6, clipped_share: 0.5 } })))?.text).toBe(
      "You asked for −10 pts paving; about 6 pts fits — in 50% of the places the rest is beyond what the space allows.",
    );
  });

  it("averages for level modes, and says when nothing changes", () => {
    const floor = e({ mode: "floor", amount: 30 });
    expect(compileSentence(floor, ctx([floor], compile({ Pct_Canopy: { mean_requested: 8.24, predicted_mean_realised: 8.24 } })))?.text).toBe("You asked for +8.24 pts tree canopy on average; all of it fits.");
    expect(compileSentence(floor, ctx([floor], compile({ Pct_Canopy: { n_cells: 0, mean_requested: 0, predicted_mean_realised: 0 } })))).toEqual({
      text: "Every place you chose is already at or above this level, so nothing changes.",
      tone: "nothing",
    });
  });

  it("names the other changes when a lever is used twice (compile numbers are per lever)", () => {
    const a = e({ amount: 10 });
    const b = e({ amount: 20, where: { kind: "zones", values: [2] } });
    expect(compileSentence(a, ctx([a, b], compile({ Pct_Canopy: { mean_requested: 15, predicted_mean_realised: 12, clipped_share: 0.4 } })))?.text).toBe(
      "Together with your other tree canopy changes, you asked for +15 pts tree canopy on average; about 12 pts fits — in 40% of the places the rest is beyond what the space allows.",
    );
  });

  it("is absent before the compile, for unfinished changes and for levers not compiled", () => {
    const edit = e();
    expect(compileSentence(edit, ctx([edit], null))).toBeNull();
    const unfinished = e({ amount: undefined });
    expect(compileSentence(unfinished, ctx([unfinished], compile({ Pct_Canopy: {} })))).toBeNull();
    expect(compileSentence(e({ lever: "Albedo", amount: 0.1 }), ctx([edit], compile({ Pct_Canopy: {} })))).toBeNull();
  });
});

// ---------------------------------------------------------------- the steps against a stub workbench

type Stub = WorkbenchModel & { setEdit: ReturnType<typeof vi.fn>; addEdit: ReturnType<typeof vi.fn>; removeEdit: ReturnType<typeof vi.fn>; setTint: ReturnType<typeof vi.fn> };

function stub(edits: Edit[], over: Partial<WorkbenchModel> = {}): Stub {
  return {
    rid: RID,
    grid: { meta: gridMeta() } as unknown as GridData,
    levers: LEVERS3,
    regions: [{ id: "rg_1", name: "Downtown", spec: { kind: "zones", values: [1] }, n_cells: 3 }],
    columns: [],
    planRefs: [],
    temperatureColumn: "pred:target",
    populationColumn: "layer:people",
    mapPick: null,
    compile: null,
    compiling: false,
    doc: doc({ edits }),
    setEdit: vi.fn(),
    addEdit: vi.fn(() => edits.length),
    removeEdit: vi.fn(),
    setTint: vi.fn(),
    ...over,
  } as unknown as Stub;
}

const radios = (root: ParentNode, group: string) => [...root.querySelectorAll<HTMLButtonElement>(`[role="radiogroup"][aria-label="${group}"] [role="radio"]`)];

describe("Step 01 · What to change", () => {
  it("creates the change from a lever card in the lever's cooling direction", async () => {
    const m = stub([]);
    const { container } = render(<StepWhat model={m} index={0} idPrefix="edit-0" />);
    expect(container.querySelectorAll(".stage-num").length).toBe(1);
    expect(byText(container, "[data-lever] strong", "Less paving")).not.toBeNull();
    const dot = container.querySelector('[data-lever="Albedo"] .flow-step-trust')!;
    expect(dot.getAttribute("data-tone")).toBe("warn");
    // the dot's words are read with the card; on hover they show with what they mean, in a tooltip (no native title)
    expect(dot.hasAttribute("title")).toBe(false);
    act(() => {
      dot.dispatchEvent(new MouseEvent("mouseover", { bubbles: true, relatedTarget: null }));
    });
    const tip = await waitFor(() => document.querySelector('[role="tooltip"]'), 2000, "trust tooltip");
    expect(tip.textContent).toContain(TRUST_TIP);
    act(() => {
      dot.dispatchEvent(new MouseEvent("mouseout", { bubbles: true, relatedTarget: null }));
    });
    click(container.querySelector('[data-lever="Pct_Impervious"]'));
    expect(m.addEdit).toHaveBeenCalledWith("Pct_Impervious", { amount: -10 });
  });

  it("swaps the lever of an existing change", () => {
    const edit = e({ where: { kind: "zones", values: [1] } });
    const m = stub([edit]);
    const { container } = render(<StepWhat model={m} index={0} idPrefix="edit-0" />);
    expect(container.querySelector('[data-lever="Pct_Canopy"]')!.getAttribute("aria-pressed")).toBe("true");
    click(container.querySelector('[data-lever="Albedo"]'));
    expect(m.setEdit).toHaveBeenCalledWith(0, { lever: "Albedo", mode: "add", amount: 0.1, where: { kind: "zones", values: [1] } });
  });
});

describe("Step 02 · Where", () => {
  it("offers the presets the run has and writes the preset's selection", () => {
    const edit = e({ where: { kind: "zones", values: [1] } });
    const m = stub([edit], { populationColumn: null });
    const { container } = render(<StepWhere model={m} index={0} idPrefix="edit-0" />);
    const r = radios(container, "Where");
    expect(r.map((b) => b.textContent)).toEqual(["Whole city", "A neighbourhood", "Hottest places", "Draw on the map"]);
    expect(r[1].getAttribute("aria-checked")).toBe("true");
    expect(r[1].tabIndex).toBe(0);
    // the neighbourhood picker: saved areas and zones
    expect(byText(container, ".flow-step-areas .chip", "Downtown")!.getAttribute("aria-pressed")).toBe("false");
    expect(byText(container, ".flow-step-areas .chip", "Zone 1")!.getAttribute("aria-pressed")).toBe("true");
    click(byText(container, ".flow-step-areas .chip", "Downtown"));
    expect(m.setEdit).toHaveBeenLastCalledWith(0, { ...edit, where: { op: "or", args: [{ kind: "zones", values: [1] }, { kind: "region", id: "rg_1" }] } }, undefined);
    click(r[2]);
    expect(m.setEdit).toHaveBeenLastCalledWith(0, { ...edit, where: { kind: "top", column: "pred:target", frac: 0.1, direction: "highest" } }, undefined);
    // arrow keys move the choice
    key(r[1], "ArrowLeft");
    expect(m.setEdit).toHaveBeenLastCalledWith(0, { ...edit, where: { kind: "all" } }, undefined);
    expect(container.textContent).toContain("Applies in zone 1.");
  });

  it("sets the share of the hottest places", () => {
    const edit = e({ where: { kind: "top", column: "pred:target", frac: 0.1, direction: "highest" } });
    const m = stub([edit]);
    const { container } = render(<StepWhere model={m} index={0} idPrefix="edit-0" />);
    expect(container.querySelector('[data-step="where"]')!.getAttribute("data-preset")).toBe("hottest");
    const share = container.querySelector<HTMLInputElement>("#edit-0-share")!;
    expect(share.value).toBe("10");
    typeInto(share, "20");
    key(share, "Enter");
    expect(m.setEdit).toHaveBeenLastCalledWith(0, { ...edit, where: { kind: "top", column: "pred:target", frac: 0.2, direction: "highest", k: undefined } }, "share-0");
    expect(container.textContent).toContain("Applies in the hottest 10% of places.");
  });

  it("uses the next shape drawn on the map while 'Draw on the map' is chosen", () => {
    const edit = e();
    const m = stub([edit]);
    const { container, rerender } = render(<StepWhere model={m} index={0} idPrefix="edit-0" />);
    click(byText(container, '[role="radio"]', "Draw on the map"));
    expect(m.setEdit).not.toHaveBeenCalled(); // nothing drawn yet
    expect(container.querySelector('[data-step="where"]')!.getAttribute("data-preset")).toBe("draw");
    expect(container.textContent).toContain("Until you draw, this change still applies across the whole city.");
    const pick = { spec: { kind: "circle", crs: "EPSG:4326", center: [-71.4, 41.8], radius_m: 300 } as SelectionSpec, label: "Circle 300 m" };
    rerender(<StepWhere model={{ ...m, mapPick: pick }} index={0} idPrefix="edit-0" />);
    expect(m.setEdit).toHaveBeenCalledWith(0, { ...edit, where: pick.spec }, undefined);
  });

  it("asks for a lever first on a change not started yet", () => {
    const { container } = render(<StepWhere model={stub([])} index={0} idPrefix="edit-0" />);
    expect(container.querySelector('[data-pending="true"]')).not.toBeNull();
    expect(container.querySelector('[role="radiogroup"]')).toBeNull();
  });
});

describe("Step 03 · How much", () => {
  it("shows the planner modes, a slider and #edit-0-amount in natural units, and the fit sentence", () => {
    const edit = e({ where: { kind: "zones", values: [1] } });
    const m = stub([edit], { compile: compile({ Pct_Canopy: { predicted_mean_realised: 9.2, clipped_share: 0.08 } }) });
    const { container } = render(<StepHowMuch model={m} index={0} idPrefix="edit-0" />);
    expect(radios(container, "How to change it").map((b) => b.textContent)).toEqual(["Add", "Bring up to at least", "Use available planting space"]);
    const amount = container.querySelector<HTMLInputElement>("#edit-0-amount")!;
    expect(amount.value).toBe("10");
    expect(container.querySelector('label[for="edit-0-amount"]')!.textContent).toBe("How much to add");
    const slider = container.querySelector<HTMLInputElement>('input[type="range"]')!;
    expect([slider.min, slider.max, slider.step]).toEqual(["0", "100", "1"]);
    expect(slider.getAttribute("aria-valuetext")).toBe("10 pts");
    typeInto(slider, "25");
    expect(m.setEdit).toHaveBeenLastCalledWith(0, { ...edit, amount: 25 }, "amount-0");
    // typical amounts (the lever's doses) are their own undo step
    click(byText(container, ".flow-step-doses .chip", "20 pts"));
    expect(m.setEdit).toHaveBeenLastCalledWith(0, { ...edit, amount: 20 }, undefined);
    const fit = container.querySelector("[data-compile-sentence]")!;
    expect(fit.getAttribute("data-tone")).toBe("clipped");
    expect(fit.textContent).toBe("You asked for +10 pts tree canopy; about 9.2 pts fits — in 8% of the places the rest is beyond what the space allows.");
    click(radios(container, "How to change it")[2]);
    expect(m.setEdit).toHaveBeenLastCalledWith(0, { lever: "Pct_Canopy", mode: "fill_headroom", amount: 0.5, where: { kind: "zones", values: [1] } });
  });

  it("takes paving away (stored negative) and caps it; planting space is canopy-only", () => {
    const edit = e({ lever: "Pct_Impervious", amount: -10 });
    const m = stub([edit]);
    const { container } = render(<StepHowMuch model={m} index={0} idPrefix="edit-0" />);
    expect(radios(container, "How to change it").map((b) => b.textContent)).toEqual(["Take away", "Bring down to at most"]);
    expect(container.querySelector<HTMLInputElement>("#edit-0-amount")!.value).toBe("10");
    typeInto(container.querySelector('input[type="range"]'), "30");
    expect(m.setEdit).toHaveBeenLastCalledWith(0, { ...edit, amount: -30 }, "amount-0");
    const albedo = render(<StepHowMuch model={stub([e({ lever: "Albedo", amount: 0.1 })])} index={0} idPrefix="edit-0" />);
    expect(radios(albedo.container, "How to change it").map((b) => b.textContent)).toEqual(["Add", "Bring up to at least"]);
  });

  it("shows the planting share as a percentage", () => {
    const edit = e({ mode: "fill_headroom", amount: 0.5 });
    const m = stub([edit]);
    const { container } = render(<StepHowMuch model={m} index={0} idPrefix="change-0" />);
    expect(container.querySelector<HTMLInputElement>("#change-0-amount")!.value).toBe("50");
    typeInto(container.querySelector('input[type="range"]'), "75");
    expect(m.setEdit).toHaveBeenLastCalledWith(0, { ...edit, amount: 0.75 }, "amount-0");
  });

  it("points a change in another mode to Advanced options, with a way back", () => {
    const edit = e({ mode: "set", amount: 40 });
    const m = stub([edit]);
    const { container } = render(<StepHowMuch model={m} index={0} idPrefix="edit-0" />);
    expect(container.textContent).toContain("“Set to exactly”, set in Advanced options");
    expect(container.querySelector("#edit-0-amount")).toBeNull();
    click(byText(container, "button", "Switch to a simple amount"));
    expect(m.setEdit).toHaveBeenLastCalledWith(0, { lever: "Pct_Canopy", mode: "add", amount: 10, where: { kind: "all" } });
  });

  it("explains a blocking compile warning for this change", () => {
    const edit = e();
    const m = stub([edit], { compile: compile({ Pct_Canopy: {} }, { warnings: [{ code: "x", message: "edit 1 selects no cells", edit_index: 0, blocking: true }] }) });
    const { container } = render(<StepHowMuch model={m} index={0} idPrefix="edit-0" />);
    expect(container.querySelector(".flow-step-blocked")!.textContent).toContain("Edit 1 selects no places.");
  });
});

describe("A change collapsed to one sentence", () => {
  it("reads as one sentence with Edit and Remove, and flags an unfinished change", () => {
    const onEdit = vi.fn();
    const onRemove = vi.fn();
    const m = stub([e({ where: { kind: "region", id: "rg_1" }, label: "street trees" }), e({ lever: "Albedo", amount: undefined })]);
    const { container } = render(
      <>
        <EditSentence model={m} index={0} onEdit={onEdit} onRemove={onRemove} />
        <EditSentence model={m} index={1} onEdit={onEdit} onRemove={onRemove} />
      </>,
    );
    const [first, second] = container.querySelectorAll(".lab-edit-sentence");
    expect(first.querySelector(".flow-step-sentence-text")!.textContent).toBe("Add 10 pts tree canopy in Downtown · street trees");
    expect(second.textContent).toContain("Add … surface reflectivity across the whole city — not finished");
    expect(second.getAttribute("data-unfinished")).toBe("true");
    click(byText(first, "button", "Edit"));
    click(byText(first, "button", "Remove"));
    expect(onEdit).toHaveBeenCalledTimes(1);
    expect(onRemove).toHaveBeenCalledTimes(1);
    expect(byText(first, "button", "Edit")!.getAttribute("aria-describedby")).toBe(first.querySelector(".flow-step-sentence-text")!.id);
  });
});
