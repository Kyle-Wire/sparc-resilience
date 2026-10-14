// The quick estimate in plain words (phase 3): model/quick.ts turns the preview summary and the compile
// into the big number, the three facts and the trust line; PreviewPane and RunFullModel lay them out
// (one Preview region, one accent "Run full model" button, the engine line, the build-estimate empty
// state); the Design map shows a hatch key "Rough estimate here" while the preview is hatched.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { pack, packBits } from "../../../api/binary";
import type { CompileResult, PreviewResult, PreviewSummary } from "../../../api/lab";
import { ApiError } from "../../../api/client";
import { clearResources } from "../../../api/resource";
import type { Action } from "../../../api/types";
import type { GridData } from "../../../map/grid";
import { navigate } from "../../../router";
import { useJobs } from "../../../stores/jobs";
import { byText, click, flush, mockFetch, render, waitFor, type MockRoute } from "../../../test/render";
import { exactAdvice } from "../components/CompilePanel";
import { PreviewPane } from "../components/flow/PreviewPane";
import { RunFullModel } from "../components/flow/RunFullModel";
import { forgetDrafts } from "../model/draft";
import {
  HATCH_KEY,
  QUICK_WORDS,
  approxDuration,
  approxTemp,
  areaLine,
  areaWords,
  bigNumber,
  buildLabel,
  changeTone,
  cityText,
  costLine,
  costWords,
  engineTone,
  fullModelUpToDate,
  hasCostModel,
  heatLine,
  quickView,
  residentsText,
  roundSig,
  runDisabledReason,
  runLabel,
  trustView,
  unfinishedNote,
} from "../model/quick";
import type { WorkbenchModel } from "../model/useWorkbench";
import { LAB_WORDS, PREVIEW_TRUST_LINE, noQuickEstimateText } from "../model/words";
import { LEVERS, RID, doc, gridMeta, result, runRoutes, scenario } from "../__fixtures__/lab";

const CELL_KM2 = 0.0009; // a 30 m cell

function summary(extra: Partial<PreviewSummary> = {}): PreviewSummary {
  return { mean: -0.0512, edited_mean: -0.4234, n_edited: 3, outside_share: 0.1, trust: "good", hatched: false, reasons: [], request_seq: 1, ...extra };
}

function compile(extra: Partial<CompileResult> = {}): CompileResult {
  return {
    content_hash: "h",
    portable: true,
    levers: { Pct_Canopy: { n_cells: 3, mean_requested: 10, total_requested: 30, predicted_mean_realised: 9.2, clipped_share: 0.08, est_cost: 27.6 } },
    union_cells: 3,
    people: 35,
    warnings: [],
    est_exact_s: 120,
    emulator: { usable: true, hatched: false, reasons: [] },
    ...extra,
  };
}

const NO_ADVICE = { recommend: false, reasons: [] };

describe("quick estimate words (model/quick.ts)", () => {
  it("rounds the big number for reading, cool for cooling and warm for warming", () => {
    expect(bigNumber(-0.4234, "degF", 3)).toEqual({ text: "About −0.42 °F", tone: "cool", numeric: true });
    expect(bigNumber(1.26, "degF", 3)).toEqual({ text: "About +1.3 °F", tone: "warm", numeric: true });
    expect(bigNumber(-12.4, "degC", 3).text).toBe("About −12 °C");
    expect(bigNumber(0.001, "degF", 3)).toEqual({ text: QUICK_WORDS.noNoticeable, tone: "neutral", numeric: false });
    expect(bigNumber(Number.NaN, "degF", 0)).toEqual({ text: QUICK_WORDS.noChangeYet, tone: "neutral", numeric: false });
    expect(bigNumber(-0.4, "degF", 0).text).toBe(QUICK_WORDS.noChangeYet);
    expect(changeTone(-0.004)).toBe("neutral");
    expect(changeTone(null)).toBe("neutral");
  });

  it("words the citywide figure, saying 'less than 0.01' rather than a row of zeros", () => {
    expect(cityText(-0.0512, "degF")).toBe("−0.05 °F");
    expect(cityText(-0.002, "degF")).toBe("Less than 0.01 °F");
    expect(cityText(Number.NaN, "degF")).toBe("—");
    expect(approxTemp(0.004, "degF")).toBe("less than 0.01 °F");
  });

  it("gives the area changed in km² or hectares, its share of the city and the residents", () => {
    expect(areaWords(3 * CELL_KM2)).toBe("0.27 ha");
    expect(areaWords(0.456)).toBe("46 ha");
    expect(areaWords(0.0456)).toBe("4.6 ha");
    expect(areaWords(2.345)).toBe("2.3 km²");
    expect(areaWords(12.6)).toBe("13 km²");
    expect(areaLine(3, CELL_KM2, 7, 35)).toBe("0.27 ha (43% of the city) · 35 residents");
    expect(areaLine(3, null, 7, null)).toBe("3 places (43% of the city)");
    expect(areaLine(1, null, null, 1)).toBe("1 place · 1 resident");
    expect(areaLine(7, null, 7, 12345.6)).toBe("7 places (the whole city) · about 12,300 residents");
    expect(areaLine(2, 1, 1000, null)).toBe("2.0 km² (under 1% of the city)");
    expect(residentsText(null)).toBeNull();
    expect(roundSig(12345.6, 3)).toBe(12300);
  });

  it("shows a cost only with a cost model (a scenario price, or a catalogue price other than the default 1)", () => {
    const c = compile();
    expect(hasCostModel(c, LEVERS, {})).toBe(false); // canopy costs the default 1 per unit
    expect(costLine(c, LEVERS, {})).toBeNull();
    expect(costLine(compile({ levers: { Pct_Canopy: { ...c.levers.Pct_Canopy, est_cost: 27600 } } }), LEVERS, { Pct_Canopy: { per_unit: 920 } })).toBe("about 27,600");
    const albedo = compile({ levers: { Albedo: { n_cells: 3, mean_requested: 0.1, total_requested: 0.3, predicted_mean_realised: 0.1, clipped_share: 0, est_cost: 1_234_000 } } });
    expect(hasCostModel(albedo, LEVERS, undefined)).toBe(true); // 100 per unit in the catalogue
    expect(costLine(albedo, LEVERS, undefined)).toBe("about 1.2 million");
    expect(costLine(compile({ levers: { Albedo: { ...albedo.levers.Albedo, est_cost: 0 } } }), LEVERS, undefined)).toBeNull();
    expect(costLine(null, LEVERS, undefined)).toBeNull();
    expect(costWords(42.4)).toBe("about 42");
    expect(costWords(3.25)).toBe("about 3.3");
  });

  it("words the heat-risk shift as the heat preview card does", () => {
    expect(heatLine({ ecBefore: 1200.4, ecAfter: 1080, measure: "people" })).toBe("Moves about 120 residents out of Extreme caution or worse");
    expect(heatLine({ ecBefore: 15432, ecAfter: 0, measure: "people" })).toBe("Moves about 15,400 residents out of Extreme caution or worse");
    expect(heatLine({ ecBefore: 2, ecAfter: 5, measure: "cells" })).toBe("Puts about 3 more places at Extreme caution or worse");
    expect(heatLine({ ecBefore: 40, ecAfter: 40.2, measure: "people" })).toBe("No change in residents at Extreme caution or worse");
    expect(heatLine(null)).toBeNull();
  });

  it("says the quick estimate is reliable, rough, or rough in the hatched areas, with exactAdvice() reasons in plain words", () => {
    expect(trustView(summary(), NO_ADVICE, LEVERS)).toEqual({ tone: "good", line: PREVIEW_TRUST_LINE.good, reasons: [] });
    // the map hatches the changed places
    const big = "Preview unreliable at this scale (12,000 edited cells > 5,000) — run exact";
    const hatched = summary({ hatched: true, trust: "rough", reasons: [big] });
    expect(trustView(hatched, NO_ADVICE, LEVERS)).toEqual({
      tone: "hatched",
      line: PREVIEW_TRUST_LINE.hatched,
      reasons: ["The change covers a large area (12,000 places), where the quick estimate is less reliable."],
    });
    // a rough lever (from the real exactAdvice), not hatched
    const albedo = compile({ levers: { Albedo: { n_cells: 3, mean_requested: 0.1, total_requested: 0.3, predicted_mean_realised: 0.1, clipped_share: 0, est_cost: 300 } } });
    const advice = exactAdvice(albedo, summary(), LEVERS);
    // the generic "<lever>: preview is rough" only restates the rough line (the lever card's dot names the lever)
    expect(advice.reasons).toEqual(["Albedo: preview is rough"]);
    expect(trustView(summary(), advice, LEVERS)).toEqual({ tone: "rough", line: PREVIEW_TRUST_LINE.rough, reasons: [] });
    // hatched with a reason naming the lever: the reasons never repeat the trust line
    const hatchedCanopy = summary({ hatched: true, trust: "rough", reasons: ["Preview unreliable at this scale (uniform Pct_Canopy rel. error 60%)"] });
    const both = { recommend: true, reasons: ["Preview unreliable at this scale (uniform Pct_Canopy rel. error 60%)", "Canopy cover: preview is rough"] };
    const v = trustView(hatchedCanopy, both, LEVERS);
    expect(v.line).toBe(PREVIEW_TRUST_LINE.hatched);
    expect(v.reasons).toEqual(["Across the whole city the quick estimate for trees & canopy can be off by about 60%."]);
    expect(v.reasons.join(" ")).not.toMatch(/is rough for/);
    // the preview's own trust is enough to say rough before the compile arrives
    expect(trustView(summary({ trust: "rough" }), NO_ADVICE, LEVERS).tone).toBe("rough");
  });

  it("composes the whole pane", () => {
    const v = quickView({ summary: summary(), unit: "degF", compile: compile(), levers: LEVERS, costs: {}, cellAreaKm2: CELL_KM2, nTotal: 7, heat: { ecBefore: 30, ecAfter: 18, measure: "people" }, advice: NO_ADVICE });
    expect(v).toEqual({
      big: { text: "About −0.42 °F", tone: "cool", numeric: true },
      city: "−0.05 °F",
      area: "0.27 ha (43% of the city) · 35 residents",
      cost: null,
      heat: "Moves about 12 residents out of Extreme caution or worse",
      trust: { tone: "good", line: PREVIEW_TRUST_LINE.good, reasons: [] },
    });
  });

  it("labels the run with a rounded duration and says why it cannot run, in plain words", () => {
    expect(runLabel(120)).toBe("Run full model (≈ 2 min)");
    expect(runLabel(150)).toBe("Run full model (≈ 3 min)");
    expect(runLabel(30)).toBe("Run full model (under 1 min)");
    expect(runLabel(5400)).toBe("Run full model (≈ 1.5 h)");
    expect(runLabel(null)).toBe("Run full model");
    expect(approxDuration(Number.NaN)).toBeNull();
    expect(buildLabel("fast", 2)).toBe("Build quick estimate (≈1 min)");
    expect(buildLabel(null, 2)).toBe("Build quick estimate (≈1 min)");
    expect(buildLabel("full", 2)).toBe("Build quick estimate (≈8–10 min)");
    const base = { hasChanges: true, unfinished: 0, runReason: null, blocking: [] };
    expect(runDisabledReason(base)).toBeNull();
    expect(runDisabledReason({ ...base, hasChanges: false })).toBe(QUICK_WORDS.needsChange);
    expect(runDisabledReason({ ...base, blocking: [{ edit_index: 1 }, { edit_index: 1 }, { edit_index: null }] })).toBe("Change 2 cannot run as set — see Advanced options for why.");
    expect(runDisabledReason({ ...base, blocking: [{ edit_index: 2 }, { edit_index: 0 }] })).toBe("Changes 1 and 3 cannot run as set — see Advanced options for why.");
    expect(runDisabledReason({ ...base, blocking: [{ edit_index: null }] })).toBe("Some changes cannot run as set — see Advanced options for why.");
    // the engine line already says the run has no model
    expect(runDisabledReason({ ...base, hasChanges: false, runReason: "This run has no checkpoint, so only the preview is available." })).toBeNull();
    expect(unfinishedNote({ hasChanges: true, unfinished: 2 })).toBe("2 changes are not finished yet, so they are left out.");
    expect(unfinishedNote({ hasChanges: false, unfinished: 1 })).toBeNull();
    expect(engineTone("ready")).toBe("good");
    expect(engineTone("loading")).toBe("busy");
    expect(engineTone("no_checkpoint")).toBe("off");
    expect(engineTone(null)).toBe("idle");
  });
});

// ---------------------------------------------------------------- the panes, on a stub model

function preview(s: PreviewSummary): PreviewResult {
  return { delta: new Float32Array(7), edited: Uint8Array.from([1, 1, 1, 0, 0, 0, 0]), summary: s };
}

function model(over: Partial<WorkbenchModel> = {}): WorkbenchModel {
  const d = doc();
  return {
    rid: RID,
    unit: "degF",
    levers: LEVERS,
    grid: { n: 7, meta: gridMeta() } as unknown as GridData,
    doc: d,
    usable: d.edits,
    store: { brushLayers: { isEmpty: () => true } },
    preview: preview(summary()),
    previewBusy: false,
    previewError: null,
    noEmulator: false,
    emulator: { present: true, kernel_cells: 41, levers: {}, action: null },
    compile: compile(),
    cellAreaKm2: CELL_KM2,
    heatPreview: null,
    advice: NO_ADVICE,
    blocking: [],
    canRun: true,
    runReason: null,
    running: false,
    runExact: vi.fn(async () => {}),
    engineState: "ready",
    engine: { state: "ready", progress: null, status: undefined, reload: async () => {} },
    shownJob: null,
    ...over,
  } as unknown as WorkbenchModel;
}

/** The <dd> after the <dt> reading `term` (the e2e `dd_of` helper). */
function ddOf(root: ParentNode, term: string): HTMLElement | null {
  const dt = [...root.querySelectorAll("dt")].find((x) => x.textContent?.trim() === term);
  return (dt?.nextElementSibling as HTMLElement | null) ?? null;
}

let fetchMock: ReturnType<typeof mockFetch> | null = null;

beforeEach(() => {
  clearResources();
  forgetDrafts();
  useJobs.getState().reset();
});

afterEach(() => {
  fetchMock?.restore();
  fetchMock = null;
});

describe("PreviewPane", () => {
  it("shows the big number, the citywide figure and the facts as dt/dd rows, with a reliable trust line", () => {
    const { container } = render(<PreviewPane model={model({ heatPreview: { shift: { ecBefore: 30, ecAfter: 18, dangerBefore: 0, dangerAfter: 0, meanHiChange: -0.2, total: 100, measure: "people" }, dewpointC: 20, source: null, caveats: [] } })} />);
    const region = container.querySelector('[aria-label="Preview"]')!;
    expect(region).not.toBeNull();
    expect(region.querySelector(".kicker")!.textContent).toBe("QUICK ESTIMATE");
    expect(region.getAttribute("data-trust")).toBe("good");
    expect(region.hasAttribute("data-hatched")).toBe(false);
    const big = region.querySelector("[data-edited-mean]")!;
    expect(big.getAttribute("data-edited-mean")).toBe("-0.4234");
    expect(big.textContent).toBe("About −0.42 °F");
    expect(ddOf(region, "where you made changes")).toBe(big);
    expect(region.querySelector("[data-city-mean]")!.getAttribute("data-city-mean")).toBe("-0.0512");
    expect(ddOf(region, "citywide")!.textContent).toBe("−0.05 °F"); // 3 of 7 places changed: the citywide figure says something new
    expect(region.querySelector('[data-n-edited="3"]')).not.toBeNull();
    expect(ddOf(region, QUICK_WORDS.area)!.textContent).toBe("0.27 ha (43% of the city) · 35 residents");
    expect(ddOf(region, QUICK_WORDS.heat)!.textContent).toBe("Moves about 12 residents out of Extreme caution or worse");
    expect(region.querySelector("[data-cost]")).toBeNull(); // no cost model on the default price
    expect(region.querySelector(".qe-trust")!.textContent).toContain(PREVIEW_TRUST_LINE.good);
  });

  it("says rough in the hatched areas, with the reasons in plain words, and flags unproven heat risk", () => {
    const s = summary({ hatched: true, trust: "rough", reasons: ["Preview unreliable at this scale (12,000 edited cells > 5,000) — run exact"] });
    const m = model({
      preview: preview(s),
      advice: exactAdvice(compile({ emulator: { usable: true, hatched: true, reasons: s.reasons } }), s, LEVERS),
      heatPreview: { shift: { ecBefore: 30, ecAfter: 18, dangerBefore: 0, dangerAfter: 0, meanHiChange: -0.2, total: 100, measure: "people" }, dewpointC: 20, source: null, caveats: ["Canopy cover: not established."] },
    });
    const { container } = render(<PreviewPane model={m} />);
    const region = container.querySelector('[aria-label="Preview"]')!;
    expect(region.getAttribute("data-hatched")).toBe("true");
    expect(region.getAttribute("data-trust")).toBe("hatched");
    const trust = region.querySelector(".qe-trust")!;
    expect(trust.textContent).toContain(PREVIEW_TRUST_LINE.hatched);
    expect([...trust.querySelectorAll("li")].map((li) => li.textContent)).toEqual(["The change covers a large area (12,000 places), where the quick estimate is less reliable."]);
    expect(region.querySelector(".qe-flag")!.textContent).toBe(QUICK_WORDS.heatUnproven);
  });

  it("shows a cost with a cost model, and words instead of a number when nothing changed", () => {
    const d = doc({ costs: { Pct_Canopy: { per_unit: 1000 } } });
    const { container } = render(<PreviewPane model={model({ doc: d, compile: compile({ levers: { Pct_Canopy: { n_cells: 0, mean_requested: 0, total_requested: 0, predicted_mean_realised: 0, clipped_share: 0, est_cost: 27600 } } }), preview: preview(summary({ n_edited: 0, edited_mean: Number.NaN })) })} />);
    const cost = ddOf(container, QUICK_WORDS.cost)!;
    expect(cost.querySelector(".qe-value")!.textContent).toBe("about 27,600");
    // what the cost units are, in an ⓘ (hover, focus or tap), not a native title
    expect(cost.querySelector("button.infotip")!.getAttribute("aria-label")).toBe("What is the estimated cost?");
    expect(cost.querySelector("[title]")).toBeNull();
    const hero = container.querySelector(".qe-hero")!;
    expect(hero.hasAttribute("data-numeric")).toBe(false);
    expect(hero.querySelector(".qe-big")!.textContent).toBe(QUICK_WORDS.noChangeYet);
    expect(container.textContent).toContain(QUICK_WORDS.noChangeYetHint);
  });

  it("is a calm empty state before the first change", () => {
    const { container } = render(<PreviewPane model={model({ preview: null })} />);
    expect(container.querySelector(".qe-idle")!.textContent).toBe("Make a change to see the quick estimate.");
    const busy = render(<PreviewPane model={model({ preview: null, previewBusy: true })} />);
    expect(busy.container.querySelector('[aria-label="Preview"]')!.getAttribute("aria-busy")).toBe("true");
  });

  it("offers to build the quick estimate when the run has none, with the server's build action", async () => {
    let built = 0;
    fetchMock = mockFetch({ ...runRoutes(), [`POST /api/runs/${RID}/actions/emulator`]: () => (built++, { status: 202, body: {} }) });
    const action: Action = { kind: "build_emulator", label: "Build emulator (≈1 min fast, ≈4–5 min per lever full)", method: "POST", path: `/api/runs/${RID}/actions/emulator`, body: {} };
    const { container } = render(<PreviewPane model={model({ noEmulator: true, preview: null, emulator: { present: false, kernel_cells: null, levers: {}, action } })} />);
    const region = container.querySelector('[aria-label="Preview"]')!;
    expect(region.querySelector("[data-no-emulator]")).not.toBeNull();
    // one action (the ⓘ beside the kicker only explains the quick estimate); no native title on it
    expect(region.querySelectorAll("button:not(.infotip)").length).toBe(1);
    expect(region.querySelector("button.infotip[data-term=emulator_preview]")).not.toBeNull();
    expect(region.querySelector("button[title]")).toBeNull();
    const button = await waitFor(() => byText(region, "button", "Build quick estimate (≈1 min)"), 5000, "build button");
    expect(button.classList.contains("accent")).toBe(false); // the accent is Run full model's
    click(button);
    await waitFor(() => built > 0, 5000, "build request");
  });

  it("words the build box by the engine state, so it never contradicts the engine line", async () => {
    fetchMock = mockFetch(runRoutes());
    const action: Action = { kind: "build_emulator", label: "Build emulator", method: "POST", path: `/api/runs/${RID}/actions/emulator`, body: {} };
    const box = (over: Partial<WorkbenchModel>) => {
      const { container } = render(<PreviewPane model={model({ noEmulator: true, preview: null, emulator: { present: false, kernel_cells: null, levers: {}, action }, ...over })} />);
      return container.querySelector("[data-no-emulator]")!;
    };
    // no saved models: the estimate cannot be built either (its build needs checkpoint.pkl), so the box
    // sends to Launch instead of offering a build that would fail; the title is not repeated by the body
    const none = box({ engineState: "no_checkpoint", pid: "p_demo" });
    expect(none.querySelector(".qe-empty-title")!.textContent).toBe("No quick estimate yet");
    expect(none.querySelector(".cap")!.textContent).toBe("Launch a new run (Fast takes a few minutes) to try changes.");
    expect(none.textContent).not.toContain("either way");
    expect(none.querySelector(".cap")!.textContent).not.toMatch(/^No quick estimate/);
    expect(none.getAttribute("data-no-checkpoint")).toBe("true");
    expect(none.querySelector("button:not(.infotip)")).toBeNull();
    expect(none.querySelector("a")!.getAttribute("href")).toBe("/p/p_demo/launch");
    expect(none.querySelector("a")!.textContent).toBe("Launch a run");
    // a model that cannot load: the estimate is still the way in
    expect(noQuickEstimateText("incompatible")).toBe(LAB_WORDS.noQuickEstimateBuildOnly);
    expect(box({ engineState: "incompatible" }).querySelector("button")).not.toBeNull();
    // a model that runs: the full model works either way
    expect(box({ engineState: "ready" }).querySelector(".cap")!.textContent).toBe(LAB_WORDS.noQuickEstimate);
    expect(LAB_WORDS.noQuickEstimate).toContain("either way");
    // unknown or failed: no promise
    expect(noQuickEstimateText(null)).not.toContain("either way");
    expect(noQuickEstimateText("error")).not.toContain("either way");
    await flush();
  });

  it("treats a no_emulator preview error as the build state, and words other errors plainly", async () => {
    fetchMock = mockFetch(runRoutes());
    const action: Action = { kind: "build_emulator", label: "Build emulator", method: "POST", path: `/api/runs/${RID}/actions/emulator`, body: {} };
    const noEmu = render(<PreviewPane model={model({ previewError: new ApiError(404, "no_emulator", "Build the emulator first", null, action) })} />);
    await flush();
    expect(byText(noEmu.container, "button", "Build quick estimate")).not.toBeNull();
    const failed = render(<PreviewPane model={model({ previewError: new ApiError(422, "invalid", "edit 1: amount is not a number") })} />);
    expect(failed.container.textContent).toContain(QUICK_WORDS.unavailable);
    expect(failed.container.textContent).toContain("edit 1: amount is not a number");
    expect(failed.container.querySelector("[data-edited-mean]")).toBeNull();
  });
});

/** A draft store stand-in: saved (`dirty` false) at revision 1 unless told otherwise. */
function draftStore(state: { dirty?: boolean; revision?: number | null } = {}) {
  const v = state.dirty ? 2 : 1;
  const store = { brushLayers: { isEmpty: () => true }, getState: () => ({ version: v, savedVersion: 1, revision: state.revision === undefined ? 1 : state.revision }), subscribe: () => () => {} };
  return store as unknown as WorkbenchModel["store"];
}

describe("the full model is up to date", () => {
  const r = result("res_1");
  it("only for a current, saved, exact result of the open scenario revision with no run under way", () => {
    expect(fullModelUpToDate(r, { sid: "sc_a", revision: 1, dirty: false })).toBe(true);
    expect(fullModelUpToDate(undefined, { sid: "sc_a", revision: 1, dirty: false })).toBe(false);
    expect(fullModelUpToDate(r, { sid: "sc_a", revision: 1, dirty: true })).toBe(false); // unsaved changes
    expect(fullModelUpToDate(r, { sid: "sc_a", revision: 1, dirty: false, running: true })).toBe(false);
    expect(fullModelUpToDate(r, { sid: "sc_b", revision: 1, dirty: false })).toBe(false); // another scenario
    expect(fullModelUpToDate(r, { sid: "sc_a", revision: 2, dirty: false })).toBe(false); // another revision
    expect(fullModelUpToDate(r, { sid: "sc_a", revision: null, dirty: false })).toBe(true); // revision not known yet
    expect(fullModelUpToDate({ ...r, stale: true }, { sid: "sc_a", revision: 1, dirty: false })).toBe(false); // the model changed
    expect(fullModelUpToDate({ ...r, summary: { ...r.summary, stale: true } }, { sid: "sc_a", revision: 1, dirty: false })).toBe(false);
    expect(fullModelUpToDate({ ...r, summary: { ...r.summary, kind: "sweep_point" } }, { sid: "sc_a", revision: 1, dirty: false })).toBe(false);
  });

  it("drops the accent: the button goes to the answer, one line says it is up to date, without the number", () => {
    const m = model({ result: r, storeSid: "sc_a", store: draftStore() });
    const { container } = render(<RunFullModel model={m} />);
    const buttons = [...container.querySelectorAll<HTMLButtonElement>("button:not(.infotip)")];
    expect(buttons.map((b) => b.textContent)).toEqual([LAB_WORDS.seeFullResult]);
    expect(buttons[0].classList.contains("accent")).toBe(false);
    click(buttons[0]);
    expect(m.runExact).not.toHaveBeenCalled();
    const line = container.querySelector("[data-full-result]")!;
    expect(line.getAttribute("data-full-result")).toBe("current");
    expect(line.textContent).toBe("Full model is up to date for these changes.");
    expect(container.textContent).not.toMatch(/0\.62|−0\.6/);
    // unsaved changes since: the accent Run full model is back, and the line says the result is out of date
    const dirty = render(<RunFullModel model={model({ result: r, storeSid: "sc_a", store: draftStore({ dirty: true }) })} />);
    const run = dirty.container.querySelector("button")!;
    expect(run.textContent).toBe("Run full model (≈ 2 min)");
    expect(run.classList.contains("accent")).toBe(true);
    expect(dirty.container.querySelector("[data-full-result]")!.getAttribute("data-full-result")).toBe("outdated");
    expect(dirty.container.querySelector("[data-full-result]")!.textContent).not.toMatch(/\d/);
  });

  it("lets the quick estimate step back once the full model is up to date", () => {
    const { container } = render(<PreviewPane model={model({ result: result("res_1"), storeSid: "sc_a", store: draftStore() })} />);
    const region = container.querySelector('[aria-label="Preview"]')!;
    expect(region.getAttribute("data-superseded")).toBe("true");
    expect(region.querySelector(".kicker")!.textContent).toBe("Quick estimate (before the full model)");
    expect(region.querySelector(".qe-hero")!.getAttribute("data-superseded")).toBe("true");
    // before the full model: the plain kicker, the hero leads
    const before = render(<PreviewPane model={model()} />);
    expect(before.container.querySelector('[aria-label="Preview"]')!.hasAttribute("data-superseded")).toBe(false);
    expect(before.container.querySelector(".kicker")!.textContent).toBe("QUICK ESTIMATE");
  });
});

describe("RunFullModel", () => {
  it("is the one accent button, runs the full model and shows the engine as one line", () => {
    const m = model();
    const { container } = render(<RunFullModel model={m} />);
    // one action; the engine line's ⓘ only explains what the full model is
    const buttons = [...container.querySelectorAll<HTMLButtonElement>("button:not(.infotip)")];
    expect(buttons.length).toBe(1);
    expect(container.querySelector("[data-engine] button.infotip")!.getAttribute("data-term")).toBe("exact_engine");
    expect(buttons[0].hasAttribute("title")).toBe(false);
    expect(buttons[0].textContent).toBe("Run full model (≈ 2 min)");
    expect(buttons[0].classList.contains("accent")).toBe(true);
    expect(buttons[0].disabled).toBe(false);
    click(buttons[0]);
    expect(m.runExact).toHaveBeenCalledTimes(1);
    const engine = container.querySelector("[data-engine]")!;
    expect(engine.getAttribute("data-engine")).toBe("ready");
    expect(engine.textContent).toBe("Model ready");
    expect(container.querySelector("[data-run-why]")).toBeNull();
  });

  it("says why it cannot run, in plain words", () => {
    const none = render(<RunFullModel model={model({ usable: [], doc: doc({ edits: [] }), canRun: false, compile: null })} />);
    expect(none.container.querySelector("button")!.disabled).toBe(true);
    expect(none.container.querySelector("button")!.textContent).toBe("Run full model");
    expect(none.container.querySelector("[data-run-why]")!.textContent).toBe(QUICK_WORDS.needsChange);
    const blocked = render(<RunFullModel model={model({ blocking: [{ code: "x", message: "Albedo above 1", edit_index: 0, blocking: true }] })} />);
    expect(blocked.container.querySelector("button")!.disabled).toBe(true);
    expect(blocked.container.querySelector("[data-run-why]")!.textContent).toBe("Change 1 cannot run as set — see Advanced options for why.");
    const loading = render(<RunFullModel model={model({ engineState: "loading", engine: { state: "loading", progress: 0.4, status: undefined, reload: async () => {} } })} />);
    expect(loading.container.querySelector("[data-engine]")!.textContent).toBe("Loading model… 40%");
    // no checkpoint: no button that can never be pressed; the engine line says it, once
    const noModel = render(<RunFullModel model={model({ canRun: false, engineState: "no_checkpoint", runReason: "This run has no checkpoint, so only the preview is available." })} />);
    expect(noModel.container.querySelector("button:not(.infotip)")).toBeNull();
    expect(noModel.container.querySelector("[data-no-model]")).not.toBeNull();
    expect(noModel.container.querySelector("[data-engine]")!.textContent).toBe("Model not available for this run — quick estimate only");
    expect(noModel.container.querySelector("[data-run-why]")).toBeNull();
  });

  it("notes unfinished changes and shows the job while it runs", () => {
    const d = doc({ edits: [...doc().edits, { lever: "Albedo", mode: "add", amount: Number.NaN, where: { kind: "all" } }] });
    useJobs.getState().upsert({
      id: "j_run", kind: "engine.scenario", lane: "engine", executor: "engine", label: "Full model", status: "running", project_id: null, run_id: RID, study_id: null, scenario_id: "sc_a",
      parent_job_id: null, after_job_id: null, priority: 0, params: {}, created_utc: "2026-10-01T22:00:00Z", started_utc: null, finished_utc: null,
      progress: 0.5, eta_s: 30, eta_lo: null, eta_hi: null, stage: null, current_path: null, exit_code: null, error: null, blocked: null, result: null, peak_rss_mb: null, threads: null,
    });
    const { container } = render(<RunFullModel model={model({ doc: d, usable: doc().edits, shownJob: "j_run" })} />);
    expect(container.textContent).toContain("1 change is not finished yet, so it is left out.");
    expect(container.querySelector(".jobstrip")).not.toBeNull();
  });
});

// ---------------------------------------------------------------- the map's hatch key, on the page

/** JSON with non-ASCII escaped, as the server sends it in a header (headers are Latin-1 only). */
function asciiJson(v: unknown): string {
  return JSON.stringify(v).replace(/[\u007f-\uffff]/g, (c) => `\\u${c.charCodeAt(0).toString(16).padStart(4, "0")}`);
}

function hatchedPreview(seq: number): MockRoute {
  const delta = [-0.6, -0.5, -0.4, 0, 0, 0, -0.1];
  const { buffer, offsets } = pack([
    { name: "delta", dtype: "float32", data: Float32Array.from(delta) },
    { name: "edited", dtype: "uint8", data: packBits([1, 1, 1, 0, 0, 0, 0]) },
  ]);
  const s = { mean: -0.23, edited_mean: -0.5, n_edited: 3, outside_share: 0.1, trust: "rough", hatched: true, reasons: ["Preview unreliable at this scale (12,000 edited cells > 5,000) — run exact"], request_seq: seq };
  return { raw: buffer, headers: { "content-type": "application/octet-stream", "X-SPARC-Offsets": JSON.stringify(offsets), "X-SPARC-Summary": asciiJson(s) } };
}

describe("Design map hatch key", () => {
  it("shows 'Rough estimate here' with the hatch swatch while the quick estimate is hatched", async () => {
    fetchMock = mockFetch({
      ...runRoutes(),
      "GET /api/scenarios/sc_a": { body: scenario("sc_a") },
      [`POST /api/runs/${RID}/preview`]: (_u, init) => hatchedPreview(JSON.parse(String(init.body)).request_seq as number),
      [`POST /api/runs/${RID}/compile`]: { body: compile({ emulator: { usable: true, hatched: true, reasons: ["Preview unreliable at this scale (12,000 edited cells > 5,000) — run exact"] } }) },
    });
    navigate(`/r/${RID}/lab/s/sc_a`, { replace: true });
    const { default: Design } = await import("../Design"); // the page (and the other flow steps) only for this test
    const { container } = render(<Design />);
    const key = await waitFor(() => container.querySelector("[data-hatch-key]"), 8000, "hatch key");
    expect(key.textContent).toBe(HATCH_KEY.preview.label); // one line: the swatch and its label, the trust text stays in the right pane
    expect(key.querySelector(".qe-hatch")).not.toBeNull();
    const region = await waitFor(() => container.querySelector('[aria-label="Preview"][data-trust="hatched"]'), 8000, "hatched trust line");
    expect(region.textContent).toContain(PREVIEW_TRUST_LINE.hatched);
    expect(container.querySelectorAll('[aria-label="Preview"]').length).toBe(1);
  }, 30000);
});
