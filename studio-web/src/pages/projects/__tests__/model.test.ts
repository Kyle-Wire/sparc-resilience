// Pure logic of the projects pages: scenario names (U+2212) and slugs, dose scale, the launch
// stage rules, plan reasons, YAML path → line, suggestions, dotted paths, EPSG search and the
// wizard's completion dots.
import { describe, expect, it } from "vitest";
import { blockingPreflight, isNumericDtype } from "../../../api/projects";
import { changedSections, getPath, jsonEqual, setPath } from "../model/dotted";
import { doseScaleRows, dosesBeyondSd } from "../model/doseScale";
import { normaliseEpsg, searchEpsg, utmForLonLat } from "../model/epsg";
import { planNodeViews, planReasonText, unitSummary } from "../model/plan";
import { configuredScenarioNames, ladderNames, pyG, scenarioSlug, scenarioSlugs } from "../model/scenarioNames";
import { checklistRows, parseSelection, requestStages, toggleStage } from "../model/stages";
import { stepDot, stepIssues } from "../model/steps";
import { applySuggestion } from "../model/suggest";
import { locatePath, yamlPathIndex } from "../model/yamlPaths";
import { CHECK, DEMO_YAML, demoRaw, ISSUES, PREFLIGHT_ERROR, planNodes, projectDetail, runPlan, SUGGESTION } from "../__fixtures__/api";

const MINUS = "−";

describe("scenario names", () => {
  it("formats increments like Python's {:g}", () => {
    expect([5, 10, 0.05, 0.1, 0.0001, 0.00001, 123456, 1234567, 2.5e-5, 9.999999, 1.5].map(pyG)).toEqual([
      "5", "10", "0.05", "0.1", "0.0001", "1e-05", "123456", "1.23457e+06", "2.5e-05", "10", "1.5",
    ]);
  });

  it("ladder names use U+2212 for decrease levers and + for increase", () => {
    expect(ladderNames({ name: "Impervious Decrease", direction: "decrease", increments: [10, 20] })).toEqual([`Impervious Decrease ${MINUS}10`, `Impervious Decrease ${MINUS}20`]);
    expect(ladderNames({ name: "Albedo Increase", direction: "increase", increments: [0.05, 0.1] })).toEqual(["Albedo Increase +0.05", "Albedo Increase +0.1"]);
    for (const n of ladderNames({ name: "X", direction: "decrease", increments: [5] })) expect(n).not.toContain("-");
  });

  it("lists every configured scenario in core's order and slugs them like catalog.scenario_slug", () => {
    const names = configuredScenarioNames(demoRaw());
    expect(names).toEqual([
      "Canopy Increase +5", "Canopy Increase +10", "Canopy Increase +20",
      `Impervious Decrease ${MINUS}10`, `Impervious Decrease ${MINUS}20`,
      "Albedo Increase +0.05", "Albedo Increase +0.1", "Cooling package",
    ]);
    expect(scenarioSlug(`Impervious Decrease ${MINUS}10`)).toBe("impervious-decrease-minus-10");
    expect(scenarioSlug("Albedo Increase +0.1")).toBe("albedo-increase-plus-0-1");
    expect(scenarioSlug("Canopy Increase +10", ["canopy-increase-plus-10"])).toBe("canopy-increase-plus-10-2");
    expect(scenarioSlugs(["A b", "A-b"]).map((s) => s.slug)).toEqual(["a-b", "a-b-2"]);
  });
});

describe("dose scale", () => {
  it("flags doses beyond 1 sd", () => {
    const rows = doseScaleRows(CHECK.dose_scale.canopy);
    expect(rows.map((r) => r.beyondSd)).toEqual([false, false, true, true, true, true]);
    expect(rows[0]).toEqual({ dose: 5, inSd: 0.42, percentile: 63, beyondSd: false });
    expect(dosesBeyondSd(CHECK.dose_scale.impervious)).toEqual([]);
    expect(dosesBeyondSd(CHECK.dose_scale.canopy)).toEqual([15, 20, 30, 40]);
  });

  it("derives dose ÷ sd when the server sent null (sd 0 stays unknown)", () => {
    const rows = doseScaleRows({ sd: 2, doses: [1, 4], doses_in_sd: [null as unknown as number, 2], percentile_reached: [] });
    expect(rows.map((r) => r.inSd)).toEqual([0.5, 2]);
    expect(rows.map((r) => r.beyondSd)).toEqual([false, true]);
    expect(doseScaleRows({ sd: 0, doses: [3], doses_in_sd: [], percentile_reached: [] })[0].inSd).toBeNull();
  });
});

describe("launch stage rules", () => {
  it("S0 and S1 always run; S2–S3 is required by any later stage; S4 by S6 or S7", () => {
    const rows = checklistRows(["S0", "S1", "S6"]);
    const by = Object.fromEntries(rows.map((r) => [r.id, r]));
    expect(by.S0).toMatchObject({ checked: true, locked: true, reason: "always runs" });
    expect(by.S1).toMatchObject({ checked: true, locked: true });
    expect(by.S2_S3).toMatchObject({ checked: true, locked: true, reason: "required by S6" });
    expect(by.S4).toMatchObject({ checked: true, locked: true, reason: "required by S6" });
    expect(by.S5).toMatchObject({ checked: false, locked: false });
    expect(by.S7).toMatchObject({ checked: false, locked: false });
    expect(checklistRows(["S5"]).find((r) => r.id === "S2_S3")!.reason).toBe("required by S5");
    expect(checklistRows(["S7"]).find((r) => r.id === "S4")!.reason).toBe("required by S7");
  });

  it("locked rows cannot be toggled; the request carries the explicit choice only", () => {
    expect(toggleStage(["S0", "S1", "S6"], "S4", false)).toEqual(["S0", "S1", "S6"]);
    expect(toggleStage(["S0", "S1", "S2_S3"], "S2_S3", false)).toEqual(["S0", "S1"]);
    expect(toggleStage(["S0", "S1"], "S5", true)).toEqual(["S0", "S1", "S5"]);
    expect(requestStages(["S0", "S1", "S2_S3", "S6"])).toEqual(["S0", "S1", "S2", "S3", "S6"]);
    expect(requestStages(["S0", "S1", "S6"])).toEqual(["S0", "S1", "S6"]);
    expect(parseSelection(["S6", "bogus", "S0"])).toEqual(["S0", "S6"]);
  });
});

describe("plan view model", () => {
  it("phrases every reason of api.md §1", () => {
    expect(planReasonText("required_by:S6")).toBe("required by S6");
    expect(planReasonText("required_by:S2_S3")).toBe("required by S2–S3");
    expect(planReasonText("disabled_by_config:cv.baselines")).toBe("disabled in the config (cv.baselines)");
    expect(planReasonText("checkpoint")).toBe("from the checkpoint");
    expect(planReasonText("not_requested")).toBe("not requested");
    expect(planReasonText("requires_S5")).toBe("needs S5 (scenarios)");
    expect(planReasonText("no_budget")).toContain("no budget");
    expect(planReasonText(null)).toBeNull();
  });

  it("orders nodes, groups units and shows estimates only for nodes that run", () => {
    const views = planNodeViews([...planNodes()].reverse());
    expect(views.map((v) => v.id)).toEqual(["S0", "S1", "S2_S3", "baselines", "cv_curve", "S4", "S5", "climate", "S6", "S7", "finish"]);
    const s4 = views.find((v) => v.id === "S4")!;
    expect(s4.reason).toBe("required by S6");
    expect(s4.units).toBe("22 engine passes · 1 checkpoint save");
    expect(s4.estimate).toBe("≈4–6 min");
    expect(views.find((v) => v.id === "S1")!.estimate).toBeNull();
    expect(unitSummary({ "base_fit:mgwr": 3, "base_fit:gam": 3, "causal_step:dml": 1 })).toBe("6 base fits · 1 causal step");
  });

  it("a failed error-severity preflight blocks the launch; warnings do not", () => {
    expect(blockingPreflight(runPlan())).toEqual([]);
    expect(blockingPreflight(runPlan({ preflight: [PREFLIGHT_ERROR] }))).toHaveLength(1);
    expect(blockingPreflight(runPlan({ preflight: [{ ...PREFLIGHT_ERROR, severity: "warn" }] }))).toEqual([]);
  });
});

describe("YAML paths", () => {
  const index = yamlPathIndex(DEMO_YAML);
  it("indexes block maps, compact and flow sequences under core:", () => {
    expect(locatePath(index, "data.target")).toBe(5);
    expect(locatePath(index, "data.crs")).toBe(11);
    expect(locatePath(index, "predictors.1")).toBe(15);
    expect(locatePath(index, "actionable.canopy.doses")).toBe(20);
    expect(locatePath(index, "actionable.canopy.doses.2")).toBe(20);
    expect(locatePath(index, "stacker.tune_lambda.1")).toBe(25);
    expect(locatePath(index, "scenarios.0.variable")).toBe(28);
    expect(locatePath(index, "scenarios.0.increments.1")).toBe(31);
  });
  it("falls back to the nearest ancestor, or null", () => {
    expect(locatePath(index, "data.join.0.path")).toBe(3);
    expect(locatePath(index, "climate.table")).toBe(1);
    expect(locatePath(yamlPathIndex(""), "data")).toBeNull();
  });
  it("handles indented sequences, item maps, comments and block scalars", () => {
    const y = ["a:", "  list:", "    - x: 1 # note", "      y: [1, {k: 2}]", "    - x: 3", "  text: |", "    not: a key", "  after: 1"].join("\n");
    const ix = yamlPathIndex(y);
    expect(ix.get("a.list.0.x")).toBe(3);
    expect(ix.get("a.list.0.y.1.k")).toBe(4);
    expect(ix.get("a.list.1.x")).toBe(5);
    expect(ix.has("a.text.not")).toBe(false);
    expect(ix.get("a.after")).toBe(8);
  });
});

describe("suggestions", () => {
  it("pre-fills empty mapping fields, predictors and roles, and names what it filled", () => {
    const { raw, filled } = applySuggestion({ data: { path: "data/city.csv" } }, SUGGESTION);
    expect(getPath(raw, "data.target")).toBe("T");
    expect(getPath(raw, "data.x")).toBe("x");
    expect(getPath(raw, "data.zone")).toBe("district");
    expect(getPath(raw, "data.crs")).toBeUndefined();
    expect(getPath(raw, "predictors")).toEqual(SUGGESTION.predictors);
    expect(getPath(raw, "physics.roles.canopy")).toBe("canopy");
    expect(filled).toEqual(expect.arrayContaining(["data.target", "data.id", "data.x", "data.y", "data.zone", "data.coord_unit", "predictors", "physics.roles.canopy"]));
  });
  it("keeps values the user set unless overwriting", () => {
    const start = { data: { target: "AAT", x: "POINT_X" } };
    expect(getPath(applySuggestion(start, SUGGESTION).raw, "data.target")).toBe("AAT");
    expect(getPath(applySuggestion(start, SUGGESTION, { overwrite: true }).raw, "data.target")).toBe("T");
  });
});

describe("dotted paths", () => {
  it("sets immutably, creates containers, deletes on undefined", () => {
    const a = { data: { x: "x" }, list: [1, 2, 3] };
    const b = setPath(a, "data.y", "y");
    expect(a.data).toEqual({ x: "x" });
    expect(b).toEqual({ data: { x: "x", y: "y" }, list: [1, 2, 3] });
    expect(setPath(a, "list.1", undefined)).toEqual({ data: { x: "x" }, list: [1, 3] });
    expect(setPath({}, "data.join.0.path", "j.csv")).toEqual({ data: { join: [{ path: "j.csv" }] } });
    expect(setPath(a, ["qa", "clip", "a.b"], [0, 1])).toMatchObject({ qa: { clip: { "a.b": [0, 1] } } });
    expect(changedSections({ a: 1, b: { c: 1 } }, { a: 1, b: { c: 2 }, d: 0 })).toEqual(["b", "d"]);
    expect(jsonEqual({ a: 1, b: undefined }, { a: 1 })).toBe(true);
  });
  it("recognises numeric dtypes", () => {
    expect(["int64", "float64", "Int32", "uint8", "double"].every(isNumericDtype)).toBe(true);
    expect(["object", "string", "bool", "datetime64[ns]"].some(isNumericDtype)).toBe(false);
  });
});

describe("EPSG search", () => {
  it("finds codes and names, accepts typed codes and hints a UTM zone from a centroid", () => {
    expect(searchEpsg("3438")[0].code).toBe("EPSG:3438");
    expect(searchEpsg("utm 19N").some((e) => e.code === "EPSG:32619")).toBe(true);
    expect(searchEpsg("EPSG:99999")[0]).toMatchObject({ code: "EPSG:99999", name: "custom code" });
    expect(normaliseEpsg("epsg 2263")).toBe("EPSG:2263");
    expect(normaliseEpsg("UTM")).toBeNull();
    expect(utmForLonLat(-71.4, 41.8)).toBe("EPSG:32619");
    expect(utmForLonLat(151.2, -33.9)).toBe("EPSG:32756");
  });
});

describe("completion dots", () => {
  const readiness = projectDetail().readiness;
  it("come from the readiness spine where a row exists (worst row wins)", () => {
    expect(stepDot("data", readiness, demoRaw(), [])).toBe("ok");
    expect(stepDot("physics", readiness, demoRaw(), [])).toBe("warn");
    expect(stepDot("inputs", readiness, demoRaw(), [])).toBe("missing");
  });
  it("are derived from the draft for scenarios, analysis and about", () => {
    expect(stepDot("scenarios", readiness, demoRaw(), [])).toBe("ok");
    expect(stepDot("scenarios", readiness, demoRaw(), ISSUES)).toBe("missing");
    expect(stepDot("scenarios", readiness, { scenarios: [] }, [])).toBe("warn");
    expect(stepDot("analysis", readiness, demoRaw(), ISSUES)).toBe("missing");
    expect(stepDot("about", readiness, demoRaw(), [])).toBe("ok");
    expect(stepIssues("levers", ISSUES).map((i) => i.path)).toEqual(["actionable.canopy.doses.6"]);
  });
});
