// The shared glossary (sparc/core/glossary.json via src/content/glossary.ts): helpers, the shape
// of every entry, and that the Lab's current words match the glossary verbatim (until those
// modules read from it directly, so the two never drift).
import { describe, expect, it } from "vitest";
import { GLOSSARY, confidence, heatCategory, isTermId, lever, mode, periodLabel, shape, ssp, sspLabel, term, trustWord, type TermId } from "../content/glossary";
import { TRUST_GRADE_LABEL, trustGrade } from "../content/trust";
import { HATCH_KEY } from "../pages/lab/model/quick";
import { FLOW_WORDS } from "../pages/lab/model/flow";
import { CONFIDENCE_WORDS, ENGINE_NO_CHECKPOINT, ENGINE_NO_MODEL_NO_ESTIMATE, ENGINE_WORDS, LEVER_WORDS, MODE_WORDS, PREVIEW_TRUST_LINE, TRUST_TIP, TRUST_WORDS, emissionsWord, engineLine } from "../pages/lab/model/words";

describe("glossary terms", () => {
  it("has a plain label and a short tip for every term", () => {
    const ids = Object.keys(GLOSSARY.terms);
    expect(ids.length).toBeGreaterThanOrEqual(35);
    for (const id of ids) {
      const t = term(id as TermId);
      expect(t.label.trim(), id).not.toBe("");
      expect(t.tip.trim(), id).not.toBe("");
      expect(t.tip.length, id).toBeLessThanOrEqual(240);
    }
  });

  it("gives the plain label with the technical name kept for the tooltip", () => {
    expect(term("coverage")).toMatchObject({ label: "Range reliability", technical: "90% interval coverage" });
    expect(term("r2").label).toBe("Accuracy score");
    expect(term("rmse").label).toBe("Typical error");
    expect(term("emulator_preview").label).toBe("Quick estimate");
    expect(term("run_exact").label).toBe("Run full model");
    expect(term("compile").label).toBe("Check the change");
    expect(term("interval_95").label).toBe("Likely range");
  });

  it("throws for an unknown term id in tests (a typo fails loudly)", () => {
    expect(() => term("not_a_term" as TermId)).toThrow(/unknown term "not_a_term"/);
    expect(isTermId("coverage")).toBe(true);
    expect(isTermId("not_a_term")).toBe(false);
    expect(isTermId("toString")).toBe(false); // own keys only
  });
});

describe("glossary sections", () => {
  it("levers by role (or a lever whose var names a role); null otherwise", () => {
    expect(lever("canopy")).toEqual({ name: "Trees & canopy", oneLine: LEVER_WORDS.canopy.oneLine, icon: "trees", unitWord: "pts", measure: "tree canopy" });
    expect(lever({ role: "IMPERVIOUS", var: "Pct_Impervious" })?.name).toBe("Less paving");
    expect(lever({ role: null, var: "albedo" })?.icon).toBe("roofs");
    expect(lever("Pct_Canopy")).toBeNull();
    expect(lever(undefined)).toBeNull();
  });

  it("modes, confidence, shapes, heat categories and trust grades", () => {
    expect(mode("add")).toMatchObject({ label: "Add", planner: true });
    expect(mode("fill_headroom").label).toBe("Use available planting space");
    expect(mode("take_away").label).toBe("Take away");
    expect(mode("nope")).toBeNull();
    expect(confidence("could_be_zero").chip).toBe("Might be no real effect");
    expect(confidence("nope")).toBeNull();
    expect([shape("saturating").label, shape("linear").label, shape("sigmoid").label, shape("censored").label]).toEqual(["Diminishing", "Steady", "Slow start, then strong", "Hit the limit"]);
    expect(heatCategory("extreme_caution").label).toBe("Extreme caution");
    expect(heatCategory(2)).toBe(heatCategory("extreme_caution"));
    expect(heatCategory(0).label).toBe("Below caution");
    expect(heatCategory(4).label).toBe("Extreme danger");
    expect(heatCategory(7)).toBeNull();
    expect(trustWord("fair").label).toBe("Fair");
  });

  it("future climates in plain words, the code kept as the technical name", () => {
    expect(sspLabel("ssp126")).toBe("low emissions");
    expect(sspLabel("SSP2-4.5")).toBe("moderate emissions");
    expect(sspLabel("ssp370")).toBe("high emissions");
    expect(sspLabel("SSP5-8.5")).toBe("very high emissions");
    expect(sspLabel("RCP 4.5")).toBe("moderate emissions");
    expect(sspLabel("mystery")).toBe("mystery");
    expect(ssp("ssp245")).toMatchObject({ technical: "SSP2-4.5" });
    expect(ssp("ssp245")?.tip).toMatch(/2\.7 °C \(likely 2\.1–3\.5 °C\)/);
    expect(periodLabel("2041-2060")).toBe("mid-century (2041–2060)");
    expect(periodLabel("2081–2100")).toBe("late century (2081–2100)");
    expect(periodLabel("2061-2080")).toBe("2061–2080");
    expect(periodLabel("today")).toBe("today");
  });
});

describe("the Lab's words match the glossary verbatim", () => {
  it("levers and modes", () => {
    for (const [role, w] of Object.entries(LEVER_WORDS)) expect(lever(role), role).toEqual(w);
    for (const [m, w] of Object.entries(MODE_WORDS)) expect(mode(m), m).toEqual({ label: w.label, help: w.hint, verb: w.verb, planner: w.planner });
    expect(mode("take_away")).toMatchObject({ label: FLOW_WORDS.takeAway, help: FLOW_WORDS.takeAwayHint });
    expect(mode("bring_down")).toMatchObject({ label: FLOW_WORDS.bringDown, help: FLOW_WORDS.bringDownHint });
  });

  it("confidence, quick-estimate trust, hatching and engine lines", () => {
    for (const [c, w] of Object.entries(CONFIDENCE_WORDS)) expect(confidence(c), c).toEqual({ chip: w.label, sentence: w.tip });
    for (const [k, w] of Object.entries(TRUST_WORDS)) expect(GLOSSARY.quickTrust.dot[k as keyof typeof TRUST_WORDS], k).toBe(w.label);
    expect(GLOSSARY.quickTrust.tip).toBe(TRUST_TIP);
    expect(GLOSSARY.quickTrust.line).toEqual(PREVIEW_TRUST_LINE);
    expect(GLOSSARY.hatch).toEqual(HATCH_KEY);
    expect(GLOSSARY.engine.states).toEqual(ENGINE_WORDS);
    expect(GLOSSARY.engine.noModelNoEstimate).toBe(ENGINE_NO_MODEL_NO_ESTIMATE);
    expect(GLOSSARY.engine.noCheckpoint).toBe(ENGINE_NO_CHECKPOINT);
    expect(GLOSSARY.engine.checking).toBe(engineLine(null, null));
  });

  it("emissions words and the run's trust grade", () => {
    for (const code of Object.keys(GLOSSARY.emissions)) expect(sspLabel(code), code).toBe(emissionsWord(code));
    expect(trustWord("good")).toEqual({ label: TRUST_GRADE_LABEL.good, sentence: trustGrade({ r2: 0.8 })!.line });
    expect(trustWord("fair")).toEqual({ label: TRUST_GRADE_LABEL.fair, sentence: trustGrade({ r2: 0.5 })!.line });
    expect(trustWord("limited")).toEqual({ label: TRUST_GRADE_LABEL.limited, sentence: trustGrade({ r2: 0.1 })!.line });
    expect(trustGrade({ r2: 0.8, coverage: 0.6 })!.line).toBe(`${GLOSSARY.trust.good.sentence} ${GLOSSARY.trustRanges.narrow}`);
  });
});
