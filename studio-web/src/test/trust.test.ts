// The run's trust grade (src/content/trust.ts): R² bands, the ±5-point coverage check, and
// the plain one-line explanation; the scenario verdicts in the glossary's plain words.
import { describe, expect, it } from "vitest";
import { GLOSSARY, term } from "../content/glossary";
import { r2Grade, TRUST_GRADE_LABEL, TRUST_RULE, trustGrade, VERDICT_TERMS, verdictWord } from "../content/trust";

describe("trustGrade", () => {
  it("is null while the held-out R² is unknown", () => {
    expect(trustGrade({ r2: null })).toBeNull();
    expect(trustGrade({ r2: undefined, coverage: 0.9 })).toBeNull();
    expect(trustGrade({ r2: Number.NaN })).toBeNull();
  });

  it("grades R² ≥ 0.6 Good, ≥ 0.4 Fair, else Limited", () => {
    expect(TRUST_RULE.good).toBe(0.6);
    expect(TRUST_RULE.fair).toBe(0.4);
    expect(r2Grade(0.81)).toBe("good");
    expect(r2Grade(0.6)).toBe("good");
    expect(r2Grade(0.5999)).toBe("fair");
    expect(r2Grade(0.4)).toBe("fair");
    expect(r2Grade(0.3999)).toBe("limited");
    expect(r2Grade(-0.2)).toBe("limited");
    expect(trustGrade({ r2: 0.81 })).toMatchObject({ grade: "good", label: "Good", r2Grade: "good", ranges: null });
    expect(trustGrade({ r2: 0.45 })).toMatchObject({ grade: "fair", label: "Fair" });
    expect(trustGrade({ r2: 0.1 })).toMatchObject({ grade: "limited", label: "Limited" });
  });

  it("keeps the grade when coverage is within 5 points of its target (inclusive)", () => {
    expect(trustGrade({ r2: 0.81, coverage: 0.89, coverageTarget: 0.9 })).toMatchObject({ grade: "good", ranges: "ok" });
    expect(trustGrade({ r2: 0.81, coverage: 0.85, coverageTarget: 0.9 })).toMatchObject({ grade: "good", ranges: "ok" });
    expect(trustGrade({ r2: 0.81, coverage: 0.95, coverageTarget: 0.9 })).toMatchObject({ grade: "good", ranges: "ok" });
    // the target defaults to 90%
    expect(trustGrade({ r2: 0.81, coverage: 0.86 })).toMatchObject({ grade: "good", ranges: "ok" });
    expect(trustGrade({ r2: 0.81, coverage: 0.86, coverageTarget: null })).toMatchObject({ grade: "good", ranges: "ok" });
  });

  it("drops one step when the ranges miss their target by more than 5 points", () => {
    expect(trustGrade({ r2: 0.81, coverage: 0.84, coverageTarget: 0.9 })).toMatchObject({ grade: "fair", r2Grade: "good", ranges: "narrow" });
    expect(trustGrade({ r2: 0.81, coverage: 0.99, coverageTarget: 0.9 })).toMatchObject({ grade: "fair", ranges: "wide" });
    expect(trustGrade({ r2: 0.5, coverage: 0.7, coverageTarget: 0.9 })).toMatchObject({ grade: "limited", r2Grade: "fair" });
    expect(trustGrade({ r2: 0.2, coverage: 0.5, coverageTarget: 0.9 })).toMatchObject({ grade: "limited", r2Grade: "limited" });
    // a different target
    expect(trustGrade({ r2: 0.7, coverage: 0.74, coverageTarget: 0.8 })).toMatchObject({ grade: "fair", ranges: "narrow" });
    expect(trustGrade({ r2: 0.7, coverage: 0.76, coverageTarget: 0.8 })).toMatchObject({ grade: "good", ranges: "ok" });
  });

  it("says it in one plain line without statistics, and explains a step down", () => {
    const good = trustGrade({ r2: 0.81, coverage: 0.89, coverageTarget: 0.9 })!;
    expect(good.line).toMatch(/^Predicts well/);
    expect(good.line).toContain("prediction ranges hold");
    const fair = trustGrade({ r2: 0.81, coverage: 0.7, coverageTarget: 0.9 })!;
    expect(fair.label).toBe("Fair");
    expect(fair.line).toContain("too narrow");
    expect(trustGrade({ r2: 0.1 })!.line).toContain("rough guide");
    for (const a of [good, fair, trustGrade({ r2: 0.45 })!]) {
      expect(a.line).not.toMatch(/R²|RMSE|coverage|\d/);
    }
  });
});

describe("the words come from the glossary", () => {
  it("grade labels and sentences", () => {
    expect(TRUST_GRADE_LABEL).toEqual({ good: GLOSSARY.trust.good.label, fair: GLOSSARY.trust.fair.label, limited: GLOSSARY.trust.limited.label });
    expect(trustGrade({ r2: 0.5, coverage: 0.99 })!.line).toBe(`${GLOSSARY.trust.fair.sentence} ${GLOSSARY.trustRanges.wide}`);
  });

  it("scenario verdicts in plain words, the server's label kept for the rest", () => {
    expect(verdictWord("robust")).toEqual({ label: "Solid", term: "verdict_robust" });
    expect(verdictWord("direction", "Direction only")).toEqual({ label: "Direction clear", term: "verdict_direction" });
    expect(verdictWord("not_established").label).toBe("Not proven");
    // the technical name stays in the tooltip
    expect(Object.values(VERDICT_TERMS).map((id) => term(id).technical)).toEqual(["Robust", "Direction only", "Not established"]);
    expect(verdictWord("unknown", "Unknown")).toEqual({ label: "Unknown", term: null });
    expect(verdictWord("toString")).toEqual({ label: "toString", term: null });
    expect(verdictWord("no_data")).toEqual({ label: "no data", term: null });
  });
});
