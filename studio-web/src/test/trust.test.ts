// The run's trust grade (src/content/trust.ts): R² bands, the ±5-point coverage check, and
// the plain one-line explanation.
import { describe, expect, it } from "vitest";
import { r2Grade, TRUST_RULE, trustGrade } from "../content/trust";

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
    expect(good.line).toContain("likely ranges hold");
    const fair = trustGrade({ r2: 0.81, coverage: 0.7, coverageTarget: 0.9 })!;
    expect(fair.label).toBe("Fair");
    expect(fair.line).toContain("too narrow");
    expect(trustGrade({ r2: 0.1 })!.line).toContain("rough guide");
    for (const a of [good, fair, trustGrade({ r2: 0.45 })!]) {
      expect(a.line).not.toMatch(/R²|RMSE|coverage|\d/);
    }
  });
});
