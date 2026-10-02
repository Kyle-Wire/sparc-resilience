// Plain-language wording (SPEC §7.7): "Confident it cools" iff hi < 0, "Could be zero" when the
// 95% interval spans 0, "Confident it warms" iff lo > 0, and the extrapolation qualifier only
// above 20% of the edited cells.
import { describe, expect, it } from "vitest";
import type { Result } from "../../../api/lab";
import type { Likely } from "../../../api/types";
import { CONFIDENCE_TEXT, confidenceOf } from "../../../components/ui/PlainResult";
import { buysLines, cityLine, confidenceWord, extrapolationQualifier, headline, interval95, plainWording, withRange } from "../model/plain";

const L = (estimate: number, lo: number | null, hi: number | null, se: number | null = null): Likely => ({ estimate, lo, hi, se, confidence: "unknown", phrase: "" });

describe("confidence wording", () => {
  it("is 'Confident it cools' exactly when hi < 0", () => {
    expect(confidenceWord(L(-0.41, -0.52, -0.3))).toBe("Confident it cools");
    expect(confidenceWord(L(-0.41, -0.52, -1e-9))).toBe("Confident it cools");
    expect(confidenceWord(L(-0.41, -0.52, 0))).toBe("Could be zero"); // hi = 0 is not < 0
    expect(confidenceWord(L(-0.05, -0.2, 0.1))).toBe("Could be zero");
    expect(confidenceWord(L(0.3, 0.1, 0.5))).toBe("Confident it warms");
    expect(confidenceWord(L(0.3, 0, 0.5))).toBe("Could be zero");
    expect(confidenceWord(L(0.3, null, null))).toBe("No uncertainty estimate");
  });

  it("the inspector card words every SE-only result exactly as confidenceWord does", () => {
    // The card (foundation PlainResult) reads lo/hi; withRange fills them from ±1.96·SE.
    for (const est of [-0.5, -0.2, -0.05, 0, 0.05, 0.3])
      for (const se of [0.01, 0.1, 0.3]) {
        const l = L(est, null, null, se);
        expect(CONFIDENCE_TEXT[confidenceOf(withRange(l))]).toBe(`${confidenceWord(l)}.`);
      }
    const given = L(-0.4, -0.5, -0.3, 0.05);
    expect(withRange(given)).toBe(given); // an explicit range is kept as sent
    expect(withRange(L(-0.4, null, null, null))).toEqual(L(-0.4, null, null, null));
  });

  it("falls back to est ± 1.96·SE when lo/hi are missing", () => {
    expect(interval95(L(-0.5, null, null, 0.1))).toEqual([-0.5 - 0.196, -0.5 + 0.196]);
    expect(confidenceWord(L(-0.5, null, null, 0.1))).toBe("Confident it cools");
    expect(confidenceWord(L(-0.1, null, null, 0.1))).toBe("Could be zero");
  });

  it("for every interval, cools iff hi < 0 and could-be-zero iff it spans 0", () => {
    const vals = [-2, -1, -0.5, -0.01, 0, 0.01, 0.5, 1, 2];
    for (const lo of vals)
      for (const hi of vals) {
        if (hi < lo) continue;
        const w = confidenceWord(L((lo + hi) / 2, lo, hi));
        expect(w === "Confident it cools").toBe(hi < 0);
        expect(w === "Could be zero").toBe(lo <= 0 && hi >= 0);
        expect(w === "Confident it warms").toBe(lo > 0);
      }
  });
});

describe("qualifiers and headline", () => {
  it("adds the extrapolation qualifier only above 20% of the edited cells", () => {
    expect(extrapolationQualifier(0.2)).toBeNull();
    expect(extrapolationQualifier(0.19)).toBeNull();
    expect(extrapolationQualifier(null)).toBeNull();
    expect(extrapolationQualifier(0.23)).toBe("partly outside observed conditions (23% of edited cells)");
    const w = plainWording({ edited: L(-0.62, -0.8, -0.44), city: L(-0.021, -0.03, -0.012), unit: "degF", fracExtrapolatedEdited: 0.23 });
    expect(w.qualifiers).toEqual(["partly outside observed conditions (23% of edited cells)"]);
    expect(plainWording({ edited: L(-0.62, -0.8, -0.44), city: L(-0.021, -0.03, -0.012), unit: "degF", fracExtrapolatedEdited: 0.2 }).qualifiers).toEqual([]);
  });

  it("writes the headline in 'by how much' terms with the likely range", () => {
    const w = plainWording({ edited: L(-0.62, -0.8, -0.44), city: L(-0.021, -0.03, -0.012), unit: "degF", fracExtrapolatedEdited: 0.05, causalDisagrees: true, previewOnly: true });
    expect(w.headline).toBe("Cools the edited area by 0.62 °F (likely range 0.44–0.80 °F).");
    expect(w.city).toBe("City-wide: 0.021 °F cooler.");
    expect(w.confidence).toBe("Confident it cools");
    expect(w.qualifiers).toEqual(["independent causal check disagrees", "preview only — not verified"]);
    expect(headline(L(0.3, 0.1, 0.5), "degF", "the city")).toBe("Warms the city by 0.30 °F (likely range 0.10–0.50 °F).");
    expect(cityLine(L(0.004, null, null), "degF")).toBe("City-wide: 0.004 °F warmer.");
    const cityOnly = plainWording({ edited: null, city: L(-0.1, -0.2, 0.05), unit: "degF", fracExtrapolatedEdited: null });
    expect(cityOnly.headline).toMatch(/^Cools the city by 0\.10 °F/);
    expect(cityOnly.confidence).toBe("Could be zero");
    expect(cityOnly.city).toBe("");
  });

  it("states what the scenario buys", () => {
    const r = {
      cost: { total: 5000, per_lever: {}, cooling_per_cost: -0.0004 },
      impacts: {
        thresholds: [90],
        exposure: [
          { case: "today", adapted: false, person_mean_temp: 88, people_ge: { "90": 1000 }, share_people_ge: { "90": 0.1 } },
          { case: "today", adapted: true, person_mean_temp: 87.8, people_ge: { "90": 900 }, share_people_ge: { "90": 0.09 } },
          { case: "ssp245 2041-2060", adapted: false, person_mean_temp: 90, people_ge: { "90": 5000 }, share_people_ge: { "90": 0.5 } },
          { case: "ssp245 2041-2060", adapted: true, person_mean_temp: 89.8, people_ge: { "90": 4600 }, share_people_ge: { "90": 0.46 } },
        ],
        equity: {},
        hot_days: null,
        hot_days_action: null,
        zones: [],
        hexes: { "250": [], "500": [] },
        climate_offset: [{ experiment: "ssp245", period: "2041-2060", offset_share: 0.41 }],
      },
    } as Pick<Result, "cost" | "impacts">;
    expect(buysLines(r, "degF")).toEqual([
      "100 residents moved below 90 °F today, 400 by mid-century (SSP2-4.5).",
      "Offsets 41% of SSP245 2041-2060 median warming.",
      "0.400 °F·cells of cooling per 1,000 cost units.",
    ]);
  });
});
