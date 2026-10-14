// Plain words of the Design flow (phase 3): lever names by role, the engine line, and exactAdvice()
// reasons rewritten for a planner.
import { describe, expect, it } from "vitest";
import { LEVERS, lever } from "../__fixtures__/lab";
import { exactAdvice } from "../components/CompilePanel";
import { CONFIDENCE_WORDS, LAB_WORDS, MODE_WORDS, PLANNER_MODES, emptyChangesText, engineLine, futureCaseLabel, futureLabel, humiditySourceWord, leverWords, modeFits, plainAdvice } from "../model/words";

describe("Lab words", () => {
  it("names levers by role, falling back to the catalogue label in title case", () => {
    expect(leverWords(lever("Pct_Canopy")).name).toBe("Trees & canopy");
    expect(leverWords(lever("Albedo")).name).toBe("Cool roofs & surfaces");
    expect(leverWords(lever("Pct_Impervious", { role: "impervious", label: "Impervious" })).name).toBe("Less paving");
    expect(leverWords(lever("water_frac", { role: null, label: "water fraction" })).name).toBe("Water Fraction");
  });

  it("offers three planner modes; planting space only for canopy with the planner layers", () => {
    expect(PLANNER_MODES.map((m) => MODE_WORDS[m].label)).toEqual(["Add", "Bring up to at least", "Use available planting space"]);
    expect(modeFits("fill_headroom", LEVERS[0])).toBe(true);
    expect(modeFits("fill_headroom", LEVERS[1])).toBe(false);
  });

  it("words the engine state as one line", () => {
    expect(engineLine("ready", null)).toBe("Model ready");
    expect(engineLine("loading", 0.4)).toBe("Loading model… 40%");
    expect(engineLine("no_checkpoint", null)).toBe("Model not available for this run — quick estimate only");
  });

  it("never words cooling as a warning", () => {
    expect(CONFIDENCE_WORDS.confident_cools).toMatchObject({ label: "Confident it cools", tone: "cool" });
    expect(CONFIDENCE_WORDS.could_be_zero.label).toBe("Might be no real effect");
  });

  it("rewrites exactAdvice reasons in plain words", () => {
    const compile = {
      content_hash: "h",
      portable: true,
      levers: { Albedo: { n_cells: 3, mean_requested: 0.1, total_requested: 0.3, predicted_mean_realised: 0.05, clipped_share: 0.5, est_cost: 1 } },
      union_cells: 3,
      people: null,
      warnings: [],
      est_exact_s: 2,
      emulator: { usable: false, hatched: true, reasons: ["Preview unreliable: Albedo has a rough emulator and edits 1,200 cells (> 500) — run exact", "Preview unreliable at this scale (uniform Albedo rel. error 246%) — run exact"] },
    };
    const advice = exactAdvice(compile, null, LEVERS);
    expect(plainAdvice(advice.reasons, LEVERS)).toEqual([
      "There is no quick estimate for this kind of change.",
      "The quick estimate is rough for cool roofs & surfaces over this many places (1,200).",
      "Across the whole city the quick estimate for cool roofs & surfaces can be off by about 246%.",
      "The quick estimate is rough for cool roofs & surfaces.",
      "More than a fifth of what you asked for does not fit the space available, and the quick estimate does not account for that.",
    ]);
    expect(plainAdvice(["Preview unreliable at this scale (12,000 edited cells > 5,000) — run exact", "the emulator has no Pct_Canopy lever", "edits extrapolate past the data"], LEVERS)).toEqual([
      "The change covers a large area (12,000 places), where the quick estimate is less reliable.",
      "There is no quick estimate for trees & canopy.",
      "Changes go beyond measured conditions past the data.",
    ]);
  });

  it("names future climates in plain words, keeping the pathway code out", () => {
    expect(futureLabel("ssp245", "2041-2060")).toBe("Mid-century, moderate emissions (2041–2060)");
    expect(futureLabel("ssp585", "2081-2100")).toBe("End of century, very high emissions (2081–2100)");
    expect(futureLabel("ssp126", "2021-2040")).toBe("Near-term, low emissions (2021–2040)");
    expect(futureCaseLabel("ssp245 2041-2060")).toBe("Mid-century, moderate emissions (2041–2060)");
    expect(futureCaseLabel("SSP2-4.5 2041-2060")).toBe("Mid-century, moderate emissions (2041–2060)");
    expect(futureCaseLabel("today")).toBeNull();
    expect(humiditySourceWord("campaign")).toBe("in the field survey");
    expect(humiditySourceWord("station")).toBe("at the weather station");
    expect(humiditySourceWord(null)).toBe("");
  });

  it("never promises a quick estimate the run does not have", () => {
    expect(emptyChangesText(false, "ready")).toBe(LAB_WORDS.emptyChanges);
    expect(emptyChangesText(true, "no_checkpoint")).toBe("First build the quick estimate (right), then pick what to change.");
    expect(emptyChangesText(true, "incompatible")).toBe(LAB_WORDS.emptyChangesBuildFirst);
    expect(emptyChangesText(true, "ready")).not.toContain("updates as you go");
  });
});
