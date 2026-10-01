// View-model formatting (SPEC §14.3 "run hub: view-model formatting"): units on every number,
// cooler/warmer wording, likely ranges, KPI tiles, producers, curve reconstruction, hex
// means, the box mask, conversions and the compare helpers.
import { describe, expect, it } from "vitest";
import { encodeSelection, decodeSelection } from "../../../stores/selection";
import { hexKey } from "../../../map/grid";
import { grid3 } from "../../../test/grid";
import { configTree, flattenScalars } from "../Compare";
import { conversionsFor } from "../Files";
import { residSelection } from "../Accuracy";
import { blpWords } from "../Causal";
import { internalPath } from "../Overview";
import { spillText } from "../Response";
import { envelopeCaption } from "../Uncertainty";
import {
  agreementWords,
  confidenceWords,
  curvePoints,
  fmtDistance,
  hexMeans,
  isTempUnit,
  kpiDisplay,
  likelyText,
  maskFromBox,
  missingText,
  outputStateStatus,
  producedByText,
  rangeWords,
  sigmoidWidth,
} from "../format";

const L = (estimate: number, lo: number | null, hi: number | null) => ({ estimate, lo, hi, se: null, confidence: "unknown" as const, phrase: "" });

describe("temperature wording", () => {
  it("words changes as cooler/warmer with a likely range", () => {
    expect(likelyText(L(-0.62, -0.94, -0.3), "°F")).toBe("0.62 °F cooler (likely range 0.30–0.94 °F cooler)");
    expect(likelyText(L(0.2, 0.1, 0.3), "degF")).toBe("0.20 °F warmer (likely range 0.10–0.30 °F warmer)");
    expect(likelyText(L(-0.05, -0.1, 0.05), "°F")).toBe("0.05 °F cooler (likely range 0.10 °F cooler to 0.05 °F warmer)");
    expect(likelyText(L(-0.05, null, null), "°F")).toBe("0.05 °F cooler");
    expect(likelyText(null, "°F")).toBe("—");
  });

  it("uses signed numbers with units for non-temperatures", () => {
    expect(isTempUnit("pp")).toBe(false);
    expect(likelyText(L(-2, -3, -1), "pp")).toBe("−2.00 pp (likely range −3.00 to −1.00 pp)");
    expect(rangeWords(-3, -1, "pp")).toBe("−3.00 to −1.00 pp");
  });

  it("states confidence from the range when the server does not", () => {
    expect(confidenceWords({ lo: -1, hi: -0.1, confidence: "unknown" })).toBe("Confident it cools.");
    expect(confidenceWords({ lo: -1, hi: 0.1, confidence: "unknown" })).toBe("Could be zero.");
    expect(confidenceWords({ lo: null, hi: null, confidence: "confident_warms" })).toBe("Confident it warms.");
  });
});

describe("KPI tiles", () => {
  const units = { target: "degF", levers: {} };
  it("formats deltas, percentages with targets, and plain numbers with units", () => {
    expect(kpiDisplay({ id: "h", label: "Headline", value: -1.0316, format: "delta", likely: L(-1.03, -1.58, -0.48) }, units)).toEqual({
      value: "1.03 °F cooler",
      unit: "",
      note: "likely range 0.48–1.58 °F cooler",
    });
    expect(kpiDisplay({ id: "c", label: "Coverage", value: 0.8902, format: "percent", decimals: 1, target: 0.9, target_label: "target" }, units)).toEqual({
      value: "89.0%",
      unit: "",
      note: "target 90.0%",
    });
    expect(kpiDisplay({ id: "r", label: "RMSE", value: 0.342, unit: "degF" }, units)).toEqual({ value: "0.34", unit: "°F", note: null });
    expect(kpiDisplay({ id: "w", label: "Warming", value: 3.213, unit: "degF", format: "signed" }, units).value).toBe("+3.21");
    expect(kpiDisplay({ id: "t", label: "Objective", value: "total cooling", format: "text" }, units).value).toBe("total cooling");
    expect(kpiDisplay({ id: "p", label: "Package", value: -1.3, format: "delta", band: { lo: -0.6, hi: -0.2, label: "causal band" } }, units).note).toBe(
      "causal band 0.20–0.60 °F cooler",
    );
  });
});

describe("producers and states", () => {
  it("names the producing stage, action or study", () => {
    expect(producedByText("stage:S6")).toBe("stage S6");
    expect(producedByText("post:planner")).toBe("the planner post-run action");
    expect(producedByText("study:placebo")).toBe("the placebo study");
    expect(missingText({ output: "causal", produced_by: "stage:S6", action: null })).toBe("causal — produced by stage S6");
    expect(outputStateStatus("writing")).toBe("running");
    expect(outputStateStatus("stale")).toBe("stale");
  });

  it("formats distances", () => {
    expect(fmtDistance(315)).toBe("315 m");
    expect(fmtDistance(1250)).toBe("1.3 km");
    expect(fmtDistance(null)).toBe("—");
  });
});

describe("cell curves", () => {
  it("keeps sampled curves and rebuilds a saturating curve from A and d_s", () => {
    const sampled = { model: "linear", A: null, ds: null, inflection: null, d90: null, dmax: 40, dose: [0, 10], benefit: [0, 0.3] };
    expect(curvePoints(sampled)).toEqual({ dose: [0, 10], benefit: [0, 0.3] });
    const sat = curvePoints({ model: "saturating", A: 2, ds: 10, inflection: null, d90: 23, dmax: 40, dose: [], benefit: [] }, 5);
    expect(sat.dose).toEqual([0, 10, 20, 30, 40]);
    expect(sat.benefit[0]).toBe(0);
    expect(sat.benefit[1]).toBeCloseTo(2 * (1 - Math.exp(-1)), 10);
    expect(sat.benefit[4]).toBeCloseTo(2 * (1 - Math.exp(-4)), 10);
    // a censored sigmoid (no d90) and a linear fit cannot be rebuilt from the parameters
    expect(curvePoints({ model: "sigmoid", A: 2, ds: null, inflection: 20, d90: null, dmax: 40, dose: [], benefit: [] })).toEqual({ dose: [], benefit: [] });
    expect(curvePoints({ model: "linear", A: null, ds: null, inflection: null, d90: null, dmax: 40, dose: [], benefit: [] })).toEqual({ dose: [], benefit: [] });
  });

  it("rebuilds a sigmoid from A, the inflection and d90 (core fit_saturation form)", () => {
    // core: B = A·[σ((D−D0)/w) − σ(−D0/w)] / [1 − σ(−D0/w)]; d90 = D0 + w·logit(0.9·(1−s0) + s0)
    const sig = (z: number) => 1 / (1 + Math.exp(-z));
    const [A, D0, w] = [1.5, 20, 4];
    const s0 = sig(-D0 / w);
    const p90 = 0.9 * (1 - s0) + s0;
    const d90 = D0 + w * Math.log(p90 / (1 - p90));
    expect(sigmoidWidth(D0, d90)).toBeCloseTo(w, 6);
    expect(sigmoidWidth(20, 15)).toBeNull(); // d90 before the inflection: no width fits
    const c = curvePoints({ model: "sigmoid", A, ds: null, inflection: D0, d90, dmax: 40, dose: [], benefit: [] }, 41);
    expect(c.dose[0]).toBe(0);
    expect(c.dose[40]).toBe(40);
    expect(c.benefit[0]).toBeCloseTo(0, 10);
    for (const i of [10, 20, 25, 40]) expect(c.benefit[i]).toBeCloseTo((A * (sig((c.dose[i] - D0) / w) - s0)) / (1 - s0), 6);
    expect(c.benefit[20]).toBeLessThan(0.5 * A + 1e-9); // the inflection sits half-way up (less the s0 offset)
  });
});

describe("hex mode and masks", () => {
  it("paints every cell with the mean of its hexagon", () => {
    const g = grid3();
    const v = Float32Array.from([1, 2, 3, 4, 5, 6, NaN]);
    const key = (r: number) => hexKey(g.meta.x0_m + g.ix[r] * g.meta.dx_m, g.meta.y0_m + g.iy[r] * g.meta.dx_m, 60);
    const out = hexMeans(g, v, 60);
    for (let r = 0; r < 7; r++) {
      const peers = [...Array(7).keys()].filter((q) => key(q) === key(r) && Number.isFinite(v[q]));
      const want = peers.length ? peers.reduce((a, q) => a + v[q], 0) / peers.length : NaN;
      if (Number.isNaN(want)) expect(Number.isNaN(out[r])).toBe(true);
      else expect(out[r]).toBeCloseTo(want, 5);
    }
    expect(new Set([...Array(7).keys()].map(key)).size).toBeGreaterThan(1); // several hexagons
    // A hexagon smaller than a cell keeps each cell's own value.
    const own = hexMeans(g, v, 10);
    expect([...own.slice(0, 6)]).toEqual([1, 2, 3, 4, 5, 6]);
    expect(Number.isNaN(own[6])).toBe(true);
  });

  it("selects rows inside a brushed box", () => {
    expect([...maskFromBox([1, 2, 3, NaN], [5, 6, 7, 8], { x: [3, 1.5], y: [5.5, 7] })]).toEqual([0, 1, 1, 0]);
  });

  it("turns the residual brush into a portable URL selection", () => {
    const spec = residSelection(0.5, 1.25);
    expect(decodeSelection(encodeSelection(spec))).toEqual({ kind: "filter", column: "pred:resid", op: "between", value: [0.5, 1.25] });
  });
});

describe("files and compare helpers", () => {
  it("offers the conversions api.md allows", () => {
    expect(conversionsFor("predictions.parquet", ["parquet", "csv", "geojson"])).toEqual(["csv", "geojson"]);
    expect(conversionsFor("baselines.json")).toEqual(["csv"]);
    expect(conversionsFor("methods.md", ["md", "html"])).toEqual(["html"]);
    expect(conversionsFor("checkpoint.pkl")).toEqual([]);
  });

  it("groups config differences into a tree by section", () => {
    const tree = configTree([
      { path: "stacker.epochs", a: 200, b: 400 },
      { path: "cv.n_folds", a: 3, b: 5 },
      { path: "stacker.tune_lambda", a: [0], b: [0, 1] },
      { path: "data.join[0].path", a: "a.csv", b: "b.csv" },
    ]);
    expect(tree.map((t) => [t.section, t.rows.length])).toEqual([
      ["cv", 1],
      ["data", 1],
      ["stacker", 2],
    ]);
  });

  it("flattens nested summaries to scalar leaves", () => {
    expect(flattenScalars({ warming: { ssp245: { median: 3.2 } }, n: 6, label: "x" })).toEqual([
      ["warming.ssp245.median", 3.2],
      ["n", 6],
      ["label", "x"],
    ]);
  });

  it("reads rank agreement in words", () => {
    expect(agreementWords(0.72)).toBe("strong agreement (τ +0.72)");
    expect(agreementWords(0.25)).toBe("weak agreement (τ +0.25)");
    expect(agreementWords(null)).toBe("—");
  });
});

describe("computed captions", () => {
  it("words the footprint / own-cell ratio from its value", () => {
    expect(spillText(4)).toBe("Footprint / own-cell ratio 4.0: 75% of the effect of a change lands in neighbouring cells, more than in the changed cell itself.");
    expect(spillText(1.25)).toBe("Footprint / own-cell ratio 1.3: 20% of the effect of a change lands in neighbouring cells.");
    expect(spillText(0.5)).toContain("offsetting part of the own-cell effect");
    expect(spillText(-2)).toContain("outweigh the own-cell effect");
    expect(spillText(null)).toBe("No footprint / own-cell ratio for this lever.");
  });

  it("counts cooling, warming and could-be-zero envelopes separately", () => {
    const row = (label: string, lo: number, hi: number) => ({ id: label, label, estimate: (lo + hi) / 2, layers: [{ id: "envelope", label: "Envelope", lo, hi }] });
    expect(envelopeCaption([row("A", -1, -0.1), row("B", 0.1, 0.4), row("C", -0.5, 0.2)])).toBe(
      "1 of 3 scenarios still cool under the widest (envelope) interval (A); 1 still warms (B); 1 could be zero.",
    );
    expect(envelopeCaption([row("A", -1, -0.1)])).toBe("1 of 1 scenarios still cool under the widest (envelope) interval (A).");
    expect(envelopeCaption([{ id: "x", label: "X", estimate: -1, layers: [{ id: "estimation", label: "Estimation", lo: -2, hi: -0.5 }] }])).toContain("no envelope interval yet");
  });

  it("reads the BLP calibration test from its p-value, not the slope alone", () => {
    expect(blpWords({ coef: 0.9, se: 0.2, p: 0.001 })).toBe("BLP calibration slope β₂ = 0.90 (SE 0.20, p = 0.001): the effect differs between cells as the CATE says.");
    expect(blpWords({ coef: 0.9, se: 0.8, p: 0.26 })).toContain("no evidence that the effect differs between cells");
    expect(blpWords({ coef: 0.3, se: 0.1, p: 0.01 })).toContain("the CATE exaggerates the spread");
    expect(blpWords({ coef: null, se: null, p: null })).toContain("not identified");
    expect(blpWords(null)).toBe("No BLP calibration test in this run.");
  });

  it("links a finding only to an app path", () => {
    expect(internalPath("/r/r1/map?layer=obs")).toBe("/r/r1/map?layer=obs");
    expect(internalPath("//evil.example/x")).toBeNull();
    expect(internalPath("/\\evil.example/x")).toBeNull(); // browsers read "/\" as "//"
    expect(internalPath("https://evil.example/")).toBeNull();
    expect(internalPath("javascript:alert(1)")).toBeNull();
    expect(internalPath("")).toBeNull();
  });
});
