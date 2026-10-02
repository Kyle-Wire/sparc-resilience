// Client-side climate maps on the 3×3 test lattice (7 observed cells, two empty): future
// temperature = observed + the selected warming statistic + the adaptation's ΔT, and ≥ T.
import { describe, expect, it } from "vitest";
import type { WarmingRow } from "../../../api/lab";
import { colorize, mapPalette } from "../../../map/colour";
import { computeDomain } from "../../../map/domain";
import { grid3 } from "../../../test/grid";
import { decodeStatistic, encodeStatistic, exceedance, futureTemperature, shareAtOrAbove, warmingValue } from "../model/climate";
import { syntheticMeta } from "../model/layers";

const OBS = Float32Array.from([86, 88, 89.5, 90, 91, NaN, 87]);
const ADAPT = Float32Array.from([-0.5, 0, -1.2, -0.3, NaN, -0.4, 0]);
const ROW: WarmingRow = {
  experiment: "ssp245",
  label: "SSP2-4.5",
  period: "2041-2060",
  n_models: 3,
  median: 3.6,
  p10: 2.7,
  p90: 4.5,
  min: 2.5,
  max: 4.9,
  by_model: { "ACCESS-CM2": 4.1, "MIROC6": 2.9, "NorESM2-LM": 3.6 },
};

describe("future temperature (client side)", () => {
  it("is obs + the selected statistic + the adaptation Δ for each of the 3×3 cells", () => {
    const g = grid3();
    expect(g.n).toBe(OBS.length);
    for (const stat of ["median", "p10", "p90", { model: "ACCESS-CM2" }] as const) {
      const w = warmingValue(ROW, stat)!;
      const fut = futureTemperature(OBS, w, ADAPT);
      for (let i = 0; i < g.n; i++) {
        const d = Number.isFinite(ADAPT[i]) ? ADAPT[i] : 0;
        if (!Number.isFinite(OBS[i])) expect(Number.isNaN(fut[i])).toBe(true);
        else expect(fut[i]).toBeCloseTo(OBS[i] + w + d, 4);
      }
    }
    expect(warmingValue(ROW, "median")).toBe(3.6);
    expect(warmingValue(ROW, "p10")).toBe(2.7);
    expect(warmingValue(ROW, "p90")).toBe(4.5);
    expect(warmingValue(ROW, { model: "MIROC6" })).toBe(2.9);
    expect(warmingValue(ROW, { model: "unknown" })).toBeNull();
    // exact values for the median, cell by cell
    expect(Array.from(futureTemperature(OBS, 3.6, ADAPT)).map((v) => (Number.isNaN(v) ? null : Number(v.toFixed(2))))).toEqual([89.1, 91.6, 91.9, 93.3, 94.6, null, 90.6]);
    // no adaptation: obs + warming
    expect(Array.from(futureTemperature(OBS, 3.6, null)).slice(0, 2).map((v) => Number(v.toFixed(2)))).toEqual([89.6, 91.6]);
  });

  it("maps ≥ T exceedance and its share, and paints both on the grid", () => {
    const g = grid3();
    const fut = futureTemperature(OBS, 3.6, ADAPT);
    const ex = exceedance(fut, 92);
    expect(Array.from(ex)).toEqual([0, 0, 0, 1, 1, 255, 0]);
    expect(shareAtOrAbove(fut, 92)).toBeCloseTo(2 / 6);
    const meta = syntheticMeta({ key: "climate:future", label: "Future", unit: "degF", scale: "div", center: 89.5 }, fut);
    const dom = computeDomain(meta, fut);
    expect(dom.kind).toBe("div");
    const out = new Uint32Array(g.nx * g.ny);
    colorize(out, g.rowToPix, fut, dom, mapPalette(false));
    // the missing observation (row 5) and the two empty lattice cells stay transparent
    let painted = 0;
    for (const px of out) if (px !== 0) painted++;
    expect(painted).toBe(6);
  });

  it("encodes the statistic for the URL", () => {
    for (const s of ["median", "p10", "p90", { model: "MIROC6" }] as const) expect(decodeStatistic(encodeStatistic(s))).toEqual(s);
    expect(decodeStatistic(null)).toBe("median");
    expect(decodeStatistic("model:")).toBe("median");
  });
});
