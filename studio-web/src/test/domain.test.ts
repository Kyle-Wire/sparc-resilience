import { describe, expect, it } from "vitest";
import type { LayerMeta } from "../api/types";
import { computeDomain, layerSummary, quantiles, tToValue, valueToT } from "../map/domain";

function meta(p: Partial<LayerMeta>): LayerMeta {
  return {
    key: "x", group: "g", label: "X", unit: "degF", scale: "seq", center: null, decimals: 2, mult: 1, zero_blank: false, labels: null,
    desc: "", sign_note: null, source: null, dtype: "float32",
    stats: { n: 100, lo: 0, hi: 100, mean: 50, p1: 1, p2: 2, p50: 50, p98: 98, p99: 99 },
    ...p,
  };
}

describe("domain rules (SPEC §6.3)", () => {
  it("clips to the 2nd–98th percentiles from the layer stats", () => {
    const d = computeDomain(meta({}));
    expect([d.lo, d.hi]).toEqual([2, 98]);
  });
  it("computes the 2–98% clip from values when stats lack percentiles", () => {
    const vals = Float32Array.from({ length: 101 }, (_, i) => i); // 0..100
    const d = computeDomain(meta({ stats: { n: 101, lo: 0, hi: 100, mean: 50, p1: null, p2: null, p50: null, p98: null, p99: null } }), vals);
    expect(d.lo).toBeCloseTo(2, 9);
    expect(d.hi).toBeCloseTo(98, 9);
  });
  it("makes diverging domains symmetric about the centre", () => {
    const temp = computeDomain(meta({ scale: "div", center: 88, stats: { n: 9, lo: 80, hi: 99, mean: 88, p1: 81, p2: 84, p50: 88, p98: 95, p99: 96 } }));
    expect(temp.center).toBe(88);
    expect([temp.lo, temp.hi]).toEqual([81, 95]); // span = max(|84−88|, |95−88|) = 7
    const delta = computeDomain(meta({ scale: "div", center: 0, stats: { n: 9, lo: -2, hi: 0.5, mean: -0.3, p1: -1.9, p2: -1.5, p50: -0.2, p98: 0.3, p99: 0.4 } }));
    expect([delta.lo, delta.center, delta.hi]).toEqual([-1.5, 0, 1.5]);
    expect(valueToT(0, delta)).toBe(0.5);
    expect(valueToT(-1.5, delta)).toBe(0);
    expect(valueToT(0.75, delta)).toBeCloseTo(0.75, 12);
  });
  it("zero_blank starts at 0 and clips on the painted (positive) cells", () => {
    const vals = new Float32Array(1000); // 990 untreated zeros…
    for (let i = 0; i < 10; i++) vals[990 + i] = i + 1; // …and doses 1..10
    const d = computeDomain(meta({ zero_blank: true, stats: { n: 1000, lo: 0, hi: 10, mean: 0.055, p1: 0, p2: 0, p50: 0, p98: 0, p99: 1 } }), vals);
    expect(d.zeroBlank).toBe(true);
    expect(d.lo).toBe(0);
    expect(d.hi).toBeCloseTo(quantiles(Array.from({ length: 10 }, (_, i) => i + 1), [0.98])[0]!, 6);
    expect(d.hi).toBeGreaterThan(9);
  });
  it("applies mult as display scaling", () => {
    const d = computeDomain(meta({ scale: "div", center: 0, mult: 0.01, stats: { n: 9, lo: -50, hi: 10, mean: -5, p1: -45, p2: -40, p50: -3, p98: 8, p99: 9 } }));
    expect(d.mult).toBe(0.01);
    expect(d.lo).toBeCloseTo(-0.4, 12);
    expect(d.hi).toBeCloseTo(0.4, 12);
    const s = layerSummary(Float32Array.from([-40, 0, 8]), d);
    expect(s.mean).toBeCloseTo(-0.32 / 3 * 1, 6);
  });
  it("honours a manual range and a locked scale", () => {
    const m = computeDomain(meta({}), null, { manual: [10, 20] });
    expect([m.lo, m.hi]).toEqual([10, 20]);
    const locked = computeDomain(meta({}));
    const other = computeDomain(meta({ stats: { n: 1, lo: 0, hi: 1, mean: 0, p1: 0, p2: 0.1, p50: 0.5, p98: 0.9, p99: 1 } }), null, { lock: locked });
    expect([other.lo, other.hi]).toEqual([2, 98]);
  });
  it("categorical layers span their classes", () => {
    const d = computeDomain(meta({ scale: "cat", labels: ["insufficient", "saturating", "linear", "S-shaped"], dtype: "uint8" }));
    expect(d.kind).toBe("cat");
    expect(d.nCat).toBe(4);
  });
  it("tToValue inverts valueToT", () => {
    const d = computeDomain(meta({ scale: "div", center: 88, stats: { n: 9, lo: 80, hi: 99, mean: 88, p1: 81, p2: 84, p50: 88, p98: 95, p99: 96 } }));
    for (const v of [81, 85, 88, 90.5, 95]) expect(tToValue(valueToT(v, d), d)).toBeCloseTo(v, 9);
  });
});
