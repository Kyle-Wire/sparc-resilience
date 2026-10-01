import { describe, expect, it } from "vitest";
import { bandScale, durationTicks, extent, formatDurationTick, formatTick, linearScale, logScale, logTicks, niceDomain, niceTicks, padDomain, stepDecimals, tickStep } from "../charts/scales";

describe("tick steps", () => {
  it("chooses 1/2/5 × 10^k", () => {
    expect(tickStep(0, 10, 5)).toBe(2);
    expect(tickStep(0, 1, 5)).toBe(0.2);
    expect(tickStep(0, 100, 10)).toBe(10);
    expect(tickStep(0, 7, 5)).toBe(1);
    expect(tickStep(-0.37, 0.52, 5)).toBe(0.2);
    expect(tickStep(0, 2.5e6, 5)).toBe(500000);
  });
  it("knows how many decimals a step needs", () => {
    expect(stepDecimals(0.2)).toBe(1);
    expect(stepDecimals(0.05)).toBe(2);
    expect(stepDecimals(5)).toBe(0);
  });
});

describe("nice ticks", () => {
  it("covers the domain with round values", () => {
    expect(niceTicks(0, 10, 5)).toEqual([0, 2, 4, 6, 8, 10]);
    expect(niceTicks(0.13, 0.91, 4)).toEqual([0.2, 0.4, 0.6, 0.8]);
    expect(niceTicks(-0.37, 0.52, 5)).toEqual([-0.2, 0, 0.2, 0.4]);
    expect(niceTicks(85.2, 93.7, 5)).toEqual([86, 88, 90, 92]);
    expect(niceTicks(10, 0, 5)).toEqual([10, 8, 6, 4, 2, 0]);
    expect(niceTicks(3, 3)).toEqual([3]);
  });
  it("has no floating-point noise", () => {
    for (const t of niceTicks(0, 0.7, 7)) expect(String(t).length).toBeLessThan(5);
  });
  it("extends domains to ticks", () => {
    expect(niceDomain(0.13, 0.91, 4)).toEqual([0, 1]);
    expect(niceDomain(-0.37, 0.52, 5)).toEqual([-0.4, 0.6]);
    expect(niceDomain(5, 5)).toEqual([4.5, 5.5]);
  });
  it("log ticks list decades (and 2×, 5× when narrow)", () => {
    expect(logTicks(100, 10000)).toEqual([100, 200, 500, 1000, 2000, 5000, 10000]);
    expect(logTicks(10, 100000)).toEqual([10, 100, 1000, 10000, 100000]);
  });
  it("formats ticks with the step's decimals and U+2212", () => {
    expect(formatTick(-0.2, 0.2)).toBe("−0.2");
    expect(formatTick(1000, 500)).toBe("1,000");
    expect(formatTick(0.4, 0.2, true)).toBe("+0.4");
  });
});

describe("scales", () => {
  it("maps and inverts linearly", () => {
    const s = linearScale([0, 10], [100, 200]);
    expect(s(5)).toBe(150);
    expect(s.invert(175)).toBe(7.5);
    const y = linearScale([0, 1], [300, 0]);
    expect(y(0.25)).toBe(225);
    expect(y.ticks(5)).toEqual([0, 0.2, 0.4, 0.6, 0.8, 1]);
  });
  it("maps on a log axis", () => {
    const s = logScale([100, 10000], [0, 200]);
    expect(s(1000)).toBeCloseTo(100, 9);
    expect(s.invert(100)).toBeCloseTo(1000, 6);
    expect(s.ticks()).toEqual([100, 200, 500, 1000, 2000, 5000, 10000]);
  });
  it("places bands", () => {
    const b = bandScale(["a", "b", "c"], [0, 300], 0.2, 0);
    expect(b.step).toBeCloseTo(300 / 2.8, 9);
    expect(b("a")).toBe(0);
    expect(b.bandwidth).toBeCloseTo(b.step * 0.8, 9);
    expect(b.at(b("b") + 1)).toBe("b");
  });
  it("computes extents and padding", () => {
    expect(extent([3, null, -1, NaN, 7])).toEqual([-1, 7]);
    expect(extent([null])).toBeNull();
    expect(padDomain([0, 10], 0.1, [0])).toEqual([0, 11]);
  });
  it("ticks elapsed time", () => {
    const { ticks, step } = durationTicks(0, 3600, 6);
    expect(step).toBe(600);
    expect(ticks).toEqual([0, 600, 1200, 1800, 2400, 3000, 3600]);
    expect(durationTicks(0, 100, 5).step).toBe(30);
    expect(formatDurationTick(1800, 900)).toBe("30 min");
    expect(formatDurationTick(5400, 3600)).toBe("1 h 30");
  });
});
