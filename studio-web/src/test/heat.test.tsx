// The browser heat index (theme/heat.ts) against sparc.core.heat.heat_index_f, and the resident shift a
// design's per-cell change makes across the NWS categories.
import { describe, expect, it } from "vitest";
import { heatCategory, heatIndexAt, heatIndexF, heatShift, rhFromDewpoint } from "../theme/heat";

describe("NWS heat index", () => {
  it("matches the Python implementation", () => {
    const t = [90, 96, 80, 70, 100, 85, 82];
    const r = [70, 40, 50, 50, 10, 90, 95];
    const py = [105.922, 100.8861, 80.8029, 69.05, 94.1225, 101.7808, 93.9722];
    t.forEach((x, i) => expect(heatIndexF(x, r[i])).toBeCloseTo(py[i], 3));
    expect(rhFromDewpoint(30, 16.7)).toBeCloseTo(44.8036, 3);
    expect(rhFromDewpoint(35, 16.7)).toBeCloseTo(33.795, 3);
  });

  it("puts category edges where the NWS does", () => {
    expect([79.9, 80, 89.9, 90, 102.9, 103, 124.9, 125].map(heatCategory)).toEqual([0, 1, 1, 2, 2, 3, 3, 4]);
  });

  it("converts Celsius targets", () => {
    expect(heatIndexAt(35, "degC", 16.7)).toBeCloseTo(heatIndexAt(95, "degF", 16.7), 6);
  });

  it("counts residents a cooling moves out of Extreme caution", () => {
    const obs = Float32Array.from([88, 92, 94, 96]);
    const people = Float32Array.from([10, 20, 30, 40]);
    const none = heatShift(obs, new Float32Array(4), people, "degF", 16.7);
    expect(none.ecBefore).toBe(none.ecAfter);
    expect(none.meanHiChange).toBe(0);
    const cool = heatShift(obs, Float32Array.from([-3, -3, -3, -3]), people, "degF", 16.7);
    const before = [88, 92, 94, 96].map((x, i) => (heatIndexAt(x, "degF", 16.7) >= 90 ? [10, 20, 30, 40][i] : 0)).reduce((a, b) => a + b, 0);
    expect(cool.ecBefore).toBe(before);
    expect(cool.ecAfter).toBeLessThan(cool.ecBefore);
    expect(cool.meanHiChange).toBeLessThan(0);
    expect(cool.total).toBe(100);
    expect(heatShift(obs, new Float32Array(4), null, "degF", 16.7).measure).toBe("cells");
  });
});

describe("HeatPreviewCard", () => {
  it("says how many residents the draft moves out of Extreme caution, marked as a preview", async () => {
    const { render } = await import("./render");
    const { HeatPreviewCard } = await import("../pages/lab/components/CompilePanel");
    const shift = { ecBefore: 36669, ecAfter: 20000, dangerBefore: 0, dangerAfter: 0, meanHiChange: -0.8, total: 173660, measure: "people" as const };
    const { container } = render(<HeatPreviewCard heat={{ shift, dewpointC: 16.7, source: "station", caveats: ["Canopy: unproven"] }} />);
    const text = container.textContent ?? "";
    expect(text).toContain("Moves about 16,669 residents out of Extreme caution or worse.");
    expect(text).toContain("36,669 → 20,000");
    expect(text).toContain("preview");
    expect(text).toContain("−0.80 °F");
    expect(container.querySelector('[aria-label="Evidence caveats"]')!.textContent).toContain("Canopy: unproven");
  });

  it("names a lever whose configured scenarios are not established", async () => {
    const { leverCaveats } = await import("../pages/lab/model/heatPreview");
    const v = (verdict: "robust" | "direction" | "not_established") => ({ verdict, label: verdict, reasons: [], qualifiers: [] });
    const row = (lever: string, verdict: "robust" | "direction" | "not_established") =>
      ({ kind: "single", lever, verdict: v(verdict) }) as unknown as import("../api/runs").ScenarioRow;
    const rows = [row("canopy", "not_established"), row("canopy", "not_established"), row("imperv", "robust"), row("imperv", "direction"), row("albedo", "direction")];
    const out = leverCaveats(rows, ["canopy", "imperv", "albedo", "other"], (x) => x.toUpperCase());
    expect(out).toHaveLength(2);
    expect(out[0]).toMatch(/^CANOPY: no configured scenario/);
    expect(out[1]).toMatch(/^ALBEDO: .*Direction only/);
    expect(leverCaveats(null, ["canopy"], (x) => x)).toEqual([]);
  });
});
