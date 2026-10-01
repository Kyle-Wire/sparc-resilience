// Every kit chart with missing values (null/NaN), empty data and degenerate domains: it must
// render without React errors and without NaN or Infinity in any SVG attribute.
import { describe, expect, it } from "vitest";
import type { ComponentType } from "react";
import { KIT, LineBand, type KitName } from "../charts";
import { render } from "./render";

/** Edge-case props per chart; the Record type forces an entry for every chart in the kit. */
const MISSING: Record<KitName, Record<string, unknown>[]> = {
  LineBand: [
    { title: "a", series: [{ id: "s", label: "s", x: [1, 2, 3], y: [1, null, 3], lo: [0.5, null, 2], hi: [1.5, null, 4] }], xLabel: "x", yLabel: "y" },
    { title: "b", series: [], xLabel: "x", yLabel: "y" },
    { title: "c", series: [{ id: "s", label: "s", x: [1], y: [null] }], xLabel: "x", yLabel: "y" },
    { title: "d", series: [{ id: "s", label: "s", x: [0, 10, 100], y: [1, 2, 3] }], xLabel: "x", yLabel: "y", xLog: true },
  ],
  Bars: [
    { title: "a", categories: ["x", "y"], series: [{ id: "a", label: "A", values: [null, 2], lo: [null, 1], hi: [null, 3] }], valueLabel: "v" },
    { title: "b", categories: [], series: [], valueLabel: "v" },
    { title: "c", categories: ["x"], series: [{ id: "a", label: "A", values: [null] }], valueLabel: "v", mode: "percent" },
  ],
  DotRange: [
    { title: "a", rows: [{ id: "a", label: "A", est: null }, { id: "b", label: "B", est: 1, lo: null, hi: 2 }], valueLabel: "v" },
    { title: "b", rows: [], valueLabel: "v" },
  ],
  Forest: [
    { title: "a", rows: [{ id: "a", label: "A", est: null, se: null }, { id: "b", label: "B", est: 0.1, se: null }], valueLabel: "v" },
    { title: "b", rows: [], valueLabel: "v" },
  ],
  Heatmap: [
    { title: "a", rows: ["r"], cols: ["c1", "c2"], values: [[null, null]] },
    { title: "b", rows: [], cols: [], values: [] },
    { title: "c", rows: ["r"], cols: ["c"], values: [[5]], scale: "div", center: 5 },
  ],
  Histogram: [
    { title: "a", values: Float32Array.from([NaN, NaN]), xLabel: "x" },
    { title: "b", values: [], xLabel: "x" },
    { title: "c", values: [3, 3, 3], xLabel: "x" },
  ],
  HexbinScatter: [
    { title: "a", points: { x: Float32Array.from([NaN, 1]), y: Float32Array.from([1, NaN]) }, xLabel: "x", yLabel: "y" },
    { title: "b", points: { x: [], y: [] }, xLabel: "x", yLabel: "y" },
  ],
  BoxStrip: [
    { title: "a", groups: [{ label: "g", q: null }], valueLabel: "v" },
    { title: "b", groups: [], valueLabel: "v" },
  ],
  Gantt: [
    { title: "a", rows: [] },
    { title: "b", rows: [{ id: "r", parentId: null, label: "run", start: 5, end: null, status: "running" }] },
  ],
  Sparkline: [
    { title: "a", values: [null, null] },
    { title: "b", values: [] },
    { title: "c", values: [2] },
  ],
  Pareto: [
    { title: "a", points: [], xLabel: "x", yLabel: "y" },
    { title: "b", points: [{ x: 1, y: 1 }], xLabel: "x", yLabel: "y", selected: 5 },
  ],
  Rose: [
    { title: "a", petals: [{ angle_deg: 0, value: null }] },
    { title: "b", petals: [] },
    { title: "c", petals: [{ angle_deg: 0, value: 0 }] },
  ],
  Gauge: [{ title: "a", value: null, label: "x" }, { title: "b", value: 5, min: 0, max: 0, label: "x" }],
  IntervalStack: [
    { title: "a", rows: [{ id: "a", label: "A", estimate: null, layers: [{ id: "l", label: "l", lo: null, hi: null }] }], valueLabel: "v" },
    { title: "b", rows: [], valueLabel: "v" },
  ],
  RingProfile: [
    { title: "a", rings: [{ r0_m: 0, r1_m: 0, mean: null }] },
    { title: "b", rings: [] },
  ],
  SmallMultiples: [
    { title: "a", items: [], panelTitle: () => "", renderPanel: () => null, table: { columns: [], rows: [] } },
    { title: "b", items: [{ id: "x" }], panelTitle: () => "x", renderPanel: () => <LineBand bare title="x" series={[{ id: "x", label: "x", x: [1, 2], y: [null, null] }]} xLabel="x" yLabel="y" />, table: { columns: [], rows: [] } },
  ],
};

describe("charts with missing or empty data", () => {
  for (const name of Object.keys(KIT) as KitName[]) {
    MISSING[name].forEach((props, i) => {
      it(`${name} #${i} renders without NaN`, () => {
        const C = KIT[name] as ComponentType<Record<string, unknown>>;
        const errors: unknown[] = [];
        const orig = console.error;
        console.error = (...a: unknown[]) => errors.push(a);
        let html = "";
        try {
          const { container } = render(<C {...props} />);
          html = container.innerHTML;
        } finally {
          console.error = orig;
        }
        const bad = /="[^"]*(NaN|Infinity)[^"]*"/.exec(html);
        expect(bad?.[0] ?? null, `${name} #${i}`).toBeNull();
        expect(errors.map((e) => String((e as unknown[])[0]).slice(0, 200)), `${name} #${i}`).toEqual([]);
      });
    });
  }
});
