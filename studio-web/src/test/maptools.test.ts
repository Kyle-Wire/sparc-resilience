import { describe, expect, it } from "vitest";
import { applyBrush, brushTool, circleTool, editedRows, hexKeyOfRow, pickSelection, pickTool, polygonTool, rectTool, zoneValue, type BrushSettings } from "../map/tools";
import { hexCenter, hexKey, makeGrid, niceLength, rowsInRadius } from "../map/grid";
import { gridMeta, grid3 } from "./grid";

describe("brush: Float32 edits, clamped, idempotent", () => {
  const base = Float32Array.from([10, 95.5, 0, 50, NaN, 99.99]);
  const settings = (p: Partial<BrushSettings> = {}): BrushSettings => ({ amount: 10, mode: "add", radius_m: 60, base, bounds: [0, 100], ...p });

  it("writes clamp(base + amount) − base as Float32", () => {
    const edit = new Float32Array(6);
    applyBrush(edit, [0, 1, 2, 3, 5], settings());
    expect(edit[0]).toBe(10);
    expect(edit[1]).toBe(Math.fround(100 - 95.5)); // clamped at the upper bound
    expect(edit[3]).toBe(10);
    for (const r of [0, 1, 2, 3, 5]) expect(Math.fround(base[r] + edit[r])).toBeLessThanOrEqual(100);
    expect(edit[5]).toBeGreaterThanOrEqual(0);
    expect(edit).toBeInstanceOf(Float32Array);
  });
  it("repainting the same cells does not accumulate", () => {
    const edit = new Float32Array(6);
    applyBrush(edit, [0, 3], settings());
    const once = Float32Array.from(edit);
    const changed = applyBrush(edit, [0, 3], settings());
    applyBrush(edit, [0, 3], settings());
    expect(changed).toEqual([]);
    expect(edit).toEqual(once);
  });
  it("clamps decreases at the lower bound and skips missing inputs", () => {
    const edit = new Float32Array(6);
    applyBrush(edit, [0, 2, 4], settings({ amount: -25 }));
    expect(edit[0]).toBe(-10);
    expect(edit[2]).toBe(0);
    expect(edit[4]).toBe(0); // NaN base: never edited
  });
  it("erases", () => {
    const edit = new Float32Array(6);
    applyBrush(edit, [0, 3], settings());
    applyBrush(edit, [3], settings({ mode: "erase" }));
    expect([edit[0], edit[3]]).toEqual([10, 0]);
    expect(editedRows(edit)).toBe(1);
  });
  it("strokes paint every cell within the radius along the path", () => {
    const g = grid3();
    const edit = new Float32Array(g.n);
    const b = Float32Array.from({ length: g.n }, () => 20);
    const t = brushTool(g, edit, () => ({ amount: 5, mode: "add", radius_m: 20, base: b, bounds: [0, 100] }));
    t.down({ px: 0.5, py: 0.5 });
    t.move({ px: 2.5, py: 0.5 }, 1);
    const res = t.up({ px: 2.5, py: 0.5 });
    expect(res?.kind).toBe("edit");
    expect([...edit]).toEqual([5, 5, 0, 0, 0, 0, 0]); // rows 0 and 1 (row 2's cell at px 2 is empty)
    expect(rowsInRadius(g, 1.5, 1.5, 35).sort()).toEqual([1, 2, 3, 5]); // 1.17 cells: the four edge neighbours
    expect(rowsInRadius(g, 1.5, 1.5, 45)).toHaveLength(7); // 1.5 cells: diagonals too
  });
});

describe("selection tools", () => {
  const g = grid3();
  it("rect produces a lon/lat rect and a local mask", () => {
    const t = rectTool(g);
    t.down({ px: 0, py: 0 });
    t.move({ px: 2, py: 2 }, 1);
    expect(t.preview(null)).toEqual({ kind: "rect", x0: 0, y0: 0, x1: 2, y1: 2 });
    const r = t.up({ px: 2, py: 2 });
    if (!r || r.kind !== "selection") throw new Error("expected a selection");
    expect(r.spec).toMatchObject({ kind: "rect", crs: "EPSG:4326" });
    expect([...r.mask]).toEqual([1, 1, 1, 0, 0, 0, 0]); // centres inside [0,2]²: rows 0, 1, 2
    expect(r.portable).toBe(true);
  });
  it("circle uses metres for the radius", () => {
    const t = circleTool(g);
    t.down({ px: 1.5, py: 1.5 });
    const r = t.up({ px: 2.5, py: 1.5 });
    if (!r || r.kind !== "selection") throw new Error("expected a selection");
    expect(r.spec).toMatchObject({ kind: "circle", crs: "EPSG:4326", radius_m: 30 });
    expect([...r.mask]).toEqual([0, 1, 1, 1, 0, 1, 0]);
  });
  it("polygon closes on Enter/double-click and works without a CRS (run_xy_m, non-portable)", () => {
    const g2 = makeGrid(gridMeta(3, 3, 7, { has_lonlat: false, corners: null, crs: null }), g.ix, g.iy);
    const t = polygonTool(g2);
    for (const p of [{ px: 0, py: 0 }, { px: 3, py: 0 }, { px: 0, py: 3 }]) t.down(p);
    const r = t.finish();
    if (!r || r.kind !== "selection") throw new Error("expected a selection");
    if (!("kind" in r.spec) || r.spec.kind !== "polygon") throw new Error("expected a polygon");
    expect(r.spec.crs).toBe("run_xy_m");
    expect(r.spec.rings[0].length).toBe(4); // closed ring
    expect([...r.mask]).toEqual([1, 1, 1, 0, 0, 0, 0]);
    expect(r.portable).toBe(false);
  });
  it("pick selects a zone (shift adds) or a hexagon", () => {
    let shift = false;
    const t = pickTool(g, { by: "zone" }, () => shift);
    const a = t.up({ px: 0.5, py: 0.5 });
    expect(a && a.kind === "selection" && a.spec).toEqual({ kind: "zones", values: [1] });
    shift = true;
    const b = t.up({ px: 2.5, py: 2.5 });
    expect(b && b.kind === "selection" && b.spec).toEqual({ kind: "zones", values: [1, 2] });
    const h = pickSelection(g, { by: "hex", size_m: 250 }, [hexKeyOfRow(g, 0, 250)]);
    if (h.kind !== "selection") throw new Error("expected a selection");
    expect(h.spec).toMatchObject({ kind: "hex", size_m: 250 });
    expect(h.mask[0]).toBe(1);
  });
  it("reads grid.bin zone as an index into GridMeta.zones for string and numeric codes (−1 = none)", () => {
    const ix = Int32Array.from([0, 1, 2]);
    const iy = Int32Array.from([0, 0, 0]);
    // A numeric code beyond int16 and a row without a zone
    const num = makeGrid(gridMeta(3, 1, 3, { zones: [7, 90210] }), ix, iy, undefined, undefined, Int16Array.from([1, 0, -1]));
    expect([0, 1, 2].map((r) => zoneValue(num, r))).toEqual([90210, 7, null]);
    const t = pickTool(num, { by: "zone" });
    const a = t.up({ px: 0.5, py: 0.5 });
    if (!a || a.kind !== "selection") throw new Error("expected a selection");
    expect(a.spec).toEqual({ kind: "zones", values: [90210] });
    expect([...a.mask]).toEqual([1, 0, 0]);
    expect(t.up({ px: 2.5, py: 0.5 })).toBeNull();
    const str = makeGrid(gridMeta(3, 1, 3, { zones: ["Downtown", "Elmwood"] }), ix, iy, undefined, undefined, Int16Array.from([0, 1, 1]));
    const b = pickSelection(str, { by: "zone" }, ["Elmwood"]);
    if (b.kind !== "selection") throw new Error("expected a selection");
    expect([...b.mask]).toEqual([0, 1, 1]);
    // No zone array in grid.bin: every row has no zone
    expect(zoneValue(makeGrid(gridMeta(3, 1, 3, { zones: [1] }), ix, iy), 0)).toBeNull();
  });
});

describe("hex keys match sparc.core.planner.hex_ids", () => {
  // Golden keys printed by the Python function for these points.
  const x = [0, 125, 250, -310.5, 1000, 1234.5, 299712.3, 216.50635094610965];
  const y = [0, 0, 216.5, 400, -777, 4321, 4637211.9, 125];
  const golden: Record<number, number[]> = {
    250: [100000100000, 100000100000, 100000100001, 99998100002, 100006099996, 99995100020, 90490121418, 100000100001],
    500: [100000100000, 100000100000, 100000100001, 99999100001, 100003099998, 99997100010, 95245110709, 100000100000],
  };
  for (const size of [250, 500]) {
    it(`size ${size} m`, () => {
      expect(x.map((xi, i) => hexKey(xi, y[i], size))).toEqual(golden[size]);
    });
  }
  it("hexCenter inverts hexKey", () => {
    const k = hexKey(1234.5, 4321, 250);
    const [cx, cy] = hexCenter(k, 250);
    expect(hexKey(cx, cy, 250)).toBe(k);
  });
  it("scale bars use 1/2/5 × 10^k metres", () => {
    expect([niceLength(730), niceLength(1999), niceLength(4900), niceLength(60)]).toEqual([500, 1000, 2000, 50]);
  });
});
