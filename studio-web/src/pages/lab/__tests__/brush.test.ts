// Brush layers: strokes overwrite (idempotent), clamp to the lever bounds against the exact
// Float32 inputs, snapshot/restore, and their wire forms (preview SparseEdit, edit blob body).
import { describe, expect, it } from "vitest";
import { decodeSparseEdit } from "../../../api/binary";
import { BrushLayers, applySupportRule, leverBounds, snapshotsEqual } from "../model/brush";
import { applyBrush } from "../../../map/tools/brush";

// Inputs that are not exactly representable in Float32 (as in a real predictors table).
const BASE = Float32Array.from([12.3, 97.7, 0.1, 55.55, 99.99, 3.3]);
const BOUNDS: [number, number] = [0, 100];

describe("brush strokes", () => {
  it("repainting the same cells does not accumulate", () => {
    const b = new BrushLayers(BASE.length);
    const stroke = { mode: "add" as const, amount: 10, base: BASE, bounds: BOUNDS };
    const first = b.paint("canopy", [0, 2, 3], stroke);
    expect(first).toEqual([0, 2, 3]);
    const after1 = Float32Array.from(b.array("canopy"));
    for (let k = 0; k < 5; k++) expect(b.paint("canopy", [0, 2, 3], stroke)).toEqual([]); // nothing changes
    expect(Array.from(b.array("canopy"))).toEqual(Array.from(after1));
    // overlapping second stroke: the overlap is overwritten, not added
    b.paint("canopy", [3, 5], stroke);
    expect(b.array("canopy")[3]).toBe(after1[3]);
    expect(Math.fround(BASE[3] + b.array("canopy")[3])).toBeCloseTo(65.55, 3);
  });

  it("clamps to the lever bounds using the Float32 inputs, never leaving them", () => {
    const b = new BrushLayers(BASE.length);
    b.paint("canopy", [0, 1, 2, 3, 4, 5], { mode: "add", amount: 10, base: BASE, bounds: BOUNDS });
    const e = b.array("canopy");
    for (let i = 0; i < BASE.length; i++) {
      const v = Math.fround(BASE[i] + e[i]);
      expect(v).toBeLessThanOrEqual(100);
      expect(v).toBeGreaterThanOrEqual(0);
    }
    // 97.7 + 10 is capped at 100: the edit is 100 − base computed in Float32
    expect(Math.fround(BASE[1] + e[1])).toBe(100);
    expect(e[1]).toBe(Math.fround(100 - BASE[1]));
    expect(Math.fround(BASE[4] + e[4])).toBe(100);
    // a decrease clamps at the lower bound
    b.paint("canopy", [2, 5], { mode: "add", amount: -50, base: BASE, bounds: BOUNDS });
    expect(Math.fround(BASE[2] + e[2])).toBe(0);
    expect(Math.fround(BASE[5] + e[5])).toBe(0);
  });

  it("changing the amount replaces the painted change; erase sets cells back to 0", () => {
    const b = new BrushLayers(BASE.length);
    b.paint("canopy", [0], { mode: "add", amount: 10, base: BASE, bounds: BOUNDS });
    b.paint("canopy", [0], { mode: "add", amount: 4, base: BASE, bounds: BOUNDS });
    expect(Math.fround(BASE[0] + b.array("canopy")[0])).toBeCloseTo(16.3, 4);
    b.paint("canopy", [0], { mode: "erase", amount: 0, base: BASE, bounds: BOUNDS });
    expect(b.array("canopy")[0]).toBe(0);
    expect(b.isEmpty()).toBe(true);
  });

  it("missing inputs are never painted and unbounded levers are not clamped", () => {
    const b = new BrushLayers(3);
    b.paint("albedo", [0, 1, 2], { mode: "add", amount: 0.1, base: Float32Array.from([0.2, NaN, 0.95]), bounds: leverBounds({ min: null, max: null }) });
    const e = b.array("albedo");
    expect(e[1]).toBe(0);
    expect(Math.fround(0.95 + e[2])).toBeCloseTo(1.05, 5);
    expect(leverBounds(undefined)).toEqual([-Infinity, Infinity]);
    expect(leverBounds({ min: 0, max: 1 })).toEqual([0, 1]);
  });

  it("layers are per lever and versioned", () => {
    const b = new BrushLayers(4);
    const v0 = b.version("canopy");
    b.paint("canopy", [0], { mode: "add", amount: 1, base: new Float32Array(4), bounds: BOUNDS });
    b.paint("albedo", [3], { mode: "add", amount: 0.1, base: new Float32Array(4), bounds: [0, 1] });
    expect(b.version("canopy")).toBe(v0 + 1);
    expect(b.levers().sort()).toEqual(["albedo", "canopy"]);
    expect(b.editedCount("canopy")).toBe(1);
    b.clear("canopy");
    expect(b.levers()).toEqual(["albedo"]);
  });

  it("snapshots restore in place and compare cell for cell", () => {
    const b = new BrushLayers(4);
    const arr = b.array("canopy");
    b.paint("canopy", [1, 2], { mode: "add", amount: 5, base: new Float32Array(4), bounds: BOUNDS });
    const snap = b.snapshot();
    expect(Array.from(snap.canopy.idx)).toEqual([1, 2]);
    b.paint("canopy", [3], { mode: "add", amount: 5, base: new Float32Array(4), bounds: BOUNDS });
    expect(snapshotsEqual(snap, b.snapshot())).toBe(false);
    b.restore(snap);
    expect(b.array("canopy")).toBe(arr);
    expect(Array.from(arr)).toEqual([0, 5, 5, 0]);
    expect(snapshotsEqual(snap, b.snapshot())).toBe(true);
    b.restore({});
    expect(b.isEmpty()).toBe(true);
  });

  it("travels as a sparse preview payload and as an Int32+Float32 edit blob", () => {
    const b = new BrushLayers(6);
    b.paint("canopy", [4, 1], { mode: "add", amount: 2.5, base: new Float32Array(6), bounds: BOUNDS });
    const payload = b.previewPayload();
    const sp = decodeSparseEdit(payload.canopy);
    expect(Array.from(sp.idx)).toEqual([1, 4]);
    expect(Array.from(sp.val)).toEqual([2.5, 2.5]);
    const { body, count } = b.blobBody("canopy");
    expect(count).toBe(2);
    const dv = new DataView(body.buffer, body.byteOffset, body.byteLength);
    expect([dv.getInt32(0, true), dv.getInt32(4, true)]).toEqual([1, 4]);
    expect([dv.getFloat32(8, true), dv.getFloat32(12, true)]).toEqual([2.5, 2.5]);
  });

  it("cells already outside the bounds follow the engine's support rule (never pushed out, never moved against the stroke)", () => {
    // albedo bounds [0.1, 0.9]; row 0 is above (0.95), row 1 below (0.05), row 2 inside
    const base = Float32Array.from([0.95, 0.05, 0.5]);
    const bounds: [number, number] = [0.1, 0.9];
    const b = new BrushLayers(3);
    expect(b.paint("albedo", [0, 1, 2], { mode: "add", amount: 0.1, base, bounds })).toEqual([1, 2]);
    const e = b.array("albedo");
    expect(e[0]).toBe(0); // adding cannot lower a cell above the bound (strict clamping would give −0.05)
    expect(Math.fround(base[1] + e[1])).toBeCloseTo(0.15, 6); // moves up toward the bounds
    expect(Math.fround(base[2] + e[2])).toBeCloseTo(0.6, 6);
    for (let k = 0; k < 3; k++) expect(b.paint("albedo", [0, 1, 2], { mode: "add", amount: 0.1, base, bounds })).toEqual([]); // idempotent
    // a decrease lowers the high cell (down to the lower bound at most) and leaves the low one
    b.paint("albedo", [0, 1], { mode: "add", amount: -2, base, bounds });
    expect(Math.fround(base[0] + e[0])).toBeCloseTo(0.1, 6);
    expect(e[1]).toBe(0);
    // the live map tool path: applyBrush (strict) then the rule, as DesignMap does on each dab
    const live = new Float32Array(3);
    const s = { mode: "add" as const, amount: 0.1, radius_m: 60, base, bounds };
    applyBrush(live, [0, 1, 2], s);
    expect(live[0]).toBeLessThan(0); // what the map kit alone would paint
    expect(applySupportRule(live, [0, 1, 2], s)).toEqual([0]);
    const once = new BrushLayers(3);
    once.paint("albedo", [0, 1, 2], { mode: "add", amount: 0.1, base, bounds });
    expect(Array.from(live)).toEqual(Array.from(once.array("albedo")));
  });
});
