// ScenarioDoc serialisation round-trips every edit mode and every SelectionSpec primitive and
// combinator; validation; the selection tree editing helpers; descriptions; item refs.
import { describe, expect, it } from "vitest";
import { EDIT_MODES, type Edit, type ScenarioDoc } from "../../../api/lab";
import type { SelectionSpec } from "../../../api/types";
import {
  DocError,
  combineSelection,
  contentKey,
  decodeItemRef,
  describeSelection,
  editProblems,
  encodeItemRef,
  itemLayerKey,
  newEdit,
  parseDoc,
  replaceSelection,
  selectionAt,
  serializeDoc,
  usableEdits,
  validateSelection,
  withMode,
  wrapSelection,
} from "../model/doc";

const PRIMITIVES: SelectionSpec[] = [
  { kind: "all" },
  { kind: "zones", values: [3, "Downtown"] },
  { kind: "polygon", crs: "EPSG:4326", rings: [[[-71.43, 41.82], [-71.42, 41.82], [-71.42, 41.83], [-71.43, 41.82]]] },
  { kind: "circle", crs: "run_xy_m", center: [1200, 2400], radius_m: 250 },
  { kind: "rect", crs: "EPSG:4326", min: [-71.5, 41.7], max: [-71.4, 41.8] },
  { kind: "cells", ids: [101, 102, "c-7"] },
  { kind: "blob", blob_id: "bl_9" },
  { kind: "hex", size_m: 500, keys: [100000100001, 100001100000] },
  { kind: "filter", column: "layer:lc_built", op: ">=", value: 0.5 },
  { kind: "filter", column: "predictor:Pct_Canopy", op: "between", value: [10, 30] },
  { kind: "filter", column: "pred:resid", op: "in", value: [1, 2, "x"] },
  { kind: "filter", column: "zone", op: "==", value: "A" },
  { kind: "top", column: "pred:target", frac: 0.1, direction: "highest", within: { kind: "zones", values: [3] } },
  { kind: "top", column: "response:Pct_Canopy:footprint_effect_per_unit", k: 500, direction: "lowest" },
  { kind: "buffer", of: { kind: "region", id: "rg_12" }, radius_m: 120 },
  { kind: "buffer", of: { kind: "blob", blob_id: "bl_2" }, lever_range: "Pct_Canopy" },
  { kind: "region", id: "rg_12" },
];

const COMBINATORS: SelectionSpec[] = [
  { op: "and", args: [{ kind: "zones", values: [3] }, { kind: "filter", column: "layer:lc_built", op: ">=", value: 0.5 }] },
  { op: "or", args: [{ kind: "region", id: "rg_1" }, { kind: "hex", size_m: 250, keys: [1] }] },
  { op: "minus", args: [{ kind: "all" }, { kind: "zones", values: [1] }] },
  { op: "not", arg: { kind: "filter", column: "pred:halfwidth", op: ">", value: 2 } },
  // nested: (Z3 and built) or not (top 10% within Z1 minus buffer)
  {
    op: "or",
    args: [
      { op: "and", args: [{ kind: "zones", values: [3] }, { kind: "filter", column: "layer:lc_built", op: ">=", value: 0.5 }] },
      { op: "not", arg: { op: "minus", args: [{ kind: "top", column: "pred:target", frac: 0.1, direction: "highest", within: { kind: "zones", values: [1] } }, { kind: "buffer", of: { kind: "all" }, radius_m: 60 }] } },
    ],
  },
];

function edit(mode: Edit["mode"], where: SelectionSpec): Edit {
  const e: Edit = { lever: mode === "fill_headroom" ? "Pct_Canopy" : "Albedo", mode, where, label: `${mode} edit` };
  if (mode === "to_percentile") e.percentile = 80;
  else if (mode === "per_cell") e.per_cell_ref = "plan:pl_1";
  else e.amount = mode === "fill_headroom" ? 0.5 : mode === "scale" ? 1.2 : 0.35;
  if (mode === "fill_headroom") e.paved_share = 0.2;
  return e;
}

function docWith(edits: Edit[]): ScenarioDoc {
  return {
    name: "Downtown cool corridor",
    notes: "two edits",
    tags: ["district", "ladder:abc"],
    anchor_run_id: "20261001-090000-full-a1b2",
    edits,
    regions: { Westside: PRIMITIVES[2], Ring: COMBINATORS[0] },
    costs: { Pct_Canopy: { per_unit: 1 }, Albedo: { per_unit: 100 } },
    options: { clip_to_support: true, mediators: false, expert: false },
  };
}

describe("ScenarioDoc serialisation", () => {
  it("round-trips every edit mode", () => {
    const doc = docWith(EDIT_MODES.map((m, i) => edit(m, PRIMITIVES[i % PRIMITIVES.length])));
    expect(doc.edits.map((e) => e.mode)).toEqual([...EDIT_MODES]);
    const text = serializeDoc(doc);
    const back = parseDoc(text);
    expect(back).toEqual(doc);
    expect(serializeDoc(back)).toBe(text);
  });

  it("round-trips every selection primitive and combinator as an edit's where and as a region", () => {
    for (const where of [...PRIMITIVES, ...COMBINATORS]) {
      expect(validateSelection(where)).toEqual([]);
      const doc = docWith([edit("add", where)]);
      doc.regions = { r: where };
      const back = parseDoc(serializeDoc(doc));
      expect(back.edits[0].where).toEqual(where);
      expect(back.regions?.r).toEqual(where);
      expect(serializeDoc(back)).toBe(serializeDoc(doc));
    }
  });

  it("serialises with stable key order regardless of construction order", () => {
    const a = parseDoc({ edits: [{ mode: "add", lever: "Albedo", amount: 0.1 }], name: "x" });
    const b = parseDoc({ name: "x", edits: [{ amount: 0.1, lever: "Albedo", mode: "add" }] });
    expect(serializeDoc(a)).toBe(serializeDoc(b));
    expect(serializeDoc(a)).toBe('{"edits":[{"amount":0.1,"lever":"Albedo","mode":"add"}],"name":"x"}');
  });

  it("the content key ignores name, tags and notes but sees edits, regions, costs and options", () => {
    const doc = docWith([edit("add", { kind: "all" })]);
    expect(contentKey({ ...doc, name: "other", tags: [], notes: "n" })).toBe(contentKey(doc));
    expect(contentKey({ ...doc, options: { ...doc.options, mediators: true } })).not.toBe(contentKey(doc));
    expect(contentKey({ ...doc, costs: {} })).not.toBe(contentKey(doc));
  });

  it("rejects invalid docs with every problem listed", () => {
    expect(() => parseDoc("not json")).toThrow(DocError);
    try {
      parseDoc({ name: 3, edits: [{ lever: "", mode: "grow", amount: "x", where: { kind: "zones", values: [] } }, { lever: "A", mode: "add", where: { op: "and", args: [] } }] });
      throw new Error("should not parse");
    } catch (e) {
      expect(e).toBeInstanceOf(DocError);
      const p = (e as DocError).problems.join("\n");
      expect(p).toMatch(/name: missing/);
      expect(p).toMatch(/edits\[0\]\.lever/);
      expect(p).toMatch(/edits\[0\]\.mode: unknown mode "grow"/);
      expect(p).toMatch(/edits\[0\]\.amount: not a number/);
      expect(p).toMatch(/edits\[0\]\.where: choose at least one zone/);
      expect(p).toMatch(/edits\[1\]\.where: "and" needs at least one condition/);
    }
    expect(validateSelection({ kind: "filter", column: "x", op: "between", value: 3 })).toEqual(["selection: between needs two numbers"]);
    expect(validateSelection({ kind: "top", column: "x", direction: "highest" })).toHaveLength(1);
    expect(validateSelection({ kind: "buffer", of: { kind: "all" } })).toHaveLength(1);
    expect(validateSelection({ op: "xor", args: [] })).toEqual(['selection: unknown combinator "xor"']);
  });
});

describe("edits", () => {
  it("new edits per mode are complete; switching modes keeps lever, where and label", () => {
    for (const m of EDIT_MODES) {
      const e = newEdit("Pct_Canopy", m);
      if (m === "per_cell") expect(editProblems(e)).toEqual(["Per-cell edits need a plan, brush or CSV source."]);
      else expect(editProblems(e)).toEqual([]);
    }
    const e = { ...newEdit("Pct_Canopy", "add", 10), where: { kind: "zones", values: [3] } as SelectionSpec, label: "trees" };
    const f = withMode(e, "to_percentile");
    expect(f).toMatchObject({ lever: "Pct_Canopy", mode: "to_percentile", percentile: 75, where: { kind: "zones", values: [3] }, label: "trees" });
    expect(f.amount).toBeUndefined();
    expect(editProblems({ ...newEdit("Pct_Canopy", "fill_headroom"), amount: 1.5 })).toEqual(["The share of headroom must be between 0 and 1."]);
    expect(usableEdits([newEdit("A", "add", 1), { lever: "A", mode: "add" }])).toHaveLength(1);
  });
});

describe("selection trees", () => {
  const root: SelectionSpec = COMBINATORS[4];

  it("addresses nodes by path", () => {
    expect(selectionAt(root, [0, 1])).toEqual({ kind: "filter", column: "layer:lc_built", op: ">=", value: 0.5 });
    expect(selectionAt(root, [1, "arg", 0, "within"])).toEqual({ kind: "zones", values: [1] });
    expect(selectionAt(root, [1, "arg", 1, "of"])).toEqual({ kind: "all" });
    expect(selectionAt(root, [5])).toBeNull();
  });

  it("replaces, removes (collapsing one-arm and/or) and wraps", () => {
    const r1 = replaceSelection(root, [0, 0], { kind: "zones", values: [4] });
    expect(selectionAt(r1, [0, 0])).toEqual({ kind: "zones", values: [4] });
    expect(selectionAt(root, [0, 0])).toEqual({ kind: "zones", values: [3] }); // immutable
    const r2 = replaceSelection(root, [0, 1], null);
    expect(selectionAt(r2, [0])).toEqual({ kind: "zones", values: [3] });
    const r3 = wrapSelection({ kind: "zones", values: [3] }, [], "not");
    expect(r3).toEqual({ op: "not", arg: { kind: "zones", values: [3] } });
    const r4 = wrapSelection({ kind: "zones", values: [3] }, [], "and", { kind: "region", id: "rg_1" });
    expect(r4).toEqual({ op: "and", args: [{ kind: "zones", values: [3] }, { kind: "region", id: "rg_1" }] });
    expect(replaceSelection(root, [], null)).toEqual({ kind: "all" });
    const r5 = replaceSelection(COMBINATORS[1], [0], null);
    expect(r5).toEqual({ kind: "hex", size_m: 250, keys: [1] });
    const top = replaceSelection(PRIMITIVES[12], ["within"], null);
    expect(top).toEqual({ kind: "top", column: "pred:target", frac: 0.1, direction: "highest" });
  });

  it("combines a map pick with an existing where", () => {
    const z: SelectionSpec = { kind: "zones", values: [3] };
    const f: SelectionSpec = { kind: "filter", column: "layer:lc_built", op: ">=", value: 0.5 };
    expect(combineSelection(undefined, z, "and")).toEqual(z);
    expect(combineSelection({ kind: "all" }, z, "minus")).toEqual({ op: "minus", args: [{ kind: "all" }, z] });
    expect(combineSelection(z, f, "and")).toEqual({ op: "and", args: [z, f] });
    expect(combineSelection({ op: "and", args: [z] }, f, "and")).toEqual({ op: "and", args: [z, f] });
    expect(combineSelection(z, f, "replace")).toEqual(f);
    // a filter primitive has an `op` too: it is not mistaken for a combinator
    expect(combineSelection(f, z, "or")).toEqual({ op: "or", args: [f, z] });
  });

  it("describes selections in plain language", () => {
    expect(describeSelection(COMBINATORS[0])).toBe("zone 3 and lc_built ≥ 0.5");
    expect(describeSelection(COMBINATORS[3])).toBe("not halfwidth > 2");
    expect(describeSelection(PRIMITIVES[12])).toBe("highest 10% by target within zone 3");
    expect(describeSelection(PRIMITIVES[14], { regionName: (id) => (id === "rg_12" ? "Westside" : null) })).toBe("120 m around Westside");
    expect(describeSelection(PRIMITIVES[9])).toBe("Pct_Canopy between 10 and 30");
    expect(describeSelection({ kind: "filter", column: "pred:resid", op: "<", value: -1.5 })).toBe("resid < −1.5");
  });
});

describe("item refs", () => {
  it("encode and decode every kind", () => {
    for (const r of [{ kind: "result", id: "res_1" }, { kind: "configured", slug: "canopy-plus-10" }, { kind: "plan", id: "pl_2" }, { kind: "baseline" }] as const) {
      expect(decodeItemRef(encodeItemRef(r))).toEqual(r);
    }
    expect(decodeItemRef("bogus")).toBeNull();
    expect(decodeItemRef("res:")).toBeNull();
    expect(itemLayerKey({ kind: "configured", slug: "x" })).toBe("sc:x");
    expect(itemLayerKey({ kind: "result", id: "res_1" })).toBe("res:res_1:delta");
    expect(itemLayerKey({ kind: "baseline" })).toBeNull();
  });
});
