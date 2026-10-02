// ScenarioDoc helpers (SPEC §7.2, §7.3, §6.6; api.md §7.1): edit-mode metadata, canonical
// serialisation and validated parsing (round-trips every edit mode and SelectionSpec
// combinator), the content key that matches the server's content_hash inputs, selection
// tree editing by path, plain-language descriptions and compare-tray item refs.
import type { Edit, EditMode, ItemRef, ScenarioDoc } from "../../../api/lab";
import { EDIT_MODES } from "../../../api/lab";
import type { SelectionSpec } from "../../../api/types";

// ---------------------------------------------------------------- edit modes

export type ModeField = "amount" | "percentile" | "per_cell_ref";

export type ModeInfo = { label: string; field: ModeField; fieldLabel: string; hint: string; unit: "lever" | "factor" | "fraction" | "percentile" | "ref" };

export const MODE_INFO: Record<EditMode, ModeInfo> = {
  add: { label: "Add", field: "amount", fieldLabel: "Change", unit: "lever", hint: "Adds the amount to every selected cell." },
  set: { label: "Set to", field: "amount", fieldLabel: "Value", unit: "lever", hint: "Sets every selected cell to the value." },
  scale: { label: "Scale by", field: "amount", fieldLabel: "Factor", unit: "factor", hint: "Multiplies every selected cell by the factor." },
  floor: { label: "Raise to at least", field: "amount", fieldLabel: "Floor", unit: "lever", hint: "Raises cells below the value up to it; cells above are untouched." },
  ceiling: { label: "Cap at", field: "amount", fieldLabel: "Ceiling", unit: "lever", hint: "Lowers cells above the value down to it; cells below are untouched." },
  fill_headroom: {
    label: "Fill plantable headroom",
    field: "amount",
    fieldLabel: "Share of headroom",
    unit: "fraction",
    hint: "Plants this share (0–1) of each cell's plantable headroom. Canopy only; needs the planner layers.",
  },
  to_percentile: {
    label: "Raise to percentile",
    field: "percentile",
    fieldLabel: "Percentile",
    unit: "percentile",
    hint: "Moves each selected cell to the city's p-th percentile in the lever's direction; cells already past it are untouched.",
  },
  per_cell: { label: "Per cell", field: "per_cell_ref", fieldLabel: "Source", unit: "ref", hint: "Per-cell changes from a plan, a brush or a design CSV." },
};

/** A fresh edit for a lever in a mode, with a sensible starting amount. */
export function newEdit(lever: string, mode: EditMode = "add", amount?: number): Edit {
  const e: Edit = { lever, mode, where: { kind: "all" } };
  if (mode === "to_percentile") e.percentile = 75;
  else if (mode === "fill_headroom") e.amount = amount ?? 0.5;
  else if (mode === "scale") e.amount = amount ?? 1.1;
  else if (mode !== "per_cell") e.amount = amount ?? 0;
  return e;
}

/** Switch an edit's mode, keeping the lever, selection and label and resetting the amount fields. */
export function withMode(e: Edit, mode: EditMode, defaultAmount?: number): Edit {
  const next = newEdit(e.lever, mode, defaultAmount);
  next.where = e.where;
  if (e.label) next.label = e.label;
  if (mode === "fill_headroom" && e.paved_share !== undefined) next.paved_share = e.paved_share;
  return next;
}

/** Problems that make an edit unusable for preview, compile and save (empty when it is complete). */
export function editProblems(e: Edit): string[] {
  const out: string[] = [];
  if (!e.lever) out.push("Choose a lever.");
  if (!EDIT_MODES.includes(e.mode)) out.push(`Unknown mode "${String(e.mode)}".`);
  const info = MODE_INFO[e.mode];
  if (info?.field === "amount" && !Number.isFinite(e.amount)) out.push(`Enter ${info.fieldLabel.toLowerCase()}.`);
  if (e.mode === "fill_headroom" && Number.isFinite(e.amount) && ((e.amount as number) < 0 || (e.amount as number) > 1)) out.push("The share of headroom must be between 0 and 1.");
  if (e.mode === "to_percentile" && !(Number.isFinite(e.percentile) && (e.percentile as number) >= 0 && (e.percentile as number) <= 100)) out.push("The percentile must be between 0 and 100.");
  if (e.mode === "per_cell" && !(e.per_cell_ref && /^(plan|blob|csv):.+/.test(e.per_cell_ref))) out.push("Per-cell edits need a plan, brush or CSV source.");
  if (e.where) out.push(...validateSelection(e.where, "where"));
  return out;
}

/** The edits that are complete enough to evaluate. */
export function usableEdits(edits: Edit[]): Edit[] {
  return edits.filter((e) => editProblems(e).length === 0);
}

// ---------------------------------------------------------------- canonical JSON

/** Recursively sort object keys and drop undefined values (arrays keep their order). */
export function canonical(v: unknown): unknown {
  if (Array.isArray(v)) return v.map(canonical);
  if (v && typeof v === "object") {
    const out: Record<string, unknown> = {};
    for (const k of Object.keys(v as Record<string, unknown>).sort()) {
      const x = (v as Record<string, unknown>)[k];
      if (x !== undefined) out[k] = canonical(x);
    }
    return out;
  }
  return v;
}

export function canonicalJson(v: unknown): string {
  return JSON.stringify(canonical(v));
}

/**
 * The part of a doc that defines what is evaluated (the server's content_hash inputs:
 * edits, regions, costs, options). Name, tags and notes are excluded.
 */
export function contentKey(doc: ScenarioDoc): string {
  return canonicalJson({ edits: doc.edits, regions: doc.regions ?? {}, costs: doc.costs ?? {}, options: doc.options ?? {} });
}

/** Canonical serialisation of a ScenarioDoc (stable key order). */
export function serializeDoc(doc: ScenarioDoc): string {
  return canonicalJson(doc);
}

export class DocError extends Error {
  readonly problems: string[];
  constructor(problems: string[]) {
    super(problems[0] ?? "Invalid scenario");
    this.name = "DocError";
    this.problems = problems;
  }
}

const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const isPair = (v: unknown): v is [number, number] => Array.isArray(v) && v.length === 2 && isNum(v[0]) && isNum(v[1]);
const CRS = new Set(["EPSG:4326", "run_xy_m"]);
const FILTER_OPS = new Set(["<", "<=", ">", ">=", "==", "between", "in"]);

/** Every problem in a SelectionSpec tree (kinds, combinators and field values), with paths. */
export function validateSelection(spec: unknown, path = "selection"): string[] {
  if (!spec || typeof spec !== "object") return [`${path}: not a selection`];
  const o = spec as Record<string, unknown>;
  const bad = (m: string) => [`${path}: ${m}`];
  if (!("kind" in o)) {
    if (o.op === "not") return validateSelection(o.arg, `${path}.arg`);
    if (o.op === "and" || o.op === "or" || o.op === "minus") {
      if (!Array.isArray(o.args) || o.args.length === 0) return bad(`"${o.op}" needs at least one condition`);
      return o.args.flatMap((a, i) => validateSelection(a, `${path}.args[${i}]`));
    }
    return bad(`unknown combinator "${String(o.op)}"`);
  }
  switch (o.kind) {
    case "all":
      return [];
    case "zones":
      return Array.isArray(o.values) && o.values.length > 0 && o.values.every((v) => typeof v === "string" || isNum(v)) ? [] : bad("choose at least one zone");
    case "polygon":
      return CRS.has(o.crs as string) && Array.isArray(o.rings) && o.rings.length > 0 && o.rings.every((r) => Array.isArray(r) && r.length >= 3 && r.every(isPair)) ? [] : bad("a polygon needs a CRS and rings of at least 3 points");
    case "circle":
      return CRS.has(o.crs as string) && isPair(o.center) && isNum(o.radius_m) && o.radius_m > 0 ? [] : bad("a circle needs a CRS, a centre and a positive radius");
    case "rect":
      return CRS.has(o.crs as string) && isPair(o.min) && isPair(o.max) ? [] : bad("a rectangle needs a CRS and two corners");
    case "cells":
      return Array.isArray(o.ids) && o.ids.every((v) => typeof v === "string" || isNum(v)) ? [] : bad("cells need a list of ids");
    case "blob":
      return typeof o.blob_id === "string" && o.blob_id.length > 0 ? [] : bad("a brushed selection needs its blob id");
    case "hex":
      return (o.size_m === 250 || o.size_m === 500) && Array.isArray(o.keys) && o.keys.every(isNum) ? [] : bad("hexagons need a size of 250 or 500 m and keys");
    case "filter": {
      if (typeof o.column !== "string" || !o.column) return bad("choose a column");
      if (!FILTER_OPS.has(o.op as string)) return bad(`unknown operator "${String(o.op)}"`);
      if (o.op === "between") return isPair(o.value) ? [] : bad("between needs two numbers");
      if (o.op === "in") return Array.isArray(o.value) && o.value.length > 0 ? [] : bad("in needs at least one value");
      return isNum(o.value) || (o.op === "==" && typeof o.value === "string") ? [] : bad("enter a value");
    }
    case "top": {
      if (typeof o.column !== "string" || !o.column) return bad("choose a column");
      if (o.direction !== "highest" && o.direction !== "lowest") return bad("choose highest or lowest");
      const hasFrac = isNum(o.frac) && o.frac > 0 && o.frac <= 1;
      const hasK = isNum(o.k) && o.k >= 1;
      if (!hasFrac && !hasK) return bad("enter a share (0–1] or a count");
      // A stored doc echoes an unset `within` as null (api.md §1): null means no limit.
      return o.within === undefined || o.within === null ? [] : validateSelection(o.within, `${path}.within`);
    }
    case "buffer": {
      const radius = isNum(o.radius_m) && o.radius_m > 0;
      const lever = typeof o.lever_range === "string" && o.lever_range.length > 0;
      if (!radius && !lever) return bad("a buffer needs a radius or a lever range");
      return validateSelection(o.of, `${path}.of`);
    }
    case "region":
      return typeof o.id === "string" && o.id.length > 0 ? [] : bad("choose a saved region");
    default:
      return bad(`unknown kind "${String(o.kind)}"`);
  }
}

function editFromJson(v: unknown, i: number): { edit: Edit | null; problems: string[] } {
  const p = `edits[${i}]`;
  if (!v || typeof v !== "object") return { edit: null, problems: [`${p}: not an object`] };
  // A stored doc echoes unset optional fields as null (api.md §1): read null as absent.
  const o = Object.fromEntries(Object.entries(v as Record<string, unknown>).filter(([, x]) => x !== null));
  const problems: string[] = [];
  if (typeof o.lever !== "string" || !o.lever) problems.push(`${p}.lever: missing`);
  if (!EDIT_MODES.includes(o.mode as EditMode)) problems.push(`${p}.mode: unknown mode "${String(o.mode)}"`);
  for (const k of ["amount", "percentile", "paved_share"] as const) if (o[k] !== undefined && !isNum(o[k])) problems.push(`${p}.${k}: not a number`);
  if (o.per_cell_ref !== undefined && typeof o.per_cell_ref !== "string") problems.push(`${p}.per_cell_ref: not a string`);
  if (o.label !== undefined && typeof o.label !== "string") problems.push(`${p}.label: not a string`);
  if (o.where !== undefined) problems.push(...validateSelection(o.where, `${p}.where`));
  if (problems.length) return { edit: null, problems };
  const e: Edit = { lever: o.lever as string, mode: o.mode as EditMode };
  for (const k of ["amount", "percentile", "paved_share"] as const) if (o[k] !== undefined) e[k] = o[k] as number;
  if (o.per_cell_ref !== undefined) e.per_cell_ref = o.per_cell_ref as string;
  if (o.where !== undefined) e.where = o.where as SelectionSpec;
  if (o.label !== undefined) e.label = o.label as string;
  return { edit: e, problems: [] };
}

/** Parse and validate a ScenarioDoc (from JSON text or an object); throws DocError listing every problem. */
export function parseDoc(input: string | unknown): ScenarioDoc {
  let v: unknown = input;
  if (typeof input === "string") {
    try {
      v = JSON.parse(input);
    } catch {
      throw new DocError(["not valid JSON"]);
    }
  }
  if (!v || typeof v !== "object" || Array.isArray(v)) throw new DocError(["a scenario must be a JSON object"]);
  const o = v as Record<string, unknown>;
  const problems: string[] = [];
  if (typeof o.name !== "string") problems.push("name: missing");
  if (!Array.isArray(o.edits)) problems.push("edits: must be a list");
  if (o.notes !== undefined && typeof o.notes !== "string") problems.push("notes: not a string");
  if (o.tags !== undefined && !(Array.isArray(o.tags) && o.tags.every((t) => typeof t === "string"))) problems.push("tags: must be a list of strings");
  if (o.anchor_run_id !== undefined && o.anchor_run_id !== null && typeof o.anchor_run_id !== "string") problems.push("anchor_run_id: not a string");
  const edits: Edit[] = [];
  if (Array.isArray(o.edits))
    o.edits.forEach((e, i) => {
      const r = editFromJson(e, i);
      if (r.edit) edits.push(r.edit);
      problems.push(...r.problems);
    });
  if (o.regions !== undefined) {
    if (!o.regions || typeof o.regions !== "object" || Array.isArray(o.regions)) problems.push("regions: must be an object");
    else for (const [k, s] of Object.entries(o.regions)) problems.push(...validateSelection(s, `regions.${k}`));
  }
  if (o.costs !== undefined) {
    if (!o.costs || typeof o.costs !== "object" || Array.isArray(o.costs)) problems.push("costs: must be an object");
    else for (const [k, c] of Object.entries(o.costs)) if (!c || typeof c !== "object" || !isNum((c as { per_unit?: unknown }).per_unit)) problems.push(`costs.${k}.per_unit: not a number`);
  }
  if (o.options !== undefined) {
    if (!o.options || typeof o.options !== "object") problems.push("options: must be an object");
    else for (const k of ["clip_to_support", "mediators", "expert"]) {
      const x = (o.options as Record<string, unknown>)[k];
      if (x !== undefined && typeof x !== "boolean") problems.push(`options.${k}: not true/false`);
    }
  }
  if (problems.length) throw new DocError(problems);
  const doc: ScenarioDoc = { name: o.name as string, edits };
  if (o.notes !== undefined) doc.notes = o.notes as string;
  if (o.tags !== undefined) doc.tags = [...(o.tags as string[])];
  if (o.anchor_run_id !== undefined) doc.anchor_run_id = o.anchor_run_id as string | null;
  if (o.regions !== undefined) doc.regions = o.regions as Record<string, SelectionSpec>;
  if (o.costs !== undefined) doc.costs = o.costs as Record<string, { per_unit: number }>;
  if (o.options !== undefined) doc.options = { ...(o.options as ScenarioDoc["options"]) };
  return doc;
}

export function emptyDoc(name = "Untitled scenario", runId: string | null = null): ScenarioDoc {
  return { name, tags: [], notes: "", anchor_run_id: runId, edits: [], regions: {}, costs: {}, options: { clip_to_support: true, mediators: true, expert: false } };
}

// ---------------------------------------------------------------- selection trees

/** A path into a selection tree: combinator arg index, "arg" (not), "within" (top) or "of" (buffer). */
export type SelPath = (number | "arg" | "within" | "of")[];

type Combinator = Extract<SelectionSpec, { args: SelectionSpec[] }> | Extract<SelectionSpec, { arg: SelectionSpec }>;
type Primitive = Exclude<SelectionSpec, Combinator>;

/** Combinators have no `kind` (a filter primitive has an `op` too, so `op` alone does not tell). */
export const isCombinator = (s: SelectionSpec): s is Combinator => !("kind" in s);
export const isPrimitive = (s: SelectionSpec): s is Primitive => "kind" in s;
const isNot = (s: SelectionSpec): s is Extract<SelectionSpec, { op: "not" }> => isCombinator(s) && s.op === "not";
const isMulti = (s: SelectionSpec): s is Extract<SelectionSpec, { args: SelectionSpec[] }> => isCombinator(s) && s.op !== "not";

export function selectionAt(root: SelectionSpec, path: SelPath): SelectionSpec | null {
  let cur: SelectionSpec | undefined = root;
  for (const step of path) {
    if (!cur) return null;
    if (typeof step === "number") cur = isMulti(cur) ? cur.args[step] : undefined;
    else if (step === "arg") cur = isNot(cur) ? cur.arg : undefined;
    else if (step === "within") cur = "kind" in cur && cur.kind === "top" ? (cur.within ?? undefined) : undefined;
    else cur = "kind" in cur && cur.kind === "buffer" ? cur.of : undefined;
  }
  return cur ?? null;
}

/** A copy of `root` with the node at `path` replaced (or removed when `node` is null). */
export function replaceSelection(root: SelectionSpec, path: SelPath, node: SelectionSpec | null): SelectionSpec {
  if (path.length === 0) return node ?? { kind: "all" };
  const [step, ...rest] = path;
  if (typeof step === "number") {
    if (!isMulti(root)) return root;
    const args = [...root.args];
    if (rest.length === 0 && node === null) args.splice(step, 1);
    else args[step] = replaceSelection(args[step], rest, node);
    if (args.length === 0) return { kind: "all" };
    if (args.length === 1 && root.op !== "minus") return args[0]; // a one-arm and/or is that arm
    return { op: root.op, args };
  }
  if (step === "arg" && isNot(root)) {
    if (rest.length === 0 && node === null) return { kind: "all" };
    return { op: "not", arg: replaceSelection(root.arg, rest, node) };
  }
  if (step === "within" && "kind" in root && root.kind === "top") {
    if (rest.length === 0 && node === null) {
      const { within: _drop, ...top } = root;
      void _drop;
      return top;
    }
    return { ...root, within: replaceSelection(root.within ?? { kind: "all" }, rest, node) };
  }
  if (step === "of" && "kind" in root && root.kind === "buffer") return { ...root, of: replaceSelection(root.of, rest, node ?? { kind: "all" }) };
  return root;
}

/** Wrap the node at `path` in a combinator; "not" negates it, the others add `extra` (default all) as a second arm. */
export function wrapSelection(root: SelectionSpec, path: SelPath, op: "and" | "or" | "minus" | "not", extra?: SelectionSpec): SelectionSpec {
  const node = selectionAt(root, path) ?? { kind: "all" };
  const wrapped: SelectionSpec = op === "not" ? { op: "not", arg: node } : { op, args: [node, extra ?? { kind: "all" }] };
  return replaceSelection(root, path, wrapped);
}

/**
 * Combine a new selection with an existing one. "replace" returns the new one; against no
 * selection (all cells) "and"/"or" also mean the new one and "minus" means everything but it.
 */
export function combineSelection(base: SelectionSpec | undefined, next: SelectionSpec, how: "replace" | "and" | "or" | "minus"): SelectionSpec {
  if (how === "replace") return next;
  const isAll = !base || ("kind" in base && base.kind === "all");
  if (isAll) return how === "minus" ? { op: "minus", args: [{ kind: "all" }, next] } : next;
  if (isMulti(base!) && base.op === how && how !== "minus") return { op: how, args: [...base.args, next] };
  return { op: how, args: [base!, next] };
}

export type DescribeContext = { regionName?: (id: string) => string | null; columnLabel?: (column: string) => string };

const OP_TEXT: Record<string, string> = { "<": "<", "<=": "≤", ">": ">", ">=": "≥", "==": "=", between: "between", in: "in" };

function num(v: unknown): string {
  return typeof v === "number" ? String(Number(v.toPrecision(6))).replace("-", "−") : String(v);
}

/** Plain-language description of a selection ("Zone 3 and lc_built ≥ 0.5"). */
export function describeSelection(spec: SelectionSpec | undefined, ctx: DescribeContext = {}, depth = 0): string {
  if (!spec) return "all cells";
  const col = (c: string) => ctx.columnLabel?.(c) ?? c.replace(/^(predictor|layer|pred|planner):/, "");
  const paren = (s: string) => (depth > 0 ? `(${s})` : s);
  if (isCombinator(spec)) {
    if (spec.op === "not") return `not ${describeSelection(spec.arg, ctx, depth + 1)}`;
    const parts = spec.args.map((a) => describeSelection(a, ctx, depth + 1));
    if (spec.op === "minus") return paren(parts.length ? `${parts[0]} minus ${parts.slice(1).join(" and ")}` : "nothing");
    return paren(parts.join(spec.op === "and" ? " and " : " or "));
  }
  switch (spec.kind) {
    case "all":
      return "all cells";
    case "zones":
      return spec.values.length === 1 ? `zone ${spec.values[0]}` : `zones ${spec.values.join(", ")}`;
    case "polygon":
      return `a drawn polygon${spec.crs === "run_xy_m" ? " (run frame)" : ""}`;
    case "circle":
      return `a ${num(Math.round(spec.radius_m))} m circle`;
    case "rect":
      return "a drawn rectangle";
    case "cells":
      return `${spec.ids.length} listed cells`;
    case "blob":
      return "brushed cells";
    case "hex":
      return `${spec.keys.length} hexagon${spec.keys.length === 1 ? "" : "s"} (${spec.size_m} m)`;
    case "filter": {
      const v = spec.value;
      if (spec.op === "between" && Array.isArray(v)) return `${col(spec.column)} between ${num(v[0])} and ${num(v[1])}`;
      if (spec.op === "in" && Array.isArray(v)) return `${col(spec.column)} in ${v.map(num).join(", ")}`;
      return `${col(spec.column)} ${OP_TEXT[spec.op]} ${num(v)}`;
    }
    case "top": {
      const how = spec.frac != null ? `${num(Math.round(spec.frac * 1000) / 10)}%` : `${spec.k} cells`;
      const within = spec.within ? ` within ${describeSelection(spec.within, ctx, depth + 1)}` : "";
      return `${spec.direction} ${how} by ${col(spec.column)}${within}`;
    }
    case "buffer":
      return `${spec.radius_m != null ? `${num(spec.radius_m)} m` : `${spec.lever_range} range`} around ${describeSelection(spec.of, ctx, depth + 1)}`;
    case "region":
      return ctx.regionName?.(spec.id) ?? `region ${spec.id}`;
  }
}

// ---------------------------------------------------------------- compare-tray item refs

/** "res:<id>" · "configured:<slug>" · "plan:<id>" · "baseline". */
export function encodeItemRef(r: ItemRef): string {
  switch (r.kind) {
    case "result":
      return `res:${r.id}`;
    case "configured":
      return `configured:${r.slug}`;
    case "plan":
      return `plan:${r.id}`;
    case "baseline":
      return "baseline";
  }
}

export function decodeItemRef(s: string): ItemRef | null {
  if (s === "baseline") return { kind: "baseline" };
  const i = s.indexOf(":");
  if (i <= 0) return null;
  const head = s.slice(0, i);
  const rest = s.slice(i + 1);
  if (!rest) return null;
  if (head === "res") return { kind: "result", id: rest };
  if (head === "configured") return { kind: "configured", slug: rest };
  if (head === "plan") return { kind: "plan", id: rest };
  return null;
}

export function sameItem(a: ItemRef, b: ItemRef): boolean {
  return encodeItemRef(a) === encodeItemRef(b);
}

/** The run layer that maps an item's per-cell ΔT (api.md §6.2 layer keys), or null for the baseline. */
export function itemLayerKey(r: ItemRef): string | null {
  switch (r.kind) {
    case "result":
      return `res:${r.id}:delta`;
    case "configured":
      return `sc:${r.slug}`;
    case "plan":
      return `plan:${r.id}:closed_loop_delta`;
    case "baseline":
      return null;
  }
}
