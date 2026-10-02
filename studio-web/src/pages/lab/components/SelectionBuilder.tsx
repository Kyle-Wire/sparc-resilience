// SelectionBuilder (SPEC §6.6, §7.3): the `where` of an edit as a SelectionSpec tree. Zones,
// filters (namespaced columns with a live cell count), top-k / top-%, buffers (metres or a
// lever's influence range), saved regions and the map tools' shapes (rect, circle, polygon,
// zone/hex pick, brushed masks) combine with and / or / minus / not. The live count resolves
// the whole tree on the server (`POST /selection/resolve`, debounced) and reports cells, area,
// residents and medians; "Save as region" stores it for reuse across runs.
import { useEffect, useMemo, useRef, useState } from "react";
import { errorMessage } from "../../../api/client";
import { invalidate } from "../../../api/resource";
import { createRegion, resolveSelection, selectionMask, type Lever, type Region, type SelectionReply } from "../../../api/lab";
import type { LayerGroup, SelectionSpec } from "../../../api/types";
import { Button } from "../../../components/ui/Button";
import { IconButton } from "../../../components/ui/IconButton";
import { NumberField } from "../../../components/ui/NumberField";
import type { GridData } from "../../../map/grid";
import { toast } from "../../../stores/ui";
import { fmtInt, fmtNum } from "../../../theme/format";
import { combineSelection, describeSelection, replaceSelection, validateSelection, wrapSelection, type SelPath } from "../model/doc";
import { debounce } from "../model/timing";

export type ColumnOption = { value: string; label: string; group: string };

const PRED: Record<string, string> = { obs: "pred:target", pred: "pred:pred", resid: "pred:resid", halfwidth: "pred:halfwidth", dist_train_m: "pred:dist_train_m" };

/** Namespaced filter columns (api.md §1) from the run's layer catalogue. */
export function columnOptions(groups: LayerGroup[]): ColumnOption[] {
  const out: ColumnOption[] = [];
  for (const g of groups)
    for (const l of g.layers) {
      if (l.scale === "cat" && l.key !== "zone") continue;
      let value: string | null = null;
      if (g.id === "inputs" && l.key !== "zone") value = `predictor:${l.key}`;
      else if (g.id === "planner") value = `layer:${l.key}`;
      else if (g.id === "temperature") value = PRED[l.key] ?? l.key;
      else if (l.key.startsWith("sc:")) value = `configured:${l.key.slice(3)}`;
      else if (/^res:[^:]+:delta$/.test(l.key)) value = `result:${l.key.slice(4, -6)}:delta`;
      else if (g.id === "effects" || g.id === "budget" || g.id === "causal") value = l.key;
      if (value) out.push({ value, label: `${l.label}${l.unit ? ` (${l.unit})` : ""}`, group: g.label });
    }
  return out;
}

export type MapPick = { spec: SelectionSpec; label: string };

export type SelectionBuilderProps = {
  rid: string;
  grid: GridData;
  value: SelectionSpec | undefined;
  onChange: (spec: SelectionSpec) => void;
  columns: ColumnOption[];
  regions: Region[];
  levers: Lever[];
  /** The latest selection drawn or picked on the map, offered for use here. */
  mapPick?: MapPick | null;
  /** The resolved mask (map tint) while this builder is focused. */
  onMask?: (mask: Uint8Array | null) => void;
  /** Id prefix for labels. */
  idPrefix: string;
};

type Ctx = Omit<SelectionBuilderProps, "value" | "onChange" | "mapPick" | "onMask" | "idPrefix" | "rid"> & { root: SelectionSpec; set: (s: SelectionSpec) => void; idPrefix: string };

function defaultFilter(columns: ColumnOption[]): SelectionSpec {
  return { kind: "filter", column: columns[0]?.value ?? "", op: ">=", value: 0 };
}

function PrimitiveEditor({ node, path, ctx }: { node: Extract<SelectionSpec, { kind: string }>; path: SelPath; ctx: Ctx }) {
  const put = (next: SelectionSpec) => ctx.set(replaceSelection(ctx.root, path, next));
  const id = `${ctx.idPrefix}-${path.join("-") || "root"}`;
  switch (node.kind) {
    case "all":
      return <span className="cap">All cells</span>;
    case "zones": {
      const zones = ctx.grid.meta.zones;
      const on = new Set(node.values.map(String));
      return (
        <div className="chips" role="group" aria-label="Zones">
          {zones.map((z) => (
            <button
              key={String(z)}
              type="button"
              className="chip"
              aria-pressed={on.has(String(z))}
              onClick={() => {
                const values = on.has(String(z)) ? node.values.filter((v) => String(v) !== String(z)) : [...node.values, z];
                put({ kind: "zones", values });
              }}
            >
              Zone {String(z)}
            </button>
          ))}
          {!zones.length ? <span className="cap">This run has no zones.</span> : null}
        </div>
      );
    }
    case "filter": {
      const v = node.value;
      return (
        <div className="sel-form">
          <label className="sr-only" htmlFor={`${id}-col`}>
            Column
          </label>
          <input id={`${id}-col`} list={`${ctx.idPrefix}-columns`} value={node.column} placeholder="column, e.g. layer:lc_built" onChange={(e) => put({ ...node, column: e.target.value })} aria-label="Filter column" />
          <select aria-label="Operator" value={node.op} onChange={(e) => {
            const op = e.target.value as typeof node.op;
            const value = op === "between" ? (Array.isArray(v) && v.length === 2 ? v : [typeof v === "number" ? v : 0, typeof v === "number" ? v : 1]) : op === "in" ? (Array.isArray(v) ? v : [typeof v === "number" ? v : 0]) : Array.isArray(v) ? (v[0] as number) : v;
            put({ ...node, op, value: value as typeof node.value });
          }}>
            {["<", "<=", ">", ">=", "==", "between", "in"].map((o) => (
              <option key={o} value={o}>
                {o === "<=" ? "≤" : o === ">=" ? "≥" : o}
              </option>
            ))}
          </select>
          {node.op === "between" && Array.isArray(v) ? (
            <>
              <NumberField label="From" value={typeof v[0] === "number" ? v[0] : null} onChange={(x) => put({ ...node, value: [x ?? 0, (v[1] as number) ?? 0] })} />
              <NumberField label="To" value={typeof v[1] === "number" ? v[1] : null} onChange={(x) => put({ ...node, value: [(v[0] as number) ?? 0, x ?? 0] })} />
            </>
          ) : node.op === "in" ? (
            <input
              aria-label="Values (comma separated)"
              defaultValue={Array.isArray(v) ? v.join(", ") : String(v)}
              onBlur={(e) => {
                const vals = e.target.value.split(",").map((s) => s.trim()).filter(Boolean).map((s) => (Number.isFinite(Number(s)) ? Number(s) : s));
                put({ ...node, value: vals });
              }}
            />
          ) : (
            <NumberField label="Value" value={typeof v === "number" ? v : null} onChange={(x) => put({ ...node, value: x ?? 0 })} />
          )}
        </div>
      );
    }
    case "top": {
      const byFrac = node.frac !== undefined || node.k === undefined;
      return (
        <div className="sel-form">
          <select aria-label="Direction" value={node.direction} onChange={(e) => put({ ...node, direction: e.target.value as "highest" | "lowest" })}>
            <option value="highest">Highest</option>
            <option value="lowest">Lowest</option>
          </select>
          {byFrac ? (
            <NumberField label="Share (%)" unit="%" min={0.1} max={100} value={node.frac !== undefined ? Number((node.frac * 100).toPrecision(6)) : 10} onChange={(x) => put({ ...node, frac: (x ?? 10) / 100, k: undefined })} />
          ) : (
            <NumberField label="Cells" min={1} step={1} value={node.k ?? 100} onChange={(x) => put({ ...node, k: Math.max(1, Math.round(x ?? 100)), frac: undefined })} />
          )}
          <button type="button" className="btn small ghost" onClick={() => put(byFrac ? { ...node, frac: undefined, k: 100 } : { ...node, k: undefined, frac: 0.1 })}>
            {byFrac ? "use a count" : "use a share"}
          </button>
          <span className="cap">by</span>
          <input list={`${ctx.idPrefix}-columns`} value={node.column} aria-label="Rank by column" placeholder="column" onChange={(e) => put({ ...node, column: e.target.value })} />
          {node.within ? (
            <div className="sel-child">
              <span className="eyebrow">within</span>
              <NodeEditor node={node.within} path={[...path, "within"]} ctx={ctx} />
            </div>
          ) : (
            <button type="button" className="btn small ghost" onClick={() => put({ ...node, within: { kind: "zones", values: ctx.grid.meta.zones.slice(0, 1) } })}>
              limit to…
            </button>
          )}
        </div>
      );
    }
    case "buffer": {
      const byLever = node.lever_range !== undefined;
      return (
        <div className="sel-form">
          {byLever ? (
            <select aria-label="Lever range" value={node.lever_range} onChange={(e) => put({ ...node, lever_range: e.target.value })}>
              {ctx.levers.map((l) => (
                <option key={l.var} value={l.var}>
                  {l.label} influence range
                </option>
              ))}
            </select>
          ) : (
            <NumberField label="Radius" unit="m" min={1} value={node.radius_m ?? 120} onChange={(x) => put({ ...node, radius_m: Math.max(1, x ?? 120) })} />
          )}
          <button
            type="button"
            className="btn small ghost"
            onClick={() => put(byLever ? { kind: "buffer", of: node.of, radius_m: 120 } : { kind: "buffer", of: node.of, lever_range: ctx.levers[0]?.var ?? "" })}
            disabled={!byLever && !ctx.levers.length}
          >
            {byLever ? "use metres" : "use a lever's range"}
          </button>
          <div className="sel-child">
            <span className="eyebrow">around</span>
            <NodeEditor node={node.of} path={[...path, "of"]} ctx={ctx} />
          </div>
        </div>
      );
    }
    case "region":
      return (
        <select aria-label="Saved region" value={node.id} onChange={(e) => put({ kind: "region", id: e.target.value })}>
          {!ctx.regions.some((r) => r.id === node.id) ? <option value={node.id}>{node.id}</option> : null}
          {ctx.regions.map((r) => (
            <option key={r.id} value={r.id}>
              {r.name} ({fmtInt(r.n_cells)} cells{r.portable === false ? ", this run only" : ""})
            </option>
          ))}
        </select>
      );
    default:
      return <span className="cap">{describeSelection(node, { regionName: (rid) => ctx.regions.find((r) => r.id === rid)?.name ?? null })}</span>;
  }
}

function NodeEditor({ node, path, ctx }: { node: SelectionSpec; path: SelPath; ctx: Ctx }) {
  const remove = () => ctx.set(replaceSelection(ctx.root, path, null));
  const wrap = (op: "and" | "or" | "minus" | "not") => ctx.set(wrapSelection(ctx.root, path, op, op === "not" ? undefined : defaultFilter(ctx.columns)));
  const actions = (
    <span className="sel-actions">
      <select
        aria-label="Combine this condition"
        value=""
        onChange={(e) => {
          const v = e.target.value;
          if (v === "buffer") ctx.set(replaceSelection(ctx.root, path, { kind: "buffer", of: node, radius_m: 120 }));
          else if (v) wrap(v as "and" | "or" | "minus" | "not");
          e.target.value = "";
        }}
      >
        <option value="">Combine…</option>
        <option value="and">… and another condition</option>
        <option value="or">… or another condition</option>
        <option value="minus">… minus another condition</option>
        <option value="not">… not (invert)</option>
        <option value="buffer">… plus a buffer around it</option>
      </select>
      {path.length || !("kind" in node && node.kind === "all") ? <IconButton icon="x" size="small" label="Remove this condition" onClick={remove} /> : null}
    </span>
  );
  if (!("kind" in node)) {
    if (node.op === "not") {
      return (
        <div className="sel-node" data-op="not">
          <div className="row">
            <strong>not</strong>
            {actions}
          </div>
          <div className="sel-child">
            <NodeEditor node={node.arg} path={[...path, "arg"]} ctx={ctx} />
          </div>
        </div>
      );
    }
    return (
      <div className="sel-node" data-op={node.op}>
        <div className="row">
          <select aria-label="Combinator" value={node.op} onChange={(e) => ctx.set(replaceSelection(ctx.root, path, { op: e.target.value as "and" | "or" | "minus", args: node.args }))}>
            <option value="and">all of (and)</option>
            <option value="or">any of (or)</option>
            <option value="minus">first, minus the rest</option>
          </select>
          {actions}
        </div>
        {node.args.map((a, i) => (
          <div className="sel-child" key={i}>
            <NodeEditor node={a} path={[...path, i]} ctx={ctx} />
          </div>
        ))}
        <button type="button" className="btn small ghost" onClick={() => ctx.set(replaceSelection(ctx.root, path, { op: node.op, args: [...node.args, defaultFilter(ctx.columns)] }))}>
          + condition
        </button>
      </div>
    );
  }
  return (
    <div className="sel-node" data-kind={node.kind}>
      <div className="row">
        <span className="eyebrow">{node.kind === "top" ? "top" : node.kind}</span>
        {actions}
      </div>
      <PrimitiveEditor node={node} path={path} ctx={ctx} />
    </div>
  );
}

/** "1,284 cells · 1.16 km² · 6,420 residents · median canopy 12" */
export function countText(r: SelectionReply, columnLabel: (c: string) => string = (c) => c): string {
  const parts = [`${fmtInt(r.n_cells)} cells`, `${fmtNum(r.area_km2, 2)} km²`];
  if (r.people !== null && Number.isFinite(r.people)) parts.push(`${fmtInt(Math.round(r.people))} residents`);
  for (const [k, v] of Object.entries(r.medians).slice(0, 2)) if (v !== null && Number.isFinite(v)) parts.push(`median ${columnLabel(k)} ${fmtNum(v, 2)}`);
  return parts.join(" · ");
}

export function SelectionBuilder(props: SelectionBuilderProps) {
  const { rid, grid, value, onChange, columns, regions, levers, mapPick, onMask, idPrefix } = props;
  const root: SelectionSpec = value ?? { kind: "all" };
  const [reply, setReply] = useState<SelectionReply | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [counting, setCounting] = useState(false);
  const [open, setOpen] = useState(false);
  const [how, setHow] = useState<"replace" | "and" | "or" | "minus">("and");
  const ctrl = useRef<AbortController | null>(null);
  const problems = useMemo(() => validateSelection(root, "where"), [root]);
  const key = JSON.stringify(root);
  const columnLabel = (c: string) => columns.find((o) => o.value === c || o.value.endsWith(":" + c))?.label ?? c;

  const count = useMemo(
    () =>
      debounce((spec: SelectionSpec) => {
        ctrl.current?.abort();
        const c = new AbortController();
        ctrl.current = c;
        setCounting(true);
        resolveSelection(rid, spec, c.signal).then(
          (r) => {
            if (c.signal.aborted) return;
            setReply(r);
            setError(null);
            setCounting(false);
            onMask?.(selectionMask(r, grid.n));
          },
          (e: unknown) => {
            if (c.signal.aborted) return;
            setError(errorMessage(e));
            setReply(null);
            setCounting(false);
          },
        );
      }, 300),
    [rid, grid.n, onMask],
  );

  useEffect(() => {
    if (problems.length) {
      count.cancel();
      setReply(null);
      return;
    }
    count(root);
    return () => count.cancel();
    // `key` (the spec's JSON) stands for `root`, which is a new object on every render.
  }, [key, problems.length, count]);

  useEffect(() => () => ctrl.current?.abort(), []);

  const ctx: Ctx = { grid, columns, regions, levers, root, set: onChange, idPrefix };
  const add = (node: SelectionSpec) => onChange(combineSelection(value, node, how));
  const [naming, setNaming] = useState<string | null>(null);
  const save = async (name: string) => {
    if (!name.trim()) return;
    try {
      await createRegion(rid, name.trim(), root);
      invalidate(`run:${rid}:regions`);
      toast("success", `Saved region "${name.trim()}"`);
      setNaming(null);
    } catch (e) {
      toast("error", "Could not save the region", { body: errorMessage(e) });
    }
  };

  return (
    <div className="selbuilder" onFocus={() => reply && onMask?.(selectionMask(reply, grid.n))}>
      <datalist id={`${idPrefix}-columns`}>
        {columns.map((c) => (
          <option key={c.value} value={c.value}>
            {c.label} · {c.group}
          </option>
        ))}
      </datalist>
      <div className="row sel-summary">
        <span className="sel-desc">{describeSelection(root, { regionName: (id) => regions.find((r) => r.id === id)?.name ?? null, columnLabel })}</span>
        <button type="button" className="btn small ghost" aria-expanded={open} onClick={() => setOpen((v) => !v)}>
          {open ? "Done" : "Edit"}
        </button>
      </div>
      <p className="cap sel-count" aria-live="polite" data-testid="sel-count">
        {problems.length ? problems[0] : error ? `Could not count: ${error}` : reply ? countText(reply, columnLabel) : counting ? "Counting…" : ""}
        {reply && !reply.portable ? " · this run only (no CRS)" : ""}
        {reply?.n_cells === 0 ? " · empty selection" : ""}
      </p>
      {reply?.warnings.length ? <p className="cap">{reply.warnings.join(" ")}</p> : null}
      {open ? (
        <div className="stack" style={{ gap: 8 }}>
          <NodeEditor node={root} path={[]} ctx={ctx} />
          <div className="row">
            <select aria-label="How to add" value={how} onChange={(e) => setHow(e.target.value as typeof how)}>
              <option value="and">Add with and</option>
              <option value="or">Add with or</option>
              <option value="minus">Subtract</option>
              <option value="replace">Replace</option>
            </select>
            <Button size="small" onClick={() => add({ kind: "zones", values: grid.meta.zones.slice(0, 1) })} disabled={!grid.meta.zones.length}>
              Zone
            </Button>
            <Button size="small" onClick={() => add(defaultFilter(columns))}>
              Filter
            </Button>
            <Button size="small" onClick={() => add({ kind: "top", column: columns[0]?.value ?? "pred:target", frac: 0.1, direction: "highest" })}>
              Top %
            </Button>
            <Button size="small" onClick={() => add({ kind: "region", id: regions[0]?.id ?? "" })} disabled={!regions.length} title={regions.length ? undefined : "No saved regions yet"}>
              Region
            </Button>
            {mapPick ? (
              <Button size="small" variant="primary" onClick={() => add(mapPick.spec)} title={mapPick.label}>
                Map selection
              </Button>
            ) : null}
          </div>
          <div className="row">
            {naming !== null ? (
              <>
                <input aria-label="Region name" value={naming} onChange={(e) => setNaming(e.target.value)} onKeyDown={(e) => e.key === "Enter" && void save(naming)} />
                <Button size="small" variant="primary" onClick={() => void save(naming)} disabled={!naming.trim()}>
                  Save region
                </Button>
                <Button size="small" variant="ghost" onClick={() => setNaming(null)}>
                  Cancel
                </Button>
              </>
            ) : (
              <Button size="small" variant="ghost" onClick={() => setNaming(describeSelection(root))} disabled={problems.length > 0}>
                Save as region
              </Button>
            )}
            <Button size="small" variant="ghost" onClick={() => onChange({ kind: "all" })}>
              Reset to all cells
            </Button>
          </div>
        </div>
      ) : mapPick ? (
        <div className="row">
          <span className="cap">Map: {mapPick.label}</span>
          <Button size="small" onClick={() => onChange(mapPick.spec)}>
            Use
          </Button>
          <Button size="small" variant="ghost" onClick={() => onChange(combineSelection(value, mapPick.spec, "and"))}>
            and
          </Button>
          <Button size="small" variant="ghost" onClick={() => onChange(combineSelection(value, mapPick.spec, "or"))}>
            or
          </Button>
        </div>
      ) : null}
    </div>
  );
}
