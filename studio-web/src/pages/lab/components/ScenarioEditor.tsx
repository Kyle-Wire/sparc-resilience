// Left pane of the Design workbench (SPEC §7.2–§7.4): the scenario's name, tags and notes,
// its edits (EditRow each), brushed layers, reporting regions, cost model and options, plus
// undo/redo (50 steps), the autosave status and design CSV import/export.
import { useRef, useState } from "react";
import { useStore } from "zustand";
import { errorMessage } from "../../../api/client";
import { designCsvUrl, importDesignCsv, type CompileResult, type Edit, type Lever, type Region } from "../../../api/lab";
import type { SelectionSpec } from "../../../api/types";
import { Button } from "../../../components/ui/Button";
import { IconButton } from "../../../components/ui/IconButton";
import { NumberField } from "../../../components/ui/NumberField";
import type { GridData } from "../../../map/grid";
import { toast } from "../../../stores/ui";
import { fmtInt, unitLabel } from "../../../theme/format";
import { newEdit } from "../model/doc";
import { canRedo, canUndo, isDirty, type DraftStore } from "../model/draft";
import { EditRow } from "./EditRow";
import { SelectionBuilder, type ColumnOption, type MapPick } from "./SelectionBuilder";

export type ScenarioEditorProps = {
  rid: string;
  store: DraftStore;
  levers: Lever[];
  grid: GridData;
  columns: ColumnOption[];
  regions: Region[];
  planRefs: { ref: string; label: string }[];
  predictors: string[];
  compile: CompileResult | null;
  mapPick: MapPick | null;
  onMask: (mask: Uint8Array | null) => void;
  onLadder?: (editIndex: number) => void;
  /** Called when focus leaves the editor (autosave on blur). */
  onBlur?: () => void;
};

function SaveStatus({ store }: { store: DraftStore }) {
  const s = {
    save: useStore(store, (x) => x.save),
    error: useStore(store, (x) => x.saveError),
    dirty: useStore(store, isDirty),
    sid: useStore(store, (x) => x.sid),
    revision: useStore(store, (x) => x.revision),
  };
  const text = s.save === "saving" ? "Saving…" : s.save === "error" ? `Not saved: ${s.error ?? "error"}` : s.dirty ? (s.sid ? "Unsaved changes" : "Not saved yet") : s.sid ? "All changes saved" : "New draft";
  return (
    <span className="cap save-status" role="status" data-save={s.save} data-dirty={s.dirty || undefined}>
      {text}
      {s.revision ? ` · revision ${s.revision}` : ""}
    </span>
  );
}

export function ScenarioEditor(p: ScenarioEditorProps) {
  const { store, levers } = p;
  const doc = useStore(store, (s) => s.doc);
  const undoable = useStore(store, canUndo);
  const redoable = useStore(store, canRedo);
  const sid = useStore(store, (s) => s.sid);
  const brushVersion = useStore(store, (s) => s.brush);
  const change = store.getState().change;
  const [newLever, setNewLever] = useState<string>("");
  const [regionName, setRegionName] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);
  const expert = !!doc.options?.expert;
  const warningsFor = (i: number) => p.compile?.warnings.filter((w) => w.edit_index === i) ?? [];
  const global = p.compile?.warnings.filter((w) => w.edit_index === null) ?? [];

  const setEdit = (i: number, e: Edit, coalesce?: string) => change((d) => ({ ...d, edits: d.edits.map((x, k) => (k === i ? e : x)) }), { coalesce });
  const addEdit = () => {
    const lever = levers.find((l) => l.var === newLever) ?? levers[0];
    if (!lever) return;
    change((d) => ({ ...d, edits: [...d.edits, newEdit(lever.var, "add", lever.design_dose ?? 0)] }));
  };

  const importCsv = async (file: File) => {
    try {
      const r = await importDesignCsv(p.rid, file);
      change((d) => ({
        ...d,
        edits: [...d.edits, ...Object.entries(r.blobs).map(([lever, blob]): Edit => ({ lever, mode: "per_cell", per_cell_ref: `blob:${blob}`, label: `design CSV (${file.name})` }))],
      }));
      toast(r.unknown_ids.length ? "warning" : "success", `Imported ${fmtInt(r.n_rows)} rows for ${r.levers.join(", ")}`, {
        body: `${r.mode === "value" ? "Absolute values were converted to changes on this run. " : ""}${r.unknown_ids.length ? `${fmtInt(r.unknown_ids.length)} ids are not in this run and were skipped.` : ""}` || undefined,
      });
    } catch (e) {
      toast("error", "Could not import the design CSV", { body: errorMessage(e) });
    }
  };

  const layers = store.brushLayers;
  const brushed = layers.levers();
  void brushVersion; // re-render when the committed brush changes

  return (
    <div
      className="scenario-editor stack"
      onBlur={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget as Node | null)) p.onBlur?.();
      }}
    >
      <div className="row">
        <IconButton icon="undo" label="Undo (Ctrl+Z)" disabled={!undoable} onClick={() => store.getState().undo()} />
        <IconButton icon="redo" label="Redo (Ctrl+Shift+Z)" disabled={!redoable} onClick={() => store.getState().redo()} />
        <span className="spacer" />
        <SaveStatus store={store} />
      </div>
      <label className="field">
        <span className="field-label">Name</span>
        <input value={doc.name} onChange={(e) => change((d) => ({ ...d, name: e.target.value }), { coalesce: "name" })} />
      </label>
      <label className="field">
        <span className="field-label">Tags (comma separated)</span>
        <input
          defaultValue={(doc.tags ?? []).join(", ")}
          key={(doc.tags ?? []).join(",")}
          onBlur={(e) => change((d) => ({ ...d, tags: e.target.value.split(",").map((t) => t.trim()).filter(Boolean) }))}
        />
      </label>
      <label className="field">
        <span className="field-label">Notes</span>
        <textarea rows={2} value={doc.notes ?? ""} onChange={(e) => change((d) => ({ ...d, notes: e.target.value }), { coalesce: "notes" })} />
      </label>

      <section className="stack" style={{ gap: 8 }} aria-label="Edits">
        <h3>Edits</h3>
        {!doc.edits.length && !brushed.length ? <p className="cap">No edits yet. Add one below, brush on the map, or start from a template.</p> : null}
        {doc.edits.map((e, i) => (
          <EditRow
            key={i}
            rid={p.rid}
            index={i}
            count={doc.edits.length}
            edit={e}
            levers={levers}
            expert={expert}
            grid={p.grid}
            columns={p.columns}
            regions={p.regions}
            planRefs={p.planRefs}
            predictors={p.predictors}
            warnings={warningsFor(i)}
            mapPick={p.mapPick}
            onMask={p.onMask}
            onChange={(next, coalesce) => setEdit(i, next, coalesce)}
            onRemove={() => change((d) => ({ ...d, edits: d.edits.filter((_, k) => k !== i) }))}
            onDuplicate={() => change((d) => ({ ...d, edits: [...d.edits.slice(0, i + 1), JSON.parse(JSON.stringify(e)) as Edit, ...d.edits.slice(i + 1)] }))}
            onMove={(delta) =>
              change((d) => {
                const edits = [...d.edits];
                const j = i + delta;
                if (j < 0 || j >= edits.length) return d;
                [edits[i], edits[j]] = [edits[j], edits[i]];
                return { ...d, edits };
              })
            }
            onLadder={p.onLadder && sid ? () => p.onLadder!(i) : undefined}
          />
        ))}
        <div className="row">
          <select aria-label="Lever for the new edit" value={newLever || levers[0]?.var || ""} onChange={(e) => setNewLever(e.target.value)}>
            {levers.map((l) => (
              <option key={l.var} value={l.var}>
                {l.label}
              </option>
            ))}
          </select>
          <Button size="small" icon="plus" onClick={addEdit} disabled={!levers.length}>
            Add edit
          </Button>
        </div>
        {brushed.length ? (
          <div className="brushed card flat">
            <p className="eyebrow">Brushed on the map</p>
            {brushed.map((l) => (
              <div className="row" key={l}>
                <span>
                  {levers.find((x) => x.var === l)?.label ?? l}: {fmtInt(layers.editedCount(l))} cells
                </span>
                <span className="spacer" />
                <Button
                  size="small"
                  variant="ghost"
                  onClick={() => {
                    layers.clear(l);
                    store.getState().commitBrush();
                  }}
                >
                  Clear
                </Button>
              </div>
            ))}
            <p className="cap">Saved as a per-cell edit of each lever (label “brush”).</p>
          </div>
        ) : null}
        {global.length ? (
          <ul className="edit-issues">
            {global.map((w, i) => (
              <li key={i} data-level={w.blocking ? "error" : "warn"}>
                {w.message}
              </li>
            ))}
          </ul>
        ) : null}
      </section>

      <details className="card flat">
        <summary>Reporting regions ({Object.keys(doc.regions ?? {}).length})</summary>
        <p className="cap">Named areas reported separately in exact results (besides the edited cells, a ring around them and every zone).</p>
        {Object.entries(doc.regions ?? {}).map(([name, spec]) => (
          <div key={name} className="stack" style={{ gap: 4 }}>
            <div className="row">
              <strong>{name}</strong>
              <span className="spacer" />
              <IconButton
                icon="x"
                size="small"
                label={`Remove region ${name}`}
                onClick={() =>
                  change((d) => {
                    const regions = { ...(d.regions ?? {}) };
                    delete regions[name];
                    return { ...d, regions };
                  })
                }
              />
            </div>
            <SelectionBuilder
              rid={p.rid}
              grid={p.grid}
              value={spec}
              onChange={(s: SelectionSpec) => change((d) => ({ ...d, regions: { ...(d.regions ?? {}), [name]: s } }))}
              columns={p.columns}
              regions={p.regions}
              levers={levers}
              mapPick={p.mapPick}
              onMask={p.onMask}
              idPrefix={`region-${name.replace(/\W+/g, "-")}`}
            />
          </div>
        ))}
        <div className="row">
          <input aria-label="New region name" placeholder="Region name" value={regionName} onChange={(e) => setRegionName(e.target.value)} />
          <Button
            size="small"
            disabled={!regionName.trim() || !!doc.regions?.[regionName.trim()]}
            onClick={() => {
              const name = regionName.trim();
              change((d) => ({ ...d, regions: { ...(d.regions ?? {}), [name]: p.mapPick?.spec ?? { kind: "all" } } }));
              setRegionName("");
            }}
          >
            Add region{p.mapPick ? " from the map" : ""}
          </Button>
        </div>
      </details>

      <details className="card flat">
        <summary>Costs</summary>
        <p className="cap">Cost per unit of realised change, used for the cost estimate and cooling per cost.</p>
        {levers.map((l) => (
          <div className="row" key={l.var}>
            <span style={{ minWidth: 120 }}>{l.label}</span>
            <NumberField
              label={`Cost per unit of ${l.label}`}
              value={doc.costs?.[l.var]?.per_unit ?? null}
              placeholder={String(l.cost_per_unit)}
              unit={l.unit ? `per ${unitLabel(l.unit)}` : undefined}
              nullable
              onChange={(v) =>
                change((d) => {
                  const costs = { ...(d.costs ?? {}) };
                  if (v === null) delete costs[l.var];
                  else costs[l.var] = { per_unit: v };
                  return { ...d, costs };
                })
              }
            />
          </div>
        ))}
      </details>

      <details className="card flat">
        <summary>Options</summary>
        <label className="row">
          <input type="checkbox" checked={doc.options?.clip_to_support ?? true} onChange={(e) => change((d) => ({ ...d, options: { ...(d.options ?? {}), clip_to_support: e.target.checked } }))} />
          Clip to the observed support (never push a cell further outside its bounds)
        </label>
        <label className="row">
          <input type="checkbox" checked={doc.options?.mediators ?? true} onChange={(e) => change((d) => ({ ...d, options: { ...(d.options ?? {}), mediators: e.target.checked } }))} />
          Let mediators follow (e.g. NDVI follows canopy)
        </label>
        <label className="row">
          <input type="checkbox" checked={expert} onChange={(e) => change((d) => ({ ...d, options: { ...(d.options ?? {}), expert: e.target.checked } }))} />
          Expert mode (edit non-actionable predictors and mediators)
        </label>
      </details>

      <details className="card flat">
        <summary>Design CSV</summary>
        <p className="cap">
          Import per-cell changes (<span className="mono">id,lever,change</span>) or absolute targets (<span className="mono">id,lever,value</span>); export this scenario's realised per-cell edit.
        </p>
        <div className="row">
          <input
            ref={fileRef}
            type="file"
            accept=".csv,text/csv"
            hidden
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) void importCsv(f);
              e.target.value = "";
            }}
          />
          <Button size="small" icon="file" onClick={() => fileRef.current?.click()}>
            Import CSV
          </Button>
          {sid ? (
            <a className="btn small" href={designCsvUrl(sid, p.rid)} download>
              Export CSV
            </a>
          ) : (
            <span className="cap">Export after the first save.</span>
          )}
        </div>
      </details>
      <p className="cap">
        {doc.anchor_run_id ? `Designed on run ${doc.anchor_run_id}` : "Not tied to a run"}
        {sid ? ` · ${sid}` : ""}
      </p>
    </div>
  );
}
