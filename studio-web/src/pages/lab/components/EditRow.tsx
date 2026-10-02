// One edit of the scenario (SPEC §7.3): lever (with its unit, bounds, dose scale and preview
// trust), mode, amount and where (SelectionBuilder), plus the compile warnings for this edit.
import type { CompileWarning, Edit, EditMode, Lever, Region } from "../../../api/lab";
import { EDIT_MODES } from "../../../api/lab";
import type { SelectionSpec } from "../../../api/types";
import { IconButton } from "../../../components/ui/IconButton";
import { NumberField } from "../../../components/ui/NumberField";
import type { GridData } from "../../../map/grid";
import { fmtNum, unitLabel } from "../../../theme/format";
import { MODE_INFO, editProblems, withMode } from "../model/doc";
import { SelectionBuilder, type ColumnOption, type MapPick } from "./SelectionBuilder";
import { TrustBadge } from "./TrustBadge";

export type EditRowProps = {
  rid: string;
  index: number;
  count: number;
  edit: Edit;
  levers: Lever[];
  expert: boolean;
  grid: GridData;
  columns: ColumnOption[];
  regions: Region[];
  planRefs: { ref: string; label: string }[];
  warnings: CompileWarning[];
  mapPick: MapPick | null;
  onChange: (e: Edit, coalesce?: string) => void;
  onRemove: () => void;
  onMove: (delta: -1 | 1) => void;
  onDuplicate: () => void;
  onLadder?: () => void;
  onMask: (mask: Uint8Array | null) => void;
  /** Predictor columns for expert-mode edits of non-lever predictors. */
  predictors: string[];
};

/** "0–100 pp · design dose 10 · 1 sd = 12" */
export function leverFacts(l: Lever | undefined): string {
  if (!l) return "not an actionable lever (expert mode)";
  const u = unitLabel(l.unit);
  const parts: string[] = [];
  if (l.min !== null || l.max !== null) parts.push(`${l.min !== null ? fmtNum(l.min, 2) : "−∞"}–${l.max !== null ? fmtNum(l.max, 2) : "∞"}${u ? " " + u : ""}`);
  parts.push(l.direction === "increase" ? "increases cool" : "decreases cool");
  if (l.design_dose !== null) parts.push(`design dose ${fmtNum(l.design_dose, 2)}`);
  if (l.sd !== null) parts.push(`1 sd = ${fmtNum(l.sd, 2)}`);
  return parts.join(" · ");
}

export function EditRow(p: EditRowProps) {
  const { edit, levers, index } = p;
  const lever = levers.find((l) => l.var === edit.lever);
  const info = MODE_INFO[edit.mode];
  const problems = editProblems(edit);
  const id = `edit-${index}`;
  const unit = info.unit === "lever" ? lever?.unit ?? "" : info.unit === "factor" ? "×" : info.unit === "percentile" ? "pct" : "";
  const set = (patch: Partial<Edit>, coalesce?: string) => p.onChange({ ...edit, ...patch }, coalesce);
  const modeOk = (m: EditMode) => m !== "fill_headroom" || (lever?.role === "canopy" && lever.headroom_available);
  return (
    <article className="edit-row card tight" aria-label={`Edit ${index + 1}: ${lever?.label ?? edit.lever}`}>
      <header className="row">
        <span className="eyebrow">Edit {index + 1}</span>
        {edit.label ? <span className="cap">{edit.label}</span> : null}
        <span className="spacer" />
        <IconButton icon="chevronUp" size="small" label="Move up" disabled={index === 0} onClick={() => p.onMove(-1)} />
        <IconButton icon="chevronDown" size="small" label="Move down" disabled={index === p.count - 1} onClick={() => p.onMove(1)} />
        <IconButton icon="copy" size="small" label="Duplicate edit" onClick={p.onDuplicate} />
        {p.onLadder && info.field !== "per_cell_ref" ? <IconButton icon="layers" size="small" label="Make a ladder of this edit" onClick={p.onLadder} /> : null}
        <IconButton icon="x" size="small" label="Remove edit" onClick={p.onRemove} />
      </header>
      <div className="edit-grid">
        <label htmlFor={`${id}-lever`}>Lever</label>
        <div className="row">
          {p.expert ? (
            <>
              <input id={`${id}-lever`} list={`${id}-predictors`} value={edit.lever} onChange={(e) => set({ lever: e.target.value }, `lever-${index}`)} />
              <datalist id={`${id}-predictors`}>
                {[...levers.map((l) => l.var), ...p.predictors.filter((c) => !levers.some((l) => l.var === c))].map((c) => (
                  <option key={c} value={c} />
                ))}
              </datalist>
            </>
          ) : (
            <select id={`${id}-lever`} value={edit.lever} onChange={(e) => set({ lever: e.target.value })}>
              {!lever ? <option value={edit.lever}>{edit.lever || "Choose…"}</option> : null}
              {levers.map((l) => (
                <option key={l.var} value={l.var}>
                  {l.label}
                  {l.unit ? ` (${unitLabel(l.unit)})` : ""}
                </option>
              ))}
            </select>
          )}
          {lever ? <TrustBadge trust={lever.emulator.trust} relErr={lever.emulator.uniform_rel_err} /> : null}
        </div>
        <span />
        <span className="cap">{leverFacts(lever)}</span>
        <label htmlFor={`${id}-mode`}>Mode</label>
        <select id={`${id}-mode`} value={edit.mode} onChange={(e) => p.onChange(withMode(edit, e.target.value as EditMode, e.target.value === "add" ? lever?.design_dose ?? 0 : undefined))}>
          {EDIT_MODES.map((m) => (
            <option key={m} value={m} disabled={!modeOk(m) && m !== edit.mode} title={!modeOk(m) ? "Canopy only, and needs the planner layers" : undefined}>
              {MODE_INFO[m].label}
            </option>
          ))}
        </select>
        <span />
        <span className="cap">{info.hint}</span>
        <label htmlFor={`${id}-amount`}>{info.fieldLabel}</label>
        {info.field === "amount" ? (
          <NumberField
            id={`${id}-amount`}
            value={edit.amount ?? null}
            unit={unit}
            min={edit.mode === "fill_headroom" ? 0 : undefined}
            max={edit.mode === "fill_headroom" ? 1 : undefined}
            step={edit.mode === "fill_headroom" ? 0.05 : edit.mode === "scale" ? 0.05 : 1}
            nullable
            onChange={(v) => set({ amount: v ?? undefined }, `amount-${index}`)}
          />
        ) : info.field === "percentile" ? (
          <NumberField id={`${id}-amount`} value={edit.percentile ?? null} min={0} max={100} step={5} unit="pct" nullable onChange={(v) => set({ percentile: v ?? undefined }, `amount-${index}`)} />
        ) : (
          <>
            <input id={`${id}-amount`} list={`${id}-refs`} value={edit.per_cell_ref ?? ""} placeholder="plan:<id> · blob:<id> · csv:<path>" onChange={(e) => set({ per_cell_ref: e.target.value || undefined }, `ref-${index}`)} />
            <datalist id={`${id}-refs`}>
              {p.planRefs.map((r) => (
                <option key={r.ref} value={r.ref}>
                  {r.label}
                </option>
              ))}
            </datalist>
          </>
        )}
        {edit.mode === "fill_headroom" ? (
          <>
            <label htmlFor={`${id}-paved`}>Paved share</label>
            <NumberField id={`${id}-paved`} value={edit.paved_share ?? null} min={0} max={1} step={0.05} nullable placeholder="config default" onChange={(v) => set({ paved_share: v ?? undefined }, `paved-${index}`)} />
          </>
        ) : null}
        <label htmlFor={`${id}-label`}>Label</label>
        <input id={`${id}-label`} value={edit.label ?? ""} placeholder="optional, e.g. street trees" onChange={(e) => set({ label: e.target.value || undefined }, `label-${index}`)} />
      </div>
      <div className="edit-where">
        <span className="eyebrow">Where</span>
        <SelectionBuilder
          rid={p.rid}
          grid={p.grid}
          value={edit.where}
          onChange={(where: SelectionSpec) => set({ where })}
          columns={p.columns}
          regions={p.regions}
          levers={levers}
          mapPick={p.mapPick}
          onMask={p.onMask}
          idPrefix={id}
        />
      </div>
      {problems.length || p.warnings.length ? (
        <ul className="edit-issues">
          {problems.map((m) => (
            <li key={m} data-level="error">
              {m}
            </li>
          ))}
          {p.warnings.map((w, i) => (
            <li key={i} data-level={w.blocking ? "error" : "warn"}>
              {w.blocking ? "Blocked: " : ""}
              {w.message}
            </li>
          ))}
        </ul>
      ) : null}
    </article>
  );
}
