// Step 02 · Where: presets (Whole city · A neighbourhood · Hottest places · Where most people are
// exposed · Draw on the map) plus "Custom filter…", which opens the existing SelectionBuilder for
// this change. Presets build the selection trees of model/flow.ts (shapes the server already
// accepts); a preset whose data the run lacks is not offered. Contract: ./types.ts.
import { useEffect, useMemo, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import type { SelectionSpec } from "../../../../api/types";
import { Button } from "../../../../components/ui/Button";
import { NumberField } from "../../../../components/ui/NumberField";
import { StageNum } from "../../../../components/ui/StageRail";
import {
  FLOW_WORDS,
  TOP_SHARE,
  fill,
  neighbourhoodOf,
  presetSelection,
  toggleNeighbourhood,
  topShareOf,
  wherePhrase,
  wherePresetOf,
  wherePresets,
  withTopShare,
  type SentenceContext,
} from "../../model/flow";
import type { WorkbenchModel } from "../../model/useWorkbench";
import { LAB_WORDS, WHERE_PRESETS, type WherePreset } from "../../model/words";
import { SelectionBuilder } from "../SelectionBuilder";
import type { StepProps } from "./types";
import "./flow.css";

export type StepWhereProps = StepProps & {
  /** Told whether "Draw on the map" is the chosen preset (false again when the step closes), so
   *  the page can show the map's Brush and Pick settings only while drawing. */
  onDrawingChange?: (drawing: boolean) => void;
};

/** The sentence and preset context of a workbench (zones, saved areas, preset columns, map pick, levers). */
export function flowContext(model: Pick<WorkbenchModel, "grid" | "regions" | "temperatureColumn" | "populationColumn" | "mapPick" | "columns" | "levers" | "planRefs">): SentenceContext {
  return {
    zones: model.grid.meta.zones,
    regions: model.regions,
    temperatureColumn: model.temperatureColumn,
    populationColumn: model.populationColumn,
    mapPick: model.mapPick,
    columns: model.columns,
    levers: model.levers,
    planRefs: model.planRefs,
  };
}

export type ChoiceOption<T extends string> = { value: T; label: ReactNode; hint?: string; disabled?: boolean };

/**
 * A single choice as wrapping pills (role="radiogroup"): arrow keys, Home and End move the choice,
 * one pill is in the tab order (the chosen one, else the first). With no option chosen (a change
 * that reads as none of them) nothing is checked and the first pill takes focus.
 */
export function ChoiceGroup<T extends string>({ label, value, options, onChange, className }: { label: string; value: T | null; options: readonly ChoiceOption<T>[]; onChange: (v: T) => void; className?: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const enabled = options.filter((o) => !o.disabled);
  const chosen = enabled.find((o) => o.value === value) ?? null;
  const tabbable = chosen ?? enabled[0] ?? null;
  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    if (!enabled.length) return;
    const i = Math.max(0, enabled.findIndex((o) => o.value === (chosen ?? tabbable)?.value));
    let next: ChoiceOption<T> | undefined;
    if (e.key === "ArrowRight" || e.key === "ArrowDown") next = enabled[(i + 1) % enabled.length];
    else if (e.key === "ArrowLeft" || e.key === "ArrowUp") next = enabled[(i + enabled.length - 1) % enabled.length];
    else if (e.key === "Home") next = enabled[0];
    else if (e.key === "End") next = enabled[enabled.length - 1];
    if (!next) return;
    e.preventDefault();
    if (next.value !== value) onChange(next.value);
    (ref.current?.querySelectorAll<HTMLButtonElement>("button")[options.indexOf(next)])?.focus();
  };
  return (
    <div ref={ref} className={["lab-flow-choices", className ?? ""].filter(Boolean).join(" ")} role="radiogroup" aria-label={label} onKeyDown={onKey}>
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={o.value === value}
          disabled={o.disabled}
          title={o.hint}
          tabIndex={o === tabbable ? 0 : -1}
          data-value={o.value}
          onClick={() => o.value !== value && onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

/**
 * Where change `index` applies. Presets write `edit.where` through `model.setEdit`. "A
 * neighbourhood" picks saved areas and zones (several combine with "or"); "Hottest places" and
 * "Where most people are exposed" take the top share by `model.temperatureColumn` /
 * `model.populationColumn` and are only offered when those exist. "Draw on the map" uses
 * `model.mapPick`, and while it is chosen each new shape or pick on the map becomes the place.
 * "Custom filter…" opens the full SelectionBuilder (idPrefix `${idPrefix}-where`, map tint via
 * `model.setTint`); a selection no preset can show opens it directly.
 */
export function StepWhere({ model, index, idPrefix, onDrawingChange }: StepWhereProps) {
  const edit = model.doc.edits[index];
  const [builderOpen, setBuilderOpen] = useState(false);
  // A preset chosen before it could build a selection (drawing before anything is drawn), tied to
  // the selection it was chosen on: once the selection changes (a pick, undo) the selection decides.
  const [wanted, setWanted] = useState<{ preset: WherePreset; key: string } | null>(null);
  const ctx = useMemo(
    () => flowContext(model),
    [model.grid, model.regions, model.temperatureColumn, model.populationColumn, model.mapPick, model.columns, model.levers, model.planRefs],
  );
  const whereKey = JSON.stringify(edit?.where ?? null);
  const read = edit ? wherePresetOf(edit.where, ctx) : "city";
  const current: WherePreset = wanted && wanted.key === whereKey ? wanted.preset : read;
  const setWhere = (where: SelectionSpec, coalesce?: string) => edit && model.setEdit(index, { ...edit, where }, coalesce);

  // Tell the page while "Draw on the map" is chosen (and stop when this step closes).
  const drawing = !!edit && current === "draw";
  useEffect(() => {
    if (!onDrawingChange) return;
    onDrawingChange(drawing);
    return () => onDrawingChange(false);
  }, [drawing, onDrawingChange]);

  // "Draw on the map": a new shape or pick on the map becomes the place of this change.
  const lastPick = useRef(model.mapPick);
  const drawRef = useRef({ edit, current, setWhere });
  drawRef.current = { edit, current, setWhere };
  useEffect(() => {
    if (model.mapPick === lastPick.current) return;
    lastPick.current = model.mapPick;
    const { edit: e, current: c, setWhere: set } = drawRef.current;
    if (e && model.mapPick && c === "draw" && JSON.stringify(model.mapPick.spec) !== JSON.stringify(e.where ?? null)) set(model.mapPick.spec);
  }, [model.mapPick]);

  const headingId = `${idPrefix}-where-head`;
  const head = (
    <h3 className="lab-step-head" id={headingId}>
      <StageNum index={1} count={3} status={edit ? "done" : "todo"} />
      {LAB_WORDS.stepWhere}
    </h3>
  );
  if (!edit) {
    return (
      <section className="lab-step flow-step" aria-labelledby={headingId} data-step="where" data-pending="true">
        {head}
        <p className="cap">{FLOW_WORDS.pickFirst}</p>
      </section>
    );
  }

  const presets = wherePresets(ctx);
  const choose = (p: WherePreset) => {
    const s = presetSelection(p, ctx, topShareOf(edit.where) ?? TOP_SHARE);
    if (s && JSON.stringify(s) !== whereKey) {
      setWanted(null);
      setWhere(s);
    } else setWanted({ preset: p, key: whereKey });
  };
  const custom = current === "custom";
  const showBuilder = builderOpen || custom;
  const closeBuilder = () => {
    setBuilderOpen(false);
    model.setTint(null);
  };
  const phrase = wherePhrase(edit.where, ctx);

  let detail: ReactNode = null;
  if (current === "neighbourhood") {
    const n = neighbourhoodOf(edit.where) ?? { zones: [], regions: [] };
    const toggle = (item: { zone: number | string } | { region: string }) => {
      const next = toggleNeighbourhood(edit.where, item);
      if (next) setWhere(next);
    };
    detail = (
      <div className="chips flow-step-areas" role="group" aria-label={FLOW_WORDS.neighbourhoods}>
        {ctx.regions.map((r) => (
          <button key={`r:${r.id}`} type="button" className="chip" aria-pressed={n.regions.includes(r.id)} onClick={() => toggle({ region: r.id })}>
            {r.name}
          </button>
        ))}
        {ctx.zones.map((z) => (
          <button key={`z:${String(z)}`} type="button" className="chip" aria-pressed={n.zones.some((v) => String(v) === String(z))} onClick={() => toggle({ zone: z })}>
            Zone {String(z)}
          </button>
        ))}
      </div>
    );
  } else if ((current === "hottest" || current === "people") && edit.where && "kind" in edit.where && edit.where.kind === "top") {
    const top = edit.where;
    const share = topShareOf(top);
    detail = (
      <div className="flow-step-inline">
        <label htmlFor={`${idPrefix}-share`}>{current === "hottest" ? FLOW_WORDS.hottestShare : FLOW_WORDS.peopleShare}</label>
        <NumberField
          id={`${idPrefix}-share`}
          value={share !== null ? Number((share * 100).toPrecision(6)) : null}
          unit="%"
          min={1}
          max={100}
          step={1}
          onChange={(v) => v !== null && setWhere(withTopShare(top, v / 100), `share-${index}`)}
        />
        <span>{FLOW_WORDS.ofTheCity}</span>
      </div>
    );
  } else if (current === "draw") {
    const pick = model.mapPick;
    const using = pick && JSON.stringify(pick.spec) === whereKey;
    detail = (
      <div className="flow-step-detail">
        <p className="cap">
          {WHERE_PRESETS.draw.hint} {FLOW_WORDS.brushHint}
        </p>
        {pick && !using ? (
          <Button size="small" onClick={() => setWhere(pick.spec)}>
            {fill(FLOW_WORDS.usePick, { label: pick.label })}
          </Button>
        ) : null}
        {read !== "draw" ? <p className="cap flow-step-pending">{fill(FLOW_WORDS.drawPending, { where: phrase })}</p> : null}
      </div>
    );
  }

  return (
    <section className="lab-step flow-step" aria-labelledby={headingId} data-step="where" data-preset={current}>
      {head}
      <ChoiceGroup<WherePreset>
        label={LAB_WORDS.stepWhere}
        value={custom ? null : current}
        onChange={choose}
        options={presets.map((p) => ({ value: p, label: WHERE_PRESETS[p].label, hint: WHERE_PRESETS[p].hint }))}
      />
      {detail}
      {!custom && (current !== "draw" || read === "draw") ? <p className="cap flow-step-applies">{fill(FLOW_WORDS.applies, { where: phrase })}</p> : null}
      {custom ? null : (
        <button type="button" className="flow-step-link" aria-expanded={showBuilder} aria-controls={`${idPrefix}-where-custom`} title={WHERE_PRESETS.custom.hint} onClick={() => (showBuilder ? closeBuilder() : setBuilderOpen(true))}>
          {showBuilder ? FLOW_WORDS.customClose : WHERE_PRESETS.custom.label}
        </button>
      )}
      {showBuilder ? (
        <div className="flow-step-custom" id={`${idPrefix}-where-custom`}>
          <SelectionBuilder
            rid={model.rid}
            grid={model.grid}
            value={edit.where}
            onChange={(s) => setWhere(s)}
            columns={model.columns}
            regions={model.regions}
            levers={model.levers}
            mapPick={model.mapPick}
            onMask={model.setTint}
            idPrefix={`${idPrefix}-where`}
          />
        </div>
      ) : null}
    </section>
  );
}
