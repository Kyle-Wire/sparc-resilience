// Step 03 · How much: the planner modes (Add · Bring up to at least · Use available planting space;
// for levers that cool by going down: Take away · Bring down to at most), a slider and a number in
// the lever's natural units, typical amounts, and the live sentence from the compile ("You asked
// for +20 pts tree canopy; about 14 pts fits — the rest is beyond what the space allows."). That
// sentence is the only place per-lever compile numbers show before running. Contract: ./types.ts.
import { useId } from "react";
import type { EditMode } from "../../../../api/lab";
import { NumberField } from "../../../../components/ui/NumberField";
import { StageNum } from "../../../../components/ui/StageRail";
import {
  FLOW_WORDS,
  amountLabel,
  amountScale,
  compileSentence,
  fill,
  fmtAmount,
  fromDisplay,
  modeHint,
  modeLabel,
  plannerModes,
  roundToStep,
  switchMode,
  toDisplay,
  typicalAmounts,
} from "../../model/flow";
import { LAB_WORDS, MODE_WORDS, plainAdvice } from "../../model/words";
import { ChoiceGroup } from "./StepWhere";
import type { StepProps } from "./types";
import "./flow.css";

export type StepHowMuchProps = StepProps;

/**
 * How much of change `index`. The mode control offers the lever's planner modes (model/flow.ts
 * plannerModes); a change in another mode (set, scale, to_percentile, per_cell, …) shows a note
 * pointing to Advanced options and a way back to a simple amount. The number input has id
 * `${idPrefix}-amount`; slider drags and typing use the coalesce key `amount-${index}`.
 */
export function StepHowMuch({ model, index, idPrefix }: StepHowMuchProps) {
  const uid = useId();
  const edit = model.doc.edits[index];
  const headingId = `${idPrefix}-howmuch-head`;
  const head = (
    <h3 className="lab-step-head" id={headingId}>
      <StageNum index={2} count={3} status={edit && Number.isFinite(edit.amount) ? "done" : "todo"} />
      {LAB_WORDS.stepHowMuch}
    </h3>
  );
  if (!edit) {
    return (
      <section className="lab-step flow-step" aria-labelledby={headingId} data-step="howmuch" data-pending="true">
        {head}
        <p className="cap">{FLOW_WORDS.pickFirst}</p>
      </section>
    );
  }
  const lever = model.levers.find((l) => l.var === edit.lever);
  const modes = plannerModes(lever);
  const fit = compileSentence(edit, { compile: model.compile, levers: model.levers, edits: model.doc.edits });
  const blocked = (model.compile?.warnings ?? []).filter((w) => w.blocking && w.edit_index === index);
  const status = (
    <>
      {fit ? (
        <p className="flow-step-fit" data-compile-sentence="true" data-tone={fit.tone} aria-live="polite" aria-busy={model.compiling || undefined}>
          {fit.text}
        </p>
      ) : model.compiling ? (
        <p className="cap" aria-live="polite">
          {FLOW_WORDS.checking}
        </p>
      ) : null}
      {blocked.length ? (
        <div className="flow-step-blocked" role="alert">
          <strong>{FLOW_WORDS.blocked}</strong>
          <ul>
            {plainAdvice(
              blocked.map((w) => w.message),
              model.levers,
            ).map((m) => (
              <li key={m}>{m}</li>
            ))}
          </ul>
        </div>
      ) : null}
    </>
  );

  if (!modes.includes(edit.mode)) {
    return (
      <section className="lab-step flow-step" aria-labelledby={headingId} data-step="howmuch" data-mode={edit.mode}>
        {head}
        <p className="cap">{fill(FLOW_WORDS.otherMode, { mode: MODE_WORDS[edit.mode]?.label ?? edit.mode })}</p>
        {lever ? (
          <button type="button" className="flow-step-link" onClick={() => model.setEdit(index, switchMode(edit, "add", lever))}>
            {FLOW_WORDS.backToSimple}
          </button>
        ) : null}
        {status}
      </section>
    );
  }

  const scale = amountScale(edit.mode, lever);
  const shown = toDisplay(edit, lever);
  const sliderMin = Math.min(scale.min, shown ?? scale.min);
  const sliderMax = Math.max(scale.max, shown ?? scale.max);
  // Slider drags and typing coalesce into one undo step; a typical-amount pick is a step of its own.
  const setShown = (v: number | null, coalesce: string | null = `amount-${index}`) =>
    model.setEdit(index, { ...edit, amount: v === null ? undefined : fromDisplay(v, edit.mode, lever) }, coalesce ?? undefined);
  const label = amountLabel(edit.mode, lever);
  const labelId = `${uid}-label`;
  const doses = edit.mode === "add" ? typicalAmounts(lever) : [];
  const text = (v: number) => fmtAmount(v, scale.unit);
  return (
    <section className="lab-step flow-step" aria-labelledby={headingId} data-step="howmuch" data-mode={edit.mode}>
      {head}
      {modes.length > 1 ? (
        <ChoiceGroup<EditMode>
          label={FLOW_WORDS.howToChange}
          value={edit.mode}
          onChange={(m) => model.setEdit(index, switchMode(edit, m, lever))}
          options={modes.map((m) => ({ value: m, label: modeLabel(m, lever), hint: modeHint(m, lever) }))}
        />
      ) : null}
      <p className="cap flow-step-hint">{modeHint(edit.mode, lever)}</p>
      <div className="flow-step-amount">
        <label className="flow-step-amount-label" id={labelId} htmlFor={`${idPrefix}-amount`}>
          {label}
        </label>
        <div className="flow-step-amount-row">
          <input
            type="range"
            className="flow-step-range"
            aria-labelledby={labelId}
            aria-valuetext={shown !== null ? text(shown) : undefined}
            min={sliderMin}
            max={sliderMax}
            step={scale.step}
            value={shown ?? sliderMin}
            onChange={(e) => setShown(roundToStep(Number(e.target.value), scale.step))}
          />
          <NumberField id={`${idPrefix}-amount`} value={shown} unit={scale.unit || null} step={scale.step} min={edit.mode === "add" ? undefined : sliderMin} max={edit.mode === "add" ? undefined : sliderMax} nullable onChange={(v) => setShown(v)} />
        </div>
        {/* the typical amounts are chips only (no slider ticks: they bunched at the slider's low end) */}
        {doses.length ? (
          <div className="flow-step-doses" role="group" aria-label={FLOW_WORDS.typical}>
            <span className="cap">{FLOW_WORDS.typical}</span>
            {doses.map((d) => (
              <button key={d} type="button" className="chip" aria-pressed={shown === d} onClick={() => setShown(d, null)}>
                {text(d)}
              </button>
            ))}
          </div>
        ) : null}
      </div>
      {status}
    </section>
  );
}
