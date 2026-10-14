// Step 01 · What to change: one card per lever (glyph, plain name, one-line effect, trust dot).
// Picking a card on a change not started yet creates it; on an existing change it swaps the lever
// (model/flow.ts switchLever). Contract: ./types.ts.
import { useId } from "react";
import type { Lever } from "../../../../api/lab";
import { StageNum } from "../../../../components/ui/StageRail";
import { FLOW_WORDS, newChangeInit, switchLever } from "../../model/flow";
import { LAB_WORDS, TRUST_TIP, TRUST_WORDS, leverWords, type LeverIcon } from "../../model/words";
import type { StepProps } from "./types";
import "./flow.css";

export type StepWhatProps = StepProps;

const GLYPHS: Record<LeverIcon, string> = {
  // a broad-crowned tree
  trees: "M12 21v-6.5M12 14.5c-4 0-7-2.5-7-5.8C5 5.5 8.1 3 12 3s7 2.5 7 5.7c0 3.3-3 5.8-7 5.8zM12 12.5l-2.5-2.5M12 11l2.5-2.5M8.5 21h7",
  // paving slabs with a gap of soil and a sprout
  paving: "M3.5 5.5h17v13h-17zM3.5 12h6M14.5 12h6M9.5 5.5v13M14.5 5.5v13M12 16.5v-3M12 14c-1.2 0-2-.8-2-2M12 13.5c1 0 1.8-.7 1.8-1.7",
  // a roof reflecting sunlight
  roofs: "M3 13.5l9-6.5 9 6.5M6 11.5V20h12v-8.5M12 2.5V4M17 4.2l-.9 1.1M7 4.2l.9 1.1",
  // two sliders
  lever: "M4 8h9M17 8h3M13 6v4M4 16h3M11 16h9M7 14v4",
};

/** The 24×24 line glyph of a lever card (stroke = currentColor). */
export function LeverGlyph({ icon, size = 20 }: { icon: LeverIcon; size?: number }) {
  return (
    <svg viewBox="0 0 24 24" width={size} height={size} fill="none" stroke="currentColor" strokeWidth={1.7} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false">
      <path d={GLYPHS[icon] ?? GLYPHS.lever} />
    </svg>
  );
}

function LeverCard({ lever, pressed, onPick }: { lever: Lever; pressed: boolean; onPick: () => void }) {
  const id = useId();
  const w = leverWords(lever);
  const t = TRUST_WORDS[lever.emulator.trust] ?? TRUST_WORDS.none;
  return (
    <button
      type="button"
      className="flow-step-lever card tight"
      aria-pressed={pressed}
      aria-labelledby={`${id}-name`}
      aria-describedby={`${id}-line ${id}-trust`}
      onClick={onPick}
      data-lever={lever.var}
      data-icon={w.icon}
    >
      <span className="flow-step-lever-icon">
        <LeverGlyph icon={w.icon} />
      </span>
      <span className="flow-step-lever-text">
        <strong id={`${id}-name`}>{w.name}</strong>
        <span className="cap" id={`${id}-line`}>
          {w.oneLine}
        </span>
      </span>
      <span className="flow-step-trust" data-tone={t.tone} title={`${t.label}. ${TRUST_TIP}`}>
        <span className="sr-only" id={`${id}-trust`}>
          {t.label}.
        </span>
      </span>
    </button>
  );
}

/**
 * Lever cards for change `index`. On a change not started yet a pick creates it
 * (`model.addEdit(lever, { amount })`: an "add" of the design dose in the lever's cooling
 * direction); on an existing change it swaps the lever, keeping where (model/flow.ts switchLever).
 */
export function StepWhat({ model, index, idPrefix }: StepWhatProps) {
  const edit = model.doc.edits[index];
  const headingId = `${idPrefix}-what`;
  const known = !edit || model.levers.some((l) => l.var === edit.lever);
  const pick = (l: Lever) => {
    if (!edit) {
      model.addEdit(l.var, newChangeInit(l));
      return;
    }
    if (edit.lever === l.var) return;
    model.setEdit(index, switchLever(edit, l));
  };
  return (
    <section className="lab-step flow-step" aria-labelledby={headingId} data-step="what">
      <h3 className="lab-step-head" id={headingId}>
        <StageNum index={0} count={3} status={edit ? "done" : "todo"} />
        {LAB_WORDS.stepWhat}
      </h3>
      {model.levers.length ? (
        <div className="flow-step-levers" role="group" aria-label={LAB_WORDS.stepWhat}>
          {model.levers.map((l) => (
            <LeverCard key={l.var} lever={l} pressed={edit?.lever === l.var} onPick={() => pick(l)} />
          ))}
        </div>
      ) : (
        <p className="cap">{FLOW_WORDS.noLevers}</p>
      )}
      {!known ? (
        <p className="cap" data-expert-lever={edit!.lever}>
          {FLOW_WORDS.notALever} <code>{edit!.lever}</code>
        </p>
      ) : null}
    </section>
  );
}
