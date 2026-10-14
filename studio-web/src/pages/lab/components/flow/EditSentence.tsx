// A completed change collapsed to one sentence ("Add 10 pts tree canopy in Downtown") with Edit and
// Remove. No compile numbers here: the "how much fits" sentence of step 03 is their only place
// before running. A change that is not finished (no amount yet) says it is left out of the estimate.
import { useId } from "react";
import type { Edit } from "../../../../api/lab";
import { Button } from "../../../../components/ui/Button";
import { editProblems } from "../../model/doc";
import { FLOW_WORDS, editSentence as sentenceOf } from "../../model/flow";
import type { WorkbenchModel } from "../../model/useWorkbench";
import { LAB_WORDS, leverWords } from "../../model/words";
import { LeverGlyph } from "./StepWhat";
import { flowContext } from "./StepWhere";
import "./flow.css";

export type EditSentenceProps = {
  model: WorkbenchModel;
  /** Index of the change in `model.doc.edits` (always an existing change). */
  index: number;
  /** Make this the active change (the page then shows its three steps). */
  onEdit: () => void;
  /** Remove this change. The page passes a handler that calls `model.removeEdit(index)` and moves the active change. */
  onRemove: () => void;
};

type SentenceModel = Pick<WorkbenchModel, "levers" | "regions"> & Partial<Pick<WorkbenchModel, "grid" | "temperatureColumn" | "populationColumn" | "mapPick" | "columns" | "planRefs">>;

/**
 * "Add 10 pts tree canopy in zone 1" (the plain one-line summary of a change; model/flow.ts
 * editSentence). Fields missing from `model` read as "none" (no zones, no preset columns).
 */
export function editSentence(model: SentenceModel, edit: Edit): string {
  return sentenceOf(
    edit,
    flowContext({
      grid: model.grid ?? ({ meta: { zones: [] } } as unknown as WorkbenchModel["grid"]),
      regions: model.regions,
      temperatureColumn: model.temperatureColumn ?? null,
      populationColumn: model.populationColumn ?? null,
      mapPick: model.mapPick ?? null,
      columns: model.columns ?? [],
      levers: model.levers,
      planRefs: model.planRefs ?? [],
    }),
  );
}

export function EditSentence({ model, index, onEdit, onRemove }: EditSentenceProps) {
  const id = useId();
  const edit = model.doc.edits[index];
  if (!edit) return null;
  const lever = model.levers.find((l) => l.var === edit.lever);
  const unfinished = editProblems(edit).length > 0;
  return (
    <div className="lab-edit-sentence flow-step-sentence" data-unfinished={unfinished || undefined}>
      <span className="flow-step-sentence-icon">
        <LeverGlyph icon={leverWords(lever, edit.lever).icon} size={18} />
      </span>
      <p className="flow-step-sentence-text" id={`${id}-text`}>
        {editSentence(model, edit)}
        {edit.label ? <span className="cap"> · {edit.label}</span> : null}
        {unfinished ? <span className="flow-step-sentence-flag"> — {FLOW_WORDS.notFinished}</span> : null}
      </p>
      <span className="flow-step-sentence-actions">
        <Button size="small" variant="ghost" onClick={onEdit} aria-describedby={`${id}-text`}>
          {LAB_WORDS.editChange}
        </Button>
        <Button size="small" variant="ghost" onClick={onRemove} aria-describedby={`${id}-text`}>
          {LAB_WORDS.removeChange}
        </Button>
      </span>
    </div>
  );
}
