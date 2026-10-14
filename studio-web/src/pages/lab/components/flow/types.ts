// Shared props contract of the Design step flow (phase 3). The Design page renders, for the active
// change, StepWhat → StepWhere → StepHowMuch; every other change collapses to an EditSentence.
import type { WorkbenchModel } from "../../model/useWorkbench";

/**
 * Props of the three steps (StepWhat, StepWhere, StepHowMuch).
 *
 * - `index` is the change's position in `model.doc.edits`. It may equal `model.doc.edits.length`:
 *   a change not started yet. StepWhat creates it on the first lever pick (`model.addEdit`);
 *   StepWhere and StepHowMuch then render a short muted placeholder ("Pick what to change first").
 * - Edit the change only through `model.setEdit(index, next, coalesce?)` (one undo step; pass a
 *   coalesce key such as `amount-${index}` for slider drags and typing), `model.addEdit` and
 *   `model.removeEdit`. Never touch the store directly.
 * - `idPrefix` prefixes every element id the step renders. StepHowMuch's amount number input MUST
 *   use `${idPrefix}-amount` (the e2e hook `#edit-0-amount`). The page passes `edit-${index}` while
 *   Advanced options is closed and `change-${index}` while it is open (the ScenarioEditor's edit rows
 *   own the `edit-*` ids then).
 * - Each step renders its own `StageNum` (01, 02, 03; `count={3}`) and heading from words.ts
 *   (LAB_WORDS.stepWhat / stepWhere / stepHowMuch), inside a `<section aria-labelledby>`.
 */
export type StepProps = {
  model: WorkbenchModel;
  index: number;
  idPrefix: string;
};
