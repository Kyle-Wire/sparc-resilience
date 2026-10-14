// Right pane, bottom: THE primary action of the Design page, "Run full model (≈ N min)" (.btn.accent,
// the same handler "Run exact" had: model.runExact flushes the autosave, then starts the exact run),
// the engine state as one muted line under it, why it cannot run yet in plain words, and the
// fold-tick JobStrip while it runs. Once the full model has a result for the changes as they stand,
// there is nothing new to run: the accent goes, the button becomes a secondary "See full model
// result" (it brings the answer card into view) and one line says the full model is up to date. The
// figure itself shows only on the answer card. Wording: model/quick.ts and model/words.ts.
import { useCallback, useSyncExternalStore } from "react";
import { Button } from "../../../../components/ui/Button";
import { JobStrip } from "../../../../components/ui/JobStrip";
import { isDirty } from "../../model/draft";
import { engineTone, fullModelUpToDate, runDisabledReason, runLabel, unfinishedNote } from "../../model/quick";
import type { WorkbenchModel } from "../../model/useWorkbench";
import { LAB_WORDS, engineLine } from "../../model/words";
import { revealAnswerCard } from "../result/AnswerCard";
import "./quick.css";

// Kept for importers of the shell's module.
export { runLabel } from "../../model/quick";

type DraftView = { dirty: boolean; revision: number | null };
const CLEAN: DraftView = { dirty: false, revision: null };
const NO_STORE: Partial<WorkbenchModel["store"]> = {};

/**
 * Whether the shown full-model result answers the changes as they stand (model/quick.ts
 * fullModelUpToDate): read from the draft store (unsaved edits or strokes, the open revision), the
 * scenario id, the run state and the result. Shared by the run block and the quick-estimate pane.
 */
export function useFullModelUpToDate(model: Pick<WorkbenchModel, "store" | "storeSid" | "result" | "running" | "activeJob" | "shownJob">): boolean {
  const store: Partial<WorkbenchModel["store"]> = model.store ?? NO_STORE;
  const subscribe = useCallback((cb: () => void) => (typeof store.subscribe === "function" ? store.subscribe(cb) : () => {}), [store]);
  const dirty = useSyncExternalStore(subscribe, () => (typeof store.getState === "function" ? isDirty(store.getState()) : CLEAN.dirty));
  const revision = useSyncExternalStore(subscribe, () => (typeof store.getState === "function" ? (store.getState().revision ?? null) : CLEAN.revision));
  return fullModelUpToDate(model.result, { sid: model.storeSid ?? null, revision, dirty, running: model.running || !!model.activeJob });
}

/**
 * Props of the run block: the whole workbench model.
 *
 * Reads: `model.compile?.est_exact_s` (the ≈ duration), `model.canRun`, `model.blocking`,
 * `model.runReason`, `model.usable` / `model.doc.edits` / `model.store.brushLayers` (what is there to
 * run), `model.running`, `model.engineState` / `model.engine.progress` (words.engineLine),
 * `model.shownJob` (JobStrip), `model.noEmulator` (the engine line), `model.result` +
 * `model.storeSid` + the draft store (whether the result is up to date: useFullModelUpToDate).
 * Acts: `model.runExact()` (engine refusals surface as `model.refusal`, whose dialogs the page renders).
 *
 * DOM contract: while there is something new to run, exactly one button whose text starts with
 * "Run full model", variant "accent"; disabled while `!model.canRun || model.blocking.length > 0`.
 * A run with no usable full model (engine state no_checkpoint | incompatible) shows no run button at
 * all, only the engine line (the section carries `data-no-model`).
 * Once the result is up to date the button reads "See full model result" (default variant, it scrolls
 * to and focuses `.answer-card`). The engine line carries `data-engine={state}`, the plain reason it
 * cannot run `data-run-why`, the result line `data-full-result` = "current" ("Full model is up to date
 * for these changes.") or "outdated" (a result that is stale or older than unsaved changes); it never
 * repeats the result's number.
 */
export type RunFullModelProps = { model: WorkbenchModel };

export function RunFullModel({ model }: RunFullModelProps) {
  const upToDate = useFullModelUpToDate(model);
  const blocked = model.blocking.length > 0;
  const disabled = !model.canRun || blocked;
  const hasChanges = model.usable.length > 0 || !model.store.brushLayers.isEmpty();
  const state = { hasChanges, unfinished: model.doc.edits.length - model.usable.length, runReason: model.runReason, blocking: model.blocking };
  const why = disabled && !upToDate ? runDisabledReason(state) : null;
  const note = unfinishedNote(state);
  const whyId = "lab-run-why";
  // no usable full model for this run: a button that can never be pressed would only point a second
  // way; the engine line says it once
  const noModel = model.engineState === "no_checkpoint" || model.engineState === "incompatible";
  return (
    <section className="lab-run-full qe-run" aria-label="Full model" data-up-to-date={upToDate || undefined} data-no-model={noModel || undefined}>
      {upToDate ? (
        <Button icon="chevronDown" onClick={() => revealAnswerCard({ focus: true })}>
          {LAB_WORDS.seeFullResult}
        </Button>
      ) : noModel ? null : (
        <Button
          variant="accent"
          icon="play"
          busy={model.running}
          disabled={disabled}
          onClick={() => void model.runExact()}
          title={model.runReason ?? undefined}
          aria-describedby={why ? whyId : undefined}
        >
          {runLabel(model.compile?.est_exact_s)}
        </Button>
      )}
      <p className="cap qe-engine" data-engine={model.engineState ?? "unknown"} data-tone={engineTone(model.engineState)}>
        <span className="qe-engine-dot" aria-hidden="true" />
        {engineLine(model.engineState, model.engine.progress, { noEmulator: model.noEmulator })}
      </p>
      {model.result && (upToDate || !(model.running || model.activeJob)) ? (
        <p className="cap qe-result" data-full-result={upToDate ? "current" : "outdated"}>
          {upToDate ? LAB_WORDS.fullModelUpToDate : LAB_WORDS.fullModelOutdated}
        </p>
      ) : null}
      {why ? (
        <p className="cap qe-why" id={whyId} data-run-why="true">
          {why}
        </p>
      ) : null}
      {note ? <p className="cap qe-why">{note}</p> : null}
      {model.shownJob ? <JobStrip jobId={model.shownJob} /> : null}
    </section>
  );
}
