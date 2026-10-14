// Scenario Lab — Design (`/r/:rid/lab`) and Scenario + result (`/r/:rid/lab/s/:sid`), SPEC §7,
// §2 J6, rebuilt in phase 3 as a straightforward flow for a city planner:
//   steps rail (~380 px) | map | quick estimate & run (~320 px), then the full-model result.
// - The steps rail shows, for the active change, 01 What to change → 02 Where → 03 How much
//   (components/flow/*); every other change collapses to one sentence. "+ Add another change"
//   stacks another. One "Advanced options" disclosure holds everything rare: the full scenario
//   editor (all eight modes, the selection tree, costs, regions, design CSV, name/tags/notes),
//   the realistic-limits / knock-on / expert toggles, the per-change compile table and warnings,
//   and the model (engine) controls.
// - The right pane is the live quick estimate (PreviewPane) over THE primary action, Run full
//   model (RunFullModel). The full-model result (ResultView) opens full width under the panes.
// All state and behaviour live in model/useWorkbench.ts (draft + autosave + undo, preview
// controller, compile, brush, map picks, exact run); this file is layout only.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useStore } from "zustand";
import { useLevers, useScenario, type CompileResult, type Lever } from "../../api/lab";
import { Button } from "../../components/ui/Button";
import { EmptyState } from "../../components/ui/EmptyState";
import { IconButton } from "../../components/ui/IconButton";
import { Table } from "../../components/ui/Table";
import { useRunDetail } from "../../layouts/resources";
import { findLayer } from "../../map/LayerPicker";
import { useRunGrid, useRunLayers } from "../../map/data";
import { Link, navigate, useRoute } from "../../router";
import { useUi } from "../../stores/ui";
import { fmtDuration, fmtInt, fmtNum, fmtPct, unitLabel } from "../../theme/format";
import { CompareTray } from "./components/CompareTray";
import { DesignMap, PREVIEW_KEY } from "./components/DesignMap";
import { LadderDialog } from "./components/Dialogs";
import { EngineChip, MemoryDialog, TrustDialog } from "./components/EngineChip";
import { EditSentence } from "./components/flow/EditSentence";
import { PreviewPane } from "./components/flow/PreviewPane";
import { RunFullModel, useFullModelUpToDate } from "./components/flow/RunFullModel";
import { StepHowMuch } from "./components/flow/StepHowMuch";
import { StepWhat } from "./components/flow/StepWhat";
import { StepWhere } from "./components/flow/StepWhere";
import { LabFrame, labHref } from "./components/LabFrame";
import { ResultView } from "./components/result/ResultView";
import { ScenarioEditor } from "./components/ScenarioEditor";
import { TemplateGallery } from "./components/TemplateGallery";
import { TrustBadge } from "./components/TrustBadge";
import { canRedo, canUndo, isDirty, type DraftStore } from "./model/draft";
import { useTray } from "./model/tray";
import { useWorkbench, type WorkbenchInput, type WorkbenchModel } from "./model/useWorkbench";
import { ADVANCED_WORDS, LAB_WORDS, emptyChangesText, leverWords } from "./model/words";

// Kept for importers of the old page module.
export { latestResultOn } from "./model/useWorkbench";

// "Advanced options" stays open across scenarios (a fork or "New scenario" remounts the workbench).
let advancedRemembered = false;
function useAdvancedOpen(): [boolean, (open: boolean) => void] {
  const [open, setOpen] = useState(advancedRemembered);
  return [
    open,
    (next) => {
      advancedRemembered = next;
      setOpen(next);
    },
  ];
}

/** Undo / redo and the save state, in the page header (the editor in Advanced options has its own too). */
function DraftControls({ store }: { store: DraftStore }) {
  const undoable = useStore(store, canUndo);
  const redoable = useStore(store, canRedo);
  const save = useStore(store, (s) => s.save);
  const saveError = useStore(store, (s) => s.saveError);
  const dirty = useStore(store, isDirty);
  const sid = useStore(store, (s) => s.sid);
  const text = save === "saving" ? "Saving…" : save === "error" ? `Not saved: ${saveError ?? "error"}` : dirty ? (sid ? "Unsaved changes" : "Not saved yet") : sid ? "All changes saved" : "New draft";
  return (
    <>
      <span className="cap lab-save" data-save={save}>
        {text}
      </span>
      <IconButton icon="undo" label="Undo (Ctrl+Z)" disabled={!undoable} onClick={() => store.getState().undo()} />
      <IconButton icon="redo" label="Redo (Ctrl+Shift+Z)" disabled={!redoable} onClick={() => store.getState().redo()} />
    </>
  );
}

/** The before-run compile in Advanced options: totals, the per-change table and every warning. */
function CompileDetails({ compile, compileError, compiling, levers }: { compile: CompileResult | null; compileError: string | null; compiling: boolean; levers: Lever[] }) {
  const c = compile;
  const rows = c ? Object.entries(c.levers).map(([v, x]) => ({ v, ...x, lever: levers.find((l) => l.var === v) })) : [];
  return (
    <section className="compile-panel stack" aria-label={ADVANCED_WORDS.details} aria-busy={compiling || undefined}>
      <h4>{ADVANCED_WORDS.details}</h4>
      {compileError ? (
        <p className="callout" data-tone="crit">
          {compileError}
        </p>
      ) : null}
      {c ? (
        <>
          <dl className="kv">
            <dt>{ADVANCED_WORDS.cellsChanged}</dt>
            <dd className="num">{fmtInt(c.union_cells)}</dd>
            {c.people !== null ? (
              <>
                <dt>{ADVANCED_WORDS.residents}</dt>
                <dd className="num">{fmtInt(Math.round(c.people))}</dd>
              </>
            ) : null}
            <dt>{ADVANCED_WORDS.cost}</dt>
            <dd className="num">{fmtNum(rows.reduce((a, r) => a + r.est_cost, 0), 0)}</dd>
            <dt>{ADVANCED_WORDS.fullModelTime}</dt>
            <dd className="num">≈ {fmtDuration(c.est_exact_s)}</dd>
            <dt>{ADVANCED_WORDS.portable}</dt>
            <dd>{c.portable ? ADVANCED_WORDS.portableYes : ADVANCED_WORDS.portableNo}</dd>
          </dl>
          {rows.length ? (
            <Table
              caption={ADVANCED_WORDS.detailsCaption}
              csvName={null}
              rowKey={(r) => r.v}
              columns={[
                {
                  key: "v",
                  label: "Change",
                  value: (r) => leverWords(r.lever, r.v).name,
                  render: (r) => (
                    <span className="row" style={{ gap: 4 }}>
                      {leverWords(r.lever, r.v).name}
                      {r.lever ? <TrustBadge trust={r.lever.emulator.trust} relErr={r.lever.emulator.uniform_rel_err} /> : null}
                    </span>
                  ),
                },
                { key: "n", label: "Places", align: "right", value: (r) => r.n_cells, render: (r) => fmtInt(r.n_cells) },
                { key: "req", label: "Asked for (mean)", align: "right", value: (r) => r.mean_requested, render: (r) => `${fmtNum(r.mean_requested, 2)} ${unitLabel(r.lever?.unit ?? "")}` },
                { key: "real", label: "Fits (mean)", align: "right", value: (r) => r.predicted_mean_realised, render: (r) => `${fmtNum(r.predicted_mean_realised, 2)} ${unitLabel(r.lever?.unit ?? "")}` },
                { key: "clip", label: "Does not fit", align: "right", value: (r) => r.clipped_share, render: (r) => fmtPct(r.clipped_share) },
                { key: "cost", label: "Cost", align: "right", value: (r) => r.est_cost, render: (r) => fmtNum(r.est_cost, 0) },
              ]}
              rows={rows}
            />
          ) : null}
          {c.warnings.length ? (
            <ul className="edit-issues">
              {c.warnings.map((w, i) => (
                <li key={i} data-level={w.blocking ? "error" : "warn"}>
                  {w.edit_index !== null && !/^edit \d/i.test(w.message) ? `Change ${w.edit_index + 1}: ` : ""}
                  {w.blocking ? "Blocked — " : ""}
                  {w.message}
                </li>
              ))}
            </ul>
          ) : null}
        </>
      ) : !compileError ? (
        <p className="cap">{compiling ? "Checking the changes…" : "Make a change to see its details."}</p>
      ) : null}
    </section>
  );
}

/** One disclosure for everything rare (rendered only while open, so its editor ids never clash with the steps'). */
function AdvancedOptions({ m, open, onToggle }: { m: WorkbenchModel; open: boolean; onToggle: (open: boolean) => void }) {
  const options = m.doc.options ?? {};
  const setOption = (k: "clip_to_support" | "mediators" | "expert", v: boolean) => m.store.getState().change((d) => ({ ...d, options: { ...(d.options ?? {}), [k]: v } }));
  return (
    <details className="lab-advanced card flat" open={open}>
      <summary
        onClick={(e) => {
          e.preventDefault();
          onToggle(!open);
        }}
      >
        {LAB_WORDS.advanced}
      </summary>
      {open ? (
        <div className="lab-advanced-body stack">
          <fieldset className="stack lab-advanced-toggles">
            <label className="row" title={ADVANCED_WORDS.clipTip}>
              <input type="checkbox" checked={options.clip_to_support ?? true} onChange={(e) => setOption("clip_to_support", e.target.checked)} />
              {ADVANCED_WORDS.clip}
            </label>
            <label className="row" title={ADVANCED_WORDS.mediatorsTip}>
              <input type="checkbox" checked={options.mediators ?? true} onChange={(e) => setOption("mediators", e.target.checked)} />
              {ADVANCED_WORDS.mediators}
            </label>
            <label className="row" title={ADVANCED_WORDS.expertTip}>
              <input type="checkbox" checked={!!options.expert} onChange={(e) => setOption("expert", e.target.checked)} />
              {ADVANCED_WORDS.expert}
            </label>
          </fieldset>
          <CompileDetails compile={m.compile} compileError={m.compileError} compiling={m.compiling} levers={m.levers} />
          <section className="stack" aria-label={ADVANCED_WORDS.model}>
            <h4>{ADVANCED_WORDS.model}</h4>
            <EngineChip rid={m.rid} />
          </section>
          <section className="stack" aria-label={ADVANCED_WORDS.editor}>
            <h4>{ADVANCED_WORDS.editor}</h4>
            <ScenarioEditor
              rid={m.rid}
              store={m.store}
              levers={m.levers}
              grid={m.grid}
              columns={m.columns}
              regions={m.regions}
              planRefs={m.planRefs}
              predictors={m.predictors}
              compile={m.compile}
              mapPick={m.mapPick}
              onMask={m.setTint}
              onLadder={(i) => m.setLadder(i)}
              onBlur={() => {
                m.setTint(null);
                void m.autosave.flush();
              }}
            />
          </section>
        </div>
      ) : null}
    </details>
  );
}

/** Brushed levers are saved as per-place changes; the rail lists them so painting shows up as a change. */
function BrushedChanges({ m }: { m: WorkbenchModel }) {
  const layers = m.store.brushLayers;
  void m.brush; // re-render with the committed brush
  const brushed = layers.levers();
  if (!brushed.length) return null;
  return (
    <ul className="lab-brushed cap" aria-label="Painted on the map">
      {brushed.map((l) => (
        <li key={l}>
          Painted on the map: {leverWords(m.levers.find((x) => x.var === l), l).name.toLowerCase()} in {fmtInt(layers.editedCount(l))} places
        </li>
      ))}
    </ul>
  );
}

/**
 * The map layer to open on: the full model's temperature change while its result is up to date
 * (`currentResultId`: the answer card then shows that result), else the quick estimate once it shows a
 * change, else the observed temperature (where the city is hot).
 */
export function defaultLayerKey(groups: WorkbenchModel["groups"], nEdited: number | null | undefined, currentResultId?: string | null): string {
  if (currentResultId) return `res:${currentResultId}:delta`;
  return (nEdited ?? 0) > 0 || !findLayer(groups, "obs") ? PREVIEW_KEY : "obs";
}

function Workbench(input: WorkbenchInput) {
  const m = useWorkbench(input);
  const { rid, pid, grid, store, doc, storeSid } = m;
  const [advancedOpen, setAdvancedOpen] = useAdvancedOpen();
  // Step 02 "Draw on the map" is chosen on the open change: the map shows its Brush and Pick settings.
  const [drawing, setDrawing] = useState(false);
  const onDrawingChange = useCallback((d: boolean) => setDrawing(d), []);
  // The map opens on the observed temperature and switches to the quick estimate with the first change,
  // and to the full model's temperature change while its result is up to date (unless a layer was
  // chosen: choosing the default layer stores nothing, so the switch still happens).
  const upToDate = useFullModelUpToDate(m);
  const currentResult = upToDate ? m.resultId : null;
  const defaultLayer = useMemo(() => defaultLayerKey(m.groups, m.preview?.summary.n_edited, currentResult), [m.groups, m.preview, currentResult]);
  // A full-model result that finishes while the page is open is brought into view (not one the page opened with).
  const [reveal, setReveal] = useState<string | null>(null);
  const lastResult = useRef(m.resultId);
  useEffect(() => {
    if (m.resultId && m.resultId !== lastResult.current) setReveal(m.resultId);
    lastResult.current = m.resultId;
  }, [m.resultId]);
  const trayCount = useTray((s) => s.byRun[rid]?.length ?? 0);
  // The active change: its three steps show; the others collapse to a sentence. -1 = all collapsed;
  // doc.edits.length = a change not started yet (StepWhat creates it on the first pick).
  const [active, setActive] = useState(0);
  const n = doc.edits.length;
  const current = n === 0 ? 0 : active < 0 ? -1 : Math.min(active, n);
  const idPrefix = (i: number) => (advancedOpen ? `change-${i}` : `edit-${i}`);
  const steps = (i: number) => (
    <>
      <StepWhat model={m} index={i} idPrefix={idPrefix(i)} />
      <StepWhere model={m} index={i} idPrefix={idPrefix(i)} onDrawingChange={onDrawingChange} />
      <StepHowMuch model={m} index={i} idPrefix={idPrefix(i)} />
    </>
  );
  const changeLabel = (i: number) => {
    const e = doc.edits[i];
    return `Change ${i + 1}${e ? `: ${leverWords(m.levers.find((l) => l.var === e.lever), e.lever).name}` : ""}`;
  };

  return (
    <LabFrame
      rid={rid}
      hideEngine
      title={doc.name || "Untitled scenario"}
      actions={
        <>
          <DraftControls store={store} />
          {storeSid ? (
            <Link className="btn small" to={labHref(rid)} onClick={() => void m.autosave.flush()}>
              New scenario
            </Link>
          ) : null}
          <IconButton icon="chevronLeft" label={m.panes.left ? "Hide the steps" : "Show the steps"} pressed={!m.panes.left} onClick={() => m.setPanes((p) => ({ ...p, left: !p.left }))} />
          <IconButton icon="chevronRight" label={m.panes.right ? "Hide the estimate" : "Show the estimate"} pressed={!m.panes.right} onClick={() => m.setPanes((p) => ({ ...p, right: !p.right }))} />
        </>
      }
    >
      <div className="lab-flow" data-tray={trayCount ? "true" : undefined}>
        <div className="lab-panes" data-left={m.panes.left ? undefined : "collapsed"} data-right={m.panes.right ? undefined : "collapsed"}>
          <aside className="lab-pane lab-steps" aria-label="Steps">
            {doc.edits.map((_, i) =>
              i === current ? (
                <article key={i} className="lab-change edit-row" data-active="true" aria-label={changeLabel(i)}>
                  <header className="row">
                    <span className="kicker">Change {i + 1}</span>
                    <span className="spacer" />
                    <Button
                      size="small"
                      variant="ghost"
                      onClick={() => {
                        m.removeEdit(i);
                        setActive(-1);
                      }}
                    >
                      {LAB_WORDS.removeChange}
                    </Button>
                    <Button size="small" variant="ghost" onClick={() => setActive(-1)}>
                      Done
                    </Button>
                  </header>
                  {steps(i)}
                </article>
              ) : (
                <article key={i} className="lab-change edit-row" aria-label={changeLabel(i)}>
                  <EditSentence
                    model={m}
                    index={i}
                    onEdit={() => setActive(i)}
                    onRemove={() => {
                      m.removeEdit(i);
                      setActive((a) => (a > i ? a - 1 : a));
                    }}
                  />
                </article>
              ),
            )}
            <BrushedChanges m={m} />
            {current === n ? (
              <article className="lab-change edit-row" data-active="true" data-pending="true" aria-label={changeLabel(n)}>
                {n > 0 ? (
                  <header className="row">
                    <span className="kicker">Change {n + 1}</span>
                    <span className="spacer" />
                    <Button size="small" variant="ghost" onClick={() => setActive(-1)}>
                      Cancel
                    </Button>
                  </header>
                ) : (
                  <p className="cap" data-intro="true">
                    {emptyChangesText(m.noEmulator, m.engineState)}
                  </p>
                )}
                {steps(n)}
              </article>
            ) : (
              <Button icon="plus" onClick={() => setActive(n)} disabled={!m.levers.length}>
                {LAB_WORDS.addChange}
              </Button>
            )}
            {/* Advanced options sit directly under step 03 (and "+ Add another change"); the examples come last */}
            <AdvancedOptions m={m} open={advancedOpen} onToggle={setAdvancedOpen} />
            {m.showTemplates ? (
              <section className="lab-examples stack" aria-label={LAB_WORDS.startFromExample}>
                <h3>{LAB_WORDS.startFromExample}</h3>
                <TemplateGallery
                  pid={pid}
                  rid={rid}
                  levers={m.levers}
                  context={{ hasCrs: !!grid.meta.crs, hasLayers: m.groups.length ? !!m.populationColumn : undefined }}
                  onCreated={m.onTemplateCreated}
                />
              </section>
            ) : null}
          </aside>
          <section className="lab-pane lab-map" aria-label="Map">
            <DesignMap
              rid={rid}
              grid={grid}
              groups={m.groups}
              levers={m.levers}
              store={store}
              brushTick={m.brushTick}
              brushState={m.brush}
              preview={m.preview}
              previewBusy={m.previewBusy}
              previewError={m.previewError ? m.previewError.message : null}
              resultId={m.resultId}
              realizedVars={m.realizedVars}
              tint={m.tint}
              layerKey={m.layerKey ?? defaultLayer}
              onLayerChange={(k) => m.setLayerKey(k === defaultLayer ? null : k)}
              onBrushChange={m.onBrushChange}
              onSelection={m.onSelection}
              showPreviewCard={false}
              drawing={drawing}
            />
          </section>
          <aside className="lab-pane lab-run" aria-label="Quick estimate and full model">
            <PreviewPane model={m} />
            <RunFullModel model={m} />
          </aside>
        </div>
      </div>
      {m.resultId ? (
        <ResultView rid={rid} resId={m.resultId} pid={pid} unit={m.unit} nFolds={grid.meta.n_folds} nCells={grid.n} reveal={reveal === m.resultId} onShowLayer={(k) => m.setLayerKey(k)} />
      ) : storeSid ? (
        <p className="cap">No full model result yet. Run the full model for a firm answer with its likely range, heat-risk impacts, comparisons and the decision pack.</p>
      ) : null}
      <CompareTray rid={rid} />
      {m.refusal?.code === "untrusted_pickle" ? (
        <TrustDialog
          rid={rid}
          error={m.refusal}
          onClose={() => m.setRefusal(null)}
          onTrusted={() => {
            m.setRefusal(null);
            void m.runExact();
          }}
        />
      ) : null}
      {m.refusal?.code === "engine_memory" ? <MemoryDialog error={m.refusal} onClose={() => m.setRefusal(null)} /> : null}
      {m.ladder !== null && storeSid ? <LadderDialog sid={storeSid} rid={rid} edits={doc.edits} editIndex={m.ladder} onClose={() => m.setLadder(null)} /> : null}
    </LabFrame>
  );
}

export default function Design() {
  const { params } = useRoute();
  const rid = params.rid ?? "";
  const sid = params.sid ?? null;
  const detail = useRunDetail(rid);
  const ctxPid = useUi((s) => s.context.projectId);
  const pid = detail.data?.run.project_id ?? ctxPid;
  const grid = useRunGrid(rid);
  const layers = useRunLayers(rid);
  const levers = useLevers(rid);
  const scenario = useScenario(sid);
  if (grid.error) return <EmptyState error={grid.error} />;
  if (sid && scenario.error && !scenario.data) return <EmptyState error={scenario.error} />;
  if (levers.error && !levers.data) return <EmptyState error={levers.error} />;
  if (!grid.data || !levers.data || (sid && !scenario.data)) {
    return (
      <p className="cap" role="status">
        Loading the Lab…
      </p>
    );
  }
  return (
    <>
      {/* Keyed by scenario: opening another one starts with its own preview, compile and job
          (its draft, with undo history, lives on in the draft registry). */}
      <Workbench key={`${rid}|${sid ?? "new"}`} rid={rid} sid={sid} pid={pid} grid={grid.data} groups={layers.data ?? []} levers={levers.data} scenario={scenario.data} />
      {!pid ? (
        <p className="cap">
          This run is not in a project, so scenarios cannot be saved. <Button size="small" variant="ghost" onClick={() => navigate("/projects")}>Projects</Button>
        </p>
      ) : null}
    </>
  );
}
