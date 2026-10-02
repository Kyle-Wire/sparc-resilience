// Scenario Lab — Design (`/r/:rid/lab`) and Scenario + result (`/r/:rid/lab/s/:sid`), SPEC §7,
// §2 J6. Three collapsible panes (editor 360 px · map · compile 340 px) over the compare tray:
// - the draft autosaves (PATCH every 2 s and on blur); a 409 conflict_revision forks a new
//   revision and editing continues on it; undo/redo keeps 50 steps;
// - the preview (POST /preview, debounced 120 ms, latest request_seq wins) paints ΔT on the
//   map; the brush paints per-lever edit arrays;
// - the compile panel recommends an exact run when the preview cannot be trusted, and Run
//   exact shows fold ticks; the latest exact result on this run opens in the inspector.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useStore } from "zustand";
import { ApiError, errorMessage } from "../../api/client";
import { packBits } from "../../api/binary";
import { invalidate, mutate } from "../../api/resource";
import {
  compileScenario,
  createScenario,
  forkScenario,
  patchScenario,
  postPreview,
  runScenarioExact,
  uploadEditBlob,
  uploadMaskBlob,
  useEmulator,
  useLevers,
  usePlans,
  useRegions,
  useResult,
  useScenario,
  type CompileResult,
  type Lever,
  type PreviewResult,
  type Scenario,
} from "../../api/lab";
import type { LayerGroup } from "../../api/types";
import { Button } from "../../components/ui/Button";
import { ActionButton, EmptyState } from "../../components/ui/EmptyState";
import { IconButton } from "../../components/ui/IconButton";
import { useRunDetail } from "../../layouts/resources";
import type { MapSelectionEvent } from "../../map/MapView";
import { useRunGrid, useRunLayers } from "../../map/data";
import type { GridData } from "../../map/grid";
import { Link, codecs, navigate, useRoute, useUrlState } from "../../router";
import { useJobs } from "../../stores/jobs";
import { toast, useUi } from "../../stores/ui";
import { CompareTray } from "./components/CompareTray";
import { CompilePanel } from "./components/CompilePanel";
import { DesignMap, PREVIEW_KEY } from "./components/DesignMap";
import { LadderDialog } from "./components/Dialogs";
import { MemoryDialog, TrustDialog, useEngineState } from "./components/EngineChip";
import { LabFrame, labHref } from "./components/LabFrame";
import { ResultInspector } from "./components/ResultInspector";
import { ScenarioEditor } from "./components/ScenarioEditor";
import { columnOptions, type MapPick } from "./components/SelectionBuilder";
import { TemplateGallery } from "./components/TemplateGallery";
import { contentKey, serializeDoc, usableEdits } from "./model/doc";
import { createAutosave, docForSave, getDraft, isDirty, moveDraft, uploadedBrushRefs, type DraftStore } from "./model/draft";
import { PreviewController } from "./model/preview";
import { debounce } from "./model/timing";

/** The latest exact result of a scenario on this run (or the one named in `?res=`). */
export function latestResultOn(s: Scenario | undefined, rid: string, wanted: string | null): string | null {
  if (!s) return null;
  if (wanted && s.results.some((r) => r.id === wanted)) return wanted;
  const mine = s.results.filter((r) => r.run_id === rid && r.kind === "exact").sort((a, b) => b.created_utc.localeCompare(a.created_utc));
  return mine[0]?.id ?? null;
}

function useKeyboardUndo(store: DraftStore) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!(e.ctrlKey || e.metaKey)) return;
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable)) return; // native text undo
      const k = e.key.toLowerCase();
      if (k === "z" && !e.shiftKey) {
        e.preventDefault();
        store.getState().undo();
      } else if ((k === "z" && e.shiftKey) || k === "y") {
        e.preventDefault();
        store.getState().redo();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [store]);
}

type WorkbenchProps = {
  rid: string;
  sid: string | null;
  pid: string | null;
  grid: GridData;
  groups: LayerGroup[];
  levers: Lever[];
  scenario: Scenario | undefined;
};

function Workbench({ rid, sid, pid, grid, groups, levers, scenario }: WorkbenchProps) {
  const store = useMemo(() => getDraft(rid, sid, grid.n), [rid, sid, grid.n]);
  const doc = useStore(store, (s) => s.doc);
  const brush = useStore(store, (s) => s.brush);
  const storeSid = useStore(store, (s) => s.sid);
  const { query } = useRoute();
  const emulator = useEmulator(rid);
  const regions = useRegions(rid);
  const plans = usePlans(rid);
  const engine = useEngineState(rid);
  const [layerKey, setLayerKey] = useUrlState("layer", codecs.optString());
  const [preview, setPreview] = useState<PreviewResult | null>(null);
  const [previewBusy, setPreviewBusy] = useState(false);
  const [previewError, setPreviewError] = useState<ApiError | null>(null);
  const [compile, setCompile] = useState<{ key: string; value: CompileResult } | null>(null);
  const [compileError, setCompileError] = useState<string | null>(null);
  const [compiling, setCompiling] = useState(false);
  const [mapPick, setMapPick] = useState<MapPick | null>(null);
  const [tint, setTint] = useState<Uint8Array | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [ladder, setLadder] = useState<number | null>(null);
  const [panes, setPanes] = useState<{ left: boolean; right: boolean }>({ left: true, right: true });
  const [brushTick, setBrushTick] = useState(0);
  const [refusal, setRefusal] = useState<ApiError | null>(null);

  // Load the server copy into a fresh draft, or into a clean one the server has moved past
  // (renamed in the library, say). A server update that only adds results or changes the
  // status keeps the local draft and its undo history.
  useEffect(() => {
    if (!scenario) return;
    const s = store.getState();
    if (s.sid !== scenario.id) {
      s.load(scenario);
      return;
    }
    if (isDirty(s) || !store.brushLayers.isEmpty() || s.serverUpdated === scenario.updated_utc) return;
    if (serializeDoc(scenario.doc) === serializeDoc(s.doc)) s.rebase(scenario, s.forkedFrom);
    else s.load(scenario);
  }, [scenario, store]);

  // Autosave: every 2 s, on editor blur, when the window loses focus and when leaving.
  const autosave = useMemo(
    () =>
      createAutosave(
        store,
        {
          create: (d) => {
            if (!pid) return Promise.reject(new Error("Open this run from its project to save scenarios."));
            return createScenario(pid, d);
          },
          patch: (id, d) => patchScenario(id, { doc: d }),
          fork: (id, d) => forkScenario(id, { doc: d }),
          uploadBrush: (_lever, body, count) => uploadEditBlob(rid, body, count),
        },
        {
          onCreated: (s) => {
            mutate(`scenario:${s.id}`, s);
            moveDraft(rid, null, s.id);
            invalidate(`project:${s.project_id}:scenarios`);
            navigate(`${labHref(rid, `s/${encodeURIComponent(s.id)}`)}${window.location.search}`, { replace: true, scroll: false });
          },
          onForked: (s, from) => {
            mutate(`scenario:${s.id}`, s);
            moveDraft(rid, from, s.id);
            invalidate(`project:${s.project_id}:scenarios`);
            toast("info", `Your edits continue on revision ${s.revision}`, { body: "The previous revision has an exact result, so it was kept and a new revision was forked from it." });
            navigate(labHref(rid, `s/${encodeURIComponent(s.id)}`), { replace: true, scroll: false });
          },
        },
      ),
    [store, pid, rid],
  );
  useEffect(() => {
    autosave.start();
    const onHide = () => void autosave.flush();
    const onVis = () => document.visibilityState === "hidden" && onHide();
    window.addEventListener("blur", onHide);
    document.addEventListener("visibilitychange", onVis);
    return () => {
      window.removeEventListener("blur", onHide);
      document.removeEventListener("visibilitychange", onVis);
      autosave.stop();
      void autosave.flush();
    };
  }, [autosave]);
  useKeyboardUndo(store);

  // Preview: debounced, latest request_seq wins.
  const controller = useMemo(
    () =>
      new PreviewController({
        send: (body) => postPreview(rid, body),
        onResult: (r) => {
          setPreview(r);
          setPreviewError(null);
        },
        onError: (e) => setPreviewError(e instanceof ApiError ? e : new ApiError(0, "internal", errorMessage(e))),
        onBusy: setPreviewBusy,
      }),
    [rid],
  );
  useEffect(() => () => controller.dispose(), [controller]);
  const usable = useMemo(() => usableEdits(doc.edits), [doc.edits]);
  // Preview only once the run is known to have an emulator (no 404 no_emulator on every open of a run without one).
  const emuPresent = emulator.data?.present === true;
  const requestPreview = useCallback(() => {
    const brushPayload = store.brushLayers.previewPayload();
    if (!emuPresent) return;
    if (!usable.length && !Object.keys(brushPayload).length) {
      controller.reset();
      setPreview(null);
      return;
    }
    const sid = store.getState().sid;
    controller.request({ edits: usable, brush: Object.keys(brushPayload).length ? brushPayload : undefined, options: doc.options, ...(sid ? { scenario_id: sid } : {}) });
  }, [controller, store, usable, doc.options, emuPresent]);
  const previewKey = useMemo(() => contentKey({ ...doc, edits: usable }), [doc, usable]);
  // Re-preview only when the evaluated content or the brush changes (not on name or notes).
  const requestRef = useRef(requestPreview);
  requestRef.current = requestPreview;
  useEffect(() => requestRef.current(), [previewKey, brush, emuPresent]);

  // Compile: debounced 300 ms on the evaluated content (brushed levers count once uploaded).
  const savedVersion = useStore(store, (s) => s.savedVersion);
  const compileDoc = useMemo(() => docForSave({ ...doc, edits: usable }, uploadedBrushRefs(store)), [doc, usable, store, savedVersion, brush]);
  const compileKey = useMemo(() => contentKey(compileDoc), [compileDoc]);
  const runCompile = useMemo(
    () =>
      debounce((key: string, d: typeof doc) => {
        setCompiling(true);
        compileScenario(rid, d).then(
          (value) => {
            setCompile({ key, value });
            setCompileError(null);
            setCompiling(false);
          },
          (e: unknown) => {
            setCompileError(errorMessage(e));
            setCompiling(false);
          },
        );
      }, 300),
    [rid],
  );
  useEffect(() => {
    if (!compileDoc.edits.length) {
      runCompile.cancel();
      setCompile(null);
      setCompileError(null);
      return;
    }
    runCompile(compileKey, compileDoc);
    return () => runCompile.cancel();
    // compileKey covers compileDoc
  }, [compileKey, runCompile]);

  // Brush strokes: preview live, commit to the undo history (one step per stroke).
  const commitBrush = useMemo(() => debounce(() => store.getState().commitBrush({ coalesce: "brush" }), 150), [store]);
  const onBrushChange = useCallback(() => {
    requestPreview();
    commitBrush();
    setBrushTick((t) => t + 1);
  }, [requestPreview, commitBrush]);
  useEffect(() => () => commitBrush.flush(), [commitBrush]);

  const onSelection = useCallback(
    (e: MapSelectionEvent) => {
      if (e.spec) setMapPick({ spec: e.spec, label: e.label });
      else if (e.mask) {
        const mask = e.mask;
        uploadMaskBlob(rid, packBits(mask)).then(
          (spec) => setMapPick({ spec, label: e.label || "brushed cells" }),
          (err: unknown) => toast("error", "Could not keep the map selection", { body: errorMessage(err) }),
        );
      } else setMapPick(null);
    },
    [rid],
  );

  const resultId = latestResultOn(scenario && scenario.id === storeSid ? scenario : undefined, rid, query.get("res"));
  const activeJob = useJobs((s) => Object.values(s.jobs).find((j) => j.scenario_id !== null && j.scenario_id === storeSid && j.run_id === rid && ["queued", "starting", "running", "cancelling", "blocked"].includes(j.status)) ?? null);
  const shownJob = jobId ?? activeJob?.id ?? null;

  const runExact = async () => {
    setRunning(true);
    try {
      await autosave.flush();
      const id = store.getState().sid;
      if (!id) throw new Error(store.getState().saveError ?? "Save the scenario first.");
      const r = await runScenarioExact(id, rid);
      if (r.cached) {
        toast("success", "Exact result was already computed (cached)");
        invalidate(`scenario:${id}`);
      }
      if (r.job) {
        useJobs.getState().upsert(r.job);
        setJobId(r.job.id);
      }
    } catch (e) {
      if (e instanceof ApiError && (e.code === "untrusted_pickle" || e.code === "engine_memory")) {
        setRefusal(e);
        return;
      }
      const warnings = e instanceof ApiError && Array.isArray(e.detail?.warnings) ? (e.detail!.warnings as { message: string }[]).map((w) => w.message).join(" ") : "";
      toast("error", "Could not run exact", { body: [errorMessage(e), warnings].filter(Boolean).join(" "), action: e instanceof ApiError && e.action ? e.action : undefined });
    } finally {
      setRunning(false);
    }
  };

  const engineState = engine.state;
  const runReason = engineState === "no_checkpoint" ? "This run has no checkpoint, so only the preview is available." : engineState === "incompatible" ? "The checkpoint cannot be loaded by this code." : null;
  const columns = useMemo(() => columnOptions(groups), [groups]);
  const predictors = useMemo(() => groups.find((g) => g.id === "inputs")?.layers.map((l) => l.key).filter((k) => k !== "zone") ?? [], [groups]);
  const planRefs = useMemo(() => (plans.data ?? []).map((pl) => ({ ref: `plan:${pl.id}`, label: pl.name })), [plans.data]);
  const result = useResult(resultId, rid);
  const realizedVars = useMemo(() => Object.keys(result.data?.realized ?? {}), [result.data]);
  const showTemplates = !doc.edits.length && store.brushLayers.isEmpty();
  const noEmulator = emulator.data && !emulator.data.present;

  return (
    <LabFrame
      rid={rid}
      hideEngine
      title={doc.name || "Untitled scenario"}
      actions={
        <>
          {storeSid ? (
            <Link className="btn small" to={labHref(rid)} onClick={() => void autosave.flush()}>
              New scenario
            </Link>
          ) : null}
          <IconButton icon="chevronLeft" label={panes.left ? "Hide the editor" : "Show the editor"} pressed={!panes.left} onClick={() => setPanes((p) => ({ ...p, left: !p.left }))} />
          <IconButton icon="chevronRight" label={panes.right ? "Hide the compile panel" : "Show the compile panel"} pressed={!panes.right} onClick={() => setPanes((p) => ({ ...p, right: !p.right }))} />
        </>
      }
    >
      <div className="panes-3 lab-panes" data-left={panes.left ? undefined : "collapsed"} data-right={panes.right ? undefined : "collapsed"}>
        <aside className="lab-pane" aria-label="Scenario editor">
          <ScenarioEditor
            rid={rid}
            store={store}
            levers={levers}
            grid={grid}
            columns={columns}
            regions={regions.data ?? []}
            planRefs={planRefs}
            predictors={predictors}
            compile={compile?.value ?? null}
            mapPick={mapPick}
            onMask={setTint}
            onLadder={(i) => setLadder(i)}
            onBlur={() => {
              setTint(null);
              void autosave.flush();
            }}
          />
          {showTemplates ? (
            <TemplateGallery
              pid={pid}
              rid={rid}
              levers={levers}
              onCreated={(s) => {
                mutate(`scenario:${s.id}`, s);
                invalidate(`project:${s.project_id}:scenarios`);
                navigate(labHref(rid, `s/${encodeURIComponent(s.id)}`));
              }}
            />
          ) : null}
        </aside>
        <section className="lab-pane lab-map" aria-label="Map">
          {noEmulator ? (
            <div className="callout row" role="note">
              <span>No emulator for this run, so there is no instant preview. Exact runs still work.</span>
              {emulator.data?.action ? <ActionButton action={emulator.data.action} size="small" variant="default" /> : null}
            </div>
          ) : null}
          <DesignMap
            rid={rid}
            grid={grid}
            groups={groups}
            levers={levers}
            store={store}
            brushTick={brushTick}
            brushState={brush}
            preview={preview}
            previewBusy={previewBusy}
            previewError={previewError ? previewError.message : null}
            resultId={resultId}
            realizedVars={realizedVars}
            tint={tint}
            layerKey={layerKey ?? PREVIEW_KEY}
            onLayerChange={(k) => setLayerKey(k === PREVIEW_KEY ? null : k)}
            onBrushChange={onBrushChange}
            onSelection={onSelection}
          />
          {previewError?.action ? <ActionButton action={previewError.action} size="small" variant="default" /> : null}
        </section>
        <aside className="lab-pane" aria-label="Compile and run">
          <CompilePanel
            rid={rid}
            compile={compile?.value ?? null}
            compileError={compileError}
            compiling={compiling}
            preview={preview?.summary ?? null}
            levers={levers}
            unit={grid.meta.units.target}
            canRun={(usable.length > 0 || !store.brushLayers.isEmpty()) && !runReason}
            runReason={runReason}
            running={running}
            jobId={shownJob}
            onRunExact={() => void runExact()}
          />
        </aside>
      </div>
      {resultId ? (
        <ResultInspector rid={rid} resId={resultId} pid={pid} unit={grid.meta.units.target} nFolds={grid.meta.n_folds} onShowLayer={(k) => setLayerKey(k)} />
      ) : storeSid ? (
        <p className="cap">No exact result on this run yet. Run exact for uncertainty, impacts, comparisons and the decision pack.</p>
      ) : null}
      <CompareTray rid={rid} />
      {refusal?.code === "untrusted_pickle" ? (
        <TrustDialog
          rid={rid}
          error={refusal}
          onClose={() => setRefusal(null)}
          onTrusted={() => {
            setRefusal(null);
            void runExact();
          }}
        />
      ) : null}
      {refusal?.code === "engine_memory" ? <MemoryDialog error={refusal} onClose={() => setRefusal(null)} /> : null}
      {ladder !== null && storeSid ? <LadderDialog sid={storeSid} rid={rid} edits={doc.edits} editIndex={ladder} onClose={() => setLadder(null)} /> : null}
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
