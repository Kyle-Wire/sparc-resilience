// The Design workbench's state and behaviour (SPEC §7, §2 J6), moved out of Design.tsx so the page
// is layout only and the step-flow components (components/flow/*, components/result/*) all read one
// typed model. The logic is unchanged from the old in-page Workbench:
// - the draft autosaves (PATCH every 2 s and on blur); a 409 conflict_revision forks a new
//   revision and editing continues on it; undo/redo keeps 50 steps (Ctrl+Z / Ctrl+Shift+Z);
// - the preview (POST /preview, debounced 120 ms, latest request_seq wins) paints ΔT on the
//   map; the brush paints per-lever edit arrays and commits one undo step per stroke;
// - the compile (debounced 300 ms) runs on the evaluated content; Run exact flushes the
//   autosave first and shows fold ticks; the latest exact result on this run is the result.
// Only the derived conveniences at the end (advice, canRun, edit helpers, preset columns) are new,
// and they only read the state above or call the draft's own `change`.
import { useCallback, useEffect, useMemo, useRef, useState, type Dispatch, type SetStateAction } from "react";
import { useStore } from "zustand";
import { ApiError, errorMessage } from "../../../api/client";
import { packBits } from "../../../api/binary";
import { invalidate, mutate } from "../../../api/resource";
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
  usePlans,
  useRegions,
  useResult,
  type CompileResult,
  type CompileWarning,
  type Edit,
  type EmulatorInfo,
  type EngineState,
  type Lever,
  type PreviewResult,
  type Region,
  type Result,
  type Scenario,
  type ScenarioDoc,
} from "../../../api/lab";
import type { Job, LayerGroup } from "../../../api/types";
import type { MapSelectionEvent } from "../../../map/MapView";
import type { GridData } from "../../../map/grid";
import { codecs, navigate, useRoute, useUrlState } from "../../../router";
import { useJobs } from "../../../stores/jobs";
import { toast } from "../../../stores/ui";
import { exactAdvice, type ExactAdvice } from "../components/CompilePanel";
import { useEngineState } from "../components/EngineChip";
import { labHref } from "../components/LabFrame";
import { columnOptions, type ColumnOption, type MapPick } from "../components/SelectionBuilder";
import type { BrushSnapshot } from "./brush";
import { contentKey, newEdit, serializeDoc, usableEdits } from "./doc";
import { createAutosave, docForSave, getDraft, isDirty, moveDraft, uploadedBrushRefs, type Autosave, type DraftStore } from "./draft";
import { useHeatPreview, type HeatPreview } from "./heatPreview";
import { PreviewController } from "./preview";
import { debounce } from "./timing";

/** The latest exact result of a scenario on this run (or the one named in `?res=`). */
export function latestResultOn(s: Scenario | undefined, rid: string, wanted: string | null): string | null {
  if (!s) return null;
  if (wanted && s.results.some((r) => r.id === wanted)) return wanted;
  const mine = s.results.filter((r) => r.run_id === rid && r.kind === "exact").sort((a, b) => b.created_utc.localeCompare(a.created_utc));
  return mine[0]?.id ?? null;
}

export function useKeyboardUndo(store: DraftStore) {
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

/** What the workbench is opened on: loaded by the Design page before the workbench mounts. */
export type WorkbenchInput = {
  rid: string;
  sid: string | null;
  pid: string | null;
  grid: GridData;
  groups: LayerGroup[];
  levers: Lever[];
  scenario: Scenario | undefined;
};

/** Collapsible panes of the three-pane layout (steps rail · map · preview & run). */
export type Panes = { left: boolean; right: boolean };

/**
 * Everything the Design page and its flow components read and do. One instance per opened
 * scenario (the page keys the workbench by `rid|sid`). Fields are grouped as in the old Workbench.
 */
export type WorkbenchModel = {
  // ---- what it is opened on
  rid: string;
  /** The route's scenario id (null on /lab before the first save). */
  sid: string | null;
  /** Project id (null: the run is not in a project, scenarios cannot be saved). */
  pid: string | null;
  grid: GridData;
  groups: LayerGroup[];
  levers: Lever[];
  scenario: Scenario | undefined;
  /** Target unit of the run (e.g. "degF"). */
  unit: string;
  /** Plain lever label for a lever var (the catalogue label, or the var itself). */
  leverLabel: (v: string) => string;

  // ---- the draft (zustand store: doc, undo/redo, brush layers, save state)
  store: DraftStore;
  doc: ScenarioDoc;
  /** Committed brush state (changes with strokes, undo, redo, load). */
  brush: BrushSnapshot;
  /** The draft's saved scenario id (follows creates and forks). */
  storeSid: string | null;
  /** Edits complete enough to evaluate (preview, compile, save). */
  usable: Edit[];
  autosave: Autosave;
  /** Replace edit i (one undo step; `coalesce` merges rapid changes such as typing or sliding). */
  setEdit: (i: number, e: Edit, coalesce?: string) => void;
  /** Append a new "add" edit of `lever` (default: the first lever) at its design dose; returns its index (-1 without levers). */
  addEdit: (lever?: string, init?: Partial<Edit>) => number;
  removeEdit: (i: number) => void;

  // ---- run resources
  emulator: EmulatorInfo | undefined;
  /** The run has an emulator: previews are requested. */
  emuPresent: boolean;
  /** The run is known to have no emulator (no quick estimate; exact runs still work). */
  noEmulator: boolean;
  /** Saved regions of the project (SelectionBuilder "region" picks, neighbourhood presets). */
  regions: Region[];
  planRefs: { ref: string; label: string }[];
  /** Namespaced filter columns (SelectionBuilder). */
  columns: ColumnOption[];
  /** Predictor keys (expert-mode edits). */
  predictors: string[];
  /** Filter column of the observed temperature ("pred:target"), null when the run has none. */
  temperatureColumn: string | null;
  /** Filter column of the residents layer ("layer:people"), null when the run has none. */
  populationColumn: string | null;
  /** Area of one grid cell in km² (null when the cell size is unknown). */
  cellAreaKm2: number | null;

  // ---- engine
  engine: ReturnType<typeof useEngineState>;
  engineState: EngineState | null;
  /** Why an exact run is impossible on this run (no checkpoint, incompatible), else null. */
  runReason: string | null;

  // ---- preview (emulator)
  preview: PreviewResult | null;
  previewBusy: boolean;
  previewError: ApiError | null;
  requestPreview: () => void;
  /** Live heat risk of the preview (runs with campaign humidity), else null. */
  heatPreview: HeatPreview | null;

  // ---- compile
  compile: CompileResult | null;
  compileError: string | null;
  compiling: boolean;
  /** Why the full model should be run (exactAdvice of compile + preview); reasons are technical, see words.plainAdvice. */
  advice: ExactAdvice;
  /** Blocking compile warnings (Run is disabled while any exist). */
  blocking: CompileWarning[];
  /** There is something to run and the run can run exactly (blocking warnings aside). */
  canRun: boolean;

  // ---- map
  layerKey: string | null;
  setLayerKey: (k: string | null) => void;
  mapPick: MapPick | null;
  setMapPick: Dispatch<SetStateAction<MapPick | null>>;
  /** The map tint of the selection being worked on (SelectionBuilder focus), else null. */
  tint: Uint8Array | null;
  setTint: Dispatch<SetStateAction<Uint8Array | null>>;
  brushTick: number;
  onBrushChange: () => void;
  onSelection: (e: MapSelectionEvent) => void;

  // ---- exact run and result
  running: boolean;
  runExact: () => Promise<void>;
  jobId: string | null;
  activeJob: Job | null;
  /** The job to show fold ticks for (just started, or still active on this scenario). */
  shownJob: string | null;
  /** An engine refusal to explain (untrusted_pickle / engine_memory dialogs). */
  refusal: ApiError | null;
  setRefusal: Dispatch<SetStateAction<ApiError | null>>;
  resultId: string | null;
  result: Result | undefined;
  realizedVars: string[];

  // ---- misc UI state
  ladder: number | null;
  setLadder: Dispatch<SetStateAction<number | null>>;
  panes: Panes;
  setPanes: Dispatch<SetStateAction<Panes>>;
  /** Nothing edited or brushed yet: offer the examples (templates). */
  showTemplates: boolean;
  /** A template created a scenario: open it. */
  onTemplateCreated: (s: Scenario) => void;
};

export function useWorkbench({ rid, sid, pid, grid, groups, levers, scenario }: WorkbenchInput): WorkbenchModel {
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
  const [panes, setPanes] = useState<Panes>({ left: true, right: true });
  const [brushTick, setBrushTick] = useState(0);
  const [refusal, setRefusal] = useState<ApiError | null>(null);
  const heatLevers = useMemo(() => Object.keys(compile?.value.levers ?? {}), [compile]);
  const leverLabel = useCallback((v: string) => levers.find((l) => l.var === v)?.label ?? v, [levers]);
  const heatPreview = useHeatPreview(rid, grid, groups, preview?.delta ?? null, heatLevers, leverLabel);

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
        toast("success", "The full model already had a result for these changes");
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
      toast("error", "Could not run the full model", { body: [errorMessage(e), warnings].filter(Boolean).join(" "), action: e instanceof ApiError && e.action ? e.action : undefined });
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
  const noEmulator = !!(emulator.data && !emulator.data.present);

  // ---- derived conveniences for the step flow (read-only views of the state above, plus edit
  // helpers that go through the draft's own `change`, exactly as the ScenarioEditor does)
  const compileValue = compile?.value ?? null;
  const advice = useMemo(() => exactAdvice(compileValue, preview?.summary ?? null, levers), [compileValue, preview, levers]);
  const blocking = useMemo(() => compileValue?.warnings.filter((w) => w.blocking) ?? [], [compileValue]);
  const canRun = (usable.length > 0 || !store.brushLayers.isEmpty()) && !runReason;
  const setEdit = useCallback(
    (i: number, e: Edit, coalesce?: string) => store.getState().change((d) => ({ ...d, edits: d.edits.map((x, k) => (k === i ? e : x)) }), { coalesce }),
    [store],
  );
  const addEdit = useCallback(
    (leverVar?: string, init?: Partial<Edit>) => {
      const lever = levers.find((l) => l.var === leverVar) ?? levers[0];
      if (!lever) return -1;
      const index = store.getState().doc.edits.length;
      store.getState().change((d) => ({ ...d, edits: [...d.edits, { ...newEdit(lever.var, "add", lever.design_dose ?? 0), ...init }] }));
      return index;
    },
    [store, levers],
  );
  const removeEdit = useCallback((i: number) => store.getState().change((d) => ({ ...d, edits: d.edits.filter((_, k) => k !== i) })), [store]);
  const temperatureColumn = useMemo(() => columns.find((c) => c.value === "pred:target")?.value ?? null, [columns]);
  const populationColumn = useMemo(() => columns.find((c) => c.value === "layer:people")?.value ?? null, [columns]);
  const dx = grid.meta.dx_m;
  const cellAreaKm2 = Number.isFinite(dx) && dx > 0 ? (dx * dx) / 1e6 : null;
  const onTemplateCreated = useCallback(
    (s: Scenario) => {
      mutate(`scenario:${s.id}`, s);
      invalidate(`project:${s.project_id}:scenarios`);
      navigate(labHref(rid, `s/${encodeURIComponent(s.id)}`));
    },
    [rid],
  );

  return {
    rid,
    sid,
    pid,
    grid,
    groups,
    levers,
    scenario,
    unit: grid.meta.units.target,
    leverLabel,
    store,
    doc,
    brush,
    storeSid,
    usable,
    autosave,
    setEdit,
    addEdit,
    removeEdit,
    emulator: emulator.data,
    emuPresent,
    noEmulator,
    regions: regions.data ?? [],
    planRefs,
    columns,
    predictors,
    temperatureColumn,
    populationColumn,
    cellAreaKm2,
    engine,
    engineState,
    runReason,
    preview,
    previewBusy,
    previewError,
    requestPreview,
    heatPreview,
    compile: compileValue,
    compileError,
    compiling,
    advice,
    blocking,
    canRun,
    layerKey,
    setLayerKey,
    mapPick,
    setMapPick,
    tint,
    setTint,
    brushTick,
    onBrushChange,
    onSelection,
    running,
    runExact,
    jobId,
    activeJob,
    shownJob,
    refusal,
    setRefusal,
    resultId,
    result: result.data,
    realizedVars,
    ladder,
    setLadder,
    panes,
    setPanes,
    showTemplates,
    onTemplateCreated,
  };
}
