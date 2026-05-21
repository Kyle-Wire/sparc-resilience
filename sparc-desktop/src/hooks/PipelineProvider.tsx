import { createContext, useContext, useState, useCallback, useRef, useEffect } from "react";
import type { ReactNode } from "react";
import type { PipelineEvent } from "@/lib/types";
import { getRunEvents, approveDag, rejectDag, cancelRun } from "@/lib/api";
import { WS_ORIGIN } from "@/lib/server";
import { getToken } from "@/stores/tokenStore";
import { notifyManifestArtifactWritten } from "@/hooks/useManifest";

// ---- Training telemetry derived state ----
export interface CapacityResult {
  hidden_dim: number;
  r2: number;
}

export interface EpochEntry {
  epoch: number;
  n_epochs: number;
  total_loss: number;
  train_phase: "cv" | "retrain" | "swa";
  components?: Record<string, number>;
}

export interface StageStatus {
  stage: number;
  status: "pending" | "running" | "complete" | "failed";
  started_at?: number;
  completed_at?: number;
  error?: string;
  traceback?: string;
  eta_seconds?: number;
  elapsed_seconds?: number;
}

export interface TrainingHealthWarning {
  warning: string;
  component?: string;
  detail?: string;
  timestamp: number;
}

export interface TrainingTelemetry {
  capacityResults: CapacityResult[];
  epochHistory: EpochEntry[];
  curriculumStage: string | null;
  curriculumLabel: string | null;
  convergenceStatus: string | null;
  healthWarnings: TrainingHealthWarning[];
}

export interface PipelineState {
  events: PipelineEvent[];
  isRunning: boolean;
  error: string | null;
  currentStage: number | null;
  training: TrainingTelemetry;
  /** Per-stage status map for the StageStatusTracker. */
  stageStatuses: Record<number, StageStatus>;
  /** Per-stage progress percentage (0–100) from tqdm output. */
  stageProgress: Record<number, number>;
  /** True when MC³ is done and the pipeline is paused awaiting DAG approval. */
  dagApprovalPending: boolean;
  /** Wall-clock ms when the current pipeline run started (persists through page navigation). */
  runStartedAt: number | null;
  /** Wall-clock ms when the current pipeline run ended (success or error). Null while running. */
  runEndedAt: number | null;
  startStage: (stage: number, opts?: { fast?: boolean; skip_gwen?: boolean }) => void;
  /** Run a sequence of stages in order, chaining each on completion. */
  startPipeline: (stages: number[], opts?: { fast?: boolean; skip_gwen?: boolean }) => void;
  cancel: () => void;
  /** Approve the discovered DAG and resume the pipeline. */
  handleApproveDag: () => Promise<void>;
  /** Reject the DAG and cancel the pipeline. */
  handleRejectDag: () => Promise<void>;
}

const PipelineContext = createContext<PipelineState | null>(null);

export function usePipeline(): PipelineState {
  const ctx = useContext(PipelineContext);
  if (!ctx) throw new Error("usePipeline must be used within <PipelineProvider>");
  return ctx;
}

export function PipelineProvider({ children, serverReady, serverLost }: { children: ReactNode; serverReady: boolean; serverLost?: boolean }) {
  const [events, setEvents] = useState<PipelineEvent[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentStage, setCurrentStage] = useState<number | null>(null);
  const [training, setTraining] = useState<TrainingTelemetry>({
    capacityResults: [],
    epochHistory: [],
    curriculumStage: null,
    curriculumLabel: null,
    convergenceStatus: null,
    healthWarnings: [],
  });
  const wsRef = useRef<WebSocket | null>(null);
  const [dagApprovalPending, setDagApprovalPending] = useState(false);
  const [stageStatuses, setStageStatuses] = useState<Record<number, StageStatus>>({});
  const [stageProgress, setStageProgress] = useState<Record<number, number>>({});
  const currentStageRef = useRef<number | null>(null);
  const [runStartedAt, setRunStartedAt] = useState<number | null>(null);
  const [runEndedAt, setRunEndedAt] = useState<number | null>(null);
  /** Remaining stages to run after the current one completes. */
  const stageQueueRef = useRef<number[]>([]);
  /** Options to reuse when chaining to the next stage. */
  const optsRef = useRef<{ fast: boolean; skip_gwen: boolean }>({ fast: false, skip_gwen: false });

  // Terminal event buffering — same rAF approach as usePipelineStream:
  // push into a ref (zero render cost per message), flush to state at display
  // frame rate. Telemetry state updates (training, stageStatuses) remain
  // synchronous since they update small bounded objects.
  const eventsBufferRef = useRef<PipelineEvent[]>([]);
  const eventsRafRef = useRef<number | null>(null);

  const scheduleEventsFlush = useCallback(() => {
    if (eventsRafRef.current !== null) return;
    eventsRafRef.current = requestAnimationFrame(() => {
      eventsRafRef.current = null;
      setEvents([...eventsBufferRef.current]);
    });
  }, []);

  /** Process a single pipeline event and update training telemetry state. */
  const processEvent = useCallback((event: PipelineEvent) => {
    // Stamp with wall-clock time at receipt so terminal timestamps are frozen
    const stamped = { ...event, receivedAt: Date.now() };
    eventsBufferRef.current.push(stamped);
    scheduleEventsFlush();

    if (event.stage !== undefined) {
      setCurrentStage(event.stage);
      currentStageRef.current = event.stage;
    }

    if (event.progress_pct !== undefined) {
      const s = event.stage ?? currentStageRef.current;
      if (s != null) {
        setStageProgress((prev) => ({ ...prev, [s]: event.progress_pct! }));
      }
    }

    // Training telemetry events
    switch (event.type) {
      case "capacity_result":
        if (event.hidden_dim != null && event.r2 != null) {
          setTraining((t) => ({
            ...t,
            capacityResults: [...t.capacityResults, { hidden_dim: event.hidden_dim!, r2: event.r2! }],
          }));
        }
        break;
      case "epoch_update":
        if (event.epoch != null && event.n_epochs != null && event.total_loss != null) {
          setTraining((t) => ({
            ...t,
            epochHistory: [
              ...t.epochHistory,
              {
                epoch: event.epoch!,
                n_epochs: event.n_epochs!,
                total_loss: event.total_loss!,
                train_phase: event.train_phase ?? "cv",
                components: event.components,
              },
            ],
          }));
          // Update ETA on current stage if present
          if (event.stage != null && (event.eta_seconds != null || event.elapsed_seconds != null)) {
            setStageStatuses((prev) => ({
              ...prev,
              [event.stage!]: {
                ...prev[event.stage!],
                stage: event.stage!,
                status: prev[event.stage!]?.status ?? "running",
                eta_seconds: event.eta_seconds,
                elapsed_seconds: event.elapsed_seconds,
              },
            }));
          }
        }
        break;
      case "curriculum_stage":
        setTraining((t) => ({
          ...t,
          curriculumStage: event.curriculum ?? null,
          curriculumLabel: event.label ?? null,
        }));
        break;
      case "convergence":
        setTraining((t) => ({
          ...t,
          convergenceStatus: event.status ?? null,
        }));
        break;
      case "dag_approval_requested":
        setDagApprovalPending(true);
        break;
      case "stage_status":
        if (event.stage != null && event.status) {
          setStageStatuses((prev) => ({
            ...prev,
            [event.stage!]: {
              stage: event.stage!,
              status: event.status as StageStatus["status"],
              started_at: event.started_at ?? prev[event.stage!]?.started_at,
              completed_at: event.completed_at,
              error: event.error,
              traceback: event.traceback,
              eta_seconds: event.eta_seconds,
              elapsed_seconds: event.elapsed_seconds,
            },
          }));
          // Immediately refresh the manifest when a stage finishes so the
          // Insights panels light up as early as possible (T43).
          if (event.status === "complete") {
            notifyManifestArtifactWritten();
          }
        }
        break;
      case "training_health":
        if (event.warning) {
          setTraining((t) => ({
            ...t,
            healthWarnings: [
              ...t.healthWarnings,
              {
                warning: event.warning!,
                component: event.component,
                detail: event.detail,
                timestamp: Date.now(),
              },
            ],
          }));
        }
        break;
    }

    if (event.type === "complete") {
      // Flush immediately so the terminal shows the final event without delay.
      if (eventsRafRef.current !== null) {
        cancelAnimationFrame(eventsRafRef.current);
        eventsRafRef.current = null;
      }
      setEvents([...eventsBufferRef.current]);
      // If there are more stages queued, don't stop the pipeline
      if (stageQueueRef.current.length === 0) {
        setIsRunning(false);
        // Freeze the elapsed timer at the moment of completion
        setRunEndedAt(Date.now());
      }
      return true; // signal to close current socket
    }
    if (event.type === "error") {
      // Flush immediately on error so the terminal shows the failure.
      if (eventsRafRef.current !== null) {
        cancelAnimationFrame(eventsRafRef.current);
        eventsRafRef.current = null;
      }
      setEvents([...eventsBufferRef.current]);
      setIsRunning(false);
      setRunEndedAt(Date.now());
      setError(event.message ?? "Unknown error");
      stageQueueRef.current = []; // clear remaining stages on error
      return true;
    }
    return false;
  }, [scheduleEventsFlush]);

  // On mount (or server reconnect), rehydrate from buffered events
  useEffect(() => {
    if (!serverReady) return;
    getRunEvents()
      .then((data) => {
        if (data.events.length > 0) {
          eventsBufferRef.current = [...data.events];
          setEvents(data.events);
          setCurrentStage(data.current_stage);
          setIsRunning(data.is_running);
          // If still running, reconnect the WebSocket to get live updates
          if (data.is_running && data.current_stage != null) {
            _connectWebSocket(data.current_stage, data.events.length);
          }
        }
      })
      .catch(() => {
        // Server may not support /run/events yet — ignore
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [serverReady]);

  // On server recovery after mid-session loss, reload run state
  const prevServerLost = useRef(false);
  useEffect(() => {
    const wasLost = prevServerLost.current;
    prevServerLost.current = serverLost ?? false;
    if (wasLost && !serverLost) {
      // Server just came back — restore pipeline state
      getRunEvents()
        .then((data) => {
          if (data.events.length > 0) {
            eventsBufferRef.current = [...data.events];
            setEvents(data.events);
            setCurrentStage(data.current_stage);
            setIsRunning(data.is_running);
            if (data.is_running && data.current_stage != null) {
              _connectWebSocket(data.current_stage, data.events.length);
            }
          }
        })
        .catch(() => {});
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [serverLost]);

  const _connectWebSocket = useCallback((stage: number, skipCount = 0) => {
    const ws = new WebSocket(`${WS_ORIGIN}/run/stream?token=${encodeURIComponent(getToken())}`);
    wsRef.current = ws;

    ws.onopen = () => {
      // If reconnecting (skipCount > 0), we only listen — don't send start
      if (skipCount === 0) {
        ws.send(JSON.stringify({ stage, fast: false, skip_gwen: false }));
      }
    };

    ws.onmessage = (msg) => {
      const event: PipelineEvent = JSON.parse(msg.data);
      if (processEvent(event)) ws.close();
    };

    ws.onerror = () => {
      setError("WebSocket connection failed");
      setIsRunning(false);
    };

    ws.onclose = () => {
      // Only clear running state if we didn't get a "complete" event
      // (the onmessage handler already handles that case)
    };
  }, [processEvent]);

  /** Open a WebSocket and run a single stage. If `resetState` is true,
   *  clears all previous events/statuses (used for the first stage in a run). */
  const _runSingleStage = useCallback(
    (stage: number, opts: { fast?: boolean; skip_gwen?: boolean }, resetState: boolean) => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }

      if (resetState) {
        // Clear the event buffer and any pending rAF flush before starting fresh.
        if (eventsRafRef.current !== null) {
          cancelAnimationFrame(eventsRafRef.current);
          eventsRafRef.current = null;
        }
        eventsBufferRef.current = [];
        setDagApprovalPending(false);
        setEvents([]);
        setError(null);
        setStageStatuses({});
        setStageProgress({});
        setRunStartedAt(Date.now());
        setRunEndedAt(null);
        setTraining({
          capacityResults: [],
          epochHistory: [],
          curriculumStage: null,
          curriculumLabel: null,
          convergenceStatus: null,
          healthWarnings: [],
        });
      }

      setIsRunning(true);
      setCurrentStage(stage);

      const ws = new WebSocket(`${WS_ORIGIN}/run/stream?token=${encodeURIComponent(getToken())}`);
      wsRef.current = ws;

      ws.onopen = () => {
        ws.send(
          JSON.stringify({
            stage,
            fast: opts.fast ?? false,
            skip_gwen: opts.skip_gwen ?? false,
          }),
        );
      };

      ws.onmessage = (msg) => {
        const event: PipelineEvent = JSON.parse(msg.data);
        const shouldClose = processEvent(event);
        if (shouldClose) {
          ws.close();
          // Chain to next queued stage on success
          if (event.type === "complete" && stageQueueRef.current.length > 0) {
            const nextStage = stageQueueRef.current.shift()!;
            // Small delay to let the backend finish cleanup
            setTimeout(() => _runSingleStage(nextStage, optsRef.current, false), 200);
          }
        }
      };

      ws.onerror = () => {
        setError("WebSocket connection failed");
        setIsRunning(false);
        stageQueueRef.current = [];
      };

      ws.onclose = () => {
        // noop — state managed by onmessage
      };
    },
    [processEvent],
  );

  const startStage = useCallback(
    (stage: number, opts: { fast?: boolean; skip_gwen?: boolean } = {}) => {
      stageQueueRef.current = [];
      optsRef.current = { fast: opts.fast ?? false, skip_gwen: opts.skip_gwen ?? false };
      _runSingleStage(stage, optsRef.current, true);
    },
    [_runSingleStage],
  );

  const startPipeline = useCallback(
    (stages: number[], opts: { fast?: boolean; skip_gwen?: boolean } = {}) => {
      if (stages.length === 0) return;
      const sorted = [...stages].sort((a, b) => a - b);
      stageQueueRef.current = sorted.slice(1);
      optsRef.current = { fast: opts.fast ?? false, skip_gwen: opts.skip_gwen ?? false };
      _runSingleStage(sorted[0], optsRef.current, true);
    },
    [_runSingleStage],
  );

  const cancel = useCallback(() => {
    stageQueueRef.current = [];
    // Tell the backend to mark the run as idle so a new run can start
    // immediately. The daemon thread finishes its current operation naturally.
    cancelRun().catch(() => { /* best-effort */ });
    wsRef.current?.close();
    wsRef.current = null;
    setIsRunning(false);
    setDagApprovalPending(false);
    // Freeze elapsed time at cancellation point (keep runStartedAt for display)
    setRunEndedAt((prev) => prev ?? Date.now());
  }, []);

  const handleApproveDag = useCallback(async () => {
    await approveDag();
    setDagApprovalPending(false);
  }, []);

  const handleRejectDag = useCallback(async () => {
    await rejectDag();
    setDagApprovalPending(false);
    setIsRunning(false);
  }, []);

  return (
    <PipelineContext.Provider value={{
      events, isRunning, error, currentStage, training, stageStatuses, stageProgress,
      dagApprovalPending, runStartedAt, runEndedAt, startStage, startPipeline, cancel, handleApproveDag, handleRejectDag,
    }}>
      {children}
    </PipelineContext.Provider>
  );
}
