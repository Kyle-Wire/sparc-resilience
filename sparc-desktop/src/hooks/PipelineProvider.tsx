import { createContext, useContext, useState, useCallback, useRef, useEffect } from "react";
import type { ReactNode } from "react";
import type { PipelineEvent } from "@/lib/types";
import { getRunEvents, approveDag, rejectDag, cancelRun } from "@/lib/api";
import { WS_ORIGIN } from "@/lib/server";
import { getToken } from "@/stores/tokenStore";
import { notifyManifestArtifactWritten } from "@/hooks/useManifest";
import {
  processPipelineEvent,
  initialPipelineReducerState,
  type PipelineReducerState,
} from "@/hooks/pipelineReducer";

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
  /** True when Stage 1 is done and the pipeline is awaiting GWEN variable approval. */
  gwenApprovalPending: boolean;
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
  const [runStartedAt, setRunStartedAt] = useState<number | null>(null);
  const [runEndedAt, setRunEndedAt] = useState<number | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // Reducer-managed state — all event-driven fields live here so they can be
  // exercised by pipelineReducer.test.ts without mounting any React component.
  const [reducerState, setReducerState] = useState<PipelineReducerState>(
    initialPipelineReducerState,
  );
  // Convenience destructures so the rest of the component reads identically.
  const { currentStage, training, stageStatuses, stageProgress,
          dagApprovalPending, gwenApprovalPending } = reducerState;

  const currentStageRef = useRef<number | null>(null);
  /** Remaining stages to run after the current one completes. */
  const stageQueueRef = useRef<number[]>([]);
  /** Options to reuse when chaining to the next stage. */
  const optsRef = useRef<{ fast: boolean; skip_gwen: boolean }>({ fast: false, skip_gwen: false });

  // Terminal event buffering — same rAF approach as usePipelineStream:
  // push into a ref (zero render cost per message), flush to state at display
  // frame rate. Reducer state updates remain synchronous since they update
  // small bounded objects.
  const eventsBufferRef = useRef<PipelineEvent[]>([]);
  const eventsRafRef = useRef<number | null>(null);

  const scheduleEventsFlush = useCallback(() => {
    if (eventsRafRef.current !== null) return;
    eventsRafRef.current = requestAnimationFrame(() => {
      eventsRafRef.current = null;
      setEvents([...eventsBufferRef.current]);
    });
  }, []);

  /** Process a single pipeline event via the pure reducer, then apply state. */
  const processEvent = useCallback((event: PipelineEvent) => {
    // Stamp with wall-clock time at receipt so terminal timestamps are frozen
    const stamped = { ...event, receivedAt: Date.now() };
    eventsBufferRef.current.push(stamped);
    scheduleEventsFlush();

    setReducerState((prev) => {
      const { next, stageCompleted } = processPipelineEvent(prev, event);
      // Side-effect: notify manifest when a stage finishes (must live outside
      // the pure reducer since notifyManifestArtifactWritten is impure).
      if (stageCompleted) notifyManifestArtifactWritten();
      // Keep the imperative currentStageRef in sync for _connectWebSocket.
      if (next.currentStage !== prev.currentStage) {
        currentStageRef.current = next.currentStage;
      }
      return next;
    });

    if (event.stage !== undefined) {
      currentStageRef.current = event.stage;
    }

    if (event.type === "complete") {
      if (eventsRafRef.current !== null) {
        cancelAnimationFrame(eventsRafRef.current);
        eventsRafRef.current = null;
      }
      setEvents([...eventsBufferRef.current]);
      if (stageQueueRef.current.length === 0) {
        setIsRunning(false);
        setRunEndedAt(Date.now());
      }
      return true;
    }
    if (event.type === "error") {
      if (eventsRafRef.current !== null) {
        cancelAnimationFrame(eventsRafRef.current);
        eventsRafRef.current = null;
      }
      setEvents([...eventsBufferRef.current]);
      setIsRunning(false);
      setRunEndedAt(Date.now());
      setError(event.message ?? "Unknown error");
      stageQueueRef.current = [];
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
          setReducerState((prev) => ({ ...prev, currentStage: data.current_stage }));
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
            setReducerState((prev) => ({ ...prev, currentStage: data.current_stage }));
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
        setEvents([]);
        setError(null);
        setRunStartedAt(Date.now());
        setRunEndedAt(null);
        setReducerState(initialPipelineReducerState());
      }

      setIsRunning(true);
      setReducerState((prev) => ({ ...prev, currentStage: stage }));

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
    setReducerState((prev) => ({ ...prev, dagApprovalPending: false, gwenApprovalPending: false }));
    // Freeze elapsed time at cancellation point (keep runStartedAt for display)
    setRunEndedAt((prev) => prev ?? Date.now());
  }, []);

  const handleApproveDag = useCallback(async () => {
    await approveDag();
    setReducerState((prev) => ({ ...prev, dagApprovalPending: false }));
  }, []);

  const handleRejectDag = useCallback(async () => {
    await rejectDag();
    setReducerState((prev) => ({ ...prev, dagApprovalPending: false }));
    setIsRunning(false);
  }, []);

  return (
    <PipelineContext.Provider value={{
      events, isRunning, error, currentStage, training, stageStatuses, stageProgress,
      dagApprovalPending, gwenApprovalPending, runStartedAt, runEndedAt, startStage, startPipeline, cancel, handleApproveDag, handleRejectDag,
    }}>
      {children}
    </PipelineContext.Provider>
  );
}
