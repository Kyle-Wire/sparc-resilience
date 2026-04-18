import { createContext, useContext, useState, useCallback, useRef, useEffect } from "react";
import type { ReactNode } from "react";
import type { PipelineEvent } from "@/lib/types";
import { getRunEvents, approveDag, rejectDag } from "@/lib/api";

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
  /** True when MC³ is done and the pipeline is paused awaiting DAG approval. */
  dagApprovalPending: boolean;
  startStage: (stage: number, opts?: { fast?: boolean; skip_gwen?: boolean }) => void;
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

export function PipelineProvider({ children, serverReady }: { children: ReactNode; serverReady: boolean }) {
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

  /** Process a single pipeline event and update training telemetry state. */
  const processEvent = useCallback((event: PipelineEvent) => {
    setEvents((prev) => [...prev, event]);

    if (event.stage !== undefined) {
      setCurrentStage(event.stage);
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

    if (event.type === "complete" || event.type === "error") {
      setIsRunning(false);
      if (event.type === "error") setError(event.message ?? "Unknown error");
      return true; // signal to close socket
    }
    return false;
  }, []);

  // On mount (or server reconnect), rehydrate from buffered events
  useEffect(() => {
    if (!serverReady) return;
    getRunEvents()
      .then((data) => {
        if (data.events.length > 0) {
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

  const _connectWebSocket = useCallback((stage: number, skipCount = 0) => {
    const ws = new WebSocket("ws://127.0.0.1:8008/run/stream");
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

  const startStage = useCallback(
    (stage: number, opts: { fast?: boolean; skip_gwen?: boolean } = {}) => {
      // Close any existing socket
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }

      setDagApprovalPending(false);
      setEvents([]);
      setError(null);
      setIsRunning(true);
      setCurrentStage(stage);
      setStageStatuses({});
      setTraining({
        capacityResults: [],
        epochHistory: [],
        curriculumStage: null,
        curriculumLabel: null,
        convergenceStatus: null,
        healthWarnings: [],
      });

      const ws = new WebSocket("ws://127.0.0.1:8008/run/stream");
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
        if (processEvent(event)) ws.close();
      };

      ws.onerror = () => {
        setError("WebSocket connection failed");
        setIsRunning(false);
      };

      ws.onclose = () => {
        // noop — state managed by onmessage
      };
    },
    [processEvent],
  );

  const cancel = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
    setIsRunning(false);
    setDagApprovalPending(false);
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
      events, isRunning, error, currentStage, training, stageStatuses,
      dagApprovalPending, startStage, cancel, handleApproveDag, handleRejectDag,
    }}>
      {children}
    </PipelineContext.Provider>
  );
}
