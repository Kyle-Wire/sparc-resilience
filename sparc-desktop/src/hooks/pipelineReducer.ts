/**
 * pipelineReducer — pure event-processing logic extracted from PipelineProvider.
 *
 * `processPipelineEvent` is a pure function: given a state snapshot and one
 * PipelineEvent it returns the next state, a `terminal` flag (close the
 * WebSocket), and an optional `errorMsg`. No React, no side effects.
 *
 * PipelineProvider applies the returned patch to its React state; tests can
 * drive the reducer directly without mounting any component.
 */
import type { PipelineEvent } from "@/lib/types";
import type {
  CapacityResult,
  EpochEntry,
  TrainingTelemetry,
  TrainingHealthWarning,
  StageStatus,
} from "./PipelineProvider";

// Re-export so callers only need this file.
export type { CapacityResult, EpochEntry, TrainingTelemetry, TrainingHealthWarning, StageStatus };

export interface PipelineReducerState {
  currentStage: number | null;
  training: TrainingTelemetry;
  stageStatuses: Record<number, StageStatus>;
  stageProgress: Record<number, number>;
  dagApprovalPending: boolean;
  gwenApprovalPending: boolean;
}

export interface ReducerResult {
  next: PipelineReducerState;
  /** True when the event signals the stream is done (complete or error). */
  terminal: boolean;
  /** Non-null when `terminal` is true and the run ended with an error. */
  errorMsg: string | null;
  /** True when a stage just completed (caller should notify manifest). */
  stageCompleted: boolean;
}

export function initialPipelineReducerState(): PipelineReducerState {
  return {
    currentStage: null,
    training: {
      capacityResults: [],
      epochHistory: [],
      curriculumStage: null,
      curriculumLabel: null,
      convergenceStatus: null,
      healthWarnings: [],
    },
    stageStatuses: {},
    stageProgress: {},
    dagApprovalPending: false,
    gwenApprovalPending: false,
  };
}

export function processPipelineEvent(
  state: PipelineReducerState,
  event: PipelineEvent,
): ReducerResult {
  // Start with a shallow copy; we build patches below.
  let next: PipelineReducerState = { ...state };
  let terminal = false;
  let errorMsg: string | null = null;
  let stageCompleted = false;

  // ── Current stage / progress tracking ────────────────────────────────
  if (event.stage !== undefined) {
    next = { ...next, currentStage: event.stage };
  }

  // ── Event-type dispatch ───────────────────────────────────────────────
  switch (event.type) {
    case "run_progress": {
      const s = event.stage ?? state.currentStage;
      if (s != null && event.pct !== undefined) {
        next = {
          ...next,
          stageProgress: { ...next.stageProgress, [s]: event.pct },
        };
      }
      break;
    }
    case "capacity_result":
      if (event.hidden_dim != null && event.r2 != null) {
        next = {
          ...next,
          training: {
            ...next.training,
            capacityResults: [
              ...next.training.capacityResults,
              { hidden_dim: event.hidden_dim, r2: event.r2 },
            ],
          },
        };
      }
      break;

    case "epoch_update":
      if (event.epoch != null && event.n_epochs != null && event.total_loss != null) {
        const entry: EpochEntry = {
          epoch: event.epoch,
          n_epochs: event.n_epochs,
          total_loss: event.total_loss,
          train_phase: event.train_phase ?? "cv",
          components: event.components,
        };
        let statuses = next.stageStatuses;
        if (event.stage != null && (event.eta_seconds != null || event.elapsed_seconds != null)) {
          statuses = {
            ...statuses,
            [event.stage]: {
              ...statuses[event.stage],
              stage: event.stage,
              status: statuses[event.stage]?.status ?? "running",
              eta_seconds: event.eta_seconds,
              elapsed_seconds: event.elapsed_seconds,
            },
          };
        }
        next = {
          ...next,
          training: {
            ...next.training,
            epochHistory: [...next.training.epochHistory, entry],
          },
          stageStatuses: statuses,
        };
      }
      break;

    case "curriculum_stage":
      next = {
        ...next,
        training: {
          ...next.training,
          curriculumStage: event.curriculum ?? null,
          curriculumLabel: event.label ?? null,
        },
      };
      break;

    case "convergence":
      next = {
        ...next,
        training: {
          ...next.training,
          convergenceStatus: event.status ?? null,
        },
      };
      break;

    case "dag_approval_requested":
      next = { ...next, dagApprovalPending: true };
      break;

    // gwen_approval_requested is not yet in the PipelineEvent union type but
    // the server sends it; cast guards against future type drift.
    case "gwen_approval_requested" as PipelineEvent["type"]:
      next = { ...next, gwenApprovalPending: true };
      break;

    case "stage_status":
      if (event.stage != null && event.status) {
        const prev = state.stageStatuses[event.stage];
        next = {
          ...next,
          stageStatuses: {
            ...next.stageStatuses,
            [event.stage]: {
              stage: event.stage,
              status: event.status as StageStatus["status"],
              started_at: event.started_at ?? prev?.started_at,
              completed_at: event.completed_at,
              error: event.error,
              traceback: event.traceback,
              eta_seconds: event.eta_seconds,
              elapsed_seconds: event.elapsed_seconds,
            },
          },
        };
        if (event.status === "complete") stageCompleted = true;
      }
      break;

    case "training_health":
      if (event.warning) {
        const warning: TrainingHealthWarning = {
          warning: event.warning,
          component: event.component,
          detail: event.detail,
          timestamp: Date.now(),
        };
        next = {
          ...next,
          training: {
            ...next.training,
            healthWarnings: [...next.training.healthWarnings, warning],
          },
        };
      }
      break;

    case "complete":
      terminal = true;
      break;

    case "error":
      terminal = true;
      errorMsg = event.message ?? "Unknown error";
      break;

    default:
      // log / metric / unknown — no state change beyond stage/progress above
      break;
  }

  return { next, terminal, errorMsg, stageCompleted };
}
