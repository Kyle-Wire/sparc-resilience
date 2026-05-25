import type { PipelineEvent } from "./types";

const STAGE_NAMES: Record<number, string> = {
  0: "Correlogram",
  1: "GWEN",
  2: "Validation",
  3: "Inference",
  4: "Simulation",
};

const STAGE_DESCRIPTIONS: Record<number, string> = {
  0: "Spatial autocorrelation analysis + pipeline config",
  1: "Variable selection via GWEN weighting",
  2: "Enhanced spatial cross-validation",
  3: "Causal validation (MC3 / NUTS)",
  4: "Scenario simulation + counterfactuals",
};

type LogLevel = "info" | "success" | "warn" | "error" | "debug" | "milestone" | "log";
export type LogLine = { text: string; level: LogLevel; ts: string };

function getTs(evt: PipelineEvent & { receivedAt?: number }): string {
  const t = evt.receivedAt ?? Date.now();
  return new Date(t).toTimeString().slice(0, 8);
}

// ---------------------------------------------------------------------------
// Typed handlers — one per event type
// ---------------------------------------------------------------------------

type AnyEvent = PipelineEvent & Record<string, unknown>;

function handleStageStatus(e: AnyEvent, ts: string): LogLine | null {
  const name = STAGE_NAMES[e.stage as number] ?? `Stage ${e.stage}`;
  const desc = STAGE_DESCRIPTIONS[e.stage as number] ?? "";
  if (e.status === "running") {
    return { text: `[STAGE] Starting ${name} — ${desc}`, level: "info", ts };
  }
  if (e.status === "complete") {
    const dur = e.elapsed_seconds ? ` (${(e.elapsed_seconds as number).toFixed(1)}s)` : "";
    return { text: `✓ ${name} completed${dur}`, level: "success", ts };
  }
  if (e.status === "failed") {
    return { text: `✕ ${name} failed: ${(e.error as string) ?? "unknown error"}`, level: "error", ts };
  }
  return null;
}

function handleEpochUpdate(e: AnyEvent, ts: string): LogLine {
  const eta = e.eta_seconds ? `  eta=${Math.round(e.eta_seconds as number)}s` : "";
  const phase = e.train_phase ? ` [${e.train_phase}]` : "";
  const loss = typeof e.total_loss === "number" ? (e.total_loss as number).toFixed(4) : "?";
  return {
    text: `[EPOCH]${phase} ${e.epoch} / ${e.n_epochs}  loss=${loss}${eta}`,
    level: "debug",
    ts,
  };
}

function handleMetric(e: AnyEvent, ts: string): LogLine {
  const fold = e.fold != null ? `  fold ${e.fold}` : "";
  const model = e.model ? `  ${e.model}` : "";
  const val = typeof e.value === "number" ? (e.value as number).toFixed(4) : e.value;
  return {
    text: `[METRIC]${model}${fold}  —  ${(e.metric as string) ?? "metric"} = ${val}`,
    level: "info",
    ts,
  };
}

function handleFoldStart(e: AnyEvent, ts: string): LogLine {
  const counts =
    e.n_train != null && e.n_test != null
      ? `  —  ${(e.n_train as number).toLocaleString()} train  /  ${(e.n_test as number).toLocaleString()} test`
      : "";
  return { text: `[FOLD] Fold ${e.fold} / ${e.n_folds}${counts}`, level: "info", ts };
}

function handleFoldComplete(e: AnyEvent, ts: string): LogLine {
  const dur = e.elapsed_seconds != null ? `  (${Math.round(e.elapsed_seconds as number)}s)` : "";
  return { text: `✓ Fold ${e.fold} complete${dur}`, level: "success", ts };
}

function handleModelResult(e: AnyEvent, ts: string): LogLine {
  return {
    text: `[MODEL] ${(e.model as string) ?? "Model"}  R²=${typeof e.r2 === "number" ? (e.r2 as number).toFixed(4) : "?"}  RMSE=${typeof e.rmse === "number" ? (e.rmse as number).toFixed(4) : "?"}`,
    level: "success",
    ts,
  };
}

function handleModelStart(e: AnyEvent, ts: string): LogLine {
  return { text: `[MODEL] Training ${(e.model_name as string) ?? "model"}`, level: "info", ts };
}

function handleModelComplete(e: AnyEvent, ts: string): LogLine {
  const r2 = e.r2 != null ? `  R²=${(e.r2 as number).toFixed(4)}` : "";
  return { text: `✓ ${(e.model_name as string) ?? "Model"} complete${r2}`, level: "success", ts };
}

function handleLog(e: AnyEvent, ts: string): LogLine | null {
  const msg = (e.message as string) ?? "";
  if (!msg.trim()) return null;
  return { text: msg, level: "log", ts };
}

function handleCheckpoint(e: AnyEvent, ts: string): LogLine {
  const lvl: LogLevel =
    e.level === "success" ? "success" : e.level === "warn" ? "warn" : "milestone";
  return { text: (e.message as string) ?? "Checkpoint", level: lvl, ts };
}

// ---------------------------------------------------------------------------
// Dispatch map — eliminates the 20-branch if-chain
// ---------------------------------------------------------------------------

type Handler = (e: AnyEvent, ts: string) => LogLine | null;

const EVENT_HANDLERS: Partial<Record<string, Handler>> = {
  stage_status: handleStageStatus,
  epoch_update: handleEpochUpdate,
  metric: handleMetric,
  fold_start: handleFoldStart,
  fold_complete: handleFoldComplete,
  model_result: handleModelResult,
  model_start: handleModelStart,
  model_complete: handleModelComplete,
  convergence: (e, ts) => ({ text: `[INFO] Convergence: ${(e.status as string) ?? "unknown"}`, level: "info", ts }),
  curriculum_stage: (e, ts) => ({ text: `[INFO] Curriculum phase — ${(e.label as string) ?? (e.curriculum as string) ?? "next phase"}`, level: "info", ts }),
  capacity_result: (e, ts) => ({ text: `[INFO] Capacity check  dim=${e.hidden_dim}  R²=${typeof e.r2 === "number" ? (e.r2 as number).toFixed(4) : "?"}`, level: "debug", ts }),
  error: (e, ts) => ({ text: `[ERROR] ${(e.message as string) ?? "An error occurred"}`, level: "error", ts }),
  complete: (_e, ts) => ({ text: "✓ Pipeline complete", level: "success", ts }),
  dag_approval_requested: (_e, ts) => ({ text: "[WARN] DAG approval required — review on the DAG page", level: "warn", ts }),
  training_health: (e, ts) => ({ text: `[WARN] ${(e.warning as string) ?? "Training health warning"}`, level: "warn", ts }),
  progress: (e, ts) => ({ text: `[INFO] ${(e.message as string) ?? `Progress: ${(e.pct as number) ?? 0}%`}`, level: "debug", ts }),
  checkpoint: handleCheckpoint,
  log: handleLog,
};

export function eventToLogLine(
  evt: PipelineEvent & { receivedAt?: number },
): LogLine | null {
  const ts = getTs(evt);
  const type = (evt as AnyEvent).type as string ?? "";
  const handler = EVENT_HANDLERS[type];
  return handler ? handler(evt as AnyEvent, ts) : null;
}
