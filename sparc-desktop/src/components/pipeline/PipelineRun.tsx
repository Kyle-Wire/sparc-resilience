import { useState, useEffect, useRef } from "react";
import { usePipeline } from "@/hooks/PipelineProvider";
import type { PipelineEvent } from "@/lib/types";
import { CapacitySweepView } from "@/components/training/CapacitySweepView";
import { EpochLossChart } from "@/components/training/EpochLossChart";
import { ConvergenceBadge } from "@/components/training/ConvergenceBadge";

const STAGES = [
  { value: 0, label: "0 — Correlogram" },
  { value: 1, label: "1 — GWEN" },
  { value: 2, label: "2 — Spatial CV" },
  { value: 3, label: "3 — Causal" },
  { value: 4, label: "4 — Scenarios" },
];

/**
 * Model-level weight map for Stage 2 progress.
 * Each model contributes a slice of the overall stage progress.
 */
const STAGE2_MODEL_WEIGHTS: Record<string, [number, number]> = {
  ols:     [5,  15],
  gwr:     [15, 35],
  gwrf:    [35, 55],
  ggpgam:  [55, 70],
};

/** Heuristic phase ordering for non-Stage-2 stages. */
const STAGE_PHASES: Record<number, string[]> = {
  0: ["Correlogram analysis", "Analyzing variable", "Pipeline configuration"],
  1: ["GWEN variable selection", "GWEN results"],
  3: ["Causal validation"],
  4: ["Scenario simulation"],
};

function phaseProgress(stage: number | undefined, phase: string | undefined): number {
  if (stage == null || !phase) return 0;
  const phases = STAGE_PHASES[stage];
  if (!phases) return 0;
  const idx = phases.findIndex((p) => phase.startsWith(p));
  if (idx < 0) return 0;
  return Math.round(((idx + 1) / phases.length) * 100);
}

/** Extract model-checkpoint progress for Stage 2. */
function modelCheckpointProgress(events: PipelineEvent[]): number | null {
  for (let i = events.length - 1; i >= 0; i--) {
    const e = events[i];
    if (e.progress_pct !== undefined && e.model) {
      return e.progress_pct;
    }
  }
  return null;
}

/** Build a list of model milestones for the progress bar visualization. */
function getModelMilestones(events: PipelineEvent[]): { name: string; done: boolean; pct: number }[] {
  const models = Object.entries(STAGE2_MODEL_WEIGHTS).map(([name, [, endPct]]) => ({
    name: name.toUpperCase(),
    done: false,
    pct: endPct,
  }));

  const doneModels = new Set<string>();
  for (const e of events) {
    if (e.phase?.includes("complete") && e.model) {
      doneModels.add(e.model);
    }
  }
  for (const m of models) {
    if (doneModels.has(m.name.toLowerCase())) {
      m.done = true;
    }
  }

  return models;
}

export default function PipelineRun() {
  const {
    events, isRunning, error, currentStage, training,
    dagApprovalPending, startStage, cancel, handleApproveDag, handleRejectDag,
  } = usePipeline();
  const [logOpen, setLogOpen] = useState(true);
  const logRef = useRef<HTMLDivElement>(null);

  const handleRun = (stage: number) => {
    setLogOpen(true);
    startStage(stage);
  };

  // Auto-scroll log to bottom
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [events]);

  // Derived state
  const metrics = events.filter((e): e is PipelineEvent & { type: "metric" } => e.type === "metric");
  const lastMetric = metrics[metrics.length - 1];
  const logs = events.filter((e) => e.type === "log");
  const complete = events.find((e) => e.type === "complete");

  // Progress: prefer model checkpoint progress for Stage 2, else phase heuristic
  const latestPhaseEvent = [...events].reverse().find((e) => e.phase);
  const currentPhase = latestPhaseEvent?.phase ?? null;
  const latestStage = currentStage ?? latestPhaseEvent?.stage ?? lastMetric?.stage;

  const checkpointPct = latestStage === 2 ? modelCheckpointProgress(events) : null;
  const explicitPct = checkpointPct ?? [...events].reverse().find((e) => e.progress_pct !== undefined)?.progress_pct;
  const progressPct = explicitPct ?? phaseProgress(latestStage, currentPhase ?? undefined);

  // Model milestones for Stage 2
  const modelMilestones = latestStage === 2 ? getModelMilestones(events) : [];

  // Current model info
  const latestModelEvent = [...events].reverse().find((e) => e.model);
  const currentModel = latestModelEvent?.model?.toUpperCase() ?? null;

  // Latest epoch info for progress display
  const latestEpoch = training.epochHistory.length > 0
    ? training.epochHistory[training.epochHistory.length - 1]
    : null;

  // Build a descriptive phase label
  let progressLabel = complete ? "Complete" : (currentPhase ?? "Running…");
  if (!complete && isRunning) {
    if (latestEpoch && training.epochHistory.length > 0) {
      progressLabel = `${currentPhase ?? "Neural meta-learner"} — Epoch ${latestEpoch.epoch}/${latestEpoch.n_epochs} · loss ${latestEpoch.total_loss.toFixed(4)}`;
    } else if (currentModel) {
      progressLabel += ` — ${currentModel}`;
    }
  }

  return (
    <div>
      <h1 className="mb-6 text-2xl font-bold">Run Pipeline</h1>

      {/* Stage buttons */}
      <div className="mb-6 flex flex-wrap gap-2">
        {STAGES.map((s) => (
          <button
            key={s.value}
            onClick={() => handleRun(s.value)}
            disabled={isRunning}
            className="rounded border border-sparc-gray-300 px-4 py-2 text-sm font-medium transition-colors hover:bg-black hover:text-white disabled:opacity-40"
          >
            {s.label}
          </button>
        ))}
        <button
          onClick={() => handleRun(-1)}
          disabled={isRunning}
          className="rounded bg-black px-4 py-2 text-sm font-medium text-white hover:bg-sparc-gray-800 disabled:opacity-40"
        >
          Run All
        </button>
        {isRunning && (
          <button
            onClick={cancel}
            className="rounded border border-sparc-crimson px-4 py-2 text-sm font-medium text-sparc-crimson hover:bg-red-50"
          >
            Cancel
          </button>
        )}
      </div>

      {/* Progress bar */}
      {(isRunning || complete) && (
        <div className="mb-4">
          <div className="mb-1 flex items-center justify-between text-xs">
            <span className="font-medium text-sparc-gray-700">
              {progressLabel}
            </span>
            <span className="tabular-nums text-sparc-gray-500">
              {complete ? "100" : progressPct}%
            </span>
          </div>
          <div className="relative h-2 w-full overflow-hidden rounded-full bg-sparc-gray-200">
            <div
              className={`h-full rounded-full transition-all duration-500 ${complete ? "bg-green-500" : "bg-sparc-purple"}`}
              style={{ width: `${complete ? 100 : progressPct}%` }}
            />
            {/* Model milestone markers for Stage 2 */}
            {modelMilestones.map((m) => (
              <div
                key={m.name}
                className="absolute top-0 h-full w-px bg-sparc-gray-400"
                style={{ left: `${m.pct}%` }}
                title={m.name}
              />
            ))}
          </div>
          {/* Model milestone labels for Stage 2 */}
          {modelMilestones.length > 0 && (
            <div className="relative mt-1 h-4 text-[9px] text-sparc-gray-500">
              {modelMilestones.map((m) => (
                <span
                  key={m.name}
                  className={`absolute -translate-x-1/2 ${m.done ? "font-bold text-green-600" : ""}`}
                  style={{ left: `${m.pct}%` }}
                >
                  {m.done ? "✓ " : ""}{m.name}
                </span>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Live metric dashboard */}
      {lastMetric && (
        <div className="mb-4 rounded border border-sparc-gray-200 bg-sparc-gray-100 p-4">
          <div className="grid grid-cols-4 gap-4 text-center">
            <div>
              <div className="text-xs text-sparc-gray-600">Stage</div>
              <div className="text-xl font-bold">{lastMetric.stage ?? "—"}</div>
            </div>
            <div>
              <div className="text-xs text-sparc-gray-600">Fold</div>
              <div className="text-xl font-bold">{lastMetric.fold ?? "—"}</div>
            </div>
            <div>
              <div className="text-xs text-sparc-gray-600">{lastMetric.metric?.toUpperCase()}</div>
              <div className="text-xl font-bold">{lastMetric.value?.toFixed(4)}</div>
            </div>
            <div>
              <div className="text-xs text-sparc-gray-600">Progress</div>
              <div className="text-xl font-bold">{explicitPct ?? progressPct}%</div>
            </div>
          </div>
        </div>
      )}

      {/* Training telemetry — visible during Stage 2 */}
      {(latestStage === 2 || training.epochHistory.length > 0 || training.capacityResults.length > 0) && (
        <div className="mb-4 space-y-4">
          <div className="flex items-center gap-2">
            <h2 className="text-sm font-semibold text-sparc-gray-700">Neural Training</h2>
            <ConvergenceBadge status={training.convergenceStatus ?? (isRunning && training.epochHistory.length > 0 ? "training" : null)} />
          </div>
          <CapacitySweepView results={training.capacityResults} />
          <EpochLossChart
            epochHistory={training.epochHistory}
            curriculumStage={training.curriculumStage}
            curriculumLabel={training.curriculumLabel}
          />
        </div>
      )}

      {/* DAG approval gate banner */}
      {dagApprovalPending && (
        <div className="mb-4 rounded border border-amber-300 bg-amber-50 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-semibold text-amber-800">
                DAG Review Required — Pipeline Paused
              </p>
              <p className="text-xs text-amber-700 mt-1">
                MC³ structure learning is complete. Review the discovered DAG in the DAG Editor tab,
                then approve to continue to NUTS posterior sampling.
              </p>
            </div>
            <div className="flex gap-2">
              <button
                onClick={handleRejectDag}
                className="rounded border border-red-300 px-3 py-1.5 text-xs font-medium text-red-600 hover:bg-red-50"
              >
                Reject
              </button>
              <button
                onClick={handleApproveDag}
                className="rounded bg-green-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-green-700"
              >
                Approve DAG
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Status banners */}
      {complete && (
        <div className="mb-4 rounded border border-green-300 bg-green-50 p-3 text-sm text-green-800">
          Pipeline complete.
        </div>
      )}
      {error && (
        <div className="mb-4 rounded border border-sparc-crimson bg-red-50 p-3 text-sm text-sparc-crimson">
          {error}
        </div>
      )}

      {/* Collapsible log output */}
      <div className="rounded border border-sparc-gray-200">
        <button
          onClick={() => setLogOpen(!logOpen)}
          className="flex w-full items-center justify-between bg-sparc-gray-100 px-3 py-2 text-left text-xs font-medium text-sparc-gray-700 hover:bg-sparc-gray-200"
        >
          <span>Terminal Output ({logs.length} lines)</span>
          <span className="text-sparc-gray-500">{logOpen ? "▲ Hide" : "▼ Show"}</span>
        </button>
        {logOpen && (
          <div
            ref={logRef}
            className="h-80 overflow-auto bg-sparc-gray-100 p-3 font-mono text-xs"
          >
            {logs.length === 0 && !isRunning && (
              <span className="text-sparc-gray-600">No output yet. Start a stage to begin.</span>
            )}
            {logs.map((e, i) => (
              <div key={i} className="py-0.5">
                {e.message}
              </div>
            ))}
            {isRunning && <div className="animate-pulse text-sparc-pink">Running...</div>}
          </div>
        )}
      </div>
    </div>
  );
}
