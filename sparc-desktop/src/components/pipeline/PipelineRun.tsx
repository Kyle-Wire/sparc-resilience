import { useState, useEffect, useRef } from "react";
import { usePipelineStream } from "@/hooks/usePipelineStream";
import type { PipelineEvent } from "@/lib/types";

const STAGES = [
  { value: 0, label: "0 — Correlogram" },
  { value: 1, label: "1 — GWEN" },
  { value: 2, label: "2 — Spatial CV" },
  { value: 3, label: "3 — Causal" },
  { value: 4, label: "4 — Scenarios" },
];

/** Heuristic phase ordering used to derive a visual progress %. */
const STAGE_PHASES: Record<number, string[]> = {
  0: ["Correlogram analysis", "Analyzing variable", "Pipeline configuration"],
  1: ["GWEN variable selection", "GWEN results"],
  2: ["Loading data", "Loading spatial folds", "Training model", "Model complete", "OOF predictions", "Spatial autocorrelation", "Deep Kriging CV", "Stage 2 complete"],
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

export default function PipelineRun() {
  const { events, isRunning, error, startStage, cancel } = usePipelineStream();
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

  // Progress: use explicit progress_pct if available, else derive from phase
  const latestPhaseEvent = [...events].reverse().find((e) => e.phase);
  const currentPhase = latestPhaseEvent?.phase ?? null;
  const currentStage = latestPhaseEvent?.stage ?? lastMetric?.stage;
  const explicitPct = [...events].reverse().find((e) => e.progress_pct !== undefined)?.progress_pct;
  const progressPct = explicitPct ?? phaseProgress(currentStage, currentPhase ?? undefined);

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
              {complete ? "Complete" : currentPhase ?? "Running…"}
            </span>
            <span className="tabular-nums text-sparc-gray-500">
              {complete ? "100" : progressPct}%
            </span>
          </div>
          <div className="h-2 w-full overflow-hidden rounded-full bg-sparc-gray-200">
            <div
              className={`h-full rounded-full transition-all duration-500 ${complete ? "bg-green-500" : "bg-sparc-purple"}`}
              style={{ width: `${complete ? 100 : progressPct}%` }}
            />
          </div>
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

      {/* Status banners */}
      {complete && (
        <div className="mb-4 rounded border border-green-300 bg-green-50 p-3 text-sm text-green-800">
          Stage {complete.stage} complete.
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
