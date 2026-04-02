import { usePipelineStream } from "@/hooks/usePipelineStream";
import type { PipelineEvent } from "@/lib/types";

const STAGES = [
  { value: 0, label: "0 — Correlogram" },
  { value: 1, label: "1 — GWEN" },
  { value: 2, label: "2 — Spatial CV" },
  { value: 3, label: "3 — Causal" },
  { value: 4, label: "4 — Scenarios" },
];

export default function PipelineRun() {
  const { events, isRunning, error, startStage, cancel } = usePipelineStream();

  const handleRun = (stage: number) => {
    startStage(stage);
  };

  // Latest metrics for a live dashboard
  const metrics = events.filter((e): e is PipelineEvent & { type: "metric" } => e.type === "metric");
  const lastMetric = metrics[metrics.length - 1];
  const logs = events.filter((e) => e.type === "log");
  const complete = events.find((e) => e.type === "complete");

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
          onClick={() => handleRun(-1)} // -1 = all
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
              <div className="text-xl font-bold">
                {events.find((e) => e.progress_pct !== undefined)?.progress_pct ?? "—"}%
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Status */}
      {complete && <div className="mb-4 rounded border border-green-300 bg-green-50 p-3 text-sm text-green-800">Stage {complete.stage} complete.</div>}
      {error && <div className="mb-4 rounded border border-sparc-crimson bg-red-50 p-3 text-sm text-sparc-crimson">{error}</div>}

      {/* Log output */}
      <div className="h-80 overflow-auto rounded border border-sparc-gray-200 bg-sparc-gray-100 p-3 font-mono text-xs">
        {logs.length === 0 && !isRunning && <span className="text-sparc-gray-600">No output yet. Start a stage to begin.</span>}
        {logs.map((e, i) => (
          <div key={i} className="py-0.5">
            {e.message}
          </div>
        ))}
        {isRunning && <div className="animate-pulse text-sparc-pink">Running...</div>}
      </div>
    </div>
  );
}
