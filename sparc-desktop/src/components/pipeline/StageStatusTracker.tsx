import type { StageStatus } from "@/hooks/PipelineProvider";

const STAGE_LABELS: Record<number, string> = {
  0: "Correlogram",
  1: "GWEN",
  2: "Spatial CV",
  3: "Causal",
  4: "Scenarios",
};

const STATUS_STYLES: Record<string, { dot: string; line: string; text: string }> = {
  pending:  { dot: "bg-sparc-gray-300",  line: "bg-sparc-gray-200", text: "text-sparc-gray-400" },
  running:  { dot: "bg-sparc-purple animate-pulse", line: "bg-sparc-purple/30", text: "text-sparc-purple font-semibold" },
  complete: { dot: "bg-green-500",       line: "bg-green-300",      text: "text-green-700" },
  failed:   { dot: "bg-red-500",         line: "bg-red-300",        text: "text-red-600" },
};

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return s > 0 ? `${m}m ${s}s` : `${m}m`;
}

interface Props {
  stageStatuses: Record<number, StageStatus>;
  currentStage: number | null;
  isRunning: boolean;
}

export default function StageStatusTracker({ stageStatuses, currentStage, isRunning }: Props) {
  const stages = [0, 1, 2, 3, 4];

  const getStatus = (stage: number): StageStatus => {
    if (stageStatuses[stage]) return stageStatuses[stage];
    return { stage, status: "pending" };
  };

  return (
    <div className="rounded border border-sparc-gray-200 bg-white p-4">
      <h3 className="mb-3 text-xs font-semibold uppercase tracking-wide text-sparc-gray-500">
        Pipeline Progress
      </h3>

      <div className="space-y-0">
        {stages.map((stage, idx) => {
          const ss = getStatus(stage);
          const style = STATUS_STYLES[ss.status] ?? STATUS_STYLES.pending;
          const isLast = idx === stages.length - 1;

          return (
            <div key={stage} className="flex gap-3">
              {/* Timeline column */}
              <div className="flex flex-col items-center">
                <div className={`h-3 w-3 rounded-full ${style.dot} shrink-0 mt-0.5`} />
                {!isLast && (
                  <div className={`w-0.5 flex-1 ${style.line} min-h-[24px]`} />
                )}
              </div>

              {/* Content column */}
              <div className={`flex-1 pb-3 ${isLast ? "" : ""}`}>
                <div className="flex items-center justify-between">
                  <span className={`text-sm ${style.text}`}>
                    Stage {stage} — {STAGE_LABELS[stage] ?? `Stage ${stage}`}
                  </span>

                  {/* Duration / ETA */}
                  <div className="flex items-center gap-2 text-xs text-sparc-gray-500">
                    {ss.status === "running" && ss.eta_seconds != null && (
                      <span className="text-sparc-purple">
                        ETA {formatDuration(ss.eta_seconds)}
                      </span>
                    )}
                    {ss.status === "running" && ss.elapsed_seconds != null && (
                      <span>{formatDuration(ss.elapsed_seconds)} elapsed</span>
                    )}
                    {ss.status === "complete" && ss.started_at && ss.completed_at && (
                      <span className="text-green-600">
                        {formatDuration(ss.completed_at - ss.started_at)}
                      </span>
                    )}
                    {ss.status === "failed" && (
                      <span className="text-red-500 font-medium">Failed</span>
                    )}
                  </div>
                </div>

                {/* Error detail */}
                {ss.status === "failed" && ss.error && (
                  <p className="mt-1 text-xs text-red-500 line-clamp-2">{ss.error}</p>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
