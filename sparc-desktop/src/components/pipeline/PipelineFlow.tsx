/**
 * PipelineFlow — Horizontal node-edge pipeline diagram with progress rings.
 * Matches the SPARC Labs prototype design exactly.
 */

interface StageInfo {
  label: string;
  status: "done" | "running" | "queued";
  /** 0–1 progress fraction for running stage */
  progress: number;
  /** Duration string e.g. "12.4s" */
  duration?: string;
}

interface PipelineFlowProps {
  stages: StageInfo[];
  currentStageIndex: number;
}

function ProgressRing({ progress, status }: { progress: number; status: string }) {
  const angle = Math.round(progress * 360);
  if (status !== "running") return null;
  return (
    <div
      style={{
        position: "absolute",
        inset: -4,
        borderRadius: 12,
        background: `conic-gradient(var(--color-sparc-crimson) ${angle}deg, transparent 0)`,
        WebkitMask: "radial-gradient(circle, transparent 62%, black 63%)",
        mask: "radial-gradient(circle, transparent 62%, black 63%)",
      }}
    />
  );
}

export default function PipelineFlow({ stages }: PipelineFlowProps) {
  return (
    <div style={{ display: "flex", alignItems: "flex-start", gap: 0, padding: "8px 0 16px" }}>
      {stages.map((stage, i) => {
        const isLast = i === stages.length - 1;
        const nodeColor =
          stage.status === "done"
            ? "var(--color-sparc-ink)"
            : stage.status === "running"
              ? "var(--color-sparc-crimson)"
              : "#e8e2d8";
        const textColor =
          stage.status === "done" || stage.status === "running" ? "#fff" : "var(--color-sparc-muted)";

        // Edge between this node and the next
        const edgeFill =
          stage.status === "done"
            ? "var(--color-sparc-ink)"
            : stage.status === "running"
              ? "var(--color-sparc-crimson)"
              : "var(--color-sparc-line)";

        // Next stage status for edge styling
        const nextRunning = i < stages.length - 1 && stages[i + 1].status === "running";

        return (
          <div key={i} style={{ display: "flex", alignItems: "center", flex: isLast ? "0 0 auto" : 1 }}>
            {/* Node */}
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 6, minWidth: 56 }}>
              <div
                style={{
                  position: "relative",
                  width: 44,
                  height: 44,
                  borderRadius: 8,
                  background: nodeColor,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  boxShadow:
                    stage.status === "running"
                      ? "0 0 0 6px rgba(231,60,37,0.14)"
                      : "none",
                  transition: "background 0.3s, box-shadow 0.3s",
                }}
              >
                <ProgressRing progress={stage.progress} status={stage.status} />
                <span
                  className="mono"
                  style={{
                    position: "relative",
                    zIndex: 1,
                    fontSize: 14,
                    fontWeight: 700,
                    color: textColor,
                  }}
                >
                  {stage.status === "done" ? "✓" : i}
                </span>
              </div>
              {/* Label */}
              <span
                style={{
                  fontSize: 10.5,
                  fontWeight: 600,
                  color: "var(--color-sparc-ink)",
                  textAlign: "center",
                  lineHeight: 1.2,
                  maxWidth: 80,
                }}
              >
                {stage.label}
              </span>
              {/* Duration / progress */}
              <span
                className="mono"
                style={{
                  fontSize: 9.5,
                  color: "var(--color-sparc-muted)",
                  marginTop: -3,
                }}
              >
                {stage.status === "running"
                  ? `${Math.round(stage.progress * 100)}%`
                  : stage.duration ?? "—"}
              </span>
            </div>

            {/* Edge connector */}
            {!isLast && (
              <div
                style={{
                  flex: 1,
                  height: 2,
                  minWidth: 16,
                  position: "relative",
                  background: "var(--color-sparc-line)",
                  marginTop: -24,
                  marginLeft: 4,
                  marginRight: 4,
                }}
              >
                {/* Filled portion */}
                <div
                  style={{
                    position: "absolute",
                    top: 0,
                    left: 0,
                    height: "100%",
                    width:
                      stage.status === "done"
                        ? "100%"
                        : stage.status === "running"
                          ? `${Math.round(stage.progress * 100)}%`
                          : "0%",
                    background: edgeFill,
                    transition: "width 0.4s ease-out",
                  }}
                />
                {/* Animated dot on running edge */}
                {(stage.status === "running" || nextRunning) && (
                  <div
                    style={{
                      position: "absolute",
                      top: -2,
                      left:
                        stage.status === "running"
                          ? `calc(${Math.round(stage.progress * 100)}% - 3px)`
                          : "0%",
                      width: 6,
                      height: 6,
                      borderRadius: 3,
                      background: "var(--color-sparc-crimson)",
                      boxShadow: "0 0 0 3px rgba(231,60,37,0.25)",
                      transition: "left 0.4s ease-out",
                    }}
                  />
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
