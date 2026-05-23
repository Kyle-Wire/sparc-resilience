/**
 * PipelineNavigator — sticky timeline strip across the top of every stage page.
 *
 * Stateless and presentational. The parent (StageResultsPage) owns which stage
 * is active and passes the full stages array. Locked stages are aria-disabled
 * and their onClick is a no-op so that fireEvent.click in tests also won't fire.
 */

export interface StageNode {
  /** 0–4 */
  id: number;
  /** Display name, e.g. "Geometry" */
  label: string;
  /** Hex or CSS variable colour for the stage */
  color: string;
  status: "active" | "complete" | "locked";
}

interface PipelineNavigatorProps {
  stages: StageNode[];
  onStageClick: (id: number) => void;
}

const RAMP =
  "linear-gradient(90deg,#602468 0%,#9e337d 14%,#e94d9b 28%,#e94461 42%,#e73c25 56%,#e76c25 70%,#e79024 82%,#f0b632 92%,#fbdd46 100%)";

export default function PipelineNavigator({ stages, onStageClick }: PipelineNavigatorProps) {
  return (
    <nav
      aria-label="Pipeline stages"
      style={{
        position: "sticky",
        top: 0,
        zIndex: 40,
        background: "var(--paper, #f7f4ee)",
        borderBottom: "1px solid var(--line, #c9c2b3)",
      }}
    >
      {/* 2 px ramp strip */}
      <div style={{ height: 2, background: RAMP }} />

      {/* Stage nodes */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 4,
          padding: "6px 16px",
        }}
      >
        {stages.map((stage, i) => (
          <span key={stage.id} style={{ display: "contents" }}>
            {/* Connector line between nodes */}
            {i > 0 && (
              <div
                style={{
                  flex: 1,
                  height: 1,
                  background:
                    stage.status === "locked"
                      ? "var(--line, #c9c2b3)"
                      : `linear-gradient(90deg,${stages[i - 1].color},${stage.color})`,
                  opacity: stage.status === "locked" ? 0.4 : 1,
                }}
              />
            )}

            <button
              data-status={stage.status}
              aria-current={stage.status === "active" ? "step" : undefined}
              aria-disabled={stage.status === "locked"}
              onClick={() => {
                if (stage.status !== "locked") onStageClick(stage.id);
              }}
              style={{
                background: stage.status === "active" ? stage.color : "none",
                color: stage.status === "active" ? "#fff" : stage.color,
                border: `1px ${stage.status === "locked" ? "dashed" : "solid"} ${stage.color}`,
                borderRadius: 4,
                padding: "3px 10px",
                fontSize: 11,
                fontFamily: "var(--font-mono, 'JetBrains Mono', monospace)",
                cursor: stage.status === "locked" ? "not-allowed" : "pointer",
                opacity: stage.status === "locked" ? 0.45 : 1,
                whiteSpace: "nowrap",
                transition: "opacity 0.15s",
              }}
            >
              {stage.id} · {stage.label}
            </button>
          </span>
        ))}
      </div>
    </nav>
  );
}
