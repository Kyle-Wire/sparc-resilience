import { usePipeline } from "@/hooks/PipelineProvider";
import { useAuthStore } from "@/stores/authStore";
import type { HealthResponse } from "@/lib/types";

const STAGE_NAMES: Record<number, string> = {
  0: "Correlogram", 1: "GWEN", 2: "Spatial CV", 3: "Causal", 4: "Scenarios",
};

interface TopbarProps {
  page: string;
  onNavigateSettings?: () => void;
  serverStatus?: HealthResponse | null;
}

export default function Topbar({ page, onNavigateSettings, serverStatus }: TopbarProps) {
  const user = useAuthStore((s) => s.user);
  const pipeline = usePipeline();

  const healthy = serverStatus != null;
  const doneCount = Object.values(pipeline.stageStatuses).filter(
    (s) => s.status === "complete",
  ).length;

  // Build status pill content
  let pillText: string;
  let pillBg: string;
  let pillColor: string;
  let pillBorder: string;

  if (pipeline.isRunning) {
    const stageName =
      pipeline.currentStage != null ? STAGE_NAMES[pipeline.currentStage] ?? `Stage ${pipeline.currentStage}` : "…";
    pillText = `Stage ${doneCount + 1}/5 · ${stageName}`;
    pillBg = "#fff8e1";
    pillColor = "#7c4a00";
    pillBorder = "1px solid var(--amber, #e79024)";
  } else if (!healthy) {
    pillText = "sidecar offline";
    pillBg = "#fff0f0";
    pillColor = "#c62828";
    pillBorder = "1px solid #ef9a9a";
  } else {
    pillText = "healthy";
    pillBg = "#f1f8e9";
    pillColor = "#2e7d32";
    pillBorder = "1px solid #a5d6a7";
  }

  const dotColor = pipeline.isRunning ? "var(--amber, #e79024)" : healthy ? "#43a047" : "#c62828";

  return (
    <div
      style={{
        height: 42,
        flexShrink: 0,
        display: "flex",
        alignItems: "center",
        borderBottom: "1px solid var(--line)",
        background: "var(--paper)",
        paddingRight: 10,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "0 16px", flex: 1 }}>
        <span
          className="mono"
          style={{ fontSize: 10, color: "var(--muted)", letterSpacing: "0.12em", textTransform: "uppercase" }}
        >
          sparc · pipeline
        </span>
        <span style={{ fontSize: 11, color: "var(--line)" }}>/</span>
        <span style={{ fontSize: 12, fontWeight: 600 }}>{page}</span>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 10, paddingRight: 4 }}>
        {/* Server / pipeline status pill */}
        <div
          aria-label={pillText}
          style={{
            display: "flex",
            alignItems: "center",
            gap: 4,
            border: pillBorder,
            background: pillBg,
            borderRadius: 999,
            padding: "3px 8px",
          }}
        >
          <div
            style={{
              width: 6,
              height: 6,
              borderRadius: "50%",
              background: dotColor,
              flexShrink: 0,
            }}
          />
          <span style={{ fontSize: 10, fontWeight: 600, color: pillColor, whiteSpace: "nowrap" }}>
            {pillText}
          </span>
        </div>

        {/* User avatar */}
        {user && (
          <button
            onClick={onNavigateSettings}
            title={user.email ?? "Account settings"}
            style={{
              width: 26,
              height: 26,
              borderRadius: "50%",
              border: "none",
              background: "var(--crimson)",
              color: "#fff",
              fontSize: 11,
              fontWeight: 700,
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              flexShrink: 0,
            }}
          >
            {user.email?.[0]?.toUpperCase() ?? "?"}
          </button>
        )}
      </div>
    </div>
  );
}

