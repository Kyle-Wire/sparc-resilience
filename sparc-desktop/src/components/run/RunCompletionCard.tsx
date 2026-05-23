/**
 * RunCompletionCard — shown inline in RunPage when the pipeline finishes.
 * Purely presentational: accepts computed values and a callback; owns no hooks.
 */

interface RunCompletionCardProps {
  elapsedSeconds: number;
  stagesComplete: number;
  totalStages: number;
  errorCount?: number;
  onViewResults: () => void;
}

function formatTime(s: number): string {
  const m = Math.floor(s / 60);
  const sec = s % 60;
  return `${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
}

export default function RunCompletionCard({
  elapsedSeconds,
  stagesComplete,
  totalStages,
  errorCount = 0,
  onViewResults,
}: RunCompletionCardProps) {
  return (
    <div
      style={{
        border: "1px solid #a5d6a7",
        background: "#f1f8e9",
        borderRadius: 10,
        padding: "20px 24px",
        marginBottom: 16,
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: 20,
        flexWrap: "wrap",
      }}
    >
      <div>
        <h2
          style={{
            fontSize: 16,
            fontWeight: 800,
            color: "#1b5e20",
            margin: 0,
            marginBottom: 8,
          }}
        >
          Run complete
        </h2>
        <div style={{ display: "flex", gap: 20, alignItems: "center", flexWrap: "wrap" }}>
          <span style={{ fontSize: 12, color: "#2e7d32" }}>
            ⏱ {formatTime(elapsedSeconds)}
          </span>
          <span style={{ fontSize: 12, color: "#2e7d32" }}>
            {stagesComplete} / {totalStages} stages
          </span>
          {errorCount > 0 && (
            <span style={{ fontSize: 12, color: "#c62828" }}>
              ⚠ {errorCount} stage{errorCount > 1 ? "s" : ""} failed
            </span>
          )}
        </div>
      </div>

      <button
        onClick={onViewResults}
        style={{
          background: "#1b5e20",
          color: "white",
          border: "none",
          borderRadius: 6,
          padding: "9px 20px",
          fontSize: 13,
          fontWeight: 700,
          cursor: "pointer",
          fontFamily: "inherit",
          whiteSpace: "nowrap",
          flexShrink: 0,
        }}
      >
        View Results →
      </button>
    </div>
  );
}
