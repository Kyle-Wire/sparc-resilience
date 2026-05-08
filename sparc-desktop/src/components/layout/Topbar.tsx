import type { HealthResponse } from "@/lib/types";
import { useArtifactStream } from "@/hooks/useArtifactStream";
import { useAuthStore } from "@/stores/authStore";

interface TopbarProps {
  page: string;
  status?: HealthResponse | null;
  onNavigateSettings?: () => void;
}

export default function Topbar({ page, status, onNavigateSettings }: TopbarProps) {
  const sidecarReady = !!status;
  const port = "8008";
  const hasApiKey = !!localStorage.getItem("anthropic-api-key");
  const { status: liveStatus, lastEvent } = useArtifactStream();
  const user = useAuthStore((s) => s.user);

  const liveColor =
    liveStatus === "open"
      ? "var(--crimson)"
      : liveStatus === "connecting"
      ? "var(--amber)"
      : "var(--muted)";
  const liveValue =
    liveStatus === "open"
      ? lastEvent
        ? `live · ${lastEvent.artifact_id}`
        : "live · idle"
      : liveStatus;

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

      <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
        <StatusPill
          color={sidecarReady ? "var(--crimson)" : "var(--muted)"}
          label="Sidecar"
          value={sidecarReady ? `ready · :${port}` : "offline"}
        />
        <StatusPill color={liveColor} label="Stream" value={liveValue} />
        <StatusPill color="var(--amber)" label="GPU" value="CUDA · 11.8" />
        <StatusPill
          color={hasApiKey ? "var(--color-sparc-purple, #602468)" : "var(--muted)"}
          label="Claude"
          value={hasApiKey ? "sonnet-4.6" : "no key"}
        />

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

function StatusPill({ color, label, value }: { color: string; label: string; value: string }) {
  return (
    <div
      className="mono"
      style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 10, color: "var(--muted)" }}
    >
      <span style={{ width: 6, height: 6, borderRadius: "50%", background: color }} />
      <span style={{ letterSpacing: "0.1em", textTransform: "uppercase" }}>{label}</span>
      <span style={{ color: "var(--ink-2)" }}>{value}</span>
    </div>
  );
}
