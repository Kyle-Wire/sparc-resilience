export type SplashStep = "sidecar" | "token" | "session" | "ready" | "failed";

interface SplashProps {
  step: SplashStep;
  parallaxEnabled?: boolean;
  onRetry?: () => void;
}

const STEP_LABELS: Record<SplashStep, string> = {
  sidecar: "Starting sidecar…",
  token: "Securing session…",
  session: "Restoring session…",
  ready: "Ready",
  failed: "Failed to start sidecar",
};

const STEP_PROGRESS: Record<SplashStep, number> = {
  sidecar: 25,
  token: 50,
  session: 75,
  ready: 100,
  failed: 25,
};

export default function Splash({ step, onRetry }: SplashProps) {
  const progress = STEP_PROGRESS[step];
  const failed = step === "failed";

  return (
    <div
      style={{
        width: "100vw",
        height: "100vh",
        position: "relative",
        overflow: "hidden",
        background: "#fff",
      }}
    >
      {/* Full-bleed animated splash image */}
      <img
        src="/splash-logo.png"
        alt=""
        style={{
          position: "absolute",
          inset: 0,
          width: "100%",
          height: "100%",
          objectFit: "cover",
          opacity: failed ? 0.4 : 1,
          transition: "opacity 0.4s",
        }}
      />

      {/* Progress bar overlaid at bottom */}
      <div
        style={{
          position: "absolute",
          bottom: 32,
          left: "50%",
          transform: "translateX(-50%)",
          width: 240,
          display: "flex",
          flexDirection: "column",
          gap: 8,
          alignItems: "center",
        }}
      >
        <div
          style={{
            width: "100%",
            height: 3,
            background: "rgba(0,0,0,0.12)",
            borderRadius: 2,
            overflow: "hidden",
          }}
        >
          <div
            style={{
              height: "100%",
              width: `${progress}%`,
              background: failed ? "var(--danger, #c0392b)" : "var(--crimson)",
              borderRadius: 2,
              transition: "width 0.5s cubic-bezier(0.4, 0, 0.2, 1)",
            }}
          />
        </div>
        <div
          className="mono"
          style={{ fontSize: 9.5, color: failed ? "var(--danger, #c0392b)" : "var(--muted)", letterSpacing: "0.08em" }}
        >
          {STEP_LABELS[step]}
        </div>
        {failed && onRetry && (
          <button
            onClick={onRetry}
            style={{
              marginTop: 6,
              padding: "6px 18px",
              borderRadius: 6,
              border: "none",
              background: "var(--crimson, #8b1a2c)",
              color: "#fff",
              fontSize: 12,
              fontWeight: 600,
              cursor: "pointer",
              letterSpacing: "0.04em",
            }}
          >
            Retry
          </button>
        )}
      </div>
    </div>
  );
}
