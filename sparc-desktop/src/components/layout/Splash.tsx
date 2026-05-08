export type SplashStep = "sidecar" | "session" | "ready";

interface SplashProps {
  step: SplashStep;
  parallaxEnabled?: boolean;
}

const STEP_LABELS: Record<SplashStep, string> = {
  sidecar: "Starting sidecar…",
  session: "Restoring session…",
  ready: "Ready",
};

const STEP_PROGRESS: Record<SplashStep, number> = {
  sidecar: 33,
  session: 66,
  ready: 100,
};

export default function Splash({ step }: SplashProps) {
  const progress = STEP_PROGRESS[step];

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
          gap: 6,
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
              background: "var(--crimson)",
              borderRadius: 2,
              transition: "width 0.5s cubic-bezier(0.4, 0, 0.2, 1)",
            }}
          />
        </div>
        <div
          className="mono"
          style={{ fontSize: 9.5, color: "var(--muted)", letterSpacing: "0.08em" }}
        >
          {STEP_LABELS[step]}
        </div>
      </div>
    </div>
  );
}
