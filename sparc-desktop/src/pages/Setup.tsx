/**
 * SPARC First-Run Setup Wizard
 *
 * Step 1 - Welcome
 * Step 2 - Install Info
 * Step 3 - Downloading / Installing (APNG loops, progress bar)
 * Step 4 - Ready
 *
 * Decoration-less window; handles its own drag region.
 */
import { useState, useEffect, useCallback, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

// ── Brand tokens ─────────────────────────────────────────────────────────────
const C = {
  crimson: "#e73c25",
  white:   "#ffffff",
  text:    "#1a1a1a",
  muted:   "#666666",
  dim:     "rgba(0,0,0,0.07)",
};

const ACCENT = "linear-gradient(135deg, #e73c25 0%, #f0b632 100%)";

// ── Types ─────────────────────────────────────────────────────────────────────
type Step = 1 | 2 | 3 | 4;

interface DlState {
  status: "idle" | "running" | "error" | "done";
  progress: number;
  message: string;
  error: string | null;
}

// ── APNG Logo ─────────────────────────────────────────────────────────────────
// APNGs with loop_count == 0 loop automatically in browsers.
// We also force-restart the element every APNG_PLAY_MS so it keeps looping
// even if the APNG was encoded with loop_count == 1.
const APNG_PLAY_MS = 3_000;

function Logo({ running = false, size = "100%" }: { running?: boolean; size?: string | number }) {
  const [key, setKey] = useState(0);

  useEffect(() => {
    if (!running) return;
    const t = setInterval(() => setKey((k) => k + 1), APNG_PLAY_MS);
    return () => clearInterval(t);
  }, [running]);

  return (
    <img
      key={key}
      src="/splash-logo.png"
      alt="SPARC"
      style={{
        width: size,
        height: size,
        objectFit: "contain",
        display: "block",
      }}
    />
  );
}

// ── Progress bar ──────────────────────────────────────────────────────────────
function ProgressBar({ value }: { value: number }) {
  return (
    <div style={{
      width: "100%", height: 6, borderRadius: 3,
      background: "rgba(0,0,0,0.1)", overflow: "hidden",
    }}>
      <div style={{
        height: "100%",
        width: `${Math.min(100, Math.max(0, value))}%`,
        background: ACCENT,
        borderRadius: 3,
        transition: "width 300ms ease",
      }} />
    </div>
  );
}

// ── Button ────────────────────────────────────────────────────────────────────
function Btn({
  children, onClick, variant = "primary", disabled,
}: {
  children: React.ReactNode;
  onClick?: () => void;
  variant?: "primary" | "ghost";
  disabled?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        padding: "10px 28px",
        borderRadius: 8,
        border: variant === "ghost" ? "1.5px solid rgba(0,0,0,0.15)" : "none",
        cursor: disabled ? "not-allowed" : "pointer",
        fontFamily: "inherit",
        fontSize: 14,
        fontWeight: 600,
        transition: "opacity 150ms",
        opacity: disabled ? 0.5 : 1,
        background: variant === "primary" ? C.crimson : "transparent",
        color: variant === "primary" ? C.white : C.text,
      }}
    >
      {children}
    </button>
  );
}

// ── Main wizard ───────────────────────────────────────────────────────────────
export default function Setup() {
  const [step, setStep] = useState<Step>(1);
  const [dl, setDl] = useState<DlState>({
    status: "idle", progress: 0, message: "", error: null,
  });
  const unlistenRef = useRef<(() => void) | null>(null);

  const runInstall = useCallback(async () => {
    setDl({ status: "running", progress: 0, message: "Starting\u2026", error: null });

    const unlisten = await listen<string>("setup://progress", (event) => {
      const line = event.payload;
      setDl((prev) => {
        const pct = line.match(/(\d+)%/);
        const progress = pct ? parseInt(pct[1], 10) : prev.progress;
        return { ...prev, message: line.slice(0, 80), progress };
      });
    });
    unlistenRef.current = unlisten;

    try {
      setDl((p) => ({ ...p, message: "Creating Python environment\u2026", progress: 5 }));
      await invoke("setup_create_venv");

      setDl((p) => ({ ...p, message: "Downloading SPARC engine\u2026", progress: 20 }));
      await invoke("setup_install_engine");

      await invoke("setup_mark_complete");

      setDl({ status: "done", progress: 100, message: "Installation complete.", error: null });
      setStep(4);
    } catch (err) {
      setDl((p) => ({
        ...p,
        status: "error",
        error: String(err),
        message: "Installation failed.",
      }));
    } finally {
      unlisten();
      unlistenRef.current = null;
    }
  }, []);

  const handleRetry = useCallback(async () => {
    await invoke("setup_cleanup_env").catch(() => {});
    runInstall();
  }, [runInstall]);

  const handleFinish = useCallback(() => {
    invoke("setup_finish").catch(() => {});
  }, []);

  // Drag region — sits at very top so the window can be moved
  const dragStyle: React.CSSProperties = {
    WebkitAppRegion: "drag",
    position: "absolute",
    top: 0, left: 0, right: 0,
    height: 36,
    zIndex: 10,
  } as React.CSSProperties;

  // ── Shared layout: white bg, logo fills top portion, content panel below ──
  const shell = (
    logoNode: React.ReactNode,
    contentNode: React.ReactNode,
    logoFlex = 0.65,
  ): React.ReactElement => (
    <div style={{
      position: "relative",
      width: "100%", height: "100%",
      display: "flex", flexDirection: "column",
      background: C.white,
      overflow: "hidden",
    }}>
      <div style={dragStyle} />

      {/* Logo area */}
      <div style={{
        flex: logoFlex,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        overflow: "hidden",
        padding: "8px 8px 0",
      }}>
        {logoNode}
      </div>

      {/* Content panel */}
      <div style={{
        flex: 1 - logoFlex,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        padding: "0 40px 24px",
        gap: 12,
      }}>
        {contentNode}
      </div>
    </div>
  );

  // ── Step 1: Welcome ───────────────────────────────────────────────────────
  const renderStep1 = () => shell(
    <Logo size="100%" />,
    <>
      <div style={{ textAlign: "center" }}>
        <h1 style={{ fontSize: 26, fontWeight: 700, margin: 0, color: C.text }}>
          Welcome to SPARC
        </h1>
        <p style={{ fontSize: 14, color: C.muted, margin: "6px 0 0" }}>
          Spatial Analysis &amp; Research Core
        </p>
      </div>
      <Btn onClick={() => setStep(2)}>Get Started &rarr;</Btn>
    </>,
  );

  // ── Step 2: Install Info ──────────────────────────────────────────────────
  const renderStep2 = () => shell(
    <Logo size="80%" />,
    <>
      <div style={{
        width: "100%",
        background: C.dim,
        borderRadius: 10,
        padding: "14px 18px",
        display: "flex",
        flexDirection: "column",
        gap: 8,
      }}>
        <p style={{ color: C.text, fontSize: 13, margin: 0, lineHeight: 1.55 }}>
          SPARC will install its analysis engine (~400&nbsp;MB) into a local
          Python environment. This one-time setup takes 1&ndash;3&nbsp;minutes.
        </p>
        <div>
          <span style={{ color: C.muted, fontSize: 11 }}>Install location</span>
          <br />
          <code style={{
            background: "rgba(0,0,0,0.06)", color: "#b45309",
            padding: "2px 8px", borderRadius: 4, fontSize: 12,
          }}>
            ~/.sparc/env
          </code>
        </div>
      </div>
      <div style={{ display: "flex", gap: 12 }}>
        <Btn variant="ghost" onClick={() => setStep(1)}>&larr; Back</Btn>
        <Btn onClick={() => { setStep(3); runInstall(); }}>Install Now</Btn>
      </div>
    </>,
    0.5,
  );

  // ── Step 3: Downloading ───────────────────────────────────────────────────
  const renderStep3 = () => shell(
    <Logo running={dl.status === "running"} size="100%" />,
    <>
      <div style={{ width: "100%" }}>
        <ProgressBar value={dl.progress} />
      </div>
      <p style={{
        fontSize: 13,
        color: dl.status === "error" ? C.crimson : C.muted,
        margin: 0,
        textAlign: "center",
        minHeight: 18,
      }}>
        {dl.status === "error" ? `Error: ${dl.error}` : dl.message}
      </p>
      {dl.status === "error" && (
        <div style={{ display: "flex", gap: 10, marginTop: 4 }}>
          <Btn variant="ghost" onClick={() => invoke("setup_finish").catch(() => {})}>Quit</Btn>
          <Btn onClick={handleRetry}>Try Again</Btn>
        </div>
      )}
    </>,
  );

  // ── Step 4: Ready ─────────────────────────────────────────────────────────
  const renderStep4 = () => shell(
    <Logo size="80%" />,
    <>
      <div style={{ textAlign: "center" }}>
        <div style={{
          width: 52, height: 52, borderRadius: "50%",
          background: ACCENT,
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 26, color: C.white,
          margin: "0 auto 10px",
        }}>
          &#10003;
        </div>
        <h2 style={{ fontSize: 22, fontWeight: 700, margin: 0, color: C.text }}>
          SPARC is ready.
        </h2>
        <p style={{ fontSize: 13, color: C.muted, margin: "6px 0 0" }}>
          Your analysis engine is installed.
        </p>
      </div>
      <Btn onClick={handleFinish}>Launch SPARC &rarr;</Btn>
    </>,
    0.55,
  );

  const steps: Record<Step, () => React.ReactElement> = {
    1: renderStep1,
    2: renderStep2,
    3: renderStep3,
    4: renderStep4,
  };

  return (
    <div style={{
      width: "100vw", height: "100vh",
      fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
      overflow: "hidden",
      userSelect: "none",
      position: "relative",
    }}>
      {steps[step]()}

      {/* Step indicator dots */}
      {dl.status !== "error" && (
        <div style={{
          position: "absolute", bottom: 10,
          left: 0, right: 0,
          display: "flex", justifyContent: "center", gap: 5,
          pointerEvents: "none",
        }}>
          {([1, 2, 3, 4] as Step[]).map((s) => (
            <div key={s} style={{
              width: 5, height: 5, borderRadius: "50%",
              background: s === step ? C.crimson : "rgba(0,0,0,0.18)",
              transition: "background 300ms",
            }} />
          ))}
        </div>
      )}
    </div>
  );
}
