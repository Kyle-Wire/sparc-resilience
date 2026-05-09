/**
 * SPARC First-Run Setup Wizard
 *
 * Step 1 — Welcome
 * Step 2 — Install Info (shows ~/.sparc/env, read-only)
 * Step 3 — Downloading (APNG logo loops with 6s pause, progress bar, status text)
 * Step 4 — Ready
 *
 * This page renders in the dedicated "setup" Tauri window (decorations: false).
 * It handles its own window drag region and close logic.
 */
import { useState, useEffect, useRef, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

// ── Brand tokens (inline — no dependency on main app stores) ─────────────────
const C = {
  purple:  "#602468",
  magenta: "#9e337d",
  pink:    "#e94d9b",
  red:     "#e94461",
  crimson: "#e73c25",
  gold:    "#f0b632",
  yellow:  "#fbdd46",
  white:   "#ffffff",
  offWhite:"#f7f3ff",
};

const GRADIENT = `linear-gradient(135deg, ${C.purple} 0%, ${C.magenta} 40%, ${C.red} 75%, ${C.gold} 100%)`;
const DARK_BG  = C.purple;

// ── Types ─────────────────────────────────────────────────────────────────────
type Step = 1 | 2 | 3 | 4;

interface DownloadState {
  status: "idle" | "running" | "error" | "done";
  progress: number;   // 0–100
  message: string;
  error: string | null;
}

// ── APNG logo with 6-second pause between plays ───────────────────────────────
// We don't know the APNG duration precisely, so we allow it to finish
// naturally (key never changes mid-play) and restart via a timer.
// APNG_CYCLE_MS = estimated play duration + 6 000 ms pause.
// Adjust APNG_PLAY_MS if you know the exact animation length.
const APNG_PLAY_MS  = 3_000;  // estimated single-play duration
const APNG_PAUSE_MS = 6_000;  // pause between plays
const APNG_CYCLE_MS = APNG_PLAY_MS + APNG_PAUSE_MS;

function AnimatedLogo({ running }: { running: boolean }) {
  const [key, setKey] = useState(0);

  useEffect(() => {
    if (!running) return;
    const t = setInterval(() => setKey((k) => k + 1), APNG_CYCLE_MS);
    return () => clearInterval(t);
  }, [running]);

  return (
    <img
      key={key}
      src="/splash-logo.png"
      alt="SPARC"
      style={{ width: 140, height: 140, objectFit: "contain" }}
    />
  );
}

// ── Progress bar ──────────────────────────────────────────────────────────────
function ProgressBar({ value }: { value: number }) {
  return (
    <div style={{
      width: "100%",
      height: 6,
      borderRadius: 3,
      background: "rgba(255,255,255,0.15)",
      overflow: "hidden",
    }}>
      <div style={{
        height: "100%",
        width: `${Math.min(100, Math.max(0, value))}%`,
        background: GRADIENT,
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
        border: "none",
        cursor: disabled ? "not-allowed" : "pointer",
        fontFamily: "inherit",
        fontSize: 14,
        fontWeight: 600,
        transition: "opacity 150ms",
        opacity: disabled ? 0.5 : 1,
        background: variant === "primary" ? C.crimson : "rgba(255,255,255,0.15)",
        color: C.white,
      }}
    >
      {children}
    </button>
  );
}

// ── Main wizard ───────────────────────────────────────────────────────────────
export default function Setup() {
  const [step, setStep] = useState<Step>(1);
  const [dl, setDl] = useState<DownloadState>({
    status: "idle", progress: 0, message: "", error: null,
  });
  const unlistenRef = useRef<(() => void) | null>(null);

  // ── Step 3: drive the install ─────────────────────────────────────────────
  const runInstall = useCallback(async () => {
    setDl({ status: "running", progress: 0, message: "Starting…", error: null });

    // Listen for progress lines emitted from Rust
    const unlisten = await listen<string>("setup://progress", (event) => {
      const line = event.payload;
      setDl((prev) => {
        // Parse uv percentage out of lines like "  [=====>  ] 47%"
        const pct = line.match(/(\d+)%/);
        const progress = pct ? parseInt(pct[1], 10) : prev.progress;
        return { ...prev, message: line.slice(0, 80), progress };
      });
    });
    unlistenRef.current = unlisten;

    try {
      setDl((p) => ({ ...p, message: "Creating Python environment…", progress: 5 }));
      await invoke("setup_create_venv");

      setDl((p) => ({ ...p, message: "Downloading SPARC engine…", progress: 20 }));
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

  // Drag region style for decoration-less window
  const dragStyle: React.CSSProperties = {
    WebkitAppRegion: "drag",
    position: "absolute",
    top: 0, left: 0, right: 0,
    height: 36,
  } as React.CSSProperties;

  // ── Step renderers ────────────────────────────────────────────────────────

  const renderStep1 = () => (
    <div style={{
      display: "flex", flexDirection: "column", alignItems: "center",
      justifyContent: "center", height: "100%", gap: 24,
      background: GRADIENT, padding: 40,
    }}>
      <div style={dragStyle} />
      <img src="/splash-logo.png" alt="SPARC" style={{ width: 100, height: 100, objectFit: "contain" }} />
      <div style={{ textAlign: "center" }}>
        <h1 style={{ color: C.white, fontSize: 32, fontWeight: 700, margin: 0 }}>
          Welcome to SPARC
        </h1>
        <p style={{ color: "rgba(255,255,255,0.8)", fontSize: 15, marginTop: 8 }}>
          Spatial Analysis &amp; Research Core
        </p>
      </div>
      <Btn onClick={() => setStep(2)}>Get Started →</Btn>
    </div>
  );

  const renderStep2 = () => (
    <div style={{
      display: "flex", flexDirection: "column", height: "100%",
      background: DARK_BG, padding: 40, gap: 20,
    }}>
      <div style={dragStyle} />
      <h2 style={{ color: C.white, fontSize: 20, fontWeight: 700, margin: 0 }}>
        SPARC Engine Installation
      </h2>
      <div style={{
        background: "rgba(255,255,255,0.08)", borderRadius: 10,
        padding: "20px 24px", display: "flex", flexDirection: "column", gap: 12,
      }}>
        <p style={{ color: "rgba(255,255,255,0.85)", fontSize: 14, margin: 0, lineHeight: 1.6 }}>
          SPARC will install its analysis engine (~400 MB) on your computer.
          This is a one-time setup that takes 1–3 minutes depending on your connection.
        </p>
        <div>
          <p style={{ color: "rgba(255,255,255,0.5)", fontSize: 12, margin: "0 0 4px" }}>
            Install location
          </p>
          <code style={{
            background: "rgba(0,0,0,0.3)", color: C.gold,
            padding: "4px 10px", borderRadius: 4, fontSize: 13,
            display: "inline-block",
          }}>
            ~/.sparc/env
          </code>
        </div>
      </div>
      <div style={{ marginTop: "auto", display: "flex", gap: 12 }}>
        <Btn variant="ghost" onClick={() => setStep(1)}>← Back</Btn>
        <Btn onClick={() => { setStep(3); runInstall(); }}>Install Now</Btn>
      </div>
    </div>
  );

  const renderStep3 = () => (
    <div style={{
      display: "flex", flexDirection: "column", alignItems: "center",
      justifyContent: "center", height: "100%", gap: 24,
      background: DARK_BG, padding: 40,
    }}>
      <div style={dragStyle} />

      <AnimatedLogo running={dl.status === "running"} />

      <div style={{ width: "100%", maxWidth: 380, display: "flex", flexDirection: "column", gap: 10 }}>
        <ProgressBar value={dl.progress} />
        <p style={{
          color: dl.status === "error" ? C.crimson : "rgba(255,255,255,0.75)",
          fontSize: 13, margin: 0, textAlign: "center", minHeight: 20,
        }}>
          {dl.status === "error" ? `Error: ${dl.error}` : dl.message}
        </p>
      </div>

      {dl.status === "error" && (
        <div style={{ display: "flex", gap: 12, marginTop: 8 }}>
          <Btn variant="ghost" onClick={() => invoke("setup_finish").catch(() => {})}>
            Quit
          </Btn>
          <Btn onClick={handleRetry}>Try Again</Btn>
        </div>
      )}
    </div>
  );

  const renderStep4 = () => (
    <div style={{
      display: "flex", flexDirection: "column", alignItems: "center",
      justifyContent: "center", height: "100%", gap: 24,
      background: GRADIENT, padding: 40,
    }}>
      <div style={dragStyle} />
      <div style={{
        width: 64, height: 64, borderRadius: "50%",
        background: "rgba(255,255,255,0.2)",
        display: "flex", alignItems: "center", justifyContent: "center",
        fontSize: 32,
      }}>
        ✓
      </div>
      <div style={{ textAlign: "center" }}>
        <h1 style={{ color: C.white, fontSize: 28, fontWeight: 700, margin: 0 }}>
          SPARC is ready.
        </h1>
        <p style={{ color: "rgba(255,255,255,0.8)", fontSize: 14, marginTop: 8 }}>
          Your analysis engine is installed and ready to use.
        </p>
      </div>
      <Btn onClick={handleFinish}>Launch SPARC →</Btn>
    </div>
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
      overflow: "hidden", userSelect: "none",
      position: "relative",
    }}>
      {steps[step]()}

      {/* Step indicator dots (steps 1-4, hidden on error) */}
      {dl.status !== "error" && (
        <div style={{
          position: "absolute", bottom: 18,
          left: 0, right: 0,
          display: "flex", justifyContent: "center", gap: 6,
        }}>
          {([1, 2, 3, 4] as Step[]).map((s) => (
            <div key={s} style={{
              width: 6, height: 6, borderRadius: "50%",
              background: s === step ? C.gold : "rgba(255,255,255,0.3)",
              transition: "background 300ms",
            }} />
          ))}
        </div>
      )}
    </div>
  );
}
