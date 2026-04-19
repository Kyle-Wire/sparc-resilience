import type { ReactNode } from "react";

export interface Tweaks {
  logoHue: "ink" | "red" | "purple" | "amber";
  logoDensity: number;
  paperTone: "warm" | "cool" | "white";
  accent: "crimson";
}

export const DEFAULT_TWEAKS: Tweaks = {
  logoHue: "ink",
  logoDensity: 1,
  paperTone: "warm",
  accent: "crimson",
};

interface TweaksPanelProps {
  tweaks: Tweaks;
  setTweaks: (updater: (prev: Tweaks) => Tweaks) => void;
  onClose: () => void;
}

export default function TweaksPanel({ tweaks, setTweaks, onClose }: TweaksPanelProps) {
  const set = <K extends keyof Tweaks>(k: K, v: Tweaks[K]) =>
    setTweaks((t) => ({ ...t, [k]: v }));

  return (
    <div
      style={{
        position: "fixed",
        right: 20,
        bottom: 20,
        width: 260,
        background: "#fff",
        border: "1px solid var(--ink)",
        borderRadius: 8,
        zIndex: 60,
        boxShadow: "0 12px 32px rgba(0,0,0,0.18)",
        fontFamily: "inherit",
      }}
    >
      <div
        style={{
          padding: "10px 12px",
          borderBottom: "1px solid var(--line)",
          display: "flex",
          alignItems: "center",
          background: "var(--ink)",
          color: "#fff",
          borderRadius: "8px 8px 0 0",
        }}
      >
        <span className="mono" style={{ fontSize: 10, letterSpacing: "0.18em", fontWeight: 700 }}>
          TWEAKS
        </span>
        <button
          onClick={onClose}
          style={{
            marginLeft: "auto",
            color: "#fff",
            background: "transparent",
            border: "none",
            cursor: "pointer",
            fontSize: 14,
          }}
        >
          ×
        </button>
      </div>
      <div style={{ padding: 12, display: "flex", flexDirection: "column", gap: 12 }}>
        <TweakField label="Logo colour">
          <div style={{ display: "flex", gap: 4 }}>
            {(["ink", "red", "purple", "amber"] as const).map((h) => (
              <button
                key={h}
                onClick={() => set("logoHue", h)}
                style={{
                  flex: 1,
                  padding: "5px 0",
                  fontSize: 10.5,
                  fontFamily: "inherit",
                  border: "1px solid " + (tweaks.logoHue === h ? "var(--ink)" : "var(--line)"),
                  background: tweaks.logoHue === h ? "var(--ink)" : "#fff",
                  color: tweaks.logoHue === h ? "#fff" : "var(--ink-2)",
                  borderRadius: 4,
                  cursor: "pointer",
                  textTransform: "uppercase",
                  letterSpacing: "0.05em",
                }}
              >
                {h}
              </button>
            ))}
          </div>
        </TweakField>
        <TweakField label={`Matter density · ${tweaks.logoDensity.toFixed(2)}`}>
          <input
            type="range"
            min="0.3"
            max="2"
            step="0.05"
            value={tweaks.logoDensity}
            onChange={(e) => set("logoDensity", parseFloat(e.target.value))}
            style={{ width: "100%" }}
          />
        </TweakField>
        <TweakField label="Paper tone">
          <div style={{ display: "flex", gap: 4 }}>
            {(["warm", "cool", "white"] as const).map((t) => (
              <button
                key={t}
                onClick={() => set("paperTone", t)}
                style={{
                  flex: 1,
                  padding: "5px 0",
                  fontSize: 10.5,
                  fontFamily: "inherit",
                  border: "1px solid " + (tweaks.paperTone === t ? "var(--ink)" : "var(--line)"),
                  background: tweaks.paperTone === t ? "var(--ink)" : "#fff",
                  color: tweaks.paperTone === t ? "#fff" : "var(--ink-2)",
                  borderRadius: 4,
                  cursor: "pointer",
                  textTransform: "uppercase",
                  letterSpacing: "0.05em",
                }}
              >
                {t}
              </button>
            ))}
          </div>
        </TweakField>
      </div>
    </div>
  );
}

function TweakField({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div>
      <div
        className="mono"
        style={{
          fontSize: 9.5,
          color: "var(--muted)",
          letterSpacing: "0.1em",
          textTransform: "uppercase",
          marginBottom: 5,
        }}
      >
        {label}
      </div>
      {children}
    </div>
  );
}
