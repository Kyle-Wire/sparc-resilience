import { useState, useCallback, useEffect } from "react";
import { SectionHeader, Card, Btn, KeyVal } from "@/components/ui/DesignSystem";
import { LOGO_HUES, PAPER_TONES, type LogoHue, type PaperTone, type ThemeSettings, loadTheme, applyTheme } from "@/lib/theme";
import EasterEgg from "@/components/common/EasterEgg";
import { useNotification } from "@/hooks/useNotifications";

export default function SettingsPage() {
  const [theme, setTheme] = useState<ThemeSettings>(loadTheme);
  const [apiKey, setApiKey] = useState(() => localStorage.getItem("sparc_api_key") ?? "");
  const [serverPort, setServerPort] = useState("8008");
  const [showSnake, setShowSnake] = useState(false);
  const { notify } = useNotification();

  // Apply theme on change
  useEffect(() => {
    applyTheme(theme);
    localStorage.setItem("sparc_theme", JSON.stringify(theme));
  }, [theme]);

  const handleHueChange = useCallback((hue: LogoHue) => {
    setTheme((prev) => ({ ...prev, logoHue: hue }));
  }, []);

  const handleToneChange = useCallback((tone: PaperTone) => {
    setTheme((prev) => ({ ...prev, paperTone: tone }));
  }, []);

  const handleSaveApiKey = useCallback(() => {
    localStorage.setItem("sparc_api_key", apiKey);
    notify("success", "API key saved");
  }, [apiKey, notify]);

  return (
    <div>
      <SectionHeader kicker="settings" label="Settings" />

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
        {/* Theme */}
        <Card title="Appearance" subtitle="logo hue and paper tone">
          <div style={{ marginBottom: 16 }}>
            <div
              className="mono"
              style={{
                fontSize: 9.5,
                color: "var(--muted)",
                textTransform: "uppercase",
                letterSpacing: "0.1em",
                marginBottom: 8,
              }}
            >
              Logo Hue
            </div>
            <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
              {LOGO_HUES.map((hue) => (
                <button
                  key={hue.key}
                  onClick={() => handleHueChange(hue.key)}
                  style={{
                    width: 32,
                    height: 32,
                    borderRadius: 6,
                    border: `2px solid ${theme.logoHue === hue.key ? "var(--ink)" : "var(--line)"}`,
                    background: hue.color,
                    cursor: "pointer",
                  }}
                  title={hue.label}
                />
              ))}
            </div>
          </div>

          <div>
            <div
              className="mono"
              style={{
                fontSize: 9.5,
                color: "var(--muted)",
                textTransform: "uppercase",
                letterSpacing: "0.1em",
                marginBottom: 8,
              }}
            >
              Paper Tone
            </div>
            <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
              {PAPER_TONES.map((tone) => (
                <button
                  key={tone.key}
                  onClick={() => handleToneChange(tone.key)}
                  style={{
                    width: 32,
                    height: 32,
                    borderRadius: 6,
                    border: `2px solid ${theme.paperTone === tone.key ? "var(--ink)" : "var(--line)"}`,
                    background: tone.paper,
                    cursor: "pointer",
                  }}
                  title={tone.label}
                />
              ))}
            </div>
          </div>
        </Card>

        {/* Connection */}
        <Card title="Connection" subtitle="API and server configuration">
          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <div>
              <div
                className="mono"
                style={{
                  fontSize: 9.5,
                  color: "var(--muted)",
                  textTransform: "uppercase",
                  letterSpacing: "0.1em",
                  marginBottom: 4,
                }}
              >
                API Key
              </div>
              <div style={{ display: "flex", gap: 8 }}>
                <input
                  type="password"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder="sk-..."
                  style={{
                    border: "1px solid var(--line)",
                    borderRadius: 4,
                    padding: "6px 8px",
                    fontSize: 12,
                    fontFamily: "'JetBrains Mono', monospace",
                    flex: 1,
                    background: "#fff",
                  }}
                />
                <Btn small onClick={handleSaveApiKey}>Save</Btn>
              </div>
            </div>

            <div>
              <div
                className="mono"
                style={{
                  fontSize: 9.5,
                  color: "var(--muted)",
                  textTransform: "uppercase",
                  letterSpacing: "0.1em",
                  marginBottom: 4,
                }}
              >
                Server Port
              </div>
              <input
                type="text"
                value={serverPort}
                onChange={(e) => setServerPort(e.target.value)}
                style={{
                  border: "1px solid var(--line)",
                  borderRadius: 4,
                  padding: "6px 8px",
                  fontSize: 12,
                  fontFamily: "'JetBrains Mono', monospace",
                  width: 100,
                  background: "#fff",
                }}
              />
            </div>

            <KeyVal label="Backend URL" value={`http://127.0.0.1:${serverPort}`} />
            <KeyVal label="WebSocket" value={`ws://127.0.0.1:${serverPort}/run/stream`} />
          </div>
        </Card>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginTop: 14 }}>
        {/* About */}
        <Card title="About" subtitle="SPARC Labs Desktop">
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <KeyVal label="Version" value="2.1.0-beta" />
            <KeyVal label="Framework" value="Tauri v2 + React 19" />
            <KeyVal label="Backend" value="FastAPI + Python 3.11" />
            <KeyVal label="License" value="MIT" />
            <div style={{ borderTop: "1px dashed var(--line)", paddingTop: 8, marginTop: 4 }}>
              <p style={{ fontSize: 12, color: "var(--muted)", lineHeight: 1.6, margin: 0 }}>
                <strong style={{ color: "var(--ink-2)" }}>SPARC</strong> — Spatial Predictive Analytics for
                Resilience and Causality. A physics-informed geospatial modeling toolkit for
                infrastructure resilience planning.
              </p>
            </div>
          </div>
        </Card>

        {/* Easter egg */}
        <Card title="Extras" subtitle="hidden features">
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div>
                <div style={{ fontSize: 12.5, fontWeight: 600 }}>Snake Game</div>
                <div className="mono" style={{ fontSize: 10, color: "var(--muted)", marginTop: 2 }}>
                  or press ↑↑↓↓←→←→BA (Konami code)
                </div>
              </div>
              <Btn small onClick={() => setShowSnake(!showSnake)}>
                {showSnake ? "Hide" : "Play"}
              </Btn>
            </div>
            {showSnake && (
              <div
                style={{
                  border: "1px solid var(--line)",
                  borderRadius: 6,
                  padding: 8,
                  background: "#1a1416",
                }}
              >
                <EasterEgg onClose={() => setShowSnake(false)} />
              </div>
            )}

            <div style={{ borderTop: "1px dashed var(--line)", paddingTop: 10 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div>
                  <div style={{ fontSize: 12.5, fontWeight: 600 }}>Keyboard Shortcuts</div>
                  <div className="mono" style={{ fontSize: 10, color: "var(--muted)", marginTop: 2 }}>
                    quick navigation
                  </div>
                </div>
              </div>
              <div style={{ marginTop: 8, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 4 }}>
                {["⌘1 Project", "⌘2 Data", "⌘3 Processing", "⌘4 DAG", "⌘5 Variables", "⌘6 Physics",
                  "⌘7 CRS", "⌘8 Scenarios", "⌘9 Models", "⌘0 Run", "⌘- Results", "⌘= Report",
                ].map((s) => (
                  <div key={s} className="mono" style={{ fontSize: 10, color: "var(--muted)", padding: "2px 0" }}>
                    {s}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
