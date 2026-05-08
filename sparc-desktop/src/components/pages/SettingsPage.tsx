import { useState, useCallback } from "react";
import { SectionHeader, Card, Btn, KeyVal, Tag } from "@/components/ui/DesignSystem";
import { THEME_PRESETS, loadThemeKey, applyTheme } from "@/lib/theme";
import EasterEgg from "@/components/common/EasterEgg";
import OnboardingTour, { resetOnboarding } from "@/components/common/OnboardingTour";
import { useNotification } from "@/hooks/useNotifications";
import { useHardwareProfile } from "@/hooks/useHardwareProfile";
import { updatePreferences, type PerformancePreset } from "@/lib/api";
import { pickFolder } from "@/lib/fileDialogs";
import { getWorkspaceDir, setWorkspaceDir } from "@/lib/workspacePrefs";
import { useAuthStore } from "@/stores/authStore";

interface SettingsPageProps {
  onNavigate?: (page: "Performance") => void;
  onSignOut?: () => Promise<void>;
  parallaxEnabled?: boolean;
  onParallaxToggle?: (enabled: boolean) => void;
}

export default function SettingsPage({
  onNavigate,
  onSignOut,
  parallaxEnabled = true,
  onParallaxToggle,
}: SettingsPageProps = {}) {
  const [activeTheme, setActiveTheme] = useState<string>(loadThemeKey);
  const [apiKey, setApiKey] = useState(() => localStorage.getItem("sparc_api_key") ?? "");
  const [serverPort, setServerPort] = useState("8008");
  const [showSnake, setShowSnake] = useState(false);
  const [tourOpen, setTourOpen] = useState(false);
  const [workspaceDir, setWorkspaceDirState] = useState<string | null>(() => getWorkspaceDir());
  const { notify } = useNotification();
  const { data: hwData, refresh: refreshHw } = useHardwareProfile(true);
  const { user, signOut } = useAuthStore();

  const handlePickWorkspace = useCallback(async () => {
    const picked = await pickFolder();
    if (!picked) return;
    setWorkspaceDir(picked);
    setWorkspaceDirState(picked);
    notify("success", "Default workspace folder saved");
  }, [notify]);

  const handleClearWorkspace = useCallback(() => {
    setWorkspaceDir(null);
    setWorkspaceDirState(null);
    notify("info", "Workspace folder cleared");
  }, [notify]);

  const handlePresetQuick = useCallback(
    async (preset: PerformancePreset) => {
      try {
        await updatePreferences({ performance: { preset } });
        await refreshHw();
        notify("success", `Performance preset set to ${preset}.`);
      } catch (err) {
        notify("error", `Failed to update preset: ${err instanceof Error ? err.message : String(err)}`);
      }
    },
    [notify, refreshHw],
  );

  const handleThemeChange = useCallback((key: string) => {
    applyTheme(key);
    setActiveTheme(key);
    notify("success", `Theme changed to ${THEME_PRESETS.find((p) => p.key === key)?.label ?? key}`);
  }, [notify]);

  const handleSaveApiKey = useCallback(() => {
    localStorage.setItem("sparc_api_key", apiKey);
    notify("success", "API key saved");
  }, [apiKey, notify]);

  const handleSignOut = useCallback(async () => {
    await signOut();
    if (onSignOut) await onSignOut();
  }, [signOut, onSignOut]);

  // If user is not authenticated, show only the login redirect card
  if (!user) {
    return (
      <div>
        <SectionHeader kicker="settings" label="Settings" />
        <div style={{ maxWidth: 420, marginTop: 24 }}>
          <Card title="Account" subtitle="sign in to access settings">
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              <p style={{ fontSize: 12, color: "var(--muted)", lineHeight: 1.6, margin: 0 }}>
                You need to sign in to configure SPARC Labs Desktop. Create an account or sign in at{" "}
                <a
                  href="https://sparclabs.co"
                  target="_blank"
                  rel="noreferrer"
                  style={{ color: "var(--crimson)", textDecoration: "none", fontWeight: 600 }}
                >
                  sparclabs.co
                </a>
                , then return here to sign in.
              </p>
              <div style={{ display: "flex", gap: 8 }}>
                <Btn onClick={handleSignOut}>Sign In</Btn>
                <button
                  onClick={() => window.open("https://sparclabs.co", "_blank")}
                  style={{
                    padding: "6px 12px",
                    fontSize: 12,
                    border: "1px solid var(--line)",
                    borderRadius: 4,
                    background: "transparent",
                    color: "var(--ink-2)",
                    cursor: "pointer",
                  }}
                >
                  sparclabs.co →
                </button>
              </div>
            </div>
          </Card>
        </div>
      </div>
    );
  }

  return (
    <div>
      <SectionHeader kicker="settings" label="Settings" />

      {/* Account */}
      <div style={{ marginBottom: 14 }}>
        <Card title="Account" subtitle="your SPARC Labs identity">
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 12 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              {/* Avatar circle */}
              <div
                style={{
                  width: 40,
                  height: 40,
                  borderRadius: "50%",
                  background: "var(--crimson)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  color: "#fff",
                  fontWeight: 700,
                  fontSize: 16,
                  flexShrink: 0,
                }}
              >
                {user.email?.[0]?.toUpperCase() ?? "?"}
              </div>
              <div>
                <div style={{ fontSize: 13, fontWeight: 600 }}>{user.email}</div>
                <div className="mono" style={{ fontSize: 10, color: "var(--muted)", marginTop: 2 }}>
                  {user.id?.slice(0, 8)}…
                </div>
              </div>
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <button
                onClick={() => window.open("https://sparclabs.co/account", "_blank")}
                style={{
                  padding: "6px 12px",
                  fontSize: 12,
                  border: "1px solid var(--line)",
                  borderRadius: 4,
                  background: "transparent",
                  color: "var(--ink-2)",
                  cursor: "pointer",
                }}
              >
                Manage account →
              </button>
              <Btn small onClick={handleSignOut}>Sign Out</Btn>
            </div>
          </div>
        </Card>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
        {/* Theme */}
        <Card title="Appearance" subtitle="choose a theme">
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {THEME_PRESETS.map((preset) => {
              const active = activeTheme === preset.key;
              return (
                <button
                  key={preset.key}
                  onClick={() => handleThemeChange(preset.key)}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 12,
                    padding: "8px 10px",
                    border: `2px solid ${active ? "var(--ink)" : "var(--line)"}`,
                    borderRadius: 7,
                    background: active ? "var(--ink)" : "transparent",
                    cursor: "pointer",
                    textAlign: "left",
                    transition: "border-color 0.15s, background 0.15s",
                  }}
                >
                  {/* Swatch */}
                  <div
                    style={{
                      width: 28,
                      height: 28,
                      borderRadius: 5,
                      background: preset.swatch.bg,
                      border: "1px solid rgba(0,0,0,0.12)",
                      flexShrink: 0,
                      position: "relative",
                      overflow: "hidden",
                    }}
                  >
                    <div
                      style={{
                        position: "absolute",
                        bottom: 0,
                        right: 0,
                        width: 12,
                        height: 12,
                        background: preset.swatch.accent,
                        borderRadius: "3px 0 5px 0",
                      }}
                    />
                  </div>
                  <span
                    style={{
                      fontSize: 12.5,
                      fontWeight: 600,
                      color: active ? "var(--paper)" : "var(--ink)",
                    }}
                  >
                    {preset.label}
                  </span>
                  {active && (
                    <span style={{ marginLeft: "auto", fontSize: 10, color: "var(--paper)", opacity: 0.7 }}>
                      ✓ active
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        </Card>

        {/* Interactivity */}
        <Card title="Interactivity" subtitle="spatial effects & motion">
          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
              <div>
                <div style={{ fontSize: 12.5, fontWeight: 600 }}>4D Mouse Parallax</div>
                <div className="mono" style={{ fontSize: 10, color: "var(--muted)", marginTop: 2, lineHeight: 1.5 }}>
                  3D depth effect that follows your cursor on splash, login, and shell background
                </div>
              </div>
              <button
                onClick={() => onParallaxToggle?.(!parallaxEnabled)}
                style={{
                  flexShrink: 0,
                  width: 44,
                  height: 24,
                  borderRadius: 12,
                  border: "none",
                  background: parallaxEnabled ? "var(--crimson)" : "var(--line)",
                  cursor: "pointer",
                  position: "relative",
                  transition: "background 0.2s",
                }}
                aria-label="Toggle parallax"
              >
                <span
                  style={{
                    position: "absolute",
                    top: 3,
                    left: parallaxEnabled ? 23 : 3,
                    width: 18,
                    height: 18,
                    borderRadius: "50%",
                    background: "#fff",
                    transition: "left 0.2s",
                    boxShadow: "0 1px 3px rgba(0,0,0,0.2)",
                  }}
                />
              </button>
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
                    background: "var(--paper-2, #fff)",
                    color: "var(--ink)",
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
                  background: "var(--paper-2, #fff)",
                  color: "var(--ink)",
                }}
              />
            </div>

            <KeyVal label="Backend URL" value={`http://127.0.0.1:${serverPort}`} />
            <KeyVal label="WebSocket" value={`ws://127.0.0.1:${serverPort}/run/stream`} />
          </div>
        </Card>
      </div>

      {/* Workspace folder */}
      <div style={{ marginTop: 14 }}>
        <Card title="Workspace folder" subtitle="default location for new projects">
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <div
              className="mono"
              style={{
                fontSize: 11,
                color: workspaceDir ? "var(--ink-2)" : "var(--muted)",
                padding: "6px 8px",
                border: "1px dashed var(--line)",
                borderRadius: 4,
                background: "var(--paper-2, #fff)",
                wordBreak: "break-all",
              }}
            >
              {workspaceDir ?? "Not set — you'll be asked per project"}
            </div>
            <div style={{ display: "flex", gap: 6 }}>
              <Btn small onClick={handlePickWorkspace}>
                Choose folder…
              </Btn>
              {workspaceDir && (
                <Btn small onClick={handleClearWorkspace}>
                  Clear
                </Btn>
              )}
            </div>
            <div className="mono" style={{ fontSize: 10, color: "var(--muted)" }}>
              When set, clicking a template scaffolds <code>&lt;workspace&gt;/&lt;template&gt;_project</code> with no prompt.
            </div>
          </div>
        </Card>
      </div>

      {/* Performance */}
      <div style={{ marginTop: 14 }}>
        <Card
          title="Performance"
          subtitle="hardware tier, parallelism, memory"
          actions={
            <Btn small onClick={() => onNavigate?.("Performance")}>
              Open advanced…
            </Btn>
          }
        >
          {hwData ? (
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 14, alignItems: "center" }}>
                <Tag>{hwData.detected.tier}</Tag>
                <KeyVal
                  label="Auto-detected"
                  value={`${hwData.detected.total_ram_gb.toFixed(1)} GB RAM · ${hwData.detected.cpu_count} cores`}
                />
                <KeyVal
                  label="Effective"
                  value={`${hwData.effective.max_workers} workers · batch ${hwData.effective.batch_size}`}
                />
                <KeyVal label="Source" value={hwData.effective.source} />
              </div>

              <div>
                <div
                  className="mono"
                  style={{
                    fontSize: 9.5,
                    color: "var(--muted)",
                    textTransform: "uppercase",
                    letterSpacing: "0.1em",
                    marginBottom: 6,
                  }}
                >
                  Quick preset
                </div>
                <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                  {(["eco", "balanced", "performance", "max"] as PerformancePreset[]).map((p) => {
                    const active = hwData.effective.preset === p;
                    return (
                      <button
                        key={p}
                        onClick={() => handlePresetQuick(p)}
                        style={{
                          padding: "5px 10px",
                          fontSize: 11,
                          fontWeight: 600,
                          borderRadius: 4,
                          border: `1px solid ${active ? "var(--ink)" : "var(--line)"}`,
                          background: active ? "var(--ink)" : "transparent",
                          color: active ? "var(--paper)" : "var(--ink-2)",
                          cursor: "pointer",
                          textTransform: "capitalize",
                        }}
                      >
                        {p}
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>
          ) : (
            <div className="mono" style={{ fontSize: 11, color: "var(--muted)" }}>
              Loading hardware profile…
            </div>
          )}
        </Card>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginTop: 14 }}>
        {/* About */}
        <Card title="About" subtitle="SPARC Labs Desktop">
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <KeyVal label="Version" value={`v${__APP_VERSION__}`} />
            <KeyVal label="Framework" value="Tauri v2 + React 19" />
            <KeyVal label="Backend" value="FastAPI + Python 3.11" />
            <KeyVal label="License" value="Proprietary" />
            <div style={{ borderTop: "1px dashed var(--line)", paddingTop: 8, marginTop: 4 }}>
              <p style={{ fontSize: 12, color: "var(--muted)", lineHeight: 1.6, margin: 0 }}>
                <strong style={{ color: "var(--ink-2)" }}>SPARC</strong> — Spatial Predictive Analytics for
                Resilience and Causality. A physics-informed geospatial modeling toolkit for
                infrastructure resilience planning.
              </p>
            </div>
          </div>
        </Card>

        {/* Extras */}
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
                  <div style={{ fontSize: 12.5, fontWeight: 600 }}>Onboarding tour</div>
                  <div className="mono" style={{ fontSize: 10, color: "var(--muted)", marginTop: 2 }}>
                    replay the first-run walkthrough
                  </div>
                </div>
                <Btn small onClick={() => { resetOnboarding(); setTourOpen(true); notify("info", "Tour restarted"); }}>
                  Restart
                </Btn>
              </div>
            </div>

            <div style={{ borderTop: "1px dashed var(--line)", paddingTop: 10 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div>
                  <div style={{ fontSize: 12.5, fontWeight: 600 }}>Keyboard Shortcuts</div>
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

      <OnboardingTour open={tourOpen} onClose={() => setTourOpen(false)} />
    </div>
  );
}
