import { useState, useCallback, useEffect } from "react";
import { open as openExternal } from "@tauri-apps/plugin-shell";
import { SectionHeader, Card, Btn, KeyVal, Tag } from "@/components/ui/DesignSystem";
import { LOGO_HUES, PAPER_TONES, type LogoHue, type PaperTone, type ThemeSettings, loadTheme, applyTheme } from "@/lib/theme";
import EasterEgg from "@/components/common/EasterEgg";
import OnboardingTour, { resetOnboarding } from "@/components/common/OnboardingTour";
import { useNotification } from "@/hooks/useNotifications";
import { useAuth } from "@/lib/auth/AuthProvider";
import { secretGet, secretSet, SECRET_KEYS } from "@/lib/auth/secrets";

import { LEMON_CUSTOMER_PORTAL_URL as LEMON_PORTAL_URL, LEMON_UPGRADE_URL } from "@/lib/auth/commerce";

export default function SettingsPage() {
  const [theme, setTheme] = useState<ThemeSettings>(loadTheme);
  const [apiKey, setApiKey] = useState("");
  const [serverPort, setServerPort] = useState("8008");
  const [showSnake, setShowSnake] = useState(false);
  const [tourOpen, setTourOpen] = useState(false);
  const { notify } = useNotification();
  const auth = useAuth();

  // Apply theme on change
  useEffect(() => {
    applyTheme(theme);
    localStorage.setItem("sparc_theme", JSON.stringify(theme));
  }, [theme]);

  // Load Anthropic API key from keyring; one-time migration from
  // legacy `localStorage` storage. Plaintext-in-localStorage keys
  // are wiped after a successful migration.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const fromKeyring = await secretGet(SECRET_KEYS.anthropicApiKey);
      if (cancelled) return;
      if (fromKeyring) {
        setApiKey(fromKeyring);
        return;
      }
      const legacy = localStorage.getItem("sparc_api_key");
      if (legacy) {
        await secretSet(SECRET_KEYS.anthropicApiKey, legacy);
        localStorage.removeItem("sparc_api_key");
        if (!cancelled) setApiKey(legacy);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const handleHueChange = useCallback((hue: LogoHue) => {
    setTheme((prev) => ({ ...prev, logoHue: hue }));
  }, []);

  const handleToneChange = useCallback((tone: PaperTone) => {
    setTheme((prev) => ({ ...prev, paperTone: tone }));
  }, []);

  const handleSaveApiKey = useCallback(async () => {
    try {
      await secretSet(SECRET_KEYS.anthropicApiKey, apiKey);
      notify("success", "API key saved to system keychain");
    } catch (e) {
      notify("error", e instanceof Error ? e.message : "Failed to save API key");
    }
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

      {/* License & Account */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginTop: 14 }}>
        <Card title="Account" subtitle="signed-in user">
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <KeyVal label="Email" value={auth.email ?? "—"} />
            <KeyVal label="Provider" value={auth.provider ?? "—"} />
            <KeyVal label="Status" value={auth.state} />
            <div style={{ display: "flex", gap: 8, justifyContent: "flex-end", marginTop: 6 }}>
              <Btn small onClick={() => void auth.signOut()}>
                Sign out
              </Btn>
            </div>
          </div>
        </Card>

        <Card title="License" subtitle="plan, status, refresh">
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <Tag color={planColor(auth.license?.plan)}>{(auth.license?.plan ?? "free").toUpperCase()}</Tag>
              <Tag color={statusColor(auth.license?.status)}>
                {auth.license?.status ?? "—"}
              </Tag>
            </div>
            <KeyVal
              label="License key"
              value={auth.license?.license_key_last4 ? `••••${auth.license.license_key_last4}` : "—"}
            />
            <KeyVal label="Expires" value={fmtDate(auth.license?.expiry)} />
            <KeyVal
              label="Grace remaining"
              value={`${Math.max(0, auth.daysRemaining)} day${auth.daysRemaining === 1 ? "" : "s"}`}
            />
            <KeyVal label="Last verified" value={fmtDate(auth.license?.last_verified)} />
            <div style={{ display: "flex", gap: 8, justifyContent: "flex-end", marginTop: 6, flexWrap: "wrap" }}>
              <Btn small onClick={() => void openExternal(LEMON_PORTAL_URL)}>
                Manage subscription
              </Btn>
              {auth.license?.plan === "free" && (
                <Btn small onClick={() => void openExternal(LEMON_UPGRADE_URL)}>
                  Upgrade
                </Btn>
              )}
              <Btn small primary onClick={() => void auth.refresh({ force: true })} disabled={auth.busy}>
                {auth.busy ? "Refreshing…" : "Refresh license"}
              </Btn>
            </div>
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

      <OnboardingTour open={tourOpen} onClose={() => setTourOpen(false)} />
    </div>
  );
}

// ─── Display helpers ────────────────────────────────────────────────

function fmtDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (isNaN(d.getTime())) return "—";
  return d.toLocaleString();
}

function planColor(plan: string | undefined): string {
  switch (plan) {
    case "team":
      return "#5b3aa1";
    case "pro":
      return "#1e7a5a";
    case "free":
    default:
      return "var(--muted)";
  }
}

function statusColor(status: string | undefined): string {
  switch (status) {
    case "active":
    case "trialing":
      return "#1e7a5a";
    case "cancelled":
    case "past_due":
      return "#c0392b";
    default:
      return "var(--muted)";
  }
}
