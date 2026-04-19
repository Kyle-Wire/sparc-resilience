// Sidebar + Topbar + Window chrome for the SPARC desktop app.

const SECTIONS = [
  { label: "Setup", pages: ["Project", "Data", "Processing"] },
  { label: "Analysis", pages: ["DAG", "Variables", "Physics", "CRS", "Scenarios", "Models"] },
  { label: "Pipeline", pages: ["Run", "Results", "Report"] },
];

function Sidebar({ currentPage, onNavigate, onToggleChat, chatOpen }) {
  let idx = 0;
  return (
    <aside style={{
      width: 224, flexShrink: 0, height: "100%",
      display: "flex", flexDirection: "column",
      background: "#fdfbf7",
      borderRight: "1px solid var(--line)",
    }}>
      {/* Brand lockup */}
      <div style={{
        display: "flex", alignItems: "center", gap: 12,
        padding: "14px 16px",
        borderBottom: "1px solid var(--line)",
      }}>
        <div style={{ width: 40, height: 40, marginTop: -2 }}>
          <CubeLogo size={40} density={0.6} />
        </div>
        <div style={{ display: "flex", flexDirection: "column", lineHeight: 1, gap: 3 }}>
          <span style={{ fontSize: 13, fontWeight: 800, letterSpacing: "-0.01em" }}>SPARC</span>
          <span style={{ fontSize: 9.5, fontWeight: 600, letterSpacing: "0.18em", color: "var(--muted)" }}>LABS</span>
        </div>
      </div>

      {/* Project pill */}
      <div style={{ padding: "10px 12px 4px" }}>
        <div style={{
          border: "1px dashed var(--line)",
          background: "#fff",
          borderRadius: 8,
          padding: "8px 10px",
        }}>
          <div className="mono" style={{ fontSize: 9, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.12em" }}>
            active project
          </div>
          <div style={{ fontSize: 12, fontWeight: 700, marginTop: 3, display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--crimson)" }}></span>
            Brown UHI — 30m
          </div>
          <div className="mono" style={{ fontSize: 10, color: "var(--muted)", marginTop: 2 }}>uhi · EPSG:3438</div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="scroll" style={{ flex: 1, overflowY: "auto", padding: "6px 0 12px" }}>
        {SECTIONS.map((section, si) => (
          <div key={section.label} style={{ marginTop: si === 0 ? 4 : 10 }}>
            <div style={{
              padding: "4px 18px",
              fontSize: 9.5, fontWeight: 700,
              letterSpacing: "0.16em",
              color: "var(--muted)",
              textTransform: "uppercase",
              display: "flex", justifyContent: "space-between", alignItems: "center",
            }}>
              <span>{section.label}</span>
              <span className="mono" style={{ fontSize: 9, opacity: 0.6 }}>{String(si + 1).padStart(2, "0")}</span>
            </div>
            {section.pages.map((p) => {
              const n = ++idx;
              const active = p === currentPage;
              return (
                <button
                  key={p}
                  onClick={() => onNavigate(p)}
                  style={{
                    display: "flex", alignItems: "center", gap: 10,
                    width: "100%", textAlign: "left", border: "none",
                    padding: "7px 12px 7px 16px",
                    fontFamily: "inherit",
                    fontSize: 13,
                    cursor: "pointer",
                    background: active ? "var(--ink)" : "transparent",
                    color: active ? "#fff" : "var(--ink-2)",
                    fontWeight: active ? 600 : 500,
                    transition: "background 0.15s",
                  }}
                  onMouseEnter={(e) => { if (!active) e.currentTarget.style.background = "rgba(0,0,0,0.05)"; }}
                  onMouseLeave={(e) => { if (!active) e.currentTarget.style.background = "transparent"; }}
                >
                  <span className="mono" style={{
                    display: "inline-flex", alignItems: "center", justifyContent: "center",
                    width: 20, height: 18, borderRadius: 3,
                    fontSize: 10, fontWeight: 600,
                    background: active ? "var(--crimson)" : "rgba(0,0,0,0.06)",
                    color: active ? "#fff" : "var(--muted)",
                  }}>{String(n).padStart(2, "0")}</span>
                  {p}
                </button>
              );
            })}
          </div>
        ))}
      </nav>

      {/* AI assistant + settings */}
      <div style={{ borderTop: "1px solid var(--line)", padding: 8 }}>
        <button
          onClick={onToggleChat}
          style={{
            display: "flex", alignItems: "center", gap: 10,
            width: "100%", textAlign: "left",
            padding: "8px 10px", borderRadius: 6,
            border: "1px solid " + (chatOpen ? "var(--ink)" : "transparent"),
            background: chatOpen ? "var(--ink)" : "transparent",
            color: chatOpen ? "#fff" : "var(--ink-2)",
            fontSize: 12.5, fontWeight: 600, cursor: "pointer",
            fontFamily: "inherit",
          }}
        >
          <span style={{ width: 18, height: 18, display: "inline-flex", alignItems: "center", justifyContent: "center" }}>
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M2 4h12v7H9l-3 3v-3H2V4z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round"/>
            </svg>
          </span>
          Assistant
          <span className="mono" style={{
            marginLeft: "auto", fontSize: 9, letterSpacing: "0.08em",
            padding: "2px 6px", borderRadius: 3,
            background: chatOpen ? "rgba(255,255,255,0.15)" : "rgba(0,0,0,0.06)",
          }}>⌘K</span>
        </button>
      </div>
    </aside>
  );
}

function Topbar({ page, status }) {
  return (
    <div style={{
      height: 42, flexShrink: 0,
      display: "flex", alignItems: "center",
      borderBottom: "1px solid var(--line)",
      background: "#fdfbf7",
      paddingRight: 10,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "0 16px", flex: 1 }}>
        <span className="mono" style={{ fontSize: 10, color: "var(--muted)", letterSpacing: "0.12em", textTransform: "uppercase" }}>
          sparc · pipeline
        </span>
        <span style={{ fontSize: 11, color: "var(--line)" }}>/</span>
        <span style={{ fontSize: 12, fontWeight: 600 }}>{page}</span>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
        <StatusPill color="var(--crimson)" label="Sidecar" value="ready · :17123" />
        <StatusPill color="var(--amber)" label="GPU" value="CUDA · 11.8" />
        <StatusPill color="var(--purple)" label="Claude" value="haiku-4.5" />
      </div>
    </div>
  );
}

function StatusPill({ color, label, value }) {
  return (
    <div className="mono" style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 10, color: "var(--muted)" }}>
      <span style={{ width: 6, height: 6, borderRadius: "50%", background: color }}></span>
      <span style={{ letterSpacing: "0.1em", textTransform: "uppercase" }}>{label}</span>
      <span style={{ color: "var(--ink-2)" }}>{value}</span>
    </div>
  );
}

// Window chrome (mac-style) — presented as a running desktop app
function WindowChrome({ children, title = "SPARC Labs — Resilience Core" }) {
  return (
    <div style={{
      width: "100%", height: "100%",
      maxWidth: 1440, maxHeight: 900,
      borderRadius: 14,
      overflow: "hidden",
      boxShadow: "0 1px 0 rgba(255,255,255,0.5) inset, 0 20px 60px rgba(0,0,0,0.22), 0 0 0 1px rgba(0,0,0,0.18)",
      display: "flex", flexDirection: "column",
      background: "var(--paper)",
    }}>
      <div style={{
        height: 36, display: "flex", alignItems: "center",
        background: "linear-gradient(180deg, #e8e2d4 0%, #dcd5c4 100%)",
        borderBottom: "1px solid rgba(0,0,0,0.18)",
        position: "relative", flexShrink: 0,
      }}>
        <div style={{ display: "flex", gap: 8, padding: "0 14px" }}>
          {["#ff5f57", "#febc2e", "#28c840"].map((c, i) => (
            <span key={i} style={{
              width: 12, height: 12, borderRadius: "50%", background: c,
              border: "0.5px solid rgba(0,0,0,0.15)",
            }}/>
          ))}
        </div>
        <div style={{
          position: "absolute", left: 0, right: 0, textAlign: "center",
          fontSize: 12, fontWeight: 600, color: "var(--ink-2)",
          letterSpacing: "0.01em", pointerEvents: "none",
        }}>{title}</div>
        <div style={{ marginLeft: "auto", display: "flex", gap: 10, padding: "0 14px" }}>
          <span className="mono" style={{ fontSize: 10, color: "var(--muted)" }}>v0.4.2 · desktop</span>
        </div>
      </div>
      <div style={{ flex: 1, display: "flex", minHeight: 0 }}>
        {children}
      </div>
    </div>
  );
}

window.Sidebar = Sidebar;
window.Topbar = Topbar;
window.WindowChrome = WindowChrome;
