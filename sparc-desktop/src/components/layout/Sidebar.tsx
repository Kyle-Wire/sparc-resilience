import CubeLogo from "../brand/CubeLogo";

const SECTIONS = [
  { label: "Setup", pages: ["Project", "Data", "Processing"] },
  { label: "Analysis", pages: ["DAG", "Variables", "Physics", "CRS", "Scenarios", "Models"] },
  { label: "Pipeline", pages: ["Run", "Results", "Decisions", "Compare", "Report"] },
] as const;

const PAGES = SECTIONS.flatMap((s) => s.pages) as unknown as readonly PageName[];

export type PageName =
  | "Project" | "Data" | "Processing"
  | "DAG" | "Variables" | "Physics" | "CRS" | "Scenarios" | "Models"
  | "Run" | "Results" | "Decisions" | "Compare" | "Report";

interface SidebarProps {
  currentPage: PageName;
  onNavigate: (page: PageName | "Settings") => void;
  onToggleChat?: () => void;
  chatOpen?: boolean;
  projectLoaded?: boolean;
  projectName?: string;
  projectDomain?: string;
  projectEpsg?: string;
}

export default function Sidebar({
  currentPage,
  onNavigate,
  onToggleChat,
  chatOpen,
  projectLoaded,
  projectName,
  projectDomain,
  projectEpsg,
}: SidebarProps) {
  let idx = 0;
  return (
    <aside
      style={{
        width: 224,
        flexShrink: 0,
        height: "100%",
        display: "flex",
        flexDirection: "column",
        background: "#fdfbf7",
        borderRight: "1px solid var(--line)",
      }}
    >
      {/* Brand lockup */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 12,
          padding: "14px 16px",
          borderBottom: "1px solid var(--line)",
        }}
      >
        <div style={{ width: 40, height: 40, marginTop: -2 }}>
          <CubeLogo size={40} />
        </div>
        <div style={{ display: "flex", flexDirection: "column", lineHeight: 1, gap: 3 }}>
          <span style={{ fontSize: 13, fontWeight: 800, letterSpacing: "-0.01em" }}>SPARC</span>
          <span style={{ fontSize: 9.5, fontWeight: 600, letterSpacing: "0.18em", color: "var(--muted)" }}>LABS</span>
        </div>
      </div>

      {/* Project pill */}
      <div style={{ padding: "10px 12px 4px" }}>
        <div
          style={{
            border: "1px dashed var(--line)",
            background: "#fff",
            borderRadius: 8,
            padding: "8px 10px",
          }}
        >
          <div
            className="mono"
            style={{ fontSize: 9, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.12em" }}
          >
            active project
          </div>
          <div style={{ fontSize: 12, fontWeight: 700, marginTop: 3, display: "flex", alignItems: "center", gap: 6 }}>
            <span
              style={{
                width: 6,
                height: 6,
                borderRadius: "50%",
                background: projectLoaded ? "var(--crimson)" : "var(--muted)",
              }}
            />
            {projectLoaded && projectName ? projectName : "No project"}
          </div>
          {projectLoaded && (projectDomain || projectEpsg) && (
            <div className="mono" style={{ fontSize: 10, color: "var(--muted)", marginTop: 2 }}>
              {projectDomain}{projectDomain && projectEpsg ? " · " : ""}{projectEpsg ? `EPSG:${projectEpsg}` : ""}
            </div>
          )}
        </div>
      </div>

      {/* Navigation */}
      <nav className="scroll" style={{ flex: 1, overflowY: "auto", padding: "6px 0 12px" }}>
        {SECTIONS.map((section, si) => (
          <div key={section.label} style={{ marginTop: si === 0 ? 4 : 10 }}>
            <div
              style={{
                padding: "4px 18px",
                fontSize: 9.5,
                fontWeight: 700,
                letterSpacing: "0.16em",
                color: "var(--muted)",
                textTransform: "uppercase",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              <span>{section.label}</span>
              <span className="mono" style={{ fontSize: 9, opacity: 0.6 }}>
                {String(si + 1).padStart(2, "0")}
              </span>
            </div>
            {section.pages.map((p) => {
              const n = ++idx;
              const active = p === currentPage;
              const disabled = p !== "Project" && !projectLoaded;
              return (
                <button
                  key={p}
                  onClick={() => !disabled && onNavigate(p as PageName)}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 10,
                    width: "100%",
                    textAlign: "left",
                    border: "none",
                    padding: "7px 12px 7px 16px",
                    fontFamily: "inherit",
                    fontSize: 13,
                    cursor: disabled ? "default" : "pointer",
                    background: active ? "var(--ink)" : "transparent",
                    color: active ? "#fff" : disabled ? "var(--muted)" : "var(--ink-2)",
                    fontWeight: active ? 600 : 500,
                    opacity: disabled ? 0.5 : 1,
                    transition: "background 0.15s",
                  }}
                  onMouseEnter={(e) => {
                    if (!active && !disabled) e.currentTarget.style.background = "rgba(0,0,0,0.05)";
                  }}
                  onMouseLeave={(e) => {
                    if (!active && !disabled) e.currentTarget.style.background = "transparent";
                  }}
                >
                  <span
                    className="mono"
                    style={{
                      display: "inline-flex",
                      alignItems: "center",
                      justifyContent: "center",
                      width: 20,
                      height: 18,
                      borderRadius: 3,
                      fontSize: 10,
                      fontWeight: 600,
                      background: active ? "var(--crimson)" : "rgba(0,0,0,0.06)",
                      color: active ? "#fff" : "var(--muted)",
                    }}
                  >
                    {String(n).padStart(2, "0")}
                  </span>
                  {p}
                </button>
              );
            })}
          </div>
        ))}
      </nav>

      {/* Settings gear */}
      <div style={{ padding: "4px 8px 0" }}>
        <button
          onClick={() => onNavigate("Settings" as any)}
          style={{
            display: "flex",
            alignItems: "center",
            gap: 10,
            width: "100%",
            textAlign: "left",
            padding: "8px 10px",
            borderRadius: 6,
            border: "none",
            background: currentPage === ("Settings" as any) ? "var(--ink)" : "transparent",
            color: currentPage === ("Settings" as any) ? "#fff" : "var(--ink-2)",
            fontSize: 12.5,
            fontWeight: 600,
            cursor: "pointer",
            fontFamily: "inherit",
            transition: "background 0.15s",
          }}
          onMouseEnter={(e) => {
            if (currentPage !== ("Settings" as any)) e.currentTarget.style.background = "rgba(0,0,0,0.05)";
          }}
          onMouseLeave={(e) => {
            if (currentPage !== ("Settings" as any)) e.currentTarget.style.background = "transparent";
          }}
        >
          <span style={{ width: 18, height: 18, display: "inline-flex", alignItems: "center", justifyContent: "center" }}>
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M8 10a2 2 0 100-4 2 2 0 000 4z" stroke="currentColor" strokeWidth="1.3" />
              <path d="M13.5 8c0-.3-.2-.6-.4-.8l1-1.6-1.2-1.2-1.6 1c-.2-.2-.5-.4-.8-.4V3.5h-1V5c-.3 0-.6.2-.8.4l-1.6-1L5.9 5.6l1 1.6c-.2.2-.4.5-.4.8H5v1h1.5c0 .3.2.6.4.8l-1 1.6 1.2 1.2 1.6-1c.2.2.5.4.8.4v1.5h1V12c.3 0 .6-.2.8-.4l1.6 1 1.2-1.2-1-1.6c.2-.2.4-.5.4-.8H14v-1h-1.5z" stroke="currentColor" strokeWidth="1.2" />
            </svg>
          </span>
          Settings
          <span
            className="mono"
            style={{
              marginLeft: "auto",
              fontSize: 9,
              letterSpacing: "0.08em",
              padding: "2px 6px",
              borderRadius: 3,
              background: currentPage === ("Settings" as any) ? "rgba(255,255,255,0.15)" : "rgba(0,0,0,0.06)",
            }}
          >
            ⌘,
          </span>
        </button>
      </div>

      {/* AI assistant */}
      <div style={{ borderTop: "1px solid var(--line)", padding: 8 }}>
        <button
          onClick={onToggleChat}
          style={{
            display: "flex",
            alignItems: "center",
            gap: 10,
            width: "100%",
            textAlign: "left",
            padding: "8px 10px",
            borderRadius: 6,
            border: "1px solid " + (chatOpen ? "var(--ink)" : "transparent"),
            background: chatOpen ? "var(--ink)" : "transparent",
            color: chatOpen ? "#fff" : "var(--ink-2)",
            fontSize: 12.5,
            fontWeight: 600,
            cursor: "pointer",
            fontFamily: "inherit",
          }}
        >
          <span style={{ width: 18, height: 18, display: "inline-flex", alignItems: "center", justifyContent: "center" }}>
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M2 4h12v7H9l-3 3v-3H2V4z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
            </svg>
          </span>
          Assistant
          <span
            className="mono"
            style={{
              marginLeft: "auto",
              fontSize: 9,
              letterSpacing: "0.08em",
              padding: "2px 6px",
              borderRadius: 3,
              background: chatOpen ? "rgba(255,255,255,0.15)" : "rgba(0,0,0,0.06)",
            }}
          >
            ⌘K
          </span>
        </button>
      </div>
    </aside>
  );
}

export { PAGES };
