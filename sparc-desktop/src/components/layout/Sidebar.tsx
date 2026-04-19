import CubeLogo from "../brand/CubeLogo";

// ---------------------------------------------------------------------------
// Grouped navigation sections — matches design prototype exactly
// ---------------------------------------------------------------------------
const SECTIONS = [
  { label: "Setup",    pages: ["Project", "Data", "Processing"] },
  { label: "Analysis", pages: ["DAG", "Variables", "Physics", "CRS", "Scenarios", "Models"] },
  { label: "Pipeline", pages: ["Run", "Results", "Report"] },
] as const;

/** Flat list derived from sections — keeps downstream consumers working. */
const PAGES = SECTIONS.flatMap((s) => s.pages) as unknown as readonly PageName[];

export type PageName =
  | "Project" | "Data" | "Processing" | "Config" | "Variables" | "CRS"
  | "DAG" | "Physics" | "Scenarios" | "Models"
  | "Run" | "Results" | "Report";

interface SidebarProps {
  currentPage: PageName;
  onNavigate: (page: PageName) => void;
  onSettings?: () => void;
  onToggleChat?: () => void;
  chatOpen?: boolean;
  projectLoaded?: boolean;
  projectName?: string;
  projectEpsg?: string;
  projectDomain?: string;
}

export default function Sidebar({
  currentPage,
  onNavigate,
  onToggleChat,
  chatOpen,
  projectLoaded,
  projectName,
  projectEpsg,
  projectDomain,
}: SidebarProps) {
  let idx = 0;

  return (
    <aside
      style={{
        width: 240,
        flexShrink: 0,
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        background: '#fdfbf7',
        borderRight: '1px solid var(--color-sparc-line)',
      }}
    >
      {/* Brand lockup */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 12,
          padding: '14px 16px',
          borderBottom: '1px solid var(--color-sparc-line)',
        }}
      >
        <div style={{ width: 64, height: 64, marginTop: -2, marginLeft: -6 }}>
          <CubeLogo size={64} animate hue="ink" intensity={0.7} />
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', lineHeight: 1, gap: 4 }}>
          <span style={{ fontSize: 15, fontWeight: 800, letterSpacing: '-0.01em' }}>SPARC</span>
          <span style={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.2em', color: 'var(--color-sparc-muted)' }}>LABS</span>
        </div>
      </div>

      {/* Project pill */}
      <div style={{ padding: '10px 12px 4px' }}>
        <div
          style={{
            border: '1px dashed var(--color-sparc-line)',
            background: '#fff',
            borderRadius: 8,
            padding: '8px 10px',
          }}
        >
          <div
            className="mono"
            style={{ fontSize: 9, color: 'var(--color-sparc-muted)', textTransform: 'uppercase', letterSpacing: '0.12em' }}
          >
            active project
          </div>
          <div style={{ fontSize: 12, fontWeight: 700, marginTop: 3, display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ width: 6, height: 6, borderRadius: '50%', background: projectLoaded ? 'var(--color-sparc-crimson)' : 'var(--color-sparc-amber)' }} />
            {projectLoaded ? (projectName ?? 'Unnamed Project') : 'No project loaded'}
          </div>
          {projectLoaded && (
            <div className="mono" style={{ fontSize: 10, color: 'var(--color-sparc-muted)', marginTop: 2 }}>
              {projectDomain ?? 'project'} · {projectEpsg ?? 'EPSG:4326'}
            </div>
          )}
        </div>
      </div>

      {/* Navigation */}
      <nav className="scroll" style={{ flex: 1, overflowY: 'auto', padding: '6px 0 12px' }}>
        {SECTIONS.map((section, si) => (
          <div key={section.label} style={{ marginTop: si === 0 ? 4 : 10 }}>
            {/* Section header */}
            <div
              style={{
                padding: '4px 18px',
                fontSize: 9.5,
                fontWeight: 700,
                letterSpacing: '0.16em',
                color: 'var(--color-sparc-muted)',
                textTransform: 'uppercase',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
              }}
            >
              <span>{section.label}</span>
              <span className="mono" style={{ fontSize: 9, opacity: 0.6 }}>
                {String(si + 1).padStart(2, '0')}
              </span>
            </div>

            {/* Section pages */}
            {section.pages.map((page) => {
              const n = ++idx;
              const active = page === currentPage;
              const disabled = page !== 'Project' && !projectLoaded;
              return (
                <button
                  key={page}
                  onClick={() => !disabled && onNavigate(page as PageName)}
                  disabled={disabled}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 10,
                    width: '100%',
                    textAlign: 'left',
                    border: 'none',
                    padding: '7px 12px 7px 16px',
                    fontFamily: 'inherit',
                    fontSize: 13,
                    cursor: disabled ? 'not-allowed' : 'pointer',
                    background: active ? 'var(--color-sparc-ink)' : 'transparent',
                    color: disabled
                      ? 'var(--color-sparc-line)'
                      : active
                        ? '#fff'
                        : 'var(--color-sparc-ink-2)',
                    fontWeight: active ? 600 : 500,
                    transition: 'background 0.15s',
                    opacity: disabled ? 0.5 : 1,
                  }}
                  onMouseEnter={(e) => {
                    if (!active && !disabled) e.currentTarget.style.background = 'rgba(0,0,0,0.05)';
                  }}
                  onMouseLeave={(e) => {
                    if (!active && !disabled) e.currentTarget.style.background = 'transparent';
                  }}
                >
                  <span
                    className="mono"
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      width: 20,
                      height: 18,
                      borderRadius: 3,
                      fontSize: 10,
                      fontWeight: 600,
                      background: disabled
                        ? 'rgba(0,0,0,0.03)'
                        : active
                          ? 'var(--color-sparc-crimson)'
                          : 'rgba(0,0,0,0.06)',
                      color: disabled
                        ? 'var(--color-sparc-line)'
                        : active
                          ? '#fff'
                          : 'var(--color-sparc-muted)',
                    }}
                  >
                    {String(n).padStart(2, '0')}
                  </span>
                  {page}
                </button>
              );
            })}
          </div>
        ))}
      </nav>

      {/* AI Assistant toggle */}
      <div style={{ borderTop: '1px solid var(--color-sparc-line)', padding: 8 }}>
        <button
          onClick={onToggleChat}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            width: '100%',
            textAlign: 'left',
            padding: '8px 10px',
            borderRadius: 6,
            border: `1px solid ${chatOpen ? 'var(--color-sparc-ink)' : 'transparent'}`,
            background: chatOpen ? 'var(--color-sparc-ink)' : 'transparent',
            color: chatOpen ? '#fff' : 'var(--color-sparc-ink-2)',
            fontSize: 12.5,
            fontWeight: 600,
            cursor: 'pointer',
            fontFamily: 'inherit',
          }}
        >
          <span style={{ width: 18, height: 18, display: 'inline-flex', alignItems: 'center', justifyContent: 'center' }}>
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M2 4h12v7H9l-3 3v-3H2V4z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
            </svg>
          </span>
          Assistant
          <span
            className="mono"
            style={{
              marginLeft: 'auto',
              fontSize: 9,
              letterSpacing: '0.08em',
              padding: '2px 6px',
              borderRadius: 3,
              background: chatOpen ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.06)',
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
