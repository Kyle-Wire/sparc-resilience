import { useState } from "react";
import CubeLogo from "../brand/CubeLogo";

// ---------------------------------------------------------------------------
// Grouped navigation sections with kicker numbering
// ---------------------------------------------------------------------------
const SECTIONS = [
  { label: "Setup",    kicker: "01–04", pages: ["Project", "Data", "Processing", "Config"] },
  { label: "Analysis", kicker: "05–10", pages: ["DAG", "Variables", "Physics", "CRS", "Scenarios", "Models"] },
  { label: "Pipeline", kicker: "11–13", pages: ["Run", "Results", "Report"] },
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
}

export default function Sidebar({
  currentPage,
  onNavigate,
  onSettings,
  onToggleChat,
  chatOpen,
  projectLoaded,
  projectName,
}: SidebarProps) {
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});

  const toggleSection = (label: string) =>
    setCollapsed((prev) => ({ ...prev, [label]: !prev[label] }));

  let pageIndex = 0;

  return (
    <aside
      className="flex h-full flex-col border-r"
      style={{
        width: 240,
        background: '#fdfbf7',
        borderColor: 'var(--color-sparc-line)',
      }}
    >
      {/* Logo lockup: animated CubeLogo + wordmark */}
      <div
        className="flex items-center gap-3 border-b px-4 py-4"
        style={{ borderColor: 'var(--color-sparc-line)' }}
      >
        <CubeLogo size={36} animate hue="ink" intensity={0.6} />
        <div className="flex flex-col leading-none">
          <span className="text-sm font-extrabold tracking-tight" style={{ color: 'var(--color-sparc-ink)' }}>SPARC</span>
          <span
            className="mono text-[9.5px] font-medium uppercase"
            style={{ letterSpacing: '0.2em', color: 'var(--color-sparc-muted)' }}
          >
            Labs · v0.4.2
          </span>
        </div>
      </div>

      {/* Project pill */}
      <div
        className="mx-3 mt-3 rounded-lg border px-3 py-2"
        style={{
          borderColor: projectLoaded ? 'var(--color-sparc-line)' : 'var(--color-sparc-amber)',
          background: projectLoaded ? '#fff' : 'rgba(231,144,36,0.08)',
        }}
      >
        <div
          className="mono text-[9px] font-semibold uppercase"
          style={{ letterSpacing: '0.12em', color: 'var(--color-sparc-muted)' }}
        >
          Active project
        </div>
        <div
          className="mt-0.5 text-[12px] font-semibold truncate"
          style={{ color: projectLoaded ? 'var(--color-sparc-ink)' : 'var(--color-sparc-amber)' }}
        >
          {projectLoaded ? (projectName ?? 'Unnamed Project') : 'No project loaded'}
        </div>
      </div>

      {/* Sectioned navigation */}
      <nav className="scroll flex-1 overflow-y-auto py-2">
        {SECTIONS.map((section, sectionIdx) => {
          const isCollapsed = collapsed[section.label] ?? false;
          const sectionActive = (section.pages as readonly string[]).includes(currentPage);

          return (
            <div key={section.label} className={sectionIdx > 0 ? 'mt-1' : ''}>
              {/* Section header */}
              <button
                onClick={() => toggleSection(section.label)}
                className="flex w-full items-center justify-between px-4 py-1.5 group"
              >
                <span className="flex items-center gap-2">
                  <span
                    className="mono text-[9px] font-medium"
                    style={{ letterSpacing: '0.1em', color: 'var(--color-sparc-muted)' }}
                  >
                    {section.kicker}
                  </span>
                  <span
                    className="text-[10px] font-bold uppercase"
                    style={{
                      letterSpacing: '0.12em',
                      color: sectionActive ? 'var(--color-sparc-crimson)' : 'var(--color-sparc-muted)',
                    }}
                  >
                    {section.label}
                  </span>
                </span>
                <span
                  className="text-[10px] transition-transform"
                  style={{
                    color: 'var(--color-sparc-line)',
                    transform: isCollapsed ? 'rotate(-90deg)' : 'rotate(0deg)',
                  }}
                >
                  ▾
                </span>
              </button>

              {/* Section pages */}
              {!isCollapsed &&
                section.pages.map((page) => {
                  const idx = pageIndex++;
                  const active = page === currentPage;
                  const disabled = page !== 'Project' && !projectLoaded;
                  const num = String(idx + 1).padStart(2, '0');
                  return (
                    <button
                      key={page}
                      onClick={() => !disabled && onNavigate(page as PageName)}
                      disabled={disabled}
                      title={disabled ? 'Load a project first' : `Go to ${page}`}
                      className="group flex w-full items-center gap-2.5 px-4 py-[5px] text-left transition-colors"
                      style={{
                        color: disabled
                          ? 'var(--color-sparc-line)'
                          : active
                            ? '#fff'
                            : 'var(--color-sparc-ink-2)',
                        background: active ? 'var(--color-sparc-ink)' : 'transparent',
                        borderRadius: active ? 4 : 0,
                        marginLeft: active ? 8 : 0,
                        marginRight: active ? 8 : 0,
                        cursor: disabled ? 'not-allowed' : 'pointer',
                        fontSize: 12.5,
                        fontWeight: active ? 600 : 400,
                      }}
                    >
                      <span
                        className="mono flex items-center justify-center rounded text-[10px] font-bold"
                        style={{
                          width: 22,
                          height: 22,
                          borderRadius: 4,
                          background: disabled
                            ? 'rgba(0,0,0,0.03)'
                            : active
                              ? 'var(--color-sparc-crimson)'
                              : 'rgba(0,0,0,0.05)',
                          color: disabled
                            ? 'var(--color-sparc-line)'
                            : active
                              ? '#fff'
                              : 'var(--color-sparc-muted)',
                        }}
                      >
                        {num}
                      </span>
                      {page}
                    </button>
                  );
                })}

              {isCollapsed && (() => { pageIndex += section.pages.length; return null; })()}
            </div>
          );
        })}
      </nav>

      {/* AI Assistant toggle */}
      <div className="border-t px-2 pt-2" style={{ borderColor: 'var(--color-sparc-line)' }}>
        <button
          onClick={onToggleChat}
          className="flex w-full items-center gap-2.5 rounded-md px-3 py-2 text-left text-[12px] font-semibold transition-colors"
          style={{
            background: chatOpen ? 'var(--color-sparc-ink)' : 'transparent',
            color: chatOpen ? '#fff' : 'var(--color-sparc-ink-2)',
            border: chatOpen ? 'none' : '1px solid var(--color-sparc-line)',
          }}
        >
          <CubeLogo size={20} animate={false} hue={chatOpen ? 'amber' : 'ink'} />
          SPARC Assistant
        </button>
      </div>

      {/* Settings */}
      <div className="border-t p-2" style={{ borderColor: 'var(--color-sparc-line)' }}>
        <button
          onClick={onSettings}
          className="flex w-full items-center gap-2.5 rounded px-3 py-2 text-left text-[12px] transition-colors"
          style={{ color: 'var(--color-sparc-muted)' }}
        >
          <span className="text-xs">⚙</span>
          Settings
        </button>
      </div>
    </aside>
  );
}

export { PAGES };
