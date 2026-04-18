import { useState } from "react";

// ---------------------------------------------------------------------------
// Grouped navigation sections
// ---------------------------------------------------------------------------
const SECTIONS = [
  { label: "Setup",    pages: ["Project", "Data", "Processing", "Config"] },
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
}

export default function Sidebar({
  currentPage,
  onNavigate,
  onSettings,
  onToggleChat,
  chatOpen,
  projectLoaded,
}: SidebarProps) {
  // Track collapsed sections — all expanded by default
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});

  const toggleSection = (label: string) =>
    setCollapsed((prev) => ({ ...prev, [label]: !prev[label] }));

  // Running page counter across all sections for numbered badges
  let pageIndex = 0;

  return (
    <aside className="flex h-full w-52 flex-col border-r border-sparc-gray-200 bg-white">
      {/* Logo lockup: cube icon + wordmark */}
      <div className="flex items-center gap-3 border-b border-sparc-gray-200 px-4 py-4">
        <img src="/logo.svg" alt="SPARC" className="h-10 w-10" />
        <div className="flex flex-col leading-none">
          <span className="text-sm font-bold tracking-tight">SPARC</span>
          <span className="text-[10px] font-medium tracking-widest text-sparc-gray-600 uppercase">Labs</span>
        </div>
      </div>

      {/* No-project badge */}
      {!projectLoaded && (
        <div className="mx-3 mt-2 rounded bg-sparc-amber/15 px-3 py-1.5 text-center text-[10px] font-medium text-sparc-orange">
          No project loaded
        </div>
      )}

      {/* Sectioned navigation */}
      <nav className="flex-1 overflow-y-auto py-1">
        {SECTIONS.map((section, sectionIdx) => {
          const isCollapsed = collapsed[section.label] ?? false;
          // Check if any page in this section is active
          const sectionActive = (section.pages as readonly string[]).includes(currentPage);

          // Snapshot the starting index for this section
          const sectionStartIdx = pageIndex;

          return (
            <div key={section.label} className={sectionIdx > 0 ? "mt-1" : ""}>
              {/* Section header — click to collapse/expand */}
              <button
                onClick={() => toggleSection(section.label)}
                className="flex w-full items-center justify-between px-4 py-1.5 group"
              >
                <span
                  className={`text-[10px] font-semibold uppercase tracking-wider ${
                    sectionActive ? "text-sparc-purple" : "text-sparc-gray-400"
                  }`}
                >
                  {section.label}
                </span>
                <span className="text-[10px] text-sparc-gray-400 group-hover:text-sparc-gray-600 transition-transform"
                  style={{ transform: isCollapsed ? "rotate(-90deg)" : "rotate(0deg)" }}
                >
                  ▾
                </span>
              </button>

              {/* Section pages */}
              {!isCollapsed &&
                section.pages.map((page) => {
                  const idx = pageIndex++;
                  // only used for the badge number
                  void sectionStartIdx;
                  const active = page === currentPage;
                  const disabled = page !== "Project" && !projectLoaded;
                  return (
                    <button
                      key={page}
                      onClick={() => !disabled && onNavigate(page as PageName)}
                      disabled={disabled}
                      title={disabled ? "Load a project first from the Project page" : `Go to ${page}`}
                      className={`group flex w-full items-center gap-2.5 px-4 py-1.5 text-left text-sm transition-colors ${
                        disabled
                          ? "text-sparc-gray-400 cursor-not-allowed"
                          : active
                            ? "bg-black text-white font-medium"
                            : "text-sparc-gray-600 hover:bg-sparc-gray-100 hover:text-black"
                      }`}
                    >
                      <span
                        className={`flex h-5 w-5 items-center justify-center rounded text-[10px] font-medium tabular-nums ${
                          disabled
                            ? "bg-sparc-gray-100 text-sparc-gray-400"
                            : active
                              ? "bg-sparc-purple text-white"
                              : "bg-sparc-gray-200 text-sparc-gray-600 group-hover:bg-sparc-gray-300"
                        }`}
                      >
                        {idx + 1}
                      </span>
                      {page}
                    </button>
                  );
                })}

              {/* Bump counter even when collapsed so numbers stay consistent */}
              {isCollapsed && (() => { pageIndex += section.pages.length; return null; })()}
            </div>
          );
        })}
      </nav>

      {/* AI Assistant toggle — above Settings */}
      <div className="border-t border-sparc-gray-200 px-2 pt-2">
        <button
          onClick={onToggleChat}
          className={`flex w-full items-center gap-2.5 rounded px-4 py-2 text-left text-sm transition-colors ${
            chatOpen
              ? "bg-sparc-purple text-white font-medium"
              : "text-sparc-gray-600 hover:bg-sparc-gray-100 hover:text-black"
          }`}
        >
          <span className="flex h-5 w-5 items-center justify-center text-xs">💬</span>
          AI Assistant
        </button>
      </div>

      {/* Settings link at bottom */}
      <div className="border-t border-sparc-gray-200 p-2">
        <button
          onClick={onSettings}
          className="flex w-full items-center gap-2.5 rounded px-4 py-2 text-left text-sm text-sparc-gray-600 hover:bg-sparc-gray-100 hover:text-black"
        >
          <span className="flex h-5 w-5 items-center justify-center text-xs">⚙</span>
          Settings
        </button>
      </div>
    </aside>
  );
}

export { PAGES };
