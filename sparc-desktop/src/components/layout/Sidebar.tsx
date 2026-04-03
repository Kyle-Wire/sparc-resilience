const PAGES = [
  "Project",
  "Data",
  "Variables",
  "CRS",
  "DAG",
  "Physics",
  "Scenarios",
  "Models",
  "Run",
  "Results",
  "Report",
] as const;

export type PageName = (typeof PAGES)[number];

interface SidebarProps {
  currentPage: PageName;
  onNavigate: (page: PageName) => void;
  onSettings?: () => void;
  projectLoaded?: boolean;
}

export default function Sidebar({ currentPage, onNavigate, onSettings, projectLoaded }: SidebarProps) {
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

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto py-2">
        {PAGES.map((page, i) => {
          const active = page === currentPage;
          const disabled = page !== "Project" && !projectLoaded;
          return (
            <button
              key={page}
              onClick={() => !disabled && onNavigate(page)}
              disabled={disabled}
              className={`group flex w-full items-center gap-2.5 px-4 py-2 text-left text-sm transition-colors ${
                disabled
                  ? "text-sparc-gray-400 cursor-not-allowed"
                  : active
                    ? "bg-black text-white font-medium"
                    : "text-sparc-gray-600 hover:bg-sparc-gray-100 hover:text-black"
              }`}
            >
              <span className={`flex h-5 w-5 items-center justify-center rounded text-[10px] font-medium tabular-nums ${
                disabled
                  ? "bg-sparc-gray-100 text-sparc-gray-400"
                  : active
                    ? "bg-sparc-purple text-white"
                    : "bg-sparc-gray-200 text-sparc-gray-600 group-hover:bg-sparc-gray-300"
              }`}>{i + 1}</span>
              {page}
            </button>
          );
        })}
      </nav>

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
