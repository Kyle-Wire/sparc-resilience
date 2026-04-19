import type { ReactNode } from "react";
import Sidebar, { type PageName } from "./Sidebar";
import Topbar from "./Topbar";
import type { HealthResponse } from "@/lib/types";

interface ShellProps {
  currentPage: PageName;
  onNavigate: (page: PageName) => void;
  onSettings?: () => void;
  onToggleChat?: () => void;
  chatOpen?: boolean;
  status: HealthResponse | null;
  projectLoaded?: boolean;
  projectName?: string;
  projectEpsg?: string;
  projectDomain?: string;
  children: ReactNode;
}

export default function Shell({
  currentPage,
  onNavigate,
  onToggleChat,
  chatOpen,
  status,
  projectLoaded,
  projectName,
  projectEpsg,
  projectDomain,
  children,
}: ShellProps) {
  return (
    <div className="flex h-screen w-screen overflow-hidden" style={{ background: 'var(--color-sparc-paper)' }}>
      <Sidebar
        currentPage={currentPage}
        onNavigate={onNavigate}
        onToggleChat={onToggleChat}
        chatOpen={chatOpen}
        projectLoaded={projectLoaded}
        projectName={projectName}
        projectEpsg={projectEpsg}
        projectDomain={projectDomain}
      />
      <div className="flex flex-1 flex-col min-w-0" style={{ position: 'relative' }}>
        <Topbar status={status} currentPage={currentPage} />
        <main className="scroll relative flex-1 overflow-auto grid-paper" style={{ padding: '20px 22px' }}>
          {children}
        </main>
      </div>
    </div>
  );
}
