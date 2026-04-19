import type { ReactNode } from "react";
import type { HealthResponse } from "@/lib/types";
import Sidebar, { type PageName } from "./Sidebar";
import Topbar from "./Topbar";

interface ShellProps {
  currentPage: PageName;
  onNavigate: (page: PageName | "Settings") => void;
  onToggleChat?: () => void;
  chatOpen?: boolean;
  children: ReactNode;
  projectLoaded?: boolean;
  projectName?: string;
  projectDomain?: string;
  projectEpsg?: string;
  status?: HealthResponse | null;
}

export default function Shell({
  currentPage,
  onNavigate,
  onToggleChat,
  chatOpen,
  children,
  projectLoaded,
  projectName,
  projectDomain,
  projectEpsg,
  status,
}: ShellProps) {
  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        display: "flex",
        background: "var(--paper)",
      }}
    >
      <Sidebar
        currentPage={currentPage}
        onNavigate={onNavigate}
        onToggleChat={onToggleChat}
        chatOpen={chatOpen}
        projectLoaded={projectLoaded}
        projectName={projectName}
        projectDomain={projectDomain}
        projectEpsg={projectEpsg}
      />
      <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>
        <Topbar page={currentPage} status={status} />
        <main
          className="scroll"
          style={{
            flex: 1,
            overflow: "auto",
            padding: "20px 22px",
            background: "var(--paper)",
            backgroundImage:
              "linear-gradient(rgba(0,0,0,0.035) 1px, transparent 1px), linear-gradient(90deg, rgba(0,0,0,0.035) 1px, transparent 1px)",
            backgroundSize: "24px 24px",
            position: "relative",
          }}
        >
          {children}
        </main>
      </div>
    </div>
  );
}
