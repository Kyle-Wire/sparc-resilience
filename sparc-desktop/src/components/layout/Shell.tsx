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
  children: ReactNode;
}

export default function Shell({ currentPage, onNavigate, onSettings, onToggleChat, chatOpen, status, projectLoaded, children }: ShellProps) {
  return (
    <div className="flex h-screen w-screen overflow-hidden bg-sparc-gray-100">
      <Sidebar currentPage={currentPage} onNavigate={onNavigate} onSettings={onSettings} onToggleChat={onToggleChat} chatOpen={chatOpen} projectLoaded={projectLoaded} />
      <div className="flex flex-1 flex-col">
        <Topbar status={status} />
        <main className="relative flex-1 overflow-auto bg-white p-6 grid-paper">{children}</main>
      </div>
    </div>
  );
}
