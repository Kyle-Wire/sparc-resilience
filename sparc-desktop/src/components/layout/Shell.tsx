import type { ReactNode } from "react";
import Sidebar, { type PageName } from "./Sidebar";
import Topbar from "./Topbar";
import type { HealthResponse } from "@/lib/types";

interface ShellProps {
  currentPage: PageName;
  onNavigate: (page: PageName) => void;
  onSettings?: () => void;
  status: HealthResponse | null;
  children: ReactNode;
}

export default function Shell({ currentPage, onNavigate, onSettings, status, children }: ShellProps) {
  return (
    <div className="flex h-screen w-screen overflow-hidden bg-sparc-gray-100">
      <Sidebar currentPage={currentPage} onNavigate={onNavigate} onSettings={onSettings} />
      <div className="flex flex-1 flex-col">
        <Topbar status={status} />
        <main className="relative flex-1 overflow-auto bg-white p-6 grid-paper">{children}</main>
      </div>
    </div>
  );
}
