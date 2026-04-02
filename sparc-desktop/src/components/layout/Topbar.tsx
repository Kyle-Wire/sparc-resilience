import type { ReactNode } from "react";
import type { HealthResponse } from "@/lib/types";

interface TopbarProps {
  status: HealthResponse | null;
  children?: ReactNode;
}

export default function Topbar({ status, children }: TopbarProps) {
  return (
    <header className="flex h-10 items-center justify-between border-b border-sparc-gray-200 bg-white px-4">
      <div className="flex items-center gap-3 text-xs">
        <span className={`inline-flex items-center gap-1.5 ${
          status?.project_loaded ? "text-sparc-gray-800" : "text-sparc-gray-600"
        }`}>
          <span className={`h-1.5 w-1.5 rounded-full ${
            status?.project_loaded ? "bg-green-500" : "bg-sparc-gray-300"
          }`} />
          {status?.project_loaded ? "Project loaded" : "No project"}
        </span>
        {status?.is_running && (
          <span className="inline-flex items-center gap-1.5 font-medium text-sparc-crimson">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-sparc-crimson" />
            Stage {status.current_stage}
          </span>
        )}
      </div>
      <div className="flex items-center gap-2">{children}</div>
    </header>
  );
}
