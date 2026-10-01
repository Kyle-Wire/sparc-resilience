// Wraps every /p/:pid/* page: sets the shell's project context and provides the project
// (GET /api/projects/{pid}) to the page through useProjectContext().
import { createContext, useContext, useEffect, type ReactNode } from "react";
import { EmptyState } from "../components/ui/EmptyState";
import { useUi } from "../stores/ui";
import { useProject, type ProjectDetail } from "./resources";

export type ProjectContextValue = { pid: string; detail: ProjectDetail | undefined; loading: boolean; reload: () => Promise<void> };

const ProjectCtx = createContext<ProjectContextValue | null>(null);

/** The current project inside a ProjectLayout (null elsewhere). */
export function useProjectContext(): ProjectContextValue | null {
  return useContext(ProjectCtx);
}

export function ProjectLayout({ pid, children }: { pid: string; children: ReactNode }) {
  const setContext = useUi((s) => s.setContext);
  const res = useProject(pid);
  useEffect(() => setContext({ projectId: pid, runId: null }), [pid, setContext]);
  if (res.error && !res.data) return <EmptyState error={res.error} title={res.error && "status" in res.error && res.error.status === 404 ? "Project not found" : undefined} />;
  return <ProjectCtx.Provider value={{ pid, detail: res.data, loading: res.loading, reload: res.reload }}>{children}</ProjectCtx.Provider>;
}
