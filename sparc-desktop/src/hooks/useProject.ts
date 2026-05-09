import { useState, useEffect, useCallback, useRef } from "react";
import { loadProject as apiLoadProject, health } from "@/lib/api";
import { addRecent } from "@/lib/recentProjects";
import { useResultCacheStore } from "@/stores/resultCache";
import { resetManifest } from "@/hooks/useManifest";

const STORAGE_KEY = "sparc-current-project";

interface UseProjectReturn {
  /** Absolute path to loaded project.yml, or null. */
  projectPath: string | null;
  /** True once the server has the project loaded. */
  projectLoaded: boolean;
  /** True while the hook is auto-rehydrating from localStorage on startup. */
  rehydrating: boolean;
  /** Load (or reload) a project and persist the path. */
  openProject: (
    path: string,
    meta?: { name?: string; template?: string },
  ) => Promise<void>;
  /** Unload the current project from local state (server state stays until next load). */
  clearProject: () => void;
}

export function useProject(serverReady: boolean): UseProjectReturn {
  const [projectPath, setProjectPath] = useState<string | null>(
    () => localStorage.getItem(STORAGE_KEY),
  );
  const [projectLoaded, setProjectLoaded] = useState(false);
  const [rehydrating, setRehydrating] = useState(true);
  const didRehydrate = useRef(false);

  // Persist path changes to localStorage
  useEffect(() => {
    if (projectPath) {
      localStorage.setItem(STORAGE_KEY, projectPath);
    } else {
      localStorage.removeItem(STORAGE_KEY);
    }
  }, [projectPath]);

  // Auto-rehydrate: once server is ready, check if it already has the project
  // loaded (e.g. server didn't restart) or re-load from our stored path.
  useEffect(() => {
    if (!serverReady || didRehydrate.current) return;
    didRehydrate.current = true;

    (async () => {
      try {
        const h = await health();
        if (h.project_loaded) {
          // Server already has a project — trust it
          setProjectLoaded(true);
          setRehydrating(false);
          return;
        }

        // Server has no project but we remember one — reload it
        if (projectPath) {
          await apiLoadProject(projectPath);
          setProjectLoaded(true);
        }
      } catch {
        // Server unreachable or project file gone — clear stale path
        setProjectPath(null);
        setProjectLoaded(false);
      } finally {
        setRehydrating(false);
      }
    })();
  }, [serverReady]); // eslint-disable-line react-hooks/exhaustive-deps

  const openProject = useCallback(
    async (path: string, meta?: { name?: string; template?: string }) => {
      // Flush stale results from previous project before loading new one.
      useResultCacheStore.getState().clear();
      resetManifest();
      await apiLoadProject(path);
      setProjectPath(path);
      setProjectLoaded(true);
      addRecent({
        path,
        name: meta?.name ?? path.split(/[\/]/).slice(-2, -1)[0] ?? "Project",
        template: meta?.template ?? "",
      });
    },
    [],
  );

  const clearProject = useCallback(() => {
    useResultCacheStore.getState().clear();
    resetManifest();
    setProjectPath(null);
    setProjectLoaded(false);
  }, []);

  return { projectPath, projectLoaded, rehydrating, openProject, clearProject };
}
