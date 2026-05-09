import { create } from "zustand";
import { loadProject as apiLoadProject, health } from "@/lib/api";
import { addRecent } from "@/lib/recentProjects";
import { useResultCacheStore } from "./resultCache";
import { resetManifest } from "@/hooks/useManifest";

const STORAGE_KEY = "sparc-current-project";

interface ProjectState {
  /** Absolute path to loaded project.yml, or null. */
  projectPath: string | null;
  /** True once the server has the project loaded. */
  projectLoaded: boolean;
  /** True while the store is auto-rehydrating from localStorage on startup. */
  rehydrating: boolean;

  /**
   * Call once when the sidecar becomes ready (serverReady → true).
   * Checks if the server already has a project loaded; if not, attempts to
   * reload from persisted localStorage path.
   */
  rehydrate: () => Promise<void>;

  /** Load (or reload) a project and persist the path. */
  openProject: (
    path: string,
    meta?: { name?: string; template?: string },
  ) => Promise<void>;

  /** Unload the current project from local state. */
  clearProject: () => void;
}

export const useProjectStore = create<ProjectState>((set, get) => ({
  projectPath: localStorage.getItem(STORAGE_KEY),
  projectLoaded: false,
  rehydrating: true,

  rehydrate: async () => {
    // Guard: only run once
    if (!get().rehydrating) return;
    try {
      const h = await health();
      if (h.project_loaded) {
        set({ projectLoaded: true, rehydrating: false });
        return;
      }
      const { projectPath } = get();
      if (projectPath) {
        await apiLoadProject(projectPath);
        set({ projectLoaded: true });
      }
    } catch {
      // Server unreachable or project file gone — clear stale path
      localStorage.removeItem(STORAGE_KEY);
      set({ projectPath: null, projectLoaded: false });
    } finally {
      set({ rehydrating: false });
    }
  },

  openProject: async (path, meta) => {
    useResultCacheStore.getState().clear();
    resetManifest();
    await apiLoadProject(path);
    localStorage.setItem(STORAGE_KEY, path);
    set({ projectPath: path, projectLoaded: true });
    addRecent({
      path,
      name: meta?.name ?? path.split(/[/\\]/).slice(-2, -1)[0] ?? "Project",
      template: meta?.template ?? "",
    });
  },

  clearProject: () => {
    useResultCacheStore.getState().clear();
    resetManifest();
    localStorage.removeItem(STORAGE_KEY);
    set({ projectPath: null, projectLoaded: false });
  },
}));
