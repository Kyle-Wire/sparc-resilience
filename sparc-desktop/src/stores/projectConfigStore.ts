/**
 * projectConfigStore — single Zustand seam for ProjectConfig.
 *
 * All reads and writes to the project.yml configuration flow through this
 * store. Callers never call `getConfig()` directly; they read from
 * `useProjectConfigStore` and mutate via `updateConfig` / `saveNow`.
 *
 * This eliminates the 4+ independent `getConfig()` fetches that previously
 * fired on mount across useProjectConfig, useProjectContext, DataPage, and
 * VariablesPage — the server is now hit exactly once per project load.
 */
import { create } from "zustand";
import { getConfig, saveConfig } from "@/lib/api";
import type { ProjectConfig } from "@/lib/types";

interface ProjectConfigState {
  config: ProjectConfig | null;
  loading: boolean;
  error: string | null;

  /**
   * Fetch config from the server and cache it.
   * No-ops if config is already loaded, unless `force: true` is passed.
   */
  fetchConfig: (opts?: { force?: boolean }) => Promise<void>;

  /**
   * Optimistic in-memory patch + debounced server save.
   * Callers should use this for interactive edits (form fields etc.).
   */
  updateConfig: (patch: Partial<ProjectConfig>) => void;

  /**
   * Patch in-memory and persist to the server immediately (no debounce).
   */
  saveNow: (patch: Partial<ProjectConfig>) => Promise<void>;

  /**
   * Clear cached config (call on project unload).
   */
  reset: () => void;
}

let _debounceTimer: ReturnType<typeof setTimeout> | null = null;

export const useProjectConfigStore = create<ProjectConfigState>((set, get) => ({
  config: null,
  loading: false,
  error: null,

  fetchConfig: async ({ force = false } = {}) => {
    const { config } = get();
    if (config !== null && !force) return; // already loaded, skip

    set({ loading: true, error: null });
    try {
      const c = await getConfig();
      set({ config: c, loading: false });
    } catch (err) {
      set({
        loading: false,
        error: err instanceof Error ? err.message : String(err),
      });
    }
  },

  updateConfig: (patch) => {
    set((s) => ({
      config: s.config ? { ...s.config, ...patch } : (patch as ProjectConfig),
    }));

    if (_debounceTimer !== null) clearTimeout(_debounceTimer);
    _debounceTimer = setTimeout(async () => {
      _debounceTimer = null;
      try {
        await saveConfig(patch);
      } catch {
        // Debounced save failure is best-effort; callers can use saveNow for
        // critical writes that need error feedback.
      }
    }, 300);
  },

  saveNow: async (patch) => {
    set((s) => ({
      config: s.config ? { ...s.config, ...patch } : (patch as ProjectConfig),
    }));
    await saveConfig(patch);
  },

  reset: () => {
    if (_debounceTimer !== null) {
      clearTimeout(_debounceTimer);
      _debounceTimer = null;
    }
    set({ config: null, loading: false, error: null });
  },
}));
