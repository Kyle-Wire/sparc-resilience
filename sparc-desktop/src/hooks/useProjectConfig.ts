/**
 * Hook for reading and writing project.yml configuration.
 *
 * Thin adapter over `useProjectConfigStore`. All state lives in the Zustand
 * store so it is shared across every consumer — no redundant `getConfig()`
 * fetches, and mutations are immediately visible app-wide.
 */
import { useEffect, useCallback } from "react";
import { useNotification } from "@/hooks/useNotifications";
import { useProjectConfigStore } from "@/stores/projectConfigStore";
import type { ProjectConfig } from "@/lib/types";

export function useProjectConfig(projectLoaded: boolean) {
  const { config, loading, fetchConfig, updateConfig: storeUpdate,
          saveNow: storeSave, reset } = useProjectConfigStore();
  const { notify } = useNotification();

  // Drive a fetch when the project becomes loaded; clear on unload.
  useEffect(() => {
    if (!projectLoaded) {
      reset();
      return;
    }
    fetchConfig();
  }, [projectLoaded, fetchConfig, reset]);

  // Debounced save with UI notification
  const updateConfig = useCallback(
    (patch: Partial<ProjectConfig>) => {
      storeUpdate(patch);
      // Notification is best-effort; debounced save errors are silently ignored
      // in the store (same behaviour as the old implementation).
    },
    [storeUpdate],
  );

  // Immediate save with UI notification
  const saveNow = useCallback(
    async (patch: Partial<ProjectConfig>) => {
      try {
        await storeSave(patch);
        notify("success", "Project configuration saved");
      } catch (e) {
        notify("error", e instanceof Error ? e.message : "Failed to save config");
      }
    },
    [storeSave, notify],
  );

  // Reload (force re-fetch)
  const reload = useCallback(async () => {
    await fetchConfig({ force: true });
  }, [fetchConfig]);

  return { config, loading, updateConfig, saveNow, reload };
}
