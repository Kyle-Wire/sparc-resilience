/**
 * Hook for reading and writing project.yml configuration with debounced persistence.
 */
import { useState, useEffect, useCallback, useRef } from "react";
import { getConfig, saveConfig } from "@/lib/api";
import { useNotification } from "@/hooks/useNotifications";
import type { ProjectConfig } from "@/lib/types";

export function useProjectConfig(projectLoaded: boolean) {
  const [config, setConfig] = useState<ProjectConfig | null>(null);
  const [loading, setLoading] = useState(false);
  const { notify } = useNotification();
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Load config when project is loaded
  useEffect(() => {
    if (!projectLoaded) {
      setConfig(null);
      return;
    }
    setLoading(true);
    getConfig()
      .then((c) => setConfig(c))
      .catch(() => {
        /* config not available yet */
      })
      .finally(() => setLoading(false));
  }, [projectLoaded]);

  // Debounced save
  const updateConfig = useCallback(
    (patch: Partial<ProjectConfig>) => {
      setConfig((prev) => (prev ? { ...prev, ...patch } : (patch as ProjectConfig)));

      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(async () => {
        try {
          await saveConfig(patch);
          notify("success", "Project configuration saved");
        } catch (e) {
          notify("error", e instanceof Error ? e.message : "Failed to save config");
        }
      }, 300);
    },
    [notify],
  );

  // Immediate save (no debounce)
  const saveNow = useCallback(
    async (patch: Partial<ProjectConfig>) => {
      setConfig((prev) => (prev ? { ...prev, ...patch } : (patch as ProjectConfig)));
      try {
        await saveConfig(patch);
        notify("success", "Project configuration saved");
      } catch (e) {
        notify("error", e instanceof Error ? e.message : "Failed to save config");
      }
    },
    [notify],
  );

  // Reload config from server
  const reload = useCallback(async () => {
    try {
      const c = await getConfig();
      setConfig(c);
    } catch {
      /* ignore */
    }
  }, []);

  return { config, loading, updateConfig, saveNow, reload };
}
