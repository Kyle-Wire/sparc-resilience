import { useCallback, useEffect, useState } from "react";
import {
  getHardwareProfile,
  type HardwareProfileResponse,
} from "@/lib/api";

/**
 * Fetches the merged hardware profile (detected + global prefs + project overrides + effective).
 * Re-fetch with `refresh()` after mutating preferences or loading a different project.
 */
export function useHardwareProfile(enabled: boolean = true) {
  const [data, setData] = useState<HardwareProfileResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await getHardwareProfile();
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (enabled) refresh();
  }, [enabled, refresh]);

  return { data, loading, error, refresh };
}
