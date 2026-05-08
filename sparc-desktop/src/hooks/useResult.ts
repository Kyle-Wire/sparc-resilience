/**
 * useResult<T>(key, fetcher) — fetch-with-cache hook.
 *
 * On first call for a given `key` the fetcher runs and the result is
 * stored in the global Zustand cache.  Subsequent renders (including
 * after React Router navigation re-mounts the component) read straight
 * from the cache and return instantly with no network request.
 *
 * Invalidation flow:
 *   1. WebSocket "artifact_written" event → useArtifactStream calls
 *      `invalidateStage(stage)` on the store.
 *   2. The Zustand store evicts all keys with the matching prefix.
 *   3. `entry` becomes `undefined` → this hook re-fires the fetcher.
 *
 * Usage:
 *   const { data, loading, error } = useResult<CorrelogramData>(
 *     present ? "s0:correlogram" : null,
 *     getCorrelogramData,
 *   );
 *
 * Pass `null` as `key` to suppress the fetch (e.g. when the artifact
 * isn't ready yet). The hook returns `{ data: null, loading: false }`.
 */
import { useEffect, useRef, useState } from "react";
import { useResultCacheStore } from "@/stores/resultCache";

export interface UseResultReturn<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

export function useResult<T>(
  key: string | null,
  fetcher: () => Promise<T>,
): UseResultReturn<T> {
  // Reactive subscription to the specific cache slot.
  // Zustand re-renders the component whenever the slot changes.
  const entry = useResultCacheStore((s) => (key ? s.cache[key] : undefined));
  const setCache = useResultCacheStore((s) => s.set);

  const [loading, setLoading] = useState<boolean>(
    () => entry === undefined && key !== null,
  );
  const [error, setError] = useState<string | null>(null);
  // Track which key is currently in-flight to avoid stale responses.
  const inflightRef = useRef<string | null>(null);

  useEffect(() => {
    if (key === null) {
      setLoading(false);
      setError(null);
      inflightRef.current = null;
      return;
    }

    if (entry !== undefined) {
      // Cache hit — data is already in the store, nothing to do.
      setLoading(false);
      setError(null);
      return;
    }

    // Deduplicate: don't fire again if we're already fetching this key.
    if (inflightRef.current === key) return;

    inflightRef.current = key;
    setLoading(true);
    setError(null);
    let cancelled = false;

    fetcher()
      .then((data) => {
        if (cancelled) return;
        setCache(key, data);
        setLoading(false);
        inflightRef.current = null;
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "request failed");
        setLoading(false);
        inflightRef.current = null;
      });

    return () => {
      cancelled = true;
      // Don't null out inflightRef here: we want the dedup guard to stay
      // until the response returns (even if the component unmounts), so
      // the response is still written to the cache for the next mount.
    };
  }, [key, entry]); // eslint-disable-line react-hooks/exhaustive-deps

  return {
    data: (entry?.data as T) ?? null,
    loading,
    error,
  };
}
