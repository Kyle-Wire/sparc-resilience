/**
 * Polls the SPARC server's `/results/manifest` and exposes the typed
 * `RunManifest`. The poll cadence is adaptive: 2s when the document is
 * focused (active), 30s when hidden (idle). The frontend uses this as the
 * single source of truth for which Results panels have data — every panel
 * gates its fetch on the manifest entry being present.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import type { ArtifactEntry, RunManifest, StageManifest } from "../lib/types";
import { getManifest, rescanManifest } from "../lib/api";
import { useArtifactEvent } from "./useArtifactStream";

const ACTIVE_INTERVAL_MS = 2_000;
const IDLE_INTERVAL_MS = 30_000;

export interface UseManifestResult {
  manifest: RunManifest | null;
  loading: boolean;
  error: string | null;
  /** Timestamp of the most recent successful manifest fetch, or null. */
  lastUpdated: Date | null;
  /** True if the project is loaded but the manifest has zero artifacts so
   *  far. Lets the UI distinguish "no project" from "project loaded but
   *  no stages run yet". */
  empty: boolean;
  /** Force a re-fetch (e.g. after a stage completes). */
  refetch: (opts?: { rescan?: boolean }) => Promise<void>;
  /** Trigger the server-side disk walk + re-fetch. */
  rescan: () => Promise<void>;
  /** Lookup helper: returns the artifact entry or `null`. */
  lookup: (stage: string | number, artifactId: string) => ArtifactEntry | null;
  /** Lookup helper: returns the per-stage manifest or `null`. */
  stage: (stage: string | number) => StageManifest | null;
}

export function useManifest(enabled: boolean = true): UseManifestResult {
  const [manifest, setManifest] = useState<RunManifest | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  // Track most recent in-flight request so we can ignore stale responses.
  const reqIdRef = useRef(0);

  const fetchOnce = useCallback(async (opts: { rescan?: boolean } = {}) => {
    if (!enabled) return;
    const myId = ++reqIdRef.current;
    setLoading(true);
    try {
      const m = await getManifest({ refresh: true, rescan: opts.rescan });
      // Drop response if a newer request started after us.
      if (myId !== reqIdRef.current) return;
      setManifest(m);
      setError(null);
      setLastUpdated(new Date());
    } catch (err) {
      if (myId !== reqIdRef.current) return;
      // 400 "No project loaded" is expected during boot; surface as null
      // manifest with no error message rather than a red banner.
      const msg = err instanceof Error ? err.message : String(err);
      if (msg.startsWith("400")) {
        setManifest(null);
        setError(null);
      } else {
        setError(msg);
      }
    } finally {
      if (myId === reqIdRef.current) setLoading(false);
    }
  }, [enabled]);

  const refetch = useCallback(
    (opts: { rescan?: boolean } = {}) => fetchOnce(opts),
    [fetchOnce],
  );

  const rescan = useCallback(async () => {
    if (!enabled) return;
    try {
      await rescanManifest();
    } catch {
      /* server-side rescan failure is non-fatal; refetch will surface state */
    }
    await fetchOnce();
  }, [enabled, fetchOnce]);

  // Adaptive polling: 2s active, 30s when document hidden.
  useEffect(() => {
    if (!enabled) return;
    let timer: ReturnType<typeof setTimeout> | null = null;
    let cancelled = false;

    const tick = async () => {
      if (cancelled) return;
      await fetchOnce();
      if (cancelled) return;
      const interval = document.hidden ? IDLE_INTERVAL_MS : ACTIVE_INTERVAL_MS;
      timer = setTimeout(tick, interval);
    };

    tick();

    const onVisibility = () => {
      if (document.hidden) return;
      // When the user returns to the tab, refresh immediately.
      if (timer) clearTimeout(timer);
      tick();
    };
    document.addEventListener("visibilitychange", onVisibility);

    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [enabled, fetchOnce]);

  const lookup = useCallback(
    (stage: string | number, artifactId: string): ArtifactEntry | null => {
      const sm = manifest?.stages?.[String(stage)];
      return sm?.artifacts?.[artifactId] ?? null;
    },
    [manifest],
  );

  // Coalesce burst-y artifact_written events into a single refetch so we
  // never spam /results/manifest. The 2s/30s polling above remains as a
  // backstop in case the WebSocket is down.
  const burstTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useArtifactEvent(() => {
    if (!enabled) return;
    if (burstTimerRef.current) return;
    burstTimerRef.current = setTimeout(() => {
      burstTimerRef.current = null;
      fetchOnce();
    }, 250);
  });
  useEffect(() => () => {
    if (burstTimerRef.current) clearTimeout(burstTimerRef.current);
  }, []);

  const stage = useCallback(
    (s: string | number): StageManifest | null =>
      manifest?.stages?.[String(s)] ?? null,
    [manifest],
  );

  const empty =
    manifest !== null &&
    Object.values(manifest.stages ?? {}).every(
      (sm) => Object.keys(sm.artifacts ?? {}).length === 0,
    );

  return { manifest, loading, error, lastUpdated, empty, refetch, rescan, lookup, stage };
}
