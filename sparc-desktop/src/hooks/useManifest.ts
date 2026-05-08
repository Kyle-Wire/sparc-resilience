/**
 * Polls the SPARC server's `/results/manifest` and exposes the typed
 * `RunManifest`. The poll cadence is adaptive: 2s when the document is
 * focused (active), 30s when hidden (idle). The frontend uses this as the
 * single source of truth for which Results panels have data — every panel
 * gates its fetch on the manifest entry being present.
 *
 * Implementation note: this hook uses a **module-level singleton** so that
 * all consumers (panels, pages, overlays) share one polling loop and one
 * cached manifest value. When a panel unmounts and remounts (e.g. page
 * navigation), it immediately receives the current cached manifest from the
 * module-level state rather than waiting up to 2 seconds for a fresh poll.
 * This eliminates the blank / loading flash that previously appeared every
 * time the user navigated back to the Insights page.
 */
import { useCallback, useEffect, useState } from "react";
import type { ArtifactEntry, RunManifest, StageManifest } from "../lib/types";
import { getManifest, rescanManifest } from "../lib/api";
import { addManifestArtifactListener } from "./useArtifactStream";

const ACTIVE_INTERVAL_MS = 2_000;
const IDLE_INTERVAL_MS = 30_000;
/** Debounce window for artifact_written burst events before re-fetching. */
const ARTIFACT_BURST_MS = 250;

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

// ---------------------------------------------------------------------------
// Module-level singleton state — survives component unmount/remount cycles.
// ---------------------------------------------------------------------------

interface SharedState {
  manifest: RunManifest | null;
  loading: boolean;
  error: string | null;
  lastUpdated: Date | null;
}

let _state: SharedState = {
  manifest: null,
  loading: false,
  error: null,
  lastUpdated: null,
};

type Listener = (s: SharedState) => void;
const _listeners = new Set<Listener>();

function _notify(patch: Partial<SharedState>) {
  _state = { ..._state, ...patch };
  _listeners.forEach((l) => { try { l(_state); } catch { /* ignore */ } });
}

let _inFlight: Promise<void> | null = null;
let _timer: ReturnType<typeof setTimeout> | null = null;
let _burstTimer: ReturnType<typeof setTimeout> | null = null;
let _started = false;

async function _fetchOnce(opts: { rescan?: boolean } = {}): Promise<void> {
  if (_inFlight && !opts.rescan) return;
  const run = async () => {
    _notify({ loading: true });
    try {
      const m = await getManifest({ refresh: true, rescan: opts.rescan });
      _notify({ manifest: m, error: null, lastUpdated: new Date(), loading: false });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      if (msg.startsWith("400")) {
        // "No project loaded" during boot — not an error.
        _notify({ manifest: null, error: null, loading: false });
      } else {
        _notify({ error: msg, loading: false });
      }
    } finally {
      _inFlight = null;
    }
  };
  _inFlight = run();
  return _inFlight;
}

function _scheduleNext() {
  if (_timer) clearTimeout(_timer);
  if (_listeners.size === 0) {
    _timer = null;
    return;
  }
  const delay = (typeof document !== "undefined" && document.hidden)
    ? IDLE_INTERVAL_MS
    : ACTIVE_INTERVAL_MS;
  _timer = setTimeout(async () => {
    await _fetchOnce();
    _scheduleNext();
  }, delay);
}

function _start() {
  if (_started) return;
  _started = true;

  // Kick off the first fetch immediately.
  _fetchOnce().then(_scheduleNext);

  // Resume fast polling when the window regains focus.
  if (typeof document !== "undefined") {
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) return;
      if (_timer) clearTimeout(_timer);
      _fetchOnce().then(_scheduleNext);
    });
  }
}

/** Called by the ArtifactStream singleton whenever an artifact is written. */
export function notifyManifestArtifactWritten() {
  if (_burstTimer) return;
  _burstTimer = setTimeout(() => {
    _burstTimer = null;
    _fetchOnce().then(() => {
      _scheduleNext();
    });
  }, ARTIFACT_BURST_MS);
}

/**
 * Reset the singleton manifest to null and notify all subscribers.
 * Call this when a new project is loaded to clear stale manifest immediately.
 */
export function resetManifest() {
  if (_timer) clearTimeout(_timer);
  _timer = null;
  _inFlight = null;
  _notify({ manifest: null, loading: false, error: null, lastUpdated: null });
  // Restart polling so the new project's manifest is fetched promptly.
  _started = false;
  _start();
}

// Wire to the artifact stream at module load time so artifact events trigger
// a manifest refetch without needing a React component alive.
if (typeof window !== "undefined") {
  addManifestArtifactListener(notifyManifestArtifactWritten);
}

// ---------------------------------------------------------------------------
// React hook — thin subscriber over the module singleton
// ---------------------------------------------------------------------------

export function useManifest(_enabled: boolean = true): UseManifestResult {
  const [localState, setLocalState] = useState<SharedState>(() => _state);

  useEffect(() => {
    // Sync immediately in case state changed between render and effect.
    setLocalState(_state);
    _listeners.add(setLocalState);
    // Ensure the shared polling loop is running.
    _start();
    return () => {
      _listeners.delete(setLocalState);
    };
  }, []);

  const { manifest, loading, error, lastUpdated } = localState;

  const refetch = useCallback(async (opts: { rescan?: boolean } = {}) => {
    if (_timer) clearTimeout(_timer);
    await _fetchOnce(opts);
    _scheduleNext();
  }, []);

  const rescan = useCallback(async () => {
    try { await rescanManifest(); } catch { /* non-fatal */ }
    if (_timer) clearTimeout(_timer);
    await _fetchOnce();
    _scheduleNext();
  }, []);

  const lookup = useCallback(
    (stage: string | number, artifactId: string): ArtifactEntry | null => {
      const sm = manifest?.stages?.[String(stage)];
      return sm?.artifacts?.[artifactId] ?? null;
    },
    [manifest],
  );

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
