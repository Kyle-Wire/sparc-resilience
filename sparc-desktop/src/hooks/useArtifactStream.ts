/**
 * Singleton WebSocket subscription to the SPARC server's `/run/stream` in
 * artifact-events mode. The server pushes one ``artifact_written`` message
 * every time ``RunRegistry.register_artifact`` is called, so the desktop can
 * invalidate caches and refresh visualizations the moment new data lands in
 * the database — instead of waiting for the next 2s manifest poll.
 *
 * Usage:
 *   const lastEvent = useArtifactStream(); // global last event
 *   useArtifactEvent((evt) => { ... });    // subscribe a callback
 *
 * The WebSocket is shared across all hook consumers; reconnect uses
 * exponential backoff (1s, 2s, 4s ... capped at 30s). The 2s/30s manifest
 * polling in `useManifest` remains as a backstop.
 */
import { useEffect, useRef, useState } from "react";
import { useResultCacheStore } from "@/stores/resultCache";
import { WS_ORIGIN } from "@/lib/server";
import { getToken } from "@/stores/tokenStore";

export interface ArtifactWrittenEvent {
  type: "artifact_written";
  stage: string;
  artifact_id: string;
  kind?: string | null;
  format?: string | null;
  size_bytes?: number | null;
  row_count?: number | null;
  content_hash?: string | null;
  written_at?: string | null;
}

export type ConnectionStatus = "idle" | "connecting" | "open" | "closed" | "error";

type Listener = (event: ArtifactWrittenEvent) => void;

const MAX_BACKOFF_MS = 30_000;
const INITIAL_BACKOFF_MS = 1_000;

class ArtifactStreamClient {
  private ws: WebSocket | null = null;
  private listeners = new Set<Listener>();
  private statusListeners = new Set<(s: ConnectionStatus) => void>();
  private backoff = INITIAL_BACKOFF_MS;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private status: ConnectionStatus = "idle";
  private lastEvent: ArtifactWrittenEvent | null = null;
  private started = false;

  start() {
    if (this.started) return;
    this.started = true;
    this.connect();
  }

  private setStatus(s: ConnectionStatus) {
    this.status = s;
    this.statusListeners.forEach((l) => {
      try { l(s); } catch { /* ignore */ }
    });
  }

  getStatus(): ConnectionStatus { return this.status; }
  getLastEvent(): ArtifactWrittenEvent | null { return this.lastEvent; }

  private connect() {
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }
    this.setStatus("connecting");
    let ws: WebSocket;
    try {
      ws = new WebSocket(`${WS_ORIGIN}/run/stream?token=${encodeURIComponent(getToken())}`);
    } catch {
      this.scheduleReconnect();
      return;
    }
    this.ws = ws;

    ws.onopen = () => {
      this.backoff = INITIAL_BACKOFF_MS;
      this.setStatus("open");
      try {
        ws.send(JSON.stringify({ subscribe: "artifacts" }));
      } catch { /* ignore */ }
    };

    ws.onmessage = (msg) => {
      let parsed: { type?: string } & Partial<ArtifactWrittenEvent>;
      try {
        parsed = JSON.parse(msg.data);
      } catch {
        return;
      }
      if (parsed.type !== "artifact_written") return;
      const evt = parsed as ArtifactWrittenEvent;
      this.lastEvent = evt;
      this.listeners.forEach((l) => {
        try { l(evt); } catch { /* ignore */ }
      });
    };

    ws.onerror = () => {
      // Don't surface "error" as a visible state — the reconnect path will
      // handle it. Surfacing it causes a brief red/grey flicker in the UI
      // every time the socket churns.
    };

    ws.onclose = () => {
      this.ws = null;
      // If we were previously open, jump straight to "connecting" instead
      // of "closed" so the UI pill doesn't flicker between states during
      // an immediate reconnect. We only show "closed" if no reconnect is
      // pending (e.g. tab is unloading).
      this.setStatus("connecting");
      this.scheduleReconnect();
    };
  }

  private scheduleReconnect() {
    if (this.reconnectTimer) return;
    const delay = this.backoff;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.backoff = Math.min(this.backoff * 2, MAX_BACKOFF_MS);
      this.connect();
    }, delay);
  }

  addListener(l: Listener): () => void {
    this.listeners.add(l);
    return () => this.listeners.delete(l);
  }

  addStatusListener(l: (s: ConnectionStatus) => void): () => void {
    this.statusListeners.add(l);
    // Push current status immediately so subscribers don't have to wait.
    try { l(this.status); } catch { /* ignore */ }
    return () => this.statusListeners.delete(l);
  }
}

const client = new ArtifactStreamClient();
// Begin connecting as soon as this module is imported.
if (typeof window !== "undefined") {
  client.start();
  // Invalidate the specific cached artifact the moment an artifact_written
  // event arrives.  Artifact-scoped invalidation means unrelated panels on
  // the same stage are not forced to refetch.
  client.addListener((evt) => {
    useResultCacheStore.getState().invalidateKey(`s${evt.stage}:${evt.artifact_id}`);
  });
}

/**
 * Register a callback that fires whenever any artifact_written event is
 * received. Used by the useManifest singleton to debounce-refetch the
 * manifest without needing a React component alive.
 *
 * Returns an unsubscribe function.
 */
export function addManifestArtifactListener(fn: () => void): () => void {
  // Wrap so we only fire on artifact_written, not other event types.
  const wrapped: Listener = () => fn();
  return client.addListener(wrapped);
}

/** Subscribe a callback to ``artifact_written`` events. */
export function useArtifactEvent(handler: Listener): void {
  const ref = useRef(handler);
  ref.current = handler;
  useEffect(() => {
    const wrapped: Listener = (e) => ref.current(e);
    return client.addListener(wrapped);
  }, []);
}

/** Subscribe to a specific (stage, artifact_id) pair. */
export function useArtifactWritten(
  stage: string | number,
  artifactId: string,
  handler: (event: ArtifactWrittenEvent) => void,
): void {
  const ref = useRef(handler);
  ref.current = handler;
  const stageStr = String(stage);
  useEffect(() => {
    return client.addListener((e) => {
      if (e.stage === stageStr && e.artifact_id === artifactId) {
        ref.current(e);
      }
    });
  }, [stageStr, artifactId]);
}

/** Returns the most recent artifact event seen on the singleton stream. */
export function useArtifactStream(): {
  lastEvent: ArtifactWrittenEvent | null;
  status: ConnectionStatus;
} {
  const [lastEvent, setLastEvent] = useState<ArtifactWrittenEvent | null>(client.getLastEvent());
  const [status, setStatus] = useState<ConnectionStatus>(client.getStatus());
  useEffect(() => {
    const off1 = client.addListener((e) => setLastEvent(e));
    const off2 = client.addStatusListener((s) => setStatus(s));
    return () => { off1(); off2(); };
  }, []);
  return { lastEvent, status };
}
