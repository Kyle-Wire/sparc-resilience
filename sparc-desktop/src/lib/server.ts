/**
 * Single source of truth for the SPARC sidecar endpoint.
 *
 * Override at build-time or runtime by setting VITE_SERVER_ORIGIN in .env.local:
 *   VITE_SERVER_ORIGIN=http://127.0.0.1:8008
 */
export const SERVER_ORIGIN: string =
  (import.meta.env.VITE_SERVER_ORIGIN as string | undefined) ?? "http://127.0.0.1:8008";

/** WebSocket origin derived from SERVER_ORIGIN (http → ws, https → wss). */
export const WS_ORIGIN: string = SERVER_ORIGIN.replace(/^http/, "ws");
