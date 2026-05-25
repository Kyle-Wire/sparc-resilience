import { useState, useEffect, useCallback, useRef } from "react";
import { health } from "@/lib/api";
import type { HealthResponse } from "@/lib/types";

const FAST_MS = 500;    // poll interval before server is ready
const SLOW_MS = 30_000; // poll interval after server is ready (mid-session loss detection)
const STARTUP_TIMEOUT_MS = 15_000; // max wait before showing "failed" splash
const MAX_RESPAWN_ATTEMPTS = 3;    // auto-respawn retries on mid-session loss

/** Invoke the Tauri restart_sidecar command. No-op if Tauri isn't available (browser dev). */
async function invokeRestartSidecar(): Promise<void> {
  try {
    const { invoke } = await import("@tauri-apps/api/core");
    await invoke("restart_sidecar");
  } catch {
    // Running in browser/dev without Tauri — ignore.
  }
}

/**
 * Polls the FastAPI server's /health endpoint until it responds, then
 * continues polling at 30 s to detect mid-session server loss.
 *
 * Returns:
 *   ready        — false until first successful poll
 *   serverLost   — true when a poll fails AFTER the server was already ready
 *                  and all respawn attempts are exhausted
 *   startupFailed — true when /health never responds within STARTUP_TIMEOUT_MS
 *   status       — last known HealthResponse
 *   restart      — call to manually trigger a sidecar restart + reset state
 */
export function useServer() {
  const [ready, setReady] = useState(false);
  const [serverLost, setServerLost] = useState(false);
  const [startupFailed, setStartupFailed] = useState(false);
  const [respawning, setRespawning] = useState(false);
  const [status, setStatus] = useState<HealthResponse | null>(null);
  const readyRef = useRef(false);
  const intervalRef = useRef<ReturnType<typeof setInterval>>(undefined);
  const startupTimerRef = useRef<ReturnType<typeof setTimeout>>(undefined);
  const respawnAttemptsRef = useRef(0);
  const respawningRef = useRef(false);

  const poll = useCallback(async () => {
    try {
      const s = await health();
      setStatus(s);
      if (!readyRef.current) {
        // Cancel the startup failure timer — we made it.
        if (startupTimerRef.current) clearTimeout(startupTimerRef.current);
        readyRef.current = true;
        setReady(true);
        setStartupFailed(false);
        // Switch from fast to slow polling
        if (intervalRef.current) clearInterval(intervalRef.current);
        intervalRef.current = setInterval(poll, SLOW_MS);
      }
      // Recover from a transient loss
      setServerLost(false);
      respawnAttemptsRef.current = 0;
      respawningRef.current = false;
    } catch {
      if (readyRef.current && !respawningRef.current) {
        // Server was previously healthy — try to auto-respawn.
        if (respawnAttemptsRef.current < MAX_RESPAWN_ATTEMPTS) {
          respawningRef.current = true;
          setRespawning(true);
          respawnAttemptsRef.current += 1;
          const delay = Math.min(1000 * respawnAttemptsRef.current, 4000);
          setTimeout(async () => {
            await invokeRestartSidecar();
            respawningRef.current = false;
            setRespawning(false);
          }, delay);
        } else {
          // All retries exhausted — surface the error to the user.
          setServerLost(true);
        }
      }
    }
  }, []);

  const restart = useCallback(async () => {
    // Reset all state and try again from scratch.
    if (intervalRef.current) clearInterval(intervalRef.current);
    if (startupTimerRef.current) clearTimeout(startupTimerRef.current);
    readyRef.current = false;
    respawnAttemptsRef.current = 0;
    respawningRef.current = false;
    setReady(false);
    setServerLost(false);
    setStartupFailed(false);
    setRespawning(false);

    await invokeRestartSidecar();

    // Resume fast polling.
    intervalRef.current = setInterval(poll, FAST_MS);
    // Re-arm the startup timeout.
    startupTimerRef.current = setTimeout(() => {
      if (!readyRef.current) setStartupFailed(true);
    }, STARTUP_TIMEOUT_MS);
  }, [poll]);

  useEffect(() => {
    poll();
    intervalRef.current = setInterval(poll, FAST_MS);
    // Startup timeout: if we haven't heard from the sidecar in 15 s, show error.
    startupTimerRef.current = setTimeout(() => {
      if (!readyRef.current) setStartupFailed(true);
    }, STARTUP_TIMEOUT_MS);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (startupTimerRef.current) clearTimeout(startupTimerRef.current);
    };
  }, [poll]);

  const refresh = useCallback(async () => {
    try {
      const s = await health();
      setStatus(s);
      setServerLost(false);
    } catch {
      if (readyRef.current) setServerLost(true);
    }
  }, []);

  return { ready, serverLost, startupFailed, respawning, status, refresh, restart };
}
