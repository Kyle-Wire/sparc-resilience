import { useState, useEffect, useCallback, useRef } from "react";
import { health } from "@/lib/api";
import type { HealthResponse } from "@/lib/types";

const FAST_MS = 500;   // poll interval before server is ready
const SLOW_MS = 30_000; // poll interval after server is ready (mid-session loss detection)

/**
 * Polls the FastAPI server's /health endpoint until it responds, then
 * continues polling at 30 s to detect mid-session server loss.
 *
 * Returns:
 *   ready      — false until first successful poll
 *   serverLost — true when a poll fails AFTER the server was already ready
 *   status     — last known HealthResponse
 */
export function useServer() {
  const [ready, setReady] = useState(false);
  const [serverLost, setServerLost] = useState(false);
  const [status, setStatus] = useState<HealthResponse | null>(null);
  const readyRef = useRef(false);
  const intervalRef = useRef<ReturnType<typeof setInterval>>(undefined);

  const poll = useCallback(async () => {
    try {
      const s = await health();
      setStatus(s);
      if (!readyRef.current) {
        readyRef.current = true;
        setReady(true);
        // Switch from fast to slow polling
        if (intervalRef.current) clearInterval(intervalRef.current);
        intervalRef.current = setInterval(poll, SLOW_MS);
      }
      // Recover from loss
      setServerLost(false);
    } catch {
      if (readyRef.current) {
        // Server was ready but now fails — signal loss
        setServerLost(true);
      }
    }
  }, []);

  useEffect(() => {
    poll();
    intervalRef.current = setInterval(poll, FAST_MS);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
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

  return { ready, serverLost, status, refresh };
}
