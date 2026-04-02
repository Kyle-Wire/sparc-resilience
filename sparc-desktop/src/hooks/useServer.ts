import { useState, useEffect, useCallback, useRef } from "react";
import { health } from "@/lib/api";
import type { HealthResponse } from "@/lib/types";

/**
 * Polls the FastAPI server's /health endpoint until it responds.
 * Returns { ready, status } — `ready` is false while the splash screen should show.
 */
export function useServer() {
  const [ready, setReady] = useState(false);
  const [status, setStatus] = useState<HealthResponse | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval>>(undefined);

  const poll = useCallback(async () => {
    try {
      const s = await health();
      setStatus(s);
      setReady(true);
      if (intervalRef.current) clearInterval(intervalRef.current);
    } catch {
      // server not up yet
    }
  }, []);

  useEffect(() => {
    poll();
    intervalRef.current = setInterval(poll, 500);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [poll]);

  const refresh = useCallback(async () => {
    try {
      const s = await health();
      setStatus(s);
    } catch {
      // ignore
    }
  }, []);

  return { ready, status, refresh };
}
