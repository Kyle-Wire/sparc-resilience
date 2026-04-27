import { useState, useCallback, useRef } from "react";
import { withWsAuth } from "@/lib/api";
import type { PipelineEvent } from "@/lib/types";

/**
 * Opens a WebSocket to /run/stream and yields structured pipeline events.
 */
export function usePipelineStream() {
  const [events, setEvents] = useState<PipelineEvent[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const startStage = useCallback(
    (stage: number, opts: { fast?: boolean; skip_gwen?: boolean } = {}) => {
      setEvents([]);
      setError(null);
      setIsRunning(true);

      const ws = new WebSocket(withWsAuth("ws://127.0.0.1:8008/run/stream"));
      wsRef.current = ws;

      ws.onopen = () => {
        ws.send(JSON.stringify({ stage, fast: opts.fast ?? false, skip_gwen: opts.skip_gwen ?? false }));
      };

      ws.onmessage = (msg) => {
        const event: PipelineEvent = JSON.parse(msg.data);
        setEvents((prev) => [...prev, event]);

        if (event.type === "complete" || event.type === "error") {
          setIsRunning(false);
          if (event.type === "error") setError(event.message ?? "Unknown error");
          ws.close();
        }
      };

      ws.onerror = () => {
        setError("WebSocket connection failed");
        setIsRunning(false);
      };

      ws.onclose = () => {
        setIsRunning(false);
      };
    },
    [],
  );

  const cancel = useCallback(() => {
    wsRef.current?.close();
    setIsRunning(false);
  }, []);

  return { events, isRunning, error, startStage, cancel };
}
