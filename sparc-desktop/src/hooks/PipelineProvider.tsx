import { createContext, useContext, useState, useCallback, useRef, useEffect } from "react";
import type { ReactNode } from "react";
import type { PipelineEvent } from "@/lib/types";
import { getRunEvents } from "@/lib/api";

export interface PipelineState {
  events: PipelineEvent[];
  isRunning: boolean;
  error: string | null;
  currentStage: number | null;
  startStage: (stage: number, opts?: { fast?: boolean; skip_gwen?: boolean }) => void;
  cancel: () => void;
}

const PipelineContext = createContext<PipelineState | null>(null);

export function usePipeline(): PipelineState {
  const ctx = useContext(PipelineContext);
  if (!ctx) throw new Error("usePipeline must be used within <PipelineProvider>");
  return ctx;
}

export function PipelineProvider({ children, serverReady }: { children: ReactNode; serverReady: boolean }) {
  const [events, setEvents] = useState<PipelineEvent[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentStage, setCurrentStage] = useState<number | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // On mount (or server reconnect), rehydrate from buffered events
  useEffect(() => {
    if (!serverReady) return;
    getRunEvents()
      .then((data) => {
        if (data.events.length > 0) {
          setEvents(data.events);
          setCurrentStage(data.current_stage);
          setIsRunning(data.is_running);
          // If still running, reconnect the WebSocket to get live updates
          if (data.is_running && data.current_stage != null) {
            _connectWebSocket(data.current_stage, data.events.length);
          }
        }
      })
      .catch(() => {
        // Server may not support /run/events yet — ignore
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [serverReady]);

  const _connectWebSocket = useCallback((stage: number, skipCount = 0) => {
    const ws = new WebSocket("ws://127.0.0.1:8008/run/stream");
    wsRef.current = ws;

    ws.onopen = () => {
      // If reconnecting (skipCount > 0), we only listen — don't send start
      if (skipCount === 0) {
        ws.send(JSON.stringify({ stage, fast: false, skip_gwen: false }));
      }
    };

    ws.onmessage = (msg) => {
      const event: PipelineEvent = JSON.parse(msg.data);
      setEvents((prev) => [...prev, event]);

      if (event.stage !== undefined) {
        setCurrentStage(event.stage);
      }

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
      // Only clear running state if we didn't get a "complete" event
      // (the onmessage handler already handles that case)
    };
  }, []);

  const startStage = useCallback(
    (stage: number, opts: { fast?: boolean; skip_gwen?: boolean } = {}) => {
      // Close any existing socket
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }

      setEvents([]);
      setError(null);
      setIsRunning(true);
      setCurrentStage(stage);

      const ws = new WebSocket("ws://127.0.0.1:8008/run/stream");
      wsRef.current = ws;

      ws.onopen = () => {
        ws.send(
          JSON.stringify({
            stage,
            fast: opts.fast ?? false,
            skip_gwen: opts.skip_gwen ?? false,
          }),
        );
      };

      ws.onmessage = (msg) => {
        const event: PipelineEvent = JSON.parse(msg.data);
        setEvents((prev) => [...prev, event]);

        if (event.stage !== undefined) {
          setCurrentStage(event.stage);
        }

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
        // noop — state managed by onmessage
      };
    },
    [],
  );

  const cancel = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
    setIsRunning(false);
  }, []);

  return (
    <PipelineContext.Provider value={{ events, isRunning, error, currentStage, startStage, cancel }}>
      {children}
    </PipelineContext.Provider>
  );
}
