import { useState, useCallback, useEffect, useRef } from "react";
import type { PipelineEvent } from "@/lib/types";

/**
 * Opens a WebSocket to /run/stream and yields structured pipeline events.
 *
 * Performance: incoming messages are accumulated in a ref (no React re-render
 * cost per message) and flushed to state via requestAnimationFrame, capping
 * re-renders at display frame rate (~60 fps) regardless of message volume.
 * Full event history is always retained — the live terminal shows everything.
 */
export function usePipelineStream() {
  const [events, setEvents] = useState<PipelineEvent[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  // Accumulator: all events for this run, mutated without triggering renders.
  const bufferRef = useRef<PipelineEvent[]>([]);
  // rAF handle so we never schedule more than one pending flush at a time.
  const rafRef = useRef<number | null>(null);

  /** Schedule a single rAF flush if one isn't already pending. */
  const scheduleFlush = useCallback(() => {
    if (rafRef.current !== null) return;
    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = null;
      // Snapshot the buffer into state (one array copy per frame, not per message).
      setEvents([...bufferRef.current]);
    });
  }, []);

  const startStage = useCallback(
    (stage: number, opts: { fast?: boolean; skip_gwen?: boolean } = {}) => {
      // Cancel any in-flight rAF from a previous run.
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      bufferRef.current = [];
      setEvents([]);
      setError(null);
      setIsRunning(true);

      const ws = new WebSocket("ws://127.0.0.1:8008/run/execute");
      wsRef.current = ws;

      ws.onopen = () => {
        ws.send(JSON.stringify({ stage, fast: opts.fast ?? false, skip_gwen: opts.skip_gwen ?? false }));
      };

      ws.onmessage = (msg) => {
        const event: PipelineEvent = JSON.parse(msg.data);
        bufferRef.current.push(event);
        scheduleFlush();

        if (event.type === "complete" || event.type === "error") {
          // Flush immediately on terminal events so the final state is applied
          // without waiting for the next animation frame.
          if (rafRef.current !== null) {
            cancelAnimationFrame(rafRef.current);
            rafRef.current = null;
          }
          setEvents([...bufferRef.current]);
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
    [scheduleFlush],
  );

  // On unmount: drop the browser-side WebSocket and cancel any pending rAF.
  // We do NOT call /run/cancel — the server pipeline thread keeps running so
  // the user can navigate elsewhere and come back to see results.
  useEffect(() => {
    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      // Close without the server-side abort — pipeline continues.
      wsRef.current?.close();
    };
  }, []);

  const cancel = useCallback(() => {
    wsRef.current?.close();
    setIsRunning(false);
  }, []);

  return { events, isRunning, error, startStage, cancel };
}
