/**
 * useAvailability — adaptive polling of `/panels/availability`.
 *
 * Polls every 2s while the pipeline is running (or the document is
 * focused and any panel is in a non-ready state) and every 30s when the
 * pipeline is idle and all panels have settled. Returns a per-panel
 * lookup so individual panels can render a uniform `<PanelGate>`.
 *
 * The hook is a singleton-per-page: components share one polling loop
 * via a module-level cache so 14+ panels mounting at once don't fan out
 * into 14 separate intervals.
 */
import { useEffect, useState } from "react";
import {
  getPanelsAvailability,
  type PanelAvailabilityEntry,
  type PanelAvailabilityResponse,
} from "@/lib/api";

let cached: PanelAvailabilityResponse | null = null;
let inFlight: Promise<PanelAvailabilityResponse | null> | null = null;
let timer: ReturnType<typeof setTimeout> | null = null;
const subscribers = new Set<(snapshot: PanelAvailabilityResponse | null) => void>();

const FAST_INTERVAL_MS = 2_000;
const SLOW_INTERVAL_MS = 30_000;

function nextDelay(snap: PanelAvailabilityResponse | null): number {
  if (!snap) return FAST_INTERVAL_MS;
  if (snap.is_running) return FAST_INTERVAL_MS;
  // Fast cadence while at least one panel is still mid-flight.
  for (const p of Object.values(snap.panels)) {
    if (p.status === "running" || p.status === "partial") return FAST_INTERVAL_MS;
  }
  return SLOW_INTERVAL_MS;
}

async function refresh(): Promise<void> {
  if (inFlight) return;
  inFlight = getPanelsAvailability().catch(() => null);
  const snap = await inFlight;
  inFlight = null;
  cached = snap;
  for (const s of subscribers) s(cached);
}

function scheduleNext() {
  if (timer) clearTimeout(timer);
  if (subscribers.size === 0) {
    timer = null;
    return;
  }
  const delay = nextDelay(cached);
  timer = setTimeout(async () => {
    await refresh();
    scheduleNext();
  }, delay);
}

export interface UseAvailabilityResult {
  snapshot: PanelAvailabilityResponse | null;
  panel: (panelId: string) => PanelAvailabilityEntry | null;
  refetch: () => Promise<void>;
}

export function useAvailability(): UseAvailabilityResult {
  const [snapshot, setSnapshot] = useState<PanelAvailabilityResponse | null>(cached);

  useEffect(() => {
    subscribers.add(setSnapshot);
    if (subscribers.size === 1) {
      // First subscriber kicks off the polling loop.
      void refresh().then(scheduleNext);
    } else if (cached) {
      // Late subscribers hydrate from the cache immediately.
      setSnapshot(cached);
    }
    return () => {
      subscribers.delete(setSnapshot);
      if (subscribers.size === 0 && timer) {
        clearTimeout(timer);
        timer = null;
      }
    };
  }, []);

  return {
    snapshot,
    panel: (panelId: string) => snapshot?.panels[panelId] ?? null,
    refetch: refresh,
  };
}
