/**
 * Tracks the free-tier monthly run cap.
 *
 * Backed by `tauri-plugin-store`, so the count survives app restarts
 * but resets at the top of each calendar month.
 */
import { useCallback, useEffect, useState } from "react";
import {
  incrementRunCounter,
  readRunCounter,
  type RunCounter,
} from "@/lib/auth/cache";
import { useEntitlements } from "@/hooks/useEntitlements";

export interface RunCapState {
  counter: RunCounter;
  /** True when the free-tier user has hit the monthly cap. */
  capped: boolean;
  /** Runs left this month, or `null` for unlimited. */
  remaining: number | null;
  /** Increment after a successful pipeline run. */
  recordRun(): Promise<void>;
}

export function useDailyRunCounter(): RunCapState {
  const ent = useEntitlements();
  const [counter, setCounter] = useState<RunCounter>({ month: "", count: 0 });

  useEffect(() => {
    void readRunCounter().then(setCounter);
  }, []);

  const recordRun = useCallback(async () => {
    const next = await incrementRunCounter();
    setCounter(next);
  }, []);

  const unlimited = ent.maxRunsPerMonth < 0;
  const remaining = unlimited ? null : Math.max(0, ent.maxRunsPerMonth - counter.count);
  const capped = !unlimited && counter.count >= ent.maxRunsPerMonth;

  return { counter, capped, remaining, recordRun };
}
