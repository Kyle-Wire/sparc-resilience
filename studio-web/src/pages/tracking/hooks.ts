// Small hooks shared by the tracking pages: a ticking clock, /api/meta, the reduced-motion
// preference and the moment a job entered `cancelling` (for the Force stop grace period).
import { useEffect, useRef, useState } from "react";
import { api } from "../../api/client";
import { useResource } from "../../api/resource";
import type { Job, Meta } from "../../api/types";
import { prefersReducedMotion } from "../../components/ui/StatusChip";

/** Force stop is offered this long after a cancel was requested (SPEC §5.8, §5.9). */
export const FORCE_STOP_GRACE_S = 90;

/** `Date.now()` re-read every `ms` while `active` (elapsed clocks, grace periods). */
export function useNow(ms = 1000, active = true): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    setNow(Date.now());
    const h = setInterval(() => setNow(Date.now()), ms);
    return () => clearInterval(h);
  }, [ms, active]);
  return now;
}

/** `GET /api/meta` (stage labels, output catalog, warning code views); fetched once per tab. */
export function useMeta() {
  return useResource<Meta>("meta", (s) => api.get<Meta>("/api/meta", undefined, s), { immutable: true });
}

/** The `prefers-reduced-motion` media preference, live. */
export function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(prefersReducedMotion);
  useEffect(() => {
    if (typeof window === "undefined" || !window.matchMedia) return;
    let mq: MediaQueryList;
    try {
      mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    } catch {
      return;
    }
    const on = () => setReduced(mq.matches);
    mq.addEventListener?.("change", on);
    return () => mq.removeEventListener?.("change", on);
  }, []);
  return reduced;
}

/**
 * When the job entered `cancelling`, in ms since the epoch: the `cancel.requested` time the
 * events carry, else the moment this page first saw the status (a snapshot without events).
 */
export function useCancellingSince(job: Pick<Job, "id" | "status"> | null, requestedTs: number | null): number | null {
  const seen = useRef<{ id: string; at: number } | null>(null);
  if (!job || job.status !== "cancelling") {
    seen.current = null;
    return null;
  }
  if (requestedTs !== null) return requestedTs * 1000;
  if (!seen.current || seen.current.id !== job.id) seen.current = { id: job.id, at: Date.now() };
  return seen.current.at;
}
