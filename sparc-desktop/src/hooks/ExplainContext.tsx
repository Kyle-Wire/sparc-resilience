import { createContext, useCallback, useContext, useMemo, useState } from "react";
import type { ReactNode } from "react";

/**
 * Cross-cutting "Explain this" / hypothesis bus.
 *
 * Pages call ``useExplain().requestExplain(text, mode)`` to seed the
 * chat panel with a focused question; App.tsx reads the seed and opens
 * the chat with the appropriate Claude system suffix.
 */

export type ExplainMode = "narrator" | "hypothesis";

export interface ExplainSeed {
  message: string;
  mode: ExplainMode;
  nonce: number;
}

export interface ExplainContextValue {
  seed: ExplainSeed | null;
  requestExplain: (message: string, mode?: ExplainMode) => void;
  consumeSeed: () => void;
}

export const ExplainContext = createContext<ExplainContextValue | null>(null);

/** Hook to be called from any page/component below an ExplainProvider. */
export function useExplain(): ExplainContextValue {
  const ctx = useContext(ExplainContext);
  if (!ctx) throw new Error("useExplain must be used inside <ExplainProvider>");
  return ctx;
}

/** Standalone provider (manages its own state). */
export function ExplainProvider({ children }: { children: ReactNode }) {
  const [seed, setSeed] = useState<ExplainSeed | null>(null);
  const requestExplain = useCallback((message: string, mode: ExplainMode = "narrator") => {
    setSeed({ message, mode, nonce: Date.now() });
  }, []);
  const consumeSeed = useCallback(() => setSeed(null), []);
  const value = useMemo(
    () => ({ seed, requestExplain, consumeSeed }),
    [seed, requestExplain, consumeSeed],
  );
  return <ExplainContext.Provider value={value}>{children}</ExplainContext.Provider>;
}

/** Helper hook the host (App) uses to manage the seed itself so it can
 * react to seed changes (e.g. open the chat panel automatically). */
export function useExplainHost() {
  const [seed, setSeed] = useState<ExplainSeed | null>(null);
  const requestExplain = useCallback((message: string, mode: ExplainMode = "narrator") => {
    setSeed({ message, mode, nonce: Date.now() });
  }, []);
  const consumeSeed = useCallback(() => setSeed(null), []);
  const value = useMemo<ExplainContextValue>(
    () => ({ seed, requestExplain, consumeSeed }),
    [seed, requestExplain, consumeSeed],
  );
  return { seed, value };
}
