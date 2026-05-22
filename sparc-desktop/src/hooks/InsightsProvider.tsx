/**
 * InsightsProvider — context for the Insights workspace.
 *
 * Holds two pieces of cross-panel state:
 *   1. `audience` — Practitioner or Researcher mode toggle. Persisted to
 *      localStorage per-project (key = "sparc-insights-audience").
 *   2. `selection` — the brushed/linked selection (point ids, bbox, active
 *      variable, active scenario). Panels read this and re-render to filter
 *      their content; panels write to it when the user brushes/clicks.
 *
 * The provider is intentionally framework-light: no manifest fetches, no
 * data loading. Panels still use `useManifest()` directly. This keeps the
 * context small and re-render-friendly.
 */
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { useNavigationStore } from "@/stores/navigationStore";

export type Audience = "practitioner" | "researcher" | "public";

/** A bbox in the same CRS as the predictions GeoJSON. `[minX, minY, maxX, maxY]`. */
export type BBox = [number, number, number, number];

export interface LinkedSelection {
  /** Subset of feature ids selected via brushing. Empty array = no filter. */
  ids: string[];
  /** Geographic bbox; lets panels filter without enumerating ids. */
  bbox: BBox | null;
  /** Active treatment variable for PDP / CATE / dose-response panels. */
  variable: string | null;
  /** Active scenario name. */
  scenario: string | null;
}

const EMPTY_SELECTION: LinkedSelection = {
  ids: [],
  bbox: null,
  variable: null,
  scenario: null,
};

export interface InsightsContextValue {
  audience: Audience;
  setAudience: (a: Audience) => void;

  selection: LinkedSelection;
  setSelection: (next: Partial<LinkedSelection>) => void;
  clearSelection: () => void;

  /** Convenience: true when any selection is active. */
  hasSelection: boolean;

  /** Navigate to another app page (e.g. "Run"). Undefined if not provided. */
  navigate: ((page: string) => void) | undefined;
}

const Ctx = createContext<InsightsContextValue | null>(null);

const AUDIENCE_KEY = "sparc-insights-audience";

function loadAudience(): Audience {
  if (typeof window === "undefined") return "practitioner";
  const v = window.localStorage.getItem(AUDIENCE_KEY);
  if (v === "researcher" || v === "public") return v;
  return "practitioner";
}

interface ProviderProps {
  children: ReactNode;
  /** Optional initial audience override (e.g. from project config). */
  initialAudience?: Audience;
}

export function InsightsProvider({ children, initialAudience }: ProviderProps) {
  const [audience, setAudienceState] = useState<Audience>(
    () => initialAudience ?? loadAudience(),
  );
  const [selection, setSelectionState] = useState<LinkedSelection>(EMPTY_SELECTION);

  useEffect(() => {
    if (typeof window !== "undefined") {
      window.localStorage.setItem(AUDIENCE_KEY, audience);
    }
  }, [audience]);

  const setAudience = useCallback((a: Audience) => setAudienceState(a), []);

  const setSelection = useCallback((next: Partial<LinkedSelection>) => {
    setSelectionState((prev) => ({ ...prev, ...next }));
  }, []);

  const clearSelection = useCallback(() => setSelectionState(EMPTY_SELECTION), []);

  const { navigate } = useNavigationStore();

  const value = useMemo<InsightsContextValue>(
    () => ({
      audience,
      setAudience,
      selection,
      setSelection,
      clearSelection,
      hasSelection:
        selection.ids.length > 0 ||
        selection.bbox !== null ||
        selection.variable !== null ||
        selection.scenario !== null,
      navigate,
    }),
    [audience, setAudience, selection, setSelection, clearSelection, navigate],
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useInsights(): InsightsContextValue {
  const v = useContext(Ctx);
  if (!v) {
    throw new Error("useInsights must be used inside <InsightsProvider>");
  }
  return v;
}

/** Convenience: just the audience. */
export function useAudience(): [Audience, (a: Audience) => void] {
  const { audience, setAudience } = useInsights();
  return [audience, setAudience];
}

/** Convenience: navigate to another app page. Returns undefined if not wired up. */
export function useInsightsNavigate(): ((page: string) => void) | undefined {
  return useInsights().navigate;
}
