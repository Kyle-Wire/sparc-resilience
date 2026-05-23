import { create } from "zustand";
import type { PageName } from "@/components/layout/Sidebar";
import { useProjectStore } from "./projectStore";

export type AppPage = PageName | "Settings" | "Performance";

/** Pages that can be entered without a loaded project. */
const UNGATED_PAGES = new Set<AppPage>(["Project", "Settings", "Performance"]);

const STORAGE_KEY = "sparc-active-page";

export interface NavigateResult {
  ok: boolean;
  /** Present when ok is false; describes why navigation was blocked. */
  reason?: "no_project";
}

interface NavigationState {
  currentPage: AppPage;
  /** Map of page → arbitrary UI state (tab, scroll offset, etc.) */
  pageUIState: Record<string, unknown>;

  /**
   * Navigate to a page. Returns { ok: true } on success, or
   * { ok: false, reason } when navigation is blocked (e.g. no project loaded).
   * Callers can use the reason to surface a tooltip or notification.
   */
  navigate: (page: AppPage | string) => NavigateResult;

  /** Persist arbitrary UI state for a page (e.g. active tab, scroll offset). */
  setPageUIState: <T>(page: string, state: T) => void;

  /** Retrieve typed UI state for a page. */
  getPageUIState: <T>(page: string) => T | undefined;
}

export const useNavigationStore = create<NavigationState>((set, get) => ({
  currentPage: (() => {
    const saved = localStorage.getItem(STORAGE_KEY) as AppPage | null;
    // Only restore safe pages on cold start — project gate handles the rest
    const safeOnColdStart: AppPage[] = ["Project", "Settings", "Performance"];
    return saved && safeOnColdStart.includes(saved) ? saved : "Project";
  })(),

  pageUIState: {},

  navigate: (page) => {
    const target = page as AppPage;
    if (!UNGATED_PAGES.has(target) && !useProjectStore.getState().projectLoaded) {
      return { ok: false, reason: "no_project" };
    }
    set({ currentPage: target });
    localStorage.setItem(STORAGE_KEY, target);
    return { ok: true };
  },

  setPageUIState: (page, state) => {
    set((s) => ({ pageUIState: { ...s.pageUIState, [page]: state } }));
  },

  getPageUIState: (page) => {
    return get().pageUIState[page] as never;
  },
}));
