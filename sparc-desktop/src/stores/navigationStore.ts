import { create } from "zustand";
import type { PageName } from "@/components/layout/Sidebar";
import { localStore } from "@/lib/localStore";
import { useProjectStore } from "./projectStore";

export type AppPage = PageName | "Settings" | "Performance";

/** Pages that can be entered without a loaded project. */
const UNGATED_PAGES = new Set<AppPage>(["Project", "Settings", "Performance"]);

export interface NavigateResult {
  ok: boolean;
  /** Present when ok is false; describes why navigation was blocked. */
  reason?: "no_project";
}

/**
 * Advisory workflow progress — never blocks navigation, only surfaces context
 * to the sidebar status spine and "next step" hints.
 *
 *   awaiting_data       → project open, no data ingested yet
 *   ready_to_configure  → data loaded; analysis configuration unlocked
 *   ready_to_run        → analysis configured; pipeline ready to run
 *   results_available   → pipeline complete; insights ready to view
 */
export type WorkflowStep =
  | "awaiting_data"
  | "ready_to_configure"
  | "ready_to_run"
  | "results_available";

interface NavigationState {
  currentPage: AppPage;
  /** Map of page → arbitrary UI state (tab, scroll offset, etc.) */
  pageUIState: Record<string, unknown>;
  /** Advisory workflow progress spine. */
  workflowStep: WorkflowStep;

  /**
   * Navigate to a page. Returns { ok: true } on success, or
   * { ok: false, reason } when navigation is blocked (e.g. no project loaded).
   */
  navigate: (page: AppPage | string) => NavigateResult;

  /** Persist arbitrary UI state for a page (e.g. active tab, scroll offset). */
  setPageUIState: <T>(page: string, state: T) => void;

  /** Retrieve typed UI state for a page. */
  getPageUIState: <T>(page: string) => T | undefined;

  /** Advance workflow: data has been loaded into the project. */
  markDataReady: () => void;
  /** Advance workflow: analysis has been configured (variables, DAG, scenarios). */
  markAnalysisConfigured: () => void;
  /** Advance workflow: pipeline has completed a run. */
  markPipelineRun: () => void;
  /** Reset workflow spine (called on project clear). */
  resetWorkflow: () => void;
}

export const useNavigationStore = create<NavigationState>((set, get) => ({
  currentPage: (() => {
    const saved = localStore.get("active-page") as AppPage | null;
    // Only restore safe pages on cold start — project gate handles the rest
    const safeOnColdStart: AppPage[] = ["Project", "Settings", "Performance"];
    return saved && safeOnColdStart.includes(saved) ? saved : "Project";
  })(),

  pageUIState: {},

  workflowStep: "awaiting_data",

  navigate: (page) => {
    const target = page as AppPage;
    if (!UNGATED_PAGES.has(target) && !useProjectStore.getState().projectLoaded) {
      return { ok: false, reason: "no_project" };
    }
    set({ currentPage: target });
    localStore.set("active-page", target);
    return { ok: true };
  },

  setPageUIState: (page, state) => {
    set((s) => ({ pageUIState: { ...s.pageUIState, [page]: state } }));
  },

  getPageUIState: (page) => {
    return get().pageUIState[page] as never;
  },

  markDataReady: () => {
    if (get().workflowStep === "awaiting_data") {
      set({ workflowStep: "ready_to_configure" });
    }
  },

  markAnalysisConfigured: () => {
    if (get().workflowStep === "ready_to_configure") {
      set({ workflowStep: "ready_to_run" });
    }
  },

  markPipelineRun: () => {
    if (get().workflowStep === "ready_to_run") {
      set({ workflowStep: "results_available" });
    }
  },

  resetWorkflow: () => {
    set({ workflowStep: "awaiting_data" });
  },
}));
