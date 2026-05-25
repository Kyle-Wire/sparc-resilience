/**
 * WorkflowState tests — advisory progress tracking across the SPARC workflow.
 *
 * WorkflowStep:
 *   awaiting_data       — project loaded, no data yet
 *   ready_to_configure  — data loaded; user can configure analysis
 *   ready_to_run        — analysis configured; pipeline ready
 *   results_available   — pipeline has run; insights ready to view
 *
 * The spine is advisory: it never blocks navigation, only surfaces progress.
 */
import { describe, it, expect, beforeEach, vi } from "vitest";

vi.mock("@/stores/projectStore", () => ({
  useProjectStore: {
    getState: () => ({ projectLoaded: true }),
  },
}));

vi.mock("@/components/layout/Sidebar", () => ({}));

import { useNavigationStore } from "@/stores/navigationStore";

function resetNav() {
  localStorage.clear();
  useNavigationStore.setState({
    currentPage: "Project",
    pageUIState: {},
    workflowStep: "awaiting_data",
  });
}

describe("WorkflowState", () => {
  beforeEach(() => resetNav());

  it("starts at awaiting_data", () => {
    expect(useNavigationStore.getState().workflowStep).toBe("awaiting_data");
  });

  it("markDataReady advances to ready_to_configure", () => {
    useNavigationStore.getState().markDataReady();
    expect(useNavigationStore.getState().workflowStep).toBe("ready_to_configure");
  });

  it("markAnalysisConfigured advances to ready_to_run from ready_to_configure", () => {
    useNavigationStore.getState().markDataReady();
    useNavigationStore.getState().markAnalysisConfigured();
    expect(useNavigationStore.getState().workflowStep).toBe("ready_to_run");
  });

  it("markPipelineRun advances to results_available from ready_to_run", () => {
    useNavigationStore.getState().markDataReady();
    useNavigationStore.getState().markAnalysisConfigured();
    useNavigationStore.getState().markPipelineRun();
    expect(useNavigationStore.getState().workflowStep).toBe("results_available");
  });

  it("markAnalysisConfigured is a no-op from awaiting_data", () => {
    useNavigationStore.getState().markAnalysisConfigured();
    expect(useNavigationStore.getState().workflowStep).toBe("awaiting_data");
  });

  it("markPipelineRun is a no-op from ready_to_configure", () => {
    useNavigationStore.getState().markDataReady();
    useNavigationStore.getState().markPipelineRun();
    expect(useNavigationStore.getState().workflowStep).toBe("ready_to_configure");
  });

  it("markDataReady is idempotent — calling twice stays at ready_to_configure", () => {
    useNavigationStore.getState().markDataReady();
    useNavigationStore.getState().markDataReady();
    expect(useNavigationStore.getState().workflowStep).toBe("ready_to_configure");
  });

  it("resetWorkflow returns to awaiting_data from any state", () => {
    useNavigationStore.getState().markDataReady();
    useNavigationStore.getState().markAnalysisConfigured();
    useNavigationStore.getState().resetWorkflow();
    expect(useNavigationStore.getState().workflowStep).toBe("awaiting_data");
  });
});
