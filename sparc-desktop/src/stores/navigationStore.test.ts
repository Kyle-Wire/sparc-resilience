/**
 * T48 — navigationStore unit tests
 *
 * Verifies navigate (project gate), setPageUIState / getPageUIState,
 * and localStorage persistence.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";

// ── Mock projectStore so we can control projectLoaded ────────────────────────

const mockProjectLoaded = { value: false };

vi.mock("@/stores/projectStore", () => ({
  useProjectStore: {
    getState: () => ({ projectLoaded: mockProjectLoaded.value }),
  },
}));

// ── Mock Sidebar type — PageName is just a union of strings ─────────────────

vi.mock("@/components/layout/Sidebar", () => ({}));

import { useNavigationStore } from "@/stores/navigationStore";

function resetNav() {
  localStorage.clear();
  useNavigationStore.setState({
    currentPage: "Project",
    pageUIState: {},
  });
}

describe("navigationStore", () => {
  beforeEach(() => {
    resetNav();
    mockProjectLoaded.value = false;
    vi.clearAllMocks();
  });

  describe("navigate — project gate", () => {
    it("allows navigation to ungated pages without a loaded project", () => {
      useNavigationStore.getState().navigate("Settings");
      expect(useNavigationStore.getState().currentPage).toBe("Settings");
    });

    it("blocks navigation to gated pages when project is not loaded", () => {
      useNavigationStore.getState().navigate("Data");
      expect(useNavigationStore.getState().currentPage).toBe("Project");
    });

    it("allows navigation to gated pages when project is loaded", () => {
      mockProjectLoaded.value = true;
      useNavigationStore.getState().navigate("Data");
      expect(useNavigationStore.getState().currentPage).toBe("Data");
    });
  });

  describe("localStorage persistence", () => {
    it("writes active page to localStorage on navigate", () => {
      mockProjectLoaded.value = true;
      useNavigationStore.getState().navigate("Run");
      expect(localStorage.getItem("sparc-active-page")).toBe("Run");
    });

    it("does NOT persist blocked navigation", () => {
      useNavigationStore.getState().navigate("Run"); // blocked — no project
      expect(localStorage.getItem("sparc-active-page")).toBeNull();
    });
  });

  describe("pageUIState", () => {
    it("stores and retrieves typed UI state per page", () => {
      useNavigationStore.getState().setPageUIState("Data", { tab: "csv" });
      const state = useNavigationStore.getState().getPageUIState<{ tab: string }>("Data");
      expect(state?.tab).toBe("csv");
    });

    it("returns undefined for a page with no saved UI state", () => {
      const state = useNavigationStore.getState().getPageUIState("Run");
      expect(state).toBeUndefined();
    });
  });
});
