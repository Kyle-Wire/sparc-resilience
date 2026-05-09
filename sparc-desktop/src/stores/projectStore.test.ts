/**
 * T47 — projectStore unit tests
 *
 * Verifies openProject, clearProject, and rehydrate logic in isolation.
 * All external dependencies (api, recentProjects, manifest, resultCache) are mocked.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";

// ── Mock external dependencies ──────────────────────────────────────────────

vi.mock("@/lib/api", () => ({
  loadProject: vi.fn(async () => ({ status: "loaded", project: {}, columns: [], row_count: 0 })),
  health: vi.fn(async () => ({ status: "ok", version: "test", project_loaded: false })),
}));

vi.mock("@/lib/recentProjects", () => ({ addRecent: vi.fn() }));
vi.mock("@/hooks/useManifest", () => ({ resetManifest: vi.fn() }));
vi.mock("@/stores/resultCache", () => ({
  useResultCacheStore: { getState: () => ({ clear: vi.fn() }) },
}));

// ── jsdom localStorage is available; re-import store fresh each test ────────
import { useProjectStore } from "@/stores/projectStore";
import { loadProject as mockLoadProject } from "@/lib/api";

// Reset store + localStorage between tests
function resetStore() {
  localStorage.clear();
  useProjectStore.setState({
    projectPath: null,
    projectLoaded: false,
    rehydrating: true,
  });
}

describe("projectStore", () => {
  beforeEach(() => {
    resetStore();
    vi.clearAllMocks();
  });

  describe("openProject", () => {
    it("sets projectLoaded and persists path to localStorage", async () => {
      const { openProject } = useProjectStore.getState();
      await openProject("/tmp/myproject.yml");

      const state = useProjectStore.getState();
      expect(state.projectLoaded).toBe(true);
      expect(state.projectPath).toBe("/tmp/myproject.yml");
      expect(localStorage.getItem("sparc-current-project")).toBe("/tmp/myproject.yml");
    });

    it("calls apiLoadProject with the given path", async () => {
      await useProjectStore.getState().openProject("/tmp/foo.yml");
      expect(mockLoadProject).toHaveBeenCalledWith("/tmp/foo.yml");
    });
  });

  describe("clearProject", () => {
    it("resets state and removes localStorage key", async () => {
      await useProjectStore.getState().openProject("/tmp/myproject.yml");
      useProjectStore.getState().clearProject();

      const state = useProjectStore.getState();
      expect(state.projectLoaded).toBe(false);
      expect(state.projectPath).toBeNull();
      expect(localStorage.getItem("sparc-current-project")).toBeNull();
    });
  });

  describe("rehydrate", () => {
    it("sets projectLoaded when health says project_loaded=true", async () => {
      const { health } = await import("@/lib/api");
      (health as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        status: "ok",
        version: "test",
        project_loaded: true,
      });

      await useProjectStore.getState().rehydrate();
      expect(useProjectStore.getState().projectLoaded).toBe(true);
    });

    it("clears stale path when server is unreachable", async () => {
      const { health } = await import("@/lib/api");
      (health as ReturnType<typeof vi.fn>).mockRejectedValueOnce(new Error("offline"));
      localStorage.setItem("sparc-current-project", "/stale/path.yml");
      useProjectStore.setState({ projectPath: "/stale/path.yml", rehydrating: true });

      await useProjectStore.getState().rehydrate();
      expect(useProjectStore.getState().projectPath).toBeNull();
      expect(localStorage.getItem("sparc-current-project")).toBeNull();
    });
  });
});
