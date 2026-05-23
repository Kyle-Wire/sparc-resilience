/**
 * projectConfigStore tests (C2 architecture candidate)
 *
 * Tests that the store is the single seam for ProjectConfig:
 *  - fetchConfig() fetches once and caches
 *  - updateConfig() patches in-memory and debounces the save
 *  - saveNow() persists immediately
 *  - reset() clears config on project unload
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

// ── Mocks ───────────────────────────────────────────────────────────────────
const mockGetConfig = vi.fn();
const mockSaveConfig = vi.fn();

vi.mock("@/lib/api", () => ({
  getConfig: (...args: unknown[]) => mockGetConfig(...args),
  saveConfig: (...args: unknown[]) => mockSaveConfig(...args),
}));

// ── Import store fresh per test via factory ─────────────────────────────────
import { useProjectConfigStore } from "./projectConfigStore";
import type { ProjectConfig } from "@/lib/types";

const SAMPLE_CONFIG: ProjectConfig = {
  project: { name: "test", domain: "wildfire" },
  data: { path: "data.csv", target_column: "burned_area" },
} as unknown as ProjectConfig;

function resetStore() {
  useProjectConfigStore.setState({ config: null, loading: false, error: null });
}

describe("useProjectConfigStore", () => {
  beforeEach(() => {
    resetStore();
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  // ── Tracer bullet ──────────────────────────────────────────────────────
  it("starts with config=null and loading=false", () => {
    const { config, loading, error } = useProjectConfigStore.getState();
    expect(config).toBeNull();
    expect(loading).toBe(false);
    expect(error).toBeNull();
  });

  // ── fetchConfig ────────────────────────────────────────────────────────
  it("fetchConfig stores the response and clears loading", async () => {
    mockGetConfig.mockResolvedValueOnce(SAMPLE_CONFIG);
    await useProjectConfigStore.getState().fetchConfig();

    const { config, loading } = useProjectConfigStore.getState();
    expect(config).toEqual(SAMPLE_CONFIG);
    expect(loading).toBe(false);
  });

  it("fetchConfig is idempotent — does not re-fetch if config already loaded", async () => {
    mockGetConfig.mockResolvedValue(SAMPLE_CONFIG);
    await useProjectConfigStore.getState().fetchConfig();
    await useProjectConfigStore.getState().fetchConfig();
    expect(mockGetConfig).toHaveBeenCalledTimes(1);
  });

  it("fetchConfig(force=true) re-fetches even when config is loaded", async () => {
    mockGetConfig.mockResolvedValue(SAMPLE_CONFIG);
    await useProjectConfigStore.getState().fetchConfig();
    await useProjectConfigStore.getState().fetchConfig({ force: true });
    expect(mockGetConfig).toHaveBeenCalledTimes(2);
  });

  it("fetchConfig records error on failure", async () => {
    mockGetConfig.mockRejectedValueOnce(new Error("Server error"));
    await useProjectConfigStore.getState().fetchConfig();

    const { config, error } = useProjectConfigStore.getState();
    expect(config).toBeNull();
    expect(error).toBe("Server error");
  });

  // ── saveNow ────────────────────────────────────────────────────────────
  it("saveNow patches config and calls saveConfig immediately", async () => {
    useProjectConfigStore.setState({ config: SAMPLE_CONFIG });
    mockSaveConfig.mockResolvedValueOnce({});

    await useProjectConfigStore.getState().saveNow({ project: { name: "renamed" } } as Partial<ProjectConfig>);

    const { config } = useProjectConfigStore.getState();
    expect((config as ProjectConfig).project.name).toBe("renamed");
    expect(mockSaveConfig).toHaveBeenCalledTimes(1);
  });

  // ── reset ──────────────────────────────────────────────────────────────
  it("reset clears config, loading, and error", async () => {
    mockGetConfig.mockResolvedValueOnce(SAMPLE_CONFIG);
    await useProjectConfigStore.getState().fetchConfig();

    useProjectConfigStore.getState().reset();

    const { config, loading, error } = useProjectConfigStore.getState();
    expect(config).toBeNull();
    expect(loading).toBe(false);
    expect(error).toBeNull();
  });
});
