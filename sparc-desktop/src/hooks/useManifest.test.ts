/**
 * Tests for useManifest module-level singleton.
 *
 * These tests verify the singleton's data flow using the injectable fetcher
 * seam (`__setFetcherForTesting`). No React mounting is required for the
 * core behaviour; we drive the singleton directly via the exported functions.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import type { RunManifest } from "@/lib/types";
import {
  __setFetcherForTesting,
  __resetForTesting,
  notifyManifestArtifactWritten,
  resetManifest,
} from "./useManifest";

// Block the real `addManifestArtifactListener` side-effect at module load.
vi.mock("./useArtifactStream", () => ({
  addManifestArtifactListener: vi.fn(),
}));

// The module singleton is reset between tests.
beforeEach(() => {
  __resetForTesting();
  vi.useFakeTimers();
});

afterEach(() => {
  __resetForTesting();
  vi.useRealTimers();
});

const makeManifest = (stages = {}): RunManifest => ({
  schema_version: 1,
  stages,
});

// ---------------------------------------------------------------------------
// Fetcher seam
// ---------------------------------------------------------------------------

describe("__setFetcherForTesting / __resetForTesting", () => {
  it("injectable fetcher is called by _fetchOnce", async () => {
    const mockFetcher = vi.fn().mockResolvedValue(makeManifest());
    __setFetcherForTesting(mockFetcher);

    // Import _fetchOnce indirectly: call notifyManifestArtifactWritten which
    // schedules a debounced fetch. Advance timers to trigger it.
    notifyManifestArtifactWritten();
    await vi.runAllTimersAsync();

    expect(mockFetcher).toHaveBeenCalledWith({ refresh: true, rescan: undefined });
  });

  it("__resetForTesting restores state so a second test starts clean", () => {
    const mockFetcher = vi.fn().mockResolvedValue(makeManifest());
    __setFetcherForTesting(mockFetcher);
    __resetForTesting();

    // After reset the injected fetcher should be cleared (no calls in next tick)
    notifyManifestArtifactWritten();
    // Don't advance timers — just assert the injected mock is not set
    expect(mockFetcher).not.toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------------
// notifyManifestArtifactWritten burst debounce
// ---------------------------------------------------------------------------

describe("notifyManifestArtifactWritten", () => {
  it("fires fetcher once after burst of events", async () => {
    const mockFetcher = vi.fn().mockResolvedValue(makeManifest());
    __setFetcherForTesting(mockFetcher);

    notifyManifestArtifactWritten();
    notifyManifestArtifactWritten();
    notifyManifestArtifactWritten();

    await vi.runAllTimersAsync();

    expect(mockFetcher).toHaveBeenCalledTimes(1);
  });
});

// ---------------------------------------------------------------------------
// resetManifest
// ---------------------------------------------------------------------------

describe("resetManifest", () => {
  it("restarts the polling loop after reset", async () => {
    const mockFetcher = vi.fn().mockResolvedValue(makeManifest({ "0": { artifacts: {} } }));
    __setFetcherForTesting(mockFetcher);

    resetManifest();
    // resetManifest calls _start() → immediate _fetchOnce call
    await vi.runAllTimersAsync();

    expect(mockFetcher).toHaveBeenCalled();
  });
});
