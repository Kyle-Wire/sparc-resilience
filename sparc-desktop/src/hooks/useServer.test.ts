/**
 * useServer — respawning state tests.
 *
 * Verifies that the hook exposes a `respawning` boolean that is true while
 * an auto-respawn retry is in-flight (server was previously ready but health
 * polls now fail). This gives callers UI-visible feedback during the retry
 * window instead of showing a hard "server lost" error immediately.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useServer } from "./useServer";

// -- Timing constants mirrored from useServer.ts ----------------------------
const FAST_MS = 500;
const SLOW_MS = 30_000;

// -- Mocks ------------------------------------------------------------------

const mockHealth = vi.hoisted(() => vi.fn());

vi.mock("@/lib/api", () => ({ health: mockHealth }));

// Tauri is unavailable in jsdom — the hook's dynamic-import already has a
// try/catch that swallows the error, so no explicit mock needed.

// -- Tests ------------------------------------------------------------------

describe("useServer — respawning", () => {
  const HEALTHY: import("@/lib/types").HealthResponse = {
    version: "test",
    project_loaded: false,
  } as never;

  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
  });

  it("exposes respawning=false when the server is healthy", async () => {
    mockHealth.mockResolvedValue(HEALTHY);

    const { result } = renderHook(() => useServer());

    // Flush the mount-time poll
    await act(async () => {});

    expect(result.current.respawning).toBe(false);
    expect(result.current.ready).toBe(true);
  });

  it("sets respawning=true after the first health failure mid-session", async () => {
    mockHealth
      .mockResolvedValueOnce(HEALTHY)       // first poll → ready
      .mockRejectedValue(new Error("gone")); // all subsequent polls fail

    const { result } = renderHook(() => useServer());

    // Flush initial poll → ready=true
    await act(async () => {});
    expect(result.current.ready).toBe(true);

    // Advance to the slow-interval tick → health fails → respawn starts
    await act(async () => {
      vi.advanceTimersByTime(SLOW_MS + 1);
    });

    expect(result.current.respawning).toBe(true);
    expect(result.current.serverLost).toBe(false);
  });
});
