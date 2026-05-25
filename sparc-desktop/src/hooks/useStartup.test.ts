/**
 * T49 — useStartup token sequencing tests
 *
 * Verifies the startup state machine includes a "token" step between
 * "sidecar" and "session" so the auth token is acquired before any
 * API calls go out.
 *
 * Observable contracts:
 *   1. Starts at "sidecar".
 *   2. While the Tauri token call is pending, does NOT advance to "session".
 *   3. setToken() is called with the resolved token value before "session" is entered.
 *   4. When Tauri is unavailable, still advances to "session" (graceful degradation).
 *
 * Public interface under test: `useStartup(serverReady, startupFailed, restart)`
 * which returns `{ splashStep, splashDone, ... }`.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";

// ── Deferred promise helper ───────────────────────────────────────────────
function deferred<T>() {
  let resolve!: (v: T) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<T>((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}

// ── Mocks ────────────────────────────────────────────────────────────────

let invokeDeferred = deferred<string>();

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn((_cmd: string) => invokeDeferred.promise),
}));

vi.mock("@/stores/tokenStore", () => ({
  setToken: vi.fn(),
}));

vi.mock("@/stores/authStore", () => ({
  useAuthStore: () => ({ init: vi.fn(async () => {}) }),
}));

vi.mock("@/lib/theme", () => ({
  loadThemeKey: vi.fn(() => "dark"),
  applyTheme: vi.fn(),
}));

import { useStartup } from "@/hooks/useStartup";
import { setToken } from "@/stores/tokenStore";

describe("useStartup — token sequencing", () => {
  beforeEach(() => {
    invokeDeferred = deferred<string>();
    vi.clearAllMocks();
    localStorage.clear();
  });

  it("starts at 'sidecar' before server is ready", () => {
    const { result } = renderHook(() =>
      useStartup(false, false, vi.fn()),
    );
    expect(result.current.splashStep).toBe("sidecar");
    expect(result.current.splashDone).toBe(false);
  });

  it("does not reach 'session' while the token call is still pending", async () => {
    // invokeDeferred never resolves → token step stays pending
    const { result, rerender } = renderHook(
      ({ serverReady }: { serverReady: boolean }) =>
        useStartup(serverReady, false, vi.fn()),
      { initialProps: { serverReady: false } },
    );

    await act(async () => {
      rerender({ serverReady: true });
    });

    // The machine must NOT have jumped straight to "session" while token is pending
    expect(result.current.splashStep).not.toBe("session");
    expect(result.current.splashStep).not.toBe("ready");
  });

  it("calls setToken with the resolved value and advances past the token step", async () => {
    const { result, rerender } = renderHook(
      ({ serverReady }: { serverReady: boolean }) =>
        useStartup(serverReady, false, vi.fn()),
      { initialProps: { serverReady: false } },
    );

    // Trigger sidecar → token
    await act(async () => {
      rerender({ serverReady: true });
    });

    // Resolve the token and flush the microtask chain
    await act(async () => {
      invokeDeferred.resolve("my-secret-token");
      await invokeDeferred.promise;
    });

    // setToken must have been called before session/ready
    expect(setToken).toHaveBeenCalledWith("my-secret-token");
    // Machine must have advanced past the token step (initAuth mock resolves immediately → "ready")
    expect(result.current.splashStep).not.toBe("sidecar");
    expect(result.current.splashStep).not.toBe("token");
  });

  it("advances past the token step even when Tauri is unavailable (graceful degradation)", async () => {
    const { result, rerender } = renderHook(
      ({ serverReady }: { serverReady: boolean }) =>
        useStartup(serverReady, false, vi.fn()),
      { initialProps: { serverReady: false } },
    );

    await act(async () => {
      rerender({ serverReady: true });
      invokeDeferred.reject(new Error("Tauri not available"));
      // Drain the rejection so there's no unhandled rejection warning
      await invokeDeferred.promise.catch(() => {});
    });

    // Machine must have advanced past the token step
    expect(result.current.splashStep).not.toBe("sidecar");
    expect(result.current.splashStep).not.toBe("token");
    expect(result.current.splashStep).not.toBe("failed");
  });
});
