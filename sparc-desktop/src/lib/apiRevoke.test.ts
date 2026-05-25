/**
 * Tests for revokeGwenApproval error handling.
 *
 * Behavior: revokeGwenApproval should reject when the server returns a
 * non-OK HTTP status, so callers can surface the error to the user.
 */
import { describe, it, expect, vi, afterEach } from "vitest";

const mockGetToken = vi.fn(() => "");
const mockGetState = vi.fn(() => ({ session: null as null | { access_token: string } }));

vi.mock("@/stores/tokenStore", () => ({
  getToken: () => mockGetToken(),
}));

vi.mock("@/stores/authStore", () => ({
  useAuthStore: { getState: () => mockGetState() },
}));

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn() }));
vi.mock("@tauri-apps/plugin-dialog", () => ({ open: vi.fn() }));

import { revokeGwenApproval } from "@/lib/api";

describe("revokeGwenApproval", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it("resolves with response body on 200 OK", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: () => Promise.resolve({ revoked: true }),
      }),
    );
    const result = await revokeGwenApproval();
    expect(result).toEqual({ revoked: true });
  });

  it("rejects with an error when server returns 403", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 403,
        statusText: "Forbidden",
        json: () => Promise.resolve({ detail: "Not authorized" }),
      }),
    );
    await expect(revokeGwenApproval()).rejects.toThrow();
  });

  it("rejects with an error when server returns 500", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
        statusText: "Internal Server Error",
        json: () => Promise.resolve({ detail: "Server error" }),
      }),
    );
    await expect(revokeGwenApproval()).rejects.toThrow();
  });

  it("rejects when fetch itself throws (network failure)", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("Network error")));
    await expect(revokeGwenApproval()).rejects.toThrow("Network error");
  });
});
