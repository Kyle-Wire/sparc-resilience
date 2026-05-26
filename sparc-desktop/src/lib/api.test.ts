/**
 * T45 — Verify that API requests include the correct auth headers.
 *
 * We mock fetch and the two auth stores so we can assert that:
 *   1. x-sparc-token is forwarded from tokenStore
 *   2. Authorization: Bearer <jwt> is forwarded when a Supabase session exists
 *   3. Neither header appears when no token / session is set
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

// ---------------------------------------------------------------------------
// Minimal store stubs — must be set up BEFORE importing api.ts
// ---------------------------------------------------------------------------

const mockGetToken = vi.fn(() => "");
const mockGetState = vi.fn(() => ({ session: null as null | { access_token: string } }));

vi.mock("@/stores/tokenStore", () => ({
  getToken: () => mockGetToken(),
}));

vi.mock("@/stores/authStore", () => ({
  useAuthStore: { getState: () => mockGetState() },
}));

// Stub Tauri and other browser APIs that jsdom doesn't provide
vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn() }));
vi.mock("@tauri-apps/plugin-dialog", () => ({ open: vi.fn() }));

// ---------------------------------------------------------------------------
// Now import api helpers — they will use the mocked stores
// ---------------------------------------------------------------------------
import { health, selectDataFile } from "@/lib/api";

// ---------------------------------------------------------------------------
// Fetch mock
// ---------------------------------------------------------------------------

function makeFetchSpy(ok = true) {
  const spy = vi.fn().mockResolvedValue({
    ok,
    json: () => Promise.resolve({ status: "ok", version: "test" }),
    text: () => Promise.resolve(""),
  } as Partial<Response> as Response);
  return spy;
}

describe("authHeaders forwarding", () => {
  let fetchSpy: ReturnType<typeof makeFetchSpy>;

  beforeEach(() => {
    fetchSpy = makeFetchSpy();
    vi.stubGlobal("fetch", fetchSpy);
    mockGetToken.mockReturnValue("");
    mockGetState.mockReturnValue({ session: null });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it("sends x-sparc-token when a token is present", async () => {
    mockGetToken.mockReturnValue("test-sparc-token");
    await health();
    const [, init] = fetchSpy.mock.calls[0];
    const headers = (init as RequestInit).headers as Record<string, string>;
    expect(headers["x-sparc-token"]).toBe("test-sparc-token");
  });

  it("sends Authorization: Bearer when a Supabase session exists", async () => {
    mockGetState.mockReturnValue({ session: { access_token: "my-jwt-token" } });
    await health();
    const [, init] = fetchSpy.mock.calls[0];
    const headers = (init as RequestInit).headers as Record<string, string>;
    expect(headers["Authorization"]).toBe("Bearer my-jwt-token");
  });

  it("sends both headers when both token and JWT are set", async () => {
    mockGetToken.mockReturnValue("sparc-tok");
    mockGetState.mockReturnValue({ session: { access_token: "jwt-tok" } });
    await health();
    const [, init] = fetchSpy.mock.calls[0];
    const headers = (init as RequestInit).headers as Record<string, string>;
    expect(headers["x-sparc-token"]).toBe("sparc-tok");
    expect(headers["Authorization"]).toBe("Bearer jwt-tok");
  });

  it("omits auth headers when neither token nor session is set", async () => {
    await health();
    const [, init] = fetchSpy.mock.calls[0];
    const headers = (init as RequestInit).headers as Record<string, string>;
    expect(headers["x-sparc-token"]).toBeUndefined();
    expect(headers["Authorization"]).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// T-selectDataFile — path sent in request body (not query-string only)
// ---------------------------------------------------------------------------
describe("selectDataFile", () => {
  let fetchSpy: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    fetchSpy = vi.fn().mockResolvedValue({
      ok: true,
      json: () =>
        Promise.resolve({ status: "selected", path: "/tmp/test.csv", columns: [], row_count: 0 }),
      text: () => Promise.resolve(""),
    } as Partial<Response> as Response);
    vi.stubGlobal("fetch", fetchSpy);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it("sends the file path in the JSON request body", async () => {
    await selectDataFile("/tmp/test.csv");
    const [, init] = fetchSpy.mock.calls[0];
    const body = JSON.parse((init as RequestInit).body as string);
    expect(body).toEqual({ path: "/tmp/test.csv" });
  });

  it("includes Content-Type: application/json", async () => {
    await selectDataFile("/tmp/test.csv");
    const [, init] = fetchSpy.mock.calls[0];
    const headers = (init as RequestInit).headers as Record<string, string>;
    expect(headers["Content-Type"]).toBe("application/json");
  });
});
