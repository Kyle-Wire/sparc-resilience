/**
 * LoginScreen — URL constant test.
 *
 * Verifies that clicking "Go to sparclabs.co" in signup mode calls
 * shellOpen with the exported SPARC_SIGNUP_URL constant rather than a
 * hardcoded string literal.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { SPARC_SIGNUP_URL } from "@/lib/constants";
import LoginScreen from "./LoginScreen";

// Mock Tauri shell plugin
const mockShellOpen = vi.hoisted(() => vi.fn());
vi.mock("@tauri-apps/plugin-shell", () => ({ open: mockShellOpen }));

// Mock Tauri mouse-parallax (uses window events — not needed for this test)
vi.mock("@/hooks/useMouseParallax", () => ({ useMouseParallax: vi.fn() }));

// Minimal auth store stub — not under test here
vi.mock("@/stores/authStore", () => ({
  useAuthStore: () => ({
    signIn: vi.fn(),
    loading: false,
    error: null,
    clearError: vi.fn(),
  }),
}));

describe("LoginScreen — signup URL", () => {
  beforeEach(() => vi.clearAllMocks());

  it("opens SPARC_SIGNUP_URL when 'Go to sparclabs.co' is clicked in signup mode", async () => {
    render(<LoginScreen parallaxEnabled={false} />);

    // Switch to signup tab
    fireEvent.click(screen.getByRole("button", { name: /create account/i }));

    // Click the CTA
    fireEvent.click(screen.getByRole("button", { name: /sparclabs\.co/i }));

    expect(mockShellOpen).toHaveBeenCalledWith(SPARC_SIGNUP_URL);
    expect(mockShellOpen).toHaveBeenCalledTimes(1);
  });
});
