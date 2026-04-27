/**
 * Tests for the pure helpers in `cache.ts` that the verifyLicense
 * state machine depends on. The store-backed I/O functions
 * (readLicenseCache / writeLicenseCache / incrementRunCounter) are
 * exercised via integration in the renderer at runtime; we don't
 * mock the Tauri store here.
 */
import { describe, it, expect } from "vitest";
import { graceDaysRemaining, isWithinGrace, buildCache } from "./cache";
import type { LicenseCache } from "./types";

const ONE_DAY_MS = 24 * 60 * 60 * 1000;

function cacheVerifiedAgo(daysAgo: number): LicenseCache {
  return {
    valid: true,
    plan: "pro",
    status: "active",
    expiry: null,
    last_verified: new Date(Date.now() - daysAgo * ONE_DAY_MS).toISOString(),
    email: "u@example.com",
    provider: "google",
    license_key_last4: null,
  };
}

describe("graceDaysRemaining", () => {
  it("returns ~15 immediately after verification", () => {
    const c = cacheVerifiedAgo(0);
    expect(graceDaysRemaining(c)).toBe(15);
  });

  it("decreases by 1 per elapsed day", () => {
    expect(graceDaysRemaining(cacheVerifiedAgo(5))).toBe(10);
    expect(graceDaysRemaining(cacheVerifiedAgo(14))).toBe(1);
  });

  it("hits zero exactly at the 15-day boundary", () => {
    expect(graceDaysRemaining(cacheVerifiedAgo(15))).toBeLessThanOrEqual(0);
  });

  it("goes negative once expired", () => {
    expect(graceDaysRemaining(cacheVerifiedAgo(20))).toBeLessThan(0);
  });
});

describe("isWithinGrace", () => {
  it("true for fresh caches", () => {
    expect(isWithinGrace(cacheVerifiedAgo(1))).toBe(true);
    expect(isWithinGrace(cacheVerifiedAgo(14))).toBe(true);
  });

  it("false for expired caches", () => {
    expect(isWithinGrace(cacheVerifiedAgo(15))).toBe(false);
    expect(isWithinGrace(cacheVerifiedAgo(30))).toBe(false);
  });
});

describe("buildCache", () => {
  it("masks the license key down to last 4 chars", () => {
    const c = buildCache({
      plan: "pro",
      status: "active",
      valid: true,
      expiry: null,
      email: "u@example.com",
      provider: "license_key",
      licenseKey: "ABCD-1234-EFGH-5678",
    });
    expect(c.license_key_last4).toBe("5678");
  });

  it("emits null last4 when no key is present", () => {
    const c = buildCache({
      plan: "free",
      status: "active",
      valid: true,
      expiry: null,
      email: "u@example.com",
      provider: "google",
    });
    expect(c.license_key_last4).toBeNull();
  });

  it("stamps last_verified with current ISO time", () => {
    const before = Date.now();
    const c = buildCache({
      plan: "team",
      status: "active",
      valid: true,
      expiry: null,
      email: "u@example.com",
      provider: "github",
    });
    const stamped = new Date(c.last_verified).getTime();
    expect(stamped).toBeGreaterThanOrEqual(before);
    expect(stamped).toBeLessThanOrEqual(Date.now() + 100);
  });
});
