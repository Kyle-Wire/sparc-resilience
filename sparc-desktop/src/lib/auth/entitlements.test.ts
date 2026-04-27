/**
 * Entitlement matrix tests. Pure data, no mocks.
 * Locks the tier matrix so accidental changes break CI.
 */
import { describe, it, expect } from "vitest";
import { featuresFor } from "./entitlements";

describe("featuresFor", () => {
  it("free tier is locked down", () => {
    const f = featuresFor("free");
    expect(f.plan).toBe("free");
    expect(f.maxProjects).toBe(1);
    expect(f.maxRunsPerMonth).toBe(5);
    expect(f.templates).toBe("blank_only");
    expect(f.scenarioRunner).toBe(false);
    expect(f.causalDiscoveryLearn).toBe(false);
    expect(f.aiChat).toBe(false);
    expect(f.reportExport).toBe(false);
    expect(f.multiSeat).toBe(false);
  });

  it("pro tier unlocks everything except multi-seat", () => {
    const f = featuresFor("pro");
    expect(f.plan).toBe("pro");
    expect(f.maxProjects).toBe(-1);
    expect(f.maxRunsPerMonth).toBe(-1);
    expect(f.templates).toBe("all");
    expect(f.scenarioRunner).toBe(true);
    expect(f.causalDiscoveryLearn).toBe(true);
    expect(f.aiChat).toBe(true);
    expect(f.reportExport).toBe(true);
    expect(f.multiSeat).toBe(false);
  });

  it("team tier unlocks everything including multi-seat", () => {
    const f = featuresFor("team");
    expect(f.plan).toBe("team");
    expect(f.multiSeat).toBe(true);
    // Spot-check the rest are also unlocked.
    expect(f.maxProjects).toBe(-1);
    expect(f.maxRunsPerMonth).toBe(-1);
    expect(f.aiChat).toBe(true);
    expect(f.reportExport).toBe(true);
  });

  it("unknown plan strings fall back to free", () => {
    // Cast through `any` to simulate a stale cache with an unknown
    // plan string from a future version.
    const f = featuresFor("nonsense" as any);
    expect(f.plan).toBe("free");
    expect(f.aiChat).toBe(false);
  });
});
