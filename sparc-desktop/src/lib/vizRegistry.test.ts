/**
 * Tests for vizRegistry C4 — VizEntry component field depth.
 *
 * Verifies that:
 *  1. The `component` field is optional (existing entries work without it).
 *  2. A registry entry CAN carry a lazy-loaded React component reference.
 *  3. `getVizComponent` returns the component for an entry that has one,
 *     and `null` for an entry that does not.
 */
import { describe, it, expect, vi } from "vitest";
import type { VizEntry } from "./vizRegistry";
import {
  VIZ_REGISTRY,
  getVizEntry,
  getVizComponent,
  registerVizComponent,
} from "./vizRegistry";

describe("VizEntry component field", () => {
  it("existing registry entries are valid without a component field", () => {
    const allValid = VIZ_REGISTRY.every(
      (e) => typeof e.stage === "string" && typeof e.artifactId === "string",
    );
    expect(allValid).toBe(true);
  });

  it("VizEntry type accepts an optional component field", () => {
    // Compile-time check via type assignment — if this compiles, the field exists.
    const entry: VizEntry = {
      stage: "0",
      artifactId: "test_artifact",
      kind: "data_histogram",
      label: "Test",
      component: null, // explicitly null is allowed
    };
    expect(entry.component).toBeNull();
  });

  it("getVizComponent returns null for an entry with no component", () => {
    const result = getVizComponent("0", "data");
    expect(result).toBeNull();
  });

  it("registerVizComponent attaches a component and getVizComponent returns it", () => {
    const FakeComponent = vi.fn();
    registerVizComponent("0", "data", FakeComponent as unknown as React.LazyExoticComponent<never>);
    const result = getVizComponent("0", "data");
    expect(result).toBe(FakeComponent);
    // Cleanup: detach so other tests are not affected
    registerVizComponent("0", "data", null);
  });

  it("getVizComponent returns null after component is unregistered", () => {
    const FakeComponent = vi.fn();
    registerVizComponent("0", "correlogram_results", FakeComponent as unknown as React.LazyExoticComponent<never>);
    registerVizComponent("0", "correlogram_results", null);
    expect(getVizComponent("0", "correlogram_results")).toBeNull();
  });
});
