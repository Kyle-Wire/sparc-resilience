/**
 * DAGView — project state synchronisation tests.
 *
 * T4: When no project is loaded (server returns 400 "No project loaded"),
 *     the component must NOT show a red error banner — it should show a
 *     friendly empty/no-project message.
 *
 * T5: When projectStore.projectLoaded flips from false → true, the
 *     component must automatically call getDag() again to populate the
 *     graph.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor, act } from "@testing-library/react";
import React from "react";

// ---------------------------------------------------------------------------
// Environment stubs — must be set BEFORE importing anything that uses them
// ---------------------------------------------------------------------------

// Tauri APIs (pulled in transitively through api.ts / workspacePrefs / etc.)
vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn() }));
vi.mock("@tauri-apps/plugin-dialog", () => ({ open: vi.fn() }));
vi.mock("@tauri-apps/plugin-shell", () => ({ Command: { create: vi.fn() } }));

// ReactFlow uses ResizeObserver which jsdom doesn't provide
vi.stubGlobal("ResizeObserver", class {
  observe() {}
  unobserve() {}
  disconnect() {}
});

// Minimal WebSocket stub so PipelineProvider doesn't throw
class StubWS {
  onopen: (() => void) | null = null;
  onmessage: ((e: MessageEvent) => void) | null = null;
  onerror: ((e: Event) => void) | null = null;
  onclose: (() => void) | null = null;
  close() {}
  send() {}
}
vi.stubGlobal("WebSocket", StubWS);

// ---------------------------------------------------------------------------
// Imports
// ---------------------------------------------------------------------------

import DAGView from "./DagEditor";
import { PipelineProvider } from "@/hooks/PipelineProvider";
import { useProjectStore } from "@/stores/projectStore";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Build a fetch stub that returns 400 "No project loaded" for /dag and
 *  sensible fallbacks for everything else PipelineProvider polls. */
function makeNoProjectFetch() {
  return vi.fn(async (url: string) => {
    const u = String(url);
    if (u.includes("/dag")) {
      return {
        ok: false,
        status: 400,
        text: async () => "No project loaded",
        json: async () => ({}),
      } as unknown as Response;
    }
    // PipelineProvider polls /run/events on mount — return empty list
    return {
      ok: true,
      status: 200,
      text: async () => "",
      json: async () => ({ events: [], current_stage: null, is_running: false }),
    } as unknown as Response;
  });
}

/** Valid minimal DAG payload returned by a healthy server. */
const EMPTY_DAG = { nodes: [], edges: [] };

function renderDagView() {
  return render(
    React.createElement(
      PipelineProvider,
      { serverReady: false, children: null },
      React.createElement(DAGView),
    ),
  );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("DAGView — project state sync", () => {
  beforeEach(() => {
    // Reset project store to clean state before each test
    useProjectStore.setState({ projectLoaded: false, projectPath: null, rehydrating: false });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
    // Restore ResizeObserver and WebSocket stubs for subsequent tests
    vi.stubGlobal("ResizeObserver", class { observe() {} unobserve() {} disconnect() {} });
    vi.stubGlobal("WebSocket", StubWS);
  });

  // T4 -----------------------------------------------------------------------
  it("shows a no-project empty state (not a red error) when server returns 400 No project loaded", async () => {
    vi.stubGlobal("fetch", makeNoProjectFetch());

    renderDagView();

    // Wait for loadDag to complete (loading spinner disappears)
    await waitFor(() =>
      expect(screen.queryByText(/loading dag/i)).not.toBeInTheDocument(),
    );

    // Should NOT show a red error banner with the raw error text
    expect(screen.queryByText(/No project loaded/i)).not.toBeInTheDocument();

    // Should show a friendly empty/no-project indicator
    expect(screen.getByText(/no project/i)).toBeInTheDocument();
  });

  // T5 -----------------------------------------------------------------------
  it("reloads the DAG automatically when a project is opened", async () => {
    // Phase 1: no project — server returns 400
    const fetchSpy = makeNoProjectFetch();
    vi.stubGlobal("fetch", fetchSpy);

    renderDagView();

    // Wait for initial load attempt to finish
    await waitFor(() =>
      expect(screen.queryByText(/loading dag/i)).not.toBeInTheDocument(),
    );

    const callsBefore = fetchSpy.mock.calls.filter(([u]) => String(u).includes("/dag")).length;
    expect(callsBefore).toBe(1); // one call on mount

    // Phase 2: project loads — server now returns a valid (empty) DAG
    fetchSpy.mockImplementation(async (url: string) => {
      const u = String(url);
      if (u.includes("/dag")) {
        return {
          ok: true,
          status: 200,
          text: async () => "",
          json: async () => EMPTY_DAG,
        } as unknown as Response;
      }
      return {
        ok: true, status: 200,
        text: async () => "",
        json: async () => ({ events: [], current_stage: null, is_running: false }),
      } as unknown as Response;
    });

    // Flip the store flag — simulates a successful project open
    act(() => {
      useProjectStore.setState({ projectLoaded: true });
    });

    // DAG should reload — second /dag call should happen
    await waitFor(() => {
      const dagCalls = fetchSpy.mock.calls.filter(([u]) => String(u).includes("/dag"));
      expect(dagCalls.length).toBeGreaterThan(1);
    });
  });
});
