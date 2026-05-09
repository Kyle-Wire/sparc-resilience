/**
 * T46 — PipelineProvider state transition tests
 *
 * Tests the processEvent callback logic by calling usePipeline after
 * wrapping in PipelineProvider with serverReady=true.
 *
 * External dependencies (WebSocket, api, tokenStore, server) are mocked.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, act } from "@testing-library/react";
import React from "react";

// ── Mocks ────────────────────────────────────────────────────────────────────

vi.mock("@/lib/api", () => ({
  getRunEvents: vi.fn(async () => ({ events: [], current_stage: null, is_running: false })),
  approveDag: vi.fn(async () => {}),
  rejectDag: vi.fn(async () => {}),
  cancelRun: vi.fn(async () => {}),
}));

vi.mock("@/lib/server", () => ({
  WS_ORIGIN: "ws://127.0.0.1:8008",
  SERVER_ORIGIN: "http://127.0.0.1:8008",
}));

vi.mock("@/stores/tokenStore", () => ({ getToken: () => "test-token" }));
vi.mock("@/hooks/useManifest", () => ({ notifyManifestArtifactWritten: vi.fn() }));

// Minimal WebSocket stub — never fires events automatically
class StubWebSocket {
  onopen: (() => void) | null = null;
  onmessage: ((e: MessageEvent) => void) | null = null;
  onerror: ((e: Event) => void) | null = null;
  onclose: (() => void) | null = null;
  close() {}
  send() {}
}

// ── System under test ────────────────────────────────────────────────────────

import { PipelineProvider, usePipeline } from "@/hooks/PipelineProvider";

// Helper: capture pipeline context value
function capturePipeline() {
  let ctx: ReturnType<typeof usePipeline> | null = null;

  function Probe() {
    ctx = usePipeline();
    return null;
  }

  function Wrapper() {
    return React.createElement(
      PipelineProvider,
      { serverReady: true, children: React.createElement(Probe) },
    );
  }

  render(React.createElement(Wrapper));
  return () => ctx!;
}

describe("PipelineProvider", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.stubGlobal("WebSocket", StubWebSocket);
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("starts with isRunning=false and empty events", () => {
    const getCtx = capturePipeline();
    expect(getCtx().isRunning).toBe(false);
    expect(getCtx().events).toEqual([]);
  });

  it("sets isRunning=true when startStage is called", async () => {
    const getCtx = capturePipeline();
    await act(async () => {
      getCtx().startStage(0);
    });
    expect(getCtx().isRunning).toBe(true);
  });

  it("resets isRunning and clears error on cancel", async () => {
    const getCtx = capturePipeline();
    await act(async () => {
      getCtx().startStage(0);
    });
    act(() => getCtx().cancel());
    expect(getCtx().isRunning).toBe(false);
    expect(getCtx().error).toBeNull();
  });

  it("dagApprovalPending starts as false", () => {
    const getCtx = capturePipeline();
    expect(getCtx().dagApprovalPending).toBe(false);
  });
});
