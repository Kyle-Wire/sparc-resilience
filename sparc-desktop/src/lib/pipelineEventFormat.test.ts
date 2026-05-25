/**
 * Tests for pipelineEventFormat — event-to-log-line mapping.
 *
 * Behavior: each PipelineEvent type maps to a predictable log line shape.
 * The function returns null for unknown or empty events.
 */
import { describe, it, expect } from "vitest";
import { eventToLogLine } from "./pipelineEventFormat";
import type { PipelineEvent } from "./types";

function makeEvent(type: string, extra: Record<string, unknown> = {}): PipelineEvent {
  return { type, ...extra } as unknown as PipelineEvent;
}

describe("eventToLogLine", () => {
  describe("stage_status events", () => {
    it("returns info line when stage is running", () => {
      const evt = makeEvent("stage_status", { stage: 0, status: "running" });
      const line = eventToLogLine(evt);
      expect(line).not.toBeNull();
      expect(line!.level).toBe("info");
      expect(line!.text).toMatch(/Correlogram/);
    });

    it("returns success line when stage is complete", () => {
      const evt = makeEvent("stage_status", { stage: 1, status: "complete" });
      const line = eventToLogLine(evt);
      expect(line!.level).toBe("success");
      expect(line!.text).toMatch(/GWEN/i);
    });

    it("includes elapsed seconds in success line when provided", () => {
      const evt = makeEvent("stage_status", {
        stage: 1,
        status: "complete",
        elapsed_seconds: 42.3,
      });
      const line = eventToLogLine(evt);
      expect(line!.text).toMatch(/42\.3s/);
    });

    it("returns error line when stage failed", () => {
      const evt = makeEvent("stage_status", {
        stage: 2,
        status: "failed",
        error: "OOM",
      });
      const line = eventToLogLine(evt);
      expect(line!.level).toBe("error");
      expect(line!.text).toMatch(/OOM/);
    });

    it("returns null for unknown stage status", () => {
      const evt = makeEvent("stage_status", { stage: 0, status: "pending" });
      const line = eventToLogLine(evt);
      expect(line).toBeNull();
    });
  });

  describe("epoch_update events", () => {
    it("returns debug line with epoch and loss info", () => {
      const evt = makeEvent("epoch_update", {
        epoch: 5,
        n_epochs: 50,
        total_loss: 0.1234,
      });
      const line = eventToLogLine(evt);
      expect(line!.level).toBe("debug");
      expect(line!.text).toMatch(/5 \/ 50/);
      expect(line!.text).toMatch(/0\.1234/);
    });
  });

  describe("metric events", () => {
    it("returns info line with metric name and value", () => {
      const evt = makeEvent("metric", { metric: "r2", value: 0.8765, fold: 2 });
      const line = eventToLogLine(evt);
      expect(line!.level).toBe("info");
      expect(line!.text).toMatch(/r2/);
      expect(line!.text).toMatch(/0\.8765/);
    });
  });

  describe("fold events", () => {
    it("returns info line for fold_start with counts", () => {
      const evt = makeEvent("fold_start", {
        fold: 1,
        n_folds: 5,
        n_train: 1000,
        n_test: 200,
      });
      const line = eventToLogLine(evt);
      expect(line!.level).toBe("info");
      expect(line!.text).toMatch(/1 \/ 5/);
    });

    it("returns success line for fold_complete", () => {
      const evt = makeEvent("fold_complete", { fold: 3, elapsed_seconds: 12 });
      const line = eventToLogLine(evt);
      expect(line!.level).toBe("success");
      expect(line!.text).toMatch(/Fold 3/);
    });
  });

  describe("model events", () => {
    it("returns success line for model_result with r2 and rmse", () => {
      const evt = makeEvent("model_result", {
        model: "gwr",
        r2: 0.91,
        rmse: 0.05,
      });
      const line = eventToLogLine(evt);
      expect(line!.level).toBe("success");
      expect(line!.text).toMatch(/gwr/i);
      expect(line!.text).toMatch(/0\.9100/);
    });
  });

  describe("terminal events", () => {
    it("returns success line for complete event", () => {
      const evt = makeEvent("complete");
      const line = eventToLogLine(evt);
      expect(line!.level).toBe("success");
    });

    it("returns error line for error event", () => {
      const evt = makeEvent("error", { message: "something broke" });
      const line = eventToLogLine(evt);
      expect(line!.level).toBe("error");
      expect(line!.text).toMatch(/something broke/);
    });
  });

  describe("log events", () => {
    it("returns null for empty log message", () => {
      const evt = makeEvent("log", { message: "   " });
      expect(eventToLogLine(evt)).toBeNull();
    });

    it("returns log line for non-empty message", () => {
      const evt = makeEvent("log", { message: "some output" });
      const line = eventToLogLine(evt);
      expect(line!.level).toBe("log");
      expect(line!.text).toBe("some output");
    });
  });

  describe("timestamp", () => {
    it("uses receivedAt when present", () => {
      const knownTime = new Date("2026-01-01T10:30:45Z").getTime();
      const evt = { ...makeEvent("complete"), receivedAt: knownTime };
      const line = eventToLogLine(evt as PipelineEvent & { receivedAt: number });
      // Time string should be HH:MM:SS format
      expect(line!.ts).toMatch(/^\d{2}:\d{2}:\d{2}$/);
    });

    it("falls back to current time when receivedAt is absent", () => {
      const line = eventToLogLine(makeEvent("complete"))!;
      expect(line.ts).toMatch(/^\d{2}:\d{2}:\d{2}$/);
    });
  });
});
