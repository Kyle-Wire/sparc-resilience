/**
 * pipelineReducer — pure event-processing logic for PipelineProvider.
 *
 * These tests verify behavior through the public interface of the reducer:
 * given a state and an event, what state comes back and is the socket terminal?
 * No React, no WebSocket, no mocks needed.
 */
import { describe, it, expect } from "vitest";
import {
  processPipelineEvent,
  initialPipelineReducerState,
} from "./pipelineReducer";
import type { PipelineEvent } from "@/lib/types";

function event(overrides: Partial<PipelineEvent>): PipelineEvent {
  return { type: "log", ...overrides };
}

describe("processPipelineEvent", () => {
  // ── Tracer bullet ───────────────────────────────────────────────────────
  it("returns unchanged state for a plain log event", () => {
    const state = initialPipelineReducerState();
    const { next, terminal } = processPipelineEvent(state, event({ type: "log", message: "hello" }));
    expect(terminal).toBe(false);
    expect(next.dagApprovalPending).toBe(false);
    expect(next.training.capacityResults).toHaveLength(0);
  });

  // ── Training telemetry ──────────────────────────────────────────────────
  it("appends a capacity_result to training.capacityResults", () => {
    const state = initialPipelineReducerState();
    const { next } = processPipelineEvent(
      state,
      event({ type: "capacity_result", hidden_dim: 64, r2: 0.85 }),
    );
    expect(next.training.capacityResults).toHaveLength(1);
    expect(next.training.capacityResults[0]).toEqual({ hidden_dim: 64, r2: 0.85 });
  });

  it("appends epoch_update to epochHistory", () => {
    const state = initialPipelineReducerState();
    const { next } = processPipelineEvent(
      state,
      event({ type: "epoch_update", epoch: 1, n_epochs: 100, total_loss: 0.5, train_phase: "cv" }),
    );
    expect(next.training.epochHistory).toHaveLength(1);
    expect(next.training.epochHistory[0].total_loss).toBe(0.5);
  });

  it("accumulates multiple epoch_updates immutably", () => {
    let state = initialPipelineReducerState();
    for (let i = 1; i <= 3; i++) {
      const { next } = processPipelineEvent(
        state,
        event({ type: "epoch_update", epoch: i, n_epochs: 10, total_loss: i * 0.1, train_phase: "cv" }),
      );
      state = next;
    }
    expect(state.training.epochHistory).toHaveLength(3);
    expect(state.training.epochHistory[2].epoch).toBe(3);
  });

  it("sets curriculumStage on curriculum_stage event", () => {
    const { next } = processPipelineEvent(
      initialPipelineReducerState(),
      event({ type: "curriculum_stage", curriculum: "Stage B", label: "Fine-tune" }),
    );
    expect(next.training.curriculumStage).toBe("Stage B");
    expect(next.training.curriculumLabel).toBe("Fine-tune");
  });

  it("sets convergenceStatus on convergence event", () => {
    const { next } = processPipelineEvent(
      initialPipelineReducerState(),
      event({ type: "convergence", status: "converged" }),
    );
    expect(next.training.convergenceStatus).toBe("converged");
  });

  it("appends training_health warning", () => {
    const { next } = processPipelineEvent(
      initialPipelineReducerState(),
      event({ type: "training_health", warning: "loss_explosion", component: "pde", detail: "gradient spike" }),
    );
    expect(next.training.healthWarnings).toHaveLength(1);
    expect(next.training.healthWarnings[0].warning).toBe("loss_explosion");
  });

  // ── Stage tracking ──────────────────────────────────────────────────────
  it("updates stageStatuses on stage_status event", () => {
    const { next } = processPipelineEvent(
      initialPipelineReducerState(),
      event({ type: "stage_status", stage: 2, status: "running" }),
    );
    expect(next.stageStatuses[2].status).toBe("running");
  });

  it("updates stageProgress when run_progress event is received", () => {
    const { next } = processPipelineEvent(
      initialPipelineReducerState(),
      event({ type: "run_progress", stage: 1, pct: 42 }),
    );
    expect(next.stageProgress[1]).toBe(42);
  });

  it("updates currentStage when stage field is present", () => {
    const { next } = processPipelineEvent(
      initialPipelineReducerState(),
      event({ type: "log", stage: 3 }),
    );
    expect(next.currentStage).toBe(3);
  });

  // ── Approval gates ──────────────────────────────────────────────────────
  it("sets dagApprovalPending on dag_approval_requested", () => {
    const { next } = processPipelineEvent(
      initialPipelineReducerState(),
      event({ type: "dag_approval_requested" }),
    );
    expect(next.dagApprovalPending).toBe(true);
  });

  it("sets gwenApprovalPending on gwen_approval_requested", () => {
    const { next } = processPipelineEvent(
      initialPipelineReducerState(),
      event({ type: "gwen_approval_requested" as PipelineEvent["type"] }),
    );
    expect(next.gwenApprovalPending).toBe(true);
  });

  // ── Terminal signals ────────────────────────────────────────────────────
  it("signals terminal=true on complete event", () => {
    const { terminal, errorMsg } = processPipelineEvent(
      initialPipelineReducerState(),
      event({ type: "complete" }),
    );
    expect(terminal).toBe(true);
    expect(errorMsg).toBeNull();
  });

  it("signals terminal=true and captures errorMsg on error event", () => {
    const { terminal, errorMsg } = processPipelineEvent(
      initialPipelineReducerState(),
      event({ type: "error", message: "CUDA OOM" }),
    );
    expect(terminal).toBe(true);
    expect(errorMsg).toBe("CUDA OOM");
  });

  it("error event without message defaults to 'Unknown error'", () => {
    const { errorMsg } = processPipelineEvent(
      initialPipelineReducerState(),
      event({ type: "error" }),
    );
    expect(errorMsg).toBe("Unknown error");
  });

  // ── Immutability ────────────────────────────────────────────────────────
  it("does not mutate the input state", () => {
    const state = initialPipelineReducerState();
    const before = JSON.stringify(state);
    processPipelineEvent(state, event({ type: "capacity_result", hidden_dim: 32, r2: 0.7 }));
    expect(JSON.stringify(state)).toBe(before);
  });
});
