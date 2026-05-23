/**
 * PipelineNavigator tests.
 * Verifies stage node rendering, active/locked states, and click emission
 * through the public props interface only.
 */
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import PipelineNavigator, { type StageNode } from "./PipelineNavigator";

const STAGES: StageNode[] = [
  { id: 0, label: "Geometry",   color: "#602468", status: "active"   },
  { id: 1, label: "Selection",  color: "#e94d9b", status: "complete" },
  { id: 2, label: "Prediction", color: "#e73c25", status: "locked"   },
  { id: 3, label: "Causation",  color: "#e79024", status: "locked"   },
  { id: 4, label: "Decision",   color: "#f0b632", status: "locked"   },
];

describe("PipelineNavigator", () => {
  it("renders a node for every stage provided", () => {
    render(<PipelineNavigator stages={STAGES} onStageClick={() => {}} />);
    // Each stage gets a button
    expect(screen.getAllByRole("button")).toHaveLength(STAGES.length);
  });

  it("marks the active stage node with data-status=active", () => {
    render(<PipelineNavigator stages={STAGES} onStageClick={() => {}} />);
    const activeBtn = screen.getByRole("button", { name: /geometry/i });
    expect(activeBtn).toHaveAttribute("data-status", "active");
  });

  it("marks locked stage nodes with data-status=locked", () => {
    render(<PipelineNavigator stages={STAGES} onStageClick={() => {}} />);
    const lockedBtns = screen
      .getAllByRole("button")
      .filter((b) => b.getAttribute("data-status") === "locked");
    expect(lockedBtns).toHaveLength(3);
  });

  it("marks complete stage nodes with data-status=complete", () => {
    render(<PipelineNavigator stages={STAGES} onStageClick={() => {}} />);
    const selectionBtn = screen.getByRole("button", { name: /selection/i });
    expect(selectionBtn).toHaveAttribute("data-status", "complete");
  });

  it("calls onStageClick with the stage id when an active stage is clicked", () => {
    const onStageClick = vi.fn();
    render(<PipelineNavigator stages={STAGES} onStageClick={onStageClick} />);
    fireEvent.click(screen.getByRole("button", { name: /geometry/i }));
    expect(onStageClick).toHaveBeenCalledWith(0);
  });

  it("calls onStageClick with the stage id when a complete stage is clicked", () => {
    const onStageClick = vi.fn();
    render(<PipelineNavigator stages={STAGES} onStageClick={onStageClick} />);
    fireEvent.click(screen.getByRole("button", { name: /selection/i }));
    expect(onStageClick).toHaveBeenCalledWith(1);
  });

  it("does not call onStageClick when a locked stage is clicked", () => {
    const onStageClick = vi.fn();
    render(<PipelineNavigator stages={STAGES} onStageClick={onStageClick} />);
    fireEvent.click(screen.getByRole("button", { name: /prediction/i }));
    expect(onStageClick).not.toHaveBeenCalled();
  });

  it("has aria-label for the nav landmark", () => {
    render(<PipelineNavigator stages={STAGES} onStageClick={() => {}} />);
    expect(screen.getByRole("navigation", { name: /pipeline stages/i })).toBeInTheDocument();
  });
});
