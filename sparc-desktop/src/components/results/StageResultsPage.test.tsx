/**
 * StageResultsPage integration tests.
 * Verifies stage-switching via PipelineNavigator and default stage rendering.
 * Stage page components are mocked to isolate the container's own behaviour.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";

// Mock all stage pages — smoke tests for these live in stage-pages.test.tsx
vi.mock("./stages/Stage0Page", () => ({ default: () => <div data-testid="stage-0-page" /> }));
vi.mock("./stages/Stage1Page", () => ({ default: () => <div data-testid="stage-1-page" /> }));
vi.mock("./stages/Stage2Page", () => ({ default: () => <div data-testid="stage-2-page" /> }));
vi.mock("./stages/Stage3Page", () => ({ default: () => <div data-testid="stage-3-page" /> }));
vi.mock("./stages/Stage4Page", () => ({ default: () => <div data-testid="stage-4-page" /> }));

import StageResultsPage from "./StageResultsPage";

describe("StageResultsPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders Stage 0 content by default", () => {
    render(<StageResultsPage />);
    expect(screen.getByTestId("stage-0-page")).toBeInTheDocument();
  });

  it("does not render other stage pages on initial mount", () => {
    render(<StageResultsPage />);
    expect(screen.queryByTestId("stage-1-page")).not.toBeInTheDocument();
    expect(screen.queryByTestId("stage-2-page")).not.toBeInTheDocument();
  });

  it("renders the pipeline navigator", () => {
    render(<StageResultsPage />);
    expect(screen.getByRole("navigation", { name: /pipeline stages/i })).toBeInTheDocument();
  });

  it("shows stage 1 content after clicking the stage 1 navigator node", () => {
    render(<StageResultsPage />);
    // The stage 1 node label is "1 · Selection"
    fireEvent.click(screen.getByRole("button", { name: /selection/i }));
    expect(screen.getByTestId("stage-1-page")).toBeInTheDocument();
    expect(screen.queryByTestId("stage-0-page")).not.toBeInTheDocument();
  });

  it("shows stage 3 content after clicking the stage 3 navigator node", () => {
    render(<StageResultsPage />);
    fireEvent.click(screen.getByRole("button", { name: /causation/i }));
    expect(screen.getByTestId("stage-3-page")).toBeInTheDocument();
  });

  it("marks the active stage as active in the navigator", () => {
    render(<StageResultsPage />);
    const geometryBtn = screen.getByRole("button", { name: /geometry/i });
    expect(geometryBtn).toHaveAttribute("data-status", "active");
  });

  it("marks the previously active stage as complete after switching", () => {
    render(<StageResultsPage />);
    fireEvent.click(screen.getByRole("button", { name: /selection/i }));
    const geometryBtn = screen.getByRole("button", { name: /geometry/i });
    expect(geometryBtn).toHaveAttribute("data-status", "complete");
  });
});
