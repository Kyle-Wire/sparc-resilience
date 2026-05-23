/**
 * RunCompletionCard tests.
 *
 * Verifies the post-run CTA card: renders completion summary, elapsed time,
 * stage count, and fires the onViewResults callback when clicked.
 * The component is purely presentational — no hooks to mock.
 */
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import RunCompletionCard from "./RunCompletionCard";

describe("RunCompletionCard", () => {
  const baseProps = {
    elapsedSeconds: 272,
    stagesComplete: 5,
    totalStages: 5,
    onViewResults: vi.fn(),
  };

  it("renders a success heading", () => {
    render(<RunCompletionCard {...baseProps} />);
    expect(screen.getByRole("heading", { name: /run complete/i })).toBeInTheDocument();
  });

  it("shows the formatted elapsed time", () => {
    render(<RunCompletionCard {...baseProps} />);
    expect(screen.getByText(/04:32/)).toBeInTheDocument();
  });

  it("shows stages complete out of total", () => {
    render(<RunCompletionCard {...baseProps} />);
    expect(screen.getByText(/5\s*\/\s*5/)).toBeInTheDocument();
  });

  it("renders the View Results CTA button", () => {
    render(<RunCompletionCard {...baseProps} />);
    expect(screen.getByRole("button", { name: /view results/i })).toBeInTheDocument();
  });

  it("calls onViewResults when the CTA is clicked", () => {
    const onViewResults = vi.fn();
    render(<RunCompletionCard {...baseProps} onViewResults={onViewResults} />);
    fireEvent.click(screen.getByRole("button", { name: /view results/i }));
    expect(onViewResults).toHaveBeenCalledOnce();
  });

  it("shows an error count when there were stage failures", () => {
    render(<RunCompletionCard {...baseProps} stagesComplete={4} totalStages={5} errorCount={1} />);
    expect(screen.getByText(/1 stage failed/i)).toBeInTheDocument();
  });

  it("does not render an error count when no failures", () => {
    render(<RunCompletionCard {...baseProps} />);
    expect(screen.queryByText(/stage failed/i)).not.toBeInTheDocument();
  });
});
