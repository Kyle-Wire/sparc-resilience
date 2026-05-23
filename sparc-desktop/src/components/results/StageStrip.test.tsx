/**
 * StageStrip tests.
 * Verifies show/hide based on the `visible` prop (controlled by the parent
 * via an IntersectionObserver watching the hero).  The strip is purely
 * presentational — observation logic lives outside it.
 */
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import StageStrip from "./StageStrip";

describe("StageStrip", () => {
  it("is not visible when visible=false (hero still on screen)", () => {
    render(<StageStrip label="Stage 0 · Geometry" visible={false} />);
    expect(screen.getByText("Stage 0 · Geometry")).not.toBeVisible();
  });

  it("is visible when visible=true (hero has scrolled out of view)", () => {
    render(<StageStrip label="Stage 0 · Geometry" visible={true} />);
    expect(screen.getByText("Stage 0 · Geometry")).toBeVisible();
  });

  it("always keeps the label in the DOM regardless of visibility", () => {
    render(<StageStrip label="Stage 3 · Causation" visible={false} />);
    expect(screen.getByText("Stage 3 · Causation")).toBeInTheDocument();
  });

  it("updates visibility reactively when visible prop changes", () => {
    const { rerender } = render(
      <StageStrip label="Stage 0 · Geometry" visible={false} />,
    );
    expect(screen.getByText("Stage 0 · Geometry")).not.toBeVisible();

    rerender(<StageStrip label="Stage 0 · Geometry" visible={true} />);
    expect(screen.getByText("Stage 0 · Geometry")).toBeVisible();
  });
});
