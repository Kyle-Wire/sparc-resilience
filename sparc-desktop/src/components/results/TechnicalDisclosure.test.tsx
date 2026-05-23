/**
 * TechnicalDisclosure — collapsible "What this means technically" section.
 * Tests verify open/close behavior and aria state through the public interface.
 */
import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import TechnicalDisclosure from "./TechnicalDisclosure";

describe("TechnicalDisclosure", () => {
  it("starts collapsed — children are in DOM but not visible", () => {
    render(
      <TechnicalDisclosure>
        <span>secret content</span>
      </TechnicalDisclosure>,
    );
    expect(screen.getByText("secret content")).not.toBeVisible();
  });

  it("shows the default toggle label", () => {
    render(
      <TechnicalDisclosure>
        <span>x</span>
      </TechnicalDisclosure>,
    );
    expect(screen.getByRole("button")).toHaveTextContent("What this means technically");
  });

  it("accepts a custom label", () => {
    render(
      <TechnicalDisclosure label="Correlogram & Matérn fit details">
        <span>x</span>
      </TechnicalDisclosure>,
    );
    expect(screen.getByRole("button")).toHaveTextContent("Correlogram & Matérn fit details");
  });

  it("marks button aria-expanded=false when collapsed", () => {
    render(
      <TechnicalDisclosure>
        <span>x</span>
      </TechnicalDisclosure>,
    );
    expect(screen.getByRole("button")).toHaveAttribute("aria-expanded", "false");
  });

  it("expands on click — children become visible", () => {
    render(
      <TechnicalDisclosure>
        <span>secret content</span>
      </TechnicalDisclosure>,
    );
    fireEvent.click(screen.getByRole("button"));
    expect(screen.getByText("secret content")).toBeVisible();
  });

  it("marks button aria-expanded=true when expanded", () => {
    render(
      <TechnicalDisclosure>
        <span>x</span>
      </TechnicalDisclosure>,
    );
    fireEvent.click(screen.getByRole("button"));
    expect(screen.getByRole("button")).toHaveAttribute("aria-expanded", "true");
  });

  it("collapses again on second click — children no longer visible", () => {
    render(
      <TechnicalDisclosure>
        <span>secret content</span>
      </TechnicalDisclosure>,
    );
    const btn = screen.getByRole("button");
    fireEvent.click(btn);
    fireEvent.click(btn);
    expect(screen.getByText("secret content")).not.toBeVisible();
  });

  it("always renders children in the DOM (never conditionally unmounts)", () => {
    render(
      <TechnicalDisclosure>
        <canvas data-testid="my-canvas" />
      </TechnicalDisclosure>,
    );
    // Canvas must be mounted even while collapsed so WebGL/Canvas contexts can initialise
    expect(screen.getByTestId("my-canvas")).toBeInTheDocument();
  });
});
