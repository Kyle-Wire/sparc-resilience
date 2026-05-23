/**
 * NarrativeCallout tests.
 * Verifies skeleton state, AI text rendering, and fallback text — all through
 * the public props interface.
 */
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import NarrativeCallout from "./NarrativeCallout";

describe("NarrativeCallout", () => {
  it("always renders the headline", () => {
    render(<NarrativeCallout headline="Spatial geometry of your data" />);
    expect(screen.getByText("Spatial geometry of your data")).toBeInTheDocument();
  });

  it("shows the skeleton loader while loading", () => {
    render(<NarrativeCallout headline="Correlogram" loading />);
    expect(screen.getByTestId("narrative-skeleton")).toBeInTheDocument();
  });

  it("does not show body text while loading", () => {
    render(<NarrativeCallout headline="Correlogram" body="Spatial autocorrelation decays at 4 km" loading />);
    expect(screen.queryByText("Spatial autocorrelation decays at 4 km")).not.toBeInTheDocument();
  });

  it("renders body text when provided and not loading", () => {
    render(
      <NarrativeCallout
        headline="Correlogram"
        body="Spatial autocorrelation decays at 4 km"
      />,
    );
    expect(screen.getByText("Spatial autocorrelation decays at 4 km")).toBeInTheDocument();
  });

  it("renders fallback text when no body is provided", () => {
    render(
      <NarrativeCallout
        headline="Correlogram"
        fallback="Configure a Claude API key to generate narrative insights."
      />,
    );
    expect(
      screen.getByText("Configure a Claude API key to generate narrative insights."),
    ).toBeInTheDocument();
  });

  it("prefers body over fallback when both are provided", () => {
    render(
      <NarrativeCallout
        headline="Correlogram"
        body="AI-generated insight"
        fallback="Fallback text"
      />,
    );
    expect(screen.getByText("AI-generated insight")).toBeInTheDocument();
    expect(screen.queryByText("Fallback text")).not.toBeInTheDocument();
  });

  it("renders nothing in the body area when neither body nor fallback is provided", () => {
    render(<NarrativeCallout headline="Correlogram" />);
    // No skeleton, no body text — just headline
    expect(screen.queryByTestId("narrative-skeleton")).not.toBeInTheDocument();
  });
});
