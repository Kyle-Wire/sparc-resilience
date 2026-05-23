/**
 * Stage page smoke tests.
 * Verifies that each stage page (0–4) renders without crashing.
 * No external panel dependencies to mock since the initial stage page
 * implementations use only NarrativeCallout and TechnicalDisclosure.
 */
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";

import Stage0Page from "./stages/Stage0Page";
import Stage1Page from "./stages/Stage1Page";
import Stage2Page from "./stages/Stage2Page";
import Stage3Page from "./stages/Stage3Page";
import Stage4Page from "./stages/Stage4Page";

describe("Stage page smoke tests — each stage renders without crashing", () => {
  it("Stage0Page renders with an accessible article landmark", () => {
    render(<Stage0Page />);
    expect(screen.getByRole("article", { name: /stage 0/i })).toBeInTheDocument();
  });

  it("Stage0Page renders its narrative callout headline", () => {
    render(<Stage0Page />);
    expect(screen.getByText("Spatial autocorrelation")).toBeInTheDocument();
  });

  it("Stage0Page renders the TechnicalDisclosure toggle", () => {
    render(<Stage0Page />);
    expect(screen.getByRole("button", { name: /matérn/i })).toBeInTheDocument();
  });

  it("Stage1Page renders with an accessible article landmark", () => {
    render(<Stage1Page />);
    expect(screen.getByRole("article", { name: /stage 1/i })).toBeInTheDocument();
  });

  it("Stage1Page renders its narrative callout headline", () => {
    render(<Stage1Page />);
    expect(screen.getByText("Variable importance")).toBeInTheDocument();
  });

  it("Stage2Page renders with an accessible article landmark", () => {
    render(<Stage2Page />);
    expect(screen.getByRole("article", { name: /stage 2/i })).toBeInTheDocument();
  });

  it("Stage2Page renders its narrative callout headline", () => {
    render(<Stage2Page />);
    expect(screen.getByText("Spatial predictions")).toBeInTheDocument();
  });

  it("Stage3Page renders with an accessible article landmark", () => {
    render(<Stage3Page />);
    expect(screen.getByRole("article", { name: /stage 3/i })).toBeInTheDocument();
  });

  it("Stage3Page renders its narrative callout headline", () => {
    render(<Stage3Page />);
    expect(screen.getByText("Dose-response")).toBeInTheDocument();
  });

  it("Stage4Page renders with an accessible article landmark", () => {
    render(<Stage4Page />);
    expect(screen.getByRole("article", { name: /stage 4/i })).toBeInTheDocument();
  });

  it("Stage4Page renders its narrative callout headline", () => {
    render(<Stage4Page />);
    expect(screen.getByText("Best intervention")).toBeInTheDocument();
  });
});
