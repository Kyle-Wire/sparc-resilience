/**
 * Sidebar accessibility tests.
 *
 * Verifies that locked nav buttons (pages gated behind a loaded project) are
 * excluded from the keyboard tab order so screen-reader/keyboard users cannot
 * accidentally land on non-functional controls.
 */
import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import Sidebar from "./Sidebar";

const noop = () => {};

describe("Sidebar — tab order", () => {
  it("'Project' button is always in the tab order", () => {
    render(
      <Sidebar
        currentPage="Project"
        onNavigate={noop}
        projectLoaded={false}
      />,
    );
    const btn = screen.getByRole("button", { name: /\bProject\b/ });
    expect(btn).not.toHaveAttribute("tabindex", "-1");
  });

  it("locked page buttons have tabIndex={-1} when no project is loaded", () => {
    render(
      <Sidebar
        currentPage="Project"
        onNavigate={noop}
        projectLoaded={false}
      />,
    );
    // Spot-check one page from each section (Setup / Configure / Pipeline)
    const lockedPages = [/\bConfigure\b/, /\bRun\b/, /\bReport\b/];
    for (const name of lockedPages) {
      const btn = screen.getByRole("button", { name });
      expect(btn).toHaveAttribute("tabindex", "-1");
    }
  });

  it("all page buttons are in the tab order when a project is loaded", () => {
    render(
      <Sidebar
        currentPage="Project"
        onNavigate={noop}
        projectLoaded={true}
      />,
    );
    const pages = [/\bConfigure\b/, /\bRun\b/, /\bReport\b/];
    for (const name of pages) {
      const btn = screen.getByRole("button", { name });
      expect(btn).not.toHaveAttribute("tabindex", "-1");
    }
  });
});
