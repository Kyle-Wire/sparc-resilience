/**
 * CommandPalette — focus trap tests.
 *
 * The palette is a modal-like overlay; focus must not escape to the page
 * behind it. Tests verify:
 *   1. The palette root has role=dialog + aria-modal="true"
 *   2. The search input receives focus when the palette opens
 *   3. Tab while the palette is open does not move focus outside it
 */
import { describe, it, expect } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import CommandPalette, { type PaletteItem } from "./CommandPalette";

const noop = () => {};

const ITEMS: PaletteItem[] = [
  { id: "p1", label: "Project", kind: "page", page: "Project" },
  { id: "p2", label: "Run",     kind: "page", page: "Run"     },
];

describe("CommandPalette — focus trap", () => {
  it("has role=dialog and aria-modal on the palette container", () => {
    render(
      <CommandPalette open={true} items={ITEMS} onClose={noop} onNavigate={noop} />,
    );
    const dialog = screen.getByRole("dialog");
    expect(dialog).toBeInTheDocument();
    expect(dialog).toHaveAttribute("aria-modal", "true");
  });

  it("focuses the search input when the palette opens", async () => {
    render(
      <CommandPalette open={true} items={ITEMS} onClose={noop} onNavigate={noop} />,
    );
    const input = screen.getByRole("textbox");
    await waitFor(() => expect(input).toHaveFocus());
  });

  it("Tab does not move focus outside the palette", async () => {
    const user = userEvent.setup();
    render(
      <>
        <button>outside</button>
        <CommandPalette open={true} items={ITEMS} onClose={noop} onNavigate={noop} />
      </>,
    );

    const input = screen.getByRole("textbox");
    await waitFor(() => expect(input).toHaveFocus());

    await user.tab();

    // Focus must remain within the palette, not on the outside button
    expect(screen.getByRole("button", { name: "outside" })).not.toHaveFocus();
  });

  it("Shift+Tab does not move focus outside the palette", async () => {
    const user = userEvent.setup();
    render(
      <>
        <button>outside</button>
        <CommandPalette open={true} items={ITEMS} onClose={noop} onNavigate={noop} />
      </>,
    );

    const input = screen.getByRole("textbox");
    await waitFor(() => expect(input).toHaveFocus());

    await user.tab({ shift: true });

    expect(screen.getByRole("button", { name: "outside" })).not.toHaveFocus();
  });
});
