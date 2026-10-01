// Focus helpers for modal surfaces (dialog, drawer, command palette).
import type { KeyboardEvent as ReactKeyboardEvent } from "react";

const FOCUSABLE =
  'a[href], button:not([disabled]), input:not([disabled]):not([type="hidden"]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

export function focusables(root: HTMLElement): HTMLElement[] {
  return [...root.querySelectorAll<HTMLElement>(FOCUSABLE)].filter((el) => !el.hasAttribute("hidden") && el.getAttribute("aria-hidden") !== "true");
}

/** Keep Tab / Shift-Tab inside `root`. Call from a keydown handler. */
export function trapTab(e: KeyboardEvent | ReactKeyboardEvent, root: HTMLElement | null): void {
  if (e.key !== "Tab" || !root) return;
  const els = focusables(root);
  if (!els.length) {
    e.preventDefault();
    return;
  }
  const first = els[0];
  const last = els[els.length - 1];
  const active = document.activeElement as HTMLElement | null;
  if (e.shiftKey && (active === first || !root.contains(active))) {
    e.preventDefault();
    last.focus();
  } else if (!e.shiftKey && (active === last || !root.contains(active))) {
    e.preventDefault();
    first.focus();
  }
}
