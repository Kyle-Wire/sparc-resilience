// Tooltips (components/ui/Tooltip.tsx), InfoTip (ⓘ) and Term: open on focus at once and on hover
// after a short delay, toggle on tap, close on Escape, role="tooltip" + aria-describedby, rendered
// in a portal on <body>, placed above or below the anchor and clamped into the viewport.
import { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { term } from "../content/glossary";
import { InfoTip } from "../components/ui/InfoTip";
import { Term } from "../components/ui/Term";
import { TOOLTIP_OPEN_DELAY_MS, Tooltip, placeTooltip } from "../components/ui/Tooltip";
import { click, key, render } from "./render";

const tooltip = () => document.querySelector<HTMLElement>('[role="tooltip"]:not([hidden])');

function focus(el: HTMLElement) {
  act(() => el.focus());
}
function blur(el: HTMLElement) {
  act(() => el.blur());
}
function hover(el: Element) {
  act(() => {
    el.dispatchEvent(new MouseEvent("mouseover", { bubbles: true, relatedTarget: null }));
  });
}
function unhover(el: Element) {
  act(() => {
    el.dispatchEvent(new MouseEvent("mouseout", { bubbles: true, relatedTarget: null }));
  });
}
function tap(el: Element) {
  act(() => {
    const down = typeof PointerEvent === "function" ? new PointerEvent("pointerdown", { bubbles: true, pointerType: "touch" }) : Object.assign(new MouseEvent("pointerdown", { bubbles: true }), { pointerType: "touch" });
    if (typeof PointerEvent !== "function") Object.defineProperty(down, "pointerType", { value: "touch" });
    el.dispatchEvent(down);
  });
  click(el);
}

const rect = (left: number, top: number, width: number, height: number): DOMRect =>
  ({ left, top, width, height, right: left + width, bottom: top + height, x: left, y: top, toJSON: () => ({}) }) as DOMRect;

/** Mock layout: the bubble is 200×60, every other element sits at `anchor`. */
function mockLayout(anchor: DOMRect) {
  return vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(function (this: HTMLElement) {
    return this.classList.contains("tip-bubble") ? rect(0, 0, 200, 60) : anchor;
  });
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.useRealTimers();
});

describe("InfoTip", () => {
  it("is an ⓘ button named after the term, with no tooltip until asked", () => {
    const { container } = render(<InfoTip term="coverage" />);
    const btn = container.querySelector("button")!;
    expect(btn.getAttribute("aria-label")).toBe("What is Range reliability?");
    expect(btn.getAttribute("type")).toBe("button");
    expect(btn.hasAttribute("aria-describedby")).toBe(false);
    expect(btn.textContent).toBe(""); // the icon only: no tip text in the page's text
    expect(document.querySelector('[role="tooltip"]')).toBeNull();
  });

  it("opens on focus at once, in a portal, describing the button; Escape closes it", () => {
    const { container } = render(<InfoTip term="coverage" />);
    const btn = container.querySelector("button")!;
    focus(btn);
    const tip = tooltip()!;
    expect(tip).not.toBeNull();
    expect(container.contains(tip)).toBe(false); // on <body>, so no card can clip it
    expect(tip.parentElement).toBe(document.body);
    expect(btn.getAttribute("aria-describedby")).toBe(tip.id);
    expect(tip.querySelector(".tip-text")!.textContent).toBe(term("coverage").tip);
    expect(tip.querySelector(".tip-tech")!.textContent).toBe("Technical: 90% interval coverage");
    expect(tip.querySelector(".tip-tech code")!.textContent).toBe("90% interval coverage");
    key(btn, "Escape");
    expect(tooltip()).toBeNull();
    expect(btn.hasAttribute("aria-describedby")).toBe(false);
    // focus again after leaving: open again
    blur(btn);
    focus(btn);
    expect(tooltip()).not.toBeNull();
    blur(btn);
    expect(tooltip()).toBeNull();
  });

  it("opens on hover after the delay, closes when the pointer leaves, and on Escape anywhere", () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    const { container } = render(<InfoTip term="r2" />);
    const btn = container.querySelector("button")!;
    hover(btn);
    act(() => vi.advanceTimersByTime(TOOLTIP_OPEN_DELAY_MS - 1));
    expect(tooltip()).toBeNull(); // a pointer passing over does not flash a tooltip
    act(() => vi.advanceTimersByTime(1));
    expect(tooltip()!.textContent).toContain(term("r2").tip);
    unhover(btn);
    expect(tooltip()).toBeNull();
    // leaving before the delay never opens it
    hover(btn);
    unhover(btn);
    act(() => vi.advanceTimersByTime(TOOLTIP_OPEN_DELAY_MS * 2));
    expect(tooltip()).toBeNull();
    // Escape on the page closes a hovered tooltip
    hover(btn);
    act(() => vi.advanceTimersByTime(TOOLTIP_OPEN_DELAY_MS));
    expect(tooltip()).not.toBeNull();
    key(document.body, "Escape");
    expect(tooltip()).toBeNull();
  });

  it("toggles on tap (touch), and a tap elsewhere closes it", () => {
    const { container } = render(
      <div>
        <InfoTip term="gini" />
        <p id="elsewhere">text</p>
      </div>,
    );
    const btn = container.querySelector("button")!;
    tap(btn);
    expect(tooltip()!.textContent).toContain(term("gini").tip);
    tap(btn);
    expect(tooltip()).toBeNull();
    tap(btn);
    expect(tooltip()).not.toBeNull();
    act(() => {
      container.querySelector("#elsewhere")!.dispatchEvent(new MouseEvent("pointerdown", { bubbles: true }));
    });
    expect(tooltip()).toBeNull();
  });

  it("takes a one-off text, label and technical override; clicks do not reach the control around it", () => {
    const { container } = render(
      <details>
        <summary>
          Ranges <InfoTip label="this range" text="From 12 test areas." technical={null} />
        </summary>
        body
      </details>,
    );
    const btn = container.querySelector("button")!;
    expect(btn.getAttribute("aria-label")).toBe("What is this range?");
    click(btn);
    expect(container.querySelector("details")!.open).toBe(false);
    focus(btn);
    expect(tooltip()!.textContent).toBe("From 12 test areas.");
    expect(tooltip()!.querySelector(".tip-tech")).toBeNull();
  });

  it("throws for an unknown term (a typo fails the test)", () => {
    const err = vi.spyOn(console, "error").mockImplementation(() => {});
    expect(() => render(<InfoTip term={"not_a_term" as "coverage"} />)).toThrow(/unknown term/);
    err.mockRestore();
  });
});

describe("Term", () => {
  it("shows the plain label inline with a dotted underline and the same tooltip", () => {
    const { container } = render(
      <p>
        The <Term id="coverage" /> is good.
      </p>,
    );
    const t = container.querySelector<HTMLElement>(".term")!;
    expect(t.tagName).toBe("BUTTON");
    expect(t.textContent).toBe("Range reliability");
    expect(container.querySelector("p")!.textContent).toBe("The Range reliability is good.");
    focus(t);
    expect(t.getAttribute("aria-describedby")).toBe(tooltip()!.id);
    expect(tooltip()!.querySelector(".tip-tech")!.textContent).toBe("Technical: 90% interval coverage");
    key(t, "Escape");
    expect(tooltip()).toBeNull();
  });

  it("keeps the children as the visible words", () => {
    const { container } = render(<Term id="interval_95">likely range</Term>);
    expect(container.querySelector(".term")!.textContent).toBe("likely range");
  });
});

describe("placement", () => {
  const vp = { width: 1024, height: 768 };
  const box = (left: number, top: number, width = 18, height = 18) => ({ left, top, width, bottom: top + height });

  it("goes above when it fits, else below; centred and clamped into the viewport", () => {
    expect(placeTooltip(box(500, 400), { width: 200, height: 60 }, vp)).toEqual({ top: 334, left: 409, side: "top", arrowX: 100 });
    expect(placeTooltip(box(500, 4), { width: 200, height: 60 }, vp)).toMatchObject({ top: 28, side: "bottom" });
    expect(placeTooltip(box(500, 740), { width: 200, height: 60 }, vp, "bottom")).toMatchObject({ side: "top" });
    expect(placeTooltip(box(1010, 400), { width: 200, height: 60 }, vp)).toMatchObject({ left: 816, arrowX: 190 });
    expect(placeTooltip(box(0, 400), { width: 200, height: 60 }, vp)).toMatchObject({ left: 8, arrowX: 10 });
    // too tall for either side: the roomier one, kept on screen
    const tall = placeTooltip(box(500, 300), { width: 200, height: 700 }, vp);
    expect(tall.side).toBe("bottom");
    expect(tall.top).toBe(60); // its bottom edge 8 px above the viewport's
  });

  it("flips below an anchor near the top edge (mocked layout)", () => {
    mockLayout(rect(500, 4, 18, 18));
    const { container } = render(<InfoTip term="coverage" />);
    focus(container.querySelector("button")!);
    const tip = tooltip()!;
    expect(tip.getAttribute("data-side")).toBe("bottom");
    expect(tip.style.top).toBe("28px");
    expect(tip.style.left).toBe("409px");
    expect(tip.style.visibility).toBe("");
  });

  it("stays above an anchor with room, and clamps at the right edge", () => {
    mockLayout(rect(1010, 400, 18, 18));
    const { container } = render(<InfoTip term="coverage" />);
    focus(container.querySelector("button")!);
    const tip = tooltip()!;
    expect(tip.getAttribute("data-side")).toBe("top");
    expect(tip.style.top).toBe("334px");
    expect(tip.style.left).toBe("816px");
    expect(tip.style.getPropertyValue("--tip-arrow-x")).toBe("190px");
  });
});

describe("Tooltip (wrapper API)", () => {
  beforeEach(() => {
    vi.useRealTimers();
  });

  it("keeps a hidden bubble in place while closed and moves it to <body> while open", () => {
    const { container } = render(
      <Tooltip content="Needs a run with a checkpoint">
        <span tabIndex={0}>Scenario Lab</span>
      </Tooltip>,
    );
    const anchor = container.querySelector<HTMLElement>(".tip-anchor")!;
    const inPlace = container.querySelector<HTMLElement>('[role="tooltip"]')!;
    expect(inPlace.hasAttribute("hidden")).toBe(true);
    expect(anchor.getAttribute("aria-describedby")).toBe(inPlace.id);
    focus(container.querySelector<HTMLElement>("[tabindex]")!);
    expect(container.querySelector('[role="tooltip"]')).toBeNull();
    const open = tooltip()!;
    expect(open.parentElement).toBe(document.body);
    expect(open.textContent).toBe("Needs a run with a checkpoint");
    expect(anchor.getAttribute("aria-describedby")).toBe(open.id);
    key(container.querySelector("[tabindex]"), "Escape");
    expect(tooltip()).toBeNull();
    expect(container.querySelector('[role="tooltip"]')!.hasAttribute("hidden")).toBe(true);
  });

  it("Escape on the trigger closes the tooltip, not the dialog around it", () => {
    const onKey = vi.fn();
    const { container } = render(
      <div onKeyDown={onKey}>
        <Tooltip content="tip">
          <button type="button">x</button>
        </Tooltip>
      </div>,
    );
    const b = container.querySelector("button")!;
    focus(b);
    key(b, "Escape");
    expect(tooltip()).toBeNull();
    expect(onKey).not.toHaveBeenCalled();
    key(b, "Escape"); // closed already: Escape goes on to the dialog
    expect(onKey).toHaveBeenCalledTimes(1);
  });
});
