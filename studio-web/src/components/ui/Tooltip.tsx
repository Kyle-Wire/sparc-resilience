import {
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
  type CSSProperties,
  type FocusEvent,
  type KeyboardEvent,
  type MouseEvent,
  type PointerEvent,
  type ReactNode,
  type RefObject,
} from "react";
import { createPortal } from "react-dom";

// Tooltips: short plain text that explains a control or a term, on hover (after a short delay),
// on keyboard focus (at once) and on tap (toggles). Escape closes. While open the bubble is
// rendered in a portal on document.body with position: fixed, above or below its anchor (whichever
// fits) and clamped into the viewport, so no card or table can clip it. Keep the content short;
// use a Dialog or Drawer for anything interactive.
//
// - <Tooltip content=…>children</Tooltip> wraps any trigger in a span (the original API).
// - useTooltip() gives the same behaviour to a component's own trigger (InfoTip, Term).

/** Hover delay before a tooltip opens (focus and tap open it at once). */
export const TOOLTIP_OPEN_DELAY_MS = 150;
/** Gap between the anchor and the bubble (px). */
const GAP = 6;
/** Room kept clear at the viewport's edges (px). */
const MARGIN = 8;

export type TooltipSide = "top" | "bottom";
export type TooltipPlacement = { top: number; left: number; side: TooltipSide; arrowX: number };
type Box = { top: number; bottom: number; left: number; width: number };

const clamp = (v: number, lo: number, hi: number) => Math.min(Math.max(v, lo), Math.max(lo, hi));

/**
 * Where a bubble goes: on the preferred side when it fits there, else on the other side when it
 * fits, else on the roomier side; centred on the anchor and clamped into the viewport. `arrowX` is
 * the anchor's centre within the bubble (for the arrow). Pure, for tests.
 */
export function placeTooltip(anchor: Box, bubble: { width: number; height: number }, viewport: { width: number; height: number }, prefer: TooltipSide = "top"): TooltipPlacement {
  const room = { top: anchor.top - GAP - MARGIN, bottom: viewport.height - anchor.bottom - GAP - MARGIN };
  const other: TooltipSide = prefer === "top" ? "bottom" : "top";
  const side: TooltipSide = bubble.height <= room[prefer] ? prefer : bubble.height <= room[other] ? other : room[prefer] >= room[other] ? prefer : other;
  const rawTop = side === "top" ? anchor.top - GAP - bubble.height : anchor.bottom + GAP;
  const top = clamp(rawTop, MARGIN, viewport.height - MARGIN - bubble.height);
  const centre = anchor.left + anchor.width / 2;
  const left = clamp(centre - bubble.width / 2, MARGIN, viewport.width - MARGIN - bubble.width);
  const arrowX = clamp(centre - left, 10, bubble.width - 10);
  return { top: Math.round(top), left: Math.round(left), side, arrowX: Math.round(arrowX) };
}

function viewportSize(): { width: number; height: number } {
  const de = document.documentElement;
  return { width: de.clientWidth || window.innerWidth, height: de.clientHeight || window.innerHeight };
}

export type TooltipOptions = {
  content: ReactNode;
  /** Preferred side (default above); flips when it does not fit. */
  side?: TooltipSide;
  /** Hover delay in ms (default 150). */
  delay?: number;
  /**
   * Keep the (hidden) bubble next to the trigger while closed, so aria-describedby always resolves
   * and the text is in the DOM (the original Tooltip did this). Off: the bubble exists only while open.
   */
  keepInPlace?: boolean;
  /** Extra class on the bubble. */
  bubbleClassName?: string;
};

export type TooltipTriggerProps = {
  "aria-describedby"?: string;
  onMouseEnter: (e: MouseEvent<HTMLElement>) => void;
  onMouseLeave: (e: MouseEvent<HTMLElement>) => void;
  onFocus: (e: FocusEvent<HTMLElement>) => void;
  onBlur: (e: FocusEvent<HTMLElement>) => void;
  onKeyDown: (e: KeyboardEvent<HTMLElement>) => void;
  onPointerDown: (e: PointerEvent<HTMLElement>) => void;
  onClick: (e: MouseEvent<HTMLElement>) => void;
};

export type TooltipControl<T extends HTMLElement = HTMLElement> = {
  id: string;
  open: boolean;
  setOpen: (open: boolean) => void;
  /** Attach to the trigger (the element the bubble is placed against). */
  anchorRef: RefObject<T | null>;
  /** Spread on the trigger. */
  triggerProps: TooltipTriggerProps;
  /** Render next to the trigger: the hidden in-place bubble (keepInPlace) or the open bubble's portal. */
  bubble: ReactNode;
};

/** Tooltip behaviour for a trigger the caller renders (see the file comment). */
export function useTooltip<T extends HTMLElement = HTMLElement>({ content, side = "top", delay = TOOLTIP_OPEN_DELAY_MS, keepInPlace = false, bubbleClassName }: TooltipOptions): TooltipControl<T> {
  const id = useId();
  const [open, setOpenState] = useState(false);
  const [place, setPlace] = useState<TooltipPlacement | null>(null);
  const anchorRef = useRef<T | null>(null);
  const bubbleRef = useRef<HTMLSpanElement | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const hovered = useRef(false);
  const focused = useRef(false);
  /** The current press is a touch: hover and focus are left to the tap (click) toggle. */
  const touch = useRef(false);

  const clearTimer = useCallback(() => {
    if (timer.current !== null) clearTimeout(timer.current);
    timer.current = null;
  }, []);

  const setOpen = useCallback(
    (next: boolean) => {
      clearTimer();
      if (!next) {
        hovered.current = false;
        focused.current = false;
        setPlace(null);
      }
      setOpenState(next);
    },
    [clearTimer],
  );

  useEffect(() => clearTimer, [clearTimer]);

  // Place the bubble before paint; again on scroll and resize while open.
  const reposition = useCallback(() => {
    const a = anchorRef.current;
    const b = bubbleRef.current;
    if (!a || !b) return;
    const ar = a.getBoundingClientRect();
    const br = b.getBoundingClientRect();
    const next = placeTooltip({ top: ar.top, bottom: ar.bottom, left: ar.left, width: ar.width }, { width: br.width, height: br.height }, viewportSize(), side);
    setPlace((p) => (p && p.top === next.top && p.left === next.left && p.side === next.side && p.arrowX === next.arrowX ? p : next));
  }, [side]);

  useLayoutEffect(() => {
    if (open) reposition();
  }, [open, content, reposition]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: globalThis.KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    const onDown = (e: globalThis.PointerEvent | globalThis.MouseEvent) => {
      const t = e.target as Node | null;
      if (t && anchorRef.current?.contains(t)) return;
      setOpen(false);
    };
    window.addEventListener("scroll", reposition, true);
    window.addEventListener("resize", reposition);
    document.addEventListener("keydown", onKey);
    document.addEventListener("pointerdown", onDown);
    return () => {
      window.removeEventListener("scroll", reposition, true);
      window.removeEventListener("resize", reposition);
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("pointerdown", onDown);
    };
  }, [open, reposition, setOpen]);

  const triggerProps: TooltipTriggerProps = {
    "aria-describedby": keepInPlace || open ? id : undefined,
    onMouseEnter: () => {
      if (touch.current) return;
      hovered.current = true;
      clearTimer();
      timer.current = setTimeout(() => {
        timer.current = null;
        if (hovered.current) setOpenState(true);
      }, delay);
    },
    onMouseLeave: () => {
      touch.current = false;
      hovered.current = false;
      clearTimer();
      if (!focused.current) setOpen(false);
    },
    onFocus: () => {
      if (touch.current) return; // the tap's click toggles it
      focused.current = true;
      clearTimer();
      setOpenState(true);
    },
    onBlur: (e) => {
      // focus moving inside the trigger (a wrapped control) keeps it open
      if (e.relatedTarget instanceof Node && anchorRef.current?.contains(e.relatedTarget)) return;
      focused.current = false;
      if (!hovered.current) setOpen(false);
    },
    onKeyDown: (e) => {
      if (e.key === "Escape" && open) {
        e.stopPropagation(); // close the tooltip, not the dialog or drawer around it
        setOpen(false);
      }
    },
    onPointerDown: (e) => {
      touch.current = e.pointerType === "touch";
    },
    onClick: () => {
      if (!touch.current) return;
      touch.current = false;
      setOpen(!open);
    },
  };

  let bubble: ReactNode = null;
  if (open && typeof document !== "undefined") {
    const style = {
      top: place?.top ?? 0,
      left: place?.left ?? 0,
      visibility: place ? undefined : "hidden",
      "--tip-arrow-x": place ? `${place.arrowX}px` : undefined,
    } as CSSProperties;
    bubble = createPortal(
      <span ref={bubbleRef} role="tooltip" id={id} className={["tip-bubble", bubbleClassName ?? ""].filter(Boolean).join(" ")} data-open="true" data-side={place?.side ?? side} style={style}>
        {content}
      </span>,
      document.body,
    );
  } else if (keepInPlace) {
    bubble = (
      <span role="tooltip" id={id} className={["tip-bubble", bubbleClassName ?? ""].filter(Boolean).join(" ")} hidden>
        {content}
      </span>
    );
  }

  return { id, open, setOpen, anchorRef, triggerProps, bubble };
}

/**
 * Hover/focus/tap tooltip around any trigger. The wrapper gets aria-describedby, so the text is
 * read on focus too; the hidden bubble stays in place while closed.
 */
export function Tooltip({ content, children, className, side }: { content: ReactNode; children: ReactNode; className?: string; side?: TooltipSide }) {
  const t = useTooltip<HTMLSpanElement>({ content, side, keepInPlace: true });
  return (
    <span ref={t.anchorRef} className={["tip-anchor", className ?? ""].filter(Boolean).join(" ")} {...t.triggerProps}>
      {children}
      {t.bubble}
    </span>
  );
}
