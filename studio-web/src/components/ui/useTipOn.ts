// A tooltip on a component's own element (StatusChip, Badge, Pill, IconButton, Chips, Seg, the map's
// tool buttons), in place of a native title="…": the same bubble as <Tooltip> (hover after a short
// delay, keyboard focus, tap, Escape), but with no wrapper span, so layout and CSS selectors stay as
// they were. The bubble only exists while open (a portal on <body>), so nothing is added in place.
import type { ReactNode, SyntheticEvent } from "react";
import { useTooltip, type TooltipControl, type TooltipSide, type TooltipTriggerProps } from "./Tooltip";

export type TipHandlerKey = Exclude<keyof TooltipTriggerProps, "aria-describedby">;
const KEYS: readonly TipHandlerKey[] = ["onMouseEnter", "onMouseLeave", "onFocus", "onBlur", "onKeyDown", "onPointerDown", "onClick"];

/** An element's own handlers, run before the tooltip's (e.g. a button's onClick and onFocus). */
export type OwnHandlers = Partial<Record<TipHandlerKey, ((e: never) => void) | undefined>>;

export type TipOptions = {
  side?: TooltipSide;
  /**
   * The element is a control (a button that acts on click): a tap acts and does not toggle the
   * tooltip, which then shows on hover and keyboard focus only.
   */
  control?: boolean;
  /** False when the tip only repeats the element's accessible name (no aria-describedby). */
  describe?: boolean;
};

export type TipOn<T extends HTMLElement> = {
  /** There is a tip to show (non-empty content). */
  on: boolean;
  ref: TooltipControl<T>["anchorRef"];
  /**
   * Props for the element: its own handlers chained with the tooltip's, and aria-describedby while
   * the tip shows. Just the element's own handlers when there is no tip.
   */
  props: (own?: OwnHandlers) => Partial<TooltipTriggerProps>;
  /** Render next to the element (the open bubble's portal; null while closed). */
  bubble: ReactNode;
};

const empty = (c: ReactNode) => c === null || c === undefined || c === false || c === "";

/** Tooltip behaviour for an element a component renders itself (see the file comment). */
export function useTipOn<T extends HTMLElement = HTMLElement>(content: ReactNode, opts: TipOptions = {}): TipOn<T> {
  const t = useTooltip<T>({ content, side: opts.side });
  const on = !empty(content);
  const props = (own: OwnHandlers = {}): Partial<TooltipTriggerProps> => {
    if (!on) return own as Partial<TooltipTriggerProps>;
    const out: Partial<TooltipTriggerProps> = {};
    for (const k of KEYS) {
      const mine = own[k];
      // A control acts on a tap; the tooltip's tap toggle (pointerdown + click) is left out.
      const theirs = opts.control && (k === "onClick" || k === "onPointerDown") ? undefined : (t.triggerProps[k] as (e: SyntheticEvent) => void);
      if (!mine && !theirs) continue;
      (out as Record<string, unknown>)[k] = (e: SyntheticEvent) => {
        mine?.(e as never);
        theirs?.(e);
      };
    }
    if (opts.describe !== false) out["aria-describedby"] = t.triggerProps["aria-describedby"];
    return out;
  };
  return { on, ref: t.anchorRef, props, bubble: on ? t.bubble : null };
}
