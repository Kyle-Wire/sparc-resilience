// A tooltip on any inline content of the Lab (a status chip, a clamped line, a dot, the wrapper of a
// button), in place of a native `title`: hover (after a short delay), keyboard focus or tap opens it,
// Escape closes it (components/ui/Tooltip useTooltip). The bubble exists only while open, in a portal,
// so it adds nothing to the page's text. `focusable` puts an anchor that is not a control in the tab
// order, so keyboard users can read the tip too; around a control leave it off (the control's own
// focus opens it). The wrapper stays when the tip is empty (no tooltip then), so a control inside it
// keeps its focus when the tip comes and goes. For a term with a glossary entry use InfoTip or Term.
import type { ReactNode } from "react";
import { useTooltip, type TooltipSide } from "../../../components/ui/Tooltip";

export type TipProps = {
  /** The tooltip's content; empty (null, undefined, "") shows none. */
  tip: ReactNode;
  children: ReactNode;
  /** Put the anchor in the tab order (for content that is not a control). */
  focusable?: boolean;
  side?: TooltipSide;
  className?: string;
};

export function Tip({ tip, children, focusable = false, side, className }: TipProps) {
  const t = useTooltip<HTMLSpanElement>({ content: tip, side });
  const empty = tip === null || tip === undefined || tip === "" || tip === false;
  return (
    <span ref={t.anchorRef} className={["tip-anchor", className ?? ""].filter(Boolean).join(" ")} tabIndex={focusable && !empty ? 0 : undefined} data-tip={empty ? undefined : ""} {...(empty ? {} : t.triggerProps)}>
      {children}
      {empty ? null : t.bubble}
    </span>
  );
}
