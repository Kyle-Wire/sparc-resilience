import type { MouseEvent, ReactNode } from "react";
import { term as glossaryTerm, type TermId } from "../../content/glossary";
import { TermTipBody } from "./InfoTip";
import { useTooltip, type TooltipSide } from "./Tooltip";

export type TermProps = {
  /** Glossary term id. */
  id: TermId;
  /** The words shown inline; default the term's plain label. */
  children?: ReactNode;
  /** Replaces the glossary tip (e.g. with run-specific numbers). */
  text?: ReactNode;
  /** The technical name under the tip; defaults to the glossary's. Null hides it. */
  technical?: string | null;
  side?: TooltipSide;
  className?: string;
};

/**
 * A term in running text with a dotted underline: <Term id="coverage" /> shows "Range
 * reliability"; hover, focus or tap shows the same tooltip as <InfoTip term="coverage" />.
 * It is a button (focusable, tappable), so do not put it inside a link or another button.
 */
export function Term({ id, children, text, technical, side, className }: TermProps) {
  const entry = glossaryTerm(id);
  const tech = technical === undefined ? entry.technical : technical;
  const t = useTooltip<HTMLButtonElement>({ content: <TermTipBody tip={text ?? entry.tip} technical={tech} />, side, bubbleClassName: "tip-term" });
  const onClick = (e: MouseEvent<HTMLButtonElement>) => {
    e.preventDefault();
    e.stopPropagation();
    t.triggerProps.onClick(e);
  };
  return (
    <>
      <button ref={t.anchorRef} type="button" className={["term", className ?? ""].filter(Boolean).join(" ")} data-term={id} {...t.triggerProps} onClick={onClick}>
        {children ?? entry.label}
      </button>
      {t.bubble}
    </>
  );
}
