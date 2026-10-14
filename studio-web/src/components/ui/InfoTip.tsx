import type { MouseEvent, ReactNode } from "react";
import { term as glossaryTerm, type TermId } from "../../content/glossary";
import { Icon } from "./Icon";
import { useTooltip, type TooltipSide } from "./Tooltip";

/**
 * The body of a glossary tooltip: the plain tip, then the technical name in mono as a footnote
 * ("Technical: 90% interval coverage"). Shared by InfoTip and Term.
 */
export function TermTipBody({ tip, technical }: { tip: ReactNode; technical?: string | null }) {
  return (
    <>
      <span className="tip-text">{tip}</span>
      {technical ? (
        <span className="tip-tech">
          Technical: <code>{technical}</code>
        </span>
      ) : null}
    </>
  );
}

type Source =
  /** A glossary term: its label names the button, its tip (or `text`) fills the bubble. */
  | { term: TermId; label?: string; text?: ReactNode }
  /** A one-off tip with no glossary entry. */
  | { term?: undefined; label: string; text: ReactNode };

export type InfoTipProps = Source & {
  /** The technical name under the tip; defaults to the glossary's. Null hides it. */
  technical?: string | null;
  /** Preferred side of the bubble (default above). */
  side?: TooltipSide;
  className?: string;
};

/**
 * A small ⓘ button that explains a term: <InfoTip term="coverage" />. Hover (after a short delay),
 * focus or tap opens the tooltip; Escape closes it. The button is named "What is <label>?" and is
 * described by the tip while it shows. Clicking it never triggers the control around it.
 */
export function InfoTip(props: InfoTipProps) {
  const entry = props.term !== undefined ? glossaryTerm(props.term) : null;
  const label = props.label ?? entry?.label ?? "";
  const tip = props.text ?? entry?.tip ?? "";
  const technical = props.technical === undefined ? entry?.technical : props.technical;
  const t = useTooltip<HTMLButtonElement>({ content: <TermTipBody tip={tip} technical={technical} />, side: props.side, bubbleClassName: "tip-term" });
  const onClick = (e: MouseEvent<HTMLButtonElement>) => {
    // inside a <summary>, <label> or clickable card the ⓘ explains; it does not toggle or submit
    e.preventDefault();
    e.stopPropagation();
    t.triggerProps.onClick(e);
  };
  return (
    <>
      <button
        ref={t.anchorRef}
        type="button"
        className={["infotip", props.className ?? ""].filter(Boolean).join(" ")}
        aria-label={`What is ${label}?`}
        data-term={props.term}
        {...t.triggerProps}
        onClick={onClick}
      >
        <Icon name="info" size={14} />
      </button>
      {t.bubble}
    </>
  );
}
