// Small tooltip helpers of the shell and the project pages: plain chips that explain themselves on
// hover, keyboard focus or tap (components/ui/Tooltip.tsx) in place of native `title`s, with the
// glossary's words (content/glossary.ts).
import type { ReactElement, ReactNode, Ref, TdHTMLAttributes } from "react";
import { Badge, modeLabel } from "../components/ui/Badge";
import { TermTipBody } from "../components/ui/InfoTip";
import { useTooltip, type TooltipSide } from "../components/ui/Tooltip";
import { useTipOn } from "../components/ui/useTipOn";
import { term, type TermId } from "../content/glossary";
import { Link, type LinkProps } from "../router";

/** Link with a ref (React 19 passes `ref` as a prop, and Link spreads its props onto the <a>). */
export const RefLink = Link as (props: LinkProps & { ref?: Ref<HTMLAnchorElement> }) => ReactElement;

/** The glossary term of a run mode ("fast" → run_fast), or null for another mode. */
export function runModeTerm(mode: string | null | undefined): TermId | null {
  return mode === "fast" ? "run_fast" : mode === "coarse" ? "run_coarse" : mode === "full" ? "run_full" : null;
}

/**
 * A chip (badge, status, short text) that explains itself on hover, keyboard focus or tap, in
 * place of a native title; the bubble exists only while open, so it adds nothing to the page's
 * text. Focusable so keyboard users can read it too. Without a tip the chip renders as it is.
 */
export function TipSpan({ tip, technical, side, className, children }: { tip: ReactNode; technical?: string | null; side?: TooltipSide; className?: string; children: ReactNode }) {
  const t = useTooltip<HTMLSpanElement>({ content: <TermTipBody tip={tip} technical={technical} />, side, bubbleClassName: "tip-term" });
  if (tip === null || tip === undefined || tip === "") return <>{children}</>;
  return (
    <span ref={t.anchorRef} className={["tip-anchor", className ?? ""].filter(Boolean).join(" ")} tabIndex={0} data-tip-chip="" {...t.triggerProps}>
      {children}
      {t.bubble}
    </span>
  );
}

/** A run's mode badge (FAST, COARSE 60, FULL) with what the mode means in a tooltip. */
export function ModeBadge({ mode, coarse_m }: { mode: string | null | undefined; coarse_m?: number | null }) {
  const id = runModeTerm(mode);
  const badge = <Badge tone="mode">{modeLabel(mode, coarse_m)}</Badge>;
  if (!id) return badge;
  const e = term(id);
  return (
    <TipSpan tip={`${e.label} run: ${e.tip}`} technical={e.technical}>
      {badge}
    </TipSpan>
  );
}

/** The DEMO badge, with what demo data means in a tooltip (not a native title). */
export function DemoTag() {
  return (
    <TipSpan tip={term("demo").tip} technical={term("demo").technical}>
      <Badge tone="demo">DEMO</Badge>
    </TipSpan>
  );
}

/**
 * A table cell (a grid square of a status matrix) that shows its label in a tooltip on hover or
 * tap, in place of a native title. The label is also the cell's aria-label, so the tooltip does not
 * describe it again.
 */
export function TipCell({ tip, children, ...rest }: Omit<TdHTMLAttributes<HTMLTableCellElement>, "title"> & { tip: string; children?: ReactNode }) {
  const t = useTipOn<HTMLTableCellElement>(tip, { describe: false });
  return (
    <td ref={t.ref} aria-label={tip} {...rest} {...t.props()}>
      {children}
      {t.bubble}
    </td>
  );
}
