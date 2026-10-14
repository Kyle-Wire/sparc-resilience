import type { ReactNode } from "react";
import { term } from "../../content/glossary";
import { TermTipBody } from "./InfoTip";
import { useTipOn } from "./useTipOn";

/** Small uppercase badge: mode (FAST / COARSE 60 / FULL), DEMO, LIVE, STALE… The run mode uses
 *  tone "mode" (neutral ink on a hairline): crimson stays for alerts, the active state and the CTA.
 *  `tip` explains it in a tooltip (hover, keyboard focus or tap; the badge becomes focusable). */
export function Badge({ tone, children, tip, title }: { tone?: "accent" | "warn" | "crit" | "demo" | "mode"; children: ReactNode; tip?: ReactNode; /** @deprecated Use `tip` (the same tooltip; no longer a native title). */ title?: string }) {
  const t = useTipOn<HTMLSpanElement>(tip ?? title);
  return (
    <>
      <span ref={t.ref} className="badge" data-tone={tone} tabIndex={t.on ? 0 : undefined} {...t.props()}>
        {children}
      </span>
      {t.bubble}
    </>
  );
}

/** Run mode badge text: FAST, COARSE 60, FULL, CUSTOM. */
export function modeLabel(mode: string | null | undefined, coarse_m?: number | null): string {
  if (mode === "coarse") return coarse_m ? `COARSE ${Math.round(coarse_m)}` : "COARSE";
  return (mode ?? "run").toUpperCase();
}

/** DEMO badge; its tooltip is the glossary's "Demo data" tip. */
export function DemoBadge() {
  const d = term("demo");
  return (
    <Badge tone="demo" tip={<TermTipBody tip={d.tip} technical={d.technical} />}>
      DEMO
    </Badge>
  );
}
