import type { ReactNode } from "react";
import { Icon, type IconName } from "./Icon";
import { useTipOn } from "./useTipOn";

export type Tone = "good" | "warn" | "crit" | "accent" | "neutral";

/**
 * Rounded pill with an optional icon (status is never colour alone). `tip` explains it in a
 * tooltip (hover, keyboard focus or tap; the pill becomes focusable).
 */
export function Pill({ tone = "neutral", icon, children, tip, title }: { tone?: Tone; icon?: IconName; children: ReactNode; tip?: ReactNode; /** @deprecated Use `tip` (the same tooltip; no longer a native title). */ title?: string }) {
  const t = useTipOn<HTMLSpanElement>(tip ?? title);
  return (
    <>
      <span ref={t.ref} className="pill" data-tone={tone === "neutral" ? undefined : tone} tabIndex={t.on ? 0 : undefined} {...t.props()}>
        {icon ? <Icon name={icon} /> : null}
        {children}
      </span>
      {t.bubble}
    </>
  );
}
