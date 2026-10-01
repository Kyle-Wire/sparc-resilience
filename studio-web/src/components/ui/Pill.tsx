import type { ReactNode } from "react";
import { Icon, type IconName } from "./Icon";

export type Tone = "good" | "warn" | "crit" | "accent" | "neutral";

/** Rounded pill with an optional icon (status is never colour alone). */
export function Pill({ tone = "neutral", icon, children, title }: { tone?: Tone; icon?: IconName; children: ReactNode; title?: string }) {
  return (
    <span className="pill" data-tone={tone === "neutral" ? undefined : tone} title={title}>
      {icon ? <Icon name={icon} /> : null}
      {children}
    </span>
  );
}
