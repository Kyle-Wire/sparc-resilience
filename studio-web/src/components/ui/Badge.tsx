import type { ReactNode } from "react";

/** Small uppercase badge: mode (FAST / COARSE 60 / FULL), DEMO, LIVE, STALE… */
export function Badge({ tone, children, title }: { tone?: "accent" | "warn" | "crit" | "demo"; children: ReactNode; title?: string }) {
  return (
    <span className="badge" data-tone={tone} title={title}>
      {children}
    </span>
  );
}

/** Run mode badge text: FAST, COARSE 60, FULL, CUSTOM. */
export function modeLabel(mode: string | null | undefined, coarse_m?: number | null): string {
  if (mode === "coarse") return coarse_m ? `COARSE ${Math.round(coarse_m)}` : "COARSE";
  return (mode ?? "run").toUpperCase();
}

export function DemoBadge() {
  return (
    <Badge tone="demo" title="Synthetic demo data at a fictional location">
      DEMO
    </Badge>
  );
}
