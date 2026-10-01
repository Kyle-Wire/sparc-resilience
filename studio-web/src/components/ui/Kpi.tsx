import type { ReactNode } from "react";

export type KpiProps = {
  label: ReactNode;
  value: ReactNode;
  unit?: ReactNode;
  note?: ReactNode;
  tone?: "good" | "warn" | "crit";
  title?: string;
};

/** KPI tile: label, big tabular value with unit, optional note. */
export function Kpi({ label, value, unit, note, tone, title }: KpiProps) {
  return (
    <div className="kpi" data-tone={tone} title={title}>
      <span className="label">{label}</span>
      <span className="value">
        {value}
        {unit ? <small>{unit}</small> : null}
      </span>
      {note ? <span className="note">{note}</span> : null}
    </div>
  );
}

export function KpiRow({ children, label }: { children: ReactNode; label?: string }) {
  return (
    <section className="kpis" aria-label={label ?? "Key numbers"}>
      {children}
    </section>
  );
}
