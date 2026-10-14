import { Children, type CSSProperties, type ReactNode } from "react";

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

/**
 * How many columns a row of `n` tiles uses at most, so its last line is never one lone tile:
 * up to 5 in one line, else the widest of 5, 4 or 3 that divides them evenly (6 → 3 × 2,
 * 8 → 4 × 2, 9 → 3 × 3), else the widest whose last line is at most one short (7 → 4 + 3),
 * else 5. Narrow screens still drop to fewer columns (base.css `.kpis[data-cols]`).
 */
export function balancedKpiCols(n: number): number {
  if (n <= 5) return Math.max(1, n);
  const opts = [5, 4, 3];
  return opts.find((c) => n % c === 0) ?? opts.find((c) => n % c >= c - 1) ?? 5;
}

export function KpiRow({ children, label }: { children: ReactNode; label?: string }) {
  const cols = balancedKpiCols(Children.toArray(children).length);
  return (
    <section className="kpis" aria-label={label ?? "Key numbers"} data-cols={cols} style={{ "--kpi-cols": cols } as CSSProperties}>
      {children}
    </section>
  );
}
