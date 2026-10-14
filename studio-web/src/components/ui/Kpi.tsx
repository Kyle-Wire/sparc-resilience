import { Children, type CSSProperties, type ReactNode } from "react";
import type { TermId } from "../../content/glossary";
import { InfoTip } from "./InfoTip";

export type KpiProps = {
  label: ReactNode;
  value: ReactNode;
  unit?: ReactNode;
  note?: ReactNode;
  tone?: "good" | "warn" | "crit";
  /** Glossary term the number is: an InfoTip (ⓘ) after the label explains it. */
  term?: TermId;
  /** A one-off explanation in an InfoTip (ⓘ) after the label; with `term`, it replaces the glossary tip. */
  tip?: ReactNode;
  /** @deprecated Use `tip`: shown the same way (an InfoTip), no longer a native title. */
  title?: string;
};

/** KPI tile: label (with an optional ⓘ), big tabular value with unit, optional note. */
export function Kpi({ label, value, unit, note, tone, term, tip, title }: KpiProps) {
  const text = tip ?? (title || undefined);
  const name = typeof label === "string" || typeof label === "number" ? String(label) : "this number";
  const info = term ? <InfoTip term={term} text={text ?? undefined} /> : text !== undefined && text !== null && text !== "" && text !== false ? <InfoTip label={name} text={text} /> : null;
  return (
    <div className="kpi" data-tone={tone}>
      <span className="label">
        {label}
        {info}
      </span>
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
