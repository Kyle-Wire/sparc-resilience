/** Determinate (0..1) or indeterminate (value null) progress bar with an accessible value. */
export function ProgressBar({ value, label, tone, className }: { value: number | null | undefined; label: string; tone?: "good" | "crit"; className?: string }) {
  const v = value === null || value === undefined || !Number.isFinite(value) ? null : Math.min(1, Math.max(0, value));
  return (
    <div
      className={["progress", className ?? ""].filter(Boolean).join(" ")}
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={v === null ? undefined : Math.round(v * 100)}
      aria-valuetext={v === null ? "in progress" : `${Math.round(v * 100)}%`}
      data-indeterminate={v === null ? "true" : undefined}
      data-tone={tone}
    >
      <i style={{ width: `${(v ?? 0.3) * 100}%` }} />
    </div>
  );
}
