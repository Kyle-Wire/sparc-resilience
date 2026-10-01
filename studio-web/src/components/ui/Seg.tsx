import { useRef, type KeyboardEvent, type ReactNode } from "react";

export type SegOption<T extends string | number> = { value: T; label: ReactNode; disabled?: boolean; title?: string };

export type SegProps<T extends string | number> = {
  options: readonly SegOption<T>[];
  value: T;
  onChange: (v: T) => void;
  label: string;
  size?: "normal" | "small";
  className?: string;
};

/** Segmented single choice (pill buttons with aria-pressed, arrow keys move the choice). */
export function Seg<T extends string | number>({ options, value, onChange, label, size = "normal", className }: SegProps<T>) {
  const ref = useRef<HTMLDivElement>(null);
  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    const enabled = options.filter((o) => !o.disabled);
    const i = enabled.findIndex((o) => o.value === value);
    const next = enabled[(i + (e.key === "ArrowRight" ? 1 : enabled.length - 1)) % enabled.length];
    if (next) {
      e.preventDefault();
      onChange(next.value);
      const idx = options.indexOf(next);
      (ref.current?.querySelectorAll("button")[idx] as HTMLButtonElement | undefined)?.focus();
    }
  };
  return (
    <div ref={ref} className={["seg", size === "small" ? "small" : "", className ?? ""].filter(Boolean).join(" ")} role="group" aria-label={label} onKeyDown={onKey}>
      {options.map((o) => (
        <button
          key={String(o.value)}
          type="button"
          aria-pressed={o.value === value}
          disabled={o.disabled}
          title={o.title}
          tabIndex={o.value === value ? 0 : -1}
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
