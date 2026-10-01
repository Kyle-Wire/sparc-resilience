import type { ReactNode } from "react";

export type ChipItem<T extends string> = { value: T; label: ReactNode; title?: string };

/**
 * Multi-select chips (aria-pressed toggles), or removable chips when `onRemove` is given.
 */
export function Chips<T extends string>({
  items,
  selected,
  onToggle,
  onRemove,
  label,
}: {
  items: readonly ChipItem<T>[];
  selected?: readonly T[];
  onToggle?: (v: T, on: boolean) => void;
  onRemove?: (v: T) => void;
  label: string;
}) {
  return (
    <div className="chips" role="group" aria-label={label}>
      {items.map((it) => {
        if (onToggle) {
          const on = !!selected?.includes(it.value);
          return (
            <button key={it.value} type="button" className="chip" aria-pressed={on} title={it.title} onClick={() => onToggle(it.value, !on)}>
              {it.label}
            </button>
          );
        }
        return (
          <span key={it.value} className="chip" title={it.title}>
            {it.label}
            {onRemove ? (
              <button type="button" className="x" aria-label={`Remove ${typeof it.label === "string" ? it.label : it.value}`} onClick={() => onRemove(it.value)}>
                ×
              </button>
            ) : null}
          </span>
        );
      })}
    </div>
  );
}
