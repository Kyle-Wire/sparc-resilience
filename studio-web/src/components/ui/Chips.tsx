import type { ReactNode } from "react";
import { useTipOn } from "./useTipOn";

export type ChipItem<T extends string> = {
  value: T;
  label: ReactNode;
  /** What the chip means, in a tooltip (hover, keyboard focus or tap). */
  tip?: ReactNode;
  /** @deprecated Use `tip` (the same tooltip; no longer a native title). */
  title?: string;
};

/** A toggle chip; its tip shows on hover and keyboard focus (a tap toggles the chip). */
function ToggleChip({ tip, on, onToggle, children }: { tip: ReactNode; on: boolean; onToggle: () => void; children: ReactNode }) {
  const t = useTipOn<HTMLButtonElement>(tip, { control: true });
  return (
    <>
      <button ref={t.ref} type="button" className="chip" aria-pressed={on} {...t.props({ onClick: onToggle })}>
        {children}
      </button>
      {t.bubble}
    </>
  );
}

/** A tag chip, with an optional remove button; with a tip it is focusable and shows it. */
function TagChip({ tip, children }: { tip: ReactNode; children: ReactNode }) {
  const t = useTipOn<HTMLSpanElement>(tip);
  return (
    <>
      <span ref={t.ref} className="chip" tabIndex={t.on ? 0 : undefined} {...t.props()}>
        {children}
      </span>
      {t.bubble}
    </>
  );
}

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
            <ToggleChip key={it.value} tip={it.tip ?? it.title} on={on} onToggle={() => onToggle(it.value, !on)}>
              {it.label}
            </ToggleChip>
          );
        }
        return (
          <TagChip key={it.value} tip={it.tip ?? it.title}>
            {it.label}
            {onRemove ? (
              <button type="button" className="x" aria-label={`Remove ${typeof it.label === "string" ? it.label : it.value}`} onClick={() => onRemove(it.value)}>
                ×
              </button>
            ) : null}
          </TagChip>
        );
      })}
    </div>
  );
}
