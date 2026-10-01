import { useId, useRef, type KeyboardEvent, type ReactNode } from "react";

export type TabItem<T extends string> = { id: T; label: ReactNode; badge?: ReactNode; disabled?: boolean };

/**
 * ARIA tabs (roving tabindex, arrow/Home/End keys). Render the active panel as `children`;
 * it is labelled by the active tab.
 */
export function Tabs<T extends string>({
  items,
  value,
  onChange,
  label,
  children,
}: {
  items: readonly TabItem<T>[];
  value: T;
  onChange: (id: T) => void;
  label: string;
  children?: ReactNode;
}) {
  const base = useId();
  const listRef = useRef<HTMLDivElement>(null);
  const enabled = items.filter((i) => !i.disabled);
  const onKey = (e: KeyboardEvent) => {
    const i = enabled.findIndex((t) => t.id === value);
    let next: TabItem<T> | undefined;
    if (e.key === "ArrowRight") next = enabled[(i + 1) % enabled.length];
    else if (e.key === "ArrowLeft") next = enabled[(i - 1 + enabled.length) % enabled.length];
    else if (e.key === "Home") next = enabled[0];
    else if (e.key === "End") next = enabled[enabled.length - 1];
    if (!next) return;
    e.preventDefault();
    onChange(next.id);
    const idx = items.indexOf(next);
    (listRef.current?.querySelectorAll('[role="tab"]')[idx] as HTMLElement | undefined)?.focus();
  };
  return (
    <div>
      <div className="tabs" role="tablist" aria-label={label} ref={listRef} onKeyDown={onKey}>
        {items.map((t) => (
          <button
            key={t.id}
            type="button"
            role="tab"
            id={`${base}-tab-${t.id}`}
            aria-selected={t.id === value}
            aria-controls={`${base}-panel`}
            tabIndex={t.id === value ? 0 : -1}
            disabled={t.disabled}
            onClick={() => onChange(t.id)}
          >
            {t.label}
            {t.badge}
          </button>
        ))}
      </div>
      {children !== undefined ? (
        <div className="tabpanel" role="tabpanel" id={`${base}-panel`} aria-labelledby={`${base}-tab-${value}`} tabIndex={0}>
          {children}
        </div>
      ) : null}
    </div>
  );
}
