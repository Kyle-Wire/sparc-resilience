import { useId, useRef, type KeyboardEvent, type ReactNode } from "react";
import { useTipOn } from "./useTipOn";

export type SegOption<T extends string | number> = {
  value: T;
  label: ReactNode;
  disabled?: boolean;
  /** What the option does, or why it is disabled, in a tooltip (hover and keyboard focus). */
  tip?: ReactNode;
  /** @deprecated Use `tip` (the same tooltip; no longer a native title). */
  title?: string;
};

export type SegProps<T extends string | number> = {
  options: readonly SegOption<T>[];
  value: T;
  onChange: (v: T) => void;
  label: string;
  size?: "normal" | "small";
  className?: string;
};

/**
 * One option. With a tip it is wrapped in a span that shows the tip on hover and keyboard focus
 * (a disabled button gets no pointer or focus events of its own; focus bubbles to the wrapper), and
 * the tip text describes the button for screen readers. Without one it is the plain button.
 */
function SegButton({ tip, pressed, disabled, tabIndex, onPick, children }: { tip: ReactNode; pressed: boolean; disabled?: boolean; tabIndex: number; onPick: () => void; children: ReactNode }) {
  const descId = useId();
  const hasTip = tip !== undefined && tip !== null && tip !== "";
  const t = useTipOn<HTMLSpanElement>(hasTip ? tip : null, { control: true, describe: false });
  const button = (
    <button type="button" aria-pressed={pressed} disabled={disabled} aria-describedby={hasTip ? descId : undefined} tabIndex={tabIndex} onClick={onPick}>
      {children}
    </button>
  );
  if (!hasTip) return button;
  return (
    <>
      <span ref={t.ref} className="seg-tip" style={{ display: "inline-flex" }} {...t.props()}>
        {button}
        <span id={descId} hidden>
          {tip}
        </span>
      </span>
      {t.bubble}
    </>
  );
}

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
        <SegButton key={String(o.value)} tip={o.tip ?? o.title} pressed={o.value === value} disabled={o.disabled} tabIndex={o.value === value ? 0 : -1} onPick={() => onChange(o.value)}>
          {o.label}
        </SegButton>
      ))}
    </div>
  );
}
