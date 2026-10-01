import { useEffect, useState, type InputHTMLAttributes } from "react";
import { unitLabel } from "../../theme/format";

export type NumberFieldProps = Omit<InputHTMLAttributes<HTMLInputElement>, "value" | "onChange" | "type" | "min" | "max"> & {
  value: number | null;
  onChange: (v: number | null) => void;
  min?: number;
  max?: number;
  step?: number | "any";
  unit?: string | null;
  /** Allow an empty field (onChange(null)). */
  nullable?: boolean;
  label?: string;
};

function parse(text: string): number | null {
  const t = text.trim().replace(/−/g, "-").replace(/,/g, "");
  if (t === "" || t === "-" || t === ".") return null;
  const v = Number(t);
  return Number.isFinite(v) ? v : null;
}

/**
 * Numeric input that accepts U+2212, commas and partial typing; commits on blur/Enter,
 * clamps to [min, max] and shows the unit.
 */
export function NumberField({ value, onChange, min, max, step = "any", unit, nullable, label, onBlur, onKeyDown, ...rest }: NumberFieldProps) {
  const [text, setText] = useState(value === null ? "" : String(value));
  useEffect(() => setText(value === null ? "" : String(value)), [value]);
  const commit = () => {
    let v = parse(text);
    if (v === null) {
      if (nullable) onChange(null);
      else setText(value === null ? "" : String(value));
      return;
    }
    if (min !== undefined) v = Math.max(min, v);
    if (max !== undefined) v = Math.min(max, v);
    setText(String(v));
    if (v !== value) onChange(v);
  };
  const invalid = text.trim() !== "" && parse(text) === null;
  return (
    <span className="numfield">
      <input
        {...rest}
        type="text"
        inputMode="decimal"
        aria-label={rest["aria-label"] ?? label}
        aria-invalid={invalid || rest["aria-invalid"] || undefined}
        value={text}
        data-step={step}
        onChange={(e) => setText(e.target.value)}
        onBlur={(e) => {
          commit();
          onBlur?.(e);
        }}
        onKeyDown={(e) => {
          if (e.key === "Enter") commit();
          if ((e.key === "ArrowUp" || e.key === "ArrowDown") && typeof step === "number") {
            e.preventDefault();
            const cur = parse(text) ?? value ?? 0;
            let v = cur + (e.key === "ArrowUp" ? step : -step);
            if (min !== undefined) v = Math.max(min, v);
            if (max !== undefined) v = Math.min(max, v);
            v = Number(v.toPrecision(12));
            setText(String(v));
            onChange(v);
          }
          onKeyDown?.(e);
        }}
      />
      {unit ? <span className="unit">{unitLabel(unit)}</span> : null}
    </span>
  );
}
