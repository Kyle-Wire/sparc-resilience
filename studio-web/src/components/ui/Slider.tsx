import { useId } from "react";

export type SliderProps = {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step?: number;
  /** Log scale: the thumb moves in log10 space (min must be > 0). */
  log?: boolean;
  format?: (v: number) => string;
  disabled?: boolean;
  hideLabel?: boolean;
};

const STEPS = 1000;

/** Range input with a live readout; optional log scale for budgets. */
export function Slider({ label, value, onChange, min, max, step, log, format, disabled, hideLabel }: SliderProps) {
  const id = useId();
  const toPos = (v: number) => (log ? ((Math.log10(Math.max(v, min)) - Math.log10(min)) / (Math.log10(max) - Math.log10(min))) * STEPS : v);
  const fromPos = (p: number) => {
    if (!log) return p;
    const v = Math.pow(10, Math.log10(min) + (p / STEPS) * (Math.log10(max) - Math.log10(min)));
    return step ? Math.round(v / step) * step : Number(v.toPrecision(3));
  };
  const text = format ? format(value) : String(value);
  return (
    <div className="field">
      <label htmlFor={id} className={hideLabel ? "sr-only" : undefined}>
        {label}
      </label>
      <div className="slider">
        <input
          id={id}
          type="range"
          min={log ? 0 : min}
          max={log ? STEPS : max}
          step={log ? 1 : step ?? "any"}
          value={toPos(value)}
          disabled={disabled}
          aria-valuetext={text}
          onChange={(e) => onChange(fromPos(Number(e.target.value)))}
        />
        <output htmlFor={id}>{text}</output>
      </div>
    </div>
  );
}
