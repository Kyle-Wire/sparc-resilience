import type { SelectHTMLAttributes } from "react";

export type SelectOption<T extends string> = { value: T; label: string; disabled?: boolean; group?: string };

export type SelectProps<T extends string> = Omit<SelectHTMLAttributes<HTMLSelectElement>, "onChange" | "value"> & {
  options: readonly SelectOption<T>[];
  value: T | null;
  onChange: (v: T) => void;
  /** Accessible name when there is no visible <label for>. */
  label?: string;
  placeholder?: string;
};

/** Native select (option groups from `group`), typed values. */
export function Select<T extends string>({ options, value, onChange, label, placeholder, ...rest }: SelectProps<T>) {
  const groups = new Map<string, SelectOption<T>[]>();
  const loose: SelectOption<T>[] = [];
  for (const o of options) {
    if (o.group) {
      const g = groups.get(o.group) ?? [];
      g.push(o);
      groups.set(o.group, g);
    } else loose.push(o);
  }
  const opt = (o: SelectOption<T>) => (
    <option key={o.value} value={o.value} disabled={o.disabled}>
      {o.label}
    </option>
  );
  return (
    <select {...rest} aria-label={rest["aria-label"] ?? label} value={value ?? ""} onChange={(e) => onChange(e.target.value as T)}>
      {value === null || placeholder ? (
        <option value="" disabled>
          {placeholder ?? "Choose…"}
        </option>
      ) : null}
      {loose.map(opt)}
      {[...groups.entries()].map(([g, list]) => (
        <optgroup key={g} label={g}>
          {list.map(opt)}
        </optgroup>
      ))}
    </select>
  );
}
