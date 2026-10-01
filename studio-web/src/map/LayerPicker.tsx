// Theme → layer picker (SPEC §6.5): theme pills, then a searchable layer list. Searching
// spans every theme.
import { useId, useMemo, useState } from "react";
import type { LayerGroup, LayerMeta } from "../api/types";
import { Seg } from "../components/ui/Seg";

export type LayerPickerProps = {
  groups: LayerGroup[];
  value: string | null;
  onChange: (key: string) => void;
  label?: string;
};

export function findLayer(groups: LayerGroup[], key: string | null): LayerMeta | null {
  if (!key) return null;
  for (const g of groups) for (const l of g.layers) if (l.key === key) return l;
  return null;
}

export function LayerPicker({ groups, value, onChange, label = "Layer" }: LayerPickerProps) {
  const id = useId();
  const current = groups.find((g) => g.layers.some((l) => l.key === value)) ?? groups[0];
  const [theme, setTheme] = useState<string | null>(null);
  const [q, setQ] = useState("");
  const activeTheme = groups.find((g) => g.id === theme) ?? current;
  const matches = useMemo(() => {
    const words = q.toLowerCase().split(/\s+/).filter(Boolean);
    if (!words.length) return activeTheme ? [{ group: activeTheme, layers: activeTheme.layers }] : [];
    return groups
      .map((g) => ({ group: g, layers: g.layers.filter((l) => words.every((w) => `${l.label} ${l.key} ${l.desc} ${g.label}`.toLowerCase().includes(w))) }))
      .filter((x) => x.layers.length);
  }, [groups, activeTheme, q]);
  if (!groups.length) return <p className="cap">No layers yet.</p>;
  return (
    <div className="map-controls" role="group" aria-label="Choose a map layer">
      {groups.length <= 9 ? (
        <Seg
          label="Map theme"
          size="small"
          options={groups.map((g) => ({ value: g.id, label: g.label }))}
          value={activeTheme?.id ?? groups[0].id}
          onChange={(gid) => {
            setTheme(gid);
            setQ("");
            const g = groups.find((x) => x.id === gid);
            if (g?.layers.length && !g.layers.some((l) => l.key === value)) onChange(g.layers[0].key);
          }}
        />
      ) : (
        <select aria-label="Map theme" value={activeTheme?.id} onChange={(e) => setTheme(e.target.value)}>
          {groups.map((g) => (
            <option key={g.id} value={g.id}>
              {g.label}
            </option>
          ))}
        </select>
      )}
      <input type="search" aria-label="Search layers" placeholder="Search layers…" value={q} onChange={(e) => setQ(e.target.value)} style={{ width: "12em" }} />
      <label className="cap" htmlFor={id}>
        {label}
      </label>
      <select id={id} value={value ?? ""} onChange={(e) => onChange(e.target.value)} style={{ maxWidth: "22em" }}>
        {value && !matches.some((m) => m.layers.some((l) => l.key === value)) ? <option value={value}>{findLayer(groups, value)?.label ?? value}</option> : null}
        {matches.length === 1
          ? matches[0].layers.map((l) => (
              <option key={l.key} value={l.key}>
                {l.label}
              </option>
            ))
          : matches.map((m) => (
              <optgroup key={m.group.id} label={m.group.label}>
                {m.layers.map((l) => (
                  <option key={l.key} value={l.key}>
                    {l.label}
                  </option>
                ))}
              </optgroup>
            ))}
      </select>
    </div>
  );
}
