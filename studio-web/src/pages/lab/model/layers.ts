// Lab map layers computed or held in the browser (preview ΔT, brush edits, future
// temperature, exceedance) and Studio result/plan layers, as LayerMeta entries the map kit
// understands. Percentile stats are left null so the domain is computed from the values.
import type { LayerGroup, LayerMeta, LayerStats } from "../../../api/types";

export function quickStats(values: ArrayLike<number> | null): LayerStats {
  let n = 0;
  let lo = Infinity;
  let hi = -Infinity;
  let sum = 0;
  if (values)
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      if (!Number.isFinite(v)) continue;
      n++;
      sum += v;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  return { n, lo: n ? lo : null, hi: n ? hi : null, mean: n ? sum / n : null, p1: null, p2: null, p50: null, p98: null, p99: null };
}

export type SyntheticLayer = {
  key: string;
  label: string;
  unit: string;
  group?: string;
  scale?: LayerMeta["scale"];
  center?: number | null;
  decimals?: number;
  desc?: string;
  sign_note?: string | null;
  labels?: string[] | null;
  zero_blank?: boolean;
  dtype?: LayerMeta["dtype"];
};

/** A LayerMeta for values computed or held client-side. */
export function syntheticMeta(l: SyntheticLayer, values: ArrayLike<number> | null): LayerMeta {
  return {
    key: l.key,
    group: l.group ?? "lab",
    label: l.label,
    unit: l.unit,
    scale: l.scale ?? "div",
    center: l.center === undefined ? (l.scale === "seq" || l.scale === "cat" ? null : 0) : l.center,
    decimals: l.decimals ?? 2,
    mult: 1,
    zero_blank: l.zero_blank ?? false,
    labels: l.labels ?? null,
    desc: l.desc ?? "",
    sign_note: l.sign_note ?? null,
    source: null,
    dtype: l.dtype ?? "float32",
    stats: l.scale === "cat" ? { n: values?.length ?? 0, lo: 0, hi: (l.labels?.length ?? 2) - 1, mean: null, p1: null, p2: null, p50: null, p98: null, p99: null } : quickStats(values),
  };
}

/** Result layer fields (api.md §7.5) with their labels. */
export function resultLayerMetas(resId: string, unit: string, realizedVars: string[], leverUnits: Record<string, string> = {}, leverNames: Record<string, string> = {}): LayerMeta[] {
  const base = `res:${resId}`;
  const list: SyntheticLayer[] = [
    { key: `${base}:delta`, label: "Temperature change (full model)", unit, sign_note: "negative = cooler", desc: "The full model's estimate of the temperature change in each place." },
    { key: `${base}:delta_sd`, label: "Spread between model runs", unit, scale: "seq", desc: "How much the model's repeat runs disagree in each place; more spread means less sure." },
    { key: `${base}:extrapolation`, label: "Beyond measured conditions", unit: "", scale: "seq", desc: "Places where the change goes beyond anything measured in this city (hatched)." },
    ...realizedVars.map((v): SyntheticLayer => ({ key: `${base}:realized_${v}`, label: `What changed: ${leverNames[v] ?? v}`, unit: leverUnits[v] ?? "", desc: "The change the full model actually applied, after keeping it within realistic limits." })),
  ];
  return list.map((l) => syntheticMeta({ ...l, group: "result" }, null));
}

/** Insert or replace a leading group in a catalogue. */
export function withGroup(groups: LayerGroup[], g: LayerGroup | null): LayerGroup[] {
  const rest = groups.filter((x) => !g || x.id !== g.id);
  return g && g.layers.length ? [g, ...rest] : rest;
}
