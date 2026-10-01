// Pick: click a cell to select its zone or its hexagon (250/500 m, same keys as
// sparc.core.planner.hex_ids). Shift-click adds to the current pick.
import type { SelectionSpec } from "../../api/types";
import { fmtInt } from "../../theme/format";
import { hexKey, rasterToXY, rowAt, type GridData } from "../grid";
import type { MapTool, ToolResult } from "./types";

export type PickMode = { by: "zone" } | { by: "hex"; size_m: 250 | 500 };

export function hexKeyOfRow(g: GridData, r: number, size_m: number): number {
  const [x, y] = rasterToXY(g, g.ix[r] + 0.5, g.ny - 1 - g.iy[r] + 0.5);
  return hexKey(x, y, size_m);
}

/** Selection for a set of zones or hex keys, with its local mask. */
export function pickSelection(g: GridData, mode: PickMode, keys: (number | string)[]): ToolResult {
  const set = new Set(keys.map(String));
  const mask = new Uint8Array(g.n);
  let n = 0;
  for (let r = 0; r < g.n; r++) {
    const k = mode.by === "zone" ? zoneValue(g, r) : hexKeyOfRow(g, r, mode.size_m);
    if (k !== null && set.has(String(k))) {
      mask[r] = 1;
      n++;
    }
  }
  const spec: SelectionSpec = mode.by === "zone" ? { kind: "zones", values: keys } : { kind: "hex", size_m: mode.size_m, keys: keys.map(Number) };
  const what = mode.by === "zone" ? (keys.length === 1 ? `Zone ${keys[0]}` : `${keys.length} zones`) : `${keys.length} hex${keys.length === 1 ? "" : "es"} (${mode.size_m} m)`;
  return { kind: "selection", spec, mask, label: `${what} · ${fmtInt(n)} cells`, portable: true };
}

/**
 * Zone of a row, or null. grid.bin's `zone` is an index into GridMeta.zones for numeric and
 * string zones alike (−1 = no zone, api.md §6.2), so codes that do not fit int16 survive.
 */
export function zoneValue(g: GridData, r: number): number | string | null {
  const z = g.zone[r];
  const zones = g.meta.zones;
  return z >= 0 && z < zones.length ? zones[z] : null;
}

export function pickTool(g: GridData, mode: PickMode, additive: () => boolean = () => false): MapTool {
  let keys: (number | string)[] = [];
  return {
    id: "pick",
    pans: false,
    down: () => null,
    move: () => null,
    up(p) {
      const r = rowAt(g, p.px, p.py);
      if (r < 0) return null;
      const k = mode.by === "zone" ? zoneValue(g, r) : hexKeyOfRow(g, r, mode.size_m);
      if (k === null) return null; // a cell without a zone picks nothing
      if (additive()) {
        keys = keys.some((x) => String(x) === String(k)) ? keys.filter((x) => String(x) !== String(k)) : [...keys, k];
      } else keys = [k];
      return keys.length ? pickSelection(g, mode, keys) : null;
    },
    finish: () => null,
    cancel() {
      keys = [];
    },
    preview: () => null,
  };
}
