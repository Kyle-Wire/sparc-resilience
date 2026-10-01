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
    if (set.has(String(k))) {
      mask[r] = 1;
      n++;
    }
  }
  const spec: SelectionSpec = mode.by === "zone" ? { kind: "zones", values: keys } : { kind: "hex", size_m: mode.size_m, keys: keys.map(Number) };
  const what = mode.by === "zone" ? (keys.length === 1 ? `Zone ${keys[0]}` : `${keys.length} zones`) : `${keys.length} hex${keys.length === 1 ? "" : "es"} (${mode.size_m} m)`;
  return { kind: "selection", spec, mask, label: `${what} · ${fmtInt(n)} cells`, portable: true };
}

/** Zone label of a row: GridMeta.zones maps zone codes to their names when present. */
export function zoneValue(g: GridData, r: number): number | string {
  const z = g.zone[r];
  const names = g.meta.zones;
  return names.length && z >= 0 && z < names.length && typeof names[0] === "string" ? names[z] : z;
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
