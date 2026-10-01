// Shared pieces of the Map tab's analysis-tools drawer (SPEC §6.5): the props every tool
// receives, a layer select, and the "select cells from a mask" helper (upload as a blob and
// store it as the run's selection).
import type { LayerGroup, LayerMeta } from "../../../api/types";
import { errorMessage } from "../../../api/client";
import { uploadMask } from "../../../api/analysis";
import type { LayerValues } from "../../../map/data";
import type { GridData } from "../../../map/grid";
import type { RunSelection } from "../../../stores/selection";
import { toast } from "../../../stores/ui";
import { fmtCount } from "../../../theme/format";

export type ToolProps = {
  rid: string;
  grid: GridData;
  groups: LayerGroup[];
  /** The layer on the map. */
  layerKey: string | null;
  selection: RunSelection;
  loadLayer: (meta: LayerMeta) => Promise<LayerValues>;
  /** Display unit of the target ("°F"). */
  unit: string;
};

export function allLayers(groups: LayerGroup[]): LayerMeta[] {
  return groups.flatMap((g) => g.layers);
}

export function LayerSelect({
  groups,
  value,
  onChange,
  label,
  filter,
}: {
  groups: LayerGroup[];
  value: string | null;
  onChange: (key: string) => void;
  label: string;
  filter?: (l: LayerMeta) => boolean;
}) {
  return (
    <label className="row cap">
      {label}
      <select value={value ?? ""} onChange={(e) => onChange(e.target.value)} aria-label={label} style={{ maxWidth: "18em" }}>
        {value === null ? <option value="">Choose…</option> : null}
        {groups.map((g) => {
          const ls = g.layers.filter((l) => !filter || filter(l));
          return ls.length ? (
            <optgroup key={g.id} label={g.label}>
              {ls.map((l) => (
                <option key={l.key} value={l.key}>
                  {l.label}
                </option>
              ))}
            </optgroup>
          ) : null;
        })}
      </select>
    </label>
  );
}

/** Store a row mask as the run's selection (uploaded as a mask blob so it fits in the URL). */
export async function selectMask(rid: string, selection: RunSelection, mask: Uint8Array, label: string, source: string): Promise<boolean> {
  try {
    const up = await uploadMask(rid, mask);
    selection.set(up.spec, { mask, n_cells: up.n_cells, label, source });
    return true;
  } catch (e) {
    toast("error", "Could not keep the selection", { body: errorMessage(e) });
    return false;
  }
}

/** Plain description of the current selection. */
export function selectionText(s: RunSelection): string {
  if (!s.spec) return "No selection: draw on the map, brush the legend histogram or a chart.";
  const n = s.nCells !== null ? fmtCount(s.nCells, "cell") : "cells";
  return `${s.label ? `${s.label} · ` : ""}${n}`;
}
