// Relationships (SPEC §6.5): hexbin of layer X vs layer Y with the current selection
// highlighted, Spearman ρ and the binned-mean line (POST /api/runs/{rid}/stats/hexbin).
// Brushing a rectangle on the plot selects those cells on the map.
import { useMemo, useState } from "react";
import { hexbin, useAnalysis, type HexbinRequest } from "../../../api/analysis";
import { HexbinScatter } from "../../../charts";
import { EmptyState } from "../../../components/ui/EmptyState";
import { fmtNum, unitLabel } from "../../../theme/format";
import { maskFromBox } from "../format";
import { allLayers, LayerSelect, selectMask, type ToolProps } from "./shared";

export function RelationshipsTool({ rid, groups, layerKey, selection, loadLayer }: ToolProps) {
  const layers = useMemo(() => allLayers(groups).filter((l) => l.scale !== "cat"), [groups]);
  const [x, setX] = useState<string | null>(null);
  const [y, setY] = useState<string | null>(null);
  const yKey = y ?? (layerKey && layers.some((l) => l.key === layerKey) ? layerKey : (layers[0]?.key ?? null));
  const xKey = x ?? layers.find((l) => l.key === "dist_train_m" && l.key !== yKey)?.key ?? layers.find((l) => l.key !== yKey)?.key ?? null;
  const xm = layers.find((l) => l.key === xKey);
  const ym = layers.find((l) => l.key === yKey);
  const body: HexbinRequest | null = xKey && yKey ? { x: xKey, y: yKey, bins: 60, ...(selection.spec ? { selection: selection.spec } : {}) } : null;
  const res = useAnalysis(rid, "hexbin", body, hexbin);
  const [busy, setBusy] = useState(false);

  const onBrush = async (box: { x: [number, number]; y: [number, number] } | null) => {
    if (!box) {
      selection.clear();
      return;
    }
    if (!xm || !ym) return;
    setBusy(true);
    try {
      const [xv, yv] = await Promise.all([loadLayer(xm), loadLayer(ym)]);
      // Plot coordinates are display values (× mult); compare in the same units.
      const sx = Array.from(xv, (v) => v * (xm.mult || 1));
      const sy = Array.from(yv, (v) => v * (ym.mult || 1));
      const mask = maskFromBox(sx, sy, box);
      const label = `${xm.label} ${fmtNum(box.x[0], xm.decimals)}–${fmtNum(box.x[1], xm.decimals)}, ${ym.label} ${fmtNum(box.y[0], ym.decimals)}–${fmtNum(box.y[1], ym.decimals)}`;
      await selectMask(rid, selection, mask, label, "relationships");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="stack" data-tool="relationships">
      <div className="row">
        <LayerSelect groups={groups} value={xKey} onChange={setX} label="X" filter={(l) => l.scale !== "cat"} />
        <LayerSelect groups={groups} value={yKey} onChange={setY} label="Y" filter={(l) => l.scale !== "cat"} />
        {busy ? <span className="cap">Selecting…</span> : null}
      </div>
      {res.error ? <EmptyState error={res.error} /> : null}
      {!res.data && !res.error && body ? <p className="cap">Computing…</p> : null}
      {res.data && xm && ym ? (
        <HexbinScatter
          title={`${ym.label} vs ${xm.label}`}
          bins={{ x_edges: res.data.x_edges, y_edges: res.data.y_edges, counts: res.data.counts, sel_counts: res.data.sel_counts }}
          xLabel={xm.label}
          yLabel={ym.label}
          xUnit={unitLabel(xm.unit) || undefined}
          yUnit={unitLabel(ym.unit) || undefined}
          spearman={res.data.spearman}
          meanLine={res.data.binned_mean}
          onBrush={(b) => void onBrush(b)}
          caption={`Spearman ρ = ${fmtNum(res.data.spearman, 2)}; the line is the mean of ${ym.label} in each ${xm.label} bin.${res.data.sel_counts ? " Orange: the current selection." : ""} Drag a rectangle to select those cells.`}
        />
      ) : null}
    </div>
  );
}
