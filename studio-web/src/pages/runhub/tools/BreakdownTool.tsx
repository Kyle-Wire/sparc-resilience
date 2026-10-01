// Breakdown (SPEC §6.5): any layer grouped by zone, quantile of any layer, hexagon
// (250/500 m), category of a categorical layer, or CV fold; a box or bar chart, optionally
// people-weighted (POST /api/runs/{rid}/stats/breakdown).
import { useMemo, useState } from "react";
import { breakdown, useAnalysis, type BreakdownBy, type BreakdownRequest } from "../../../api/analysis";
import { Bars, BoxStrip } from "../../../charts";
import { EmptyState } from "../../../components/ui/EmptyState";
import { Seg } from "../../../components/ui/Seg";
import { fmtInt, unitLabel } from "../../../theme/format";
import { allLayers, LayerSelect, type ToolProps } from "./shared";

type ByKind = BreakdownBy["kind"];

export function BreakdownTool({ rid, grid, groups, layerKey }: ToolProps) {
  const layers = useMemo(() => allLayers(groups), [groups]);
  const numeric = layers.filter((l) => l.scale !== "cat");
  const cats = layers.filter((l) => l.scale === "cat");
  const hasZones = grid.meta.zones.length > 0;
  const [value, setValue] = useState<string | null>(null);
  const v = value ?? (layerKey && numeric.some((l) => l.key === layerKey) ? layerKey : (numeric[0]?.key ?? null));
  const [kind, setKind] = useState<ByKind>(hasZones ? "zone" : "quantile");
  const [byLayer, setByLayer] = useState<string | null>(null);
  const [q, setQ] = useState(5);
  const [size, setSize] = useState<250 | 500>(250);
  const [weights, setWeights] = useState(false);
  const [stat, setStat] = useState<"box" | "mean">("box");
  const qLayer = byLayer && numeric.some((l) => l.key === byLayer) ? byLayer : (numeric.find((l) => l.key !== v)?.key ?? v);
  const cLayer = byLayer && cats.some((l) => l.key === byLayer) ? byLayer : (cats[0]?.key ?? null);
  let by: BreakdownBy | null = null;
  if (kind === "zone" || kind === "fold") by = { kind };
  else if (kind === "hex") by = { kind, size_m: size };
  else if (kind === "quantile" && qLayer) by = { kind, layer: qLayer, q };
  else if (kind === "category" && cLayer) by = { kind, layer: cLayer };
  const body: BreakdownRequest | null = v && by ? { value: v, by, weights: weights ? "people" : null, stat } : null;
  const res = useAnalysis(rid, "breakdown", body, breakdown);
  const meta = layers.find((l) => l.key === v);
  const u = unitLabel(meta?.unit ?? "");
  const byLabel =
    kind === "zone"
      ? "Zone"
      : kind === "fold"
        ? "CV fold"
        : kind === "hex"
          ? `Hexagon (${size} m)`
          : kind === "quantile"
            ? `${layers.find((l) => l.key === qLayer)?.label ?? "layer"} quantile`
            : (layers.find((l) => l.key === cLayer)?.label ?? "Category");
  const groupsOut = res.data?.groups ?? [];
  const shown = kind === "hex" ? groupsOut.slice(0, 40) : groupsOut;
  return (
    <div className="stack" data-tool="breakdown">
      <div className="row">
        <LayerSelect groups={groups} value={v} onChange={setValue} label="Value" filter={(l) => l.scale !== "cat"} />
        <Seg<ByKind>
          label="Group by"
          size="small"
          value={kind}
          onChange={setKind}
          options={[
            { value: "zone", label: "Zone", disabled: !hasZones, title: hasZones ? undefined : "This run has no zones" },
            { value: "quantile", label: "Quantile" },
            { value: "hex", label: "Hexagon" },
            { value: "category", label: "Category", disabled: !cats.length },
            { value: "fold", label: "Fold" },
          ]}
        />
      </div>
      <div className="row">
        {kind === "quantile" ? (
          <>
            <LayerSelect groups={groups} value={qLayer} onChange={setByLayer} label="Quantiles of" filter={(l) => l.scale !== "cat"} />
            <Seg<number> label="Number of quantiles" size="small" value={q} onChange={setQ} options={[4, 5, 10].map((n) => ({ value: n, label: String(n) }))} />
          </>
        ) : null}
        {kind === "category" ? <LayerSelect groups={groups} value={cLayer} onChange={setByLayer} label="Categories of" filter={(l) => l.scale === "cat"} /> : null}
        {kind === "hex" ? (
          <Seg<250 | 500>
            label="Hexagon size"
            size="small"
            value={size}
            onChange={setSize}
            options={[
              { value: 250, label: "250 m" },
              { value: 500, label: "500 m" },
            ]}
          />
        ) : null}
        <Seg<"box" | "mean">
          label="Chart"
          size="small"
          value={stat}
          onChange={setStat}
          options={[
            { value: "box", label: "Box" },
            { value: "mean", label: "Mean" },
          ]}
        />
        <label className="row cap">
          <input type="checkbox" checked={weights} onChange={(e) => setWeights(e.target.checked)} /> people-weighted
        </label>
      </div>
      {res.error ? <EmptyState error={res.error} /> : null}
      {!res.data && !res.error && body ? <p className="cap">Computing…</p> : null}
      {res.data && meta ? (
        stat === "box" ? (
          <BoxStrip
            title={`${meta.label} by ${byLabel.toLowerCase()}`}
            units={u}
            groups={shown.map((g) => ({ label: g.label, q: g.q, n: g.n, mean: g.mean }))}
            valueLabel={meta.label}
            unit={u}
            groupLabel={byLabel}
            decimals={meta.decimals}
            caption={`${fmtInt(groupsOut.length)} groups${groupsOut.length > shown.length ? ` (first ${shown.length} shown)` : ""}${weights ? ", people-weighted" : ""}.`}
          />
        ) : (
          <Bars
            title={`Mean ${meta.label} by ${byLabel.toLowerCase()}`}
            units={u}
            categories={shown.map((g) => g.label)}
            categoryLabel={byLabel}
            series={[{ id: "mean", label: `Mean ${meta.label}`, values: shown.map((g) => g.mean) }]}
            valueLabel={meta.label}
            unit={u}
            decimals={meta.decimals}
            caption={`${fmtInt(groupsOut.length)} groups${weights ? ", people-weighted means" : ""}.`}
          />
        )
      ) : null}
    </div>
  );
}
