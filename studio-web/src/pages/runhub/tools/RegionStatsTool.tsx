// Region stats (SPEC §6.5): for the current selection, n, area and residents; mean, sd and
// p10/p50/p90 inside vs the mean outside for the chosen layers; configured-scenario means
// with jackknife SE where folds exist. "Save as region" turns the selection into a reusable
// SelectionSpec region.
import { useMemo, useState } from "react";
import { createRegion, regionStats, useAnalysis, type LayerRegionStats, type RegionStatsRequest } from "../../../api/analysis";
import { errorMessage } from "../../../api/client";
import { invalidate } from "../../../api/resource";
import { Button } from "../../../components/ui/Button";
import { Chips } from "../../../components/ui/Chips";
import { EmptyState } from "../../../components/ui/EmptyState";
import { Table } from "../../../components/ui/Table";
import { toast } from "../../../stores/ui";
import { fmtInt, fmtNum, fmtSigned, unitLabel } from "../../../theme/format";
import { PinButton } from "../common";
import { confidenceWords, likelyText } from "../format";
import { allLayers, LayerSelect, selectionText, type ToolProps } from "./shared";

type LayerRow = { key: string; label: string; unit: string; decimals: number } & LayerRegionStats;

export function RegionStatsTool({ rid, groups, layerKey, selection, unit }: ToolProps) {
  const layers = useMemo(() => allLayers(groups), [groups]);
  const [extra, setExtra] = useState<string[]>([]);
  const [weights, setWeights] = useState(false);
  const configured = useMemo(() => layers.filter((l) => l.key.startsWith("sc:")).map((l) => ({ ref: `configured:${l.key.slice(3)}`, label: l.label })), [layers]);
  const [scen, setScen] = useState<string[] | null>(null);
  const scenarios = scen ?? configured.slice(0, 3).map((c) => c.ref);
  const wanted = [...new Set([layerKey, ...extra].filter((k): k is string => !!k && layers.some((l) => l.key === k && l.scale !== "cat")))];
  const body: RegionStatsRequest | null = selection.spec ? { selection: selection.spec, layers: wanted, weights: weights ? "people" : null, scenarios } : null;
  const res = useAnalysis(rid, "region", body, regionStats);
  const [name, setName] = useState("");
  const [saving, setSaving] = useState(false);

  const rows: LayerRow[] = res.data
    ? Object.entries(res.data.layers).map(([k, v]) => {
        const m = layers.find((l) => l.key === k);
        return { key: k, label: m?.label ?? k, unit: unitLabel(m?.unit ?? ""), decimals: m?.decimals ?? 2, ...v };
      })
    : [];
  const scenRows = res.data ? Object.entries(res.data.scenarios).map(([k, v]) => ({ ref: k, label: configured.find((c) => c.ref === k)?.label ?? k, ...v })) : [];

  const save = async () => {
    if (!selection.spec || !name.trim()) return;
    setSaving(true);
    try {
      const r = await createRegion(rid, name.trim(), selection.spec);
      invalidate(`run:${rid}:regions`);
      selection.set({ kind: "region", id: r.id }, { mask: selection.mask, n_cells: r.n_cells, label: r.name, source: "region" });
      toast("success", `Saved region "${r.name}"`, { body: "Scenarios and other runs of the project can reuse it." });
      setName("");
    } catch (e) {
      toast("error", "Could not save the region", { body: errorMessage(e) });
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="stack" data-tool="region">
      <p className="cap" role="status">
        {selectionText(selection)}
      </p>
      <div className="row">
        <LayerSelect groups={groups} value={null} onChange={(k) => k && setExtra((x) => (x.includes(k) ? x : [...x, k]))} label="Add a layer" filter={(l) => l.scale !== "cat"} />
        <label className="row cap">
          <input type="checkbox" checked={weights} onChange={(e) => setWeights(e.target.checked)} /> people-weighted
        </label>
      </div>
      {extra.length ? (
        <Chips
          label="Extra layers"
          items={extra.map((k) => ({ value: k, label: layers.find((l) => l.key === k)?.label ?? k }))}
          onRemove={(k) => setExtra((x) => x.filter((y) => y !== k))}
        />
      ) : null}
      {configured.length ? (
        <Chips
          label="Configured scenarios"
          items={configured.map((c) => ({ value: c.ref, label: c.label }))}
          selected={scenarios}
          onToggle={(v, on) => setScen(on ? [...scenarios, v] : scenarios.filter((x) => x !== v))}
        />
      ) : null}
      {!selection.spec ? (
        <EmptyState
          title="Select cells first"
          body="Draw a rectangle, circle or polygon, pick zones, or brush the legend histogram; the statistics then compare inside and outside."
        />
      ) : null}
      {res.error ? <EmptyState error={res.error} /> : null}
      {selection.spec && !res.data && !res.error ? <p className="cap">Computing…</p> : null}
      {res.data ? (
        <>
          <dl className="kv" aria-label="Selection summary">
            <dt>Cells</dt>
            <dd>{fmtInt(res.data.n_cells)}</dd>
            <dt>Area</dt>
            <dd>{fmtNum(res.data.area_km2, 2)} km²</dd>
            <dt>Residents</dt>
            <dd>{res.data.people === null ? "no people layer" : fmtInt(res.data.people)}</dd>
          </dl>
          <Table<LayerRow>
            caption="Inside vs outside the selection"
            csvName="region-stats"
            rowKey={(r) => r.key}
            columns={[
              { key: "label", label: "Layer", value: (r) => r.label },
              { key: "mean", label: "Mean inside", align: "right", value: (r) => r.mean, render: (r) => `${fmtNum(r.mean, r.decimals)} ${r.unit}`.trim() },
              { key: "sd", label: "sd", align: "right", value: (r) => r.sd, render: (r) => fmtNum(r.sd, r.decimals) },
              { key: "p10", label: "p10", align: "right", value: (r) => r.p10, render: (r) => fmtNum(r.p10, r.decimals) },
              { key: "p50", label: "Median", align: "right", value: (r) => r.p50, render: (r) => fmtNum(r.p50, r.decimals) },
              { key: "p90", label: "p90", align: "right", value: (r) => r.p90, render: (r) => fmtNum(r.p90, r.decimals) },
              { key: "out", label: "Mean outside", align: "right", value: (r) => r.mean_outside, render: (r) => `${fmtNum(r.mean_outside, r.decimals)} ${r.unit}`.trim() },
              {
                key: "diff",
                label: "Inside − outside",
                align: "right",
                value: (r) => (r.mean !== null && r.mean_outside !== null ? r.mean - r.mean_outside : null),
                render: (r) => (r.mean !== null && r.mean_outside !== null ? fmtSigned(r.mean - r.mean_outside, r.decimals) : "—"),
              },
            ]}
            rows={rows}
          />
          {scenRows.length ? (
            <Table
              caption="Scenario change inside vs outside"
              csvName="region-scenarios"
              rowKey={(r) => r.ref}
              columns={[
                { key: "label", label: "Scenario", value: (r) => r.label },
                { key: "in", label: "Inside", value: (r) => r.inside.estimate, render: (r) => `${likelyText(r.inside, unit)} ${confidenceWords(r.inside)}` },
                { key: "out", label: "Outside", value: (r) => r.outside.estimate, render: (r) => likelyText(r.outside, unit) },
                { key: "folds", label: "Jackknife SE", value: (r) => (r.has_folds ? "yes" : "no (no folds)") },
              ]}
              rows={scenRows}
            />
          ) : null}
          <div className="row" style={{ justifyContent: "space-between" }}>
            <form
              className="row"
              onSubmit={(e) => {
                e.preventDefault();
                void save();
              }}
            >
              <input type="text" value={name} onChange={(e) => setName(e.target.value)} placeholder="Region name" aria-label="Region name" />
              <Button type="submit" size="small" busy={saving} disabled={!name.trim()}>
                Save as region
              </Button>
            </form>
            <PinButton title="Region stats" snapshot={{ selection: selection.spec, label: selection.label, stats: res.data }} />
          </div>
        </>
      ) : null}
    </div>
  );
}
