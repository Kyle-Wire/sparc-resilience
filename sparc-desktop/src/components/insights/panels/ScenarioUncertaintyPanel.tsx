/**
 * ScenarioUncertaintyPanel — MC quantile spread map for the active
 * scenario, plus a one-line aggregate consensus diagnostic.
 */
import { useEffect, useMemo, useState } from "react";
import { Panel, PanelEmpty, Pill, Stat, StatGrid } from "@/components/ui/DesignSystem";
import { PanelGate } from "@/components/ui/PanelGate";
import { useManifest } from "@/hooks/useManifest";
import { useLinkedSelection } from "@/hooks/useLinkedSelection";
import { getScenarioUncertainty, type ScenarioUncertaintyResponse } from "@/lib/api";
import SpatialMap from "@/components/map/SpatialMap";
import { MAP_HEIGHT_DEFAULT } from "@/lib/design-tokens";

export default function ScenarioUncertaintyPanel() {
  const manifest = useManifest();
  const linked = useLinkedSelection();
  const present = !!manifest.stage("4");
  const [data, setData] = useState<ScenarioUncertaintyResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const scenarioId = linked.selection.scenario;

  useEffect(() => {
    if (!present) {
      setData(null);
      return;
    }
    setError(null);
    getScenarioUncertainty(scenarioId ?? undefined)
      .then(setData)
      .catch((e) => {
        setData(null);
        setError(e instanceof Error ? e.message : "no uncertainty");
      });
  }, [present, scenarioId, manifest.lastUpdated]);

  const consensus = useMemo<Record<string, unknown> | null>(() => {
    const rows = data?.mc_consensus_summary;
    if (!rows?.length) return null;
    return rows[0];
  }, [data]);

  const layerOptions = useMemo(() => {
    const f0 = data?.geojson?.features?.[0];
    if (!f0) return [];
    const skip = new Set(["lon", "lat", "geometry", "id"]);
    return Object.keys(f0.properties ?? {}).filter(
      (k) => !skip.has(k) && typeof (f0.properties ?? {})[k] === "number",
    );
  }, [data]);

  const defaultLayer = useMemo(() => {
    const prefer = ["std", "q95_minus_q05", "ci_width", "iqr"];
    for (const p of prefer) if (layerOptions.includes(p)) return p;
    return layerOptions[0] ?? "";
  }, [layerOptions]);

  const [layer, setLayer] = useState<string>("");
  const resolvedLayer = layer || defaultLayer;

  if (!present) {
    return (
      <PanelGate panelId="scenario_uncertainty" title="Scenario uncertainty" subtitle="stage 4 · MC">
        <Panel title="Scenario uncertainty" subtitle="stage 4 · MC">
          <PanelEmpty
            reason="No uncertainty artifact"
            hint="Run stage 4 with Monte-Carlo uncertainty enabled."
          />
        </Panel>
      </PanelGate>
    );
  }
  if (!data) {
    return (
      <Panel title="Scenario uncertainty" subtitle="stage 4 · MC">
        <PanelEmpty reason={error ?? "Loading…"} />
      </Panel>
    );
  }

  return (
    <Panel
      title="Scenario uncertainty"
      subtitle={resolvedLayer ? `layer · ${resolvedLayer}` : "MC quantiles"}
      actions={
        <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
          {scenarioId && <Pill color="var(--purple)">scenario · {scenarioId}</Pill>}
          {layerOptions.length > 1 && (
            <select
              value={resolvedLayer}
              onChange={(e) => setLayer(e.target.value)}
              className="mono"
              style={{
                fontSize: 11,
                padding: "3px 8px",
                border: "1px solid var(--line)",
                borderRadius: 4,
                background: "#fff",
              }}
            >
              {layerOptions.map((opt) => (
                <option key={opt} value={opt}>
                  {opt}
                </option>
              ))}
            </select>
          )}
        </div>
      }
      bodyPadding={0}
    >
      <div style={{ height: MAP_HEIGHT_DEFAULT, position: "relative" }}>
        <SpatialMap
          geojson={data.geojson}
          colorField={resolvedLayer || undefined}
          mode="scatter"
          height="100%"
        />
      </div>
      {consensus && (
        <div style={{ padding: 12, borderTop: "1px solid var(--line)" }}>
          <StatGrid>
            {Object.entries(consensus)
              .filter(([, v]) => typeof v === "number")
              .slice(0, 6)
              .map(([k, v]) => (
                <Stat key={k} label={k} value={(v as number).toFixed(3)} />
              ))}
          </StatGrid>
        </div>
      )}
    </Panel>
  );
}
