/**
 * ScenarioMapPanel — predicted-value or delta surface for a saved
 * scenario (stage 4). Reads scenario from linked-selection.scenario;
 * defaults to first scenario.
 */
import { useMemo, useState } from "react";
import { Panel, PanelEmpty, Pill, Btn } from "@/components/ui/DesignSystem";
import { PanelGate } from "@/components/ui/PanelGate";
import { useManifest } from "@/hooks/useManifest";
import { useLinkedSelection } from "@/hooks/useLinkedSelection";
import { useInsightsNavigate } from "@/hooks/InsightsProvider";
import { useResult } from "@/hooks/useResult";
import { getScenarioDetail } from "@/lib/api";
import SpatialMap from "@/components/map/SpatialMap";
import ResizableMapWrapper from "@/components/map/ResizableMapWrapper";
import { MAP_HEIGHT_DEFAULT } from "@/lib/design-tokens";
import type { ScenarioDetail } from "@/lib/types";

export default function ScenarioMapPanel() {
  const manifest = useManifest();
  const linked = useLinkedSelection();
  const navigate = useInsightsNavigate();
  const present = !!manifest.stage("4");
  const { data: detail } = useResult<ScenarioDetail>(
    present ? "s4:scenario_detail" : null,
    getScenarioDetail,
  );
  const [layer, setLayer] = useState<string>("");

  const features = detail?.geojson?.features ?? [];
  const props0 = features[0]?.properties ?? {};

  const layerOptions = useMemo(
    () =>
      Object.keys(props0).filter(
        (k) => k.startsWith("delta_") || k.startsWith("pred_") || k === "predicted",
      ),
    [props0],
  );

  const resolvedLayer =
    layer ||
    (linked.selection.scenario && layerOptions.includes(`pred_${linked.selection.scenario}`)
      ? `pred_${linked.selection.scenario}`
      : layerOptions[0] ?? "predicted");

  if (!present) {
    return (
      <PanelGate panelId="scenario_map" title="Scenario map" subtitle="stage 4">
        <Panel title="Scenario map" subtitle="stage 4">
          <PanelEmpty
            reason="No scenarios run"
            hint="Configure scenarios on the Scenarios page and run stage 4."
            action={navigate && <Btn small onClick={() => navigate("Run")}>Go to Run page →</Btn>}
          />
        </Panel>
      </PanelGate>
    );
  }
  if (!features.length) {
    return (
      <Panel title="Scenario map" subtitle="stage 4">
        <PanelEmpty reason="Loading…" />
      </Panel>
    );
  }

  return (
    <Panel
      title="Scenario map"
      subtitle={resolvedLayer ? `layer · ${resolvedLayer}` : "stage 4"}
      actions={
        <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
          {linked.selection.bbox && <Pill color="var(--crimson)">brushed</Pill>}
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
      <ResizableMapWrapper defaultHeight={MAP_HEIGHT_DEFAULT}>
        <SpatialMap
          geojson={detail?.geojson ?? null}
          colorField={resolvedLayer}
          mode="scatter"
          height="100%"
          expandable
          onFeatureClick={(p) => {
            const id = p.id != null ? String(p.id) : null;
            if (id) linked.setIds([id]);
          }}
        />
      </ResizableMapWrapper>
    </Panel>
  );
}
