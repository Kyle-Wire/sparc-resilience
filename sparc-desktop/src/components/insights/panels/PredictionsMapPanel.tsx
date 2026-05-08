/**
 * PredictionsMapPanel — canonical spatial predictions map (stage 2).
 *
 * Wraps the existing <SpatialMap> component and feeds it the active
 * layer (model variant). Brushing on the map writes the bbox into the
 * linked-selection store so other panels can filter on it.
 */
import { useMemo, useState } from "react";
import { Panel, PanelEmpty, Pill } from "@/components/ui/DesignSystem";
import { useManifest } from "@/hooks/useManifest";
import { useLinkedSelection } from "@/hooks/useLinkedSelection";
import { useResult } from "@/hooks/useResult";
import { getSpatialCvPredictions } from "@/lib/api";
import SpatialMap from "@/components/map/SpatialMap";
import { MAP_HEIGHT_DEFAULT } from "@/lib/design-tokens";
import type { GeoJsonData } from "@/lib/types";

export default function PredictionsMapPanel() {
  const manifest = useManifest();
  const linked = useLinkedSelection();
  const present = !!manifest.lookup("2", "spatial_cv_predictions");
  const { data: geo } = useResult<GeoJsonData>(
    present ? "s2:spatial_cv_predictions" : null,
    getSpatialCvPredictions,
  );
  const [layer, setLayer] = useState<string>("");

  // Available numeric columns in the GeoJSON properties → layer options.
  const layerOptions = useMemo(() => {
    const f0 = geo?.features?.[0];
    if (!f0) return [];
    const skip = new Set(["lon", "lat", "geometry", "id"]);
    return Object.entries(f0.properties ?? {})
      .filter(
        ([k, v]) => !skip.has(k) && typeof v === "number" && Number.isFinite(v as number),
      )
      .map(([k]) => k);
  }, [geo]);

  const resolvedLayer = useMemo(() => {
    if (layer && layerOptions.includes(layer)) return layer;
    // Prefer increasingly-fancy ensemble outputs, then any numeric column.
    const preference = [
      "predicted_full",
      "predicted",
      "pred_ensemble",
      "pred_meta",
      "pred_v2_neural",
      "pred_neural",
      "pred_gwrf",
      "pred_mgwr",
      "pred_ggpgam",
      "pred_ols",
    ];
    for (const candidate of preference) {
      if (layerOptions.includes(candidate)) return candidate;
    }
    // Fall back to the first column whose name suggests it's a prediction.
    const predLike = layerOptions.find((c) => /^pred(_|icted)/i.test(c));
    return predLike ?? layerOptions[0] ?? "";
  }, [layer, layerOptions]);

  if (!present) {
    return (
      <Panel title="Predictions map" subtitle="stage 2 · spatial CV">
        <PanelEmpty
          reason="No predictions yet"
          hint="Run stage 2 (Models) to produce spatial cross-validation predictions."
        />
      </Panel>
    );
  }

  return (
    <Panel
      title="Predictions map"
      subtitle={resolvedLayer ? `layer · ${resolvedLayer}` : "spatial CV"}
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
      <div style={{ height: MAP_HEIGHT_DEFAULT, position: "relative" }}>
        <SpatialMap
          geojson={geo}
          colorField={resolvedLayer || undefined}
          mode="scatter"
          height="100%"
          onFeatureClick={(props) => {
            const id = props.id != null ? String(props.id) : null;
            if (id) linked.setIds([id]);
          }}
        />
      </div>
    </Panel>
  );
}
