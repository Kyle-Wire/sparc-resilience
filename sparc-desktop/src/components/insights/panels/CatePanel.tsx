/**
 * CatePanel — heterogeneous treatment effect (CATE) map for the
 * active treatment variable.
 *
 * Variable list comes from `/results/cate/variables`; spatial surface
 * comes from `/results/cate/map?variable=...`. Toggling the
 * uncertainty overlay refetches with `with_uncertainty=1`.
 *
 * Wired to the linked-selection store: picking a variable here also
 * filters DoseResponsePanel + PdpPanel.
 */
import { useMemo, useState } from "react";
import { Panel, PanelEmpty, Pill, Btn } from "@/components/ui/DesignSystem";
import { useManifest } from "@/hooks/useManifest";
import { useLinkedSelection } from "@/hooks/useLinkedSelection";
import { useInsightsNavigate } from "@/hooks/InsightsProvider";
import { useResult } from "@/hooks/useResult";
import { getCateMap, getCateMapVariables } from "@/lib/api";
import SpatialMap from "@/components/map/SpatialMap";
import ResizableMapWrapper from "@/components/map/ResizableMapWrapper";
import { MAP_HEIGHT_DEFAULT } from "@/lib/design-tokens";
import type { GeoJsonData } from "@/lib/types";

export default function CatePanel() {
  const manifest = useManifest();
  const linked = useLinkedSelection();
  const navigate = useInsightsNavigate();
  const stage3 = manifest.stage("3");
  const present = !!stage3 && Object.keys(stage3.artifacts ?? {}).some((a) => a.startsWith("cate_multiplier::"));

  const [withUncertainty, setWithUncertainty] = useState<boolean>(true);

  const { data: varsData } = useResult<{ variables: string[]; empty_reason?: string }>(
    present ? "s3:cate_variables" : null,
    getCateMapVariables,
  );
  const vars: string[] = varsData?.variables ?? [];
  const emptyHint = useMemo(() => {
    if (!varsData || vars.length > 0) return null;
    return (varsData as { empty_reason?: string }).empty_reason ?? null;
  }, [varsData, vars.length]);

  const activeVar = linked.selection.variable;
  const resolvedVar = useMemo(() => {
    if (!vars.length) return null;
    return vars.find((v) => v === activeVar) ?? vars[0];
  }, [vars, activeVar]);

  const mapKey = resolvedVar ? `s3:cate_map:${resolvedVar}:${withUncertainty}` : null;
  const { data: geo } = useResult<GeoJsonData>(
    mapKey,
    () => getCateMap(resolvedVar!, withUncertainty) as Promise<GeoJsonData>,
  );

  // Derive the actual property name the backend writes (cate_{variable})
  const colorField = resolvedVar ? `cate_${resolvedVar}` : undefined;
  // Significance mask: coef_ns_{variable} is true when the 90% CI brackets zero
  const dimField = (withUncertainty && resolvedVar) ? `coef_ns_${resolvedVar}` : undefined;

  // Compute symmetric domain centred at zero so the rdbu palette's neutral
  // midpoint always lands at zero (no treatment effect). Falls back to
  // undefined (auto) when no data is loaded yet.
  const cateDomain = useMemo<[number, number] | undefined>(() => {
    if (!geo || !colorField) return undefined;
    let min = Infinity, max = -Infinity;
    for (const f of geo.features) {
      const v = f.properties[colorField];
      if (typeof v === "number" && isFinite(v)) {
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
    if (!isFinite(min)) return undefined;
    const maxAbs = Math.max(Math.abs(min), Math.abs(max));
    return [-maxAbs, maxAbs];
  }, [geo, colorField]);

  if (!present) {
    return (
      <Panel title="CATE — heterogeneous treatment effects" subtitle="stage 3 · causal">
        <PanelEmpty
          reason="No CATE multipliers"
          hint="Run the Causal stage with treatment variables to produce CATE surfaces."
          action={navigate && <Btn small onClick={() => navigate("Run")}>Go to Run page →</Btn>}
        />
      </Panel>
    );
  }

  if (vars.length === 0) {
    return (
      <Panel title="CATE — heterogeneous treatment effects" subtitle="stage 3 · causal">
        <PanelEmpty reason={emptyHint ?? "Loading…"} hint={emptyHint ? "Re-run the Causal stage to populate." : undefined} />
      </Panel>
    );
  }

  return (
    <Panel
      title="CATE — heterogeneous treatment effects"
      subtitle={resolvedVar ? `treatment · ${resolvedVar}` : "stage 3"}
      actions={
        <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
          {resolvedVar && resolvedVar === activeVar && <Pill color="var(--purple)">linked</Pill>}
          <label className="mono" style={{ fontSize: 11, color: "var(--ink-2)", display: "flex", gap: 4 }}>
            <input
              type="checkbox"
              checked={withUncertainty}
              onChange={(e) => setWithUncertainty(e.target.checked)}
            />
            uncertainty
          </label>
          <select
            value={resolvedVar ?? ""}
            onChange={(e) => linked.setVariable(e.target.value)}
            className="mono"
            style={{
              fontSize: 11,
              padding: "3px 8px",
              border: "1px solid var(--line)",
              borderRadius: 4,
              background: "#fff",
            }}
          >
            {vars.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </div>
      }
      bodyPadding={0}
    >
      <ResizableMapWrapper defaultHeight={MAP_HEIGHT_DEFAULT}>
        <SpatialMap
          geojson={geo}
          colorField={colorField}
          dimField={dimField}
          palette="rdbu"
          domainOverride={cateDomain}
          mode="scatter"
          height="100%"
          expandable
          onFeatureClick={(props) => {
            const id = props.id != null ? String(props.id) : null;
            if (id) linked.setIds([id]);
          }}
        />
      </ResizableMapWrapper>
    </Panel>
  );
}
