/**
 * PdpPanel — partial-dependence (response curve) for the active
 * variable across the available sources (neural_pde / gwrf /
 * causal_dose_response). Source switcher + variable switcher.
 */
import { useMemo, useState } from "react";
import { Panel, PanelEmpty, Pill, Btn } from "@/components/ui/DesignSystem";
import { PanelGate } from "@/components/ui/PanelGate";
import { useManifest } from "@/hooks/useManifest";
import { useLinkedSelection } from "@/hooks/useLinkedSelection";
import { useInsightsNavigate } from "@/hooks/InsightsProvider";
import { useResult } from "@/hooks/useResult";
import { getPdpCurves } from "@/lib/api";
import { SPARC_RAMP_HEX } from "@/lib/design-tokens";
import { useCanvas } from "./_useCanvas";
import type { PdpResponse, PdpSource } from "@/lib/types";

interface CurvePayload {
  grid_values?: number[];
  pdp_values?: number[];
  pdp_std?: number[];
  curve_fit?: { saturation_point?: number };
}

export default function PdpPanel() {
  const manifest = useManifest();
  const linked = useLinkedSelection();
  const navigate = useInsightsNavigate();
  const stage2Arts = manifest.stage("2")?.artifacts ?? {};
  const present =
    !!manifest.lookup("3", "dose_response_curves") ||
    !!manifest.lookup("2", "gwrf_condition_curves") ||
    Object.keys(stage2Arts).some((a) => a.startsWith("v2_neural_pdp::"));

  const { data } = useResult<PdpResponse>(
    present ? "s3:pdp_curves" : null,
    getPdpCurves,
  );
  const [source, setSource] = useState<PdpSource | null>(null);

  const meta = data?._meta;
  const availableSources = useMemo<PdpSource[]>(() => {
    return (meta?.available_sources ?? []).filter((s) => {
      const payload = meta?.by_source?.[s];
      return payload && Object.keys(payload).length > 0;
    });
  }, [meta]);

  const effectiveSource = useMemo<PdpSource | null>(() => {
    if (source && availableSources.includes(source)) return source;
    return availableSources[0] ?? null;
  }, [source, availableSources]);

  const curves = useMemo<Record<string, CurvePayload> | null>(() => {
    if (effectiveSource && meta?.by_source) return meta.by_source[effectiveSource] ?? null;
    if (!data) return null;
    // Fallback: treat all non-meta top-level keys as curve variables
    return Object.fromEntries(
      Object.entries(data).filter(([k]) => !k.startsWith("_")) as [string, CurvePayload][],
    );
  }, [data, meta, effectiveSource]);

  const variables = curves ? Object.keys(curves) : [];
  const activeVar = linked.selection.variable;
  const selectedVar = useMemo(() => {
    if (!variables.length) return null;
    return variables.find((v) => v === activeVar) ?? variables[0];
  }, [variables, activeVar]);

  const canvasRef = useCanvas((ctx, w, h) => {
    if (!selectedVar || !curves) return;
    const curve = curves[selectedVar];
    const gridVals = curve?.grid_values ?? [];
    const pdpVals = curve?.pdp_values ?? [];
    const pdpStd = curve?.pdp_std ?? [];
    if (!gridVals.length || !pdpVals.length) return;

    ctx.strokeStyle = "var(--line, #c9c2b3)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(40, 10);
    ctx.lineTo(40, h - 25);
    ctx.lineTo(w - 10, h - 25);
    ctx.stroke();

    const minG = Math.min(...gridVals),
      maxG = Math.max(...gridVals);
    const minP = Math.min(...pdpVals.map((v, i) => v - (pdpStd[i] ?? 0)));
    const maxP = Math.max(...pdpVals.map((v, i) => v + (pdpStd[i] ?? 0)));
    const rG = maxG - minG || 1,
      rP = maxP - minP || 1;

    if (pdpStd.length === pdpVals.length) {
      ctx.fillStyle = SPARC_RAMP_HEX[0] + "22";
      ctx.beginPath();
      gridVals.forEach((g, i) => {
        const x = 42 + ((g - minG) / rG) * (w - 54);
        const y = h - 35 - ((pdpVals[i] + pdpStd[i] - minP) / rP) * (h - 45) + 10;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      for (let i = gridVals.length - 1; i >= 0; i--) {
        const x = 42 + ((gridVals[i] - minG) / rG) * (w - 54);
        const y = h - 35 - ((pdpVals[i] - pdpStd[i] - minP) / rP) * (h - 45) + 10;
        ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.fill();
    }

    ctx.strokeStyle = SPARC_RAMP_HEX[0];
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    gridVals.forEach((g, i) => {
      const x = 42 + ((g - minG) / rG) * (w - 54);
      const y = h - 35 - ((pdpVals[i] - minP) / rP) * (h - 45) + 10;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    ctx.fillStyle = "var(--ink-2, #6e6358)";
    ctx.font = "9px 'JetBrains Mono', monospace";
    ctx.textAlign = "center";
    ctx.fillText(minG.toFixed(2), 42, h - 10);
    ctx.fillText(maxG.toFixed(2), w - 10, h - 10);
    ctx.textAlign = "right";
    ctx.fillText(maxP.toFixed(2), 38, 18);
    ctx.fillText(minP.toFixed(2), 38, h - 28);

    const satPoint = curve?.curve_fit?.saturation_point;
    if (satPoint != null && Number.isFinite(satPoint) && satPoint >= minG && satPoint <= maxG) {
      const sx = 42 + ((satPoint - minG) / rG) * (w - 54);
      ctx.save();
      ctx.strokeStyle = "var(--crimson, #dc2626)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(sx, 12);
      ctx.lineTo(sx, h - 28);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "var(--crimson, #dc2626)";
      ctx.font = "bold 8px 'JetBrains Mono', monospace";
      ctx.textAlign = "center";
      ctx.fillText("sat.", sx, 10);
      ctx.restore();
    }
  }, [selectedVar, curves]);

  if (!present) {
    return (
      <PanelGate panelId="pdp" title="Response curves" subtitle="PDP">
        <Panel title="Response curves" subtitle="PDP">
          <PanelEmpty
            reason="No PDP curves yet"
            hint="Run stage 2 (Models) or stage 3 (Causal) to generate response curves."
            action={navigate && <Btn small onClick={() => navigate("Run")}>Go to Run page →</Btn>}
          />
        </Panel>
      </PanelGate>
    );
  }
  if (!variables.length) {
    return (
      <Panel title="Response curves" subtitle={effectiveSource ?? "PDP"}>
        <PanelEmpty reason="Loading…" />
      </Panel>
    );
  }

  return (
    <Panel
      title="Response curves"
      subtitle={`${effectiveSource ?? "pdp"} · ${selectedVar}`}
      actions={
        <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
          {selectedVar === activeVar && <Pill color="var(--purple)">linked</Pill>}
          {availableSources.length > 1 && (
            <select
              value={effectiveSource ?? ""}
              onChange={(e) => setSource(e.target.value as PdpSource)}
              className="mono"
              style={{
                fontSize: 11,
                padding: "3px 8px",
                border: "1px solid var(--line)",
                borderRadius: 4,
                background: "#fff",
              }}
            >
              {availableSources.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          )}
          <select
            value={selectedVar ?? ""}
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
            {variables.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </div>
      }
    >
      <canvas ref={canvasRef} style={{ width: "100%", height: 280, display: "block" }} />
    </Panel>
  );
}
