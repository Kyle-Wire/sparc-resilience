/**
 * CorrelogramPanel — Moran's I vs lag distance, multi-variable.
 * Stage 0 spatial autocorrelation overview.
 */
import { Panel, PanelEmpty, Btn } from "@/components/ui/DesignSystem";
import { useManifest } from "@/hooks/useManifest";
import { useResult } from "@/hooks/useResult";
import { useInsightsNavigate } from "@/hooks/InsightsProvider";
import { getCorrelogramData } from "@/lib/api";
import { SPARC_RAMP_HEX } from "@/lib/design-tokens";
import { useCanvas } from "./_useCanvas";
import type { CorrelogramData } from "@/lib/types";

export default function CorrelogramPanel() {
  const manifest = useManifest();
  const navigate = useInsightsNavigate();
  const present = !!manifest.lookup("0", "correlogram_results");
  const { data } = useResult<CorrelogramData>(
    present ? "s0:correlogram" : null,
    getCorrelogramData,
  );

  const results = data?.individual_results;
  const varNames = results ? Object.keys(results) : [];

  const canvasRef = useCanvas((ctx, w, h) => {
    if (!varNames.length || !results) return;

    const pad = { top: 20, right: 20, bottom: 35, left: 50 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    let allMinI = 0,
      allMaxI = 0,
      allMaxLag = 0;
    for (const name of varNames) {
      const cr = results[name].correlogram_results;
      allMinI = Math.min(allMinI, ...cr.morans_i_values);
      allMaxI = Math.max(allMaxI, ...cr.morans_i_values);
      allMaxLag = Math.max(allMaxLag, ...cr.lag_distances);
    }
    const rangeI = allMaxI - allMinI || 1;

    // Frame
    ctx.strokeStyle = "var(--line, #c9c2b3)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top);
    ctx.lineTo(pad.left, h - pad.bottom);
    ctx.lineTo(w - pad.right, h - pad.bottom);
    ctx.stroke();

    // Axis labels
    ctx.fillStyle = "var(--ink-2, #6e6358)";
    ctx.font = "9px 'JetBrains Mono', monospace";
    ctx.textAlign = "center";
    ctx.fillText("Lag distance", w / 2, h - 6);
    ctx.save();
    ctx.translate(12, h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Moran's I", 0, 0);
    ctx.restore();

    // Zero reference at I=0 if in range
    if (allMinI < 0 && allMaxI > 0) {
      const zeroY = pad.top + plotH - ((0 - allMinI) / rangeI) * plotH;
      ctx.setLineDash([4, 3]);
      ctx.strokeStyle = "rgba(0,0,0,0.3)";
      ctx.beginPath();
      ctx.moveTo(pad.left, zeroY);
      ctx.lineTo(w - pad.right, zeroY);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Tick labels
    ctx.fillStyle = "var(--ink-2, #6e6358)";
    ctx.font = "8px 'JetBrains Mono', monospace";
    ctx.textAlign = "center";
    for (let i = 0; i <= 4; i++) {
      const lagVal = (allMaxLag * i) / 4;
      const x = pad.left + (plotW * i) / 4;
      ctx.fillText(lagVal.toFixed(0), x, h - pad.bottom + 12);
    }
    ctx.textAlign = "right";
    for (let i = 0; i <= 4; i++) {
      const iVal = allMinI + (rangeI * (4 - i)) / 4;
      const y = pad.top + (plotH * i) / 4;
      ctx.fillText(iVal.toFixed(2), pad.left - 4, y + 3);
      ctx.strokeStyle = "rgba(0,0,0,0.04)";
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(w - pad.right, y);
      ctx.stroke();
    }

    const top = Math.min(varNames.length, 6);
    for (let vi = 0; vi < top; vi++) {
      const name = varNames[vi];
      const cr = results[name].correlogram_results;
      const lags = cr.lag_distances;
      const morans = cr.morans_i_values;
      const pvals = cr.p_values;

      const color = SPARC_RAMP_HEX[vi % SPARC_RAMP_HEX.length];
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (let i = 0; i < lags.length; i++) {
        const x = pad.left + (lags[i] / allMaxLag) * plotW;
        const y = pad.top + plotH - ((morans[i] - allMinI) / rangeI) * plotH;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      for (let i = 0; i < lags.length; i++) {
        if (pvals[i] < 0.05) {
          const x = pad.left + (lags[i] / allMaxLag) * plotW;
          const y = pad.top + plotH - ((morans[i] - allMinI) / rangeI) * plotH;
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(x, y, 3, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }
  }, [data, varNames.length]);

  if (!present) {
    return (
      <Panel title="Correlogram" subtitle="stage 0 · spatial autocorrelation">
        <PanelEmpty
          reason="Awaiting stage 0"
          hint="Run the Correlogram stage to characterise spatial autocorrelation in the input variables."
          action={navigate && <Btn small onClick={() => navigate("Run")}>Go to Run page →</Btn>}
        />
      </Panel>
    );
  }

  if (!varNames.length) {
    return (
      <Panel title="Correlogram" subtitle="stage 0 · spatial autocorrelation">
        <PanelEmpty reason="Loading…" />
      </Panel>
    );
  }

  return (
    <Panel
      title="Correlogram"
      subtitle={`Moran's I · ${varNames.length} variable${varNames.length === 1 ? "" : "s"}`}
    >
      <canvas ref={canvasRef} style={{ width: "100%", height: 280, display: "block" }} />
      <div className="mono" style={{ fontSize: 10, color: "var(--ink-2)", marginTop: 6 }}>
        Filled markers = significant at p &lt; 0.05.
      </div>
    </Panel>
  );
}
