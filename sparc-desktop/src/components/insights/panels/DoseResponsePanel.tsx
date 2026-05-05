/**
 * DoseResponsePanel — marginal causal effect dY/dT vs treatment level
 * for the active CATE/treatment variable.
 *
 * Reads the linked-selection `variable`. Falls back to the first
 * available curve when nothing is selected.
 */
import { useEffect, useMemo, useState } from "react";
import { Panel, PanelEmpty, Pill } from "@/components/ui/DesignSystem";
import { useManifest } from "@/hooks/useManifest";
import { useLinkedSelection } from "@/hooks/useLinkedSelection";
import { getDoseResponseCurves } from "@/lib/api";
import { useCanvas } from "./_useCanvas";
import type { DoseResponseData } from "@/lib/types";

export default function DoseResponsePanel() {
  const manifest = useManifest();
  const linked = useLinkedSelection();
  const present = !!manifest.lookup("3", "dose_response_curves");
  const [data, setData] = useState<DoseResponseData | null>(null);

  useEffect(() => {
    if (!present) {
      setData(null);
      return;
    }
    getDoseResponseCurves()
      .then((d) => setData(d as DoseResponseData))
      .catch(() => setData(null));
  }, [present, manifest.lastUpdated]);

  const treatments = useMemo(() => (data ? Object.keys(data) : []), [data]);
  const activeVar = linked.selection.variable;
  const treatmentName = useMemo(() => {
    if (!treatments.length) return null;
    return treatments.find((t) => t === activeVar) ?? treatments[0];
  }, [treatments, activeVar]);
  const curve = treatmentName && data ? data[treatmentName] : null;

  const canvasRef = useCanvas((ctx, w, h) => {
    if (!curve || !curve.dose_levels?.length || !curve.marginal_effects?.length) return;
    const xs = curve.dose_levels;
    const ys = curve.marginal_effects;
    const pad = { top: 16, right: 14, bottom: 30, left: 44 };
    const xMin = Math.min(...xs),
      xMax = Math.max(...xs);
    const yMin = Math.min(...ys, 0),
      yMax = Math.max(...ys, 0);
    const xRange = xMax - xMin || 1;
    const yRange = yMax - yMin || 1;
    const toX = (v: number) => pad.left + ((v - xMin) / xRange) * (w - pad.left - pad.right);
    const toY = (v: number) => h - pad.bottom - ((v - yMin) / yRange) * (h - pad.top - pad.bottom);

    // Grid
    ctx.strokeStyle = "rgba(0,0,0,0.06)";
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (i / 4) * (h - pad.top - pad.bottom);
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(w - pad.right, y);
      ctx.stroke();
    }
    if (yMin < 0 && yMax > 0) {
      ctx.strokeStyle = "rgba(0,0,0,0.25)";
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.moveTo(pad.left, toY(0));
      ctx.lineTo(w - pad.right, toY(0));
      ctx.stroke();
    }

    // Curve
    ctx.strokeStyle = "var(--purple, #6b1aa1)";
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    xs.forEach((x: number, i: number) => {
      if (i === 0) ctx.moveTo(toX(x), toY(ys[i]));
      else ctx.lineTo(toX(x), toY(ys[i]));
    });
    ctx.stroke();
    ctx.fillStyle = "var(--purple, #6b1aa1)";
    xs.forEach((x: number, i: number) => {
      ctx.beginPath();
      ctx.arc(toX(x), toY(ys[i]), 2.5, 0, Math.PI * 2);
      ctx.fill();
    });

    // Axes labels
    ctx.fillStyle = "var(--ink-2, #6e6358)";
    ctx.font = "9px 'JetBrains Mono', monospace";
    ctx.textAlign = "center";
    ctx.fillText(treatmentName!.replace(/_/g, " "), (pad.left + w - pad.right) / 2, h - 8);
    ctx.save();
    ctx.translate(10, (pad.top + h - pad.bottom) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("dY/dT", 0, 0);
    ctx.restore();
    ctx.textAlign = "right";
    for (let i = 0; i <= 4; i++) {
      const v = yMin + (i / 4) * yRange;
      const y = h - pad.bottom - (i / 4) * (h - pad.top - pad.bottom);
      ctx.fillText(v.toFixed(2), pad.left - 4, y + 3);
    }
    ctx.textAlign = "center";
    [0, 0.5, 1].forEach((t) => {
      const v = xMin + t * xRange;
      ctx.fillText(v.toFixed(2), pad.left + t * (w - pad.left - pad.right), h - pad.bottom + 12);
    });

    if (curve.is_nonlinear) {
      ctx.fillStyle = "var(--crimson, #b91c1c)";
      ctx.textAlign = "right";
      ctx.fillText("nonlinear", w - pad.right - 2, pad.top + 8);
    }
  }, [curve, treatmentName]);

  if (!present) {
    return (
      <Panel title="Dose-response" subtitle="stage 3 · causal">
        <PanelEmpty
          reason="No dose-response curves"
          hint="Run the Causal stage with treatment variables configured."
        />
      </Panel>
    );
  }

  if (!treatments.length) {
    return (
      <Panel title="Dose-response" subtitle="stage 3 · causal">
        <PanelEmpty reason="Loading…" />
      </Panel>
    );
  }

  return (
    <Panel
      title="Dose-response"
      subtitle={treatmentName ? `treatment · ${treatmentName}` : "stage 3"}
      actions={
        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          {activeVar && treatmentName === activeVar && <Pill color="var(--purple)">linked</Pill>}
          {treatments.length > 1 && (
            <select
              value={treatmentName ?? ""}
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
              {treatments.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
          )}
        </div>
      }
    >
      <canvas ref={canvasRef} style={{ width: "100%", height: 260, display: "block" }} />
    </Panel>
  );
}
