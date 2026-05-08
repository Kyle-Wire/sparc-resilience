/**
 * EquityCostPanel — practitioner-facing summary of cost vs equity
 * across configured intervention candidates.
 *
 * No optimizer call here — just visualises the candidate table so the
 * user can see the cost / equity / mean-effect spread before drilling
 * into Decision Support.
 */
import { useMemo } from "react";
import { Panel, PanelEmpty } from "@/components/ui/DesignSystem";
import { useResult } from "@/hooks/useResult";
import { getDecisionCandidates, type InterventionCandidate } from "@/lib/api";
import { SPARC_RAMP_HEX } from "@/lib/design-tokens";
import { useCanvas } from "./_useCanvas";

export default function EquityCostPanel() {
  const { data: candidatesData, error } = useResult(
    "meta:decision_candidates",
    getDecisionCandidates,
  );
  const candidates: InterventionCandidate[] = candidatesData?.candidates ?? [];

  const stats = useMemo(() => {
    if (!candidates.length) return null;
    const xs = candidates.map((c) => c.cost);
    const ys = candidates.map((c) => c.equity_weight);
    return {
      xMin: Math.min(...xs),
      xMax: Math.max(...xs),
      yMin: Math.min(...ys),
      yMax: Math.max(...ys),
      effMin: Math.min(...candidates.map((c) => c.mean_effect)),
      effMax: Math.max(...candidates.map((c) => c.mean_effect)),
    };
  }, [candidates]);

  const canvasRef = useCanvas((ctx, w, h) => {
    if (!stats || !candidates.length) return;
    const pad = { top: 14, right: 14, bottom: 32, left: 50 };
    const xR = stats.xMax - stats.xMin || 1;
    const yR = stats.yMax - stats.yMin || 1;
    const eR = stats.effMax - stats.effMin || 1;
    const toX = (v: number) => pad.left + ((v - stats.xMin) / xR) * (w - pad.left - pad.right);
    const toY = (v: number) => h - pad.bottom - ((v - stats.yMin) / yR) * (h - pad.top - pad.bottom);

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
    ctx.fillText("cost", (pad.left + w - pad.right) / 2, h - 8);
    ctx.save();
    ctx.translate(12, (pad.top + h - pad.bottom) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("equity weight", 0, 0);
    ctx.restore();

    ctx.textAlign = "right";
    ctx.fillText(stats.yMax.toFixed(2), pad.left - 4, pad.top + 4);
    ctx.fillText(stats.yMin.toFixed(2), pad.left - 4, h - pad.bottom);
    ctx.textAlign = "center";
    ctx.fillText(stats.xMin.toFixed(1), pad.left, h - pad.bottom + 12);
    ctx.fillText(stats.xMax.toFixed(1), w - pad.right, h - pad.bottom + 12);

    // Bubbles — colour by mean_effect (cooler→warmer ramp)
    candidates.forEach((c) => {
      const x = toX(c.cost);
      const y = toY(c.equity_weight);
      const t = (c.mean_effect - stats.effMin) / eR;
      const ci = Math.min(SPARC_RAMP_HEX.length - 1, Math.max(0, Math.floor(t * (SPARC_RAMP_HEX.length - 1))));
      const r = 6 + Math.min(14, Math.abs(c.mean_effect / (eR || 1)) * 12);
      ctx.fillStyle = SPARC_RAMP_HEX[ci] + "cc";
      ctx.strokeStyle = SPARC_RAMP_HEX[ci];
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = "var(--ink, #1a1416)";
      ctx.font = "9px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.fillText(c.name, x + r + 2, y + 3);
    });
  }, [candidates, stats]);

  if (error || !candidates.length) {
    return (
      <Panel title="Equity & cost" subtitle="decision support">
        <PanelEmpty
          reason={error ?? "No intervention candidates"}
          hint="Configure interventions on the Decision Support page to populate this view."
        />
      </Panel>
    );
  }

  return (
    <Panel
      title="Equity & cost"
      subtitle={`${candidates.length} candidate${candidates.length === 1 ? "" : "s"}`}
    >
      <canvas ref={canvasRef} style={{ width: "100%", height: 280, display: "block" }} />
      <div className="mono" style={{ fontSize: 10, color: "var(--ink-2)", marginTop: 6 }}>
        Bubble area ∝ |mean effect|, hue = effect ramp.
      </div>
    </Panel>
  );
}
