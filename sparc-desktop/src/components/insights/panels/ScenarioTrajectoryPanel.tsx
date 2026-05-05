/**
 * ScenarioTrajectoryPanel — predicted-value trajectory over time `t`
 * for the active scenario, plotted as one line per geometry (sampled).
 */
import { useEffect, useMemo, useState } from "react";
import { Panel, PanelEmpty, Pill } from "@/components/ui/DesignSystem";
import { useManifest } from "@/hooks/useManifest";
import { useLinkedSelection } from "@/hooks/useLinkedSelection";
import { getScenarioTrajectory, type TrajectoryRecord } from "@/lib/api";
import { SPARC_RAMP_HEX } from "@/lib/design-tokens";
import { useCanvas } from "./_useCanvas";

export default function ScenarioTrajectoryPanel() {
  const manifest = useManifest();
  const linked = useLinkedSelection();
  const present = !!manifest.stage("4");
  const [records, setRecords] = useState<TrajectoryRecord[]>([]);
  const [error, setError] = useState<string | null>(null);

  const scenarioId = linked.selection.scenario;

  useEffect(() => {
    if (!present) {
      setRecords([]);
      return;
    }
    setError(null);
    getScenarioTrajectory(scenarioId ?? undefined)
      .then((r) => setRecords(r.records ?? []))
      .catch((e) => {
        setRecords([]);
        setError(e instanceof Error ? e.message : "no trajectory");
      });
  }, [present, scenarioId, manifest.lastUpdated]);

  // Group records by geometry_id, sort by t.
  const series = useMemo(() => {
    const by = new Map<string, { t: number; v: number }[]>();
    for (const r of records) {
      const key = String(r.geometry_id ?? "_all");
      const t = typeof r.t === "number" ? r.t : Number(r.t);
      if (!Number.isFinite(t) || !Number.isFinite(r.value)) continue;
      const arr = by.get(key) ?? [];
      arr.push({ t, v: r.value });
      by.set(key, arr);
    }
    const out = Array.from(by.entries()).map(([id, pts]) => ({
      id,
      pts: pts.sort((a, b) => a.t - b.t),
    }));
    return out;
  }, [records]);

  // Sample at most 40 series for legibility.
  const sampled = useMemo(() => {
    if (series.length <= 40) return series;
    const step = Math.ceil(series.length / 40);
    return series.filter((_, i) => i % step === 0);
  }, [series]);

  const canvasRef = useCanvas((ctx, w, h) => {
    if (!sampled.length) return;
    const pad = { top: 14, right: 12, bottom: 26, left: 44 };
    let tMin = Infinity,
      tMax = -Infinity,
      vMin = Infinity,
      vMax = -Infinity;
    for (const s of sampled) {
      for (const p of s.pts) {
        if (p.t < tMin) tMin = p.t;
        if (p.t > tMax) tMax = p.t;
        if (p.v < vMin) vMin = p.v;
        if (p.v > vMax) vMax = p.v;
      }
    }
    const tR = tMax - tMin || 1;
    const vR = vMax - vMin || 1;
    const toX = (t: number) => pad.left + ((t - tMin) / tR) * (w - pad.left - pad.right);
    const toY = (v: number) => h - pad.bottom - ((v - vMin) / vR) * (h - pad.top - pad.bottom);

    // Frame
    ctx.strokeStyle = "var(--line, #c9c2b3)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top);
    ctx.lineTo(pad.left, h - pad.bottom);
    ctx.lineTo(w - pad.right, h - pad.bottom);
    ctx.stroke();

    // Lines
    sampled.forEach((s, i) => {
      const color = SPARC_RAMP_HEX[i % SPARC_RAMP_HEX.length] + "aa";
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.1;
      ctx.beginPath();
      s.pts.forEach((p, j) => {
        const x = toX(p.t),
          y = toY(p.v);
        if (j === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    });

    // Tick labels
    ctx.fillStyle = "var(--ink-2, #6e6358)";
    ctx.font = "9px 'JetBrains Mono', monospace";
    ctx.textAlign = "center";
    ctx.fillText(tMin.toFixed(1), pad.left, h - 8);
    ctx.fillText(tMax.toFixed(1), w - pad.right, h - 8);
    ctx.textAlign = "right";
    ctx.fillText(vMax.toFixed(2), pad.left - 4, pad.top + 6);
    ctx.fillText(vMin.toFixed(2), pad.left - 4, h - pad.bottom);
  }, [sampled]);

  if (!present) {
    return (
      <Panel title="Scenario trajectory" subtitle="stage 4">
        <PanelEmpty
          reason="No trajectory artifact"
          hint="Run stage 4 with trajectory output enabled."
        />
      </Panel>
    );
  }
  if (!records.length) {
    return (
      <Panel title="Scenario trajectory" subtitle="stage 4">
        <PanelEmpty reason={error ?? "Loading…"} />
      </Panel>
    );
  }

  return (
    <Panel
      title="Scenario trajectory"
      subtitle={`${series.length} cells${sampled.length < series.length ? ` (sampled to ${sampled.length})` : ""}`}
      actions={scenarioId ? <Pill color="var(--purple)">scenario · {scenarioId}</Pill> : null}
    >
      <canvas ref={canvasRef} style={{ width: "100%", height: 240, display: "block" }} />
    </Panel>
  );
}
