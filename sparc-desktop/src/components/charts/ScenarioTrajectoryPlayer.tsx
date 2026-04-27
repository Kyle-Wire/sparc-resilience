/**
 * ScenarioTrajectoryPlayer — animated time-series of scenario predictions.
 *
 * Combines a polyline plot of value(t) per geometry with a play/pause time
 * slider (`MapTimeSlider`). Each geometry is a thin polyline; the active
 * timestep is highlighted with a vertical scrubber line.
 */
import { useMemo, useState } from "react";
import type { TrajectoryRecord } from "@/lib/api";
import MapTimeSlider from "@/components/map/MapTimeSlider";

interface ScenarioTrajectoryPlayerProps {
  records: TrajectoryRecord[];
  scenarioId?: number | string;
  width?: number;
  height?: number;
  /** Maximum number of geometries to render polylines for. */
  maxLines?: number;
}

export default function ScenarioTrajectoryPlayer({
  records,
  scenarioId,
  width = 560,
  height = 280,
  maxLines = 30,
}: ScenarioTrajectoryPlayerProps) {
  const filtered = useMemo(() => {
    return scenarioId !== undefined
      ? records.filter((r) => String(r.scenario_id) === String(scenarioId))
      : records;
  }, [records, scenarioId]);

  const { steps, byGeom, vMin, vMax } = useMemo(() => {
    const stepSet = new Set<string | number>();
    const groups = new Map<string | number, TrajectoryRecord[]>();
    let lo = Infinity, hi = -Infinity;
    for (const r of filtered) {
      stepSet.add(r.t);
      const gid = r.geometry_id ?? "_all";
      let arr = groups.get(gid);
      if (!arr) { arr = []; groups.set(gid, arr); }
      arr.push(r);
      if (typeof r.value === "number" && isFinite(r.value)) {
        if (r.value < lo) lo = r.value;
        if (r.value > hi) hi = r.value;
      }
    }
    const steps = Array.from(stepSet).sort((a, b) => Number(a) - Number(b));
    for (const arr of groups.values()) {
      arr.sort((a, b) => Number(a.t) - Number(b.t));
    }
    return { steps, byGeom: groups, vMin: lo, vMax: hi };
  }, [filtered]);

  const [tIdx, setTIdx] = useState(0);

  if (!steps.length) {
    return <div style={{ padding: 20, fontSize: 12, color: "#888" }}>No trajectory data.</div>;
  }

  const padL = 40, padR = 12, padT = 14, padB = 28;
  const w = width - padL - padR;
  const h = height - padT - padB;
  const tToX = (t: string | number) => padL + (steps.indexOf(t) / Math.max(1, steps.length - 1)) * w;
  const yRange = vMax - vMin || 1;
  const vToY = (v: number) => padT + h - ((v - vMin) / yRange) * h;

  const lines = Array.from(byGeom.entries()).slice(0, maxLines);
  const extra = byGeom.size - lines.length;
  const scrubX = tToX(steps[tIdx] ?? steps[0]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <svg width={width} height={height} role="img" aria-label="Scenario trajectory">
        <line x1={padL} x2={padL} y1={padT} y2={padT + h} stroke="#bbb" />
        <line x1={padL} x2={padL + w} y1={padT + h} y2={padT + h} stroke="#bbb" />
        <text x={4} y={padT + 4} fontSize={9} fill="#666">{vMax.toFixed(2)}</text>
        <text x={4} y={padT + h} fontSize={9} fill="#666">{vMin.toFixed(2)}</text>
        {lines.map(([gid, arr], i) => (
          <polyline
            key={String(gid)}
            points={arr.map((r) => `${tToX(r.t)},${vToY(r.value)}`).join(" ")}
            fill="none"
            stroke="#7a3a3a"
            strokeOpacity={0.25 + 0.5 / Math.max(1, lines.length - i)}
            strokeWidth={0.8}
          />
        ))}
        <line x1={scrubX} x2={scrubX} y1={padT} y2={padT + h} stroke="#7a3a3a" strokeWidth={1.5} strokeDasharray="3,2" />
        <text x={scrubX} y={padT + h + 14} fontSize={10} fill="#7a3a3a" textAnchor="middle">
          t = {String(steps[tIdx])}
        </text>
        {extra > 0 && (
          <text x={width - padR} y={height - 4} fontSize={9} fill="#888" textAnchor="end">
            +{extra} more series hidden
          </text>
        )}
      </svg>
      <MapTimeSlider
        steps={steps}
        index={tIdx}
        onIndexChange={setTIdx}
        speedMs={500}
        label="Trajectory step"
      />
    </div>
  );
}
