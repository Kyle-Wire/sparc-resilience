/**
 * SpatialHeatmap — Canvas-rendered heatmap with SPARC 9-color ramp.
 * Displays real spatial data from the API, falling back to gaussian noise.
 * Scenario-reactive: shifts the field based on selected scenario.
 * Matches the SPARC Labs prototype design.
 */
import { useRef, useEffect, useState, useCallback } from "react";
import { getSpatialCvPredictions } from "@/lib/api";
import type { GeoJsonData } from "@/lib/types";
import { SPARC_HEX } from "./RampLegend";

const CELL = 6;

/** Scenario shift values matching the prototype */
const SCENARIO_SHIFTS: Record<string, number> = {
  baseline: 0,
  "canopy+10": -0.18,
  "impervious-20": -0.25,
  "albedo+0.1": -0.12,
};

/** Interpolate between two hex colors */
function lerpColor(a: string, b: string, t: number): string {
  const ah = parseInt(a.slice(1), 16);
  const bh = parseInt(b.slice(1), 16);
  const ar = (ah >> 16) & 0xff, ag = (ah >> 8) & 0xff, ab = ah & 0xff;
  const br = (bh >> 16) & 0xff, bg = (bh >> 8) & 0xff, bb = bh & 0xff;
  const r = Math.round(ar + (br - ar) * t);
  const g = Math.round(ag + (bg - ag) * t);
  const b2 = Math.round(ab + (bb - ab) * t);
  return `rgb(${r},${g},${b2})`;
}

/** Map a 0–1 value to the SPARC color ramp */
function rampColor(t: number): string {
  const clamped = Math.max(0, Math.min(1, t));
  const idx = clamped * (SPARC_HEX.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.min(lo + 1, SPARC_HEX.length - 1);
  return lerpColor(SPARC_HEX[lo], SPARC_HEX[hi], idx - lo);
}

/** Generate gaussian hotspot/coolspot noise field */
function generateField(cols: number, rows: number, shift: number): Float32Array {
  const field = new Float32Array(cols * rows);

  // Hotspots
  const spots = [
    { x: 0.25, y: 0.3, sigma: 0.15, amp: 0.8 },
    { x: 0.6, y: 0.2, sigma: 0.12, amp: 0.9 },
    { x: 0.8, y: 0.6, sigma: 0.18, amp: 0.7 },
    { x: 0.4, y: 0.7, sigma: 0.14, amp: 0.85 },
    { x: 0.15, y: 0.8, sigma: 0.1, amp: 0.75 },
    // Coolspots
    { x: 0.5, y: 0.5, sigma: 0.2, amp: -0.4 },
    { x: 0.9, y: 0.1, sigma: 0.15, amp: -0.3 },
  ];

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const nx = c / cols;
      const ny = r / rows;
      let v = 0.5 + shift;
      for (const s of spots) {
        const dx = nx - s.x;
        const dy = ny - s.y;
        v += s.amp * Math.exp(-(dx * dx + dy * dy) / (2 * s.sigma * s.sigma));
      }
      // Add noise
      v += (Math.sin(c * 3.7 + r * 2.1) * 0.05) + (Math.cos(c * 1.3 + r * 4.7) * 0.03);
      field[r * cols + c] = v;
    }
  }

  // Normalize to 0–1
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < field.length; i++) {
    if (field[i] < min) min = field[i];
    if (field[i] > max) max = field[i];
  }
  const range = max - min || 1;
  for (let i = 0; i < field.length; i++) {
    field[i] = (field[i] - min) / range;
  }

  return field;
}

interface SpatialHeatmapProps {
  scenario?: string;
  height?: number;
}

export default function SpatialHeatmap({ scenario = "baseline", height = 320 }: SpatialHeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [geojson, setGeojson] = useState<GeoJsonData | null>(null);

  useEffect(() => {
    getSpatialCvPredictions().then(setGeojson).catch(() => null);
  }, []);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const dpr = Math.min(window.devicePixelRatio, 2);
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    const ctx = canvas.getContext("2d")!;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    const shift = SCENARIO_SHIFTS[scenario] ?? 0;

    if (geojson && geojson.features.length > 0) {
      // Render real spatial data
      const features = geojson.features;
      const vals = features.map((f) => Number(f.properties.predicted ?? f.properties.value ?? 0));
      let min = Infinity, max = -Infinity;
      for (const v of vals) {
        if (v < min) min = v;
        if (v > max) max = v;
      }
      const range = max - min || 1;

      // Get coordinate bounds
      const coords = features.map((f) => {
        const c = f.geometry.coordinates as number[];
        return { x: c[0], y: c[1] };
      });
      const xMin = Math.min(...coords.map((c) => c.x));
      const xMax = Math.max(...coords.map((c) => c.x));
      const yMin = Math.min(...coords.map((c) => c.y));
      const yMax = Math.max(...coords.map((c) => c.y));
      const xRange = xMax - xMin || 1;
      const yRange = yMax - yMin || 1;

      features.forEach((f, i) => {
        const c = f.geometry.coordinates as number[];
        const px = ((c[0] - xMin) / xRange) * (w - CELL) ;
        const py = ((yMax - c[1]) / yRange) * (h - CELL);
        const t = (vals[i] - min + shift * range) / range;
        ctx.globalAlpha = 0.82;
        ctx.fillStyle = rampColor(t);
        ctx.fillRect(px, py, CELL, CELL);
      });
      ctx.globalAlpha = 1;
    } else {
      // Gaussian noise field fallback
      const cols = Math.floor(w / CELL);
      const rows = Math.floor(h / CELL);
      const field = generateField(cols, rows, shift);

      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          ctx.globalAlpha = 0.82;
          ctx.fillStyle = rampColor(field[r * cols + c]);
          ctx.fillRect(c * CELL, r * CELL, CELL, CELL);
        }
      }
      ctx.globalAlpha = 1;
    }

    // Street grid overlay
    ctx.strokeStyle = "rgba(0,0,0,0.08)";
    ctx.lineWidth = 1;
    const vLines = 14, hLines = 10;
    for (let i = 1; i < vLines; i++) {
      const x = (i / vLines) * w;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
    for (let i = 1; i < hLines; i++) {
      const y = (i / hLines) * h;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    // Dashed border
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = "rgba(0,0,0,0.45)";
    ctx.lineWidth = 1;
    ctx.strokeRect(0.5, 0.5, w - 1, h - 1);
    ctx.setLineDash([]);
  }, [geojson, scenario]);

  useEffect(() => {
    draw();
  }, [draw]);

  // Redraw on resize
  useEffect(() => {
    const obs = new ResizeObserver(() => draw());
    if (canvasRef.current) obs.observe(canvasRef.current);
    return () => obs.disconnect();
  }, [draw]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width: "100%", height, display: "block", borderRadius: 4 }}
    />
  );
}
