import { useEffect, useRef } from "react";
import { SPARC_HEX } from "./RampLegend";

interface SpatialMapProps {
  scenario: string;
}

export default function SpatialMap({ scenario }: SpatialMapProps) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = w * DPR;
    canvas.height = h * DPR;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(DPR, DPR);

    const hotspots = [
      { x: 0.35, y: 0.45, s: 0.18, a: 1.0 },
      { x: 0.62, y: 0.38, s: 0.14, a: 0.85 },
      { x: 0.78, y: 0.64, s: 0.16, a: 0.7 },
      { x: 0.22, y: 0.72, s: 0.20, a: 0.6 },
      { x: 0.55, y: 0.78, s: 0.12, a: 0.9 },
    ];
    const coolspots = [
      { x: 0.15, y: 0.30, s: 0.18, a: 0.8 },
      { x: 0.85, y: 0.20, s: 0.12, a: 0.7 },
    ];
    const cell = 6;
    const shift =
      scenario === "canopy+10"
        ? -0.18
        : scenario === "impervious-20"
        ? -0.25
        : scenario === "albedo+0.1"
        ? -0.12
        : 0;

    for (let y = 0; y < h; y += cell) {
      for (let x = 0; x < w; x += cell) {
        const nx = x / w,
          ny = y / h;
        let v = 0;
        for (const hs of hotspots) {
          const dx = nx - hs.x,
            dy = ny - hs.y;
          v += hs.a * Math.exp(-(dx * dx + dy * dy) / (2 * hs.s * hs.s));
        }
        for (const cs of coolspots) {
          const dx = nx - cs.x,
            dy = ny - cs.y;
          v -= cs.a * Math.exp(-(dx * dx + dy * dy) / (2 * cs.s * cs.s));
        }
        v += shift;
        const t = Math.max(0, Math.min(1, (v + 0.5) / 1.6));
        const idx = t * (SPARC_HEX.length - 1);
        const i0 = Math.floor(idx);
        const i1 = Math.min(SPARC_HEX.length - 1, i0 + 1);
        const f = idx - i0;
        const c0 = SPARC_HEX[i0];
        const c1 = SPARC_HEX[i1];
        const r = Math.round(parseInt(c0.slice(1, 3), 16) * (1 - f) + parseInt(c1.slice(1, 3), 16) * f);
        const g = Math.round(parseInt(c0.slice(3, 5), 16) * (1 - f) + parseInt(c1.slice(3, 5), 16) * f);
        const b = Math.round(parseInt(c0.slice(5, 7), 16) * (1 - f) + parseInt(c1.slice(5, 7), 16) * f);
        ctx.fillStyle = `rgba(${r},${g},${b},0.82)`;
        ctx.fillRect(x, y, cell, cell);
      }
    }

    ctx.strokeStyle = "rgba(0,0,0,0.08)";
    ctx.lineWidth = 1;
    for (let i = 0; i < 14; i++) {
      ctx.beginPath();
      ctx.moveTo((i / 14) * w, 0);
      ctx.lineTo((i / 14) * w + 20, h);
      ctx.stroke();
    }
    for (let i = 0; i < 10; i++) {
      ctx.beginPath();
      ctx.moveTo(0, (i / 10) * h);
      ctx.lineTo(w, (i / 10) * h - 12);
      ctx.stroke();
    }

    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = "rgba(0,0,0,0.45)";
    ctx.lineWidth = 1;
    ctx.strokeRect(0.5, 0.5, w - 1, h - 1);
  }, [scenario]);

  return <canvas ref={ref} style={{ width: "100%", height: "100%", display: "block" }} />;
}
