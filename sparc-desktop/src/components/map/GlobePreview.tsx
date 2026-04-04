import { useRef, useEffect, useCallback } from "react";

interface GlobePreviewProps {
  /** Center longitude of the bbox highlight */
  longitude?: number;
  /** Center latitude of the bbox highlight */
  latitude?: number;
  /** Size in pixels */
  size?: number;
}

const TAU = Math.PI * 2;
const LAND_COLOR = "#d4d4d4";
const OCEAN_COLOR = "#f5f5f5";
const GRID_COLOR = "rgba(0,0,0,0.06)";
const HIGHLIGHT_COLOR = "rgba(96, 36, 104, 0.6)";
const HIGHLIGHT_STROKE = "#602468";

// Very simplified continental outlines — just enough to recognize Earth
// Each array is [lng, lat] pairs forming a closed polygon outline
// We only draw circles and a simple highlight here for performance

export default function GlobePreview({ longitude = 0, latitude = 0, size = 200 }: GlobePreviewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rotRef = useRef(longitude);
  const frameRef = useRef<number>(0);
  const hoverRef = useRef(false);

  const project = useCallback(
    (lng: number, lat: number, rot: number, r: number, cx: number, cy: number): [number, number, boolean] => {
      const lam = ((lng - rot) * Math.PI) / 180;
      const phi = (lat * Math.PI) / 180;
      const x = cx + r * Math.cos(phi) * Math.sin(lam);
      const y = cy - r * Math.sin(phi);
      const visible = Math.cos(phi) * Math.cos(lam) > 0;
      return [x, y, visible];
    },
    []
  );

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, w: number, h: number, rot: number) => {
      const cx = w / 2;
      const cy = h / 2;
      const r = Math.min(cx, cy) - 8;

      ctx.clearRect(0, 0, w, h);

      // Ocean
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, TAU);
      ctx.fillStyle = OCEAN_COLOR;
      ctx.fill();
      ctx.strokeStyle = LAND_COLOR;
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Grid lines (meridians and parallels)
      ctx.strokeStyle = GRID_COLOR;
      ctx.lineWidth = 0.5;
      for (let lng = -180; lng <= 180; lng += 30) {
        ctx.beginPath();
        for (let lat = -90; lat <= 90; lat += 3) {
          const [x, y, vis] = project(lng, lat, rot, r, cx, cy);
          if (vis) {
            if (lat === -90 || !project(lng, lat - 3, rot, r, cx, cy)[2]) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          }
        }
        ctx.stroke();
      }
      for (let lat = -60; lat <= 60; lat += 30) {
        ctx.beginPath();
        for (let lng = -180; lng <= 180; lng += 3) {
          const [x, y, vis] = project(lng, lat, rot, r, cx, cy);
          if (vis) {
            if (lng === -180 || !project(lng - 3, lat, rot, r, cx, cy)[2]) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          }
        }
        ctx.stroke();
      }

      // Highlight area (project bbox as a dot/circle at lat/lon)
      if (longitude !== undefined && latitude !== undefined) {
        const [hx, hy, hv] = project(longitude, latitude, rot, r, cx, cy);
        if (hv) {
          ctx.beginPath();
          ctx.arc(hx, hy, 8, 0, TAU);
          ctx.fillStyle = HIGHLIGHT_COLOR;
          ctx.fill();
          ctx.strokeStyle = HIGHLIGHT_STROKE;
          ctx.lineWidth = 2;
          ctx.stroke();

          // Pulsing ring
          ctx.beginPath();
          ctx.arc(hx, hy, 14, 0, TAU);
          ctx.strokeStyle = "rgba(96, 36, 104, 0.25)";
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }
    },
    [longitude, latitude, project]
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const animate = () => {
      if (!hoverRef.current) {
        rotRef.current += 0.15;
      }
      draw(ctx, canvas.width, canvas.height, rotRef.current);
      frameRef.current = requestAnimationFrame(animate);
    };

    frameRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(frameRef.current);
  }, [draw]);

  return (
    <canvas
      ref={canvasRef}
      width={size}
      height={size}
      className="cursor-pointer"
      onMouseEnter={() => (hoverRef.current = true)}
      onMouseLeave={() => (hoverRef.current = false)}
      title="Globe preview — hover to pause rotation"
    />
  );
}
