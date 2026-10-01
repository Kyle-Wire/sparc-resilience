// Map overlays (SPEC §6.5): logger sites, before/after links, zone outlines, the hex grid, CV
// block lines, an influence circle and the selection outline, all SVG in raster units inside
// GridCanvas's view transform; plus the screen-space scale bar (1/2/5×10^k) and north arrow.
import { useMemo } from "react";
import { outlinePath } from "./colour";
import { fmtMeters, hexCenter, hexKey, niceLength, rasterToXY, rowCenter, xyToRaster, type GridData } from "./grid";

type Tone = "s1" | "s2" | "s3" | "ink";
type PointRef = { row?: number; x_m?: number; y_m?: number };

export type OverlayFeature =
  | { type: "points"; id: string; label: string; points: (PointRef & { label?: string; tone?: Tone })[] }
  | { type: "links"; id: string; label: string; links: { from: PointRef; to: PointRef }[]; tone?: Tone }
  | { type: "outline"; id: string; label: string; classOf: (row: number) => number | string | null; tone?: Tone }
  | { type: "gridlines"; id: string; label: string; spacing_m: number; origin_m?: [number, number] }
  | { type: "hexgrid"; id: string; label: string; size_m: number }
  | { type: "circle"; id: string; label: string; center: PointRef; radius_m: number; tone?: Tone };

const toneVar = (t?: Tone) => `var(--${t === "ink" || !t ? "ink" : t})`;

function at(g: GridData, p: PointRef): { px: number; py: number } | null {
  if (p.row !== undefined && p.row >= 0 && p.row < g.n) return rowCenter(g, p.row);
  if (p.x_m !== undefined && p.y_m !== undefined) return xyToRaster(g, p.x_m, p.y_m);
  return null;
}

function HexGrid({ g, size_m }: { g: GridData; size_m: number }) {
  const d = useMemo(() => {
    // Every hex touching the grid extent, drawn as pointy-top outlines.
    const [xa, ya] = rasterToXY(g, 0, g.ny);
    const [xb, yb] = rasterToXY(g, g.nx, 0);
    const R = size_m / Math.sqrt(3);
    const keys = new Set<number>();
    const step = size_m / 2;
    for (let x = xa - size_m; x <= xb + size_m; x += step) for (let y = ya - size_m; y <= yb + size_m; y += step) keys.add(hexKey(x, y, size_m));
    const parts: string[] = [];
    for (const k of keys) {
      const [cx, cy] = hexCenter(k, size_m);
      const pts = Array.from({ length: 6 }, (_, i) => {
        const a = ((60 * i - 30) * Math.PI) / 180;
        const p = xyToRaster(g, cx + R * Math.cos(a), cy + R * Math.sin(a));
        return `${p.px.toFixed(2)},${p.py.toFixed(2)}`;
      });
      parts.push(`M${pts.join("L")}Z`);
    }
    return parts.join("");
  }, [g, size_m]);
  return <path d={d} fill="none" style={{ stroke: "var(--ink-2)" }} strokeOpacity={0.45} strokeWidth={0.8} vectorEffect="non-scaling-stroke" />;
}

function GridLines({ g, spacing_m, origin_m }: { g: GridData; spacing_m: number; origin_m?: [number, number] }) {
  const d = useMemo(() => {
    const [ox, oy] = origin_m ?? [g.meta.x0_m - g.meta.dx_m / 2, g.meta.y0_m - g.meta.dx_m / 2];
    const [xa, ya] = rasterToXY(g, 0, g.ny);
    const [xb, yb] = rasterToXY(g, g.nx, 0);
    const parts: string[] = [];
    for (let x = ox + Math.ceil((xa - ox) / spacing_m) * spacing_m; x <= xb; x += spacing_m) {
      const p = xyToRaster(g, x, 0);
      parts.push(`M${p.px.toFixed(2)},0V${g.ny}`);
    }
    for (let y = oy + Math.ceil((ya - oy) / spacing_m) * spacing_m; y <= yb; y += spacing_m) {
      const p = xyToRaster(g, 0, y);
      parts.push(`M0,${p.py.toFixed(2)}H${g.nx}`);
    }
    return parts.join("");
  }, [g, spacing_m, origin_m]);
  return <path d={d} fill="none" style={{ stroke: "var(--ink)" }} strokeOpacity={0.35} strokeDasharray="4 3" strokeWidth={1} vectorEffect="non-scaling-stroke" />;
}

function Outline({ g, classOf, tone }: { g: GridData; classOf: (row: number) => number | string | null; tone?: Tone }) {
  const d = useMemo(() => outlinePath(g.cellToPt, g.nx, g.ny, classOf), [g, classOf]);
  return <path d={d} fill="none" style={{ stroke: toneVar(tone) }} strokeWidth={1.4} vectorEffect="non-scaling-stroke" />;
}

/** Overlay features in raster units; `scale` keeps markers a constant screen size. */
export function Overlay({ grid, features, scale }: { grid: GridData; features: OverlayFeature[]; scale: number }) {
  const r = 5 / Math.max(scale, 1e-6);
  return (
    <g>
      {features.map((f) => {
        switch (f.type) {
          case "points":
            return (
              <g key={f.id}>
                {f.points.map((p, i) => {
                  const q = at(grid, p);
                  return q ? <circle key={i} cx={q.px} cy={q.py} r={r} style={{ fill: toneVar(p.tone ?? "s2"), stroke: "var(--surface)" }} strokeWidth={1.5} vectorEffect="non-scaling-stroke" /> : null;
                })}
              </g>
            );
          case "links":
            return (
              <g key={f.id}>
                {f.links.map((l, i) => {
                  const a = at(grid, l.from);
                  const b = at(grid, l.to);
                  return a && b ? <line key={i} x1={a.px} y1={a.py} x2={b.px} y2={b.py} style={{ stroke: toneVar(f.tone ?? "ink") }} strokeWidth={1.2} vectorEffect="non-scaling-stroke" /> : null;
                })}
              </g>
            );
          case "outline":
            return <Outline key={f.id} g={grid} classOf={f.classOf} tone={f.tone} />;
          case "gridlines":
            return <GridLines key={f.id} g={grid} spacing_m={f.spacing_m} origin_m={f.origin_m} />;
          case "hexgrid":
            return <HexGrid key={f.id} g={grid} size_m={f.size_m} />;
          case "circle": {
            const c = at(grid, f.center);
            return c ? <circle key={f.id} cx={c.px} cy={c.py} r={f.radius_m / grid.meta.dx_m} fill="none" style={{ stroke: toneVar(f.tone ?? "ink") }} strokeWidth={1.5} strokeDasharray="5 4" vectorEffect="non-scaling-stroke" /> : null;
          }
        }
      })}
    </g>
  );
}

/** Scale bar (1/2/5×10^k m) and north arrow in screen space. */
export function MapChrome({ grid, scale, size }: { grid: GridData; scale: number; size: { w: number; h: number } }) {
  const pxPerM = scale / grid.meta.dx_m;
  const len_m = niceLength((size.w * 0.28) / pxPerM);
  const len = len_m * pxPerM;
  const x0 = 14;
  const y0 = size.h - 16;
  return (
    <g>
      <rect x={x0 - 6} y={y0 - 22} width={len + 58} height={30} rx={6} style={{ fill: "var(--surface)" }} fillOpacity={0.85} />
      <line x1={x0} x2={x0 + len} y1={y0} y2={y0} style={{ stroke: "var(--ink)" }} strokeWidth={2} />
      {[x0, x0 + len / 2, x0 + len].map((x) => (
        <line key={x} x1={x} x2={x} y1={y0 - 5} y2={y0} style={{ stroke: "var(--ink)" }} strokeWidth={1.5} />
      ))}
      <text x={x0} y={y0 - 9} fontSize={11} style={{ fill: "var(--ink)", fontFamily: "var(--font-mono)" }}>
        0
      </text>
      <text x={x0 + len + 4} y={y0 - 9} fontSize={11} style={{ fill: "var(--ink)", fontFamily: "var(--font-mono)" }}>
        {fmtMeters(len_m)}
      </text>
      <g transform={`translate(${size.w - 24},${40})`}>
        <path d="M0,-16l6,16l-6,-4l-6,4z" style={{ fill: "var(--ink)" }} />
        <text y={14} fontSize={11} textAnchor="middle" style={{ fill: "var(--ink)", fontFamily: "var(--font-mono)" }}>
          N
        </text>
      </g>
    </g>
  );
}
