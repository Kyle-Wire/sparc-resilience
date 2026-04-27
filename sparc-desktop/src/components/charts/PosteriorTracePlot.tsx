/**
 * PosteriorTracePlot — small-multiples MCMC trace plot.
 *
 * Each parameter draws as a thin polyline of draw_index → value. Up to 4
 * chains are colour-coded. Pure SVG.
 */
import { useMemo } from "react";

export interface TraceDraw {
  param: string;
  chain: number;
  draw: number;
  value: number;
}

interface PosteriorTracePlotProps {
  draws: TraceDraw[];
  width?: number;
  height?: number;
  /** Maximum parameters to render (others get a "+N more" footer). */
  maxParams?: number;
}

const CHAIN_COLORS = ["#7a3a3a", "#3a5e7a", "#5e7a3a", "#7a5e3a"];

export default function PosteriorTracePlot({
  draws,
  width = 560,
  height = 320,
  maxParams = 6,
}: PosteriorTracePlotProps) {
  const grouped = useMemo(() => {
    const byParam = new Map<string, Map<number, TraceDraw[]>>();
    for (const d of draws) {
      if (!isFinite(d.value)) continue;
      let chains = byParam.get(d.param);
      if (!chains) { chains = new Map(); byParam.set(d.param, chains); }
      let arr = chains.get(d.chain);
      if (!arr) { arr = []; chains.set(d.chain, arr); }
      arr.push(d);
    }
    for (const chains of byParam.values()) {
      for (const arr of chains.values()) arr.sort((a, b) => a.draw - b.draw);
    }
    return byParam;
  }, [draws]);

  const params = Array.from(grouped.keys()).slice(0, maxParams);
  const extra = grouped.size - params.length;

  if (!params.length) {
    return <div style={{ padding: 20, fontSize: 12, color: "#888" }}>No posterior draws.</div>;
  }

  const cols = params.length <= 2 ? params.length : params.length <= 4 ? 2 : 3;
  const rows = Math.ceil(params.length / cols);
  const padX = 8, padY = 24;
  const cellW = (width - padX * (cols + 1)) / cols;
  const cellH = (height - padY * (rows + 1) - (extra > 0 ? 16 : 0)) / rows;

  return (
    <svg width={width} height={height} role="img" aria-label="Posterior trace plot">
      {params.map((param, i) => {
        const r = Math.floor(i / cols), c = i % cols;
        const x = padX + c * (cellW + padX);
        const y = padY + r * (cellH + padY);
        const chains = grouped.get(param)!;
        const all = Array.from(chains.values()).flat();
        const lo = Math.min(...all.map((d) => d.value));
        const hi = Math.max(...all.map((d) => d.value));
        const drawLo = Math.min(...all.map((d) => d.draw));
        const drawHi = Math.max(...all.map((d) => d.draw));
        const xRange = drawHi - drawLo || 1;
        const yRange = hi - lo || 1;
        const toX = (v: number) => x + ((v - drawLo) / xRange) * cellW;
        const toY = (v: number) => y + cellH - ((v - lo) / yRange) * cellH;
        return (
          <g key={param}>
            <text x={x} y={y - 6} fontSize={11} fontWeight={600} fill="#333">{param}</text>
            <rect x={x} y={y} width={cellW} height={cellH} fill="none" stroke="#ddd" />
            {Array.from(chains.entries()).map(([chain, arr]) => (
              <polyline
                key={chain}
                points={arr.map((d) => `${toX(d.draw)},${toY(d.value)}`).join(" ")}
                fill="none"
                stroke={CHAIN_COLORS[chain % CHAIN_COLORS.length]}
                strokeWidth={0.7}
                strokeOpacity={0.85}
              />
            ))}
            <text x={x} y={y + cellH + 10} fontSize={9} fill="#888">{lo.toFixed(2)}</text>
            <text x={x + cellW} y={y + cellH + 10} fontSize={9} fill="#888" textAnchor="end">{hi.toFixed(2)}</text>
          </g>
        );
      })}
      {extra > 0 && (
        <text x={padX} y={height - 4} fontSize={10} fill="#888">+{extra} more parameters not shown</text>
      )}
    </svg>
  );
}
