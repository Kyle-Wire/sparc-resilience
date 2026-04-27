/**
 * MapLegend — gradient-and-tick legend for SpatialMap and friends.
 *
 * Reusable so the map, the comparison view, and any other deck.gl/maplibre
 * surface can share a single legend implementation. Supports linear and
 * quantile (5-class) modes; both get tick labels along the gradient.
 */
import type { CSSProperties } from "react";

export type LegendMode = "linear" | "quantile";

interface MapLegendProps {
  label: string;
  domain: [number, number];
  rampCss: string;
  mode?: LegendMode;
  /** Optional pre-computed quantile breakpoints (5 values when mode=quantile). */
  breaks?: number[];
  /** Number of ticks for linear mode (default 5). */
  ticks?: number;
  /** Override container style (e.g. position absolute relocates the chip). */
  style?: CSSProperties;
  /** Hide the label row (compact mode). */
  compact?: boolean;
}

function fmt(v: number): string {
  const abs = Math.abs(v);
  if (abs >= 1000 || (abs > 0 && abs < 0.01)) return v.toExponential(1);
  return v.toFixed(2);
}

export default function MapLegend({
  label,
  domain,
  rampCss,
  mode = "linear",
  breaks,
  ticks = 5,
  style,
  compact,
}: MapLegendProps) {
  const [lo, hi] = domain;
  const tickValues: number[] = (() => {
    if (mode === "quantile" && breaks && breaks.length > 0) return breaks;
    if (ticks <= 1) return [lo, hi];
    return Array.from({ length: ticks }, (_, i) => lo + ((hi - lo) * i) / (ticks - 1));
  })();

  const baseStyle: CSSProperties = {
    background: "rgba(255,255,255,0.95)",
    border: "1px solid rgba(0,0,0,0.06)",
    borderRadius: 6,
    padding: "6px 8px",
    boxShadow: "0 1px 3px rgba(0,0,0,0.08)",
    backdropFilter: "blur(4px)",
    fontFamily: "system-ui, sans-serif",
    color: "#444",
    minWidth: 160,
    ...style,
  };

  return (
    <div style={baseStyle}>
      {!compact && (
        <div
          style={{
            fontSize: 10,
            fontWeight: 600,
            textTransform: "uppercase",
            letterSpacing: "0.06em",
            color: "#666",
            marginBottom: 4,
          }}
        >
          {label}
        </div>
      )}
      <div
        style={{
          height: 10,
          width: "100%",
          borderRadius: 2,
          background: rampCss,
        }}
      />
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          marginTop: 2,
          fontSize: 9,
          color: "#666",
          fontFamily: "ui-monospace, monospace",
        }}
      >
        {tickValues.map((v, i) => (
          <span key={i}>{fmt(v)}</span>
        ))}
      </div>
    </div>
  );
}
