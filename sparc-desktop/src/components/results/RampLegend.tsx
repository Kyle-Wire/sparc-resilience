const SPARC_HEX = [
  "#602468", "#9e337d", "#e94d9b", "#e94461", "#e73c25",
  "#e76c25", "#e79024", "#f0b632", "#fbdd46",
];

interface RampLegendProps {
  min?: string;
  max?: string;
  label?: string;
}

export default function RampLegend({
  label = "ΔTemperature (z-score)",
  min = "−1.2",
  max = "+1.2",
}: RampLegendProps) {
  return (
    <div>
      <div
        className="mono"
        style={{
          fontSize: 9.5,
          color: "var(--muted)",
          letterSpacing: "0.08em",
          textTransform: "uppercase",
          marginBottom: 4,
        }}
      >
        {label}
      </div>
      <div style={{ height: 10, borderRadius: 2, background: `linear-gradient(90deg, ${SPARC_HEX.join(", ")})` }} />
      <div
        className="mono"
        style={{
          display: "flex",
          justifyContent: "space-between",
          fontSize: 10,
          color: "var(--muted)",
          marginTop: 2,
        }}
      >
        <span>{min}</span>
        <span>0</span>
        <span>{max}</span>
      </div>
    </div>
  );
}

export { SPARC_HEX };
