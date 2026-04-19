interface ModelScore {
  name: string;
  r2: number;
  hi?: boolean;
}

const DEMO_DATA: ModelScore[] = [
  { name: "OLS", r2: 0.294 },
  { name: "GWR", r2: 0.828 },
  { name: "GWRF", r2: 0.898 },
  { name: "GGPGAM", r2: 0.839 },
  { name: "Meta (std)", r2: 0.902 },
  { name: "Meta (+L)", r2: 0.915, hi: true },
];

interface ModelBarChartProps {
  models?: ModelScore[];
}

export default function ModelBarChart({ models }: ModelBarChartProps) {
  const data = models && models.length > 0 ? models : DEMO_DATA;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6, padding: "4px 0" }}>
      {data.map((d) => (
        <div
          key={d.name}
          style={{ display: "grid", gridTemplateColumns: "92px 1fr 44px", alignItems: "center", gap: 10 }}
        >
          <span className="mono" style={{ fontSize: 10.5, color: "var(--ink-2)" }}>
            {d.name}
          </span>
          <div
            style={{
              height: 14,
              background: "rgba(0,0,0,0.05)",
              borderRadius: 2,
              position: "relative",
              overflow: "hidden",
            }}
          >
            <div
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                bottom: 0,
                width: `${d.r2 * 100}%`,
                background: d.hi ? "var(--crimson)" : "var(--ink-2)",
              }}
            />
            {d.hi && (
              <div
                style={{
                  position: "absolute",
                  inset: 0,
                  backgroundImage:
                    "repeating-linear-gradient(-45deg, transparent 0 3px, rgba(255,255,255,0.15) 3px 4px)",
                }}
              />
            )}
          </div>
          <span
            className="mono"
            style={{
              fontSize: 10.5,
              textAlign: "right",
              fontWeight: 600,
              color: d.hi ? "var(--crimson)" : "var(--ink-2)",
            }}
          >
            {d.r2.toFixed(3)}
          </span>
        </div>
      ))}
    </div>
  );
}
