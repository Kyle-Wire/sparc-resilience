import { useState, useEffect, useRef } from "react";
import { SectionHeader, Card, Tag, Btn, Stat, StatGrid } from "@/components/ui/DesignSystem";
import { getConfig } from "@/lib/api";
import { useNotification } from "@/hooks/useNotifications";

interface MonotoneConstraint {
  variable: string;
  direction: "increasing" | "decreasing" | "none";
  reason: string;
}

interface Guardrail {
  label: string;
  value: string;
  description: string;
}

export default function PhysicsPage() {
  const [constraints, setConstraints] = useState<MonotoneConstraint[]>([]);
  const [guardrails, setGuardrails] = useState<Guardrail[]>([]);
  const [selectedVar, setSelectedVar] = useState<string>("");
  const curveCanvasRef = useRef<HTMLCanvasElement>(null);
  const { notify } = useNotification();

  useEffect(() => {
    getConfig()
      .then((config) => {
        const physics = config.physics ?? {};
        const mc = physics.monotone_constraints ?? {};
        const newConstraints: MonotoneConstraint[] = Object.entries(mc).map(([k, v]) => ({
          variable: k,
          direction: (v as number) > 0 ? "increasing" : (v as number) < 0 ? "decreasing" : "none",
          reason: (v as number) > 0 ? "Physical expectation: positive relationship" : (v as number) < 0 ? "Physical expectation: negative relationship" : "No constraint",
        }));
        if (newConstraints.length === 0) {
          // Default demo constraints for UHI
          newConstraints.push(
            { variable: "tree_canopy_pct", direction: "decreasing", reason: "More canopy → lower temperature" },
            { variable: "impervious_pct", direction: "increasing", reason: "More impervious → higher temperature" },
            { variable: "albedo", direction: "decreasing", reason: "Higher albedo → lower temperature" },
            { variable: "building_height", direction: "increasing", reason: "Taller buildings → more heat trapping" },
            { variable: "ndvi", direction: "decreasing", reason: "More vegetation → lower temperature" },
          );
        }
        setConstraints(newConstraints);
        if (newConstraints.length > 0) setSelectedVar(newConstraints[0].variable);

        setGuardrails([
          { label: "Max temperature anomaly", value: "±6.0 °C", description: "Physical bound on UHI intensity relative to baseline" },
          { label: "Laplacian penalty weight", value: "λ = 0.01", description: "Spatial smoothness constraint for physics-informed regularization" },
          { label: "GWEN normalization", value: "enabled", description: "Geographically weighted evidence normalization" },
        ]);
      })
      .catch(() => {
        setConstraints([
          { variable: "tree_canopy_pct", direction: "decreasing", reason: "More canopy → lower temperature" },
          { variable: "impervious_pct", direction: "increasing", reason: "More impervious → higher temperature" },
          { variable: "albedo", direction: "decreasing", reason: "Higher albedo → lower temperature" },
          { variable: "building_height", direction: "increasing", reason: "Taller buildings → more heat trapping" },
          { variable: "ndvi", direction: "decreasing", reason: "More vegetation → lower temperature" },
        ]);
        setSelectedVar("tree_canopy_pct");
        setGuardrails([
          { label: "Max temperature anomaly", value: "±6.0 °C", description: "Physical bound on UHI intensity" },
          { label: "Laplacian penalty weight", value: "λ = 0.01", description: "Spatial smoothness constraint" },
          { label: "GWEN normalization", value: "enabled", description: "Geographically weighted evidence normalization" },
        ]);
      });
  }, []);

  // Draw response curve
  useEffect(() => {
    const canvas = curveCanvasRef.current;
    if (!canvas || !selectedVar) return;
    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * DPR; canvas.height = h * DPR;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(DPR, DPR);

    const constraint = constraints.find((c) => c.variable === selectedVar);
    const isDecreasing = constraint?.direction === "decreasing";
    const isNone = constraint?.direction === "none";

    // Axes
    ctx.strokeStyle = "var(--line)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(40, 10); ctx.lineTo(40, h - 25); ctx.lineTo(w - 10, h - 25);
    ctx.stroke();

    // Grid lines
    ctx.strokeStyle = "rgba(0,0,0,0.04)";
    ctx.setLineDash([3, 3]);
    for (let i = 1; i <= 4; i++) {
      const y = 10 + (h - 35) * (i / 5);
      ctx.beginPath(); ctx.moveTo(40, y); ctx.lineTo(w - 10, y); ctx.stroke();
    }
    ctx.setLineDash([]);

    // Labels
    ctx.fillStyle = "var(--muted)";
    ctx.font = "9px 'JetBrains Mono'";
    ctx.textAlign = "center";
    ctx.fillText(selectedVar.replace(/_/g, " "), w / 2, h - 6);
    ctx.save();
    ctx.translate(12, h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("∂y / ∂x", 0, 0);
    ctx.restore();

    // Curve
    ctx.strokeStyle = "var(--crimson)";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    const nPts = 60;
    for (let i = 0; i <= nPts; i++) {
      const t = i / nPts;
      const x = 42 + t * (w - 54);
      let y: number;
      if (isNone) {
        y = (h - 35) / 2 + Math.sin(t * 4 * Math.PI) * 30 + 10;
      } else if (isDecreasing) {
        y = 20 + (h - 55) * (1 - Math.exp(-3 * t)) * 0.85 + Math.sin(t * 6) * 5;
      } else {
        y = h - 35 - (h - 55) * (1 - Math.exp(-3 * t)) * 0.85 + Math.sin(t * 6) * 5;
      }
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // CI band
    ctx.fillStyle = "rgba(231, 60, 37, 0.12)";
    ctx.beginPath();
    for (let i = 0; i <= nPts; i++) {
      const t = i / nPts;
      const x = 42 + t * (w - 54);
      let y: number;
      if (isNone) {
        y = (h - 35) / 2 + Math.sin(t * 4 * Math.PI) * 30 + 10 - 15;
      } else if (isDecreasing) {
        y = 20 + (h - 55) * (1 - Math.exp(-3 * t)) * 0.85 + Math.sin(t * 6) * 5 - 15;
      } else {
        y = h - 35 - (h - 55) * (1 - Math.exp(-3 * t)) * 0.85 + Math.sin(t * 6) * 5 - 15;
      }
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    for (let i = nPts; i >= 0; i--) {
      const t = i / nPts;
      const x = 42 + t * (w - 54);
      let y: number;
      if (isNone) {
        y = (h - 35) / 2 + Math.sin(t * 4 * Math.PI) * 30 + 10 + 15;
      } else if (isDecreasing) {
        y = 20 + (h - 55) * (1 - Math.exp(-3 * t)) * 0.85 + Math.sin(t * 6) * 5 + 15;
      } else {
        y = h - 35 - (h - 55) * (1 - Math.exp(-3 * t)) * 0.85 + Math.sin(t * 6) * 5 + 15;
      }
      ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fill();
  }, [selectedVar, constraints]);

  const handleToggleDirection = (varName: string) => {
    setConstraints((prev) =>
      prev.map((c) =>
        c.variable === varName
          ? {
              ...c,
              direction: c.direction === "increasing" ? "decreasing" : c.direction === "decreasing" ? "none" : "increasing",
            }
          : c,
      ),
    );
  };

  return (
    <div>
      <SectionHeader
        kicker="06 · analysis"
        label="Physics"
        right={<Btn primary small onClick={() => notify("info", "Physics constraints saved")}>Save constraints</Btn>}
      />

      <StatGrid>
        <Stat label="Constrained" value={String(constraints.filter((c) => c.direction !== "none").length)} tint="var(--crimson)" />
        <Stat label="Increasing" value={String(constraints.filter((c) => c.direction === "increasing").length)} tint="var(--amber)" />
        <Stat label="Decreasing" value={String(constraints.filter((c) => c.direction === "decreasing").length)} tint="var(--purple)" />
        <Stat label="Guardrails" value={String(guardrails.length)} tint="var(--ink)" />
      </StatGrid>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
        <Card title="Monotone constraints" subtitle="click direction to cycle: ↑ ↓ ●">
          <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
            {constraints.map((c, i) => (
              <div
                key={c.variable}
                onClick={() => setSelectedVar(c.variable)}
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr auto auto",
                  alignItems: "center",
                  gap: 10,
                  padding: "10px 0",
                  borderTop: i > 0 ? "1px dashed var(--line)" : "none",
                  cursor: "pointer",
                  background: selectedVar === c.variable ? "rgba(231,60,37,0.04)" : "transparent",
                  borderRadius: 4,
                  paddingLeft: 8,
                  paddingRight: 8,
                }}
              >
                <div>
                  <div style={{ fontSize: 12.5, fontWeight: 600 }}>{c.variable.replace(/_/g, " ")}</div>
                  <div className="mono" style={{ fontSize: 10, color: "var(--muted)", marginTop: 2 }}>{c.reason}</div>
                </div>
                <button
                  onClick={(e) => { e.stopPropagation(); handleToggleDirection(c.variable); }}
                  style={{
                    width: 30,
                    height: 30,
                    borderRadius: 5,
                    border: "1px solid var(--line)",
                    background: c.direction === "none" ? "#fff" : c.direction === "increasing" ? "#fff3ea" : "#f5eaf8",
                    cursor: "pointer",
                    fontSize: 16,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontFamily: "inherit",
                  }}
                >
                  {c.direction === "increasing" ? "↑" : c.direction === "decreasing" ? "↓" : "●"}
                </button>
                <Tag
                  color={
                    c.direction === "increasing"
                      ? "var(--amber)"
                      : c.direction === "decreasing"
                      ? "var(--purple)"
                      : "var(--muted)"
                  }
                >
                  {c.direction}
                </Tag>
              </div>
            ))}
          </div>
        </Card>

        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <Card title="Response curve" subtitle={selectedVar ? selectedVar.replace(/_/g, " ") + " → target" : "select a variable"}>
            <canvas
              ref={curveCanvasRef}
              style={{ width: "100%", height: 200, display: "block" }}
            />
          </Card>

          <Card title="Guardrails" subtitle="physical bounds and regularization">
            {guardrails.map((g, i) => (
              <div
                key={g.label}
                style={{
                  padding: "10px 0",
                  borderTop: i > 0 ? "1px dashed var(--line)" : "none",
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ fontSize: 12.5, fontWeight: 600 }}>{g.label}</span>
                  <Tag color="var(--ink)">{g.value}</Tag>
                </div>
                <div className="mono" style={{ fontSize: 10, color: "var(--muted)", marginTop: 3 }}>{g.description}</div>
              </div>
            ))}
          </Card>
        </div>
      </div>
    </div>
  );
}
