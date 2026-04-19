import { useState, useEffect, useRef, useCallback } from "react";
import { SectionHeader, Card, Tag, Btn, Stat, StatGrid } from "@/components/ui/DesignSystem";
import { getConfig, saveConfig, getPdpCurves } from "@/lib/api";
import { useNotification } from "@/hooks/useNotifications";
import type { PdpCurves } from "@/lib/types";

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
  const [pdpData, setPdpData] = useState<PdpCurves | null>(null);
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
        setConstraints(newConstraints);
        if (newConstraints.length > 0) setSelectedVar(newConstraints[0].variable);

        // Derive guardrails from config
        const bounds = physics.variable_bounds;
        const litWeight = physics.literature_weight;
        const gails: Guardrail[] = [];
        if (bounds && typeof bounds === "object") {
          for (const [k, v] of Object.entries(bounds)) {
            const b = v as { min?: number; max?: number };
            gails.push({ label: `Bound: ${k}`, value: `${b.min ?? "−∞"} … ${b.max ?? "∞"}`, description: `Physical bounds on ${k}` });
          }
        }
        if (typeof litWeight === "number") {
          gails.push({ label: "Literature weight", value: String(litWeight), description: "Weight given to literature-derived constraints" });
        }
        setGuardrails(gails);
      })
      .catch(() => {});

    // Load real PDP curves
    getPdpCurves()
      .then((pdp) => setPdpData(pdp))
      .catch(() => {});
  }, []);

  // Draw response curve from real PDP data
  useEffect(() => {
    const canvas = curveCanvasRef.current;
    if (!canvas || !selectedVar) return;
    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * DPR; canvas.height = h * DPR;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(DPR, DPR);

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

    const pdpVar = pdpData?.[selectedVar];
    const gridVals = pdpVar?.pdp?.grid_values ?? pdpVar?.grid_values;
    const pdpVals = pdpVar?.pdp?.pdp_values ?? pdpVar?.pdp_values;
    const pdpStd = pdpVar?.pdp?.pdp_std;

    if (!gridVals || !pdpVals || gridVals.length === 0) {
      ctx.fillStyle = "var(--muted)";
      ctx.font = "12px Inter";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("Run pipeline to see response curves", w / 2, h / 2);
      return;
    }

    const xMin = Math.min(...gridVals), xMax = Math.max(...gridVals);
    const yMin = Math.min(...pdpVals), yMax = Math.max(...pdpVals);
    const xRange = xMax - xMin || 1, yRange = yMax - yMin || 1;
    const plotL = 42, plotR = w - 12, plotT = 14, plotB = h - 28;

    const toX = (v: number) => plotL + ((v - xMin) / xRange) * (plotR - plotL);
    const toY = (v: number) => plotB - ((v - yMin) / yRange) * (plotB - plotT);

    // CI band from pdp_std
    if (pdpStd && pdpStd.length === pdpVals.length) {
      ctx.fillStyle = "rgba(231, 60, 37, 0.12)";
      ctx.beginPath();
      for (let i = 0; i < gridVals.length; i++) {
        const x = toX(gridVals[i]), y = toY(pdpVals[i] + pdpStd[i]);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      for (let i = gridVals.length - 1; i >= 0; i--) {
        ctx.lineTo(toX(gridVals[i]), toY(pdpVals[i] - pdpStd[i]));
      }
      ctx.closePath();
      ctx.fill();
    }

    // Main curve
    ctx.strokeStyle = "var(--crimson)";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    for (let i = 0; i < gridVals.length; i++) {
      const x = toX(gridVals[i]), y = toY(pdpVals[i]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }, [selectedVar, constraints, pdpData]);

  const handleSave = useCallback(async () => {
    const mc: Record<string, number> = {};
    for (const c of constraints) {
      mc[c.variable] = c.direction === "increasing" ? 1 : c.direction === "decreasing" ? -1 : 0;
    }
    try {
      await saveConfig({ physics: { monotone_constraints: mc } });
      notify("success", "Physics constraints saved");
    } catch {
      notify("error", "Failed to save constraints");
    }
  }, [constraints, notify]);

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
        right={<Btn primary small onClick={handleSave}>Save constraints</Btn>}
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
