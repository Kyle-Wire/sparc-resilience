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

interface PdeWeights {
  heat_diffusion: number;
  energy_balance: number;
  directional: number;
  anisotropy: number;
  gradient_flux: number;
  gaussian_curv: number;
  alpha_smooth: number;
  alpha_prior: number;
}

interface VariableBound {
  variable: string;
  min: string;
  max: string;
}

const PDE_DEFAULTS: PdeWeights = {
  heat_diffusion: 1.0,
  energy_balance: 0.50,
  directional: 0.20,
  anisotropy: 0.10,
  gradient_flux: 0.10,
  gaussian_curv: 0.05,
  alpha_smooth: 0.10,
  alpha_prior: 0.10,
};

const PDE_EQUATIONS: Record<keyof PdeWeights, { label: string; eq: string; activates: string }> = {
  heat_diffusion: { label: "Heat Diffusion", eq: "α∇²T − S ≈ 0", activates: "Epoch 1" },
  energy_balance: { label: "Energy Balance", eq: "Q* − QH − QE ≈ 0", activates: "Epoch 5" },
  directional: { label: "Directional Curvature", eq: "∂²T/∂x² + ∂²T/∂y² consistent", activates: "Epoch 10" },
  anisotropy: { label: "Anisotropy Penalty", eq: "penalize spurious isotropy", activates: "Epoch 10" },
  gradient_flux: { label: "Gradient Flux", eq: "Fourier flux: q = −k∇T", activates: "Epoch 15" },
  gaussian_curv: { label: "Gaussian Curvature", eq: "det(H) regularizer", activates: "Epoch 15" },
  alpha_smooth: { label: "α Smoothness", eq: "‖∇α(s)‖² penalty", activates: "Epoch 15" },
  alpha_prior: { label: "α Prior", eq: "‖α(s) − α̅‖² from mixture prior", activates: "Epoch 15" },
};

export default function PhysicsPage() {
  const [constraints, setConstraints] = useState<MonotoneConstraint[]>([]);
  const [pdeWeights, setPdeWeights] = useState<PdeWeights>({ ...PDE_DEFAULTS });
  const [variableBounds, setVariableBounds] = useState<VariableBound[]>([]);
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

        // Load PDE weights
        const pw = (physics as any).pde_weights ?? {};
        setPdeWeights({
          heat_diffusion: pw.heat_diffusion ?? PDE_DEFAULTS.heat_diffusion,
          energy_balance: pw.energy_balance ?? PDE_DEFAULTS.energy_balance,
          directional: pw.directional ?? PDE_DEFAULTS.directional,
          anisotropy: pw.anisotropy ?? PDE_DEFAULTS.anisotropy,
          gradient_flux: pw.gradient_flux ?? PDE_DEFAULTS.gradient_flux,
          gaussian_curv: pw.gaussian_curv ?? PDE_DEFAULTS.gaussian_curv,
          alpha_smooth: pw.alpha_smooth ?? PDE_DEFAULTS.alpha_smooth,
          alpha_prior: pw.alpha_prior ?? PDE_DEFAULTS.alpha_prior,
        });

        // Load variable bounds as editable guardrails
        const bounds = physics.variable_bounds ?? {};
        const newBounds: VariableBound[] = Object.entries(bounds).map(([k, v]) => {
          const b = v as { min?: number; max?: number };
          return { variable: k, min: b.min != null ? String(b.min) : "", max: b.max != null ? String(b.max) : "" };
        });
        setVariableBounds(newBounds);
      })
      .catch(() => {});

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
    const bounds: Record<string, { min: number | null; max: number | null }> = {};
    for (const b of variableBounds) {
      bounds[b.variable] = {
        min: b.min !== "" ? parseFloat(b.min) : null,
        max: b.max !== "" ? parseFloat(b.max) : null,
      };
    }
    try {
      await saveConfig({
        physics: {
          monotone_constraints: mc,
          pde_weights: pdeWeights,
          variable_bounds: bounds,
        } as any,
      });
      notify("success", "Physics settings saved");
    } catch {
      notify("error", "Failed to save physics settings");
    }
  }, [constraints, pdeWeights, variableBounds, notify]);

  const handleAddBound = () => {
    setVariableBounds((prev) => [...prev, { variable: "", min: "", max: "" }]);
  };

  const handleRemoveBound = (i: number) => {
    setVariableBounds((prev) => prev.filter((_, idx) => idx !== i));
  };

  const handleBoundChange = (i: number, field: "variable" | "min" | "max", val: string) => {
    setVariableBounds((prev) => prev.map((b, idx) => idx === i ? { ...b, [field]: val } : b));
  };

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
        right={<Btn primary small onClick={handleSave}>Save physics</Btn>}
      />

      <StatGrid>
        <Stat label="Constrained" value={String(constraints.filter((c) => c.direction !== "none").length)} tint="var(--crimson)" />
        <Stat label="Increasing" value={String(constraints.filter((c) => c.direction === "increasing").length)} tint="var(--amber)" />
        <Stat label="Decreasing" value={String(constraints.filter((c) => c.direction === "decreasing").length)} tint="var(--purple)" />
        <Stat label="Guardrails" value={String(variableBounds.length)} tint="var(--ink)" />
      </StatGrid>

      {/* Row 1: constraints + response curve */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
        <Card title="Monotone constraints" subtitle="click direction to cycle: ↑ ↓ ●">
          <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
            {constraints.length === 0 && (
              <div style={{ color: "var(--muted)", fontSize: 12, fontStyle: "italic", textAlign: "center", padding: 20 }}>
                Configure predictor variables in the Data page first
              </div>
            )}
            {constraints.map((c, i) => (
              <div
                key={c.variable}
                onClick={() => setSelectedVar(c.variable)}
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr auto auto",
                  alignItems: "center",
                  gap: 10,
                  padding: "10px 8px",
                  borderTop: i > 0 ? "1px dashed var(--line)" : "none",
                  cursor: "pointer",
                  background: selectedVar === c.variable ? "rgba(231,60,37,0.04)" : "transparent",
                  borderRadius: 4,
                }}
              >
                <div>
                  <div style={{ fontSize: 12.5, fontWeight: 600 }}>{c.variable.replace(/_/g, " ")}</div>
                  <div className="mono" style={{ fontSize: 10, color: "var(--muted)", marginTop: 2 }}>{c.reason}</div>
                </div>
                <button
                  onClick={(e) => { e.stopPropagation(); handleToggleDirection(c.variable); }}
                  style={{
                    width: 30, height: 30, borderRadius: 5, border: "1px solid var(--line)",
                    background: c.direction === "none" ? "#fff" : c.direction === "increasing" ? "#fff3ea" : "#f5eaf8",
                    cursor: "pointer", fontSize: 16, display: "flex", alignItems: "center", justifyContent: "center",
                  }}
                >
                  {c.direction === "increasing" ? "↑" : c.direction === "decreasing" ? "↓" : "●"}
                </button>
                <Tag color={c.direction === "increasing" ? "var(--amber)" : c.direction === "decreasing" ? "var(--purple)" : "var(--muted)"}>
                  {c.direction}
                </Tag>
              </div>
            ))}
          </div>
        </Card>

        <Card title="Response curve" subtitle={selectedVar ? selectedVar.replace(/_/g, " ") + " → target" : "select a variable"}>
          <canvas ref={curveCanvasRef} style={{ width: "100%", height: 200, display: "block" }} />
        </Card>
      </div>

      {/* Row 2: PDE weights + editable guardrails */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginTop: 14 }}>
        <Card title="PDE loss weights" subtitle="8-term physics-informed loss · staged sub-curriculum">
          <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
            {(Object.keys(PDE_DEFAULTS) as Array<keyof PdeWeights>).map((key, i) => {
              const info = PDE_EQUATIONS[key];
              const val = pdeWeights[key];
              const maxDefault = 1.0;
              return (
                <div
                  key={key}
                  style={{
                    padding: "9px 0",
                    borderTop: i > 0 ? "1px dashed var(--line)" : "none",
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
                    <div>
                      <span style={{ fontSize: 12, fontWeight: 600 }}>{info.label}</span>
                      <span className="mono" style={{ fontSize: 10, color: "var(--muted)", marginLeft: 8 }}>{info.eq}</span>
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                      <Tag color="var(--ink-2)">{info.activates}</Tag>
                      <input
                        type="number"
                        step="0.01"
                        min="0"
                        max="10"
                        value={val}
                        onChange={(e) => setPdeWeights((prev) => ({ ...prev, [key]: parseFloat(e.target.value) || 0 }))}
                        className="mono"
                        style={{
                          width: 58, padding: "3px 6px", border: "1px solid var(--line)", borderRadius: 4,
                          fontSize: 12, fontWeight: 700, textAlign: "right", fontFamily: "inherit",
                        }}
                      />
                    </div>
                  </div>
                  <div style={{ height: 4, background: "rgba(0,0,0,0.05)", borderRadius: 2, overflow: "hidden" }}>
                    <div
                      style={{
                        width: `${Math.min((val / maxDefault) * 100, 100)}%`,
                        height: "100%",
                        background: val > 0 ? "var(--crimson)" : "var(--muted)",
                        transition: "width 0.2s",
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </Card>

        <Card
          title="Variable guardrails"
          subtitle="physical bounds per predictor — applied during training"
          actions={
            <button
              onClick={handleAddBound}
              style={{ padding: "3px 10px", background: "var(--ink)", color: "#fff", border: "none", borderRadius: 4, fontSize: 11, cursor: "pointer", fontWeight: 600 }}
            >
              + Add
            </button>
          }
        >
          <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
            {variableBounds.length === 0 && (
              <div style={{ color: "var(--muted)", fontSize: 12, fontStyle: "italic", textAlign: "center", padding: 14 }}>
                No guardrails set. Click <strong>+ Add</strong> to constrain a variable range.
              </div>
            )}
            {variableBounds.map((b, i) => (
              <div
                key={i}
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr 80px 80px auto",
                  gap: 6,
                  alignItems: "center",
                  padding: "8px 0",
                  borderTop: i > 0 ? "1px dashed var(--line)" : "none",
                }}
              >
                <input
                  type="text"
                  placeholder="variable name"
                  value={b.variable}
                  onChange={(e) => handleBoundChange(i, "variable", e.target.value)}
                  style={{ border: "1px solid var(--line)", borderRadius: 4, padding: "4px 6px", fontSize: 11, fontFamily: "inherit" }}
                />
                <input
                  type="number"
                  placeholder="min"
                  value={b.min}
                  onChange={(e) => handleBoundChange(i, "min", e.target.value)}
                  className="mono"
                  style={{ border: "1px solid var(--line)", borderRadius: 4, padding: "4px 6px", fontSize: 11, fontFamily: "inherit", textAlign: "right" }}
                />
                <input
                  type="number"
                  placeholder="max"
                  value={b.max}
                  onChange={(e) => handleBoundChange(i, "max", e.target.value)}
                  className="mono"
                  style={{ border: "1px solid var(--line)", borderRadius: 4, padding: "4px 6px", fontSize: 11, fontFamily: "inherit", textAlign: "right" }}
                />
                <button
                  onClick={() => handleRemoveBound(i)}
                  style={{ width: 22, height: 22, border: "none", background: "none", cursor: "pointer", color: "var(--muted)", fontSize: 14, display: "flex", alignItems: "center", justifyContent: "center" }}
                >
                  ×
                </button>
              </div>
            ))}
            {variableBounds.length > 0 && (
              <div style={{ display: "grid", gridTemplateColumns: "1fr 80px 80px auto", gap: 6, padding: "4px 0 0" }}>
                <div className="mono" style={{ fontSize: 9.5, color: "var(--muted)" }}>Variable</div>
                <div className="mono" style={{ fontSize: 9.5, color: "var(--muted)", textAlign: "right" }}>Min</div>
                <div className="mono" style={{ fontSize: 9.5, color: "var(--muted)", textAlign: "right" }}>Max</div>
                <div />
              </div>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}
