import { useState, useEffect, useRef, useCallback } from "react";
import { SectionHeader, Card, Tag, Btn, Stat, StatGrid } from "@/components/ui/DesignSystem";
import { getConfig, saveConfig, getPdpCurves, dataSummary } from "@/lib/api";
import { useNotification } from "@/hooks/useNotifications";
import { usePipeline } from "@/hooks/PipelineProvider";
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

type BcType = "none" | "dirichlet" | "neumann" | "robin";
interface BoundaryConditions {
  type: BcType;
  // Dirichlet: fixed value at boundary (e.g. ambient temperature)
  value: string;
  // Neumann: flux (e.g. heat flux W/m²)
  flux: string;
  // Robin: h (convective coefficient) and T_inf (ambient)
  h: string;
  t_inf: string;
  // loss weight for the BC term
  weight: string;
}

const BC_DEFAULTS: BoundaryConditions = {
  type: "none",
  value: "",
  flux: "0",
  h: "10",
  t_inf: "",
  weight: "0.10",
};

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

// ── small helper: section header with ? tooltip ────────────────────────
function SectionHelp({ title, tip }: { title: string; tip: string }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
      {title}
      <span
        title={tip}
        style={{
          display: "inline-flex",
          alignItems: "center",
          justifyContent: "center",
          width: 15,
          height: 15,
          borderRadius: "50%",
          border: "1px solid var(--muted)",
          color: "var(--muted)",
          fontSize: 9,
          fontWeight: 700,
          cursor: "default",
          lineHeight: 1,
          fontFamily: "inherit",
          flexShrink: 0,
        }}
      >
        ?
      </span>
    </span>
  );
}

export default function PhysicsPage() {
  const [constraints, setConstraints] = useState<MonotoneConstraint[]>([]);
  const [pdeWeights, setPdeWeights] = useState<PdeWeights>({ ...PDE_DEFAULTS });
  const [variableBounds, setVariableBounds] = useState<VariableBound[]>([]);
  const [bc, setBc] = useState<BoundaryConditions>({ ...BC_DEFAULTS });
  const [literatureWeight, setLiteratureWeight] = useState<number>(0.25);
  const [selectedVar, setSelectedVar] = useState<string>("");
  const [pdpData, setPdpData] = useState<PdpCurves | null>(null);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [addConstraintVar, setAddConstraintVar] = useState<string>("");
  const [availablePredictors, setAvailablePredictors] = useState<string[]>([]);
  const curveCanvasRef = useRef<HTMLCanvasElement>(null);
  const { notify } = useNotification();
  const pipeline = usePipeline();

  useEffect(() => {
    Promise.all([getConfig(), dataSummary().catch(() => null)])
      .then(([config, summary]) => {
        const physics = config.physics ?? {};
        const mc = physics.monotone_constraints ?? {};
        const newConstraints: MonotoneConstraint[] = Object.entries(mc).map(([k, v]) => ({
          variable: k,
          direction: (v as number) > 0 ? "increasing" : (v as number) < 0 ? "decreasing" : "none",
          reason: (v as number) > 0 ? "Physical expectation: positive relationship" : (v as number) < 0 ? "Physical expectation: negative relationship" : "No constraint",
        }));
        setConstraints(newConstraints);
        if (newConstraints.length > 0) setSelectedVar(newConstraints[0].variable);

        setLiteratureWeight(
          typeof (physics as Record<string, unknown>).literature_weight === "number"
            ? (physics as Record<string, unknown>).literature_weight as number
            : 0.25,
        );

        // Load PDE weights
        const pw = (physics as Record<string, unknown>).pde_weights as Record<string, number> ?? {};
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

        // Load variable bounds
        const bounds = physics.variable_bounds ?? {};
        const newBounds: VariableBound[] = Object.entries(bounds).map(([k, v]) => {
          const b = v as { min?: number; max?: number };
          return { variable: k, min: b.min != null ? String(b.min) : "", max: b.max != null ? String(b.max) : "" };
        });
        setVariableBounds(newBounds);

        // Load boundary conditions
        const bcCfg = (physics as Record<string, unknown>).boundary_conditions as Record<string, unknown> ?? {};
        const bcType = (bcCfg.type ?? "none") as BcType;
        setBc({
          type: ["none", "dirichlet", "neumann", "robin"].includes(bcType) ? bcType : "none",
          value: bcCfg.value != null ? String(bcCfg.value) : "",
          flux: bcCfg.flux != null ? String(bcCfg.flux) : "0",
          h: bcCfg.h != null ? String(bcCfg.h) : "10",
          t_inf: bcCfg.t_inf != null ? String(bcCfg.t_inf) : "",
          weight: bcCfg.weight != null ? String(bcCfg.weight) : "0.10",
        });

        // Build available predictors for the add-constraint picker
        const rawPreds = config.predictors as unknown;
        const predList: string[] = Array.isArray(rawPreds)
          ? rawPreds
          : Array.isArray((rawPreds as Record<string, unknown>)?.base_model)
          ? ((rawPreds as Record<string, unknown>).base_model as string[])
          : summary?.columns ?? [];
        setAvailablePredictors(predList);
        if (!addConstraintVar && predList.length > 0) setAddConstraintVar(predList[0]);
      })
      .catch(() => {});

    getPdpCurves()
      .then((pdp) => setPdpData(pdp))
      .catch(() => {});
    // Refresh whenever a new pipeline run finishes so neural PDP CSVs are picked up.
  }, [pipeline.runEndedAt]); // eslint-disable-line react-hooks/exhaustive-deps

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
    const parseNum = (s: string): number | null => {
      if (s === "" || s == null) return null;
      const n = parseFloat(s);
      return Number.isFinite(n) ? n : null;
    };
    const bcPayload: Record<string, unknown> = { type: bc.type, weight: parseNum(bc.weight) ?? 0.1 };
    if (bc.type === "dirichlet") bcPayload.value = parseNum(bc.value);
    if (bc.type === "neumann") bcPayload.flux = parseNum(bc.flux);
    if (bc.type === "robin") {
      bcPayload.h = parseNum(bc.h);
      bcPayload.t_inf = parseNum(bc.t_inf);
    }
    try {
      await saveConfig({
        physics: {
          monotone_constraints: mc,
          literature_weight: literatureWeight,
          pde_weights: pdeWeights,
          variable_bounds: bounds,
          boundary_conditions: bcPayload,
        } as never,
      });
      notify("success", "Physics settings saved");
    } catch {
      notify("error", "Failed to save physics settings");
    }
  }, [constraints, literatureWeight, pdeWeights, variableBounds, bc, notify]);

  const handleAddBound = () => {
    setVariableBounds((prev) => [...prev, { variable: "", min: "", max: "" }]);
  };

  const handleRemoveBound = (i: number) => {
    setVariableBounds((prev) => prev.filter((_, idx) => idx !== i));
  };

  const handleBoundChange = (i: number, field: "variable" | "min" | "max", val: string) => {
    setVariableBounds((prev) => prev.map((b, idx) => idx === i ? { ...b, [field]: val } : b));
  };

  const handleAddConstraint = () => {
    if (!addConstraintVar) return;
    if (constraints.some((c) => c.variable === addConstraintVar)) {
      notify("info", `${addConstraintVar} already has a constraint`);
      return;
    }
    const newConstraint: MonotoneConstraint = {
      variable: addConstraintVar,
      direction: "none",
      reason: "User-defined",
    };
    setConstraints((prev) => [...prev, newConstraint]);
    setSelectedVar(addConstraintVar);
  };

  const handleRemoveConstraint = (varName: string) => {
    setConstraints((prev) => prev.filter((c) => c.variable !== varName));
    setSelectedVar((prev) => (prev === varName ? "" : prev));
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

      {/* Physics strength */}
      <div style={{ marginBottom: 14 }}>
        <Card
          title={
            <SectionHelp
              title="Physics strength"
              tip="Controls the balance between data-driven learning and physics prior knowledge. 0 = fully data-driven, 1 = fully physics-constrained. Start at 0.25 and increase if the model violates known physical relationships."
            /> as unknown as string
          }
          subtitle={`literature_weight = ${literatureWeight.toFixed(2)}`}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 14, padding: "4px 0" }}>
            <span className="mono" style={{ fontSize: 10, color: "var(--muted)", whiteSpace: "nowrap" }}>
              Data-driven
            </span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={literatureWeight}
              onChange={(e) => setLiteratureWeight(parseFloat(e.target.value))}
              style={{ flex: 1, accentColor: "var(--crimson)" }}
            />
            <span className="mono" style={{ fontSize: 10, color: "var(--muted)", whiteSpace: "nowrap" }}>
              Physics-informed
            </span>
            <span
              className="mono"
              style={{
                fontSize: 13,
                fontWeight: 700,
                color: "var(--crimson)",
                minWidth: 36,
                textAlign: "right",
              }}
            >
              {literatureWeight.toFixed(2)}
            </span>
          </div>
        </Card>
      </div>

      {/* Row 1: constraints + response curve */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
        <Card
          title={
            <SectionHelp
              title="Monotone constraints"
              tip="Enforce a direction on how each predictor affects the outcome. ↑ = must increase the target, ↓ = must decrease it. These constraints are injected into gradient-boosted and neural models to prevent physically implausible predictions."
            /> as unknown as string
          }
          subtitle="click direction to cycle: ↑ ↓ ●"
          actions={
            availablePredictors.length > 0 ? (
              <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                <select
                  value={addConstraintVar}
                  onChange={(e) => setAddConstraintVar(e.target.value)}
                  style={{
                    border: "1px solid var(--line)",
                    borderRadius: 4,
                    padding: "3px 6px",
                    fontSize: 11,
                    fontFamily: "inherit",
                    background: "#fff",
                    maxWidth: 130,
                  }}
                >
                  {availablePredictors.map((p) => (
                    <option key={p} value={p}>{p}</option>
                  ))}
                </select>
                <Btn small onClick={handleAddConstraint}>+ Add</Btn>
              </div>
            ) : undefined
          }
        >
          <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
            {constraints.length === 0 && (
              <div style={{ color: "var(--muted)", fontSize: 12, fontStyle: "italic", textAlign: "center", padding: 20 }}>
                Select a variable above and click <strong>+ Add</strong> to define a constraint
              </div>
            )}
            {constraints.map((c, i) => (
              <div
                key={c.variable}
                onClick={() => setSelectedVar(c.variable)}
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr auto auto auto",
                  alignItems: "center",
                  gap: 8,
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
                  title="Click to cycle: increasing → decreasing → none"
                  style={{
                    width: 30, height: 30, borderRadius: 5,
                    border: "1px solid " + (c.direction === "none" ? "var(--line)" : c.direction === "increasing" ? "var(--amber)" : "var(--purple)"),
                    background: c.direction === "none" ? "#fff" : c.direction === "increasing" ? "#fff3ea" : "#f5eaf8",
                    cursor: "pointer", fontSize: 16, display: "flex", alignItems: "center", justifyContent: "center",
                    transition: "all 0.15s",
                  }}
                >
                  {c.direction === "increasing" ? "↑" : c.direction === "decreasing" ? "↓" : "●"}
                </button>
                <Tag color={c.direction === "increasing" ? "var(--amber)" : c.direction === "decreasing" ? "var(--purple)" : "var(--muted)"}>
                  {c.direction}
                </Tag>
                <button
                  onClick={(e) => { e.stopPropagation(); handleRemoveConstraint(c.variable); }}
                  title="Remove constraint"
                  style={{
                    width: 20, height: 20, border: "none", background: "none",
                    cursor: "pointer", color: "var(--muted)", fontSize: 14,
                    display: "flex", alignItems: "center", justifyContent: "center",
                    borderRadius: 3,
                  }}
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        </Card>

        <Card
          title={
            <SectionHelp
              title="Response curve"
              tip="Partial dependence plot (PDP) showing the marginal effect of the selected predictor on the target. The shaded band is ±1 standard deviation across spatial folds. Only available after a pipeline run."
            /> as unknown as string
          }
          subtitle={selectedVar ? selectedVar.replace(/_/g, " ") + " → target" : "select a variable"}
          actions={(() => {
            const v = selectedVar ? pdpData?.[selectedVar] : undefined;
            const src = (v as { source?: string } | undefined)?.source;
            if (!src) return null;
            const isNeural = src === "neural_pde";
            return (
              <Tag color={isNeural ? "var(--purple)" : "var(--muted)"}>
                {isNeural ? "PINN · PDE" : "GWRF"}
              </Tag>
            );
          })()}
        >
          <canvas ref={curveCanvasRef} style={{ width: "100%", height: 200, display: "block" }} />
        </Card>
      </div>

      {/* Row 2: variable guardrails */}
      <div style={{ marginTop: 14 }}>
        <Card
          title={
            <SectionHelp
              title="Variable guardrails"
              tip="Hard physical bounds applied per predictor during training. Predictions that would require inputs outside these ranges are clipped. Use these to prevent extrapolation to physically impossible values (e.g. negative tree cover)."
            /> as unknown as string
          }
          subtitle="physical bounds per predictor — applied during training"
          actions={<Btn small onClick={handleAddBound}>+ Add</Btn>}
        >
          <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
            {variableBounds.length === 0 && (
              <div style={{ color: "var(--muted)", fontSize: 12, fontStyle: "italic", textAlign: "center", padding: 14 }}>
                No guardrails set. Click <strong>+ Add</strong> to constrain a variable range.
              </div>
            )}
            {variableBounds.length > 0 && (
              <div style={{ display: "grid", gridTemplateColumns: "1fr 80px 80px auto", gap: 6, padding: "0 0 6px" }}>
                <div className="mono" style={{ fontSize: 9.5, color: "var(--muted)" }}>Variable</div>
                <div className="mono" style={{ fontSize: 9.5, color: "var(--muted)", textAlign: "right" }}>Min</div>
                <div className="mono" style={{ fontSize: 9.5, color: "var(--muted)", textAlign: "right" }}>Max</div>
                <div />
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
                  padding: "6px 0",
                  borderTop: i > 0 ? "1px dashed var(--line)" : "none",
                }}
              >
                <input
                  type="text"
                  placeholder="variable name"
                  value={b.variable}
                  onChange={(e) => handleBoundChange(i, "variable", e.target.value)}
                  style={{ border: "1px solid var(--line)", borderRadius: 4, padding: "4px 6px", fontSize: 11, fontFamily: "inherit", background: "#fff" }}
                />
                <input
                  type="number"
                  placeholder="min"
                  value={b.min}
                  onChange={(e) => handleBoundChange(i, "min", e.target.value)}
                  className="mono"
                  style={{ border: "1px solid var(--line)", borderRadius: 4, padding: "4px 6px", fontSize: 11, fontFamily: "inherit", textAlign: "right", background: "#fff" }}
                />
                <input
                  type="number"
                  placeholder="max"
                  value={b.max}
                  onChange={(e) => handleBoundChange(i, "max", e.target.value)}
                  className="mono"
                  style={{ border: "1px solid var(--line)", borderRadius: 4, padding: "4px 6px", fontSize: 11, fontFamily: "inherit", textAlign: "right", background: "#fff" }}
                />
                <button
                  onClick={() => handleRemoveBound(i)}
                  title="Remove guardrail"
                  style={{ width: 22, height: 22, border: "none", background: "none", cursor: "pointer", color: "var(--muted)", fontSize: 14, display: "flex", alignItems: "center", justifyContent: "center" }}
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* Advanced section: PDE weights + boundary conditions */}
      <div style={{ marginTop: 14 }}>
        <button
          onClick={() => setAdvancedOpen((v) => !v)}
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            background: "none",
            border: "none",
            cursor: "pointer",
            padding: "6px 0",
            fontFamily: "inherit",
            color: "var(--muted)",
            fontSize: 11,
            fontWeight: 600,
            letterSpacing: "0.1em",
            textTransform: "uppercase",
          }}
        >
          <span style={{ fontSize: 10 }}>{advancedOpen ? "▼" : "▶"}</span>
          Advanced physics
          <span
            title="PDE loss weights and boundary conditions. These control the physics-informed neural network (PINN) training curriculum. Only adjust if you understand the underlying partial differential equations."
            style={{
              display: "inline-flex", alignItems: "center", justifyContent: "center",
              width: 15, height: 15, borderRadius: "50%", border: "1px solid var(--muted)",
              color: "var(--muted)", fontSize: 9, fontWeight: 700, cursor: "default",
              lineHeight: 1, fontFamily: "inherit",
            }}
          >
            ?
          </span>
        </button>

        {advancedOpen && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginTop: 10 }}>
            <Card
              title={
                <SectionHelp
                  title="PDE loss weights"
                  tip="8-term physics-informed loss function staged across training epochs. Each weight controls how strongly that physical equation is enforced. The curriculum activates terms progressively — earlier epochs use simpler physics before adding complex constraints."
                /> as unknown as string
              }
              subtitle="8-term physics-informed loss · staged sub-curriculum"
            >
              <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
                {(Object.keys(PDE_DEFAULTS) as Array<keyof PdeWeights>).map((key, i) => {
                  const info = PDE_EQUATIONS[key];
                  const val = pdeWeights[key];
                  const maxDefault = 1.0;
                  return (
                    <div
                      key={key}
                      style={{ padding: "9px 0", borderTop: i > 0 ? "1px dashed var(--line)" : "none" }}
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
                              fontSize: 12, fontWeight: 700, textAlign: "right", fontFamily: "inherit", background: "#fff",
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
              title={
                <SectionHelp
                  title="Boundary conditions"
                  tip="Constrains what happens at the edges of your study area. None = no edge constraint. Dirichlet = fixed known value at boundary (e.g. ambient temperature). Neumann = fixed flux (e.g. no-flux insulated edge). Robin = convective cooling at boundary."
                /> as unknown as string
              }
              subtitle="physics-informed constraint applied at domain edges"
            >
              <div style={{ display: "grid", gridTemplateColumns: "200px 1fr", gap: 14 }}>
                <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  {([
                    { id: "none", label: "None", hint: "no BC loss" },
                    { id: "dirichlet", label: "Dirichlet", hint: "fixed value  T = Tᵇ" },
                    { id: "neumann", label: "Neumann", hint: "flux  ∂T/∂n = q" },
                    { id: "robin", label: "Robin", hint: "convective  −k∇T·n = h(T − T∞)" },
                  ] as const).map((opt) => (
                    <button
                      key={opt.id}
                      onClick={() => setBc((prev) => ({ ...prev, type: opt.id }))}
                      style={{
                        textAlign: "left", padding: "8px 10px",
                        border: "1px solid " + (bc.type === opt.id ? "var(--crimson)" : "var(--line)"),
                        background: bc.type === opt.id ? "rgba(231,60,37,0.05)" : "#fff",
                        borderRadius: 5, cursor: "pointer", fontFamily: "inherit",
                        transition: "all 0.15s",
                      }}
                    >
                      <div style={{ fontSize: 12, fontWeight: 600 }}>{opt.label}</div>
                      <div className="mono" style={{ fontSize: 10, color: "var(--muted)", marginTop: 2 }}>{opt.hint}</div>
                    </button>
                  ))}
                </div>

                <div>
                  {bc.type === "none" && (
                    <div style={{ color: "var(--muted)", fontSize: 12, fontStyle: "italic", padding: "20px 10px" }}>
                      No boundary constraint applied. PDE loss will not penalize behaviour at the
                      fishnet edge. Suitable when the study area is large relative to edge effects
                      or when the outcome has no meaningful boundary value.
                    </div>
                  )}
                  {bc.type === "dirichlet" && (
                    <div style={{ display: "grid", gap: 10 }}>
                      <BcField label="Boundary value  Tᵇ" hint="fixed value at edge (e.g. ambient temperature °C)" value={bc.value} onChange={(v) => setBc((prev) => ({ ...prev, value: v }))} placeholder="e.g. 24.0" />
                      <BcField label="BC loss weight  λᴾᶜ" hint="multiplier for BC loss term (relative to diffusion)" value={bc.weight} onChange={(v) => setBc((prev) => ({ ...prev, weight: v }))} placeholder="0.10" />
                    </div>
                  )}
                  {bc.type === "neumann" && (
                    <div style={{ display: "grid", gap: 10 }}>
                      <BcField label="Boundary flux  q" hint="normal-derivative value (0 → insulated / no-flux)" value={bc.flux} onChange={(v) => setBc((prev) => ({ ...prev, flux: v }))} placeholder="0.0" />
                      <BcField label="BC loss weight  λᴾᶜ" hint="multiplier for BC loss term" value={bc.weight} onChange={(v) => setBc((prev) => ({ ...prev, weight: v }))} placeholder="0.10" />
                    </div>
                  )}
                  {bc.type === "robin" && (
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                      <BcField label="Convective coef.  h" hint="surface heat-transfer (W m⁻² K⁻¹)" value={bc.h} onChange={(v) => setBc((prev) => ({ ...prev, h: v }))} placeholder="10" />
                      <BcField label="Ambient  T∞" hint="far-field reference value" value={bc.t_inf} onChange={(v) => setBc((prev) => ({ ...prev, t_inf: v }))} placeholder="e.g. 24.0" />
                      <BcField label="BC loss weight  λᴾᶜ" hint="multiplier for BC loss term" value={bc.weight} onChange={(v) => setBc((prev) => ({ ...prev, weight: v }))} placeholder="0.10" />
                    </div>
                  )}
                </div>
              </div>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}

function BcField({
  label, hint, value, onChange, placeholder,
}: {
  label: string;
  hint: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
}) {
  return (
    <div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 8, marginBottom: 4 }}>
        <span style={{ fontSize: 11.5, fontWeight: 600 }}>{label}</span>
        <span className="mono" style={{ fontSize: 10, color: "var(--muted)" }}>{hint}</span>
      </div>
      <input
        type="number"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="mono"
        style={{
          width: "100%",
          padding: "6px 8px",
          border: "1px solid var(--line)",
          borderRadius: 4,
          fontSize: 12,
          fontFamily: "inherit",
        }}
      />
    </div>
  );
}
