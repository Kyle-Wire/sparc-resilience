import { useState, useEffect, useRef, useMemo } from "react";
import { SectionHeader, Card, Btn, Stat, StatGrid, KeyVal } from "@/components/ui/DesignSystem";
import {
  optimizeBudget,
  getCateMapVariables,
  getLocalCoefVariables,
  dataSummary,
  type BudgetOptimizeResponse,
} from "@/lib/api";
import { EmptyState } from "@/components/common/EmptyState";
import { SPARC_RAMP_HEX } from "@/lib/design-tokens";
import { useNotification } from "@/hooks/useNotifications";

type BenefitSource = "cate" | "local_coef" | "uniform";
type Solver = "auto" | "greedy" | "greedy_2opt" | "milp";

export default function BudgetOptimizerPage() {
  const { notify } = useNotification();

  const [benefitSource, setBenefitSource] = useState<BenefitSource>("cate");
  const [variable, setVariable] = useState<string>("");
  const [budget, setBudget] = useState<number>(100);
  const [solver, setSolver] = useState<Solver>("auto");
  const [costColumn, setCostColumn] = useState<string>("");
  const [xMaxColumn, setXMaxColumn] = useState<string>("");
  const [paretoSweep, setParetoSweep] = useState<boolean>(true);

  const [cateVars, setCateVars] = useState<string[]>([]);
  const [coefVars, setCoefVars] = useState<string[]>([]);
  const [columns, setColumns] = useState<string[]>([]);

  const [running, setRunning] = useState(false);
  const [response, setResponse] = useState<BudgetOptimizeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const allocCanvasRef = useRef<HTMLCanvasElement>(null);
  const paretoCanvasRef = useRef<HTMLCanvasElement>(null);

  // Load benefit source variable lists + dataset columns
  useEffect(() => {
    getCateMapVariables()
      .then((res) => setCateVars(res?.variables ?? []))
      .catch(() => setCateVars([]));
    getLocalCoefVariables()
      .then((res) => setCoefVars(res?.variables ?? []))
      .catch(() => setCoefVars([]));
    dataSummary()
      .then((s) => setColumns(s?.columns ?? []))
      .catch(() => setColumns([]));
  }, []);

  // Reset variable when source changes
  useEffect(() => {
    if (benefitSource === "uniform") {
      setVariable("");
      return;
    }
    const list = benefitSource === "cate" ? cateVars : coefVars;
    if (list.length > 0 && !list.includes(variable)) setVariable(list[0]);
  }, [benefitSource, cateVars, coefVars]); // eslint-disable-line react-hooks/exhaustive-deps

  const sourceList = benefitSource === "cate" ? cateVars : benefitSource === "local_coef" ? coefVars : [];
  const sourceUnavailable = benefitSource !== "uniform" && sourceList.length === 0;

  const canRun = budget > 0 && !running && (benefitSource === "uniform" || !!variable) && !sourceUnavailable;

  const onRun = async () => {
    if (!canRun) return;
    setRunning(true);
    setError(null);
    try {
      const resp = await optimizeBudget({
        budget,
        benefit_source: benefitSource,
        variable: benefitSource === "uniform" ? undefined : variable,
        cost_column: costColumn || undefined,
        x_max_column: xMaxColumn || undefined,
        solver,
        pareto_sweep: paretoSweep,
      });
      setResponse(resp);
      notify(
        "success",
        `Optimized in ${(resp.result.solve_time_s * 1000).toFixed(0)} ms — treated ${resp.result.n_treated}/${resp.n_cells} cells`,
      );
    } catch (e: any) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
      notify("error", msg);
    } finally {
      setRunning(false);
    }
  };

  // ----- Allocation histogram (canvas) -----
  useEffect(() => {
    const canvas = allocCanvasRef.current;
    if (!canvas) return;
    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * DPR; canvas.height = h * DPR;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(DPR, DPR);
    ctx.clearRect(0, 0, w, h);
    if (!response) {
      ctx.fillStyle = "#f5f1eb";
      ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = "#6e6358";
      ctx.font = "12px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Run the optimizer to see the allocation distribution", w / 2, h / 2);
      return;
    }

    const alloc = response.result.allocation;
    const nz = alloc.filter((v) => v > 1e-9);
    if (nz.length === 0) {
      ctx.fillStyle = "#f5f1eb";
      ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = "#a04050";
      ctx.font = "12px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Optimizer treated zero cells — increase budget?", w / 2, h / 2);
      return;
    }

    const bins = 32;
    const max = Math.max(...nz);
    const counts = new Array(bins).fill(0);
    for (const v of nz) {
      const i = Math.min(bins - 1, Math.floor((v / max) * bins));
      counts[i]++;
    }
    const maxCount = Math.max(...counts, 1);
    const pad = { top: 8, bottom: 18, left: 28, right: 8 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;
    const barW = plotW / bins;

    for (let i = 0; i < bins; i++) {
      const bh = (counts[i] / maxCount) * plotH;
      const color = SPARC_RAMP_HEX[Math.floor((i / bins) * SPARC_RAMP_HEX.length)] ?? SPARC_RAMP_HEX[0];
      ctx.fillStyle = color;
      ctx.fillRect(pad.left + i * barW + 1, pad.top + plotH - bh, Math.max(1, barW - 2), bh);
    }

    // Axis labels
    ctx.fillStyle = "#6e6358";
    ctx.font = "9px Inter, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("0", pad.left, pad.top + plotH + 12);
    ctx.textAlign = "right";
    ctx.fillText(max.toFixed(2), pad.left + plotW, pad.top + plotH + 12);
    ctx.textAlign = "left";
    ctx.fillText("allocation per cell (treated only)", pad.left, h - 4);
  }, [response]);

  // ----- Pareto frontier (canvas) -----
  useEffect(() => {
    const canvas = paretoCanvasRef.current;
    if (!canvas) return;
    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * DPR; canvas.height = h * DPR;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(DPR, DPR);
    ctx.clearRect(0, 0, w, h);

    if (!response?.pareto?.points?.length) {
      ctx.fillStyle = "#f5f1eb";
      ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = "#6e6358";
      ctx.font = "12px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(
        response ? "Pareto sweep was disabled" : "Run the optimizer to see the Pareto frontier",
        w / 2,
        h / 2,
      );
      return;
    }

    const pts = response.pareto.points;
    const xs = pts.map((p) => p.budget);
    const ys = pts.map((p) => p.total_benefit);
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    const yMin = 0, yMax = Math.max(...ys);
    const pad = { top: 12, bottom: 26, left: 38, right: 12 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;
    const sx = (x: number) => pad.left + ((x - xMin) / (xMax - xMin || 1)) * plotW;
    const sy = (y: number) => pad.top + plotH - ((y - yMin) / (yMax - yMin || 1)) * plotH;

    // Axis
    ctx.strokeStyle = "#d6cfc1";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top);
    ctx.lineTo(pad.left, pad.top + plotH);
    ctx.lineTo(pad.left + plotW, pad.top + plotH);
    ctx.stroke();

    // Curve
    ctx.strokeStyle = "#602468";
    ctx.lineWidth = 2;
    ctx.beginPath();
    pts.forEach((p, i) => {
      const x = sx(p.budget);
      const y = sy(p.total_benefit);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Points + labels (highlight current budget)
    pts.forEach((p) => {
      const x = sx(p.budget);
      const y = sy(p.total_benefit);
      const isCurrent = Math.abs(p.budget - response.result.budget) < 1e-6;
      ctx.fillStyle = isCurrent ? "#e73c25" : "#602468";
      ctx.beginPath();
      ctx.arc(x, y, isCurrent ? 5 : 3, 0, Math.PI * 2);
      ctx.fill();
    });

    // Labels
    ctx.fillStyle = "#6e6358";
    ctx.font = "9px Inter, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(`B=${xMin.toFixed(1)}`, pad.left - 4, pad.top + plotH + 12);
    ctx.textAlign = "right";
    ctx.fillText(`B=${xMax.toFixed(1)}`, pad.left + plotW, pad.top + plotH + 12);
    ctx.textAlign = "left";
    ctx.save();
    ctx.translate(8, pad.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillText("total benefit", 0, 0);
    ctx.restore();
    ctx.textAlign = "center";
    ctx.fillText("budget", pad.left + plotW / 2, h - 4);
  }, [response]);

  // ----- Equity quintiles -----
  const quintiles = useMemo(() => {
    if (!response) return null;
    const a = [...response.result.allocation].sort((x, y) => x - y);
    if (a.length === 0) return null;
    const out: { label: string; share: number }[] = [];
    const total = a.reduce((s, v) => s + v, 0) || 1;
    for (let q = 0; q < 5; q++) {
      const lo = Math.floor((q / 5) * a.length);
      const hi = Math.floor(((q + 1) / 5) * a.length);
      const sum = a.slice(lo, hi).reduce((s, v) => s + v, 0);
      out.push({ label: `Q${q + 1}`, share: sum / total });
    }
    return out;
  }, [response]);

  return (
    <div>
      <SectionHeader
        kicker="12 · pipeline"
        label="Budget Optimizer"
        right={
          <Btn primary disabled={!canRun} onClick={onRun}>
            {running ? "Optimizing…" : "Run optimizer"}
          </Btn>
        }
      />

      <Card title="Setup" subtitle="benefits come from your causal model; cost defaults to 1 per cell">
        <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 12 }}>
          <Field label="Benefit source">
            <select
              value={benefitSource}
              onChange={(e) => setBenefitSource(e.target.value as BenefitSource)}
              style={selectStyle}
            >
              <option value="cate">CATE (NUTS posterior — most defensible)</option>
              <option value="local_coef">PDE / MGWR local coefficient</option>
              <option value="uniform">Uniform (sanity reference)</option>
            </select>
          </Field>

          <Field label="Variable">
            {benefitSource === "uniform" ? (
              <span style={{ fontSize: 11, color: "var(--muted)" }}>not used for uniform source</span>
            ) : sourceList.length === 0 ? (
              <span style={{ fontSize: 11, color: "var(--crimson)" }}>
                no {benefitSource === "cate" ? "CATE" : "local-coefficient"} variables yet
              </span>
            ) : (
              <select value={variable} onChange={(e) => setVariable(e.target.value)} style={selectStyle}>
                {sourceList.map((v) => (
                  <option key={v} value={v}>{v}</option>
                ))}
              </select>
            )}
          </Field>

          <Field label="Budget">
            <input
              type="number"
              value={budget}
              min={0}
              step={1}
              onChange={(e) => setBudget(parseFloat(e.target.value) || 0)}
              style={inputStyle}
            />
          </Field>

          <Field label="Solver">
            <select value={solver} onChange={(e) => setSolver(e.target.value as Solver)} style={selectStyle}>
              <option value="auto">auto (recommended)</option>
              <option value="greedy">greedy</option>
              <option value="greedy_2opt">greedy + 2-opt</option>
              <option value="milp">MILP (exact, requires PuLP)</option>
            </select>
          </Field>

          <Field label="Cost column (optional)">
            <select value={costColumn} onChange={(e) => setCostColumn(e.target.value)} style={selectStyle}>
              <option value="">— uniform cost = 1 —</option>
              {columns.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </Field>

          <Field label="Max-allocation column (optional)">
            <select value={xMaxColumn} onChange={(e) => setXMaxColumn(e.target.value)} style={selectStyle}>
              <option value="">— uniform x_max = 1 —</option>
              {columns.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </Field>
        </div>

        <label style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 12, fontSize: 11 }}>
          <input
            type="checkbox"
            checked={paretoSweep}
            onChange={(e) => setParetoSweep(e.target.checked)}
          />
          Sweep budget at ×0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0 to expose diminishing returns
        </label>

        {sourceUnavailable && (
          <div style={{ marginTop: 12 }}>
            <EmptyState
              tone="waiting"
              compact
              title={`No ${benefitSource === "cate" ? "CATE" : "local-coefficient"} data yet.`}
              body={
                benefitSource === "cate"
                  ? "Run the pipeline through NUTS posterior sampling — that's where CATE comes from."
                  : "Run Stage 2b (MGWR/GWR) to generate local coefficients."
              }
            />
          </div>
        )}
      </Card>

      {/* ------------------------- Result summary ------------------------- */}
      {error && (
        <Card title="Optimizer error">
          <EmptyState tone="blocked" title="Optimizer failed." body={error} />
        </Card>
      )}

      {response && !error && (
        <>
          <StatGrid>
            <Stat
              label="Total benefit"
              value={response.result.total_benefit.toFixed(2)}
              tint="var(--crimson)"
              sub={response.benefit_description}
            />
            <Stat
              label="Spent"
              value={response.result.total_cost.toFixed(2)}
              tint="var(--purple)"
              sub={`of ${response.result.budget.toFixed(2)} budget`}
            />
            <Stat
              label="Cells treated"
              value={`${response.result.n_treated} / ${response.n_cells}`}
              tint="var(--ink)"
              sub={`${response.result.n_fully_treated} at max`}
            />
            <Stat
              label="Allocation Gini"
              value={response.result.gini.toFixed(3)}
              tint="var(--amber)"
              sub={response.result.gini > 0.7 ? "highly concentrated" : response.result.gini > 0.4 ? "moderately concentrated" : "spread evenly"}
            />
          </StatGrid>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
            <Card title="Allocation distribution" subtitle="histogram across treated cells">
              <canvas ref={allocCanvasRef} style={{ width: "100%", height: 220, display: "block" }} />
            </Card>

            <Card title="Pareto frontier" subtitle="benefit vs. budget across the sweep">
              <canvas ref={paretoCanvasRef} style={{ width: "100%", height: 220, display: "block" }} />
            </Card>
          </div>

          <Card title="Equity report" subtitle="share of budget by quintile (sorted ascending)">
            {quintiles ? (
              <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 8 }}>
                {quintiles.map((q, i) => (
                  <div key={q.label} style={{ textAlign: "center" }}>
                    <div className="mono" style={{ fontSize: 10, color: "var(--muted)" }}>{q.label}</div>
                    <div
                      style={{
                        height: 60,
                        marginTop: 4,
                        display: "flex",
                        alignItems: "flex-end",
                        justifyContent: "center",
                        background: "#fbf6ee",
                        borderRadius: 4,
                      }}
                    >
                      <div
                        style={{
                          width: "60%",
                          height: `${Math.max(2, q.share * 100)}%`,
                          background: SPARC_RAMP_HEX[i * 2] ?? SPARC_RAMP_HEX[0],
                          borderRadius: "2px 2px 0 0",
                        }}
                      />
                    </div>
                    <div className="mono" style={{ fontSize: 10, marginTop: 4 }}>
                      {(q.share * 100).toFixed(1)}%
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <span style={{ fontSize: 11, color: "var(--muted)" }}>—</span>
            )}
          </Card>

          <Card title="Run details">
            <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 8 }}>
              <KeyVal k="Solver">{response.result.solver}</KeyVal>
              <KeyVal k="Solve time">{(response.result.solve_time_s * 1000).toFixed(1)} ms</KeyVal>
              <KeyVal k="Benefit source">{response.benefit_source}</KeyVal>
              <KeyVal k="Cells">{String(response.n_cells)}</KeyVal>
            </div>
            {response.result.notes && (
              <p style={{ marginTop: 8, fontSize: 11, color: "var(--amber)" }}>{response.result.notes}</p>
            )}
          </Card>
        </>
      )}

      {!response && !error && (
        <Card>
          <EmptyState
            tone="awkward"
            title="No optimization run yet."
            body="Pick a benefit source, set a budget, and press Run. We'll allocate to maximize total benefit, then sweep the budget to show diminishing returns."
            action={
              <Btn primary disabled={!canRun} onClick={onRun}>
                Run optimizer
              </Btn>
            }
          />
        </Card>
      )}
    </div>
  );
}

// ---- small local helpers -------------------------------------------------
function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <span style={{ fontSize: 10, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
        {label}
      </span>
      {children}
    </label>
  );
}

const selectStyle: React.CSSProperties = {
  padding: "5px 8px",
  fontSize: 12,
  borderRadius: 4,
  border: "1px solid var(--line)",
  background: "#fff",
  fontFamily: "inherit",
};

const inputStyle: React.CSSProperties = {
  padding: "5px 8px",
  fontSize: 12,
  borderRadius: 4,
  border: "1px solid var(--line)",
  background: "#fff",
  fontFamily: "inherit",
};
