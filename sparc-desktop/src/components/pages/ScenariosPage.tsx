import { useState, useEffect, useCallback, useRef } from "react";
import { SectionHeader, Card, Tag, Btn, Stat, StatGrid, thStyle, tdStyle } from "@/components/ui/DesignSystem";
import {
  getConfig,
  getScenarioDetail,
  getScenarioLibrary,
  appendScenarioToLibrary,
  getScenarioConfig,
  updateProjectScenarios,
  deleteProjectScenario,
  dataSummary,
  runScenarioChain,
  type ScenarioTimeline,
  type ChainAction,
  type ChainStepResult,
  type SweepScenarioSpec,
  type JointScenarioSpec,
  type JointScenarioIntervention,
} from "@/lib/api";
import { useNotification } from "@/hooks/useNotifications";
import { SPARC_RAMP_HEX } from "@/lib/design-tokens";

type ScenariosTab = "configure" | "library" | "timeline";

/** Infer the natural absolute range for a variable from its name. */
function naturalRange(col: string): { lo: number; hi: number; step: number; unit: string } | null {
  const lower = col.toLowerCase();
  if (/pct|percent|impervious|canopy|cover|urban|green/.test(lower)) {
    return { lo: 0, hi: 100, step: 1, unit: "%" };
  }
  if (/ndvi|evi|savi|lai|fpar|vegetation.index/.test(lower)) {
    return { lo: -1, hi: 1, step: 0.05, unit: "" };
  }
  if (/albedo|reflect/.test(lower)) {
    return { lo: 0, hi: 1, step: 0.02, unit: "" };
  }
  return null;
}


interface Scenario {
  id: string;
  name: string;
  interventions: Record<string, number>;
  delta: number;
  status: "draft" | "computed" | "baseline";
}

interface InterventionSlider {
  variable: string;
  min: number;
  max: number;
  step: number;
  unit: string;
  value: number;
  baseline: number;
}

export default function ScenariosPage() {
  const { notify } = useNotification();
  const [tab, setTab] = useState<ScenariosTab>("configure");
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [activeIdx, setActiveIdx] = useState(0);
  const [sliders, setSliders] = useState<InterventionSlider[]>([]);
  const [_configRaw, setConfigRaw] = useState<Record<string, unknown> | null>(null);
  const [library, setLibrary] = useState<ScenarioTimeline | null>(null);
  const [libParent, setLibParent] = useState<string | null>(null);
  const [libComment, setLibComment] = useState("");
  const [libAuthor, setLibAuthor] = useState("me");
  const histRef = useRef<HTMLCanvasElement>(null);

  // Defined scenarios (from project.yml, written back on save)
  const [definedSweeps, setDefinedSweeps] = useState<SweepScenarioSpec[]>([]);
  const [definedJoints, setDefinedJoints] = useState<JointScenarioSpec[]>([]);
  const [showRerunBanner, setShowRerunBanner] = useState(false);

  // Sweep builder state
  type BuilderMode = "sweep" | "joint" | null;
  const [builderMode, setBuilderMode] = useState<BuilderMode>(null);
  const [sweepName, setSweepName] = useState("");
  const [sweepVariable, setSweepVariable] = useState("");
  const [sweepDirection, setSweepDirection] = useState<"increase" | "decrease">("increase");
  const [sweepMin, setSweepMin] = useState("0");
  const [sweepMax, setSweepMax] = useState("50");
  const [sweepStep, setSweepStep] = useState("5");
  const [sweepUnit, setSweepUnit] = useState("");

  // Joint builder state
  const [jointName, setJointName] = useState("");
  const [jointPropagate, setJointPropagate] = useState(true);
  const [jointRows, setJointRows] = useState<JointScenarioIntervention[]>([
    { variable: "", direction: "increase", increment: 10 },
  ]);

  // Timeline tab state
  const [chainActions, setChainActions] = useState<ChainAction[]>([]);
  const [chainRunning, setChainRunning] = useState(false);
  const [chainResult, setChainResult] = useState<ChainStepResult[] | null>(null);
  const [chainCumulative, setChainCumulative] = useState<number | null>(null);
  const [chainTreatment, setChainTreatment] = useState("");
  const [chainDeltaX, setChainDeltaX] = useState("0");

  useEffect(() => {
    // Load scenarios from API results
    getScenarioDetail()
      .then((detail: any) => {
        if (detail?.scenarios && Array.isArray(detail.scenarios) && detail.scenarios.length > 0) {
          setScenarios(
            detail.scenarios.map((sc: any, i: number) => ({
              id: `s${i}`,
              name: sc.name ?? `Scenario ${i + 1}`,
              interventions: sc.interventions ?? {},
              delta: sc.delta ?? sc.mean_delta ?? 0,
              status: "computed" as const,
            })),
          );
        }
      })
      .catch(() => {});

    // Load defined scenarios from project.yml via /scenarios/config
    getScenarioConfig()
      .then((cfg) => {
        setDefinedSweeps(cfg.scenarios ?? []);
        setDefinedJoints(cfg.joint_scenarios ?? []);
        // Pre-populate sweep builder variable from first defined sweep
        if ((cfg.scenarios ?? []).length > 0 && cfg.scenarios[0].variable) {
          setSweepVariable(cfg.scenarios[0].variable);
        }
      })
      .catch(() => {});

    // Load config for slider predictors + preset hints
    getConfig()
      .then((config) => {
        setConfigRaw(config as unknown as Record<string, unknown>);
        const cols = config.predictors ?? [];

        // Build sliders from predictors in config, using domain-inferred natural ranges
        if (cols.length > 0) {
          const top = cols.slice(0, 6);
          dataSummary()
            .then((ds) => {
              const ns = ds?.numeric_summary ?? {};
              setSliders(
                top.map((col: string) => {
                  const stats = ns[col];
                  const mean = stats ? Number(stats.mean ?? 0) : 0;
                  const baseline = Number.isFinite(mean) ? mean : 0;

                  const nr = naturalRange(col);
                  let lo: number;
                  let hi: number;
                  let step: number;
                  let unit: string;

                  if (nr) {
                    lo = nr.lo;
                    hi = nr.hi;
                    step = nr.step;
                    unit = nr.unit;
                  } else {
                    const dataMin = stats ? Number(stats.min ?? baseline) : baseline - 0.5;
                    const dataMax = stats ? Number(stats.max ?? baseline) : baseline + 0.5;
                    lo = Number.isFinite(dataMin) ? dataMin : baseline - 0.5;
                    hi = Number.isFinite(dataMax) ? dataMax : baseline + 0.5;
                    step = Math.max((hi - lo) / 40, 1e-4);
                    unit = "";
                  }

                  return {
                    variable: col,
                    min: lo,
                    max: hi,
                    step,
                    unit,
                    value: Math.min(Math.max(baseline, lo), hi),
                    baseline: Math.min(Math.max(baseline, lo), hi),
                  };
                }),
              );
            })
            .catch(() => {
              setSliders(
                top.map((col: string) => ({
                  variable: col,
                  min: 0,
                  max: 100,
                  step: 1,
                  unit: "",
                  value: 0,
                  baseline: 0,
                })),
              );
            });
        }
      })
      .catch(() => {});
  }, []);

  // Posterior histogram
  useEffect(() => {
    const canvas = histRef.current;
    if (!canvas) return;
    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * DPR; canvas.height = h * DPR;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(DPR, DPR);

    const active = scenarios[activeIdx];
    if (!active) {
      ctx.fillStyle = "#6e6358";
      ctx.font = "12px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Add and compute scenarios to see results", w / 2, h / 2);
      return;
    }

    if (active.status !== "computed" || active.delta === 0) {
      ctx.fillStyle = "#6e6358";
      ctx.font = "12px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Compute scenarios to see Δ distribution", w / 2, h / 2);
      return;
    }

    // -- Posterior Gaussian density ---------------------------------------
    // Pick the dominant intervention variable; lookup its NUTS posterior std
    // (per-unit). Std of Δ = |intervention| × std_per_unit.  If no posterior
    // available, fall back to an assumed CV of 25% so the density is still
    // informative.
    const interventionEntries = Object.entries(active.interventions);
    interventionEntries.sort(
      (a, b) => Math.abs(b[1]) - Math.abs(a[1]),
    );
    let sigma = Math.max(Math.abs(active.delta) * 0.25, 1e-6);
    const posteriorMatched = false;

    const mu = active.delta;
    const xMin = mu - 4 * sigma;
    const xMax = mu + 4 * sigma;
    const xRange = Math.max(xMax - xMin, 1e-9);

    // Compute densities
    const N = 120;
    const xs = Array.from({ length: N }, (_, i) => xMin + (i / (N - 1)) * xRange);
    const ys = xs.map((x) => Math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * Math.sqrt(2 * Math.PI)));
    const yMax = Math.max(...ys);

    const plotL = 36, plotR = w - 12, plotT = 14, plotB = h - 28;
    const toX = (x: number) => plotL + ((x - xMin) / xRange) * (plotR - plotL);
    const toY = (y: number) => plotB - (y / yMax) * (plotB - plotT);

    // Axes
    ctx.strokeStyle = "#c9c2b3";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(plotL, plotT); ctx.lineTo(plotL, plotB); ctx.lineTo(plotR, plotB);
    ctx.stroke();

    // Zero reference line
    if (xMin <= 0 && xMax >= 0) {
      ctx.strokeStyle = "#a59f93";
      ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(toX(0), plotT); ctx.lineTo(toX(0), plotB); ctx.stroke();
      ctx.setLineDash([]);
    }

    // 90% CI shading (mu ± 1.645 σ)
    const ciLo = mu - 1.645 * sigma;
    const ciHi = mu + 1.645 * sigma;
    ctx.fillStyle = SPARC_RAMP_HEX[mu < 0 ? 0 : SPARC_RAMP_HEX.length - 1] + "33";
    ctx.beginPath();
    ctx.moveTo(toX(ciLo), plotB);
    for (let i = 0; i < xs.length; i++) {
      if (xs[i] < ciLo || xs[i] > ciHi) continue;
      ctx.lineTo(toX(xs[i]), toY(ys[i]));
    }
    ctx.lineTo(toX(ciHi), plotB);
    ctx.closePath();
    ctx.fill();

    // Density curve
    ctx.strokeStyle = SPARC_RAMP_HEX[mu < 0 ? 0 : SPARC_RAMP_HEX.length - 1];
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < xs.length; i++) {
      const px = toX(xs[i]), py = toY(ys[i]);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();

    // Mean marker
    ctx.strokeStyle = "#1a1416";
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(toX(mu), plotT); ctx.lineTo(toX(mu), plotB); ctx.stroke();

    // Axis labels (x-axis)
    ctx.fillStyle = "#6e6358";
    ctx.font = "9px 'JetBrains Mono'";
    ctx.textAlign = "center";
    ctx.fillText(xMin.toFixed(2), plotL, h - 12);
    ctx.fillText(xMax.toFixed(2), plotR, h - 12);
    ctx.fillText("Δ response", w / 2, h - 2);

    // Mean + CI label
    ctx.fillStyle = "#1a1416";
    ctx.font = "bold 10px 'JetBrains Mono'";
    ctx.textAlign = "center";
    ctx.fillText(
      `μ=${mu.toFixed(2)}  90% CI [${ciLo.toFixed(2)}, ${ciHi.toFixed(2)}]`,
      w / 2,
      plotT - 2,
    );
    if (!posteriorMatched) {
      ctx.fillStyle = "#9a8e75";
      ctx.font = "8px 'JetBrains Mono'";
      ctx.fillText("σ estimated (no NUTS posterior matched)", w / 2, plotT + 10);
    }
  }, [scenarios, activeIdx]);


  const refreshLibrary = useCallback(() => {
    getScenarioLibrary().then(setLibrary).catch(() => setLibrary(null));
  }, []);


  useEffect(() => { refreshLibrary(); }, [refreshLibrary]);

  const handleSaveActiveToLibrary = useCallback(async () => {
    const sc = scenarios[activeIdx];
    if (!sc) { notify("warning", "No scenario selected"); return; }
    try {
      const entry = await appendScenarioToLibrary(
        { name: sc.name, interventions: sc.interventions, delta: sc.delta, status: sc.status },
        { author: libAuthor, comment: libComment || sc.name, parent_id: libParent },
      );
      notify("success", `Saved → ${entry.id}`);
      setLibComment("");
      refreshLibrary();
    } catch (err) {
      notify("error", err instanceof Error ? err.message : String(err));
    }
  }, [scenarios, activeIdx, libAuthor, libComment, libParent, notify, refreshLibrary]);

  /** Generate increments list from sweep builder range inputs. */
  const buildIncrements = useCallback((): number[] => {
    const lo = parseFloat(sweepMin);
    const hi = parseFloat(sweepMax);
    const st = parseFloat(sweepStep);
    if (!Number.isFinite(lo) || !Number.isFinite(hi) || !Number.isFinite(st) || st <= 0) return [];
    const out: number[] = [];
    for (let v = lo; v <= hi + 1e-9; v += st) {
      out.push(Math.round(v * 1e6) / 1e6);
    }
    return out;
  }, [sweepMin, sweepMax, sweepStep]);

  const handleSaveSweep = useCallback(async () => {
    if (!sweepName.trim() || !sweepVariable.trim()) {
      notify("warning", "Name and variable required");
      return;
    }
    const increments = buildIncrements();
    if (increments.length === 0) {
      notify("warning", "Invalid range — check min/max/step");
      return;
    }
    const newSweep: SweepScenarioSpec = {
      name: sweepName.trim(),
      variable: sweepVariable.trim(),
      direction: sweepDirection,
      min_val: parseFloat(sweepMin),
      max_val: parseFloat(sweepMax),
      unit: sweepUnit,
      increments,
    };
    const updated = definedSweeps.some((s) => s.name === newSweep.name)
      ? definedSweeps.map((s) => (s.name === newSweep.name ? newSweep : s))
      : [...definedSweeps, newSweep];
    try {
      await updateProjectScenarios({ scenarios: updated, joint_scenarios: definedJoints });
      setDefinedSweeps(updated);
      setBuilderMode(null);
      setSweepName("");
      setShowRerunBanner(true);
      notify("success", `Sweep "${newSweep.name}" saved to project.yml`);
    } catch (err) {
      notify("error", err instanceof Error ? err.message : String(err));
    }
  }, [sweepName, sweepVariable, sweepDirection, sweepMin, sweepMax, sweepStep, sweepUnit, buildIncrements, definedSweeps, definedJoints, notify]);

  const handleSaveJoint = useCallback(async () => {
    if (!jointName.trim()) { notify("warning", "Name required"); return; }
    const validRows = jointRows.filter((r) => r.variable.trim());
    if (validRows.length === 0) { notify("warning", "Add at least one variable"); return; }
    const newJoint: JointScenarioSpec = {
      name: jointName.trim(),
      auto_propagate_dag: jointPropagate,
      interventions: validRows,
    };
    const updated = definedJoints.some((j) => j.name === newJoint.name)
      ? definedJoints.map((j) => (j.name === newJoint.name ? newJoint : j))
      : [...definedJoints, newJoint];
    try {
      await updateProjectScenarios({ scenarios: definedSweeps, joint_scenarios: updated });
      setDefinedJoints(updated);
      setBuilderMode(null);
      setJointName("");
      setJointRows([{ variable: "", direction: "increase", increment: 10 }]);
      setShowRerunBanner(true);
      notify("success", `Joint scenario "${newJoint.name}" saved to project.yml`);
    } catch (err) {
      notify("error", err instanceof Error ? err.message : String(err));
    }
  }, [jointName, jointPropagate, jointRows, definedSweeps, definedJoints, notify]);

  const handleDeleteSweep = useCallback(async (name: string) => {
    try {
      await deleteProjectScenario(name, "sweep");
      setDefinedSweeps((prev) => prev.filter((s) => s.name !== name));
      setShowRerunBanner(true);
      notify("success", `Removed "${name}"`);
    } catch (err) {
      notify("error", err instanceof Error ? err.message : String(err));
    }
  }, [notify]);

  const handleDeleteJoint = useCallback(async (name: string) => {
    try {
      await deleteProjectScenario(name, "joint");
      setDefinedJoints((prev) => prev.filter((j) => j.name !== name));
      setShowRerunBanner(true);
      notify("success", `Removed "${name}"`);
    } catch (err) {
      notify("error", err instanceof Error ? err.message : String(err));
    }
  }, [notify]);

  return (
    <div>
      <SectionHeader
        kicker="08 · analysis"
        label="Scenarios"
        right={
          <div style={{ display: "flex", gap: 6 }}>
            <Btn small onClick={() => setBuilderMode("sweep")}>+ Sweep</Btn>
            <Btn small onClick={() => setBuilderMode("joint")}>+ Joint</Btn>
          </div>
        }
      />

      {/* Tab bar — Configure / Library */}
      <div
        role="tablist"
        style={{
          display: "flex",
          gap: 4,
          borderBottom: "1px solid var(--line)",
          marginBottom: 12,
        }}
      >
        {(
          [
            ["configure", "Configure"],
            ["library", "Library"],
            ["timeline", "Timeline"],
          ] as const
        ).map(([key, label]) => (
          <button
            key={key}
            role="tab"
            aria-selected={tab === key}
            onClick={() => setTab(key)}
            style={{
              padding: "6px 14px",
              fontSize: 12,
              border: 0,
              borderBottom:
                tab === key ? "2px solid var(--crimson, #e73c25)" : "2px solid transparent",
              background: "transparent",
              cursor: "pointer",
              color: tab === key ? "var(--ink)" : "var(--muted)",
              fontWeight: tab === key ? 700 : 500,
              marginBottom: -1,
            }}
          >
            {label}
          </button>
        ))}
      </div>

      <StatGrid>
        <Stat label="Defined sweeps" value={String(definedSweeps.length)} tint="var(--ink)" />
        <Stat label="Joint scenarios" value={String(definedJoints.length)} tint="var(--amber)" />
        <Stat label="Computed" value={String(scenarios.filter((s) => s.status === "computed").length)} tint="var(--crimson)" />
        <Stat label="Best Δ" value={scenarios.length ? `${Math.min(...scenarios.map((s) => s.delta)).toFixed(1)}` : "—"} tint="var(--purple)" />
      </StatGrid>

      {/* Re-run banner */}
      {showRerunBanner && (
        <div style={{
          background: "rgba(91,58,140,.08)",
          border: "1px solid var(--purple)",
          borderRadius: 6,
          padding: "10px 14px",
          marginBottom: 12,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 10,
        }}>
          <span style={{ fontSize: 12 }}>
            ✓ Scenarios saved to <span className="mono">project.yml</span>. Re-run Stage 4 to compute updated results.
          </span>
          <div style={{ display: "flex", gap: 6 }}>
            <Btn small primary onClick={() => { setShowRerunBanner(false); setTab("timeline"); }}>Re-run</Btn>
            <Btn small onClick={() => setShowRerunBanner(false)}>Dismiss</Btn>
          </div>
        </div>
      )}

      {/* ------------------------------------------------------------ */}
      {/* Configure tab — Defined (project.yml) + Computed (pipeline)  */}
      {/* ------------------------------------------------------------ */}
      {tab === "configure" && (
      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>

        {/* Builder panel (shown when + Sweep or + Joint clicked) */}
        {builderMode === "sweep" && (
          <Card title="New sweep scenario" subtitle="single variable · range-based · written to project.yml">
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
              <label style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                <span style={{ fontSize: 11, fontWeight: 600 }}>Scenario name</span>
                <input value={sweepName} onChange={(e) => setSweepName(e.target.value)}
                  placeholder="e.g. Canopy Increase"
                  style={{ border: "1px solid var(--line)", borderRadius: 4, padding: "5px 8px", fontSize: 12, fontFamily: "inherit" }} />
              </label>
              <label style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                <span style={{ fontSize: 11, fontWeight: 600 }}>Variable</span>
                {sliders.length > 0 ? (
                  <select value={sweepVariable} onChange={(e) => setSweepVariable(e.target.value)}
                    style={{ border: "1px solid var(--line)", borderRadius: 4, padding: "5px 8px", fontSize: 12, fontFamily: "inherit" }}>
                    <option value="">— select —</option>
                    {sliders.map((s) => <option key={s.variable} value={s.variable}>{s.variable}</option>)}
                  </select>
                ) : (
                  <input value={sweepVariable} onChange={(e) => setSweepVariable(e.target.value)}
                    placeholder="e.g. Pct_Canopy"
                    style={{ border: "1px solid var(--line)", borderRadius: 4, padding: "5px 8px", fontSize: 12, fontFamily: "inherit" }} />
                )}
              </label>
              <label style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                <span style={{ fontSize: 11, fontWeight: 600 }}>Direction</span>
                <select value={sweepDirection} onChange={(e) => setSweepDirection(e.target.value as "increase" | "decrease")}
                  style={{ border: "1px solid var(--line)", borderRadius: 4, padding: "5px 8px", fontSize: 12, fontFamily: "inherit" }}>
                  <option value="increase">increase</option>
                  <option value="decrease">decrease</option>
                </select>
              </label>
              <label style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                <span style={{ fontSize: 11, fontWeight: 600 }}>Unit (optional)</span>
                <input value={sweepUnit} onChange={(e) => setSweepUnit(e.target.value)}
                  placeholder="e.g. percentage points"
                  style={{ border: "1px solid var(--line)", borderRadius: 4, padding: "5px 8px", fontSize: 12, fontFamily: "inherit" }} />
              </label>
              <label style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                <span style={{ fontSize: 11, fontWeight: 600 }}>Range min</span>
                <input type="number" value={sweepMin} onChange={(e) => setSweepMin(e.target.value)} step="any"
                  style={{ border: "1px solid var(--line)", borderRadius: 4, padding: "5px 8px", fontSize: 12, fontFamily: "'JetBrains Mono', monospace" }} />
              </label>
              <label style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                <span style={{ fontSize: 11, fontWeight: 600 }}>Range max</span>
                <input type="number" value={sweepMax} onChange={(e) => setSweepMax(e.target.value)} step="any"
                  style={{ border: "1px solid var(--line)", borderRadius: 4, padding: "5px 8px", fontSize: 12, fontFamily: "'JetBrains Mono', monospace" }} />
              </label>
              <label style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                <span style={{ fontSize: 11, fontWeight: 600 }}>Step</span>
                <input type="number" value={sweepStep} onChange={(e) => setSweepStep(e.target.value)} step="any" min="0.001"
                  style={{ border: "1px solid var(--line)", borderRadius: 4, padding: "5px 8px", fontSize: 12, fontFamily: "'JetBrains Mono', monospace" }} />
              </label>
              <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                <span style={{ fontSize: 11, fontWeight: 600 }}>Preview</span>
                <span className="mono" style={{ fontSize: 10, color: "var(--muted)", paddingTop: 6 }}>
                  {buildIncrements().slice(0, 8).join(", ")}{buildIncrements().length > 8 ? ` … (${buildIncrements().length} total)` : ""}
                </span>
              </div>
            </div>
            <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
              <Btn small primary onClick={handleSaveSweep}>Save to project.yml</Btn>
              <Btn small onClick={() => setBuilderMode(null)}>Cancel</Btn>
            </div>
          </Card>
        )}

        {builderMode === "joint" && (
          <Card title="New joint scenario" subtitle="multi-variable · fixed increment per variable · written to project.yml">
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 12 }}>
              <label style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                <span style={{ fontSize: 11, fontWeight: 600 }}>Scenario name</span>
                <input value={jointName} onChange={(e) => setJointName(e.target.value)}
                  placeholder="e.g. Green Infrastructure Package"
                  style={{ border: "1px solid var(--line)", borderRadius: 4, padding: "5px 8px", fontSize: 12, fontFamily: "inherit" }} />
              </label>
              <label style={{ display: "flex", flexDirection: "column", gap: 3, justifyContent: "flex-end" }}>
                <span style={{ fontSize: 11, fontWeight: 600 }}>Auto-propagate DAG</span>
                <label style={{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer" }}>
                  <input type="checkbox" checked={jointPropagate} onChange={(e) => setJointPropagate(e.target.checked)} />
                  <span style={{ fontSize: 11 }}>Propagate through causal DAG</span>
                </label>
              </label>
            </div>
            {jointRows.map((row, i) => (
              <div key={i} style={{ display: "grid", gridTemplateColumns: "2fr 1fr 1fr auto", gap: 8, marginBottom: 8, alignItems: "flex-end" }}>
                {sliders.length > 0 ? (
                  <select value={row.variable} onChange={(e) => setJointRows((prev) => prev.map((r, j) => j === i ? { ...r, variable: e.target.value } : r))}
                    style={{ border: "1px solid var(--line)", borderRadius: 4, padding: "5px 8px", fontSize: 12, fontFamily: "inherit" }}>
                    <option value="">— variable —</option>
                    {sliders.map((s) => <option key={s.variable} value={s.variable}>{s.variable}</option>)}
                  </select>
                ) : (
                  <input value={row.variable} onChange={(e) => setJointRows((prev) => prev.map((r, j) => j === i ? { ...r, variable: e.target.value } : r))}
                    placeholder="variable"
                    style={{ border: "1px solid var(--line)", borderRadius: 4, padding: "5px 8px", fontSize: 12, fontFamily: "inherit" }} />
                )}
                <select value={row.direction} onChange={(e) => setJointRows((prev) => prev.map((r, j) => j === i ? { ...r, direction: e.target.value as "increase" | "decrease" } : r))}
                  style={{ border: "1px solid var(--line)", borderRadius: 4, padding: "5px 8px", fontSize: 12, fontFamily: "inherit" }}>
                  <option value="increase">increase</option>
                  <option value="decrease">decrease</option>
                </select>
                <input type="number" value={row.increment} onChange={(e) => setJointRows((prev) => prev.map((r, j) => j === i ? { ...r, increment: parseFloat(e.target.value) || 0 } : r))}
                  placeholder="increment" step="any"
                  style={{ border: "1px solid var(--line)", borderRadius: 4, padding: "5px 8px", fontSize: 12, fontFamily: "'JetBrains Mono', monospace" }} />
                <button onClick={() => setJointRows((prev) => prev.filter((_, j) => j !== i))}
                  style={{ background: "none", border: 0, color: "var(--muted)", cursor: "pointer", fontSize: 16, padding: 0, lineHeight: 1 }}
                  title="Remove row">×</button>
              </div>
            ))}
            <Btn small onClick={() => setJointRows((prev) => [...prev, { variable: "", direction: "increase", increment: 10 }])}>+ Variable</Btn>
            <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
              <Btn small primary onClick={handleSaveJoint}>Save to project.yml</Btn>
              <Btn small onClick={() => setBuilderMode(null)}>Cancel</Btn>
            </div>
          </Card>
        )}

        {/* Defined scenarios from project.yml */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
          <Card title="Defined sweeps" subtitle="single-variable · from project.yml · re-run to compute">
            {definedSweeps.length === 0 ? (
              <div style={{ fontSize: 11, color: "var(--muted)", padding: "8px 0" }}>
                No sweeps defined. Click <strong>+ Sweep</strong> to create one.
              </div>
            ) : definedSweeps.map((sc) => (
              <div key={sc.name} style={{
                display: "grid",
                gridTemplateColumns: "1fr auto auto",
                alignItems: "center",
                gap: 8,
                padding: "9px 0",
                borderTop: "1px dashed var(--line)",
              }}>
                <div>
                  <div style={{ fontSize: 12.5, fontWeight: 600 }}>{sc.name}</div>
                  <div className="mono" style={{ fontSize: 10, color: "var(--muted)", marginTop: 2 }}>
                    {sc.direction} {sc.variable} · {sc.increments.length} increment{sc.increments.length !== 1 ? "s" : ""}{sc.unit ? ` (${sc.unit})` : ""}
                  </div>
                  <div className="mono" style={{ fontSize: 9.5, color: "var(--muted)" }}>
                    [{sc.increments.slice(0, 5).join(", ")}{sc.increments.length > 5 ? "…" : ""}]
                  </div>
                </div>
                <Tag color="var(--ink)">defined</Tag>
                <button onClick={() => handleDeleteSweep(sc.name)}
                  style={{ background: "none", border: 0, color: "var(--muted)", cursor: "pointer", fontSize: 15, padding: 0, lineHeight: 1 }}
                  title="Delete">×</button>
              </div>
            ))}
          </Card>

          <Card title="Defined joint scenarios" subtitle="multi-variable · from project.yml · re-run to compute">
            {definedJoints.length === 0 ? (
              <div style={{ fontSize: 11, color: "var(--muted)", padding: "8px 0" }}>
                No joint scenarios. Click <strong>+ Joint</strong> to create one.
              </div>
            ) : definedJoints.map((sc) => (
              <div key={sc.name} style={{
                display: "grid",
                gridTemplateColumns: "1fr auto auto",
                alignItems: "center",
                gap: 8,
                padding: "9px 0",
                borderTop: "1px dashed var(--line)",
              }}>
                <div>
                  <div style={{ fontSize: 12.5, fontWeight: 600 }}>{sc.name}</div>
                  <div className="mono" style={{ fontSize: 10, color: "var(--muted)", marginTop: 2 }}>
                    {sc.interventions.map((v) => `${v.direction === "decrease" ? "−" : "+"}${v.increment} ${v.variable}`).join(", ")}
                  </div>
                </div>
                <Tag color="var(--amber)">{sc.auto_propagate_dag ? "DAG" : "no-DAG"}</Tag>
                <button onClick={() => handleDeleteJoint(sc.name)}
                  style={{ background: "none", border: 0, color: "var(--muted)", cursor: "pointer", fontSize: 15, padding: 0, lineHeight: 1 }}
                  title="Delete">×</button>
              </div>
            ))}
          </Card>
        </div>

        {/* Computed results from pipeline */}
        <Card title="Computed results" subtitle="from last pipeline run · click to inspect">
          {scenarios.filter((s) => s.status === "computed").length === 0 ? (
            <div style={{ fontSize: 11, color: "var(--muted)", padding: "8px 0" }}>
              No computed results yet. Define scenarios above, then re-run the pipeline.
            </div>
          ) : scenarios.filter((s) => s.status === "computed").map((sc, i) => (
            <div
              key={sc.id}
              onClick={() => setActiveIdx(scenarios.indexOf(sc))}
              style={{
                display: "grid",
                gridTemplateColumns: "1fr auto auto",
                alignItems: "center",
                gap: 10,
                padding: "10px 8px",
                borderTop: i > 0 ? "1px dashed var(--line)" : "none",
                cursor: "pointer",
                background: activeIdx === scenarios.indexOf(sc) ? "rgba(231,60,37,0.04)" : "transparent",
                borderRadius: 4,
              }}
            >
              <div>
                <div style={{ fontSize: 12.5, fontWeight: 600 }}>{sc.name}</div>
                <div className="mono" style={{ fontSize: 10, color: "var(--muted)", marginTop: 2 }}>
                  {Object.entries(sc.interventions)
                    .map(([k, v]) => `${k}: ${v > 0 ? "+" : ""}${v}`)
                    .join(", ") || "—"}
                </div>
              </div>
              <span className="mono" style={{
                fontSize: 13, fontWeight: 700,
                color: sc.delta < 0 ? "var(--purple)" : sc.delta > 0 ? "var(--crimson)" : "var(--muted)",
              }}>
                {sc.delta === 0 ? "—" : `${sc.delta > 0 ? "+" : ""}${sc.delta.toFixed(1)}`}
              </span>
              <Tag color="var(--ink)">computed</Tag>
            </div>
          ))}
          {scenarios.filter((s) => s.status === "computed").length > 0 && (
            <Card title="Posterior distribution" subtitle={scenarios[activeIdx]?.name ?? "select a scenario"}>
              <canvas ref={histRef} style={{ width: "100%", height: 220, display: "block" }} />
              <div className="mono" style={{ fontSize: 10, color: "var(--muted)", marginTop: 8, textAlign: "center" }}>
                {scenarios[activeIdx]?.status === "computed"
                  ? `Median Δ = ${scenarios[activeIdx]?.delta.toFixed(2)}`
                  : "Not yet computed"}
              </div>
            </Card>
          )}
        </Card>

      </div>
      )}

      {/* ------------------------------------------------------------ */}
      {/* Library tab — versioned, append-only scenario journal         */}
      {/* ------------------------------------------------------------ */}
      {tab === "library" && (
      <Card
        title="Versioned scenario library"
        subtitle={library ? `${library.count} entr${library.count === 1 ? "y" : "ies"} · append-only journal` : "loading…"}
        actions={
          <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
            <input
              value={libAuthor}
              onChange={(e) => setLibAuthor(e.target.value)}
              placeholder="author"
              style={{ fontSize: 11, padding: "3px 6px", borderRadius: 3, border: "1px solid var(--line)", width: 80, fontFamily: "'JetBrains Mono', monospace" }}
            />
            <input
              value={libComment}
              onChange={(e) => setLibComment(e.target.value)}
              placeholder="comment (optional)"
              style={{ fontSize: 11, padding: "3px 6px", borderRadius: 3, border: "1px solid var(--line)", width: 220 }}
            />
            <Btn small primary onClick={handleSaveActiveToLibrary}>Save active</Btn>
          </div>
        }
      >
        {!library || library.count === 0 ? (
          <div style={{ fontSize: 11, color: "var(--muted)" }}>
            No saved entries. Scenarios are auto-appended here when saved to project.yml.
          </div>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
              <thead>
                <tr style={{ background: "#fdf6e9" }}>
                  {["", "id", "parent", "author", "comment", "name", "created"].map((h) => (
                    <th key={h} style={{ ...thStyle, textAlign: "left" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {library.entries.slice().reverse().map((e) => {
                  const sc = (e.scenario as { name?: string }) || {};
                  const selected = libParent === e.id;
                  return (
                    <tr key={e.id} style={{ borderBottom: "1px dotted var(--line)", background: selected ? "rgba(91,58,140,.06)" : undefined }}>
                      <td style={tdStyle}>
                        <input
                          type="radio"
                          name="lib-parent"
                          checked={selected}
                          onChange={() => setLibParent(e.id)}
                          title="branch from this entry"
                        />
                      </td>
                      <td style={tdStyle} className="mono">{e.id}</td>
                      <td style={tdStyle} className="mono">{e.parent_id || "—"}</td>
                      <td style={tdStyle}>{e.author}</td>
                      <td style={tdStyle}>{e.comment || <span style={{ color: "var(--muted)" }}>—</span>}</td>
                      <td style={tdStyle}>{sc.name || "—"}</td>
                      <td style={tdStyle} className="mono">{e.created_utc.slice(0, 19).replace("T", " ")}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            {libParent && (
              <div style={{ fontSize: 10.5, color: "var(--muted)", marginTop: 6 }}>
                Branching from <span className="mono">{libParent}</span> ·{" "}
                <button onClick={() => setLibParent(null)} style={{ background: "none", border: 0, color: "var(--purple)", textDecoration: "underline", cursor: "pointer", fontSize: 10.5 }}>clear</button>
              </div>
            )}
          </div>
        )}
      </Card>
      )}

      {/* ------------------------------------------------------------ */}
      {/* Timeline tab — multi-step latent rollout (JD-4)               */}
      {/* ------------------------------------------------------------ */}
      {tab === "timeline" && (
      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        <Card title="Multi-step rollout" subtitle="chain interventions sequentially through the latent space">
          {/* Step builder */}
          <div style={{ display: "flex", gap: 8, alignItems: "flex-end", marginBottom: 12 }}>
            <label style={{ display: "flex", flexDirection: "column", gap: 3, flex: 2 }}>
              <span style={{ fontSize: 11, fontWeight: 600 }}>Treatment variable</span>
              <input
                type="text"
                value={chainTreatment}
                onChange={(e) => setChainTreatment(e.target.value)}
                placeholder="e.g. Pct_Canopy"
                style={{ border: "1px solid var(--line)", borderRadius: 4, padding: "5px 8px", fontSize: 12, fontFamily: "inherit" }}
              />
            </label>
            <label style={{ display: "flex", flexDirection: "column", gap: 3, flex: 1 }}>
              <span style={{ fontSize: 11, fontWeight: 600 }}>Δ value</span>
              <input
                type="number"
                value={chainDeltaX}
                onChange={(e) => setChainDeltaX(e.target.value)}
                step="any"
                style={{ border: "1px solid var(--line)", borderRadius: 4, padding: "5px 8px", fontSize: 12, fontFamily: "'JetBrains Mono', monospace" }}
              />
            </label>
            <Btn
              small
              onClick={() => {
                if (!chainTreatment.trim()) return;
                const dx = parseFloat(chainDeltaX);
                if (isNaN(dx)) return;
                setChainActions((prev) => [...prev, { treatment: chainTreatment.trim(), delta_x: dx, delta_t: 1.0 }]);
                setChainTreatment("");
                setChainDeltaX("0");
              }}
            >
              + Add step
            </Btn>
          </div>

          {/* Step list */}
          {chainActions.length === 0 ? (
            <div style={{ fontSize: 11, color: "var(--muted)", marginBottom: 12 }}>
              No steps yet. Add at least one intervention above.
            </div>
          ) : (
            <div style={{ marginBottom: 12 }}>
              {chainActions.map((a, i) => (
                <div
                  key={i}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 10,
                    padding: "7px 0",
                    borderTop: i > 0 ? "1px dashed var(--line)" : undefined,
                  }}
                >
                  <span
                    className="mono"
                    style={{ width: 22, height: 22, background: "var(--crimson)", color: "#fff", borderRadius: "50%", display: "inline-flex", alignItems: "center", justifyContent: "center", fontSize: 11, fontWeight: 700, flexShrink: 0 }}
                  >
                    {i + 1}
                  </span>
                  <span style={{ flex: 1, fontSize: 12, fontWeight: 600 }}>{a.treatment}</span>
                  <span className="mono" style={{ fontSize: 12, color: a.delta_x >= 0 ? "var(--crimson)" : "var(--purple)" }}>
                    {a.delta_x >= 0 ? "+" : ""}{a.delta_x}
                  </span>
                  <button
                    onClick={() => setChainActions((prev) => prev.filter((_, j) => j !== i))}
                    style={{ background: "none", border: 0, color: "var(--muted)", cursor: "pointer", fontSize: 14, padding: 0, lineHeight: 1 }}
                    title="Remove step"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          )}

          <div style={{ display: "flex", gap: 8 }}>
            <Btn
              primary
              small
              disabled={chainActions.length === 0 || chainRunning}
              onClick={async () => {
                setChainRunning(true);
                setChainResult(null);
                setChainCumulative(null);
                try {
                  const res = await runScenarioChain(chainActions);
                  setChainResult(res.steps);
                  setChainCumulative(res.cumulative_mean_delta);
                } catch (err) {
                  notify("error", err instanceof Error ? err.message : "Chain rollout failed");
                } finally {
                  setChainRunning(false);
                }
              }}
            >
              {chainRunning ? "Running…" : "Run chain"}
            </Btn>
            {chainActions.length > 0 && (
              <Btn
                small
                onClick={() => { setChainActions([]); setChainResult(null); setChainCumulative(null); }}
              >
                Clear all
              </Btn>
            )}
          </div>
        </Card>

        {chainResult && chainResult.length > 0 && (
          <Card
            title="Rollout results"
            subtitle={`${chainResult.length} steps · cumulative Δ = ${chainCumulative != null ? (chainCumulative >= 0 ? "+" : "") + chainCumulative.toFixed(3) : "—"}`}
          >
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr>
                  {["Step", "Treatment", "Δ applied", "Mean Δ response", "90% range"].map((h) => (
                    <th key={h} style={thStyle}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {chainResult.map((s) => (
                  <tr key={s.step} style={{ borderBottom: "1px dotted var(--line)" }}>
                    <td style={tdStyle} className="mono">{s.step}</td>
                    <td style={tdStyle}>{s.treatment}</td>
                    <td style={tdStyle} className="mono">{s.delta_x >= 0 ? "+" : ""}{s.delta_x}</td>
                    <td
                      style={{ ...tdStyle, fontWeight: 700, color: s.mean_delta < 0 ? "var(--purple)" : s.mean_delta > 0 ? "var(--crimson)" : "var(--muted)" }}
                      className="mono"
                    >
                      {s.mean_delta >= 0 ? "+" : ""}{s.mean_delta.toFixed(3)}
                    </td>
                    <td style={{ color: "var(--muted)", ...tdStyle }} className="mono">
                      [{s.p5_delta.toFixed(3)}, {s.p95_delta.toFixed(3)}]
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        )}
      </div>
      )}
    </div>
  );
}
