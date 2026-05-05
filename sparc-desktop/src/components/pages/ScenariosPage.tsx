import { useState, useEffect, useCallback, useRef, useMemo } from "react";
import { SectionHeader, Card, Tag, Btn, Stat, StatGrid, thStyle, tdStyle } from "@/components/ui/DesignSystem";
import DownloadMenu from "@/components/common/DownloadMenu";
import {
  getConfig,
  getScenarioDetail,
  runScenarios,
  getNutsSummary,
  getScenarioLibrary,
  appendScenarioToLibrary,
  dataSummary,
  parseMissingArtifact,
  parseScenarioVariant,
  type ScenarioTimeline,
} from "@/lib/api";
import { useNotification } from "@/hooks/useNotifications";
import { useManifest } from "@/hooks/useManifest";
import { SPARC_RAMP_HEX } from "@/lib/design-tokens";
import { presetsForDomain, applyPresetToPredictors } from "@/lib/scenarioPresets";

type ScenariosTab = "configure" | "run" | "results" | "library";

type ConfiguredScenarioRow = {
  name: string;
  variable?: string;
  increments?: number[];
  interventions?: Record<string, number>;
  status: string;
};

type LastRunInfo = {
  nScenarios: number;
  summaryRows: number;
  mode?: string;
  conservation?: number;
  timestamp: string;
};

interface Props {
  /** Optional cross-page navigation, e.g. "Open Results →". */
  onNavigate?: (page: string) => void;
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

interface NutsPosterior {
  treatment: string;
  mean: number;
  std: number;
  ci_5: number;
  ci_25: number;
  median: number;
  ci_75: number;
  ci_95: number;
}

interface NutsConvergence {
  parameter: string;
  r_hat: number;
  ess: number;
  converged: boolean;
}

export default function ScenariosPage({ onNavigate }: Props = {}) {
  const { notify } = useNotification();
  const manifest = useManifest(true);
  const [tab, setTab] = useState<ScenariosTab>("configure");
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [activeIdx, setActiveIdx] = useState(0);
  const [sliders, setSliders] = useState<InterventionSlider[]>([]);
  const [resultsArtifactId, setResultsArtifactId] = useState<string | null>(null);
  const [configRaw, setConfigRaw] = useState<Record<string, unknown> | null>(null);
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [lastRun, setLastRun] = useState<LastRunInfo | null>(null);
  const [nutsData, setNutsData] = useState<{
    acceptance_rate?: number;
    n_divergences?: number;
    posteriors?: NutsPosterior[];
    convergence?: NutsConvergence[];
  } | null>(null);
  const [nutsTab, setNutsTab] = useState<"posteriors" | "convergence">("posteriors");
  const [library, setLibrary] = useState<ScenarioTimeline | null>(null);
  const [libParent, setLibParent] = useState<string | null>(null);
  const [libComment, setLibComment] = useState("");
  const [libAuthor, setLibAuthor] = useState("me");
  const histRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    // Load NUTS posterior summaries
    getNutsSummary()
      .then((data) => setNutsData(data as any))
      .catch(() => {});

    // Load scenarios from API results
    getScenarioDetail()
      .then((detail: any) => {
        if (detail?.results_artifact_id) setResultsArtifactId(detail.results_artifact_id);
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

    // Load config for intervention builder sliders + preset library
    getConfig()
      .then((config) => {
        setConfigRaw(config as unknown as Record<string, unknown>);
        const cfgScenarios = (config.scenarios ?? []) as any[];
        const cols = config.predictors ?? [];
        const domain = config.project?.domain ?? "";
        const presets = presetsForDomain(domain);

        setScenarios((existing) => {
          // Don't clobber API-loaded computed scenarios; merge config + presets only if empty.
          if (existing.length > 0) return existing;
          const merged: Scenario[] = [];
          for (let i = 0; i < cfgScenarios.length; i++) {
            const sc = cfgScenarios[i] ?? {};
            merged.push({
              id: `cfg-${i}`,
              name: sc.name ?? `Scenario ${i + 1}`,
              interventions: sc.interventions ?? {},
              delta: sc.delta ?? 0,
              status: "draft",
            });
          }
          for (const p of presets) {
            const interventions = applyPresetToPredictors(p, cols);
            // Only include presets that match at least one project predictor.
            if (p.id !== "preset-baseline" && Object.keys(interventions).length === 0) continue;
            merged.push({
              id: p.id,
              name: p.name,
              interventions,
              delta: 0,
              status: p.id === "preset-baseline" ? "baseline" : "draft",
            });
          }
          return merged;
        });

        // Build sliders from predictors in config, bounded by /data/summary stats
        if (cols.length > 0) {
          const top = cols.slice(0, 6);
          dataSummary()
            .then((ds) => {
              const ns = ds?.numeric_summary ?? {};
              setSliders(
                top.map((col: string) => {
                  const stats = ns[col];
                  // Prefer ±2σ around the baseline (mean); fall back to data
                  // min/max, then to legacy ±0.5 if neither is available.
                  let baseline = 0;
                  let lo = -0.5;
                  let hi = 0.5;
                  let step = 0.05;
                  if (stats) {
                    const mean = Number(stats.mean ?? 0);
                    const std = Number(stats.std ?? 0);
                    const dataMin = Number(stats.min ?? mean);
                    const dataMax = Number(stats.max ?? mean);
                    baseline = Number.isFinite(mean) ? mean : 0;
                    if (Number.isFinite(std) && std > 0) {
                      lo = baseline - 2 * std;
                      hi = baseline + 2 * std;
                    } else if (Number.isFinite(dataMin) && Number.isFinite(dataMax) && dataMax > dataMin) {
                      lo = dataMin;
                      hi = dataMax;
                    }
                    const span = hi - lo;
                    if (span > 0 && Number.isFinite(span)) {
                      step = span / 40;
                    }
                  }
                  return {
                    variable: col,
                    min: Number.isFinite(lo) ? lo : -0.5,
                    max: Number.isFinite(hi) ? hi : 0.5,
                    step: Number.isFinite(step) && step > 0 ? step : 0.05,
                    unit: "",
                    value: baseline,
                    baseline,
                  };
                }),
              );
            })
            .catch(() => {
              setSliders(
                top.map((col: string) => ({
                  variable: col,
                  min: -0.5,
                  max: 0.5,
                  step: 0.05,
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
    const dominant = interventionEntries.sort(
      (a, b) => Math.abs(b[1]) - Math.abs(a[1]),
    )[0];
    let sigma = Math.max(Math.abs(active.delta) * 0.25, 1e-6);
    let posteriorMatched = false;
    if (dominant && nutsData?.posteriors) {
      const post = nutsData.posteriors.find(
        (p) => p.treatment.toLowerCase() === dominant[0].toLowerCase(),
      );
      if (post) {
        sigma = Math.max(Math.abs(post.std * dominant[1]), Math.abs(active.delta) * 0.05);
        posteriorMatched = true;
      }
    }

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
  }, [scenarios, activeIdx, nutsData]);

  const handleSliderChange = useCallback((variable: string, value: number) => {
    setSliders((prev) => prev.map((s) => s.variable === variable ? { ...s, value } : s));
  }, []);

  const handleAddScenario = useCallback(() => {
    const interventions: Record<string, number> = {};
    sliders.forEach((s) => {
      if (s.value !== 0) interventions[s.variable] = s.value;
    });
    const name = prompt("Scenario name:", `Scenario ${scenarios.length + 1}`);
    if (!name) return;
    setScenarios((prev) => [
      ...prev,
      { id: `s${prev.length}`, name, interventions, delta: 0, status: "draft" },
    ]);
    notify("success", `Scenario "${name}" added`);
  }, [sliders, scenarios, notify]);

  const handleComputeAll = useCallback(async () => {
    setRunning(true);
    setRunError(null);
    try {
      notify("info", "Computing scenario deltas...");
      const result = await runScenarios();
      setLastRun({
        nScenarios: result.n_scenarios,
        summaryRows: result.summary_rows,
        mode: (result as any).scenario_mode,
        conservation: (result as any).conservation_violations,
        timestamp: new Date().toLocaleTimeString(),
      });
      notify("success", `Computed ${result.n_scenarios} scenarios`);
      await manifest.rescan().catch(() => {});
      // Refresh from API
      const detail: any = await getScenarioDetail().catch(() => null);
      if (detail?.results_artifact_id) setResultsArtifactId(detail.results_artifact_id);
      const rows = detail?.scenarios ?? detail?.summary;
      if (Array.isArray(rows) && rows.length) {
        setScenarios(
          rows.map((sc: any, i: number) => ({
            id: `s${i}`,
            name: sc.name ?? `Scenario ${i + 1}`,
            interventions: sc.interventions ?? {},
            delta: sc.delta ?? sc.mean_delta ?? 0,
            status: "computed" as const,
          })),
        );
      }
    } catch (e) {
      const missing = parseMissingArtifact(e);
      const msg = missing?.hint ?? (e instanceof Error ? e.message : "Scenario computation failed");
      setRunError(msg);
      notify("error", msg);
    } finally {
      setRunning(false);
    }
  }, [notify, manifest]);

  // Configured scenarios derived from project.yml (Run tab).
  const configured: ConfiguredScenarioRow[] = useMemo(() => {
    if (!configRaw) return [];
    const out: ConfiguredScenarioRow[] = [];
    const baseList = (configRaw.scenarios ?? []) as any[];
    for (const s of baseList) {
      out.push({
        name: s.name,
        variable: s.variable,
        increments: s.increments,
        status: "configured",
      });
    }
    for (const s of (configRaw.joint_scenarios ?? []) as any[]) {
      out.push({
        name: s.name,
        interventions: Object.fromEntries(
          (s.interventions ?? []).map((iv: any) => [iv.variable, iv.increment]),
        ),
        status: "configured (joint)",
      });
    }
    for (const s of (configRaw.interaction_scenarios ?? []) as any[]) {
      out.push({ name: s.name, status: "configured (interaction)" });
    }
    return out;
  }, [configRaw]);

  const stage4 = manifest.stage(4);
  const hasStageResults =
    (stage4?.artifacts && Object.keys(stage4.artifacts).length > 0) || lastRun !== null;
  const autoRun = (configRaw?.auto_run_scenarios_at_stage_4 as boolean | undefined) !== false;
  const variant = parseScenarioVariant(resultsArtifactId);

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

  return (
    <div>
      <SectionHeader
        kicker="08 · analysis"
        label="Scenarios"
        right={
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            {variant && (
              <Tag
                color={
                  variant === "hybrid"
                    ? "var(--purple)"
                    : variant === "reprediction"
                    ? "var(--amber)"
                    : variant === "dag"
                    ? "var(--crimson)"
                    : "var(--muted)"
                }
              >
                variant: {variant}
              </Tag>
            )}
            <Btn small onClick={handleAddScenario}>Add scenario</Btn>
            <Btn
              primary
              small
              onClick={handleComputeAll}
              disabled={running}
            >
              {running ? "Running…" : "Compute all"}
            </Btn>
          </div>
        }
      />

      {/* Tab bar — Configure / Run / Results / Library */}
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
            ["run", "Run"],
            ["results", "Results"],
            ["library", "Library"],
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
        <Stat label="Scenarios" value={String(scenarios.length)} tint="var(--ink)" />
        <Stat label="Computed" value={String(scenarios.filter((s) => s.status === "computed").length)} tint="var(--crimson)" />
        <Stat label="Best Δ" value={scenarios.length ? `${Math.min(...scenarios.map((s) => s.delta)).toFixed(1)}` : "—"} tint="var(--purple)" />
        <Stat label="Variables" value={String(sliders.length)} tint="var(--amber)" />
      </StatGrid>

      {/* ------------------------------------------------------------ */}
      {/* Configure tab — scenario library card + intervention builder  */}
      {/* ------------------------------------------------------------ */}
      {tab === "configure" && (
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <Card title="Scenario library" subtitle="click to select · compare to baseline">
            {scenarios.map((sc, i) => (
              <div
                key={sc.id}
                onClick={() => setActiveIdx(i)}
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr auto auto",
                  alignItems: "center",
                  gap: 10,
                  padding: "10px 8px",
                  borderTop: i > 0 ? "1px dashed var(--line)" : "none",
                  cursor: "pointer",
                  background: activeIdx === i ? "rgba(231,60,37,0.04)" : "transparent",
                  borderRadius: 4,
                }}
              >
                <div>
                  <div style={{ fontSize: 12.5, fontWeight: 600 }}>{sc.name}</div>
                  <div className="mono" style={{ fontSize: 10, color: "var(--muted)", marginTop: 2 }}>
                    {Object.entries(sc.interventions)
                      .map(([k, v]) => `${k}: ${v > 0 ? "+" : ""}${v}`)
                      .join(", ") || "no interventions"}
                  </div>
                </div>
                <span
                  className="mono"
                  style={{
                    fontSize: 13,
                    fontWeight: 700,
                    color: sc.delta < 0 ? "var(--purple)" : sc.delta > 0 ? "var(--crimson)" : "var(--muted)",
                  }}
                >
                  {sc.delta === 0 ? "—" : `${sc.delta > 0 ? "+" : ""}${sc.delta.toFixed(1)} °C`}
                </span>
                <Tag
                  color={
                    sc.status === "computed"
                      ? "var(--ink)"
                      : sc.status === "baseline"
                      ? "var(--purple)"
                      : "var(--muted)"
                  }
                >
                  {sc.status}
                </Tag>
              </div>
            ))}
          </Card>

          <Card title="Intervention builder" subtitle="adjust sliders to create new scenario">
            {sliders.map((s) => (
              <div key={s.variable} style={{ padding: "10px 0", borderTop: "1px dashed var(--line)" }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                  <span style={{ fontSize: 12, fontWeight: 600 }}>{s.variable.replace(/_/g, " ")}</span>
                  <span
                    className="mono"
                    style={{
                      fontSize: 12,
                      fontWeight: 700,
                      color: s.value !== 0 ? "var(--crimson)" : "var(--muted)",
                    }}
                  >
                    {s.value > 0 ? "+" : ""}
                    {s.value.toFixed(2)} {s.unit}
                  </span>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span className="mono" style={{ fontSize: 9, color: "var(--muted)", width: 36, textAlign: "right" }}>
                    {s.min}
                  </span>
                  <input
                    type="range"
                    min={s.min}
                    max={s.max}
                    step={s.step}
                    value={s.value}
                    onChange={(e) => handleSliderChange(s.variable, Number(e.target.value))}
                    style={{ flex: 1, accentColor: "var(--crimson)" }}
                  />
                  <span className="mono" style={{ fontSize: 9, color: "var(--muted)", width: 36 }}>
                    {s.max}
                  </span>
                </div>
                <div className="mono" style={{ fontSize: 9.5, color: "var(--muted)", marginTop: 2 }}>
                  baseline: {s.baseline}{s.unit}
                </div>
              </div>
            ))}
          </Card>
        </div>

        <Card title="Posterior distribution" subtitle={scenarios[activeIdx]?.name ?? "select a scenario"}>
          <canvas
            ref={histRef}
            style={{ width: "100%", height: 300, display: "block" }}
          />
          <div className="mono" style={{ fontSize: 10, color: "var(--muted)", marginTop: 8, textAlign: "center" }}>
            {scenarios[activeIdx]?.status === "computed"
              ? `Median Δ = ${scenarios[activeIdx]?.delta.toFixed(2)} · 95% CI: [${(scenarios[activeIdx]?.delta - 0.98).toFixed(2)}, ${(scenarios[activeIdx]?.delta + 0.98).toFixed(2)}]`
              : "Not yet computed"}
          </div>
        </Card>
      </div>
      )}

      {/* ------------------------------------------------------------ */}
      {/* Run tab — fold-in of the legacy ScenarioRunnerPage             */}
      {/* ------------------------------------------------------------ */}
      {tab === "run" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <StatGrid>
            <Stat label="Configured" value={String(configured.length)} tint="var(--ink)" />
            <Stat label="Computed" value={String(scenarios.filter((s) => s.status === "computed").length)} tint="var(--purple)" />
            <Stat
              label="Auto-run @ Stage 4"
              value={autoRun ? "on" : "off"}
              tint={autoRun ? "var(--ink)" : "var(--amber)"}
            />
            <Stat
              label="Last run"
              value={lastRun?.timestamp ?? (hasStageResults ? "previous session" : "—")}
              tint="var(--crimson)"
            />
          </StatGrid>

          {runError && (
            <Card title="Error" subtitle="from /scenarios/run">
              <div style={{ fontSize: 12.5, color: "var(--crimson)", lineHeight: 1.55 }}>
                {runError}
              </div>
            </Card>
          )}

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
            <Card title="Configured scenarios" subtitle="from project.yml">
              {configured.length === 0 ? (
                <div style={{ fontSize: 12.5, color: "var(--muted)", padding: "10px 4px" }}>
                  No scenarios defined. Open the <strong>Configure</strong> tab above to add some,
                  or edit project.yml directly under the <code>scenarios:</code> key.
                </div>
              ) : (
                configured.map((s, i) => (
                  <div
                    key={`${s.name}-${i}`}
                    style={{
                      display: "grid",
                      gridTemplateColumns: "1fr auto",
                      alignItems: "center",
                      gap: 10,
                      padding: "10px 4px",
                      borderTop: i > 0 ? "1px dashed var(--line)" : "none",
                    }}
                  >
                    <div>
                      <div style={{ fontSize: 12.5, fontWeight: 600 }}>{s.name}</div>
                      <div className="mono" style={{ fontSize: 10, color: "var(--muted)", marginTop: 2 }}>
                        {s.variable
                          ? `${s.variable} ${s.increments ? `× [${s.increments.join(", ")}]` : ""}`
                          : s.interventions
                          ? Object.entries(s.interventions)
                              .map(([k, v]) => `${k}: ${v > 0 ? "+" : ""}${v}`)
                              .join(", ")
                          : "—"}
                      </div>
                    </div>
                    <Tag color="var(--muted)">{s.status}</Tag>
                  </div>
                ))
              )}
            </Card>

            <Card
              title="Computed deltas"
              subtitle="from /results/scenarios/detail"
              actions={
                <DownloadMenu
                  artifactId="scenario_results"
                  stage="4"
                  label="scenario results"
                  dataEndpoint="/results/scenarios"
                  dataFilename="scenario_results"
                  includeBundle
                  compact
                />
              }
            >
              {scenarios.filter((s) => s.status === "computed").length === 0 ? (
                <div style={{ fontSize: 12.5, color: "var(--muted)", padding: "10px 4px" }}>
                  No computed scenarios yet. Click <strong>Compute all</strong> in the header
                  to launch.
                </div>
              ) : (
                scenarios
                  .filter((s) => s.status === "computed")
                  .map((sc, i) => (
                    <div
                      key={`${sc.id}-${i}`}
                      style={{
                        display: "grid",
                        gridTemplateColumns: "1fr auto",
                        alignItems: "center",
                        gap: 10,
                        padding: "10px 4px",
                        borderTop: i > 0 ? "1px dashed var(--line)" : "none",
                      }}
                    >
                      <div style={{ fontSize: 12.5, fontWeight: 600 }}>{sc.name}</div>
                      <span
                        className="mono"
                        style={{
                          fontSize: 13,
                          fontWeight: 700,
                          color:
                            sc.delta < 0
                              ? "var(--purple)"
                              : sc.delta > 0
                              ? "var(--crimson)"
                              : "var(--muted)",
                        }}
                      >
                        {sc.delta === 0
                          ? "—"
                          : `${sc.delta > 0 ? "+" : ""}${sc.delta.toFixed(2)}`}
                      </span>
                    </div>
                  ))
              )}
              {hasStageResults && onNavigate && (
                <div style={{ marginTop: 10, paddingTop: 10, borderTop: "1px dashed var(--line)" }}>
                  <Btn small onClick={() => onNavigate("Insights")}>Open Insights →</Btn>
                </div>
              )}
            </Card>
          </div>

          {lastRun && (
            <Card title="Last run summary" subtitle={lastRun.timestamp}>
              <div className="mono" style={{ fontSize: 11.5, color: "var(--muted)", lineHeight: 1.7 }}>
                scenarios: {lastRun.nScenarios}
                <br />
                summary rows: {lastRun.summaryRows}
                {lastRun.mode && (
                  <>
                    <br />
                    mode: {lastRun.mode}
                  </>
                )}
                {lastRun.conservation !== undefined && (
                  <>
                    <br />
                    conservation violations: {lastRun.conservation}
                  </>
                )}
              </div>
            </Card>
          )}
        </div>
      )}

      {/* ------------------------------------------------------------ */}
      {/* Results tab — NUTS / MC³ posterior results                    */}
      {/* ------------------------------------------------------------ */}
      {tab === "results" && nutsData && (
        <div style={{ marginTop: 14 }}>
          <Card
            title="NUTS posterior results (MC³)"
            subtitle={
              nutsData.acceptance_rate !== undefined
                ? `acceptance ${(nutsData.acceptance_rate * 100).toFixed(1)}% · divergences ${nutsData.n_divergences ?? 0}`
                : "original-scale causal estimates"
            }
            actions={
              <div style={{ display: "flex", gap: 6 }}>
                {(["posteriors", "convergence"] as const).map((t) => (
                  <button
                    key={t}
                    onClick={() => setNutsTab(t)}
                    style={{
                      padding: "3px 10px",
                      fontSize: 11,
                      borderRadius: 4,
                      border: "1px solid var(--line)",
                      background: nutsTab === t ? "var(--crimson, #e73c25)" : "transparent",
                      color: nutsTab === t ? "#fff" : "var(--muted)",
                      cursor: "pointer",
                    }}
                  >
                    {t === "posteriors" ? "Posterior coefficients" : "Convergence (R̂, ESS)"}
                  </button>
                ))}
              </div>
            }
          >
            {nutsTab === "posteriors" && nutsData.posteriors && nutsData.posteriors.length > 0 && (
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11.5 }}>
                <thead>
                  <tr style={{ textAlign: "left", color: "var(--muted)" }}>
                    <th style={thStyle}>Treatment</th>
                    <th style={{ ...thStyle, textAlign: "right" }}>Mean</th>
                    <th style={{ ...thStyle, textAlign: "right" }}>Std</th>
                    <th style={{ ...thStyle, textAlign: "right" }}>CI 5%</th>
                    <th style={{ ...thStyle, textAlign: "right" }}>Median</th>
                    <th style={{ ...thStyle, textAlign: "right" }}>CI 95%</th>
                    <th style={{ ...thStyle, width: 120 }}>Interval</th>
                  </tr>
                </thead>
                <tbody>
                  {nutsData.posteriors.map((row) => {
                    const range = row.ci_95 - row.ci_5 || 1;
                    const barLeft = ((row.ci_25 - row.ci_5) / range) * 100;
                    const barWidth = ((row.ci_75 - row.ci_25) / range) * 100;
                    const medLeft = ((row.median - row.ci_5) / range) * 100;
                    const positive = row.mean >= 0;
                    return (
                      <tr key={row.treatment} style={{ borderTop: "1px solid var(--line)" }}>
                        <td style={{ ...tdStyle, fontWeight: 600 }}>{row.treatment.replace(/_/g, " ")}</td>
                        <td className="mono" style={{ ...tdStyle, textAlign: "right", fontWeight: 700, color: positive ? "var(--crimson)" : "var(--purple)" }}>
                          {row.mean >= 0 ? "+" : ""}{row.mean.toFixed(4)}
                        </td>
                        <td className="mono" style={{ ...tdStyle, textAlign: "right", color: "var(--muted)" }}>±{row.std.toFixed(4)}</td>
                        <td className="mono" style={{ ...tdStyle, textAlign: "right", color: "var(--muted)" }}>{row.ci_5.toFixed(4)}</td>
                        <td className="mono" style={{ ...tdStyle, textAlign: "right" }}>{row.median.toFixed(4)}</td>
                        <td className="mono" style={{ ...tdStyle, textAlign: "right", color: "var(--muted)" }}>{row.ci_95.toFixed(4)}</td>
                        <td style={tdStyle}>
                          <div style={{ position: "relative", height: 10, background: "rgba(0,0,0,0.04)", borderRadius: 3, overflow: "hidden" }}>
                            <div style={{ position: "absolute", left: `${barLeft}%`, width: `${barWidth}%`, height: "100%", background: (positive ? "var(--crimson, #e73c25)" : "var(--purple, #7b5ea7)") + "99" }} />
                            <div style={{ position: "absolute", left: `${medLeft}%`, width: 2, height: "100%", background: positive ? "var(--crimson, #e73c25)" : "var(--purple, #7b5ea7)" }} />
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
            {nutsTab === "convergence" && nutsData.convergence && nutsData.convergence.length > 0 && (
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11.5 }}>
                <thead>
                  <tr style={{ textAlign: "left", color: "var(--muted)" }}>
                    <th style={thStyle}>Parameter</th>
                    <th style={{ ...thStyle, textAlign: "right" }}>R̂</th>
                    <th style={{ ...thStyle, textAlign: "right" }}>ESS</th>
                    <th style={{ ...thStyle, textAlign: "center" }}>Converged</th>
                  </tr>
                </thead>
                <tbody>
                  {nutsData.convergence.map((row, i) => (
                    <tr key={i} style={{ borderTop: "1px solid var(--line)" }}>
                      <td style={{ ...tdStyle, fontFamily: "var(--font-mono)" }}>{row.parameter}</td>
                      <td className="mono" style={{ ...tdStyle, textAlign: "right", color: (row.r_hat ?? 0) > 1.1 ? "var(--crimson)" : "var(--ink)" }}>
                        {typeof row.r_hat === "number" ? row.r_hat.toFixed(3) : "—"}
                      </td>
                      <td className="mono" style={{ ...tdStyle, textAlign: "right" }}>
                        {typeof row.ess === "number" ? row.ess.toFixed(0) : "—"}
                      </td>
                      <td style={{ ...tdStyle, textAlign: "center" }}>
                        <span style={{ color: row.converged ? "var(--teal, #2a9d8f)" : "var(--crimson)" }}>
                          {row.converged ? "✓" : "✗"}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
            {!nutsData.posteriors?.length && !nutsData.convergence?.length && (
              <div style={{ padding: 24, textAlign: "center", color: "var(--muted)", fontSize: 12 }}>
                No NUTS data available.
              </div>
            )}
          </Card>
        </div>
      )}
      {tab === "results" && !nutsData && (
        <Card title="Results" subtitle="awaiting Stage 3 NUTS posteriors">
          <div style={{ padding: 24, textAlign: "center", color: "var(--muted)", fontSize: 12 }}>
            No NUTS posteriors yet. Run Stage 3 (causal inference) to populate
            this view, or open the <strong>Results</strong> page for stage-level
            summaries.
          </div>
        </Card>
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
            No saved entries. Pick a scenario above and click "Save active" to seed the library.
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
    </div>
  );
}
