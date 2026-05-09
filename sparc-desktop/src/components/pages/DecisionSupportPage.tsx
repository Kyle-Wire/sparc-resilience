import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { SectionHeader, Card, Btn, Stat, StatGrid, Tag } from "@/components/ui/DesignSystem";
import { EmptyState } from "@/components/common/EmptyState";
import { SPARC_RAMP_HEX, MAP_HEIGHT_DEFAULT } from "@/lib/design-tokens";
import SpatialMap from "@/components/map/SpatialMap";
import ResizableMapWrapper from "@/components/map/ResizableMapWrapper";
import { ExportBlockButton } from "@/components/common/ExportBlockButton";
import { usePipeline } from "@/hooks/PipelineProvider";
import { useNotification } from "@/hooks/useNotifications";
import {
  getDecisionCandidates,
  optimizeDecisions,
  decisionUncertainty,
  computeEquity,
  getCensusEquity,
  getCateMapVariables,
  getTargeting,
  optimizeBudget,
  dataSummary,
  type InterventionCandidate,
  type OptimizerResponse,
  type RankedIntervention,
  type EquityResponse,
  type CensusEquityResponse,
  type TargetingResponse,
  type UncertaintyRecord,
  type BudgetOptimizeResponse,
} from "@/lib/api";

/**
 * Decision Support — budget & equity optimization.
 *
 * Fine-tunes already-run scenarios to find the optimal locations and
 * intervention mix given a budget and equity constraints:
 *   1. Scenarios      — review candidates loaded from pipeline outputs
 *   2. Budget & Equity — set budget cap, robustness, equity weighting
 *   3. Results         — ranked options + spatial targeting + Pareto
 *
 * Wizard state persists in localStorage under WIZ_KEY so users can
 * step away and return without losing their work.
 */

type Step = 1 | 2 | 3;
type BenefitSource = "cate" | "uniform";
type Solver = "auto" | "greedy" | "greedy_2opt" | "milp";

interface PersistedState {
  budget: string;
  robustness: number;
  minimise: boolean;
  benefitSource: BenefitSource;
  benefitVar: string;
  solver: Solver;
  costColumn: string;
  xMaxColumn: string;
  paretoSweep: boolean;
  nDraws: number;
  layerText: Record<string, string>;
  layerInvert: Record<string, boolean>;
  topK: number;
  activeTargetVar: string;
  candidateEdits: Record<string, Partial<InterventionCandidate>>;
}

const WIZ_KEY = "sparc:decision-support:wizard:v1";

const DEFAULT_STATE: PersistedState = {
  budget: "",
  robustness: 0,
  minimise: false,
  benefitSource: "cate",
  benefitVar: "",
  solver: "auto",
  costColumn: "",
  xMaxColumn: "",
  paretoSweep: true,
  nDraws: 500,
  layerText: { population: "", vulnerability: "" },
  layerInvert: { population: false, vulnerability: false },
  topK: 50,
  activeTargetVar: "",
  candidateEdits: {},
};

function loadPersisted(): PersistedState {
  try {
    const raw = localStorage.getItem(WIZ_KEY);
    if (!raw) return { ...DEFAULT_STATE };
    const parsed = { ...DEFAULT_STATE, ...JSON.parse(raw) } as PersistedState;
    // v4 migration: the legacy "local_coef" benefit source (MGWR/GWR
    // correlation-based coefficients) is gone — collapse onto "cate".
    if ((parsed.benefitSource as string) === "local_coef") {
      parsed.benefitSource = "cate";
    }
    return parsed;
  } catch {
    return { ...DEFAULT_STATE };
  }
}

export default function DecisionSupportPage() {
  const { runEndedAt } = usePipeline();
  const { notify } = useNotification();

  const [step, setStep] = useState<Step>(1);
  const [wiz, setWiz] = useState<PersistedState>(() => loadPersisted());

  // Persist wizard config whenever it changes.
  useEffect(() => {
    try {
      localStorage.setItem(WIZ_KEY, JSON.stringify(wiz));
    } catch {
      /* storage unavailable — carry on */
    }
  }, [wiz]);

  const updateWiz = useCallback((patch: Partial<PersistedState>) => {
    setWiz((prev) => ({ ...prev, ...patch }));
  }, []);

  // ──────────────────────────────────────────────────────────────
  // Step 1: Candidates
  // ──────────────────────────────────────────────────────────────
  const [candidates, setCandidates] = useState<InterventionCandidate[]>([]);
  const [loadingCandidates, setLoadingCandidates] = useState(false);
  const [candidateError, setCandidateError] = useState<string | null>(null);

  const mergedCandidates = useMemo<InterventionCandidate[]>(
    () =>
      candidates.map((c) => {
        const patch = wiz.candidateEdits[c.name];
        return patch ? { ...c, ...patch } : c;
      }),
    [candidates, wiz.candidateEdits],
  );

  const loadCandidates = useCallback(async () => {
    setLoadingCandidates(true);
    setCandidateError(null);
    try {
      const res = await getDecisionCandidates();
      setCandidates(res.candidates ?? []);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setCandidateError(msg);
      setCandidates([]);
    } finally {
      setLoadingCandidates(false);
    }
  }, []);

  useEffect(() => {
    loadCandidates();
  }, [loadCandidates, runEndedAt]);

  const editCandidate = useCallback(
    (name: string, patch: Partial<InterventionCandidate>) => {
      setWiz((prev) => ({
        ...prev,
        candidateEdits: {
          ...prev.candidateEdits,
          [name]: { ...(prev.candidateEdits[name] ?? {}), ...patch },
        },
      }));
    },
    [],
  );

  // ──────────────────────────────────────────────────────────────
  // Step 2 metadata loaders (benefit source variables + columns)
  // ──────────────────────────────────────────────────────────────
  const [cateVars, setCateVars] = useState<string[]>([]);
  const [columns, setColumns] = useState<string[]>([]);

  useEffect(() => {
    getCateMapVariables()
      .then((res) => setCateVars(res?.variables ?? []))
      .catch(() => setCateVars([]));
    dataSummary()
      .then((s) => setColumns(s?.columns ?? []))
      .catch(() => setColumns([]));
  }, [runEndedAt]);

  // Keep benefitVar pointing at something valid as cateVars changes.
  useEffect(() => {
    if (wiz.benefitSource === "uniform") return;
    if (cateVars.length > 0 && !cateVars.includes(wiz.benefitVar)) {
      updateWiz({ benefitVar: cateVars[0] });
    }
  }, [wiz.benefitSource, wiz.benefitVar, cateVars, updateWiz]);

  // Keep activeTargetVar valid vs. cateVars.
  useEffect(() => {
    if (cateVars.length > 0 && !cateVars.includes(wiz.activeTargetVar)) {
      updateWiz({ activeTargetVar: cateVars[0] });
    }
  }, [cateVars, wiz.activeTargetVar, updateWiz]);

  // ──────────────────────────────────────────────────────────────
  // Step 3: Recommendation — run optimizers
  // ──────────────────────────────────────────────────────────────
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [ranked, setRanked] = useState<OptimizerResponse | null>(null);
  const [uncertainty, setUncertainty] = useState<UncertaintyRecord[]>([]);
  const [equityResult, setEquityResult] = useState<EquityResponse | null>(null);
  const [census, setCensus] = useState<CensusEquityResponse | null>(null);
  const [budgetResp, setBudgetResp] = useState<BudgetOptimizeResponse | null>(null);
  const [targeting, setTargeting] = useState<TargetingResponse | null>(null);
  const [targetingLoading, setTargetingLoading] = useState(false);

  // Re-fetch targeting when the active treatment variable or top-K changes.
  useEffect(() => {
    if (step !== 3 || !wiz.activeTargetVar) {
      return;
    }
    setTargetingLoading(true);
    getTargeting(wiz.activeTargetVar, wiz.topK)
      .then((g) => setTargeting(g))
      .catch(() => setTargeting(null))
      .finally(() => setTargetingLoading(false));
  }, [step, wiz.activeTargetVar, wiz.topK]);

  const runRecommendation = useCallback(async () => {
    if (mergedCandidates.length === 0) {
      notify("warning", "No candidates loaded");
      return;
    }
    setRunning(true);
    setRunError(null);
    try {
      const parsedBudget = wiz.budget.trim() === "" ? null : Number(wiz.budget);
      const optBody = {
        candidates: mergedCandidates,
        budget: parsedBudget,
        robustness_lambda: wiz.robustness,
        minimise: wiz.minimise,
      };

      // Fire causal optimizer + budget allocator + uncertainty + equity
      // in parallel.  Failures are surfaced per-task without aborting.
      const [
        optRes,
        budgetSettled,
        mcSettled,
        equitySettled,
        censusSettled,
      ] = await Promise.all([
        optimizeDecisions(optBody),
        wiz.benefitSource && (parsedBudget ?? 0) > 0
          ? optimizeBudget({
              budget: parsedBudget ?? 0,
              benefit_source: wiz.benefitSource,
              variable: wiz.benefitSource === "uniform" ? undefined : wiz.benefitVar,
              cost_column: wiz.costColumn || undefined,
              x_max_column: wiz.xMaxColumn || undefined,
              solver: wiz.solver,
              pareto_sweep: wiz.paretoSweep,
            }).then(
              (r) => ({ ok: true as const, value: r }),
              (e: unknown) => ({ ok: false as const, error: e }),
            )
          : Promise.resolve({ ok: true as const, value: null as BudgetOptimizeResponse | null }),
        decisionUncertainty({ ...optBody, n_draws: wiz.nDraws }).then(
          (r) => ({ ok: true as const, value: r }),
          (e: unknown) => ({ ok: false as const, error: e }),
        ),
        runEquityLayers(mergedCandidates, wiz.layerText, wiz.layerInvert),
        getCensusEquity().then(
          (r) => ({ ok: true as const, value: r }),
          (e: unknown) => ({ ok: false as const, error: e }),
        ),
      ]);

      setRanked(optRes);
      if (budgetSettled.ok) setBudgetResp(budgetSettled.value ?? null);
      else notify("warning", `Budget optimizer failed: ${errorMsg(budgetSettled.error)}`);

      if (mcSettled.ok) setUncertainty(mcSettled.value?.uncertainty ?? []);
      else {
        notify("warning", `Uncertainty MC failed: ${errorMsg(mcSettled.error)}`);
        setUncertainty([]);
      }

      if (equitySettled.ok) setEquityResult(equitySettled.value);
      else if (equitySettled.reason) notify("info", equitySettled.reason);

      if (censusSettled.ok) setCensus(censusSettled.value);

      notify("success", `Ranked ${optRes.ranked.length} candidates`);
    } catch (err) {
      const msg = errorMsg(err);
      setRunError(msg);
      notify("error", msg);
    } finally {
      setRunning(false);
    }
  }, [mergedCandidates, wiz, notify]);

  // Auto-run when user advances to step 3 for the first time.
  const autoRanRef = useRef(false);
  useEffect(() => {
    if (step === 3 && !autoRanRef.current && mergedCandidates.length > 0 && !running) {
      autoRanRef.current = true;
      void runRecommendation();
    }
  }, [step, mergedCandidates, running, runRecommendation]);

  const goToStep = useCallback((next: Step) => {
    setStep(next);
  }, []);

  const resetWizard = useCallback(() => {
    setWiz({ ...DEFAULT_STATE });
    setRanked(null);
    setBudgetResp(null);
    setUncertainty([]);
    setEquityResult(null);
    setCensus(null);
    setTargeting(null);
    autoRanRef.current = false;
    setStep(1);
  }, []);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <SectionHeader
        kicker="Budget & Equity"
        label="Decision Support"
        right={<Btn small onClick={resetWizard}>Reset</Btn>}
      />

      <Stepper current={step} onNavigate={goToStep} />

      {step === 1 && (
        <Step1Candidates
          candidates={mergedCandidates}
          rawCandidates={candidates}
          loading={loadingCandidates}
          error={candidateError}
          onReload={loadCandidates}
          onEdit={editCandidate}
          onNext={() => goToStep(2)}
        />
      )}

      {step === 2 && (
        <Step2Constraints
          wiz={wiz}
          updateWiz={updateWiz}
          cateVars={cateVars}
          columns={columns}
          onBack={() => goToStep(1)}
          onNext={() => {
            autoRanRef.current = false;
            goToStep(3);
          }}
        />
      )}

      {step === 3 && (
        <Step3Recommendation
          running={running}
          error={runError}
          ranked={ranked}
          uncertainty={uncertainty}
          equity={equityResult}
          census={census}
          budget={budgetResp}
          targeting={targeting}
          targetingLoading={targetingLoading}
          cateVars={cateVars}
          wiz={wiz}
          updateWiz={updateWiz}
          onBack={() => goToStep(2)}
          onRerun={runRecommendation}
        />
      )}
    </div>
  );
}


// ════════════════════════════════════════════════════════════════
// Stepper
// ════════════════════════════════════════════════════════════════
function Stepper({ current, onNavigate }: { current: Step; onNavigate: (s: Step) => void }) {
  const steps: Array<{ n: Step; label: string; sub: string }> = [
    { n: 1, label: "Scenarios", sub: "Review modelled interventions" },
    { n: 2, label: "Budget & Equity", sub: "Set constraints and allocation rules" },
    { n: 3, label: "Results", sub: "Ranked options and spatial targeting" },
  ];
  return (
    <div style={{ display: "flex", gap: 12 }}>
      {steps.map((s) => {
        const active = s.n === current;
        const done = s.n < current;
        return (
          <button
            key={s.n}
            onClick={() => onNavigate(s.n)}
            disabled={s.n > current && current < 3}
            style={{
              flex: 1,
              textAlign: "left",
              padding: "12px 14px",
              borderRadius: 10,
              border: `1px solid ${active ? "var(--accent, #c9662b)" : "var(--line)"}`,
              background: active ? "#fff4ec" : done ? "#f7f3eb" : "#fff",
              cursor: s.n > current ? "default" : "pointer",
              opacity: s.n > current && current < 3 ? 0.6 : 1,
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span
                style={{
                  width: 22,
                  height: 22,
                  borderRadius: "50%",
                  background: active ? "var(--accent, #c9662b)" : done ? "#7a8a5a" : "var(--line)",
                  color: "#fff",
                  display: "inline-flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: 12,
                  fontWeight: 700,
                }}
              >
                {done ? "✓" : s.n}
              </span>
              <span style={{ fontSize: 13, fontWeight: 600 }}>{s.label}</span>
            </div>
            <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 4 }}>{s.sub}</div>
          </button>
        );
      })}
    </div>
  );
}

// ════════════════════════════════════════════════════════════════
// Step 1 — Candidates
// ════════════════════════════════════════════════════════════════
function Step1Candidates({
  candidates,
  rawCandidates,
  loading,
  error,
  onReload,
  onEdit,
  onNext,
}: {
  candidates: InterventionCandidate[];
  rawCandidates: InterventionCandidate[];
  loading: boolean;
  error: string | null;
  onReload: () => void;
  onEdit: (name: string, patch: Partial<InterventionCandidate>) => void;
  onNext: () => void;
}) {
  const canAdvance = candidates.length > 0;
  return (
    <Card
      title="Step 1 — Modelled Scenarios"
      subtitle="Review your pipeline scenarios. Adjust cost and equity weight to reflect real-world priorities before optimizing."
      actions={
        <div style={{ display: "flex", gap: 8 }}>
          <Btn small onClick={onReload} disabled={loading}>
            {loading ? "Loading…" : "Reload"}
          </Btn>
          <Btn primary onClick={onNext} disabled={!canAdvance}>
            Next: Budget & Equity →
          </Btn>
        </div>
      }
    >
      {error && (
        <div style={{ padding: 10, background: "#fee8e0", borderRadius: 6, marginBottom: 12, fontSize: 12 }}>
          {error}
        </div>
      )}
      {candidates.length === 0 && !loading ? (
        <EmptyState
          title="No scenarios found"
          body="Run the pipeline through Stage 4 first — scenarios and CATE estimates will load automatically."
        />
      ) : (
        <>
          {/* Legend */}
          <div style={{ display: "flex", gap: 16, marginBottom: 10, fontSize: 11, color: "var(--muted)" }}>
            <span>Pipeline outputs are read-only.</span>
            <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <span style={{ display: "inline-block", width: 10, height: 10, borderRadius: 2, background: "#fff4ec", border: "1px solid var(--accent,#c9662b)" }} />
              Shaded columns are editable — changes apply only to this optimization.
            </span>
          </div>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
              <colgroup>
                {/* Read-only zone */}
                <col /><col /><col /><col /><col />
                {/* Editable zone */}
                <col style={{ background: "#fffaf6" }} /><col style={{ background: "#fffaf6" }} />
              </colgroup>
              <thead>
                <tr>
                  <th colSpan={5} style={{ ...thStyle, borderBottom: "none", paddingBottom: 2, color: "var(--muted)", fontWeight: 400, fontSize: 10, letterSpacing: "0.06em", textTransform: "uppercase" }}>
                    From pipeline
                  </th>
                  <th colSpan={2} style={{ ...thStyle, borderBottom: "none", paddingBottom: 2, color: "var(--accent,#c9662b)", fontWeight: 600, fontSize: 10, letterSpacing: "0.06em", textTransform: "uppercase", background: "#fffaf6", borderRadius: "6px 6px 0 0" }}>
                    Your inputs
                  </th>
                </tr>
                <tr style={{ textAlign: "left", borderBottom: "1px solid var(--line)" }}>
                  <th style={thStyle}>Scenario</th>
                  <th style={thStyle}>Treatment</th>
                  <th style={thStyle}>Magnitude</th>
                  <th style={thStyle}>Mean effect</th>
                  <th style={thStyle}>Uncertainty (σ)</th>
                  <th style={{ ...thStyle, background: "#fffaf6" }}>Cost</th>
                  <th style={{ ...thStyle, background: "#fffaf6" }}>Equity weight</th>
                </tr>
              </thead>
              <tbody>
                {candidates.map((c) => {
                  const baseline = rawCandidates.find((r) => r.name === c.name);
                  const costChanged = Math.abs(c.cost - (baseline?.cost ?? c.cost)) > 1e-9;
                  const eqChanged = Math.abs(c.equity_weight - (baseline?.equity_weight ?? c.equity_weight)) > 1e-9;
                  return (
                    <tr key={c.name} style={{ borderBottom: "1px solid #eee" }}>
                      <td style={{ ...tdStyle, fontWeight: 600 }}>{c.name}</td>
                      <td style={tdStyle}><Tag>{c.treatment}</Tag></td>
                      <td style={tdStyle}>{c.magnitude.toFixed(3)}</td>
                      <td style={tdStyle}>{c.mean_effect.toFixed(4)}</td>
                      <td style={{ ...tdStyle, color: "var(--muted)" }}>{c.effect_std.toFixed(4)}</td>
                      <td style={{ ...tdStyle, background: "#fffaf6" }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                          <NumberField
                            value={c.cost}
                            baseline={baseline?.cost ?? c.cost}
                            step={0.1}
                            onChange={(v) => onEdit(c.name, { cost: v })}
                          />
                          {costChanged && (
                            <button
                              onClick={() => onEdit(c.name, { cost: baseline?.cost ?? c.cost })}
                              title="Reset to pipeline value"
                              style={{ background: "none", border: "none", cursor: "pointer", fontSize: 11, color: "var(--muted)", padding: "0 2px" }}
                            >↺</button>
                          )}
                        </div>
                      </td>
                      <td style={{ ...tdStyle, background: "#fffaf6" }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                          <NumberField
                            value={c.equity_weight}
                            baseline={baseline?.equity_weight ?? c.equity_weight}
                            step={0.05}
                            onChange={(v) => onEdit(c.name, { equity_weight: v })}
                          />
                          {eqChanged && (
                            <button
                              onClick={() => onEdit(c.name, { equity_weight: baseline?.equity_weight ?? c.equity_weight })}
                              title="Reset to pipeline value"
                              style={{ background: "none", border: "none", cursor: "pointer", fontSize: 11, color: "var(--muted)", padding: "0 2px" }}
                            >↺</button>
                          )}
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </Card>
  );
}

// ════════════════════════════════════════════════════════════════
// Step 2 — Budget & Equity
// ════════════════════════════════════════════════════════════════
function Step2Constraints({
  wiz,
  updateWiz,
  cateVars,
  columns,
  onBack,
  onNext,
}: {
  wiz: PersistedState;
  updateWiz: (p: Partial<PersistedState>) => void;
  cateVars: string[];
  columns: string[];
  onBack: () => void;
  onNext: () => void;
}) {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const sourceList = wiz.benefitSource === "cate" ? cateVars : [];
  const sourceUnavailable = wiz.benefitSource !== "uniform" && sourceList.length === 0;

  return (
    <Card
      title="Step 2 — Budget & Equity"
      subtitle="Set your spending limit and equity priorities. Advanced settings are optional."
    >
      {/* ── Core controls ── */}
      <FieldRow label="Total budget (optional)">
        <input
          type="number"
          value={wiz.budget}
          onChange={(e) => updateWiz({ budget: e.target.value })}
          placeholder="Leave blank for unconstrained"
          style={{ ...inputStyle, width: 220 }}
        />
      </FieldRow>

      <FieldRow label="Equity priority">
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: 11, color: "var(--muted)", whiteSpace: "nowrap" }}>Efficiency first</span>
          <input
            type="range" min={0} max={1} step={0.05}
            value={wiz.robustness / 2}
            onChange={(e) => updateWiz({ robustness: Number(e.target.value) * 2 })}
            style={{ flex: 1 }}
          />
          <span style={{ fontSize: 11, color: "var(--muted)", whiteSpace: "nowrap" }}>Equity first</span>
          <span style={{ width: 36, textAlign: "right", fontSize: 12, fontWeight: 600 }}>
            {Math.round((wiz.robustness / 2) * 100)}%
          </span>
        </div>
        <div style={{ fontSize: 10, color: "var(--muted)", marginTop: 3 }}>
          Higher values prioritise underserved areas over maximum aggregate benefit.
        </div>
      </FieldRow>

      <FieldRow label="Objective">
        <label style={{ fontSize: 12, display: "inline-flex", gap: 6, alignItems: "center" }}>
          <input
            type="checkbox"
            checked={wiz.minimise}
            onChange={(e) => updateWiz({ minimise: e.target.checked })}
          />
          Minimise the outcome (e.g. reduce risk) — uncheck to maximise
        </label>
      </FieldRow>

      {/* ── Advanced toggle ── */}
      <button
        onClick={() => setShowAdvanced((v) => !v)}
        style={{
          marginTop: 8,
          background: "none",
          border: "none",
          padding: 0,
          fontSize: 12,
          color: "var(--accent, #c9662b)",
          cursor: "pointer",
          display: "flex",
          alignItems: "center",
          gap: 4,
        }}
      >
        <span style={{ fontSize: 10 }}>{showAdvanced ? "▾" : "▸"}</span>
        {showAdvanced ? "Hide advanced settings" : "Advanced settings"}
      </button>

      {showAdvanced && (
        <div style={{ marginTop: 12, display: "flex", flexDirection: "column", gap: 0, borderTop: "1px solid var(--line)", paddingTop: 12 }}>
          <FieldRow label={`Robustness λ — ${wiz.robustness.toFixed(2)}`}>
            <input
              type="range" min={0} max={2} step={0.1}
              value={wiz.robustness}
              onChange={(e) => updateWiz({ robustness: Number(e.target.value) })}
              style={{ width: "100%" }}
            />
            <div style={{ fontSize: 10, color: "var(--muted)", marginTop: 2 }}>
              Penalises high-variance candidates. 0 = pure mean effect, 2 = strongly risk-averse.
            </div>
          </FieldRow>

          <FieldRow label="Benefit source for spatial allocation">
            <div style={{ display: "flex", gap: 6 }}>
              {(["cate", "uniform"] as BenefitSource[]).map((src) => (
                <button
                  key={src}
                  onClick={() => updateWiz({ benefitSource: src })}
                  style={{
                    padding: "4px 10px", fontSize: 11, borderRadius: 6, cursor: "pointer",
                    border: `1px solid ${wiz.benefitSource === src ? "var(--accent, #c9662b)" : "var(--line)"}`,
                    background: wiz.benefitSource === src ? "#fff4ec" : "#fff",
                  }}
                >
                  {src === "cate" ? "Causal effect map" : "Uniform"}
                </button>
              ))}
            </div>
          </FieldRow>

          {wiz.benefitSource !== "uniform" && (
            <FieldRow label={`Treatment variable${sourceUnavailable ? " — none available yet" : ""}`}>
              <select
                value={wiz.benefitVar}
                onChange={(e) => updateWiz({ benefitVar: e.target.value })}
                style={inputStyle}
                disabled={sourceUnavailable}
              >
                {sourceList.map((v) => <option key={v} value={v}>{v}</option>)}
              </select>
            </FieldRow>
          )}

          <FieldRow label="Solver">
            <select
              value={wiz.solver}
              onChange={(e) => updateWiz({ solver: e.target.value as Solver })}
              style={inputStyle}
            >
              <option value="auto">Auto (recommended)</option>
              <option value="greedy">Greedy</option>
              <option value="greedy_2opt">Greedy + 2-opt</option>
              <option value="milp">MILP — exact, slow for n &gt; 5000</option>
            </select>
          </FieldRow>

          <FieldRow label="Cost column (optional)">
            <select
              value={wiz.costColumn}
              onChange={(e) => updateWiz({ costColumn: e.target.value })}
              style={inputStyle}
            >
              <option value="">— uniform cost —</option>
              {columns.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </FieldRow>

          <FieldRow label="Max-allocation column (optional)">
            <select
              value={wiz.xMaxColumn}
              onChange={(e) => updateWiz({ xMaxColumn: e.target.value })}
              style={inputStyle}
            >
              <option value="">— no cap —</option>
              {columns.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </FieldRow>

          <FieldRow label={`Uncertainty draws — ${wiz.nDraws}`}>
            <input
              type="range" min={100} max={2000} step={100}
              value={wiz.nDraws}
              onChange={(e) => updateWiz({ nDraws: Number(e.target.value) })}
              style={{ width: "100%" }}
            />
          </FieldRow>

          <FieldRow label="Pareto sweep">
            <label style={{ fontSize: 12, display: "inline-flex", gap: 6, alignItems: "center" }}>
              <input
                type="checkbox"
                checked={wiz.paretoSweep}
                onChange={(e) => updateWiz({ paretoSweep: e.target.checked })}
              />
              Compute benefit vs. budget curve
            </label>
          </FieldRow>
        </div>
      )}

      <div style={{ display: "flex", gap: 8, justifyContent: "space-between", marginTop: 20 }}>
        <Btn small onClick={onBack}>← Back</Btn>
        <Btn primary onClick={onNext}>Run Optimization →</Btn>
      </div>
    </Card>
  );
}

// ════════════════════════════════════════════════════════════════
// Step 3 — Recommendation
// ════════════════════════════════════════════════════════════════
function Step3Recommendation({
  running,
  error,
  ranked,
  uncertainty,
  equity,
  census,
  budget,
  targeting,
  targetingLoading,
  cateVars,
  wiz,
  updateWiz,
  onBack,
  onRerun,
}: {
  running: boolean;
  error: string | null;
  ranked: OptimizerResponse | null;
  uncertainty: UncertaintyRecord[];
  equity: EquityResponse | null;
  census: CensusEquityResponse | null;
  budget: BudgetOptimizeResponse | null;
  targeting: TargetingResponse | null;
  targetingLoading: boolean;
  cateVars: string[];
  wiz: PersistedState;
  updateWiz: (p: Partial<PersistedState>) => void;
  onBack: () => void;
  onRerun: () => void;
}) {
  const targetingBlockRef = useRef<HTMLDivElement>(null);
  const rankedRows: RankedIntervention[] = ranked?.ranked ?? [];
  const topPick = rankedRows[0];
  const uncByName = useMemo(() => {
    const m = new Map<string, UncertaintyRecord>();
    for (const u of uncertainty) m.set(u.candidate, u);
    return m;
  }, [uncertainty]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <Card
        title="Step 3 — Results"
        subtitle={running ? "Running optimization…" : "Scenarios ranked by projected benefit, cost, and equity."}
        actions={
          <div style={{ display: "flex", gap: 8 }}>
            <Btn small onClick={onBack}>← Back</Btn>
            <Btn small primary onClick={onRerun} disabled={running}>
              {running ? "Running…" : "Re-run"}
            </Btn>
          </div>
        }
      >
        {error && (
          <div style={{ padding: 10, background: "#fee8e0", borderRadius: 6, marginBottom: 12, fontSize: 12 }}>
            {error}
          </div>
        )}

        {ranked && (
          <StatGrid>
            <Stat label="Scenarios evaluated" value={`${rankedRows.length}`} />
            <Stat label="Recommended" value={`${rankedRows.filter((r) => r.selected).length}`} />
            <Stat
              label="Top scenario"
              value={topPick ? topPick.risk_adjusted_effect.toFixed(4) : "—"}
              sub={topPick?.name}
            />
            <Stat
              label="Equity distribution (Gini)"
              value={ranked.equity_summary.gini == null ? "—" : ranked.equity_summary.gini.toFixed(3)}
            />
          </StatGrid>
        )}
      </Card>

      {ranked && (
        <Card
          title="Ranked Scenarios"
          subtitle="Scenarios are ordered by projected benefit adjusted for cost and equity. Recommended scenarios are highlighted."
        >
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
              <thead>
                <tr style={{ textAlign: "left", borderBottom: "1px solid var(--line)" }}>
                  <th style={thStyle}>Rank</th>
                  <th style={thStyle}>Scenario</th>
                  <th style={thStyle}>Projected benefit</th>
                  <th style={thStyle}>Cost</th>
                  <th style={thStyle}>Equity weight</th>
                  <th style={thStyle}>Uncertainty range</th>
                  <th style={{ ...thStyle, textAlign: "center" }}>Recommended</th>
                </tr>
              </thead>
              <tbody>
                {rankedRows.map((r, i) => {
                  const u = uncByName.get(r.name);
                  const isTop = i === 0 && r.selected;
                  return (
                    <tr
                      key={r.name}
                      style={{
                        borderBottom: "1px solid #eee",
                        background: isTop ? "#eaf4d9" : r.selected ? "#f1f7e8" : undefined,
                      }}
                    >
                      <td style={{ ...tdStyle, fontWeight: r.selected ? 700 : undefined }}>{i + 1}</td>
                      <td style={{ ...tdStyle, fontWeight: r.selected ? 700 : undefined }}>{r.name}</td>
                      <td style={tdStyle}>{r.risk_adjusted_effect.toFixed(4)}</td>
                      <td style={tdStyle}>{r.cost.toFixed(2)}</td>
                      <td style={tdStyle}>{r.equity_weight.toFixed(2)}</td>
                      <td style={{ ...tdStyle, color: "var(--muted)" }}>
                        {u ? `${u.effect_p10.toFixed(3)} – ${u.effect_p90.toFixed(3)}` : "—"}
                      </td>
                      <td style={{ ...tdStyle, textAlign: "center" }}>
                        {r.selected ? (
                          <span style={{
                            display: "inline-block",
                            padding: "2px 8px",
                            borderRadius: 10,
                            fontSize: 11,
                            fontWeight: 600,
                            background: isTop ? "#c8e6a0" : "#dcedc8",
                            color: "#2e7d32",
                          }}>
                            {isTop ? "★ Top pick" : "✓ Yes"}
                          </span>
                        ) : null}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {ranked && !budget && !running && (
        <Card title="Budget Allocation" subtitle="No budget set — add one below to see where to spend it.">
          <div style={{ display: "flex", gap: 10, alignItems: "flex-end", flexWrap: "wrap" }}>
            <div>
              <div style={{ fontSize: 11, color: "var(--muted)", marginBottom: 4 }}>
                Total budget ($)
              </div>
              <input
                type="number"
                min={0}
                value={wiz.budget}
                onChange={(e) => updateWiz({ budget: e.target.value })}
                placeholder="e.g. 500000"
                style={{ ...inputStyle, width: 180 }}
                autoFocus
              />
            </div>
            <Btn
              primary
              onClick={onRerun}
              disabled={!wiz.budget || Number(wiz.budget) <= 0}
            >
              Run allocation →
            </Btn>
          </div>
          <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 8 }}>
            This shows which areas to prioritise and how much to spend in each, based on your ranked scenarios and equity settings.
          </div>
        </Card>
      )}

      {budget && (
        <Card
          title="Budget Allocation"
          subtitle={`${budget.benefit_description} — solver: ${budget.result.solver}`}
        >
          <StatGrid>
            <Stat label="Areas targeted" value={`${budget.result.n_treated} of ${budget.n_cells}`} />
            <Stat label="Fully treated" value={`${budget.result.n_fully_treated}`} />
            <Stat label="Total projected benefit" value={budget.result.total_benefit.toFixed(3)} />
            <Stat label="Budget used" value={`$${budget.result.total_cost.toFixed(0)} of $${budget.result.budget.toFixed(0)}`} />
            <Stat
              label="Distribution equity"
              value={giniLabel(budget.result.gini)}
              sub={`Gini ${budget.result.gini.toFixed(3)}`}
            />
            <Stat label="Solve time" value={`${(budget.result.solve_time_s * 1000).toFixed(0)} ms`} />
          </StatGrid>
          <AllocationBars allocation={budget.result.allocation} />
          {budget.pareto && budget.pareto.points.length > 0 && (
            <ParetoPlot points={budget.pareto.points} />
          )}
        </Card>
      )}

      {equity && (
        <Card title="Equity layers" subtitle={`Disparity index: ${equity.disparity_index.toFixed(3)}`}>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
              <thead>
                <tr style={{ textAlign: "left", borderBottom: "1px solid var(--line)" }}>
                  <th style={thStyle}>Candidate</th>
                  <th style={thStyle}>Weight</th>
                  <th style={thStyle}>Breakdown</th>
                </tr>
              </thead>
              <tbody>
                {equity.scores.map((s) => (
                  <tr key={s.candidate} style={{ borderBottom: "1px solid #eee" }}>
                    <td style={tdStyle}>{s.candidate}</td>
                    <td style={tdStyle}>{s.weight.toFixed(3)}</td>
                    <td style={{ ...tdStyle, fontFamily: "monospace", fontSize: 11 }}>
                      {Object.entries(s.layer_breakdown)
                        .map(([k, v]) => `${k}:${v.toFixed(2)}`)
                        .join("  ")}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {census && (
        <Card
          title="Community Equity Profile"
          subtitle={`Source: US Census ACS · ${census.context.n_counties} ${census.context.n_counties === 1 ? "county" : "counties"} · ${census.context.vintage} vintage`}
        >
          {(() => {
            const ctx = census.context;
            const pov = ctx.poverty_rate ?? null;
            const inc = ctx.median_income ?? null;
            const mh = ctx.mobile_home_share ?? null;
            // Income deprivation: 0 at $80k+, 1 at $0
            const incDep = inc == null ? null : Math.max(0, Math.min(1, 1 - inc / 80_000));
            const known = [pov, incDep, mh].filter((v) => v != null) as number[];
            const weights = [0.4, 0.4, 0.2].slice(0, known.length);
            const weightSum = weights.reduce((a, b) => a + b, 0);
            const score = known.length > 0
              ? known.reduce((s, v, i) => s + v * weights[i], 0) / weightSum
              : null;
            const { label: vulnLabel, color: vulnColor, bg: vulnBg } = score == null
              ? { label: "No data", color: "#6e6358", bg: "#f5f0ea" }
              : score < 0.2
              ? { label: "Low vulnerability", color: "#2e7d32", bg: "#eaf4d9" }
              : score < 0.4
              ? { label: "Moderate vulnerability", color: "#b45309", bg: "#fef3c7" }
              : { label: "High vulnerability", color: "#b91c1c", bg: "#fee2e2" };

            return (
              <>
                {/* Composite score banner */}
                <div style={{
                  display: "flex", alignItems: "center", gap: 14,
                  padding: "12px 14px", borderRadius: 8,
                  background: vulnBg, marginBottom: 14,
                }}>
                  <div style={{ fontSize: 28, fontWeight: 800, color: vulnColor, lineHeight: 1 }}>
                    {score == null ? "—" : Math.round(score * 100)}
                    {score != null && <span style={{ fontSize: 14, fontWeight: 400 }}>/100</span>}
                  </div>
                  <div>
                    <div style={{ fontSize: 13, fontWeight: 700, color: vulnColor }}>{vulnLabel}</div>
                    <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 2 }}>
                      Composite equity score — weighted from poverty rate, income level, and housing insecurity
                    </div>
                  </div>
                </div>

                {/* Underlying indicators */}
                <StatGrid>
                  <Stat
                    label="Poverty rate"
                    value={pov == null ? "—" : `${(pov * 100).toFixed(1)}%`}
                    sub={pov == null ? undefined : pov > 0.2 ? "Above national avg" : "Below national avg"}
                  />
                  <Stat
                    label="Median household income"
                    value={inc == null ? "—" : `$${inc.toLocaleString()}`}
                    sub={inc == null ? undefined : inc < 50_000 ? "Below national median" : "Above national median"}
                  />
                  <Stat
                    label="Housing insecurity"
                    value={mh == null ? "—" : `${(mh * 100).toFixed(1)}%`}
                    sub="Share of mobile / manufactured homes"
                  />
                  <Stat
                    label="Population"
                    value={ctx.population == null ? "—" : ctx.population.toLocaleString()}
                    sub="Affected residents"
                  />
                </StatGrid>
              </>
            );
          })()}
        </Card>
      )}

      <Card
        title="Priority Targeting Map"
        subtitle="Highest-priority areas for intervention based on causal effect and cost. Darker colour = higher priority."
        actions={
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <select
              value={wiz.activeTargetVar}
              onChange={(e) => updateWiz({ activeTargetVar: e.target.value })}
              style={{ ...inputStyle, maxWidth: 160 }}
              disabled={cateVars.length === 0}
            >
              {cateVars.length === 0
                ? <option value="">— run pipeline first —</option>
                : cateVars.map((v) => <option key={v} value={v}>{v}</option>)}
            </select>
            <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <span style={{ fontSize: 11, color: "var(--muted)" }}>Top</span>
              <input
                type="number"
                value={wiz.topK}
                min={1}
                max={500}
                onChange={(e) => updateWiz({ topK: Math.max(1, Number(e.target.value) || 1) })}
                style={{ ...inputStyle, width: 64 }}
              />
              <span style={{ fontSize: 11, color: "var(--muted)" }}>areas</span>
            </div>
          </div>
        }
      >
        {targetingLoading ? (
          <div style={{ padding: 24, textAlign: "center", color: "var(--muted)" }}>Loading…</div>
        ) : targeting ? (
          <div ref={targetingBlockRef} style={{ position: "relative" }}>
            <div style={{ position: "absolute", top: 8, right: 50, zIndex: 6 }}>
              <ExportBlockButton
                targetRef={targetingBlockRef}
                artifactId="decision_targeting"
                label={`targeting_${wiz.activeTargetVar || "map"}`}
                compact
              />
            </div>
            <ResizableMapWrapper defaultHeight={MAP_HEIGHT_DEFAULT}>
              <SpatialMap
                geojson={targeting}
                colorField="priority"
                palette="viridis"
                expandable
              />
            </ResizableMapWrapper>
          </div>
        ) : (
          <EmptyState title="No targeting map" body="Select a treatment variable above to see priority areas." />
        )}
      </Card>
    </div>
  );
}

// ════════════════════════════════════════════════════════════════
// Small reusable bits
// ════════════════════════════════════════════════════════════════
function AllocationBars({ allocation }: { allocation: number[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = w * DPR;
    canvas.height = h * DPR;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(DPR, DPR);
    ctx.clearRect(0, 0, w, h);
    const nz = allocation.filter((v) => v > 1e-9);
    if (nz.length === 0) {
      ctx.fillStyle = "#6e6358";
      ctx.font = "12px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("No cells allocated", w / 2, h / 2);
      return;
    }
    const max = Math.max(...nz);
    const nbins = 24;
    const bins = new Array(nbins).fill(0);
    for (const v of nz) {
      const idx = Math.min(nbins - 1, Math.floor((v / max) * nbins));
      bins[idx] += 1;
    }
    const maxCount = Math.max(...bins);
    const bw = w / nbins;
    bins.forEach((c, i) => {
      const bh = (c / maxCount) * (h - 20);
      const rampIdx = Math.min(
        SPARC_RAMP_HEX.length - 1,
        Math.floor((i / nbins) * SPARC_RAMP_HEX.length),
      );
      ctx.fillStyle = SPARC_RAMP_HEX[rampIdx];
      ctx.fillRect(i * bw + 1, h - bh - 4, bw - 2, bh);
    });
  }, [allocation]);
  return <canvas ref={canvasRef} style={{ width: "100%", height: 120, marginTop: 12 }} />;
}

function ParetoPlot({ points }: { points: Array<{ budget: number; total_benefit: number }> }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || points.length === 0) return;
    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = w * DPR;
    canvas.height = h * DPR;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(DPR, DPR);
    ctx.clearRect(0, 0, w, h);
    const pad = 28;
    const xs = points.map((p) => p.budget);
    const ys = points.map((p) => p.total_benefit);
    const xMin = Math.min(...xs);
    const xMax = Math.max(...xs);
    const yMin = Math.min(...ys);
    const yMax = Math.max(...ys);
    const sx = (x: number) => pad + ((x - xMin) / Math.max(1e-9, xMax - xMin)) * (w - pad * 1.5);
    const sy = (y: number) => h - pad - ((y - yMin) / Math.max(1e-9, yMax - yMin)) * (h - pad * 1.5);
    ctx.strokeStyle = "#cfc4b3";
    ctx.beginPath();
    ctx.moveTo(pad, pad / 2);
    ctx.lineTo(pad, h - pad);
    ctx.lineTo(w - pad / 2, h - pad);
    ctx.stroke();
    ctx.strokeStyle = "var(--accent, #c9662b)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    points.forEach((p, i) => {
      const x = sx(p.budget);
      const y = sy(p.total_benefit);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = "#6e6358";
    ctx.font = "10px Inter, sans-serif";
    ctx.fillText("Budget →", w - 60, h - 6);
    ctx.save();
    ctx.translate(10, h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Benefit", 0, 0);
    ctx.restore();
  }, [points]);
  return <canvas ref={ref} style={{ width: "100%", height: 160, marginTop: 12 }} />;
}

function NumberField({
  value,
  baseline,
  step = 0.1,
  onChange,
}: {
  value: number;
  baseline: number;
  step?: number;
  onChange: (v: number) => void;
}) {
  const changed = Math.abs(value - baseline) > 1e-9;
  return (
    <input
      type="number"
      value={value}
      step={step}
      onChange={(e) => {
        const n = Number(e.target.value);
        if (Number.isFinite(n)) onChange(n);
      }}
      style={{
        ...inputStyle,
        width: 84,
        borderColor: changed ? "var(--accent, #c9662b)" : "var(--line)",
        background: changed ? "#fff8f2" : "#fff",
      }}
    />
  );
}

function FieldRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ fontSize: 11, color: "var(--muted)", marginBottom: 4 }}>{label}</div>
      {children}
    </div>
  );
}

const inputStyle: React.CSSProperties = {
  padding: "4px 8px",
  fontSize: 12,
  border: "1px solid var(--line)",
  borderRadius: 6,
  background: "#fff",
};

const thStyle: React.CSSProperties = { padding: "6px 8px", fontSize: 11, fontWeight: 600, color: "var(--muted)" };
const tdStyle: React.CSSProperties = { padding: "6px 8px" };

// ════════════════════════════════════════════════════════════════
// Helpers
// ════════════════════════════════════════════════════════════════
function errorMsg(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

function giniLabel(gini: number): string {
  if (gini < 0.3) return "Fairly distributed";
  if (gini < 0.5) return "Moderately concentrated";
  return "Highly concentrated";
}

type EquitySettled =
  | { ok: true; value: EquityResponse }
  | { ok: false; reason?: string };

async function runEquityLayers(
  candidates: InterventionCandidate[],
  layerText: Record<string, string>,
  layerInvert: Record<string, boolean>,
): Promise<EquitySettled> {
  const names = candidates.map((c) => c.name);
  const layers: Record<string, number[]> = {};
  const invert: Record<string, boolean> = {};
  for (const [name, raw] of Object.entries(layerText)) {
    const parsed = raw.split(/[\s,]+/).map(Number).filter((v) => Number.isFinite(v));
    if (parsed.length === names.length) {
      layers[name] = parsed;
      invert[name] = !!layerInvert[name];
    }
  }
  if (Object.keys(layers).length === 0) {
    return { ok: false, reason: undefined };
  }
  try {
    const eq = await computeEquity({ candidate_names: names, layers, invert });
    return { ok: true, value: eq };
  } catch (e) {
    return { ok: false, reason: `Equity layers failed: ${errorMsg(e)}` };
  }
}
