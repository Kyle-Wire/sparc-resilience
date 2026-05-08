import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { SectionHeader, Card, Btn, Stat, StatGrid, Tag } from "@/components/ui/DesignSystem";
import { EmptyState } from "@/components/common/EmptyState";
import { SPARC_RAMP_HEX, MAP_HEIGHT_DEFAULT } from "@/lib/design-tokens";
import SpatialMap from "@/components/map/SpatialMap";
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
  optimizeScenario,
  getEquityLayer,
  type InterventionCandidate,
  type OptimizerResponse,
  type RankedIntervention,
  type EquityResponse,
  type CensusEquityResponse,
  type TargetingResponse,
  type UncertaintyRecord,
  type BudgetOptimizeResponse,
  type OptimizeScenarioResponse,
  type EquityLayerRecord,
  type RuntimeScenarioSpec,
} from "@/lib/api";

/**
 * Decision Support — Phase 7 unified wizard.
 *
 * Consolidates the former Decisions + BudgetOptimizer pages into a
 * 3-step flow:
 *   1. Candidates   — load, edit cost/equity, pick treatment variables
 *   2. Constraints  — budget, robustness λ, direction, benefit source,
 *                     solver, equity layers
 *   3. Recommendation — runs causal optimizer + budget allocator in
 *                     parallel, shows ranked table + spatial targeting
 *                     + allocation histogram + Pareto frontier
 *
 * Wizard state persists in localStorage under WIZ_KEY so users can
 * step away and return without losing their edits.
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

  const [mode, setMode] = useState<"portfolio" | "scenario">("portfolio");

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <SectionHeader
        kicker="Phase 7"
        label="Decision Support"
        right={
          <div style={{ display: "flex", gap: 8 }}>
            {mode === "portfolio" && <Btn small onClick={resetWizard}>Reset</Btn>}
          </div>
        }
      />

      {/* Mode tabs */}
      <div style={{ display: "flex", gap: 0, borderBottom: "1px solid var(--line)" }}>
        {(["portfolio", "scenario"] as const).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            style={{
              padding: "8px 18px",
              fontSize: 13,
              fontWeight: mode === m ? 700 : 400,
              border: "none",
              borderBottom: mode === m ? "2px solid var(--accent, #c9662b)" : "2px solid transparent",
              background: "transparent",
              color: mode === m ? "var(--accent, #c9662b)" : "var(--muted)",
              cursor: "pointer",
            }}
          >
            {m === "portfolio" ? "Portfolio Optimizer" : "Scenario Optimizer"}
          </button>
        ))}
      </div>

      {mode === "portfolio" && (
        <>
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
        </>
      )}

      {mode === "scenario" && <ScenarioOptimizer cateVars={cateVars} />}
    </div>
  );
}

// ════════════════════════════════════════════════════════════════
// Scenario Optimizer — 4-step wizard
// ════════════════════════════════════════════════════════════════

type ScenStep = 1 | 2 | 3 | 4;

interface ScenarioWizState {
  /** Step 1: variable → magnitude map */
  interventions: Record<string, number>;
  scenarioName: string;
  /** Step 2: budget + unit costs + equity slider */
  budget: string;
  unitCosts: Record<string, number>;
  equityFocus: number;
  solver: "auto" | "greedy" | "greedy_2opt" | "milp";
  paretoSweep: boolean;
}

const SCEN_WIZ_KEY = "sparc:scenario-optimizer:wizard:v1";
const SCEN_DEFAULT: ScenarioWizState = {
  interventions: {},
  scenarioName: "My Scenario",
  budget: "",
  unitCosts: {},
  equityFocus: 0,
  solver: "auto",
  paretoSweep: true,
};

function loadScenWiz(): ScenarioWizState {
  try {
    const raw = localStorage.getItem(SCEN_WIZ_KEY);
    return raw ? { ...SCEN_DEFAULT, ...JSON.parse(raw) } : { ...SCEN_DEFAULT };
  } catch {
    return { ...SCEN_DEFAULT };
  }
}

function ScenarioOptimizer({ cateVars }: { cateVars: string[] }) {
  const { notify } = useNotification();
  const [step, setStep] = useState<ScenStep>(1);
  const [wiz, setWiz] = useState<ScenarioWizState>(() => loadScenWiz());
  const updateWiz = useCallback((patch: Partial<ScenarioWizState>) => {
    setWiz((prev) => ({ ...prev, ...patch }));
  }, []);

  useEffect(() => {
    try { localStorage.setItem(SCEN_WIZ_KEY, JSON.stringify(wiz)); } catch { /* ignore */ }
  }, [wiz]);

  // Step 3: equity layer
  const [equityRecords, setEquityRecords] = useState<EquityLayerRecord[]>([]);
  const [equityLoading, setEquityLoading] = useState(false);
  const [equityError, setEquityError] = useState<string | null>(null);

  const fetchEquity = useCallback(async () => {
    setEquityLoading(true);
    setEquityError(null);
    try {
      const res = await getEquityLayer();
      setEquityRecords(res.records ?? []);
    } catch (e) {
      setEquityError(e instanceof Error ? e.message : String(e));
      setEquityRecords([]);
    } finally {
      setEquityLoading(false);
    }
  }, []);

  // Step 4: results
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<OptimizeScenarioResponse | null>(null);
  const [runError, setRunError] = useState<string | null>(null);

  const runOptimize = useCallback(async () => {
    const budget = Number(wiz.budget);
    if (!Number.isFinite(budget) || budget <= 0) {
      notify("warning", "Enter a positive budget first");
      return;
    }
    if (Object.keys(wiz.interventions).length === 0) {
      notify("warning", "Add at least one intervention in Step 1");
      return;
    }
    setRunning(true);
    setRunError(null);
    try {
      const spec: RuntimeScenarioSpec = {
        name: wiz.scenarioName,
        interventions: wiz.interventions,
        unit_costs: wiz.unitCosts,
        budget,
        equity_focus: wiz.equityFocus,
      };
      const res = await optimizeScenario(spec, {
        solver: wiz.solver,
        pareto_sweep: wiz.paretoSweep,
      });
      setResult(res);
      notify("success", `Allocation complete — ${res.summary.n_cells_treated} cells treated`);
      setStep(4);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setRunError(msg);
      notify("error", msg);
    } finally {
      setRunning(false);
    }
  }, [wiz, notify]);

  const goTo = useCallback((s: ScenStep) => setStep(s), []);
  const reset = useCallback(() => {
    setWiz({ ...SCEN_DEFAULT });
    setResult(null);
    setRunError(null);
    setEquityRecords([]);
    setStep(1);
  }, []);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      {/* Step pills */}
      <div style={{ display: "flex", gap: 8 }}>
        {([
          [1, "1. Scenario Builder"],
          [2, "2. Budget & Costs"],
          [3, "3. Equity Layer"],
          [4, "4. Allocation"],
        ] as [ScenStep, string][]).map(([n, label]) => {
          const active = n === step;
          const done = n < step;
          return (
            <button
              key={n}
              onClick={() => goTo(n)}
              disabled={n > step}
              style={{
                flex: 1, padding: "10px 12px", borderRadius: 8, textAlign: "left",
                border: `1px solid ${active ? "var(--accent,#c9662b)" : "var(--line)"}`,
                background: active ? "#fff4ec" : done ? "#f7f3eb" : "#fff",
                cursor: n > step ? "default" : "pointer",
                opacity: n > step ? 0.5 : 1,
                fontSize: 12, fontWeight: active ? 700 : 400,
              }}
            >
              {done ? "✓ " : ""}{label}
            </button>
          );
        })}
      </div>

      {step === 1 && (
        <ScenStep1Builder
          vars={cateVars}
          interventions={wiz.interventions}
          scenarioName={wiz.scenarioName}
          onChange={(v) => updateWiz({ interventions: v })}
          onNameChange={(v) => updateWiz({ scenarioName: v })}
          onNext={() => goTo(2)}
        />
      )}

      {step === 2 && (
        <ScenStep2Budget
          wiz={wiz}
          updateWiz={updateWiz}
          onBack={() => goTo(1)}
          onNext={() => {
            void fetchEquity();
            goTo(3);
          }}
        />
      )}

      {step === 3 && (
        <ScenStep3Equity
          records={equityRecords}
          loading={equityLoading}
          error={equityError}
          equityFocus={wiz.equityFocus}
          onRefetch={fetchEquity}
          onEquityChange={(v) => updateWiz({ equityFocus: v })}
          onBack={() => goTo(2)}
          onRunOptimize={runOptimize}
          running={running}
        />
      )}

      {step === 4 && (
        <ScenStep4Allocation
          result={result}
          error={runError}
          running={running}
          onBack={() => goTo(3)}
          onReset={reset}
          onRerun={runOptimize}
        />
      )}
    </div>
  );
}

// ── ScenStep1Builder ───────────────────────────────────────────
function ScenStep1Builder({
  vars,
  interventions,
  scenarioName,
  onChange,
  onNameChange,
  onNext,
}: {
  vars: string[];
  interventions: Record<string, number>;
  scenarioName: string;
  onChange: (v: Record<string, number>) => void;
  onNameChange: (n: string) => void;
  onNext: () => void;
}) {
  const [newVar, setNewVar] = useState(vars[0] ?? "");
  useEffect(() => { if (!newVar && vars.length) setNewVar(vars[0]); }, [vars, newVar]);

  const addVar = () => {
    if (!newVar || newVar in interventions) return;
    onChange({ ...interventions, [newVar]: 1 });
  };

  const removeVar = (v: string) => {
    const next = { ...interventions };
    delete next[v];
    onChange(next);
  };

  const setMag = (v: string, mag: number) => {
    onChange({ ...interventions, [v]: mag });
  };

  const hasInterventions = Object.keys(interventions).length > 0;

  return (
    <Card>
      <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>
        Define Interventions
      </div>

      <div style={{ marginBottom: 14 }}>
        <label style={{ fontSize: 11, color: "var(--muted)", display: "block", marginBottom: 4 }}>
          Scenario name
        </label>
        <input
          value={scenarioName}
          onChange={(e) => onNameChange(e.target.value)}
          style={{ ...inputStyle, width: "100%", boxSizing: "border-box" }}
          placeholder="My Scenario"
        />
      </div>

      {/* Variable picker */}
      <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
        <select
          value={newVar}
          onChange={(e) => setNewVar(e.target.value)}
          style={{ ...inputStyle, flex: 1 }}
        >
          {vars.map((v) => <option key={v} value={v}>{v}</option>)}
          {vars.length === 0 && <option value="">— run Stage 3 first —</option>}
        </select>
        <Btn small onClick={addVar} disabled={!newVar || newVar in interventions}>
          Add
        </Btn>
      </div>

      {/* Intervention sliders */}
      {hasInterventions ? (
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
          <thead>
            <tr style={{ borderBottom: "1px solid var(--line)" }}>
              <th style={thStyle}>Variable</th>
              <th style={thStyle}>Change magnitude</th>
              <th style={{ ...thStyle, textAlign: "right" }}></th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(interventions).map(([v, mag]) => (
              <tr key={v} style={{ borderBottom: "1px solid var(--line)" }}>
                <td style={tdStyle}><Tag>{v}</Tag></td>
                <td style={tdStyle}>
                  <input
                    type="range"
                    min={-5} max={5} step={0.1}
                    value={mag}
                    onChange={(e) => setMag(v, Number(e.target.value))}
                    style={{ width: 120 }}
                  />
                  <span style={{ marginLeft: 8, color: mag < 0 ? "#c0392b" : "#2e7d32" }}>
                    {mag > 0 ? "+" : ""}{mag.toFixed(1)}
                  </span>
                </td>
                <td style={{ ...tdStyle, textAlign: "right" }}>
                  <button onClick={() => removeVar(v)} style={{ ...inputStyle, color: "#c0392b", background: "none", border: "none", cursor: "pointer" }}>✕</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <EmptyState title="No interventions yet" body="Add at least one treatment variable to continue." />
      )}

      <div style={{ display: "flex", justifyContent: "flex-end", marginTop: 16 }}>
        <Btn onClick={onNext} disabled={!hasInterventions}>Next →</Btn>
      </div>
    </Card>
  );
}

// ── ScenStep2Budget ────────────────────────────────────────────
function ScenStep2Budget({
  wiz,
  updateWiz,
  onBack,
  onNext,
}: {
  wiz: ScenarioWizState;
  updateWiz: (p: Partial<ScenarioWizState>) => void;
  onBack: () => void;
  onNext: () => void;
}) {
  const setUnitCost = (v: string, cost: number) => {
    updateWiz({ unitCosts: { ...wiz.unitCosts, [v]: cost } });
  };

  const budgetValid = Number(wiz.budget) > 0;

  return (
    <Card>
      <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>Budget & Costs</div>

      <div style={{ marginBottom: 14 }}>
        <label style={{ fontSize: 11, color: "var(--muted)", display: "block", marginBottom: 4 }}>
          Total budget ($)
        </label>
        <input
          type="number"
          min={0}
          value={wiz.budget}
          onChange={(e) => updateWiz({ budget: e.target.value })}
          style={{ ...inputStyle, width: 180 }}
          placeholder="e.g. 500000"
        />
      </div>

      {/* Unit costs per intervention variable */}
      {Object.keys(wiz.interventions).length > 0 && (
        <div style={{ marginBottom: 14 }}>
          <div style={{ fontSize: 11, color: "var(--muted)", marginBottom: 6 }}>
            Unit cost per 1.0 change ($)
          </div>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: "1px solid var(--line)" }}>
                <th style={thStyle}>Variable</th>
                <th style={thStyle}>Cost / unit</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(wiz.interventions).map(([v]) => (
                <tr key={v} style={{ borderBottom: "1px solid var(--line)" }}>
                  <td style={tdStyle}><Tag>{v}</Tag></td>
                  <td style={tdStyle}>
                    <input
                      type="number"
                      min={0}
                      value={wiz.unitCosts[v] ?? ""}
                      onChange={(e) => setUnitCost(v, Number(e.target.value))}
                      style={{ ...inputStyle, width: 100 }}
                      placeholder="e.g. 5000"
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <div style={{ fontSize: 10, color: "var(--muted)", marginTop: 4 }}>
            Leave blank to use abstract cost = 1 (budget = cell count cap).
          </div>
        </div>
      )}

      {/* Equity focus */}
      <div style={{ marginBottom: 14 }}>
        <label style={{ fontSize: 11, color: "var(--muted)", display: "block", marginBottom: 4 }}>
          Equity focus (α) — blends efficiency ↔ equity-first
        </label>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: 11 }}>Efficiency</span>
          <input
            type="range" min={0} max={1} step={0.05}
            value={wiz.equityFocus}
            onChange={(e) => updateWiz({ equityFocus: Number(e.target.value) })}
            style={{ flex: 1 }}
          />
          <span style={{ fontSize: 11 }}>Equity</span>
          <span style={{ width: 36, textAlign: "right", fontSize: 12, fontWeight: 600 }}>
            {(wiz.equityFocus * 100).toFixed(0)}%
          </span>
        </div>
      </div>

      {/* Solver */}
      <div style={{ marginBottom: 14 }}>
        <label style={{ fontSize: 11, color: "var(--muted)", display: "block", marginBottom: 4 }}>
          Solver
        </label>
        <select
          value={wiz.solver}
          onChange={(e) => updateWiz({ solver: e.target.value as ScenarioWizState["solver"] })}
          style={inputStyle}
        >
          <option value="auto">Auto (recommended)</option>
          <option value="greedy">Greedy</option>
          <option value="greedy_2opt">Greedy + 2-opt</option>
          <option value="milp">MILP (slow, exact, n≤5000)</option>
        </select>
      </div>

      <div style={{ display: "flex", justifyContent: "space-between", marginTop: 16 }}>
        <Btn small onClick={onBack}>← Back</Btn>
        <Btn onClick={onNext} disabled={!budgetValid}>Next →</Btn>
      </div>
    </Card>
  );
}

// ── ScenStep3Equity ────────────────────────────────────────────
function ScenStep3Equity({
  records,
  loading,
  error,
  equityFocus,
  onRefetch,
  onEquityChange,
  onBack,
  onRunOptimize,
  running,
}: {
  records: EquityLayerRecord[];
  loading: boolean;
  error: string | null;
  equityFocus: number;
  onRefetch: () => void;
  onEquityChange: (v: number) => void;
  onBack: () => void;
  onRunOptimize: () => void;
  running: boolean;
}) {
  const avgEquity = records.length
    ? records.reduce((s, r) => s + r.equity_score, 0) / records.length
    : null;
  const avgPoverty = records.filter((r) => r.poverty_rate != null).length
    ? records.filter((r) => r.poverty_rate != null).reduce((s, r) => s + (r.poverty_rate ?? 0), 0) /
      records.filter((r) => r.poverty_rate != null).length
    : null;
  const avgMinority = records.filter((r) => r.minority_pct != null).length
    ? records.filter((r) => r.minority_pct != null).reduce((s, r) => s + (r.minority_pct ?? 0), 0) /
      records.filter((r) => r.minority_pct != null).length
    : null;

  return (
    <Card>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <div style={{ fontSize: 14, fontWeight: 700 }}>Equity Layer</div>
        <Btn small onClick={onRefetch} disabled={loading}>
          {loading ? "Loading…" : "Refresh"}
        </Btn>
      </div>

      {error && (
        <div style={{ padding: "8px 12px", borderRadius: 6, background: "#fff0f0", color: "#c0392b", fontSize: 12, marginBottom: 10 }}>
          {error} — equity weighting will use uniform scores.
        </div>
      )}

      {records.length > 0 && (
        <StatGrid>
          <Stat label="Cells" value={records.length.toLocaleString()} />
          {avgEquity != null && <Stat label="Avg equity score" value={(avgEquity * 100).toFixed(1) + "%"} />}
          {avgPoverty != null && <Stat label="Avg poverty rate" value={(avgPoverty * 100).toFixed(1) + "%"} />}
          {avgMinority != null && <Stat label="Avg minority %" value={(avgMinority * 100).toFixed(1) + "%"} />}
        </StatGrid>
      )}

      {records.length === 0 && !loading && !error && (
        <div style={{ fontSize: 12, color: "var(--muted)", padding: "12px 0" }}>
          No equity layer loaded. Allocation will use uniform equity scores (0.5).
          Click Refresh to fetch Census tract data.
        </div>
      )}

      <div style={{ marginTop: 14 }}>
        <label style={{ fontSize: 11, color: "var(--muted)", display: "block", marginBottom: 4 }}>
          Equity focus (α) — override from Step 2
        </label>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: 11 }}>Efficiency</span>
          <input
            type="range" min={0} max={1} step={0.05}
            value={equityFocus}
            onChange={(e) => onEquityChange(Number(e.target.value))}
            style={{ flex: 1 }}
          />
          <span style={{ fontSize: 11 }}>Equity</span>
          <span style={{ width: 36, textAlign: "right", fontSize: 12, fontWeight: 600 }}>
            {(equityFocus * 100).toFixed(0)}%
          </span>
        </div>
      </div>

      <div style={{ display: "flex", justifyContent: "space-between", marginTop: 16 }}>
        <Btn small onClick={onBack}>← Back</Btn>
        <Btn onClick={onRunOptimize} disabled={running}>
          {running ? "Running…" : "Run Allocation →"}
        </Btn>
      </div>
    </Card>
  );
}

// ── ScenStep4Allocation ───────────────────────────────────────
function ScenStep4Allocation({
  result,
  error,
  running,
  onBack,
  onReset,
  onRerun,
}: {
  result: OptimizeScenarioResponse | null;
  error: string | null;
  running: boolean;
  onBack: () => void;
  onReset: () => void;
  onRerun: () => void;
}) {
  if (running) {
    return (
      <Card>
        <EmptyState title="Running optimizer…" body="Computing equity-modulated allocation. This may take a moment." tone="waiting" />
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <div style={{ color: "#c0392b", fontSize: 13, marginBottom: 12 }}>Error: {error}</div>
        <div style={{ display: "flex", gap: 8 }}>
          <Btn small onClick={onBack}>← Back</Btn>
          <Btn small onClick={onRerun}>Retry</Btn>
        </div>
      </Card>
    );
  }

  if (!result) {
    return (
      <Card>
        <EmptyState title="No allocation results yet" body="Run the optimizer from Step 3." />
        <div style={{ marginTop: 12 }}><Btn small onClick={onBack}>← Back</Btn></div>
      </Card>
    );
  }

  const s = result.summary;
  const paretoPoints = result.pareto?.points ?? [];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <Card>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
          <div style={{ fontSize: 14, fontWeight: 700 }}>Allocation: {result.scenario_name}</div>
          <div style={{ display: "flex", gap: 8 }}>
            <Btn small onClick={onBack}>← Back</Btn>
            <Btn small onClick={onRerun}>Re-run</Btn>
            <Btn small onClick={onReset}>New scenario</Btn>
          </div>
        </div>
        <StatGrid>
          <Stat label="Budget" value={`$${result.budget.toLocaleString()}`} />
          <Stat label="Total cost" value={`$${s.total_estimated_cost.toLocaleString(undefined, { maximumFractionDigits: 0 })}`} />
          <Stat label="Cells treated" value={s.n_cells_treated.toLocaleString()} />
          <Stat label="Fully treated" value={s.n_cells_fully_treated.toLocaleString()} />
          <Stat label="Projected Δ" value={s.total_projected_delta.toFixed(3)} />
          <Stat label="Allocation Gini" value={s.allocation_gini.toFixed(3)} />
          <Stat label="Equity Gini" value={s.equity_gini.toFixed(3)} />
          <Stat label="Solver" value={s.solver} />
        </StatGrid>
      </Card>

      {/* Allocation map */}
      {result.allocation_geojson && (
        <Card>
          <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>Allocation Map</div>
          <SpatialMap
            geojson={result.allocation_geojson}
            colorField="allocation"
            palette="viridis"
            height={typeof MAP_HEIGHT_DEFAULT === "number" ? `${MAP_HEIGHT_DEFAULT}px` : MAP_HEIGHT_DEFAULT}
          />
          <div style={{ fontSize: 10, color: "var(--muted)", marginTop: 6 }}>
            Colour: allocation fraction [0–1] per grid cell.
            Hover for equity_score, projected_delta, estimated_cost.
          </div>
        </Card>
      )}

      {/* Pareto frontier */}
      {paretoPoints.length > 0 && (
        <Card>
          <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 4 }}>Pareto Frontier — budget vs. benefit</div>
          <ParetoChart points={paretoPoints} />
        </Card>
      )}

      {/* Export */}
      <Card>
        <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>Export</div>
        <a
          href={URL.createObjectURL(new Blob([JSON.stringify(result.allocation_geojson)], { type: "application/geo+json" }))}
          download={`allocation_${result.scenario_name.replace(/\s+/g, "_")}.geojson`}
          style={{ fontSize: 12, color: "var(--accent,#c9662b)", textDecoration: "underline" }}
        >
          Download allocation GeoJSON
        </a>
      </Card>
    </div>
  );
}

// Reuse ParetoChart type-safely
interface _AP { budget: number; total_benefit: number; n_treated: number; gini: number }
function ParetoChart({ points }: { points: _AP[] }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || points.length === 0) return;
    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth || 400;
    const h = 160;
    canvas.width = w * DPR;
    canvas.height = h * DPR;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(DPR, DPR);
    ctx.clearRect(0, 0, w, h);
    const pad = 30;
    const xs = points.map((p) => p.budget);
    const ys = points.map((p) => p.total_benefit);
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    const yMin = Math.min(...ys), yMax = Math.max(...ys);
    const sx = (x: number) => pad + ((x - xMin) / Math.max(1e-9, xMax - xMin)) * (w - pad * 2);
    const sy = (y: number) => h - pad - ((y - yMin) / Math.max(1e-9, yMax - yMin)) * (h - pad * 2);
    ctx.strokeStyle = "#ccc";
    ctx.beginPath();
    ctx.moveTo(pad, pad / 2);
    ctx.lineTo(pad, h - pad);
    ctx.lineTo(w - pad / 2, h - pad);
    ctx.stroke();
    ctx.strokeStyle = "var(--accent, #c9662b)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    points.forEach((p, i) => { i === 0 ? ctx.moveTo(sx(p.budget), sy(p.total_benefit)) : ctx.lineTo(sx(p.budget), sy(p.total_benefit)); });
    ctx.stroke();
    ctx.fillStyle = "var(--accent, #c9662b)";
    points.forEach((p) => {
      ctx.beginPath();
      ctx.arc(sx(p.budget), sy(p.total_benefit), 3, 0, Math.PI * 2);
      ctx.fill();
    });
    ctx.fillStyle = "#6e6358";
    ctx.font = "10px Inter, sans-serif";
    ctx.fillText("Budget →", w - 56, h - 6);
  }, [points]);
  return <canvas ref={ref} style={{ width: "100%", height: 160 }} />;
}

// ════════════════════════════════════════════════════════════════
// Stepper
// ════════════════════════════════════════════════════════════════
function Stepper({ current, onNavigate }: { current: Step; onNavigate: (s: Step) => void }) {
  const steps: Array<{ n: Step; label: string; sub: string }> = [
    { n: 1, label: "Candidates", sub: "Load + edit intervention list" },
    { n: 2, label: "Constraints", sub: "Budget, robustness, equity" },
    { n: 3, label: "Recommendation", sub: "Ranked portfolio + spatial targeting" },
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
      title="Step 1 — Candidate Interventions"
      subtitle="Loaded from scenario + posterior outputs. Edit cost or equity weight in place."
      actions={
        <div style={{ display: "flex", gap: 8 }}>
          <Btn small onClick={onReload} disabled={loading}>
            {loading ? "Loading…" : "Reload"}
          </Btn>
          <Btn small primary onClick={onNext} disabled={!canAdvance}>
            Next: Constraints →
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
          title="No candidates yet"
          body="Run the pipeline through Stage 4 so scenarios + CATE posterior means are available."
        />
      ) : (
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
            <thead>
              <tr style={{ textAlign: "left", borderBottom: "1px solid var(--line)" }}>
                <th style={thStyle}>Name</th>
                <th style={thStyle}>Treatment</th>
                <th style={thStyle}>Magnitude</th>
                <th style={thStyle}>Mean effect</th>
                <th style={thStyle}>σ effect</th>
                <th style={thStyle}>Cost</th>
                <th style={thStyle}>Equity wt.</th>
              </tr>
            </thead>
            <tbody>
              {candidates.map((c) => {
                const baseline = rawCandidates.find((r) => r.name === c.name);
                return (
                  <tr key={c.name} style={{ borderBottom: "1px solid #eee" }}>
                    <td style={tdStyle}>{c.name}</td>
                    <td style={tdStyle}>
                      <Tag>{c.treatment}</Tag>
                    </td>
                    <td style={tdStyle}>{c.magnitude.toFixed(3)}</td>
                    <td style={tdStyle}>{c.mean_effect.toFixed(4)}</td>
                    <td style={tdStyle}>{c.effect_std.toFixed(4)}</td>
                    <td style={tdStyle}>
                      <NumberField
                        value={c.cost}
                        baseline={baseline?.cost ?? c.cost}
                        step={0.1}
                        onChange={(v) => onEdit(c.name, { cost: v })}
                      />
                    </td>
                    <td style={tdStyle}>
                      <NumberField
                        value={c.equity_weight}
                        baseline={baseline?.equity_weight ?? c.equity_weight}
                        step={0.05}
                        onChange={(v) => onEdit(c.name, { equity_weight: v })}
                      />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}

// ════════════════════════════════════════════════════════════════
// Step 2 — Constraints
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
  const sourceList = wiz.benefitSource === "cate" ? cateVars : [];
  const sourceUnavailable = wiz.benefitSource !== "uniform" && sourceList.length === 0;

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
      <Card title="Step 2a — Budget & Direction" subtitle="Controls the causal ranking.">
        <FieldRow label="Budget (optional, total cost cap)">
          <input
            type="number"
            value={wiz.budget}
            onChange={(e) => updateWiz({ budget: e.target.value })}
            placeholder="Unconstrained"
            style={inputStyle}
          />
        </FieldRow>
        <FieldRow label={`Robustness λ — ${wiz.robustness.toFixed(2)}`}>
          <input
            type="range"
            min={0}
            max={2}
            step={0.1}
            value={wiz.robustness}
            onChange={(e) => updateWiz({ robustness: Number(e.target.value) })}
            style={{ width: "100%" }}
          />
        </FieldRow>
        <FieldRow label="Objective">
          <label style={{ fontSize: 12, display: "inline-flex", gap: 6, alignItems: "center" }}>
            <input
              type="checkbox"
              checked={wiz.minimise}
              onChange={(e) => updateWiz({ minimise: e.target.checked })}
            />
            Minimise effect (otherwise maximise)
          </label>
        </FieldRow>
        <FieldRow label={`Uncertainty draws — ${wiz.nDraws}`}>
          <input
            type="range"
            min={100}
            max={2000}
            step={100}
            value={wiz.nDraws}
            onChange={(e) => updateWiz({ nDraws: Number(e.target.value) })}
            style={{ width: "100%" }}
          />
        </FieldRow>
      </Card>

      <Card title="Step 2b — Budget Allocator" subtitle="Allocates capital across grid cells.">
        <FieldRow label="Benefit source">
          <div style={{ display: "flex", gap: 6 }}>
            {(["cate", "uniform"] as BenefitSource[]).map((src) => (
              <button
                key={src}
                onClick={() => updateWiz({ benefitSource: src })}
                style={{
                  padding: "4px 10px",
                  fontSize: 11,
                  borderRadius: 6,
                  border: `1px solid ${wiz.benefitSource === src ? "var(--accent, #c9662b)" : "var(--line)"}`,
                  background: wiz.benefitSource === src ? "#fff4ec" : "#fff",
                  cursor: "pointer",
                }}
                title={
                  src === "cate"
                    ? "Bayesian local treatment effect β(s) from Stage 3 NUTS posterior"
                    : "Uniform unit benefit — sanity check / equity reference only"
                }
              >
                {src === "cate" ? "Bayesian β(s)" : "Uniform"}
              </button>
            ))}
          </div>
        </FieldRow>
        {wiz.benefitSource !== "uniform" && (
          <FieldRow label={`Treatment variable${sourceUnavailable ? " (none available)" : ""}`}>
            <select
              value={wiz.benefitVar}
              onChange={(e) => updateWiz({ benefitVar: e.target.value })}
              style={inputStyle}
              disabled={sourceUnavailable}
            >
              {sourceList.map((v) => (
                <option key={v} value={v}>{v}</option>
              ))}
            </select>
          </FieldRow>
        )}
        <FieldRow label="Solver">
          <select
            value={wiz.solver}
            onChange={(e) => updateWiz({ solver: e.target.value as Solver })}
            style={inputStyle}
          >
            <option value="auto">Auto</option>
            <option value="greedy">Greedy</option>
            <option value="greedy_2opt">Greedy + 2-opt</option>
            <option value="milp">MILP (exact)</option>
          </select>
        </FieldRow>
        <FieldRow label="Cost column (optional)">
          <select
            value={wiz.costColumn}
            onChange={(e) => updateWiz({ costColumn: e.target.value })}
            style={inputStyle}
          >
            <option value="">— uniform —</option>
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
      </Card>

      <Card
        title="Step 2c — Equity Layers"
        subtitle="Provide one numeric value per candidate (comma or space separated). Leaving blank skips the layer."
        style={{ gridColumn: "1 / -1" }}
      >
        {Object.keys(wiz.layerText).map((name) => (
          <FieldRow key={name} label={name}>
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <input
                type="text"
                value={wiz.layerText[name]}
                onChange={(e) => updateWiz({ layerText: { ...wiz.layerText, [name]: e.target.value } })}
                placeholder="e.g. 120, 80, 200, 50"
                style={{ ...inputStyle, flex: 1 }}
              />
              <label style={{ fontSize: 11, display: "inline-flex", gap: 4 }}>
                <input
                  type="checkbox"
                  checked={!!wiz.layerInvert[name]}
                  onChange={(e) =>
                    updateWiz({ layerInvert: { ...wiz.layerInvert, [name]: e.target.checked } })
                  }
                />
                Invert
              </label>
            </div>
          </FieldRow>
        ))}
      </Card>

      <div style={{ gridColumn: "1 / -1", display: "flex", gap: 8, justifyContent: "flex-end" }}>
        <Btn small onClick={onBack}>← Back</Btn>
        <Btn small primary onClick={onNext}>Next: Recommendation →</Btn>
      </div>
    </div>
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
        title="Step 3 — Recommendation"
        subtitle={running ? "Crunching causal + budget optimizers…" : "Results from all parallel jobs."}
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
            <Stat label="Ranked" value={`${rankedRows.length}`} />
            <Stat label="Selected" value={`${rankedRows.filter((r) => r.selected).length}`} />
            <Stat
              label="Top pick effect"
              value={topPick ? topPick.risk_adjusted_effect.toFixed(4) : "—"}
              sub={topPick?.name}
            />
            <Stat
              label="Equity Gini"
              value={ranked.equity_summary.gini == null ? "—" : ranked.equity_summary.gini.toFixed(3)}
            />
          </StatGrid>
        )}
      </Card>

      {ranked && (
        <Card title="Ranked Portfolio" subtitle="Risk-adjusted causal effects × equity weights.">
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
              <thead>
                <tr style={{ textAlign: "left", borderBottom: "1px solid var(--line)" }}>
                  <th style={thStyle}>Rank</th>
                  <th style={thStyle}>Name</th>
                  <th style={thStyle}>Score</th>
                  <th style={thStyle}>Risk-adj effect</th>
                  <th style={thStyle}>Cost</th>
                  <th style={thStyle}>Equity wt.</th>
                  <th style={thStyle}>Effect p10 / p90</th>
                  <th style={thStyle}>Selected</th>
                </tr>
              </thead>
              <tbody>
                {rankedRows.map((r, i) => {
                  const u = uncByName.get(r.name);
                  return (
                    <tr
                      key={r.name}
                      style={{ borderBottom: "1px solid #eee", background: r.selected ? "#f1f7e8" : undefined }}
                    >
                      <td style={tdStyle}>{i + 1}</td>
                      <td style={tdStyle}>{r.name}</td>
                      <td style={tdStyle}>{r.score.toFixed(4)}</td>
                      <td style={tdStyle}>{r.risk_adjusted_effect.toFixed(4)}</td>
                      <td style={tdStyle}>{r.cost.toFixed(2)}</td>
                      <td style={tdStyle}>{r.equity_weight.toFixed(2)}</td>
                      <td style={tdStyle}>
                        {u ? `${u.effect_p10.toFixed(3)} / ${u.effect_p90.toFixed(3)}` : "—"}
                      </td>
                      <td style={tdStyle}>{r.selected ? "✓" : ""}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {ranked && !budget && !running && (
        <Card
          title="Budget Optimization"
          subtitle="not yet computed for this run"
        >
          <div style={{ fontSize: 12.5, color: "var(--muted)", lineHeight: 1.6 }}>
            To produce a spatial budget allocation, return to{" "}
            <strong>Step 2b — Budget Allocator</strong> and provide a positive
            budget value plus a benefit source (Bayesian β or uniform). Then
            re-run the recommendation.
          </div>
        </Card>
      )}

      {budget && (
        <Card
          title="Budget Optimization · Allocation"
          subtitle={`${budget.benefit_description} — solver: ${budget.result.solver}`}
        >
          <StatGrid>
            <Stat label="Treated cells" value={`${budget.result.n_treated}/${budget.n_cells}`} />
            <Stat label="Fully treated" value={`${budget.result.n_fully_treated}`} />
            <Stat label="Total benefit" value={budget.result.total_benefit.toFixed(3)} />
            <Stat label="Spend" value={`${budget.result.total_cost.toFixed(2)} / ${budget.result.budget.toFixed(2)}`} />
            <Stat label="Gini" value={budget.result.gini.toFixed(3)} />
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
          title="ACS Demographic Context"
          subtitle={`${census.context.n_counties} county/ies · vintage ${census.context.vintage}`}
        >
          <StatGrid>
            <Stat
              label="Population"
              value={census.context.population == null ? "—" : census.context.population.toLocaleString()}
            />
            <Stat
              label="Median income"
              value={
                census.context.median_income == null
                  ? "—"
                  : `$${census.context.median_income.toLocaleString()}`
              }
            />
            <Stat
              label="Poverty rate"
              value={
                census.context.poverty_rate == null
                  ? "—"
                  : `${(census.context.poverty_rate * 100).toFixed(1)}%`
              }
            />
            <Stat
              label="Mobile-home share"
              value={
                census.context.mobile_home_share == null
                  ? "—"
                  : `${(census.context.mobile_home_share * 100).toFixed(1)}%`
              }
            />
          </StatGrid>
        </Card>
      )}

      <Card
        title="Spatial Targeting"
        subtitle="Top-K grid cells ranked by CATE ÷ cost."
        actions={
          <div style={{ display: "flex", gap: 8 }}>
            <select
              value={wiz.activeTargetVar}
              onChange={(e) => updateWiz({ activeTargetVar: e.target.value })}
              style={{ ...inputStyle, maxWidth: 160 }}
              disabled={cateVars.length === 0}
            >
              {cateVars.length === 0 ? <option value="">— no CATE maps —</option> : cateVars.map((v) => (
                <option key={v} value={v}>{v}</option>
              ))}
            </select>
            <input
              type="number"
              value={wiz.topK}
              min={1}
              max={500}
              onChange={(e) => updateWiz({ topK: Math.max(1, Number(e.target.value) || 1) })}
              style={{ ...inputStyle, width: 80 }}
            />
          </div>
        }
      >
        {targetingLoading ? (
          <div style={{ padding: 24, textAlign: "center", color: "var(--muted)" }}>Loading targeting…</div>
        ) : targeting ? (
          <div ref={targetingBlockRef} style={{ height: MAP_HEIGHT_DEFAULT, position: "relative" }}>
            <div style={{ position: "absolute", top: 8, right: 50, zIndex: 6 }}>
              <ExportBlockButton targetRef={targetingBlockRef} artifactId="decision_targeting" label={`targeting_${wiz.activeTargetVar || "map"}`} compact />
            </div>
            <SpatialMap geojson={targeting} expandable />
          </div>
        ) : (
          <EmptyState title="No targeting map" body="Pick a treatment variable with a CATE map." />
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
