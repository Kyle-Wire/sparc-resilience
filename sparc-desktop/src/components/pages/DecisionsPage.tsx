import { useCallback, useEffect, useMemo, useState } from "react";
import { SectionHeader, Card, Btn, Stat, StatGrid, KeyVal, Tag } from "@/components/ui/DesignSystem";
import ExplainButton from "@/components/common/ExplainButton";
import {
  getDecisionCandidates,
  optimizeDecisions,
  computeEquity,
  getCensusEquity,
  getCateMapVariables,
  getTargeting,
  decisionUncertainty,
  type InterventionCandidate,
  type OptimizerResponse,
  type RankedIntervention,
  type EquityResponse,
  type CensusEquityResponse,
  type TargetingResponse,
  type UncertaintyRecord,
} from "@/lib/api";
import { usePipeline } from "@/hooks/PipelineProvider";
import { useNotification } from "@/hooks/useNotifications";
import SpatialMap from "@/components/map/SpatialMap";

/**
 * Decisions page (Phase 9).
 *
 * Surfaces the causal optimizer: ranks candidate interventions by
 * (mean causal effect − λ·σ) / cost · equity_weight.  Inputs come from
 * existing scenario + NUTS posterior outputs so the score is causal,
 * not correlation-based.
 */
export default function DecisionsPage() {
  const { runEndedAt } = usePipeline();
  const { notify } = useNotification();

  const [candidates, setCandidates] = useState<InterventionCandidate[]>([]);
  const [optimized, setOptimized] = useState<OptimizerResponse | null>(null);
  const [budget, setBudget] = useState<string>("");
  const [robustness, setRobustness] = useState(0);
  const [minimise, setMinimise] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Equity layer state — raw text per layer, parsed lazily on apply.
  const [layerText, setLayerText] = useState<Record<string, string>>({
    population: "",
    vulnerability: "",
  });
  const [layerInvert, setLayerInvert] = useState<Record<string, boolean>>({
    population: false,
    vulnerability: false,
  });
  const [equityResult, setEquityResult] = useState<EquityResponse | null>(null);

  // Targeting state — spatial deployment priority based on CATE ÷ cost.
  const [targetingVars, setTargetingVars] = useState<string[]>([]);
  const [activeTargetVar, setActiveTargetVar] = useState<string>("");
  const [topK, setTopK] = useState<number>(50);
  const [targeting, setTargeting] = useState<TargetingResponse | null>(null);
  const [targetingLoading, setTargetingLoading] = useState(false);
  const [uncertainty, setUncertainty] = useState<UncertaintyRecord[]>([]);
  const [nDraws, setNDraws] = useState<number>(500);

  // Census ACS auto-fetch — area-wide demographic context.
  const [census, setCensus] = useState<CensusEquityResponse | null>(null);
  const [censusLoading, setCensusLoading] = useState(false);

  const loadCandidates = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await getDecisionCandidates();
      setCandidates(res.candidates ?? []);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      setCandidates([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { loadCandidates(); }, [loadCandidates, runEndedAt]);

  // Load list of treatment variables that have a spatial CATE map.
  useEffect(() => {
    getCateMapVariables()
      .then((res) => {
        const vars = res.variables ?? [];
        setTargetingVars(vars);
        if (vars.length > 0 && !activeTargetVar) setActiveTargetVar(vars[0]);
      })
      .catch(() => setTargetingVars([]));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runEndedAt]);

  // Refetch targeting GeoJSON when variable or top-K changes.
  useEffect(() => {
    if (!activeTargetVar) { setTargeting(null); return; }
    setTargetingLoading(true);
    getTargeting(activeTargetVar, topK)
      .then((g) => setTargeting(g))
      .catch(() => setTargeting(null))
      .finally(() => setTargetingLoading(false));
  }, [activeTargetVar, topK]);

  const runOptimizer = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const requestBody = {
        candidates,
        budget: budget.trim() === "" ? null : Number(budget),
        robustness_lambda: robustness,
        minimise,
      };
      const res = await optimizeDecisions(requestBody);
      setOptimized(res);
      notify("success", `Ranked ${res.ranked.length} candidates`);
      // Run Monte-Carlo uncertainty in parallel; failure here shouldn't block.
      try {
        const mc = await decisionUncertainty({ ...requestBody, n_draws: nDraws });
        setUncertainty(mc.uncertainty);
      } catch (mcErr) {
        const msg = mcErr instanceof Error ? mcErr.message : String(mcErr);
        notify("warning", `Uncertainty MC failed: ${msg}`);
        setUncertainty([]);
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      notify("error", msg);
    } finally {
      setLoading(false);
    }
  }, [candidates, budget, robustness, minimise, nDraws, notify]);

  const ranked: RankedIntervention[] = optimized?.ranked ?? [];
  const topPick = ranked[0];

  const updateCandidate = useCallback(
    (idx: number, patch: Partial<InterventionCandidate>) => {
      setCandidates((prev) => prev.map((c, i) => (i === idx ? { ...c, ...patch } : c)));
    },
    [],
  );

  const totalSelectedCost = useMemo(
    () => ranked.filter((r) => r.selected).reduce((acc, r) => acc + r.cost, 0),
    [ranked],
  );

  const uncertaintyByName = useMemo(() => {
    const m = new Map<string, UncertaintyRecord>();
    for (const u of uncertainty) m.set(u.candidate, u);
    return m;
  }, [uncertainty]);

  const applyEquityLayers = useCallback(async () => {
    if (candidates.length === 0) {
      notify("warning", "Load candidates first");
      return;
    }
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
      notify("warning", `Provide ${names.length} numeric values per layer (comma or space separated)`);
      return;
    }
    try {
      const eq = await computeEquity({ candidate_names: names, layers, invert });
      setEquityResult(eq);
      // Patch candidate equity_weight from the equity service.
      setCandidates((prev) =>
        prev.map((c) => {
          const m = eq.scores.find((s) => s.candidate === c.name);
          return m ? { ...c, equity_weight: m.weight } : c;
        }),
      );
      notify("success", `Applied equity — disparity ${eq.disparity_index.toFixed(3)}`);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      notify("error", msg);
    }
  }, [candidates, layerText, layerInvert, notify]);

  // Census ACS auto-fetch — pulls area demographics, pre-fills layer
  // textareas with the same value for each candidate (uniform area
  // context), so the user can immediately Apply.
  const fetchCensus = useCallback(async () => {
    if (candidates.length === 0) {
      notify("warning", "Load candidates first");
      return;
    }
    setCensusLoading(true);
    try {
      const res = await getCensusEquity();
      setCensus(res);
      // Pre-fill textareas: replicate area-wide values per candidate.
      const n = candidates.length;
      setLayerText((prev) => {
        const next = { ...prev };
        for (const [name, value] of Object.entries(res.layers)) {
          next[name] = Array(n).fill(value.toFixed(4)).join(", ");
        }
        return next;
      });
      setLayerInvert((prev) => ({ ...prev, ...res.invert }));
      const ctx = res.context;
      const counties = ctx.counties.map((c) => c.name).join(", ") || "(none)";
      notify(
        "success",
        `Census ${ctx.vintage}: ${counties} · pop ${ctx.population?.toLocaleString() ?? "—"} · poverty ${((ctx.poverty_rate ?? 0) * 100).toFixed(1)}%`,
      );
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      notify("error", `Census fetch failed: ${msg}`);
    } finally {
      setCensusLoading(false);
    }
  }, [candidates, notify]);

  return (
    <div style={{ padding: 14, display: "flex", flexDirection: "column", gap: 12 }}>
      <SectionHeader
        kicker="DECISIONS"
        label="rank candidate interventions by causal benefit per cost"
        right={
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            {topPick && (
              <ExplainButton
                prompt={
                  `Explain why ${topPick.name} is the top-ranked intervention. ` +
                  `Score=${topPick.score.toFixed(3)} (mean effect µ=${topPick.mean_effect.toFixed(3)}, ` +
                  `σ=${topPick.effect_std.toFixed(3)}, cost=${topPick.cost}, equity_weight=${topPick.equity_weight.toFixed(2)}). ` +
                  `What is the causal mechanism, and how should the user interpret the uncertainty?`
                }
                label="Explain top pick"
              />
            )}
            <Btn onClick={loadCandidates} disabled={loading}>
              Refresh candidates
            </Btn>
            <Btn primary onClick={runOptimizer} disabled={loading || candidates.length === 0}>
              {loading ? "Optimizing…" : "Run optimizer"}
            </Btn>
          </div>
        }
      />

      <StatGrid>
        <Stat label="Candidates" value={String(candidates.length)} tint="var(--ink)" />
        <Stat
          label="Top score"
          value={topPick ? topPick.score.toFixed(3) : "—"}
          sub={topPick?.name}
          tint="var(--crimson)"
        />
        <Stat
          label="Selected"
          value={optimized ? String(optimized.settings.n_selected) : "—"}
          sub={optimized && optimized.settings.budget != null
            ? `Σcost ${totalSelectedCost.toFixed(1)} ≤ ${optimized.settings.budget}`
            : undefined}
          tint="var(--purple)"
        />
        <Stat
          label="Equity gini"
          value={optimized?.equity_summary.gini != null ? optimized.equity_summary.gini.toFixed(3) : "—"}
          tint="var(--amber)"
          sub="0 = perfect equality"
        />
      </StatGrid>

      {error && (
        <Card title="Optimizer error" subtitle="last attempt failed">
          <div style={{ fontSize: 12, color: "var(--crimson)", fontFamily: "'JetBrains Mono', monospace" }}>
            {error}
          </div>
        </Card>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", gap: 14 }}>
        <Card title="Optimizer settings" subtitle="risk-neutral by default">
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <KeyVal
              label="Budget cap"
              value={
                <input
                  type="number"
                  value={budget}
                  onChange={(e) => setBudget(e.target.value)}
                  placeholder="∞"
                  style={{
                    border: "1px solid var(--line)", borderRadius: 4,
                    padding: "3px 6px", fontSize: 12, width: 100,
                    fontFamily: "'JetBrains Mono', monospace",
                  }}
                />
              }
            />
            <KeyVal
              label="Robustness λ"
              value={
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <input
                    type="range"
                    min={0} max={3} step={0.1}
                    value={robustness}
                    onChange={(e) => setRobustness(Number(e.target.value))}
                    style={{ width: 120 }}
                  />
                  <span className="mono" style={{ fontSize: 11, minWidth: 32 }}>
                    {robustness.toFixed(1)}
                  </span>
                </div>
              }
            />
            <KeyVal
              label="Direction"
              value={
                <div style={{ display: "flex", gap: 6 }}>
                  <button
                    onClick={() => setMinimise(false)}
                    style={pillBtn(!minimise)}
                  >
                    maximise effect
                  </button>
                  <button
                    onClick={() => setMinimise(true)}
                    style={pillBtn(minimise)}
                  >
                    minimise (e.g. cool UHI)
                  </button>
                </div>
              }
            />
            <KeyVal
              label="MC draws"
              value={
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <input
                    type="range"
                    min={50} max={2000} step={50}
                    value={nDraws}
                    onChange={(e) => setNDraws(Number(e.target.value))}
                    style={{ width: 120 }}
                  />
                  <span className="mono" style={{ fontSize: 11, minWidth: 40 }}>{nDraws}</span>
                </div>
              }
            />
            <div style={{ marginTop: 6, fontSize: 10.5, color: "var(--muted)", lineHeight: 1.5 }}>
              <strong>Score</strong> = equity_weight · (μ − λ·σ) / cost.<br />
              μ comes from the dose-response / scenario posterior (causal),
              σ from the NUTS posterior std. λ &gt; 0 prefers tighter
              credible intervals (robust to uncertainty).
            </div>
          </div>
        </Card>

        <Card title="Equity summary" subtitle={optimized ? `n=${optimized.equity_summary.n}` : "run the optimizer"}>
          {optimized ? (
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              <KeyVal label="Mean weight" value={optimized.equity_summary.mean?.toFixed(3) ?? "—"} />
              <KeyVal label="Min" value={optimized.equity_summary.min?.toFixed(3) ?? "—"} />
              <KeyVal label="Max" value={optimized.equity_summary.max?.toFixed(3) ?? "—"} />
              <KeyVal label="Gini" value={optimized.equity_summary.gini?.toFixed(3) ?? "—"} />
              {equityResult && (
                <KeyVal
                  label="Disparity index"
                  value={`${equityResult.disparity_index.toFixed(3)} (CV-based)`}
                />
              )}
            </div>
          ) : (
            <div style={{ fontSize: 11, color: "var(--muted)" }}>
              Equity statistics appear after the optimizer runs.
            </div>
          )}
        </Card>
      </div>

      <Card
        title="Equity layers"
        subtitle={`map disadvantage proxies onto candidate weights · paste ${candidates.length || "N"} values per layer`}
        actions={
          <Btn onClick={fetchCensus} disabled={censusLoading || candidates.length === 0}>
            {censusLoading ? "Fetching ACS…" : "Auto-fetch ACS"}
          </Btn>
        }
      >
        {census && (
          <div
            style={{
              marginBottom: 10, padding: "8px 10px",
              border: "1px solid var(--line)", borderRadius: 4,
              background: "#fdf6e9", display: "flex", flexWrap: "wrap",
              gap: 12, fontSize: 11,
            }}
          >
            <div className="mono" style={{ color: "var(--purple)", fontSize: 9.5, letterSpacing: "0.1em", textTransform: "uppercase" }}>
              Census ACS {census.context.vintage}
            </div>
            <div>
              <strong>{census.context.n_counties}</strong> {census.context.n_counties === 1 ? "county" : "counties"}
              {census.context.counties.length > 0 && (
                <span style={{ color: "var(--muted)" }}>
                  {" "}· {census.context.counties.map((c) => c.name).join(", ")}
                </span>
              )}
            </div>
            {census.context.population != null && (
              <div>pop <strong>{census.context.population.toLocaleString()}</strong></div>
            )}
            {census.context.median_income != null && (
              <div>median income <strong>${Math.round(census.context.median_income).toLocaleString()}</strong></div>
            )}
            {census.context.poverty_rate != null && (
              <div>poverty <strong>{(census.context.poverty_rate * 100).toFixed(1)}%</strong></div>
            )}
            {census.context.mobile_home_share != null && (
              <div>mobile homes <strong>{(census.context.mobile_home_share * 100).toFixed(1)}%</strong></div>
            )}
            {census.context.notes && (
              <div style={{ flexBasis: "100%", color: "var(--crimson)", fontSize: 10 }}>{census.context.notes}</div>
            )}
          </div>
        )}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr auto", gap: 12, alignItems: "start" }}>
          {Object.keys(layerText).map((name) => (
            <div key={name} style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <div className="mono" style={{ fontSize: 9.5, color: "var(--muted)", letterSpacing: "0.1em", textTransform: "uppercase" }}>
                {name}
              </div>
              <textarea
                value={layerText[name]}
                onChange={(e) => setLayerText((p) => ({ ...p, [name]: e.target.value }))}
                placeholder={`e.g. 0.4, 0.7, 0.2 … (one per candidate)`}
                rows={3}
                style={{
                  border: "1px solid var(--line)", borderRadius: 4,
                  padding: "6px 8px", fontSize: 11, fontFamily: "'JetBrains Mono', monospace",
                  resize: "vertical", minHeight: 56,
                }}
              />
              <label style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 10, color: "var(--muted)" }}>
                <input
                  type="checkbox"
                  checked={!!layerInvert[name]}
                  onChange={(e) => setLayerInvert((p) => ({ ...p, [name]: e.target.checked }))}
                  style={{ accentColor: "var(--purple)" }}
                />
                invert (advantage → disadvantage)
              </label>
            </div>
          ))}
          <div style={{ display: "flex", flexDirection: "column", gap: 6, alignSelf: "stretch", justifyContent: "flex-end" }}>
            <Btn primary onClick={applyEquityLayers} disabled={candidates.length === 0}>
              Apply to candidates
            </Btn>
            <div style={{ fontSize: 10, color: "var(--muted)", maxWidth: 180, lineHeight: 1.4 }}>
              Each layer is min-max normalised; weights are the convex mean.
            </div>
          </div>
        </div>
      </Card>

      <Card
        title="Spatial targeting"
        subtitle={
          targeting?.summary
            ? `${targeting.summary.variable} · top ${targeting.summary.top_k}/${targeting.summary.n_features} cells · |CATE| ÷ cost`
            : (targetingVars.length === 0 ? "no spatial CATE maps available" : "pick a treatment variable")
        }
        actions={
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            {targetingVars.length > 0 && (
              <select
                value={activeTargetVar}
                onChange={(e) => setActiveTargetVar(e.target.value)}
                style={{
                  fontSize: 11, padding: "3px 6px", borderRadius: 3,
                  border: "1px solid var(--line)", fontFamily: "inherit",
                }}
              >
                {targetingVars.map((v) => (
                  <option key={v} value={v}>{v.replace(/_/g, " ")}</option>
                ))}
              </select>
            )}
            <span className="mono" style={{ fontSize: 10, color: "var(--muted)" }}>top-K</span>
            <input
              type="range" min={10} max={500} step={10}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              style={{ width: 100 }}
            />
            <span className="mono" style={{ fontSize: 11, minWidth: 32 }}>{topK}</span>
          </div>
        }
      >
        <div style={{ height: 360, borderRadius: 4, overflow: "hidden", border: "1px solid var(--line)", position: "relative" }}>
          {targetingLoading && (
            <div className="mono" style={{
              position: "absolute", top: 8, left: 8, zIndex: 10,
              fontSize: 9, padding: "3px 6px", background: "rgba(255,255,255,0.92)",
              border: "1px solid var(--line)", borderRadius: 3, color: "var(--muted)",
            }}>
              loading…
            </div>
          )}
          {targeting && targeting.features.length > 0 ? (
            <SpatialMap
              geojson={targeting}
              colorField="priority"
              mode="scatter"
              height="100%"
              palette="sparc"
            />
          ) : (
            <div style={{
              display: "flex", height: "100%", alignItems: "center", justifyContent: "center",
              color: "var(--muted)", fontSize: 12, background: "#faf8f4",
            }}>
              {targetingVars.length === 0
                ? "Run the causal stage to generate spatial CATE maps"
                : "Pick a variable to compute deployment priority"}
            </div>
          )}
        </div>
      </Card>

      <Card
        title="Candidate interventions"
        subtitle={candidates.length ? `editable — derived from scenarios + NUTS posteriors` : "no scenarios available yet"}
      >        {candidates.length === 0 ? (
          <div style={{ fontSize: 12, color: "var(--muted)" }}>
            Run the pipeline through Stage 4 (Scenarios) to populate intervention candidates.
          </div>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
              <thead>
                <tr style={{ background: "#fdf6e9" }}>
                  {["#", "Name", "Treatment", "μ effect", "σ", "Cost", "Equity w", "Score", "P(select)", "Effect 10–90%"].map((h) => (
                    <th key={h} style={th}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {candidates.map((c, i) => {
                  const r = ranked.find((rr) => rr.name === c.name && rr.treatment === c.treatment);
                  const u = uncertaintyByName.get(c.name);
                  return (
                    <tr key={`${c.name}-${i}`} style={{ borderBottom: "1px dotted var(--line)", background: r?.selected ? "#fff8ef" : undefined }}>
                      <td style={td}>{r ? ranked.indexOf(r) + 1 : ""}</td>
                      <td style={td}>{c.name}</td>
                      <td style={td}><Tag color="var(--purple)">{c.treatment}</Tag></td>
                      <td style={td} className="mono">{c.mean_effect.toFixed(3)}</td>
                      <td style={td} className="mono">{c.effect_std.toFixed(3)}</td>
                      <td style={td}>
                        <input
                          type="number"
                          value={c.cost}
                          step={0.5}
                          onChange={(e) => updateCandidate(i, { cost: Number(e.target.value) || 0 })}
                          style={cellInput}
                        />
                      </td>
                      <td style={td}>
                        <input
                          type="number"
                          value={c.equity_weight}
                          step={0.1}
                          onChange={(e) => updateCandidate(i, { equity_weight: Number(e.target.value) || 0 })}
                          style={cellInput}
                        />
                      </td>
                      <td style={{ ...td, fontWeight: 600, color: r ? "var(--ink)" : "var(--muted)" }} className="mono">
                        {r ? r.score.toFixed(3) : "—"}
                      </td>
                      <td style={td}>
                        {u ? (
                          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                            <div style={{
                              width: 48, height: 6, background: "var(--line)", borderRadius: 3, overflow: "hidden",
                            }}>
                              <div style={{
                                width: `${Math.round(u.selection_probability * 100)}%`,
                                height: "100%",
                                background: u.selection_probability > 0.66
                                  ? "var(--purple)"
                                  : u.selection_probability > 0.33
                                    ? "var(--amber)"
                                    : "var(--crimson)",
                              }} />
                            </div>
                            <span className="mono" style={{ fontSize: 10 }}>
                              {(u.selection_probability * 100).toFixed(0)}%
                            </span>
                          </div>
                        ) : <span style={{ color: "var(--muted)" }}>—</span>}
                      </td>
                      <td style={td} className="mono" title={u ? `median ${u.effect_p50.toFixed(3)}` : ""}>
                        {u ? `${u.effect_p10.toFixed(2)} … ${u.effect_p90.toFixed(2)}` : "—"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}

const th: React.CSSProperties = {
  textAlign: "left",
  padding: "6px 8px",
  fontSize: 9.5,
  textTransform: "uppercase",
  letterSpacing: "0.08em",
  color: "var(--muted)",
  borderBottom: "1px solid var(--line)",
};
const td: React.CSSProperties = {
  padding: "6px 8px",
  verticalAlign: "middle",
};
const cellInput: React.CSSProperties = {
  border: "1px solid var(--line)",
  borderRadius: 3,
  padding: "2px 4px",
  width: 64,
  fontFamily: "'JetBrains Mono', monospace",
  fontSize: 11,
};
const pillBtn = (active: boolean): React.CSSProperties => ({
  padding: "3px 8px",
  fontSize: 10,
  borderRadius: 3,
  border: "1px solid " + (active ? "var(--crimson)" : "var(--line)"),
  background: active ? "var(--crimson)" : "#fff",
  color: active ? "#fff" : "var(--ink-2)",
  cursor: "pointer",
  fontFamily: "inherit",
  fontWeight: 600,
});
