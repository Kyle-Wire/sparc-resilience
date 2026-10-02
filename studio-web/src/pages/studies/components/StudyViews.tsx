// Result views of the study kinds (SPEC §8, api.md §9 `GET /api/studies/{stid}/view`): placebo
// verdict table and Δ per sd vs real, simcheck grid + share strip + coverage + false positives +
// bias-correction card, multiverse heatmaps and R² by variant, the reproduce checklist and the
// benchmark effect shares. Shared by the Validation tab cards and the study page.
import type { ReactNode } from "react";
import {
  useStudyView,
  type BenchmarkView,
  type BiasCorrection,
  type MultiverseView,
  type PlaceboRow,
  type PlaceboView,
  type ReproduceView,
  type SimcheckGeneratorSummary,
  type SimcheckView,
  type ViewKind,
} from "../../../api/studies";
import { Bars, BoxStrip, DotRange, Heatmap } from "../../../charts";
import { EmptyState } from "../../../components/ui/EmptyState";
import { Kpi, KpiRow } from "../../../components/ui/Kpi";
import { StatusChip } from "../../../components/ui/StatusChip";
import { Link } from "../../../router";
import { fmtDuration, fmtNum, fmtPct, fmtSigned, unitLabel } from "../../../theme/format";
import { quantiles, sharesByGenerator, simGrid } from "../model/simcheck";
import { SimcheckGrid } from "./SimcheckGrid";

const runHref = (rid: string) => `/r/${encodeURIComponent(rid)}`;

// ---------------------------------------------------------------- placebo

/** Model Δ at the dose closest to +1 sd, with its SE (the verdict's numbers when present). */
export function oneSd(r: PlaceboRow): { d: number | null; se: number | null } {
  if (r.verdict && r.verdict.delta_1sd !== null && r.verdict.delta_1sd !== undefined) return { d: r.verdict.delta_1sd, se: r.verdict.se_1sd ?? null };
  if (!r.model?.length) return { d: null, se: null };
  const m = r.model.reduce((a, b) => (Math.abs(b.dose_sd - 1) < Math.abs(a.dose_sd - 1) ? b : a));
  return { d: m.mean_delta, se: m.se };
}

export function placeboVerdictText(r: PlaceboRow): string {
  if (!r.placebo) return "real effect (reference)";
  const v = r.verdict;
  if (!v) return "verdict pending";
  return `${v.model_pass ? "model passes" : "model fails"} · ${v.causal_ci_covers_zero ? "causal passes" : "causal fails"}`;
}

export function PlaceboResult({ view, units }: { view: PlaceboView; units: string }) {
  const rows = view.rows ?? [];
  const u = unitLabel(units);
  const nPl = view.n_placebos ?? rows.filter((r) => r.placebo).length;
  const corr = Object.entries(view.layer_correlation ?? {});
  return (
    <div className="stack">
      {nPl ? (
        <p className="vt-headline">
          {view.n_pass_model ?? 0} of {nPl} placebo layer{nPl === 1 ? "" : "s"} pass the model check (no effect within 2 SE, or under 10% of the real one); {view.n_pass_causal ?? 0} of {nPl} pass
          the causal check (its interval covers zero).
        </p>
      ) : null}
      {rows.length ? (
        <>
          <div className="tablewrap">
            <table className="tbl" aria-label="Placebo verdicts">
              <thead>
                <tr>
                  <th scope="col">Re-fit</th>
                  <th scope="col">Layer</th>
                  <th scope="col" className="r">
                    Model Δ at +1 sd{u ? ` (${u})` : ""}
                  </th>
                  <th scope="col" className="r">
                    Ratio to real
                  </th>
                  <th scope="col" className="r">
                    Causal θ per sd
                  </th>
                  <th scope="col">Verdict</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r, i) => {
                  const { d, se } = oneSd(r);
                  const v = r.verdict;
                  return (
                    <tr key={`${r.kind}-${r.variable}-${i}`} className={r.placebo ? undefined : "hl"}>
                      <td>{r.kind}</td>
                      <td>
                        {r.variable}
                        {r.placebo ? "" : " (real)"}
                      </td>
                      <td className="r num">
                        {fmtSigned(d, 3)}
                        {se !== null ? ` ± ${fmtNum(se, 3)}` : ""}
                      </td>
                      <td className="r num">{v && v.ratio_to_real !== null ? fmtPct(v.ratio_to_real) : "—"}</td>
                      <td className="r num">
                        {fmtSigned(r.causal_theta_sum_per_sd, 3)}
                        {r.causal_se_per_sd !== null && r.causal_se_per_sd !== undefined ? ` ± ${fmtNum(r.causal_se_per_sd, 3)}` : ""}
                      </td>
                      <td>
                        {r.placebo ? (
                          <span className="row" style={{ gap: 4 }}>
                            <StatusChip status={v ? (v.model_pass ? "done" : "failed") : "planned"} text={v ? (v.model_pass ? "model ✓" : "model ✗") : "pending"} />
                            {v ? <StatusChip status={v.causal_ci_covers_zero ? "done" : "failed"} text={v.causal_ci_covers_zero ? "causal ✓" : "causal ✗"} /> : null}
                          </span>
                        ) : (
                          <span className="cap">reference</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <DotRange
            title="Δ per sd: placebo vs real"
            units={`city-mean ΔT at +1 sd of the layer${u ? `, ${u}` : ""} (negative = cooler)`}
            rows={rows.map((r, i) => {
              const { d, se } = oneSd(r);
              const c = r.causal_theta_sum_per_sd;
              const cse = r.causal_se_per_sd;
              return {
                id: `${r.kind}-${r.variable}-${i}`,
                label: `${r.kind}: ${r.variable}${r.placebo ? "" : " (real)"}`,
                est: d,
                lo: d !== null && se !== null ? d - 1.96 * se : null,
                hi: d !== null && se !== null ? d + 1.96 * se : null,
                muted: r.placebo,
                check: c !== null && c !== undefined && Number.isFinite(c) ? { est: c, lo: cse !== null && cse !== undefined ? c - 1.96 * cse : null, hi: cse !== null && cse !== undefined ? c + 1.96 * cse : null, label: "causal θ per sd" } : null,
              };
            })}
            valueLabel="ΔT at +1 sd"
            unit={u}
            signed
            decimals={3}
            rangeLabel="95% interval"
            caption={placeboCaption(rows)}
          />
        </>
      ) : (
        <p className="cap">The verdict table fills in as the re-fits finish.</p>
      )}
      {corr.length ? (
        <p className="cap">
          Correlation of each moved layer with the original:{" "}
          {corr.map(([k, v], i) => (
            <span key={k}>
              {i ? "; " : ""}
              {k} {fmtNum(v, 2)}
            </span>
          ))}
          . Values near zero mean the placebo really broke the layer's link to place.
        </p>
      ) : null}
      {view.children?.length ? (
        <ul className="sx-list" aria-label="Placebo child runs">
          {view.children.map((c) => (
            <li key={c.kind} className="row">
              <strong>{c.kind}</strong>
              <StatusChip status={c.status} />
              {c.verdict ? <span className="cap">{c.verdict}</span> : null}
              {c.run_id ? <Link to={runHref(c.run_id)}>child run</Link> : null}
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

function placeboCaption(rows: PlaceboRow[]): string {
  const real = rows.filter((r) => !r.placebo);
  const pl = rows.filter((r) => r.placebo);
  const big = pl.filter((r) => r.verdict && !r.verdict.model_pass);
  const parts = [`Blue: the real layers' effects (reference); grey: placebo layers, which should sit near zero.`];
  if (real.length && pl.length) parts.push(big.length ? `${big.length} placebo layer${big.length === 1 ? "" : "s"} show an effect beyond 2 SE and above 10% of the real one.` : "No placebo layer shows an effect beyond 2 SE and 10% of the real one.");
  parts.push("Orange diamond: the causal estimate per sd.");
  return parts.join(" ");
}

// ---------------------------------------------------------------- simcheck

/** The null generators of a summary: "null", plus "null/<product>" when a study ran other products. */
function nullGenerators(gens: Record<string, SimcheckGeneratorSummary>): [string, SimcheckGeneratorSummary][] {
  return Object.entries(gens).filter(([k]) => k === "null" || k.startsWith("null/"));
}

export function SimcheckResult({ view, workers = 1, units }: { view: SimcheckView; workers?: number; units: string }) {
  const grid = simGrid(view);
  // Null generators ("null", "null/<product>") plant no effect: they have false positives, not shares.
  const strips = sharesByGenerator(grid).filter((s) => !s.generator.startsWith("null"));
  const gens = view.generators ?? {};
  const covNames = Object.keys(gens).filter((g) => !g.startsWith("null"));
  const nulls = nullGenerators(gens);
  const bc = view.bias_correction ?? null;
  const u = unitLabel(units);
  return (
    <div className="stack">
      <SimcheckGrid view={view} workers={workers} />
      {strips.some((s) => s.shares.length) ? (
        <BoxStrip
          title="Effect share per generator"
          units="recovered ÷ planted city-mean effect (1 = exact)"
          groups={strips.map((s) => ({ label: s.generator, n: s.shares.length, q: quantiles(s.shares), points: s.shares }))}
          valueLabel="Effect share"
          groupLabel="Generator"
          whiskerLabel="min–max"
          decimals={2}
          caption={shareCaption(gens)}
        />
      ) : null}
      {covNames.length ? (
        <Bars
          title="Interval coverage of the planted truth"
          units="% of replicates (target 95%)"
          categories={covNames}
          series={[
            { id: "ci", label: "Model 95% CI covers truth", values: covNames.map((g) => pct(gens[g].ci_coverage)) },
            { id: "causal", label: "Causal 95% CI covers truth", values: covNames.map((g) => pct(gens[g].causal_ci_coverage)) },
          ]}
          orientation="h"
          mode="grouped"
          valueLabel="Coverage"
          unit="%"
          categoryLabel="Generator"
          decimals={0}
          domain={[0, 100]}
          caption={coverageCaption(gens, covNames)}
        />
      ) : null}
      <div className="grid2">
        {nulls.length ? (
          <section className="card tight" aria-label="False positives">
            <header>
              <h3>False positives (null generator)</h3>
            </header>
            {nulls.map(([name, g]) => (
              <KpiRow key={name} label={`False positives: ${name}`}>
                <Kpi
                  label={nulls.length > 1 ? `Model significant (${name})` : "Model significant"}
                  value={fmtPct(g.false_positive_rate ?? null)}
                  note={`of ${g.n} no-effect replicates`}
                  tone={tone(g.false_positive_rate, 0.1)}
                />
                <Kpi label="Causal significant" value={fmtPct(g.causal_false_positive_rate ?? null)} tone={tone(g.causal_false_positive_rate, 0.1)} />
                <Kpi
                  label="Mean spurious Δ"
                  value={fmtSigned(g.null_mean_delta ?? null, 3)}
                  unit={u || undefined}
                  note={g.null_mean_delta_se !== null && g.null_mean_delta_se !== undefined ? `± ${fmtNum(g.null_mean_delta_se, 3)} SE` : undefined}
                />
              </KpiRow>
            ))}
            <p className="cap">With no planted effect, about 5% of replicates should come out significant at 95%.</p>
          </section>
        ) : null}
        {bc && bc.share_range ? <BiasCorrectionCard bc={bc} /> : null}
      </div>
      {Object.keys(gens).length ? (
        <details>
          <summary className="cap">Per-generator summary</summary>
          <GeneratorTable generators={gens} />
        </details>
      ) : null}
    </div>
  );
}

/** simcheck_summary's bias-correction block: median share across generators, range, stability. */
export function BiasCorrectionCard({ bc }: { bc: BiasCorrection }) {
  if (!bc.share_range) return null;
  return (
    <section className="card tight" aria-label="Bias correction">
      <header>
        <h3>Bias correction</h3>
        <StatusChip status={bc.stable ? "done" : "stale"} text={bc.stable ? "stable across generators" : "varies by generator"} />
      </header>
      <KpiRow label="Bias correction">
        <Kpi label="Median share" value={fmtNum(bc.share_median_across_generators ?? null, 2)} note={`${bc.n_generators ?? "?"} generators`} />
        <Kpi label="Share range" value={`${fmtNum(bc.share_range[0], 2)}–${fmtNum(bc.share_range[1], 2)}`} />
        <Kpi label="Relative spread" value={fmtPct(bc.relative_spread ?? null)} tone={bc.stable ? "good" : "warn"} />
        {bc.correction_factor ? <Kpi label="Correction factor" value={`× ${fmtNum(bc.correction_factor, 2)}`} /> : null}
      </KpiRow>
      <p className="cap">
        {bc.stable && bc.correction_factor
          ? `City-wide canopy effects can be divided by ${fmtNum(bc.share_median_across_generators ?? null, 2)} (multiplied by ${fmtNum(bc.correction_factor, 2)}).`
          : (bc.n_generators ?? 2) < 2
            ? "Only one non-null generator so far: no correction."
            : "No single correction: read city-wide canopy effects as a range across these shares."}
      </p>
    </section>
  );
}

/** One row per generator of a simcheck summary (merged or single study). */
export function GeneratorTable({ generators }: { generators: Record<string, SimcheckGeneratorSummary> }) {
  const names = Object.keys(generators);
  return (
    <div className="tablewrap">
      <table className="tbl" aria-label="Simulation check by generator">
        <thead>
          <tr>
            <th scope="col">Generator</th>
            <th scope="col" className="r">
              Replicates
            </th>
            <th scope="col" className="r">
              Gate passed
            </th>
            <th scope="col" className="r">
              Median share
            </th>
            <th scope="col" className="r">
              Share IQR
            </th>
            <th scope="col" className="r">
              Model CI coverage
            </th>
            <th scope="col" className="r">
              Causal CI coverage
            </th>
            <th scope="col" className="r">
              False positives
            </th>
          </tr>
        </thead>
        <tbody>
          {names.map((g) => {
            const r = generators[g];
            return (
              <tr key={g} data-generator={g}>
                <td>{g}</td>
                <td className="r num">{r.n}</td>
                <td className="r num">{r.n_gate_pass}</td>
                <td className="r num">{fmtNum(r.share_median, 2)}</td>
                <td className="r num">{r.share_iqr ? `${fmtNum(r.share_iqr[0], 2)}–${fmtNum(r.share_iqr[1], 2)}` : "—"}</td>
                <td className="r num">{fmtPct(r.ci_coverage)}</td>
                <td className="r num">{fmtPct(r.causal_ci_coverage)}</td>
                <td className="r num">{r.false_positive_rate === undefined ? "—" : fmtPct(r.false_positive_rate)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

const pct = (v: number | null | undefined) => (v === null || v === undefined || !Number.isFinite(v) ? null : v * 100);

function tone(v: number | null | undefined, warnAbove: number): "good" | "warn" | undefined {
  if (v === null || v === undefined || !Number.isFinite(v)) return undefined;
  return v > warnAbove ? "warn" : "good";
}

function shareCaption(gens: Record<string, SimcheckGeneratorSummary>): string {
  const med = Object.entries(gens)
    .filter(([k, g]) => !k.startsWith("null") && g.share_median !== null)
    .map(([k, g]) => `${k} ${fmtNum(g.share_median, 2)}`);
  return med.length ? `Median share (gate-passing replicates): ${med.join(", ")}. Below 1: the model attenuates the planted effect.` : "Each dot is one replicate; box = interquartile range.";
}

function coverageCaption(gens: Record<string, SimcheckGeneratorSummary>, names: string[]): string {
  const low = names.filter((g) => gens[g].ci_coverage !== null && (gens[g].ci_coverage as number) < 0.8);
  return low.length ? `The model interval covers the truth in under 80% of replicates for ${low.join(", ")}: its SE understates the attribution error there.` : "Model intervals cover the planted truth in at least 80% of replicates for every generator.";
}

// ---------------------------------------------------------------- multiverse

export function MultiverseResult({ view, units }: { view: MultiverseView; units: string }) {
  const variants = view.variants ?? [];
  const effects = view.effects ?? {};
  const scenarios = Object.keys(effects);
  const labelOf = (name: string) => variants.find((v) => v.name === name)?.label || name;
  const names = [...new Set([...variants.map((v) => v.name), ...scenarios.flatMap((s) => Object.keys(effects[s].values ?? {}))])];
  const priority = view.priority ?? {};
  const pVariants = Object.keys(priority);
  const levers = [...new Set(pVariants.flatMap((v) => Object.keys(priority[v] ?? {})))];
  const u = unitLabel(units);
  const st = view.stability ?? null;
  const done = variants.filter((v) => v.r2 !== null);
  return (
    <div className="stack">
      {st ? (
        <KpiRow label="Multiverse stability">
          <Kpi label="Sign stability (worst scenario)" value={fmtPct(st.sign_stability_min ?? null)} tone={st.sign_stability_min === 1 ? "good" : st.sign_stability_min !== null && st.sign_stability_min !== undefined ? "warn" : undefined} />
          <Kpi label="Median Kendall τ" value={fmtNum(st.median_kendall_tau ?? null, 2)} note="priority map vs baseline" />
          <Kpi label="Median top-decile overlap" value={fmtNum(st.median_top_decile_jaccard ?? null, 2)} />
        </KpiRow>
      ) : null}
      {variants.length ? (
        <div className="tablewrap">
          <table className="tbl" aria-label="Multiverse variants">
            <thead>
              <tr>
                <th scope="col">Variant</th>
                <th scope="col">Status</th>
                <th scope="col" className="r">
                  R²
                </th>
                <th scope="col" className="r">
                  RMSE
                </th>
                <th scope="col" className="r">
                  Time
                </th>
                <th scope="col">Run</th>
              </tr>
            </thead>
            <tbody>
              {variants.map((v) => (
                <tr key={v.name} className={v.name === "baseline" ? "hl" : undefined}>
                  <td>{v.label || v.name}</td>
                  <td>
                    <StatusChip status={v.status} />
                  </td>
                  <td className="r num">{fmtNum(v.r2, 3)}</td>
                  <td className="r num">{fmtNum(v.rmse, 3)}</td>
                  <td className="r num">{fmtDuration(v.seconds)}</td>
                  <td>{v.run_id ? <Link to={runHref(v.run_id)}>open</Link> : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
      {scenarios.length && names.length ? (
        <Heatmap
          title="Scenario effect by variant"
          units={`city-mean ΔT${u ? `, ${u}` : ""} (negative = cooler)`}
          rows={names.map(labelOf)}
          cols={scenarios}
          values={names.map((n) => scenarios.map((s) => effects[s].values?.[n] ?? null))}
          scale="div"
          center={0}
          unit={u}
          valueLabel="Mean ΔT"
          rowLabel="Variant"
          colLabel="Scenario"
          labels={names.length * scenarios.length <= 80}
          caption={signCaption(effects)}
        />
      ) : null}
      {pVariants.length && levers.length ? (
        <div className="grid2">
          <Heatmap
            title="Priority agreement with the baseline (Kendall τ)"
            units="rank correlation of each lever's priority map with the baseline's"
            rows={pVariants.map(labelOf)}
            cols={levers}
            values={pVariants.map((v) => levers.map((l) => priority[v]?.[l]?.kendall_tau ?? null))}
            scale="seq"
            domain={[0, 1]}
            valueLabel="Kendall τ"
            rowLabel="Variant"
            colLabel="Lever"
            labels
            caption="Kendall τ between each variant's priority map and the baseline's. Near 1: the same places come first."
          />
          <Heatmap
            title="Top-decile overlap with the baseline"
            units="Jaccard index of the top 10% priority cells"
            rows={pVariants.map(labelOf)}
            cols={levers}
            values={pVariants.map((v) => levers.map((l) => priority[v]?.[l]?.top_decile_jaccard ?? null))}
            scale="seq"
            domain={[0, 1]}
            valueLabel="Jaccard"
            rowLabel="Variant"
            colLabel="Lever"
            labels
            caption="Share of the top-decile cells the variant and the baseline have in common (1 = the same cells)."
          />
        </div>
      ) : null}
      {done.length ? (
        <Bars
          title="Held-out R² by variant"
          categories={done.map((v) => v.label || v.name)}
          series={[{ id: "r2", label: "Held-out R²", values: done.map((v) => v.r2) }]}
          highlight={done.filter((v) => v.name === "baseline").map((v) => v.label || v.name)}
          orientation="h"
          valueLabel="Held-out R²"
          categoryLabel="Variant"
          decimals={3}
          caption={r2Caption(done)}
        />
      ) : (
        <p className="cap">Variants fill in the table and heatmaps as they finish.</p>
      )}
    </div>
  );
}

function signCaption(effects: NonNullable<MultiverseView["effects"]>): string {
  const rows = Object.entries(effects);
  const unstable = rows.filter(([, e]) => e.sign_stability !== null && e.sign_stability < 1).map(([s, e]) => `${s} (${fmtPct(e.sign_stability)})`);
  return unstable.length ? `Sign changes under some variants for ${unstable.join(", ")}.` : `Every scenario keeps its sign under every variant (${rows.length} scenario${rows.length === 1 ? "" : "s"}).`;
}

function r2Caption(done: { name: string; label: string; r2: number | null }[]): string {
  const base = done.find((v) => v.name === "baseline");
  const vals = done.map((v) => v.r2).filter((x): x is number => x !== null);
  if (!vals.length) return "";
  const lo = Math.min(...vals);
  const hi = Math.max(...vals);
  return `${base ? `Baseline R² ${fmtNum(base.r2, 3)}; ` : ""}variants range ${fmtNum(lo, 3)}–${fmtNum(hi, 3)}.`;
}

// ---------------------------------------------------------------- reproduce

export function ReproduceResult({ view, childRunId }: { view: ReproduceView; childRunId?: string | null }) {
  const checks = view.checks ?? [];
  const hardFail = checks.filter((c) => c.hard && !c.ok).length;
  return (
    <div className="stack">
      <div className="row">
        {view.pass === true || view.pass === false ? (
          <StatusChip status={view.pass ? "done" : "failed"} text={view.pass ? "reproduces" : `${hardFail} hard check${hardFail === 1 ? " differs" : "s differ"}`} />
        ) : (
          <StatusChip status="planned" text="checks pending" />
        )}
        {childRunId ? <Link to={runHref(childRunId)}>Reproduction run</Link> : null}
      </div>
      {checks.length ? (
        <div className="tablewrap">
          <table className="tbl" aria-label="Reproduction checks">
            <thead>
              <tr>
                <th scope="col">Check</th>
                <th scope="col">Kind</th>
                <th scope="col">Result</th>
                <th scope="col">Detail</th>
              </tr>
            </thead>
            <tbody>
              {checks.map((c) => (
                <tr key={c.check}>
                  <td>{c.check}</td>
                  <td>{c.hard ? "hard" : "soft"}</td>
                  <td>
                    <StatusChip status={c.ok ? "done" : c.hard ? "failed" : "stale"} text={c.ok ? "matches" : "differs"} />
                  </td>
                  <td className="cap">{c.detail}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="cap">The checklist (CV design, R² per model, scenario effects and input data; code and package versions as soft checks) fills in when the reproduction finishes.</p>
      )}
      {view.original || view.reproduction ? (
        <p className="cap">
          {view.original ? (
            <>
              Original: <span className="sx-mono">{view.original}</span>.{" "}
            </>
          ) : null}
          {view.reproduction ? (
            <>
              Reproduction: <span className="sx-mono">{view.reproduction}</span>.
            </>
          ) : null}
        </p>
      ) : null}
    </div>
  );
}

// ---------------------------------------------------------------- benchmark

const SETTING_LABELS: Record<string, string> = { standard: "Standard", spatial_plus: "Spatial+ (MGWR)" };

export function BenchmarkResult({ view }: { view: BenchmarkView }) {
  const runs = view.runs ?? {};
  const settings = Object.keys(runs);
  if (!settings.length) return <p className="cap">The benchmark shares appear when it finishes.</p>;
  const models = [...new Set(settings.flatMap((s) => Object.keys(runs[s].models ?? {})))];
  const cats = [...models, "stack", ...(settings.some((s) => runs[s].footprint) ? ["footprint map"] : [])];
  const value = (s: string, m: string) => (m === "stack" ? runs[s].stack?.share ?? null : m === "footprint map" ? runs[s].footprint?.share ?? null : runs[s].models?.[m]?.share ?? null);
  const stackTxt = settings.map((s) => `${SETTING_LABELS[s] ?? s} ${fmtNum(runs[s].stack?.share ?? null, 2)}`).join(", ");
  return (
    <div className="stack">
      <Bars
        title="Effect share recovered on the synthetic city"
        units="model mean Δ ÷ planted mean Δ (1 = unbiased)"
        categories={cats}
        series={settings.map((s) => ({ id: s, label: SETTING_LABELS[s] ?? s, values: cats.map((m) => value(s, m)) }))}
        highlight={["stack"]}
        orientation="h"
        mode="grouped"
        valueLabel="Effect share"
        categoryLabel="Model"
        decimals={2}
        caption={`Stacked share: ${stackTxt}. Below 1 the model attenuates the planted canopy effect; the footprint row compares the per-unit footprint map with the truth.`}
      />
    </div>
  );
}

// ---------------------------------------------------------------- dispatch

/** Loads a study's view and renders it for its kind. */
export function StudyViewPanel({ kind, studyId, units, workers, childRunId, empty }: { kind: ViewKind; studyId: string; units: string; workers?: number; childRunId?: string | null; empty?: ReactNode }) {
  const res = useStudyView(studyId, kind);
  if (res.error && !res.data) return <EmptyState error={res.error} title="Could not load the study results" />;
  if (!res.data) return res.loading ? <p className="cap">Loading results…</p> : <>{empty ?? null}</>;
  const v = res.data;
  if (kind === "placebo") return <PlaceboResult view={v as PlaceboView} units={units} />;
  if (kind === "simcheck") return <SimcheckResult view={v as SimcheckView} workers={workers} units={units} />;
  if (kind === "multiverse") return <MultiverseResult view={v as MultiverseView} units={units} />;
  if (kind === "reproduce") return <ReproduceResult view={v as ReproduceView} childRunId={childRunId} />;
  return <BenchmarkResult view={v as BenchmarkView} />;
}
