// Causal audit (SPEC §6.4): per treatment, the forest of θ_own, θ_nbr and θ_sum (±1.96 SE)
// beside the model slopes, audit verdict chips, the doubly-robust dose–response with the
// model's own-cell partial-dependence curve overlaid, the CATE quantile box with BLP
// calibration, CATE and model-slope maps, sensitivity, controls and the DAG audit. The complete
// page is Model details › Cause-and-effect audit; How much to trust this reads one line per
// lever from `causalAgreement`.
// Plain words (phase 4): Model details keeps the technical names (θ, CATE, BLP, E-value, DAG
// audit, partial dependence); every block and chart has an ⓘ or a glossary <Term> beside it.
import type { CausalTreatment, Sections } from "../../api/runs";
import { BoxStrip, Forest, LineBand, type LineSeries } from "../../charts";
import { InfoTip } from "../../components/ui/InfoTip";
import { Pill } from "../../components/ui/Pill";
import { Term } from "../../components/ui/Term";
import { lever as leverWords } from "../../content/glossary";
import { Seg } from "../../components/ui/Seg";
import { Link, codecs, useUrlState } from "../../router";
import { fmtNum, fmtPct, unitLabel } from "../../theme/format";
import { Block, GenericTableView, OlderCode, Section, ViewPage, useRid, useUnits, type DetailPageProps } from "./common";
import { fmtDistance, humanize } from "./format";
import { RunLayerMap } from "./RunLayerMap";

type Treatment = Sections<CausalTreatment> & { label: string; unit: string };
type Blp = NonNullable<NonNullable<CausalTreatment["cate"]>["blp"]>;

/**
 * The units of a treatment. The server's `unit` is already the effect unit, target per lever
 * unit ("°F per pp"), so it is shown as it is; the dose (the x of the dose–response curve) is
 * in the lever's own unit ("pp").
 */
export function treatmentUnits(unit: string | null | undefined, name: string, targetUnit: string, leverUnits: Record<string, string>): { effect: string; dose: string } {
  const u = unit ?? "";
  const i = u.indexOf(" per ");
  const target = unitLabel(i >= 0 ? u.slice(0, i) : targetUnit);
  const lever = i >= 0 ? u.slice(i + " per ".length) : u;
  const effect = `${target || "effect"} per ${unitLabel(lever) || "unit"}`;
  const dose = unitLabel(leverUnits[name]) || (lever && lever !== "unit" ? unitLabel(lever) : "");
  return { effect, dose };
}

/**
 * The best-linear-predictor calibration test in words (core `causal.blp_calibration`):
 * R_Y = β₁·R_T + β₂·R_T·(s − s̄); β₂ ≈ 1 means the CATE is a calibrated predictor of the
 * effect, β₂ ≈ 0 that it carries no real heterogeneity; p tests β₂ = 0.
 */
export function blpWords(blp: Blp | null | undefined): string {
  if (!blp) return "No BLP calibration test in this run.";
  if (blp.coef === null) return "The CATE is constant across cells, so the BLP calibration test is not identified.";
  const stats = [blp.se !== null ? `SE ${fmtNum(blp.se, 2)}` : null, blp.p !== null ? `p = ${fmtNum(blp.p, 3)}` : null].filter((x): x is string => x !== null);
  const head = `BLP calibration slope β₂ = ${fmtNum(blp.coef, 2)}${stats.length ? ` (${stats.join(", ")})` : ""}`;
  if (blp.p !== null && blp.p < 0.05 && blp.coef > 0)
    return `${head}: the effect differs between cells as the CATE says${blp.coef < 0.5 ? ", though the CATE exaggerates the spread" : blp.coef > 1.5 ? ", more strongly than the CATE suggests" : ""}.`;
  return `${head}: no evidence that the effect differs between cells; read the spread of the box as noise.`;
}

/** How a lever's independent (causal) estimate compares with the model's, in one plain line. */
export type CausalAgreement = {
  lever: string;
  label: string;
  /** agrees: every audit check consistent; size: same direction, different size or curve shape; conflict: opposite directions; unknown: no audit. */
  status: "agrees" | "size" | "conflict" | "unknown";
  line: string;
};

/** The audit check that compares the whole change (own cell + neighbours), else the first with both numbers. */
function primaryCheck(audit: CausalTreatment["audit"]): CausalTreatment["audit"][number] | null {
  return audit.find((a) => a.check === "adoption_vs_theta_sum") ?? audit.find((a) => a.model !== null && a.causal !== null) ?? null;
}

/**
 * One step of a lever, in words, for "… °F cooler per percentage point of tree canopy": a pp lever
 * counts percentage points of its glossary measure, a 0–1 lever (albedo) whole units of it.
 */
export function perStepPhrase(name: string, t: Pick<Treatment, "unit" | "label">, leverUnits: Record<string, string>): string {
  const w = leverWords(name);
  const measure = w?.measure ?? (t.label || humanize(name)).toLowerCase();
  const u = t.unit ?? "";
  const i = u.indexOf(" per ");
  const lu = unitLabel(leverUnits[name]) || (i >= 0 ? unitLabel(u.slice(i + " per ".length)) : "");
  if (lu === "pp" || lu === "%") return `percentage point of ${measure}`;
  if (!lu || lu === "unit" || lu === "units" || lu === "albedo" || lu === "1") return `unit of ${measure}${lu === "albedo" ? " (0–1 scale)" : ""}`;
  return `${lu} of ${measure}`;
}

/** "cooler" / "warmer" for an effect per step (negative = cooler). */
const coolWord = (v: number) => (v < 0 ? "cooler" : "warmer");

export function causalAgreement(name: string, t: Treatment, targetUnit: string, leverUnits: Record<string, string>): CausalAgreement {
  // the lever's plain name from the glossary ("Trees & canopy"), else the catalogue label
  const label = leverWords(name)?.name ?? (t.label || humanize(name));
  const audit = t.audit ?? [];
  if (!audit.length) return { lever: name, label, status: "unknown", line: `${label}: the cause-and-effect check did not test this lever in this run.` };
  const u = unitLabel(treatmentUnits(t.unit, name, targetUnit, leverUnits).effect.split(" per ")[0]) || unitLabel(targetUnit);
  const per = perStepPhrase(name, t, leverUnits);
  const p = primaryCheck(audit);
  const m = p && p.model !== null ? p.model : null;
  const c = p && p.causal !== null ? p.causal : null;
  const same = m !== null && c !== null && Math.sign(m) === Math.sign(c);
  const nums =
    m === null || c === null
      ? ""
      : same
        ? ` (model ${fmtNum(Math.abs(m), 2)} ${u}, check ${fmtNum(Math.abs(c), 2)} ${u} ${coolWord(m)} per ${per})`
        : ` (model ${fmtNum(Math.abs(m), 2)} ${u} ${coolWord(m)}, check ${fmtNum(Math.abs(c), 2)} ${u} ${coolWord(c)} per ${per})`;
  if (audit.some((a) => /sign/i.test(a.verdict))) {
    return { lever: name, label, status: "conflict", line: `${label}: the check points the other way${nums}. Treat this lever's effect as unproven.` };
  }
  if (audit.some((a) => a.flag && !/shape/i.test(a.verdict))) {
    const agrees = same ? `agrees it ${m! < 0 ? "cools" : "warms"}` : "agrees on the direction";
    const size = same ? `a ${Math.abs(c!) < Math.abs(m!) ? "smaller" : "larger"} effect` : "a different size";
    return { lever: name, label, status: "size", line: `${label}: the check ${agrees} but finds ${size}${nums}.` };
  }
  if (audit.some((a) => a.flag)) {
    return { lever: name, label, status: "size", line: `${label}: the check agrees on the size${nums}, but the effect builds up differently as the amount grows.` };
  }
  return { lever: name, label, status: "agrees", line: `${label}: the check agrees with the model${nums}.` };
}

function DrCurve({ t, name }: { t: Treatment; name: string }) {
  const units = useUnits();
  const u = unitLabel(units.target);
  const dose = treatmentUnits(t.unit, name, units.target, units.levers).dose;
  const rid = useRid();
  const map = `/r/${encodeURIComponent(rid)}/map?layer=`;
  const dr = t.dr_curve;
  if (!dr) return <OlderCode title="Dose–response curve" />;
  const series: LineSeries[] = [{ id: "dr", label: "Doubly-robust estimate (own exposure)", x: dr.t, y: dr.theta, lo: dr.lo, hi: dr.hi, emphasis: true }];
  const pd = t.model_pd_curve;
  if (pd) {
    series.push({
      id: "pd",
      label: "Model own-cell partial dependence",
      x: pd.t,
      y: pd.y,
      lo: pd.y.map((y, i) => (y === null || pd.se[i] === null ? null : y - 1.96 * (pd.se[i] as number))),
      hi: pd.y.map((y, i) => (y === null || pd.se[i] === null ? null : y + 1.96 * (pd.se[i] as number))),
      dashed: true,
      color: "var(--s2)",
    });
  }
  return (
    <div className="stack" style={{ gap: 6 }}>
      <LineBand
        title={`${t.label}: dose–response`}
        note={
          <>
            <Term id="dose_response">Dose–response</Term> from the separate causal analysis
            {pd ? (
              <>
                ; dashed: the model&apos;s <Term id="own_cell">own-cell</Term> <Term id="partial_dependence">partial dependence</Term>
              </>
            ) : null}
            .
          </>
        }
        units={u}
        series={series}
        xLabel={t.label}
        xUnit={dose}
        yLabel="Temperature"
        yUnit={u}
        decimals={2}
        caption={`Effective sample size ${fmtNum(dr.ess, 0)}; ${fmtPct(dr.clipped_frac, 0)} of the weights clipped.${dr.note ? ` ${dr.note}` : ""}${pd ? " The dashed curve is what the model implies for the same own-cell exposure." : ""}`}
      />
      {!pd ? <OlderCode title="Model partial-dependence overlay" /> : null}
      {t.cate_layer ? (
        <Block title={`${t.label}: CATE and model slopes per cell`} tip={<InfoTip term="cate" label="CATE" />}>
          <p className="cap">
            The causal effect per cell (CATE) beside the model's own slopes, from causal_cells.parquet:{" "}
            <Link to={map + encodeURIComponent(t.cate_layer)}>CATE</Link> · <Link to={map + encodeURIComponent(`mslope_${name}`)}>model adoption slope</Link> ·{" "}
            <Link to={map + encodeURIComponent(`mslope_own_${name}`)}>model own-cell slope</Link>.
          </p>
          <RunLayerMap rid={rid} keys={[t.cate_layer, `mslope_${name}`, `mslope_own_${name}`]} title={`${t.label}: CATE and model slopes`} height={340} />
        </Block>
      ) : (
        <OlderCode title="CATE and model-slope maps" />
      )}
    </div>
  );
}

function TreatmentView({ name, t }: { name: string; t: Treatment }) {
  const units = useUnits();
  const per = treatmentUnits(t.unit, name, units.target, units.levers).effect;
  return (
    <div className="stack">
      <div className="grid2">
        <Section title="Effect estimates" data={t.forest}>
          {(f) => (
            <Forest
              title={`${t.label}: causal estimates vs model`}
              note={
                <>
                  The separate analysis&apos;s effect on the spot (own), on the neighbours (nbr) and in total (sum), beside the model&apos;s.
                  <InfoTip
                    label="θ own, θ nbr and θ sum"
                    text="θ is the effect of one unit more of the lever, estimated from the readings alone: in the changed place (own), in the places around it (nbr), and both together (sum)."
                    technical="Double machine learning estimates θ_own, θ_nbr, θ_sum (±1.96 SE)"
                  />
                </>
              }
              units={per}
              rows={f.map((r) => ({ id: r.id, label: r.label, est: r.est, lo: r.lo, hi: r.hi, se: r.se, compare: r.model }))}
              valueLabel="Effect per unit"
              unit={per}
              compareLabel="model slope"
              decimals={4}
              caption={(() => {
                const sum = f.find((r) => r.id === "sum") ?? f[f.length - 1];
                return sum && sum.est !== null
                  ? `${sum.label}: ${fmtNum(sum.est, 4)} ${per}${sum.model !== null ? ` against the model's ${fmtNum(sum.model, 4)}` : ""}. Bars are 95% intervals; squares the model.`
                  : undefined;
              })()}
            />
          )}
        </Section>
        <Section title="Audit" data={t.audit}>
          {(a) => (
            <Block title="Audit verdicts" tip={<InfoTip term="causal_audit" label="Causal audit" />}>
              <ul className="stack" style={{ listStyle: "none", padding: 0, margin: 0, gap: 6 }}>
                {a.map((x) => (
                  <li key={x.check} className="row">
                    <Pill tone={x.flag ? "warn" : "good"} icon={x.flag ? "alert" : "check"}>
                      {x.verdict}
                    </Pill>
                    <span>{x.label}</span>
                    {x.model !== null && x.causal !== null ? (
                      <span className="cap num">
                        model {fmtNum(x.model, 4)} · causal {fmtNum(x.causal, 4)}
                      </span>
                    ) : null}
                  </li>
                ))}
              </ul>
            </Block>
          )}
        </Section>
      </div>
      <DrCurve t={t} name={name} />
      <div className="grid2">
        <Section title="CATE" data={t.cate}>
          {(c) => (
            <BoxStrip
              title={`${t.label}: effect heterogeneity (CATE)`}
              note={
                <>
                  How much the effect varies from place to place.
                  <InfoTip term="cate" label="CATE" />
                </>
              }
              units={per}
              groups={[{ label: t.label, q: c.q, mean: c.mean }]}
              valueLabel="CATE"
              unit={per}
              whiskerLabel="5th–95th percentile"
              decimals={4}
              zeroLine
              caption={blpWords(c.blp)}
            />
          )}
        </Section>
        <Section title="Sensitivity" data={t.sensitivity}>
          {(s) => (
            <Block title="Sensitivity to hidden confounding" tip={<InfoTip term="e_value" label="Sensitivity to hidden confounding" />}>
              <dl className="kv" aria-label="Sensitivity">
                <dt>E-value</dt>
                <dd>{fmtNum(s.e_value, 2)}</dd>
                <dt>E-value of the CI bound</dt>
                <dd>{fmtNum(s.e_value_ci, 2)}</dd>
                <dt>
                  Robustness value
                  <InfoTip
                    label="Robustness value"
                    text="The share of the unexplained variation in both the lever and the temperature that a hidden factor would have to explain to remove the effect (at α = 0.05: to make it no longer significant)."
                    technical="Robustness value RV_q (Cinelli & Hazlett)"
                  />
                </dt>
                <dd>{fmtPct(s.rv_q, 0)}</dd>
                <dt>Robustness value (α = 0.05)</dt>
                <dd>{fmtPct(s.rv_q_alpha, 0)}</dd>
                <dt>
                  Design effect
                  <InfoTip
                    label="Design effect"
                    text="How much the clustering of nearby readings widens the uncertainty, compared with the same number of independent readings. 1 means no widening."
                    technical="Design effect (spatial clustering)"
                  />
                </dt>
                <dd>{fmtNum(s.design_effect, 2)}</dd>
              </dl>
              <p className="cap">
                An unmeasured confounder would need to explain {fmtPct(s.rv_q_alpha, 0)} of the residual variance of both treatment and temperature to make the effect
                insignificant.
              </p>
            </Block>
          )}
        </Section>
      </div>
      <Section title="Controls" data={t.controls}>
        {(c) => (
          <Block
            title="Controls"
            tip={
              <InfoTip
                label="Controls"
                text="The other map layers, and a smooth map-wide pattern, that the causal analysis holds fixed so they cannot pass for the lever's effect."
                technical="Confounders and spatial basis of the double machine learning"
              />
            }
          >
            <dl className="kv" aria-label="Controls">
              <dt>Controls</dt>
              <dd>{c.names.join(", ") || "—"}</dd>
              <dt>Spatial basis scale</dt>
              <dd>{fmtDistance(c.basis_scale_m)}</dd>
              <dt>
                Hole-scale ratio
                <InfoTip
                  label="Hole-scale ratio"
                  text="The size of the smooth map-wide pattern next to the size of the lever's patches. Near or below 1, the pattern may soak up part of the lever's own effect."
                  technical="Spatial basis scale ÷ treatment hole scale"
                />
              </dt>
              <dd>
                {fmtNum(c.hole_scale_ratio, 2)}{" "}
                {c.hole_warning ? (
                  <Pill tone="warn" icon="alert">
                    basis may absorb the effect
                  </Pill>
                ) : null}
              </dd>
              {Object.entries(c.nuisance_r2).map(([k, v]) => (
                <div key={k} style={{ display: "contents" }}>
                  <dt>Nuisance R², {k}</dt>
                  <dd>{fmtNum(v, 2)}</dd>
                </div>
              ))}
            </dl>
          </Block>
        )}
      </Section>
    </div>
  );
}

export default function Causal({
  title = "Causal audit",
  intro = (
    <>
      A <Term id="causal_audit">cause-and-effect check of the model&apos;s effects</Term>: double machine learning on the observed variation, with neighbourhood spillover,
      heterogeneity (<Term id="cate">CATE</Term>) and sensitivity analysis.
    </>
  ),
}: DetailPageProps = {}) {
  const [tSel, setTSel] = useUrlState("t", codecs.optString());
  return (
    <ViewPage
      view="causal"
      title={title}
      intro={intro}
    >
      {(s) => (
        <>
          <Section title="Treatments" data={s.treatments}>
            {(ts) => {
              const names = Object.keys(ts);
              if (!names.length) return <p className="cap">No treatments were analysed.</p>;
              const cur = tSel && ts[tSel] ? tSel : names[0];
              return (
                <div className="stack">
                  {names.length > 1 ? (
                    <Seg<string> label="Treatment" value={cur} onChange={(v) => setTSel(v)} options={names.map((n) => ({ value: n, label: ts[n].label || n }))} />
                  ) : null}
                  <TreatmentView name={cur} t={ts[cur]} />
                </div>
              );
            }}
          </Section>
          <Section title="Flags" data={s.flags}>
            {(f) =>
              f.length ? (
                <Block
                  title="Flags"
                  tip={<InfoTip label="causal flags" text="Audit checks where the model and the separate causal analysis disagree, by lever." technical="Causal audit flags" />}
                >
                  <ul style={{ margin: 0, paddingLeft: 18 }}>
                    {f.map((x, i) => (
                      <li key={i}>
                        {x.treatment}: {humanize(x.check)} — {x.verdict}
                      </li>
                    ))}
                  </ul>
                </Block>
              ) : (
                <p className="cap">No causal flags.</p>
              )
            }
          </Section>
          {s.dag_audit ? (
            <Block title="DAG audit" tip={<InfoTip term="dag_audit" label="DAG audit" />}>
              <GenericTableView table={s.dag_audit} caption="DAG audit" csvName="dag-audit" />
            </Block>
          ) : (
            <p className="cap">No DAG audit in this run (it runs only when causal.dag_audit is on).</p>
          )}
        </>
      )}
    </ViewPage>
  );
}
