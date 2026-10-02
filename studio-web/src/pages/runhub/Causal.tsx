// Causal audit (SPEC §6.4): per treatment, the forest of θ_own, θ_nbr and θ_sum (±1.96 SE)
// beside the model slopes, audit verdict chips, the doubly-robust dose–response with the
// model's own-cell partial-dependence curve overlaid, the CATE quantile box with BLP
// calibration, CATE and model-slope maps, sensitivity, controls and the DAG audit.
import type { CausalTreatment, Sections } from "../../api/runs";
import { BoxStrip, Forest, LineBand, type LineSeries } from "../../charts";
import { Pill } from "../../components/ui/Pill";
import { Seg } from "../../components/ui/Seg";
import { Link, codecs, useUrlState } from "../../router";
import { fmtNum, fmtPct, unitLabel } from "../../theme/format";
import { Block, GenericTableView, OlderCode, Section, ViewPage, useRid, useUnits } from "./common";
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
        <Block title={`${t.label}: CATE and model slopes per cell`}>
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
            <Block title="Audit verdicts">
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
            <Block title="Sensitivity to hidden confounding">
              <dl className="kv" aria-label="Sensitivity">
                <dt>E-value</dt>
                <dd>{fmtNum(s.e_value, 2)}</dd>
                <dt>E-value of the CI bound</dt>
                <dd>{fmtNum(s.e_value_ci, 2)}</dd>
                <dt>Robustness value</dt>
                <dd>{fmtPct(s.rv_q, 0)}</dd>
                <dt>Robustness value (α = 0.05)</dt>
                <dd>{fmtPct(s.rv_q_alpha, 0)}</dd>
                <dt>Design effect</dt>
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
          <Block title="Controls">
            <dl className="kv" aria-label="Controls">
              <dt>Controls</dt>
              <dd>{c.names.join(", ") || "—"}</dd>
              <dt>Spatial basis scale</dt>
              <dd>{fmtDistance(c.basis_scale_m)}</dd>
              <dt>Hole-scale ratio</dt>
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

export default function Causal() {
  const [tSel, setTSel] = useUrlState("t", codecs.optString());
  return (
    <ViewPage
      view="causal"
      title="Causal audit"
      intro="An independent check of the model's effects: double machine learning on the observed variation, with neighbourhood spillover, heterogeneity and sensitivity analysis."
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
                <Block title="Flags">
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
            <Block title="DAG audit">
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
