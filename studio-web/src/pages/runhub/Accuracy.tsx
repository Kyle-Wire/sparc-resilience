// Accuracy (SPEC §6.4): per-model table with blend weights, obs-vs-pred hexbin, a residual
// histogram brushable into the run's selection (shown on the map), residuals by zone and
// fold, interval honesty, the stacker panel, physics card, advection verdict and the CV
// design card. The forcing card (model input, not accuracy) is on Data & checks (./DataQa). Usable mid-run: the server computes metrics from
// predictions.parquet and marks the ViewModel `live`. The complete page is Model details ›
// Accuracy; How much to trust this reuses `finalModel` and `ObsPredChart`.
// Plain words (phase 4): this is Model details, so the technical names stay (R², RMSE, stacker,
// physics card, advection, CV design), each explained by an ⓘ or a glossary <Term> beside it.
import type { AccuracySections, CoverageRow, ModelRow } from "../../api/runs";
import type { SelectionSpec } from "../../api/types";
import { Bars, BoxStrip, DotRange, HexbinScatter, Histogram, SmallMultiples } from "../../charts";
import { Badge } from "../../components/ui/Badge";
import { InfoTip } from "../../components/ui/InfoTip";
import { Pill } from "../../components/ui/Pill";
import { Table } from "../../components/ui/Table";
import { Term } from "../../components/ui/Term";
import { Tooltip } from "../../components/ui/Tooltip";
import { Link, navigate } from "../../router";
import { encodeSelection, useRunSelection } from "../../stores/selection";
import { fmtInt, fmtNum, fmtPct, fmtSigned, fmtValue, unitLabel } from "../../theme/format";
import { Block, Section, ViewPage, useRid, useUnits, type DetailPageProps } from "./common";

function ModelsTable({ rows }: { rows: ModelRow[] }) {
  const u = unitLabel(useUnits().target);
  const maxW = Math.max(1e-9, ...rows.map((r) => r.weight ?? 0));
  return (
    <Table<ModelRow>
      caption="Held-out accuracy per model"
      csvName="model-accuracy"
      rowKey={(r) => r.model}
      highlight={(r) => r.kind === "stack"}
      initialSort={{ key: "r2", dir: "desc" }}
      columns={[
        { key: "model", label: "Model", value: (r) => r.label },
        { key: "r2", label: "R²", align: "right", value: (r) => r.r2, render: (r) => fmtNum(r.r2, 3) },
        { key: "rmse", label: "RMSE", unit: u, align: "right", value: (r) => r.rmse, render: (r) => fmtNum(r.rmse, 3) },
        { key: "mae", label: "MAE", unit: u, align: "right", value: (r) => r.mae, render: (r) => fmtNum(r.mae, 3) },
        { key: "bias", label: "Bias (pred − obs)", unit: u, align: "right", value: (r) => r.bias, render: (r) => fmtSigned(r.bias, 3) },
        {
          key: "weight",
          label: "Blend weight",
          value: (r) => r.weight,
          render: (r) =>
            r.weight === null ? (
              "—"
            ) : (
              <span>
                <span className="wbar" style={{ width: `${Math.max(1, (r.weight / maxW) * 80)}px` }} aria-hidden="true" />
                {fmtPct(r.weight, 0)}
              </span>
            ),
        },
      ]}
      rows={rows}
    />
  );
}

/** The run's final model row: the stack, else the model with the highest held-out R². */
export function finalModel(rows: readonly ModelRow[] | null | undefined): ModelRow | null {
  if (!rows?.length) return null;
  const stack = rows.find((r) => r.kind === "stack");
  if (stack) return stack;
  return rows.reduce<ModelRow | null>((best, r) => (r.r2 !== null && (!best || best.r2 === null || r.r2 > best.r2) ? r : best), null) ?? rows[0];
}

/** Observed vs predicted held-out values (hexbin) with the 1:1 line. */
export function ObsPredChart({
  bins,
  unit,
  title = "Observed vs predicted",
  caption,
  height,
}: {
  bins: NonNullable<AccuracySections["obs_pred_bins"]>;
  /** Target unit label (°F). */
  unit: string;
  title?: string;
  caption?: string;
  height?: number;
}) {
  const sp = bins.spearman;
  return (
    <HexbinScatter
      title={title}
      bins={bins}
      xLabel="Predicted"
      yLabel="Observed"
      xUnit={unit || undefined}
      yUnit={unit || undefined}
      diagonal
      spearman={sp ?? null}
      height={height}
      caption={
        caption ??
        (sp !== undefined && sp !== null ? `Spearman ρ = ${fmtNum(sp, 2)} between held-out predictions and observations; the dashed line is 1:1.` : "The dashed line is 1:1.")
      }
    />
  );
}

/** The residual-histogram brush as a portable selection (`pred:resid` between lo and hi). */
export function residSelection(lo: number, hi: number): SelectionSpec {
  return { kind: "filter", column: "pred:resid", op: "between", value: [lo, hi] };
}

function brushOf(spec: SelectionSpec | null): [number, number] | null {
  if (spec && "kind" in spec && spec.kind === "filter" && spec.column === "pred:resid" && spec.op === "between" && Array.isArray(spec.value) && spec.value.length === 2) {
    const [a, b] = spec.value as [number, number];
    return typeof a === "number" && typeof b === "number" ? [a, b] : null;
  }
  return null;
}

function ResidualHistogram({ hist }: { hist: NonNullable<AccuracySections["resid_hist"]> }) {
  const rid = useRid();
  const units = useUnits();
  const u = unitLabel(units.target);
  const sel = useRunSelection(rid);
  const brush = brushOf(sel.spec);
  const n = hist.counts.reduce((a, b) => a + b, 0);
  const inBrush = brush ? hist.counts.reduce((a, c, i) => a + (hist.edges[i] >= brush[0] - 1e-9 && hist.edges[i + 1] <= brush[1] + 1e-9 ? c : 0), 0) : 0;
  const mapHref = brush ? `/r/${encodeURIComponent(rid)}/map?layer=resid&sel=${encodeURIComponent(encodeSelection(residSelection(brush[0], brush[1])) ?? "")}` : null;
  return (
    <div className="stack" style={{ gap: 6 }}>
      <Histogram
        title="Residuals (observed − predicted)"
        note={
          <>
            How far each test place&apos;s reading is from the prediction.
            <InfoTip term="residual" label="Residual" />
          </>
        }
        units={u}
        bins={hist}
        xLabel="Residual"
        unit={u}
        decimals={2}
        refLines={[{ value: 0, label: "0" }]}
        brush={brush}
        onBrush={(r) =>
          r ? sel.set(residSelection(r[0], r[1]), { label: `Residual ${fmtSigned(r[0], 2)} to ${fmtSigned(r[1], 2)} ${u}`, n_cells: null, source: "accuracy" }) : sel.clear()
        }
        caption={`${fmtInt(n)} held-out cells. Positive residuals: the model predicts too cool there. Drag across bars (or press Enter on one) to select those cells.`}
      />
      {brush ? (
        <p className="cap" role="status">
          Selected {fmtInt(inBrush)} cells with residuals {fmtSigned(brush[0], 2)} to {fmtSigned(brush[1], 2)} {u}. {mapHref ? <Link to={mapHref}>Show on the map</Link> : null}
        </p>
      ) : null}
    </div>
  );
}

function CoveragePanels({ ih }: { ih: NonNullable<AccuracySections["interval_honesty"]> }) {
  const panels: { id: string; label: string; rows: CoverageRow[] }[] = [
    { id: "fold", label: "By CV fold", rows: ih.by_fold },
    { id: "distance", label: "By distance to training data", rows: ih.by_distance },
    { id: "zone", label: "By zone", rows: ih.by_zone },
  ].filter((p) => p.rows.length);
  const target = ih.target;
  const under = panels.flatMap((p) => p.rows.filter((r) => r.global !== null && r.global < target - 0.05).map((r) => `${p.label.toLowerCase().replace(/^by /, "")} ${r.label}`));
  const caption = `Overall ${fmtPct(ih.global, 1)} of held-out cells fall inside their interval (adaptive ${fmtPct(ih.adaptive, 1)}) against a ${fmtPct(target, 0)} target${ih.halfwidth !== null ? `; mean half-width ${fmtNum(ih.halfwidth, 2)}` : ""}. ${under.length ? `More than 5 points under target: ${under.join(", ")}.` : "No group is more than 5 points under target."}`;
  if (!panels.length) return <p className="cap">{caption}</p>;
  return (
    <SmallMultiples
      title="Interval honesty"
      note={
        <>
          Do the prediction ranges hold in each test area, at each distance and in each zone?
          <InfoTip term="interval_honesty" label="Interval honesty" />
        </>
      }
      units="% of held-out cells inside the 90% interval"
      items={panels}
      panelTitle={(p) => p.label}
      renderPanel={(p) => (
        <Bars
          bare
          title={p.label}
          categories={p.rows.map((r) => r.label)}
          series={[
            { id: "global", label: "Global interval", values: p.rows.map((r) => (r.global === null ? null : r.global * 100)) },
            { id: "adaptive", label: "Adaptive interval", values: p.rows.map((r) => (r.adaptive === null ? null : r.adaptive * 100)) },
          ]}
          highlight={p.rows.filter((r) => r.global !== null && r.global < target - 0.05).map((r) => r.label)}
          valueLabel="Coverage"
          unit="%"
          decimals={1}
          domain={[0, 100]}
          height={200}
        />
      )}
      table={{
        columns: [
          { key: "g", label: "Grouping" },
          { key: "l", label: "Group" },
          { key: "n", label: "n" },
          { key: "gl", label: "Global coverage", unit: "%" },
          { key: "ad", label: "Adaptive coverage", unit: "%" },
        ],
        rows: panels.flatMap((p) => p.rows.map((r) => [p.label, r.label, r.n, r.global === null ? null : r.global * 100, r.adaptive === null ? null : r.adaptive * 100])),
      }}
      legend={
        <>
          <span>
            <i className="box" style={{ background: "var(--s1)" }} />
            global interval
          </span>
          <span>
            <i className="box" style={{ background: "var(--s2)" }} />
            adaptive interval
            <InfoTip
              label="Global and adaptive intervals"
              text="The global prediction range has the same width everywhere. The adaptive one is wider where the model is less sure and narrower where it is surer."
              technical="Global vs locally adaptive conformal interval"
            />
          </span>
          <span>outlined = more than 5 points under the {fmtPct(target, 0)} target</span>
        </>
      }
      caption={caption}
    />
  );
}

function StackerPanel({ st }: { st: NonNullable<AccuracySections["stacker"]> }) {
  const u = unitLabel(useUnits().target);
  const kept = st.folds.filter((f) => f.residual_kept).length;
  const best = st.candidates.reduce<{ name: string; rmse: number | null } | null>((m, c) => (c.rmse !== null && (!m || m.rmse === null || c.rmse < m.rmse) ? c : m), null);
  return (
    <div className="stack">
      <div className="row">
        {st.spatial_plus.length ? <Pill tone="accent">Spatial+ on: {st.spatial_plus.join(", ")}</Pill> : <Pill>Spatial+ off</Pill>}
        <Pill tone={kept ? "accent" : "neutral"}>
          residual kept in {kept}/{st.folds.length} folds
        </Pill>
        {st.chosen ? <Pill>chosen: {st.chosen}</Pill> : null}
      </div>
      <div className="grid2">
        <Bars
          title="Stacker candidates"
          note={
            <>
              The ways of blending the models that were tried; the lowest <Term id="rmse">RMSE</Term> wins.
            </>
          }
          units={u}
          categories={st.candidates.map((c) => c.name)}
          categoryLabel="Candidate"
          series={[{ id: "rmse", label: "Held-out RMSE", values: st.candidates.map((c) => c.rmse) }]}
          highlight={st.chosen ? [st.chosen] : best ? [best.name] : []}
          valueLabel="Held-out RMSE"
          unit={u}
          decimals={3}
          caption={best ? `Lowest held-out RMSE: ${best.name} (${fmtValue(best.rmse, u, 3)}).` : undefined}
        />
        <Bars
          title="Blend weights per fold"
          note={
            <>
              How much each model counts in each test area&apos;s blend.
              <InfoTip term="blend_weights" label="Blend weights" />
            </>
          }
          categories={st.folds.map((f) => `Fold ${f.fold + 1}`)}
          series={st.models.map((m) => ({ id: m, label: m, values: st.folds.map((f) => f.weights[m] ?? 0) }))}
          mode="percent"
          valueLabel="weight"
          decimals={0}
          caption={`Non-negative least-squares weights of the base models in each fold (each bar sums to 100%).`}
        />
      </div>
      <Table
        caption="Residual network per fold"
        csvName="stacker-folds"
        rowKey={(r) => r.fold}
        columns={[
          { key: "fold", label: "Fold", value: (r) => r.fold + 1 },
          { key: "kept", label: "Residual network", value: (r) => (r.residual_kept === null ? null : r.residual_kept ? "kept" : "gated off") },
          { key: "base", label: "Validation MSE, blend", align: "right", value: (r) => r.val_mse_base, render: (r) => fmtNum(r.val_mse_base, 4) },
          { key: "with", label: "Validation MSE, with residual", align: "right", value: (r) => r.val_mse_with_residual, render: (r) => fmtNum(r.val_mse_with_residual, 4) },
          { key: "epoch", label: "Best epoch", align: "right", value: (r) => r.best_epoch },
        ]}
        rows={st.folds}
      />
    </div>
  );
}

function PhysicsCard({ ph }: { ph: NonNullable<AccuracySections["physics"]> }) {
  const byParam = new Map(ph.folds.map((f) => [f.param, f.values]));
  const withFolds = ph.params.filter((p) => byParam.has(p.name));
  return (
    <div className="stack">
      <Table
        caption="Physics parameters"
        csvName="physics-parameters"
        rowKey={(r) => r.name}
        columns={[
          {
            key: "label",
            label: "Parameter",
            value: (r) => r.label,
            render: (r) => (
              <Tooltip content={
                <>
                  Technical: <code>{r.name}</code>
                </>
              }>
                {r.label}
              </Tooltip>
            ),
          },
          {
            key: "mean",
            label: "Mean ± sd",
            align: "right",
            value: (r) => r.mean,
            render: (r) => `${fmtNum(r.mean, 3)} ± ${fmtNum(r.sd, 3)}${r.unit ? " " + unitLabel(r.unit) : ""}`,
          },
          { key: "prior", label: "Prior", value: (r) => r.prior ?? "—" },
        ]}
        rows={ph.params}
      />
      {withFolds.length ? (
        <SmallMultiples
          title="Physics parameters per fold"
          items={withFolds}
          panelTitle={(p) => `${p.label}${p.unit ? ` (${unitLabel(p.unit)})` : ""}`}
          renderPanel={(p) => (
            <DotRange
              bare
              title={p.label}
              rows={[
                {
                  id: p.name,
                  label: "folds",
                  est: p.mean,
                  lo: p.mean !== null && p.sd !== null ? p.mean - p.sd : null,
                  hi: p.mean !== null && p.sd !== null ? p.mean + p.sd : null,
                  strip: (byParam.get(p.name) ?? []).filter((v): v is number => v !== null),
                },
              ]}
              valueLabel={p.label}
              unit={p.unit ? unitLabel(p.unit) : undefined}
              rangeLabel="mean ± sd"
              decimals={3}
            />
          )}
          table={{
            columns: [
              { key: "p", label: "Parameter" },
              { key: "f", label: "Fold" },
              { key: "v", label: "Value" },
            ],
            rows: withFolds.flatMap((p) => (byParam.get(p.name) ?? []).map((v, i) => [p.label, i + 1, v])),
          }}
          caption="Grey dots are the per-fold fits; the bar is mean ± sd."
        />
      ) : null}
      {ph.warnings.length ? (
        <ul className="callout" style={{ margin: 0, paddingLeft: 28 }}>
          {ph.warnings.map((w, i) => (
            <li key={i}>{w}</li>
          ))}
        </ul>
      ) : (
        <p className="cap">No physics fit warnings.</p>
      )}
    </div>
  );
}

export default function Accuracy({
  title = "Accuracy",
  intro = (
    <>
      Held-out accuracy from <Term id="cv_design">spatially blocked cross-validation</Term>: every prediction here was made without the cell&apos;s own block.
    </>
  ),
}: DetailPageProps = {}) {
  const rid = useRid();
  return (
    <ViewPage
      view="accuracy"
      title={title}
      intro={intro}
      live={(vm) => vm.sections.live === true}
      liveText="Live: these metrics are computed from predictions.parquet while the run continues. They refresh as outputs are written."
    >
      {(s, vm) => (
        <>
          <Section title="Per-model accuracy" data={s.models}>
            {(m) => (
              <Block
                title="Per-model accuracy"
                tip={
                  <InfoTip
                    label="Per-model accuracy"
                    text="Each model scored on areas it was not trained on. R²: the accuracy score (1 is perfect). RMSE and MAE: the typical error. Bias: how much it predicts too warm (+) or too cool (−) on average. Blend weight: how much it counts in the final blend."
                    technical="Held-out R², RMSE, MAE, bias and stacker weight per model"
                  />
                }
              >
                <ModelsTable rows={m} />
              </Block>
            )}
          </Section>
          <div className="grid2">
            <Section title="Observed vs predicted" data={s.obs_pred_bins}>
              {(b) => <ObsPredChart bins={b} unit={unitLabel(vm.units.target)} />}
            </Section>
            <Section title="Residual histogram" data={s.resid_hist}>
              {(h) => <ResidualHistogram hist={h} />}
            </Section>
          </div>
          {/* Without zones, Residuals by fold takes the full width instead of half a row on its own. */}
          <div className={s.resid_by_zone?.length ? "grid2" : "stack"}>
            <Section title="Residuals by zone" data={s.resid_by_zone}>
              {(z) =>
                z.length ? (
                  <BoxStrip
                    title="Residuals by zone"
                    note={
                      <>
                        Does the model miss more in some zones than others?
                        <InfoTip term="residual" label="Residual" />
                      </>
                    }
                    units={`${unitLabel(vm.units.target)} (observed − predicted)`}
                    groups={z}
                    valueLabel="Residual"
                    unit={unitLabel(vm.units.target)}
                    decimals={2}
                    zeroLine
                    groupLabel="Zone"
                    caption={`${z.length} zones; boxes are quartiles, whiskers the 10th–90th percentile.`}
                  />
                ) : (
                  <p className="cap" style={{ gridColumn: "1 / -1" }}>This run has no zones.</p>
                )
              }
            </Section>
            <Section title="Residuals by fold" data={s.resid_by_fold}>
              {(f) => (
                <BoxStrip
                  title="Residuals by fold"
                  note={
                    <>
                      The same misses, one box per <Term id="held_out_fold">test area</Term> (fold).
                    </>
                  }
                  units={`${unitLabel(vm.units.target)} (observed − predicted)`}
                  groups={f}
                  valueLabel="Residual"
                  unit={unitLabel(vm.units.target)}
                  decimals={2}
                  zeroLine
                  groupLabel="Fold"
                  onGroupClick={() => navigate(`/r/${encodeURIComponent(rid)}/map?layer=fold`)}
                  caption="A fold whose box sits away from zero is biased there. Select a fold to see the fold layer on the map."
                />
              )}
            </Section>
          </div>
          <Section title="Interval honesty" data={s.interval_honesty}>
            {(ih) => <CoveragePanels ih={ih} />}
          </Section>
          <Section title="Stacker" data={s.stacker}>
            {(st) => (
              <Block title="Stacker" tip={<InfoTip term="stacker" label="Stacker" />}>
                <StackerPanel st={st} />
              </Block>
            )}
          </Section>
          <Section title="Physics card" data={s.physics}>
            {(ph) => (
              <Block title="Physics card" tip={<InfoTip term="physics_card" label="Physics card" />}>
                <PhysicsCard ph={ph} />
              </Block>
            )}
          </Section>
          {/* The advection verdict and the CV design share a row (the forcing card is on Data & checks). */}
          <div className="grid2">
            <Section title="Advection verdict" data={s.advection}>
              {(a) => {
                const verdict = (
                  <>
                    <strong>Advection:</strong>
                    <InfoTip term="advection" label="Advection" /> {a.verdict}{" "}
                    {a.selected !== null ? <Badge tone={a.selected ? "accent" : undefined}>{a.selected ? "advection kept" : "advection off"}</Badge> : null}
                  </>
                );
                return a.fold_delta_rmse.length ? (
                  <DotRange
                    title="Advection ΔRMSE per fold"
                    note={verdict}
                    units={`${unitLabel(vm.units.target)} (with − without advection)`}
                    rows={a.fold_delta_rmse.map((v, i) => ({ id: `f${i}`, label: `Fold ${i + 1}`, est: v }))}
                    valueLabel="ΔRMSE"
                    unit={unitLabel(vm.units.target)}
                    signed
                    decimals={3}
                    caption={`Dots left of zero: advection lowered the held-out error in that fold (${a.fold_delta_rmse.filter((v) => v !== null && v < 0).length} of ${a.fold_delta_rmse.length}).`}
                  />
                ) : (
                  <div className="card">
                    <p className="row">{verdict}</p>
                  </div>
                );
              }}
            </Section>
            <Section title="CV design" data={s.cv_design}>
              {(cv) => (
                <Block
                  title="CV design"
                  tip={<InfoTip term="cv_design" label="CV design" />}
                  actions={<Link to={`/r/${encodeURIComponent(rid)}/map?layer=fold`}>Show folds on the map</Link>}
                >
                  <dl className="kv" aria-label="CV design">
                    <dt>Folds</dt>
                    <dd>{fmtInt(cv.n_folds)}</dd>
                    <dt>Block size</dt>
                    <dd>{fmtValue(cv.block_m, "m", 0)}</dd>
                    <dt>Buffer</dt>
                    <dd>{fmtValue(cv.buffer_m, "m", 0)}</dd>
                    <dt>Blocks</dt>
                    <dd>{fmtInt(cv.n_blocks)}</dd>
                    <dt>Test cells per fold</dt>
                    <dd>{cv.test_sizes.map((n) => fmtInt(n)).join(" · ") || "—"}</dd>
                  </dl>
                </Block>
              )}
            </Section>
          </div>
        </>
      )}
    </ViewPage>
  );
}
