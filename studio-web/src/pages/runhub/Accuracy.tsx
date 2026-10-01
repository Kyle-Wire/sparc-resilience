// Accuracy (SPEC §6.4): per-model table with blend weights, obs-vs-pred hexbin, a residual
// histogram brushable into the run's selection (shown on the map), residuals by zone and
// fold, interval honesty, the stacker panel, physics card, advection verdict, forcing card
// and the CV design card. Usable mid-run: the server computes metrics from
// predictions.parquet and marks the ViewModel `live`.
import type { AccuracySections, CoverageRow, ModelRow } from "../../api/runs";
import type { SelectionSpec } from "../../api/types";
import { Bars, BoxStrip, DotRange, HexbinScatter, Histogram, SmallMultiples } from "../../charts";
import { Badge } from "../../components/ui/Badge";
import { Pill } from "../../components/ui/Pill";
import { Table } from "../../components/ui/Table";
import { Link, navigate } from "../../router";
import { encodeSelection, useRunSelection } from "../../stores/selection";
import { fmtInt, fmtNum, fmtPct, fmtSigned, fmtValue, unitLabel } from "../../theme/format";
import { Block, Section, ViewPage, useRid, useUnits } from "./common";

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
          { key: "label", label: "Parameter", value: (r) => r.label, render: (r) => <span title={r.name}>{r.label}</span> },
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

function WindArrow({ deg, speed }: { deg: number | null; speed: number | null }) {
  if (deg === null) return null;
  // The arrow points where the wind blows to (from `deg`, clockwise from north).
  return (
    <svg width="40" height="40" viewBox="-20 -20 40 40" role="img" aria-label={`Wind from ${fmtNum(deg, 0)}° at ${fmtNum(speed, 1)} m/s`}>
      <circle r="18" fill="none" style={{ stroke: "var(--line-strong)" }} />
      <g transform={`rotate(${deg + 180})`}>
        <path d="M0,-14 L5,-4 L1.5,-4 L1.5,12 L-1.5,12 L-1.5,-4 L-5,-4 Z" style={{ fill: "var(--ink-2)" }} />
      </g>
    </svg>
  );
}

export default function Accuracy() {
  const rid = useRid();
  return (
    <ViewPage
      view="accuracy"
      title="Accuracy"
      intro="Held-out accuracy from spatially blocked cross-validation: every prediction here was made without the cell's own block."
      live={(vm) => vm.sections.live === true}
      liveText="Live: these metrics are computed from predictions.parquet while the run continues. They refresh as outputs are written."
    >
      {(s, vm) => (
        <>
          <Section title="Per-model accuracy" data={s.models}>
            {(m) => (
              <Block title="Per-model accuracy">
                <ModelsTable rows={m} />
              </Block>
            )}
          </Section>
          <div className="grid2">
            <Section title="Observed vs predicted" data={s.obs_pred_bins}>
              {(b) => (
                <HexbinScatter
                  title="Observed vs predicted"
                  bins={b}
                  xLabel="Predicted"
                  yLabel="Observed"
                  xUnit={unitLabel(vm.units.target) || undefined}
                  yUnit={unitLabel(vm.units.target) || undefined}
                  diagonal
                  spearman={b.spearman ?? null}
                  caption={
                    b.spearman !== undefined && b.spearman !== null
                      ? `Spearman ρ = ${fmtNum(b.spearman, 2)} between held-out predictions and observations; the dashed line is 1:1.`
                      : "The dashed line is 1:1."
                  }
                />
              )}
            </Section>
            <Section title="Residual histogram" data={s.resid_hist}>
              {(h) => <ResidualHistogram hist={h} />}
            </Section>
          </div>
          <div className="grid2">
            <Section title="Residuals by zone" data={s.resid_by_zone}>
              {(z) =>
                z.length ? (
                  <BoxStrip
                    title="Residuals by zone"
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
                  <p className="cap">This run has no zones.</p>
                )
              }
            </Section>
            <Section title="Residuals by fold" data={s.resid_by_fold}>
              {(f) => (
                <BoxStrip
                  title="Residuals by fold"
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
              <Block title="Stacker">
                <StackerPanel st={st} />
              </Block>
            )}
          </Section>
          <Section title="Physics card" data={s.physics}>
            {(ph) => (
              <Block title="Physics card">
                <PhysicsCard ph={ph} />
              </Block>
            )}
          </Section>
          <div className="grid2">
            <Section title="Advection verdict" data={s.advection}>
              {(a) => (
                <div className="stack">
                  <p className="row">
                    <strong>Advection:</strong> {a.verdict}{" "}
                    {a.selected !== null ? <Badge tone={a.selected ? "accent" : undefined}>{a.selected ? "advection kept" : "advection off"}</Badge> : null}
                  </p>
                  {a.fold_delta_rmse.length ? (
                    <DotRange
                      title="Advection ΔRMSE per fold"
                      units={`${unitLabel(vm.units.target)} (with − without advection)`}
                      rows={a.fold_delta_rmse.map((v, i) => ({ id: `f${i}`, label: `Fold ${i + 1}`, est: v }))}
                      valueLabel="ΔRMSE"
                      unit={unitLabel(vm.units.target)}
                      signed
                      decimals={3}
                      caption={`Dots left of zero: advection lowered the held-out error in that fold (${a.fold_delta_rmse.filter((v) => v !== null && v < 0).length} of ${a.fold_delta_rmse.length}).`}
                    />
                  ) : null}
                </div>
              )}
            </Section>
            <Section title="Forcing card" data={s.forcing}>
              {(f) => (
                <Block title="Forcing">
                  <div className="row" style={{ alignItems: "flex-start" }}>
                    <WindArrow deg={f.wind_dir_deg} speed={f.wind_speed} />
                    <dl className="kv" aria-label="Forcing">
                      <dt>Date</dt>
                      <dd>{f.date ?? "—"}</dd>
                      <dt>Hours</dt>
                      <dd>{f.hours ?? "—"}</dd>
                      <dt>Shortwave down</dt>
                      <dd>{fmtValue(f.sw_down, "W/m²", 0)}</dd>
                      <dt>Net longwave</dt>
                      <dd>{fmtValue(f.lw_net, "W/m²", 0)}</dd>
                      <dt>Wind</dt>
                      <dd>{f.wind_speed !== null ? `${fmtValue(f.wind_speed, "m/s", 1)}${f.wind_dir_deg !== null ? ` from ${fmtNum(f.wind_dir_deg, 0)}°` : ""}` : "—"}</dd>
                      <dt>Station</dt>
                      <dd>{f.station ?? "—"}</dd>
                    </dl>
                  </div>
                  {f.checks.length ? (
                    <ul className="cap" style={{ margin: 0, paddingLeft: 18 }}>
                      {f.checks.map((c, i) => (
                        <li key={i}>{c}</li>
                      ))}
                    </ul>
                  ) : null}
                </Block>
              )}
            </Section>
          </div>
          <Section title="CV design" data={s.cv_design}>
            {(cv) => (
              <Block title="CV design" actions={<Link to={`/r/${encodeURIComponent(rid)}/map?layer=fold`}>Show folds on the map</Link>}>
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
        </>
      )}
    </ViewPage>
  );
}
