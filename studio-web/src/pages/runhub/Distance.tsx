// Distance & baselines (SPEC §6.4): skill vs CV block size on a log axis (stack with its
// fold band, base models and baselines in grey, the main design marked, the random-points
// "leaky reference"), the baseline forest (ΔMSE ± 2 SE), the verdict and block wins. A
// section of a stage the run skipped says why, with the Overview's "Re-run with …" remedy.
import type { DistanceSections } from "../../api/runs";
import { Bars, Forest, LineBand, type LineSeries, type RefLine } from "../../charts";
import { fmtNum, fmtPct, unitLabel } from "../../theme/format";
import { Section, ViewPage, useRid, useSkippedStage, useUnits } from "./common";
import { fmtDistance } from "./format";

function SkillCurve({ c }: { c: NonNullable<DistanceSections["curve"]> }) {
  const u = unitLabel(useUnits().target);
  const keep = c.block_m.map((b, i) => (b > 0 ? i : -1)).filter((i) => i >= 0);
  const x = keep.map((i) => c.block_m[i]);
  const pick = (a: (number | null)[] | null | undefined) => (a ? keep.map((i) => a[i] ?? null) : undefined);
  const series: LineSeries[] = c.series.map((s) => ({
    id: s.id,
    label: s.label,
    x,
    y: pick(s.values) ?? [],
    lo: s.kind === "stack" ? pick(s.lo) : undefined,
    hi: s.kind === "stack" ? pick(s.hi) : undefined,
    muted: s.kind !== "stack",
    emphasis: s.kind === "stack",
    dashed: s.kind === "baseline",
  }));
  const refLines: RefLine[] = [];
  if (c.main_block_m !== null) refLines.push({ axis: "x", value: c.main_block_m, label: "main design" });
  for (const r of c.random) if (r.value !== null) refLines.push({ axis: "y", value: r.value, label: `${r.label} (leaky reference)` });
  const stack = c.series.find((s) => s.kind === "stack");
  const firstLast = stack ? [stack.values[keep[0]], stack.values[keep[keep.length - 1]]] : [null, null];
  const metric = c.metric === "r2" ? "R²" : "RMSE";
  const caption =
    stack && firstLast[0] !== null && firstLast[1] !== null && x.length > 1
      ? `Stack ${metric} goes from ${fmtNum(firstLast[0], 3)} with ${fmtDistance(x[0])} blocks to ${fmtNum(firstLast[1], 3)} with ${fmtDistance(x[x.length - 1])} blocks: larger blocks test prediction further from training data. The band is the fold min–max.`
      : "Skill against CV block size; the band is the fold min–max.";
  return (
    <LineBand
      title={`${metric} vs CV block size`}
      units={c.metric === "r2" ? "R² (held out)" : u}
      series={series}
      xLabel="Block size"
      xUnit="m"
      xLog
      xDecimals={0}
      yLabel={`Held-out ${metric}`}
      yUnit={c.metric === "rmse" ? u : undefined}
      refLines={refLines}
      decimals={3}
      caption={caption}
    />
  );
}

export default function Distance() {
  const rid = useRid();
  const noCurve = useSkippedStage(rid, "cv_curve", "The skill-vs-distance stage");
  const noBaselines = useSkippedStage(rid, "baselines", "The baselines stage");
  return (
    <ViewPage
      view="distance"
      title="Distance & baselines"
      intro="How skill holds up further from the training data, and whether the stack beats simple baselines on the same blocks."
    >
      {(s, vm) => {
        const u = unitLabel(vm.units.target);
        return (
        <>
          <Section title="Verdict" data={s.verdict} absent={noBaselines}>
            {(v) => (
              <div className="callout" data-tone={v.stack_wins === false ? undefined : "info"} role="status">
                <strong>Verdict:</strong> {v.text}
                {v.best_baseline ? <span className="cap"> (strongest baseline: {v.best_baseline})</span> : null}
              </div>
            )}
          </Section>
          <Section title="Skill vs block size" data={s.curve} absent={noCurve}>
            {(c) => <SkillCurve c={c} />}
          </Section>
          <div className="grid2">
            <Section title="Baseline forest" data={s.baselines} absent={noBaselines}>
              {(b) => (
                <Forest
                  title="Stack vs baselines"
                  units={`ΔMSE (baseline − stack), ${u ? `${u}²` : "target units²"}`}
                  rows={b.map((r) => ({ id: r.id, label: r.label, est: r.delta_mse, se: r.delta_mse_se }))}
                  z={2}
                  better={{ side: "positive", label: "stack better" }}
                  valueLabel="ΔMSE"
                  unit={u ? `${u}²` : undefined}
                  decimals={3}
                  caption={`${b.filter((r) => r.stack_better).length} of ${b.length} baselines are beaten by more than 2 SE; ${b.filter((r) => r.baseline_better).length} beat the stack.`}
                />
              )}
            </Section>
            <Section title="Block wins" data={s.block_wins} absent={noBaselines}>
              {(w) => (
                <Bars
                  title="Share of blocks the stack wins"
                  categories={w.map((r) => r.label)}
                  orientation="h"
                  series={[{ id: "frac", label: "Blocks won", values: w.map((r) => (r.frac === null ? null : r.frac * 100)) }]}
                  valueLabel="Blocks won"
                  unit="%"
                  domain={[0, 100]}
                  decimals={0}
                  caption={
                    w.length
                      ? `Against each baseline, the share of CV blocks where the stack has the lower error (${w.map((r) => `${r.label}: ${fmtPct(r.frac, 0)}`).join("; ")}).`
                      : undefined
                  }
                />
              )}
            </Section>
          </div>
        </>
        );
      }}
    </ViewPage>
  );
}
