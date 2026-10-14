// Uncertainty (SPEC §6.4): layered intervals per scenario (estimation, specification,
// attribution, causal band, envelope) around the estimate with a zero line and "excludes 0"
// badges, the climate spread table, and the evidence sources with attach/detach. The complete
// page is Model details › Uncertainty; How much to trust this reuses `envelopeCaption` and
// `ScenarioRanges` (the one place outside the Lab with each scenario's verdict pill).
import { useState } from "react";
import { api, errorMessage } from "../../api/client";
import { invalidate } from "../../api/resource";
import type { UncertaintySections } from "../../api/runs";
import type { Job } from "../../api/types";
import { IntervalStack } from "../../charts";
import { excludesZero } from "../../charts/IntervalStack";
import { Button } from "../../components/ui/Button";
import { StatusChip } from "../../components/ui/StatusChip";
import { Link, runHref } from "../../router";
import { useJobs } from "../../stores/jobs";
import { toast } from "../../stores/ui";
import { EMPTY, fmtSignedValue, fmtTempChange, unitLabel } from "../../theme/format";
import { Block, GenericTableView, Section, VerdictPill, ViewPage, useRid, type DetailPageProps } from "./common";
import { isTempUnit, rangeWords } from "./format";

type Source = NonNullable<UncertaintySections["sources"]>[number];
type Row = NonNullable<UncertaintySections["rows"]>[number];

const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

/** A scenario's widest (envelope) interval, null when the report has none for it. */
export function envelopeOf(r: Pick<Row, "layers">): Row["layers"][number] | null {
  return r.layers.find((l) => /envelope/i.test(l.id)) ?? null;
}

/** The widest interval a scenario has: its envelope, else the widest layer with both bounds. */
export function widestOf(r: Pick<Row, "layers">): { lo: number; hi: number } | null {
  const e = envelopeOf(r);
  if (e && isNum(e.lo) && isNum(e.hi)) return { lo: e.lo, hi: e.hi };
  let best: { lo: number; hi: number } | null = null;
  for (const l of r.layers) if (isNum(l.lo) && isNum(l.hi) && (!best || l.hi - l.lo > best.hi - best.lo)) best = { lo: l.lo, hi: l.hi };
  return best;
}

/**
 * Each scenario's change with its widest range and its verdict pill (Robust / Direction only /
 * Not established), one row per scenario. The layer-by-layer chart is the full page's.
 */
export function ScenarioRanges({ rows, unit, caption = "How wide each scenario's range is" }: { rows: readonly Row[]; unit: string; caption?: string }) {
  if (!rows.length) return <p className="cap">No scenarios in the uncertainty report.</p>;
  const verdicts = rows.some((r) => r.verdict);
  return (
    <table className="tbl" aria-label={caption} data-scenario-ranges="true">
      <thead>
        <tr>
          <th scope="col">Scenario</th>
          <th scope="col">Change, city average</th>
          <th scope="col">Widest range</th>
          {verdicts ? <th scope="col">What the evidence supports</th> : null}
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => {
          const w = widestOf(r);
          return (
            <tr key={r.id} data-scenario={r.id}>
              <td>{r.label}</td>
              <td className="num">{isNum(r.estimate) ? (isTempUnit(unit) ? fmtTempChange(r.estimate, unit, 2) : fmtSignedValue(r.estimate, unit, 2)) : EMPTY}</td>
              <td className="num">{w ? rangeWords(w.lo, w.hi, unit, 2) : EMPTY}</td>
              {verdicts ? (
                <td>
                  <VerdictPill verdict={r.verdict} />
                </td>
              ) : null}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

/**
 * Caption of the layered intervals: which scenarios keep their sign under the widest
 * (envelope) interval. ΔT < 0 is cooler, so an envelope entirely below zero still cools and
 * one entirely above zero still warms; every other scenario could be zero.
 */
export function envelopeCaption(rows: Row[]): string {
  const envelope = envelopeOf;
  const withEnv = rows.filter((r) => envelope(r) !== null);
  if (!withEnv.length) return `${rows.length} scenario${rows.length === 1 ? "" : "s"}; no envelope interval yet (attach studies below to widen the evidence).`;
  const cools = withEnv.filter((r) => {
    const e = envelope(r)!;
    return excludesZero(e) && (e.hi as number) < 0;
  });
  const warms = withEnv.filter((r) => {
    const e = envelope(r)!;
    return excludesZero(e) && (e.lo as number) > 0;
  });
  const names = (rs: Row[]) => rs.map((r) => r.label).join(", ");
  const parts = [`${cools.length} of ${withEnv.length} scenarios still cool under the widest (envelope) interval${cools.length ? ` (${names(cools)})` : ""}`];
  if (warms.length) parts.push(`${warms.length} still warm${warms.length === 1 ? "s" : ""} (${names(warms)})`);
  const zero = withEnv.length - cools.length - warms.length;
  if (zero) parts.push(`${zero} could be zero`);
  return parts.join("; ") + ".";
}

function SourceRow({ rid, s }: { rid: string; s: Source }) {
  const [busy, setBusy] = useState(false);
  const toggle = async () => {
    if (!s.study_id) return;
    setBusy(true);
    try {
      const r = await api.post<{ study: unknown; job: Job | null }>(`/api/studies/${encodeURIComponent(s.study_id)}/${s.attached ? "detach" : "attach"}`, { run_id: rid });
      if (r.job) {
        useJobs.getState().upsert(r.job);
        toast("info", "Uncertainty report re-running", { href: `/jobs/${r.job.id}`, linkLabel: "Track" });
      } else toast("success", s.attached ? `${s.label} detached` : `${s.label} attached`);
      invalidate(`run:${rid}:views`);
      invalidate(`run:${rid}:studies`);
    } catch (e) {
      toast("error", `Could not ${s.attached ? "detach" : "attach"} ${s.label}`, { body: errorMessage(e) });
    } finally {
      setBusy(false);
    }
  };
  return (
    <li className="row" style={{ justifyContent: "space-between" }}>
      <span className="row">
        <strong>{s.label}</strong>
        <span className="cap">{s.kind}</span>
        {s.state ? <StatusChip status={s.state} /> : null}
        {s.attached ? <StatusChip status="done" text="attached" /> : <StatusChip status="not_run" text="not attached" />}
      </span>
      {s.study_id ? (
        <Button size="small" busy={busy} onClick={() => void toggle()}>
          {s.attached ? "Detach" : "Attach"}
        </Button>
      ) : !s.attached ? (
        <Link to={`${runHref(rid, "trust")}#validation`} className="btn small">
          Run study
        </Link>
      ) : null}
    </li>
  );
}

export default function Uncertainty({
  title = "Uncertainty",
  intro = "Each scenario's estimate inside nested intervals: estimation (folds), specification (multiverse), attribution (simulation check), the causal band and their envelope.",
}: DetailPageProps = {}) {
  const rid = useRid();
  return (
    <ViewPage
      view="uncertainty"
      title={title}
      intro={intro}
    >
      {(s, vm) => {
        const u = unitLabel(vm.units.target);
        return (
          <>
            <Section title="Layered intervals" data={s.rows}>
              {(rows) => <IntervalStack title="Uncertainty by source" units={`${u} (negative = cooler)`} rows={rows} valueLabel="City-mean ΔT" unit={u} caption={envelopeCaption(rows)} />}
            </Section>
            <Section title="Verdicts" data={s.rows}>
              {(rows) =>
                rows.some((r) => r.verdict) ? (
                  <Block title="What the evidence supports">
                    <p className="cap prose">
                      Robust: the plausible range excludes no change and every analysis variant agrees on the sign. Direction only: the sign holds but the size could be close to
                      zero. Not established: the change cannot be told apart from what the pipeline reports when no effect is planted, the intervals disagree, or most edited
                      cells are beyond the conditions the model was trained on. * marks a qualifier (extrapolation, or outside the causal band).
                    </p>
                    <table className="tbl" aria-label="Verdict per scenario">
                      <thead>
                        <tr>
                          <th scope="col">Scenario</th>
                          <th scope="col">Verdict and why</th>
                        </tr>
                      </thead>
                      <tbody>
                        {rows.map((r) => (
                          <tr key={r.id}>
                            <td>{r.label}</td>
                            <td>
                              <VerdictPill verdict={r.verdict} detail />
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </Block>
                ) : null
              }
            </Section>
            <Section title="Climate spread" data={s.climate}>
              {(t) => (
                <Block title="Climate spread">
                  <GenericTableView table={t} caption="Climate spread" csvName="climate-spread" />
                </Block>
              )}
            </Section>
            <Section title="Sources" data={s.sources}>
              {(src) => (
                <Block title="Evidence sources">
                  {src.length ? (
                    <ul className="stack" style={{ listStyle: "none", padding: 0, margin: 0, gap: 8 }}>
                      {src.map((x) => (
                        <SourceRow key={`${x.kind}-${x.study_id ?? x.label}`} rid={rid} s={x} />
                      ))}
                    </ul>
                  ) : (
                    <p className="cap">No studies feed this report yet.</p>
                  )}
                </Block>
              )}
            </Section>
          </>
        );
      }}
    </ViewPage>
  );
}
