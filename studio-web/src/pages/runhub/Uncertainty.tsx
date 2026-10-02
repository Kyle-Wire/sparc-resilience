// Uncertainty (SPEC §6.4): layered intervals per scenario (estimation, specification,
// attribution, causal band, envelope) around the estimate with a zero line and "excludes 0"
// badges, the climate spread table, and the evidence sources with attach/detach.
import { useState } from "react";
import { api, errorMessage } from "../../api/client";
import { invalidate } from "../../api/resource";
import type { UncertaintySections } from "../../api/runs";
import type { Job } from "../../api/types";
import { IntervalStack } from "../../charts";
import { excludesZero } from "../../charts/IntervalStack";
import { Button } from "../../components/ui/Button";
import { StatusChip } from "../../components/ui/StatusChip";
import { Link } from "../../router";
import { useJobs } from "../../stores/jobs";
import { toast } from "../../stores/ui";
import { unitLabel } from "../../theme/format";
import { Block, GenericTableView, Section, ViewPage, useRid } from "./common";

type Source = NonNullable<UncertaintySections["sources"]>[number];
type Row = NonNullable<UncertaintySections["rows"]>[number];

/**
 * Caption of the layered intervals: which scenarios keep their sign under the widest
 * (envelope) interval. ΔT < 0 is cooler, so an envelope entirely below zero still cools and
 * one entirely above zero still warms; every other scenario could be zero.
 */
export function envelopeCaption(rows: Row[]): string {
  const envelope = (r: Row) => r.layers.find((l) => /envelope/i.test(l.id)) ?? null;
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
        <Link to={`/r/${encodeURIComponent(rid)}/validation`} className="btn small">
          Run study
        </Link>
      ) : null}
    </li>
  );
}

export default function Uncertainty() {
  const rid = useRid();
  return (
    <ViewPage
      view="uncertainty"
      title="Uncertainty"
      intro="Each scenario's estimate inside nested intervals: estimation (folds), specification (multiverse), attribution (simulation check), the causal band and their envelope."
    >
      {(s, vm) => {
        const u = unitLabel(vm.units.target);
        return (
          <>
            <Section title="Layered intervals" data={s.rows}>
              {(rows) => <IntervalStack title="Uncertainty by source" units={`${u} (negative = cooler)`} rows={rows} valueLabel="City-mean ΔT" unit={u} caption={envelopeCaption(rows)} />}
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
