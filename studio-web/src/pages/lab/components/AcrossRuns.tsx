// "Check across runs" output (SPEC §7.12): the latest finished check of a scenario as a dot
// plot of the city-mean ΔT ± its likely range per run, with the sign stability and the spread
// (the range of the run means is the result's specification band). A check still running
// shows its progress; runs that could not evaluate the scenario are listed with the reason.
import { useAcrossRunsJobs, type AcrossRunsResult } from "../../../api/lab";
import { isActiveStatus, type Job } from "../../../api/types";
import { DotRange } from "../../../charts";
import { JobStrip } from "../../../components/ui/JobStrip";
import { fmtDateTime, fmtNum, fmtPct, unitLabel } from "../../../theme/format";

const ofScenario = (j: Job, sid: string) => j.scenario_id === sid || (j.params as { scenario_id?: unknown } | null)?.scenario_id === sid;

/** The newest finished check of a scenario, and a newer one still running (jobs newest first). */
export function latestAcross(jobs: Job[], sid: string): { done: (Job & { result: AcrossRunsResult }) | null; active: Job | null } {
  const mine = jobs.filter((j) => ofScenario(j, sid));
  const done = mine.find((j) => j.status === "succeeded" && j.result && Array.isArray((j.result as { rows?: unknown }).rows)) as (Job & { result: AcrossRunsResult }) | undefined;
  const active = mine.find((j) => isActiveStatus(j.status)) ?? null;
  return { done: done ?? null, active };
}

export function AcrossRunsPanel({ pid, sid, unit }: { pid: string | null; sid: string; unit: string }) {
  const jobs = useAcrossRunsJobs(pid);
  const { done, active } = latestAcross(jobs.data?.items ?? [], sid);
  if (!done && !active) return null;
  const u = unitLabel(unit);
  const r = done?.result ?? null;
  const ok = r ? r.rows.filter((x) => x.ok && x.city) : [];
  const failed = r ? r.rows.filter((x) => !x.ok) : [];
  return (
    <section className="stack" aria-label="Across runs">
      <h3>Across runs</h3>
      {active ? <JobStrip jobId={active.id} /> : null}
      {r ? (
        <>
          <DotRange
            title="The same scenario on other runs"
            units={`${u} city mean (negative = cooler)`}
            rows={ok.map((x) => ({ id: x.run_id, label: x.run_id, est: x.city!.estimate, lo: x.city!.lo, hi: x.city!.hi }))}
            valueLabel="City-mean ΔT"
            unit={u}
            signed
            rangeLabel="Likely range"
            caption={`${ok.length} of ${r.rows.length} runs evaluated (${fmtDateTime(done!.finished_utc)}). Sign stability ${fmtPct(r.sign_stability)}; spread of the run means ${fmtNum(r.spread, 3)} ${u} (the result's specification band).`}
          />
          {failed.length ? (
            <ul className="edit-issues">
              {failed.map((x) => (
                <li key={x.run_id} data-level="warn">
                  {x.run_id}: {x.error ?? "not evaluated"}
                </li>
              ))}
            </ul>
          ) : null}
        </>
      ) : null}
    </section>
  );
}
