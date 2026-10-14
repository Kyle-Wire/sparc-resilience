// The run's job log (SPEC §3.2; Model details › Run log, `/r/:rid/details/runlog`, which the
// old `/r/:rid/track` redirects to): Mission Control of the run's latest job, with a switcher
// over its jobs (launch, resume, post-run actions, …). `embedded` (how Model details renders it)
// leaves out the job tabs whose facts have their own Model details section (BottomTabs).
import { useResource } from "../../api/resource";
import { listJobs } from "../../api/tracking";
import { EmptyState } from "../../components/ui/EmptyState";
import { Select } from "../../components/ui/Select";
import { codecs, useRoute, useUrlState } from "../../router";
import { fmtDateTime } from "../../theme/format";
import { JobTracker } from "./mission/JobTracker";
import "./tracking.css";

export default function RunTrack({ embedded = false }: { embedded?: boolean } = {}) {
  const { params } = useRoute();
  const rid = params.rid;
  const [jobQ, setJobQ] = useUrlState("job", codecs.optString());
  const jobs = useResource(rid ? `run:${rid}:jobs` : null, (s) => listJobs({ run: rid, limit: 100 }, s), { tags: rid ? [`run:${rid}`, "jobs"] : [] });
  if (!rid) return null;
  if (jobs.error && !jobs.data) return <EmptyState error={jobs.error} />;
  const list = jobs.data?.items ?? [];
  if (!jobs.data) {
    return (
      <p className="cap" role="status">
        Loading the run's jobs…
      </p>
    );
  }
  if (!list.length) {
    return <EmptyState title="No tracked job for this run" body="This run was imported or made outside Studio, so there is no job log to follow. Its stage timings are in the run record above." />;
  }
  const current = list.find((j) => j.id === jobQ) ?? list[0];
  return (
    <div className="stack">
      {list.length > 1 ? (
        <div className="row">
          <label htmlFor="job-switch" className="cap">
            Job
          </label>
          <Select
            id="job-switch"
            value={current.id}
            onChange={(v) => setJobQ(v === list[0].id ? null : v)}
            options={list.map((j) => ({ value: j.id, label: `${j.label || j.kind} · ${j.status} · ${fmtDateTime(j.created_utc)}` }))}
          />
        </div>
      ) : null}
      <JobTracker key={current.id} jid={current.id} embedded={embedded} />
    </div>
  );
}
