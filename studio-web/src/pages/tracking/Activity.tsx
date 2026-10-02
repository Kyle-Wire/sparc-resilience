// `/jobs` — Activity (SPEC §3.2, §10.4): the queue per lane with pause/resume and reordering,
// running jobs, interrupted jobs with Resume, and the job history with filters.
import { useEffect, useState } from "react";
import { errorMessage } from "../../api/client";
import { invalidate, useResource } from "../../api/resource";
import {
  deleteJob,
  getQueue,
  listJobs,
  pauseQueue,
  resumeQueue,
  resumeRun,
  retryJob,
  setJobPriority,
  type QueueState,
} from "../../api/tracking";
import type { Job, Page } from "../../api/types";
import { Button } from "../../components/ui/Button";
import { Card } from "../../components/ui/Card";
import { Chips } from "../../components/ui/Chips";
import { ConfirmDialog } from "../../components/ui/Dialog";
import { EmptyState } from "../../components/ui/EmptyState";
import { IconButton } from "../../components/ui/IconButton";
import { JobStrip } from "../../components/ui/JobStrip";
import { Select } from "../../components/ui/Select";
import { StatusChip } from "../../components/ui/StatusChip";
import { Table, type Column } from "../../components/ui/Table";
import { useProjects } from "../../layouts/resources";
import { codecs, Link, navigate, useUrlState } from "../../router";
import { activeJobs, useJobs } from "../../stores/jobs";
import { toast } from "../../stores/ui";
import { fmtDateTime, fmtDuration, fmtRelative } from "../../theme/format";
import { useMeta } from "./hooks";
import { queueMovePatches } from "./model";
import "./tracking.css";

const HISTORY_STATUSES = ["succeeded", "failed", "cancelled", "interrupted"] as const;

function duration(j: Job): number | null {
  if (!j.started_utc) return null;
  const end = j.finished_utc ? Date.parse(j.finished_utc) : Date.now();
  return (end - Date.parse(j.started_utc)) / 1000;
}

/**
 * Whether a finished job can be started again from here: a `run.external` pseudo-job only
 * follows a command-line run (Studio cannot start it again; its run can be rerun from the run page).
 */
export function canRestart(j: Pick<Job, "kind" | "status">): boolean {
  return j.status !== "succeeded" && j.kind !== "run.external";
}

/** Resume a run job from its checkpoint, or retry any other job; opens the new job. */
async function resumeOrRetry(j: Job): Promise<void> {
  try {
    const next = j.kind === "run.core" && j.run_id ? await resumeRun(j.run_id) : await retryJob(j.id);
    useJobs.getState().upsert(next);
    invalidate("jobs");
    navigate(`/jobs/${encodeURIComponent(next.id)}`);
  } catch (e) {
    toast("error", j.kind === "run.core" ? "Resume failed" : "Retry failed", { body: errorMessage(e) });
  }
}

function QueueCard() {
  const queue = useResource<QueueState>("queue", (s) => getQueue(s), { tags: ["jobs", "queue"] });
  const jobs = useJobs((s) => s.jobs);
  const [busy, setBusy] = useState(false);
  if (queue.error && !queue.data) return <EmptyState error={queue.error} />;
  const q = queue.data;
  const move = async (lane: string[], i: number, dir: -1 | 1) => {
    const queued = lane.map((id) => jobs[id]);
    if (queued.some((j) => !j)) return;
    try {
      for (const p of queueMovePatches(queued, i, dir)) useJobs.getState().upsert(await setJobPriority(p.id, p.priority));
    } catch (e) {
      toast("error", "Could not reorder the queue", { body: errorMessage(e) });
    }
    await queue.reload();
  };
  return (
    <Card
      title="Queue"
      actions={
        q ? (
          <Button
            size="small"
            icon={q.paused ? "play" : "pause"}
            busy={busy}
            aria-pressed={q.paused}
            onClick={async () => {
              setBusy(true);
              try {
                await (q.paused ? resumeQueue() : pauseQueue());
                await queue.reload();
              } catch (e) {
                toast("error", "Could not change the queue", { body: errorMessage(e) });
              } finally {
                setBusy(false);
              }
            }}
          >
            {q.paused ? "Resume queue" : "Pause queue"}
          </Button>
        ) : null
      }
    >
      {q?.paused ? (
        <div className="callout" role="status">
          The queue is paused: running jobs continue, queued jobs wait.
        </div>
      ) : null}
      {q ? (
        <table className="checklist" aria-label="Lanes">
          <thead>
            <tr>
              <th scope="col">Lane</th>
              <th scope="col" className="r">
                Slots
              </th>
              <th scope="col">Running</th>
              <th scope="col">Queued (first starts next)</th>
            </tr>
          </thead>
          <tbody>
            {q.lanes.map((l) => (
              <tr key={l.lane}>
                <th scope="row">{l.lane}</th>
                <td className="num r">
                  {l.running.length}/{l.slots}
                </td>
                <td>
                  {l.running.length
                    ? l.running.map((id) => (
                        <div key={id}>
                          <Link to={`/jobs/${encodeURIComponent(id)}`}>{jobs[id]?.label || id}</Link>
                        </div>
                      ))
                    : "—"}
                </td>
                <td>
                  {l.queued.length ? (
                    <ol className="stack" style={{ gap: 2, margin: 0, paddingLeft: "1.2em" }}>
                      {l.queued.map((id, i) => (
                        <li key={id}>
                          <span className="row" style={{ gap: 4 }}>
                            <Link to={`/jobs/${encodeURIComponent(id)}`}>{jobs[id]?.label || id}</Link>
                            {jobs[id]?.status === "blocked" ? <StatusChip status="blocked" title={jobs[id]?.blocked?.reason} /> : null}
                            <IconButton size="small" icon="chevronUp" label={`Move ${jobs[id]?.label || id} up`} disabled={i === 0} onClick={() => void move(l.queued, i, -1)} />
                            <IconButton size="small" icon="chevronDown" label={`Move ${jobs[id]?.label || id} down`} disabled={i === l.queued.length - 1} onClick={() => void move(l.queued, i, 1)} />
                          </span>
                        </li>
                      ))}
                    </ol>
                  ) : (
                    "—"
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="cap">Loading the queue…</p>
      )}
    </Card>
  );
}

function History() {
  const [statuses, setStatuses] = useUrlState("status", codecs.list());
  const [kind, setKind] = useUrlState("kind", codecs.string(""));
  const [project, setProject] = useUrlState("project", codecs.string(""));
  const meta = useMeta();
  const projects = useProjects();
  const status = statuses.length ? statuses : [...HISTORY_STATUSES];
  const key = `jobs:history:${status.join(",")}|${kind}|${project}`;
  const first = useResource<Page<Job>>(key, (s) => listJobs({ status, kind: kind || undefined, project: project || undefined, limit: 50 }, s), { tags: ["jobs"], keepPrevious: true });
  const [more, setMore] = useState<{ key: string; items: Job[]; next: string | null } | null>(null);
  const [toDelete, setToDelete] = useState<Job | null>(null);
  const [deleting, setDeleting] = useState(false);
  useEffect(() => setMore(null), [key]);
  const items = [...(first.data?.items ?? []), ...(more?.key === key ? more.items : [])];
  const next = more?.key === key ? more.next : first.data?.next_cursor ?? null;
  const projName = (pid: string | null) => (pid ? projects.data?.find((p) => p.id === pid)?.name ?? pid : "—");
  const cols: Column<Job>[] = [
    { key: "status", label: "Status", value: (j) => j.status, render: (j) => <StatusChip status={j.status} /> },
    { key: "label", label: "Job", value: (j) => j.label, render: (j) => <Link to={`/jobs/${encodeURIComponent(j.id)}`}>{j.label || j.kind}</Link> },
    { key: "kind", label: "Kind", value: (j) => j.kind, render: (j) => <span className="mono">{j.kind}</span> },
    { key: "project", label: "Project", value: (j) => projName(j.project_id) },
    { key: "created", label: "Created", value: (j) => j.created_utc, render: (j) => <span title={fmtDateTime(j.created_utc)}>{fmtRelative(j.created_utc)}</span> },
    { key: "duration", label: "Duration", align: "right", value: (j) => duration(j), render: (j) => fmtDuration(duration(j)) },
    { key: "error", label: "Outcome", sortable: false, value: (j) => j.error?.message ?? (j.exit_code !== null ? `exit ${j.exit_code}` : null), render: (j) => <span className="cap">{j.error?.message ?? (j.exit_code !== null ? `exit ${j.exit_code}` : "—")}</span> },
    {
      key: "actions",
      label: "Actions",
      sortable: false,
      render: (j) => (
        <span className="row" style={{ gap: 4 }}>
          {canRestart(j) ? (
            <Button size="small" variant="ghost" onClick={() => void resumeOrRetry(j)}>
              {j.kind === "run.core" ? "Resume" : "Retry"}
            </Button>
          ) : null}
          <IconButton size="small" icon="x" label={`Delete ${j.label || j.id}`} onClick={() => setToDelete(j)} />
        </span>
      ),
    },
  ];
  const kinds = [...new Set([...(meta.data?.job_kinds.map((k) => k.kind) ?? []), ...items.map((j) => j.kind)])].sort();
  return (
    <Card title="History">
      <div className="filters">
        <div className="field">
          <span className="field-label">Status</span>
          <Chips
            label="Status"
            items={HISTORY_STATUSES.map((s) => ({ value: s, label: s }))}
            selected={statuses}
            onToggle={(v, on) => setStatuses(on ? [...statuses, v] : statuses.filter((x) => x !== v))}
          />
        </div>
        <div className="field">
          <label htmlFor="hist-kind">Kind</label>
          <Select id="hist-kind" value={kind} onChange={setKind} options={[{ value: "", label: "All kinds" }, ...kinds.map((k) => ({ value: k, label: k }))]} />
        </div>
        <div className="field">
          <label htmlFor="hist-project">Project</label>
          <Select id="hist-project" value={project} onChange={setProject} options={[{ value: "", label: "All projects" }, ...(projects.data ?? []).map((p) => ({ value: p.id, label: p.name }))]} />
        </div>
      </div>
      {first.error && !first.data ? (
        <EmptyState error={first.error} />
      ) : (
        <Table columns={cols} rows={items} rowKey={(j) => j.id} caption="Finished jobs" empty={first.loading ? "Loading…" : "No finished jobs match these filters."} csvName="jobs" />
      )}
      {next ? (
        <Button
          size="small"
          onClick={async () => {
            try {
              const page = await listJobs({ status, kind: kind || undefined, project: project || undefined, limit: 50, cursor: next });
              setMore({ key, items: [...(more?.key === key ? more.items : []), ...page.items], next: page.next_cursor });
            } catch (e) {
              toast("error", "Could not load more jobs", { body: errorMessage(e) });
            }
          }}
        >
          Load more
        </Button>
      ) : null}
      <ConfirmDialog
        open={!!toDelete}
        onClose={() => setToDelete(null)}
        title="Delete this job?"
        confirmLabel="Delete job and its logs"
        danger
        busy={deleting}
        onConfirm={async () => {
          if (!toDelete) return;
          setDeleting(true);
          try {
            await deleteJob(toDelete.id, true);
            useJobs.getState().remove(toDelete.id);
            invalidate("jobs");
            toast("success", "Job deleted");
            setToDelete(null);
          } catch (e) {
            toast("error", "Could not delete the job", { body: errorMessage(e) });
          } finally {
            setDeleting(false);
          }
        }}
      >
        {toDelete ? `${toDelete.label || toDelete.kind} and its job directory (events, stdout, stderr) are removed. Run outputs are not touched.` : null}
      </ConfirmDialog>
    </Card>
  );
}

export default function Activity() {
  const jobs = useJobs((s) => s.jobs);
  const running = activeJobs(jobs).filter((j) => j.status === "running" || j.status === "starting" || j.status === "cancelling");
  const interrupted = useResource<Page<Job>>("jobs:interrupted", (s) => listJobs({ status: "interrupted", limit: 100 }, s), { tags: ["jobs"] });
  return (
    <div className="stack">
      <header className="page-head">
        <h1>Activity</h1>
        <p className="cap">Every job Studio runs or follows: queued, running, interrupted and finished.</p>
      </header>
      <QueueCard />
      <Card title="Running">
        {running.length ? (
          <div className="stack" style={{ gap: 8 }}>
            {running.map((j) => (
              <JobStrip key={j.id} jobId={j.id} />
            ))}
          </div>
        ) : (
          <p className="cap">Nothing is running.</p>
        )}
      </Card>
      {interrupted.data?.items.length ? (
        <Card title="Interrupted" eyebrow="the worker vanished (server stopped, sleep, out of memory)">
          <ul className="spine">
            {interrupted.data.items.map((j) => (
              <li key={j.id}>
                <span>
                  <StatusChip status="interrupted" /> <Link to={`/jobs/${encodeURIComponent(j.id)}`}>{j.label || j.kind}</Link>
                </span>
                <span className="cap">
                  {j.error?.message ?? "process vanished"} · {fmtRelative(j.finished_utc ?? j.created_utc)}
                </span>
                {canRestart(j) ? (
                  <Button size="small" variant="primary" icon="play" onClick={() => void resumeOrRetry(j)}>
                    {j.kind === "run.core" ? "Resume" : "Retry"}
                  </Button>
                ) : (
                  <span className="cap">started from the command line</span>
                )}
              </li>
            ))}
          </ul>
        </Card>
      ) : null}
      <History />
    </div>
  );
}
