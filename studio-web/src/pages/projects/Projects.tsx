// Projects (`/projects`, SPEC §3.2): a sortable list of every project with readiness, runs,
// last run and activity; archive / unarchive and delete (refused while jobs are active).
import { useState } from "react";
import { errorMessage } from "../../api/client";
import { deleteProject, patchProject, useProjectList } from "../../api/projects";
import { invalidate } from "../../api/resource";
import type { Project } from "../../api/types";
import { Badge, DemoBadge } from "../../components/ui/Badge";
import { Button } from "../../components/ui/Button";
import { ConfirmDialog } from "../../components/ui/Dialog";
import { EmptyState } from "../../components/ui/EmptyState";
import { StatusChip } from "../../components/ui/StatusChip";
import { Table, type Column } from "../../components/ui/Table";
import { Link } from "../../router";
import { toast } from "../../stores/ui";
import { fmtDateTime, fmtNum, fmtRelative } from "../../theme/format";
import "./projects.css";

export default function Projects() {
  const [showArchived, setShowArchived] = useState(false);
  const list = useProjectList(showArchived);
  const [deleting, setDeleting] = useState<Project | null>(null);
  const [withFiles, setWithFiles] = useState(false);
  const [busy, setBusy] = useState(false);

  const archive = async (p: Project, archived: boolean) => {
    try {
      await patchProject(p.id, { archived });
      invalidate("projects");
      invalidate(`project:${p.id}`);
      toast("success", archived ? `Archived ${p.name}` : `Restored ${p.name}`);
    } catch (e) {
      toast("error", `Could not ${archived ? "archive" : "restore"} ${p.name}`, { body: errorMessage(e) });
    }
  };

  const remove = async () => {
    if (!deleting) return;
    setBusy(true);
    try {
      await deleteProject(deleting.id, withFiles);
      invalidate("projects");
      toast("success", `Deleted ${deleting.name}`, { body: withFiles ? "The project folder was removed." : "The project folder was kept on disk." });
      setDeleting(null);
    } catch (e) {
      toast("error", `Could not delete ${deleting.name}`, { body: errorMessage(e) });
    } finally {
      setBusy(false);
    }
  };

  const columns: Column<Project>[] = [
    {
      key: "name",
      label: "Project",
      value: (p) => p.name,
      render: (p) => (
        <span className="row" style={{ gap: 6 }}>
          <Link to={`/p/${encodeURIComponent(p.id)}`}>{p.name}</Link>
          {p.demo ? <DemoBadge /> : null}
          {p.archived ? <Badge>archived</Badge> : null}
        </span>
      ),
    },
    { key: "template", label: "Template", value: (p) => p.template ?? "", render: (p) => (p.template ? p.template.replace(/_/g, " ") : "—") },
    { key: "readiness", label: "Readiness", align: "right", value: (p) => (p.readiness_score.total ? p.readiness_score.done / p.readiness_score.total : null), render: (p) => `${p.readiness_score.done}/${p.readiness_score.total}` },
    { key: "n_runs", label: "Runs", align: "right", value: (p) => p.n_runs },
    {
      key: "last_run",
      label: "Last run",
      value: (p) => p.last_run?.created_utc ?? null,
      render: (p) =>
        p.last_run ? (
          <span className="row" style={{ gap: 6 }}>
            <StatusChip status={p.last_run.status} />
            {p.last_run.r2 !== null ? <span className="num">R² {fmtNum(p.last_run.r2, 3)}</span> : null}
            <span className="cap">{fmtRelative(p.last_run.created_utc)}</span>
          </span>
        ) : (
          "—"
        ),
    },
    { key: "active_jobs", label: "Active jobs", align: "right", value: (p) => p.active_jobs },
    { key: "updated_utc", label: "Updated", value: (p) => p.updated_utc, render: (p) => <span title={fmtDateTime(p.updated_utc)}>{fmtRelative(p.updated_utc)}</span> },
    {
      key: "actions",
      label: "Actions",
      sortable: false,
      value: () => null,
      render: (p) => (
        <span className="row" style={{ gap: 4 }}>
          <Button size="small" variant="ghost" onClick={() => void archive(p, !p.archived)}>
            {p.archived ? "Restore" : "Archive"}
          </Button>
          <Button
            size="small"
            variant="ghost"
            icon="x"
            disabled={p.active_jobs > 0}
            title={p.active_jobs > 0 ? "Jobs are running on this project" : undefined}
            onClick={() => {
              setWithFiles(false);
              setDeleting(p);
            }}
          >
            Delete
          </Button>
        </span>
      ),
    },
  ];

  return (
    <div className="stack projects-page">
      <header className="page-head">
        <p className="eyebrow">Workspace</p>
        <div className="row">
          <h1>Projects</h1>
          <span className="spacer" />
          <label className="row cap">
            <input type="checkbox" checked={showArchived} onChange={(e) => setShowArchived(e.target.checked)} /> Show archived
          </label>
          <Link className="btn small" to="/">
            New project
          </Link>
        </div>
      </header>
      {list.error && !list.data ? (
        <EmptyState error={list.error} title="Could not list the projects" />
      ) : (
        <Table
          columns={columns}
          rows={list.data ?? []}
          rowKey={(p) => p.id}
          caption="Projects"
          initialSort={{ key: "updated_utc", dir: "desc" }}
          empty={list.loading ? "Loading…" : "No projects yet: start one from Home."}
          csvName="projects"
        />
      )}
      <ConfirmDialog
        open={!!deleting}
        title={`Delete ${deleting?.name ?? "project"}?`}
        confirmLabel="Delete"
        danger
        busy={busy}
        onClose={() => setDeleting(null)}
        onConfirm={() => void remove()}
      >
        <p>The project leaves Studio's index, with its runs, studies and scenarios.</p>
        <label className="row">
          <input type="checkbox" checked={withFiles} onChange={(e) => setWithFiles(e.target.checked)} /> Also delete the project folder on disk ({deleting?.dir})
        </label>
        <p className="cap">Runs imported in place from elsewhere are never deleted from their own folders.</p>
      </ConfirmDialog>
    </div>
  );
}
