// Home (`/`, SPEC §3.2, J1, J10): project cards (name, active run, readiness score, last
// activity), the quickstarts (Try a synthetic city, Open Providence example, Import a config
// or run folder with the pickle-trust confirmation, New blank project), the jobs running
// across projects, and the disk and engine strip.
import { useState, type ReactNode } from "react";
import { ApiError, errorMessage } from "../../api/client";
import { createProject, importProject, importRun, useProjectList, useStorageSummary, type CreateProjectResult } from "../../api/projects";
import { invalidate } from "../../api/resource";
import type { Project } from "../../api/types";
import { Badge, DemoBadge } from "../../components/ui/Badge";
import { Button } from "../../components/ui/Button";
import { Card } from "../../components/ui/Card";
import { ConfirmDialog, Dialog } from "../../components/ui/Dialog";
import { EmptyState } from "../../components/ui/EmptyState";
import { Icon, type IconName } from "../../components/ui/Icon";
import { JobStrip } from "../../components/ui/JobStrip";
import { ProgressBar } from "../../components/ui/ProgressBar";
import { Seg } from "../../components/ui/Seg";
import { StatusChip } from "../../components/ui/StatusChip";
import { Link, navigate } from "../../router";
import { activeJobs, useJobs } from "../../stores/jobs";
import { toast } from "../../stores/ui";
import { fmtBytes, fmtNum, fmtRelative } from "../../theme/format";
import "./projects.css";

/** `base`, or `base 2`, `base 3`… when a project already has that name. */
export function uniqueName(base: string, projects: readonly Pick<Project, "name">[]): string {
  const taken = new Set(projects.map((p) => p.name.trim().toLowerCase()));
  if (!taken.has(base.toLowerCase())) return base;
  let i = 2;
  while (taken.has(`${base} ${i}`.toLowerCase())) i++;
  return `${base} ${i}`;
}

function lines(text: string): string[] {
  return text
    .split("\n")
    .map((s) => s.trim())
    .filter(Boolean);
}

function announceCreated(r: CreateProjectResult): void {
  invalidate("projects");
  for (const w of r.warnings) toast("warning", w);
  toast("success", `Created ${r.project.name}`, {
    body: r.imported_runs.length ? `Imported ${r.imported_runs.length} existing ${r.imported_runs.length === 1 ? "run" : "runs"} in place.` : undefined,
  });
}

export function ProjectCard({ p }: { p: Project }) {
  const score = p.readiness_score;
  const frac = score.total ? score.done / score.total : null;
  return (
    <article className="card proj-card" data-project={p.id}>
      <header>
        <div>
          <h3>
            <Link to={`/p/${encodeURIComponent(p.id)}`}>{p.name}</Link>
          </h3>
          <span className="cap">
            {p.template ? p.template.replace(/_/g, " ") : "project"} · updated {fmtRelative(p.updated_utc)}
          </span>
        </div>
        <div className="row">
          {p.demo ? <DemoBadge /> : null}
          {p.archived ? <Badge>archived</Badge> : null}
        </div>
      </header>
      <div className="stack" style={{ gap: 4 }}>
        <span className="cap">
          Readiness {score.done}/{score.total}
        </span>
        <ProgressBar value={frac} label={`${p.name} readiness`} tone={frac === 1 ? "good" : undefined} />
      </div>
      <dl className="kv">
        <div className="kv-row">
          <dt>Runs</dt>
          <dd>{p.n_runs}</dd>
        </div>
        <div className="kv-row">
          <dt>Last run</dt>
          <dd>
            {p.last_run ? (
              <span className="row" style={{ justifyContent: "flex-end", gap: 6 }}>
                <StatusChip status={p.last_run.status} />
                {p.last_run.r2 !== null ? <span className="num">R² {fmtNum(p.last_run.r2, 3)}</span> : null}
                <span className="cap">{fmtRelative(p.last_run.created_utc)}</span>
              </span>
            ) : (
              "none yet"
            )}
          </dd>
        </div>
        <div className="kv-row">
          <dt>Active run</dt>
          <dd className="mono">{p.active_run_id ? <Link to={`/r/${encodeURIComponent(p.active_run_id)}`}>{p.active_run_id}</Link> : "—"}</dd>
        </div>
        {p.active_jobs ? (
          <div className="kv-row">
            <dt>Jobs</dt>
            <dd>{p.active_jobs} running or queued</dd>
          </div>
        ) : null}
      </dl>
      <div className="row">
        <Link className="btn small" to={`/p/${encodeURIComponent(p.id)}`}>
          Open
        </Link>
        <Link className="btn small ghost" to={`/p/${encodeURIComponent(p.id)}/setup/data`}>
          Setup
        </Link>
        <Link className="btn small ghost" to={`/p/${encodeURIComponent(p.id)}/launch`}>
          Launch
        </Link>
      </div>
    </article>
  );
}

function Quickstart({ icon, title, body, children }: { icon: IconName; title: string; body: ReactNode; children: ReactNode }) {
  return (
    <div className="quickstart">
      <Icon name={icon} size={20} />
      <strong>{title}</strong>
      <span className="cap">{body}</span>
      <div className="row">{children}</div>
    </div>
  );
}

function NameDialog({
  open,
  title,
  initial,
  onClose,
  onSubmit,
  children,
  submitLabel,
}: {
  open: boolean;
  title: string;
  initial: string;
  onClose: () => void;
  onSubmit: (name: string) => Promise<void>;
  children?: ReactNode;
  submitLabel: string;
}) {
  const [name, setName] = useState(initial);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const submit = async () => {
    setBusy(true);
    setError(null);
    try {
      await onSubmit(name.trim());
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  };
  return (
    <Dialog
      open={open}
      onClose={onClose}
      busy={busy}
      title={title}
      footer={
        <>
          <Button onClick={onClose} disabled={busy}>
            Cancel
          </Button>
          <Button variant="primary" busy={busy} disabled={!name.trim()} onClick={() => void submit()}>
            {submitLabel}
          </Button>
        </>
      }
    >
      <div className="stack" style={{ gap: 10 }}>
        <label className="field">
          <span className="field-label">Project name</span>
          <input data-autofocus value={name} onChange={(e) => setName(e.target.value)} onKeyDown={(e) => e.key === "Enter" && name.trim() && void submit()} aria-label="Project name" />
        </label>
        {children}
        {error ? (
          <div className="callout" data-tone="crit" role="alert">
            {error}
          </div>
        ) : null}
      </div>
    </Dialog>
  );
}

const PICKLE_RISK = "Checkpoint files execute code when loaded; import only folders you produced.";

function ImportDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [kind, setKind] = useState<"config" | "run">("config");
  const [name, setName] = useState("");
  const [configPath, setConfigPath] = useState("");
  const [copyData, setCopyData] = useState(false);
  const [runDirs, setRunDirs] = useState("");
  const [studyDirs, setStudyDirs] = useState("");
  const [runDir, setRunDir] = useState("");
  const [trust, setTrust] = useState(false);
  const [confirm, setConfirm] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const ready = kind === "config" ? !!configPath.trim() : !!runDir.trim();

  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      if (kind === "config") {
        const r = await importProject({
          config_path: configPath.trim(),
          ...(name.trim() ? { name: name.trim() } : {}),
          copy_data: copyData,
          ...(lines(runDirs).length ? { run_dirs: lines(runDirs) } : {}),
          ...(lines(studyDirs).length ? { study_dirs: lines(studyDirs) } : {}),
          trust_pickles: trust,
        });
        invalidate("projects");
        for (const w of r.warnings) toast("warning", w);
        toast("success", `Imported ${r.project.name}`, { body: `${r.runs.length} runs, ${r.studies.length} studies` });
        onClose();
        navigate(`/p/${encodeURIComponent(r.project.id)}`);
      } else {
        const r = await importRun({ dir: runDir.trim(), ...(configPath.trim() ? { config_path: configPath.trim() } : {}), trust_pickles: trust });
        invalidate("projects");
        invalidate("runs");
        toast("success", `Imported run ${r.label || r.id}`);
        onClose();
        navigate(`/r/${encodeURIComponent(r.id)}`);
      }
    } catch (e) {
      if (e instanceof ApiError && e.code === "needs_config")
        setError(`${e.message} Give the config the run was made with (it has no provenance to rebuild it from).`);
      else setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <Dialog
        open={open && !confirm}
        onClose={onClose}
        busy={busy}
        wide
        title="Import a config or run folder"
        footer={
          <>
            <Button onClick={onClose} disabled={busy}>
              Cancel
            </Button>
            <Button variant="primary" busy={busy} disabled={!ready} onClick={() => (trust ? setConfirm(true) : void run())}>
              Import
            </Button>
          </>
        }
      >
        <div className="stack" style={{ gap: 10 }}>
          <Seg
            label="What to import"
            value={kind}
            options={[
              { value: "config", label: "A config (new project)" },
              { value: "run", label: "A run folder" },
            ]}
            onChange={setKind}
          />
          {kind === "config" ? (
            <>
              <label className="field">
                <span className="field-label">Config file (absolute path to a core YAML)</span>
                <input value={configPath} onChange={(e) => setConfigPath(e.target.value)} placeholder="/home/me/city/configs/core_city.yml" aria-label="Config path" />
              </label>
              <label className="field">
                <span className="field-label">Project name (default: from the config)</span>
                <input value={name} onChange={(e) => setName(e.target.value)} aria-label="Project name" />
              </label>
              <label className="row">
                <input type="checkbox" checked={copyData} onChange={(e) => setCopyData(e.target.checked)} /> Copy the data files into the project (otherwise paths are made absolute)
              </label>
              <label className="field">
                <span className="field-label">Run folders to import in place (one per line, optional)</span>
                <textarea rows={2} value={runDirs} onChange={(e) => setRunDirs(e.target.value)} aria-label="Run folders" />
              </label>
              <label className="field">
                <span className="field-label">Study folders (placebo, simcheck, multiverse; one per line, optional)</span>
                <textarea rows={2} value={studyDirs} onChange={(e) => setStudyDirs(e.target.value)} aria-label="Study folders" />
              </label>
            </>
          ) : (
            <>
              <label className="field">
                <span className="field-label">Run folder (absolute path; it stays where it is)</span>
                <input value={runDir} onChange={(e) => setRunDir(e.target.value)} placeholder="/home/me/output/core/city" aria-label="Run folder" />
              </label>
              <label className="field">
                <span className="field-label">Config it was made with (only when the run has no provenance)</span>
                <input value={configPath} onChange={(e) => setConfigPath(e.target.value)} aria-label="Config path" />
              </label>
            </>
          )}
          <label className="row trust-pickles">
            <input type="checkbox" checked={trust} onChange={(e) => setTrust(e.target.checked)} /> Trust the checkpoint files (needed for the Scenario Lab and studies)
          </label>
          <p className="cap">{PICKLE_RISK} Without trust, the runs' outputs are browsable but their checkpoints are never loaded.</p>
          {error ? (
            <div className="callout" data-tone="crit" role="alert">
              {error}
            </div>
          ) : null}
        </div>
      </Dialog>
      <ConfirmDialog
        open={open && confirm}
        title="Trust these checkpoint files?"
        confirmLabel="I produced these folders: import"
        danger
        busy={busy}
        onClose={() => setConfirm(false)}
        onConfirm={async () => {
          await run();
          setConfirm(false);
        }}
      >
        <p>
          <b>{PICKLE_RISK}</b>
        </p>
        <p>A checkpoint is a Python pickle. Loading one from someone else can run arbitrary code on this computer.</p>
      </ConfirmDialog>
    </>
  );
}

function Quickstarts({ projects }: { projects: Project[] }) {
  const [busy, setBusy] = useState(false);
  const [dialog, setDialog] = useState<"providence" | "import" | "blank" | null>(null);
  const [importRuns, setImportRuns] = useState(true);
  const synthetic = async () => {
    setBusy(true);
    try {
      // One click, no dialog: the name must be free. Home lists only active projects, so an
      // archived "Synthetic city" (or a name with the same slug) answers 409; try the next one.
      const taken: Pick<Project, "name">[] = [...projects];
      let r: CreateProjectResult | null = null;
      for (let attempt = 0; !r; attempt++) {
        const name = uniqueName("Synthetic city", taken);
        try {
          r = await createProject({ name, template: "synthetic_demo" });
        } catch (e) {
          if (!(e instanceof ApiError && e.code === "conflict") || attempt >= 20) throw e;
          taken.push({ name });
        }
      }
      announceCreated(r);
      navigate(`/p/${encodeURIComponent(r.project.id)}`);
    } catch (e) {
      toast("error", "Could not create the synthetic city", { body: errorMessage(e) });
    } finally {
      setBusy(false);
    }
  };
  return (
    <section className="quickstarts" aria-label="Quickstarts">
      <Quickstart icon="bolt" title="Try a synthetic city" body="A DEMO city with planted effects, people layers and climate factors, ready to launch in under a minute.">
        <Button variant="primary" busy={busy} onClick={() => void synthetic()}>
          Try a synthetic city
        </Button>
      </Quickstart>
      <Quickstart icon="layers" title="Open Providence example" body="The Providence heat study: 54,701 cells, forcing, CMIP6 table and layers; optionally its existing runs.">
        <Button onClick={() => setDialog("providence")}>Open Providence example</Button>
      </Quickstart>
      <Quickstart icon="folder" title="Import a config / run folder" body="Bring an existing core YAML, run folders and studies in place.">
        <Button onClick={() => setDialog("import")}>Import…</Button>
      </Quickstart>
      <Quickstart icon="plus" title="New blank project" body="Start from your own CSV of street-level temperatures.">
        <Button onClick={() => setDialog("blank")}>New blank project</Button>
      </Quickstart>
      <NameDialog
        key={`providence:${dialog === "providence"}`}
        open={dialog === "providence"}
        title="Open the Providence example"
        initial={uniqueName("Providence", projects)}
        submitLabel="Create"
        onClose={() => setDialog(null)}
        onSubmit={async (name) => {
          const r = await createProject({ name, template: "providence_example", options: { import_existing_runs: importRuns } });
          announceCreated(r);
          setDialog(null);
          navigate(`/p/${encodeURIComponent(r.project.id)}`);
        }}
      >
        <label className="row">
          <input type="checkbox" checked={importRuns} onChange={(e) => setImportRuns(e.target.checked)} /> Import the existing Providence runs and studies in place (from a repo checkout)
        </label>
        <p className="cap">Copies brown4.csv, the 2020-07-29 forcing, the CMIP6 table and the layers into the project.</p>
      </NameDialog>
      <NameDialog
        key={`blank:${dialog === "blank"}`}
        open={dialog === "blank"}
        title="New blank project"
        initial={uniqueName("New city", projects)}
        submitLabel="Create and set up"
        onClose={() => setDialog(null)}
        onSubmit={async (name) => {
          const r = await createProject({ name, template: "blank" });
          announceCreated(r);
          setDialog(null);
          navigate(`/p/${encodeURIComponent(r.project.id)}/setup/data`);
        }}
      />
      <ImportDialog key={`import:${dialog === "import"}`} open={dialog === "import"} onClose={() => setDialog(null)} />
    </section>
  );
}

function SystemStrip() {
  const storage = useStorageSummary();
  const host = useJobs((s) => s.host);
  const s = storage.data;
  const engine = host.state ?? "absent";
  return (
    <section className="system-strip" aria-label="Disk and engine">
      <span className="row">
        <Icon name="folder" /> Workspace {s ? fmtBytes(s.workspace_bytes) : "—"} · {s ? `${fmtBytes(s.free_bytes)} free` : "disk unknown"}
      </span>
      <span className="row">
        <Icon name="bolt" /> Scenario engine <StatusChip status={engine} />
      </span>
      <Link to="/settings" className="btn small ghost">
        Settings
      </Link>
    </section>
  );
}

export default function Home() {
  const list = useProjectList(false);
  const jobs = useJobs((s) => s.jobs);
  const running = activeJobs(jobs);
  const projects = list.data ?? [];
  return (
    <div className="stack home">
      <header className="page-head">
        <p className="eyebrow">SPARC Studio</p>
        <h1>Street-level heat, from a CSV to cooling decisions</h1>
      </header>
      <Quickstarts projects={projects} />
      {running.length ? (
        <Card title="Running now" eyebrow={`${running.length} ${running.length === 1 ? "job" : "jobs"}`} actions={<Link to="/jobs">Activity</Link>}>
          <div className="stack" style={{ gap: 6 }}>
            {running.slice(0, 6).map((j) => (
              <JobStrip key={j.id} jobId={j.id} />
            ))}
          </div>
        </Card>
      ) : null}
      <section aria-label="Projects" className="stack" style={{ gap: 10 }}>
        <div className="row">
          <h2>Projects</h2>
          <span className="spacer" />
          <Link to="/projects">All projects</Link>
        </div>
        {list.error && !list.data ? (
          <EmptyState error={list.error} title="Could not list the projects" />
        ) : list.data && projects.length === 0 ? (
          <EmptyState title="No projects yet" body="Start with a synthetic city, the Providence example, or your own data." />
        ) : (
          <div className="proj-cards">
            {projects.map((p) => (
              <ProjectCard key={p.id} p={p} />
            ))}
          </div>
        )}
      </section>
      <SystemStrip />
    </div>
  );
}
