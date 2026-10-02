// The shell on every page (SPEC §3.1): sidebar (project switcher, project nav declared by
// route modules, Activity, Settings), top bar (breadcrumb, active-run chip, job tray, engine
// dot, connection pill, command palette, theme toggle), toasts, the aria-live job announcer
// and the browser tab title mirroring the most important running job.
import { useEffect, useMemo, type ReactNode } from "react";
import type { Project, RunSummary } from "../api/types";
import { CommandPalette } from "../components/ui/CommandPalette";
import { ConnectionPill } from "../components/ui/ConnectionPill";
import { Icon, type IconName } from "../components/ui/Icon";
import { IconButton } from "../components/ui/IconButton";
import { JobTray } from "../components/ui/JobTray";
import { Toasts } from "../components/ui/Toast";
import { Tooltip } from "../components/ui/Tooltip";
import { Badge, modeLabel } from "../components/ui/Badge";
import { Link, matchRoute, navigate, runTabHref, useLocation, useRegistry, useRoute, type Registry, type RouteDef } from "../router";
import { activeJobs, connectJobs, tabTitle, useJobs } from "../stores/jobs";
import { useUi, type Command, type ThemePref } from "../stores/ui";
import { fmtDate, fmtNum, fmtPct } from "../theme/format";
import { useProject, useProjects, useRecentRuns, useRunDetail } from "./resources";

const THEMES: { value: ThemePref; icon: IconName; label: string }[] = [
  { value: "system", icon: "monitor", label: "Theme: system" },
  { value: "light", icon: "sun", label: "Theme: light" },
  { value: "dark", icon: "moon", label: "Theme: dark" },
];

function ThemeToggle() {
  const theme = useUi((s) => s.theme);
  const setTheme = useUi((s) => s.setTheme);
  const i = Math.max(0, THEMES.findIndex((t) => t.value === theme));
  const cur = THEMES[i];
  const next = THEMES[(i + 1) % THEMES.length];
  return <IconButton icon={cur.icon} label={`${cur.label} (switch to ${next.value})`} onClick={() => setTheme(next.value)} />;
}

function EngineDot({ runId }: { runId: string | null }) {
  const host = useJobs((s) => s.host);
  const perRun = useJobs((s) => (runId ? s.engineByRun[runId] : undefined));
  const info = perRun ?? host;
  const state = info.state ?? "absent";
  const text =
    state === "loading" && info.progress !== null ? `loading ${fmtPct(info.progress)}` : state === "absent" ? "off" : String(state).replace(/_/g, " ");
  const shown = state === "absent" || state === "no_checkpoint" ? "cold" : state;
  return (
    <span className="engine-dot hide-narrow" data-state={shown} title={`Scenario engine: ${text}${info.rss_mb ? ` · ${fmtNum(info.rss_mb / 1024, 1)} GB` : ""}`} role="status" aria-label={`Engine ${text}`}>
      <i aria-hidden="true" />
      engine {text}
    </span>
  );
}

function ProjectSwitcher({ projectId }: { projectId: string | null }) {
  const { data } = useProjects();
  const list = (data ?? []).filter((p) => !p.archived || p.id === projectId);
  return (
    <div className="field" style={{ padding: "0 4px 6px" }}>
      <label htmlFor="project-switcher" className="eyebrow">
        Project
      </label>
      <select
        id="project-switcher"
        value={projectId ?? ""}
        onChange={(e) => {
          if (e.target.value === "__all") navigate("/projects");
          else if (e.target.value) navigate(`/p/${e.target.value}`);
        }}
      >
        <option value="">{list.length ? "Choose a project…" : "No projects yet"}</option>
        {list.map((p) => (
          <option key={p.id} value={p.id}>
            {p.name}
            {p.demo ? " (demo)" : ""}
          </option>
        ))}
        <option value="__all">All projects…</option>
      </select>
    </div>
  );
}

/**
 * The project nav entry the current page belongs to: the entry its route declared, else the
 * entry whose target is the longest prefix of the path. The project root (Overview) prefixes
 * every project page, so it is current only on its own route.
 */
export function currentProjectNav(entries: { id: string; to: string; route: RouteDef }[], pathname: string, route: RouteDef | null): string | null {
  const own = entries.find((e) => e.route === route);
  if (own) return own.id;
  let best: string | null = null;
  let len = -1;
  for (const e of entries) {
    const p = e.to.split(/[?#]/)[0].replace(/\/+$/, "");
    if (/^\/p\/[^/]+$/.test(p)) continue;
    if ((pathname === p || pathname.startsWith(p + "/")) && p.length > len) {
      best = e.id;
      len = p.length;
    }
  }
  return best;
}

/** The project nav built from route modules' `projectNav` declarations (SPEC §3.1, §12.3). */
export function ProjectNav({ registry, project }: { registry: Registry; project: Project | null }) {
  const loc = useLocation();
  if (!project) return null;
  const entries = registry.projectNav.flatMap((n) => {
    const to = n.to(project.id, project);
    return to === null ? [] : [{ n, id: n.id, to, route: n.route }];
  });
  const current = currentProjectNav(entries, loc.pathname, matchRoute(registry.compiled, loc.pathname)?.route ?? null);
  return (
    <nav aria-label="Project">
      <ul className="shell-nav">
        {entries.map(({ n, to }) => {
          const reason = n.disabledReason?.(project) ?? null;
          const emph = n.id === "lab";
          return (
            <li key={n.id}>
              {reason ? (
                <Tooltip content={reason}>
                  <span className="nav-disabled" aria-disabled="true" tabIndex={0}>
                    {n.label}
                  </span>
                </Tooltip>
              ) : (
                <Link to={to} aria-current={n.id === current ? "page" : undefined} className={emph ? "emph" : undefined}>
                  {n.label}
                </Link>
              )}
            </li>
          );
        })}
      </ul>
    </nav>
  );
}

function ActiveRunChip({ projectRuns, run, activeRunId }: { projectRuns: RunSummary[]; run: RunSummary | null; activeRunId: string | null }) {
  const current = run ?? projectRuns.find((r) => r.id === activeRunId) ?? null;
  if (!current && !projectRuns.length) return null;
  return (
    <span className="row hide-narrow" style={{ gap: 6 }}>
      {current ? (
        <>
          <Badge tone="accent">{modeLabel(current.mode, current.coarse_m)}</Badge>
          {current.demo ? <Badge tone="demo">DEMO</Badge> : null}
          <span className="cap num" title="Held-out R²">
            R² {fmtNum(current.r2, 3)}
          </span>
          <span className="cap">{fmtDate(current.created_utc)}</span>
        </>
      ) : null}
      {projectRuns.length ? (
        <select aria-label="Switch run" value={current?.id ?? ""} onChange={(e) => e.target.value && navigate(`/r/${e.target.value}`)} style={{ maxWidth: "14em" }}>
          {current ? null : <option value="">Open a run…</option>}
          {projectRuns.map((r) => (
            <option key={r.id} value={r.id}>
              {r.label || r.id} · {r.status}
            </option>
          ))}
        </select>
      ) : null}
    </span>
  );
}

function Breadcrumb({ project, run, tabTitle: tab, pageTitle }: { project: Project | null; run: RunSummary | null; tabTitle: string | null; pageTitle: string | null }) {
  const items: { label: string; to?: string }[] = [];
  if (project) items.push({ label: project.name, to: `/p/${project.id}` });
  if (run) items.push({ label: run.label || run.id, to: `/r/${run.id}` });
  if (tab) items.push({ label: tab });
  else if (pageTitle && !(run && pageTitle === "Overview")) items.push({ label: pageTitle });
  return (
    <nav className="breadcrumb" aria-label="Breadcrumb">
      <ol>
        {items.map((it, i) => (
          <li key={i} aria-current={i === items.length - 1 ? "page" : undefined}>
            {it.to && i < items.length - 1 ? <Link to={it.to}>{it.label}</Link> : it.label}
          </li>
        ))}
      </ol>
    </nav>
  );
}

function useBaseCommands(registry: Registry, project: Project | null, runId: string | null): Command[] {
  const jobs = useJobs((s) => s.jobs);
  const paletteOpen = useUi((s) => s.paletteOpen);
  const setTheme = useUi((s) => s.setTheme);
  const { data: projects } = useProjects();
  const { data: runs } = useRecentRuns(paletteOpen);
  return useMemo(() => {
    const cmds: Command[] = [
      { id: "nav:home", label: "Home", group: "Go to", run: () => navigate("/") },
      { id: "nav:projects", label: "Projects", group: "Go to", run: () => navigate("/projects") },
      { id: "nav:runs", label: "Run history", group: "Go to", run: () => navigate("/runs") },
      { id: "nav:jobs", label: "Activity (jobs)", group: "Go to", keywords: "queue", run: () => navigate("/jobs") },
      { id: "nav:findings", label: "Findings", group: "Go to", run: () => navigate("/findings") },
      { id: "nav:settings", label: "Settings", group: "Go to", run: () => navigate("/settings") },
      ...(["system", "light", "dark"] as ThemePref[]).map((t) => ({ id: `theme:${t}`, label: `Theme: ${t}`, group: "Action", keywords: "dark light mode", run: () => setTheme(t) })),
    ];
    if (project)
      for (const n of registry.projectNav) {
        const to = n.to(project.id, project);
        if (to && !n.disabledReason?.(project)) cmds.push({ id: `pnav:${n.id}`, label: n.label, group: project.name, run: () => navigate(to) });
      }
    if (runId) for (const t of registry.runTabs) cmds.push({ id: `tab:${t.id}`, label: t.label, group: "This run", keywords: t.group, run: () => navigate(runTabHref(t, runId)) });
    for (const j of activeJobs(jobs)) cmds.push({ id: `job:${j.id}`, label: j.label || j.kind, group: "Job", hint: j.status, keywords: j.kind, run: () => navigate(`/jobs/${j.id}`) });
    for (const p of projects ?? []) cmds.push({ id: `project:${p.id}`, label: p.name, group: "Project", run: () => navigate(`/p/${p.id}`) });
    for (const r of runs?.items ?? []) cmds.push({ id: `run:${r.id}`, label: r.label || r.id, group: "Run", hint: `${r.mode} · ${r.status}`, keywords: r.id, run: () => navigate(`/r/${r.id}`) });
    return cmds;
  }, [registry, project, runId, jobs, projects, runs, setTheme]);
}

export function AppShell({ children, pageTitle, fullWidth }: { children: ReactNode; pageTitle: string | null; fullWidth?: boolean }) {
  const registry = useRegistry();
  const route = useRoute();
  const context = useUi((s) => s.context);
  const sidebarOpen = useUi((s) => s.sidebarOpen);
  const setSidebarOpen = useUi((s) => s.setSidebarOpen);
  const setPaletteOpen = useUi((s) => s.setPaletteOpen);
  const liveMessage = useUi((s) => s.liveMessage);
  const jobs = useJobs((s) => s.jobs);
  const progress = useJobs((s) => s.progress);
  const { data: projectDetail } = useProject(context.projectId);
  const { data: runDetail } = useRunDetail(context.runId);
  const project = projectDetail?.project ?? null;
  const run = context.runId ? runDetail?.run ?? null : null;
  const commands = useBaseCommands(registry, project, context.runId);

  useEffect(() => connectJobs(), []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && (e.key === "k" || e.key === "K")) {
        e.preventDefault();
        setPaletteOpen(!useUi.getState().paletteOpen);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [setPaletteOpen]);

  // Close the mobile sidebar on navigation.
  useEffect(() => setSidebarOpen(false), [route.pathname, setSidebarOpen]);

  // Tab title mirrors the most important running job (SPEC §3.1).
  const title = tabTitle(jobs, progress, pageTitle);
  useEffect(() => {
    document.title = title;
  }, [title]);

  const tab = context.runId && route.route?.runTab ? route.route.runTab.label : null;

  return (
    <div className="shell">
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <aside className="shell-sidebar" data-open={sidebarOpen} aria-label="Sidebar">
        <Link to="/" className="shell-brand">
          <span className="logo" aria-hidden="true">
            S
          </span>
          SPARC Studio
        </Link>
        <ProjectSwitcher projectId={context.projectId} />
        <ProjectNav registry={registry} project={project} />
        <nav className="shell-sidebar-bottom" aria-label="Workspace">
          <ul className="shell-nav">
            <li>
              <Link to="/jobs" activeMatch="prefix">
                <Icon name="activity" /> Activity
                {activeJobs(jobs).length ? <span className="muted num">({activeJobs(jobs).length})</span> : null}
              </Link>
            </li>
            <li>
              <Link to="/settings" activeMatch="prefix">
                <Icon name="settings" /> Settings
              </Link>
            </li>
          </ul>
        </nav>
      </aside>
      {sidebarOpen ? <div className="shell-backdrop" aria-hidden="true" onClick={() => setSidebarOpen(false)} /> : null}
      <div className="shell-main">
        <header className="shell-topbar">
          <IconButton icon="menu" label="Open navigation" className="shell-menu-btn" onClick={() => setSidebarOpen(!sidebarOpen)} aria-expanded={sidebarOpen} />
          <Breadcrumb project={project} run={run} tabTitle={tab} pageTitle={pageTitle} />
          <span className="spacer" />
          {project ? <ActiveRunChip projectRuns={projectDetail?.runs ?? []} run={run} activeRunId={project.active_run_id} /> : null}
          <JobTray />
          <EngineDot runId={context.runId} />
          <ConnectionPill />
          <IconButton icon="search" label="Command palette (Ctrl+K)" onClick={() => setPaletteOpen(true)} />
          <ThemeToggle />
        </header>
        <main id="main" className={fullWidth ? "shell-content full" : "shell-content"} tabIndex={-1}>
          {children}
        </main>
      </div>
      <Toasts />
      <CommandPalette base={commands} />
      <div className="sr-only" aria-live="polite" role="status">
        {liveMessage}
      </div>
    </div>
  );
}
