// The shell on every page (SPEC §3.1): sidebar (project switcher, project nav declared by
// route modules with its collapsed Advanced group, Settings), top bar (breadcrumb, active-run
// chip, job tray, engine dot, connection pill, command palette, theme toggle), toasts, the
// aria-live job announcer and the browser tab title mirroring the most important running job.
// Brand styling (ramp strip, gradient mark, textured sidebar) lives in ./shell.css.
import { useEffect, useMemo, useState, type ReactNode } from "react";
import type { Project, RunSummary } from "../api/types";
import { CommandPalette } from "../components/ui/CommandPalette";
import { ConnectionPill } from "../components/ui/ConnectionPill";
import { Icon, type IconName } from "../components/ui/Icon";
import { IconButton } from "../components/ui/IconButton";
import { JobTray } from "../components/ui/JobTray";
import { Toasts } from "../components/ui/Toast";
import { TermTipBody } from "../components/ui/InfoTip";
import { Tooltip, useTooltip } from "../components/ui/Tooltip";
import { modeLabel } from "../components/ui/Badge";
import { term } from "../content/glossary";
import { TRUST_GRADE_LABEL, trustGrade as gradeOf, type TrustGrade } from "../content/trust";
import { Link, matchRoute, navigate, runHref, runTabHref, useLocation, useRegistry, useRoute, type ProjectNavDef, type Registry, type RouteDef } from "../router";
import { activeJobs, connectJobs, tabTitle, useJobs } from "../stores/jobs";
import { useUi, type Command, type ThemePref } from "../stores/ui";
import { fmtDateTime, fmtNum, fmtPct } from "../theme/format";
import { useProject, useProjects, useRecentRuns, useRunDetail } from "./resources";
import { DemoTag, RefLink } from "./tips";
import "./shell.css";

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

/** The engine chip's state word, after "Full model" ("Full model ready", "Full model loading 42%"). */
const ENGINE_WORD: Record<string, string> = {
  absent: "starts when needed",
  starting: "starting",
  cold: "not loaded",
  queued: "waiting",
  loading: "loading",
  ready: "ready",
  busy: "busy",
  recycling: "restarting",
  no_checkpoint: "not available",
  incompatible: "not available",
  error: "error",
};

/** What each engine state means, after the full model's glossary tip (the chip's tooltip). */
const ENGINE_TIP: Record<string, string> = {
  absent: "It is not running now; it starts by itself when a page needs it, such as the Scenario Lab.",
  starting: "It is starting up.",
  cold: "It is not loaded yet; it loads when you run a scenario.",
  queued: "It is waiting for its turn to load.",
  loading: "It is loading this run's saved models.",
  ready: "It is loaded and ready to work out scenarios.",
  busy: "It is working on a scenario; others wait their turn.",
  recycling: "It is restarting to free memory.",
  no_checkpoint: "This run saved no models, so only quick estimates are possible.",
  incompatible: "This run's saved models do not fit this version of SPARC, so only quick estimates are possible.",
  error: "It could not load. Try again from the Scenario Lab.",
};

/**
 * The top bar's full-model chip. `noModels`: the current run saved no trained models, so the chip says
 * "Full model not available", as the Scenario Lab and the Library say it, whatever the engine host is doing.
 */
export function EngineDot({ runId, noModels = false }: { runId: string | null; noModels?: boolean }) {
  const host = useJobs((s) => s.host);
  const perRun = useJobs((s) => (runId ? s.engineByRun[runId] : undefined));
  const info = perRun ?? host;
  const state = noModels ? "no_checkpoint" : (info.state ?? "absent");
  const word = ENGINE_WORD[state] ?? String(state).replace(/_/g, " ");
  const text = state === "loading" && info.progress !== null ? `${word} ${fmtPct(info.progress)}` : word;
  const shown = state === "absent" || state === "no_checkpoint" ? "cold" : state;
  const name = term("exact_engine").label;
  const mem = info.rss_mb ? ` Using ${fmtNum(info.rss_mb / 1024, 1)} GB of memory.` : "";
  const t = useTooltip<HTMLSpanElement>({
    content: <TermTipBody tip={`${term("exact_engine").tip} ${ENGINE_TIP[state] ?? ""}${mem}`.trim()} technical={`Scenario engine: ${String(state).replace(/_/g, " ")}`} />,
    side: "bottom",
    bubbleClassName: "tip-term",
  });
  return (
    <>
      <span ref={t.anchorRef} className="engine-dot hide-narrow" data-state={shown} role="status" aria-label={`${name}: ${text}`} tabIndex={0} {...t.triggerProps}>
        <i aria-hidden="true" />
        {name} {text}
      </span>
      {t.bubble}
    </>
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
 * The project nav entry the current page belongs to: the entry its route declared or claims
 * (`claims`: Launch highlights Set up), else, on a run page, Scenario Lab for `/r/:rid/lab…`
 * and Runs for every other `/r/:rid/…`, else the entry whose target is the longest prefix of
 * the path. The project root (Overview) prefixes every project page, so it is current only on
 * its own route.
 */
export function currentProjectNav(entries: { id: string; to: string; route: RouteDef; claims?: string[] }[], pathname: string, route: RouteDef | null): string | null {
  const own = entries.find((e) => e.route === route) ?? (route ? entries.find((e) => e.claims?.includes(route.path)) : undefined);
  if (own) return own.id;
  const run = /^\/r\/[^/]+(\/[^/]+)?/.exec(pathname);
  if (run) {
    const want = run[1] === "/lab" ? "lab" : "runs";
    if (entries.some((e) => e.id === want)) return want;
  }
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

/** localStorage key of the sidebar's Advanced group open state ("1" open). */
export const ADVANCED_NAV_KEY = "sparc.nav.advanced";

function readAdvancedOpen(): boolean {
  try {
    return window.localStorage.getItem(ADVANCED_NAV_KEY) === "1";
  } catch {
    return false;
  }
}

function writeAdvancedOpen(open: boolean): void {
  try {
    window.localStorage.setItem(ADVANCED_NAV_KEY, open ? "1" : "0");
  } catch {
    /* storage blocked: the group just starts collapsed next time */
  }
}

type NavItem = { n: ProjectNavDef; id: string; to: string; route: RouteDef; claims?: string[] };

function NavEntry({ item, project, current, count }: { item: NavItem; project: Project; current: string | null; count?: number }) {
  const { n, to } = item;
  const reason = n.disabledReason?.(project) ?? null;
  return (
    <li>
      {reason ? (
        <Tooltip content={reason}>
          <span className="nav-disabled" aria-disabled="true" tabIndex={0}>
            {n.label}
          </span>
        </Tooltip>
      ) : (
        <Link to={to} aria-current={n.id === current ? "page" : undefined} className={n.id === "lab" ? "emph" : undefined}>
          {n.label}
          {count ? <span className="muted num">({count})</span> : null}
        </Link>
      )}
    </li>
  );
}

/**
 * The project nav built from route modules' `projectNav` declarations (SPEC §3.1, §12.3): the
 * main entries, then the "Advanced" group (`group: "advanced"`), collapsed by default; its open
 * state persists in localStorage and it opens by itself while one of its pages is current.
 */
export function ProjectNav({ registry, project }: { registry: Registry; project: Project | null }) {
  const loc = useLocation();
  const jobs = useJobs((s) => s.jobs);
  const entries: NavItem[] = project
    ? registry.projectNav.flatMap((n) => {
        const to = n.to(project.id, project);
        return to === null ? [] : [{ n, id: n.id, to, route: n.route, claims: n.claims }];
      })
    : [];
  const current = currentProjectNav(entries, loc.pathname, matchRoute(registry.compiled, loc.pathname)?.route ?? null);
  const main = entries.filter((e) => e.n.group !== "advanced");
  const advanced = entries.filter((e) => e.n.group === "advanced");
  const onAdvanced = advanced.some((e) => e.id === current);
  // Opening on one of its pages is not remembered; a click is.
  const [open, setOpen] = useState(() => onAdvanced || readAdvancedOpen());
  useEffect(() => {
    if (onAdvanced) setOpen(true);
  }, [onAdvanced]);
  if (!project) return null;
  const running = activeJobs(jobs).length;
  return (
    <nav aria-label="Project">
      <ul className="shell-nav">
        {main.map((e) => (
          <NavEntry key={e.id} item={e} project={project} current={current} />
        ))}
      </ul>
      {advanced.length ? (
        <div className="shell-nav-group">
          <button
            type="button"
            className="shell-nav-toggle"
            aria-expanded={open}
            aria-controls="shell-nav-advanced"
            onClick={() => {
              setOpen(!open);
              writeAdvancedOpen(!open);
            }}
          >
            <Icon name={open ? "chevronDown" : "chevronRight"} size={14} />
            Advanced
          </button>
          {open ? (
            <ul className="shell-nav" id="shell-nav-advanced">
              {advanced.map((e) => (
                <NavEntry key={e.id} item={e} project={project} current={current} count={e.id === "activity" ? running : undefined} />
              ))}
            </ul>
          ) : null}
        </div>
      ) : null}
    </nav>
  );
}

/**
 * Switcher option text per run id: label, mode, start time and status, so a fast and a coarse
 * run of one config differ. Runs that still read the same get their id appended. Accuracy
 * numbers are not here: they live on How much to trust this (and Run history's columns).
 */
export function runOptionLabels(runs: RunSummary[]): Map<string, string> {
  const base = runs.map((r) =>
    [r.label || r.id, modeLabel(r.mode, r.coarse_m), r.created_utc ? fmtDateTime(r.created_utc) : null, r.status].filter((x): x is string => !!x).join(" · "),
  );
  const seen = new Map<string, number>();
  for (const t of base) seen.set(t, (seen.get(t) ?? 0) + 1);
  return new Map(runs.map((r, i) => [r.id, (seen.get(base[i]) ?? 0) > 1 && r.label ? `${base[i]} · ${r.id}` : base[i]]));
}

/**
 * What the closed switcher shows for the current run: its name and start time, short enough to
 * be read whole in the top bar. When another run of the project would read the same, the full
 * option text (mode, status, id) is used instead, so the chip still tells them apart.
 */
export function runChipLabel(run: RunSummary, runs: RunSummary[], optionLabels: Map<string, string>): string {
  const short = (r: RunSummary) => [r.label || r.id, r.created_utc ? fmtDateTime(r.created_utc) : null].filter((x): x is string => !!x).join(" · ");
  const mine = short(run);
  return runs.some((r) => r.id !== run.id && short(r) === mine) ? (optionLabels.get(run.id) ?? mine) : mine;
}

export type { TrustGrade };

/**
 * The run's trust grade for the chip's dot, by the same rule as How much to trust this and the
 * Summary card (../content/trust.ts: held-out R², stepped down when the likely ranges miss their
 * 90% target), from the cheap accuracy numbers on RunSummary; null when R² is unknown.
 */
export function trustGrade(r2: number | null | undefined, coverage?: number | null): TrustGrade | null {
  return gradeOf({ r2, coverage })?.grade ?? null;
}

/**
 * The top bar's run chip: one bordered chip holding DEMO, the run switcher (its closed face reads
 * "name · start time", runChipLabel; the open list and the tooltip give mode and status too) and,
 * right next to it, the trust grade as a dot with its word ("Good"), linking to How much to trust
 * this, so it cannot be read as a connection light. No accuracy numbers (their home is the Trust
 * tab), and no second copy of the run's name.
 */
export function ActiveRunChip({ projectRuns, run, activeRunId }: { projectRuns: RunSummary[]; run: RunSummary | null; activeRunId: string | null }) {
  const current = run ?? projectRuns.find((r) => r.id === activeRunId) ?? null;
  const optionLabels = useMemo(() => runOptionLabels(projectRuns), [projectRuns]);
  const assessed = current ? gradeOf({ r2: current.r2, coverage: current.coverage }) : null;
  // Tooltips (hover, focus or tap) in place of native titles; each bubble exists only while open.
  const runTip = useTooltip<HTMLSelectElement>({ content: current ? `Open run: ${optionLabels.get(current.id) ?? current.id}` : "Open one of this project's runs", side: "bottom" });
  const trustTip = useTooltip<HTMLAnchorElement>({
    content: assessed ? <TermTipBody tip={`${term("trust_grade").label}: ${assessed.label}. ${assessed.line} Open How much to trust this for the details.`} technical={term("trust_grade").technical} /> : null,
    side: "bottom",
    bubbleClassName: "tip-term",
  });
  if (!current && !projectRuns.length) return null;
  const grade = assessed?.grade ?? null;
  const switchable = projectRuns.length > 0;
  return (
    <span className="run-chip hide-narrow" data-grade={grade ?? undefined}>
      {current?.demo ? <DemoTag /> : null}
      {switchable ? (
        <select
          ref={runTip.anchorRef}
          aria-label="Switch run"
          className="run-chip-select"
          value={current?.id ?? ""}
          {...runTip.triggerProps}
          onChange={(e) => e.target.value && navigate(runHref(e.target.value))}
        >
          {current ? null : <option value="">Open a run…</option>}
          {projectRuns.map((r) => (
            <option key={r.id} value={r.id}>
              {r.id === current?.id ? runChipLabel(r, projectRuns, optionLabels) : optionLabels.get(r.id)}
            </option>
          ))}
        </select>
      ) : null}
      {switchable ? runTip.bubble : null}
      {current && grade ? (
        <RefLink ref={trustTip.anchorRef} to={runHref(current.id, "trust")} className="run-chip-trust" data-grade={grade} {...trustTip.triggerProps}>
          <span className="trust-dot" data-grade={grade} aria-hidden="true" />
          <span className="sr-only">Trust: </span>
          {TRUST_GRADE_LABEL[grade]}
        </RefLink>
      ) : null}
      {current && grade ? trustTip.bubble : null}
    </span>
  );
}

function Breadcrumb({ project, run, tabTitle: tab, pageTitle }: { project: Project | null; run: RunSummary | null; tabTitle: string | null; pageTitle: string | null }) {
  const items: { label: string; to?: string }[] = [];
  if (project) items.push({ label: project.name, to: `/p/${project.id}` });
  if (run) items.push({ label: run.label || run.id, to: runHref(run.id) });
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
      { id: "nav:projects", label: "All projects", group: "Go to", keywords: "projects", run: () => navigate("/projects") },
      { id: "nav:runs", label: "Run history", group: "Go to", keywords: "runs", run: () => navigate("/runs") },
      { id: "nav:jobs", label: "Activity: running and queued work", group: "Go to", keywords: "jobs queue", run: () => navigate("/jobs") },
      { id: "nav:findings", label: "Findings: pinned charts and notes", group: "Go to", keywords: "notes pins", run: () => navigate("/findings") },
      { id: "nav:settings", label: "Settings", group: "Go to", run: () => navigate("/settings") },
      ...(["system", "light", "dark"] as ThemePref[]).map((t) => ({
        id: `theme:${t}`,
        label: t === "system" ? "Theme: match this computer" : `Theme: ${t}`,
        group: "Action",
        keywords: "dark light mode system",
        run: () => setTheme(t),
      })),
    ];
    if (project)
      for (const n of registry.projectNav) {
        const to = n.to(project.id, project);
        if (to && !n.disabledReason?.(project)) cmds.push({ id: `pnav:${n.id}`, label: n.label, group: project.name, run: () => navigate(to) });
      }
    if (runId) for (const t of registry.runTabs) cmds.push({ id: `tab:${t.id}`, label: t.label, group: "This run", keywords: t.views.join(" "), run: () => navigate(runTabHref(t, runId)) });
    for (const j of activeJobs(jobs)) cmds.push({ id: `job:${j.id}`, label: j.label || j.kind, group: "Running work", hint: j.status.replace(/_/g, " "), keywords: `${j.kind} job`, run: () => navigate(`/jobs/${j.id}`) });
    for (const p of projects ?? []) cmds.push({ id: `project:${p.id}`, label: p.name, group: "Project", run: () => navigate(`/p/${p.id}`) });
    for (const r of runs?.items ?? []) cmds.push({ id: `run:${r.id}`, label: r.label || r.id, group: "Run", hint: `${r.mode} run · ${r.status.replace(/_/g, " ")}`, keywords: `${r.id} ${r.mode}`, run: () => navigate(runHref(r.id)) });
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
      <div className="shell-ramp" aria-hidden="true" />
      <aside className="shell-sidebar" data-open={sidebarOpen} aria-label="Sidebar">
        <Link to="/" className="shell-brand">
          <span className="logo" aria-hidden="true" />
          <span className="shell-wordmark">
            <b>SPARC</b> <span>Studio</span>
          </span>
        </Link>
        <ProjectSwitcher projectId={context.projectId} />
        <ProjectNav registry={registry} project={project} />
        <nav className="shell-sidebar-bottom" aria-label="Workspace">
          <ul className="shell-nav">
            {project ? null : (
              // With a project open, Activity is in the project nav's Advanced group.
              <li>
                <Link to="/jobs" activeMatch="prefix">
                  <Icon name="activity" /> Activity
                  {activeJobs(jobs).length ? <span className="muted num">({activeJobs(jobs).length})</span> : null}
                </Link>
              </li>
            )}
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
          {/* the Lab's Design page says the engine state under its Run button (its own words) */}
          {route.route?.hideEngine ? null : <EngineDot runId={context.runId} noModels={!!context.runId && runDetail?.checkpoint?.present === false} />}
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
