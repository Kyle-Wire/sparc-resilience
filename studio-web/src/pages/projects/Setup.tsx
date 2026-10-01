// Setup wizard (`/p/:pid/setup/:step`, SPEC §9.3): non-linear tabs data · levers · physics ·
// inputs · scenarios · analysis · about, each with a completion dot from the readiness spine.
// Steps edit one shared draft of the raw config (setup/store.ts) that is validated as it
// changes (`POST /config/validate {raw}`, issues shown at each field's dotted path) and saved
// atomically (`PUT /config {raw}` with If-Match; a 409 opens the conflict dialog).
import { useEffect, useState, type ReactNode } from "react";
import { ApiError, errorMessage } from "../../api/client";
import { conflictVersion, getConfig, putConfig, useProjectConfig, validateConfig } from "../../api/projects";
import { invalidate, mutate } from "../../api/resource";
import { Badge, DemoBadge } from "../../components/ui/Badge";
import { Button } from "../../components/ui/Button";
import { EmptyState } from "../../components/ui/EmptyState";
import { Tabs } from "../../components/ui/Tabs";
import { useProject } from "../../layouts/resources";
import { Link, navigate, useRoute } from "../../router";
import { toast } from "../../stores/ui";
import { CfgProvider } from "./components/cfg";
import { ConflictDialog } from "./components/ConflictDialog";
import { DOT_TEXT, isStepId, STEPS, stepDot, type DotState, type StepId } from "./model/steps";
import { AboutStep } from "./setup/AboutStep";
import { AnalysisStep } from "./setup/AnalysisStep";
import { DataStep } from "./setup/DataStep";
import { InputsStep } from "./setup/InputsStep";
import { LeversStep } from "./setup/LeversStep";
import { PhysicsStep } from "./setup/PhysicsStep";
import { ScenariosStep } from "./setup/ScenariosStep";
import { dirtySections, rawKey, useDraft, useDrafts } from "./setup/store";
import "./projects.css";

const STEP_VIEW: Record<StepId, () => ReactNode> = {
  data: () => <DataStep />,
  levers: () => <LeversStep />,
  physics: () => <PhysicsStep />,
  inputs: () => <InputsStep />,
  scenarios: () => <ScenariosStep />,
  analysis: () => <AnalysisStep />,
  about: () => <AboutStep />,
};

/** The step of the current URL (`/p/:pid/setup/:step`, or the `/p/:pid/setup/inputs` nav route). */
export function currentStep(params: Record<string, string>, pathname: string): string {
  if (params.step) return params.step;
  const last = pathname.replace(/\/+$/, "").split("/").pop() ?? "";
  return last === "setup" ? "data" : last;
}

// Moving between steps changes the URL and remounts the page; keyboard focus on the tab list
// is carried over so arrow-key navigation keeps working.
let focusTabAfterMount = false;

function Dot({ state }: { state: DotState }) {
  return <span className="step-dot" data-state={state} aria-label={DOT_TEXT[state]} role="img" />;
}

/** Validate the draft as it changes (debounced); issues land in the draft store. */
function useDraftValidation(pid: string) {
  const draft = useDraft(pid);
  const setIssues = useDrafts((s) => s.setIssues);
  const [validating, setValidating] = useState(false);
  const key = draft ? rawKey(draft.raw) : null;
  const done = !!draft && draft.issuesKey === key;
  useEffect(() => {
    if (!key || done) return;
    let live = true;
    const ctrl = new AbortController();
    const t = window.setTimeout(async () => {
      setValidating(true);
      try {
        const r = await validateConfig(pid, { raw: JSON.parse(key) as Record<string, unknown> }, ctrl.signal);
        if (live) setIssues(pid, r.issues, key);
      } catch {
        /* keep the previous issues; the save response re-validates */
      } finally {
        if (live) setValidating(false);
      }
    }, 450);
    return () => {
      live = false;
      ctrl.abort();
      window.clearTimeout(t);
    };
  }, [pid, key, done, setIssues]);
  return validating;
}

function SaveBar({ pid }: { pid: string }) {
  const draft = useDraft(pid)!;
  const st = useDrafts.getState;
  const dirty = dirtySections(draft);
  const [saving, setSaving] = useState(false);
  const [conflict, setConflict] = useState<{ server: number | null } | null>(null);

  useEffect(() => {
    if (!dirty.length) return;
    const onUnload = (e: BeforeUnloadEvent) => {
      e.preventDefault();
      e.returnValue = "";
    };
    window.addEventListener("beforeunload", onUnload);
    return () => window.removeEventListener("beforeunload", onUnload);
  }, [dirty.length]);

  const save = async (version: number) => {
    const d = st().drafts[pid];
    if (!d) return;
    setSaving(true);
    try {
      const r = await putConfig(pid, version, { raw: d.raw, note: `setup: ${dirtySections(d).join(", ")}` });
      st().markSaved(pid, r.version);
      st().setIssues(pid, r.issues, rawKey(d.raw));
      setConflict(null);
      invalidate(`project:${pid}`);
      const errors = r.issues.filter((i) => i.level === "error").length;
      toast(errors ? "warning" : "success", `Saved config version ${r.version}`, { body: errors ? `${errors} validation ${errors === 1 ? "error" : "errors"} remain.` : undefined });
    } catch (e) {
      if (e instanceof ApiError && e.code === "conflict") setConflict({ server: conflictVersion(e.detail) });
      else toast("error", "Could not save the config", { body: errorMessage(e) });
    } finally {
      setSaving(false);
    }
  };

  const reload = async () => {
    setSaving(true);
    try {
      const doc = await getConfig(pid);
      mutate(`project:${pid}:config`, doc);
      st().forget(pid);
      st().adopt(pid, doc);
      setConflict(null);
      toast("info", `Loaded config version ${doc.version}`);
    } catch (e) {
      toast("error", "Could not load the saved config", { body: errorMessage(e) });
    } finally {
      setSaving(false);
    }
  };

  return (
    <>
      {draft.serverVersion > draft.baseVersion && dirty.length ? (
        <div className="callout" data-tone="info" role="status">
          A newer config version ({draft.serverVersion}) was saved while you edited (yours is based on {draft.baseVersion}). Saving will ask before overwriting it.{" "}
          <Button size="small" variant="ghost" onClick={() => void reload()}>
            Load version {draft.serverVersion}
          </Button>
        </div>
      ) : null}
      {dirty.length ? (
        <div className="savebar" role="region" aria-label="Unsaved changes">
          <span>
            <b>Unsaved changes</b> in {dirty.join(", ")}
          </span>
          <span className="spacer" />
          <Button variant="ghost" onClick={() => st().discard(pid)} disabled={saving}>
            Discard
          </Button>
          <Button variant="primary" icon="check" busy={saving} onClick={() => void save(draft.baseVersion)}>
            Save config
          </Button>
        </div>
      ) : null}
      <ConflictDialog
        open={!!conflict}
        mine={draft.baseVersion}
        server={conflict?.server ?? null}
        busy={saving}
        onClose={() => setConflict(null)}
        onReload={() => void reload()}
        onOverwrite={() => void save(conflict?.server ?? draft.serverVersion)}
      />
    </>
  );
}

export default function Setup() {
  const { params, pathname } = useRoute();
  const pid = params.pid ?? "";
  const step = currentStep(params, pathname);
  const project = useProject(pid || null);
  const cfg = useProjectConfig(pid || null);
  const adopt = useDrafts((s) => s.adopt);
  const draft = useDraft(pid);
  const validating = useDraftValidation(pid);

  useEffect(() => {
    if (cfg.data) adopt(pid, cfg.data);
  }, [cfg.data, pid, adopt]);

  useEffect(() => {
    if (pid && !isStepId(step)) navigate(`/p/${encodeURIComponent(pid)}/setup/data`, { replace: true });
  }, [pid, step]);

  const hasDraft = !!draft;
  useEffect(() => {
    if (!hasDraft || !focusTabAfterMount) return;
    focusTabAfterMount = false;
    document.querySelector<HTMLElement>('.setup-page [role="tab"][aria-selected="true"]')?.focus();
  }, [hasDraft]);

  if (!isStepId(step)) return null;
  if (cfg.error && !draft) return <EmptyState error={cfg.error} title="Could not load the project config" />;
  const p = project.data?.project;
  const readiness = project.data?.readiness;
  const issues = draft?.issues ?? [];
  const errors = issues.filter((i) => i.level === "error").length;
  const warns = issues.filter((i) => i.level === "warn").length;

  return (
    <div className="stack setup-page">
      <header className="page-head">
        <p className="eyebrow">Setup</p>
        <div className="row">
          <h1>{p ? `Set up ${p.name}` : "Set up"}</h1>
          {p?.demo ? <DemoBadge /> : null}
          <span className="spacer" />
          <span className="cap" role="status" aria-live="polite">
            {validating ? "Validating…" : draft ? (errors || warns ? `${errors} ${errors === 1 ? "error" : "errors"} · ${warns} ${warns === 1 ? "warning" : "warnings"}` : "Config valid") : ""}
          </span>
          {draft ? <Badge title="Config version the draft is based on">v{draft.baseVersion}</Badge> : null}
          <Link className="btn small ghost" to={`/p/${encodeURIComponent(pid)}/config`}>
            Edit the YAML
          </Link>
          <Link className="btn small" to={`/p/${encodeURIComponent(pid)}/launch`}>
            Launch
          </Link>
        </div>
      </header>
      {!draft ? (
        <p className="cap" role="status">
          <span className="spinner" aria-hidden="true" /> Loading the config…
        </p>
      ) : (
        <CfgProvider value={{ pid, draft, issues }}>
          <Tabs
            label="Setup steps"
            value={step}
            onChange={(id) => {
              focusTabAfterMount = document.activeElement?.getAttribute("role") === "tab";
              navigate(`/p/${encodeURIComponent(pid)}/setup/${id}`);
            }}
            items={STEPS.map((s) => {
              const dot = stepDot(s.id, readiness, draft.raw, issues);
              return {
                id: s.id,
                label: (
                  <span className="step-tab" data-step-dot={dot}>
                    <Dot state={dot} />
                    {s.label}
                  </span>
                ),
              };
            })}
          >
            {STEP_VIEW[step]()}
          </Tabs>
          <SaveBar pid={pid} />
        </CfgProvider>
      )}
    </div>
  );
}
