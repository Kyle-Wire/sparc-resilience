// Config editor (`/p/:pid/config`, SPEC §9.4, J4): the project's config.yml in a CodeArea with
// line numbers and an issue gutter (`/config/validate` issues placed by their dotted path),
// the unsaved diff, the diff vs DEFAULTS, the version history, and an impact preview before
// every save ("2 existing runs would need a refit"). Saves send `If-Match: <version>`; a 409
// opens the conflict dialog. Comments are not preserved, and the page says so.
import { useEffect, useMemo, useState } from "react";
import { ApiError, errorMessage } from "../../api/client";
import {
  configAtVersion,
  configImpact,
  conflictVersion,
  putConfig,
  useConfigHistory,
  useProjectConfig,
  validateConfig,
  type ImpactResult,
} from "../../api/projects";
import { invalidate } from "../../api/resource";
import type { Issue } from "../../api/types";
import { Badge } from "../../components/ui/Badge";
import { Button } from "../../components/ui/Button";
import { CodeArea, type LineIssue } from "../../components/ui/CodeArea";
import { Dialog } from "../../components/ui/Dialog";
import { Diff } from "../../components/ui/Diff";
import { EmptyState } from "../../components/ui/EmptyState";
import { Icon } from "../../components/ui/Icon";
import { Tabs } from "../../components/ui/Tabs";
import { Link, useRoute } from "../../router";
import { toast } from "../../stores/ui";
import { fmtDateTime } from "../../theme/format";
import { ConflictDialog } from "./components/ConflictDialog";
import { stageShort } from "./model/plan";
import { locatePath, yamlPathIndex } from "./model/yamlPaths";
import "./projects.css";

type YamlError = { line: number; column: number | null; message: string };

/** Validation issues → gutter marks (an issue whose path is not in the file marks line 1). */
export function lineIssues(yaml: string, issues: readonly Issue[], yamlError: YamlError | null): LineIssue[] {
  const index = yamlPathIndex(yaml);
  const out: LineIssue[] = issues.map((i) => ({ line: locatePath(index, i.path) ?? 1, level: i.level, message: `${i.path ? i.path + ": " : ""}${i.message}` }));
  if (yamlError) out.push({ line: yamlError.line, level: "error", message: yamlError.message });
  return out;
}

function yamlErrorOf(e: unknown): YamlError | null {
  if (!(e instanceof ApiError) || e.code !== "yaml_error") return null;
  const line = typeof e.detail?.line === "number" ? (e.detail.line as number) : 1;
  const column = typeof e.detail?.column === "number" ? (e.detail.column as number) : null;
  return { line, column, message: `YAML: ${e.message}` };
}

function Impact({ impact }: { impact: ImpactResult }) {
  return (
    <div className="stack impact" style={{ gap: 8 }} aria-label="Impact of this change">
      <p>
        {impact.changed_sections.length ? (
          <>
            Sections changed:{" "}
            {impact.changed_sections.map((s) => (
              <Badge key={s}>{s}</Badge>
            ))}
          </>
        ) : (
          "No section that affects a run changed."
        )}
      </p>
      {impact.runs.length === 0 ? (
        <p className="cap">No existing run is affected.</p>
      ) : (
        <>
          <p>
            <b>
              {impact.runs.filter((r) => r.refit_from).length} existing {impact.runs.filter((r) => r.refit_from).length === 1 ? "run" : "runs"}
            </b>{" "}
            would need a refit to reflect this change. Existing runs are not changed: resume always uses each run's launch snapshot.
          </p>
          <table className="tbl" aria-label="Runs affected">
            <thead>
              <tr>
                <th scope="col">Run</th>
                <th scope="col">Sections changed</th>
                <th scope="col">A re-run would refit from</th>
              </tr>
            </thead>
            <tbody>
              {impact.runs.map((r) => (
                <tr key={r.run_id}>
                  <td>
                    <Link to={`/r/${encodeURIComponent(r.run_id)}`}>{r.label || r.run_id}</Link>
                  </td>
                  <td className="cap">{r.changed_sections.join(", ") || "none"}</td>
                  <td>{r.phrase || (r.refit_from ? stageShort(r.refit_from) : "no refit")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}

type Tab = "issues" | "changes" | "defaults" | "history";

export default function ConfigEditor() {
  const { params } = useRoute();
  const pid = params.pid ?? "";
  const cfg = useProjectConfig(pid || null);
  const history = useConfigHistory(pid || null);
  const [base, setBase] = useState<{ version: number; yaml: string } | null>(null);
  const [text, setText] = useState<string>("");
  const [newer, setNewer] = useState<number | null>(null);
  const [issues, setIssues] = useState<Issue[]>([]);
  const [yamlError, setYamlError] = useState<YamlError | null>(null);
  const [validating, setValidating] = useState(false);
  const [tab, setTab] = useState<Tab>("issues");
  const [review, setReview] = useState<{ impact: ImpactResult | null; error: unknown } | null>(null);
  const [note, setNote] = useState("");
  const [saving, setSaving] = useState(false);
  const [conflict, setConflict] = useState<{ server: number | null } | null>(null);
  const [picked, setPicked] = useState<{ version: number; yaml: string } | null>(null);

  const dirty = !!base && text !== base.yaml;

  // Follow the server while the text is unedited; flag a newer version otherwise.
  useEffect(() => {
    const doc = cfg.data;
    if (!doc) return;
    if (!base || text === base.yaml) {
      setBase({ version: doc.version, yaml: doc.yaml });
      setText(doc.yaml);
      setNewer(null);
    } else if (doc.version !== base.version) setNewer(doc.version);
  }, [cfg.data]);

  // Validate the text as it changes.
  useEffect(() => {
    if (!base) return;
    let live = true;
    const ctrl = new AbortController();
    const t = window.setTimeout(async () => {
      setValidating(true);
      try {
        const r = await validateConfig(pid, { yaml: text }, ctrl.signal);
        if (!live) return;
        setIssues(r.issues);
        setYamlError(null);
      } catch (e) {
        if (!live) return;
        const ye = yamlErrorOf(e);
        if (ye) setYamlError(ye);
      } finally {
        if (live) setValidating(false);
      }
    }, 450);
    return () => {
      live = false;
      ctrl.abort();
      window.clearTimeout(t);
    };
  }, [pid, text, base]);

  const gutter = useMemo(() => lineIssues(text, issues, yamlError).sort((a, b) => a.line - b.line), [text, issues, yamlError]);

  const openReview = async () => {
    setReview({ impact: null, error: null });
    try {
      setReview({ impact: await configImpact(pid, { yaml: text }), error: null });
    } catch (e) {
      setReview({ impact: null, error: e });
    }
  };

  const save = async (version: number) => {
    setSaving(true);
    try {
      const r = await putConfig(pid, version, { yaml: text, ...(note.trim() ? { note: note.trim() } : {}) });
      setBase({ version: r.version, yaml: text });
      setNewer(null);
      setIssues(r.issues);
      setReview(null);
      setConflict(null);
      setNote("");
      invalidate(`project:${pid}`);
      const errors = r.issues.filter((i) => i.level === "error").length;
      toast(errors ? "warning" : "success", `Saved config version ${r.version}`, { body: errors ? `${errors} validation ${errors === 1 ? "error" : "errors"} remain.` : undefined });
    } catch (e) {
      const ye = yamlErrorOf(e);
      if (e instanceof ApiError && e.code === "conflict") {
        setReview(null);
        setConflict({ server: conflictVersion(e.detail) });
      } else if (ye) {
        setReview(null);
        setYamlError(ye);
        toast("error", `The YAML does not parse (line ${ye.line})`, { body: e instanceof Error ? e.message : undefined });
      } else toast("error", "Could not save the config", { body: errorMessage(e) });
    } finally {
      setSaving(false);
    }
  };

  const loadSaved = async () => {
    setConflict(null);
    setBase(null);
    await cfg.reload();
  };

  if (cfg.error && !cfg.data) return <EmptyState error={cfg.error} title="Could not load the config" />;
  const doc = cfg.data;
  const errors = issues.filter((i) => i.level === "error").length + (yamlError ? 1 : 0);
  const warns = issues.filter((i) => i.level === "warn").length;

  return (
    <div className="stack config-editor">
      <header className="page-head">
        <p className="eyebrow">Config</p>
        <div className="row">
          <h1>config.yml</h1>
          {base ? <Badge title="Saved version this text is based on">v{base.version}</Badge> : null}
          {dirty ? <Badge tone="warn">unsaved</Badge> : null}
          <span className="spacer" />
          <span className="cap" role="status" aria-live="polite">
            {validating ? "Validating…" : `${errors} ${errors === 1 ? "error" : "errors"} · ${warns} ${warns === 1 ? "warning" : "warnings"}`}
          </span>
          <Link className="btn small ghost" to={`/p/${encodeURIComponent(pid)}/setup/data`}>
            Back to setup
          </Link>
        </div>
      </header>
      <div className="callout comments-notice" data-tone="info">
        <Icon name="info" /> Comments are not preserved: saving rewrites config.yml from the parsed values. Every save is kept in the History tab.
      </div>
      {newer !== null ? (
        <div className="callout" role="status">
          Version {newer} was saved while you edited (yours is based on {base?.version}).{" "}
          <Button size="small" variant="ghost" onClick={() => void loadSaved()}>
            Load version {newer} (discard mine)
          </Button>
        </div>
      ) : null}
      {!doc || !base ? (
        <p className="cap" role="status">
          <span className="spinner" aria-hidden="true" /> Loading the config…
        </p>
      ) : (
        <div className="config-grid">
          <div className="stack" style={{ gap: 8 }}>
            <CodeArea value={text} onChange={setText} label="config.yml" issues={gutter} rows={30} />
            <div className="row">
              <Button variant="primary" icon="check" disabled={!dirty} onClick={() => void openReview()}>
                Review and save
              </Button>
              <Button variant="ghost" disabled={!dirty} onClick={() => setText(base.yaml)}>
                Revert
              </Button>
            </div>
          </div>
          <div>
            <Tabs
              label="Config panels"
              value={tab}
              onChange={setTab}
              items={[
                { id: "issues", label: `Issues (${gutter.length})` },
                { id: "changes", label: "Unsaved changes" },
                { id: "defaults", label: "vs DEFAULTS" },
                { id: "history", label: "History" },
              ]}
            >
              {tab === "issues" ? (
                gutter.length ? (
                  <ul className="issue-list">
                    {gutter.map((i, k) => (
                      <li key={k} data-level={i.level}>
                        <span className="mono num">L{i.line}</span> <span>{i.message}</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="cap">No issues.</p>
                )
              ) : tab === "changes" ? (
                <Diff a={base.yaml} b={text} label="Unsaved changes" />
              ) : tab === "defaults" ? (
                <table className="tbl" aria-label="Changed from DEFAULTS">
                  <thead>
                    <tr>
                      <th scope="col">Key</th>
                      <th scope="col">Value</th>
                      <th scope="col">Default</th>
                    </tr>
                  </thead>
                  <tbody>
                    {doc.changed_from_defaults.map((r) => (
                      <tr key={r.path}>
                        <td className="mono">{r.path}</td>
                        <td className="mono">{JSON.stringify(r.value)}</td>
                        <td className="mono muted">{JSON.stringify(r.default)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <div className="stack" style={{ gap: 8 }}>
                  <ul className="history-list">
                    {(history.data ?? []).map((h) => (
                      <li key={h.version}>
                        <button
                          type="button"
                          className="linklike"
                          aria-pressed={picked?.version === h.version}
                          onClick={async () => {
                            try {
                              setPicked(await configAtVersion(pid, h.version));
                            } catch (e) {
                              toast("error", `Could not load version ${h.version}`, { body: errorMessage(e) });
                            }
                          }}
                        >
                          v{h.version}
                        </button>{" "}
                        <span className="cap">
                          {fmtDateTime(h.saved_utc)}
                          {h.note ? ` · ${h.note}` : ""}
                        </span>
                      </li>
                    ))}
                  </ul>
                  {picked ? (
                    <>
                      <div className="row">
                        <span className="cap">
                          Version {picked.version} → the editor text
                        </span>
                        <span className="spacer" />
                        <Button size="small" onClick={() => setText(picked.yaml)}>
                          Load v{picked.version} into the editor
                        </Button>
                      </div>
                      <Diff a={picked.yaml} b={text} label={`Version ${picked.version} vs the editor`} />
                    </>
                  ) : null}
                </div>
              )}
            </Tabs>
          </div>
        </div>
      )}
      <Dialog
        open={!!review}
        onClose={() => setReview(null)}
        busy={saving}
        wide
        title="Review the change before saving"
        footer={
          <>
            <Button onClick={() => setReview(null)} disabled={saving}>
              Keep editing
            </Button>
            <Button variant="primary" busy={saving} onClick={() => base && void save(base.version)} data-autofocus>
              Save version {base ? base.version + 1 : ""}
            </Button>
          </>
        }
      >
        {review?.error ? (
          <div className="callout" data-tone="crit" role="alert">
            Impact preview unavailable: {errorMessage(review.error)}
          </div>
        ) : review?.impact ? (
          <Impact impact={review.impact} />
        ) : (
          <p className="cap" role="status">
            <span className="spinner" aria-hidden="true" /> Comparing with the existing runs' checkpoints…
          </p>
        )}
        {base ? <Diff a={base.yaml} b={text} label="Changes to save" /> : null}
        <label className="field">
          <span className="field-label">Note (kept in the history)</span>
          <input value={note} onChange={(e) => setNote(e.target.value)} placeholder="e.g. tune_lambda back to [0, 0.1, 1]" aria-label="Save note" />
        </label>
        {errors ? <p className="cap">The config still has {errors} validation {errors === 1 ? "error" : "errors"}; it can be saved, but launches are refused until they are fixed.</p> : null}
      </Dialog>
      <ConflictDialog
        open={!!conflict}
        mine={base?.version ?? 0}
        server={conflict?.server ?? null}
        busy={saving}
        onClose={() => setConflict(null)}
        onReload={() => void loadSaved()}
        onOverwrite={() => void save(conflict?.server ?? newer ?? (base ? base.version + 1 : 0))}
      />
    </div>
  );
}
