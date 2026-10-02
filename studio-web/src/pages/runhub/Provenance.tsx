// Provenance (SPEC §6.4): copyable hashes (input, joins, config, code), git commit and dirty
// flag, platform; environment.txt with a diff against another run of the project; the
// effective config with its diff vs the project config and vs DEFAULTS; launch.json; and
// "Reproduce this run" (POST /api/runs/{rid}/studies/reproduce).
import { useState } from "react";
import { errorMessage } from "../../api/client";
import { reproduceRun, useProjectRuns, useRunConfig, useRunDetailFull, useRunEnvironment } from "../../api/runs";
import { Badge } from "../../components/ui/Badge";
import { Button } from "../../components/ui/Button";
import { CodeArea } from "../../components/ui/CodeArea";
import { EmptyState } from "../../components/ui/EmptyState";
import { Icon } from "../../components/ui/Icon";
import { Table } from "../../components/ui/Table";
import { copyText } from "../../components/ui/download";
import { useJobs } from "../../stores/jobs";
import { toast } from "../../stores/ui";
import { fmtDateTime } from "../../theme/format";
import { Block, Section, ViewPage, useRid } from "./common";
import { humanize } from "./format";

const show = (v: unknown) => (v === undefined ? "—" : typeof v === "string" ? v : JSON.stringify(v));

function CopyValue({ label, value }: { label: string; value: string | null }) {
  return (
    <div className="row" style={{ flexWrap: "nowrap", minWidth: 0 }}>
      <span className="mono cap" style={{ overflowWrap: "anywhere" }}>
        {value ?? "—"}
      </span>
      {value ? (
        <button
          type="button"
          className="btn small ghost"
          aria-label={`Copy ${label}`}
          onClick={async () => {
            const ok = await copyText(value);
            toast(ok ? "success" : "warning", ok ? `${label} copied` : "Copy failed: select the text instead");
          }}
        >
          <Icon name="copy" />
        </button>
      ) : null}
    </div>
  );
}

function ReproduceButton({ rid }: { rid: string }) {
  const [busy, setBusy] = useState(false);
  return (
    <Button
      variant="primary"
      busy={busy}
      icon="refresh"
      onClick={async () => {
        setBusy(true);
        try {
          const r = await reproduceRun(rid);
          useJobs.getState().upsert(r.job);
          toast("info", "Reproduction started: a child run re-fits S0–S3 and checks the results", { href: `/jobs/${r.job.id}`, linkLabel: "Track" });
        } catch (e) {
          toast("error", "Could not start the reproduction", { body: errorMessage(e) });
        } finally {
          setBusy(false);
        }
      }}
    >
      Reproduce this run
    </Button>
  );
}

function Environment({ rid, pid }: { rid: string; pid: string | null }) {
  const runs = useProjectRuns(pid);
  const [other, setOther] = useState<string>("");
  const env = useRunEnvironment(rid, other || null);
  const choices = (runs.data?.items ?? []).filter((r) => r.id !== rid);
  return (
    <Block
      title="Environment"
      actions={
        <label className="row cap">
          Diff with
          <select value={other} onChange={(e) => setOther(e.target.value)} aria-label="Run to diff the environment with">
            <option value="">no other run</option>
            {choices.map((r) => (
              <option key={r.id} value={r.id}>
                {r.label || r.id} · {fmtDateTime(r.created_utc)}
              </option>
            ))}
          </select>
        </label>
      }
    >
      {env.error ? <EmptyState error={env.error} /> : null}
      {env.data?.diff && other ? (
        <div className="stack" style={{ gap: 6 }} aria-label="Environment diff">
          {!env.data.diff.added.length && !env.data.diff.removed.length && !env.data.diff.changed.length ? <p className="cap">Identical package lists.</p> : null}
          {env.data.diff.changed.length ? (
            <Table
              caption="Changed packages"
              csvName="environment-changed"
              rowKey={(r) => r.name}
              columns={[
                { key: "name", label: "Package", value: (r) => r.name },
                { key: "a", label: "This run", value: (r) => r.a },
                { key: "b", label: "Other run", value: (r) => r.b },
              ]}
              rows={env.data.diff.changed}
            />
          ) : null}
          {env.data.diff.added.length ? (
            <p className="cap">
              <strong>Only in the other run:</strong> {env.data.diff.added.join(", ")}
            </p>
          ) : null}
          {env.data.diff.removed.length ? (
            <p className="cap">
              <strong>Only in this run:</strong> {env.data.diff.removed.join(", ")}
            </p>
          ) : null}
        </div>
      ) : null}
      {env.data ? (
        <details>
          <summary>{env.data.packages.length} packages in environment.txt</summary>
          <pre className="mono" style={{ maxHeight: 320, overflow: "auto", fontSize: "0.8rem" }}>
            {env.data.packages.join("\n")}
          </pre>
        </details>
      ) : !env.error ? (
        <p className="cap">Loading the environment…</p>
      ) : null}
    </Block>
  );
}

function Config({ rid }: { rid: string }) {
  const cfg = useRunConfig(rid);
  if (cfg.error) return <EmptyState error={cfg.error} />;
  if (!cfg.data) return <p className="cap">Loading the configuration…</p>;
  const c = cfg.data;
  return (
    <Block
      title="Effective configuration"
      actions={
        <Badge title={`Config directory: ${c.config_dir}`}>from {c.source === "launch" ? "launch.json" : c.source === "manifest" ? "the manifest" : "the import config"}</Badge>
      }
    >
      <CodeArea label="Run configuration (YAML)" value={c.yaml} readOnly rows={18} />
      <h4>Differs from the current project config</h4>
      {c.vs_project_diff.length ? (
        <Table
          caption="Run vs project config"
          csvName="config-vs-project"
          rowKey={(r) => r.path}
          columns={[
            { key: "path", label: "Setting", value: (r) => r.path, render: (r) => <span className="mono">{r.path}</span> },
            { key: "run", label: "This run", value: (r) => show(r.run) },
            { key: "project", label: "Project now", value: (r) => show(r.project) },
          ]}
          rows={c.vs_project_diff}
        />
      ) : (
        <p className="cap">Same as the project config.</p>
      )}
      <h4>Differs from the defaults</h4>
      {c.vs_defaults.length ? (
        <Table
          caption="Run vs defaults"
          csvName="config-vs-defaults"
          rowKey={(r) => r.path}
          columns={[
            { key: "path", label: "Setting", value: (r) => r.path, render: (r) => <span className="mono">{r.path}</span> },
            { key: "value", label: "This run", value: (r) => show(r.value) },
            { key: "default", label: "Default", value: (r) => show(r.default) },
          ]}
          rows={c.vs_defaults}
        />
      ) : (
        <p className="cap">Every setting is at its default.</p>
      )}
    </Block>
  );
}

export default function Provenance() {
  const rid = useRid();
  const detail = useRunDetailFull(rid);
  return (
    <ViewPage
      view="provenance"
      title="Provenance"
      intro="Everything needed to trust and repeat this run: hashes of the inputs, config and code, the software environment and the exact launch snapshot."
      actions={<ReproduceButton rid={rid} />}
    >
      {(s) => (
        <>
          <div className="grid2">
            <Section title="Hashes" data={s.hashes}>
              {(h) => (
                <Block title="Hashes">
                  <dl className="kv" aria-label="Hashes">
                    {Object.entries(h).map(([k, v]) => (
                      <div key={k} style={{ display: "contents" }}>
                        <dt>{humanize(k.replace(/_sha256$/, ""))}</dt>
                        <dd>
                          <CopyValue label={`${humanize(k)} hash`} value={v} />
                        </dd>
                      </div>
                    ))}
                  </dl>
                </Block>
              )}
            </Section>
            <div className="stack">
              <Section title="Git" data={s.git}>
                {(g) => (
                  <Block title="Code">
                    <dl className="kv" aria-label="Git">
                      <dt>Commit</dt>
                      <dd>
                        <CopyValue label="Commit" value={g.commit} />
                        {g.dirty ? (
                          <Badge tone="warn" title="Uncommitted changes were present when the run started">
                            dirty
                          </Badge>
                        ) : g.dirty === false ? (
                          <Badge>clean</Badge>
                        ) : null}
                      </dd>
                      {g.branch ? (
                        <>
                          <dt>Branch</dt>
                          <dd className="mono">{g.branch}</dd>
                        </>
                      ) : null}
                    </dl>
                  </Block>
                )}
              </Section>
              <Section title="Platform" data={s.platform}>
                {(p) => (
                  <Block title="Platform">
                    <dl className="kv" aria-label="Platform">
                      {Object.entries(p).map(([k, v]) => (
                        <div key={k} style={{ display: "contents" }}>
                          <dt>{humanize(k)}</dt>
                          <dd className="mono">{v ?? "—"}</dd>
                        </div>
                      ))}
                    </dl>
                  </Block>
                )}
              </Section>
            </div>
          </div>
          <Environment rid={rid} pid={detail.data?.run.project_id ?? null} />
          <Config rid={rid} />
          <Section title="Launch snapshot" data={s.launch}>
            {(l) => (
              <Block title="launch.json">
                <p className="cap">Resume always rebuilds the config from this snapshot, so later project edits never change what an interrupted run resumes with.</p>
                <pre className="mono" style={{ maxHeight: 360, overflow: "auto", fontSize: "0.8rem" }}>
                  {JSON.stringify(l, null, 2)}
                </pre>
              </Block>
            )}
          </Section>
        </>
      )}
    </ViewPage>
  );
}
