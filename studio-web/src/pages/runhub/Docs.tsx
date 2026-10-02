// Documents (SPEC §6.4 Docs): report, methods, model card, uncertainty, placebo, multiverse,
// simcheck and benchmark markdown rendered with `marked` (raw HTML escaped by the foundation
// Markdown). report.md is labelled "as of run end" (it is frozen). "Regenerate methods & model
// card" starts the writeup job (POST /api/runs/{rid}/actions/writeup).
import { useState } from "react";
import { errorMessage } from "../../api/client";
import type { DocEntry, DocId } from "../../api/runs";
import { regenerateWriteup, useRunDoc, useRunDocs } from "../../api/runs";
import { Badge } from "../../components/ui/Badge";
import { Button } from "../../components/ui/Button";
import { EmptyState } from "../../components/ui/EmptyState";
import { Markdown } from "../../components/ui/Markdown";
import { Link, useRoute } from "../../router";
import { useJobs } from "../../stores/jobs";
import { toast } from "../../stores/ui";
import { fmtDateTime } from "../../theme/format";
import { LiveBanner, useRid, useRunIsLive } from "./common";

const PRODUCER: Record<DocId, string> = {
  report: "the run itself (written at run end)",
  methods: "the writeup action",
  model_card: "the writeup action",
  uncertainty: "the uncertainty report (attach studies on the Uncertainty tab)",
  placebo: "the placebo study",
  multiverse: "the multiverse study",
  simcheck: "the simulation check study",
  benchmark: "the benchmark study",
};

export function RegenerateButton({ rid, label = "Regenerate methods & model card" }: { rid: string; label?: string }) {
  const [busy, setBusy] = useState(false);
  return (
    <Button
      busy={busy}
      icon="refresh"
      onClick={async () => {
        setBusy(true);
        try {
          const job = await regenerateWriteup(rid);
          useJobs.getState().upsert(job);
          toast("info", `${job.label || "Writeup"} started`, { href: `/jobs/${job.id}`, linkLabel: "Track" });
        } catch (e) {
          toast("error", "Could not regenerate the documents", { body: errorMessage(e) });
        } finally {
          setBusy(false);
        }
      }}
    >
      {label}
    </Button>
  );
}

function DocBody({ rid, doc }: { rid: string; doc: DocEntry }) {
  const content = useRunDoc(rid, doc.present ? doc.id : null);
  if (!doc.present) {
    return (
      <EmptyState title={`${doc.title}: not in this run yet`} body={`${doc.file} is produced by ${PRODUCER[doc.id] ?? "a post-run action"}.`}>
        {doc.regenerable ? <RegenerateButton rid={rid} label="Generate it now" /> : null}
      </EmptyState>
    );
  }
  if (content.error) return <EmptyState error={content.error} />;
  if (!content.data) return <p className="cap">Loading {doc.title}…</p>;
  // report.md cannot be re-rendered from the manifest (SPEC §6.4 Docs): always "as of run end".
  const frozen = content.data.frozen || doc.frozen || doc.id === "report";
  return (
    <article className="card" aria-label={doc.title}>
      <header>
        <div>
          <p className="eyebrow">{doc.file}</p>
          <h3>{doc.title}</h3>
        </div>
        <div className="row">
          {frozen ? (
            <Badge tone="warn" title="Written when the run finished; later post-run actions do not re-render it">
              as of run end
            </Badge>
          ) : null}
          <span className="cap">updated {fmtDateTime(content.data.mtime || doc.mtime)}</span>
        </div>
      </header>
      <Markdown source={content.data.markdown} />
    </article>
  );
}

export default function Docs() {
  const rid = useRid();
  const { params } = useRoute();
  const docs = useRunDocs(rid);
  const runLive = useRunIsLive(rid);
  const list = docs.data ?? [];
  const chosen = list.find((d) => d.id === params.doc) ?? list.find((d) => d.present) ?? list[0] ?? null;
  return (
    <section className="stack" aria-labelledby="docs-title">
      <header className="row" style={{ justifyContent: "space-between" }}>
        <h2 id="docs-title" style={{ margin: 0 }}>
          Documents
        </h2>
        <RegenerateButton rid={rid} />
      </header>
      {runLive ? <LiveBanner>This run is still running: documents appear as the run and its post-run actions write them.</LiveBanner> : null}
      {docs.error ? <EmptyState error={docs.error} /> : null}
      {!docs.data && !docs.error ? <p className="cap">Loading documents…</p> : null}
      {docs.data ? (
        <div className="row" style={{ alignItems: "flex-start", flexWrap: "wrap" }}>
          <nav aria-label="Documents" style={{ flex: "0 0 220px" }}>
            <ul className="stack" style={{ listStyle: "none", padding: 0, margin: 0, gap: 4 }}>
              {list.map((d) => (
                <li key={d.id}>
                  <Link to={`/r/${encodeURIComponent(rid)}/docs/${d.id}`} aria-current={chosen?.id === d.id ? "page" : undefined} className={d.present ? undefined : "muted"}>
                    {d.title}
                  </Link>
                  {!d.present ? <span className="cap"> · not yet</span> : d.frozen || d.id === "report" ? <span className="cap"> · as of run end</span> : null}
                </li>
              ))}
            </ul>
          </nav>
          <div style={{ flex: "1 1 480px", minWidth: 0 }}>{chosen ? <DocBody rid={rid} doc={chosen} /> : <p className="cap">This run has no documents.</p>}</div>
        </div>
      ) : null}
    </section>
  );
}
