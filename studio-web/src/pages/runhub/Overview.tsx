// Run overview (SPEC §6.4 Overview): header (name, mode chips, created, commit + dirty flag,
// versions, n cells, grid, run dir, DEMO), the compact tracker while a job runs, KPI row,
// stage timing bar, data-health flags, outputs availability grid, studies chips, caveats and
// model-card limitations, and the run's findings.
import { useMemo } from "react";
import type { OverviewSections, RunDetail } from "../../api/runs";
import { RUN_TAB_IDS } from "../../api/types";
import { useRunDetailFull } from "../../api/runs";
import { Bars } from "../../charts";
import { Badge, modeLabel } from "../../components/ui/Badge";
import { ActionButton } from "../../components/ui/EmptyState";
import { Icon } from "../../components/ui/Icon";
import { JobStrip } from "../../components/ui/JobStrip";
import { StatusChip } from "../../components/ui/StatusChip";
import { copyText } from "../../components/ui/download";
import { Link } from "../../router";
import { toast } from "../../stores/ui";
import { fmtDateTime, fmtDuration, fmtInt, fmtNum } from "../../theme/format";
import { Block, FlagList, KpiTiles, Section, ViewPage, useRid } from "./common";
import { OUTPUT_STATE_TEXT, outputStateStatus, producedByText } from "./format";

function Header({ detail }: { detail: RunDetail }) {
  const h = detail.header;
  const run = detail.run;
  const versions = Object.entries(h.versions ?? {});
  return (
    <Block
      title={h.name || run.label || run.id}
      actions={
        <>
          <Badge tone="accent">{modeLabel(run.mode, run.coarse_m)}</Badge>
          {h.demo ? (
            <Badge tone="demo" title="Synthetic demo data at a fictional location">
              DEMO
            </Badge>
          ) : null}
          <StatusChip status={run.status} />
        </>
      }
    >
      <dl className="kv" aria-label="Run header">
        <dt>Created</dt>
        <dd>{fmtDateTime(h.created_utc)}</dd>
        <dt>Origin</dt>
        <dd>{run.origin.replace(/_/g, " ")}</dd>
        <dt>Commit</dt>
        <dd className="mono">
          {h.git_commit ?? "—"}
          {h.git_dirty ? (
            <>
              {" "}
              <Badge tone="warn" title="The working tree had uncommitted changes when this run started">
                dirty
              </Badge>
            </>
          ) : null}
        </dd>
        <dt>Cells</dt>
        <dd>{fmtInt(h.n_points)}</dd>
        <dt>Grid</dt>
        <dd>
          {h.grid_shape ? `${fmtInt(h.grid_shape[0])} × ${fmtInt(h.grid_shape[1])} cells` : "—"}
          {h.cell_m ? ` at ${fmtNum(h.cell_m, 0)} m` : ""}
          {h.coarse_m ? ` (coarse ${fmtNum(h.coarse_m, 0)} m)` : ""}
        </dd>
        {run.duration_s !== null ? (
          <>
            <dt>Duration</dt>
            <dd>{fmtDuration(run.duration_s)}</dd>
          </>
        ) : null}
        <dt>Run directory</dt>
        <dd className="mono" style={{ overflowWrap: "anywhere" }}>
          {h.run_dir}{" "}
          <button
            type="button"
            className="btn small ghost"
            aria-label="Copy run directory"
            onClick={async () => {
              const ok = await copyText(h.run_dir);
              toast(ok ? "success" : "warning", ok ? "Run directory copied" : "Copy failed: select the path instead");
            }}
          >
            <Icon name="copy" />
          </button>
        </dd>
        {versions.length ? (
          <>
            <dt>Versions</dt>
            <dd className="mono cap">{versions.map(([k, v]) => `${k} ${v}`).join(" · ")}</dd>
          </>
        ) : null}
        {detail.checkpoint ? (
          <>
            <dt>Checkpoint</dt>
            <dd>
              {detail.checkpoint.present ? `saved, done: ${detail.checkpoint.done.join(", ") || "—"}` : (detail.checkpoint.reason ?? "none")}{" "}
              <Link to={`/r/${encodeURIComponent(run.id)}/files#checkpoint`}>details</Link>
            </dd>
          </>
        ) : null}
      </dl>
    </Block>
  );
}

type Timing = NonNullable<OverviewSections["timings"]>[number];

/** Stage timing bar: one stacked bar of stage durations; skipped stages listed greyed. */
function StageTimingBar({ timings }: { timings: Timing[] }) {
  const ran = timings.filter((t) => t.seconds !== null && t.seconds > 0);
  const total = ran.reduce((a, t) => a + (t.seconds ?? 0), 0);
  const slowest = ran.reduce<Timing | null>((m, t) => (!m || (t.seconds ?? 0) > (m.seconds ?? 0) ? t : m), null);
  const caption = ran.length
    ? `${fmtDuration(total)} in ${ran.length} stage${ran.length === 1 ? "" : "s"}${slowest ? `; ${slowest.label} took ${fmtNum(((slowest.seconds ?? 0) / total) * 100, 0)}%` : ""}.`
    : "No stage has finished yet.";
  const skipped = timings.filter((t) => t.seconds === null || t.state === "skipped" || t.state === "not_run");
  return (
    <div className="stack" style={{ gap: 6 }}>
      <Bars
        title="Stage timings"
        units="seconds"
        categories={["This run"]}
        orientation="h"
        mode="stacked"
        valueLabel="Time"
        unit="s"
        decimals={1}
        height={90}
        series={timings.map((t) => ({ id: t.stage, label: t.label, values: [t.seconds], muted: t.state === "skipped" || t.state === "cached" || t.state === "not_run" }))}
        caption={caption}
      />
      {skipped.length ? (
        <p className="cap">
          Not run:{" "}
          {skipped.map((t, i) => (
            <span key={t.stage}>
              {i ? ", " : ""}
              {t.label}
              {t.reason ? ` (${t.reason.replace(/_/g, " ")})` : ""}
            </span>
          ))}
        </p>
      ) : null}
    </div>
  );
}

type OutputCell = NonNullable<OverviewSections["outputs_grid"]>[number];

/** Where an output is shown: its run tab, or Files for viewer kinds (table, json, …). */
function outputHref(rid: string, view: string): string {
  const base = `/r/${encodeURIComponent(rid)}`;
  if (view === "overview") return base;
  return (RUN_TAB_IDS as readonly string[]).includes(view) ? `${base}/${view}` : `${base}/files`;
}

function OutputsGrid({ outputs, rid }: { outputs: OutputCell[]; rid: string }) {
  const groups = useMemo(() => {
    const m = new Map<string, OutputCell[]>();
    for (const o of outputs) m.set(o.group, [...(m.get(o.group) ?? []), o]);
    return [...m.entries()];
  }, [outputs]);
  if (!outputs.length) return <p className="cap">No catalogued outputs.</p>;
  return (
    <div className="grid3" role="list" aria-label="Outputs availability">
      {groups.map(([g, list]) => (
        <div key={g} role="listitem" className="stack" style={{ gap: 6 }}>
          <p className="eyebrow">{g}</p>
          <ul style={{ listStyle: "none", padding: 0, margin: 0 }} className="stack">
            {list.map((o) => (
              <li key={o.id} className="row" style={{ justifyContent: "space-between", gap: 6 }} data-output={o.id} data-state={o.state}>
                <span title={`Produced by ${producedByText(o.produced_by)}`}>
                  <Link to={outputHref(rid, o.view)}>{o.label}</Link>
                </span>
                <span className="row" style={{ gap: 4 }}>
                  <StatusChip status={outputStateStatus(o.state)} text={OUTPUT_STATE_TEXT[o.state] ?? o.state} />
                  {o.state === "missing" && o.action ? <ActionButton action={o.action} size="small" variant="default" /> : null}
                </span>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}

/** A finding's stored `url_state` when it is a same-origin app path ("/r/…?…"), else null. */
export function internalPath(url: string | null | undefined): string | null {
  return url && url.startsWith("/") && !url.startsWith("//") && !url.startsWith("/\\") ? url : null;
}

const STUDY_STATUS: Record<string, string> = { not_run: "not_run", queued: "queued", running: "running", done: "done", stale: "stale", failed: "failed" };

export default function Overview() {
  const rid = useRid();
  const detail = useRunDetailFull(rid);
  return (
    <div className="stack">
      <JobStrip runId={rid} />
      {detail.data ? <Header detail={detail.data} /> : null}
      <ViewPage view="overview" title="Overview" caveatsOpen>
        {(s) => (
          <>
            <Section title="Key numbers" data={s.kpis}>
              {(kpis) => <KpiTiles kpis={kpis} label="Run key numbers" />}
            </Section>
            <div className="grid2">
              <Section title="Stage timings" data={s.timings}>
                {(t) => <StageTimingBar timings={t} />}
              </Section>
              <Section title="Data health" data={s.flags}>
                {(flags) => (
                  <Block title="Data health">
                    <FlagList flags={flags} empty="No data-health flags." />
                  </Block>
                )}
              </Section>
            </div>
            <Section title="Outputs" data={s.outputs_grid}>
              {(o) => (
                <Block title="Outputs">
                  <OutputsGrid outputs={o} rid={rid} />
                </Block>
              )}
            </Section>
            <Section title="Studies" data={s.studies}>
              {(studies) => (
                <Block title="Studies" actions={<Link to={`/r/${encodeURIComponent(rid)}/validation`}>Validation</Link>}>
                  {studies.length ? (
                    <ul className="row" style={{ listStyle: "none", padding: 0, margin: 0 }} aria-label="Study status">
                      {studies.map((st) => (
                        <li key={st.kind} className="chip" title={st.headline ?? undefined}>
                          <strong>{st.kind}</strong> <StatusChip status={STUDY_STATUS[st.state] ?? st.state} />
                          {st.headline ? <span className="cap">{st.headline}</span> : null}
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="cap">No studies yet.</p>
                  )}
                </Block>
              )}
            </Section>
            <Section title="Model-card limitations" data={s.limitations}>
              {(lim) => (
                <Block title="Model-card limitations">
                  {lim.length ? (
                    <ul className="prose" style={{ margin: 0, paddingLeft: 18 }}>
                      {lim.map((l, i) => (
                        <li key={i}>{l}</li>
                      ))}
                    </ul>
                  ) : (
                    <p className="cap">The model card lists no limitations.</p>
                  )}
                </Block>
              )}
            </Section>
            <Section title="Findings" data={s.findings}>
              {(f) => (
                <Block title="Findings from this run">
                  {f.length ? (
                    <ul style={{ margin: 0, paddingLeft: 18 }}>
                      {f.map((x) => (
                        <li key={x.id}>
                          <Link to={internalPath(x.url_state) ?? `/r/${encodeURIComponent(rid)}`}>{x.title}</Link> <span className="cap">{fmtDateTime(x.created_utc)}</span>
                          {x.note_md ? <p className="cap">{x.note_md}</p> : null}
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="cap">Nothing pinned yet: use Pin on any chart to collect findings.</p>
                  )}
                </Block>
              )}
            </Section>
          </>
        )}
      </ViewPage>
    </div>
  );
}
