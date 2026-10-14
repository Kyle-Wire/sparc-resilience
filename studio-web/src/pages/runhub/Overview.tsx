// Run overview (SPEC §6.4 Overview): header (name, mode chips, created, commit + dirty flag,
// versions, n cells, grid, run dir, DEMO), the compact tracker while a job runs, KPI row,
// stage timing bar, data-health flags, outputs availability grid, studies chips, caveats and
// model-card limitations, and the run's findings. No tab routes the full page any more (the
// Summary tab is the run's landing page); its run record (header, stage timings, outputs grid)
// is RunRecord, shown in Model details › Run log (compact there: `detailsOnly`).
import { useMemo } from "react";
import { useResource } from "../../api/resource";
import type { OverviewSections, RunDetail } from "../../api/runs";
import { useRunDetailFull, useView } from "../../api/runs";
import { listJobs } from "../../api/tracking";
import { Bars } from "../../charts";
import { Badge, modeLabel } from "../../components/ui/Badge";
import { ActionButton } from "../../components/ui/EmptyState";
import { Icon } from "../../components/ui/Icon";
import { JobStrip } from "../../components/ui/JobStrip";
import { StatusChip } from "../../components/ui/StatusChip";
import { copyText } from "../../components/ui/download";
import { Link, runHref, runViewHref } from "../../router";
import { toast } from "../../stores/ui";
import { fmtDateTime, fmtDuration, fmtInt, fmtNum } from "../../theme/format";
import { stageTone } from "../../theme/palette";
import { Block, FlagList, KpiTiles, Section, ViewPage, useRid, type DetailPageProps } from "./common";
import { OUTPUT_STATE_TEXT, outputStateStatus, producedByText } from "./format";

/**
 * The run header card. `compact` (Model details › Run log) leaves out the commit, the run
 * directory and the versions: their home is Model details › Provenance.
 */
function Header({ detail, compact = false }: { detail: RunDetail; compact?: boolean }) {
  const h = detail.header;
  const run = detail.run;
  const versions = Object.entries(h.versions ?? {});
  return (
    <Block
      title={h.name || run.label || run.id}
      actions={
        <>
          <Badge tone="mode">{modeLabel(run.mode, run.coarse_m)}</Badge>
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
        {compact ? null : (
          <>
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
          </>
        )}
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
        {compact ? null : (
          <>
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
          </>
        )}
        {versions.length && !compact ? (
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
              <Link to={`${runHref(run.id, "details", "files")}#checkpoint`}>details</Link>
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
        series={timings.map((t, i) => {
          const muted = t.state === "skipped" || t.state === "cached" || t.state === "not_run";
          // Each stage keeps its stage-number badge colour: the brand ramp sampled once across all stages (no repeats).
          return { id: t.stage, label: t.label, values: [t.seconds], muted, color: muted ? undefined : stageTone(i, timings.length) };
        })}
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

/** Where an output is shown: the page of its view (VIEW_HREF), or Files for viewer kinds (table, json, …). */
function outputHref(rid: string, view: string): string {
  return runViewHref(rid, view) ?? runHref(rid, "details", "files");
}

/** The outputs availability grid; `actions` (default on) adds each missing output's remedy next to its MISSING pill. */
function OutputsGrid({ outputs, rid, actions = true }: { outputs: OutputCell[]; rid: string; actions?: boolean }) {
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
              <li key={o.id} className="output-row" data-output={o.id} data-state={o.state}>
                <span className="output-label" title={`Produced by ${producedByText(o.produced_by)}`}>
                  <Link to={outputHref(rid, o.view)}>{o.label}</Link>
                </span>
                <StatusChip status={outputStateStatus(o.state)} text={OUTPUT_STATE_TEXT[o.state] ?? o.state} />
                {actions && o.state === "missing" && o.action ? (
                  <span className="output-action">
                    <ActionButton action={o.action} size="small" variant="default" />
                  </span>
                ) : null}
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

/**
 * The run record (Model details › Run log, above the job log): the header (created, origin,
 * commit, cells, grid, duration, run directory, versions), the stage timing bar and the outputs
 * availability grid (state pills only: each remedy is offered where its result is read, e.g. the
 * skill-vs-distance re-run on How much to trust this).
 *
 * `detailsOnly` (how Model details › Run log renders it) keeps each fact in one place: no outputs
 * grid (Files is its home), no commit, versions or run directory (Provenance is theirs), and the
 * stage timing bar only when the run has no tracked job, because the job log below (RunTrack)
 * already shows the stage times on its rail and timeline. Imported runs have no job log, so that
 * is where their stage timings live.
 */
export function RunRecord({ detailsOnly = false }: DetailPageProps = {}) {
  const rid = useRid();
  const detail = useRunDetailFull(rid);
  const overview = useView(rid, "overview");
  const jobs = useResource(detailsOnly && rid ? `run:${rid}:jobs` : null, (sig) => listJobs({ run: rid, limit: 100 }, sig), { tags: rid ? [`run:${rid}`, "jobs"] : [] });
  const s = overview.data && overview.data.availability !== "missing" ? overview.data.sections : null;
  // Compact: the bar only once the run is known to have no job log (or the list cannot be read).
  const showTimings = !detailsOnly || (jobs.data ? !jobs.data.items.length : !!jobs.error);
  return (
    <div className="stack" data-run-record="" data-compact={detailsOnly || undefined}>
      {detail.data ? <Header detail={detail.data} compact={detailsOnly} /> : null}
      {s ? (
        <>
          {showTimings ? (
            <Section title="Stage timings" data={s.timings}>
              {(t) => <StageTimingBar timings={t} />}
            </Section>
          ) : null}
          {detailsOnly ? null : (
            <Section title="Outputs" data={s.outputs_grid}>
              {(o) => (
                <Block title="Outputs">
                  <OutputsGrid outputs={o} rid={rid} actions={false} />
                </Block>
              )}
            </Section>
          )}
        </>
      ) : null}
    </div>
  );
}

const STUDY_STATUS: Record<string, string> = { not_run: "not_run", queued: "queued", running: "running", done: "done", stale: "stale", failed: "failed" };

export default function Overview() {
  const rid = useRid();
  const detail = useRunDetailFull(rid);
  return (
    <div className="stack">
      <JobStrip runId={rid} />
      {detail.data ? <Header detail={detail.data} /> : null}
      <ViewPage view="overview" title="Overview" showCaveats caveatsOpen>
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
                <Block title="Studies" actions={<Link to={`${runHref(rid, "trust")}#validation`}>Validation</Link>}>
                  {studies.length ? (
                    <ul className="row" style={{ listStyle: "none", padding: 0, margin: 0 }} aria-label="Study status">
                      {studies.map((st) => (
                        <li key={st.kind} className="chip study-chip" title={st.headline ?? undefined}>
                          <strong>{st.kind}</strong> <StatusChip status={STUDY_STATUS[st.state] ?? st.state} />
                          {st.headline ? <span className="chip-note">{st.headline}</span> : null}
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
