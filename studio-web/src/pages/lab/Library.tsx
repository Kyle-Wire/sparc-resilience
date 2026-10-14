// Scenario library (`/r/:rid/lab/library`, SPEC §7.4, §7.8, §7.12), two groups, each titled
// with a kicker under the page's own title:
// - "Set up with this run" (first; phase 2, the canonical home of the run's configured
//   scenarios, formerly the Scenarios run tab): the scenarios view's sections from
//   ../runhub/Scenarios (citywide change per scenario, ladders, the table) with the Lab's row
//   actions (clone to edit, run the full model again for the uncertainty of a difference, pin to
//   compare, map).
//   `?configured=<slug>` focuses one of them (the old run tab's links redirect here).
// - "Project scenarios": every scenario of the project as a lineage tree (revisions under their
//   parents) with search, tag and status filters; batch full-model runs; per-scenario actions (open,
//   pin to compare, run the full model, fork, ladder, check across runs with its time and RAM
//   estimate, promote with the YAML diff, archive, delete); ladder charts (the change vs the amount
//   with the likely-range ribbon). Plain words (phase 4): the glossary's terms, with an ⓘ where a
//   term needs explaining; no native titles.
import { useEffect, useMemo, useState } from "react";
import { ApiError, errorMessage } from "../../api/client";
import { invalidate } from "../../api/resource";
import {
  forkScenario,
  getScenario,
  patchScenario,
  rerunConfigured,
  runBatch,
  runScenarioExact,
  useScenarios,
  type Scenario,
  type ScenarioStatus,
  type ScenarioSummary,
} from "../../api/lab";
import type { ScenarioRow } from "../../api/runs";
import type { Likely } from "../../api/types";
import { LineBand, SmallMultiples } from "../../charts";
import { Badge } from "../../components/ui/Badge";
import { Button } from "../../components/ui/Button";
import { EmptyState } from "../../components/ui/EmptyState";
import { InfoTip } from "../../components/ui/InfoTip";
import { StatusChip } from "../../components/ui/StatusChip";
import { Table } from "../../components/ui/Table";
import { term } from "../../content/glossary";
import { useRunDetail } from "../../layouts/resources";
import { useRunGrid } from "../../map/data";
import { Link, codecs, navigate, useRoute, useUrlState } from "../../router";
import { useJobs } from "../../stores/jobs";
import { toast, useUi } from "../../stores/ui";
import { fmtRelative, fmtSigned, unitLabel } from "../../theme/format";
import { CloneButton, ConfiguredScenarios, scenarioLabel } from "../runhub/Scenarios";
import { AcrossRunsDialog, DeleteScenarioDialog, LadderDialog, PromoteDialog } from "./components/Dialogs";
import { MemoryDialog, TrustDialog } from "./components/EngineChip";
import { LabFrame, labHref } from "./components/LabFrame";
import { Tip } from "./components/Tip";
import { lineage, ladderDoses, ladderGroups, type LineageNode } from "./model/library";
import { useTray } from "./model/tray";

const STATUSES: ScenarioStatus[] = ["draft", "previewed", "exact", "stale", "archived"];

/** A scenario's status in plain words (filter chips and the status column). */
export const STATUS_WORDS: Record<ScenarioStatus, string> = {
  draft: "draft",
  previewed: `${term("emulator_preview").label.toLowerCase()} only`,
  exact: term("exact_engine").label.toLowerCase(),
  stale: term("stale").label.toLowerCase(),
  archived: "archived",
};

/** "Run full model", the glossary's name for an exact run. */
const RUN_FULL = term("run_exact").label;
/** "full model", in sentences. */
const FULL = term("exact_engine").label.toLowerCase();

/** "−0.41 °F (−0.52 to −0.30)" */
export function likelyShort(l: Likely | null | undefined, unit: string, d = 3): string {
  if (!l) return "—";
  const u = unitLabel(unit);
  return `${fmtSigned(l.estimate, d)} ${u}${l.lo !== null && l.hi !== null ? ` (${fmtSigned(l.lo, d)} to ${fmtSigned(l.hi, d)})` : ""}`;
}

type Dialog = { kind: "ladder"; scenario: Scenario } | { kind: "across"; sid: string } | { kind: "promote"; sid: string } | { kind: "delete"; sid: string; name: string } | null;

/** The edit a ladder starts on: the first one with an amount (per-cell edits have none). */
export function firstLadderEdit(s: Scenario): number {
  const i = s.doc.edits.findIndex((e) => e.mode !== "per_cell");
  return i < 0 ? 0 : i;
}

/** A change (negative = cooler) as cooling (higher = cooler). */
function cooling(v: number | null | undefined): number | null {
  return typeof v === "number" && Number.isFinite(v) ? (v === 0 ? 0 : -v) : null;
}

function Ladders({ list, unit }: { list: ScenarioSummary[]; unit: string }) {
  const groups = useMemo(() => ladderGroups(list), [list]);
  const [docs, setDocs] = useState<Record<string, Scenario>>({});
  const ids = groups.flatMap((g) => g.members.map((m) => m.id)).join(",");
  useEffect(() => {
    let live = true;
    const want = ids ? ids.split(",") : [];
    Promise.all(want.map((id) => getScenario(id).catch(() => null))).then((all) => {
      if (!live) return;
      const out: Record<string, Scenario> = {};
      for (const s of all) if (s) out[s.id] = s;
      setDocs(out);
    });
    return () => {
      live = false;
    };
  }, [ids]);
  const u = unitLabel(unit);
  const ladders = groups.flatMap((g) => {
    const full = g.members.map((m) => docs[m.id]).filter((s): s is Scenario => !!s);
    if (full.length !== g.members.length) return [];
    const d = ladderDoses(full.map((s) => s.doc));
    if (!d) return [];
    const pts = g.members
      .map((m, i) => ({ dose: d.doses[i], l: m.latest?.city ?? null, ex: m.latest?.frac_extrapolated_edited ?? null }))
      .sort((a, b) => a.dose - b.dose);
    return [{ id: g.id, lever: d.lever, pts }];
  });
  if (!groups.length) return null;
  if (!ladders.length) return <p className="cap">Loading ladders…</p>;
  return (
    <SmallMultiples
      title={`${term("ladder").label}s`}
      units={`${u} of cooling, citywide (higher = cooler)`}
      items={ladders}
      panelTitle={(l) => `${l.lever} · ladder ${l.id}`}
      renderPanel={(l) => (
        <LineBand
          bare
          title={l.lever}
          series={[
            {
              id: l.id,
              label: `${l.lever} ladder`,
              x: l.pts.map((p) => p.dose),
              // drawn as cooling (higher = cooler), the same way round as What cools and the Summary
              y: l.pts.map((p) => cooling(p.l?.estimate)),
              lo: l.pts.map((p) => cooling(p.l?.hi)),
              hi: l.pts.map((p) => cooling(p.l?.lo)),
              hollow: l.pts.map((p) => (p.ex ?? 0) > 0.2),
              points: true,
            },
          ]}
          xLabel={term("dose").label}
          yLabel="Citywide cooling"
          yUnit={u}
          yInclude={[0]}
          refLines={[{ axis: "y", value: 0 }]}
          height={200}
        />
      )}
      table={{
        columns: [
          { key: "ladder", label: "Ladder" },
          { key: "dose", label: term("dose").label },
          { key: "est", label: "Citywide cooling", unit: u },
          { key: "lo", label: "Low", unit: u },
          { key: "hi", label: "High", unit: u },
        ],
        rows: ladders.flatMap((l) => l.pts.map((p) => [l.id, p.dose, cooling(p.l?.estimate), cooling(p.l?.hi), cooling(p.l?.lo)])),
      }}
      caption={
        <>
          Each ladder runs one change at growing amounts (full-model results; an amount without a result is a gap). Hollow points lie mostly{" "}
          {term("extrapolated").label.toLowerCase()}.
          <InfoTip term="ladder" />
        </>
      }
    />
  );
}

/** The Lab's actions on a configured scenario: clone to edit, re-run exactly (no fold detail), pin, map. */
function ConfiguredActions({ rid, pid, row }: { rid: string; pid: string | null; row: ScenarioRow }) {
  const pin = useTray((s) => s.pin);
  const [busy, setBusy] = useState(false);
  const rerun = async () => {
    setBusy(true);
    try {
      const job = await rerunConfigured(rid, row.slug);
      useJobs.getState().upsert(job);
      toast("info", `Running "${scenarioLabel(row)}" through the ${FULL} again`, { href: `/jobs/${job.id}`, linkLabel: "Track" });
    } catch (e) {
      toast("error", `Could not start the ${FULL} run`, { body: errorMessage(e) });
    } finally {
      setBusy(false);
    }
  };
  return (
    <span className="row" style={{ gap: 4, flexWrap: "nowrap" }}>
      <CloneButton rid={rid} slug={row.slug} pid={pid} />
      {!row.has_folds ? (
        <Tip tip={`This older run kept no result for each test area. Run the ${FULL} again to compare this scenario with others, with the ${term("paired_se").label.toLowerCase()}.`}>
          <Button size="small" busy={busy} onClick={() => void rerun()}>
            {RUN_FULL} again
          </Button>
        </Tip>
      ) : null}
      <Button size="small" variant="ghost" icon="pin" onClick={() => (pin(rid, { kind: "configured", slug: row.slug }, scenarioLabel(row)) ? toast("success", "Pinned to the compare tray") : toast("warning", "The compare tray holds 4 items"))}>
        Pin
      </Button>
      <Link className="btn small ghost" to={`/r/${encodeURIComponent(rid)}/map?layer=${encodeURIComponent(`sc:${row.slug}`)}`}>
        Map
      </Link>
    </span>
  );
}

export default function Library() {
  const { params, query } = useRoute();
  const rid = params.rid ?? "";
  const detail = useRunDetail(rid);
  const ctxPid = useUi((s) => s.context.projectId);
  const pid = detail.data?.run.project_id ?? ctxPid;
  const grid = useRunGrid(rid);
  const unit = grid.data?.meta.units.target ?? "";
  const [q, setQ] = useUrlState("q", codecs.string());
  const [status, setStatus] = useUrlState("status", codecs.list());
  const [tag, setTag] = useUrlState("tag", codecs.optString());
  const [thisRun, setThisRun] = useUrlState("this_run", codecs.bool(false));
  const [archived, setArchived] = useUrlState("archived", codecs.bool(false));
  const focus = query.get("configured");
  const list = useScenarios(pid, { q: q || undefined, tag: tag ?? undefined, run: thisRun ? rid : undefined, archived }, rid);
  const pin = useTray((s) => s.pin);
  const [selected, setSelected] = useState<string[]>([]);
  const [dialog, setDialog] = useState<Dialog>(null);
  const [busy, setBusy] = useState<string | null>(null);

  const items = useMemo(() => (list.data ?? []).filter((s) => !status.length || status.includes(s.status)), [list.data, status]);
  const tree = useMemo(() => lineage(items), [items]);
  const allTags = useMemo(() => [...new Set((list.data ?? []).flatMap((s) => s.tags))].sort(), [list.data]);

  const act = async (id: string, f: () => Promise<unknown>, ok?: string) => {
    setBusy(id);
    try {
      await f();
      if (ok) toast("success", ok);
      invalidate(`project:${pid}:scenarios`);
    } catch (e) {
      toast("error", "That did not work", { body: errorMessage(e) });
    } finally {
      setBusy(null);
    }
  };
  const [refusal, setRefusal] = useState<{ error: ApiError; retry: () => void } | null>(null);
  const runExact = (s: ScenarioSummary): Promise<void> =>
    act(s.id, async () => {
      try {
        const r = await runScenarioExact(s.id, rid);
        if (r.job) {
          useJobs.getState().upsert(r.job);
          toast("info", `Running "${s.name}" through the ${FULL}`, { href: `/jobs/${r.job.id}`, linkLabel: "Track" });
        } else toast("success", `"${s.name}" already has this ${FULL} result`);
      } catch (e) {
        // The checkpoint needs trust, or memory is short: explain with the dialog, not a toast.
        if (e instanceof ApiError && (e.code === "untrusted_pickle" || e.code === "engine_memory")) setRefusal({ error: e, retry: () => void runExact(s) });
        else throw e;
      }
    });
  const batch = () =>
    act("batch", async () => {
      const job = await runBatch(rid, selected);
      useJobs.getState().upsert(job);
      toast("info", `Running ${selected.length} scenarios through the ${FULL}`, { href: `/jobs/${job.id}`, linkLabel: "Track" });
      setSelected([]);
    });
  const openLadder = (s: ScenarioSummary) => act(s.id, async () => setDialog({ kind: "ladder", scenario: await getScenario(s.id) }));

  if (!pid && detail.data) return <EmptyState title="Not in a project" body="This run is not part of a project, so it has no scenario library." />;

  return (
    <LabFrame rid={rid} title="Scenario library" actions={<Link className="btn small primary" to={labHref(rid)}>New scenario</Link>}>
      <div className="stack composer" data-composer="library">
        <section className="composer-section" data-group="configured">
          <ConfiguredScenarios
            title={term("configured_scenario").label}
            intro={`${term("configured_scenario").tip} Negative changes are cooler.`}
            focus={focus}
            actions={(r) => <ConfiguredActions rid={rid} pid={pid} row={r} />}
          />
        </section>
        <section className="composer-section stack" data-group="project" aria-labelledby="library-project">
          <h3 id="library-project" className="kicker composer-heading">
            Project scenarios
          </h3>
          <div className="row" role="search">
            <input type="search" aria-label="Search scenarios" placeholder="Search names, tags, notes" value={q} onChange={(e) => setQ(e.target.value)} style={{ minWidth: 240 }} />
            <div className="chips" role="group" aria-label="Status">
              {STATUSES.filter((s) => s !== "archived").map((s) => (
                <button key={s} type="button" className="chip" aria-pressed={status.includes(s)} data-status={s} onClick={() => setStatus(status.includes(s) ? status.filter((x) => x !== s) : [...status, s])}>
                  {STATUS_WORDS[s]}
                </button>
              ))}
            </div>
            <select aria-label="Tag" value={tag ?? ""} onChange={(e) => setTag(e.target.value || null)}>
              <option value="">All tags</option>
              {allTags.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
            <label className="row">
              <input type="checkbox" checked={thisRun} onChange={(e) => setThisRun(e.target.checked)} /> With results on this run
            </label>
            <label className="row">
              <input type="checkbox" checked={archived} onChange={(e) => setArchived(e.target.checked)} /> Show archived
            </label>
            <span className="spacer" />
            <Button variant="primary" size="small" disabled={!selected.length} busy={busy === "batch"} onClick={() => void batch()}>
              {RUN_FULL} for the selected ({selected.length})
            </Button>
          </div>
          {list.error && !list.data ? <EmptyState error={list.error} /> : null}
          <Table<LineageNode>
            caption="Scenarios (lineage)"
            csvName="scenarios"
            rowKey={(n) => n.s.id}
            columns={[
              {
                key: "pick",
                label: "Select",
                sortable: false,
                value: () => null,
                render: (n) => (
                  <input
                    type="checkbox"
                    aria-label={`Select ${n.s.name}`}
                    checked={selected.includes(n.s.id)}
                    onChange={(e) => setSelected((cur) => (e.target.checked ? [...cur, n.s.id] : cur.filter((x) => x !== n.s.id)))}
                  />
                ),
              },
              {
                key: "name",
                label: "Scenario",
                sortable: false,
                value: (n) => n.s.name,
                render: (n) => (
                  <span className="lineage-row">
                    {n.depth ? <span className="lineage-indent" aria-hidden="true">{"  ".repeat(n.depth - 1) + "└ "}</span> : null}
                    <Link to={labHref(rid, `s/${encodeURIComponent(n.s.id)}`)}>{n.s.name}</Link>
                    {n.depth ? <span className="sr-only"> (revision of an earlier scenario)</span> : null}
                  </span>
                ),
              },
              { key: "rev", label: "Rev.", align: "right", sortable: false, value: (n) => n.s.revision },
              { key: "status", label: "Status", sortable: false, value: (n) => STATUS_WORDS[n.s.status] ?? n.s.status, render: (n) => <StatusChip status={n.s.status === "exact" ? "done" : n.s.status === "draft" ? "planned" : n.s.status} text={STATUS_WORDS[n.s.status] ?? n.s.status} /> },
              { key: "tags", label: "Tags", sortable: false, value: (n) => n.s.tags.join(" "), render: (n) => n.s.tags.map((t) => <span key={t} className="chip">{t}</span>) },
              {
                key: "latest",
                label: `Latest ${FULL} result`,
                sortable: false,
                value: (n) => n.s.latest?.city.estimate ?? null,
                render: (n) =>
                  n.s.latest ? (
                    <span>
                      {likelyShort(n.s.latest.edited ?? n.s.latest.city, unit)}
                      {n.s.latest.stale ? (
                        <Tip tip={term("stale").tip} focusable>
                          <Badge tone="warn">{STATUS_WORDS.stale}</Badge>
                        </Tip>
                      ) : null}
                      {n.s.latest.run_id !== rid ? <span className="cap"> · other run</span> : null}
                    </span>
                  ) : (
                    "—"
                  ),
              },
              { key: "updated", label: "Updated", sortable: false, value: (n) => n.s.updated_utc, render: (n) => fmtRelative(n.s.updated_utc) },
              {
                key: "actions",
                label: "Actions",
                sortable: false,
                value: () => null,
                render: (n) => {
                  const s = n.s;
                  return (
                    <span className="row" style={{ gap: 4 }}>
                      <Button size="small" busy={busy === s.id} onClick={() => void runExact(s)} disabled={s.status === "archived"}>
                        {RUN_FULL}
                      </Button>
                      {s.latest && s.latest.run_id === rid ? (
                        <Button size="small" variant="ghost" icon="pin" onClick={() => (pin(rid, { kind: "result", id: s.latest!.id }, s.name) ? toast("success", "Pinned to the compare tray") : toast("warning", "The compare tray holds 4 items"))}>
                          Pin
                        </Button>
                      ) : null}
                      <select
                        aria-label={`More actions for ${s.name}`}
                        value=""
                        onChange={(e) => {
                          const v = e.target.value;
                          e.target.value = "";
                          if (v === "fork") void act(s.id, async () => navigate(labHref(rid, `s/${encodeURIComponent((await forkScenario(s.id)).id)}`)));
                          else if (v === "ladder") void openLadder(s);
                          else if (v === "across") setDialog({ kind: "across", sid: s.id });
                          else if (v === "promote") setDialog({ kind: "promote", sid: s.id });
                          else if (v === "archive") void act(s.id, () => patchScenario(s.id, { archived: s.status !== "archived" }), s.status === "archived" ? "Restored" : "Archived");
                          else if (v === "delete") setDialog({ kind: "delete", sid: s.id, name: s.name });
                        }}
                      >
                        <option value="">More…</option>
                        <option value="fork">Copy as a variant</option>
                        <option value="ladder">Try growing amounts…</option>
                        <option value="across">Check across runs…</option>
                        <option value="promote">Add to the project settings…</option>
                        <option value="archive">{s.status === "archived" ? "Restore" : "Archive"}</option>
                        <option value="delete">Delete</option>
                      </select>
                    </span>
                  );
                },
              },
            ]}
            rows={tree}
            empty={<span className="cap">{list.data ? "No scenarios match. Start one from Design or a template." : "Loading…"}</span>}
          />
          <Ladders list={list.data ?? []} unit={unit} />
        </section>
      </div>
      {refusal?.error.code === "untrusted_pickle" ? (
        <TrustDialog
          rid={rid}
          error={refusal.error}
          onClose={() => setRefusal(null)}
          onTrusted={() => {
            const retry = refusal.retry;
            setRefusal(null);
            retry();
          }}
        />
      ) : null}
      {refusal?.error.code === "engine_memory" ? <MemoryDialog error={refusal.error} onClose={() => setRefusal(null)} /> : null}
      {dialog?.kind === "ladder" ? <LadderDialog sid={dialog.scenario.id} rid={rid} edits={dialog.scenario.doc.edits} editIndex={firstLadderEdit(dialog.scenario)} onClose={() => setDialog(null)} /> : null}
      {dialog?.kind === "delete" ? (
        <DeleteScenarioDialog
          sid={dialog.sid}
          name={dialog.name}
          onDeleted={() => {
            toast("success", "Deleted");
            setSelected((cur) => cur.filter((x) => x !== dialog.sid));
            invalidate(`project:${pid}:scenarios`);
          }}
          onClose={() => setDialog(null)}
        />
      ) : null}
      {dialog?.kind === "across" && pid ? <AcrossRunsDialog sid={dialog.sid} pid={pid} rid={rid} onClose={() => setDialog(null)} /> : null}
      {dialog?.kind === "promote" ? <PromoteDialog sid={dialog.sid} onClose={() => setDialog(null)} /> : null}
    </LabFrame>
  );
}
