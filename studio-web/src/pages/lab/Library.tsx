// Scenario library (`/r/:rid/lab/library`, SPEC §7.4, §7.8, §7.12): every scenario of the
// project as a lineage tree (revisions under their parents) with search, tag and status
// filters; batch exact runs; per-scenario actions (open, pin to compare, run exact, fork,
// ladder, check across runs with its time and RAM estimate, promote with the YAML diff,
// archive, delete); the run's "Configured in this run" group (clone to edit, re-run exactly
// for paired SE, pin); ladder charts (ΔT vs dose with the likely-range ribbon).
// `?configured=<slug>` focuses one configured scenario (the run hub's "Open in Lab").
import { useEffect, useMemo, useState } from "react";
import { ApiError, errorMessage } from "../../api/client";
import { invalidate } from "../../api/resource";
import {
  createScenario,
  forkScenario,
  getScenario,
  patchScenario,
  rerunConfigured,
  runBatch,
  runScenarioExact,
  useRunScenarios,
  useScenarios,
  type ConfiguredScenario,
  type Scenario,
  type ScenarioStatus,
  type ScenarioSummary,
} from "../../api/lab";
import type { Likely } from "../../api/types";
import { LineBand, SmallMultiples } from "../../charts";
import { Badge } from "../../components/ui/Badge";
import { Button } from "../../components/ui/Button";
import { EmptyState } from "../../components/ui/EmptyState";
import { StatusChip } from "../../components/ui/StatusChip";
import { Table } from "../../components/ui/Table";
import { useRunDetail } from "../../layouts/resources";
import { useRunGrid } from "../../map/data";
import { Link, codecs, navigate, useRoute, useUrlState } from "../../router";
import { useJobs } from "../../stores/jobs";
import { toast, useUi } from "../../stores/ui";
import { fmtPct, fmtRelative, fmtSigned, unitLabel } from "../../theme/format";
import { AcrossRunsDialog, DeleteScenarioDialog, LadderDialog, PromoteDialog } from "./components/Dialogs";
import { MemoryDialog, TrustDialog } from "./components/EngineChip";
import { LabFrame, labHref } from "./components/LabFrame";
import { lineage, ladderDoses, ladderGroups, type LineageNode } from "./model/library";
import { confidenceWord } from "./model/plain";
import { useTray } from "./model/tray";

const STATUSES: ScenarioStatus[] = ["draft", "previewed", "exact", "stale", "archived"];

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
      title="Ladders"
      units={`${u} city mean (negative = cooler)`}
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
              y: l.pts.map((p) => p.l?.estimate ?? null),
              lo: l.pts.map((p) => p.l?.lo ?? null),
              hi: l.pts.map((p) => p.l?.hi ?? null),
              hollow: l.pts.map((p) => (p.ex ?? 0) > 0.2),
              points: true,
            },
          ]}
          xLabel="Dose"
          yLabel="City-mean ΔT"
          yUnit={u}
          yInclude={[0]}
          refLines={[{ axis: "y", value: 0 }]}
          height={200}
        />
      )}
      table={{
        columns: [
          { key: "ladder", label: "Ladder" },
          { key: "dose", label: "Dose" },
          { key: "est", label: "City-mean ΔT", unit: u },
          { key: "lo", label: "Low", unit: u },
          { key: "hi", label: "High", unit: u },
        ],
        rows: ladders.flatMap((l) => l.pts.map((p) => [l.id, p.dose, p.l?.estimate ?? null, p.l?.lo ?? null, p.l?.hi ?? null])),
      }}
      caption="Each ladder repeats one edit at increasing doses (exact results; rungs without a result are gaps). Hollow points are mostly extrapolated."
    />
  );
}

function ConfiguredGroup({ rid, pid, focus, unit }: { rid: string; pid: string | null; focus: string | null; unit: string }) {
  const data = useRunScenarios(rid);
  const pin = useTray((s) => s.pin);
  const [busy, setBusy] = useState<string | null>(null);
  useEffect(() => {
    if (!focus || !data.data) return;
    const el = document.querySelector(`[data-configured="${focus.replace(/["\\]/g, "\\$&")}"]`);
    el?.scrollIntoView?.({ block: "center" });
  }, [focus, data.data]);
  if (data.error && !data.data) return <EmptyState error={data.error} />;
  const rows = data.data?.configured ?? [];
  const clone = async (c: ConfiguredScenario) => {
    if (!pid) return;
    setBusy(c.slug);
    try {
      const s = await createScenario(pid, { ...c.doc, name: `${c.name} (copy)`, anchor_run_id: rid });
      navigate(labHref(rid, `s/${encodeURIComponent(s.id)}`));
    } catch (e) {
      toast("error", "Could not clone the scenario", { body: errorMessage(e) });
    } finally {
      setBusy(null);
    }
  };
  const rerun = async (c: ConfiguredScenario) => {
    setBusy(c.slug);
    try {
      const job = await rerunConfigured(rid, c.slug);
      useJobs.getState().upsert(job);
      toast("info", `Re-running "${c.name}" exactly`, { href: `/jobs/${job.id}`, linkLabel: "Track" });
    } catch (e) {
      toast("error", "Could not start the exact re-run", { body: errorMessage(e) });
    } finally {
      setBusy(null);
    }
  };
  return (
    <section className="stack" aria-label="Configured in this run">
      <h3>Configured in this run</h3>
      <p className="cap">The scenarios of the project config, evaluated at the end of S5. Read-only: clone one to edit it.</p>
      {focus && !rows.some((c) => c.slug === focus) && data.data ? <p className="callout">This run has no configured scenario “{focus}”.</p> : null}
      <Table<ConfiguredScenario>
        caption="Configured scenarios"
        csvName="configured-scenarios"
        rowKey={(c) => c.slug}
        highlight={(c) => c.slug === focus}
        columns={[
          { key: "name", label: "Scenario", value: (c) => c.name, render: (c) => <span data-configured={c.slug}>{c.name}</span> },
          { key: "city", label: "City-mean change", unit: unitLabel(unit), align: "right", value: (c) => c.city.estimate, render: (c) => likelyShort(c.city, unit) },
          { key: "conf", label: "Confidence", value: (c) => confidenceWord(c.city) },
          { key: "ex", label: "Extrapolated", align: "right", value: (c) => c.frac_extrapolated, render: (c) => fmtPct(c.frac_extrapolated) },
          { key: "folds", label: "Paired SE", value: (c) => (c.has_folds ? "ready" : "needs exact re-run"), render: (c) => (c.has_folds ? "ready" : <Badge tone="warn">needs exact re-run</Badge>) },
          {
            key: "actions",
            label: "Actions",
            sortable: false,
            value: () => null,
            render: (c) => (
              <span className="row" style={{ gap: 4, flexWrap: "nowrap" }}>
                <Button size="small" busy={busy === c.slug} disabled={!pid} onClick={() => void clone(c)}>
                  Clone to edit
                </Button>
                {!c.has_folds ? (
                  <Button size="small" busy={busy === c.slug} onClick={() => void rerun(c)} title="Fold detail is missing (older run): re-run exactly for paired SE">
                    Re-run exactly
                  </Button>
                ) : null}
                <Button size="small" variant="ghost" icon="pin" onClick={() => (pin(rid, { kind: "configured", slug: c.slug }, c.name) ? toast("success", "Pinned to the compare tray") : toast("warning", "The compare tray holds 4 items"))}>
                  Pin
                </Button>
                <Link className="btn small ghost" to={`/r/${encodeURIComponent(rid)}/map?layer=${encodeURIComponent(c.layer_key)}`}>
                  Map
                </Link>
              </span>
            ),
          },
        ]}
        rows={rows}
        empty={<span className="cap">{data.data ? "No configured scenarios in this run." : "Loading…"}</span>}
      />
    </section>
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
          toast("info", `Running "${s.name}" exactly`, { href: `/jobs/${r.job.id}`, linkLabel: "Track" });
        } else toast("success", `"${s.name}" already has this exact result (cached)`);
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
      toast("info", `Running ${selected.length} scenarios exactly`, { href: `/jobs/${job.id}`, linkLabel: "Track" });
      setSelected([]);
    });
  const openLadder = (s: ScenarioSummary) => act(s.id, async () => setDialog({ kind: "ladder", scenario: await getScenario(s.id) }));

  if (!pid && detail.data) return <EmptyState title="Not in a project" body="This run is not part of a project, so it has no scenario library." />;

  return (
    <LabFrame rid={rid} title="Scenario library" actions={<Link className="btn small primary" to={labHref(rid)}>New scenario</Link>}>
      <div className="row" role="search">
        <input type="search" aria-label="Search scenarios" placeholder="Search names, tags, notes" value={q} onChange={(e) => setQ(e.target.value)} style={{ minWidth: 240 }} />
        <div className="chips" role="group" aria-label="Status">
          {STATUSES.filter((s) => s !== "archived").map((s) => (
            <button key={s} type="button" className="chip" aria-pressed={status.includes(s)} onClick={() => setStatus(status.includes(s) ? status.filter((x) => x !== s) : [...status, s])}>
              {s}
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
          Run selected exactly ({selected.length})
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
          { key: "status", label: "Status", sortable: false, value: (n) => n.s.status, render: (n) => <StatusChip status={n.s.status === "exact" ? "done" : n.s.status === "draft" ? "planned" : n.s.status} text={n.s.status} /> },
          { key: "tags", label: "Tags", sortable: false, value: (n) => n.s.tags.join(" "), render: (n) => n.s.tags.map((t) => <span key={t} className="chip">{t}</span>) },
          {
            key: "latest",
            label: "Latest exact",
            sortable: false,
            value: (n) => n.s.latest?.city.estimate ?? null,
            render: (n) =>
              n.s.latest ? (
                <span>
                  {likelyShort(n.s.latest.edited ?? n.s.latest.city, unit)}
                  {n.s.latest.stale ? <Badge tone="warn">stale</Badge> : null}
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
                    Run exact
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
                    <option value="fork">Fork a variant</option>
                    <option value="ladder">Make a ladder…</option>
                    <option value="across">Check across runs…</option>
                    <option value="promote">Promote to config…</option>
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
      <ConfiguredGroup rid={rid} pid={pid} focus={focus} unit={unit} />
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
