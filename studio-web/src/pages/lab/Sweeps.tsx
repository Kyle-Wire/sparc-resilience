// Sweeps (`/r/:rid/lab/sweeps/:swid?`, SPEC §7.9): one lever at custom doses, optionally in a
// region, run exactly per dose (engine.sweep). The curve shows the region and city mean ΔT
// with their likely-range ribbons (hollow points when mostly extrapolated), the fitted
// saturation overlay and the pipeline's own response curve. The overlay, its d90 line and the
// caption use the fit on the requested dose (the x axis); the fit on the neighbourhood dose
// (smaller for a regional sweep) is only quoted, in its own units.
import { useMemo, useState } from "react";
import { errorMessage } from "../../api/client";
import { invalidate } from "../../api/resource";
import { createSweep, deleteSweep, useLevers, useRegions, useSweep, useSweeps, type Lever, type SweepListItem } from "../../api/lab";
import { isActiveStatus, type JobStatus, type SelectionSpec } from "../../api/types";
import { LineBand, type LineSeries } from "../../charts";
import { Button } from "../../components/ui/Button";
import { EmptyState } from "../../components/ui/EmptyState";
import { JobStrip } from "../../components/ui/JobStrip";
import { StatusChip } from "../../components/ui/StatusChip";
import { Table } from "../../components/ui/Table";
import { useRunGrid, useRunLayers } from "../../map/data";
import type { GridData } from "../../map/grid";
import { Link, navigate, useRoute } from "../../router";
import { useJobs } from "../../stores/jobs";
import { toast } from "../../stores/ui";
import { fmtDateTime, fmtNum, fmtPct, fmtSigned, unitLabel } from "../../theme/format";
import { ConfirmDialog } from "./components/Dialogs";
import { LabFrame, labHref } from "./components/LabFrame";
import { SelectionBuilder, columnOptions } from "./components/SelectionBuilder";
import { parseNumberList } from "./model/library";
import { dominantSign, fitOverlay, overlayXs, pipelineCurve, sweepFitText } from "./model/sweep";

function NewSweep({ rid, grid, levers }: { rid: string; grid: GridData; levers: Lever[] }) {
  const regions = useRegions(rid);
  const layers = useRunLayers(rid);
  const [lever, setLever] = useState(levers[0]?.var ?? "");
  const lv = levers.find((l) => l.var === lever);
  const [doses, setDoses] = useState(() => (levers[0]?.doses.length ? levers[0].doses.join(", ") : "5, 10, 20, 30"));
  const [where, setWhere] = useState<SelectionSpec>({ kind: "all" });
  const [busy, setBusy] = useState(false);
  const parsed = parseNumberList(doses);
  const columns = useMemo(() => columnOptions(layers.data ?? []), [layers.data]);
  const start = async () => {
    setBusy(true);
    try {
      const sel = "kind" in where && where.kind === "all" ? undefined : where;
      const r = await createSweep(rid, { lever, doses: parsed.values, ...(sel ? { selection: sel } : {}) });
      useJobs.getState().upsert(r.job);
      invalidate(`run:${rid}:lab`);
      navigate(labHref(rid, `sweeps/${encodeURIComponent(r.sweep_id)}`));
    } catch (e) {
      toast("error", "Could not start the sweep", { body: errorMessage(e) });
    } finally {
      setBusy(false);
    }
  };
  return (
    <section className="card stack" aria-label="New sweep">
      <h3>New sweep</h3>
      <label className="field">
        <span className="field-label">Lever</span>
        <select
          value={lever}
          onChange={(e) => {
            setLever(e.target.value);
            const l = levers.find((x) => x.var === e.target.value);
            if (l?.doses.length) setDoses(l.doses.join(", "));
          }}
        >
          {levers.map((l) => (
            <option key={l.var} value={l.var}>
              {l.label}
            </option>
          ))}
        </select>
      </label>
      <label className="field">
        <span className="field-label">Doses{lv?.unit ? ` (${unitLabel(lv.unit)})` : ""}</span>
        <input value={doses} onChange={(e) => setDoses(e.target.value)} aria-invalid={parsed.bad.length > 0 || undefined} />
        <span className="hint">{parsed.bad.length ? `Not numbers: ${parsed.bad.join(", ")}` : "Each dose is one exact run (≈ the exact time per scenario)."}</span>
      </label>
      <div className="field">
        <span className="field-label">Where (optional)</span>
        <SelectionBuilder rid={rid} grid={grid} value={where} onChange={setWhere} columns={columns} regions={regions.data ?? []} levers={levers} idPrefix="sweep" />
      </div>
      <Button variant="primary" busy={busy} disabled={!lever || parsed.values.length < 2 || parsed.bad.length > 0} onClick={() => void start()}>
        Run sweep ({parsed.values.length} doses)
      </Button>
    </section>
  );
}

function SweepView({ rid, swid, unit, item, levers }: { rid: string; swid: string; unit: string; item: SweepListItem | undefined; levers: Lever[] }) {
  const sw = useSweep(swid, rid);
  const [confirm, setConfirm] = useState(false);
  if (sw.error && !sw.data) return <EmptyState error={sw.error} />;
  const s = sw.data;
  if (!s) return <p className="cap">Loading the sweep…</p>;
  const u = unitLabel(unit);
  const lu = unitLabel(levers.find((l) => l.var === s.params.lever)?.unit ?? "");
  const doseUnit = lu ? ` ${lu}` : "";
  const curve = [...s.curve].sort((a, b) => a.dose - b.dose);
  // only a fit on the requested dose belongs on this x axis
  const fitOnDose = s.fit && (s.fit.axis ?? "dose") === "dose" ? s.fit : null;
  const nf = s.fit_neighbourhood ?? null;
  const xs = curve.map((p) => p.dose);
  const hasRegion = curve.some((p) => p.region);
  const main = curve.map((p) => p.region ?? p.city);
  const series: LineSeries[] = [];
  if (hasRegion)
    series.push({ id: "region", label: "Region mean", x: xs, y: main.map((l) => l.estimate), lo: main.map((l) => l.lo), hi: main.map((l) => l.hi), hollow: curve.map((p) => p.frac_extrapolated > 0.2), points: true, emphasis: true });
  series.push({
    id: "city",
    label: "City mean",
    x: xs,
    y: curve.map((p) => p.city.estimate),
    lo: curve.map((p) => p.city.lo),
    hi: curve.map((p) => p.city.hi),
    hollow: curve.map((p) => p.frac_extrapolated > 0.2),
    points: true,
    muted: hasRegion,
  });
  const ox = overlayXs(xs);
  const fit = fitOverlay(fitOnDose, ox, curve.map((p, i) => ({ x: p.dose, y: main[i].estimate })));
  if (fit) series.push({ id: "fit", label: `Fit (${fitOnDose?.model})`, x: ox, y: fit, dashed: true });
  const pipe = pipelineCurve(s.pipeline_curve, dominantSign(main.map((l) => l.estimate)));
  if (pipe) series.push({ id: "pipeline", label: "Pipeline response curve", x: pipe.x, y: pipe.y, dashed: true, muted: true });
  const remove = async () => {
    await deleteSweep(swid);
    invalidate(`run:${rid}:lab`);
    navigate(labHref(rid, "sweeps"));
  };
  const active = isActiveStatus(s.status as JobStatus);
  return (
    <section className="stack" aria-label="Sweep">
      <div className="row">
        <h3>
          {String(s.params.lever)} at {s.params.doses.map((d) => fmtNum(d, 2)).join(", ")}
        </h3>
        <StatusChip status={s.status} />
        <span className="spacer" />
        <Button size="small" variant="danger" onClick={() => setConfirm(true)} disabled={active} title={active ? "Cancel the sweep's job first" : undefined}>
          Delete
        </Button>
      </div>
      {confirm ? (
        <ConfirmDialog title="Delete this sweep?" confirmLabel="Delete sweep" onConfirm={remove} onClose={() => setConfirm(false)}>
          <p>The sweep and its {s.points.length} exact point results will be deleted.</p>
        </ConfirmDialog>
      ) : null}
      {item?.job_id ? <JobStrip jobId={item.job_id} /> : null}
      {curve.length ? (
        <LineBand
          title="Response to the swept lever"
          units={`${u} (negative = cooler)`}
          series={series}
          xLabel={`Requested dose${doseUnit}`}
          yLabel="Mean ΔT"
          yUnit={u}
          yInclude={[0]}
          refLines={[{ axis: "y", value: 0 }, ...(fitOnDose?.d90 ? [{ axis: "x" as const, value: fitOnDose.d90, label: "d90" }] : [])]}
          caption={`${hasRegion ? "Region and city" : "City"} mean ΔT with the likely range at each requested dose; hollow points are mostly extrapolated.${sweepFitText(fitOnDose, nf, doseUnit)}`}
        />
      ) : (
        <p className="cap">No points yet{s.status === "running" ? ": the sweep is running." : "."}</p>
      )}
      <Table
        caption="Sweep points"
        csvName={`sweep-${swid}`}
        rowKey={(p) => String(p.dose)}
        columns={[
          { key: "dose", label: "Dose", align: "right", value: (p) => p.dose },
          { key: "city", label: "City mean", unit: u, align: "right", value: (p) => p.city.estimate, render: (p) => `${fmtSigned(p.city.estimate, 4)}${p.city.se !== null ? ` ± ${fmtNum(1.96 * p.city.se, 4)}` : ""}` },
          { key: "region", label: "Region mean", unit: u, align: "right", value: (p) => p.region?.estimate ?? null, render: (p) => (p.region ? `${fmtSigned(p.region.estimate, 3)}${p.region.se !== null ? ` ± ${fmtNum(1.96 * p.region.se, 3)}` : ""}` : "—") },
          { key: "realized", label: "Realised dose", align: "right", value: (p) => p.realized, render: (p) => fmtNum(p.realized, 2) },
          { key: "neigh", label: "Neighbourhood dose", align: "right", value: (p) => p.neighbourhood_dose ?? null, render: (p) => fmtNum(p.neighbourhood_dose ?? null, 2) },
          { key: "ex", label: "Extrapolated", align: "right", value: (p) => p.frac_extrapolated, render: (p) => fmtPct(p.frac_extrapolated) },
        ]}
        rows={curve}
      />
    </section>
  );
}

export default function Sweeps() {
  const { params } = useRoute();
  const rid = params.rid ?? "";
  const swid = params.swid ?? null;
  const grid = useRunGrid(rid);
  const levers = useLevers(rid);
  const sweeps = useSweeps(rid);
  if (grid.error) return <EmptyState error={grid.error} />;
  if (!grid.data || !levers.data) return <p className="cap">Loading…</p>;
  const unit = grid.data.meta.units.target;
  return (
    <LabFrame rid={rid} title="Sweeps">
      <div className="lab-split">
        <div className="stack">
          <NewSweep rid={rid} grid={grid.data} levers={levers.data} />
          <Table<SweepListItem>
            caption="Sweeps of this run"
            csvName={null}
            rowKey={(s) => s.id}
            highlight={(s) => s.id === swid}
            columns={[
              { key: "lever", label: "Lever", value: (s) => s.lever, render: (s) => <Link to={labHref(rid, `sweeps/${encodeURIComponent(s.id)}`)}>{s.lever}</Link> },
              { key: "doses", label: "Doses", value: (s) => s.doses.join(", ") },
              { key: "status", label: "Status", value: (s) => s.status, render: (s) => <StatusChip status={s.status} /> },
              { key: "created", label: "Created", value: (s) => s.created_utc, render: (s) => fmtDateTime(s.created_utc) },
            ]}
            rows={sweeps.data ?? []}
            empty={<span className="cap">{sweeps.data ? "No sweeps yet." : "Loading…"}</span>}
          />
        </div>
        {swid ? <SweepView rid={rid} swid={swid} unit={unit} item={sweeps.data?.find((s) => s.id === swid)} levers={levers.data} /> : <p className="cap">Choose a sweep, or run a new one.</p>}
      </div>
    </LabFrame>
  );
}
