// Planner pack (SPEC §6.4): exposure bars and the person-mean temperature table, hot-days
// small multiples (campaign-like vs lower bound), equity quintiles with concentration-index
// chips, plantable KPIs, the sortable zone table, hex choropleth links (250 / 500 m), logger
// sites and before/after pairs as map overlays, the GIS gallery (GeoTIFF previews,
// hexagons.gpkg, CSVs with lon/lat added on export) and "Re-run planner with package…".
// Hot days, equity and zones are optional parts of a pack (no station record, no adaptation
// package, no zone column): a pack without them says why, not "older code".
import { useMemo, useState } from "react";
import { api, errorMessage } from "../../api/client";
import type { PlannerSections } from "../../api/runs";
import { fileRawUrl, useGridMeta, useView } from "../../api/runs";
import type { Job } from "../../api/types";
import { Bars, LineBand, SmallMultiples } from "../../charts";
import { Button } from "../../components/ui/Button";
import { Pill } from "../../components/ui/Pill";
import type { OverlayFeature } from "../../map/Overlay";
import { Link } from "../../router";
import { useJobs } from "../../stores/jobs";
import { toast } from "../../stores/ui";
import { fmtBytes, fmtNum, fmtPct, unitLabel } from "../../theme/format";
import { Block, GenericTableView, KpiTiles, OlderCode, Section, ViewPage, useRid, useUnits, type Absence } from "./common";
import { RunLayerMap } from "./RunLayerMap";

function Exposure({ e }: { e: NonNullable<PlannerSections["exposure"]> }) {
  const u = unitLabel(useUnits().target);
  const cats = e.thresholds.map((t) => `≥ ${fmtNum(t, 0)} ${u}`);
  const pct = (a: (number | null)[] | null | undefined) => a?.map((v) => (v === null ? null : v * 100));
  const today = e.groups[0];
  return (
    <Bars
      title="Residents exposed to heat"
      units="% of residents"
      categories={cats}
      categoryLabel="Temperature"
      series={e.groups.map((g, i) => ({ id: g.id, label: g.label, values: pct(g.share) ?? [], lo: pct(g.lo), hi: pct(g.hi), muted: i > 2 }))}
      valueLabel="Residents"
      unit="%"
      decimals={1}
      caption={today ? `${today.label}: ${e.thresholds.map((t, i) => `${fmtPct(today.share[i], 0)} of residents at ≥ ${fmtNum(t, 0)} ${u}`).join(", ")}.` : undefined}
    />
  );
}

function HotDays({ h }: { h: NonNullable<PlannerSections["hot_days"]> }) {
  const u = unitLabel(useUnits().target);
  return (
    <SmallMultiples
      title="Hot days per year"
      units={h.unit}
      items={h.panels}
      panelTitle={(p) => p.label}
      renderPanel={(p) => (
        <LineBand
          bare
          title={p.label}
          series={[
            { id: "campaign", label: "Campaign-like day", x: p.thresholds, y: p.campaign },
            { id: "lower", label: "Lower bound", x: p.thresholds, y: p.lower, dashed: true },
          ]}
          xLabel="Threshold"
          xUnit={u}
          yLabel="Days"
          decimals={1}
          height={180}
        />
      )}
      table={{
        columns: [
          { key: "p", label: "Panel" },
          { key: "t", label: "Threshold" },
          { key: "c", label: "Campaign-like days" },
          { key: "l", label: "Lower-bound days" },
        ],
        rows: h.panels.flatMap((p) => p.thresholds.map((t, i) => [p.label, t, p.campaign[i] ?? null, p.lower[i] ?? null])),
      }}
      caption="Days per year at or above each threshold: if every hot day looked like the campaign day (solid), and the lower bound from the station record (dashed)."
    />
  );
}

/**
 * Why an optional part of a planner pack is absent (core `planner_pack`): hot days need a
 * station (the run config's `planner.ghcn_station`, else the forcing station) and its daily
 * history, equity an adaptation package, the zone table a `data.zone` column.
 */
const PACK_ABSENT: Record<"hot_days" | "equity" | "zones", Absence> = {
  hot_days: {
    reason:
      "The planner pack had no weather-station record to count hot days: the run's config names no GHCN-Daily station (planner.ghcn_station) or forcing station, or the station's daily history could not be downloaded.",
  },
  equity: { reason: "The planner pack had no adaptation package (a joint scenario) whose cooling it could share out by quintile; Re-run the planner pack can name one." },
  zones: { reason: "The run's data has no zone column (data.zone), so the pack has no zone table." },
};

function Equity({ eq }: { eq: NonNullable<PlannerSections["equity"]> }) {
  const u = unitLabel(useUnits().target);
  const [m, setM] = useState(eq.measures[0] ?? "");
  const cur = eq.measures.includes(m) ? m : eq.measures[0];
  return (
    <div className="stack">
      <div className="row">
        {eq.concentration.map((c) => (
          <Pill
            key={c.label}
            tone={c.value !== null && Math.abs(c.value) > 0.1 ? "warn" : "neutral"}
            title="Concentration index: 0 = evenly shared, positive = concentrated in higher quintiles"
          >
            {c.label}: {fmtNum(c.value, 2)}
          </Pill>
        ))}
      </div>
      {cur ? (
        <Bars
          title={`Cooling by ${cur} quintile`}
          units={u ? `${u} (positive = cooler)` : "positive = cooler"}
          categories={eq.quintiles.map((q) => q.label)}
          categoryLabel={`Quintile of ${cur} (1 = lowest)`}
          series={[{ id: cur, label: "Mean cooling", values: eq.quintiles.map((q) => q.values[cur] ?? null) }]}
          valueLabel="Mean cooling"
          unit={u || undefined}
          decimals={2}
          actions={
            eq.measures.length > 1 ? (
              <select aria-label="Equity measure" value={cur} onChange={(e) => setM(e.target.value)}>
                {eq.measures.map((x) => (
                  <option key={x} value={x}>
                    {x}
                  </option>
                ))}
              </select>
            ) : undefined
          }
          caption={`The package's mean cooling (positive = cooler) in each quintile of ${cur} (1 = lowest); a flat profile shares the benefit evenly.`}
        />
      ) : (
        <p className="cap">No equity measures in this planner pack.</p>
      )}
    </div>
  );
}

function Gis({ rid, gis, hexFiles }: { rid: string; gis: NonNullable<PlannerSections["gis"]>; hexFiles: NonNullable<PlannerSections["hex_files"]> | null }) {
  const [open, setOpen] = useState<string | null>(null);
  const gridMeta = useGridMeta(rid);
  const noCrs = gridMeta.data ? !gridMeta.data.crs : false;
  const files = [...gis, ...(hexFiles ?? []).map((h) => ({ relpath: h.relpath, kind: `hexagons ${h.size_m} m (${h.format})`, bytes: null, layer: null }))];
  if (!files.length) return <p className="cap">No GIS files in this planner pack.</p>;
  return (
    <div className="stack">
      <ul className="grid3" style={{ listStyle: "none", padding: 0, margin: 0 }} aria-label="GIS files">
        {files.map((f) => {
          const isCsv = /\.csv$/i.test(f.relpath);
          return (
            <li key={f.relpath} className="card" style={{ padding: 10 }}>
              <div className="mono cap" style={{ overflowWrap: "anywhere" }}>
                {f.relpath}
              </div>
              <div className="cap">
                {f.kind}
                {f.bytes !== null ? ` · ${fmtBytes(f.bytes)}` : ""}
              </div>
              <div className="row" style={{ marginTop: 6 }}>
                <a className="btn small" href={fileRawUrl(rid, f.relpath)} download>
                  Download
                </a>
                {isCsv ? (
                  <a
                    className="btn small ghost"
                    href={fileRawUrl(rid, f.relpath, "csv")}
                    download
                    title={noCrs ? "Row indices converted to ids (this run has no CRS, so no lon/lat)" : "Row indices converted to ids with lon/lat"}
                  >
                    {noCrs ? "CSV with ids" : "CSV with lon/lat"}
                  </a>
                ) : null}
                {f.layer ? (
                  <button type="button" className="btn small ghost" aria-pressed={open === f.layer} onClick={() => setOpen(open === f.layer ? null : f.layer)}>
                    Preview
                  </button>
                ) : null}
              </div>
            </li>
          );
        })}
      </ul>
      {open ? <RunLayerMap rid={rid} keys={[open]} title="GeoTIFF preview" height={300} /> : null}
    </div>
  );
}

function RerunPlanner({ rid }: { rid: string }) {
  const scen = useView(rid, "scenarios");
  const packages = useMemo(() => (scen.data?.sections.rows ?? []).filter((r) => r.kind === "joint"), [scen.data]);
  const [pkg, setPkg] = useState("");
  const [busy, setBusy] = useState(false);
  return (
    <form
      className="row"
      onSubmit={async (e) => {
        e.preventDefault();
        setBusy(true);
        try {
          const job = await api.post<Job>(`/api/runs/${encodeURIComponent(rid)}/actions/planner`, pkg ? { package: pkg } : {});
          useJobs.getState().upsert(job);
          toast("info", "Planner pack started", { href: `/jobs/${job.id}`, linkLabel: "Track" });
        } catch (err) {
          toast("error", "Could not start the planner pack", { body: errorMessage(err) });
        } finally {
          setBusy(false);
        }
      }}
    >
      <label className="row cap">
        Package
        <select value={pkg} onChange={(e) => setPkg(e.target.value)} aria-label="Package for the planner pack">
          <option value="">Default (from the config)</option>
          {packages.map((p) => (
            <option key={p.slug} value={p.slug}>
              {p.name}
            </option>
          ))}
        </select>
      </label>
      <Button type="submit" busy={busy}>
        Re-run planner with package
      </Button>
    </form>
  );
}

export default function Planner() {
  const rid = useRid();
  return (
    <ViewPage view="planner" title="Planner pack" intro="Who is exposed, where to act, and the files to take into the field and into GIS.">
      {(s) => {
        // The view read a pack (exposure is always built from one): its null parts have known reasons.
        const absent = (k: keyof typeof PACK_ABSENT) => (s.exposure !== null ? PACK_ABSENT[k] : null);
        const overlays: OverlayFeature[] = [];
        if (s.sites?.length)
          overlays.push({ type: "points", id: "sites", label: "Logger sites", points: s.sites.map((p) => ({ row: p.row, label: p.label ?? String(p.id), tone: "s2" as const })) });
        if (s.pairs?.length)
          overlays.push({ type: "links", id: "pairs", label: "Before/after pairs", links: s.pairs.map((p) => ({ from: { row: p.treated }, to: { row: p.control } })), tone: "s3" });
        return (
          <>
            <Section title="Plantable space" data={s.plantable}>
              {(k) => <KpiTiles kpis={k} label="Plantable space" />}
            </Section>
            <div className="grid2">
              <Section title="Exposure" data={s.exposure}>
                {(e) => <Exposure e={e} />}
              </Section>
              <Section title="Person-mean temperature" data={s.person_mean}>
                {(t) => (
                  <Block title="Person-mean temperature">
                    <GenericTableView table={t} caption="Person-mean temperature" csvName="person-mean-temperature" />
                  </Block>
                )}
              </Section>
            </div>
            <Section title="Hot days" data={s.hot_days} absent={absent("hot_days")}>
              {(h) => <HotDays h={h} />}
            </Section>
            <Section title="Equity" data={s.equity} absent={absent("equity")}>
              {(eq) => (
                <Block title="Equity">
                  <Equity eq={eq} />
                </Block>
              )}
            </Section>
            <Section title="Zones" data={s.zones} absent={absent("zones")}>
              {(z) => (
                <Block
                  title="Zones"
                  actions={
                    <>
                      <Link to={`/r/${encodeURIComponent(rid)}/map?hex=250`}>Hex map 250 m</Link>
                      <Link to={`/r/${encodeURIComponent(rid)}/map?hex=500`}>500 m</Link>
                    </>
                  }
                >
                  <GenericTableView table={z} caption="Zones" csvName="planner-zones" />
                </Block>
              )}
            </Section>
            <Block title="Logger sites and before/after pairs">
              {s.sites === null && s.pairs === null ? (
                <OlderCode title="Logger sites and before/after pairs" />
              ) : (
                <>
                  <p className="cap">
                    {s.sites?.length ?? 0} logger sites (orange) and {s.pairs?.length ?? 0} before/after pairs (green links).
                  </p>
                  <RunLayerMap rid={rid} keys={["people", "obs", "plantable_pp"]} title="Field kit" overlays={overlays} />
                </>
              )}
            </Block>
            <Section title="GIS gallery" data={s.gis}>
              {(g) => (
                <Block title="GIS gallery">
                  <Gis rid={rid} gis={g} hexFiles={s.hex_files} />
                </Block>
              )}
            </Section>
            <Block title="Re-run the planner pack">
              <RerunPlanner rid={rid} />
            </Block>
          </>
        );
      }}
    </ViewPage>
  );
}
