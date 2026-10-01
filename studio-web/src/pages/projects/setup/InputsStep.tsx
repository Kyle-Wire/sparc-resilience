// Setup step "inputs" (SPEC §9.3, J2 step 6): four cards, each a tracked network job with a
// host pre-check, its status, the result views and "Link into config" with a YAML diff:
// - Campaign forcing (ERA5 + ISD station; station picker from the cached index, which is
//   fetched by its own job when missing);
// - People & land cover (HRSL + WorldCover);
// - Open predictors (Sentinel-2 / open layers; bootstrap mode for a city with no predictors);
// - CMIP6 change factors.
// Per-object progress is the job's own (job tray, JobStrip, Mission Control).
import { useMemo, useState, type ReactNode } from "react";
import { ApiError, errorMessage } from "../../../api/client";
import {
  getStations,
  INPUT_JOB_KIND,
  startCmip6,
  startFeatures,
  startForcing,
  startLayers,
  useInputs,
  useInputView,
  type FeaturesTarget,
  type InputKind,
  type ScatterBins,
  type Station,
  type WindSource,
} from "../../../api/inputs";
import { listProjectJobs, useMetaInfo, useStudioSettings, type LinkKind } from "../../../api/projects";
import { useResource } from "../../../api/resource";
import { isActiveStatus, type Job } from "../../../api/types";
import { Bars, HexbinScatter } from "../../../charts";
import { Badge } from "../../../components/ui/Badge";
import { Button } from "../../../components/ui/Button";
import { Card } from "../../../components/ui/Card";
import { Chips } from "../../../components/ui/Chips";
import { ActionButton } from "../../../components/ui/EmptyState";
import { JobStrip } from "../../../components/ui/JobStrip";
import { Kpi, KpiRow } from "../../../components/ui/Kpi";
import { NumberField } from "../../../components/ui/NumberField";
import { Pill } from "../../../components/ui/Pill";
import { Seg } from "../../../components/ui/Seg";
import { StatusChip } from "../../../components/ui/StatusChip";
import { MapView } from "../../../map/MapView";
import { Link } from "../../../router";
import { useJobs } from "../../../stores/jobs";
import { toast } from "../../../stores/ui";
import { fmtInt, fmtNum, fmtRelative, fmtSig } from "../../../theme/format";
import { useCfg } from "../components/cfg";
import { LinkDialog } from "../components/LinkDialog";
import { NetCheck } from "../components/NetCheck";
import { getList, getRecord, getString } from "../model/dotted";
import { previewGroups, previewLoader, usePreviewGrid } from "./actions";
import { DEFAULT_PERIODS, SSPS } from "./AnalysisStep";

// ---------------------------------------------------------------- shared card machinery

/** The latest job of an input kind for this project: live from the job store, else the server's list. */
export function useInputJob(pid: string, kind: InputKind): Job | null {
  const jobKind = INPUT_JOB_KIND[kind];
  const jobs = useJobs((s) => s.jobs);
  const listed = useResource(`project:${pid}:jobs:${jobKind}`, (s) => listProjectJobs(pid, jobKind, 1, s), { tags: [`project:${pid}`, "jobs"] });
  return useMemo(() => {
    const live = Object.values(jobs).filter((j) => j.project_id === pid && j.kind === jobKind);
    const all = [...live, ...(listed.data?.items ?? [])];
    if (!all.length) return null;
    all.sort((a, b) => b.created_utc.localeCompare(a.created_utc));
    const top = all[0];
    // The listed copy carries `result`; the store's copy has the freshest status.
    const fromList = listed.data?.items.find((j) => j.id === top.id);
    const fromStore = jobs[top.id];
    return fromStore ? { ...(fromList ?? {}), ...fromStore, result: fromStore.result ?? fromList?.result ?? null } : top;
  }, [jobs, listed.data, pid, jobKind]);
}

function useHosts(kind: InputKind): string[] {
  const meta = useMetaInfo();
  return meta.data?.job_kinds.find((k) => k.kind === INPUT_JOB_KIND[kind])?.network_hosts ?? [];
}

async function launch(start: () => Promise<Job>, label: string): Promise<Job | null> {
  try {
    const job = await start();
    useJobs.getState().upsert(job);
    toast("info", `${job.label || label} started`, { href: `/jobs/${job.id}`, linkLabel: "Track" });
    return job;
  } catch (e) {
    toast("error", `${label} could not start`, { body: errorMessage(e), action: e instanceof ApiError && e.action ? e.action : undefined });
    return null;
  }
}

function JobState({ job }: { job: Job | null }) {
  if (!job) return <span className="cap">Not fetched yet.</span>;
  if (isActiveStatus(job.status)) return <JobStrip jobId={job.id} />;
  return (
    <div className="row">
      <StatusChip status={job.status} />
      <span className="cap">
        Last job {fmtRelative(job.finished_utc ?? job.created_utc)} · <Link to={`/jobs/${job.id}`}>details</Link>
      </span>
      {job.status === "failed" && job.error ? <span className="cap">{job.error.message}</span> : null}
    </div>
  );
}

function InputCard({
  kind,
  title,
  eyebrow,
  status,
  children,
  start,
  startLabel,
  disabledReason,
  job,
  results,
}: {
  kind: InputKind;
  title: string;
  eyebrow: string;
  status: ReactNode;
  children: ReactNode;
  start: () => Promise<Job>;
  startLabel: string;
  disabledReason?: string | null;
  job: Job | null;
  results: ReactNode;
}) {
  const hosts = useHosts(kind);
  const settings = useStudioSettings();
  const [busy, setBusy] = useState(false);
  const offline = settings.data?.offline === true;
  const running = !!job && isActiveStatus(job.status);
  const reason = offline ? "Offline mode is on (Settings): network inputs are hidden." : disabledReason ?? null;
  return (
    <Card title={title} eyebrow={eyebrow} className="input-card" data-input={kind} actions={status}>
      <NetCheck hosts={hosts} auto={!offline} />
      {children}
      <div className="row">
        <Button
          variant="primary"
          icon="download"
          busy={busy}
          disabled={!!reason || running}
          onClick={async () => {
            setBusy(true);
            await launch(start, title);
            setBusy(false);
          }}
        >
          {startLabel}
        </Button>
        {reason ? <span className="cap">{reason}</span> : running ? <span className="cap">Running: progress is in the job tray.</span> : null}
      </div>
      <JobState job={job} />
      {results}
    </Card>
  );
}

function LinkButton({ pid, kind, path, label, linked }: { pid: string; kind: LinkKind; path: string | null | undefined; label?: string; linked?: boolean }) {
  const [open, setOpen] = useState(false);
  if (!path) return null;
  return (
    <>
      <Button icon="file" variant={linked ? "ghost" : "default"} onClick={() => setOpen(true)}>
        {label ?? (linked ? "Linked: show the config diff" : "Link into config")}
      </Button>
      <LinkDialog pid={pid} kind={kind} path={path} open={open} onClose={() => setOpen(false)} />
    </>
  );
}

const str = (v: unknown): string | null => (typeof v === "string" && v ? v : null);

/** The data centroid from the last data check (null until a check ran with a CRS). */
function useCentroid(pid: string, token: string | null) {
  const grid = usePreviewGrid(pid, token);
  const b = grid.data?.meta.bounds_lonlat;
  return b ? { lon: (b[0] + b[2]) / 2, lat: (b[1] + b[3]) / 2 } : null;
}

function defaultSite(raw: unknown, centroid: { lat: number; lon: number } | null): { lat: number | null; lon: number | null } {
  const site = getList<unknown>(raw, "climate.site");
  if (typeof site[0] === "number" && typeof site[1] === "number") return { lat: site[0], lon: site[1] };
  return centroid ? { lat: Number(centroid.lat.toFixed(4)), lon: Number(centroid.lon.toFixed(4)) } : { lat: null, lon: null };
}

function LatLon({ lat, lon, onChange }: { lat: number | null; lon: number | null; onChange: (lat: number | null, lon: number | null) => void }) {
  return (
    <div className="row">
      <NumberField label="Latitude" value={lat} onChange={(v) => onChange(v, lon)} nullable min={-90} max={90} unit="°N" />
      <NumberField label="Longitude" value={lon} onChange={(v) => onChange(lat, v)} nullable min={-180} max={180} unit="°E" />
    </div>
  );
}

// ---------------------------------------------------------------- campaign forcing

function StationPicker({ pid, lat, lon, value, onChange }: { pid: string; lat: number | null; lon: number | null; value: string; onChange: (s: string) => void }) {
  const [stations, setStations] = useState<Station[] | null>(null);
  const [error, setError] = useState<ApiError | Error | null>(null);
  const [busy, setBusy] = useState(false);
  const find = async () => {
    if (lat === null || lon === null) return;
    setBusy(true);
    setError(null);
    try {
      setStations(await getStations(pid, lat, lon, 10));
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
      setStations(null);
    } finally {
      setBusy(false);
    }
  };
  const missingIndex = error instanceof ApiError && error.status === 404;
  return (
    <div className="field station-picker">
      <span className="field-label">Station (nearest ISD stations)</span>
      <div className="row">
        <input aria-label="Station id (USAF-WBAN)" placeholder="72507014765" value={value} onChange={(e) => onChange(e.target.value)} style={{ width: "12em" }} />
        <Button size="small" icon="search" busy={busy} disabled={lat === null || lon === null} onClick={() => void find()}>
          Find nearby stations
        </Button>
      </div>
      {missingIndex ? (
        <div className="callout" data-tone="info">
          The station list is not cached yet (NOAA ISD index, ≈3 MB). Fetch it once as a job; Studio never downloads inside a page request.
          <div className="row" style={{ marginTop: 6 }}>
            <ActionButton
              size="small"
              action={
                (error as ApiError).action ?? { kind: "fetch_input", label: "Fetch station list", method: "POST", path: `/api/projects/${encodeURIComponent(pid)}/inputs/stations`, body: {} }
              }
            />
          </div>
        </div>
      ) : error ? (
        <span className="cap">Station lookup failed: {errorMessage(error)}</span>
      ) : null}
      {stations ? (
        <table className="tbl" aria-label="Nearby stations">
          <thead>
            <tr>
              <th scope="col">Use</th>
              <th scope="col">Station</th>
              <th scope="col" className="r">
                Distance
              </th>
              <th scope="col">Record</th>
            </tr>
          </thead>
          <tbody>
            {stations.map((s) => (
              <tr key={s.usaf_wban} className={s.usaf_wban === value ? "hl" : undefined}>
                <td>
                  <input type="radio" name="station" aria-label={`Use ${s.name}`} checked={s.usaf_wban === value} onChange={() => onChange(s.usaf_wban)} />
                </td>
                <td>
                  {s.name} <span className="mono cap">{s.usaf_wban}</span>
                </td>
                <td className="r num">{fmtNum(s.dist_km, 1)} km</td>
                <td className="cap">
                  {s.begin}–{s.end}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
      <span className="hint">Empty: Studio picks the nearest station with data on the day.</span>
    </div>
  );
}

function localTz(): string {
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";
  } catch {
    return "UTC";
  }
}

function ForcingCard() {
  const c = useCfg();
  const pid = c.pid;
  const inputs = useInputs(pid);
  const job = useInputJob(pid, "forcing");
  const centroid = useCentroid(pid, c.draft.check?.preview_token || null);
  const site = defaultSite(c.draft.raw, centroid);
  const [date, setDate] = useState("");
  const [hours, setHours] = useState<[number | null, number | null]>([15, 16]);
  const [tz, setTz] = useState(localTz());
  const [pos, setPos] = useState<{ lat: number | null; lon: number | null } | null>(null);
  const [station, setStation] = useState("");
  const [wind, setWind] = useState<WindSource>("auto");
  const [autoLink, setAutoLink] = useState(false);
  const where = pos ?? site;
  const f = inputs.data?.forcing ?? null;
  const done = job?.status === "succeeded";
  const view = useInputView(pid, "forcing", !!f || done);
  const path = f?.path ?? str(job?.result?.path);
  const ready = /^\d{4}-\d{2}-\d{2}$/.test(date) && hours[0] !== null && hours[1] !== null;
  return (
    <InputCard
      kind="forcing"
      title="Campaign forcing"
      eyebrow="ERA5 + ISD station"
      job={job}
      status={f ? <StatusChip status={f.linked ? "done" : "present"} text={f.linked ? "linked" : "fetched"} /> : <StatusChip status="missing" />}
      startLabel="Fetch forcing"
      disabledReason={!ready ? "Set the campaign date and hours." : where.lat === null && !station ? "Set a location (or a station)." : null}
      start={() =>
        startForcing(pid, {
          date,
          hours: [hours[0] ?? 0, hours[1] ?? 0],
          tz,
          ...(where.lat !== null && where.lon !== null ? { lat: where.lat, lon: where.lon } : {}),
          ...(station ? { station } : {}),
          wind_source: wind,
          link: autoLink,
        })
      }
      results={
        f || done ? (
          <div className="stack" style={{ gap: 8 }}>
            {view.data ? (
              <>
                <table className="tbl" aria-label="ERA5 vs station">
                  <thead>
                    <tr>
                      <th scope="col">Quantity</th>
                      <th scope="col" className="r">
                        ERA5
                      </th>
                      <th scope="col" className="r">
                        Station
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {view.data.compare.map((r) => (
                      <tr key={r.name}>
                        <td>{r.name}</td>
                        <td className="r num">{fmtSig(r.era5, 4)}</td>
                        <td className="r num">{fmtSig(r.station, 4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {view.data.checks.length ? (
                  <div className="row" aria-label="Forcing checks">
                    {view.data.checks.map((ck, i) => (
                      <Pill key={i} tone="warn" icon="alert">
                        {ck}
                      </Pill>
                    ))}
                  </div>
                ) : (
                  <Pill tone="good" icon="check">
                    Forcing checks passed
                  </Pill>
                )}
              </>
            ) : view.error ? (
              <span className="cap">{errorMessage(view.error)}</span>
            ) : null}
            <div className="row">
              <LinkButton pid={pid} kind="forcing" path={path} linked={f?.linked} />
            </div>
          </div>
        ) : null
      }
    >
      <div className="grid3">
        <label className="field">
          <span className="field-label">Campaign date</span>
          <input type="date" value={date} onChange={(e) => setDate(e.target.value)} aria-label="Campaign date" />
        </label>
        <div className="field">
          <span className="field-label">Hours (local, from–to)</span>
          <div className="row">
            <NumberField label="From hour" value={hours[0]} onChange={(v) => setHours([v, hours[1]])} min={0} max={23} step={1} />
            <NumberField label="To hour" value={hours[1]} onChange={(v) => setHours([hours[0], v])} min={0} max={24} step={1} />
          </div>
        </div>
        <label className="field">
          <span className="field-label">Time zone</span>
          <input value={tz} onChange={(e) => setTz(e.target.value)} aria-label="Time zone" placeholder="America/New_York" />
        </label>
      </div>
      <div className="field">
        <span className="field-label">Location (default: climate.site or the data centroid)</span>
        <LatLon lat={where.lat} lon={where.lon} onChange={(lat, lon) => setPos({ lat, lon })} />
      </div>
      <StationPicker pid={pid} lat={where.lat} lon={where.lon} value={station} onChange={setStation} />
      <div className="row">
        <span className="cap">Wind</span>
        <Seg
          size="small"
          label="Wind source"
          value={wind}
          options={[
            { value: "auto", label: "auto" },
            { value: "station", label: "station" },
            { value: "era5", label: "ERA5" },
          ]}
          onChange={setWind}
        />
        <label className="row cap">
          <input type="checkbox" checked={autoLink} onChange={(e) => setAutoLink(e.target.checked)} /> Link into the config when done
        </label>
      </div>
    </InputCard>
  );
}

// ---------------------------------------------------------------- people & land cover

function LayersCard() {
  const c = useCfg();
  const pid = c.pid;
  const inputs = useInputs(pid);
  const job = useInputJob(pid, "layers");
  const l = inputs.data?.layers ?? null;
  const done = job?.status === "succeeded";
  const view = useInputView(pid, "layers", !!l || done);
  const crs = getString(c.draft.raw, "data.crs") ?? getString(c.draft.effective, "data.crs");
  const path = l?.path ?? str(job?.result?.path);
  const totals = view.data?.totals ?? {};
  const people = (k: string) => (typeof totals[k] === "number" ? (totals[k] as number) : null);
  const lc = Object.entries(totals).filter(([k, v]) => k.startsWith("lc_") && typeof v === "number") as [string, number][];
  const check = c.draft.check;
  const token = check?.preview_token || null;
  const grid = usePreviewGrid(pid, token);
  const groups = useMemo(() => {
    if (!check) return [];
    const want = (n: string) => n.startsWith("people") || n.startsWith("lc_");
    return previewGroups(check, c.draft.raw, null)
      .map((g) => ({ ...g, layers: g.layers.filter((x) => want(x.key)) }))
      .filter((g) => g.layers.length);
  }, [check, c.draft.raw]);
  const load = useMemo(() => (token ? previewLoader(pid, token) : null), [pid, token]);
  return (
    <InputCard
      kind="layers"
      title="People & land cover"
      eyebrow="HRSL + WorldCover"
      job={job}
      status={l ? <StatusChip status={l.linked ? "done" : "present"} text={l.linked ? "linked" : "fetched"} /> : <StatusChip status="missing" />}
      startLabel="Fetch people & land cover"
      disabledReason={crs ? null : "Needs data.crs (Data step) to place the layers on the grid."}
      start={() => startLayers(pid, { link: false })}
      results={
        l || done ? (
          <div className="stack" style={{ gap: 8 }}>
            <KpiRow label="Residents">
              <Kpi label="Residents" value={fmtInt(people("people") ?? l?.people_total ?? null)} note="HRSL population on the grid" />
              <Kpi label="Aged 60+" value={fmtInt(people("people_60_plus"))} />
              <Kpi label="Under 5" value={fmtInt(people("people_under_5"))} />
              <Kpi label="Cells" value={fmtInt(l?.n ?? null)} />
            </KpiRow>
            {lc.length ? (
              <Bars
                title="Land cover (mean share of a cell)"
                categories={lc.map(([k]) => k.replace(/^lc_/, ""))}
                series={[{ id: "share", label: "Mean share", values: lc.map(([, v]) => v * (v <= 1 ? 100 : 1)) }]}
                valueLabel="Mean share"
                unit="%"
                orientation="h"
                decimals={1}
                pin={false}
              />
            ) : null}
            {grid.data && load && groups.length ? (
              <MapView grid={grid.data} groups={groups} loadLayer={load} height={300} title="People and land cover preview" tools={["pan"]} />
            ) : (
              <p className="cap">The preview map appears here once the data check includes the layer columns.</p>
            )}
            <div className="row">
              <LinkButton pid={pid} kind="layers" path={path} linked={l?.linked} />
            </div>
          </div>
        ) : null
      }
    >
      <p className="cap">Residents (all, 60+, under 5) from the High Resolution Settlement Layer and WorldCover land-cover fractions, aggregated to the data grid. Used by the planner pack and the people objective.</p>
    </InputCard>
  );
}

// ---------------------------------------------------------------- open predictors

function ScatterCharts({ bins }: { bins: ScatterBins[] | ScatterBins | null | undefined }) {
  const list = !bins ? [] : Array.isArray(bins) ? bins : [bins];
  if (!list.length) return null;
  return (
    <div className="grid2">
      {list.map((b, i) => (
        <HexbinScatter
          key={i}
          title={b.role ? `${b.role}: city vs open` : "City vs open layer"}
          bins={{ x_edges: b.x_edges, y_edges: b.y_edges, counts: b.counts }}
          xLabel={b.city_column ?? "city layer"}
          yLabel={b.open_column ?? "open layer"}
          diagonal
          pin={false}
          height={240}
        />
      ))}
    </div>
  );
}

function AgreementTable({ rows }: { rows: Record<string, unknown>[] }) {
  if (!rows.length) return <p className="cap">No agreement table (bootstrap mode, or no roles mapped).</p>;
  const keys = [...new Set(rows.flatMap((r) => Object.keys(r)))];
  return (
    <table className="tbl" aria-label="Agreement with the city's layers">
      <thead>
        <tr>
          {keys.map((k) => (
            <th key={k} scope="col">
              {k.replace(/_/g, " ")}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((r, i) => (
          <tr key={i}>
            {keys.map((k) => (
              <td key={k} className={typeof r[k] === "number" ? "r num" : undefined}>
                {typeof r[k] === "number" ? fmtSig(r[k] as number, 3) : r[k] === null || r[k] === undefined ? "—" : String(r[k])}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function FeaturesCard() {
  const c = useCfg();
  const pid = c.pid;
  const inputs = useInputs(pid);
  const job = useInputJob(pid, "features");
  const pv = c.value("predictors");
  const bootstrap = !Array.isArray(pv) || pv.length === 0;
  const rolesMapped = Object.values(getRecord<unknown>(c.draft.raw, "physics.roles")).some((v) => typeof v === "string" && v);
  const [months, setMonths] = useState("");
  const [cloud, setCloud] = useState<number | null>(20);
  const [tiles, setTiles] = useState("");
  const [target, setTarget] = useState<FeaturesTarget | null>(null);
  const tgt: FeaturesTarget = target ?? (bootstrap ? "this_project" : "new_project");
  const fs = inputs.data?.features ?? null;
  const done = job?.status === "succeeded";
  const view = useInputView(pid, "features", !!fs || done);
  const crs = getString(c.draft.raw, "data.crs") ?? getString(c.draft.effective, "data.crs");
  const path = fs?.path ?? str(job?.result?.path);
  const openProject = str(job?.result?.open_project_id);
  const list = (s: string) =>
    s
      .split(",")
      .map((x) => x.trim())
      .filter(Boolean);
  return (
    <InputCard
      kind="features"
      title="Open predictors"
      eyebrow="Sentinel-2 + open layers"
      job={job}
      status={fs ? <StatusChip status={fs.linked ? "done" : "present"} text={fs.linked ? "linked" : "fetched"} /> : <StatusChip status="missing" />}
      startLabel="Build open predictors"
      disabledReason={crs ? null : "Needs data.crs (Data step)."}
      start={() => startFeatures(pid, { ...(list(months).length ? { months: list(months) } : {}), ...(cloud !== null ? { max_cloud: cloud } : {}), ...(list(tiles).length ? { s2_tiles: list(tiles) } : {}), target: tgt })}
      results={
        fs || done ? (
          <div className="stack" style={{ gap: 8 }}>
            {openProject ? (
              <p>
                Created <Link to={`/p/${encodeURIComponent(openProject)}`}>the _open project</Link>.
              </p>
            ) : null}
            {rolesMapped ? (
              view.data ? (
                <>
                  <AgreementTable rows={view.data.agreement} />
                  <ScatterCharts bins={view.data.scatter_bins} />
                </>
              ) : view.error ? (
                <span className="cap">{errorMessage(view.error)}</span>
              ) : null
            ) : (
              <p className="cap">Agreement with the city's own layers is shown once physics roles are mapped.</p>
            )}
            <div className="row">
              <LinkButton pid={pid} kind="features_join" path={path} linked={fs?.linked} label={fs?.linked ? "Linked: show the config diff" : "Use as this project's predictors"} />
              {!openProject ? <LinkButton pid={pid} kind="features_new_project" path={path} label="Create _open project" /> : null}
            </div>
          </div>
        ) : null
      }
    >
      {bootstrap ? (
        <div className="callout" data-tone="info">
          <Badge tone="accent">Bootstrap</Badge> This project has no predictors yet. The job builds the grid from target, id, x/y and the CRS only, and writes the open predictors into this project.
        </div>
      ) : null}
      <div className="grid3">
        <label className="field">
          <span className="field-label">Months (YYYY-MM, comma separated)</span>
          <input value={months} onChange={(e) => setMonths(e.target.value)} placeholder="2020-06, 2020-07, 2020-08" aria-label="Months" />
        </label>
        <div className="field">
          <span className="field-label">Maximum cloud cover</span>
          <NumberField label="Maximum cloud cover" value={cloud} onChange={setCloud} min={0} max={100} unit="%" nullable />
        </div>
        <label className="field">
          <span className="field-label">Sentinel-2 tiles (optional)</span>
          <input value={tiles} onChange={(e) => setTiles(e.target.value)} placeholder="19TCG" aria-label="Sentinel-2 tiles" />
        </label>
      </div>
      <div className="row">
        <span className="cap">Write to</span>
        <Seg
          size="small"
          label="Where the open predictors go"
          value={tgt}
          options={[
            { value: "this_project", label: "this project" },
            { value: "new_project", label: "a new _open project" },
          ]}
          onChange={setTarget}
        />
      </div>
    </InputCard>
  );
}

// ---------------------------------------------------------------- CMIP6

type PeriodRow = { name: string; from: number | null; to: number | null };

function Cmip6Card() {
  const c = useCfg();
  const pid = c.pid;
  const inputs = useInputs(pid);
  const job = useInputJob(pid, "cmip6");
  const centroid = useCentroid(pid, c.draft.check?.preview_token || null);
  const site = defaultSite(c.draft.raw, centroid);
  const [pos, setPos] = useState<{ lat: number | null; lon: number | null } | null>(null);
  const [ssps, setSsps] = useState<string[]>(SSPS.map((s) => s.value));
  const [periods, setPeriods] = useState<PeriodRow[]>(Object.entries(DEFAULT_PERIODS).map(([name, [a, b]]) => ({ name, from: a, to: b })));
  const [baseline, setBaseline] = useState<[number | null, number | null]>([1995, 2014]);
  const [months, setMonths] = useState<string[]>(["6", "7", "8"]);
  const [variable, setVariable] = useState<"tasmax" | "tas">("tasmax");
  const [models, setModels] = useState("");
  const [workers, setWorkers] = useState<number | null>(4);
  const where = pos ?? site;
  const cl = inputs.data?.climate ?? null;
  const done = job?.status === "succeeded";
  const view = useInputView(pid, "climate", !!cl || done);
  const path = cl?.path ?? str(job?.result?.path);
  const skipped = Array.isArray(job?.result?.skipped_models) ? (job!.result!.skipped_models as unknown[]).map(String) : [];
  const validPeriods = periods.filter((p) => p.name && p.from !== null && p.to !== null && p.to >= p.from);
  const isDefault =
    validPeriods.length === 3 && validPeriods.every((p) => DEFAULT_PERIODS[p.name] && DEFAULT_PERIODS[p.name][0] === p.from && DEFAULT_PERIODS[p.name][1] === p.to);
  return (
    <InputCard
      kind="cmip6"
      title="CMIP6 change factors"
      eyebrow="Warming per SSP and period"
      job={job}
      status={cl ? <StatusChip status={cl.linked ? "done" : "present"} text={cl.linked ? "linked" : "fetched"} /> : <StatusChip status="missing" />}
      startLabel="Fetch change factors"
      disabledReason={!ssps.length ? "Choose at least one SSP." : !validPeriods.length ? "Add a period." : where.lat === null && !getString(c.draft.raw, "data.crs") ? "Set a site (or data.crs for the centroid)." : null}
      start={() =>
        startCmip6(pid, {
          ...(where.lat !== null && where.lon !== null ? { lat: where.lat, lon: where.lon } : {}),
          experiments: ssps,
          ...(isDefault ? {} : { periods: Object.fromEntries(validPeriods.map((p) => [p.name, [p.from!, p.to!] as [number, number]])) }),
          baseline: [baseline[0] ?? 1995, baseline[1] ?? 2014],
          months: months.map(Number).sort((a, b) => a - b),
          variable,
          ...(models.trim() ? { models: models.split(",").map((m) => m.trim()).filter(Boolean) } : {}),
          workers: workers ?? 4,
          link: false,
        })
      }
      results={
        cl || done ? (
          <div className="stack" style={{ gap: 8 }}>
            {cl ? (
              <p className="cap">
                {fmtInt(cl.n_models)} models · {cl.experiments.join(", ")} · {cl.periods.join(", ")}
              </p>
            ) : null}
            {skipped.length ? <p className="cap">Skipped models (no data): {skipped.join(", ")}</p> : null}
            {view.data ? (
              <table className="tbl" aria-label="Warming by SSP and period">
                <thead>
                  <tr>
                    <th scope="col">SSP</th>
                    <th scope="col">Period</th>
                    <th scope="col" className="r">
                      Median warming
                    </th>
                    <th scope="col" className="r">
                      10–90% of models
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {view.data.summary.map((r) => (
                    <tr key={`${r.experiment}:${r.period}`}>
                      <td>{SSPS.find((s) => s.value === r.experiment)?.label ?? r.experiment}</td>
                      <td>{r.period}</td>
                      <td className="r num">+{fmtNum(r.median, 2)} K</td>
                      <td className="r num">
                        {fmtNum(r.p10, 2)}–{fmtNum(r.p90, 2)} K
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : view.error ? (
              <span className="cap">{errorMessage(view.error)}</span>
            ) : null}
            <div className="row">
              <LinkButton pid={pid} kind="climate" path={path} linked={cl?.linked} />
              {!isDefault ? <span className="cap">Non-default periods: linking also writes climate.periods.</span> : null}
            </div>
          </div>
        ) : null
      }
    >
      <div className="field">
        <span className="field-label">Site (default: climate.site or the data centroid)</span>
        <LatLon lat={where.lat} lon={where.lon} onChange={(lat, lon) => setPos({ lat, lon })} />
      </div>
      <div className="field">
        <span className="field-label">SSPs</span>
        <Chips label="SSPs" items={SSPS} selected={ssps} onToggle={(v, on) => setSsps(on ? SSPS.map((s) => s.value).filter((x) => x === v || ssps.includes(x)) : ssps.filter((x) => x !== v))} />
      </div>
      <div className="field">
        <span className="field-label">Periods</span>
        {periods.map((p, i) => (
          <div key={i} className="row">
            <input aria-label={`Period ${i + 1} name`} value={p.name} onChange={(e) => setPeriods(periods.map((x, k) => (k === i ? { ...x, name: e.target.value } : x)))} style={{ width: "9em" }} />
            <NumberField label={`Period ${i + 1} first year`} value={p.from} onChange={(v) => setPeriods(periods.map((x, k) => (k === i ? { ...x, from: v } : x)))} step={1} />
            <NumberField label={`Period ${i + 1} last year`} value={p.to} onChange={(v) => setPeriods(periods.map((x, k) => (k === i ? { ...x, to: v } : x)))} step={1} />
            <Button size="small" variant="ghost" icon="x" onClick={() => setPeriods(periods.filter((_, k) => k !== i))}>
              Remove
            </Button>
          </div>
        ))}
        <div className="row">
          <Button size="small" icon="plus" onClick={() => setPeriods([...periods, { name: "2061-2080", from: 2061, to: 2080 }])}>
            Add a period
          </Button>
        </div>
      </div>
      <div className="grid3">
        <div className="field">
          <span className="field-label">Baseline</span>
          <div className="row">
            <NumberField label="Baseline first year" value={baseline[0]} onChange={(v) => setBaseline([v, baseline[1]])} step={1} />
            <NumberField label="Baseline last year" value={baseline[1]} onChange={(v) => setBaseline([baseline[0], v])} step={1} />
          </div>
        </div>
        <div className="field">
          <span className="field-label">Variable</span>
          <Seg
            size="small"
            label="CMIP6 variable"
            value={variable}
            options={[
              { value: "tasmax", label: "tasmax" },
              { value: "tas", label: "tas" },
            ]}
            onChange={setVariable}
          />
        </div>
        <div className="field">
          <span className="field-label">Parallel downloads</span>
          <NumberField label="Workers" value={workers} onChange={setWorkers} min={1} max={16} step={1} />
        </div>
      </div>
      <div className="field">
        <span className="field-label">Months</span>
        <Chips
          label="Months"
          items={["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"].map((m, i) => ({ value: String(i + 1), label: m }))}
          selected={months}
          onToggle={(m, on) => setMonths(on ? [...months, m] : months.filter((x) => x !== m))}
        />
      </div>
      <label className="field">
        <span className="field-label">Models (optional, comma separated; empty = every model with data)</span>
        <input value={models} onChange={(e) => setModels(e.target.value)} placeholder="ACCESS-CM2, MIROC6" aria-label="Models" />
      </label>
    </InputCard>
  );
}

// ---------------------------------------------------------------- step

function Summary() {
  const c = useCfg();
  const inputs = useInputs(c.pid);
  const s = inputs.data;
  const item = (label: string, present: boolean, linked: boolean | undefined, detail: string) => (
    <li>
      <StatusChip status={present ? (linked ? "done" : "present") : "missing"} text={present ? (linked ? "linked" : "fetched") : "missing"} /> <strong>{label}</strong>{" "}
      <span className="cap">{detail}</span>
    </li>
  );
  if (!s) return null;
  return (
    <ul className="inputs-summary" aria-label="Inputs">
      {item("Campaign forcing", !!s.forcing, s.forcing?.linked, s.forcing ? `${s.forcing.date} · ${s.forcing.path}` : "")}
      {item("People & land cover", !!s.layers, s.layers?.linked, s.layers ? `${fmtInt(s.layers.people_total)} residents` : "")}
      {item("Open predictors", !!s.features, s.features?.linked, s.features?.path ?? "")}
      {item("CMIP6 change factors", !!s.climate, s.climate?.linked, s.climate ? `${fmtInt(s.climate.n_models)} models` : "")}
      {s.ghcn ? item("GHCN daily", true, true, `${s.ghcn.station} · ${s.ghcn.years.join("–")}`) : null}
    </ul>
  );
}

export function InputsStep() {
  const c = useCfg();
  const check = c.draft.check;
  return (
    <div className="stack setup-step" data-step="inputs">
      <p className="cap">
        Each input is a tracked network job: it keeps running if you leave the page, and its progress is in the job tray. Every result ends with <b>Link into config</b>, which shows the YAML
        change first.
        {check ? "" : " Run the data check (Data step) first so the location defaults to the data centroid."}
      </p>
      <Summary />
      <div className="inputs-grid">
        <ForcingCard />
        <LayersCard />
        <FeaturesCard />
        <Cmip6Card />
      </div>
    </div>
  );
}
