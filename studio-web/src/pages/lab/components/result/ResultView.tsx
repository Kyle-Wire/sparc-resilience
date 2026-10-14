// The full-model result on the Design page (phase 3), full width under the panes: the AnswerCard,
// then one accordion where each item is opened on its own and nothing repeats:
//   Where it cools (map layer toggles + reach rings + by area; the ONLY "cooling outside your
//   changes" figure; a whole-city change says in one line that there is no outside) · Heat risk & people (today) · Against future warming (the futures) · What was
//   actually changed (the ONLY realised vs requested, places held back) · How sure are we (quick vs
//   full gap, uncertainty layers, cross-run check, cause-and-effect badge → /r/:rid/trust) ·
//   Technical details (SE, folds, method, knock-on effects, every area).
// No KPI row; the cost shows once, in the card. A section's content is built the first time it is
// opened and kept after. Impacts (heat risk, exposure, equity, hot days, climate offset) are worked
// out once, on their own, when a result arrives without them (POST /api/results/{id}/impacts, then
// the cached result is updated); the card and the two sections share that one request's state, and
// only an error offers a button (try again).
import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { ApiError, errorMessage } from "../../../../api/client";
import { postImpacts, useLevers, useResult, type Lever, type Result } from "../../../../api/lab";
import { mutate } from "../../../../api/resource";
import { IntervalStack, RingProfile, type IntervalLayer, type IntervalRow } from "../../../../charts";
import { Button } from "../../../../components/ui/Button";
import { ActionButton, EmptyState } from "../../../../components/ui/EmptyState";
import { Icon } from "../../../../components/ui/Icon";
import { Table } from "../../../../components/ui/Table";
import { Link, runHref } from "../../../../router";
import { fmtDateTime, fmtInt, fmtNum, fmtPct, fmtSigned, fmtValue, unitLabel } from "../../../../theme/format";
import { futureCaseLabel, futureLabel, leverUnitWord, leverWords } from "../../model/words";
import { AcrossRunsPanel } from "../AcrossRuns";
import {
  ClimateOffsets,
  EquityBars,
  ExposureTable,
  HeatRiskTable,
  HotDays,
  UNCERTAINTY_PLAIN,
  isFutureCase,
  spillRings,
  uncertaintyLayers,
  type ResultInspectorProps,
} from "../ResultInspector";
import { AnswerCard, RESULT_WORDS, cardShowsExtrapolation, editsCoverCity, mainLikely, revealAnswerCard, showLayer, type ImpactsState } from "./AnswerCard";
import "./result.css";

/**
 * Same props as the old inspector: `rid`, `resId` (the exact result), `pid` (decision pack; null
 * disables it), `unit` (target unit), `nFolds` (technical details), `onShowLayer` (map layer
 * toggles: `res:${resId}:delta | delta_sd | extrapolation | realized_<var>`).
 * DOM contract (e2e): `.plain-result .headline` under `.result-inspector` carries the plain headline;
 * the DEMO badge is a `.badge`; each accordion item is `details.rv-section[data-section=<id>]` with
 * ids where | heat | future | changed | sure | tech (its body renders on first open).
 * - `nCells` (optional): the run's cell count (a change of the whole city reads as one in the card).
 * - `reveal` (optional): the result just finished; once loaded, the answer card scrolls into view.
 */
export type ResultViewProps = ResultInspectorProps & { nCells?: number | null; reveal?: boolean };

export type ResultSectionId = "where" | "heat" | "future" | "changed" | "sure" | "tech";

export function ResultView(p: ResultViewProps) {
  // a new result starts with every section closed and no impacts request in flight
  return <ResultBody key={p.resId} {...p} />;
}

/**
 * Working out the impacts; the cached result gains them, so every reader updates. `needed` (a loaded
 * result without impacts) starts it once on its own: never again for this result after that, and an
 * error waits for the reader's "Try again" (`compute`). A run without resident data (`needs_layers`)
 * is `noData`: nothing to count, nothing to retry.
 */
function useImpacts(resId: string, needed: boolean): ImpactsState {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const started = useRef(false);
  const compute = useCallback(async () => {
    started.current = true;
    setBusy(true);
    try {
      const impacts = await postImpacts(resId, {});
      mutate(`result:${resId}`, (prev: Result | undefined) => (prev ? { ...prev, impacts } : prev) as Result);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
    } finally {
      setBusy(false);
    }
  }, [resId]);
  useEffect(() => {
    if (needed && !started.current) void compute();
  }, [needed, compute]);
  const noData = error instanceof ApiError && error.code === "needs_layers";
  return { busy, error, compute: () => void compute(), noData };
}

function ResultBody({ rid, resId, pid, unit, nFolds, onShowLayer, nCells, reveal }: ResultViewProps) {
  const res = useResult(resId, rid);
  const levers = useLevers(rid).data;
  const impacts = useImpacts(resId, !!res.data && !res.data.impacts);
  const [open, setOpen] = useState<Partial<Record<ResultSectionId, boolean>>>({});
  const [seen, setSeen] = useState<Partial<Record<ResultSectionId, boolean>>>({});
  // A section opened from the card ("How sure are we") is scrolled to once it has rendered.
  const [scrollTo, setScrollTo] = useState<ResultSectionId | null>(null);
  const loaded = !!res.data;
  // A result that just finished comes into view once it has loaded (once).
  const revealed = useRef(false);
  useEffect(() => {
    if (!reveal || !loaded || revealed.current) return;
    revealed.current = revealAnswerCard();
  }, [reveal, loaded]);
  useEffect(() => {
    if (!scrollTo) return;
    const el = typeof document !== "undefined" ? document.querySelector<HTMLElement>(`details.rv-section[data-section="${scrollTo}"]`) : null;
    if (el && typeof el.scrollIntoView === "function") el.scrollIntoView({ block: "start", behavior: "smooth" });
    el?.querySelector<HTMLElement>("summary")?.focus({ preventScroll: true });
    setScrollTo(null);
  }, [scrollTo]);
  if (res.error && !res.data) {
    return (
      <section className="rv result-inspector stack" aria-label="Full model result">
        <EmptyState error={res.error} />
      </section>
    );
  }
  const r = res.data;
  if (!r) return <p className="cap">Loading the result…</p>;
  const toggle = (id: ResultSectionId) => {
    const next = !open[id];
    setOpen((o) => ({ ...o, [id]: next }));
    if (next) setSeen((s) => ({ ...s, [id]: true }));
  };
  const openSection = (id: ResultSectionId) => {
    setOpen((o) => ({ ...o, [id]: true }));
    setSeen((s) => ({ ...s, [id]: true }));
    setScrollTo(id);
  };
  const section = (id: ResultSectionId, title: string, teaser: string, body: () => ReactNode) => (
    <details className="rv-section" data-section={id} open={!!open[id]}>
      <summary
        onClick={(e) => {
          e.preventDefault();
          toggle(id);
        }}
      >
        <span className="rv-section-title">{title}</span>
        <span className="rv-section-teaser">{teaser}</span>
        <Icon name="chevronDown" className="rv-section-chev" />
      </summary>
      {seen[id] ? <div className="rv-section-body">{body()}</div> : null}
    </details>
  );
  return (
    <section className="rv result-inspector stack" aria-label="Full model result">
      <AnswerCard rid={rid} resId={resId} pid={pid} result={r} unit={unit} onShowLayer={onShowLayer} impacts={impacts} nCells={nCells} onOpenSection={openSection} />
      <div className="rv-sections">
        {section("where", RESULT_WORDS.where, RESULT_WORDS.whereTeaser, () => (
          <WhereItCools r={r} resId={resId} unit={unit} levers={levers} onShowLayer={onShowLayer} nCells={nCells} />
        ))}
        {section("heat", RESULT_WORDS.heat, RESULT_WORDS.heatTeaser, () => (
          <HeatAndPeople r={r} unit={unit} impacts={impacts} />
        ))}
        {section("future", RESULT_WORDS.future, RESULT_WORDS.futureTeaser, () => (
          <FutureWarming r={r} unit={unit} impacts={impacts} />
        ))}
        {section("changed", RESULT_WORDS.changed, RESULT_WORDS.changedTeaser, () => (
          <WhatChanged r={r} levers={levers} />
        ))}
        {section("sure", RESULT_WORDS.sure, RESULT_WORDS.sureTeaser, () => (
          <HowSure r={r} rid={rid} pid={pid} unit={unit} />
        ))}
        {section("tech", RESULT_WORDS.tech, RESULT_WORDS.techTeaser, () => (
          <TechnicalDetails r={r} resId={resId} unit={unit} nFolds={nFolds} nCells={nCells} />
        ))}
      </div>
    </section>
  );
}

// ---------------------------------------------------------------- Where it cools

const leverOf = (levers: readonly Lever[] | undefined, v: string) => levers?.find((l) => l.var === v);
const areaName = (name: string) => (/^zone /i.test(name) ? `Zone ${name.slice(5)}` : name);

/** The ring chart's words: the changed places and the rings around them, the likely range as lines. */
const RING_WORDS = {
  inside: "your changes",
  axis: "Distance from your changes",
  legend: "dark bar = places you changed · lighter bars = rings around them · lines = likely range",
  /** Instead of the chart when the changes cover the whole city (it would be one bar, the card's number). */
  wholeCity: "Your changes cover the whole city, so there is nowhere outside them for the cooling to reach.",
} as const;

function WhereItCools({ r, resId, unit, levers, onShowLayer, nCells }: { r: Result; resId: string; unit: string; levers: Lever[] | undefined; onShowLayer?: (key: string) => void; nCells?: number | null }) {
  const u = unitLabel(unit);
  const share = r.spill.outside_share;
  const effect = mainLikely(r).estimate > 0 ? "warming" : "cooling";
  const reach = Object.entries(r.spill.lever_ranges).filter(([, m]) => Number.isFinite(m));
  // named areas only: the changed places and the whole city are the card's numbers, the rings the chart's
  const areas = r.regions.filter((g) => !g.auto || /^zone /i.test(g.name));
  // a change of the whole city has no "outside": the share would say nothing
  const wholeCity = editsCoverCity(r, nCells);
  return (
    <>
      {wholeCity ? null : (
        <p className="rv-lead" data-outside-share={share ?? undefined}>
          {share !== null && Number.isFinite(share) ? (
            <>
              About <strong>{fmtPct(share)}</strong> of the {effect} lands outside your changes, in the places around them.
            </>
          ) : (
            `There is no ${effect} to split between your changes and the places around them.`
          )}
        </p>
      )}
      {onShowLayer ? (
        <div className="rv-layers" role="group" aria-label="Show result layers on the map">
          <span className="cap">On the map:</span>
          <Button size="small" variant="ghost" onClick={() => showLayer(onShowLayer, `res:${resId}:delta`)}>
            Temperature change
          </Button>
          <Button size="small" variant="ghost" onClick={() => showLayer(onShowLayer, `res:${resId}:delta_sd`)}>
            Spread between model runs
          </Button>
          <Button size="small" variant="ghost" onClick={() => showLayer(onShowLayer, `res:${resId}:extrapolation`)}>
            Beyond measured conditions
          </Button>
          {Object.keys(r.realized).map((v) => (
            <Button key={v} size="small" variant="ghost" onClick={() => showLayer(onShowLayer, `res:${resId}:realized_${v}`)}>
              What changed: {leverWords(leverOf(levers, v), v).name}
            </Button>
          ))}
        </div>
      ) : null}
      {wholeCity ? (
        <p className="cap" data-reach="whole-city">
          {RING_WORDS.wholeCity}
        </p>
      ) : r.spill.rings.length > 1 ? (
        // one ring is only the changed places: a single bar that repeats the card's number
        <RingProfile
          title="How far the cooling reaches"
          units={`${u} (negative = cooler)`}
          rings={spillRings(r.spill.rings)}
          unit={u}
          valueLabel="Average change"
          insideLabel={RING_WORDS.inside}
          axisTitle={RING_WORDS.axis}
          legend={<span>{RING_WORDS.legend}</span>}
          caption={`Average change in the places you changed and in rings around them.${reach.length ? ` Each change reaches about ${reach.map(([v, m]) => `${fmtInt(m)} m (${leverWords(leverOf(levers, v), v).name.toLowerCase()})`).join(", ")}.` : ""}`}
        />
      ) : null}
      {areas.length ? (
        <Table
          title="By area"
          csvName="areas"
          rowKey={(g) => g.name}
          columns={[
            { key: "name", label: "Area", value: (g) => areaName(g.name) },
            { key: "n", label: "Places", align: "right", value: (g) => g.n_cells, render: (g) => fmtInt(g.n_cells) },
            {
              key: "mean",
              label: "Average change",
              unit: u,
              align: "right",
              value: (g) => g.mean?.estimate ?? null,
              render: (g) => (g.mean ? `${fmtSigned(g.mean.estimate, 3)}${g.mean.se !== null ? ` ± ${fmtNum(1.96 * g.mean.se, 3)}` : ""}` : "—"),
            },
            { key: "pw", label: "Felt by residents", unit: u, align: "right", value: (g) => g.people_weighted, render: (g) => fmtSigned(g.people_weighted, 3) },
            { key: "c05", label: `Cooled by ${fmtNum(0.5, 1)} ${u} or more`, align: "right", value: (g) => g.frac_cooled_05, render: (g) => fmtPct(g.frac_cooled_05) },
          ]}
          rows={areas}
        />
      ) : null}
    </>
  );
}

// ---------------------------------------------------------------- Heat risk & people · Against future warming

const IMPACTS_WORDS = {
  busy: "Working out heat risk, who benefits and future warming… (a few seconds)",
  noData: "This run has no resident data, so heat risk, who benefits and future warming cannot be worked out.",
  failed: "Could not work out heat risk and future warming",
} as const;

/**
 * The impacts are not in yet (the one request ResultView starts on its own): a status line while it
 * runs; after an error, the plain reason and the only button, "Try again" (or the server's remedy for
 * a run without resident data, which a retry would not fix).
 */
function ImpactsStatus({ state }: { state: ImpactsState }) {
  if (state.error && state.noData) {
    const action = state.error instanceof ApiError ? state.error.action : null;
    return (
      <div className="rv-prompt" data-impacts="no-data">
        <p className="cap">{IMPACTS_WORDS.noData}</p>
        {action ? <ActionButton action={action} size="small" variant="default" /> : null}
      </div>
    );
  }
  if (state.error) {
    return (
      <div className="rv-prompt" role="alert" data-impacts="error">
        <p className="cap">
          {IMPACTS_WORDS.failed}: {errorMessage(state.error)}
        </p>
        <Button size="small" icon="refresh" busy={state.busy} onClick={state.compute}>
          {RESULT_WORDS.tryAgain}
        </Button>
      </div>
    );
  }
  return (
    <div className="rv-prompt" role="status" data-impacts="busy" data-busy="true">
      <p className="cap row">
        <span className="spinner" aria-hidden="true" /> {IMPACTS_WORDS.busy}
      </p>
    </div>
  );
}

function HeatAndPeople({ r, unit, impacts }: { r: Result; unit: string; impacts: ImpactsState }) {
  const imp = r.impacts;
  if (!imp) return <ImpactsStatus state={impacts} />;
  const heatRows = imp.heat ? imp.heat.rows.filter((x) => !isFutureCase(x.case)) : [];
  const exposure = imp.exposure.filter((e) => !isFutureCase(e.case));
  const any = heatRows.length > 0 || exposure.length > 0 || Object.keys(imp.equity).length > 0 || imp.hot_days !== null || imp.hot_days_action !== null;
  return (
    <>
      {imp.heat && heatRows.length ? <HeatRiskTable heat={imp.heat} rows={heatRows} future={false} unit={unit} /> : null}
      {exposure.length ? <ExposureTable impacts={imp} rows={exposure} unit={unit} future={false} /> : null}
      <EquityBars impacts={imp} unit={unit} />
      <HotDays impacts={imp} />
      {any ? null : <p className="cap">This run has no resident or heat-risk data.</p>}
    </>
  );
}

function FutureWarming({ r, unit, impacts }: { r: Result; unit: string; impacts: ImpactsState }) {
  const imp = r.impacts;
  if (!imp) return <ImpactsStatus state={impacts} />;
  const heatRows = imp.heat ? imp.heat.rows.filter((x) => isFutureCase(x.case)) : [];
  const exposure = imp.exposure.filter((e) => isFutureCase(e.case));
  const offsets = imp.climate_offset.some((o) => typeof o.offset_share === "number" && Number.isFinite(o.offset_share));
  return (
    <>
      <ClimateOffsets impacts={imp} />
      {imp.heat && heatRows.length ? <HeatRiskTable heat={imp.heat} rows={heatRows} future unit={unit} /> : null}
      {exposure.length ? <ExposureTable impacts={imp} rows={exposure} unit={unit} future /> : null}
      {offsets || heatRows.length || exposure.length ? null : <p className="cap">This run has no future-climate pathways.</p>}
    </>
  );
}

// ---------------------------------------------------------------- What was actually changed

type Realised = [string, Result["realized"][string]];

function WhatChanged({ r, levers }: { r: Result; levers: Lever[] | undefined }) {
  const rows: Realised[] = Object.entries(r.realized);
  if (!rows.length) return <p className="cap">No changes were recorded for this result.</p>;
  const held = rows.some(([, x]) => x.clipped_share > 0);
  const amount = (v: string, x: number) => {
    const w = leverUnitWord(leverOf(levers, v));
    return `${fmtSigned(x, 2)}${w ? ` ${w}` : ""}`;
  };
  return (
    <>
      <p className="rv-lead">
        {held
          ? "In some places what you asked for does not fit: no place can go beyond what is realistic there, so the full model used what fits."
          : "Everything you asked for fits the space available."}
      </p>
      <Table
        title="What you asked for and what fits"
        csvName="realised"
        rowKey={([v]) => v}
        columns={[
          { key: "v", label: "Change", value: ([v]) => leverWords(leverOf(levers, v), v).name },
          { key: "rq", label: "You asked for (average)", align: "right", value: ([, x]) => x.requested_mean, render: ([v, x]) => amount(v, x.requested_mean) },
          { key: "rl", label: "What fits (average)", align: "right", value: ([, x]) => x.realized_mean, render: ([v, x]) => amount(v, x.realized_mean) },
          { key: "clip", label: "Places held back", align: "right", value: ([, x]) => x.clipped_share, render: ([, x]) => fmtPct(x.clipped_share) },
          { key: "rqt", label: "Asked for (total)", align: "right", value: ([, x]) => x.requested_total, render: ([, x]) => fmtNum(x.requested_total, 1) },
          { key: "rlt", label: "Fits (total)", align: "right", value: ([, x]) => x.realized_total, render: ([, x]) => fmtNum(x.realized_total, 1) },
        ]}
        rows={rows}
      />
      <p className="cap">Averages are over the places each change touches; totals add them up over those places.</p>
    </>
  );
}

// ---------------------------------------------------------------- How sure are we

/** Short row names of the doubt chart (by uncertainty layer id), so none is cut short. */
const DOUBT_ROW: Record<string, string> = {
  estimation: "Model fit",
  specification: "Model versions & runs",
  attribution: "Simulation check",
  causal: "Independent check",
  envelope: "All sources together",
};

const DOUBT_LEGEND = "one bar per source of doubt: the range it allows · tick = estimate";

/**
 * One row per source of doubt (model fit · … · independent check · all sources together), each with
 * its own range, so no range hides inside a wider one. The tick on every row is the model's citywide
 * estimate.
 */
export function doubtRows(estimate: number, layers: IntervalLayer[]): IntervalRow[] {
  return layers.map((l) => ({ id: l.id, label: DOUBT_ROW[l.id] ?? l.label, estimate, layers: [l] }));
}

function HowSure({ r, rid, pid, unit }: { r: Result; rid: string; pid: string | null; unit: string }) {
  const u = unitLabel(unit);
  const pve = r.preview_vs_exact;
  const frac = r.extrapolated_edited ?? r.summary.frac_extrapolated_edited;
  const layers = r.uncertainty ? uncertaintyLayers(r.uncertainty).map((l) => ({ ...l, label: UNCERTAINTY_PLAIN[l.id] ?? l.label })) : [];
  const cc = r.causal_check;
  return (
    <>
      <ul className="rv-list">
        {pve ? (
          <li data-preview-vs-exact="true">
            The quick estimate was off by about {fmtValue(pve.mean_abs_err, unit, 3)} per place on average ({fmtPct(pve.rel_err)} of the full model&apos;s change).
          </li>
        ) : (
          <li>There was no quick estimate to compare with the full model.</li>
        )}
        {!cardShowsExtrapolation(r) && frac !== null && Number.isFinite(frac) ? (
          <li>
            {frac > 0
              ? `${fmtPct(frac)} of the changed places go beyond the conditions the model has seen (a small share).`
              : "All the changed places stay within the conditions the model has seen."}
          </li>
        ) : null}
        {r.warnings.map((w, i) => (
          <li key={i} data-warning={w.code}>
            {w.message}
          </li>
        ))}
      </ul>
      <div className="rv-causal">
        <Link to={runHref(rid, "trust")} className="pill" data-tone={cc ? (cc.model_within ? "good" : "warn") : undefined} title="How much to trust this run">
          {cc ? (cc.model_within ? "Cause-and-effect check agrees" : "Cause-and-effect check disagrees") : "No cause-and-effect check"}
        </Link>
        <span className="cap">
          {cc
            ? `An independent check puts the citywide change between ${fmtNum(cc.lo, 2)} and ${fmtValue(cc.hi, unit, 2)}; the model's estimate is ${cc.model_within ? "inside" : "outside"} that range${cc.model_within ? "" : ", so treat the size of the effect with caution"}.`
            : "There is no independent check for these changes in this run."}
        </span>
      </div>
      {layers.length ? (
        <IntervalStack
          title="How wide the range gets with each source of doubt"
          units={`${u} (negative = cooler)`}
          rows={doubtRows(r.city.estimate, layers)}
          valueLabel="Citywide change"
          unit={u}
          legend={<span>{DOUBT_LEGEND}</span>}
          caption={r.uncertainty?.envelope_excludes_zero ? "Even with every source of doubt added up, the range stays on one side of zero." : "With every source of doubt added up, the range includes zero."}
        />
      ) : null}
      {r.scenario ? <AcrossRunsPanel pid={pid} sid={r.scenario.id} unit={unit} contentHash={typeof r.spec.content_hash === "string" ? r.spec.content_hash : null} /> : null}
    </>
  );
}

// ---------------------------------------------------------------- Technical details

/** The pathway codes behind the plain future labels ("Mid-century, moderate emissions (2041–2060): ssp245 2041-2060"). */
function futureCodes(r: Result): string[] {
  const imp = r.impacts;
  if (!imp) return [];
  const cases = new Map<string, string>();
  for (const o of imp.climate_offset) cases.set(futureLabel(o.experiment, o.period), `${o.experiment} ${o.period}`);
  for (const c of [...imp.exposure.map((e) => e.case), ...(imp.heat?.rows.map((x) => x.case) ?? [])]) {
    const label = isFutureCase(c) ? futureCaseLabel(c) : null;
    if (label && !cases.has(label)) cases.set(label, c);
  }
  return [...cases].map(([label, code]) => `${label}: ${code}`);
}

function TechnicalDetails({ r, resId, unit, nFolds, nCells }: { r: Result; resId: string; unit: string; nFolds: number | null; nCells?: number | null }) {
  const u = unitLabel(unit);
  const main = mainLikely(r);
  const edited = r.regions.find((g) => g.auto && /edited/i.test(g.name) && !/ring/i.test(g.name));
  const mediators = Object.entries(r.mediators);
  // a change of the whole city: the citywide figures and the changed places are the main ones, said once
  const wholeCity = editsCoverCity(r, nCells);
  const regions = wholeCity && edited ? r.regions.filter((g) => g !== edited) : r.regions;
  const futures = futureCodes(r);
  const rows: [string, ReactNode][] = [
    ["Standard error", main.se === null ? "not available" : fmtValue(main.se, unit, 4)],
    ...(wholeCity ? [] : ([["Citywide standard error", r.city.se === null ? "not available" : fmtValue(r.city.se, unit, 4)]] as [string, ReactNode][])),
    ...(edited ? ([["Changed places (cells)", fmtInt(edited.n_cells)]] as [string, ReactNode][]) : []),
    ...(nFolds ? ([["Folds (jackknife)", fmtNum(nFolds, 0)]] as [string, ReactNode][]) : []),
    ["Method", "Exact engine on the run's fold models. SE = std over folds of the fold means × √(K−1) (jackknife); likely range = estimate ± 1.96·SE."],
    ["Cell spread (p10 to p90)", `${fmtSigned(r.p10, 3)} to ${fmtSigned(r.p90, 3)} ${u}`],
    ["Mean per-cell fold SD", fmtNum(r.mean_delta_sd, 4)],
    ["Knock-on effects (mediators)", mediators.length ? mediators.map(([v, m]) => `${v} follows ${fmtSigned(m.mean_change, 3)}`).join(" · ") : "none"],
    ...(futures.length ? ([["Future climates (pathway codes)", futures.join(" · ")]] as [string, ReactNode][]) : []),
    ["Result", `${resId} · ${fmtDateTime(r.summary.created_utc)}`],
  ];
  return (
    <>
      <dl className="rv-tech">
        {rows.map(([k, v]) => (
          <div key={k} style={{ display: "contents" }}>
            <dt>{k}</dt>
            <dd>{v}</dd>
          </div>
        ))}
      </dl>
      {regions.length ? (
        <Table
          title="Every area (cells)"
          csvName="regions"
          rowKey={(g) => g.name}
          columns={[
            { key: "name", label: "Region", value: (g) => g.name, render: (g) => (g.auto ? <em>{g.name}</em> : g.name) },
            { key: "n", label: "Cells", align: "right", value: (g) => g.n_cells, render: (g) => fmtInt(g.n_cells) },
            { key: "mean", label: "Mean ΔT", unit: u, align: "right", value: (g) => g.mean?.estimate ?? null, render: (g) => (g.mean ? `${fmtSigned(g.mean.estimate, 3)}${g.mean.se !== null ? ` ± ${fmtNum(1.96 * g.mean.se, 3)}` : ""}` : "—") },
            { key: "pw", label: "People-weighted", unit: u, align: "right", value: (g) => g.people_weighted, render: (g) => fmtSigned(g.people_weighted, 3) },
            { key: "total", label: "Total", unit: `${u}·cells`, align: "right", value: (g) => g.total, render: (g) => fmtSigned(g.total, 1) },
            { key: "c01", label: "Cooled ≥ 0.1", align: "right", value: (g) => g.frac_cooled_01, render: (g) => fmtPct(g.frac_cooled_01) },
            { key: "c05", label: "Cooled ≥ 0.5", align: "right", value: (g) => g.frac_cooled_05, render: (g) => fmtPct(g.frac_cooled_05) },
          ]}
          rows={regions}
        />
      ) : null}
    </>
  );
}
