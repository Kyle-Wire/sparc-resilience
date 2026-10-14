// The Heat view, in two parts (phase 2):
// - Heat risk › Today (default export, the canonical home of the heat-risk baseline): how hot the
//   campaign afternoon felt (NWS heat index from each cell's temperature and the campaign
//   dewpoint), who was exposed (residents per NWS category, with the cooling package), where,
//   and the heat-index distribution. The dewpoint can be changed to ask "what if the afternoon
//   were more humid?" (the view's ?dewpoint_C=, kept in the page URL so Future climate uses it).
// - HeatFutures (Heat risk › Future climate): residents per category in each CMIP6 future (a band
//   between constant dewpoint and constant relative humidity, widened by the models' 10th–90th
//   percentile warming), with and without the package. The warming itself is the Climate view's.
// How sure each scenario's cooling is (the view's verdicts) lives on How much to trust this.
// Plain words (phase 4): the heat index is the "feels-like temperature" and the NWS categories
// are heat-risk levels, each explained in an ⓘ with its feels-like range; the dewpoint control is
// the "Humidity assumption"; futures are named by emissions and period ("Moderate emissions,
// mid-century (2041–2060)"), the pathway code only in a tooltip's technical line.
import { useState } from "react";
import type { HeatCase, HeatCategoryId, HeatSections } from "../../api/runs";
import { Bars, Histogram } from "../../charts";
import { InfoTip } from "../../components/ui/InfoTip";
import { Seg } from "../../components/ui/Seg";
import { Term } from "../../components/ui/Term";
import { heatCategory, term } from "../../content/glossary";
import { Link, runHref } from "../../router";
import { fmtInt, fmtNum } from "../../theme/format";
import { emissionsName, FutureName, heatLevelsTip, HeatLevelsTip, periodName } from "./climateWords";
import { Block, KpiTiles, Section, ViewPage, useRid } from "./common";
import { heatHref, useDewpoint } from "./HeatRisk";
import { RunLayerMap } from "./RunLayerMap";

const CATS: HeatCategoryId[] = ["below", "caution", "extreme_caution", "danger", "extreme_danger"];
const CAT_COLOR: Record<HeatCategoryId, string> = {
  below: "var(--hc0)",
  caution: "var(--hc1)",
  extreme_caution: "var(--hc2)",
  danger: "var(--hc3)",
  extreme_danger: "var(--hc4)",
};
const TONE: Record<HeatSections["brief"][number]["tone"], string | undefined> = { neutral: "info", good: "good", warn: undefined, crit: "crit" };
/** Brief items and KPIs about the futures or the verdicts: their home is Future climate or Trust, not Today. */
const NOT_TODAY_BRIEF = new Set(["future", "sure"]);
const NOT_TODAY_KPIS = new Set(["ec_future"]);
/** ⓘ after the KPI labels: the heat-risk levels for the counts, the feels-like temperature for the °F tiles. */
const KPI_TIPS = { ec_today: heatLevelsTip(), ec_pkg: heatLevelsTip(), mean_hi: { term: "heat_index" }, max_hi: { term: "heat_index" } } as const;

type Humidity = "constant_dewpoint" | "constant_rh";

/** A heat-risk level in the glossary's words (the server's label when the glossary has none). */
const levelLabel = (c: { id: HeatCategoryId; label: string }) => heatCategory(c.id)?.label ?? c.label;
/** Where Extreme caution starts (°F), from the view's categories. */
const ecFrom = (categories: HeatSections["categories"] | null | undefined) => categories?.find((c) => c.id === "extreme_caution")?.lo_F ?? null;
const people = (n: number | null | undefined, measure: "people" | "cells") =>
  n === null || n === undefined ? "—" : `${n >= 10000 ? `${fmtNum(n / 1000, 0)}k` : n >= 1000 ? `${fmtNum(n / 1000, 1)}k` : fmtInt(n)}${measure === "cells" ? " places" : ""}`;
const range = (r: [number, number] | null | undefined, measure: "people" | "cells") =>
  !r ? "—" : Math.abs(r[1] - r[0]) < Math.max(1, 0.005 * r[1]) ? people(r[0], measure) : `${people(r[0], measure)} – ${people(r[1], measure)}`;

/** A dew point in °F for the page (the API and the URL keep °C, as the heat index's input). */
export const dewF = (c: number) => (c * 9) / 5 + 32;
/** A typed dew point in °F back to the °C the view takes, rounded to 0.01 °C. */
export const dewC = (f: number) => Math.round((((f - 32) * 5) / 9) * 100) / 100;

/**
 * The humidity assumption (a dew point): the measured afternoon's, a preset, or a typed value.
 * Shown and typed in °F, like every other temperature on the page; sent to the view in °C.
 */
function HumidityControl({ h, value, onChange }: { h: HeatSections["humidity"]; value: number | null; onChange: (v: number | null) => void }) {
  const [custom, setCustom] = useState<string>(value !== null ? fmtNum(dewF(value), 0) : "");
  const opts: { value: string; label: string }[] = [];
  if (h.campaign_dewpoint_C !== null) opts.push({ value: "campaign", label: `As measured (dew point ${fmtNum(dewF(h.campaign_dewpoint_C), 0)} °F)` });
  for (const p of h.presets) opts.push({ value: String(p.dewpoint_C), label: p.label });
  const cur = value === null ? (h.campaign_dewpoint_C !== null ? "campaign" : "") : String(value);
  const apply = () => {
    const f = Number(custom);
    // −22 to 95 °F: the −30 to 35 °C the view accepts
    if (custom.trim() !== "" && Number.isFinite(f) && f >= -22 && f <= 95) onChange(dewC(f));
  };
  return (
    <div className="stack" style={{ gap: 8 }}>
      {h.needs_input ? (
        <p className="callout" role="note">
          This run has no humidity reading for the measurement day. Pick a humidity level below to work out the <Term id="heat_index">feels-like temperature</Term>; the
          result is a what-if, not a measurement.
        </p>
      ) : null}
      <div className="row" style={{ flexWrap: "wrap", gap: 8 }}>
        <Seg<string> label="Humidity assumption" size="small" value={cur} onChange={(v) => onChange(v === "campaign" ? null : Number(v))} options={opts} />
        <label className="row cap" style={{ gap: 6 }}>
          Custom
          <input
            type="number"
            step={1}
            min={-22}
            max={95}
            value={custom}
            onChange={(e) => setCustom(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") apply();
            }}
            aria-label="Custom dew point in degrees Fahrenheit"
            style={{ width: 80 }}
          />
          °F
        </label>
        <button type="button" className="btn" onClick={apply}>
          Apply
        </button>
      </div>
      <p className="cap">
        {h.dewpoint_C !== null ? (
          <>
            Dew point {fmtNum(dewF(h.dewpoint_C), 0)} °F ({h.source ?? "assumed"})
            {h.rh_range ? `, so relative humidity is ${fmtNum(h.rh_range[0], 0)}–${fmtNum(h.rh_range[1], 0)}% across the city` : ""}. One humidity value is used for every place:
            moisture varies little across a city on one afternoon.
          </>
        ) : (
          "No humidity chosen yet."
        )}
      </p>
    </div>
  );
}

function Brief({ items }: { items: NonNullable<HeatSections["brief"]> }) {
  return (
    <div className="grid2" role="list" aria-label="Heat brief">
      {items.map((b) => (
        <div key={b.id} className="callout" data-tone={TONE[b.tone]} role="listitem">
          <strong style={{ display: "block", color: "var(--ink)", marginBottom: 4 }}>{b.title}</strong>
          {b.text}
        </div>
      ))}
    </div>
  );
}

/**
 * Residents (or cells) per NWS category: today and with the package ("today"), or each future
 * of one period under one humidity assumption, with and without the package ("futures").
 */
function Exposure({ s, scope }: { s: Pick<HeatSections, "today" | "futures" | "categories">; scope: "today" | "futures" }) {
  const [hum, setHum] = useState<Humidity>("constant_dewpoint");
  const periods = [...new Set(s.futures.map((f) => f.period))];
  const [per, setPer] = useState<string>(periods.find((p) => p.startsWith("2041")) ?? periods[Math.floor(periods.length / 2)] ?? "");
  const t = s.today;
  const rows: { label: string; c: HeatCase }[] = [];
  if (scope === "today") {
    rows.push({ label: "Today", c: t.unadapted });
    if (t.adapted) rows.push({ label: "Today · with the cooling package", c: t.adapted });
  } else {
    // the period is picked above the chart, so each row names its emissions future only
    for (const f of s.futures.filter((x) => x.period === per)) {
      const name = emissionsName(f.experiment, f.label);
      rows.push({ label: name, c: f.unadapted[hum] });
      if (f.adapted) rows.push({ label: `${name} · with the cooling package`, c: f.adapted[hum] });
    }
  }
  const labelOf = (id: HeatCategoryId) => {
    const c = s.categories.find((x) => x.id === id);
    return c ? levelLabel(c) : heatCategory(id)?.label ?? id;
  };
  const measure = t.measure === "people" ? "Residents" : "Places";
  const ec = t.unadapted.ec_or_worse;
  const ecF = ecFrom(s.categories);
  return (
    <Bars
      title={scope === "today" ? `${measure} by heat-risk category` : `${measure} by heat-risk category in each future`}
      units={t.measure === "people" ? "residents" : "places"}
      categories={rows.map((r) => r.label)}
      orientation="h"
      mode="stacked"
      series={CATS.map((id) => ({ id, label: labelOf(id), values: rows.map((r) => r.c.counts[id] ?? 0), color: CAT_COLOR[id] }))}
      valueLabel={measure}
      unit={t.measure === "people" ? "people" : "places"}
      decimals={0}
      highlight={scope === "today" ? ["Today"] : undefined}
      note={
        <>
          What the levels mean
          <HeatLevelsTip label="heat-risk levels" />
        </>
      }
      actions={
        scope === "futures" ? (
          <>
            {periods.length > 1 ? <Seg<string> label="Time period" size="small" value={per} onChange={setPer} options={periods.map((p) => ({ value: p, label: periodName(p) }))} /> : null}
            <Seg<Humidity>
              label="Future humidity"
              size="small"
              value={hum}
              onChange={setHum}
              options={[
                { value: "constant_dewpoint", label: "Same moisture (lower end)" },
                { value: "constant_rh", label: "Same relative humidity (upper end)" },
              ]}
            />
            <InfoTip term="future_humidity" />
          </>
        ) : undefined
      }
      caption={
        scope === "today"
          ? `Today ${people(ec, t.measure)} ${t.measure === "people" ? "residents are" : "are"} at Extreme caution or worse${ecF !== null ? ` (feels like ${fmtNum(ecF, 0)} °F or more)` : ""}${
              t.adapted ? `; with the cooling package, ${people(t.adapted.ec_or_worse, t.measure)}` : ""
            }.`
          : `Each future adds the climate models' middle estimate of warming by ${periodName(per).toLowerCase()} to the measured afternoon. Switch Future humidity to see the other end of the range.`
      }
    />
  );
}

function Distribution({ hist, categories }: { hist: NonNullable<HeatSections["hist"]>; categories: HeatSections["categories"] }) {
  const [which, setWhich] = useState<"today" | "pkg">("today");
  const counts = which === "pkg" && hist.adapted_counts ? hist.adapted_counts : hist.counts;
  const lo = hist.edges[0];
  const hi = hist.edges[hist.edges.length - 1];
  const refs = categories.filter((c) => c.lo_F !== null && c.lo_F >= lo && c.lo_F <= hi).map((c) => ({ value: c.lo_F as number, label: levelLabel(c) }));
  const starts = categories.filter((c) => c.lo_F !== null).map((c) => `${levelLabel(c)} ${fmtNum(c.lo_F, 0)} °F`);
  return (
    <Histogram
      title="Feels-like temperature across the city"
      note={
        <>
          The heat index, in °F
          <InfoTip term="heat_index" />
        </>
      }
      units={hist.measure === "people" ? "residents per band" : "places per band"}
      bins={{ edges: hist.edges, counts }}
      xLabel="Feels-like temperature"
      unit="°F"
      countLabel={hist.measure === "people" ? "Residents" : "Places"}
      decimals={0}
      refLines={refs}
      actions={
        hist.adapted_counts ? (
          <Seg<"today" | "pkg">
            label="Case"
            size="small"
            value={which}
            onChange={setWhich}
            options={[
              { value: "today", label: "Today" },
              { value: "pkg", label: "With the cooling package" },
            ]}
          />
        ) : undefined
      }
      caption={`Lines mark where each heat-risk level starts${starts.length ? ` (${starts.join(", ")})` : ""}.`}
    />
  );
}

/** Extreme caution+ and Danger+ in each future (ranges); today's counts are on Today, the warming in the Warming chart. */
function Futures({ s }: { s: Pick<HeatSections, "today" | "futures"> }) {
  const m = s.today.measure;
  return (
    <div style={{ overflowX: "auto" }}>
      <table className="tbl" aria-label="Exposure in each climate future">
        <thead>
          <tr>
            <th scope="col">Future climate</th>
            <th scope="col" className="r">
              Extreme caution or worse
            </th>
            <th scope="col" className="r">
              Danger or worse
            </th>
            {s.today.adapted ? (
              <>
                <th scope="col" className="r">
                  With the cooling package: Extreme caution or worse
                </th>
                <th scope="col" className="r">
                  With the cooling package: Danger or worse
                </th>
              </>
            ) : null}
          </tr>
        </thead>
        <tbody>
          {s.futures.map((f) => (
            <tr key={f.id}>
              <td>
                <FutureName experiment={f.experiment} period={f.period} fallback={f.label} />
              </td>
              <td className="r">{range(f.unadapted.ec_range_full, m)}</td>
              <td className="r">{range(f.unadapted.danger_range_full, m)}</td>
              {s.today.adapted ? (
                <>
                  <td className="r">{range(f.adapted?.ec_range_full, m)}</td>
                  <td className="r">{range(f.adapted?.danger_range_full, m)}</td>
                </>
              ) : null}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="cap">
        Each range covers most <Term id="climate_models">climate models</Term> (from the 10th to the 90th percentile of their warming) and both{" "}
        <Term id="future_humidity">future humidity</Term> cases: the air holds the same amount of water as today (the low end) or takes up more water as it warms and stays as muggy as today (the high end). The feels-like
        temperature climbs steeply near each level&apos;s edge, so a few tenths of a degree can move many residents across it.
      </p>
    </div>
  );
}

function Method({ s }: { s: Pick<HeatSections, "categories" | "humidity"> }) {
  return (
    <div className="stack" style={{ gap: 8 }}>
      <p className="cap prose">
        Heat risk uses the <Term id="heat_index" technical={s.humidity.method}>feels-like temperature</Term>, which combines air temperature and humidity the way the US
        National Weather Service does. Air temperature alone understates danger: humidity slows sweating, so the same air temperature feels hotter on a muggier day. For
        future years we show a range: at the low end the air holds the same amount of water as today, so it feels a little less muggy as it warms; at the high end it
        takes up more water as it warms and stays as muggy as today, so the heat feels worse.
        <InfoTip term="future_humidity" />
      </p>
      <ul className="stack" style={{ listStyle: "none", padding: 0, margin: 0, gap: 4 }} aria-label="Heat-risk levels">
        {s.categories
          .filter((c) => c.id !== "below")
          .map((c) => (
            <li key={c.id} className="row" style={{ gap: 8, alignItems: "baseline" }}>
              <span aria-hidden="true" style={{ display: "inline-block", width: 12, height: 12, borderRadius: 3, background: CAT_COLOR[c.id] }} />
              <strong>{levelLabel(c)}</strong>
              <span className="cap">
                feels like {c.hi_F === null ? `${fmtNum(c.lo_F, 0)} °F or more` : `${fmtNum(c.lo_F, 0)}–${fmtNum(c.hi_F, 0)} °F`}: {c.note}
              </span>
            </li>
          ))}
      </ul>
    </div>
  );
}

/** Heat risk › Today: the heat-risk baseline of the campaign afternoon. */
export default function Heat() {
  const rid = useRid();
  const [dewpoint, setDewpoint] = useDewpoint();
  return (
    <ViewPage
      view="heat"
      title="Heat risk today"
      intro={
        <>
          How hot the <Term id="campaign">measured afternoon</Term> felt, who was exposed and what the cooling package changes. Heat risk uses the US National Weather
          Service&apos;s <Term id="heat_index">feels-like temperature</Term> and its <Term id="heat_risk_level" text={heatLevelsTip().text}>heat-risk levels</Term>.
        </>
      }
      params={{ dewpoint_C: dewpoint }}
    >
      {(s) => (
        <>
          <Section title={term("humidity_assumption").label} data={s.humidity}>
            {(h) => (
              <Block title={term("humidity_assumption").label} tip={<InfoTip term="humidity_assumption" />}>
                <HumidityControl h={h} value={dewpoint} onChange={setDewpoint} />
              </Block>
            )}
          </Section>
          {s.brief ? <Brief items={s.brief.filter((b) => !NOT_TODAY_BRIEF.has(b.id))} /> : null}
          {s.kpis ? <KpiTiles kpis={s.kpis.filter((k) => !NOT_TODAY_KPIS.has(k.id))} label="Heat key numbers" tips={KPI_TIPS} /> : null}
          {s.today && s.categories ? <Exposure s={{ today: s.today, futures: s.futures ?? [], categories: s.categories }} scope="today" /> : null}
          <div className="grid2">
            {s.hist && s.categories ? <Distribution hist={s.hist} categories={s.categories} /> : null}
            {s.humidity && !s.humidity.user_set && !s.humidity.needs_input ? (
              <Block title="Where">
                <RunLayerMap rid={rid} keys={["heat_cat", "heat_index", "heat_cat_pkg", "heat_index_pkg"]} title="Heat risk" height={360} />
              </Block>
            ) : null}
          </div>
          {s.categories && s.humidity ? (
            <Block title="How heat risk is measured" tip={<HeatLevelsTip label="heat-risk levels" />}>
              <Method s={{ categories: s.categories, humidity: s.humidity }} />
            </Block>
          ) : null}
          <p className="cap" data-heat-next="true">
            How this changes as the climate warms: <Link to={heatHref(rid, "future", dewpoint)}>Future climate</Link>. How sure each scenario&apos;s cooling is:{" "}
            <Link to={runHref(rid, "trust")}>How much to trust this</Link>.
          </p>
        </>
      )}
    </ViewPage>
  );
}

/**
 * Heat risk › Future climate: residents per NWS category in each future, with and without the
 * package, at the dewpoint chosen under Today (the campaign's by default).
 */
export function HeatFutures() {
  const rid = useRid();
  const [dewpoint] = useDewpoint();
  const today = heatHref(rid, "today", dewpoint);
  return (
    <ViewPage
      view="heat"
      title="Heat risk as the climate warms"
      intro="Residents at each heat-risk level on a similar afternoon in each future climate, with and without the cooling package."
      params={{ dewpoint_C: dewpoint }}
    >
      {(s) => {
        if (!s.today || !s.futures || !s.categories)
          return (
            <p className="callout" role="note">
              {s.humidity?.needs_input ? (
                <>
                  This run has no measured humidity. Pick a humidity assumption under <Link to={today}>Today</Link> to count residents at heat risk in each future.
                </>
              ) : (
                "Heat risk in future climates is not available for this run."
              )}
            </p>
          );
        if (!s.futures.length) return <p className="cap">No future climates for this run: it has no climate-model projections yet (see Future climate above).</p>;
        const h = s.humidity;
        return (
          <>
            {h && h.dewpoint_C !== null ? (
              <p className="cap">
                Feels-like temperature with a dew point of {fmtNum(dewF(h.dewpoint_C), 0)} °F ({h.source ?? "assumed"}); change the humidity assumption under <Link to={today}>Today</Link>,
                where today&apos;s counts are.
              </p>
            ) : null}
            <Exposure s={{ today: s.today, futures: s.futures, categories: s.categories }} scope="futures" />
            <Block title="Extreme caution and Danger in each future" tip={<HeatLevelsTip label="heat-risk levels" />}>
              <Futures s={{ today: s.today, futures: s.futures }} />
            </Block>
          </>
        );
      }}
    </ViewPage>
  );
}
