// Heat (the heat brief): how hot the campaign afternoon felt (NWS heat index from each cell's
// temperature and the campaign dewpoint), who was exposed (residents per NWS category), how that
// changes in each CMIP6 future (a band between constant dewpoint and constant relative humidity,
// widened by the models' 10th–90th percentile warming), what the adaptation package buys, and how
// sure each scenario's cooling is (Robust / Direction only / Not established). The dewpoint can
// be changed to ask "what if the afternoon were more humid?" (the view's ?dewpoint_C=).
import { useState } from "react";
import type { HeatCase, HeatCategoryId, HeatSections } from "../../api/runs";
import { Bars, Histogram } from "../../charts";
import { Seg } from "../../components/ui/Seg";
import { fmtInt, fmtNum } from "../../theme/format";
import { Block, KpiTiles, Section, VerdictPill, ViewPage, useRid } from "./common";
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

type Humidity = "constant_dewpoint" | "constant_rh";

const period = (p: string) => p.replace("-", "–");
const people = (n: number | null | undefined, measure: "people" | "cells") =>
  n === null || n === undefined ? "—" : `${n >= 10000 ? `${fmtNum(n / 1000, 0)}k` : n >= 1000 ? `${fmtNum(n / 1000, 1)}k` : fmtInt(n)}${measure === "cells" ? " cells" : ""}`;
const range = (r: [number, number] | null | undefined, measure: "people" | "cells") =>
  !r ? "—" : Math.abs(r[1] - r[0]) < Math.max(1, 0.005 * r[1]) ? people(r[0], measure) : `${people(r[0], measure)} – ${people(r[1], measure)}`;

/** The dewpoint control: the campaign's, a preset, or a typed value (°C). */
function HumidityControl({ h, value, onChange }: { h: HeatSections["humidity"]; value: number | null; onChange: (v: number | null) => void }) {
  const [custom, setCustom] = useState<string>(value !== null ? String(value) : "");
  const opts: { value: string; label: string }[] = [];
  if (h.campaign_dewpoint_C !== null) opts.push({ value: "campaign", label: `Campaign (${fmtNum(h.campaign_dewpoint_C, 1)} °C)` });
  for (const p of h.presets) opts.push({ value: String(p.dewpoint_C), label: p.label });
  const cur = value === null ? (h.campaign_dewpoint_C !== null ? "campaign" : "") : String(value);
  const apply = () => {
    const v = Number(custom);
    if (custom.trim() !== "" && Number.isFinite(v) && v >= -30 && v <= 35) onChange(v);
  };
  return (
    <div className="stack" style={{ gap: 8 }}>
      {h.needs_input ? (
        <p className="callout" role="note">
          This run has no campaign humidity (no forcing file with a station or ERA5 dewpoint). Pick a dewpoint to turn its temperatures into a heat index; the result is a
          what-if, not a measurement.
        </p>
      ) : null}
      <div className="row" style={{ flexWrap: "wrap", gap: 8 }}>
        <Seg<string> label="Dewpoint" size="small" value={cur} onChange={(v) => onChange(v === "campaign" ? null : Number(v))} options={opts} />
        <label className="row cap" style={{ gap: 6 }}>
          Custom
          <input
            type="number"
            step={0.5}
            min={-30}
            max={35}
            value={custom}
            onChange={(e) => setCustom(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") apply();
            }}
            aria-label="Custom dewpoint in degrees Celsius"
            style={{ width: 80 }}
          />
          °C
        </label>
        <button type="button" className="btn small" onClick={apply}>
          Apply
        </button>
      </div>
      <p className="cap">
        {h.dewpoint_C !== null ? (
          <>
            Dewpoint {fmtNum(h.dewpoint_C, 1)} °C ({h.source ?? "assumed"})
            {h.rh_range ? `: relative humidity ${fmtNum(h.rh_range[0], 0)}–${fmtNum(h.rh_range[1], 0)}% across the city` : ""}. One dewpoint applies to every cell (moisture is close
            to uniform across a city on one afternoon).
          </>
        ) : (
          "No dewpoint chosen yet."
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

/** Residents (or cells) per NWS category: today, with the package, and each future. */
function Exposure({ s }: { s: Pick<HeatSections, "today" | "futures" | "categories"> }) {
  const [hum, setHum] = useState<Humidity>("constant_dewpoint");
  const periods = [...new Set(s.futures.map((f) => f.period))];
  const [per, setPer] = useState<string>(periods.find((p) => p.startsWith("2041")) ?? periods[Math.floor(periods.length / 2)] ?? "");
  const t = s.today;
  const rows: { label: string; c: HeatCase }[] = [{ label: "Today", c: t.unadapted }];
  if (t.adapted) rows.push({ label: "Today · with package", c: t.adapted });
  for (const f of s.futures.filter((x) => x.period === per)) {
    rows.push({ label: f.label, c: f.unadapted[hum] });
    if (f.adapted) rows.push({ label: `${f.label} · with package`, c: f.adapted[hum] });
  }
  const labelOf = (id: HeatCategoryId) => s.categories.find((c) => c.id === id)?.label ?? id;
  const measure = t.measure === "people" ? "Residents" : "Cells";
  const ec = t.unadapted.ec_or_worse;
  return (
    <Bars
      title={`${measure} by heat-risk category`}
      units={t.measure === "people" ? "residents" : "cells"}
      categories={rows.map((r) => r.label)}
      orientation="h"
      mode="stacked"
      series={CATS.map((id) => ({ id, label: labelOf(id), values: rows.map((r) => r.c.counts[id] ?? 0), color: CAT_COLOR[id] }))}
      valueLabel={measure}
      unit={t.measure === "people" ? "people" : "cells"}
      decimals={0}
      highlight={["Today"]}
      actions={
        s.futures.length ? (
          <>
            {periods.length > 1 ? <Seg<string> label="Period" size="small" value={per} onChange={setPer} options={periods.map((p) => ({ value: p, label: period(p) }))} /> : null}
            <Seg<Humidity>
              label="Future humidity"
              size="small"
              value={hum}
              onChange={setHum}
              options={[
                { value: "constant_dewpoint", label: "Constant dewpoint (lower)" },
                { value: "constant_rh", label: "Constant RH (upper)" },
              ]}
            />
          </>
        ) : undefined
      }
      caption={`Today ${people(ec, t.measure)} ${t.measure === "people" ? "residents are" : "are"} at Extreme caution or worse${
        t.adapted ? `; with ${t.package ?? "the package"}, ${people(t.adapted.ec_or_worse, t.measure)}` : ""
      }.${s.futures.length ? ` Futures (${period(per)}) add each pathway's median warming to the campaign afternoon; switch the humidity assumption to see the other end of the band.` : ""}`}
    />
  );
}

function Distribution({ hist, categories }: { hist: NonNullable<HeatSections["hist"]>; categories: HeatSections["categories"] }) {
  const [which, setWhich] = useState<"today" | "pkg">("today");
  const counts = which === "pkg" && hist.adapted_counts ? hist.adapted_counts : hist.counts;
  const lo = hist.edges[0];
  const hi = hist.edges[hist.edges.length - 1];
  const refs = categories.filter((c) => c.lo_F !== null && c.lo_F >= lo && c.lo_F <= hi).map((c) => ({ value: c.lo_F as number, label: c.label }));
  return (
    <Histogram
      title="Heat index across the city"
      units={hist.measure === "people" ? "residents per bin" : "cells per bin"}
      bins={{ edges: hist.edges, counts }}
      xLabel="Heat index"
      unit="°F"
      countLabel={hist.measure === "people" ? "Residents" : "Cells"}
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
              { value: "pkg", label: "With package" },
            ]}
          />
        ) : undefined
      }
      caption="Lines mark where the NWS categories start (Caution 80 °F, Extreme caution 90 °F, Danger 103 °F, Extreme danger 125 °F)."
    />
  );
}

function Futures({ s }: { s: Pick<HeatSections, "today" | "futures"> }) {
  const m = s.today.measure;
  return (
    <div style={{ overflowX: "auto" }}>
      <table className="tbl" aria-label="Exposure in each climate future">
        <thead>
          <tr>
            <th scope="col">Future</th>
            <th scope="col" className="r">
              Warming (°F)
            </th>
            <th scope="col" className="r">
              Extreme caution or worse
            </th>
            <th scope="col" className="r">
              Danger or worse
            </th>
            {s.today.adapted ? (
              <>
                <th scope="col" className="r">
                  With package: Extreme caution+
                </th>
                <th scope="col" className="r">
                  With package: Danger+
                </th>
              </>
            ) : null}
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Today (campaign afternoon)</td>
            <td className="r">0</td>
            <td className="r">{people(s.today.unadapted.ec_or_worse, m)}</td>
            <td className="r">{people(s.today.unadapted.danger_or_worse, m)}</td>
            {s.today.adapted ? (
              <>
                <td className="r">{people(s.today.adapted.ec_or_worse, m)}</td>
                <td className="r">{people(s.today.adapted.danger_or_worse, m)}</td>
              </>
            ) : null}
          </tr>
          {s.futures.map((f) => (
            <tr key={f.id}>
              <td>
                {f.label} {period(f.period)}
              </td>
              <td className="r">
                +{fmtNum(f.warming_F, 1)}
                {f.warming_lo_F !== null && f.warming_hi_F !== null ? <span className="cap"> ({fmtNum(f.warming_lo_F, 1)}–{fmtNum(f.warming_hi_F, 1)})</span> : null}
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
        Ranges span the climate models&apos; 10th–90th percentile warming and the two humidity assumptions (constant dewpoint, the lower end; constant relative humidity, the upper
        end). The heat index is steep near the category edges, so a few tenths of a degree can move many residents across one.
      </p>
    </div>
  );
}

function Verdicts({ rows }: { rows: NonNullable<HeatSections["verdicts"]> }) {
  return (
    <div style={{ overflowX: "auto" }}>
      <table className="tbl" aria-label="Verdict per adaptation scenario">
        <thead>
          <tr>
            <th scope="col">Scenario</th>
            <th scope="col" className="r">
              City-mean change
            </th>
            <th scope="col" className="r">
              Plausible range
            </th>
            <th scope="col">Verdict and why</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.slug}>
              <td>{r.scenario}</td>
              <td className="r">{r.estimate === null ? "—" : fmtNum(r.estimate, 2)}</td>
              <td className="r">{r.envelope ? `${fmtNum(r.envelope[0], 2)} to ${fmtNum(r.envelope[1], 2)}` : "—"}</td>
              <td>
                <VerdictPill verdict={r.verdict ? { verdict: r.verdict, label: r.label ?? r.verdict, reasons: r.reasons, qualifiers: r.qualifiers } : null} detail />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Method({ s }: { s: Pick<HeatSections, "categories" | "humidity"> }) {
  return (
    <div className="stack" style={{ gap: 8 }}>
      <p className="cap prose">
        {s.humidity.method}. Air temperature alone understates danger: humidity slows sweating, so the same air temperature feels hotter on a muggier day. Futures are a band
        between two standard assumptions: {s.humidity.assumptions.constant_dewpoint}; and {s.humidity.assumptions.constant_rh}.
      </p>
      <ul className="stack" style={{ listStyle: "none", padding: 0, margin: 0, gap: 4 }} aria-label="NWS heat-index categories">
        {s.categories
          .filter((c) => c.id !== "below")
          .map((c) => (
            <li key={c.id} className="row" style={{ gap: 8, alignItems: "baseline" }}>
              <span aria-hidden="true" style={{ display: "inline-block", width: 12, height: 12, borderRadius: 3, background: CAT_COLOR[c.id] }} />
              <strong>{c.label}</strong>
              <span className="cap">
                {c.hi_F === null ? `≥ ${fmtNum(c.lo_F, 0)} °F` : `${fmtNum(c.lo_F, 0)}–${fmtNum(c.hi_F, 0)} °F`}: {c.note}
              </span>
            </li>
          ))}
      </ul>
    </div>
  );
}

export default function Heat() {
  const rid = useRid();
  const [dewpoint, setDewpoint] = useState<number | null>(null);
  return (
    <ViewPage
      view="heat"
      title="Heat brief"
      intro="How hot the campaign afternoon felt, who was exposed, how that changes as the climate warms, what the adaptation package buys, and how sure each answer is. Heat risk uses the US National Weather Service heat index and its categories."
      params={{ dewpoint_C: dewpoint }}
    >
      {(s) => (
        <>
          <Section title="Humidity" data={s.humidity}>
            {(h) => (
              <Block title="Humidity">
                <HumidityControl h={h} value={dewpoint} onChange={setDewpoint} />
              </Block>
            )}
          </Section>
          {s.brief ? <Brief items={s.brief} /> : null}
          {s.kpis ? <KpiTiles kpis={s.kpis} label="Heat key numbers" /> : null}
          {s.today && s.futures && s.categories ? <Exposure s={{ today: s.today, futures: s.futures, categories: s.categories }} /> : null}
          <div className="grid2">
            {s.hist && s.categories ? <Distribution hist={s.hist} categories={s.categories} /> : null}
            {s.humidity && !s.humidity.user_set && !s.humidity.needs_input ? (
              <Block title="Where">
                <RunLayerMap rid={rid} keys={["heat_cat", "heat_index", "heat_cat_pkg", "heat_index_pkg"]} title="Heat risk" height={360} />
              </Block>
            ) : null}
          </div>
          {s.today && s.futures?.length ? (
            <Block title="As the climate warms">
              <Futures s={{ today: s.today, futures: s.futures }} />
            </Block>
          ) : null}
          <Section
            title="How sure"
            data={s.verdicts}
            absent={{
              reason: "The uncertainty envelopes are not computed for this run, so its scenarios have no verdict yet.",
              action: { kind: "run_job", label: "Compute uncertainty envelopes", method: "POST", path: `/api/runs/${encodeURIComponent(rid)}/actions/uncertainty`, body: {} },
            }}
          >
            {(v) => (
              <Block title="How sure: verdict per scenario">
                <Verdicts rows={v} />
              </Block>
            )}
          </Section>
          {s.categories && s.humidity ? (
            <Block title="Method">
              <Method s={{ categories: s.categories, humidity: s.humidity }} />
            </Block>
          ) : null}
        </>
      )}
    </ViewPage>
  );
}
