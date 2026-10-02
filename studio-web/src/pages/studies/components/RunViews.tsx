// Result charts of the post-run actions and run-level checks on the Validation tab, read from the
// run's own outputs: baselines forest (distance view), uncertainty layered intervals, literature
// panel (response view), emulator validation, planner and write-up links, and the "Truth vs
// recovered" card of synthetic demo projects (`GET /api/runs/{rid}/truth`, SPEC §8, §9.2).
import { api } from "../../../api/client";
import { useResource } from "../../../api/resource";
import { useView, type ResponseSections } from "../../../api/runs";
import { useTruth, type TruthRow } from "../../../api/studies";
import { Bars, DotRange, Forest, IntervalStack } from "../../../charts";
import { excludesZero } from "../../../charts/IntervalStack";
import { EmptyState } from "../../../components/ui/EmptyState";
import { Link } from "../../../router";
import { fmtNum, fmtPct, fmtValue, unitLabel } from "../../../theme/format";

const rh = (rid: string) => `/r/${encodeURIComponent(rid)}`;

function NotYet({ what }: { what: string }) {
  return <p className="cap">{what}</p>;
}

// ---------------------------------------------------------------- baselines

const sameText = (a: string | null | undefined, b: string | null | undefined) => !!a && !!b && a.trim().replace(/\.$/, "") === b.trim().replace(/\.$/, "");

/** `headline`: the card's own headline, so an identical verdict is not repeated. */
export function BaselinesResult({ rid, headline }: { rid: string; headline?: string | null }) {
  const vm = useView(rid, "distance");
  if (vm.error && !vm.data) return <EmptyState error={vm.error} />;
  const s = vm.data?.sections;
  if (!s) return <NotYet what="Loading…" />;
  const b = s.baselines;
  if (!b || !b.length) return <NotYet what="No baseline comparison in this run yet." />;
  const u = unitLabel(vm.data?.units.target);
  return (
    <div className="stack">
      {s.verdict && !sameText(s.verdict.text, headline) ? <p className="vt-headline">{s.verdict.text}</p> : null}
      <Forest
        title="Stack vs reference baselines"
        units={`ΔMSE (baseline − stack), ${u ? `${u}²` : "target units²"}; positive = stack better`}
        rows={b.map((r) => ({ id: r.id, label: r.label, est: r.delta_mse, se: r.delta_mse_se }))}
        z={2}
        better={{ side: "positive", label: "stack better" }}
        valueLabel="ΔMSE"
        unit={u ? `${u}²` : undefined}
        decimals={3}
        caption={`${b.filter((r) => r.stack_better).length} of ${b.length} baselines are beaten by more than 2 SE; ${b.filter((r) => r.baseline_better).length} beat the stack. Whiskers: ± 2 block-clustered SE.`}
      />
      <Link to={`${rh(rid)}/distance`}>Skill vs distance and block wins →</Link>
    </div>
  );
}

// ---------------------------------------------------------------- uncertainty

export function UncertaintyResult({ rid }: { rid: string }) {
  const vm = useView(rid, "uncertainty");
  if (vm.error && !vm.data) return <EmptyState error={vm.error} />;
  const rows = vm.data?.sections.rows;
  if (!vm.data) return <NotYet what="Loading…" />;
  if (!rows || !rows.length) return <NotYet what="No uncertainty envelopes yet: attach studies, then run the uncertainty report." />;
  const u = unitLabel(vm.data.units.target);
  const env = rows.map((r) => r.layers.find((l) => /envelope/i.test(l.id))).filter((l): l is NonNullable<typeof l> => !!l);
  const keep = env.filter(excludesZero).length;
  return (
    <div className="stack">
      <IntervalStack
        title="Layered intervals per scenario"
        units={`city-mean ΔT${u ? `, ${u}` : ""} (negative = cooler)`}
        rows={rows}
        valueLabel="City-mean ΔT"
        unit={u}
        caption={env.length ? `${keep} of ${env.length} scenarios exclude zero under the widest (envelope) interval.` : "Estimation intervals only: no specification or attribution layer yet."}
      />
      <Link to={`${rh(rid)}/uncertainty`}>Uncertainty tab (sources, climate spread) →</Link>
    </div>
  );
}

// ---------------------------------------------------------------- literature

type Lit = NonNullable<ResponseSections["literature"]>;

/** One quantity's literature rows: published ranges (grey) and SPARC ± 1.96 SE with the factor-of-2 band and the causal point. */
export function literatureRows(lit: Lit, quantity: string) {
  const pub = lit.rows.filter((r) => r.quantity === quantity);
  const ours = lit.sparc.filter((r) => r.quantity === quantity);
  return [
    ...pub.map((r) => ({
      id: r.key,
      label: r.citation ?? r.key,
      est: r.low !== null && r.high !== null ? (r.low + r.high) / 2 : r.low ?? r.high,
      lo: r.low,
      hi: r.high,
      muted: true,
    })),
    ...ours.map((r) => ({
      id: `sparc-${r.scenario}`,
      label: `SPARC: ${r.scenario}`,
      est: r.cooling,
      lo: r.cooling !== null && r.se !== null ? r.cooling - 1.96 * r.se : null,
      hi: r.cooling !== null && r.se !== null ? r.cooling + 1.96 * r.se : null,
      lo2: r.cooling === null ? null : Math.min(r.cooling / 2, r.cooling * 2),
      hi2: r.cooling === null ? null : Math.max(r.cooling / 2, r.cooling * 2),
      check: r.causal === null ? null : { est: r.causal, label: "causal estimate" },
    })),
  ];
}

export function LiteratureResult({ rid }: { rid: string }) {
  const vm = useView(rid, "response");
  if (vm.error && !vm.data) return <EmptyState error={vm.error} />;
  if (!vm.data) return <NotYet what="Loading…" />;
  const lit = vm.data.sections.literature;
  if (!lit) return <NotYet what="This run has no literature panel (older code, or no canopy/albedo lever)." />;
  const quantities = [...new Set([...lit.rows.map((r) => r.quantity), ...lit.sparc.map((r) => r.quantity)])];
  if (!quantities.length) return <NotYet what="No published values match this run's levers." />;
  return (
    <div className="stack">
      {quantities.map((q) => {
        const rows = literatureRows(lit, q);
        const ours = lit.sparc.find((r) => r.quantity === q);
        const unit = unitLabel(ours?.unit ?? lit.rows.find((r) => r.quantity === q)?.unit ?? "");
        const pub = lit.rows.filter((r) => r.quantity === q);
        const overlap = ours && ours.cooling !== null ? pub.some((r) => r.low !== null && r.low <= (ours.cooling as number) * 2 && (r.high ?? r.low) >= (ours.cooling as number) / 2) : null;
        return (
          <DotRange
            key={q}
            title={`Literature check: ${q}`}
            units={`cooling${unit ? `, ${unit}` : ""}`}
            rows={rows}
            valueLabel="Cooling"
            unit={unit}
            rangeLabel="published range / SPARC 95%"
            outerLabel="factor-of-2 band"
            decimals={2}
            caption={
              ours && ours.cooling !== null
                ? `SPARC's ${ours.scenario} cools by ${fmtValue(ours.cooling, unit, 2)}; published values ${overlap ? "overlap" : "fall outside"} its factor-of-2 band. Other cities, methods and scales: agreement is a sanity check, not validation.`
                : "Published ranges for this lever."
            }
          />
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------- emulator

type EmulatorSummary = {
  levers?: Record<string, { patch_pass_rate?: number | null; patch_mean_abs_err_median?: number | null; patch_mean_rel_err_median?: number | null; uniform_rel_err?: number | null; physics?: boolean | null }>;
};

export function EmulatorResult({ rid }: { rid: string }) {
  const res = useResource<EmulatorSummary>(`run:${rid}:outputs:emulator`, (s) => api.get<EmulatorSummary>(`/api/runs/${encodeURIComponent(rid)}/outputs/emulator`, undefined, s), {
    tags: [`run:${rid}:outputs`],
  });
  if (res.error && !res.data) return <NotYet what="No emulator yet: build it to enable the Lab's instant preview." />;
  const levers = Object.entries(res.data?.levers ?? {});
  if (!levers.length) return <NotYet what={res.loading ? "Loading…" : "The emulator has no levers."} />;
  return (
    <Bars
      title="Emulator vs exact engine"
      units="% of validation patches within tolerance"
      categories={levers.map(([v]) => v)}
      series={[{ id: "pass", label: "Patches passing", values: levers.map(([, l]) => (l.patch_pass_rate === null || l.patch_pass_rate === undefined ? null : l.patch_pass_rate * 100)) }]}
      orientation="h"
      valueLabel="Patches passing"
      unit="%"
      categoryLabel="Lever"
      domain={[0, 100]}
      decimals={0}
      caption={levers
        .map(([v, l]) => `${v}: median patch error ${fmtPct(l.patch_mean_rel_err_median ?? null)}, uniform-edit error ${fmtPct(l.uniform_rel_err ?? null)}`)
        .join("; ")}
    />
  );
}

// ---------------------------------------------------------------- planner and write-up

export function PlannerResult({ rid }: { rid: string }) {
  return (
    <p className="cap">
      Exposure, hot days, equity, plantable space, hexagons and logger sites are on the <Link to={`${rh(rid)}/planner`}>Planner tab</Link>.
    </p>
  );
}

export function WriteupResult({ rid }: { rid: string }) {
  return (
    <p className="cap">
      Read the <Link to={`${rh(rid)}/docs/methods`}>methods</Link> and the <Link to={`${rh(rid)}/docs/model_card`}>model card</Link>. The run report stays as written at the end of the run.
    </p>
  );
}

// ---------------------------------------------------------------- truth vs recovered

function truthNote(r: TruthRow): string {
  if (r.quantity === "noise_sd") return "sanity bound: the held-out RMSE should not fall below the planted noise";
  if (r.share === null) return "";
  const off = Math.abs(r.share - 1);
  return off <= 0.2 ? "recovered within 20%" : r.share < 1 ? `attenuated (${fmtPct(r.share)} of the truth)` : `overstated (${fmtPct(r.share)} of the truth)`;
}

export function TruthCard({ rid }: { rid: string }) {
  const res = useTruth(rid);
  if (res.error && !res.data) {
    if ("status" in res.error && res.error.status === 404) return <p className="cap">No planted truth for this run (truth.json is missing).</p>;
    return <EmptyState error={res.error} />;
  }
  const rows = res.data?.rows ?? [];
  if (!rows.length) return <p className="cap">{res.loading ? "Loading…" : "No planted quantities to compare."}</p>;
  const ratio = rows.filter((r) => r.share !== null);
  return (
    <div className="stack">
      <div className="tablewrap">
        <table className="tbl" aria-label="Truth vs recovered">
          <thead>
            <tr>
              <th scope="col">Quantity</th>
              <th scope="col" className="r">
                Planted truth
              </th>
              <th scope="col" className="r">
                Recovered
              </th>
              <th scope="col" className="r">
                Share
              </th>
              <th scope="col">Reading</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={`${r.quantity}-${r.scenario ?? ""}`}>
                <td>{r.label}</td>
                <td className="r num">{fmtValue(r.truth, r.unit, 3)}</td>
                <td className="r num">
                  {fmtValue(r.recovered, r.unit, 3)}
                  {r.se !== null ? ` ± ${fmtNum(r.se, 3)}` : ""}
                </td>
                <td className="r num">{r.share === null ? "—" : fmtNum(r.share, 2)}</td>
                <td className="cap">{truthNote(r)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {ratio.length ? (
        <DotRange
          title="Recovered ÷ planted"
          units="1 = recovered exactly"
          rows={ratio.map((r) => ({
            id: `${r.quantity}-${r.scenario ?? ""}`,
            label: r.label,
            est: r.share,
            lo: r.recovered !== null && r.se !== null && r.truth ? (r.recovered - 1.96 * r.se) / r.truth : null,
            hi: r.recovered !== null && r.se !== null && r.truth ? (r.recovered + 1.96 * r.se) / r.truth : null,
          }))}
          valueLabel="Share of the truth"
          decimals={2}
          rangeLabel="95% interval"
          caption={`${ratio.filter((r) => r.share !== null && Math.abs(r.share - 1) <= 0.2).length} of ${ratio.length} planted quantities are recovered within 20%.`}
        />
      ) : null}
    </div>
  );
}
