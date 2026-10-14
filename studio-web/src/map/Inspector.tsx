// Pinned cell inspector (SPEC §6.5): every layer value for the cell with units, lon/lat, zone,
// per-lever response curves rebuilt from the fitted parameters, and configured/Studio
// scenario deltas worded cooler/warmer. Data comes from GET /api/runs/{rid}/cells/{index}.
// A planner reads "place" for a grid cell (glossary `cell`); a layer's technical key and the
// curve's shape name are in tooltips.
import type { LayerMeta } from "../api/types";
import { LineBand } from "../charts/LineBand";
import { lever as leverWords, shape, term } from "../content/glossary";
import { EmptyState } from "../components/ui/EmptyState";
import { IconButton } from "../components/ui/IconButton";
import { InfoTip, TermTipBody } from "../components/ui/InfoTip";
import { Tooltip } from "../components/ui/Tooltip";
import { fmtNum, fmtSigned, fmtTempChange, unitLabel } from "../theme/format";
import { fmtLonLat } from "./grid";

export type CellCurve = {
  model: string;
  A: number | null;
  ds: number | null;
  inflection: number | null;
  d90: number | null;
  dmax: number | null;
  dose: number[];
  benefit: number[];
};

/** `GET /api/runs/{rid}/cells/{index}` (api.md §6.2). */
export type CellInfo = {
  index: number;
  id: number | string;
  lon: number | null;
  lat: number | null;
  zone: number | string | null;
  values: Record<string, number | null>;
  curves: Record<string, CellCurve>;
  scenarios: Record<string, number | null>;
};

export type InspectorProps = {
  cell: CellInfo | null;
  layers: LayerMeta[];
  targetUnit?: string;
  loading?: boolean;
  error?: unknown;
  onClose: () => void;
  /** Layer key to list first (the one on the map). */
  focusKey?: string | null;
};

function fmtLayerValue(m: LayerMeta | undefined, v: number | null): string {
  if (v === null || !Number.isFinite(v)) return "—";
  if (!m) return fmtNum(v, 3);
  if (m.scale === "cat") return m.labels?.[v] ?? String(v);
  const x = v * (m.mult || 1);
  const u = unitLabel(m.unit);
  const s = m.scale === "div" && (m.center ?? 0) === 0 ? fmtSigned(x, m.decimals) : fmtNum(x, m.decimals);
  return u ? (u === "%" ? s + "%" : `${s} ${u}`) : s;
}

/** Words of the pinned-place card. */
export const INSPECTOR_WORDS = {
  title: `Pinned ${term("cell").label.toLowerCase()}`,
  place: term("cell").label,
  unpin: "Unpin this place",
  row: "Data row",
  zone: "Neighbourhood",
  layers: "Map layers here",
  scenarios: "Scenario effects here",
  /** The y axis of a lever's curve: the cooling it brings here. */
  cooling: "Cooling",
  /** The marker where a typical place has 90% of the cooling it can get (d90). */
  d90: "90% of the cooling",
} as const;

/** A lever's plain name: the glossary's ("Trees & canopy"), else its layer's label, else the key. */
function leverName(key: string, byKey: Map<string, LayerMeta>): string {
  return leverWords(key)?.name ?? byKey.get(key)?.label ?? key;
}

export function Inspector({ cell, layers, targetUnit, loading, error, onClose, focusKey }: InspectorProps) {
  const byKey = new Map(layers.map((l) => [l.key, l]));
  const W = INSPECTOR_WORDS;
  return (
    <section className="card" aria-label={W.title}>
      <header>
        <div>
          <p className="eyebrow">
            {W.title}
            <InfoTip term="cell" />
          </p>
          <h3>{cell ? `${W.place} ${cell.id}` : loading ? "Loading…" : W.place}</h3>
        </div>
        <IconButton icon="x" label={W.unpin} onClick={onClose} size="small" />
      </header>
      {error ? <EmptyState error={error} /> : null}
      {cell ? (
        <>
          <dl className="kv">
            <dt>{W.row}</dt>
            <dd>{cell.index}</dd>
            {cell.lon !== null && cell.lat !== null ? (
              <>
                <dt>Location</dt>
                <dd>{fmtLonLat([cell.lon, cell.lat])}</dd>
              </>
            ) : null}
            {cell.zone !== null ? (
              <>
                <dt>{W.zone}</dt>
                <dd>{String(cell.zone)}</dd>
              </>
            ) : null}
          </dl>
          <details open>
            <summary>{W.layers}</summary>
            <dl className="kv">
              {Object.entries(cell.values)
                .sort(([a], [b]) => (a === focusKey ? -1 : b === focusKey ? 1 : 0))
                .map(([k, v]) => (
                  <div key={k} className="kv-row">
                    <dt>
                      {/* the layer's technical key (and what it is) on hover, never a native title */}
                      <Tooltip content={<TermTipBody tip={byKey.get(k)?.desc || byKey.get(k)?.label || k} technical={k} />}>{byKey.get(k)?.label ?? k}</Tooltip>
                    </dt>
                    <dd>{fmtLayerValue(byKey.get(k), v)}</dd>
                  </div>
                ))}
            </dl>
          </details>
          {Object.keys(cell.scenarios).length ? (
            <details open>
              <summary>{W.scenarios}</summary>
              <dl className="kv">
                {Object.entries(cell.scenarios).map(([k, v]) => (
                  <div key={k} className="kv-row">
                    <dt>{k}</dt>
                    <dd>{fmtTempChange(v, targetUnit, 2)}</dd>
                  </div>
                ))}
              </dl>
            </details>
          ) : null}
          {Object.entries(cell.curves).map(([lever, c]) => {
            const name = leverName(lever, byKey);
            return (
              <LineBand
                key={lever}
                bare
                title={`${name}: ${term("dose_response").label.toLowerCase()} here`}
                series={[{ id: lever, label: `${name} (${shape(c.model)?.label.toLowerCase() ?? c.model})`, x: c.dose, y: c.benefit }]}
                xLabel={`${name}: ${term("dose").label.toLowerCase()}`}
                yLabel={W.cooling}
                yUnit={unitLabel(targetUnit)}
                refLines={c.d90 !== null ? [{ axis: "x", value: c.d90, label: W.d90 }] : []}
                height={170}
                width={300}
              />
            );
          })}
        </>
      ) : null}
    </section>
  );
}
