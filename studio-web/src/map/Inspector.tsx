// Pinned cell inspector (SPEC §6.5): every layer value for the cell with units, lon/lat, zone,
// per-lever response curves rebuilt from the fitted parameters, and configured/Studio
// scenario deltas worded cooler/warmer. Data comes from GET /api/runs/{rid}/cells/{index}.
import type { LayerMeta } from "../api/types";
import { LineBand } from "../charts/LineBand";
import { EmptyState } from "../components/ui/EmptyState";
import { IconButton } from "../components/ui/IconButton";
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

export function Inspector({ cell, layers, targetUnit, loading, error, onClose, focusKey }: InspectorProps) {
  const byKey = new Map(layers.map((l) => [l.key, l]));
  return (
    <section className="card" aria-label="Pinned cell">
      <header>
        <div>
          <p className="eyebrow">Pinned cell</p>
          <h3>{cell ? `Cell ${cell.id}` : loading ? "Loading…" : "Cell"}</h3>
        </div>
        <IconButton icon="x" label="Unpin cell" onClick={onClose} size="small" />
      </header>
      {error ? <EmptyState error={error} /> : null}
      {cell ? (
        <>
          <dl className="kv">
            <dt>Row</dt>
            <dd>{cell.index}</dd>
            {cell.lon !== null && cell.lat !== null ? (
              <>
                <dt>Location</dt>
                <dd>{fmtLonLat([cell.lon, cell.lat])}</dd>
              </>
            ) : null}
            {cell.zone !== null ? (
              <>
                <dt>Zone</dt>
                <dd>{String(cell.zone)}</dd>
              </>
            ) : null}
          </dl>
          <details open>
            <summary>Layer values</summary>
            <dl className="kv">
              {Object.entries(cell.values)
                .sort(([a], [b]) => (a === focusKey ? -1 : b === focusKey ? 1 : 0))
                .map(([k, v]) => (
                  <span key={k} style={{ display: "contents" }}>
                    <dt title={k}>{byKey.get(k)?.label ?? k}</dt>
                    <dd>{fmtLayerValue(byKey.get(k), v)}</dd>
                  </span>
                ))}
            </dl>
          </details>
          {Object.keys(cell.scenarios).length ? (
            <details open>
              <summary>Scenario change here</summary>
              <dl className="kv">
                {Object.entries(cell.scenarios).map(([k, v]) => (
                  <span key={k} style={{ display: "contents" }}>
                    <dt>{k}</dt>
                    <dd>{fmtTempChange(v, targetUnit, 2)}</dd>
                  </span>
                ))}
              </dl>
            </details>
          ) : null}
          {Object.entries(cell.curves).map(([lever, c]) => (
            <LineBand
              key={lever}
              bare
              title={`${lever} response here`}
              series={[{ id: lever, label: `${lever} (${c.model})`, x: c.dose, y: c.benefit }]}
              xLabel={`${lever} dose`}
              yLabel="Cooling"
              yUnit={unitLabel(targetUnit)}
              refLines={c.d90 !== null ? [{ axis: "x", value: c.d90, label: "90% of max" }] : []}
              height={170}
              width={300}
            />
          ))}
        </>
      ) : null}
    </section>
  );
}
