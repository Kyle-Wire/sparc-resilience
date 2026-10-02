// Map explorer (SPEC §6.4, §6.5): the full-width MapView over every generated layer group
// (GET /layers), swipe / side-by-side / difference, lock scale, hex mode, overlays (logger
// sites, before/after pairs, zone outlines, CV block lines, influence circle), the pinned
// cell inspector (GET /cells/{index}, with curve reconstruction), PNG and layer exports, the
// per-run selection (URL `sel`, legend-histogram brushes, drawn shapes) and the analysis-
// tools drawer (Region stats, Breakdown, Relationships, Correlogram).
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { resolveMask, useRegions } from "../../api/analysis";
import { errorMessage } from "../../api/client";
import { layerExportUrl, useCell, useRunDetailFull, useView } from "../../api/runs";
import { countBits } from "../../api/binary";
import { isFinishedRun, type LayerGroup, type LayerMeta } from "../../api/types";
import { Chips } from "../../components/ui/Chips";
import { Drawer } from "../../components/ui/Drawer";
import { EmptyState } from "../../components/ui/EmptyState";
import { Seg } from "../../components/ui/Seg";
import { Tabs } from "../../components/ui/Tabs";
import { Inspector, type CellInfo } from "../../map/Inspector";
import { MapView, type CompareMode, type MapSelectionEvent } from "../../map/MapView";
import type { OverlayFeature } from "../../map/Overlay";
import type { PickMode } from "../../map/tools/pick";
import type { ToolId } from "../../map/tools/types";
import { runLayerLoader, useRunGrid, useRunLayers, type LayerValues } from "../../map/data";
import { rowAt, type GridData } from "../../map/grid";
import { runIsLive } from "../../layouts/resources";
import { codecs, useUrlState } from "../../router";
import { useRunSelection, type RunSelection } from "../../stores/selection";
import { fmtInt, unitLabel } from "../../theme/format";
import { LiveBanner, useRid } from "./common";
import { curvePoints, hexMeans } from "./format";
import { BreakdownTool } from "./tools/BreakdownTool";
import { CorrelogramTool } from "./tools/CorrelogramTool";
import { RegionStatsTool } from "./tools/RegionStatsTool";
import { RelationshipsTool } from "./tools/RelationshipsTool";
import { selectMask, selectionText, type ToolProps } from "./tools/shared";

const MODES: readonly CompareMode[] = ["none", "swipe", "side", "diff"];
const TOOLS = ["region", "breakdown", "relationships", "correlogram"] as const;
type ToolTab = (typeof TOOLS)[number];
const TOOL_LABEL: Record<ToolTab, string> = { region: "Region stats", breakdown: "Breakdown", relationships: "Relationships", correlogram: "Correlogram" };
const OVERLAYS = ["zones", "cv", "sites", "pairs", "influence"] as const;
type OverlayId = (typeof OVERLAYS)[number];
const MAP_TOOLS: ToolId[] = ["pan", "rect", "circle", "polygon", "pick"];
const PICK_ZONES: PickMode = { by: "zone" };
const NO_RANGES: never[] = [];

/**
 * Keep the store's mask in step with the URL selection: specs arriving without a mask (a
 * reload, a deep link, a saved region, the Accuracy histogram) are resolved by the server.
 */
export function useSelectionMask(rid: string, sel: RunSelection, n: number | null): { resolving: boolean; error: unknown } {
  const [state, setState] = useState<{ key: string | null; resolving: boolean; error: unknown }>({ key: null, resolving: false, error: null });
  const { spec, key, mask, resolve } = sel;
  useEffect(() => {
    if (!spec || !key || mask || n === null) return;
    const ctrl = new AbortController();
    setState({ key, resolving: true, error: null });
    resolveMask(rid, spec, n, ctrl.signal).then(
      (r) => {
        if (ctrl.signal.aborted) return;
        resolve({ mask: r.mask, n_cells: r.reply.n_cells });
        setState({ key, resolving: false, error: null });
      },
      (error: unknown) => {
        if (!ctrl.signal.aborted) setState({ key, resolving: false, error });
      },
    );
    return () => ctrl.abort();
  }, [rid, spec, key, mask, n, resolve]);
  return state.key === key ? { resolving: state.resolving, error: state.error } : { resolving: false, error: null };
}

/** The cell inspector fed by GET /cells/{index}, with each lever's curve rebuilt. */
function CellInspector({ rid, row, close, layers, focusKey, unit }: { rid: string; row: number; close: () => void; layers: LayerMeta[]; focusKey: string | null; unit: string }) {
  // Configured scenarios are listed in the catalogue as sc:<slug>; ask for their deltas here.
  const scenarios = useMemo(() => layers.filter((l) => l.key.startsWith("sc:")).map((l) => `configured:${l.key.slice(3)}`), [layers]);
  const cell = useCell(rid, row, scenarios);
  const data = useMemo<CellInfo | null>(() => {
    if (!cell.data) return null;
    const curves = Object.fromEntries(Object.entries(cell.data.curves).map(([k, c]) => [k, { ...c, ...curvePoints(c) }]));
    // Name configured scenarios as the catalogue does ("configured:<slug>" → its label).
    const label = (ref: string) => (ref.startsWith("configured:") ? (layers.find((l) => l.key === `sc:${ref.slice(11)}`)?.label ?? ref) : ref);
    const scen = Object.fromEntries(Object.entries(cell.data.scenarios).map(([k, v]) => [label(k), v]));
    return { ...cell.data, curves, scenarios: scen };
  }, [cell.data, layers]);
  return <Inspector cell={data} layers={layers} targetUnit={unit} loading={cell.loading} error={cell.error} onClose={close} focusKey={focusKey} />;
}

/**
 * The project's saved regions (api.md §6.3, listed on every run of the project): picking one
 * makes it the run's selection (`sel=rg_…`), resolved on this run by the server.
 */
function SavedRegions({ rid, sel }: { rid: string; sel: RunSelection }) {
  const regions = useRegions(rid);
  const list = regions.data ?? [];
  if (!list.length) return null;
  const cur = sel.spec && "kind" in sel.spec && sel.spec.kind === "region" ? sel.spec.id : "";
  return (
    <label className="row cap">
      Saved region
      <select
        value={cur}
        aria-label="Saved region"
        onChange={(e) => {
          const r = list.find((x) => x.id === e.target.value);
          if (r) sel.set({ kind: "region", id: r.id }, { label: r.name, source: "region" });
          else sel.clear();
        }}
      >
        <option value="">none</option>
        {list.map((r) => (
          <option key={r.id} value={r.id} title={r.portable === false ? "Drawn in run coordinates (no CRS): it may not resolve on other runs" : undefined}>
            {r.name} ({fmtInt(r.n_cells)} cells{r.portable === false ? ", not portable" : ""})
          </option>
        ))}
      </select>
    </label>
  );
}

/** A row near the middle of the grid (default centre of the influence circle). */
function centreRow(g: GridData): number {
  const r = rowAt(g, g.nx / 2, g.ny / 2);
  if (r >= 0) return r;
  let best = 0;
  let bestD = Infinity;
  for (let i = 0; i < g.n; i++) {
    const d = (g.ix[i] - g.nx / 2) ** 2 + (g.ny - 1 - g.iy[i] - g.ny / 2) ** 2;
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  }
  return best;
}

export default function MapTab() {
  const rid = useRid();
  const detail = useRunDetailFull(rid);
  const finished = isFinishedRun(detail.data?.run.status);
  const runLive = runIsLive(detail.data?.run.status);
  const grid = useRunGrid(rid, { immutable: finished });
  const layers = useRunLayers(rid);
  const [layerKey, setLayerKey] = useUrlState("layer", codecs.optString());
  const [cmpKey, setCmpKey] = useUrlState("cmp", codecs.optString());
  const [mode, setMode] = useUrlState("mode", codecs.enum(MODES, "none"));
  const [hexRaw, setHex] = useUrlState("hex", codecs.int(0));
  const hex = hexRaw === 250 || hexRaw === 500 ? hexRaw : 0;
  const [ovRaw, setOv] = useUrlState("ov", codecs.list());
  const ovKey = ovRaw.join(",");
  const ov = useMemo(() => (ovKey ? ovKey.split(",") : []).filter((x): x is OverlayId => (OVERLAYS as readonly string[]).includes(x)), [ovKey]);
  const [infl, setInfl] = useUrlState("infl", codecs.optString());
  const [toolRaw, setTool] = useUrlState("tool", codecs.optString());
  const tool = (TOOLS as readonly string[]).includes(toolRaw ?? "") ? (toolRaw as ToolTab) : null;
  const sel = useRunSelection(rid);
  const [pinned, setPinned] = useState<number | null>(null);
  const g = grid.data ?? null;
  const resolving = useSelectionMask(rid, sel, g ? g.n : null);

  const groups: LayerGroup[] = layers.data ?? [];
  const all = useMemo(() => groups.flatMap((x) => x.layers), [groups]);
  const cur = layerKey && all.some((l) => l.key === layerKey) ? layerKey : (all[0]?.key ?? null);
  const curMeta = all.find((l) => l.key === cur) ?? null;

  const base = useMemo(() => (g ? runLayerLoader(rid, g.meta.etag) : null), [rid, g]);
  // The analysis tools work on per-cell values, never on the hexagon means painted in hex mode
  // (a Relationships brush must select the cells whose own values fall in the box).
  const rawLoad = useCallback(
    (meta: LayerMeta): Promise<LayerValues> => (base ? base(meta) : Promise.reject(new Error("The grid is not loaded yet"))),
    [base],
  );
  const hexCache = useRef(new WeakMap<LayerValues, Map<number, Float32Array>>());
  const loadLayer = useCallback(
    async (meta: LayerMeta): Promise<LayerValues> => {
      if (!base || !g) throw new Error("The grid is not loaded yet");
      const v = await base(meta);
      if (!hex || meta.scale === "cat") return v;
      let per = hexCache.current.get(v);
      if (!per) hexCache.current.set(v, (per = new Map()));
      let h = per.get(hex);
      if (!h) per.set(hex, (h = hexMeans(g, v, hex)));
      return h;
    },
    [base, g, hex],
  );

  // Overlay data comes from the views that own it, fetched only while the overlay is on.
  const acc = useView(rid, ov.includes("cv") ? "accuracy" : null);
  const planner = useView(rid, ov.includes("sites") || ov.includes("pairs") ? "planner" : null);
  const influence = useView(rid, ov.includes("influence") ? "influence" : null);
  const ranges = influence.data?.sections.ranges ?? NO_RANGES;
  const inflKey = infl && ranges.some((r) => r.predictor === infl) ? infl : (ranges.find((r) => r.predictor === cur)?.predictor ?? ranges[0]?.predictor ?? null);

  const overlays = useMemo<OverlayFeature[]>(() => {
    if (!g) return [];
    const out: OverlayFeature[] = [];
    if (ov.includes("zones") && g.meta.zones.length) out.push({ type: "outline", id: "zones", label: "Zone outlines", classOf: (r) => (g.zone[r] >= 0 ? g.zone[r] : null) });
    const block = acc.data?.sections.cv_design?.block_m;
    if (ov.includes("cv") && block) out.push({ type: "gridlines", id: "cv", label: "CV blocks", spacing_m: block });
    const ps = planner.data?.sections;
    if (ov.includes("sites") && ps?.sites?.length)
      out.push({ type: "points", id: "sites", label: "Logger sites", points: ps.sites.map((s) => ({ row: s.row, label: s.label ?? String(s.id), tone: "s2" as const })) });
    if (ov.includes("pairs") && ps?.pairs?.length)
      out.push({ type: "links", id: "pairs", label: "Before/after pairs", tone: "s3", links: ps.pairs.map((p) => ({ from: { row: p.treated }, to: { row: p.control } })) });
    const r = ranges.find((x) => x.predictor === inflKey);
    if (ov.includes("influence") && r?.range_m)
      out.push({ type: "circle", id: "influence", label: `${r.label} influence range`, center: { row: pinned ?? centreRow(g) }, radius_m: r.range_m, tone: "s2" });
    if (hex) out.push({ type: "hexgrid", id: "hex", label: `${hex} m hexagons`, size_m: hex });
    return out;
  }, [g, ov, acc.data, planner.data, ranges, inflKey, pinned, hex]);

  const onSelection = useCallback(
    (e: MapSelectionEvent) => {
      if (!e.spec && !e.mask) {
        sel.clear();
        return;
      }
      if (e.spec) {
        sel.set(e.spec, { mask: e.mask, n_cells: e.mask ? countBits(e.mask) : null, label: e.label, source: e.source });
        return;
      }
      if (e.mask) void selectMask(rid, sel, e.mask, e.label, e.source);
    },
    [rid, sel],
  );

  if (grid.error) return <EmptyState error={grid.error} />;
  if (layers.error) return <EmptyState error={layers.error} />;
  if (!g || !layers.data) return <p className="cap">Loading the map…</p>;
  if (!all.length) return <EmptyState title="No map layers yet" body="Layers appear as soon as the run writes predictions.parquet (after S2_S3)." />;

  const unit = unitLabel(g.meta.units.target);
  const toolProps: ToolProps = { rid, grid: g, groups, layerKey: cur, selection: sel, loadLayer: rawLoad, unit };
  const noCrs = !g.meta.crs;
  const exportLink = (fmt: "tif" | "csv" | "geojson", label: string) =>
    cur && !(noCrs && fmt !== "csv") ? (
      <a className="btn small ghost" href={layerExportUrl(rid, cur, fmt)} download>
        {label}
      </a>
    ) : (
      <span className="btn small ghost" aria-disabled="true" title="This run has no CRS, so GIS formats are unavailable">
        {label}
      </span>
    );

  return (
    <section className="stack" aria-labelledby="map-title">
      <header className="row" style={{ justifyContent: "space-between" }}>
        <h2 id="map-title" style={{ margin: 0 }}>
          Map explorer
        </h2>
        <div className="row">
          <Seg<0 | 250 | 500>
            label="Hex mode"
            size="small"
            value={hex}
            onChange={(v) => setHex(v)}
            options={[
              { value: 0, label: "Cells" },
              { value: 250, label: "Hex 250 m" },
              { value: 500, label: "Hex 500 m" },
            ]}
          />
          <span className="cap">Export layer:</span>
          {exportLink("tif", "GeoTIFF")}
          {exportLink("csv", "CSV")}
          {exportLink("geojson", "GeoJSON")}
          <button type="button" className="btn small" aria-pressed={!!tool} onClick={() => setTool(tool ? null : "region")}>
            Analysis tools
          </button>
        </div>
      </header>
      <div className="row">
        <Chips<OverlayId>
          label="Overlays"
          items={OVERLAYS.filter((o) => o !== "zones" || g.meta.zones.length > 0).map((o) => ({
            value: o,
            label: { zones: "Zone outlines", cv: "CV blocks", sites: "Logger sites", pairs: "Before/after pairs", influence: "Influence circle" }[o],
          }))}
          selected={ov}
          onToggle={(v, on) => setOv(on ? [...ov, v] : ov.filter((x) => x !== v))}
        />
        {ov.includes("influence") && ranges.length ? (
          <label className="row cap">
            Influence of
            <select value={inflKey ?? ""} onChange={(e) => setInfl(e.target.value)} aria-label="Influence circle predictor">
              {ranges.map((r) => (
                <option key={r.predictor} value={r.predictor}>
                  {r.label}
                </option>
              ))}
            </select>
            {pinned === null ? "(centred on the map; pin a cell to move it)" : "(around the pinned cell)"}
          </label>
        ) : null}
      </div>
      {runLive ? <LiveBanner>This run is still running: new layers appear, and changed ones refresh, as each output is written.</LiveBanner> : null}
      <div className="row cap" role="status" aria-live="polite" data-testid="selection-status">
        <span>
          <strong>Selection:</strong> {selectionText(sel)}
          {resolving.resolving ? " · resolving…" : ""}
          {resolving.error ? ` · could not resolve: ${errorMessage(resolving.error)}` : ""}
        </span>
        {sel.spec ? (
          <button type="button" className="btn small ghost" onClick={() => sel.clear()}>
            Clear
          </button>
        ) : null}
        <SavedRegions rid={rid} sel={sel} />
        {hex ? <span>· showing {hex} m hexagon means</span> : null}
      </div>
      <MapView
        grid={g}
        groups={groups}
        loadLayer={loadLayer}
        layerKey={cur}
        onLayerChange={(k) => setLayerKey(k)}
        compareKey={cmpKey}
        onCompareChange={(k) => setCmpKey(k)}
        compareMode={mode}
        onCompareModeChange={(m) => setMode(m)}
        allowDiff
        selection={sel.mask}
        onSelection={onSelection}
        tools={MAP_TOOLS}
        pickMode={PICK_ZONES}
        overlays={overlays}
        onPin={(r) => setPinned(r)}
        inspector={(row, close, layer) => <CellInspector rid={rid} row={row} close={close} layers={all} focusKey={layer?.key ?? null} unit={g.meta.units.target} />}
        height={600}
        title={detail.data?.header.name ?? rid}
      />
      <Drawer open={!!tool} onClose={() => setTool(null)} title="Analysis tools" mode="inline">
        <Tabs<ToolTab> label="Analysis tool" items={TOOLS.map((t) => ({ id: t, label: TOOL_LABEL[t] }))} value={tool ?? "region"} onChange={(t) => setTool(t)}>
          {tool === "region" ? <RegionStatsTool {...toolProps} /> : null}
          {tool === "breakdown" ? <BreakdownTool {...toolProps} /> : null}
          {tool === "relationships" ? <RelationshipsTool {...toolProps} /> : null}
          {tool === "correlogram" ? <CorrelogramTool {...toolProps} /> : null}
        </Tabs>
      </Drawer>
      {curMeta ? (
        <p className="cap">
          {curMeta.label}: {fmtInt(curMeta.stats.n)} cells with values{curMeta.sign_note ? ` · ${curMeta.sign_note}` : ""}
          {curMeta.source ? ` · from ${curMeta.source.file}${curMeta.source.column ? ` (${curMeta.source.column})` : ""}` : ""}
        </p>
      ) : null}
    </section>
  );
}
