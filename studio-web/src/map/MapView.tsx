// MapView (SPEC §6.5): layer picker, GridCanvas, legend with a brushable histogram, pinned
// inspector, overlays, swipe / side-by-side / difference compare, opacity, lock scale, manual
// range, tools (pan, brush, rect, circle, polygon, pick), keyboard cell cursor and PNG export
// with the legend at 4×. Data-agnostic: the caller passes the grid, the layer catalogue and a
// loader (run hub and Lab use map/data.ts; the setup preview uses project preview binaries).
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import type { LayerGroup, LayerMeta, SelectionSpec } from "../api/types";
import { IconButton } from "../components/ui/IconButton";
import { NumberField } from "../components/ui/NumberField";
import { Seg } from "../components/ui/Seg";
import { downloadBlob, fileSlug } from "../components/ui/download";
import { errorMessage } from "../api/client";
import { toast, useUi } from "../stores/ui";
import { fmtNum, fmtSigned, unitLabel } from "../theme/format";
import { getLuts, THEME_COLORS } from "../theme/palette";
import { GridCanvas, type GridCanvasHandle } from "./GridCanvas";
import { Legend, categorySwatches, divergingEnds } from "./Legend";
import { LayerPicker, findLayer } from "./LayerPicker";
import { MapChrome, Overlay, type OverlayFeature } from "./Overlay";
import { SwipeCompare } from "./SwipeCompare";
import { maskFromRange, type LayerValues } from "./data";
import { computeDomain, type Domain } from "./domain";
import { fmtLonLat, rasterToLonLat, niceLength, fmtMeters, type GridData } from "./grid";
import { brushTool, type BrushSettings } from "./tools/brush";
import { circleTool } from "./tools/circle";
import { panTool } from "./tools/pan";
import { pickTool, type PickMode } from "./tools/pick";
import { polygonTool } from "./tools/polygon";
import { rectTool } from "./tools/rect";
import type { MapTool, ToolId, ToolResult } from "./tools/types";
import { useMapView } from "./useMapView";
import type { IconName } from "../components/ui/Icon";

export type CompareMode = "none" | "swipe" | "side" | "diff";

export type MapSelectionEvent = { spec: SelectionSpec | null; mask: Uint8Array | null; label: string; portable: boolean; source: string };

export type MapViewProps = {
  grid: GridData;
  groups: LayerGroup[];
  loadLayer: (meta: LayerMeta) => Promise<LayerValues>;
  /** Controlled layer (e.g. from the URL); uncontrolled when omitted. */
  layerKey?: string | null;
  onLayerChange?: (key: string) => void;
  compareKey?: string | null;
  onCompareChange?: (key: string | null) => void;
  compareMode?: CompareMode;
  onCompareModeChange?: (m: CompareMode) => void;
  /** Difference mode needs the same grid; false disables it with a reason. */
  allowDiff?: boolean;
  selection?: Uint8Array | null;
  onSelection?: (e: MapSelectionEvent) => void;
  hatch?: Uint8Array | null;
  tools?: ToolId[];
  brush?: { edit: Float32Array; settings: () => BrushSettings; onChange?: (rows: number[]) => void };
  pickMode?: PickMode;
  overlays?: OverlayFeature[];
  /** Pinned inspector content for a row (default: lon/lat, zone and the layer value). */
  inspector?: (row: number, close: () => void, layer: LayerMeta | null) => ReactNode;
  onPin?: (row: number | null) => void;
  sidePanel?: ReactNode;
  height?: number;
  title?: string;
  /** Hide the side column (legend etc.), e.g. for small embedded maps. */
  noSide?: boolean;
};

const TOOL_INFO: Record<ToolId, { icon: IconName; label: string }> = {
  pan: { icon: "hand", label: "Pan and pick (drag to move, click to pin a cell)" },
  brush: { icon: "brush", label: "Brush edits" },
  rect: { icon: "square", label: "Select a rectangle" },
  circle: { icon: "circle", label: "Select a circle" },
  polygon: { icon: "polygon", label: "Select a polygon (double-click or Enter to close)" },
  pick: { icon: "target", label: "Pick a zone or hexagon" },
};

function useLayerValues(meta: LayerMeta | null, load: (m: LayerMeta) => Promise<LayerValues>) {
  const [state, setState] = useState<{ key: string | null; values: LayerValues | null; error: unknown }>({ key: null, values: null, error: null });
  useEffect(() => {
    if (!meta) return;
    let live = true;
    load(meta).then(
      (values) => live && setState({ key: meta.key, values, error: null }),
      (error: unknown) => live && setState({ key: meta.key, values: null, error }),
    );
    return () => {
      live = false;
    };
  }, [meta, load]);
  return state.key === meta?.key ? state : { key: meta?.key ?? null, values: null, error: null };
}

/** Difference B − A per row (NaN where either is missing). */
export function differenceValues(a: ArrayLike<number>, b: ArrayLike<number>): Float32Array {
  const n = Math.min(a.length, b.length);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = b[i] - a[i];
  return out;
}

export function MapView(props: MapViewProps) {
  const { grid, groups } = props;
  const dark = useUi((s) => s.dark);
  const view = useMapView();
  const canvasHandle = useRef<GridCanvasHandle>(null);
  const firstKey = groups[0]?.layers[0]?.key ?? null;
  const [innerKey, setInnerKey] = useState<string | null>(props.layerKey ?? firstKey);
  const layerKey = props.layerKey !== undefined ? props.layerKey ?? firstKey : innerKey ?? firstKey;
  const setLayer = (k: string) => (props.onLayerChange ? props.onLayerChange(k) : setInnerKey(k));
  const [innerMode, setInnerMode] = useState<CompareMode>("none");
  const mode = props.compareMode ?? innerMode;
  const setMode = (m: CompareMode) => (props.onCompareModeChange ? props.onCompareModeChange(m) : setInnerMode(m));
  const [innerCmp, setInnerCmp] = useState<string | null>(null);
  const cmpKey = props.compareKey !== undefined ? props.compareKey : innerCmp;
  const setCmp = (k: string | null) => (props.onCompareChange ? props.onCompareChange(k) : setInnerCmp(k));

  const [opacity, setOpacity] = useState(1);
  const [lock, setLock] = useState<Domain | null>(null);
  const [manual, setManual] = useState<[number, number] | null>(null);
  const [toolId, setToolId] = useState<ToolId>("pan");
  const [pinned, setPinned] = useState<number | null>(null);
  const [hoverRow, setHoverRow] = useState(-1);
  const [brushRange, setBrushRange] = useState<[number, number] | null>(null);
  const shiftRef = useRef(false);

  const meta = useMemo(() => findLayer(groups, layerKey), [groups, layerKey]);
  const cmpMeta = useMemo(() => findLayer(groups, cmpKey ?? null), [groups, cmpKey]);
  const a = useLayerValues(meta, props.loadLayer);
  const b = useLayerValues(mode !== "none" ? cmpMeta : null, props.loadLayer);

  const domain = useMemo(() => (meta && a.values ? computeDomain(meta, a.values, { lock, manual }) : null), [meta, a.values, lock, manual]);
  const cmpDomain = useMemo(() => (cmpMeta && b.values ? computeDomain(cmpMeta, b.values, { lock: lock ?? (cmpMeta.scale === meta?.scale ? domain : null) }) : null), [cmpMeta, b.values, lock, domain, meta]);
  const diff = useMemo(() => (mode === "diff" && a.values && b.values ? differenceValues(a.values, b.values) : null), [mode, a.values, b.values]);
  const diffMeta = useMemo<LayerMeta | null>(
    () =>
      meta && cmpMeta && diff
        ? { ...meta, key: `diff:${meta.key}:${cmpMeta.key}`, label: `${cmpMeta.label} − ${meta.label}`, scale: "div", center: 0, zero_blank: false, labels: null, stats: { n: diff.length, lo: null, hi: null, mean: null, p1: null, p2: null, p50: null, p98: null, p99: null }, sign_note: "B − A" }
        : null,
    [meta, cmpMeta, diff],
  );
  const diffDomain = useMemo(() => (diffMeta && diff ? computeDomain(diffMeta, diff) : null), [diffMeta, diff]);

  // Reset the legend brush and a typed range when the layer changes: a range in one layer's
  // units means nothing for the next (Lock scale is the way to keep a scale across layers).
  useEffect(() => {
    setBrushRange(null);
    setManual(null);
  }, [layerKey]);

  const tool: MapTool | null = useMemo(() => {
    switch (toolId) {
      case "brush":
        return props.brush ? brushTool(grid, props.brush.edit, props.brush.settings, props.brush.onChange) : panTool();
      case "rect":
        return rectTool(grid);
      case "circle":
        return circleTool(grid);
      case "polygon":
        return polygonTool(grid);
      case "pick":
        return pickTool(grid, props.pickMode ?? { by: "zone" }, () => shiftRef.current);
      default:
        return panTool();
    }
  }, [toolId, grid, props.brush, props.pickMode]);

  const onToolResult = useCallback(
    (r: ToolResult) => {
      if (r.kind === "selection") props.onSelection?.({ spec: r.spec, mask: r.mask, label: r.label, portable: r.portable, source: toolId });
      else toast("info", r.label);
    },
    [props, toolId],
  );

  const pin = (row: number | null) => {
    setPinned(row);
    props.onPin?.(row);
  };

  const showMeta = mode === "diff" ? diffMeta : meta;
  const showValues = mode === "diff" ? diff : a.values;
  const showDomain = mode === "diff" ? diffDomain : domain;

  const describe = useCallback(
    (row: number, p: { px: number; py: number }) => {
      const ll = fmtLonLat(rasterToLonLat(grid, p.px, p.py));
      if (row < 0) return `No observation${ll ? " · " + ll : ""}`;
      const m = showMeta;
      const v = showValues ? showValues[row] : NaN;
      let txt = "—";
      if (m && Number.isFinite(v)) {
        if (m.zero_blank && v <= 0) txt = "none"; // painted as no data (e.g. untreated cells)
        else if (m.scale === "cat") txt = m.labels?.[v] ?? String(v);
        else {
          const x = v * (m.mult || 1);
          txt = `${m.scale === "div" && (m.center ?? 0) === 0 ? fmtSigned(x, m.decimals) : fmtNum(x, m.decimals)} ${unitLabel(m.unit)}`.trim();
        }
      }
      return `${m?.label ?? "Value"}: ${txt}${ll ? " · " + ll : ""} · row ${row}`;
    },
    [grid, showMeta, showValues],
  );

  const exportPng = async () => {
    try {
      const raster = canvasHandle.current?.raster();
      if (!raster || !showDomain || !showMeta) throw new Error("Map export needs canvas support in this browser");
      const S = 4;
      const legendW = 260;
      const W = grid.nx * S + legendW + 24;
      const nClasses = showDomain.kind === "cat" ? categorySwatches(showMeta, dark, showDomain.nCat).length : 0;
      const H = Math.max(grid.ny * S, 220, 20 * nClasses + 20) + 40;
      const c = document.createElement("canvas");
      c.width = W;
      c.height = H;
      const g = c.getContext("2d");
      if (!g) throw new Error("Map export needs canvas support in this browser");
      const tc = dark ? THEME_COLORS.dark : THEME_COLORS.light;
      g.fillStyle = tc.surface;
      g.fillRect(0, 0, W, H);
      g.imageSmoothingEnabled = false;
      g.drawImage(raster, 0, 30, grid.nx * S, grid.ny * S);
      g.fillStyle = tc.ink;
      g.font = "600 16px 'Public Sans', sans-serif";
      g.fillText(`${props.title ? props.title + " · " : ""}${showMeta.label}`, 8, 20);
      // legend: class swatches for categorical layers, else the ramp with its range
      const lx = grid.nx * S + 16;
      const lut = getLuts(dark)[showDomain.kind === "div" ? "div" : "seq"];
      if (showDomain.kind === "cat") {
        g.font = "12px 'Public Sans', sans-serif";
        categorySwatches(showMeta, dark, showDomain.nCat).forEach((sw, k) => {
          const y = 44 + k * 20;
          g.fillStyle = sw.color;
          g.fillRect(lx, y, 14, 14);
          g.fillStyle = tc.ink2;
          g.fillText(sw.label, lx + 20, y + 11);
        });
      } else {
        for (let i = 0; i < 200; i++) {
          const k = Math.round((i / 199) * 255) * 3;
          g.fillStyle = `rgb(${lut[k]},${lut[k + 1]},${lut[k + 2]})`;
          g.fillRect(lx + i, 50, 1.2, 16);
        }
        g.fillStyle = tc.ink2;
        g.font = "12px 'Public Sans', sans-serif";
        g.fillText(`≤ ${fmtNum(showDomain.lo, showMeta.decimals)}`, lx, 84);
        const hiTxt = `≥ ${fmtNum(showDomain.hi, showMeta.decimals)} ${unitLabel(showMeta.unit)}`;
        g.fillText(hiTxt, lx + 200 - g.measureText(hiTxt).width, 84);
        if (showDomain.kind === "div") {
          const [lowEnd, highEnd] = divergingEnds(showMeta);
          g.fillText(lowEnd, lx, 102);
          g.fillText(highEnd, lx + 200 - g.measureText(highEnd).width, 102);
        }
      }
      // scale bar
      const pxPerM = S / grid.meta.dx_m;
      const len = niceLength((grid.nx * S * 0.3) / pxPerM);
      g.fillStyle = tc.ink;
      g.fillRect(12, 30 + grid.ny * S - 14, len * pxPerM, 3);
      g.font = "12px 'IBM Plex Mono', monospace";
      g.fillText(fmtMeters(len), 16 + len * pxPerM, 30 + grid.ny * S - 10);
      const blob = await new Promise<Blob>((res, rej) => c.toBlob((bb) => (bb ? res(bb) : rej(new Error("PNG encoding failed"))), "image/png"));
      downloadBlob(blob, `${fileSlug(`${props.title ?? "map"}-${showMeta.key}`)}.png`);
    } catch (e) {
      toast("error", "Map export failed", { body: errorMessage(e) });
    }
  };

  const tools = props.tools ?? ["pan"];
  const scale = view.state.scale;
  const overlay = props.overlays?.length ? <Overlay grid={grid} features={props.overlays} scale={scale} /> : null;
  const chrome = (size: { w: number; h: number }) => <MapChrome grid={grid} scale={scale} size={size} />;
  const H = props.height ?? 520;

  const canvasProps = {
    tool,
    hatch: props.hatch,
    opacity,
    describe,
    onHover: (row: number) => setHoverRow(row),
    onPick: (row: number) => pin(row),
    onToolResult,
    overlay,
    chrome,
  };

  const allLayers = useMemo(() => groups.flatMap((g) => g.layers), [groups]);

  const sideBySide = mode === "side" && cmpMeta;
  return (
    <div className={props.noSide ? "mapview no-side" : "mapview"}>
      <div className="stack" style={{ gap: 8, minWidth: 0 }} onKeyDown={(e) => (shiftRef.current = e.shiftKey)} onPointerDown={(e) => (shiftRef.current = e.shiftKey)}>
        <LayerPicker groups={groups} value={layerKey} onChange={setLayer} label={mode === "none" ? "Layer" : "Layer A"} />
        <div className="map-controls">
          {tools.length > 1 ? (
            <div className="seg small" role="group" aria-label="Map tool">
              {tools.map((t) => (
                <button key={t} type="button" aria-pressed={toolId === t} title={TOOL_INFO[t].label} aria-label={TOOL_INFO[t].label} onClick={() => setToolId(t)} disabled={t === "brush" && !props.brush}>
                  {t}
                </button>
              ))}
            </div>
          ) : null}
          <Seg<CompareMode>
            label="Compare"
            size="small"
            value={mode}
            onChange={(m) => {
              setMode(m);
              if (m !== "none" && !cmpKey && allLayers.length > 1) setCmp(allLayers.find((l) => l.key !== layerKey)?.key ?? null);
            }}
            options={[
              { value: "none", label: "Single" },
              { value: "swipe", label: "Swipe" },
              { value: "side", label: "Side by side" },
              { value: "diff", label: "Difference", disabled: props.allowDiff === false, title: props.allowDiff === false ? "Difference needs the same grid" : "B − A" },
            ]}
          />
          {mode !== "none" ? (
            <label className="row cap">
              Layer B
              <select aria-label="Layer B" value={cmpKey ?? ""} onChange={(e) => setCmp(e.target.value || null)}>
                <option value="">Choose…</option>
                {groups.map((g) => (
                  <optgroup key={g.id} label={g.label}>
                    {g.layers.map((l) => (
                      <option key={l.key} value={l.key}>
                        {l.label}
                      </option>
                    ))}
                  </optgroup>
                ))}
              </select>
            </label>
          ) : null}
          <label className="row cap">
            Opacity
            <input type="range" min={0.2} max={1} step={0.05} value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} aria-label="Layer opacity" style={{ width: 80 }} />
          </label>
          <button type="button" className="btn small ghost" aria-pressed={!!lock} onClick={() => setLock(lock ? null : domain)} title="Keep the current colour scale when switching layers">
            {lock ? "Scale locked" : "Lock scale"}
          </button>
          <IconButton icon="image" label="Export map as PNG (4×, with legend)" size="small" onClick={() => void exportPng()} />
          <IconButton icon="target" label="Fit map" size="small" onClick={() => (canvasHandle.current ? canvasHandle.current.fit() : view.fit(grid.nx, grid.ny, 640, H))} />
        </div>
        {a.error ? <div className="callout" data-tone="crit">Could not load {meta?.label}: {errorMessage(a.error)}</div> : null}
        <div style={{ position: "relative" }}>
          {mode === "swipe" && cmpMeta ? (
            <SwipeCompare
              grid={grid}
              view={view}
              dark={dark}
              height={H}
              selection={props.selection}
              left={{ values: a.values, domain, label: meta?.label ?? "A" }}
              right={{ values: b.values, domain: cmpDomain, label: cmpMeta.label }}
              interactive={{ ...canvasProps, handleRef: canvasHandle }}
              overlay={overlay}
            />
          ) : sideBySide ? (
            <div className="side-by-side">
              <GridCanvas {...canvasProps} handleRef={canvasHandle} grid={grid} view={view} dark={dark} values={a.values} domain={domain} selection={props.selection} label={`Map A: ${meta?.label ?? ""}`} height={H} />
              <GridCanvas {...canvasProps} grid={grid} view={view} dark={dark} values={b.values} domain={cmpDomain} selection={props.selection} label={`Map B: ${cmpMeta.label}`} height={H} />
            </div>
          ) : (
            <GridCanvas
              {...canvasProps}
              handleRef={canvasHandle}
              grid={grid}
              view={view}
              dark={dark}
              values={showValues}
              domain={showDomain}
              selection={props.selection}
              label={`Map of ${showMeta?.label ?? "no layer"}`}
              height={H}
            />
          )}
          {hoverRow >= 0 && showMeta && showValues ? (
            <span className="map-readout" aria-hidden="true">
              {describe(hoverRow, { px: grid.ix[hoverRow] + 0.5, py: grid.ny - 1 - grid.iy[hoverRow] + 0.5 })}
            </span>
          ) : null}
        </div>
      </div>
      {props.noSide ? null : (
        <div className="map-side">
          {showMeta && showDomain ? (
            <section className="card">
              <h3>{showMeta.label}</h3>
              {showMeta.desc ? <p className="cap">{showMeta.desc}</p> : null}
              <Legend
                meta={showMeta}
                domain={showDomain}
                dark={dark}
                values={showValues}
                brush={brushRange}
                onBrush={
                  props.onSelection && showValues && mode !== "diff"
                    ? (r) => {
                        setBrushRange(r);
                        if (!r) {
                          props.onSelection?.({ spec: null, mask: null, label: "", portable: true, source: "legend" });
                          return;
                        }
                        // A brush reaching an end of the ramp includes the cells painted with
                        // that end colour (beyond the 2–98% clip).
                        const eps = (showDomain.hi - showDomain.lo) * 1e-9;
                        const lo = r[0] <= showDomain.lo + eps ? -Infinity : r[0];
                        const hi = r[1] >= showDomain.hi - eps ? Infinity : r[1];
                        const mask = maskFromRange(showValues, showDomain.mult, lo, hi);
                        let n = 0;
                        for (let i = 0; i < mask.length; i++) n += mask[i];
                        props.onSelection?.({ spec: null, mask, label: `${showMeta.label} ${fmtNum(r[0], showMeta.decimals)} to ${fmtNum(r[1], showMeta.decimals)} · ${n.toLocaleString("en-US")} cells`, portable: false, source: "legend" });
                      }
                    : undefined
                }
              />
              {showDomain.kind !== "cat" ? (
                <div className="row cap">
                  Range
                  <NumberField label="Range minimum" placeholder={fmtNum(showDomain.lo, showMeta.decimals)} value={manual?.[0] ?? null} nullable onChange={(v) => setManual(v === null ? null : [v, manual?.[1] ?? showDomain.hi])} />
                  <NumberField label="Range maximum" placeholder={fmtNum(showDomain.hi, showMeta.decimals)} value={manual?.[1] ?? null} nullable onChange={(v) => setManual(v === null ? null : [manual?.[0] ?? showDomain.lo, v])} />
                  {manual ? (
                    <button type="button" className="btn small ghost" onClick={() => setManual(null)}>
                      Auto
                    </button>
                  ) : null}
                </div>
              ) : null}
            </section>
          ) : (
            <section className="card">
              <p className="cap">{meta ? "Loading layer…" : "Choose a layer."}</p>
            </section>
          )}
          {pinned !== null ? (
            props.inspector ? (
              props.inspector(pinned, () => pin(null), meta)
            ) : (
              <section className="card" aria-label="Pinned cell">
                <header>
                  <h3>Pinned cell</h3>
                  <IconButton icon="x" label="Unpin cell" size="small" onClick={() => pin(null)} />
                </header>
                <p className="num">{describe(pinned, { px: grid.ix[pinned] + 0.5, py: grid.ny - 1 - grid.iy[pinned] + 0.5 })}</p>
              </section>
            )
          ) : null}
          {props.sidePanel}
        </div>
      )}
    </div>
  );
}
