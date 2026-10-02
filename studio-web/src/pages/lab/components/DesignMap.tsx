// Centre pane of the Design workbench (SPEC §7.5, §12.5): the map kit's MapView in Design mode.
// - "Preview ΔT" is the emulator's linear preview (always labelled; hatched when the preview
//   is unreliable at this scale); the edited cells are tinted.
// - The brush paints the chosen lever's Float32 edit array, clamped against the lever's exact
//   Float32 inputs (strokes overwrite, never accumulate); erase sets cells back to 0.
// - Exact result layers (ΔT, spread, extrapolation with hatching above 1, realised changes)
//   and the whole run catalogue are one click away.
// - Rect / circle / polygon / zone and hex picks become a selection offered to the edits.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getResultLayer, type Lever, type PreviewResult, type ResultField } from "../../../api/lab";
import type { LayerGroup, LayerMeta } from "../../../api/types";
import { NumberField } from "../../../components/ui/NumberField";
import { Seg } from "../../../components/ui/Seg";
import { MapView, type MapSelectionEvent } from "../../../map/MapView";
import { runLayerLoader, type LayerValues } from "../../../map/data";
import type { GridData } from "../../../map/grid";
import { BRUSH_RADII_M, type BrushSettings } from "../../../map/tools/brush";
import type { PickMode } from "../../../map/tools/pick";
import { fmtInt, fmtPct, fmtSigned, unitLabel } from "../../../theme/format";
import { leverBounds } from "../model/brush";
import type { DraftStore } from "../model/draft";
import { resultLayerMetas, syntheticMeta, withGroup } from "../model/layers";
import { TrustBadge } from "./TrustBadge";

export const PREVIEW_KEY = "lab:preview";

// Exact result layers never change once written (a stale result keeps its files), so the
// decoded arrays are kept for the session (bounded).
const resultLayers = new Map<string, Promise<Float32Array>>();
function resultLayer(resId: string, field: ResultField): Promise<Float32Array> {
  const k = `${resId}|${field}`;
  let p = resultLayers.get(k);
  if (!p) {
    p = getResultLayer(resId, field);
    p.catch(() => resultLayers.delete(k));
    resultLayers.set(k, p);
    if (resultLayers.size > 24) resultLayers.delete(resultLayers.keys().next().value as string);
  }
  return p;
}
export const brushKey = (lever: string) => `lab:brush:${lever}`;

export type DesignMapProps = {
  rid: string;
  grid: GridData;
  groups: LayerGroup[];
  levers: Lever[];
  store: DraftStore;
  /** Changes with every brush dab (live) and every committed brush state (undo, redo, load). */
  brushTick: unknown;
  brushState: unknown;
  preview: PreviewResult | null;
  previewBusy: boolean;
  previewError: string | null;
  resultId: string | null;
  realizedVars: string[];
  tint: Uint8Array | null;
  layerKey: string | null;
  onLayerChange: (key: string) => void;
  onBrushChange: () => void;
  onSelection: (e: MapSelectionEvent) => void;
};

export function DesignMap(p: DesignMapProps) {
  const { grid, levers, store, preview } = p;
  const unit = grid.meta.units.target;
  const [brushLever, setBrushLever] = useState<string>(levers[0]?.var ?? "");
  const [mode, setMode] = useState<"add" | "erase">("add");
  const [amount, setAmount] = useState<number | null>(null);
  const [radius, setRadius] = useState<number>(120);
  const [pick, setPick] = useState<PickMode>({ by: "zone" });
  const [base, setBase] = useState<{ lever: string; values: Float32Array } | null>(null);
  const [hatchMask, setHatchMask] = useState<Uint8Array | null>(null);
  const lever = levers.find((l) => l.var === brushLever) ?? levers[0];
  const loadRun = useMemo(() => runLayerLoader(p.rid, grid.meta.etag), [p.rid, grid.meta.etag]);

  useEffect(() => {
    if (!levers.some((l) => l.var === brushLever) && levers[0]) setBrushLever(levers[0].var);
  }, [levers, brushLever]);
  useEffect(() => {
    if (lever && amount === null) setAmount(lever.design_dose ?? 1);
  }, [lever, amount]);

  // The lever's exact Float32 inputs: the brush clamps against them.
  const leverMeta = useMemo(() => p.groups.flatMap((g) => g.layers).find((l) => l.key === lever?.var) ?? null, [p.groups, lever]);
  useEffect(() => {
    if (!leverMeta || leverMeta.dtype !== "float32") return;
    let live = true;
    loadRun(leverMeta).then((v) => live && v instanceof Float32Array && setBase({ lever: leverMeta.key, values: v }), () => {});
    return () => {
      live = false;
    };
  }, [leverMeta, loadRun]);

  // Brush settings are read through a ref, so the tool is not rebuilt mid-stroke.
  const settingsRef = useRef<BrushSettings | null>(null);
  settingsRef.current =
    base && lever && base.lever === lever.var ? { amount: amount ?? 0, mode, radius_m: radius, base: base.values, bounds: leverBounds(lever) } : null;
  const brushEdit = lever ? store.brushLayers.array(lever.var) : null;
  const onBrushRows = p.onBrushChange;
  const brush = useMemo(
    () =>
      brushEdit && settingsRef.current
        ? {
            edit: brushEdit,
            settings: () => settingsRef.current ?? { amount: 0, mode: "erase" as const, radius_m: 60, base: new Float32Array(grid.n), bounds: [-Infinity, Infinity] as [number, number] },
            onChange: () => {
              store.brushLayers.touch(lever!.var);
              onBrushRows();
            },
          }
        : undefined,
    // Rebuilt when the lever, its base or the edit array changes; amount/mode/radius go through the ref.
    [brushEdit, base, lever, grid.n, store, onBrushRows],
  );

  // Client-held layers: the preview and one brush layer per brushed lever.
  const designGroup = useMemo<LayerGroup>(() => {
    const layers: LayerMeta[] = [];
    layers.push(
      syntheticMeta(
        {
          key: PREVIEW_KEY,
          label: "Preview ΔT (emulator, linear)",
          unit,
          desc: "Linear emulator preview: it never saturates or clips to support. Run exact for decisions.",
          sign_note: "negative = cooler",
        },
        preview?.delta ?? null,
      ),
    );
    for (const l of new Set([...store.brushLayers.levers(), ...(lever ? [lever.var] : [])])) {
      const lv = levers.find((x) => x.var === l);
      layers.push(syntheticMeta({ key: brushKey(l), label: `Brush edit: ${lv?.label ?? l}`, unit: lv?.unit ?? "", desc: "Per-cell change painted with the brush." }, store.brushLayers.array(l)));
    }
    return { id: "design", label: "Design", layers };
    // brushTick / brushState: the brush layers' values change with strokes, undo and redo
  }, [preview, unit, store, lever, levers, p.brushTick, p.brushState]);

  const resultGroup = useMemo<LayerGroup | null>(() => {
    if (!p.resultId) return null;
    const units = Object.fromEntries(levers.map((l) => [l.var, l.unit]));
    return { id: "result", label: "Exact result", layers: resultLayerMetas(p.resultId, unit, p.realizedVars, units) };
  }, [p.resultId, p.realizedVars, levers, unit]);

  const groups = useMemo(() => withGroup(withGroup(p.groups, resultGroup), designGroup), [p.groups, resultGroup, designGroup]);

  const previewRef = useRef(preview);
  previewRef.current = preview;
  const load = useCallback(
    async (meta: LayerMeta): Promise<LayerValues> => {
      if (meta.key === PREVIEW_KEY) return previewRef.current?.delta ?? new Float32Array(grid.n).fill(NaN);
      if (meta.key.startsWith("lab:brush:")) return Float32Array.from(store.brushLayers.array(meta.key.slice("lab:brush:".length)));
      const m = /^res:([^:]+):(.+)$/.exec(meta.key);
      if (m) return resultLayer(m[1], m[2] as ResultField);
      return loadRun(meta);
    },
    [grid.n, store, loadRun],
  );

  // Hatching: extrapolated cells (score > 1) on result layers; the edited cells when the
  // preview is flagged unreliable at this scale.
  const current = p.layerKey ?? PREVIEW_KEY;
  useEffect(() => {
    let live = true;
    const m = /^res:([^:]+):/.exec(current);
    if (m) {
      resultLayer(m[1], "extrapolation").then(
        (ex) => {
          if (!live) return;
          const h = new Uint8Array(ex.length);
          for (let i = 0; i < ex.length; i++) h[i] = ex[i] > 1 ? 1 : 0;
          setHatchMask(h);
        },
        () => live && setHatchMask(null),
      );
    } else setHatchMask(current === PREVIEW_KEY && preview?.summary.hatched ? preview.edited : null);
    return () => {
      live = false;
    };
  }, [current, preview]);

  const s = preview?.summary;
  const side = (
    <>
      <section className="card" aria-label="Preview">
        <header>
          <h3>Preview</h3>
          {s ? <TrustBadge trust={s.trust} /> : null}
        </header>
        {p.previewError ? (
          <p className="cap">{p.previewError}</p>
        ) : s ? (
          <dl className="kv">
            <dt>Edited cells</dt>
            <dd className="num">{fmtInt(s.n_edited)}</dd>
            <dt>Mean in edited cells</dt>
            <dd className="num">
              {fmtSigned(s.edited_mean, 3)} {unitLabel(unit)}
            </dd>
            <dt>City mean</dt>
            <dd className="num">
              {fmtSigned(s.mean, 4)} {unitLabel(unit)}
            </dd>
            <dt>Cooling outside the edits</dt>
            <dd className="num">{fmtPct(s.outside_share)}</dd>
          </dl>
        ) : (
          <p className="cap">{p.previewBusy ? "Computing the preview…" : "Edit or brush to see the preview."}</p>
        )}
        {s?.hatched ? (
          <p className="callout" role="note" data-preview-banner="true">
            {s.reasons[0] ?? "Preview unreliable at this scale"} — run exact.
          </p>
        ) : null}
        <p className="cap">Preview is linear: it never saturates or clips to support.</p>
      </section>
      <section className="card" aria-label="Brush">
        <h3>Brush</h3>
        {lever ? (
          <div className="stack" style={{ gap: 6 }}>
            <select aria-label="Brush lever" value={lever.var} onChange={(e) => setBrushLever(e.target.value)}>
              {levers.map((l) => (
                <option key={l.var} value={l.var}>
                  {l.label}
                </option>
              ))}
            </select>
            <Seg<"add" | "erase"> label="Brush mode" size="small" value={mode} onChange={setMode} options={[{ value: "add", label: "Add" }, { value: "erase", label: "Erase" }]} />
            {mode === "add" ? <NumberField label="Brush change" unit={lever.unit} value={amount} onChange={(v) => setAmount(v)} /> : null}
            <Seg<number> label="Brush radius" size="small" value={radius} onChange={setRadius} options={BRUSH_RADII_M.map((r) => ({ value: r, label: `${r} m` }))} />
            <p className="cap">
              {base?.lever === lever.var ? "Choose the brush tool above the map and paint. Repainting replaces, never adds up." : leverMeta ? "Loading the lever's inputs…" : "This lever's input layer is not in the run catalogue."}
            </p>
          </div>
        ) : (
          <p className="cap">No actionable levers.</p>
        )}
      </section>
      <section className="card" aria-label="Pick">
        <h3>Pick</h3>
        <Seg<string>
          label="Pick by"
          size="small"
          value={pick.by === "zone" ? "zone" : String(pick.size_m)}
          onChange={(v) => setPick(v === "zone" ? { by: "zone" } : { by: "hex", size_m: Number(v) as 250 | 500 })}
          options={[
            { value: "zone", label: "Zone", disabled: !grid.meta.zones.length },
            { value: "250", label: "Hex 250 m" },
            { value: "500", label: "Hex 500 m" },
          ]}
        />
        <p className="cap">Shift-click adds to a pick. Picks and drawn shapes are offered to the edits on the left.</p>
      </section>
    </>
  );

  return (
    <MapView
      grid={grid}
      groups={groups}
      loadLayer={load}
      layerKey={current}
      onLayerChange={p.onLayerChange}
      selection={p.tint ?? preview?.edited ?? null}
      hatch={hatchMask}
      tools={["pan", "brush", "rect", "circle", "polygon", "pick"]}
      brush={brush}
      pickMode={pick}
      onSelection={p.onSelection}
      sidePanel={side}
      height={560}
      title="Scenario design"
    />
  );
}
