// Centre pane of the Design workbench (SPEC §7.5, §12.5): the map kit's MapView in Design mode.
// - "Quick estimate: temperature change" is the emulator's linear preview (always labelled;
//   hatched when the preview is unreliable at this scale); the edited cells are tinted.
// - The brush paints the chosen lever's Float32 edit array, clamped against the lever's exact
//   Float32 inputs (strokes overwrite, never accumulate); erase sets cells back to 0.
// - Exact result layers (ΔT, spread, extrapolation with hatching above 1, realised changes)
//   are one click away.
// - Rect / circle / polygon / zone and hex picks become a selection offered to the edits.
// Planner chrome by default (phase 3): the planner's layer groups only (Design, Temperature,
// Planner & people, Full model result), the ramp-only legend, no compare / opacity / lock-scale
// controls, and the Brush and Pick settings only while step 02 "Draw on the map" is chosen (or that
// tool is in use). Expert mode (Advanced options) shows the whole run catalogue and every control.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useStore } from "zustand";
import { getResultLayer, type Lever, type PreviewResult, type ResultField } from "../../../api/lab";
import type { LayerGroup, LayerMeta } from "../../../api/types";
import { NumberField } from "../../../components/ui/NumberField";
import { Seg } from "../../../components/ui/Seg";
import { InfoTip } from "../../../components/ui/InfoTip";
import { MapView, type MapSelectionEvent } from "../../../map/MapView";
import { runLayerLoader, withoutPerModelLayers, type LayerValues } from "../../../map/data";
import type { GridData } from "../../../map/grid";
import { BRUSH_RADII_M, type BrushSettings } from "../../../map/tools/brush";
import type { PickMode } from "../../../map/tools/pick";
import type { ToolId } from "../../../map/tools/types";
import { fmtInt, fmtPct, fmtSigned, unitLabel } from "../../../theme/format";
import { applySupportRule, leverBounds } from "../model/brush";
import { BRUSH_LABEL, type DraftStore } from "../model/draft";
import { resultLayerMetas, syntheticMeta, withGroup } from "../model/layers";
import { HATCH_KEY } from "../model/quick";
import { leverUnitWord, leverWords } from "../model/words";
import { TrustBadge } from "./TrustBadge";
import "./flow/quick.css";

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

/** Layer groups a planner sees on the Design map (expert mode shows the whole run catalogue). */
export const PLANNER_GROUPS: readonly string[] = ["design", "temperature", "planner", "result"];

/**
 * The planner's groups, plus the group of the layer on show (a `?layer=` link may name any layer).
 * Per-model layers (`oof_*`, `resid_*`) are left out unless on show: they belong to expert mode.
 */
export function plannerGroups(groups: readonly LayerGroup[], current: string | null): LayerGroup[] {
  return withoutPerModelLayers(
    groups.filter((g) => PLANNER_GROUPS.includes(g.id) || (current !== null && g.layers.some((l) => l.key === current))),
    current,
  );
}

/** Plain words of the map's Pick setting (the pick tool picks a zone or a hexagon of this size). */
export const PICK_WORDS = { zone: "Neighbourhood", hex250: "Small blocks (250 m)", hex500: "Large blocks (500 m)" } as const;

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
  /** Render the "Preview" summary card in the map's side panel (default true). The Design step
   *  flow passes false: its own preview pane carries the quick estimate. */
  showPreviewCard?: boolean;
  /** Step 02 "Draw on the map" is chosen: show the Brush and Pick settings (they also show while
   *  the Paint or Pick area tool is in use, and always in expert mode). Default true. */
  drawing?: boolean;
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
  const [tool, setTool] = useState<ToolId>("pan");
  const expert = useStore(store, (s) => !!s.doc.options?.expert);
  const lever = levers.find((l) => l.var === brushLever) ?? levers[0];
  // A brush edit saved in an earlier session is a per-cell blob edit of the doc; new strokes
  // become a second one and add to it (the saved strokes cannot be re-opened for painting).
  const savedBrush = useStore(store, (s) => s.doc.edits.findIndex((e) => e.mode === "per_cell" && e.label === BRUSH_LABEL && e.lever === lever?.var && !!e.per_cell_ref?.startsWith("blob:")));
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
            onChange: (rows: number[]) => {
              // The map kit clamps to [lo, hi]; cells already outside follow the engine's rule.
              if (settingsRef.current) applySupportRule(brushEdit, rows, settingsRef.current);
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
          label: "Quick estimate: temperature change",
          unit,
          desc: "Purple = cooler, orange = warmer. A quick guide — run the full model for a firm answer.",
          sign_note: "negative = cooler",
        },
        preview?.delta ?? null,
      ),
    );
    for (const l of new Set([...store.brushLayers.levers(), ...(lever ? [lever.var] : [])])) {
      const lv = levers.find((x) => x.var === l);
      layers.push(syntheticMeta({ key: brushKey(l), label: `Painted: ${leverWords(lv, l).name}`, unit: lv?.unit ?? "", desc: "The change painted place by place with the brush (a per-cell edit)." }, store.brushLayers.array(l)));
    }
    // "Your changes", not "Design": the page's own sub-tab is called Design
    return { id: "design", label: "Your changes", layers };
    // brushTick / brushState: the brush layers' values change with strokes, undo and redo
  }, [preview, unit, store, lever, levers, p.brushTick, p.brushState]);

  const resultGroup = useMemo<LayerGroup | null>(() => {
    if (!p.resultId) return null;
    const units = Object.fromEntries(levers.map((l) => [l.var, l.unit]));
    const names = Object.fromEntries(levers.map((l) => [l.var, leverWords(l).name]));
    return { id: "result", label: "Full model result", layers: resultLayerMetas(p.resultId, unit, p.realizedVars, units, names) };
  }, [p.resultId, p.realizedVars, levers, unit]);

  const allGroups = useMemo(() => withGroup(withGroup(p.groups, resultGroup), designGroup), [p.groups, resultGroup, designGroup]);

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
  const groups = useMemo(() => (expert ? allGroups : plannerGroups(allGroups, current)), [expert, allGroups, current]);
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

  // Map key for the hatching (the legend shows the colour ramp only): rough quick estimate on the
  // preview layer, beyond measured conditions on a full-model layer.
  const hatchShown = useMemo(() => !!hatchMask && hatchMask.some((v) => v === 1), [hatchMask]);
  const hatchKey = hatchShown ? HATCH_KEY[/^res:/.test(current) ? "result" : "preview"] : null;

  const s = preview?.summary;
  const drawing = p.drawing ?? true;
  const showBrush = drawing || expert || tool === "brush";
  const showPick = drawing || expert || tool === "pick";
  const unitWord = leverUnitWord(lever);
  const side = (
    <>
      {hatchKey ? (
        // one line: the hatch swatch and what it means (the right pane carries the trust message)
        <p className="qe-key" role="note" aria-label="Map key" data-hatch-key="true">
          <span className="qe-hatch" aria-hidden="true" />
          {hatchKey.label}
          <InfoTip term="hatched" label={hatchKey.label.toLowerCase()} text={hatchKey.note} />
        </p>
      ) : null}
      {p.showPreviewCard === false ? null : (
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
      )}
      {showBrush ? (
        <section className="card" aria-label="Brush">
          <h3>Brush</h3>
          {lever ? (
            <div className="stack" style={{ gap: 6 }}>
              <select aria-label="Brush lever" value={lever.var} onChange={(e) => setBrushLever(e.target.value)}>
                {levers.map((l) => (
                  <option key={l.var} value={l.var}>
                    {leverWords(l).name}
                  </option>
                ))}
              </select>
              <Seg<"add" | "erase"> label="Brush mode" size="small" value={mode} onChange={setMode} options={[{ value: "add", label: "Add" }, { value: "erase", label: "Erase" }]} />
              {mode === "add" ? <NumberField label="Brush change" unit={unitWord || null} value={amount} onChange={(v) => setAmount(v)} /> : null}
              <Seg<number> label="Brush radius" size="small" value={radius} onChange={setRadius} options={BRUSH_RADII_M.map((r) => ({ value: r, label: `${r} m` }))} />
              <p className="cap">
                {base?.lever === lever.var
                  ? "Choose Paint above the map and paint. Painting a place again replaces the amount, it never adds up."
                  : leverMeta
                    ? "Loading today's values for this change…"
                    : "This change cannot be painted: its map layer is not in this run."}
              </p>
              {savedBrush >= 0 ? (
                <p className="cap" data-saved-brush="true">
                  Change {savedBrush + 1} holds the painting of {leverWords(lever).name.toLowerCase()} from an earlier session. New painting is saved as another change and adds to it; remove change {savedBrush + 1} to start over.
                </p>
              ) : null}
            </div>
          ) : (
            <p className="cap">Nothing in this run can be painted.</p>
          )}
        </section>
      ) : null}
      {showPick ? (
        <section className="card" aria-label="Pick">
          <h3>Pick</h3>
          <Seg<string>
            label="Pick by"
            size="small"
            value={pick.by === "zone" ? "zone" : String(pick.size_m)}
            onChange={(v) => setPick(v === "zone" ? { by: "zone" } : { by: "hex", size_m: Number(v) as 250 | 500 })}
            options={[
              { value: "zone", label: PICK_WORDS.zone, disabled: !grid.meta.zones.length },
              { value: "250", label: PICK_WORDS.hex250 },
              { value: "500", label: PICK_WORDS.hex500 },
            ]}
          />
          <p className="cap">Choose Pick area above the map, then click the map. Shift-click adds to a pick. What you draw or pick can be used as the place of a change.</p>
        </section>
      ) : null}
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
      onToolChange={setTool}
      brush={brush}
      pickMode={pick}
      onSelection={p.onSelection}
      simple={!expert}
      compactLegend={!expert}
      sidePanel={side}
      height={560}
      title="Scenario design"
    />
  );
}
