// Live heat risk of the draft (the emulator preview): the per-cell ΔT of the latest preview applied to
// the run's observed temperatures, turned into the NWS heat index with the campaign dewpoint, and the
// residents (people layer) moved across the categories. Only when the run has campaign humidity; the
// exact result's impacts carry the verified numbers.
import { useEffect, useMemo, useState } from "react";
import { useView, type ScenarioRow } from "../../../api/runs";
import type { LayerGroup } from "../../../api/types";
import { runLayerLoader } from "../../../map/data";
import type { GridData } from "../../../map/grid";
import { findLayer } from "../../../map/LayerPicker";
import { heatShift, type HeatShift } from "../../../theme/heat";

export type HeatPreview = { shift: HeatShift; dewpointC: number; source: string | null; caveats: string[] };

/** Caveats from the run's verdicts for the levers a draft edits: a lever whose every configured
 *  single-lever scenario is "Not established" (or only "Direction only") is named, so a preview
 *  never reads as more certain than the evidence. */
export function leverCaveats(rows: ScenarioRow[] | null | undefined, levers: string[], label: (v: string) => string): string[] {
  const out: string[] = [];
  for (const v of levers) {
    const vs = (rows ?? []).filter((r) => r.kind === "single" && r.lever === v && r.verdict?.verdict);
    if (!vs.length) continue;
    const kinds = new Set(vs.map((r) => r.verdict!.verdict));
    if (kinds.size === 1 && kinds.has("not_established"))
      out.push(`${label(v)}: no configured scenario of this lever is established on this run (Not established: indistinguishable from no effect). Treat its heat-risk change as unproven.`);
    else if (!kinds.has("robust") && kinds.has("direction")) out.push(`${label(v)}: its configured scenarios support the direction of the change but not its size (Direction only).`);
  }
  return out;
}

export function useHeatPreview(rid: string, grid: GridData, groups: LayerGroup[], delta: Float32Array | null, levers: string[] = [], label: (v: string) => string = (v) => v): HeatPreview | null {
  const heat = useView(rid, "heat");
  const scen = useView(rid, "scenarios");
  const leverKey = levers.join("\u0000");
  const hum = heat.data?.sections.humidity ?? null;
  const dewpoint = hum?.campaign_dewpoint_C ?? null;
  const [arrays, setArrays] = useState<{ obs: ArrayLike<number>; people: ArrayLike<number> | null } | null>(null);
  const obsMeta = useMemo(() => findLayer(groups, "obs"), [groups]);
  const peopleMeta = useMemo(() => findLayer(groups, "people"), [groups]);
  useEffect(() => {
    if (dewpoint === null || !obsMeta) return;
    let live = true;
    const load = runLayerLoader(rid, grid.meta.etag);
    void Promise.all([load(obsMeta), peopleMeta ? load(peopleMeta) : Promise.resolve(null)])
      .then(([obs, people]) => {
        if (live) setArrays({ obs, people });
      })
      .catch(() => {
        if (live) setArrays(null);
      });
    return () => {
      live = false;
    };
  }, [rid, grid.meta.etag, dewpoint, obsMeta, peopleMeta]);
  const rows = scen.data?.sections.rows ?? null;
  const caveats = useMemo(() => leverCaveats(rows, leverKey ? leverKey.split("\u0000") : [], label), [rows, leverKey, label]);
  return useMemo(() => {
    if (!delta || !arrays || dewpoint === null || arrays.obs.length !== delta.length) return null;
    const people = arrays.people && arrays.people.length === delta.length ? arrays.people : null;
    return { shift: heatShift(arrays.obs, delta, people, grid.meta.units.target, dewpoint), dewpointC: dewpoint, source: hum?.campaign_source ?? null, caveats };
  }, [delta, arrays, dewpoint, grid.meta.units.target, hum?.campaign_source, caveats]);
}
