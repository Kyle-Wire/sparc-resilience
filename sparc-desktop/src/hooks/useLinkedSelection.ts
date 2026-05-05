/**
 * useLinkedSelection — thin wrapper around InsightsProvider's selection,
 * with helpers for the most common brushed-linking operations on a
 * GeoJSON FeatureCollection.
 */
import { useCallback, useMemo } from "react";
import { useInsights, type BBox, type LinkedSelection } from "./InsightsProvider";
import type { GeoJsonData } from "../lib/types";

export interface UseLinkedSelectionResult {
  selection: LinkedSelection;
  setIds: (ids: string[]) => void;
  setBbox: (bbox: BBox | null) => void;
  setVariable: (v: string | null) => void;
  setScenario: (s: string | null) => void;
  clear: () => void;
  /** True when nothing is selected (panels show all data). */
  isEmpty: boolean;
  /** Filter a GeoJSON FeatureCollection by current ids + bbox. */
  filter: (geo: GeoJsonData | null) => GeoJsonData | null;
  /** Test whether a single feature passes the current filter. */
  matches: (feature: GeoJsonData["features"][number]) => boolean;
}

function pointInBbox(coords: number[] | undefined, bbox: BBox): boolean {
  if (!coords || coords.length < 2) return false;
  const [x, y] = coords;
  return x >= bbox[0] && x <= bbox[2] && y >= bbox[1] && y <= bbox[3];
}

export function useLinkedSelection(): UseLinkedSelectionResult {
  const { selection, setSelection, clearSelection, hasSelection } = useInsights();

  const setIds = useCallback((ids: string[]) => setSelection({ ids }), [setSelection]);
  const setBbox = useCallback((bbox: BBox | null) => setSelection({ bbox }), [setSelection]);
  const setVariable = useCallback((variable: string | null) => setSelection({ variable }), [setSelection]);
  const setScenario = useCallback((scenario: string | null) => setSelection({ scenario }), [setSelection]);

  const idSet = useMemo(() => new Set(selection.ids), [selection.ids]);

  const matches = useCallback(
    (f: GeoJsonData["features"][number]): boolean => {
      // No filter active for spatial keys → pass.
      if (idSet.size === 0 && !selection.bbox) return true;
      const fid = ((f as { id?: string | number }).id ?? f.properties?.id) as
        | string
        | number
        | undefined;
      if (idSet.size > 0 && fid != null && idSet.has(String(fid))) return true;
      if (selection.bbox) {
        const coords = (f.geometry?.coordinates as number[] | undefined) ?? undefined;
        if (pointInBbox(coords, selection.bbox)) return true;
      }
      return false;
    },
    [idSet, selection.bbox],
  );

  const filter = useCallback(
    (geo: GeoJsonData | null): GeoJsonData | null => {
      if (!geo) return null;
      if (idSet.size === 0 && !selection.bbox) return geo;
      return { ...geo, features: geo.features.filter(matches) };
    },
    [idSet.size, selection.bbox, matches],
  );

  return {
    selection,
    setIds,
    setBbox,
    setVariable,
    setScenario,
    clear: clearSelection,
    isEmpty: !hasSelection,
    filter,
    matches,
  };
}
