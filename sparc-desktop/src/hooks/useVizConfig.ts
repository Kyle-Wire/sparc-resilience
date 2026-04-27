/**
 * useVizConfig — per-page viz config (palette, mode, extrusion, time index)
 * persisted to localStorage so the user's choices survive navigation.
 *
 * Wraps useState with a JSON-serialized localStorage backing store. Each
 * caller passes a stable `key` so different pages get independent configs.
 */
import { useCallback, useEffect, useState } from "react";
import type { ColorPalette, MapMode } from "@/components/map/SpatialMap";

export interface VizConfig {
  palette: ColorPalette;
  mode: MapMode;
  /** When set, choropleth is extruded by this property name. */
  extrudeField?: string;
  extrudeScale?: number;
  /** Active time/step index for time-varying data. */
  timeIndex?: number;
  /** Active scenario index (stage 4). */
  scenarioIndex?: number;
}

const DEFAULTS: VizConfig = {
  palette: "sparc",
  mode: "scatter",
  extrudeScale: 50,
  timeIndex: 0,
  scenarioIndex: 0,
};

const PREFIX = "sparc.viz.";

function load(key: string): VizConfig {
  try {
    const raw = localStorage.getItem(PREFIX + key);
    if (!raw) return { ...DEFAULTS };
    const parsed = JSON.parse(raw) as Partial<VizConfig>;
    return { ...DEFAULTS, ...parsed };
  } catch {
    return { ...DEFAULTS };
  }
}

export function useVizConfig(key: string): {
  config: VizConfig;
  setConfig: (next: Partial<VizConfig>) => void;
  reset: () => void;
} {
  const [config, setLocalConfig] = useState<VizConfig>(() => load(key));

  // Re-load when the key changes (e.g. user switches active artifact).
  useEffect(() => { setLocalConfig(load(key)); }, [key]);

  const setConfig = useCallback(
    (next: Partial<VizConfig>) => {
      setLocalConfig((prev) => {
        const merged = { ...prev, ...next };
        try { localStorage.setItem(PREFIX + key, JSON.stringify(merged)); } catch { /* ignore */ }
        return merged;
      });
    },
    [key],
  );

  const reset = useCallback(() => {
    try { localStorage.removeItem(PREFIX + key); } catch { /* ignore */ }
    setLocalConfig({ ...DEFAULTS });
  }, [key]);

  return { config, setConfig, reset };
}
