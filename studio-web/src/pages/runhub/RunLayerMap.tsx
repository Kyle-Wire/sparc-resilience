// A compact map of a few run layers inside an analysis tab (Budget dose and closed-loop maps,
// Planner sites): the foundation MapView over the run grid with the catalogue filtered to
// the given keys, linking to the full Map explorer.
import { useMemo, useState } from "react";
import type { LayerGroup } from "../../api/types";
import { EmptyState } from "../../components/ui/EmptyState";
import { MapView } from "../../map/MapView";
import type { OverlayFeature } from "../../map/Overlay";
import { runLayerLoader, useRunGrid, useRunLayers } from "../../map/data";
import { Link } from "../../router";

export function RunLayerMap({ rid, keys, title, height = 380, overlays }: { rid: string; keys: string[]; title: string; height?: number; overlays?: OverlayFeature[] }) {
  const grid = useRunGrid(rid);
  const layers = useRunLayers(rid);
  const keyList = keys.join("\u0000");
  const groups = useMemo<LayerGroup[]>(() => {
    const want = new Set(keyList.split("\u0000"));
    return (layers.data ?? []).map((g) => ({ ...g, layers: g.layers.filter((l) => want.has(l.key)) })).filter((g) => g.layers.length);
  }, [layers.data, keyList]);
  const load = useMemo(() => (grid.data ? runLayerLoader(rid, grid.data.meta.etag) : null), [rid, grid.data]);
  const [key, setKey] = useState<string | null>(null);
  if (grid.error || layers.error) return <EmptyState error={grid.error ?? layers.error} />;
  if (!grid.data || !layers.data || !load) return <p className="cap">Loading the map…</p>;
  if (!groups.length) return <p className="cap">None of these layers ({keys.join(", ")}) is in this run.</p>;
  // the first of `keys` the run has (callers list them in order of preference)
  const present = new Set(groups.flatMap((g) => g.layers.map((l) => l.key)));
  const cur = key ?? keys.find((k) => present.has(k)) ?? groups[0].layers[0].key;
  return (
    <div className="stack" style={{ gap: 6 }}>
      <MapView grid={grid.data} groups={groups} loadLayer={load} layerKey={cur} onLayerChange={setKey} height={height} title={title} overlays={overlays} />
      <p className="cap">
        <Link to={`/r/${encodeURIComponent(rid)}/map?layer=${encodeURIComponent(cur)}`}>Open in the Map explorer</Link> for swipe, tools and the cell inspector.
      </p>
    </div>
  );
}
