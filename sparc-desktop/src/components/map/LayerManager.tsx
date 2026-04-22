import { useCallback, useEffect, useMemo, useState } from "react";
import { getContextLayers, type ContextLayer, type ContextLayerCatalog } from "@/lib/api";

export interface ActiveLayer {
  layer: ContextLayer;
  opacity: number;
}

/**
 * useContextLayers — fetch the catalog (optionally for *domain*) and
 * manage the active set + per-layer opacity. Defaults are seeded from
 * `catalog.defaults` returned by the backend.
 */
export function useContextLayers(domain?: string) {
  const [catalog, setCatalog] = useState<ContextLayerCatalog | null>(null);
  const [activeIds, setActiveIds] = useState<Set<string>>(() => new Set());
  const [opacities, setOpacities] = useState<Record<string, number>>({});
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getContextLayers(domain)
      .then((c) => {
        if (cancelled) return;
        setCatalog(c);
        setActiveIds(new Set(c.defaults));
        setOpacities(Object.fromEntries(c.layers.map((l) => [l.id, l.opacity_default ?? 1.0])));
      })
      .catch((e) => !cancelled && setError(e instanceof Error ? e.message : String(e)));
    return () => { cancelled = true; };
  }, [domain]);

  const toggle = useCallback((id: string) => {
    setActiveIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const setOpacity = useCallback((id: string, v: number) => {
    setOpacities((prev) => ({ ...prev, [id]: v }));
  }, []);

  const active: ActiveLayer[] = useMemo(() => {
    if (!catalog) return [];
    return catalog.layers
      .filter((l) => activeIds.has(l.id))
      .map((l) => ({ layer: l, opacity: opacities[l.id] ?? l.opacity_default ?? 1.0 }));
  }, [catalog, activeIds, opacities]);

  return { catalog, active, activeIds, opacities, toggle, setOpacity, error };
}

interface LayerManagerProps {
  domain?: string;
  ctx: ReturnType<typeof useContextLayers>;
  compact?: boolean;
}

/**
 * LayerManager — collapsible panel UI for choosing context overlays.
 * Pass a single `ctx` (from `useContextLayers`) so the parent can also
 * read `ctx.active` for the SpatialMap `contextLayers` prop.
 */
export default function LayerManager({ ctx, compact }: LayerManagerProps) {
  const [open, setOpen] = useState(true);
  const { catalog, activeIds, opacities, toggle, setOpacity, error } = ctx;

  if (error) {
    return <div style={{ fontSize: 11, color: "var(--crimson)" }}>Layer catalog: {error}</div>;
  }
  if (!catalog) {
    return <div style={{ fontSize: 11, color: "var(--muted)" }}>Loading layer catalog…</div>;
  }

  const basemaps = catalog.layers.filter((l) => l.kind === "basemap");
  const overlays = catalog.layers.filter((l) => l.kind === "overlay");

  const headerStyle: React.CSSProperties = {
    fontSize: 9.5, color: "var(--muted)", letterSpacing: "0.1em",
    textTransform: "uppercase", marginBottom: 4,
  };

  return (
    <div style={{ background: "rgba(255,255,255,.95)", border: "1px solid var(--line)", borderRadius: 4, padding: compact ? 6 : 10, fontSize: 11, minWidth: compact ? 180 : 220 }}>
      <div
        onClick={() => setOpen((o) => !o)}
        style={{ display: "flex", justifyContent: "space-between", alignItems: "center", cursor: "pointer", marginBottom: open ? 8 : 0 }}
      >
        <strong style={{ fontSize: 11 }}>Context layers</strong>
        <span style={{ color: "var(--muted)", fontSize: 10 }}>{activeIds.size}/{catalog.layers.length} {open ? "▾" : "▸"}</span>
      </div>
      {open && (
        <>
          {basemaps.length > 0 && (
            <>
              <div style={headerStyle}>Basemap</div>
              {basemaps.map((l) => (
                <LayerRow
                  key={l.id} layer={l} active={activeIds.has(l.id)}
                  opacity={opacities[l.id] ?? l.opacity_default ?? 1}
                  onToggle={() => toggle(l.id)}
                  onOpacity={(v) => setOpacity(l.id, v)}
                />
              ))}
            </>
          )}
          {overlays.length > 0 && (
            <>
              <div style={{ ...headerStyle, marginTop: 8 }}>Overlays</div>
              {overlays.map((l) => (
                <LayerRow
                  key={l.id} layer={l} active={activeIds.has(l.id)}
                  opacity={opacities[l.id] ?? l.opacity_default ?? 1}
                  onToggle={() => toggle(l.id)}
                  onOpacity={(v) => setOpacity(l.id, v)}
                />
              ))}
            </>
          )}
          {catalog.roadmap.length > 0 && (
            <details style={{ marginTop: 8 }}>
              <summary style={{ ...headerStyle, cursor: "pointer", marginBottom: 0 }}>Roadmap (not yet wired)</summary>
              <ul style={{ margin: "6px 0 0", paddingLeft: 16, color: "var(--muted)", fontSize: 10.5, lineHeight: 1.5 }}>
                {catalog.roadmap.map((r) => (
                  <li key={r.id}><strong>{r.name}</strong> — {r.description}</li>
                ))}
              </ul>
            </details>
          )}
        </>
      )}
    </div>
  );
}

function LayerRow({ layer, active, opacity, onToggle, onOpacity }: {
  layer: ContextLayer; active: boolean; opacity: number;
  onToggle: () => void; onOpacity: (v: number) => void;
}) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 2, padding: "4px 0", borderBottom: "1px dotted var(--line)" }}>
      <label style={{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer" }} title={layer.description}>
        <input type="checkbox" checked={active} onChange={onToggle} />
        <span style={{ flex: 1 }}>{layer.name}</span>
        {layer.recommended && <span style={{ fontSize: 9, color: "var(--purple)" }}>★</span>}
      </label>
      {active && (
        <input
          type="range" min={0} max={1} step={0.05} value={opacity}
          onChange={(e) => onOpacity(parseFloat(e.target.value))}
          style={{ width: "100%", margin: 0 }}
          title={`opacity ${Math.round(opacity * 100)}%`}
        />
      )}
      <div style={{ fontSize: 9, color: "var(--muted)" }}>{layer.attribution}</div>
    </div>
  );
}
