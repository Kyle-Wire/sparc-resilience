/**
 * ScenarioLibraryTimeline — chronological tree of saved scenarios.
 *
 * Renders entries as a vertical timeline grouped by lineage (root → children).
 * Click a node to call `onSelect`. Pure SVG/HTML, no external deps.
 */
import { useMemo } from "react";
import type { ScenarioLibraryEntry, ScenarioTimeline } from "@/lib/api";

interface ScenarioLibraryTimelineProps {
  data: ScenarioTimeline | null;
  onSelect?: (entry: ScenarioLibraryEntry) => void;
  selectedId?: string;
  height?: number;
}

interface NodeRow {
  entry: ScenarioLibraryEntry;
  depth: number;
}

function flattenTree(timeline: ScenarioTimeline): NodeRow[] {
  const byId = new Map<string, ScenarioLibraryEntry>();
  for (const e of timeline.entries) byId.set(e.id, e);
  const out: NodeRow[] = [];
  const visit = (id: string, depth: number) => {
    const e = byId.get(id);
    if (!e) return;
    out.push({ entry: e, depth });
    for (const cid of timeline.children[id] ?? []) visit(cid, depth + 1);
  };
  for (const root of timeline.roots) visit(root, 0);
  return out;
}

export default function ScenarioLibraryTimeline({
  data,
  onSelect,
  selectedId,
  height = 360,
}: ScenarioLibraryTimelineProps) {
  const rows = useMemo(() => (data ? flattenTree(data) : []), [data]);

  if (!data || !rows.length) {
    return (
      <div style={{ padding: 20, fontSize: 12, color: "#888" }}>
        No saved scenarios.
      </div>
    );
  }

  return (
    <div
      style={{
        height,
        overflowY: "auto",
        border: "1px solid var(--line, rgba(0,0,0,0.08))",
        borderRadius: 6,
        background: "rgba(255,255,255,0.65)",
      }}
    >
      <ul style={{ listStyle: "none", margin: 0, padding: "8px 0" }}>
        {rows.map(({ entry, depth }) => {
          const selected = entry.id === selectedId;
          return (
            <li key={entry.id}>
              <button
                type="button"
                onClick={() => onSelect?.(entry)}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  width: "100%",
                  padding: "6px 12px",
                  paddingLeft: 12 + depth * 18,
                  background: selected ? "rgba(122, 58, 58, 0.10)" : "transparent",
                  border: "none",
                  borderLeft: selected ? "3px solid #7a3a3a" : "3px solid transparent",
                  textAlign: "left",
                  cursor: "pointer",
                  fontSize: 12,
                  color: "#222",
                }}
              >
                <span
                  aria-hidden
                  style={{
                    display: "inline-block",
                    width: 8,
                    height: 8,
                    borderRadius: "50%",
                    background: depth === 0 ? "#7a3a3a" : "#aaa",
                    flexShrink: 0,
                  }}
                />
                <span style={{ display: "flex", flexDirection: "column", gap: 1, minWidth: 0 }}>
                  <span style={{ fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {entry.comment || entry.id.slice(0, 8)}
                  </span>
                  <span style={{ fontSize: 10, color: "#777" }}>
                    {entry.author} · {new Date(entry.created_utc).toLocaleString()}
                  </span>
                </span>
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
