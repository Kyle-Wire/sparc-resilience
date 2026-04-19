import type { DagDefinition } from "@/lib/types";

interface DagDisplayNode {
  id: string;
  x: number;
  y: number;
  label: string;
  type: "t" | "m" | "c" | "o";
}

const DEMO_NODES: DagDisplayNode[] = [
  { id: "canopy", x: 60, y: 60, label: "Canopy", type: "t" },
  { id: "impv", x: 60, y: 130, label: "Impervious", type: "t" },
  { id: "albedo", x: 60, y: 200, label: "Albedo", type: "t" },
  { id: "ndvi", x: 220, y: 95, label: "NDVI", type: "m" },
  { id: "elev", x: 220, y: 165, label: "Elevation", type: "c" },
  { id: "water", x: 220, y: 225, label: "Dist. water", type: "c" },
  { id: "aat", x: 380, y: 130, label: "AAT_z", type: "o" },
];

const DEMO_EDGES: [string, string][] = [
  ["canopy", "ndvi"], ["canopy", "aat"], ["impv", "aat"], ["albedo", "aat"],
  ["ndvi", "aat"], ["elev", "aat"], ["water", "aat"], ["canopy", "impv"],
];

function typeToShort(t: string): "t" | "m" | "c" | "o" {
  if (t === "treatment") return "t";
  if (t === "mediator") return "m";
  if (t === "outcome") return "o";
  return "c";
}

function layoutFromDag(dag: DagDefinition): { nodes: DagDisplayNode[]; edges: [string, string][] } {
  const grouped: Record<string, typeof dag.nodes> = { t: [], m: [], c: [], o: [] };
  for (const n of dag.nodes) {
    grouped[typeToShort(n.type)].push(n);
  }

  const xPositions: Record<string, number> = { t: 60, m: 220, c: 220, o: 380 };
  const nodes: DagDisplayNode[] = [];

  for (const [short, list] of Object.entries(grouped)) {
    const x = xPositions[short] ?? 220;
    const spacing = 280 / Math.max(list.length, 1);
    list.forEach((n, i) => {
      nodes.push({
        id: n.name,
        x,
        y: 40 + (i + 0.5) * spacing,
        label: n.name.length > 12 ? n.name.slice(0, 10) + "…" : n.name,
        type: short as DagDisplayNode["type"],
      });
    });
  }

  const edges: [string, string][] = dag.edges.map((e) => [e.parent, e.child]);
  return { nodes, edges };
}

const color = (t: DagDisplayNode["type"]): string =>
  t === "t" ? "var(--crimson)" : t === "m" ? "var(--purple)" : t === "o" ? "var(--ink)" : "var(--muted)";

interface DagMiniProps {
  dag?: DagDefinition | null;
}

export default function DagMini({ dag }: DagMiniProps) {
  const { nodes, edges } = dag && dag.nodes.length > 0
    ? layoutFromDag(dag)
    : { nodes: DEMO_NODES, edges: DEMO_EDGES };

  const N = Object.fromEntries(nodes.map((n) => [n.id, n]));

  return (
    <svg viewBox="0 0 460 280" style={{ width: "100%", height: 260 }}>
      <defs>
        <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M0,0 L10,5 L0,10 z" fill="rgba(0,0,0,0.45)" />
        </marker>
      </defs>
      {edges.map(([a, b], i) => {
        if (!N[a] || !N[b]) return null;
        return (
          <line
            key={i}
            x1={N[a].x + 34}
            y1={N[a].y}
            x2={N[b].x - 34}
            y2={N[b].y}
            stroke="rgba(0,0,0,0.35)"
            strokeWidth="1"
            strokeDasharray="3 3"
            markerEnd="url(#arr)"
          />
        );
      })}
      {nodes.map((n) => (
        <g key={n.id}>
          <rect
            x={n.x - 42}
            y={n.y - 14}
            width="84"
            height="28"
            rx="3"
            fill="#fff"
            stroke={color(n.type)}
            strokeWidth={n.type === "o" ? 2 : 1.2}
          />
          <text
            x={n.x}
            y={n.y + 4}
            textAnchor="middle"
            fontSize="11"
            fontFamily="'JetBrains Mono', monospace"
            fill="var(--ink-2)"
          >
            {n.label}
          </text>
        </g>
      ))}
    </svg>
  );
}
