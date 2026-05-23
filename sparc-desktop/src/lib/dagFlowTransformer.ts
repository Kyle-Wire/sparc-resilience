/**
 * Converts DAG domain objects (DagDefinition, MC3Result) into ReactFlow
 * node/edge arrays ready for rendering.
 *
 * Single seam: all color mapping, node construction, edge styling, and
 * dagre layout live here. DagEditor never imports dagre directly.
 */
import { type Node, type Edge, MarkerType } from "@xyflow/react";
import dagre from "@dagrejs/dagre";
import type { DagDefinition, DagNodeType, MC3Result } from "./types";

// ── Node type → brand color ───────────────────────────────────────────────
export const NODE_COLORS: Record<string, string> = {
  treatment:        "#602468",
  mediator:         "#a44eb4",
  confounder:       "#e79024",
  outcome:          "#e73c25",
  instrument:       "#1e6fb8",
  proxy_confounder: "#f0b632",
  selection:        "#6e6358",
};

export const NODE_TYPE_OPTIONS: DagNodeType[] = [
  "treatment", "mediator", "confounder", "outcome",
  "instrument", "proxy_confounder", "selection",
];

// ── Dagre layout (hidden from callers) ───────────────────────────────────
function layoutNodes(nodes: Node[], edges: Edge[]): Node[] {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: "TB", ranksep: 90, nodesep: 70 });
  nodes.forEach((n) => g.setNode(n.id, { width: 160, height: 70 }));
  edges.forEach((e) => g.setEdge(e.source, e.target));
  dagre.layout(g);
  return nodes.map((n) => {
    const pos = g.node(n.id);
    return { ...n, position: { x: pos.x - 80, y: pos.y - 35 } };
  });
}

// ── Expert DAG → ReactFlow ────────────────────────────────────────────────
export function fromExpert(dag: DagDefinition): { nodes: Node[]; edges: Edge[] } {
  const flowNodes: Node[] = dag.nodes.map((n) => ({
    id: n.name, type: "dag", position: { x: 0, y: 0 },
    data: { label: n.name, nodeType: n.type, description: n.description },
  }));
  const nodeTypeMap = Object.fromEntries(dag.nodes.map((n) => [n.name, n.type]));
  const flowEdges: Edge[] = dag.edges.map((e) => {
    const color = NODE_COLORS[nodeTypeMap[e.parent] ?? ""] ?? "#602468";
    const labelBits: string[] = [];
    if (e.mechanism) labelBits.push(e.mechanism);
    if (e.lag) labelBits.push(`lag ${e.lag}`);
    if (e.sign_prior && e.sign_prior !== "0") labelBits.push(e.sign_prior);
    return {
      id: `e-${e.parent}-${e.child}`, source: e.parent, target: e.child,
      label: labelBits.length > 0 ? labelBits.join(" · ") : undefined,
      animated: e.kind === "time_lagged",
      style: { stroke: color, strokeWidth: 1.6 },
      markerEnd: { type: MarkerType.ArrowClosed, color },
      labelStyle: { fontSize: 9, fill: "#6e6358" },
    };
  });
  return { nodes: layoutNodes(flowNodes, flowEdges), edges: flowEdges };
}

// ── MC³ posterior → ReactFlow ─────────────────────────────────────────────
export function fromMc3(mc3: MC3Result, threshold: number): { nodes: Node[]; edges: Edge[] } {
  const { node_names, edge_probs } = mc3;
  const flowNodes: Node[] = node_names.map((name) => ({
    id: name, type: "dag", position: { x: 0, y: 0 },
    data: { label: name, nodeType: "confounder" },
  }));
  const flowEdges: Edge[] = [];
  for (let i = 0; i < node_names.length; i++) {
    for (let j = 0; j < node_names.length; j++) {
      if (i === j) continue;
      const prob = edge_probs[i][j];
      if (prob < threshold) continue;
      const color = prob >= 0.8 ? "#2d8a2d" : prob >= 0.5 ? "#e79024" : "#9ca3af";
      flowEdges.push({
        id: `mc3-${node_names[i]}-${node_names[j]}`,
        source: node_names[i], target: node_names[j],
        label: `${(prob * 100).toFixed(0)}%`,
        animated: prob >= 0.8,
        style: {
          stroke: color,
          strokeWidth: Math.max(1, prob * 2.5),
          strokeDasharray: prob < 0.5 ? "4,3" : undefined,
        },
        markerEnd: { type: MarkerType.ArrowClosed, color },
        labelStyle: { fontSize: 9, fill: color, fontWeight: prob >= 0.5 ? 600 : 400 },
      });
    }
  }
  return { nodes: layoutNodes(flowNodes, flowEdges), edges: flowEdges };
}
