import { useEffect, useState, useCallback, useMemo } from "react";
import {
  ReactFlow,
  Node,
  Edge,
  Controls,
  Background,
  BackgroundVariant,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
  MarkerType,
  Handle,
  Position,
  type NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import dagre from "@dagrejs/dagre";
import { getDag, validateDag } from "@/lib/api";
import type { DagEdge, DagDefinition, DagValidation } from "@/lib/types";

// ---------------------------------------------------------------------------
// Color palette for node types
// ---------------------------------------------------------------------------
const TYPE_COLORS: Record<string, { bg: string; border: string; text: string }> = {
  treatment:  { bg: "#f3e8f5", border: "#602468", text: "#602468" },
  mediator:   { bg: "#f5e0f8", border: "#a44eb4", text: "#7a2890" },
  confounder: { bg: "#fde8ec", border: "#f0a0b0", text: "#a04050" },
  outcome:    { bg: "#fdf8e0", border: "#fbdd46", text: "#8a7000" },
};

// ---------------------------------------------------------------------------
// Custom node component
// ---------------------------------------------------------------------------
function DagNodeComponent({ data }: NodeProps) {
  const d = data as { label: string; nodeType: string; description?: string; proposed?: boolean };
  const colors = TYPE_COLORS[d.nodeType] ?? { bg: "#f0f0f0", border: "#888", text: "#333" };

  return (
    <div
      className="rounded-lg px-4 py-3 shadow-sm min-w-[140px] text-center"
      style={{
        backgroundColor: colors.bg,
        border: `2px ${d.proposed ? "dashed" : "solid"} ${colors.border}`,
        opacity: d.proposed ? 0.7 : 1,
      }}
    >
      <Handle type="target" position={Position.Top} className="!bg-gray-400 !w-2 !h-2" />
      <div className="font-mono text-xs font-bold" style={{ color: colors.text }}>
        {d.label}
      </div>
      <div className="text-[10px] mt-0.5 capitalize" style={{ color: colors.text, opacity: 0.7 }}>
        {d.nodeType}
      </div>
      {d.description && (
        <div className="text-[9px] mt-1 text-gray-500 leading-tight">{d.description}</div>
      )}
      <Handle type="source" position={Position.Bottom} className="!bg-gray-400 !w-2 !h-2" />
    </div>
  );
}

const nodeTypes = { dag: DagNodeComponent };

// ---------------------------------------------------------------------------
// Dagre auto-layout
// ---------------------------------------------------------------------------
function layoutNodes(
  nodes: Node[],
  edges: Edge[],
  direction: "TB" | "LR" = "TB",
): Node[] {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: direction, ranksep: 80, nodesep: 60 });

  nodes.forEach((n) => g.setNode(n.id, { width: 160, height: 70 }));
  edges.forEach((e) => g.setEdge(e.source, e.target));
  dagre.layout(g);

  return nodes.map((n) => {
    const pos = g.node(n.id);
    return { ...n, position: { x: pos.x - 80, y: pos.y - 35 } };
  });
}

// ---------------------------------------------------------------------------
// Convert API DAG to ReactFlow nodes/edges
// ---------------------------------------------------------------------------
function dagToFlow(
  dag: DagDefinition,
  proposedEdges?: DagEdge[],
): { nodes: Node[]; edges: Edge[] } {
  const flowNodes: Node[] = dag.nodes.map((n) => ({
    id: n.name,
    type: "dag",
    position: { x: 0, y: 0 },
    data: { label: n.name, nodeType: n.type, description: n.description },
  }));

  const flowEdges: Edge[] = dag.edges.map((e) => ({
    id: `e-${e.parent}-${e.child}`,
    source: e.parent,
    target: e.child,
    label: e.mechanism,
    animated: false,
    style: { stroke: "#602468" },
    markerEnd: { type: MarkerType.ArrowClosed, color: "#602468" },
    labelStyle: { fontSize: 9, fill: "#666" },
  }));

  // Proposed edges from Claude (dashed, amber)
  if (proposedEdges) {
    proposedEdges.forEach((e, i) => {
      if (!flowEdges.find((fe) => fe.source === e.parent && fe.target === e.child)) {
        flowEdges.push({
          id: `proposal-${i}`,
          source: e.parent,
          target: e.child,
          label: `💡 ${e.mechanism ?? "proposed"}`,
          animated: true,
          style: { stroke: "#e79024", strokeDasharray: "5,5" },
          markerEnd: { type: MarkerType.ArrowClosed, color: "#e79024" },
          labelStyle: { fontSize: 9, fill: "#e79024" },
        });
      }
    });
  }

  const laid = layoutNodes(flowNodes, flowEdges);
  return { nodes: laid, edges: flowEdges };
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
export default function DAGView() {
  const [dag, setDag] = useState<DagDefinition | null>(null);
  const [validation, setValidation] = useState<DagValidation | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);

  // Check for Claude-proposed edges
  const proposedEdges = useMemo<DagEdge[]>(() => {
    try {
      const raw = localStorage.getItem("sparc-proposed-edges");
      if (raw) return JSON.parse(raw);
    } catch {}
    return [];
  }, []);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const d = await getDag();
      setDag(d);
      const { nodes: n, edges: e } = dagToFlow(d, proposedEdges);
      setNodes(n);
      setEdges(e);
      setError(null);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [proposedEdges, setNodes, setEdges]);

  useEffect(() => { load(); }, [load]);

  const onConnect = useCallback(
    (params: Connection) => {
      const newEdge: Edge = {
        id: `e-${params.source}-${params.target}`,
        source: params.source,
        target: params.target,
        style: { stroke: "#602468" },
        markerEnd: { type: MarkerType.ArrowClosed, color: "#602468" },
      };
      setEdges((eds) => addEdge(newEdge, eds));
    },
    [setEdges],
  );

  const validate = async () => {
    if (!dag) return;
    try {
      const v = await validateDag(dag);
      setValidation(v);
    } catch (e: any) {
      setError(e.message);
    }
  };

  const acceptProposals = () => {
    // Convert proposal edges to permanent
    setEdges((eds) =>
      eds.map((e) =>
        e.id.startsWith("proposal-")
          ? {
              ...e,
              animated: false,
              style: { stroke: "#602468" },
              markerEnd: { type: MarkerType.ArrowClosed, color: "#602468" },
              label: typeof e.label === "string" ? e.label.replace("💡 ", "") : e.label,
              labelStyle: { fontSize: 9, fill: "#666" },
            }
          : e,
      ),
    );
    localStorage.removeItem("sparc-proposed-edges");
  };

  const dismissProposals = () => {
    setEdges((eds) => eds.filter((e) => !e.id.startsWith("proposal-")));
    localStorage.removeItem("sparc-proposed-edges");
  };

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <p className="text-sm text-sparc-gray-500">Loading DAG…</p>
      </div>
    );
  }

  if (error && !dag) {
    return (
      <div className="p-6">
        <p className="text-red-600">{error}</p>
        <p className="mt-2 text-sm text-sparc-gray-600">
          Load a project with a causal DAG first.
        </p>
      </div>
    );
  }

  const hasProposals = edges.some((e) => e.id.startsWith("proposal-"));

  return (
    <div className="flex h-full flex-col">
      {/* Toolbar */}
      <div className="flex items-center justify-between border-b border-sparc-gray-200 px-4 py-3">
        <div>
          <h2 className="text-lg font-bold">Causal DAG</h2>
          <p className="text-xs text-sparc-gray-500">
            Drag between nodes to create edges. Ask the AI Assistant to propose edges.
          </p>
        </div>
        <div className="flex gap-2">
          {hasProposals && (
            <>
              <button
                onClick={acceptProposals}
                className="rounded bg-green-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-green-700"
              >
                Accept Proposals
              </button>
              <button
                onClick={dismissProposals}
                className="rounded border border-red-300 px-3 py-1.5 text-xs font-medium text-red-600 hover:bg-red-50"
              >
                Dismiss
              </button>
            </>
          )}
          <button
            onClick={validate}
            className="rounded bg-sparc-purple px-3 py-1.5 text-xs font-medium text-white hover:bg-sparc-magenta"
          >
            Validate
          </button>
          <button
            onClick={load}
            className="rounded border border-sparc-gray-300 px-3 py-1.5 text-xs hover:bg-sparc-gray-100"
          >
            Reload
          </button>
        </div>
      </div>

      {/* Validation banner */}
      {validation && (
        <div
          className={`border-b px-4 py-2 text-xs ${
            validation.valid
              ? "border-green-200 bg-green-50 text-green-800"
              : "border-red-200 bg-red-50 text-red-800"
          }`}
        >
          {validation.valid
            ? `✓ Valid — ${validation.n_nodes} nodes, ${validation.n_edges} edges`
            : `✗ ${validation.error}`}
        </div>
      )}

      {/* Legend */}
      <div className="flex gap-4 border-b border-sparc-gray-100 px-4 py-2">
        {Object.entries(TYPE_COLORS).map(([type, c]) => (
          <div key={type} className="flex items-center gap-1.5 text-[10px]">
            <span
              className="inline-block h-2.5 w-2.5 rounded-sm border-2"
              style={{ borderColor: c.border, backgroundColor: c.bg }}
            />
            <span className="capitalize">{type}</span>
          </div>
        ))}
        {hasProposals && (
          <div className="flex items-center gap-1.5 text-[10px]">
            <span className="inline-block h-2.5 w-2.5 rounded-sm border-2 border-dashed" style={{ borderColor: "#e79024" }} />
            <span>AI Proposal</span>
          </div>
        )}
      </div>

      {/* Flow canvas */}
      <div className="flex-1">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          nodeTypes={nodeTypes}
          fitView
          fitViewOptions={{ padding: 0.3 }}
          proOptions={{ hideAttribution: true }}
        >
          <Background variant={BackgroundVariant.Dots} gap={16} size={1} color="#e0e0e0" />
          <Controls showInteractive={false} />
        </ReactFlow>
      </div>
    </div>
  );
}
