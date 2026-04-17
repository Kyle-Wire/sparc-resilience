import { useEffect, useState, useCallback, useMemo, useRef } from "react";
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
import { getDag, validateDag, getMc3Result } from "@/lib/api";
import type { DagEdge, DagDefinition, DagValidation, MC3Result } from "@/lib/types";
import { usePipeline } from "@/hooks/PipelineProvider";

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
// Undo / Redo history stack
// ---------------------------------------------------------------------------
interface Snapshot {
  nodes: Node[];
  edges: Edge[];
}

function useUndoRedo(
  nodes: Node[],
  edges: Edge[],
  setNodes: (ns: Node[]) => void,
  setEdges: (es: Edge[]) => void,
) {
  const historyRef = useRef<Snapshot[]>([]);
  const futureRef = useRef<Snapshot[]>([]);
  const skipRef = useRef(false); // prevent recording when restoring

  const pushSnapshot = useCallback(() => {
    if (skipRef.current) return;
    historyRef.current.push({
      nodes: JSON.parse(JSON.stringify(nodes)),
      edges: JSON.parse(JSON.stringify(edges)),
    });
    futureRef.current = []; // clear redo stack on new action
    // Limit history size
    if (historyRef.current.length > 50) historyRef.current.shift();
  }, [nodes, edges]);

  const undo = useCallback(() => {
    if (historyRef.current.length === 0) return;
    const snapshot = historyRef.current.pop()!;
    futureRef.current.push({
      nodes: JSON.parse(JSON.stringify(nodes)),
      edges: JSON.parse(JSON.stringify(edges)),
    });
    skipRef.current = true;
    setNodes(snapshot.nodes);
    setEdges(snapshot.edges);
    requestAnimationFrame(() => { skipRef.current = false; });
  }, [nodes, edges, setNodes, setEdges]);

  const redo = useCallback(() => {
    if (futureRef.current.length === 0) return;
    const snapshot = futureRef.current.pop()!;
    historyRef.current.push({
      nodes: JSON.parse(JSON.stringify(nodes)),
      edges: JSON.parse(JSON.stringify(edges)),
    });
    skipRef.current = true;
    setNodes(snapshot.nodes);
    setEdges(snapshot.edges);
    requestAnimationFrame(() => { skipRef.current = false; });
  }, [nodes, edges, setNodes, setEdges]);

  const canUndo = historyRef.current.length > 0;
  const canRedo = futureRef.current.length > 0;

  return { pushSnapshot, undo, redo, canUndo, canRedo };
}

// ---------------------------------------------------------------------------
// Node type options for context menu
// ---------------------------------------------------------------------------
const NODE_TYPE_OPTIONS = ["treatment", "mediator", "confounder", "outcome"] as const;

// ---------------------------------------------------------------------------
// MC³ edge styling helpers
// ---------------------------------------------------------------------------
function mc3EdgeStyle(prob: number): { stroke: string; strokeDasharray?: string; strokeWidth: number } {
  if (prob >= 0.8) return { stroke: "#16a34a", strokeWidth: Math.max(1.5, prob * 3) };
  if (prob >= 0.3) return { stroke: "#d97706", strokeDasharray: "6,4", strokeWidth: Math.max(1, prob * 2.5) };
  return { stroke: "#9ca3af", strokeDasharray: "3,3", strokeWidth: 1 };
}

function mc3EdgesToFlow(mc3: MC3Result): Edge[] {
  const { node_names, edge_probs } = mc3;
  const edges: Edge[] = [];
  for (let i = 0; i < node_names.length; i++) {
    for (let j = 0; j < node_names.length; j++) {
      if (i === j) continue;
      const prob = edge_probs[i][j];
      if (prob < 0.10) continue; // skip negligible edges
      const style = mc3EdgeStyle(prob);
      edges.push({
        id: `mc3-${node_names[i]}-${node_names[j]}`,
        source: node_names[i],
        target: node_names[j],
        label: `${(prob * 100).toFixed(0)}%`,
        animated: prob >= 0.5,
        style,
        markerEnd: { type: MarkerType.ArrowClosed, color: style.stroke },
        labelStyle: { fontSize: 9, fill: style.stroke, fontWeight: prob >= 0.5 ? 600 : 400 },
      });
    }
  }
  return edges;
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
export default function DAGView() {
  const { dagApprovalPending, handleApproveDag, handleRejectDag } = usePipeline();
  const [dag, setDag] = useState<DagDefinition | null>(null);
  const [validation, setValidation] = useState<DagValidation | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [mc3, setMc3] = useState<MC3Result | null>(null);
  const [showMc3, setShowMc3] = useState(false);

  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);

  // Undo/Redo
  const { pushSnapshot, undo, redo, canUndo, canRedo } = useUndoRedo(nodes, edges, setNodes, setEdges);

  // Context menu state
  const [contextMenu, setContextMenu] = useState<{
    x: number;
    y: number;
    nodeId: string;
  } | null>(null);

  // Quick edge form
  const [quickEdge, setQuickEdge] = useState(false);
  const [qeSource, setQeSource] = useState("");
  const [qeTarget, setQeTarget] = useState("");
  const [qeMechanism, setQeMechanism] = useState("");

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

  // Fetch MC³ results when DAG approval is requested
  useEffect(() => {
    if (!dagApprovalPending) {
      setMc3(null);
      setShowMc3(false);
      return;
    }
    getMc3Result()
      .then((data) => {
        setMc3(data);
        setShowMc3(true);
        // Overlay MC³ edges onto the flow canvas
        if (data) {
          const mc3Edges = mc3EdgesToFlow(data);
          setEdges((eds) => {
            // Remove any previous mc3 overlay edges
            const cleaned = eds.filter((e) => !e.id.startsWith("mc3-"));
            return [...cleaned, ...mc3Edges];
          });
          // Ensure all MC³ nodes exist in the canvas
          const existingIds = new Set(nodes.map((n) => n.id));
          const newNodes: Node[] = [];
          for (const name of data.node_names) {
            if (!existingIds.has(name)) {
              newNodes.push({
                id: name,
                type: "dag",
                position: { x: 0, y: 0 },
                data: { label: name, nodeType: "confounder" },
              });
            }
          }
          if (newNodes.length > 0) {
            setNodes((nds) => layoutNodes([...nds, ...newNodes], edges));
          }
        }
      })
      .catch(() => {
        // MC³ result not ready yet — will retry on next render
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dagApprovalPending]);

  // Keyboard shortcuts for undo/redo
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "z" && !e.shiftKey) {
        e.preventDefault();
        undo();
      }
      if ((e.ctrlKey || e.metaKey) && e.key === "z" && e.shiftKey) {
        e.preventDefault();
        redo();
      }
      if ((e.ctrlKey || e.metaKey) && e.key === "y") {
        e.preventDefault();
        redo();
      }
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [undo, redo]);

  const onConnect = useCallback(
    (params: Connection) => {
      pushSnapshot();
      const newEdge: Edge = {
        id: `e-${params.source}-${params.target}`,
        source: params.source,
        target: params.target,
        style: { stroke: "#602468" },
        markerEnd: { type: MarkerType.ArrowClosed, color: "#602468" },
      };
      setEdges((eds) => addEdge(newEdge, eds));
    },
    [setEdges, pushSnapshot],
  );

  // Quick edge: add via form
  const addQuickEdge = useCallback(() => {
    if (!qeSource || !qeTarget || qeSource === qeTarget) return;
    // Check for duplicate
    if (edges.find((e) => e.source === qeSource && e.target === qeTarget)) return;
    pushSnapshot();
    const newEdge: Edge = {
      id: `e-${qeSource}-${qeTarget}`,
      source: qeSource,
      target: qeTarget,
      label: qeMechanism || undefined,
      style: { stroke: "#602468" },
      markerEnd: { type: MarkerType.ArrowClosed, color: "#602468" },
      labelStyle: { fontSize: 9, fill: "#666" },
    };
    setEdges((eds) => addEdge(newEdge, eds));
    setQeSource("");
    setQeTarget("");
    setQeMechanism("");
  }, [qeSource, qeTarget, qeMechanism, edges, setEdges, pushSnapshot]);

  // Context menu: right-click on node
  const onNodeContextMenu = useCallback(
    (event: React.MouseEvent, node: Node) => {
      event.preventDefault();
      setContextMenu({ x: event.clientX, y: event.clientY, nodeId: node.id });
    },
    [],
  );

  // Change node type from context menu
  const changeNodeType = useCallback(
    (nodeId: string, newType: string) => {
      pushSnapshot();
      setNodes((nds) =>
        nds.map((n) =>
          n.id === nodeId
            ? { ...n, data: { ...n.data, nodeType: newType } }
            : n,
        ),
      );
      setContextMenu(null);
    },
    [setNodes, pushSnapshot],
  );

  // Close context menu on click
  useEffect(() => {
    if (!contextMenu) return;
    const close = () => setContextMenu(null);
    document.addEventListener("click", close);
    return () => document.removeEventListener("click", close);
  }, [contextMenu]);

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
    pushSnapshot();
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
    pushSnapshot();
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
            Drag between nodes to create edges. Right-click a node to change its type.
          </p>
        </div>
        <div className="flex gap-2">
          {/* Undo/Redo */}
          <button
            onClick={undo}
            disabled={!canUndo}
            className="rounded border border-sparc-gray-300 px-2 py-1.5 text-xs hover:bg-sparc-gray-100 disabled:opacity-30"
            title="Undo (Ctrl+Z)"
          >
            ↶ Undo
          </button>
          <button
            onClick={redo}
            disabled={!canRedo}
            className="rounded border border-sparc-gray-300 px-2 py-1.5 text-xs hover:bg-sparc-gray-100 disabled:opacity-30"
            title="Redo (Ctrl+Shift+Z)"
          >
            ↷ Redo
          </button>
          <div className="mx-1 w-px bg-sparc-gray-200" />
          {/* Quick edge toggle */}
          <button
            onClick={() => setQuickEdge(!quickEdge)}
            className={`rounded border px-3 py-1.5 text-xs font-medium ${
              quickEdge
                ? "border-sparc-purple bg-sparc-purple text-white"
                : "border-sparc-gray-300 hover:bg-sparc-gray-100"
            }`}
          >
            + Edge
          </button>
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

      {/* Quick edge form */}
      {quickEdge && (
        <div className="flex items-center gap-2 border-b border-sparc-gray-100 bg-sparc-gray-50 px-4 py-2">
          <span className="text-xs text-sparc-gray-600">From:</span>
          <select
            value={qeSource}
            onChange={(e) => setQeSource(e.target.value)}
            className="rounded border border-sparc-gray-200 px-2 py-1 text-xs"
          >
            <option value="">Select…</option>
            {nodes.map((n) => (
              <option key={n.id} value={n.id}>{n.id}</option>
            ))}
          </select>
          <span className="text-xs text-sparc-gray-400">→</span>
          <span className="text-xs text-sparc-gray-600">To:</span>
          <select
            value={qeTarget}
            onChange={(e) => setQeTarget(e.target.value)}
            className="rounded border border-sparc-gray-200 px-2 py-1 text-xs"
          >
            <option value="">Select…</option>
            {nodes.filter((n) => n.id !== qeSource).map((n) => (
              <option key={n.id} value={n.id}>{n.id}</option>
            ))}
          </select>
          <input
            type="text"
            value={qeMechanism}
            onChange={(e) => setQeMechanism(e.target.value)}
            placeholder="Mechanism (optional)"
            className="rounded border border-sparc-gray-200 px-2 py-1 text-xs w-40"
          />
          <button
            onClick={addQuickEdge}
            disabled={!qeSource || !qeTarget || qeSource === qeTarget}
            className="rounded bg-sparc-purple px-3 py-1 text-xs font-medium text-white hover:bg-sparc-magenta disabled:opacity-40"
          >
            Add
          </button>
        </div>
      )}

      {/* DAG Approval gate banner */}
      {dagApprovalPending && (
        <div className="flex items-center justify-between border-b border-amber-300 bg-amber-50 px-4 py-3">
          <div>
            <p className="text-sm font-semibold text-amber-800">
              DAG Review Required — Pipeline Paused
            </p>
            <p className="text-xs text-amber-700">
              MC³ structure learning is complete. Review the discovered edges (colored by probability)
              then approve to continue to NUTS posterior sampling.
              {mc3 && ` ${mc3.median_dag.edges.length} edges above 50% threshold.`}
            </p>
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => setShowMc3(!showMc3)}
              className="rounded border border-amber-400 px-3 py-1.5 text-xs font-medium text-amber-700 hover:bg-amber-100"
            >
              {showMc3 ? "Hide MC³ Edges" : "Show MC³ Edges"}
            </button>
            <button
              onClick={handleRejectDag}
              className="rounded border border-red-300 px-3 py-1.5 text-xs font-medium text-red-600 hover:bg-red-50"
            >
              Reject & Cancel
            </button>
            <button
              onClick={handleApproveDag}
              className="rounded bg-green-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-green-700"
            >
              Approve DAG
            </button>
          </div>
        </div>
      )}

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
        {showMc3 && (
          <>
            <div className="mx-1 w-px bg-sparc-gray-200" />
            <div className="flex items-center gap-1.5 text-[10px]">
              <span className="inline-block h-0.5 w-4" style={{ backgroundColor: "#16a34a" }} />
              <span>MC³ &gt;80%</span>
            </div>
            <div className="flex items-center gap-1.5 text-[10px]">
              <span className="inline-block h-0.5 w-4 border-t-2 border-dashed" style={{ borderColor: "#d97706" }} />
              <span>MC³ 30-80%</span>
            </div>
            <div className="flex items-center gap-1.5 text-[10px]">
              <span className="inline-block h-0.5 w-4 border-t border-dotted" style={{ borderColor: "#9ca3af" }} />
              <span>MC³ &lt;30%</span>
            </div>
          </>
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
          onNodeContextMenu={onNodeContextMenu}
          nodeTypes={nodeTypes}
          fitView
          fitViewOptions={{ padding: 0.3 }}
          proOptions={{ hideAttribution: true }}
        >
          <Background variant={BackgroundVariant.Dots} gap={16} size={1} color="#e0e0e0" />
          <Controls showInteractive={false} />
        </ReactFlow>

        {/* Node type context menu */}
        {contextMenu && (
          <div
            className="fixed z-50 rounded border border-sparc-gray-200 bg-white py-1 shadow-lg"
            style={{ left: contextMenu.x, top: contextMenu.y }}
          >
            <div className="px-3 py-1 text-[10px] font-bold text-sparc-gray-400 uppercase">
              Change Type
            </div>
            {NODE_TYPE_OPTIONS.map((type) => {
              const c = TYPE_COLORS[type];
              return (
                <button
                  key={type}
                  onClick={() => changeNodeType(contextMenu.nodeId, type)}
                  className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-xs hover:bg-sparc-gray-50"
                >
                  <span
                    className="inline-block h-2.5 w-2.5 rounded-sm border-2"
                    style={{ borderColor: c.border, backgroundColor: c.bg }}
                  />
                  <span className="capitalize">{type}</span>
                </button>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
