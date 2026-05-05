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
// Color palette for node types. SPARC v4 adds three more identification-
// relevant types beyond the original four:
//   - instrument: IV — affects outcome only via a treatment.
//   - proxy_confounder: imperfect measurement of a latent confounder
//     (residual confounding warning).
//   - selection: drives sample inclusion; conditioning → collider bias.
// ---------------------------------------------------------------------------
const TYPE_COLORS: Record<string, { bg: string; border: string; text: string }> = {
  treatment:        { bg: "#f3e8f5", border: "#602468", text: "#602468" },
  mediator:         { bg: "#f5e0f8", border: "#a44eb4", text: "#7a2890" },
  confounder:       { bg: "#fde8ec", border: "#f0a0b0", text: "#a04050" },
  outcome:          { bg: "#fdf8e0", border: "#fbdd46", text: "#8a7000" },
  instrument:       { bg: "#e6f0ff", border: "#1e4fb8", text: "#1e4fb8" },
  proxy_confounder: { bg: "#fbe7d4", border: "#b86a1e", text: "#8a4a10" },
  selection:        { bg: "#e8e8e8", border: "#555",    text: "#222"    },
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

  const flowEdges: Edge[] = dag.edges.map((e) => {
    // Edge styling derives from sign_prior (hue) and confidence (opacity
    // / width). `kind` and `lag` are surfaced in the label so the
    // identification semantics are visible at a glance.
    const sign = e.sign_prior;
    const conf = typeof e.confidence === "number" ? Math.max(0, Math.min(1, e.confidence)) : null;
    const baseColor =
      sign === "+" ? "#1f7d1f"
        : sign === "-" ? "#b91c1c"
          : e.kind === "instrumental" ? "#1e4fb8"
            : e.kind === "time_lagged" ? "#7a2890"
              : "#602468";
    const opacity = conf == null ? 1 : 0.4 + 0.6 * conf;
    const strokeWidth = conf == null ? 1.4 : 1 + conf * 1.6;
    const dash = e.kind === "time_lagged" ? "5,4" : e.kind === "mediated" ? "2,3" : undefined;
    const labelBits: string[] = [];
    if (e.mechanism) labelBits.push(e.mechanism);
    if (e.lag) labelBits.push(`lag ${e.lag}`);
    if (sign && sign !== "0") labelBits.push(sign);
    if (conf != null) labelBits.push(`c=${conf.toFixed(2)}`);
    return {
      id: `e-${e.parent}-${e.child}`,
      source: e.parent,
      target: e.child,
      label: labelBits.join(" · "),
      animated: e.kind === "time_lagged",
      style: { stroke: baseColor, opacity, strokeWidth, ...(dash ? { strokeDasharray: dash } : {}) },
      markerEnd: { type: MarkerType.ArrowClosed, color: baseColor },
      labelStyle: { fontSize: 9, fill: "#666" },
      data: {
        mechanism: e.mechanism,
        kind: e.kind ?? "direct",
        lag: e.lag,
        sign_prior: e.sign_prior,
        confidence: e.confidence,
      },
    };
  });

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
const NODE_TYPE_OPTIONS = [
  "treatment",
  "mediator",
  "confounder",
  "outcome",
  "instrument",
  "proxy_confounder",
  "selection",
] as const;

// ---------------------------------------------------------------------------
// MC³ edge styling helpers
// ---------------------------------------------------------------------------
function mc3EdgeStyle(prob: number): { stroke: string; strokeDasharray?: string; strokeWidth: number } {
  if (prob >= 0.8) return { stroke: "#16a34a", strokeWidth: Math.max(1.5, prob * 3) };
  if (prob >= 0.3) return { stroke: "#d97706", strokeDasharray: "6,4", strokeWidth: Math.max(1, prob * 2.5) };
  return { stroke: "#9ca3af", strokeDasharray: "3,3", strokeWidth: 1 };
}

function mc3EdgesToFlow(mc3: MC3Result, threshold: number = 0.10): Edge[] {
  const { node_names, edge_probs } = mc3;
  const edges: Edge[] = [];
  for (let i = 0; i < node_names.length; i++) {
    for (let j = 0; j < node_names.length; j++) {
      if (i === j) continue;
      const prob = edge_probs[i][j];
      if (prob < threshold) continue; // skip edges below threshold
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
  const { dagApprovalPending, handleApproveDag, handleRejectDag, runEndedAt, stageStatuses } = usePipeline();
  const [dag, setDag] = useState<DagDefinition | null>(null);
  const [validation, setValidation] = useState<DagValidation | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [mc3, setMc3] = useState<MC3Result | null>(null);
  const [showMc3, setShowMc3] = useState(false);
  const [mc3Threshold, setMc3Threshold] = useState(0.30);
  const [mc3FetchError, setMc3FetchError] = useState<string | null>(null);

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

  // Edge tooltip state (causal assumption + SPARC v4 attrs)
  const [edgeTooltip, setEdgeTooltip] = useState<{
    x: number;
    y: number;
    source: string;
    target: string;
    mechanism?: string;
    kind?: string;
    lag?: number;
    sign_prior?: string;
    confidence?: number;
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

  // -------- Phase 20: vim-style keyboard navigation --------
  // j/k step through nodes (sorted by y, then x). gg → first, G → last.
  // e on a selected node opens the quick-edge form pre-filled with it as
  // the source. d deletes the selected node and any incident edges.
  // We bail out when focus is in an editable element so typing in inputs
  // is not hijacked.
  const [vimSelectedId, setVimSelectedId] = useState<string | null>(null);
  const lastGRef = useRef<number>(0);
  useEffect(() => {
    function isEditable(el: EventTarget | null): boolean {
      if (!(el instanceof HTMLElement)) return false;
      const tag = el.tagName;
      return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || el.isContentEditable;
    }
    function onKey(e: KeyboardEvent) {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (isEditable(e.target)) return;
      if (nodes.length === 0) return;
      const sorted = [...nodes].sort((a, b) => (a.position.y - b.position.y) || (a.position.x - b.position.x));
      const idx = vimSelectedId ? sorted.findIndex((n) => n.id === vimSelectedId) : -1;

      const select = (i: number) => {
        const id = sorted[Math.max(0, Math.min(sorted.length - 1, i))].id;
        setVimSelectedId(id);
        setNodes((ns) => ns.map((n) => ({ ...n, selected: n.id === id })));
      };

      if (e.key === "j") { e.preventDefault(); select(idx < 0 ? 0 : Math.min(sorted.length - 1, idx + 1)); }
      else if (e.key === "k") { e.preventDefault(); select(idx < 0 ? sorted.length - 1 : Math.max(0, idx - 1)); }
      else if (e.key === "G") { e.preventDefault(); select(sorted.length - 1); }
      else if (e.key === "g") {
        e.preventDefault();
        const now = Date.now();
        if (now - lastGRef.current < 400) { select(0); lastGRef.current = 0; }
        else { lastGRef.current = now; }
      } else if (e.key === "e" && vimSelectedId) {
        e.preventDefault();
        setQeSource(vimSelectedId);
        setQeTarget("");
        setQeMechanism("");
        setQuickEdge(true);
      } else if ((e.key === "d" || e.key === "Delete" || e.key === "Backspace") && vimSelectedId) {
        e.preventDefault();
        pushSnapshot();
        const id = vimSelectedId;
        setNodes((ns) => ns.filter((n) => n.id !== id));
        setEdges((es) => es.filter((ed) => ed.source !== id && ed.target !== id));
        setVimSelectedId(null);
      } else if (e.key === "Escape") {
        setVimSelectedId(null);
        setNodes((ns) => ns.map((n) => ({ ...n, selected: false })));
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [nodes, vimSelectedId, setNodes, setEdges, pushSnapshot]);

  // Fetch MC³ results when DAG approval is requested OR user toggles overlay on
  useEffect(() => {
    if (!dagApprovalPending && !showMc3) {
      setMc3FetchError(null);
      // Strip any stale mc3 overlay edges when toggle is off
      setEdges((eds) => eds.filter((e) => !e.id.startsWith("mc3-")));
      return;
    }
    getMc3Result()
      .then((data) => {
        setMc3(data);
        setMc3FetchError(null);
        if (dagApprovalPending) setShowMc3(true);
        if (data) {
          const mc3Edges = mc3EdgesToFlow(data, mc3Threshold);
          setEdges((eds) => {
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
        } else {
          setMc3FetchError("No MC³ result available yet \u2014 run the pipeline through the DAG step first.");
        }
      })
      .catch((err) => {
        setMc3(null);
        setMc3FetchError(
          err instanceof Error
            ? err.message
            : "Could not fetch MC³ result \u2014 has the pipeline reached the DAG-learning step?",
        );
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dagApprovalPending, showMc3, mc3Threshold]);

  // Load MC³ after the causal stage finishes (re-fetches even if approval gate
  // was previously cleared). Also poll every 3 s while the causal stage is
  // running so users see edges accumulate live.
  useEffect(() => {
    const causalStatus = stageStatuses[3]?.status;
    const fetchMc3 = () => {
      getMc3Result()
        .then((data) => {
          if (!data) return;
          setMc3(data);
          if (showMc3) {
            const mc3Edges = mc3EdgesToFlow(data);
            setEdges((eds) => {
              const cleaned = eds.filter((e) => !e.id.startsWith("mc3-"));
              return [...cleaned, ...mc3Edges];
            });
          }
        })
        .catch(() => {});
    };
    if (causalStatus === "running") {
      fetchMc3();
      const id = setInterval(fetchMc3, 3000);
      return () => clearInterval(id);
    }
    if (causalStatus === "complete" || runEndedAt) fetchMc3();
    return undefined;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stageStatuses[3]?.status, runEndedAt]);

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
      setEdges((eds) => addEdge({ ...params, animated: false }, eds));

      const applyMc3 = (data: import("@/lib/types").MC3Result) => {
        setMc3(data);
        setShowMc3(true);
        const mc3Edges = mc3EdgesToFlow(data);
        setEdges((eds) => {
          const cleaned = eds.filter((e) => !e.id.startsWith("mc3-"));
          return [...cleaned, ...mc3Edges];
        });
        // Use functional setNodes to read latest state and avoid stale closure
        setNodes((currentNodes) => {
          const existingIds = new Set(currentNodes.map((n) => n.id));
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
          return newNodes.length > 0
            ? layoutNodes([...currentNodes, ...newNodes], mc3Edges)
            : currentNodes;
        });
      };

      getMc3Result()
        .then((data) => { if (data) applyMc3(data); })
        .catch(() => {
          // MC³ result not ready yet — retry once after a short delay
          setTimeout(() => {
            getMc3Result()
              .then((data) => { if (data) applyMc3(data); })
              .catch(() => {});
          }, 1500);
        });
    },
    [addEdge, pushSnapshot],
  );

  // Quick edge: add edge from form inputs
  const addQuickEdge = useCallback(() => {
    if (!qeSource || !qeTarget || qeSource === qeTarget) return;
    pushSnapshot();
    const id = `qe-${qeSource}-${qeTarget}-${Date.now()}`;
    setEdges((eds) => [
      ...eds,
      {
        id,
        source: qeSource,
        target: qeTarget,
        animated: false,
        ...(qeMechanism ? { label: qeMechanism } : {}),
      },
    ]);
    setQeSource("");
    setQeTarget("");
    setQeMechanism("");
  }, [qeSource, qeTarget, qeMechanism, pushSnapshot, setEdges]);

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
    if (!contextMenu && !edgeTooltip) return;
    const close = () => { setContextMenu(null); setEdgeTooltip(null); };
    document.addEventListener("click", close);
    return () => document.removeEventListener("click", close);
  }, [contextMenu, edgeTooltip]);

  // Edge click → causal assumption tooltip (now surfaces SPARC v4
  // identification attributes carried on `edge.data`).
  const onEdgeClick = useCallback(
    (event: React.MouseEvent, edge: Edge) => {
      if (edge.id.startsWith("mc3-")) return; // skip MC³ overlay edges
      event.stopPropagation();
      const d = (edge.data ?? {}) as {
        mechanism?: string;
        kind?: string;
        lag?: number;
        sign_prior?: string;
        confidence?: number;
      };
      setEdgeTooltip({
        x: event.clientX,
        y: event.clientY,
        source: edge.source,
        target: edge.target,
        mechanism: d.mechanism ?? (typeof edge.label === "string" ? edge.label.replace("💡 ", "") : undefined),
        kind: d.kind,
        lag: d.lag,
        sign_prior: d.sign_prior,
        confidence: d.confidence,
      });
    },
    [],
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

  // Client-side structural warnings
  const structuralWarnings = useMemo<string[]>(() => {
    if (nodes.length === 0) return [];
    const warns: string[] = [];
    // Find user-defined edges (not mc3/proposal)
    const userEdges = edges.filter(
      (e) => !e.id.startsWith("mc3-") && !e.id.startsWith("proposal-"),
    );
    // Disconnected nodes
    const connected = new Set<string>();
    for (const e of userEdges) { connected.add(e.source); connected.add(e.target); }
    const disconnected = nodes.filter((n) => !connected.has(n.id));
    if (disconnected.length > 0) {
      warns.push(`Disconnected node${disconnected.length > 1 ? "s" : ""}: ${disconnected.map((n) => n.id).join(", ")}`);
    }
    // Outcome nodes with outgoing edges
    const outcomes = nodes.filter((n) => (n.data as any).nodeType === "outcome");
    for (const o of outcomes) {
      if (userEdges.some((e) => e.source === o.id)) {
        warns.push(`Outcome "${o.id}" has outgoing edges — outcomes should be terminal`);
      }
    }
    // Treatment → Outcome path exists
    const treatments = nodes.filter((n) => (n.data as any).nodeType === "treatment");
    if (treatments.length > 0 && outcomes.length > 0) {
      // BFS from any treatment to any outcome
      const adj = new Map<string, string[]>();
      for (const e of userEdges) {
        if (!adj.has(e.source)) adj.set(e.source, []);
        adj.get(e.source)!.push(e.target);
      }
      const outcomeIds = new Set(outcomes.map((o) => o.id));
      let pathExists = false;
      for (const t of treatments) {
        const visited = new Set<string>();
        const queue = [t.id];
        while (queue.length > 0) {
          const cur = queue.shift()!;
          if (outcomeIds.has(cur) && cur !== t.id) { pathExists = true; break; }
          if (visited.has(cur)) continue;
          visited.add(cur);
          for (const next of adj.get(cur) ?? []) queue.push(next);
        }
        if (pathExists) break;
      }
      if (!pathExists) {
        warns.push("No directed path from any treatment to any outcome");
      }
    }
    return warns;
  }, [nodes, edges]);

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
          <span
            className="ml-2 hidden font-mono text-[10px] text-sparc-gray-500 lg:inline"
            title="Vim-style navigation: j/k step nodes, gg/G first/last, e draw edge, d delete"
          >
            j/k · gg/G · e · d
          </span>
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
          <div className="mx-1 w-px bg-sparc-gray-200" />
          {/* MC³ overlay toggle (always available) */}
          <button
            onClick={() => setShowMc3((v) => !v)}
            className={`rounded border px-3 py-1.5 text-xs font-medium ${
              showMc3
                ? "border-emerald-500 bg-emerald-50 text-emerald-700"
                : "border-sparc-gray-300 hover:bg-sparc-gray-100"
            }`}
            title="Show MC³ posterior edge probabilities"
          >
            {showMc3 ? "◉ MC³ on" : "○ MC³ off"}
          </button>
          <button
            onClick={load}
            className="rounded border border-sparc-gray-300 px-3 py-1.5 text-xs hover:bg-sparc-gray-100"
          >
            Reload
          </button>
        </div>
      </div>

      {/* MC³ status / threshold slider */}
      {showMc3 && (
        <div className="flex items-center gap-3 border-b border-emerald-100 bg-emerald-50 px-4 py-2">
          {mc3 ? (
            <>
              <span className="text-xs font-semibold text-emerald-800">
                MC³ posterior · {mc3.node_names.length} nodes
              </span>
              <span className="text-xs text-emerald-700">
                threshold {(mc3Threshold * 100).toFixed(0)}%
              </span>
              <input
                type="range"
                min={0.05}
                max={0.95}
                step={0.05}
                value={mc3Threshold}
                onChange={(e) => setMc3Threshold(parseFloat(e.target.value))}
                className="flex-1 max-w-xs"
              />
              <span className="text-[10px] text-emerald-700">
                showing {edges.filter((e) => e.id.startsWith("mc3-")).length} edges
              </span>
            </>
          ) : mc3FetchError ? (
            <span className="text-xs text-amber-700">⚠ {mc3FetchError}</span>
          ) : (
            <span className="text-xs text-emerald-700">Loading MC³ result…</span>
          )}
        </div>
      )}

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

      {/* Structural warnings */}
      {structuralWarnings.length > 0 && (
        <div className="border-b border-amber-200 bg-amber-50 px-4 py-2">
          {structuralWarnings.map((w, i) => (
            <p key={i} className="text-xs text-amber-700">⚠ {w}</p>
          ))}
        </div>
      )}

      {/* User-asserted identification assumptions (SPARC v4) */}
      {dag?.assumptions && (
        <div className="border-b border-sparc-gray-100 bg-sparc-gray-50 px-4 py-2">
          <p className="mb-1 text-[10px] font-bold uppercase tracking-wide text-sparc-gray-500">
            Identification Assumptions
          </p>
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px]">
            {([
              ["conditional_exchangeability", "Exchangeability"],
              ["positivity", "Positivity"],
              ["consistency", "Consistency"],
              ["no_interference", "No interference"],
            ] as const).map(([k, label]) => {
              const v = (dag.assumptions as Record<string, unknown>)?.[k];
              if (v === undefined) return null;
              const ok = v === true;
              return (
                <span key={k} className="flex items-center gap-1">
                  <span className={ok ? "text-emerald-700" : "text-amber-700"}>{ok ? "✓" : "✗"}</span>
                  <span className="text-sparc-gray-700">{label}</span>
                </span>
              );
            })}
          </div>
          {dag.assumptions.notes && (
            <p className="mt-1 text-[10px] italic leading-snug text-sparc-gray-500">
              {dag.assumptions.notes}
            </p>
          )}
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
          onEdgeClick={onEdgeClick}
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

        {/* Causal assumption tooltip on edge click */}
        {edgeTooltip && (
          <div
            className="fixed z-50 max-w-xs rounded-lg border border-sparc-gray-200 bg-white p-3 shadow-lg"
            style={{ left: edgeTooltip.x + 8, top: edgeTooltip.y - 8 }}
            onClick={(e) => e.stopPropagation()}
          >
            <p className="text-xs font-semibold text-sparc-gray-800 mb-1">
              Causal Assumption
            </p>
            <p className="text-xs text-sparc-gray-600">
              <span className="font-mono font-bold text-sparc-purple">{edgeTooltip.source}</span>
              {" → "}
              <span className="font-mono font-bold text-sparc-purple">{edgeTooltip.target}</span>
            </p>
            {(edgeTooltip.kind || edgeTooltip.sign_prior || edgeTooltip.confidence != null || edgeTooltip.lag != null) && (
              <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-2 gap-y-0.5 text-[10px] text-sparc-gray-600">
                {edgeTooltip.kind && (<><dt className="font-semibold">kind</dt><dd className="font-mono">{edgeTooltip.kind}</dd></>)}
                {edgeTooltip.sign_prior && (<><dt className="font-semibold">sign</dt><dd className="font-mono">{edgeTooltip.sign_prior}</dd></>)}
                {edgeTooltip.confidence != null && (<><dt className="font-semibold">confidence</dt><dd className="font-mono">{edgeTooltip.confidence.toFixed(2)}</dd></>)}
                {edgeTooltip.lag != null && (<><dt className="font-semibold">lag</dt><dd className="font-mono">{edgeTooltip.lag}</dd></>)}
              </dl>
            )}
            <p className="text-[11px] text-sparc-gray-500 mt-1.5 leading-relaxed">
              You are asserting that <strong>{edgeTooltip.source}</strong> causally influences{" "}
              <strong>{edgeTooltip.target}</strong>.
              {edgeTooltip.mechanism && (
                <> Mechanism: <em>{edgeTooltip.mechanism}</em>.</>
              )}
              {" "}This edge will be used for backdoor adjustment and counterfactual estimation.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
