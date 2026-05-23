/**
 * useDagMutations — deep hook that owns all DAG graph mutations.
 *
 * Every mutation (addNode, addEdge, deleteNode, changeNodeType, connect,
 * acceptSuggestion, replaceGraph) automatically snapshots before applying,
 * so callers never need to call pushSnapshot() manually.
 *
 * Undo/redo history and the skipRef guard are hidden from callers.
 */
import { useCallback, useRef } from "react";
import { type Node, type Edge, addEdge, type Connection, MarkerType } from "@xyflow/react";
import type { DagNodeType, DagEdgeSuggestion } from "@/lib/types";
import { NODE_COLORS } from "@/lib/dagFlowTransformer";

type NodeSetter = (updater: Node[] | ((prev: Node[]) => Node[])) => void;
type EdgeSetter = (updater: Edge[] | ((prev: Edge[]) => Edge[])) => void;

interface Snapshot {
  nodes: Node[];
  edges: Edge[];
}

export function useDagMutations(
  nodes: Node[],
  edges: Edge[],
  setNodes: NodeSetter,
  setEdges: EdgeSetter,
) {
  const histRef = useRef<Snapshot[]>([]);
  const futRef  = useRef<Snapshot[]>([]);
  const skipRef = useRef(false);

  // ── Internal snapshot — not exposed ──────────────────────────────────
  const snapshot = useCallback(() => {
    if (skipRef.current) return;
    histRef.current.push({
      nodes: JSON.parse(JSON.stringify(nodes)),
      edges: JSON.parse(JSON.stringify(edges)),
    });
    futRef.current = [];
    if (histRef.current.length > 50) histRef.current.shift();
  }, [nodes, edges]);

  // ── Undo / redo ───────────────────────────────────────────────────────
  const undo = useCallback(() => {
    if (histRef.current.length === 0) return;
    const snap = histRef.current.pop()!;
    futRef.current.push({
      nodes: JSON.parse(JSON.stringify(nodes)),
      edges: JSON.parse(JSON.stringify(edges)),
    });
    skipRef.current = true;
    setNodes(snap.nodes);
    setEdges(snap.edges);
    requestAnimationFrame(() => { skipRef.current = false; });
  }, [nodes, edges, setNodes, setEdges]);

  const redo = useCallback(() => {
    if (futRef.current.length === 0) return;
    const snap = futRef.current.pop()!;
    histRef.current.push({
      nodes: JSON.parse(JSON.stringify(nodes)),
      edges: JSON.parse(JSON.stringify(edges)),
    });
    skipRef.current = true;
    setNodes(snap.nodes);
    setEdges(snap.edges);
    requestAnimationFrame(() => { skipRef.current = false; });
  }, [nodes, edges, setNodes, setEdges]);

  // ── Mutations ─────────────────────────────────────────────────────────
  const addNode = useCallback((name: string, nodeType: DagNodeType) => {
    snapshot();
    const maxY = nodes.reduce((m, n) => Math.max(m, n.position.y), 0);
    setNodes((prev) => [
      ...prev,
      {
        id: name,
        type: "dag",
        position: { x: 60 + (prev.length % 4) * 185, y: maxY + 130 },
        data: { label: name, nodeType },
      },
    ]);
  }, [snapshot, nodes, setNodes]);

  const addEdgeByIds = useCallback((source: string, target: string) => {
    snapshot();
    const srcType = (nodes.find((n) => n.id === source)?.data as { nodeType?: string })?.nodeType ?? "";
    const color = NODE_COLORS[srcType] ?? "#602468";
    setEdges((eds) => [
      ...eds,
      {
        id: `qe-${source}-${target}-${Date.now()}`,
        source,
        target,
        style: { stroke: color, strokeWidth: 1.6 },
        markerEnd: { type: MarkerType.ArrowClosed, color },
      },
    ]);
  }, [snapshot, nodes, setEdges]);

  const connect = useCallback((params: Connection) => {
    snapshot();
    const srcType = (nodes.find((n) => n.id === params.source)?.data as { nodeType?: string })?.nodeType ?? "";
    const color = NODE_COLORS[srcType] ?? "#602468";
    setEdges((eds) =>
      addEdge(
        { ...params, style: { stroke: color, strokeWidth: 1.6 }, markerEnd: { type: MarkerType.ArrowClosed, color } },
        eds,
      ),
    );
  }, [snapshot, nodes, setEdges]);

  const deleteNode = useCallback((nodeId: string) => {
    snapshot();
    setNodes((ns) => ns.filter((n) => n.id !== nodeId));
    setEdges((es) => es.filter((e) => e.source !== nodeId && e.target !== nodeId));
  }, [snapshot, setNodes, setEdges]);

  const changeNodeType = useCallback((nodeId: string, newType: string) => {
    snapshot();
    setNodes((nds) =>
      nds.map((n) =>
        n.id === nodeId ? { ...n, data: { ...n.data, nodeType: newType } } : n,
      ),
    );
  }, [snapshot, setNodes]);

  const acceptSuggestion = useCallback((s: DagEdgeSuggestion) => {
    snapshot();
    const srcType = (nodes.find((n) => n.id === s.parent)?.data as { nodeType?: string })?.nodeType ?? "";
    const color = NODE_COLORS[srcType] ?? "#602468";
    setEdges((eds) => [
      ...eds,
      {
        id: `suggest-${s.parent}-${s.child}-${Date.now()}`,
        source: s.parent,
        target: s.child,
        style: { stroke: color, strokeWidth: 1.6 },
        markerEnd: { type: MarkerType.ArrowClosed, color },
      },
    ]);
  }, [snapshot, nodes, setEdges]);

  const replaceGraph = useCallback((newNodes: Node[], newEdges: Edge[]) => {
    snapshot();
    setNodes(newNodes);
    setEdges(newEdges);
  }, [snapshot, setNodes, setEdges]);

  return {
    addNode,
    addEdgeByIds,
    connect,
    deleteNode,
    changeNodeType,
    acceptSuggestion,
    replaceGraph,
    undo,
    redo,
    canUndo: histRef.current.length > 0,
    canRedo: futRef.current.length > 0,
  };
}
