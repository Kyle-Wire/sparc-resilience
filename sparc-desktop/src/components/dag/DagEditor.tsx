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
import type { DagEdge, DagDefinition, DagValidation, MC3Result, DagNodeType } from "@/lib/types";
import { usePipeline } from "@/hooks/PipelineProvider";
import { Btn, SectionHeader } from "@/components/ui/DesignSystem";

// ── Node type → brand color ───────────────────────────────────────────────
const NODE_COLORS: Record<string, string> = {
  treatment:        "#602468",
  mediator:         "#a44eb4",
  confounder:       "#e79024",
  outcome:          "#e73c25",
  instrument:       "#1e6fb8",
  proxy_confounder: "#f0b632",
  selection:        "#6e6358",
};

const NODE_TYPE_OPTIONS: DagNodeType[] = [
  "treatment","mediator","confounder","outcome","instrument","proxy_confounder","selection",
];

// ── Custom DAG node ───────────────────────────────────────────────────────
function SparcDagNode({ data }: NodeProps) {
  const d = data as { label: string; nodeType: string; description?: string };
  const color = NODE_COLORS[d.nodeType] ?? "#6e6358";
  return (
    <div style={{
      background:"#fff", border:"1px solid #c9c2b3", borderLeft:`4px solid ${color}`,
      borderRadius:6, minWidth:140, maxWidth:200, padding:"8px 10px",
      boxShadow:"0 2px 8px rgba(26,20,18,0.10)", fontFamily:"inherit",
    }}>
      <Handle type="target" position={Position.Top}
        style={{ width:8, height:8, background:"#c9c2b3", border:"1.5px solid #a09880" }} />
      <div className="mono"
        style={{ fontSize:11, fontWeight:700, color:"#1a1416", letterSpacing:"-0.01em", lineHeight:1.25 }}>
        {d.label}
      </div>
      <div style={{ fontSize:8.5, marginTop:4, fontWeight:700, letterSpacing:"0.08em",
        textTransform:"uppercase", color, opacity:0.9 }}>
        {d.nodeType.replace(/_/g, " ")}
      </div>
      {d.description && (
        <div style={{ fontSize:9, marginTop:3, color:"#6e6358", lineHeight:1.4 }}>{d.description}</div>
      )}
      <Handle type="source" position={Position.Bottom}
        style={{ width:8, height:8, background:"#c9c2b3", border:"1.5px solid #a09880" }} />
    </div>
  );
}

const nodeTypes = { dag: SparcDagNode };

// ── Dagre layout ──────────────────────────────────────────────────────────
function layoutNodes(nodes: Node[], edges: Edge[]): Node[] {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir:"TB", ranksep:90, nodesep:70 });
  nodes.forEach((n) => g.setNode(n.id, { width:160, height:70 }));
  edges.forEach((e) => g.setEdge(e.source, e.target));
  dagre.layout(g);
  return nodes.map((n) => {
    const pos = g.node(n.id);
    return { ...n, position:{ x:pos.x - 80, y:pos.y - 35 } };
  });
}

// ── API DAG → ReactFlow ───────────────────────────────────────────────────
function dagToFlow(dag: DagDefinition): { nodes: Node[]; edges: Edge[] } {
  const flowNodes: Node[] = dag.nodes.map((n) => ({
    id: n.name, type: "dag", position: { x:0, y:0 },
    data: { label:n.name, nodeType:n.type, description:n.description },
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
      style: { stroke:color, strokeWidth:1.6 },
      markerEnd: { type:MarkerType.ArrowClosed, color },
      labelStyle: { fontSize:9, fill:"#6e6358" },
    };
  });
  return { nodes: layoutNodes(flowNodes, flowEdges), edges: flowEdges };
}

// ── MC³ posterior → ReactFlow ─────────────────────────────────────────────
function mc3ToFlow(mc3: MC3Result, threshold: number): { nodes: Node[]; edges: Edge[] } {
  const { node_names, edge_probs } = mc3;
  const flowNodes: Node[] = node_names.map((name) => ({
    id: name, type: "dag", position: { x:0, y:0 }, data: { label:name, nodeType:"confounder" },
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
        style: { stroke:color, strokeWidth:Math.max(1, prob * 2.5), strokeDasharray:prob < 0.5 ? "4,3" : undefined },
        markerEnd: { type:MarkerType.ArrowClosed, color },
        labelStyle: { fontSize:9, fill:color, fontWeight:prob >= 0.5 ? 600 : 400 },
      });
    }
  }
  return { nodes: layoutNodes(flowNodes, flowEdges), edges: flowEdges };
}

// ── Undo / redo ───────────────────────────────────────────────────────────
interface Snapshot { nodes: Node[]; edges: Edge[] }
function useUndoRedo(
  nodes: Node[], edges: Edge[],
  setNodes: (ns: Node[]) => void, setEdges: (es: Edge[]) => void,
) {
  const histRef = useRef<Snapshot[]>([]);
  const futRef  = useRef<Snapshot[]>([]);
  const skipRef = useRef(false);
  const pushSnapshot = useCallback(() => {
    if (skipRef.current) return;
    histRef.current.push({ nodes:JSON.parse(JSON.stringify(nodes)), edges:JSON.parse(JSON.stringify(edges)) });
    futRef.current = [];
    if (histRef.current.length > 50) histRef.current.shift();
  }, [nodes, edges]);
  const undo = useCallback(() => {
    if (histRef.current.length === 0) return;
    const snap = histRef.current.pop()!;
    futRef.current.push({ nodes:JSON.parse(JSON.stringify(nodes)), edges:JSON.parse(JSON.stringify(edges)) });
    skipRef.current = true;
    setNodes(snap.nodes); setEdges(snap.edges);
    requestAnimationFrame(() => { skipRef.current = false; });
  }, [nodes, edges, setNodes, setEdges]);
  const redo = useCallback(() => {
    if (futRef.current.length === 0) return;
    const snap = futRef.current.pop()!;
    histRef.current.push({ nodes:JSON.parse(JSON.stringify(nodes)), edges:JSON.parse(JSON.stringify(edges)) });
    skipRef.current = true;
    setNodes(snap.nodes); setEdges(snap.edges);
    requestAnimationFrame(() => { skipRef.current = false; });
  }, [nodes, edges, setNodes, setEdges]);
  return { pushSnapshot, undo, redo, canUndo:histRef.current.length > 0, canRedo:futRef.current.length > 0 };
}

// ── Shared input style ────────────────────────────────────────────────────
const inputStyle: React.CSSProperties = {
  border:"1px solid var(--line)", borderRadius:4, padding:"5px 8px",
  fontSize:11.5, fontFamily:"inherit", background:"#fff",
  outline:"none", width:"100%", boxSizing:"border-box",
};

// ── Panel card ────────────────────────────────────────────────────────────
function PanelCard({ title, subtitle, children }: { title:string; subtitle?:string; children:React.ReactNode }) {
  return (
    <div style={{ background:"#fff", border:"1px solid var(--line)", borderRadius:8 }}>
      <div style={{ padding:"10px 12px", borderBottom:"1px solid var(--line)",
        background:"#fdfbf7", borderRadius:"8px 8px 0 0" }}>
        <div style={{ fontSize:12, fontWeight:700 }}>{title}</div>
        {subtitle && <div className="mono" style={{ fontSize:9.5, color:"var(--muted)", marginTop:1 }}>{subtitle}</div>}
      </div>
      <div style={{ padding:12 }}>{children}</div>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────
export default function DAGView() {
  const { dagApprovalPending, handleApproveDag, handleRejectDag } = usePipeline();

  const [mode, setMode] = useState<"build" | "mc3">("build");

  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [loading, setLoading]       = useState(true);
  const [error, setError]           = useState<string | null>(null);
  const [validation, setValidation] = useState<DagValidation | null>(null);

  const [mc3, setMc3]               = useState<MC3Result | null>(null);
  const [mc3Loading, setMc3Loading] = useState(false);
  const [mc3Error, setMc3Error]     = useState<string | null>(null);
  const [mc3Threshold, setMc3Threshold] = useState(0.30);
  const [mc3Nodes, setMc3Nodes, onMc3NodesChange] = useNodesState<Node>([]);
  const [mc3Edges, setMc3Edges, onMc3EdgesChange] = useEdgesState<Edge>([]);

  const { pushSnapshot, undo, redo, canUndo, canRedo } = useUndoRedo(nodes, edges, setNodes, setEdges);

  const [newNodeName, setNewNodeName] = useState("");
  const [newNodeType, setNewNodeType] = useState<DagNodeType>("treatment");
  const [qeSource, setQeSource]       = useState("");
  const [qeTarget, setQeTarget]       = useState("");
  const [contextMenu, setContextMenu] = useState<{ x:number; y:number; nodeId:string } | null>(null);

  const loadDag = useCallback(async () => {
    setLoading(true);
    try {
      const d = await getDag();
      const { nodes:n, edges:e } = dagToFlow(d);
      setNodes(n); setEdges(e); setError(null);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to load DAG");
    } finally { setLoading(false); }
  }, [setNodes, setEdges]);

  useEffect(() => { loadDag(); }, [loadDag]);

  useEffect(() => {
    if (mode !== "mc3") return;
    setMc3Loading(true);
    getMc3Result()
      .then((data) => {
        setMc3(data); setMc3Error(null);
        if (data) {
          const { nodes:n, edges:e } = mc3ToFlow(data, mc3Threshold);
          setMc3Nodes(n); setMc3Edges(e);
        }
      })
      .catch((err: unknown) => {
        setMc3Error(err instanceof Error ? err.message : "MC³ result not available");
      })
      .finally(() => setMc3Loading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode]);

  useEffect(() => {
    if (!mc3) return;
    const { nodes:n, edges:e } = mc3ToFlow(mc3, mc3Threshold);
    setMc3Nodes(n); setMc3Edges(e);
  }, [mc3, mc3Threshold, setMc3Nodes, setMc3Edges]);

  useEffect(() => { if (dagApprovalPending) setMode("mc3"); }, [dagApprovalPending]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (!(e.ctrlKey || e.metaKey)) return;
      if (e.key === "z" && !e.shiftKey) { e.preventDefault(); undo(); }
      if ((e.key === "z" && e.shiftKey) || e.key === "y") { e.preventDefault(); redo(); }
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [undo, redo]);

  const onConnect = useCallback((params: Connection) => {
    pushSnapshot();
    const srcType = (nodes.find((n) => n.id === params.source)?.data as { nodeType?:string })?.nodeType ?? "";
    const color = NODE_COLORS[srcType] ?? "#602468";
    setEdges((eds) => addEdge({ ...params,
      style:{ stroke:color, strokeWidth:1.6 },
      markerEnd:{ type:MarkerType.ArrowClosed, color } }, eds));
  }, [nodes, pushSnapshot, setEdges]);

  const handleAddNode = useCallback(() => {
    const name = newNodeName.trim();
    if (!name || nodes.find((n) => n.id === name)) return;
    pushSnapshot();
    const maxY = nodes.reduce((m, n) => Math.max(m, n.position.y), 0);
    setNodes((prev) => [...prev, {
      id: name, type: "dag",
      position:{ x:60 + (prev.length % 4) * 185, y:maxY + 130 },
      data:{ label:name, nodeType:newNodeType },
    }]);
    setNewNodeName("");
  }, [newNodeName, newNodeType, nodes, pushSnapshot, setNodes]);

  const handleAddEdge = useCallback(() => {
    if (!qeSource || !qeTarget || qeSource === qeTarget) return;
    pushSnapshot();
    const srcType = (nodes.find((n) => n.id === qeSource)?.data as { nodeType?:string })?.nodeType ?? "";
    const color = NODE_COLORS[srcType] ?? "#602468";
    setEdges((eds) => [...eds, {
      id: `qe-${qeSource}-${qeTarget}-${Date.now()}`,
      source:qeSource, target:qeTarget,
      style:{ stroke:color, strokeWidth:1.6 },
      markerEnd:{ type:MarkerType.ArrowClosed, color },
    }]);
    setQeSource(""); setQeTarget("");
  }, [qeSource, qeTarget, nodes, pushSnapshot, setEdges]);

  const onNodeContextMenu = useCallback((event: React.MouseEvent, node: Node) => {
    event.preventDefault();
    setContextMenu({ x:event.clientX, y:event.clientY, nodeId:node.id });
  }, []);

  const changeNodeType = useCallback((nodeId: string, newType: string) => {
    pushSnapshot();
    setNodes((nds) => nds.map((n) =>
      n.id === nodeId ? { ...n, data:{ ...n.data, nodeType:newType } } : n));
    setContextMenu(null);
  }, [pushSnapshot, setNodes]);

  const deleteNode = useCallback((nodeId: string) => {
    pushSnapshot();
    setNodes((ns) => ns.filter((n) => n.id !== nodeId));
    setEdges((es) => es.filter((e) => e.source !== nodeId && e.target !== nodeId));
    setContextMenu(null);
  }, [pushSnapshot, setNodes, setEdges]);

  useEffect(() => {
    if (!contextMenu) return;
    const close = () => setContextMenu(null);
    document.addEventListener("click", close);
    return () => document.removeEventListener("click", close);
  }, [contextMenu]);

  const handleValidate = async () => {
    const currentDag: DagDefinition = {
      nodes: nodes.map((n) => ({
        name: n.id,
        type: ((n.data as { nodeType?:string })?.nodeType ?? "confounder") as DagNodeType,
      })),
      edges: edges.filter((e) => !e.id.startsWith("mc3-"))
        .map((e) => ({ parent:e.source, child:e.target } as DagEdge)),
    };
    try {
      const v = await validateDag(currentDag);
      setValidation(v);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Validation failed");
    }
  };

  const importMc3 = useCallback(() => {
    if (!mc3) return;
    pushSnapshot();
    const importDag: DagDefinition = {
      nodes: mc3.median_dag.nodes.map((name) => ({ name, type:"confounder" as DagNodeType })),
      edges: mc3.median_dag.edges.map((me) => ({ parent:me.source, child:me.target } as DagEdge)),
    };
    const { nodes:n, edges:e } = dagToFlow(importDag);
    setNodes(n); setEdges(e); setMode("build");
  }, [mc3, pushSnapshot, setNodes, setEdges]);

  const structuralWarnings = useMemo<string[]>(() => {
    if (nodes.length === 0) return [];
    const warns: string[] = [];
    const userEdges = edges.filter((e) => !e.id.startsWith("mc3-"));
    const connected = new Set<string>();
    for (const e of userEdges) { connected.add(e.source); connected.add(e.target); }
    const disconnected = nodes.filter((n) => !connected.has(n.id));
    if (disconnected.length > 0)
      warns.push(`Disconnected: ${disconnected.map((n) => n.id).join(", ")}`);
    const outcomes = nodes.filter((n) => (n.data as { nodeType?:string }).nodeType === "outcome");
    for (const o of outcomes)
      if (userEdges.some((e) => e.source === o.id))
        warns.push(`Outcome "${o.id}" has outgoing edges`);
    return warns;
  }, [nodes, edges]);

  const mc3Legend: [string, string][] = [
    ["#2d8a2d", "≥80% — strong"],
    ["#e79024", "50–79% — moderate"],
    ["#9ca3af", "<50% — weak"],
  ];

  if (loading) {
    return (
      <div style={{ display:"flex", alignItems:"center", justifyContent:"center", height:300, color:"var(--muted)" }}>
        <span className="mono" style={{ fontSize:12 }}>Loading DAG…</span>
      </div>
    );
  }

  return (
    <div style={{ display:"flex", flexDirection:"column", height:"calc(100vh - 56px)" }}>

      {/* Header */}
      <div style={{ flexShrink:0, paddingBottom:10 }}>
        <SectionHeader
          kicker="04 · analysis"
          label="Causal DAG"
          right={
            <div style={{ display:"flex", alignItems:"center", gap:8 }}>
              <div style={{ display:"flex", border:"1px solid var(--line)", borderRadius:5, overflow:"hidden" }}>
                {(["build", "mc3"] as const).map((m, i) => (
                  <button key={m} onClick={() => setMode(m)} style={{
                    padding:"6px 14px", fontSize:11.5, fontWeight:600,
                    border:"none", borderLeft:i > 0 ? "1px solid var(--line)" : "none",
                    background:mode === m ? "var(--ink)" : "var(--paper-2)",
                    color:mode === m ? "#fff" : "var(--ink-2)",
                    cursor:"pointer", fontFamily:"inherit",
                    transition:"background 120ms, color 120ms",
                  }}>
                    {m === "build" ? "Build" : "MC³ Review"}
                  </button>
                ))}
              </div>
              {mode === "build" && (
                <>
                  <Btn small onClick={undo} disabled={!canUndo}>↶</Btn>
                  <Btn small onClick={redo} disabled={!canRedo}>↷</Btn>
                  <Btn small onClick={loadDag}>Reload</Btn>
                  <Btn small onClick={handleValidate}>Validate</Btn>
                </>
              )}
              {dagApprovalPending && (
                <>
                  <Btn small onClick={handleRejectDag}>Reject</Btn>
                  <Btn primary onClick={handleApproveDag}>Approve DAG</Btn>
                </>
              )}
            </div>
          }
        />
      </div>

      {/* Approval banner */}
      {dagApprovalPending && (
        <div style={{ flexShrink:0, marginBottom:8, padding:"10px 14px",
          background:"#fff8ed", border:"1px solid var(--amber)", borderRadius:6 }}>
          <div style={{ fontSize:12, fontWeight:700, color:"#7a4800" }}>
            DAG Review Required — Pipeline Paused
          </div>
          <div style={{ fontSize:11, color:"#8a5800", marginTop:2, lineHeight:1.5 }}>
            MC³ structure learning complete. Review suggested edges in{" "}
            <strong>MC³ Review</strong>, then approve to continue.
            {mc3 && ` ${mc3.median_dag.edges.length} edges in median DAG.`}
          </div>
        </div>
      )}

      {/* Validation banner */}
      {validation && (
        <div style={{ flexShrink:0, marginBottom:8, padding:"8px 14px",
          background:validation.valid ? "#f0faf0" : "#fff0f0",
          border:`1px solid ${validation.valid ? "#4a9a4a" : "var(--crimson)"}`,
          borderRadius:6, fontSize:12,
          color:validation.valid ? "#2a6a2a" : "var(--crimson)",
          display:"flex", justifyContent:"space-between", alignItems:"center" }}>
          <span>
            {validation.valid
              ? `✓ Valid — ${validation.n_nodes} nodes, ${validation.n_edges} edges`
              : `✗ ${validation.error}`}
          </span>
          <button onClick={() => setValidation(null)} style={{ background:"none", border:"none",
            cursor:"pointer", fontSize:16, lineHeight:1, color:"inherit", opacity:0.5, padding:"0 4px" }}>
            ×
          </button>
        </div>
      )}

      {/* Structural warnings */}
      {structuralWarnings.length > 0 && mode === "build" && (
        <div style={{ flexShrink:0, marginBottom:8, padding:"8px 14px",
          background:"#fffbf0", border:"1px solid var(--amber)", borderRadius:6 }}>
          {structuralWarnings.map((w, i) => (
            <div key={i} style={{ fontSize:11.5, color:"#7a4800" }}>⚠ {w}</div>
          ))}
        </div>
      )}

      {/* Error banner */}
      {error && (
        <div style={{ flexShrink:0, marginBottom:8, padding:"8px 14px",
          background:"#fff0f0", border:"1px solid var(--crimson)", borderRadius:6,
          fontSize:12, color:"var(--crimson)",
          display:"flex", justifyContent:"space-between", alignItems:"center" }}>
          <span>{error}</span>
          <button onClick={() => setError(null)} style={{ background:"none", border:"none",
            cursor:"pointer", fontSize:16, lineHeight:1, color:"inherit", opacity:0.5, padding:"0 4px" }}>
            ×
          </button>
        </div>
      )}

      {/* Content row */}
      <div style={{ flex:1, display:"flex", gap:12, minHeight:0 }}>

        {/* Side panel */}
        <div style={{ width:220, flexShrink:0, display:"flex", flexDirection:"column",
          gap:10, overflowY:"auto" }}>
          {mode === "build" ? (
            <>
              <PanelCard title="Add Node">
                <div style={{ display:"flex", flexDirection:"column", gap:8 }}>
                  <input type="text" value={newNodeName}
                    onChange={(e) => setNewNodeName(e.target.value)}
                    onKeyDown={(e) => { if (e.key === "Enter") handleAddNode(); }}
                    placeholder="Variable name…" style={inputStyle} />
                  <select value={newNodeType}
                    onChange={(e) => setNewNodeType(e.target.value as DagNodeType)}
                    style={{ ...inputStyle, color:NODE_COLORS[newNodeType], fontWeight:600 }}>
                    {NODE_TYPE_OPTIONS.map((t) => (
                      <option key={t} value={t}>{t.replace(/_/g, " ")}</option>
                    ))}
                  </select>
                  <Btn primary small onClick={handleAddNode} disabled={!newNodeName.trim()}>
                    Add Node
                  </Btn>
                </div>
              </PanelCard>

              <PanelCard title="Add Edge" subtitle="or drag from a node handle">
                <div style={{ display:"flex", flexDirection:"column", gap:8 }}>
                  <select value={qeSource} onChange={(e) => setQeSource(e.target.value)} style={inputStyle}>
                    <option value="">From…</option>
                    {nodes.map((n) => <option key={n.id} value={n.id}>{n.id}</option>)}
                  </select>
                  <select value={qeTarget} onChange={(e) => setQeTarget(e.target.value)} style={inputStyle}>
                    <option value="">To…</option>
                    {nodes.filter((n) => n.id !== qeSource).map((n) => (
                      <option key={n.id} value={n.id}>{n.id}</option>
                    ))}
                  </select>
                  <Btn primary small onClick={handleAddEdge} disabled={!qeSource || !qeTarget}>
                    Add Edge
                  </Btn>
                </div>
              </PanelCard>

              <PanelCard title="Node types">
                <div style={{ display:"flex", flexDirection:"column", gap:6 }}>
                  {NODE_TYPE_OPTIONS.map((t) => (
                    <div key={t} style={{ display:"flex", alignItems:"center", gap:8 }}>
                      <span style={{ width:4, height:16, borderRadius:2, background:NODE_COLORS[t], flexShrink:0 }} />
                      <span style={{ fontSize:11, color:"var(--ink-2)", textTransform:"capitalize" }}>
                        {t.replace(/_/g, " ")}
                      </span>
                    </div>
                  ))}
                  <div style={{ marginTop:8, paddingTop:8, borderTop:"1px dashed var(--line)",
                    fontSize:10, color:"var(--muted)", lineHeight:1.6 }}>
                    Right-click any node to change its type or delete it.
                  </div>
                </div>
              </PanelCard>
            </>
          ) : (
            <>
              <PanelCard title="Threshold" subtitle="min edge probability">
                <div style={{ display:"flex", flexDirection:"column", gap:10 }}>
                  <div style={{ display:"flex", alignItems:"center", gap:8 }}>
                    <input type="range" min={0.05} max={0.95} step={0.05}
                      value={mc3Threshold}
                      onChange={(e) => setMc3Threshold(parseFloat(e.target.value))}
                      style={{ flex:1 }} />
                    <span className="mono" style={{ fontSize:13, fontWeight:700, width:40, textAlign:"right" }}>
                      {(mc3Threshold * 100).toFixed(0)}%
                    </span>
                  </div>
                  {mc3 ? (
                    <div style={{ display:"flex", flexDirection:"column", gap:4 }}>
                      <div style={{ fontSize:11, color:"var(--muted)" }}>
                        <strong style={{ color:"var(--ink)" }}>{mc3Edges.length}</strong> edges shown
                      </div>
                      <div style={{ fontSize:11, color:"var(--muted)" }}>
                        <strong style={{ color:"var(--ink)" }}>{mc3.node_names.length}</strong> nodes
                      </div>
                      <div style={{ fontSize:11, color:"var(--muted)" }}>
                        Acceptance:{" "}
                        <strong style={{ color:"var(--ink)" }}>
                          {(mc3.mc3_summary.acceptance_rate * 100).toFixed(1)}%
                        </strong>
                      </div>
                    </div>
                  ) : (
                    <div style={{ fontSize:11, color:"var(--muted)", fontStyle:"italic" }}>
                      {mc3Loading ? "Loading…" : "No results yet"}
                    </div>
                  )}
                </div>
              </PanelCard>

              <PanelCard title="Edge strength">
                <div style={{ display:"flex", flexDirection:"column", gap:8 }}>
                  {mc3Legend.map(([color, label]) => (
                    <div key={label} style={{ display:"flex", alignItems:"center", gap:10 }}>
                      <span style={{ width:24, height:2.5, background:color, flexShrink:0, borderRadius:1 }} />
                      <span style={{ fontSize:11, color:"var(--ink-2)" }}>{label}</span>
                    </div>
                  ))}
                </div>
              </PanelCard>

              <Btn primary onClick={importMc3} disabled={!mc3}>Import to Build</Btn>

              <div style={{ fontSize:10.5, color:"var(--muted)", lineHeight:1.6 }}>
                Imports the median DAG into your Build canvas. Node types default to{" "}
                <em>confounder</em> — reassign them after importing.
              </div>
            </>
          )}
        </div>

        {/* Canvas */}
        <div style={{ flex:1, minWidth:0, borderRadius:8, border:"1px solid var(--line)",
          overflow:"hidden", background:"var(--paper-2)", position:"relative" }}>
          {mode === "build" ? (
            nodes.length === 0 ? (
              <div style={{ display:"flex", flexDirection:"column", alignItems:"center",
                justifyContent:"center", height:"100%", gap:14, padding:40 }}>
                <div style={{ fontSize:36, opacity:0.18, lineHeight:1 }}>◎</div>
                <div style={{ fontSize:15, fontWeight:700, color:"var(--ink-2)" }}>No nodes yet</div>
                <div style={{ fontSize:12, color:"var(--muted)", maxWidth:300, textAlign:"center", lineHeight:1.7 }}>
                  Use <strong>Add Node</strong> to name your variables, then draw edges by
                  dragging from node handles or using <strong>Add Edge</strong>.
                </div>
              </div>
            ) : (
              <ReactFlow nodes={nodes} edges={edges}
                onNodesChange={onNodesChange} onEdgesChange={onEdgesChange}
                onConnect={onConnect} onNodeContextMenu={onNodeContextMenu}
                nodeTypes={nodeTypes} fitView fitViewOptions={{ padding:0.3 }}
                proOptions={{ hideAttribution:true }} style={{ background:"var(--paper-2)" }}>
                <Background variant={BackgroundVariant.Dots} gap={18} size={1} color="#c9c2b3" />
                <Controls showInteractive={false} />
              </ReactFlow>
            )
          ) : mc3Loading ? (
            <div style={{ display:"flex", alignItems:"center", justifyContent:"center",
              height:"100%", color:"var(--muted)" }}>
              <span className="mono" style={{ fontSize:12 }}>Loading MC³ result…</span>
            </div>
          ) : mc3Error ? (
            <div style={{ display:"flex", flexDirection:"column", alignItems:"center",
              justifyContent:"center", height:"100%", gap:12, padding:40 }}>
              <div style={{ fontSize:15, fontWeight:700, color:"var(--ink-2)" }}>MC³ not available</div>
              <div style={{ fontSize:12, color:"var(--muted)", maxWidth:320, textAlign:"center", lineHeight:1.7 }}>
                {mc3Error}. Run the pipeline through the causal stage first.
              </div>
            </div>
          ) : (
            <ReactFlow nodes={mc3Nodes} edges={mc3Edges}
              onNodesChange={onMc3NodesChange} onEdgesChange={onMc3EdgesChange}
              nodeTypes={nodeTypes} fitView fitViewOptions={{ padding:0.3 }}
              proOptions={{ hideAttribution:true }} style={{ background:"var(--paper-2)" }}
              nodesConnectable={false} nodesDraggable={true}>
              <Background variant={BackgroundVariant.Dots} gap={18} size={1} color="#c9c2b3" />
              <Controls showInteractive={false} />
            </ReactFlow>
          )}
        </div>
      </div>

      {/* Context menu */}
      {contextMenu && (
        <div onClick={(e) => e.stopPropagation()} style={{
          position:"fixed", left:contextMenu.x, top:contextMenu.y, zIndex:9999,
          background:"#fff", border:"1px solid var(--line)", borderRadius:6,
          boxShadow:"0 4px 18px rgba(26,20,18,0.16)",
          paddingTop:4, paddingBottom:4, minWidth:170,
        }}>
          <div className="mono" style={{ padding:"4px 12px 8px", fontSize:9,
            color:"var(--muted)", letterSpacing:"0.1em", textTransform:"uppercase",
            borderBottom:"1px solid var(--line)", marginBottom:4 }}>
            Change type
          </div>
          {NODE_TYPE_OPTIONS.map((type) => (
            <button key={type} onClick={() => changeNodeType(contextMenu.nodeId, type)}
              onMouseEnter={(e) => { (e.currentTarget as HTMLButtonElement).style.background = "var(--paper-2)"; }}
              onMouseLeave={(e) => { (e.currentTarget as HTMLButtonElement).style.background = "none"; }}
              style={{ display:"flex", alignItems:"center", gap:10, width:"100%",
                padding:"6px 12px", border:"none", background:"none", cursor:"pointer",
                fontSize:12, fontFamily:"inherit", textAlign:"left", color:"var(--ink-2)" }}>
              <span style={{ width:4, height:14, borderRadius:2, background:NODE_COLORS[type], flexShrink:0 }} />
              <span style={{ textTransform:"capitalize" }}>{type.replace(/_/g, " ")}</span>
            </button>
          ))}
          <div style={{ borderTop:"1px solid var(--line)", marginTop:4, paddingTop:4 }}>
            <button onClick={() => deleteNode(contextMenu.nodeId)}
              onMouseEnter={(e) => { (e.currentTarget as HTMLButtonElement).style.background = "#fff4f4"; }}
              onMouseLeave={(e) => { (e.currentTarget as HTMLButtonElement).style.background = "none"; }}
              style={{ display:"flex", alignItems:"center", gap:10, width:"100%",
                padding:"6px 12px", border:"none", background:"none", cursor:"pointer",
                fontSize:12, fontFamily:"inherit", textAlign:"left", color:"var(--crimson)" }}>
              Delete node
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
