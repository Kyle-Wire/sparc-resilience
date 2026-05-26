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
  type Connection,
  Handle,
  Position,
  type NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { fromExpert, fromMc3, NODE_COLORS, NODE_TYPE_OPTIONS } from "@/lib/dagFlowTransformer";
import { validateDagStructure } from "@/lib/dagValidate";
import { mergeMc3IntoDag } from "@/lib/dagMerge";
import { useDagMutations } from "@/hooks/useDagMutations";
import { getDag, validateDag, getMc3Result, suggestDagEdges, saveDag } from "@/lib/api";
import type { DagEdge, DagDefinition, DagValidation, MC3Result, DagNodeType, DagEdgeSuggestion } from "@/lib/types";
import { usePipeline } from "@/hooks/PipelineProvider";
import { Btn, SectionHeader } from "@/components/ui/DesignSystem";
import { useProjectStore } from "@/stores/projectStore";

// -- Custom DAG node -------------------------------------------------------
function SparcDagNode({ data }: NodeProps) {
  const d = data as { label: string; nodeType: string; description?: string };
  const color = NODE_COLORS[d.nodeType] ?? "#6e6358";
  return (
    <div style={{
      background: "#fff", border: "1px solid #c9c2b3", borderLeft: `4px solid ${color}`,
      borderRadius: 6, minWidth: 140, maxWidth: 200, padding: "8px 10px",
      boxShadow: "0 2px 8px rgba(26,20,18,0.10)", fontFamily: "inherit",
    }}>
      <Handle type="target" position={Position.Top}
        style={{ width: 8, height: 8, background: "#c9c2b3", border: "1.5px solid #a09880" }} />
      <div className="mono"
        style={{ fontSize: 11, fontWeight: 700, color: "#1a1416", letterSpacing: "-0.01em", lineHeight: 1.25 }}>
        {d.label}
      </div>
      <div style={{
        fontSize: 8.5, marginTop: 4, fontWeight: 700, letterSpacing: "0.08em",
        textTransform: "uppercase", color, opacity: 0.9,
      }}>
        {d.nodeType.replace(/_/g, " ")}
      </div>
      {d.description && (
        <div style={{ fontSize: 9, marginTop: 3, color: "#6e6358", lineHeight: 1.4 }}>{d.description}</div>
      )}
      <Handle type="source" position={Position.Bottom}
        style={{ width: 8, height: 8, background: "#c9c2b3", border: "1.5px solid #a09880" }} />
    </div>
  );
}

const nodeTypes = { dag: SparcDagNode };

const inputStyle: React.CSSProperties = {
  border: "1px solid var(--line)", borderRadius: 4, padding: "5px 8px",
  fontSize: 11.5, fontFamily: "inherit", background: "#fff",
  outline: "none", width: "100%", boxSizing: "border-box",
};

function PanelCard({ title, subtitle, children }: { title: string; subtitle?: string; children: React.ReactNode }) {
  return (
    <div style={{ background: "#fff", border: "1px solid var(--line)", borderRadius: 8 }}>
      <div style={{
        padding: "10px 12px", borderBottom: "1px solid var(--line)",
        background: "#fdfbf7", borderRadius: "8px 8px 0 0",
      }}>
        <div style={{ fontSize: 12, fontWeight: 700 }}>{title}</div>
        {subtitle && <div className="mono" style={{ fontSize: 9.5, color: "var(--muted)", marginTop: 1 }}>{subtitle}</div>}
      </div>
      <div style={{ padding: 12 }}>{children}</div>
    </div>
  );
}

// -- Main component --------------------------------------------------------
export default function DAGView() {
  const { dagApprovalPending, handleApproveDag, handleRejectDag } = usePipeline();

  const [mode, setMode] = useState<"build" | "mc3">("build");

  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [loading, setLoading]       = useState(true);
  const [error, setError]           = useState<string | null>(null);
  const [noProject, setNoProject]   = useState(false);
  const projectLoaded = useProjectStore((s) => s.projectLoaded);
  const [validation, setValidation] = useState<DagValidation | null>(null);

  const [mc3, setMc3]               = useState<MC3Result | null>(null);
  const [mc3Loading, setMc3Loading] = useState(false);
  const [mc3Error, setMc3Error]     = useState<string | null>(null);
  const [mc3Threshold, setMc3Threshold] = useState(0.30);
  const [mc3Nodes, setMc3Nodes, onMc3NodesChange] = useNodesState<Node>([]);
  const [mc3Edges, setMc3Edges, onMc3EdgesChange] = useEdgesState<Edge>([]);

  const [saveLoading, setSaveLoading] = useState(false);
  const [saveSaved, setSaveSaved]     = useState(false);

  const [newNodeName, setNewNodeName] = useState("");
  const [newNodeType, setNewNodeType] = useState<DagNodeType>("treatment");
  const [qeSource, setQeSource]       = useState("");
  const [qeTarget, setQeTarget]       = useState("");
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; nodeId: string } | null>(null);

  const [suggestions, setSuggestions] = useState<DagEdgeSuggestion[]>([]);
  const [suggestLoading, setSuggestLoading] = useState(false);

  const mutations = useDagMutations(nodes, edges, setNodes, setEdges);

  const loadDag = useCallback(async () => {
    setLoading(true);
    setNoProject(false);
    try {
      const d = await getDag();
      const { nodes: n, edges: e } = fromExpert(d);
      setNodes(n); setEdges(e); setError(null);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to load DAG";
      if (/no project loaded/i.test(msg)) {
        setNoProject(true);
        setError(null);
      } else {
        setError(msg);
      }
    } finally { setLoading(false); }
  }, [setNodes, setEdges]);

  useEffect(() => { loadDag(); }, [loadDag]);

  // Auto-reload when a project is opened
  useEffect(() => {
    if (projectLoaded) loadDag();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectLoaded]);

  useEffect(() => {
    if (mode !== "mc3") return;
    setMc3Loading(true);
    getMc3Result()
      .then((data) => {
        setMc3(data); setMc3Error(null);
        if (data) {
          const { nodes: n, edges: e } = fromMc3(data, mc3Threshold);
          setMc3Nodes(n); setMc3Edges(e);
        }
      })
      .catch((err: unknown) => {
        setMc3Error(err instanceof Error ? err.message : "MC3 result not available");
      })
      .finally(() => setMc3Loading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode]);

  useEffect(() => {
    if (!mc3) return;
    const { nodes: n, edges: e } = fromMc3(mc3, mc3Threshold);
    setMc3Nodes(n); setMc3Edges(e);
  }, [mc3, mc3Threshold, setMc3Nodes, setMc3Edges]);

  useEffect(() => { if (dagApprovalPending) setMode("mc3"); }, [dagApprovalPending]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (!(e.ctrlKey || e.metaKey)) return;
      if (e.key === "z" && !e.shiftKey) { e.preventDefault(); mutations.undo(); }
      if ((e.key === "z" && e.shiftKey) || e.key === "y") { e.preventDefault(); mutations.redo(); }
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [mutations]);

  useEffect(() => {
    if (!contextMenu) return;
    const close = () => setContextMenu(null);
    document.addEventListener("click", close);
    return () => document.removeEventListener("click", close);
  }, [contextMenu]);

  const handleAddNode = useCallback(() => {
    const name = newNodeName.trim();
    if (!name || nodes.find((n) => n.id === name)) return;
    mutations.addNode(name, newNodeType);
    setNewNodeName("");
  }, [newNodeName, newNodeType, nodes, mutations]);

  const handleAddEdge = useCallback(() => {
    if (!qeSource || !qeTarget || qeSource === qeTarget) return;
    mutations.addEdgeByIds(qeSource, qeTarget);
    setQeSource(""); setQeTarget("");
  }, [qeSource, qeTarget, mutations]);

  const onConnect = useCallback((params: Connection) => {
    mutations.connect(params);
  }, [mutations]);

  const onNodeContextMenu = useCallback((event: React.MouseEvent, node: Node) => {
    event.preventDefault();
    setContextMenu({ x: event.clientX, y: event.clientY, nodeId: node.id });
  }, []);

  const handleValidate = async () => {
    const currentDag: DagDefinition = {
      nodes: nodes.map((n) => ({
        name: n.id,
        type: ((n.data as { nodeType?: string })?.nodeType ?? "confounder") as DagNodeType,
      })),
      edges: edges.filter((e) => !e.id.startsWith("mc3-"))
        .map((e) => ({ parent: e.source, child: e.target } as DagEdge)),
    };
    try {
      const v = await validateDag(currentDag);
      setValidation(v);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Validation failed");
    }
  };

  const handleSave = async () => {
    const currentDag: DagDefinition = {
      nodes: nodes.map((n) => ({
        name: n.id,
        type: ((n.data as { nodeType?: string })?.nodeType ?? "confounder") as DagNodeType,
      })),
      edges: edges.filter((e) => !e.id.startsWith("mc3-"))
        .map((e) => ({ parent: e.source, child: e.target } as DagEdge)),
    };
    setSaveLoading(true);
    try {
      await saveDag(currentDag);
      setSaveSaved(true);
      setTimeout(() => setSaveSaved(false), 2000);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Save failed");
    } finally {
      setSaveLoading(false);
    }
  };

  const handleMergeImport = useCallback(() => {
    if (!mc3) return;
    const expertDag: DagDefinition = {
      nodes: nodes.map((n) => ({
        name: n.id,
        type: ((n.data as { nodeType?: string })?.nodeType ?? "confounder") as DagNodeType,
      })),
      edges: edges.filter((e) => !e.id.startsWith("mc3-"))
        .map((e) => ({ parent: e.source, child: e.target } as DagEdge)),
    };
    const merged = mergeMc3IntoDag(expertDag, mc3, mc3Threshold);
    const { nodes: n, edges: e } = fromExpert(merged);
    mutations.replaceGraph(n, e);
    setMode("build");
  }, [mc3, mc3Threshold, nodes, edges, mutations]);

  const structuralWarnings = useMemo(
    () => validateDagStructure(nodes, edges),
    [nodes, edges],
  );

  const mc3Legend: [string, string][] = [
    ["#2d8a2d", ">=80% - strong"],
    ["#e79024", "50-79% - moderate"],
    ["#9ca3af", "<50% - weak"],
  ];

  if (loading) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: 300, color: "var(--muted)" }}>
        <span className="mono" style={{ fontSize: 12 }}>Loading DAG...</span>
      </div>
    );
  }

  if (noProject) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: 300, color: "var(--muted)", flexDirection: "column", gap: 8 }}>
        <span className="mono" style={{ fontSize: 12 }}>No project — load a project to use the causal DAG editor.</span>
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "calc(100vh - 56px)" }}>
      <div style={{ flexShrink: 0, paddingBottom: 10 }}>
        <SectionHeader
          kicker="04 . analysis"
          label="Causal DAG"
          right={
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <div style={{ display: "flex", border: "1px solid var(--line)", borderRadius: 5, overflow: "hidden" }}>
                {(["build", "mc3"] as const).map((m, i) => (
                  <button key={m} onClick={() => setMode(m)} style={{
                    padding: "6px 14px", fontSize: 11.5, fontWeight: 600,
                    border: "none", borderLeft: i > 0 ? "1px solid var(--line)" : "none",
                    background: mode === m ? "var(--ink)" : "var(--paper-2)",
                    color: mode === m ? "#fff" : "var(--ink-2)",
                    cursor: "pointer", fontFamily: "inherit",
                    transition: "background 120ms, color 120ms",
                  }}>
                    {m === "build" ? "Build" : "MC3 Review"}
                  </button>
                ))}
              </div>
              {mode === "build" && (
                <>
                  <Btn small onClick={mutations.undo} disabled={!mutations.canUndo}>undo</Btn>
                  <Btn small onClick={mutations.redo} disabled={!mutations.canRedo}>redo</Btn>
                  <Btn small onClick={loadDag}>Reload</Btn>
                  <Btn small onClick={handleValidate}>Validate</Btn>
                  <Btn small onClick={handleSave} disabled={saveLoading}>
                    {saveLoading ? "Saving..." : saveSaved ? "Saved" : "Save"}
                  </Btn>
                  <Btn
                    small
                    disabled={suggestLoading}
                    onClick={async () => {
                      setSuggestLoading(true);
                      try {
                        const res = await suggestDagEdges({ max_suggestions: 10 });
                        setSuggestions(res.suggestions);
                      } catch {
                        setSuggestions([]);
                      } finally {
                        setSuggestLoading(false);
                      }
                    }}
                  >
                    {suggestLoading ? "..." : "Suggest edges"}
                  </Btn>
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

      {dagApprovalPending && (
        <div style={{
          flexShrink: 0, marginBottom: 8, padding: "10px 14px",
          background: "#fff8ed", border: "1px solid var(--amber)", borderRadius: 6,
        }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: "#7a4800" }}>
            DAG Review Required -- Pipeline Paused
          </div>
          <div style={{ fontSize: 11, color: "#8a5800", marginTop: 2, lineHeight: 1.5 }}>
            MC3 structure learning complete. Review suggested edges in{" "}
            <strong>MC3 Review</strong>, then approve to continue.
            {mc3 && ` ${mc3.median_dag.edges.length} edges in median DAG.`}
          </div>
        </div>
      )}

      {validation && (
        <div style={{
          flexShrink: 0, marginBottom: 8, padding: "8px 14px",
          background: validation.valid ? "#f0faf0" : "#fff0f0",
          border: `1px solid ${validation.valid ? "#4a9a4a" : "var(--crimson)"}`,
          borderRadius: 6, fontSize: 12,
          color: validation.valid ? "#2a6a2a" : "var(--crimson)",
          display: "flex", justifyContent: "space-between", alignItems: "center",
        }}>
          <span>
            {validation.valid
              ? `Valid -- ${validation.n_nodes} nodes, ${validation.n_edges} edges`
              : `${validation.error}`}
          </span>
          <button onClick={() => setValidation(null)} style={{
            background: "none", border: "none",
            cursor: "pointer", fontSize: 16, lineHeight: 1, color: "inherit", opacity: 0.5, padding: "0 4px",
          }}>x</button>
        </div>
      )}

      {structuralWarnings.length > 0 && mode === "build" && (
        <div style={{
          flexShrink: 0, marginBottom: 8, padding: "8px 14px",
          background: "#fffbf0", border: "1px solid var(--amber)", borderRadius: 6,
        }}>
          {structuralWarnings.map((w, i) => (
            <div key={i} style={{ fontSize: 11.5, color: "#7a4800" }}>! {w}</div>
          ))}
        </div>
      )}

      {error && (
        <div style={{
          flexShrink: 0, marginBottom: 8, padding: "8px 14px",
          background: "#fff0f0", border: "1px solid var(--crimson)", borderRadius: 6,
          fontSize: 12, color: "var(--crimson)",
          display: "flex", justifyContent: "space-between", alignItems: "center",
        }}>
          <span>{error}</span>
          <button onClick={() => setError(null)} style={{
            background: "none", border: "none",
            cursor: "pointer", fontSize: 16, lineHeight: 1, color: "inherit", opacity: 0.5, padding: "0 4px",
          }}>x</button>
        </div>
      )}

      <div style={{ flex: 1, display: "flex", gap: 12, minHeight: 0 }}>
        <div style={{
          width: 220, flexShrink: 0, display: "flex", flexDirection: "column",
          gap: 10, overflowY: "auto",
        }}>
          {mode === "build" ? (
            <>
              <PanelCard title="Add Node">
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  <input type="text" value={newNodeName}
                    onChange={(e) => setNewNodeName(e.target.value)}
                    onKeyDown={(e) => { if (e.key === "Enter") handleAddNode(); }}
                    placeholder="Variable name..." style={inputStyle} />
                  <select value={newNodeType}
                    onChange={(e) => setNewNodeType(e.target.value as DagNodeType)}
                    style={{ ...inputStyle, color: NODE_COLORS[newNodeType], fontWeight: 600 }}>
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
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  <select value={qeSource} onChange={(e) => setQeSource(e.target.value)} style={inputStyle}>
                    <option value="">From...</option>
                    {nodes.map((n) => <option key={n.id} value={n.id}>{n.id}</option>)}
                  </select>
                  <select value={qeTarget} onChange={(e) => setQeTarget(e.target.value)} style={inputStyle}>
                    <option value="">To...</option>
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
                <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  {NODE_TYPE_OPTIONS.map((t) => (
                    <div key={t} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <span style={{ width: 4, height: 16, borderRadius: 2, background: NODE_COLORS[t], flexShrink: 0 }} />
                      <span style={{ fontSize: 11, color: "var(--ink-2)", textTransform: "capitalize" }}>
                        {t.replace(/_/g, " ")}
                      </span>
                    </div>
                  ))}
                  <div style={{
                    marginTop: 8, paddingTop: 8, borderTop: "1px dashed var(--line)",
                    fontSize: 10, color: "var(--muted)", lineHeight: 1.6,
                  }}>
                    Right-click any node to change its type or delete it.
                  </div>
                </div>
              </PanelCard>

              {suggestions.length > 0 && (
                <PanelCard title="AI suggestions" subtitle="click to accept edge">
                  <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
                    {suggestions.map((s, i) => (
                      <div
                        key={i}
                        onClick={() => {
                          mutations.acceptSuggestion(s);
                          setSuggestions((prev) => prev.filter((_, j) => j !== i));
                        }}
                        title={s.reason}
                        style={{
                          display: "flex", alignItems: "center", gap: 6,
                          padding: "5px 4px", borderRadius: 4, cursor: "pointer",
                          borderTop: i > 0 ? "1px dashed var(--line)" : undefined,
                          transition: "background 100ms",
                        }}
                        onMouseEnter={(e) => (e.currentTarget.style.background = "rgba(0,0,0,0.04)")}
                        onMouseLeave={(e) => (e.currentTarget.style.background = "")}
                      >
                        <span className="mono" style={{
                          fontSize: 10, flex: 1, overflow: "hidden",
                          textOverflow: "ellipsis", whiteSpace: "nowrap",
                        }}>
                          {s.parent} {s.child}
                        </span>
                        <span className="mono" style={{
                          fontSize: 9, padding: "1px 5px", borderRadius: 3,
                          background: s.toward_outcome ? "rgba(231,60,37,0.12)" : "rgba(0,0,0,0.06)",
                          color: s.toward_outcome ? "var(--crimson)" : "var(--muted)",
                          flexShrink: 0,
                        }}>
                          {s.score.toFixed(2)}
                        </span>
                      </div>
                    ))}
                    <button
                      onClick={() => setSuggestions([])}
                      style={{
                        marginTop: 6, fontSize: 10, color: "var(--muted)",
                        background: "none", border: 0, cursor: "pointer", textAlign: "left", padding: 0,
                      }}
                    >
                      dismiss all
                    </button>
                  </div>
                </PanelCard>
              )}
            </>
          ) : (
            <>
              <PanelCard title="Threshold" subtitle="min edge probability">
                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <input type="range" min={0.05} max={0.95} step={0.05}
                      value={mc3Threshold}
                      onChange={(e) => setMc3Threshold(parseFloat(e.target.value))}
                      style={{ flex: 1 }} />
                    <span className="mono" style={{ fontSize: 13, fontWeight: 700, width: 40, textAlign: "right" }}>
                      {(mc3Threshold * 100).toFixed(0)}%
                    </span>
                  </div>
                  {mc3 ? (
                    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                      <div style={{ fontSize: 11, color: "var(--muted)" }}>
                        <strong style={{ color: "var(--ink)" }}>{mc3Edges.length}</strong> edges shown
                      </div>
                      <div style={{ fontSize: 11, color: "var(--muted)" }}>
                        <strong style={{ color: "var(--ink)" }}>{mc3.node_names.length}</strong> nodes
                      </div>
                      <div style={{ fontSize: 11, color: "var(--muted)" }}>
                        Acceptance:{" "}
                        <strong style={{ color: "var(--ink)" }}>
                          {(mc3.mc3_summary.acceptance_rate * 100).toFixed(1)}%
                        </strong>
                      </div>
                    </div>
                  ) : (
                    <div style={{ fontSize: 11, color: "var(--muted)", fontStyle: "italic" }}>
                      {mc3Loading ? "Loading..." : "No results yet"}
                    </div>
                  )}
                </div>
              </PanelCard>

              <PanelCard title="Edge strength">
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  {mc3Legend.map(([color, label]) => (
                    <div key={label} style={{ display: "flex", alignItems: "center", gap: 10 }}>
                      <span style={{ width: 24, height: 2.5, background: color, flexShrink: 0, borderRadius: 1 }} />
                      <span style={{ fontSize: 11, color: "var(--ink-2)" }}>{label}</span>
                    </div>
                  ))}
                </div>
              </PanelCard>

              <Btn primary onClick={handleMergeImport} disabled={!mc3}>Merge into Build</Btn>

              <div style={{ fontSize: 10.5, color: "var(--muted)", lineHeight: 1.6 }}>
                Merges MC3 discoveries into your Build canvas. Expert node types and
                edge attributes are preserved. New edges are added with their posterior
                confidence score.
              </div>
            </>
          )}
        </div>

        <div style={{
          flex: 1, minWidth: 0, borderRadius: 8, border: "1px solid var(--line)",
          overflow: "hidden", background: "var(--paper-2)", position: "relative",
        }}>
          {mode === "build" ? (
            nodes.length === 0 ? (
              <div style={{
                display: "flex", flexDirection: "column", alignItems: "center",
                justifyContent: "center", height: "100%", gap: 14, padding: 40,
              }}>
                <div style={{ fontSize: 36, opacity: 0.18, lineHeight: 1 }}>O</div>
                <div style={{ fontSize: 15, fontWeight: 700, color: "var(--ink-2)" }}>No nodes yet</div>
                <div style={{ fontSize: 12, color: "var(--muted)", maxWidth: 300, textAlign: "center", lineHeight: 1.7 }}>
                  Use <strong>Add Node</strong> to name your variables, then draw edges by
                  dragging from node handles or using <strong>Add Edge</strong>.
                </div>
              </div>
            ) : (
              <ReactFlow nodes={nodes} edges={edges}
                onNodesChange={onNodesChange} onEdgesChange={onEdgesChange}
                onConnect={onConnect} onNodeContextMenu={onNodeContextMenu}
                nodeTypes={nodeTypes} fitView fitViewOptions={{ padding: 0.3 }}
                proOptions={{ hideAttribution: true }} style={{ background: "var(--paper-2)" }}>
                <Background variant={BackgroundVariant.Dots} gap={18} size={1} color="#c9c2b3" />
                <Controls showInteractive={false} />
              </ReactFlow>
            )
          ) : mc3Loading ? (
            <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "var(--muted)" }}>
              <span className="mono" style={{ fontSize: 12 }}>Loading MC3 result...</span>
            </div>
          ) : mc3Error ? (
            <div style={{
              display: "flex", flexDirection: "column", alignItems: "center",
              justifyContent: "center", height: "100%", gap: 12, padding: 40,
            }}>
              <div style={{ fontSize: 15, fontWeight: 700, color: "var(--ink-2)" }}>MC3 not available</div>
              <div style={{ fontSize: 12, color: "var(--muted)", maxWidth: 320, textAlign: "center", lineHeight: 1.7 }}>
                {mc3Error}. Run the pipeline through the causal stage first.
              </div>
            </div>
          ) : (
            <ReactFlow nodes={mc3Nodes} edges={mc3Edges}
              onNodesChange={onMc3NodesChange} onEdgesChange={onMc3EdgesChange}
              nodeTypes={nodeTypes} fitView fitViewOptions={{ padding: 0.3 }}
              proOptions={{ hideAttribution: true }} style={{ background: "var(--paper-2)" }}
              nodesConnectable={false} nodesDraggable={true}>
              <Background variant={BackgroundVariant.Dots} gap={18} size={1} color="#c9c2b3" />
              <Controls showInteractive={false} />
            </ReactFlow>
          )}
        </div>
      </div>

      {contextMenu && (
        <div onClick={(e) => e.stopPropagation()} style={{
          position: "fixed", left: contextMenu.x, top: contextMenu.y, zIndex: 9999,
          background: "#fff", border: "1px solid var(--line)", borderRadius: 6,
          boxShadow: "0 4px 18px rgba(26,20,18,0.16)",
          paddingTop: 4, paddingBottom: 4, minWidth: 170,
        }}>
          <div className="mono" style={{
            padding: "4px 12px 8px", fontSize: 9,
            color: "var(--muted)", letterSpacing: "0.1em", textTransform: "uppercase",
            borderBottom: "1px solid var(--line)", marginBottom: 4,
          }}>
            Change type
          </div>
          {NODE_TYPE_OPTIONS.map((type) => (
            <button key={type}
              onClick={() => { mutations.changeNodeType(contextMenu.nodeId, type); setContextMenu(null); }}
              onMouseEnter={(e) => { (e.currentTarget as HTMLButtonElement).style.background = "var(--paper-2)"; }}
              onMouseLeave={(e) => { (e.currentTarget as HTMLButtonElement).style.background = "none"; }}
              style={{
                display: "flex", alignItems: "center", gap: 10, width: "100%",
                padding: "6px 12px", border: "none", background: "none", cursor: "pointer",
                fontSize: 12, fontFamily: "inherit", textAlign: "left", color: "var(--ink-2)",
              }}>
              <span style={{ width: 4, height: 14, borderRadius: 2, background: NODE_COLORS[type], flexShrink: 0 }} />
              <span style={{ textTransform: "capitalize" }}>{type.replace(/_/g, " ")}</span>
            </button>
          ))}
          <div style={{ borderTop: "1px solid var(--line)", marginTop: 4, paddingTop: 4 }}>
            <button
              onClick={() => { mutations.deleteNode(contextMenu.nodeId); setContextMenu(null); }}
              onMouseEnter={(e) => { (e.currentTarget as HTMLButtonElement).style.background = "#fff4f4"; }}
              onMouseLeave={(e) => { (e.currentTarget as HTMLButtonElement).style.background = "none"; }}
              style={{
                display: "flex", alignItems: "center", gap: 10, width: "100%",
                padding: "6px 12px", border: "none", background: "none", cursor: "pointer",
                fontSize: 12, fontFamily: "inherit", textAlign: "left", color: "var(--crimson)",
              }}>
              Delete node
            </button>
          </div>
        </div>
      )}
    </div>
  );
}