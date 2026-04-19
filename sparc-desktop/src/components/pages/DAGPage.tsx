import { useState, useEffect, useRef, useCallback } from "react";
import { SectionHeader, Card, Btn } from "@/components/ui/DesignSystem";
import { getDag, getMc3Result, approveDag, rejectDag } from "@/lib/api";
import { usePipeline } from "@/hooks/PipelineProvider";
import { useNotification } from "@/hooks/useNotifications";
import type { DagDefinition, MC3Result } from "@/lib/types";
import { SPARC_RAMP_HEX } from "@/lib/design-tokens";

interface DagPageProps {
  onNavigate?: (page: string) => void;
}

// Assign layer index: outcome = 0, then by distance from outcome
function layerNodes(
  nodeNames: string[],
  edges: { source: string; target: string }[],
  target?: string,
): Map<string, number> {
  const layers = new Map<string, number>();
  const outgoing = new Map<string, string[]>();
  for (const e of edges) {
    if (!outgoing.has(e.source)) outgoing.set(e.source, []);
    outgoing.get(e.source)!.push(e.target);
  }
  // BFS from leaves (nodes with no outgoing or target)
  const targetNode = target ?? nodeNames[nodeNames.length - 1];
  layers.set(targetNode, 0);
  const queue = [targetNode];
  while (queue.length) {
    const cur = queue.shift()!;
    const curLayer = layers.get(cur)!;
    for (const e of edges) {
      if (e.target === cur && !layers.has(e.source)) {
        layers.set(e.source, curLayer + 1);
        queue.push(e.source);
      }
    }
  }
  // Any unreached nodes get highest layer + 1
  const maxLayer = Math.max(0, ...layers.values());
  for (const n of nodeNames) {
    if (!layers.has(n)) layers.set(n, maxLayer + 1);
  }
  return layers;
}

const NODE_W = 140;
const NODE_H = 48;
const H_GAP = 30;
const V_GAP = 70;

export default function DAGPage({ onNavigate: _onNavigate }: DagPageProps) {
  const [dag, setDag] = useState<DagDefinition | null>(null);
  const [mc3, setMc3] = useState<MC3Result | null>(null);
  const [loading, setLoading] = useState(true);
  const { dagApprovalPending, handleApproveDag, handleRejectDag } = usePipeline();
  const { notify } = useNotification();
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      getDag().catch(() => null),
      getMc3Result().catch(() => null),
    ]).then(([d, m]) => {
      setDag(d);
      setMc3(m);
      setLoading(false);
    });
  }, []);

  const handleApprove = useCallback(async () => {
    try {
      await approveDag();
      handleApproveDag?.();
      notify("success", "DAG approved");
    } catch (e) {
      notify("error", e instanceof Error ? e.message : "Approval failed");
    }
  }, [handleApproveDag, notify]);

  const handleReject = useCallback(async () => {
    try {
      await rejectDag();
      handleRejectDag?.();
      notify("info", "DAG rejected — MC3 will re-run");
    } catch (e) {
      notify("error", e instanceof Error ? e.message : "Rejection failed");
    }
  }, [handleRejectDag, notify]);

  // Derive graph from MC3 median_dag or fall back to raw dag
  const graph = (() => {
    if (mc3?.median_dag) {
      return {
        nodes: mc3.median_dag.nodes,
        edges: mc3.median_dag.edges,
      };
    }
    if (dag) {
      return {
        nodes: dag.nodes.map((n) => n.name ?? (n as any).id ?? ""),
        edges: dag.edges.map((e) => ({
          source: e.parent,
          target: e.child,
          probability: 1,
        })),
      };
    }
    return null;
  })();

  // Layout positions
  const positions = (() => {
    if (!graph) return new Map<string, { x: number; y: number }>();
    const layers = layerNodes(graph.nodes, graph.edges);
    const maxLayer = Math.max(0, ...layers.values());

    // Group nodes by layer
    const byLayer = new Map<number, string[]>();
    for (const [node, layer] of layers) {
      if (!byLayer.has(layer)) byLayer.set(layer, []);
      byLayer.get(layer)!.push(node);
    }

    const pos = new Map<string, { x: number; y: number }>();
    for (let layer = maxLayer; layer >= 0; layer--) {
      const nodesInLayer = byLayer.get(layer) ?? [];
      const totalW = nodesInLayer.length * NODE_W + (nodesInLayer.length - 1) * H_GAP;
      const startX = -totalW / 2 + NODE_W / 2;
      nodesInLayer.forEach((node, i) => {
        pos.set(node, {
          x: startX + i * (NODE_W + H_GAP),
          y: (maxLayer - layer) * (NODE_H + V_GAP),
        });
      });
    }
    return pos;
  })();

  // Compute SVG viewBox
  const allPos = [...positions.values()];
  const minX = Math.min(0, ...allPos.map((p) => p.x)) - NODE_W / 2 - 20;
  const minY = Math.min(0, ...allPos.map((p) => p.y)) - 20;
  const maxX = Math.max(0, ...allPos.map((p) => p.x)) + NODE_W / 2 + 20;
  const maxY = Math.max(0, ...allPos.map((p) => p.y)) + NODE_H + 20;
  const viewBox = `${minX} ${minY} ${maxX - minX} ${maxY - minY}`;

  if (loading) {
    return (
      <div>
        <SectionHeader kicker="04 · analysis" label="DAG" />
        <Card title="Loading causal graph…" subtitle="fetching from backend">
          <div style={{ padding: 40, textAlign: "center", color: "var(--muted)", fontSize: 13 }}>
            Connecting to server…
          </div>
        </Card>
      </div>
    );
  }

  if (!graph || graph.nodes.length === 0) {
    return (
      <div>
        <SectionHeader kicker="04 · analysis" label="DAG" />
        <Card title="Causal graph" subtitle="no graph available">
          <div style={{ padding: 40, textAlign: "center", color: "var(--muted)", fontSize: 13 }}>
            Run the pipeline to generate the causal graph via MC3.
          </div>
        </Card>
      </div>
    );
  }

  return (
    <div>
      <SectionHeader
        kicker="04 · analysis"
        label="DAG"
        right={
          dagApprovalPending ? (
            <div style={{ display: "flex", gap: 8 }}>
              <Btn small onClick={handleReject}>Reject</Btn>
              <Btn primary small onClick={handleApprove}>Approve DAG</Btn>
            </div>
          ) : undefined
        }
      />

      <div style={{ display: "grid", gridTemplateColumns: "1fr 320px", gap: 14 }}>
        {/* Graph visualization */}
        <Card
          title="Causal graph"
          subtitle={mc3 ? `MC3 median DAG · ${graph.edges.length} edges · acceptance ${((mc3.mc3_summary?.acceptance_rate ?? 0) * 100).toFixed(1)}%` : `${graph.edges.length} edges`}
        >
          <svg
            ref={svgRef}
            viewBox={viewBox}
            style={{ width: "100%", height: 380, display: "block" }}
          >
            <defs>
              <marker id="arrow" viewBox="0 0 10 8" refX="9" refY="4" markerWidth="8" markerHeight="6" orient="auto-start-reverse">
                <path d="M0,0 L10,4 L0,8 z" fill="var(--ink)" opacity="0.5" />
              </marker>
            </defs>

            {/* Edges */}
            {graph.edges.map((e, i) => {
              const from = positions.get(e.source);
              const to = positions.get(e.target);
              if (!from || !to) return null;
              const prob = (e as any).probability ?? 1;
              const colorIdx = Math.min(SPARC_RAMP_HEX.length - 1, Math.floor(prob * (SPARC_RAMP_HEX.length - 1)));
              return (
                <line
                  key={i}
                  x1={from.x}
                  y1={from.y + NODE_H}
                  x2={to.x}
                  y2={to.y}
                  stroke={SPARC_RAMP_HEX[colorIdx]}
                  strokeWidth={1.5 + prob * 1.5}
                  strokeOpacity={0.3 + prob * 0.7}
                  markerEnd="url(#arrow)"
                />
              );
            })}

            {/* Nodes */}
            {graph.nodes.map((node) => {
              const pos = positions.get(node);
              if (!pos) return null;
              return (
                <g key={node}>
                  <rect
                    x={pos.x - NODE_W / 2}
                    y={pos.y}
                    width={NODE_W}
                    height={NODE_H}
                    rx={6}
                    fill="#fff"
                    stroke="var(--line)"
                    strokeWidth={1.5}
                  />
                  <text
                    x={pos.x}
                    y={pos.y + NODE_H / 2}
                    textAnchor="middle"
                    dominantBaseline="central"
                    fontSize={10}
                    fontWeight={600}
                    fontFamily="JetBrains Mono, monospace"
                    fill="var(--ink)"
                  >
                    {node.length > 16 ? node.slice(0, 14) + "…" : node}
                  </text>
                </g>
              );
            })}
          </svg>
        </Card>

        {/* Edge table */}
        <Card title="Edge table" subtitle={`${graph.edges.length} directed edges`}>
          <div style={{ maxHeight: 360, overflowY: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
              <thead>
                <tr style={{ borderBottom: "1px solid var(--line)" }}>
                  <th style={{ textAlign: "left", padding: "4px 6px", fontWeight: 700, fontSize: 10, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.08em" }}>From</th>
                  <th style={{ textAlign: "left", padding: "4px 6px", fontWeight: 700, fontSize: 10, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.08em" }}>To</th>
                  <th style={{ textAlign: "right", padding: "4px 6px", fontWeight: 700, fontSize: 10, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.08em" }}>Prob</th>
                </tr>
              </thead>
              <tbody>
                {graph.edges.map((e, i) => (
                  <tr key={i} style={{ borderBottom: "1px dashed var(--line)" }}>
                    <td className="mono" style={{ padding: "5px 6px", fontSize: 10.5 }}>{e.source}</td>
                    <td className="mono" style={{ padding: "5px 6px", fontSize: 10.5 }}>{e.target}</td>
                    <td className="mono" style={{ padding: "5px 6px", fontSize: 10.5, textAlign: "right" }}>
                      {((e as any).probability != null) ? ((e as any).probability as number).toFixed(2) : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>

      {/* MC3 summary */}
      {mc3?.mc3_summary && (
        <div style={{ marginTop: 14 }}>
          <Card title="MC3 summary" subtitle="Markov chain Monte Carlo model composition">
            <div style={{ display: "flex", gap: 24, fontSize: 12 }}>
              <div>
                <span style={{ color: "var(--muted)", fontSize: 10 }}>Accepted</span>
                <div className="mono" style={{ fontWeight: 700 }}>{mc3.mc3_summary.n_accepted.toLocaleString()}</div>
              </div>
              <div>
                <span style={{ color: "var(--muted)", fontSize: 10 }}>Total iterations</span>
                <div className="mono" style={{ fontWeight: 700 }}>{mc3.mc3_summary.n_total.toLocaleString()}</div>
              </div>
              <div>
                <span style={{ color: "var(--muted)", fontSize: 10 }}>Acceptance rate</span>
                <div className="mono" style={{ fontWeight: 700 }}>{(mc3.mc3_summary.acceptance_rate * 100).toFixed(1)}%</div>
              </div>
              <div>
                <span style={{ color: "var(--muted)", fontSize: 10 }}>Best score</span>
                <div className="mono" style={{ fontWeight: 700 }}>{mc3.mc3_summary.best_score.toFixed(2)}</div>
              </div>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}
