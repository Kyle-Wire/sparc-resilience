/**
 * Additive merge of MC³ posterior results into an expert DAG.
 *
 * The expert DAG is never destructively overwritten:
 *  - Expert node types are preserved for all shared nodes.
 *  - Expert edge attributes (mechanism, lag, sign_prior, etc.) are kept.
 *  - MC³-only nodes are added as "confounder".
 *  - MC³ edges above `threshold` that have no expert counterpart are
 *    added with `confidence` set to the posterior edge probability.
 */
import type { DagDefinition, DagNodeType, MC3Result } from "./types";

export function mergeMc3IntoDag(
  expert: DagDefinition,
  mc3: MC3Result,
  threshold: number,
): DagDefinition {
  const expertNodeTypes = new Map<string, DagNodeType>(
    expert.nodes.map((n) => [n.name, n.type]),
  );
  const expertEdgeKeys = new Set<string>(
    expert.edges.map((e) => `${e.parent}\u2192${e.child}`),
  );

  // Merged nodes: expert first, then any new MC³ nodes
  const allNodeNames = new Set<string>(expert.nodes.map((n) => n.name));
  for (const name of mc3.node_names) allNodeNames.add(name);

  const mergedNodes = Array.from(allNodeNames).map((name) => ({
    name,
    type: (expertNodeTypes.get(name) ?? "confounder") as DagNodeType,
  }));

  // Merged edges: expert edges preserved in full, net-new MC³ edges appended
  const mergedEdges = [...expert.edges];
  const { node_names, edge_probs } = mc3;

  for (let i = 0; i < node_names.length; i++) {
    for (let j = 0; j < node_names.length; j++) {
      if (i === j) continue;
      const prob = edge_probs[i][j];
      if (prob < threshold) continue;
      const key = `${node_names[i]}\u2192${node_names[j]}`;
      if (expertEdgeKeys.has(key)) continue; // expert edge takes precedence
      mergedEdges.push({
        parent: node_names[i],
        child: node_names[j],
        confidence: parseFloat(prob.toFixed(3)),
      });
    }
  }

  return {
    nodes: mergedNodes,
    edges: mergedEdges,
    assumptions: expert.assumptions,
  };
}
