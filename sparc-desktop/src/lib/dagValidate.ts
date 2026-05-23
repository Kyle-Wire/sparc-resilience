/**
 * Pure client-side structural validation for DAG graphs.
 *
 * No React dependency — safe to call from tests directly.
 */
import { type Node, type Edge } from "@xyflow/react";

/**
 * Check the current build-mode graph for structural problems.
 * Returns a list of human-readable warning strings (empty = clean).
 *
 * Rules checked:
 *  - Disconnected nodes (no incident edges)
 *  - Outcome nodes with outgoing edges
 */
export function validateDagStructure(nodes: Node[], edges: Edge[]): string[] {
  if (nodes.length === 0) return [];

  const warns: string[] = [];
  const userEdges = edges.filter((e) => !e.id.startsWith("mc3-"));

  const connected = new Set<string>();
  for (const e of userEdges) {
    connected.add(e.source);
    connected.add(e.target);
  }

  const disconnected = nodes.filter((n) => !connected.has(n.id));
  if (disconnected.length > 0) {
    warns.push(`Disconnected: ${disconnected.map((n) => n.id).join(", ")}`);
  }

  const outcomes = nodes.filter(
    (n) => (n.data as { nodeType?: string }).nodeType === "outcome",
  );
  for (const o of outcomes) {
    if (userEdges.some((e) => e.source === o.id)) {
      warns.push(`Outcome "${o.id}" has outgoing edges`);
    }
  }

  return warns;
}
