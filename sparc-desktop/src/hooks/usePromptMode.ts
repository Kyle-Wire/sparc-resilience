/**
 * usePromptMode — derives the AI prompt mode from the active page and
 * any pending explain seed.
 */
import type { ExplainSeed } from "@/hooks/ExplainContext.tsx";

export type PromptMode = "general" | "domain" | "dag" | "physics" | "results" | "narrator" | "hypothesis";

export function usePromptMode(
  page: string,
  explainSeed: ExplainSeed | null,
): PromptMode {
  if (explainSeed?.mode === "narrator") return "narrator";
  if (explainSeed?.mode === "hypothesis") return "hypothesis";
  switch (page) {
    case "Project":   return "domain";
    case "Configure": return "dag";
    case "Results":
    case "Report":    return "results";
    default:          return "general";
  }
}
