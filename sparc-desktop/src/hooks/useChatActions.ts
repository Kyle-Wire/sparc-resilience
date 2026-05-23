/**
 * useChatActions — dispatches Claude action payloads to the sidecar:
 *   suggest_template, propose_dag_edges, suggest_physics, suggest_predictors.
 *
 * Receives collaborating objects as parameters to stay hook-composition-safe
 * (no duplicate store subscriptions).
 */
import { useCallback } from "react";
import { initProject, saveConfig } from "@/lib/api";
import type { ClaudeAction } from "@/lib/types";

interface NotifHandle {
  notify: (type: "error" | "info" | "success", message: string) => void;
}

interface ProjectHandle {
  openProject: (path: string, opts: { template?: string }) => Promise<void>;
}

export function useChatActions(
  navigate: (page: string) => void,
  notif: NotifHandle,
  project: ProjectHandle,
  refresh: () => void,
) {
  const handleAction = useCallback(
    async (action: ClaudeAction) => {
      try {
        switch (action.action) {
          case "suggest_template": {
            const home = prompt("Choose output directory:", ${action.template}_project);
            if (!home) break;
            const res = await initProject(action.template, home);
            await project.openProject(res.project_yml, { template: action.template });
            navigate("Data");
            refresh();
            break;
          }
          case "propose_dag_edges": {
            localStorage.setItem("sparc-proposed-edges", JSON.stringify(action.edges));
            navigate("DAG");
            refresh();
            break;
          }
          case "suggest_physics": {
            await saveConfig({
              physics: { monotone_constraints: action.monotonic_constraints },
            });
            navigate("Physics");
            refresh();
            break;
          }
          case "suggest_predictors": {
            await saveConfig({ predictors: action.predictors });
            navigate("Variables");
            refresh();
            break;
          }
        }
      } catch (e) {
        notif.notify("error", e instanceof Error ? e.message : "Action dispatch failed");
      }
    },
    // notif and project are stable references from their respective stores/hooks
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [navigate, notif, project, refresh],
  );

  return { handleAction };
}
