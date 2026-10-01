// Issue summary at the top of a setup step: every validation issue of the step's sections
// with its dotted path, and a one-click fix when the server offers one (`Issue.fix`).
import type { Issue } from "../../../api/types";
import { Button } from "../../../components/ui/Button";
import { Icon } from "../../../components/ui/Icon";
import { useCfg } from "./cfg";

const LEVEL_ICON = { error: "x", warn: "alert", info: "info" } as const;

export function StepIssues({ issues, title = "Issues in this step" }: { issues: readonly Issue[]; title?: string }) {
  const c = useCfg();
  if (!issues.length) return null;
  const errors = issues.filter((i) => i.level === "error").length;
  const warns = issues.filter((i) => i.level === "warn").length;
  return (
    <details className="step-issues" open={errors > 0}>
      <summary>
        {title}: {errors} {errors === 1 ? "error" : "errors"}, {warns} {warns === 1 ? "warning" : "warnings"}
        {issues.length - errors - warns ? `, ${issues.length - errors - warns} notes` : ""}
      </summary>
      <ul>
        {issues.map((i, k) => (
          <li key={k} data-level={i.level}>
            <Icon name={LEVEL_ICON[i.level]} size={13} />
            <span className="mono">{i.path || "config"}</span>
            <span>{i.message}</span>
            {i.fix ? (
              <Button size="small" onClick={() => c.set(i.fix!.path, i.fix!.value)}>
                Fix: set {i.fix.path} to {JSON.stringify(i.fix.value)}
              </Button>
            ) : null}
          </li>
        ))}
      </ul>
    </details>
  );
}
