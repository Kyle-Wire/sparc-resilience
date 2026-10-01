import { useId, type ReactNode } from "react";
import type { Issue } from "../../api/types";

export type FieldProps = {
  label: ReactNode;
  hint?: ReactNode;
  /** Server issues for this field's dotted path (rendered under the control). */
  issues?: Issue[];
  children: (id: string, describedBy: string | undefined, invalid: boolean) => ReactNode;
  className?: string;
};

/** Label + control + hint + issue messages, wired with ids for assistive tech. */
export function Field({ label, hint, issues, children, className }: FieldProps) {
  const id = useId();
  const hintId = hint ? `${id}-hint` : undefined;
  const issueId = issues?.length ? `${id}-issues` : undefined;
  const describedBy = [hintId, issueId].filter(Boolean).join(" ") || undefined;
  const invalid = !!issues?.some((i) => i.level === "error");
  return (
    <div className={["field", className ?? ""].filter(Boolean).join(" ")}>
      <label htmlFor={id}>{label}</label>
      {children(id, describedBy, invalid)}
      {hint ? (
        <span className="hint" id={hintId}>
          {hint}
        </span>
      ) : null}
      {issues?.length ? (
        <div id={issueId}>
          {issues.map((i, k) => (
            <div key={k} className="issue" data-level={i.level}>
              {i.level === "error" ? "Error: " : i.level === "warn" ? "Warning: " : ""}
              {i.message}
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}

/** Issues whose path equals `path` or lies under it (`levers.Pct_Canopy` ⊂ `levers`). */
export function issuesAt(issues: Issue[] | null | undefined, path: string): Issue[] {
  if (!issues) return [];
  return issues.filter((i) => i.path === path || i.path.startsWith(path + ".") || i.path.startsWith(path + "["));
}
