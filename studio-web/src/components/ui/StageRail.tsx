// StageRail: the brand's numbered stage progression (website product StageStack). Mono
// "01", "02"… squares tinted along the purple→gold ramp (sampled at i/(n−1), so n stages run
// purple → gold once, never wrapping), joined by a thin
// rail that fills with the ramp up to the last done stage. An ordered list; the current stage
// carries aria-current="step". Status is never colour alone: the default body adds the status
// word for screen readers, and rows with their own `content` carry their own chips or controls.
import type { CSSProperties, ReactNode } from "react";
import { Link } from "../../router";
import { STAGE_TONE_STOPS, stageTone, stageToneIndex } from "../../theme/palette";
import "../../theme/stage-rail.css";

export type StageStatus = "done" | "running" | "todo" | "failed" | "skipped";

export type StageRailStage = {
  id: string;
  label: ReactNode;
  status?: StageStatus;
  meta?: ReactNode;
  href?: string;
  /** Trailing slot of the default body (an action button, a chip). */
  aside?: ReactNode;
  /** Replaces the default label · meta · aside body (rows that bring their own controls). */
  content?: ReactNode;
  /** Extra data-* attributes on the list item: `{ stage: "S4" }` → `data-stage="S4"`. */
  data?: Record<string, string | undefined>;
};

/** Number of brand-ramp stops (purple … gold) the squares are sampled from. */
export const STAGE_TONES = STAGE_TONE_STOPS;

/** Tone hooks for stage i of n: the nearest ramp stop (data-tone) and the exact colours (--stage-c, --stage-next). */
export function stageToneProps(index: number, count: number): { "data-tone": number; style: CSSProperties } {
  return {
    "data-tone": stageToneIndex(index, count),
    style: { "--stage-c": stageTone(index, count), "--stage-next": stageTone(index + 1, count) } as CSSProperties,
  };
}

const STATUS_TEXT: Record<StageStatus, string> = { done: "done", running: "running", todo: "to do", failed: "failed", skipped: "skipped" };

/** "01", "02", … for a zero-based index. */
export function stageNumber(index: number): string {
  return String(index + 1).padStart(2, "0");
}

/** Index of the last done stage, -1 when none is done. */
export function lastDoneIndex(stages: readonly { status?: StageStatus }[]): number {
  for (let i = stages.length - 1; i >= 0; i--) if (stages[i].status === "done") return i;
  return -1;
}

/**
 * The numbered square on its own, for progressions that cannot be a list (the Setup tabs).
 * `filled` marks the rail segment after it as filled; `count` is the number of stages in the
 * progression (default: the seven ramp stops), so the last stage is gold.
 */
export function StageNum({ index, status, filled, count = STAGE_TONES }: { index: number; status?: StageStatus; filled?: boolean; count?: number }) {
  return (
    <span className="stage-num" {...stageToneProps(index, count)} data-status={status ?? "todo"} data-filled={filled || undefined} aria-hidden="true">
      {stageNumber(index)}
    </span>
  );
}

export function StageRail({
  stages,
  orientation = "vertical",
  current,
  label,
  size = "md",
  className,
}: {
  stages: readonly StageRailStage[];
  orientation?: "vertical" | "horizontal";
  /** Id of the current stage (aria-current="step"). */
  current?: string | null;
  /** Accessible name of the list. */
  label?: string;
  size?: "md" | "sm";
  className?: string;
}) {
  const last = lastDoneIndex(stages);
  return (
    <ol className={className ? `stage-rail ${className}` : "stage-rail"} data-orientation={orientation} data-size={size} aria-label={label}>
      {stages.map((s, i) => {
        const data = Object.fromEntries(Object.entries(s.data ?? {}).map(([k, v]) => [`data-${k}`, v]));
        return (
          <li
            key={s.id}
            className="stage-rail-item"
            {...stageToneProps(i, stages.length)}
            data-status={s.status ?? "todo"}
            data-filled={i < last || undefined}
            aria-current={current != null && s.id === current ? "step" : undefined}
            {...data}
          >
            <StageNum index={i} status={s.status} count={stages.length} />
            {s.content !== undefined ? (
              <div className="stage-rail-content">{s.content}</div>
            ) : (
              <div className="stage-rail-body">
                <div className="stage-rail-text">
                  <span className="stage-rail-label">
                    {s.href ? <Link to={s.href}>{s.label}</Link> : s.label}
                    {s.status ? <span className="sr-only"> ({STATUS_TEXT[s.status]})</span> : null}
                  </span>
                  {s.meta ? <span className="stage-rail-meta">{s.meta}</span> : null}
                </div>
                {s.aside ? <div className="stage-rail-aside">{s.aside}</div> : null}
              </div>
            )}
          </li>
        );
      })}
    </ol>
  );
}
