// The stage rail (SPEC §5.10): eleven chips S0 … finish, each with a state icon and text (never
// colour alone), duration or estimate, the skip/cached reason as a tooltip, and checkpoint
// markers. Clicking a chip opens its live panel and zooms the Gantt. The chips sit on the
// shared numbered StageRail (01 … 11 on the brand ramp, filled up to the last done stage).
import type { StageId } from "../../../api/types";
import { StageRail as Rail, type StageStatus } from "../../../components/ui/StageRail";
import { StatusChip } from "../../../components/ui/StatusChip";
import { fmtBytes, fmtDateTime, fmtDuration, fmtPct } from "../../../theme/format";
import type { RailItem } from "../model";

function timeText(it: RailItem, nowS: number, startedTs: number | null): string {
  if (it.state === "done" || it.state === "failed" || it.state === "cancelled") return it.elapsed_s !== null ? fmtDuration(it.elapsed_s) : "";
  if (it.state === "running") {
    const so = startedTs !== null ? fmtDuration(Math.max(0, nowS - startedTs)) : "";
    return it.est_s !== null && it.est_s > 0 ? `${so} · ≈${fmtDuration(it.est_s)} left` : so;
  }
  if (it.state === "planned") return it.est_s !== null && it.est_s > 0 ? `≈${fmtDuration(it.est_s)}` : "";
  return "";
}

/** Tracker stage state → the rail square's status. */
export function railStatus(state: string): StageStatus {
  if (state === "done" || state === "cached") return "done";
  if (state === "running") return "running";
  if (state === "failed" || state === "cancelled" || state === "interrupted") return "failed";
  if (state === "skipped" || state === "disabled" || state === "not_requested") return "skipped";
  return "todo";
}

export function StageRail({
  items,
  selected,
  onSelect,
  nowS,
  started,
  mini,
  label = "Pipeline stages",
}: {
  items: RailItem[];
  selected?: StageId | null;
  onSelect?: (id: StageId) => void;
  nowS?: number;
  /** started_ts per stage (running chips show time so far). */
  started?: Record<string, number | null>;
  mini?: boolean;
  label?: string;
}) {
  const running = items.find((it) => it.state === "running");
  return (
    <Rail
      className={mini ? "rail mini" : "rail"}
      orientation="horizontal"
      size={mini ? "sm" : "md"}
      label={label}
      current={running?.id ?? null}
      stages={items.map((it) => {
        const ck = it.checkpoints[0];
        const ckText = ck ? `Checkpoint saved${ck.ts ? ` ${fmtDateTime(ck.ts)}` : ""}: done ${ck.done.join(", ") || "—"}${ck.bytes ? `, ${fmtBytes(ck.bytes)}` : ""}` : "";
        const tip = [it.label, it.reasonText ? `${it.state.replace(/_/g, " ")}: ${it.reasonText}` : "", ckText].filter(Boolean).join(" · ");
        const time = mini ? "" : timeText(it, nowS ?? Date.now() / 1000, started?.[it.id] ?? null);
        const body = (
          <>
            <span className="rail-name">
              {it.short}
              {ck ? (
                <span className="rail-ck" title={ckText} aria-label={ckText}>
                  ◆
                </span>
              ) : null}
            </span>
            <StatusChip status={it.state} meta={it.state === "running" && it.progress !== null ? fmtPct(it.progress) : undefined} title={tip} />
            {time ? <span className="rail-time">{time}</span> : null}
            {it.reasonText ? <span className="sr-only">{it.reasonText}</span> : null}
          </>
        );
        return {
          id: it.id,
          label: it.label,
          status: railStatus(it.state),
          content: onSelect ? (
            <button type="button" className="rail-chip" aria-pressed={selected === it.id} title={tip} onClick={() => onSelect(it.id)} data-stage={it.id}>
              {body}
            </button>
          ) : (
            <span className="rail-chip" title={tip} data-stage={it.id}>
              {body}
            </span>
          ),
        };
      })}
    />
  );
}
