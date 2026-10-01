// The stage rail (SPEC §5.10): eleven chips S0 … finish, each with a state icon and text (never
// colour alone), duration or estimate, the skip/cached reason as a tooltip, and checkpoint
// markers. Clicking a chip opens its live panel and zooms the Gantt.
import type { StageId } from "../../../api/types";
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
  return (
    <ol className={mini ? "rail mini" : "rail"} aria-label={label}>
      {items.map((it) => {
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
        return (
          <li key={it.id}>
            {onSelect ? (
              <button type="button" className="rail-chip" aria-pressed={selected === it.id} title={tip} onClick={() => onSelect(it.id)} data-stage={it.id}>
                {body}
              </button>
            ) : (
              <span className="rail-chip" title={tip} data-stage={it.id}>
                {body}
              </span>
            )}
          </li>
        );
      })}
    </ol>
  );
}
