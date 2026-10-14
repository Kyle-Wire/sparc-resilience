// The stage rail (SPEC §5.10): eleven chips S0 … finish, each with its plain name ("Train the
// models"), a state icon and text (never colour alone), duration or estimate, and checkpoint
// markers; the tooltip says what the stage does, the skip/cached reason and the technical name
// ("S2–S3 · Base models and stacker"). Clicking a chip opens its live panel and zooms the Gantt. The chips sit on the
// shared numbered StageRail (01 … 11 on the brand ramp, filled up to the last done stage).
import type { ReactNode } from "react";
import type { StageId } from "../../../api/types";
import { TermTipBody } from "../../../components/ui/InfoTip";
import { StageRail as Rail, type StageStatus } from "../../../components/ui/StageRail";
import { StatusChip } from "../../../components/ui/StatusChip";
import { useTooltip } from "../../../components/ui/Tooltip";
import { term } from "../../../content/glossary";
import { fmtBytes, fmtDateTime, fmtDuration, fmtPct } from "../../../theme/format";
import { STAGE_PLAIN, STAGE_TERM, type RailItem } from "../model";

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

/** The tooltip of a rail chip: what the stage does in plain words, then its state's reason and checkpoint; the technical name as the footnote. */
function chipTip(it: RailItem, ckText: string): { tip: string; technical: string } {
  const e = term(STAGE_TERM[it.id]);
  const why = it.reasonText ? ` ${it.state.replace(/_/g, " ")}: ${it.reasonText}.` : "";
  const ck = ckText ? ` ${ckText}.` : "";
  return { tip: `${e.tip}${why}${ck}`, technical: `${it.short} · ${it.label}` };
}

function RailChip({ it, selected, onSelect, body, tip }: { it: RailItem; selected?: StageId | null; onSelect?: (id: StageId) => void; body: ReactNode; tip: { tip: string; technical: string } }) {
  const content = <TermTipBody tip={tip.tip} technical={tip.technical} />;
  const btn = useTooltip<HTMLButtonElement>({ content, bubbleClassName: "tip-term" });
  const span = useTooltip<HTMLSpanElement>({ content, bubbleClassName: "tip-term" });
  if (onSelect)
    return (
      <>
        <button
          ref={btn.anchorRef}
          type="button"
          className="rail-chip"
          aria-pressed={selected === it.id}
          data-stage={it.id}
          {...btn.triggerProps}
          onClick={(e) => {
            btn.triggerProps.onClick(e);
            onSelect(it.id);
          }}
        >
          {body}
        </button>
        {btn.bubble}
      </>
    );
  return (
    <>
      <span ref={span.anchorRef} className="rail-chip" tabIndex={0} data-stage={it.id} {...span.triggerProps}>
        {body}
      </span>
      {span.bubble}
    </>
  );
}

export function StageRail({
  items,
  selected,
  onSelect,
  nowS,
  started,
  mini,
  label = "Run stages",
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
        const ckText = ck ? `Saved models updated${ck.ts ? ` ${fmtDateTime(ck.ts)}` : ""} (checkpoint: done ${ck.done.join(", ") || "—"}${ck.bytes ? `, ${fmtBytes(ck.bytes)}` : ""})` : "";
        const tip = chipTip(it, ckText);
        const time = mini ? "" : timeText(it, nowS ?? Date.now() / 1000, started?.[it.id] ?? null);
        const body = (
          <>
            <span className="rail-name">
              {mini ? it.short : STAGE_PLAIN[it.id]}
              {ck ? (
                <span className="rail-ck" aria-label={ckText}>
                  ◆
                </span>
              ) : null}
            </span>
            <StatusChip status={it.state} meta={it.state === "running" && it.progress !== null ? fmtPct(it.progress) : undefined} />
            {time ? <span className="rail-time">{time}</span> : null}
            {it.reasonText ? <span className="sr-only">{it.reasonText}</span> : null}
          </>
        );
        return {
          id: it.id,
          label: STAGE_PLAIN[it.id],
          status: railStatus(it.state),
          content: <RailChip it={it} selected={selected} onSelect={onSelect} body={body} tip={tip} />,
        };
      })}
    />
  );
}
