// PlanGraph (SPEC §3.2 Launch, §5.4): the run plan as a vertical stage graph. Each node shows
// its state (will run / skipped / cached) with an icon, text and line style (never colour
// alone), the reason ("required by S6", "disabled in the config (cv.baselines)", "from the
// checkpoint"), its estimate range and planned work units. The same nodes core will emit as
// `run.plan`, so the graph is what the run's stage rail will show.
import type { PlanNode } from "../../../api/types";
import { Icon, type IconName } from "../../../components/ui/Icon";
import { fmtDurationRange } from "../../../theme/format";
import { planCounts, planNodeViews, stageShort } from "../model/plan";

const STATE: Record<PlanNode["state"], { icon: IconName; text: string }> = {
  will_run: { icon: "play", text: "will run" },
  skipped: { icon: "skip", text: "skipped" },
  cached: { icon: "cached", text: "cached" },
};

export function PlanGraph({
  nodes,
  total,
  title = "Plan",
}: {
  nodes: readonly PlanNode[];
  total?: { lo: number | null; hi: number | null } | null;
  title?: string;
}) {
  const views = planNodeViews(nodes);
  const counts = planCounts(nodes);
  return (
    <figure className="plan-graph-wrap" aria-label={title}>
      <ol className="plan-graph">
        {views.map((n) => {
          const st = STATE[n.state];
          return (
            <li key={n.id} className="plan-node" data-node={n.id} data-state={n.state}>
              <span className="plan-mark" aria-hidden="true">
                <Icon name={st.icon} size={12} />
              </span>
              <div className="plan-body">
                <div className="plan-title">
                  <span className="mono plan-id">{stageShort(n.id)}</span> <span className="plan-label">{n.label}</span>
                </div>
                <div className="plan-meta">
                  <span className="plan-state">{st.text}</span>
                  {n.reason ? <span className="plan-reason"> · {n.reason}</span> : null}
                  {n.units ? <span className="plan-units"> · {n.units}</span> : null}
                </div>
              </div>
              <span className="plan-est num">{n.estimate ?? ""}</span>
            </li>
          );
        })}
      </ol>
      <figcaption className="cap">
        {counts.will_run} will run · {counts.skipped} skipped · {counts.cached} cached
        {total && total.lo !== null && total.hi !== null ? ` · total ≈${fmtDurationRange(total.lo, total.hi)}` : ""}
      </figcaption>
    </figure>
  );
}
