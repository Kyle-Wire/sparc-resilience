import type { TrainingHealthWarning } from "@/hooks/PipelineProvider";

const WARNING_STYLES: Record<string, { icon: string; color: string }> = {
  loss_explosion: { icon: "💥", color: "text-red-400" },
  physics_stagnation: { icon: "🔬", color: "text-amber-400" },
  gradient_spike: { icon: "⚡", color: "text-orange-400" },
};

interface Props {
  warnings: TrainingHealthWarning[];
  /** Show at most this many recent warnings. Default 3. */
  maxVisible?: number;
}

export function TrainingHealthBadge({ warnings, maxVisible = 3 }: Props) {
  if (warnings.length === 0) return null;

  const recent = warnings.slice(-maxVisible);
  const hidden = warnings.length - recent.length;

  return (
    <div className="space-y-1">
      {hidden > 0 && (
        <p className="text-[10px] text-zinc-500">
          + {hidden} earlier warning{hidden > 1 ? "s" : ""}
        </p>
      )}
      {recent.map((w, i) => {
        const style = WARNING_STYLES[w.warning] ?? { icon: "⚠", color: "text-amber-400" };
        return (
          <div
            key={i}
            className="flex items-center gap-2 rounded border border-zinc-700 bg-zinc-800/60 px-2.5 py-1.5 text-xs"
          >
            <span>{style.icon}</span>
            <span className={`font-medium ${style.color}`}>
              {w.warning.replace(/_/g, " ")}
            </span>
            {w.component && (
              <span className="text-zinc-500">({w.component})</span>
            )}
            {w.detail && (
              <span className="text-zinc-400 truncate">{w.detail}</span>
            )}
          </div>
        );
      })}
    </div>
  );
}
