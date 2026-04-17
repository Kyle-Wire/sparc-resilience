interface Props {
  status: string | null;
}

const STATUS_STYLES: Record<string, { bg: string; text: string; label: string; pulse: boolean }> = {
  training: { bg: "bg-blue-900/60", text: "text-blue-300", label: "Training", pulse: true },
  converging: { bg: "bg-amber-900/60", text: "text-amber-300", label: "Converging", pulse: true },
  converged: { bg: "bg-green-900/60", text: "text-green-300", label: "Converged", pulse: false },
};

export function ConvergenceBadge({ status }: Props) {
  if (!status) return null;

  const style = STATUS_STYLES[status] ?? STATUS_STYLES.training;

  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium ${style.bg} ${style.text}`}>
      {style.pulse && (
        <span className="relative flex h-2 w-2">
          <span className={`absolute inline-flex h-full w-full animate-ping rounded-full opacity-75 ${style.text.replace("text-", "bg-")}`} />
          <span className={`relative inline-flex h-2 w-2 rounded-full ${style.text.replace("text-", "bg-")}`} />
        </span>
      )}
      {style.label}
    </span>
  );
}
