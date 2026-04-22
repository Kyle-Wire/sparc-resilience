import { useExplain, type ExplainMode } from "@/hooks/ExplainContext";

interface ExplainButtonProps {
  /** The question to send to Claude (templated against the current artifact). */
  prompt: string;
  /** Narrator (default) or hypothesis-generation mode. */
  mode?: ExplainMode;
  /** Optional label override; defaults to "Explain". */
  label?: string;
  /** Show only an icon, useful inside dense tables. */
  compact?: boolean;
  title?: string;
}

/**
 * Inline button that opens the chat panel pre-seeded with a focused
 * "narrator" question about a specific number, chart, or map cell.
 */
export default function ExplainButton({
  prompt,
  mode = "narrator",
  label,
  compact = false,
  title,
}: ExplainButtonProps) {
  const { requestExplain } = useExplain();
  return (
    <button
      type="button"
      onClick={() => requestExplain(prompt, mode)}
      title={title ?? "Open the assistant with a focused question about this value"}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 4,
        padding: compact ? "1px 5px" : "2px 8px",
        fontSize: compact ? 9 : 10,
        fontFamily: "'JetBrains Mono', monospace",
        textTransform: "uppercase",
        letterSpacing: "0.06em",
        color: "var(--purple)",
        background: "transparent",
        border: "1px solid var(--purple)",
        borderRadius: 3,
        cursor: "pointer",
        lineHeight: 1.2,
      }}
    >
      <span aria-hidden="true">›</span> {label ?? "Explain"}
    </button>
  );
}
