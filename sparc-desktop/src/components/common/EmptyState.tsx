import type { ReactNode } from "react";

/**
 * Standardized empty-state surface used everywhere data isn't available yet.
 *
 * Tone: warm, plain-spoken, never accusatory. Always tells the user
 * *what* is missing and *how* to get it. Replaces the old habit of
 * showing "Error: failed to fetch" toasts for missing-but-expected data.
 */

export type EmptyStateTone = "awkward" | "waiting" | "blocked" | "blank";

interface EmptyStateProps {
  /** Big friendly headline. Keep it human. */
  title: string;
  /** One or two sentences explaining what's missing and how to fix it. */
  body?: ReactNode;
  /** Optional action — usually a button that takes the user to the right page. */
  action?: ReactNode;
  /** Visual mood. Affects the icon glyph and accent color only. */
  tone?: EmptyStateTone;
  /** Compact variant for cards. Default is comfortable. */
  compact?: boolean;
}

const TONE_GLYPHS: Record<EmptyStateTone, string> = {
  awkward: "◌",
  waiting: "◐",
  blocked: "◇",
  blank: "○",
};

const TONE_COLORS: Record<EmptyStateTone, string> = {
  awkward: "var(--amber)",
  waiting: "var(--purple)",
  blocked: "var(--crimson)",
  blank: "var(--muted)",
};

export function EmptyState({
  title,
  body,
  action,
  tone = "blank",
  compact = false,
}: EmptyStateProps) {
  const accent = TONE_COLORS[tone];
  const padY = compact ? 22 : 44;

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: 10,
        padding: `${padY}px 18px`,
        textAlign: "center",
        color: "var(--ink-2)",
      }}
    >
      <div
        aria-hidden
        style={{
          width: compact ? 36 : 52,
          height: compact ? 36 : 52,
          borderRadius: "50%",
          border: `1px dashed ${accent}`,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: compact ? 18 : 26,
          color: accent,
          background: `${accent}10`,
        }}
      >
        {TONE_GLYPHS[tone]}
      </div>
      <div
        style={{
          fontSize: compact ? 13 : 15,
          fontWeight: 700,
          letterSpacing: "-0.01em",
          maxWidth: 420,
        }}
      >
        {title}
      </div>
      {body && (
        <div
          style={{
            fontSize: compact ? 11.5 : 12.5,
            lineHeight: 1.55,
            color: "var(--muted)",
            maxWidth: 440,
          }}
        >
          {body}
        </div>
      )}
      {action && <div style={{ marginTop: 6 }}>{action}</div>}
    </div>
  );
}

/**
 * Shorthand for the common "you haven't run the pipeline yet" case.
 * Pass the stage name and a callback to navigate to the Run page.
 */
export function StageMissingEmptyState({
  stageLabel,
  onGoToRun,
}: {
  stageLabel: string;
  onGoToRun?: () => void;
}) {
  return (
    <EmptyState
      tone="awkward"
      title="This is awkward — nothing here yet."
      body={
        <>
          This view needs <strong>{stageLabel}</strong> to finish first. When
          you're ready, head over to the <strong>Run</strong> page and start the
          pipeline.
        </>
      }
      action={
        onGoToRun ? (
          <button
            onClick={onGoToRun}
            style={{
              border: "1px solid var(--ink)",
              background: "var(--ink)",
              color: "#fff",
              padding: "6px 14px",
              borderRadius: 5,
              fontSize: 11.5,
              fontWeight: 600,
              cursor: "pointer",
              fontFamily: "inherit",
            }}
          >
            Go to Run →
          </button>
        ) : undefined
      }
    />
  );
}
