/**
 * NarrativeCallout — sticky left-column callout card per figure section.
 *
 * Two text paths:
 *   1. AI path  — `body` is provided by the Claude integration
 *   2. Fallback — `fallback` is a templated string used when no API key
 *
 * While the AI request is in-flight the caller passes `loading={true}`
 * and a skeleton placeholder is shown instead of body text.
 */

interface NarrativeCalloutProps {
  /** Section headline — always visible. */
  headline: string;
  /** AI-generated (or computed) body text. Takes priority over fallback. */
  body?: string;
  /** Templated fallback shown when body is absent and not loading. */
  fallback?: string;
  /** True while the AI narrative is being fetched. Shows skeleton. */
  loading?: boolean;
  /** Left-border accent colour. Defaults to Stage 0 purple. */
  accentColor?: string;
}

export default function NarrativeCallout({
  headline,
  body,
  fallback,
  loading = false,
  accentColor = "var(--s0, #602468)",
}: NarrativeCalloutProps) {
  const displayText = body ?? fallback;

  return (
    <aside
      style={{
        borderLeft: `3px solid ${accentColor}`,
        paddingLeft: 16,
        position: "sticky",
        top: 112,
        alignSelf: "start",
      }}
    >
      <h3
        style={{
          margin: "0 0 8px",
          fontSize: 13,
          fontWeight: 600,
          color: "var(--ink, #1a1416)",
          fontFamily: "var(--font-sans, Inter, sans-serif)",
          textTransform: "uppercase",
          letterSpacing: "0.06em",
        }}
      >
        {headline}
      </h3>

      {loading ? (
        <div data-testid="narrative-skeleton" aria-label="Loading narrative">
          <div
            style={{
              background: "var(--paper-2, #ede8dd)",
              height: 10,
              width: "85%",
              borderRadius: 4,
              marginBottom: 6,
            }}
          />
          <div
            style={{
              background: "var(--paper-2, #ede8dd)",
              height: 10,
              width: "65%",
              borderRadius: 4,
              marginBottom: 6,
            }}
          />
          <div
            style={{
              background: "var(--paper-2, #ede8dd)",
              height: 10,
              width: "75%",
              borderRadius: 4,
            }}
          />
        </div>
      ) : displayText ? (
        <p
          style={{
            margin: 0,
            fontSize: 13,
            lineHeight: 1.6,
            color: "var(--muted, #6e6358)",
          }}
        >
          {displayText}
        </p>
      ) : null}
    </aside>
  );
}
