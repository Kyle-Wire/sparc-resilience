import { useEffect, useState, useMemo } from "react";
import { dataHistogram, type DataHistogram } from "@/lib/api";
import { SPARC_RAMP_HEX } from "@/lib/design-tokens";
import { EmptyState } from "@/components/common/EmptyState";

interface HistogramProps {
  variable: string;
  bins?: number;
  /** Pixel height of the chart (excluding labels). */
  height?: number;
  /** Compact rendering for inline use in lists. */
  compact?: boolean;
}

/**
 * Server-side histogram renderer. Pulls bin counts from /data/histogram and
 * draws SPARC-ramp gradient bars. Works on any dataset size since the server
 * returns ~`bins` ints.
 */
export function Histogram({ variable, bins = 40, height = 80, compact }: HistogramProps) {
  const [data, setData] = useState<DataHistogram | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    dataHistogram(variable, bins)
      .then((d) => {
        if (!cancelled) {
          setData(d);
          setLoading(false);
        }
      })
      .catch((e) => {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : "Failed to load histogram");
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [variable, bins]);

  const maxBin = useMemo(() => (data ? Math.max(1, ...data.bins) : 1), [data]);

  if (loading) {
    return (
      <div
        style={{
          height,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: 10.5,
          color: "var(--muted)",
          fontFamily: "var(--font-mono, monospace)",
        }}
      >
        loading…
      </div>
    );
  }

  if (error) {
    return compact ? (
      <div style={{ fontSize: 10, color: "var(--crimson)" }}>—</div>
    ) : (
      <EmptyState compact tone="awkward" title="Couldn't compute the histogram." body={error} />
    );
  }

  if (!data || data.n_finite === 0) {
    return compact ? (
      <div style={{ fontSize: 10, color: "var(--muted)" }}>no values</div>
    ) : (
      <EmptyState
        compact
        tone="blank"
        title="No numeric values in this column."
        body="Either the column is non-numeric or every value is missing."
      />
    );
  }

  const ramp = SPARC_RAMP_HEX;

  return (
    <div>
      <div
        style={{
          display: "flex",
          alignItems: "flex-end",
          gap: 1,
          height,
          padding: "0 1px",
        }}
      >
        {data.bins.map((c, i) => {
          const frac = c / maxBin;
          const colorIdx = Math.floor((i / Math.max(1, data.bins.length - 1)) * (ramp.length - 1));
          return (
            <div
              key={i}
              title={`[${data.edges[i].toFixed(2)}, ${data.edges[i + 1].toFixed(2)}] · ${c}`}
              style={{
                flex: 1,
                height: `${Math.max(2, frac * 100)}%`,
                background: ramp[colorIdx],
                borderRadius: "1px 1px 0 0",
                opacity: c === 0 ? 0.15 : 1,
              }}
            />
          );
        })}
      </div>
      {!compact && (
        <div
          className="mono"
          style={{
            display: "flex",
            justifyContent: "space-between",
            fontSize: 9.5,
            color: "var(--muted)",
            marginTop: 4,
            letterSpacing: "0.04em",
          }}
        >
          <span>{data.min !== null ? data.min.toFixed(2) : "—"}</span>
          <span>
            n={data.n_finite.toLocaleString()}
            {data.n_missing > 0 ? ` · missing ${data.n_missing.toLocaleString()}` : ""}
          </span>
          <span>{data.max !== null ? data.max.toFixed(2) : "—"}</span>
        </div>
      )}
    </div>
  );
}
