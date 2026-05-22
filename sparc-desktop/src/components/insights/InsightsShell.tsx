/**
 * InsightsShell — top-level layout for the Insights workspace.
 *
 * Structure:
 *   [ Header band: title · audience toggle · Confidence badge · actions ]
 *   [ Rail (panel groups) | Stacked panels (audience-filtered) ]
 *
 * The shell itself is layout-only. It accepts a list of panel descriptors
 * and renders them in order, hiding any whose audience does not match.
 * Panels are responsible for their own data fetching and empty-states.
 */
import { useMemo, useEffect, useState, useRef } from "react";
import type { ReactNode } from "react";
import { useAudience, type Audience } from "@/hooks/InsightsProvider";
import { Kicker, RampStrip } from "@/components/ui/DesignSystem";

export interface InsightsPanelDescriptor {
  /** Stable id; used for rail anchor + key. */
  id: string;
  /** Group label shown in the rail. */
  group: string;
  /** Display name for the rail link. */
  label: string;
  /** Audience(s) that should see this panel. */
  audience: Audience | Audience[];
  /**
   * Optional pipeline stage badge shown next to the group header in the rail
   * and as a section divider in the canvas. E.g. "Stage 0", "Stage 3".
   * Only the first panel in each group needs this set.
   */
  stage?: string;
  /** Render the panel's body. */
  render: () => ReactNode;
}

interface ShellProps {
  panels: InsightsPanelDescriptor[];
  /** Optional content for the right side of the header (e.g. confidence badge). */
  headerExtras?: ReactNode;
}

function audienceMatches(want: Audience | Audience[], current: Audience): boolean {
  return Array.isArray(want) ? want.includes(current) : want === current;
}

export default function InsightsShell({ panels, headerExtras }: ShellProps) {
  const [audience, setAudience] = useAudience();
  const [activeId, setActiveId] = useState<string | null>(null);
  const mainRef = useRef<HTMLElement>(null);

  const visible = useMemo(
    () => panels.filter((p) => audienceMatches(p.audience, audience)),
    [panels, audience],
  );

  const groups = useMemo(() => {
    const out: { group: string; stage?: string; items: InsightsPanelDescriptor[] }[] = [];
    for (const p of visible) {
      const last = out[out.length - 1];
      if (last && last.group === p.group) {
        last.items.push(p);
      } else {
        out.push({ group: p.group, stage: p.stage, items: [p] });
      }
    }
    return out;
  }, [visible]);

  // Scroll-spy: observe all panel sections and track which one is in view
  useEffect(() => {
    const ids = visible.map((p) => p.id);
    const observers: IntersectionObserver[] = [];
    // Use a map to track intersection ratios per panel
    const ratios: Record<string, number> = {};

    ids.forEach((id) => {
      const el = document.getElementById(id);
      if (!el) return;
      const obs = new IntersectionObserver(
        ([entry]) => {
          ratios[id] = entry.intersectionRatio;
          // Active = the visible panel with the highest intersection ratio
          const best = ids.reduce<string | null>((acc, cur) => {
            const r = ratios[cur] ?? 0;
            return r > (ratios[acc ?? ""] ?? 0) ? cur : acc;
          }, null);
          if (best) setActiveId(best);
        },
        { root: mainRef.current, threshold: [0, 0.1, 0.25, 0.5, 0.75, 1.0] },
      );
      obs.observe(el);
      observers.push(obs);
    });

    return () => observers.forEach((o) => o.disconnect());
  }, [visible]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16, height: "100%" }}>
      {/* Header band */}
      <div>
        <RampStrip />
        <div
          style={{
            display: "flex",
            alignItems: "flex-end",
            justifyContent: "space-between",
            gap: 16,
            padding: "14px 4px 12px",
            borderBottom: "1px solid var(--line)",
          }}
        >
          <div>
            <Kicker>Insights · linked dashboard</Kicker>
            <div style={{ fontSize: 28, fontWeight: 800, letterSpacing: "-0.02em", marginTop: 4 }}>
              {audience === "practitioner" ? "What should we do?"
                : audience === "public" ? "What does this mean for me?"
                : "What's true & how sure?"}
            </div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            {headerExtras}
            <div className="audience-toggle" role="tablist" aria-label="Audience">
              <button
                type="button"
                role="tab"
                aria-selected={audience === "practitioner"}
                className={audience === "practitioner" ? "active" : ""}
                onClick={() => setAudience("practitioner")}
              >
                Planner
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={audience === "researcher"}
                className={audience === "researcher" ? "active" : ""}
                onClick={() => setAudience("researcher")}
              >
                Researcher
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={audience === "public"}
                className={audience === "public" ? "active" : ""}
                onClick={() => setAudience("public")}
              >
                Public
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Rail + canvas */}
      <div className="insights-shell" style={{ flex: 1, minHeight: 0 }}>
        <aside className="insights-rail scroll" style={{ overflowY: "auto", maxHeight: "100%" }}>
          {groups.map((g) => (
            <div key={g.group}>
              <div className="group-label" style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 6 }}>
                <span>{g.group}</span>
                {g.stage && (
                  <span style={{
                    fontSize: 8,
                    fontWeight: 700,
                    letterSpacing: "0.08em",
                    padding: "1px 5px",
                    borderRadius: 3,
                    background: "rgba(0,0,0,0.06)",
                    color: "var(--muted)",
                    textTransform: "uppercase",
                    flexShrink: 0,
                  }}>
                    {g.stage}
                  </span>
                )}
              </div>
              {g.items.map((p) => (
                <a
                  key={p.id}
                  href={`#${p.id}`}
                  className={activeId === p.id ? "active" : ""}
                  onClick={(e) => {
                    e.preventDefault();
                    document.getElementById(p.id)?.scrollIntoView({ behavior: "smooth", block: "start" });
                  }}
                >
                  {p.label}
                </a>
              ))}
            </div>
          ))}
        </aside>

        <main
          ref={mainRef}
          className="scroll"
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 0,
            overflowY: "auto",
            paddingRight: 4,
            paddingBottom: 80,
          }}
        >
          {visible.length === 0 && (
            <div style={{ padding: 32 }}>
              <Kicker>No panels for this audience</Kicker>
            </div>
          )}
          {groups.map((g, gi) => (
            <div key={g.group}>
              {/* Group divider — visible section break in the canvas */}
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 12,
                  padding: gi === 0 ? "0 0 14px" : "28px 0 14px",
                }}
              >
                <div style={{ flex: 1, height: 1, background: "var(--line)" }} />
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  {g.stage && (
                    <span
                      className="mono"
                      style={{
                        fontSize: 9,
                        fontWeight: 700,
                        letterSpacing: "0.12em",
                        padding: "2px 7px",
                        borderRadius: 3,
                        background: "var(--ink)",
                        color: "#fff",
                        textTransform: "uppercase",
                      }}
                    >
                      {g.stage}
                    </span>
                  )}
                  <span
                    className="mono"
                    style={{
                      fontSize: 10,
                      fontWeight: 700,
                      letterSpacing: "0.14em",
                      color: "var(--muted)",
                      textTransform: "uppercase",
                    }}
                  >
                    {g.group}
                  </span>
                </div>
                <div style={{ flex: 1, height: 1, background: "var(--line)" }} />
              </div>

              {/* Panels in this group */}
              <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                {g.items.map((p) => (
                  <section key={p.id} id={p.id} style={{ scrollMarginTop: 12 }}>
                    {p.render()}
                  </section>
                ))}
              </div>
            </div>
          ))}
        </main>
      </div>
    </div>
  );
}
