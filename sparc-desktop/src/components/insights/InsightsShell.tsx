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
import { useMemo } from "react";
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

  const visible = useMemo(
    () => panels.filter((p) => audienceMatches(p.audience, audience)),
    [panels, audience],
  );

  const groups = useMemo(() => {
    const out: { group: string; items: InsightsPanelDescriptor[] }[] = [];
    for (const p of visible) {
      const last = out[out.length - 1];
      if (last && last.group === p.group) {
        last.items.push(p);
      } else {
        out.push({ group: p.group, items: [p] });
      }
    }
    return out;
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
              {audience === "practitioner" ? "What should we do?" : "What's true & how sure?"}
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
                Practitioner
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
            </div>
          </div>
        </div>
      </div>

      {/* Rail + canvas */}
      <div className="insights-shell" style={{ flex: 1, minHeight: 0 }}>
        <aside className="insights-rail scroll" style={{ overflowY: "auto", maxHeight: "100%" }}>
          {groups.map((g) => (
            <div key={g.group}>
              <div className="group-label">{g.group}</div>
              {g.items.map((p) => (
                <a
                  key={p.id}
                  href={`#${p.id}`}
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
          className="scroll"
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 16,
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
          {visible.map((p) => (
            <section key={p.id} id={p.id} style={{ scrollMarginTop: 12 }}>
              {p.render()}
            </section>
          ))}
        </main>
      </div>
    </div>
  );
}
