/**
 * StageStrip — secondary sticky bar that appears when the stage hero scrolls
 * out of the viewport.
 *
 * This component is purely presentational. The parent stage page is responsible
 * for wiring an IntersectionObserver to the hero element and passing `visible`
 * based on whether the hero is off-screen.
 *
 * Using `hidden={!visible}` keeps the element in the DOM at all times (matching
 * the same pattern as TechnicalDisclosure) while letting jest-dom's
 * `toBeVisible()` work correctly in tests.
 */

interface StageStripProps {
  /** "Stage N · Name" label text */
  label: string;
  /** Left-border accent colour */
  accentColor?: string;
  /** Controlled by the parent via an IntersectionObserver watching the hero */
  visible: boolean;
}

export default function StageStrip({
  label,
  accentColor = "var(--s0, #602468)",
  visible,
}: StageStripProps) {
  return (
    <div
      hidden={!visible}
      style={{
        position: "sticky",
        top: 60,
        zIndex: 30,
        background: "var(--paper, #f7f4ee)",
        borderBottom: "1px solid var(--line, #c9c2b3)",
        borderLeft: `3px solid ${accentColor}`,
        padding: "6px 16px",
        fontSize: 12,
        fontFamily: "var(--font-mono, 'JetBrains Mono', monospace)",
        color: "var(--muted, #6e6358)",
        letterSpacing: "0.04em",
      }}
    >
      {label}
    </div>
  );
}
