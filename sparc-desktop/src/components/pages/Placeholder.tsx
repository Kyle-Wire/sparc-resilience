import { SectionHeader } from "@/components/ui/DesignSystem";

interface PlaceholderProps {
  kicker: string;
  title: string;
  description: string;
}

export default function Placeholder({ kicker, title, description }: PlaceholderProps) {
  return (
    <div>
      <SectionHeader kicker={kicker} label={title} />
      <div
        style={{
          border: "1px dashed var(--line)",
          background: "repeating-linear-gradient(135deg, rgba(0,0,0,0.015) 0 8px, transparent 8px 16px)",
          borderRadius: 8,
          padding: 40,
          textAlign: "center",
        }}
      >
        <div
          className="mono"
          style={{ fontSize: 10, color: "var(--muted)", letterSpacing: "0.14em", textTransform: "uppercase" }}
        >
          view
        </div>
        <div style={{ fontSize: 16, fontWeight: 700, marginTop: 6 }}>{title}</div>
        <div style={{ fontSize: 12.5, color: "var(--muted)", marginTop: 6, maxWidth: 420, margin: "6px auto 0" }}>
          {description}
        </div>
      </div>
    </div>
  );
}
