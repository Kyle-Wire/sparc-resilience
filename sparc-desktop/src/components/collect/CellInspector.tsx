/**
 * CellInspector — slide-in panel that shows all variable values for a clicked
 * grid cell.  Missing values render as a dash with a warning badge.
 */
import { Btn } from "@/components/ui/DesignSystem";

interface Props {
  cellId: number | null;
  data: Record<string, unknown> | null;
  loading: boolean;
  onClose: () => void;
}

const HIDDEN_KEYS = new Set(["geometry", "__index_level_0__"]);

export default function CellInspector({ cellId, data, loading, onClose }: Props) {
  if (cellId === null) return null;

  const rows: Array<{ key: string; value: unknown }> = data
    ? Object.entries(data)
        .filter(([k]) => !HIDDEN_KEYS.has(k))
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([key, value]) => ({ key, value }))
    : [];

  return (
    <div style={{
      position: "absolute",
      top: 12,
      right: 12,
      width: 280,
      maxHeight: "calc(100% - 24px)",
      background: "var(--surface, #fff)",
      border: "1px solid var(--border)",
      borderRadius: 8,
      boxShadow: "0 4px 16px rgba(0,0,0,0.12)",
      display: "flex",
      flexDirection: "column",
      zIndex: 100,
    }}>
      {/* Header */}
      <div style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        padding: "10px 14px",
        borderBottom: "1px solid var(--border)",
      }}>
        <span style={{ fontWeight: 700, fontSize: 13 }}>Cell {cellId}</span>
        <Btn small onClick={onClose}>✕</Btn>
      </div>

      {/* Body */}
      <div style={{ overflowY: "auto", flex: 1, padding: "8px 0" }}>
        {loading && (
          <p style={{ padding: "8px 14px", fontSize: 13, color: "var(--ink-muted)" }}>
            Loading…
          </p>
        )}
        {!loading && rows.length === 0 && (
          <p style={{ padding: "8px 14px", fontSize: 13, color: "var(--ink-muted)" }}>
            No data
          </p>
        )}
        {!loading && rows.map(({ key, value }) => {
          const missing = value === null || value === undefined;
          return (
            <div key={key} style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              padding: "4px 14px",
              borderBottom: "1px solid var(--border)",
            }}>
              <span style={{ fontSize: 12, color: "var(--ink-muted)", flex: 1 }}>{key}</span>
              <span style={{
                fontSize: 12,
                fontVariantNumeric: "tabular-nums",
                color: missing ? "#c0392b" : "var(--ink)",
                fontWeight: missing ? 700 : 400,
                marginLeft: 8,
              }}>
                {missing ? "—" : formatValue(value)}
              </span>
              {missing && (
                <span title="Missing value" style={{ marginLeft: 4, fontSize: 10, color: "#c0392b" }}>⚠</span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function formatValue(value: unknown): string {
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(4);
  }
  if (typeof value === "boolean") return value ? "true" : "false";
  return String(value);
}
