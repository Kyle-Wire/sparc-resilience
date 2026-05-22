/**
 * VariableManifestTable — shows all 27+ UHI variables with status badges,
 * coverage %, source name, and a tier indicator.
 */
import type { CollectManifest, ManifestEntry } from "@/lib/api";

interface Props {
  manifest: CollectManifest;
}

const STATUS_STYLE: Record<ManifestEntry["status"], { bg: string; label: string }> = {
  PENDING:   { bg: "#e0e0e0", label: "Pending" },
  FETCHING:  { bg: "#fff3cd", label: "Fetching…" },
  COMPLETE:  { bg: "#d4edda", label: "Complete" },
  WARNING:   { bg: "#ffeeba", label: "Warning" },
  ERROR:     { bg: "#f8d7da", label: "Error" },
  SKIPPED:   { bg: "#e2e3e5", label: "Skipped" },
};

const GROUP_ORDER = [
  "aat_residual", "aat_morning", "aat_midday", "aat_night", "diurnal_aat", "era5_t2m",
  "lst",
  "ndvi", "ndbi", "mndwi", "albedo", "ndwi", "savi", "evi", "ndbai", "nbr", "ndmi", "ui", "bsi",
  "pct_impervious", "pct_canopy", "land_cover",
  "bldg_height_mean", "bldg_coverage", "svf",
  "holc_grade", "cdc_svi", "ejscreen_score",
];

export default function VariableManifestTable({ manifest }: Props) {
  // Sort variables by GROUP_ORDER, unknown ones appended at end
  const sorted = [...manifest.variables].sort((a, b) => {
    const ai = GROUP_ORDER.indexOf(a.name);
    const bi = GROUP_ORDER.indexOf(b.name);
    if (ai === -1 && bi === -1) return a.name.localeCompare(b.name);
    if (ai === -1) return 1;
    if (bi === -1) return -1;
    return ai - bi;
  });

  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
        <thead>
          <tr style={{ borderBottom: "2px solid var(--border)", background: "var(--surface-raised, #f9f9f9)" }}>
            <th style={TH}>Variable</th>
            <th style={TH}>Required</th>
            <th style={TH}>Status</th>
            <th style={TH}>Coverage</th>
            <th style={TH}>Source</th>
            <th style={TH}>Tier</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((entry) => (
            <VariableRow key={entry.name} entry={entry} />
          ))}
        </tbody>
      </table>
      {!manifest.can_build && manifest.blocking_variables.length > 0 && (
        <div style={{
          marginTop: 12,
          padding: "8px 12px",
          background: "#f8d7da",
          borderRadius: 6,
          fontSize: 13,
          color: "#721c24",
        }}>
          <strong>Cannot build dataset.</strong> Required variable(s) failed:{" "}
          {manifest.blocking_variables.join(", ")}
        </div>
      )}
    </div>
  );
}

function VariableRow({ entry }: { entry: ManifestEntry }) {
  const s = STATUS_STYLE[entry.status] ?? STATUS_STYLE.PENDING;
  const coveragePct = entry.coverage_pct != null ? `${(entry.coverage_pct * 100).toFixed(1)}%` : "—";
  return (
    <tr style={{ borderBottom: "1px solid var(--border)", verticalAlign: "middle" }}>
      <td style={TD}>
        <code style={{ fontSize: 12 }}>{entry.name}</code>
      </td>
      <td style={{ ...TD, textAlign: "center" }}>
        {entry.required ? (
          <span style={{ color: "#721c24", fontWeight: 700 }}>●</span>
        ) : (
          <span style={{ color: "var(--ink-muted)" }}>○</span>
        )}
      </td>
      <td style={TD}>
        <span style={{
          display: "inline-block",
          padding: "2px 8px",
          borderRadius: 10,
          background: s.bg,
          fontSize: 11,
          fontWeight: 600,
          whiteSpace: "nowrap",
        }}>
          {s.label}
        </span>
      </td>
      <td style={{ ...TD, textAlign: "right", fontVariantNumeric: "tabular-nums" }}>
        {entry.status === "COMPLETE" || entry.status === "WARNING" ? coveragePct : "—"}
      </td>
      <td style={{ ...TD, color: "var(--ink-muted)", maxWidth: 200, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
        {entry.source_name ?? "—"}
        {entry.error_message && (
          <span title={entry.error_message} style={{ marginLeft: 6, color: "#721c24", cursor: "help" }}>⚠</span>
        )}
      </td>
      <td style={{ ...TD, textAlign: "center", color: "var(--ink-muted)" }}>
        {entry.tier != null ? `T${entry.tier}` : "—"}
      </td>
    </tr>
  );
}

const TH: React.CSSProperties = {
  padding: "8px 10px",
  textAlign: "left",
  fontWeight: 700,
  fontSize: 12,
  color: "var(--ink-muted)",
  whiteSpace: "nowrap",
};

const TD: React.CSSProperties = {
  padding: "6px 10px",
};
