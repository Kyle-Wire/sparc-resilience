/**
 * GwenApprovalModal — surfaces GWEN variable-importance results after Stage 1
 * and lets the user approve (or skip) before Stage 2 begins.
 *
 * Triggered when the pipeline emits a `gwen_approval_requested` event, or when
 * Stage 1 completes and `gwen_approved.txt` does not yet exist.
 */
import { useEffect, useState, useCallback } from "react";
import { Btn, Panel, Pill } from "@/components/ui/DesignSystem";
import { getGwenStatus, approveGwen, type GwenStatus } from "@/lib/api";

interface Props {
  onApproved: () => void;
  onDismiss: () => void;
}

export function GwenApprovalModal({ onApproved, onDismiss }: Props) {
  const [status, setStatus] = useState<GwenStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [approving, setApproving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    getGwenStatus()
      .then(setStatus)
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  const handleApprove = useCallback(async () => {
    setApproving(true);
    try {
      await approveGwen();
      onApproved();
    } catch (e) {
      setError(String(e));
    } finally {
      setApproving(false);
    }
  }, [onApproved]);

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 1000,
        background: "rgba(0,0,0,0.45)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <div
        style={{
          background: "var(--surface)",
          borderRadius: 10,
          border: "1px solid var(--line)",
          padding: 28,
          maxWidth: 720,
          width: "90vw",
          maxHeight: "80vh",
          overflow: "auto",
          boxShadow: "0 8px 40px rgba(0,0,0,0.2)",
        }}
      >
        <div style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase", color: "var(--muted)", marginBottom: 4 }}>
            Stage 1 · GWEN variable selection
          </div>
          <h2 style={{ margin: 0, fontSize: 20, fontWeight: 800 }}>Review selected predictors</h2>
          <p style={{ margin: "8px 0 0", fontSize: 13, color: "var(--ink-2)" }}>
            GWEN has ranked your predictors. Review the importance scores and approve to continue to Stage 2 (Spatial CV), or dismiss to come back later.
          </p>
        </div>

        {error && (
          <div style={{ fontSize: 12, color: "#c0392b", padding: "8px 12px", background: "#fff0f0", borderRadius: 4, marginBottom: 12 }}>
            {error}
          </div>
        )}

        {loading ? (
          <div style={{ fontSize: 13, color: "var(--muted)", padding: "16px 0" }}>Loading GWEN results…</div>
        ) : status?.rows?.length ? (
          <Panel title="Variable importance">
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", fontSize: 12, borderCollapse: "collapse" }}>
                <thead>
                  <tr style={{ textAlign: "left", borderBottom: "1px solid var(--line)" }}>
                    <th style={{ padding: "6px 8px" }}>Variable</th>
                    <th style={{ padding: "6px 8px", textAlign: "right" }}>Importance</th>
                    <th style={{ padding: "6px 8px", textAlign: "right" }}>Rank</th>
                    <th style={{ padding: "6px 8px" }}>Selected</th>
                  </tr>
                </thead>
                <tbody>
                  {status.rows
                    .slice()
                    .sort((a: any, b: any) => (b.importance ?? 0) - (a.importance ?? 0))
                    .map((row: any, i: number) => {
                      const imp = typeof row.importance === "number" ? row.importance : null;
                      const selected = row.selected ?? row.gwen_selected ?? (imp != null && imp > 0.01);
                      return (
                        <tr key={i} style={{ borderBottom: "1px solid rgba(0,0,0,0.04)" }}>
                          <td style={{ padding: "6px 8px", fontFamily: "var(--mono)" }}>{row.variable ?? row.feature ?? row.name ?? `row ${i}`}</td>
                          <td style={{ padding: "6px 8px", textAlign: "right", fontVariantNumeric: "tabular-nums" }}>
                            {imp != null ? (
                              <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                                <span
                                  style={{
                                    display: "inline-block",
                                    width: Math.max(4, Math.round(imp * 80)),
                                    height: 8,
                                    background: "var(--crimson)",
                                    borderRadius: 2,
                                    opacity: 0.7,
                                  }}
                                />
                                {imp.toFixed(4)}
                              </span>
                            ) : "—"}
                          </td>
                          <td style={{ padding: "6px 8px", textAlign: "right", color: "var(--ink-2)" }}>{i + 1}</td>
                          <td style={{ padding: "6px 8px" }}>
                            {selected ? (
                              <Pill color="var(--green, #2f7d32)">✓ Yes</Pill>
                            ) : (
                              <Pill color="var(--ink-3, #aaa)">No</Pill>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                </tbody>
              </table>
            </div>
          </Panel>
        ) : (
          <div style={{ fontSize: 13, color: "var(--muted)", padding: "16px 0" }}>
            GWEN results are not yet available. You can approve to proceed anyway.
          </div>
        )}

        <div style={{ display: "flex", gap: 10, marginTop: 20, justifyContent: "flex-end" }}>
          <Btn onClick={onDismiss} style={{}}>Dismiss — decide later</Btn>
          <Btn primary onClick={handleApprove} disabled={approving}>
            {approving ? "Approving…" : "✓ Approve & continue to Stage 2"}
          </Btn>
        </div>
      </div>
    </div>
  );
}
