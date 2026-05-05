/**
 * DatasetProfilePanel — researcher-facing per-column diagnostics.
 *
 * Reads the Phase 4 `/results/dataset/profile` endpoint (richer than
 * `/data/summary`).  Surfaces missing-rate, distribution shape (skew /
 * kurtosis), and Pearson correlation against the project target so the
 * user can spot degenerate or weakly-related predictors before training.
 */
import { useEffect, useMemo, useState } from "react";
import { Panel, PanelEmpty, Pill, Stat, StatGrid } from "@/components/ui/DesignSystem";
import { getDatasetProfile, type DatasetColumnProfile, type DatasetProfile } from "@/lib/api";

function fmt(n: number | null | undefined, digits = 3): string {
  if (n == null || !Number.isFinite(n)) return "—";
  if (Math.abs(n) >= 1e6) return n.toExponential(2);
  return n.toFixed(digits);
}

function corrTone(r: number | null | undefined): string {
  if (r == null || !Number.isFinite(r)) return "var(--ink-3)";
  const a = Math.abs(r);
  if (a >= 0.5) return "var(--purple)";
  if (a >= 0.2) return "var(--ink-2)";
  return "var(--ink-3)";
}

function missingTone(pct: number): string {
  if (pct >= 25) return "var(--crimson, #b91c1c)";
  if (pct >= 5) return "var(--amber, #b86a1e)";
  return "var(--ink-3)";
}

export default function DatasetProfilePanel() {
  const [profile, setProfile] = useState<DatasetProfile | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    getDatasetProfile()
      .then((p) => { if (alive) setProfile(p); })
      .catch((e) => { if (alive) setError(e instanceof Error ? e.message : "no profile"); });
    return () => { alive = false; };
  }, []);

  // Sort: target first, then by |corr_target| desc, then by missing_pct desc.
  const sorted = useMemo<DatasetColumnProfile[]>(() => {
    if (!profile) return [];
    const cols = [...profile.columns];
    cols.sort((a, b) => {
      if (profile.target) {
        if (a.name === profile.target) return -1;
        if (b.name === profile.target) return 1;
      }
      const ar = a.corr_target == null ? -1 : Math.abs(a.corr_target);
      const br = b.corr_target == null ? -1 : Math.abs(b.corr_target);
      if (ar !== br) return br - ar;
      return b.missing_pct - a.missing_pct;
    });
    return cols;
  }, [profile]);

  if (error) {
    return (
      <Panel title="Dataset profile" subtitle="per-column diagnostics">
        <PanelEmpty
          reason={error}
          hint="Load a project (Setup) so the dataset profile endpoint can read it."
        />
      </Panel>
    );
  }
  if (!profile) {
    return (
      <Panel title="Dataset profile" subtitle="per-column diagnostics">
        <PanelEmpty reason="Loading…" />
      </Panel>
    );
  }

  const nMissing = profile.columns.filter((c) => c.missing_pct > 0).length;
  const nDegenerate = profile.columns.filter((c) => c.n_unique != null && c.n_unique <= 1).length;

  return (
    <Panel
      title="Dataset profile"
      subtitle={profile.target ? `target · ${profile.target}` : "per-column diagnostics"}
      actions={profile.crs ? <Pill color="var(--ink-3)">{profile.crs}</Pill> : null}
    >
      <StatGrid>
        <Stat label="rows" value={profile.n_rows.toLocaleString()} />
        <Stat label="columns" value={String(profile.n_cols)} />
        <Stat label="cols w/ missing" value={String(nMissing)} />
        <Stat label="degenerate" value={String(nDegenerate)} />
      </StatGrid>
      <div style={{ marginTop: 12, maxHeight: 320, overflowY: "auto", fontSize: 12 }}>
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead style={{ position: "sticky", top: 0, background: "var(--paper-2)" }}>
            <tr style={{ textAlign: "left", color: "var(--ink-2)" }}>
              <th style={{ padding: "4px 6px" }}>column</th>
              <th style={{ padding: "4px 6px" }}>dtype</th>
              <th style={{ padding: "4px 6px", textAlign: "right" }}>miss%</th>
              <th style={{ padding: "4px 6px", textAlign: "right" }}>unique</th>
              <th style={{ padding: "4px 6px", textAlign: "right" }}>mean</th>
              <th style={{ padding: "4px 6px", textAlign: "right" }}>std</th>
              <th style={{ padding: "4px 6px", textAlign: "right" }}>skew</th>
              <th style={{ padding: "4px 6px", textAlign: "right" }}>r·target</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((c) => {
              const isTarget = c.name === profile.target;
              return (
                <tr key={c.name} style={{ borderTop: "1px solid var(--paper-3)" }}>
                  <td style={{ padding: "4px 6px", fontFamily: "var(--font-mono)", color: isTarget ? "var(--purple)" : undefined }}>
                    {c.name}{isTarget && " ★"}
                  </td>
                  <td style={{ padding: "4px 6px", color: "var(--ink-3)" }}>{c.dtype}</td>
                  <td style={{ padding: "4px 6px", textAlign: "right", color: missingTone(c.missing_pct) }}>
                    {c.missing_pct.toFixed(1)}
                  </td>
                  <td style={{ padding: "4px 6px", textAlign: "right" }}>{c.n_unique ?? "—"}</td>
                  <td style={{ padding: "4px 6px", textAlign: "right" }}>{fmt(c.mean, 2)}</td>
                  <td style={{ padding: "4px 6px", textAlign: "right" }}>{fmt(c.std, 2)}</td>
                  <td style={{ padding: "4px 6px", textAlign: "right" }}>{fmt(c.skew, 2)}</td>
                  <td style={{ padding: "4px 6px", textAlign: "right", color: corrTone(c.corr_target) }}>
                    {fmt(c.corr_target, 2)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}
