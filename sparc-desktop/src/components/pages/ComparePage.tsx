import { useCallback, useEffect, useMemo, useState } from "react";
import { SectionHeader, Card, Btn, Stat, StatGrid, Tag } from "@/components/ui/DesignSystem";
import {
  discoverRuns,
  diffRuns,
  getProvenance,
  loadFrozenRun,
  verifyReproduction,
  type RunCandidate,
  type RunDiffResponse,
  type ProvenanceData,
  type ReproduceVerifyResponse,
} from "@/lib/api";
import { useNotification } from "@/hooks/useNotifications";

/**
 * Compare page (Phase 15).
 *
 * Diffs two SPARC output directories on disk. No re-runs needed —
 * just reads JSON artifacts and surfaces config / model / causal /
 * DAG deltas side-by-side.
 */
export default function ComparePage() {
  const { notify } = useNotification();

  const [searchRoot, setSearchRoot] = useState<string>("");
  const [runs, setRuns] = useState<RunCandidate[]>([]);
  const [runA, setRunA] = useState<string>("");
  const [runB, setRunB] = useState<string>("");
  const [diff, setDiff] = useState<RunDiffResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [provA, setProvA] = useState<ProvenanceData | null>(null);
  const [provB, setProvB] = useState<ProvenanceData | null>(null);
  const [verify, setVerify] = useState<ReproduceVerifyResponse | null>(null);

  const loadRuns = useCallback(async (root?: string) => {
    try {
      const res = await discoverRuns(root || undefined);
      setRuns(res.runs);
      setSearchRoot(res.search_root);
      if (res.runs.length >= 2 && !runA && !runB) {
        setRunA(res.runs[0].path);
        setRunB(res.runs[1].path);
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      notify("error", `Could not list runs: ${msg}`);
    }
  }, [notify, runA, runB]);

  useEffect(() => { loadRuns(); }, [loadRuns]);

  // Auto-fetch provenance whenever a run is selected
  useEffect(() => {
    if (!runA) { setProvA(null); return; }
    getProvenance(runA).then((r) => setProvA(r.provenance)).catch(() => setProvA(null));
  }, [runA]);
  useEffect(() => {
    if (!runB) { setProvB(null); return; }
    getProvenance(runB).then((r) => setProvB(r.provenance)).catch(() => setProvB(null));
  }, [runB]);

  const reproduceA = useCallback(async () => {
    if (!runA) return;
    try {
      const r = await loadFrozenRun(runA);
      notify("success", `Loaded frozen config from ${runA.split(/[\\/]/).pop()}. Re-run via the Run page, then click Verify.`);
      if (!r.config_hash) notify("warning", "Run had no recorded config hash");
    } catch (err) {
      notify("error", err instanceof Error ? err.message : String(err));
    }
  }, [runA, notify]);

  const verifyAB = useCallback(async () => {
    if (!runA || !runB) return;
    try {
      const r = await verifyReproduction(runA, runB);
      setVerify(r);
      const ok = r.config_hash_match && r.data_hash_match;
      notify(ok ? "success" : "warning",
        `Config ${r.config_hash_match ? "\u2713" : "\u2717"} · Data ${r.data_hash_match ? "\u2713" : "\u2717"} · Git ${r.git_commit_match ? "\u2713" : "\u2717"} · Sidecars ${r.sidecars_b.n_match}/${r.sidecars_b.checked.length}`);
    } catch (err) {
      notify("error", err instanceof Error ? err.message : String(err));
    }
  }, [runA, runB, notify]);

  const runDiff = useCallback(async () => {
    if (!runA || !runB) {
      notify("warning", "Pick two runs to compare");
      return;
    }
    if (runA === runB) {
      notify("warning", "Pick two different runs");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await diffRuns(runA, runB);
      setDiff(res);
      notify("success", `Diffed: ${res.config.changes.length} config changes · ${res.causal.effects.length} effects · ${res.dag.added.length + res.dag.removed.length + res.dag.changed.length} DAG edits`);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      notify("error", msg);
    } finally {
      setLoading(false);
    }
  }, [runA, runB, notify]);

  const counts = useMemo(() => ({
    config: diff?.config.changes.length ?? 0,
    models: diff?.models.rows.length ?? 0,
    causal: diff?.causal.effects.length ?? 0,
    dag: diff ? diff.dag.added.length + diff.dag.removed.length + diff.dag.changed.length : 0,
  }), [diff]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <SectionHeader
        kicker="COMPARE"
        label="Pairwise run diff"
        right={
          <Btn primary onClick={runDiff} disabled={loading || !runA || !runB}>
            {loading ? "Diffing…" : "Run diff"}
          </Btn>
        }
      />

      <StatGrid>
        <Stat label="Config changes" value={String(counts.config)} />
        <Stat label="Model rows" value={String(counts.models)} />
        <Stat label="Causal effects" value={String(counts.causal)} />
        <Stat label="DAG edits" value={String(counts.dag)} />
      </StatGrid>

      {error && (
        <Card title="Error">
          <div style={{ color: "var(--crimson)", fontSize: 11, fontFamily: "'JetBrains Mono', monospace" }}>{error}</div>
        </Card>
      )}

      <Card
        title="Run picker"
        subtitle={searchRoot ? `searching: ${searchRoot} · ${runs.length} runs found` : "loading…"}
        actions={
          <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
            <input
              type="text"
              placeholder="search root (optional)"
              value={searchRoot}
              onChange={(e) => setSearchRoot(e.target.value)}
              style={{
                fontSize: 11, padding: "3px 6px", borderRadius: 3,
                border: "1px solid var(--line)", width: 280,
                fontFamily: "'JetBrains Mono', monospace",
              }}
            />
            <Btn onClick={() => loadRuns(searchRoot)}>Rescan</Btn>
          </div>
        }
      >
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
          {(["A", "B"] as const).map((side) => {
            const value = side === "A" ? runA : runB;
            const setter = side === "A" ? setRunA : setRunB;
            return (
              <div key={side} style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <div className="mono" style={{ fontSize: 9.5, color: "var(--muted)", letterSpacing: "0.1em", textTransform: "uppercase" }}>
                  Run {side}
                </div>
                <select
                  value={value}
                  onChange={(e) => setter(e.target.value)}
                  style={{
                    fontSize: 11, padding: "4px 6px", borderRadius: 3,
                    border: "1px solid var(--line)", fontFamily: "'JetBrains Mono', monospace",
                  }}
                >
                  <option value="">— pick a run —</option>
                  {runs.map((r) => {
                    const date = r.mtime ? new Date(r.mtime * 1000).toISOString().slice(0, 16).replace("T", " ") : "—";
                    return (
                      <option key={r.path} value={r.path}>
                        {r.name} · {date}
                      </option>
                    );
                  })}
                </select>
                {value && (
                  <div className="mono" style={{ fontSize: 9, color: "var(--muted)", wordBreak: "break-all" }}>
                    {value}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </Card>

      <Card
        title="Reproducibility"
        subtitle="Frozen config + data hash + git commit + env lock"
        actions={
          <div style={{ display: "flex", gap: 6 }}>
            <Btn onClick={reproduceA} disabled={!runA}>Reproduce A</Btn>
            <Btn primary onClick={verifyAB} disabled={!runA || !runB}>Verify A↔B</Btn>
          </div>
        }
      >
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
          {([
            ["A", provA] as const,
            ["B", provB] as const,
          ]).map(([side, p]) => (
            <div key={side}>
              <div className="mono" style={{ fontSize: 9.5, color: "var(--muted)", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 4 }}>
                Run {side}
              </div>
              {p ? (
                <div style={{ fontSize: 11, lineHeight: 1.6 }}>
                  <div><span style={{ color: "var(--muted)" }}>config:</span> <span className="mono">{(p.config_hash || "").slice(0, 16)}…</span></div>
                  <div><span style={{ color: "var(--muted)" }}>data:</span> <span className="mono">{((p.data?.sha256) || "—").slice(0, 16)}…</span></div>
                  <div><span style={{ color: "var(--muted)" }}>git:</span> <span className="mono">{(p.git?.commit || "—").slice(0, 8)}{p.git?.dirty ? " +dirty" : ""}</span></div>
                  <div><span style={{ color: "var(--muted)" }}>frozen:</span> {p.timestamp_utc}</div>
                  <div><span style={{ color: "var(--muted)" }}>env pkgs:</span> {Object.keys(p.env || {}).length}</div>
                </div>
              ) : (
                <div style={{ fontSize: 11, color: "var(--muted)" }}>
                  {side === "A" ? (runA ? "No run_provenance.json found." : "Pick a run to inspect.") : (runB ? "No run_provenance.json found." : "Pick a run to inspect.")}
                </div>
              )}
            </div>
          ))}
        </div>
        {verify && (
          <div style={{ marginTop: 12, paddingTop: 12, borderTop: "1px dotted var(--line)" }}>
            <div style={{ display: "flex", gap: 14, fontSize: 11, marginBottom: 6 }}>
              <span><Tag color={verify.config_hash_match ? "var(--purple)" : "var(--crimson)"}>config {verify.config_hash_match ? "match" : "diff"}</Tag></span>
              <span><Tag color={verify.data_hash_match ? "var(--purple)" : "var(--crimson)"}>data {verify.data_hash_match ? "match" : "diff"}</Tag></span>
              <span><Tag color={verify.git_commit_match ? "var(--purple)" : "var(--amber)"}>git {verify.git_commit_match ? "match" : "diff"}</Tag></span>
              <span><Tag color={verify.sidecars_b.n_mismatch === 0 ? "var(--purple)" : "var(--crimson)"}>artifacts {verify.sidecars_b.n_match}/{verify.sidecars_b.checked.length}</Tag></span>
            </div>
            {Object.keys(verify.env_diffs).length > 0 && (
              <div style={{ fontSize: 11 }}>
                <div style={{ color: "var(--muted)", marginBottom: 4 }}>Environment differences:</div>
                {Object.entries(verify.env_diffs).slice(0, 8).map(([k, v]) => (
                  <div key={k} className="mono">
                    {k}: {v.a || "—"} → {v.b || "—"}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </Card>

      {diff && (
        <>
          <Card title="Configuration diff" subtitle={`${diff.config.changes.length} differences`}>
            {diff.config.changes.length === 0 ? (
              <div style={{ fontSize: 11, color: "var(--muted)" }}>Configurations are identical.</div>
            ) : (
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                  <thead>
                    <tr style={{ background: "#fdf6e9" }}>
                      {["Path", "Kind", "A", "B"].map((h) => (
                        <th key={h} style={th}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {diff.config.changes.map((c, i) => (
                      <tr key={i} style={{ borderBottom: "1px dotted var(--line)" }}>
                        <td style={td} className="mono">{c.path}</td>
                        <td style={td}>
                          <Tag color={c.kind === "added" ? "var(--purple)" : c.kind === "removed" ? "var(--crimson)" : "var(--amber)"}>
                            {c.kind}
                          </Tag>
                        </td>
                        <td style={td} className="mono">{formatVal(c.a)}</td>
                        <td style={td} className="mono">{formatVal(c.b)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>

          <Card title="Model performance diff" subtitle={`${diff.models.rows.length} models`}>
            {diff.models.rows.length === 0 ? (
              <div style={{ fontSize: 11, color: "var(--muted)" }}>
                No model performance file found in either run.
              </div>
            ) : (
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                  <thead>
                    <tr style={{ background: "#fdf6e9" }}>
                      {["Model", "R² A", "R² B", "ΔR²", "RMSE A", "RMSE B", "ΔRMSE"].map((h) => (
                        <th key={h} style={th}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {diff.models.rows.map((r) => (
                      <tr key={r.model} style={{ borderBottom: "1px dotted var(--line)" }}>
                        <td style={td}>{r.model}</td>
                        <td style={td} className="mono">{fmtNum(r.r2_a)}</td>
                        <td style={td} className="mono">{fmtNum(r.r2_b)}</td>
                        <td style={{ ...td, color: deltaColor(r.r2_delta, "higher_better") }} className="mono">
                          {fmtDelta(r.r2_delta)}
                        </td>
                        <td style={td} className="mono">{fmtNum(r.rmse_a)}</td>
                        <td style={td} className="mono">{fmtNum(r.rmse_b)}</td>
                        <td style={{ ...td, color: deltaColor(r.rmse_delta, "lower_better") }} className="mono">
                          {fmtDelta(r.rmse_delta)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>

          <Card title="Causal effects diff" subtitle={`${diff.causal.effects.length} effects`}>
            {diff.causal.effects.length === 0 ? (
              <div style={{ fontSize: 11, color: "var(--muted)" }}>
                No causal payload found in either run.
              </div>
            ) : (
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                  <thead>
                    <tr style={{ background: "#fdf6e9" }}>
                      {["Effect", "β A", "CI A", "β B", "CI B", "Δβ"].map((h) => (
                        <th key={h} style={th}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {diff.causal.effects.map((r) => (
                      <tr key={r.effect} style={{ borderBottom: "1px dotted var(--line)" }}>
                        <td style={td}>{r.effect}</td>
                        <td style={td} className="mono">{fmtNum(r.estimate_a)}</td>
                        <td style={td} className="mono" title={JSON.stringify(r.ci_a)}>
                          {fmtCi(r.ci_a)}
                        </td>
                        <td style={td} className="mono">{fmtNum(r.estimate_b)}</td>
                        <td style={td} className="mono" title={JSON.stringify(r.ci_b)}>
                          {fmtCi(r.ci_b)}
                        </td>
                        <td style={{ ...td, color: deltaColor(r.delta, "higher_better"), fontWeight: 600 }} className="mono">
                          {fmtDelta(r.delta)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>

          <Card title="DAG diff" subtitle={`A: ${diff.dag.n_a} edges · B: ${diff.dag.n_b} edges`}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 14 }}>
              <DagBlock title="Added in B" edges={diff.dag.added} color="var(--purple)" />
              <DagBlock title="Removed from A" edges={diff.dag.removed} color="var(--crimson)" />
              <DagBlock title="Probability changed (>0.05)" edges={diff.dag.changed} color="var(--amber)" showDelta />
            </div>
          </Card>
        </>
      )}
    </div>
  );
}

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

function DagBlock({ title, edges, color, showDelta = false }: {
  title: string;
  edges: { from: string; to: string; probability?: number | null; prob_a?: number | null; prob_b?: number | null; delta?: number | null }[];
  color: string;
  showDelta?: boolean;
}) {
  return (
    <div>
      <div className="mono" style={{ fontSize: 9.5, color, letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 6 }}>
        {title} · {edges.length}
      </div>
      {edges.length === 0 ? (
        <div style={{ fontSize: 10, color: "var(--muted)" }}>(none)</div>
      ) : (
        <ul style={{ listStyle: "none", padding: 0, margin: 0, display: "flex", flexDirection: "column", gap: 3 }}>
          {edges.map((e, i) => (
            <li key={i} className="mono" style={{ fontSize: 10, padding: "3px 6px", border: `1px solid ${color}40`, borderRadius: 3, background: "#fff" }}>
              {e.from} → {e.to}
              {showDelta && e.delta != null && (
                <span style={{ float: "right", color }}>
                  {fmtNum(e.prob_a)} → {fmtNum(e.prob_b)} ({fmtDelta(e.delta)})
                </span>
              )}
              {!showDelta && e.probability != null && (
                <span style={{ float: "right", color: "var(--muted)" }}>p={fmtNum(e.probability)}</span>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function fmtNum(v: unknown): string {
  if (v === null || v === undefined) return "—";
  const n = Number(v);
  if (!Number.isFinite(n)) return "—";
  return n.toFixed(3);
}

function fmtDelta(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return (v > 0 ? "+" : "") + v.toFixed(3);
}

function fmtCi(ci: [number | null, number | null]): string {
  const [lo, hi] = ci;
  if (lo == null && hi == null) return "—";
  return `[${fmtNum(lo)}, ${fmtNum(hi)}]`;
}

function formatVal(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

function deltaColor(v: number | null, polarity: "higher_better" | "lower_better"): string {
  if (v == null || !Number.isFinite(v) || Math.abs(v) < 1e-9) return "var(--muted)";
  const better = polarity === "higher_better" ? v > 0 : v < 0;
  return better ? "var(--purple)" : "var(--crimson)";
}

const th: React.CSSProperties = {
  textAlign: "left",
  padding: "6px 8px",
  fontSize: 9.5,
  textTransform: "uppercase",
  letterSpacing: "0.08em",
  color: "var(--muted)",
  borderBottom: "1px solid var(--line)",
};

const td: React.CSSProperties = {
  padding: "6px 8px",
  verticalAlign: "middle",
};
