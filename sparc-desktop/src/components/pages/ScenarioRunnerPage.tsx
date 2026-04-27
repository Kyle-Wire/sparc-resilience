import { useEffect, useMemo, useState, useCallback } from "react";
import { SectionHeader, Card, Tag, Btn, Stat, StatGrid } from "@/components/ui/DesignSystem";
import { useNotification } from "@/hooks/useNotifications";
import { useManifest } from "@/hooks/useManifest";
import { getConfig, runScenarios, getScenarioDetail, parseMissingArtifact } from "@/lib/api";
import { useEntitlements } from "@/hooks/useEntitlements";
import UpgradePrompt from "@/components/auth/UpgradePrompt";

type ScenarioRow = {
  name: string;
  variable?: string;
  increments?: number[];
  interventions?: Record<string, number>;
  delta?: number;
  status?: string;
};

type ConfigShape = {
  scenarios?: any[];
  joint_scenarios?: any[];
  interaction_scenarios?: any[];
  auto_run_scenarios_at_stage_4?: boolean;
};

interface Props {
  onNavigate?: (page: string) => void;
}

/**
 * Stage-4 scenario launcher.
 *
 * Surfaces all configured scenarios and lets the user trigger
 * `/scenarios/run` synchronously without leaving the desktop app.
 * Results land in the same registry consumed by the Results page,
 * so the "Open results" button just navigates there.
 */
export default function ScenarioRunnerPage({ onNavigate }: Props) {
  const ent = useEntitlements();
  const { notify } = useNotification();
  const manifest = useManifest(true);
  const [config, setConfig] = useState<ConfigShape | null>(null);
  const [scenarios, setScenarios] = useState<ScenarioRow[]>([]);
  const [running, setRunning] = useState(false);
  const [lastRun, setLastRun] = useState<{
    nScenarios: number;
    summaryRows: number;
    mode?: string;
    conservation?: number;
    timestamp: string;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Load project + currently-computed scenarios
  useEffect(() => {
    getConfig()
      .then((c) => setConfig(c as any))
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
    getScenarioDetail()
      .then((d: any) => {
        const rows = d?.scenarios ?? d?.summary;
        if (Array.isArray(rows)) {
          setScenarios(
            rows.map((sc: any) => ({
              name: sc.name,
              variable: sc.variable,
              interventions: sc.interventions,
              delta: sc.delta ?? sc.mean_delta,
              status: "computed",
            })),
          );
        }
      })
      .catch(() => {});
  }, []);

  const configured: ScenarioRow[] = useMemo(() => {
    if (!config) return [];
    const out: ScenarioRow[] = [];
    for (const s of config.scenarios ?? []) {
      out.push({
        name: s.name,
        variable: s.variable,
        increments: s.increments,
        status: "configured",
      });
    }
    for (const s of config.joint_scenarios ?? []) {
      out.push({
        name: s.name,
        interventions: Object.fromEntries(
          (s.interventions ?? []).map((iv: any) => [iv.variable, iv.increment]),
        ),
        status: "configured (joint)",
      });
    }
    for (const s of config.interaction_scenarios ?? []) {
      out.push({ name: s.name, status: "configured (interaction)" });
    }
    return out;
  }, [config]);

  const handleRun = useCallback(async () => {
    setRunning(true);
    setError(null);
    notify("info", "Launching scenarios — this can take a minute…");
    try {
      const result = await runScenarios();
      setLastRun({
        nScenarios: result.n_scenarios,
        summaryRows: result.summary_rows,
        mode: (result as any).scenario_mode,
        conservation: (result as any).conservation_violations,
        timestamp: new Date().toLocaleTimeString(),
      });
      notify("success", `Computed ${result.n_scenarios} scenarios`);
      // Refresh manifest + computed list
      await manifest.rescan().catch(() => {});
      const detail: any = await getScenarioDetail().catch(() => null);
      const rows = detail?.scenarios ?? detail?.summary;
      if (Array.isArray(rows)) {
        setScenarios(
          rows.map((sc: any) => ({
            name: sc.name,
            variable: sc.variable,
            interventions: sc.interventions,
            delta: sc.delta ?? sc.mean_delta,
            status: "computed",
          })),
        );
      }
    } catch (e) {
      const missing = parseMissingArtifact(e);
      const msg = missing?.hint ?? (e instanceof Error ? e.message : String(e));
      setError(msg);
      notify("error", msg);
    } finally {
      setRunning(false);
    }
  }, [manifest, notify]);

  const stage4 = manifest.stage(4);
  const hasResults = (stage4?.artifacts && Object.keys(stage4.artifacts).length > 0) || lastRun !== null;
  const autoRun = config?.auto_run_scenarios_at_stage_4 !== false;

  return (
    <div>
      <SectionHeader
        kicker="08 · scenario runner"
        label="Scenario Runner"
        right={
          <div style={{ display: "flex", gap: 8 }}>
            <Btn small onClick={() => onNavigate?.("Scenarios")}>Edit scenarios</Btn>
            <Btn primary small onClick={handleRun} disabled={!ent.scenarioRunner || running || configured.length === 0}>
              {running ? "Running…" : "Run all scenarios"}
            </Btn>
          </div>
        }
      />

      {!ent.scenarioRunner && (
        <UpgradePrompt
          feature="Scenario Runner"
          description="What-if scenario sweeps, joint scenarios, and counterfactual rollups are available on Pro."
        />
      )}

      <StatGrid>
        <Stat label="Configured" value={String(configured.length)} tint="var(--ink)" />
        <Stat label="Computed" value={String(scenarios.length)} tint="var(--purple)" />
        <Stat
          label="Auto-run @ Stage 4"
          value={autoRun ? "on" : "off"}
          tint={autoRun ? "var(--ink)" : "var(--amber)"}
        />
        <Stat
          label="Last run"
          value={lastRun?.timestamp ?? (hasResults ? "previous session" : "—")}
          tint="var(--crimson)"
        />
      </StatGrid>

      {error && (
        <Card title="Error" subtitle="from /scenarios/run">
          <div style={{ fontSize: 12.5, color: "var(--crimson)", lineHeight: 1.55 }}>{error}</div>
        </Card>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
        <Card title="Configured scenarios" subtitle="from project.yml">
          {configured.length === 0 ? (
            <div style={{ fontSize: 12.5, color: "var(--muted)", padding: "10px 4px" }}>
              No scenarios defined. Open the <strong>Scenarios</strong> page to add some, or edit
              project.yml directly under the <code>scenarios:</code> key.
            </div>
          ) : (
            configured.map((s, i) => (
              <div
                key={`${s.name}-${i}`}
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr auto",
                  alignItems: "center",
                  gap: 10,
                  padding: "10px 4px",
                  borderTop: i > 0 ? "1px dashed var(--line)" : "none",
                }}
              >
                <div>
                  <div style={{ fontSize: 12.5, fontWeight: 600 }}>{s.name}</div>
                  <div className="mono" style={{ fontSize: 10, color: "var(--muted)", marginTop: 2 }}>
                    {s.variable
                      ? `${s.variable} ${s.increments ? `× [${s.increments.join(", ")}]` : ""}`
                      : s.interventions
                      ? Object.entries(s.interventions)
                          .map(([k, v]) => `${k}: ${v > 0 ? "+" : ""}${v}`)
                          .join(", ")
                      : "—"}
                  </div>
                </div>
                <Tag color="var(--muted)">{s.status}</Tag>
              </div>
            ))
          )}
        </Card>

        <Card title="Computed deltas" subtitle="from /results/scenarios/detail">
          {scenarios.length === 0 ? (
            <div style={{ fontSize: 12.5, color: "var(--muted)", padding: "10px 4px" }}>
              No computed scenarios yet. Click <strong>Run all scenarios</strong> to launch.
            </div>
          ) : (
            scenarios.map((sc, i) => (
              <div
                key={`${sc.name}-${i}`}
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr auto",
                  alignItems: "center",
                  gap: 10,
                  padding: "10px 4px",
                  borderTop: i > 0 ? "1px dashed var(--line)" : "none",
                }}
              >
                <div style={{ fontSize: 12.5, fontWeight: 600 }}>{sc.name}</div>
                <span
                  className="mono"
                  style={{
                    fontSize: 13,
                    fontWeight: 700,
                    color:
                      (sc.delta ?? 0) < 0
                        ? "var(--purple)"
                        : (sc.delta ?? 0) > 0
                        ? "var(--crimson)"
                        : "var(--muted)",
                  }}
                >
                  {sc.delta === undefined || sc.delta === 0
                    ? "—"
                    : `${sc.delta > 0 ? "+" : ""}${sc.delta.toFixed(2)}`}
                </span>
              </div>
            ))
          )}
          {hasResults && (
            <div style={{ marginTop: 10, paddingTop: 10, borderTop: "1px dashed var(--line)" }}>
              <Btn small onClick={() => onNavigate?.("Results")}>Open Results →</Btn>
            </div>
          )}
        </Card>
      </div>

      {lastRun && (
        <Card title="Last run summary" subtitle={lastRun.timestamp}>
          <div className="mono" style={{ fontSize: 11.5, color: "var(--muted)", lineHeight: 1.7 }}>
            scenarios: {lastRun.nScenarios}
            <br />
            summary rows: {lastRun.summaryRows}
            {lastRun.mode && (
              <>
                <br />
                mode: {lastRun.mode}
              </>
            )}
            {lastRun.conservation !== undefined && (
              <>
                <br />
                conservation violations: {lastRun.conservation}
              </>
            )}
          </div>
        </Card>
      )}
    </div>
  );
}
