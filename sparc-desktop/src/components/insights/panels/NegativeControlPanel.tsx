/**
 * NegativeControlPanel — permutation negative-control test for the
 * active CATE/treatment variable. Tied to linked-selection.variable.
 */
import { Panel, PanelEmpty, Pill, Stat, StatGrid } from "@/components/ui/DesignSystem";
import { useLinkedSelection } from "@/hooks/useLinkedSelection";
import { useResult } from "@/hooks/useResult";
import { getCausalNegativeControl, type NegativeControlResponse } from "@/lib/api";

export default function NegativeControlPanel() {
  const linked = useLinkedSelection();
  const variable = linked.selection.variable;
  const { data, error } = useResult<NegativeControlResponse>(
    variable ? `causal:neg_control:${variable}` : null,
    () => getCausalNegativeControl(variable!, 1000),
  );

  if (!variable) {
    return (
      <Panel title="Negative control" subtitle="permutation test">
        <PanelEmpty
          reason="Pick a treatment variable"
          hint="Click a CATE row or pick from Dose-response to drive this test."
        />
      </Panel>
    );
  }
  if (error || !data) {
    return (
      <Panel title="Negative control" subtitle={`variable · ${variable}`}>
        <PanelEmpty reason={error ?? "Loading…"} />
      </Panel>
    );
  }

  return (
    <Panel
      title="Negative control"
      subtitle={`variable · ${variable}`}
      actions={
        <Pill color={data.passed ? "var(--green, #2f7d32)" : "var(--crimson, #b91c1c)"}>
          {data.passed ? "passed" : "failed"}
        </Pill>
      }
    >
      <StatGrid>
        <Stat label="z-score" value={data.z_score.toFixed(2)} />
        <Stat label="p-value" value={data.p_value.toExponential(2)} />
        <Stat label="observed" value={data.mean_observed.toFixed(3)} />
        <Stat label="null mean ± σ" value={`${data.mean_null.toFixed(3)} ± ${data.std_null.toFixed(3)}`} />
        <Stat label="permutations" value={data.n_permutations.toString()} />
        <Stat label="n samples" value={data.n.toString()} />
      </StatGrid>
      <p style={{ marginTop: 12, fontSize: 12, color: "var(--ink-2)" }}>{data.interpretation}</p>
    </Panel>
  );
}
