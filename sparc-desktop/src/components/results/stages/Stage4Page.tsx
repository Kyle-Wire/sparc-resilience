/**
 * Stage 4 — Decision
 * Wonder → Structure → Proof arc:
 *   Wonder:    Scenario comparison side-by-side hero
 *   Structure: Headline recommendation · Scenario cards · Budget frontier · Equity map
 *   Proof:     Uncertainty bands on projections
 */
import NarrativeCallout from "../NarrativeCallout";
import TechnicalDisclosure from "../TechnicalDisclosure";

const ACCENT = "var(--s4, #f0b632)";

export default function Stage4Page() {
  return (
    <article aria-label="Stage 4: Decision">
      <section
        style={{
          display: "grid",
          gridTemplateColumns: "38% 1fr",
          gap: 32,
          padding: "40px 24px",
        }}
      >
        <NarrativeCallout
          headline="Best intervention"
          fallback="The model evaluates every intervention scenario against the causal estimates and ranks them by expected impact, equity, and cost-effectiveness."
          accentColor={ACCENT}
        />
        <div data-testid="s4-scenarios-placeholder" style={{ minHeight: 200 }} />
      </section>

      <section
        style={{
          display: "grid",
          gridTemplateColumns: "38% 1fr",
          gap: 32,
          padding: "40px 24px",
          background: "var(--paper-2, #ede8dd)",
        }}
      >
        <NarrativeCallout
          headline="Equity & budget frontier"
          fallback="The budget-frontier chart shows the cost/benefit tradeoff across scenarios, and the equity map reveals the geographic distribution of intervention benefits."
          accentColor={ACCENT}
        />
        <div data-testid="s4-frontier-placeholder" style={{ minHeight: 200 }} />
      </section>

      <div style={{ padding: "0 24px 40px" }}>
        <TechnicalDisclosure label="Uncertainty bands on projections">
          <div data-testid="s4-tech-content">
            <p>Credible interval bands on all scenario projections will be rendered here.</p>
          </div>
        </TechnicalDisclosure>
      </div>
    </article>
  );
}
