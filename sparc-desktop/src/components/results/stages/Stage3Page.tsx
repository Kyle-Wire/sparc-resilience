/**
 * Stage 3 — Causation
 * Wonder → Structure → Proof arc:
 *   Wonder:    CATE spatial heterogeneity map hero
 *   Structure: Dose-response curve · ATE + credible interval · CATE map
 *   Proof:     Negative-control · Sensitivity · ELBO/KL divergence
 */
import NarrativeCallout from "../NarrativeCallout";
import TechnicalDisclosure from "../TechnicalDisclosure";

const ACCENT = "var(--s3, #e79024)";

export default function Stage3Page() {
  return (
    <article aria-label="Stage 3: Causation">
      <section
        style={{
          display: "grid",
          gridTemplateColumns: "38% 1fr",
          gap: 32,
          padding: "40px 24px",
        }}
      >
        <NarrativeCallout
          headline="Dose-response"
          fallback="The dose-response curve shows how the outcome changes as the treatment intensity increases — holding all confounders constant via the causal model."
          accentColor={ACCENT}
        />
        <div data-testid="s3-dose-response-placeholder" style={{ minHeight: 200 }} />
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
          headline="Heterogeneous effects (CATE)"
          fallback="The average treatment effect may hide strong local variation. This map shows where the intervention will be most and least effective across the study area."
          accentColor={ACCENT}
        />
        <div data-testid="s3-cate-placeholder" style={{ minHeight: 200 }} />
      </section>

      <div style={{ padding: "0 24px 40px" }}>
        <TechnicalDisclosure label="Negative-control, sensitivity & convergence">
          <div data-testid="s3-tech-content">
            <p>
              Negative-control test results, sensitivity analysis, and Bayesian convergence
              diagnostics (ELBO / KL divergence) will be rendered here.
            </p>
          </div>
        </TechnicalDisclosure>
      </div>
    </article>
  );
}
