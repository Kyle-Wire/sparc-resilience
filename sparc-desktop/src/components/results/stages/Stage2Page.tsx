/**
 * Stage 2 — Prediction
 * Wonder → Structure → Proof arc:
 *   Wonder:    Predictions map with uncertainty overlay hero
 *   Structure: OOF performance per fold · R²/RMSE · Residual map
 *   Proof:     Model comparison (if applicable)
 */
import NarrativeCallout from "../NarrativeCallout";
import TechnicalDisclosure from "../TechnicalDisclosure";

const ACCENT = "var(--s2, #e73c25)";

export default function Stage2Page() {
  return (
    <article aria-label="Stage 2: Prediction">
      <section
        style={{
          display: "grid",
          gridTemplateColumns: "38% 1fr",
          gap: 32,
          padding: "40px 24px",
        }}
      >
        <NarrativeCallout
          headline="Spatial predictions"
          fallback="The geographically weighted random forest produces a continuous prediction surface across the entire study area, capturing local variation in the outcome."
          accentColor={ACCENT}
        />
        <div data-testid="s2-predictions-placeholder" style={{ minHeight: 200 }} />
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
          headline="Out-of-fold performance"
          fallback="Cross-validation folds show whether predictive accuracy is consistent across the geography — a spatially biased model would show sharp drops in held-out regions."
          accentColor={ACCENT}
        />
        <div data-testid="s2-oof-placeholder" style={{ minHeight: 200 }} />
      </section>

      <div style={{ padding: "0 24px 40px" }}>
        <TechnicalDisclosure label="Model comparison & residuals">
          <div data-testid="s2-tech-content">
            <p>Residual map and optional model comparison cards will be rendered here.</p>
          </div>
        </TechnicalDisclosure>
      </div>
    </article>
  );
}
