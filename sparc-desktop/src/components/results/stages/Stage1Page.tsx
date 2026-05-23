/**
 * Stage 1 — Selection
 * Wonder → Structure → Proof arc:
 *   Wonder:    Variable-importance choropleth grid hero
 *   Structure: Importance ranking · Dropped variables · Stability
 *   Proof:     Bootstrap agreement · VIF table
 */
import NarrativeCallout from "../NarrativeCallout";
import TechnicalDisclosure from "../TechnicalDisclosure";

const ACCENT = "var(--s1, #e94d9b)";

export default function Stage1Page() {
  return (
    <article aria-label="Stage 1: Selection">
      <section
        style={{
          display: "grid",
          gridTemplateColumns: "38% 1fr",
          gap: 32,
          padding: "40px 24px",
        }}
      >
        <NarrativeCallout
          headline="Variable importance"
          fallback="GWEN ranks each candidate predictor by spatial signal strength and removes redundant variables so that only the most informative features enter the model."
          accentColor={ACCENT}
        />
        <div data-testid="s1-importance-placeholder" style={{ minHeight: 200 }} />
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
          headline="Dropped variables"
          fallback="Variables excluded due to multicollinearity or low spatial signal are listed here so you can audit the selection process."
          accentColor={ACCENT}
        />
        <div data-testid="s1-dropped-placeholder" style={{ minHeight: 200 }} />
      </section>

      <div style={{ padding: "0 24px 40px" }}>
        <TechnicalDisclosure label="Bootstrap stability & VIF table">
          <div data-testid="s1-tech-content">
            <p>Bootstrap agreement across folds and variance inflation factors will be rendered here.</p>
          </div>
        </TechnicalDisclosure>
      </div>
    </article>
  );
}
