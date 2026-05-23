/**
 * Stage 0 — Geometry
 *
 * Wonder → Structure → Proof arc:
 *   Wonder:    Hero — animated spatial scatter, key stats
 *   Structure: Correlogram · Effective Ranges · Anisotropy
 *   Proof:     TechnicalDisclosure — cross-correlogram heatmap + Matérn fit
 */
import NarrativeCallout from "../NarrativeCallout";
import TechnicalDisclosure from "../TechnicalDisclosure";

const ACCENT = "var(--s0, #602468)";

export default function Stage0Page() {
  return (
    <article aria-label="Stage 0: Geometry">
      {/* ── Figure 1: Correlogram ────────────────────────────────────── */}
      <section
        style={{
          display: "grid",
          gridTemplateColumns: "38% 1fr",
          gap: 32,
          padding: "40px 24px",
        }}
      >
        <NarrativeCallout
          headline="Spatial autocorrelation"
          fallback="The correlogram shows how similarity between locations decays with distance — the foundation of every kernel bandwidth choice the model makes."
          accentColor={ACCENT}
        />
        <div data-testid="s0-correlogram-placeholder" style={{ minHeight: 200 }} />
      </section>

      {/* ── Figure 2: Effective Ranges ──────────────────────────────── */}
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
          headline="Effective ranges"
          fallback="Each variable's effective range is the lag distance at which autocorrelation drops below 0.1 — the natural neighbourhood radius for that predictor."
          accentColor={ACCENT}
        />
        <div data-testid="s0-ranges-placeholder" style={{ minHeight: 200 }} />
      </section>

      {/* ── Figure 3: Anisotropy ────────────────────────────────────── */}
      <section
        style={{
          display: "grid",
          gridTemplateColumns: "38% 1fr",
          gap: 32,
          padding: "40px 24px",
        }}
      >
        <NarrativeCallout
          headline="Directional anisotropy"
          fallback="The anisotropy diagram reveals whether spatial patterns vary by compass direction — indicating dominant flow paths, prevailing winds, or infrastructure corridors."
          accentColor={ACCENT}
        />
        <div data-testid="s0-anisotropy-placeholder" style={{ minHeight: 200 }} />
      </section>

      {/* ── Technical Disclosure ────────────────────────────────────── */}
      <div style={{ padding: "0 24px 40px" }}>
        <TechnicalDisclosure label="Correlogram & Matérn fit details">
          <div data-testid="s0-tech-content">
            <p>Cross-correlogram heatmap and Matérn kernel fit will be rendered here.</p>
          </div>
        </TechnicalDisclosure>
      </div>
    </article>
  );
}
