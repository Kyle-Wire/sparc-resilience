// How reliable the quick estimate is for a lever (SPEC §7.5): good (patch pass rate ≥ 0.9 and
// uniform relative error ≤ 0.35), rough, or none (no quick estimate). A dot and the plain words
// (glossary quickTrust, the same words as the lever cards' trust dots), with an ⓘ that says what it
// means and, when known, how far off the quick estimate can be citywide. Text carries the status,
// never colour alone; no native title.
import type { EmulatorTrust } from "../../../api/lab";
import { InfoTip } from "../../../components/ui/InfoTip";
import { term } from "../../../content/glossary";
import { fmtPct } from "../../../theme/format";
import { TRUST_TIP, TRUST_WORDS } from "../model/words";

/** The ⓘ's text: what the dot means, then the citywide error when the run reports it. */
export function trustBadgeTip(trust: EmulatorTrust, relErr?: number | null): string {
  const err = relErr !== null && relErr !== undefined && Number.isFinite(relErr) ? ` For a change of the whole city it can be off by about ${fmtPct(relErr)} (${term("uniform_rel_error").label.toLowerCase()}).` : "";
  return trust === "none" ? `There is no quick estimate for this lever: only the full model can work it out. ${TRUST_TIP}` : `${TRUST_TIP}${err}`;
}

export function TrustBadge({ trust, relErr }: { trust: EmulatorTrust; relErr?: number | null }) {
  const w = TRUST_WORDS[trust] ?? TRUST_WORDS.none;
  return (
    <span className="lab-trust-badge" data-trust={trust}>
      <span className="lab-trust-dot" data-tone={w.tone} aria-hidden="true" />
      {w.label}
      <InfoTip label="quick-estimate reliability" text={trustBadgeTip(trust, relErr)} technical={`Emulator trust: ${trust}${relErr !== null && relErr !== undefined && Number.isFinite(relErr) ? `, ${term("uniform_rel_error").technical} ${fmtPct(relErr)}` : ""}`} />
    </span>
  );
}
