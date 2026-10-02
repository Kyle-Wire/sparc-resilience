// Emulator trust per lever (SPEC §7.5): good (patch pass rate ≥ 0.9 and uniform relative error
// ≤ 0.35), rough, or none (no emulator). Text carries the status, never colour alone.
import type { EmulatorTrust } from "../../../api/lab";
import { Pill } from "../../../components/ui/Pill";
import { fmtPct } from "../../../theme/format";

export function TrustBadge({ trust, relErr }: { trust: EmulatorTrust; relErr?: number | null }) {
  const err = relErr !== null && relErr !== undefined && Number.isFinite(relErr) ? ` (uniform rel. error ${fmtPct(relErr)})` : "";
  if (trust === "good") return <Pill tone="good" icon="check" title={`The preview tracks the exact engine well for this lever${err}.`}>preview good</Pill>;
  if (trust === "rough") return <Pill tone="warn" icon="alert" title={`The preview is approximate for this lever${err}; run exact before deciding.`}>preview rough</Pill>;
  return <Pill icon="minus" title="No emulator for this lever: exact only.">no preview</Pill>;
}
