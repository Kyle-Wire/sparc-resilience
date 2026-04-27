/**
 * Inline upgrade nudge.
 *
 * Drop into any feature surface where the current plan lacks the
 * required entitlement. Renders as a compact card with a
 * "Manage Subscription" link.
 */
import { Btn } from "@/components/ui/DesignSystem";
import { open as openExternal } from "@tauri-apps/plugin-shell";

import { LEMON_UPGRADE_URL } from "@/lib/auth/commerce";

const UPGRADE_URL = LEMON_UPGRADE_URL;

interface Props {
  feature: string;
  description?: string;
}

export default function UpgradePrompt({ feature, description }: Props) {
  return (
    <div
      style={{
        border: "1px dashed var(--line)",
        borderRadius: 6,
        padding: 14,
        background: "#fffaf3",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: 12,
      }}
    >
      <div>
        <div style={{ fontSize: 12.5, fontWeight: 600, color: "var(--ink-2)" }}>
          {feature} requires Pro
        </div>
        {description && (
          <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 2 }}>
            {description}
          </div>
        )}
      </div>
      <Btn primary small onClick={() => void openExternal(UPGRADE_URL)}>
        Upgrade
      </Btn>
    </div>
  );
}
