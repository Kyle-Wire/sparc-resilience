import { createSupabaseServerClient, createSupabaseAdminClient } from "@/lib/supabase/server";
import { ApiKeysClient } from "./parts";

export const dynamic = "force-dynamic";

export interface ApiKeyRow { id: string; provider: string; label: string; key_last4: string; created_at: string; }

export default async function ApiKeysPage() {
  const supabase = await createSupabaseServerClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return null;

  let keys: ApiKeyRow[] = [];
  try {
    const admin = createSupabaseAdminClient();
    const { data } = await admin.from("api_keys").select("id,provider,label,key_last4,created_at").eq("user_id", user.id).order("created_at", { ascending: false });
    keys = data ?? [];
  } catch { /* dev */ }

  return (
    <div style={{ display: "grid", gap: 18 }}>
      <div className="card" style={{ padding: 24 }}>
        <div className="kicker">byok · bring your own key</div>
        <h2 style={{ fontSize: 24, fontWeight: 800, letterSpacing: "-0.02em", margin: "8px 0 6px" }}>LLM API keys</h2>
        <p style={{ color: "var(--ink-2)", fontSize: 14, lineHeight: 1.55, margin: 0 }}>
          Keys are encrypted with Supabase Vault and only decrypted by your desktop app at runtime. SPARC never proxies your prompts.
        </p>
      </div>
      <ApiKeysClient initial={keys} />
    </div>
  );
}
