import Link from "next/link";
import { createSupabaseServerClient, createSupabaseAdminClient } from "@/lib/supabase/server";

export const dynamic = "force-dynamic";

export default async function AccountOverview() {
  const supabase = await createSupabaseServerClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return null;

  let sub: { tier: string | null; status: string | null; renews_at: string | null } | null = null;
  try {
    const admin = createSupabaseAdminClient();
    const { data } = await admin
      .from("subscriptions")
      .select("tier,status,renews_at")
      .eq("user_id", user.id)
      .order("updated_at", { ascending: false })
      .limit(1)
      .maybeSingle();
    sub = data;
  } catch { /* table may not exist yet in dev */ }

  return (
    <div style={{ display: "grid", gap: 18 }}>
      <div className="card" style={{ padding: 24 }}>
        <div className="kicker">welcome back</div>
        <h2 style={{ fontSize: 24, fontWeight: 800, letterSpacing: "-0.02em", margin: "8px 0 0" }}>
          {user.user_metadata?.full_name ?? user.email}
        </h2>
        <p style={{ color: "var(--ink-2)", margin: "8px 0 0", fontSize: 14 }}>
          Manage your subscription, license keys, and the desktop client from this dashboard.
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 14 }}>
        <Link href="/account/subscription" className="card" style={{ padding: 22, textDecoration: "none", color: "inherit" }}>
          <div className="kicker">subscription</div>
          <div style={{ fontSize: 22, fontWeight: 700, marginTop: 6, textTransform: "capitalize" }}>{sub?.tier ?? "Wander (free)"}</div>
          <div className="mono" style={{ fontSize: 11, color: "var(--muted)", marginTop: 4 }}>{sub?.status ?? "no active paid plan"}</div>
        </Link>
        <Link href="/account/downloads" className="card" style={{ padding: 22, textDecoration: "none", color: "inherit" }}>
          <div className="kicker">desktop client</div>
          <div style={{ fontSize: 22, fontWeight: 700, marginTop: 6 }}>Download SPARC</div>
          <div className="mono" style={{ fontSize: 11, color: "var(--muted)", marginTop: 4 }}>macOS · Windows · Linux</div>
        </Link>
        <Link href="/account/api-keys" className="card" style={{ padding: 22, textDecoration: "none", color: "inherit" }}>
          <div className="kicker">byok api keys</div>
          <div style={{ fontSize: 22, fontWeight: 700, marginTop: 6 }}>Manage</div>
          <div className="mono" style={{ fontSize: 11, color: "var(--muted)", marginTop: 4 }}>Anthropic · OpenAI · Mistral</div>
        </Link>
        <Link href="/account/usage" className="card" style={{ padding: 22, textDecoration: "none", color: "inherit" }}>
          <div className="kicker">usage</div>
          <div style={{ fontSize: 22, fontWeight: 700, marginTop: 6 }}>This month</div>
          <div className="mono" style={{ fontSize: 11, color: "var(--muted)", marginTop: 4 }}>runs · scenarios · tokens</div>
        </Link>
      </div>
    </div>
  );
}
