import Link from "next/link";
import { createSupabaseServerClient, createSupabaseAdminClient } from "@/lib/supabase/server";

export const dynamic = "force-dynamic";

interface Sub {
  tier: string | null;
  status: string | null;
  renews_at: string | null;
  ends_at: string | null;
  trial_ends_at: string | null;
  customer_portal_url: string | null;
}

export default async function SubscriptionPage({ searchParams }: { searchParams: Promise<{ welcome?: string; error?: string }> }) {
  const sp = await searchParams;
  const supabase = await createSupabaseServerClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return null;

  let sub: Sub | null = null;
  try {
    const admin = createSupabaseAdminClient();
    const { data } = await admin
      .from("subscriptions")
      .select("tier,status,renews_at,ends_at,trial_ends_at,customer_portal_url")
      .eq("user_id", user.id)
      .order("updated_at", { ascending: false })
      .limit(1)
      .maybeSingle();
    sub = data;
  } catch { /* dev */ }

  return (
    <div style={{ display: "grid", gap: 18 }}>
      {sp.welcome && (
        <div className="card" style={{ padding: 18, borderLeft: "3px solid var(--purple)" }}>
          <strong>Subscription confirmed.</strong> Thanks for supporting SPARC. It can take a few seconds for the webhook to land — refresh if your tier hasn&apos;t updated.
        </div>
      )}
      {sp.error === "no_portal" && (
        <div className="card" style={{ padding: 18, borderLeft: "3px solid var(--crimson)" }}>
          We couldn&apos;t find a customer portal for your account yet. If you just subscribed, give the webhook a moment.
        </div>
      )}

      <div className="card" style={{ padding: 24 }}>
        <div className="kicker">current plan</div>
        <div className="row" style={{ alignItems: "baseline", gap: 14, marginTop: 6 }}>
          <h2 style={{ fontSize: 28, fontWeight: 800, letterSpacing: "-0.02em", margin: 0, textTransform: "capitalize" }}>{sub?.tier ?? "Wander (free)"}</h2>
          {sub?.status && <span className="pill mono">{sub.status}</span>}
        </div>
        <dl className="dl" style={{ marginTop: 18 }}>
          <dt>renews</dt><dd className="mono">{sub?.renews_at ?? "—"}</dd>
          <dt>ends</dt><dd className="mono">{sub?.ends_at ?? "—"}</dd>
          <dt>trial ends</dt><dd className="mono">{sub?.trial_ends_at ?? "—"}</dd>
        </dl>
        <div className="row" style={{ marginTop: 22, gap: 10, flexWrap: "wrap" }}>
          {sub?.customer_portal_url ? (
            <Link className="btn btn-primary" href="/api/portal">Manage billing & invoices →</Link>
          ) : (
            <Link className="btn btn-accent" href="/pricing">Upgrade plan →</Link>
          )}
          <Link className="btn" href="/contact?tier=transcend">Talk about Transcend</Link>
        </div>
      </div>

      <div className="card dashed" style={{ padding: 22 }}>
        <div className="kicker">need to change something?</div>
        <p style={{ fontSize: 14, color: "var(--ink-2)", margin: "8px 0 0", lineHeight: 1.55 }}>
          Card updates, invoices, tax IDs, and cancellations are handled in the Lemon Squeezy customer portal.
          Click <em>Manage billing</em> above to open a signed link to your account.
        </p>
      </div>
    </div>
  );
}
