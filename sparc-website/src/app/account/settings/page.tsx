import { createSupabaseServerClient } from "@/lib/supabase/server";

export const dynamic = "force-dynamic";

export default async function SettingsPage() {
  const supabase = await createSupabaseServerClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return null;

  return (
    <div style={{ display: "grid", gap: 18 }}>
      <form className="card" style={{ padding: 24 }} action="/api/account/profile" method="post">
        <div className="kicker">profile</div>
        <h2 style={{ fontSize: 22, fontWeight: 700, margin: "8px 0 16px" }}>Display name</h2>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
          <div><label className="label">Email</label><input className="input mono" value={user.email ?? ""} readOnly /></div>
          <div><label className="label">Full name</label><input className="input" name="full_name" defaultValue={user.user_metadata?.full_name ?? ""} placeholder="Naomi Schaal" /></div>
          <div style={{ gridColumn: "1 / -1" }}><label className="label">Organization</label><input className="input" name="organization" defaultValue={user.user_metadata?.organization ?? ""} placeholder="UC Berkeley · Geography" /></div>
        </div>
        <div className="row" style={{ marginTop: 16, justifyContent: "flex-end" }}><button type="submit" className="btn btn-primary">Save</button></div>
      </form>

      <div className="card" style={{ padding: 24 }}>
        <div className="kicker">danger zone</div>
        <h2 style={{ fontSize: 20, fontWeight: 700, margin: "8px 0 6px" }}>Delete account</h2>
        <p style={{ color: "var(--ink-2)", fontSize: 14, lineHeight: 1.55, margin: "0 0 14px" }}>
          Permanently delete your SPARC Labs account and all associated data. Active subscriptions must be cancelled first.
        </p>
        <a href="mailto:SparcUrbanLabs@gmail.com?subject=Delete%20my%20account" className="btn">Request deletion</a>
      </div>
    </div>
  );
}
