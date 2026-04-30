import { createSupabaseServerClient, createSupabaseAdminClient } from "@/lib/supabase/server";

export const dynamic = "force-dynamic";

interface Seat { id: string; email: string; status: string; invited_at: string | null; }

export default async function SeatsPage() {
  const supabase = await createSupabaseServerClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return null;

  let seats: Seat[] = [];
  let max = 0;
  try {
    const admin = createSupabaseAdminClient();
    const { data: subs } = await admin.from("subscriptions").select("tier,seats_allowed").eq("user_id", user.id).order("updated_at", { ascending: false }).limit(1).maybeSingle();
    max = subs?.seats_allowed ?? 0;
    const { data } = await admin.from("seats").select("id,email,status,invited_at").eq("owner_id", user.id);
    seats = data ?? [];
  } catch { /* dev */ }

  return (
    <div style={{ display: "grid", gap: 18 }}>
      <div className="card" style={{ padding: 24 }}>
        <div className="kicker">team seats</div>
        <h2 style={{ fontSize: 24, fontWeight: 800, letterSpacing: "-0.02em", margin: "8px 0 6px" }}>{seats.length} of {max || "—"} seats used</h2>
        <p style={{ color: "var(--ink-2)", fontSize: 14, margin: 0, lineHeight: 1.55 }}>
          Invite teammates to your Converge workspace. Each seat gets its own license key, usage allotment, and project access.
        </p>
      </div>

      <form className="card" style={{ padding: 22 }} action="/api/seats/invite" method="post">
        <div className="kicker">invite teammate</div>
        <div className="row" style={{ marginTop: 12, gap: 8, alignItems: "flex-end" }}>
          <div style={{ flex: 1 }}>
            <label className="label">Email</label>
            <input className="input" type="email" name="email" required placeholder="teammate@university.edu" />
          </div>
          <button type="submit" className="btn btn-accent" disabled={max === 0}>Send invite</button>
        </div>
        {max === 0 && <p className="mono" style={{ marginTop: 10, fontSize: 11, color: "var(--muted)" }}>▸ upgrade to converge to invite teammates</p>}
      </form>

      <div className="card" style={{ padding: 0, overflow: "hidden" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
          <thead><tr style={{ background: "var(--paper-2)", borderBottom: "1px solid var(--line)" }}>
            <th style={{ textAlign: "left", padding: "12px 18px", fontFamily: "var(--font-mono)", fontSize: 11, letterSpacing: "0.1em", textTransform: "uppercase", color: "var(--muted)" }}>email</th>
            <th style={{ textAlign: "left", padding: "12px 18px", fontFamily: "var(--font-mono)", fontSize: 11, letterSpacing: "0.1em", textTransform: "uppercase", color: "var(--muted)" }}>status</th>
            <th style={{ textAlign: "left", padding: "12px 18px", fontFamily: "var(--font-mono)", fontSize: 11, letterSpacing: "0.1em", textTransform: "uppercase", color: "var(--muted)" }}>invited</th>
          </tr></thead>
          <tbody>
            {seats.length === 0 ? (
              <tr><td colSpan={3} style={{ padding: 22, textAlign: "center", color: "var(--muted)" }}>No seats yet.</td></tr>
            ) : seats.map((s) => (
              <tr key={s.id} style={{ borderBottom: "1px dotted var(--line)" }}>
                <td style={{ padding: "12px 18px" }}>{s.email}</td>
                <td style={{ padding: "12px 18px" }}><span className="pill mono">{s.status}</span></td>
                <td style={{ padding: "12px 18px" }} className="mono">{s.invited_at ? new Date(s.invited_at).toLocaleDateString() : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
