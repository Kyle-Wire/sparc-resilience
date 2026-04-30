import { createSupabaseServerClient, createSupabaseAdminClient } from "@/lib/supabase/server";

export const dynamic = "force-dynamic";

interface Device { id: string; device_name: string; os: string; last_seen: string; created_at: string; }

export default async function DevicesPage() {
  const supabase = await createSupabaseServerClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return null;

  let devices: Device[] = [];
  try {
    const admin = createSupabaseAdminClient();
    const { data } = await admin.from("desktop_sessions").select("id,device_name,os,last_seen,created_at").eq("user_id", user.id).order("last_seen", { ascending: false });
    devices = data ?? [];
  } catch { /* dev */ }

  return (
    <div style={{ display: "grid", gap: 18 }}>
      <div className="card" style={{ padding: 24 }}>
        <div className="kicker">authorized devices</div>
        <h2 style={{ fontSize: 24, fontWeight: 800, letterSpacing: "-0.02em", margin: "8px 0 6px" }}>Devices</h2>
        <p style={{ color: "var(--ink-2)", fontSize: 14, margin: 0, lineHeight: 1.55 }}>
          Every machine where you&apos;re signed in to the SPARC desktop client. Sign out remotely if a device is lost.
        </p>
      </div>

      <div className="card" style={{ padding: 0, overflow: "hidden" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
          <thead><tr style={{ background: "var(--paper-2)", borderBottom: "1px solid var(--line)" }}>
            {["name", "os", "last seen", "added", ""].map((h) => (
              <th key={h} style={{ textAlign: "left", padding: "12px 18px", fontFamily: "var(--font-mono)", fontSize: 11, letterSpacing: "0.1em", textTransform: "uppercase", color: "var(--muted)" }}>{h}</th>
            ))}
          </tr></thead>
          <tbody>
            {devices.length === 0 ? (
              <tr><td colSpan={5} style={{ padding: 22, textAlign: "center", color: "var(--muted)" }}>No devices signed in yet.</td></tr>
            ) : devices.map((d) => (
              <tr key={d.id} style={{ borderBottom: "1px dotted var(--line)" }}>
                <td style={{ padding: "12px 18px" }}>{d.device_name}</td>
                <td style={{ padding: "12px 18px" }} className="mono">{d.os}</td>
                <td style={{ padding: "12px 18px" }} className="mono">{new Date(d.last_seen).toLocaleString()}</td>
                <td style={{ padding: "12px 18px" }} className="mono">{new Date(d.created_at).toLocaleDateString()}</td>
                <td style={{ padding: "12px 18px", textAlign: "right" }}>
                  <form action="/api/account/devices/revoke" method="post" style={{ display: "inline" }}>
                    <input type="hidden" name="id" value={d.id} />
                    <button type="submit" className="btn btn-sm">Sign out</button>
                  </form>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
