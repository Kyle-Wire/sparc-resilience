import { createSupabaseServerClient, createSupabaseAdminClient } from "@/lib/supabase/server";

export const dynamic = "force-dynamic";

interface UsageRow { metric: string; total: number; }

export default async function UsagePage() {
  const supabase = await createSupabaseServerClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return null;

  let rows: UsageRow[] = [];
  try {
    const admin = createSupabaseAdminClient();
    const since = new Date(); since.setDate(1); since.setHours(0, 0, 0, 0);
    const { data } = await admin.from("usage_events").select("metric,quantity").eq("user_id", user.id).gte("occurred_at", since.toISOString());
    const agg = new Map<string, number>();
    (data ?? []).forEach((d: { metric: string; quantity: number }) => agg.set(d.metric, (agg.get(d.metric) ?? 0) + Number(d.quantity)));
    rows = [...agg.entries()].map(([metric, total]) => ({ metric, total }));
  } catch { /* dev */ }

  return (
    <div style={{ display: "grid", gap: 18 }}>
      <div className="card" style={{ padding: 24 }}>
        <div className="kicker">this billing period</div>
        <h2 style={{ fontSize: 24, fontWeight: 800, letterSpacing: "-0.02em", margin: "8px 0 6px" }}>Usage</h2>
        <p style={{ color: "var(--ink-2)", fontSize: 14, margin: 0, lineHeight: 1.55 }}>
          Reported by your desktop app. SPARC counts only metadata — no project contents, no rasters, no prompts.
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 14 }}>
        {(rows.length === 0 ? [
          { metric: "pipeline_runs", total: 0 },
          { metric: "scenarios", total: 0 },
          { metric: "llm_tokens", total: 0 },
        ] : rows).map((r) => (
          <div key={r.metric} className="card" style={{ padding: 22 }}>
            <div className="kicker">{r.metric.replace(/_/g, " ")}</div>
            <div style={{ fontSize: 32, fontWeight: 800, letterSpacing: "-0.025em", marginTop: 6, fontFeatureSettings: "'tnum'" }}>{r.total.toLocaleString()}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
