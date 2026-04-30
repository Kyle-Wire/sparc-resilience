import { NextRequest, NextResponse } from "next/server";
import { createSupabaseAdminClient } from "@/lib/supabase/server";

export const runtime = "nodejs";

export async function POST(req: NextRequest) {
  let body: Record<string, unknown>;
  try { body = await req.json(); } catch { return NextResponse.json({ error: "invalid json" }, { status: 400 }); }

  const email = String(body.email ?? "").trim().toLowerCase();
  if (!email || !/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email)) {
    return NextResponse.json({ error: "valid email required" }, { status: 400 });
  }

  const row = {
    email,
    name: String(body.name ?? "").slice(0, 200) || null,
    organization: String(body.org ?? "").slice(0, 200) || null,
    domain: String(body.domain ?? "").slice(0, 64) || null,
    crs: String(body.crs ?? "").slice(0, 64) || null,
    reason: String(body.reason ?? "").slice(0, 4000) || null,
    user_agent: req.headers.get("user-agent") ?? null,
    referer: req.headers.get("referer") ?? null,
  };

  try {
    const admin = createSupabaseAdminClient();
    const { data, error } = await admin.from("waitlist").upsert(row, { onConflict: "email" }).select("id").single();
    if (error) throw error;

    // Optional notification — fire-and-forget so the user isn't blocked on email delivery.
    void notify(row).catch(() => {});

    return NextResponse.json({ ok: true, ticket: String(data.id).slice(0, 8) });
  } catch (err) {
    // If Supabase isn't configured yet we still want the form to feel real in dev.
    if (process.env.NODE_ENV !== "production") {
      console.warn("[waitlist] persist skipped:", err);
      return NextResponse.json({ ok: true, ticket: "DEV-" + Math.random().toString(36).slice(2, 8) });
    }
    return NextResponse.json({ error: "could not save request" }, { status: 500 });
  }
}

async function notify(row: Record<string, unknown>) {
  const key = process.env.RESEND_API_KEY;
  const to = process.env.WAITLIST_NOTIFY_EMAIL;
  if (!key || !to) return;
  await fetch("https://api.resend.com/emails", {
    method: "POST",
    headers: { authorization: `Bearer ${key}`, "content-type": "application/json" },
    body: JSON.stringify({
      from: "SPARC Waitlist <waitlist@sparclabs.co>",
      to: [to],
      subject: `Waitlist · ${row.email}`,
      text: Object.entries(row).map(([k, v]) => `${k}: ${v ?? ""}`).join("\n"),
    }),
  });
}
