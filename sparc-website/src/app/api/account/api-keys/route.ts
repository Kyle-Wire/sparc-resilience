import { NextRequest, NextResponse } from "next/server";
import { createSupabaseServerClient, createSupabaseAdminClient } from "@/lib/supabase/server";

export const runtime = "nodejs";

export async function POST(req: NextRequest) {
  const supabase = await createSupabaseServerClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return NextResponse.json({ error: "unauthenticated" }, { status: 401 });

  const body = await req.json().catch(() => null) as { provider?: string; label?: string; secret?: string } | null;
  if (!body?.provider || !body?.label || !body?.secret) {
    return NextResponse.json({ error: "provider, label, and secret are required" }, { status: 400 });
  }

  try {
    const admin = createSupabaseAdminClient();
    // Persist via the encrypt_api_key Postgres function (defined in migration 0001).
    // The function wraps Supabase Vault and returns the row WITHOUT the secret.
    const { data, error } = await admin.rpc("encrypt_api_key", {
      p_user_id: user.id,
      p_provider: body.provider,
      p_label: body.label,
      p_secret: body.secret,
    });
    if (error) throw error;
    return NextResponse.json({ ok: true, key: data });
  } catch (err) {
    console.error("[api-keys] add failed:", err);
    return NextResponse.json({ error: "could not save key" }, { status: 500 });
  }
}

export async function DELETE(req: NextRequest) {
  const supabase = await createSupabaseServerClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return NextResponse.json({ error: "unauthenticated" }, { status: 401 });

  const id = new URL(req.url).searchParams.get("id");
  if (!id) return NextResponse.json({ error: "id required" }, { status: 400 });

  try {
    const admin = createSupabaseAdminClient();
    const { error } = await admin.from("api_keys").delete().eq("id", id).eq("user_id", user.id);
    if (error) throw error;
    return NextResponse.json({ ok: true });
  } catch (err) {
    console.error("[api-keys] delete failed:", err);
    return NextResponse.json({ error: "could not delete" }, { status: 500 });
  }
}
