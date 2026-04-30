import { NextRequest, NextResponse } from "next/server";
import { createSupabaseServerClient, createSupabaseAdminClient } from "@/lib/supabase/server";

export const runtime = "nodejs";

export async function POST(req: NextRequest) {
  const supabase = await createSupabaseServerClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return NextResponse.redirect(new URL("/login", req.url));

  const form = await req.formData();
  const id = String(form.get("id") ?? "");
  if (!id) return NextResponse.redirect(new URL("/account/devices?error=missing_id", req.url));

  try {
    const admin = createSupabaseAdminClient();
    await admin.from("desktop_sessions").delete().eq("id", id).eq("user_id", user.id);
  } catch { /* dev */ }
  return NextResponse.redirect(new URL("/account/devices?revoked=1", req.url), { status: 303 });
}
