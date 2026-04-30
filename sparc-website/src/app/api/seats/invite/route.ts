import { NextRequest, NextResponse } from "next/server";
import { createSupabaseServerClient, createSupabaseAdminClient } from "@/lib/supabase/server";

export const runtime = "nodejs";

export async function POST(req: NextRequest) {
  const supabase = await createSupabaseServerClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return NextResponse.redirect(new URL("/login", req.url));

  const form = await req.formData();
  const email = String(form.get("email") ?? "").trim().toLowerCase();
  if (!email) return NextResponse.redirect(new URL("/account/seats?error=email_required", req.url));

  try {
    const admin = createSupabaseAdminClient();
    await admin.from("seats").upsert({
      owner_id: user.id,
      email,
      status: "invited",
      invited_at: new Date().toISOString(),
    }, { onConflict: "owner_id,email" });
  } catch { /* dev */ }

  return NextResponse.redirect(new URL("/account/seats?invited=1", req.url), { status: 303 });
}
