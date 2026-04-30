import { NextRequest, NextResponse } from "next/server";
import { createSupabaseServerClient } from "@/lib/supabase/server";

export const runtime = "nodejs";

export async function POST(req: NextRequest) {
  const supabase = await createSupabaseServerClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return NextResponse.redirect(new URL("/login", req.url));

  const form = await req.formData();
  const full_name = String(form.get("full_name") ?? "").slice(0, 200);
  const organization = String(form.get("organization") ?? "").slice(0, 200);

  await supabase.auth.updateUser({ data: { full_name, organization } });
  return NextResponse.redirect(new URL("/account/settings?saved=1", req.url), { status: 303 });
}
