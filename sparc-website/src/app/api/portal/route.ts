import { NextRequest, NextResponse } from "next/server";
import { createSupabaseServerClient, createSupabaseAdminClient } from "@/lib/supabase/server";

export const runtime = "nodejs";

// Returns the customer portal URL for the current user's most recent subscription.
export async function GET(req: NextRequest) {
  const supabase = await createSupabaseServerClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return NextResponse.redirect(new URL("/login?next=/account/subscription", req.url));

  const admin = createSupabaseAdminClient();
  const { data, error } = await admin
    .from("subscriptions")
    .select("customer_portal_url")
    .eq("user_id", user.id)
    .order("updated_at", { ascending: false })
    .limit(1)
    .maybeSingle();

  if (error || !data?.customer_portal_url) {
    return NextResponse.redirect(new URL("/account/subscription?error=no_portal", req.url));
  }
  return NextResponse.redirect(data.customer_portal_url, { status: 303 });
}
