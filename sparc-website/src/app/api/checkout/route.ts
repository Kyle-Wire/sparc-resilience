import { NextRequest, NextResponse } from "next/server";
import { createSupabaseServerClient } from "@/lib/supabase/server";
import { createUserCheckout, getVariantId, type Tier } from "@/lib/lemonsqueezy";
import { SITE_URL } from "@/lib/brand";

export const runtime = "nodejs";

export async function GET(req: NextRequest) {
  const url = new URL(req.url);
  const tier = (url.searchParams.get("tier") ?? "") as Tier;
  const period = (url.searchParams.get("period") ?? "monthly") as "monthly" | "yearly";

  if (tier !== "pursue" && tier !== "converge") {
    return NextResponse.redirect(new URL("/pricing", req.url));
  }

  const supabase = await createSupabaseServerClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user || !user.email) {
    const next = encodeURIComponent(`/api/checkout?tier=${tier}&period=${period}`);
    return NextResponse.redirect(new URL(`/login?next=${next}`, req.url));
  }

  const variantId = getVariantId(tier, period);
  if (!variantId) {
    return NextResponse.json({ error: `Variant not configured for ${tier}_${period}.` }, { status: 500 });
  }

  try {
    const checkoutUrl = await createUserCheckout({
      variantId,
      userId: user.id,
      email: user.email,
      tier,
      redirectUrl: `${SITE_URL}/account/subscription?welcome=1`,
    });
    if (!checkoutUrl) throw new Error("Lemon Squeezy returned no URL.");
    return NextResponse.redirect(checkoutUrl, { status: 303 });
  } catch (err) {
    console.error("[checkout] failed:", err);
    return NextResponse.redirect(new URL("/pricing?error=checkout", req.url));
  }
}
