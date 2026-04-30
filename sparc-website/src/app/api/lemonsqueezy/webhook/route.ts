import { NextRequest, NextResponse } from "next/server";
import { createSupabaseAdminClient } from "@/lib/supabase/server";
import { verifyWebhookSignature, type LemonSqueezyEvent } from "@/lib/lemonsqueezy";

export const runtime = "nodejs";

interface LSWebhook {
  meta: {
    event_name: LemonSqueezyEvent;
    custom_data?: { user_id?: string; tier?: string };
  };
  data: {
    id: string;
    type: string;
    attributes: Record<string, unknown> & {
      user_email?: string;
      status?: string;
      product_id?: number;
      variant_id?: number;
      customer_id?: number;
      renews_at?: string | null;
      ends_at?: string | null;
      trial_ends_at?: string | null;
      cancelled?: boolean;
      urls?: { customer_portal?: string };
    };
  };
}

export async function POST(req: NextRequest) {
  const raw = await req.text();
  const sig = req.headers.get("x-signature");
  if (!verifyWebhookSignature(raw, sig)) {
    return NextResponse.json({ error: "invalid signature" }, { status: 401 });
  }

  let payload: LSWebhook;
  try { payload = JSON.parse(raw); } catch { return NextResponse.json({ error: "bad json" }, { status: 400 }); }

  const event = payload.meta.event_name;
  const userId = payload.meta.custom_data?.user_id;
  const tier = payload.meta.custom_data?.tier;
  const attr = payload.data.attributes;

  if (!userId) {
    // Order without user_id (e.g. one-off) — log but ack so LS doesn't retry forever.
    console.warn("[lemonsqueezy] event without user_id:", event);
    return NextResponse.json({ ok: true });
  }

  const admin = createSupabaseAdminClient();

  try {
    if (event.startsWith("subscription_")) {
      const status = attr.status ?? (event === "subscription_cancelled" ? "cancelled" : event === "subscription_expired" ? "expired" : "active");
      await admin.from("subscriptions").upsert({
        user_id: userId,
        tier: tier ?? null,
        ls_subscription_id: payload.data.id,
        ls_customer_id: attr.customer_id ?? null,
        ls_product_id: attr.product_id ?? null,
        ls_variant_id: attr.variant_id ?? null,
        status,
        renews_at: attr.renews_at ?? null,
        ends_at: attr.ends_at ?? null,
        trial_ends_at: attr.trial_ends_at ?? null,
        customer_portal_url: attr.urls?.customer_portal ?? null,
        raw: attr,
        updated_at: new Date().toISOString(),
      }, { onConflict: "ls_subscription_id" });
    } else if (event === "order_created") {
      await admin.from("orders").upsert({
        user_id: userId,
        ls_order_id: payload.data.id,
        tier: tier ?? null,
        raw: attr,
        created_at: new Date().toISOString(),
      }, { onConflict: "ls_order_id" });
    } else if (event === "order_refunded") {
      await admin.from("orders").update({ refunded_at: new Date().toISOString(), raw: attr }).eq("ls_order_id", payload.data.id);
    }

    return NextResponse.json({ ok: true });
  } catch (err) {
    console.error("[lemonsqueezy] persist failed:", event, err);
    return NextResponse.json({ error: "persist failed" }, { status: 500 });
  }
}
