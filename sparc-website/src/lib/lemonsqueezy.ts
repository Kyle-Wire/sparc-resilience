// Lemon Squeezy helpers — checkout creation + webhook signature verification.
import crypto from "node:crypto";
import { lemonSqueezySetup, createCheckout } from "@lemonsqueezy/lemonsqueezy.js";

let configured = false;
function ensureConfigured() {
  if (configured) return;
  const apiKey = process.env.LEMONSQUEEZY_API_KEY;
  if (!apiKey) throw new Error("LEMONSQUEEZY_API_KEY is not set.");
  lemonSqueezySetup({ apiKey });
  configured = true;
}

export type Tier = "wander" | "pursue" | "converge" | "transcend";

export function getVariantId(tier: Exclude<Tier, "wander" | "transcend">, period: "monthly" | "yearly"): string | undefined {
  const map: Record<string, string | undefined> = {
    pursue_monthly: process.env.LEMONSQUEEZY_VARIANT_PURSUE_MONTHLY,
    pursue_yearly: process.env.LEMONSQUEEZY_VARIANT_PURSUE_YEARLY,
    converge_monthly: process.env.LEMONSQUEEZY_VARIANT_CONVERGE_MONTHLY,
    converge_yearly: process.env.LEMONSQUEEZY_VARIANT_CONVERGE_YEARLY,
  };
  return map[`${tier}_${period}`];
}

export async function createUserCheckout(args: {
  variantId: string;
  userId: string;
  email: string;
  tier: Tier;
  redirectUrl: string;
}) {
  ensureConfigured();
  const storeId = process.env.LEMONSQUEEZY_STORE_ID;
  if (!storeId) throw new Error("LEMONSQUEEZY_STORE_ID is not set.");
  const result = await createCheckout(storeId, args.variantId, {
    checkoutData: {
      email: args.email,
      custom: { user_id: args.userId, tier: args.tier },
    },
    productOptions: { redirectUrl: args.redirectUrl },
    checkoutOptions: { embed: false },
  });
  return result.data?.data.attributes.url;
}

// Webhook signature verification — see https://docs.lemonsqueezy.com/api/webhooks
export function verifyWebhookSignature(rawBody: string, signature: string | null): boolean {
  const secret = process.env.LEMONSQUEEZY_WEBHOOK_SECRET;
  if (!secret || !signature) return false;
  const hmac = crypto.createHmac("sha256", secret);
  const digest = hmac.update(rawBody).digest("hex");
  try {
    return crypto.timingSafeEqual(Buffer.from(digest, "hex"), Buffer.from(signature, "hex"));
  } catch {
    return false;
  }
}

export type LemonSqueezyEvent =
  | "subscription_created"
  | "subscription_updated"
  | "subscription_cancelled"
  | "subscription_resumed"
  | "subscription_expired"
  | "subscription_paused"
  | "subscription_unpaused"
  | "order_created"
  | "order_refunded";
