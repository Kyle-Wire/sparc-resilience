// supabase/functions/lemon-webhook/index.ts
//
// Receives Lemon Squeezy webhooks and reconciles them into the
// `licenses` (subscription events) and `license_keys` (license-key
// product events) tables.
//
// HMAC verification:
//   LS signs the raw body with `LEMON_SQUEEZY_WEBHOOK_SECRET` using
//   HMAC-SHA256 and sends the hex digest in the `X-Signature` header.
//   We must verify against the *raw* body (no parsing first).
//
// Events handled (subset of LS's full event set):
//   subscription_created, subscription_updated, subscription_cancelled,
//   subscription_resumed, subscription_expired,
//   license_key_created, license_key_updated
//
// All other events are accepted with a 200 (so LS doesn't keep
// retrying) but logged and ignored.

import { createClient } from "https://esm.sh/@supabase/supabase-js@2.45.0";
import { corsHeaders, jsonResponse } from "../_shared/cors.ts";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const WEBHOOK_SECRET = Deno.env.get("LEMON_SQUEEZY_WEBHOOK_SECRET") ?? "";

// Same variant→plan map as redeem-key. Keep in sync (or migrate to DB).
const VARIANT_PLAN_MAP: Record<string, "pro" | "team"> = {
  "358410": "pro",  // SPARC Pro
  // "<team_variant_id>": "team",
};

function variantToPlan(variantId: string | undefined): "pro" | "team" {
  if (variantId && VARIANT_PLAN_MAP[variantId]) return VARIANT_PLAN_MAP[variantId];
  return "pro";
}

async function verifySignature(rawBody: string, signature: string): Promise<boolean> {
  if (!WEBHOOK_SECRET) return false;
  const enc = new TextEncoder();
  const key = await crypto.subtle.importKey(
    "raw",
    enc.encode(WEBHOOK_SECRET),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const sigBytes = await crypto.subtle.sign("HMAC", key, enc.encode(rawBody));
  const hex = Array.from(new Uint8Array(sigBytes))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
  // Constant-time compare.
  if (hex.length !== signature.length) return false;
  let diff = 0;
  for (let i = 0; i < hex.length; i++) {
    diff |= hex.charCodeAt(i) ^ signature.charCodeAt(i);
  }
  return diff === 0;
}

function statusFromLs(lsStatus: string): "active" | "trialing" | "past_due" | "cancelled" | "expired" {
  switch (lsStatus) {
    case "on_trial":
      return "trialing";
    case "active":
      return "active";
    case "past_due":
    case "unpaid":
      return "past_due";
    case "cancelled":
      return "cancelled";
    case "expired":
      return "expired";
    default:
      return "active";
  }
}

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }
  if (req.method !== "POST") {
    return jsonResponse({ error: "method_not_allowed" }, { status: 405 });
  }

  const rawBody = await req.text();
  const signature = req.headers.get("X-Signature") ?? "";
  const ok = await verifySignature(rawBody, signature);
  if (!ok) {
    return jsonResponse({ error: "invalid_signature" }, { status: 401 });
  }

  let payload: any;
  try {
    payload = JSON.parse(rawBody);
  } catch {
    return jsonResponse({ error: "invalid_json" }, { status: 400 });
  }

  const eventName: string = payload?.meta?.event_name ?? "";
  const customData: Record<string, unknown> = payload?.meta?.custom_data ?? {};
  const data = payload?.data ?? {};
  const attrs = data?.attributes ?? {};

  const admin = createClient(SUPABASE_URL, SERVICE_ROLE_KEY, {
    auth: { persistSession: false, autoRefreshToken: false },
  });

  // ─── Subscription events → licenses table ────────────────────
  if (eventName.startsWith("subscription_")) {
    // We require the checkout to set `custom_data.user_id` =
    // Supabase auth.users.id so we can map LS subscriptions back to
    // the right user. Without it the row is orphaned.
    const userId = String(customData.user_id ?? "");
    if (!userId) {
      console.warn("subscription event without user_id custom_data", eventName);
      return jsonResponse({ ok: true, ignored: "no_user_id" });
    }

    const plan = variantToPlan(String(attrs.variant_id ?? ""));
    const status = statusFromLs(String(attrs.status ?? ""));
    const renewsAt: string | null = attrs.renews_at ?? attrs.ends_at ?? null;

    const { error } = await admin.from("licenses").upsert(
      {
        user_id: userId,
        plan,
        status,
        current_period_end: renewsAt,
        ls_subscription_id: String(data.id ?? ""),
        ls_customer_id: String(attrs.customer_id ?? ""),
        ls_variant_id: String(attrs.variant_id ?? ""),
        updated_at: new Date().toISOString(),
      },
      { onConflict: "user_id" },
    );
    if (error) {
      console.error("licenses upsert failed", error);
      return jsonResponse({ error: "db_upsert_failed" }, { status: 500 });
    }
    return jsonResponse({ ok: true });
  }

  // ─── License-key events → license_keys table ─────────────────
  if (eventName.startsWith("license_key_")) {
    const key = String(attrs.key ?? "");
    if (!key) {
      return jsonResponse({ ok: true, ignored: "no_key" });
    }
    const plan = variantToPlan(String(attrs.variant_id ?? ""));
    const lsStatus = String(attrs.status ?? "active");
    const status = lsStatus === "active" ? "active"
      : lsStatus === "disabled" ? "disabled"
      : lsStatus === "expired" ? "expired"
      : "active";
    const { error } = await admin.from("license_keys").upsert(
      {
        key,
        plan,
        status,
        ls_order_id: String(attrs.order_id ?? ""),
        ls_license_key_id: String(data.id ?? ""),
        expiry: attrs.expires_at ?? null,
        updated_at: new Date().toISOString(),
      },
      { onConflict: "key" },
    );
    if (error) {
      console.error("license_keys upsert failed", error);
      return jsonResponse({ error: "db_upsert_failed" }, { status: 500 });
    }
    return jsonResponse({ ok: true });
  }

  // Unknown event — ack so LS doesn't retry forever.
  return jsonResponse({ ok: true, ignored: eventName });
});
