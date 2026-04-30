# sparclabs.co — SPARC Labs marketing site & account portal

Next.js 15 (App Router, TypeScript) site for **SPARC Labs** (_Spatial Research Labs_).
Marketing pages, gated `/account/*` portal, Supabase auth, Lemon Squeezy billing.

## Stack

- **Next.js 15** (App Router · typed routes · server components by default)
- **Supabase** for auth (magic link + Google + Microsoft) and storage; Vault for BYOK secrets
- **Lemon Squeezy** for billing (Pursue, Converge per-seat); Transcend = sales-assisted
- **Plain CSS** with [`tokens.css`](src/styles/tokens.css) as the design source of truth
- Brand strings live exclusively in [`src/lib/brand.ts`](src/lib/brand.ts)

## Setup

```bash
cd sparc-website
pnpm install          # or: npm install
cp .env.example .env.local   # fill in Supabase + Lemon Squeezy values
pnpm dev              # http://localhost:3000
```

## Environment

| Variable | Purpose |
|---|---|
| `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Browser + server Supabase clients |
| `SUPABASE_SERVICE_ROLE_KEY` | Webhook + admin operations only — never exposed to the browser |
| `LEMONSQUEEZY_API_KEY`, `LEMONSQUEEZY_STORE_ID` | Checkout creation |
| `LEMONSQUEEZY_WEBHOOK_SECRET` | HMAC verification on `/api/lemonsqueezy/webhook` |
| `LEMONSQUEEZY_VARIANT_PURSUE_{MONTHLY,YEARLY}` | Variant IDs for Pursue plan |
| `LEMONSQUEEZY_VARIANT_CONVERGE_{MONTHLY,YEARLY}` | Variant IDs for Converge plan |
| `RESEND_API_KEY`, `WAITLIST_NOTIFY_EMAIL` | Optional waitlist notifications |
| `NEXT_PUBLIC_SITE_URL` | Optional override (defaults to `https://sparclabs.co`) |

## Database

Apply [`supabase/migrations/0001_initial.sql`](supabase/migrations/0001_initial.sql)
to provision tables (`profiles`, `subscriptions`, `orders`, `seats`, `license_keys`,
`api_keys`, `desktop_sessions`, `usage_events`, `waitlist`) with strict RLS.

## Auth flow

| Surface | Path | Notes |
|---|---|---|
| Web sign-in | `/login` | Magic link or OAuth |
| Web sign-up | `/signup` | Same form, `shouldCreateUser: true` |
| Callback | `/auth/callback` | Exchanges `code` → cookie session |
| Desktop deep-link | `/auth/callback?desktop=1&state=…` | Bounces to `sparc://auth/callback?code=…&state=…` |
| Sign out | `POST /auth/signout` | |

## Billing flow

1. User picks a plan on `/pricing` → `GET /api/checkout?tier=…&period=…`
2. We create a Lemon Squeezy checkout with `custom_data.user_id`
3. LS posts to `/api/lemonsqueezy/webhook` (HMAC-verified) and we upsert `subscriptions`
4. User manages billing at `/account/subscription` → `/api/portal`

## Account portal

`/account` (gated by middleware): overview, subscription, seats, downloads & licenses,
BYOK API keys, usage, devices, settings.

## Deploy

Vercel. Set env vars; point `sparclabs.co` at the project. The Lemon Squeezy webhook
URL is `https://sparclabs.co/api/lemonsqueezy/webhook`.

## Verify

```bash
pnpm typecheck
pnpm lint
pnpm build
```
