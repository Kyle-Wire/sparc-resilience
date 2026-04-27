# Supabase backend for SPARC

This directory contains the SQL schema and three Edge Functions that
back the SPARC desktop app's auth + license system.

```
supabase/
├── migrations/
│   └── 0001_init_licenses.sql      # licenses + license_keys tables, RLS
└── functions/
    ├── _shared/cors.ts             # shared CORS + JSON helpers
    ├── verify-license/index.ts     # canonical license-state lookup
    ├── redeem-key/index.ts         # LS license-key activation
    └── lemon-webhook/index.ts      # LS webhook → DB reconciler
```

## Setup

1. **Create a Supabase project** at https://supabase.com.
2. **Enable auth providers**: Google, GitHub, Microsoft (Azure), and
   Email + Password. For each OAuth provider add the redirect URL
   `http://127.0.0.1:<random>/callback` (the desktop generates a
   random loopback port per launch via `tauri-plugin-oauth`).
3. **Run the migration**:
   ```bash
   supabase db push
   # or, manually:
   psql "$SUPABASE_DB_URL" -f supabase/migrations/0001_init_licenses.sql
   ```
4. **Set Edge Function secrets** (in the Supabase dashboard → Settings
   → Edge Functions):
   - `LEMON_SQUEEZY_API_KEY`         — server-side LS API key
   - `LEMON_SQUEEZY_STORE_ID`        — numeric LS store id
   - `LEMON_SQUEEZY_WEBHOOK_SECRET`  — HMAC secret you set in LS
5. **Deploy Edge Functions**:
   ```bash
   supabase functions deploy verify-license
   supabase functions deploy redeem-key
   supabase functions deploy lemon-webhook
   ```
6. **Configure Lemon Squeezy webhook**: point it at
   `https://<project>.supabase.co/functions/v1/lemon-webhook` with
   the same secret. Subscribe to:
   `subscription_*`, `license_key_*`.
7. **Map LS variant IDs → SPARC plans**: edit the `VARIANT_PLAN_MAP`
   constant at the top of `redeem-key/index.ts` and `lemon-webhook/index.ts`.
8. **Wire the desktop app**: copy `sparc-desktop/.env.example` to
   `.env.local` and fill in `VITE_SUPABASE_URL` + `VITE_SUPABASE_ANON_KEY`.

## How the pieces fit together

**Subscription flow (Pro/Team):**
1. User clicks *Upgrade* in Settings → opens LS checkout in the
   browser (with `custom_data.user_id` set to their auth UID).
2. LS hits `lemon-webhook` → upsert into `licenses` for that user.
3. Desktop polls `verify-license` → reads the new row → flips plan to
   `pro` / `team` in the local cache.

**License-key flow (no account required):**
1. User buys a one-time license-key product on LS.
2. User pastes the key into the desktop's "Use license key" tab.
3. Desktop calls `redeem-key` → activates with LS → upserts into
   `license_keys`.
4. Subsequent launches call `verify-license` with the key.

**Offline grace:**
The desktop caches `{plan, status, expiry, last_verified}` in
`tauri-plugin-store` and tolerates a 15-day network outage. After 15
days without a successful `verify-license`, the app blocks all pages
except Settings.

## Creator bypass (you, the maintainer)

If you build with `VITE_SPARC_CREATOR_MODE=1` (already set in
`sparc-desktop/.env.local`), the desktop mints a perpetual `team`
license at startup with no auth. You'll never see the login wall on
your own machine. Distributed binaries should NOT have this env var
set at build time.
