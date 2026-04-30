-- SPARC Labs · Initial schema
-- Tables: profiles, subscriptions, orders, seats, license_keys, api_keys, desktop_sessions, usage_events, waitlist
-- All tables have strict RLS. Service role bypasses RLS for server-side operations (webhooks etc).

create extension if not exists "pgcrypto";

-- ============================================================================
-- profiles · 1:1 with auth.users, auto-created on signup via trigger
-- ============================================================================
create table if not exists public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  email text not null,
  full_name text,
  organization text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

alter table public.profiles enable row level security;
create policy "profiles_self_read" on public.profiles for select using (auth.uid() = id);
create policy "profiles_self_update" on public.profiles for update using (auth.uid() = id);

create or replace function public.handle_new_user() returns trigger
language plpgsql security definer set search_path = public as $$
begin
  insert into public.profiles (id, email, full_name, organization)
  values (new.id, new.email, new.raw_user_meta_data->>'full_name', new.raw_user_meta_data->>'organization')
  on conflict (id) do nothing;
  return new;
end $$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created after insert on auth.users
for each row execute function public.handle_new_user();

-- ============================================================================
-- subscriptions · one row per Lemon Squeezy subscription, upserted by webhook
-- ============================================================================
create table if not exists public.subscriptions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  tier text,
  status text,
  ls_subscription_id text unique,
  ls_customer_id bigint,
  ls_product_id bigint,
  ls_variant_id bigint,
  seats_allowed int not null default 1,
  renews_at timestamptz,
  ends_at timestamptz,
  trial_ends_at timestamptz,
  customer_portal_url text,
  raw jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists subscriptions_user_idx on public.subscriptions (user_id, updated_at desc);
alter table public.subscriptions enable row level security;
create policy "subscriptions_self_read" on public.subscriptions for select using (auth.uid() = user_id);
-- writes only via service role (webhook)

-- ============================================================================
-- orders · one-off purchases / refunds
-- ============================================================================
create table if not exists public.orders (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  ls_order_id text unique,
  tier text,
  refunded_at timestamptz,
  raw jsonb,
  created_at timestamptz not null default now()
);
alter table public.orders enable row level security;
create policy "orders_self_read" on public.orders for select using (auth.uid() = user_id);

-- ============================================================================
-- seats · invitations into a Converge workspace
-- ============================================================================
create table if not exists public.seats (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null references auth.users(id) on delete cascade,
  member_id uuid references auth.users(id) on delete set null,
  email text not null,
  status text not null default 'invited', -- invited | active | revoked
  invited_at timestamptz,
  joined_at timestamptz,
  unique (owner_id, email)
);
alter table public.seats enable row level security;
create policy "seats_owner_read" on public.seats for select using (auth.uid() = owner_id);
create policy "seats_member_read" on public.seats for select using (auth.uid() = member_id);
create policy "seats_owner_write" on public.seats for all using (auth.uid() = owner_id) with check (auth.uid() = owner_id);

-- ============================================================================
-- license_keys · per-seat keys consumed by the desktop client
-- ============================================================================
create table if not exists public.license_keys (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  tier text not null,
  key_prefix text not null,         -- displayed in UI (e.g. "SP-XYZ")
  key_hash text not null,           -- bcrypt/sha hash of the full key, never the plaintext
  activated_at timestamptz,
  revoked_at timestamptz,
  created_at timestamptz not null default now()
);
create index if not exists license_keys_user_idx on public.license_keys (user_id);
alter table public.license_keys enable row level security;
create policy "license_keys_self_read" on public.license_keys for select using (auth.uid() = user_id);

-- ============================================================================
-- api_keys · BYOK secrets, encrypted via Supabase Vault
-- ============================================================================
create table if not exists public.api_keys (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  provider text not null,
  label text not null default 'default',
  vault_secret_id uuid not null,    -- references vault.secrets.id
  key_last4 text not null,
  created_at timestamptz not null default now(),
  unique (user_id, provider, label)
);
create index if not exists api_keys_user_idx on public.api_keys (user_id);
alter table public.api_keys enable row level security;
create policy "api_keys_self_read" on public.api_keys for select using (auth.uid() = user_id);
create policy "api_keys_self_delete" on public.api_keys for delete using (auth.uid() = user_id);

-- Helper RPC: encrypt and store a BYOK secret.
-- Returns the safe row (no plaintext) for client display.
create or replace function public.encrypt_api_key(
  p_user_id uuid, p_provider text, p_label text, p_secret text
) returns jsonb language plpgsql security definer set search_path = public, vault as $$
declare
  v_secret_id uuid;
  v_row record;
begin
  -- Use Supabase Vault to encrypt at rest.
  select vault.create_secret(p_secret, format('byok:%s:%s:%s', p_user_id, p_provider, p_label), null) into v_secret_id;
  insert into public.api_keys (user_id, provider, label, vault_secret_id, key_last4)
  values (p_user_id, p_provider, p_label, v_secret_id, right(p_secret, 4))
  on conflict (user_id, provider, label) do update
    set vault_secret_id = excluded.vault_secret_id, key_last4 = excluded.key_last4
  returning id, provider, label, key_last4, created_at into v_row;
  return jsonb_build_object('id', v_row.id, 'provider', v_row.provider, 'label', v_row.label, 'key_last4', v_row.key_last4, 'created_at', v_row.created_at);
end $$;

-- ============================================================================
-- desktop_sessions · sign-ins from Tauri client
-- ============================================================================
create table if not exists public.desktop_sessions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  device_name text not null,
  os text not null,
  app_version text,
  created_at timestamptz not null default now(),
  last_seen timestamptz not null default now()
);
create index if not exists desktop_sessions_user_idx on public.desktop_sessions (user_id, last_seen desc);
alter table public.desktop_sessions enable row level security;
create policy "desktop_sessions_self" on public.desktop_sessions for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

-- ============================================================================
-- usage_events · metering, append-only
-- ============================================================================
create table if not exists public.usage_events (
  id bigserial primary key,
  user_id uuid not null references auth.users(id) on delete cascade,
  metric text not null,            -- pipeline_runs | scenarios | llm_tokens | …
  quantity numeric not null default 1,
  meta jsonb,
  occurred_at timestamptz not null default now()
);
create index if not exists usage_events_user_time on public.usage_events (user_id, occurred_at desc);
alter table public.usage_events enable row level security;
create policy "usage_events_self_read" on public.usage_events for select using (auth.uid() = user_id);
-- inserts via service role only

-- ============================================================================
-- waitlist · public form submissions
-- ============================================================================
create table if not exists public.waitlist (
  id uuid primary key default gen_random_uuid(),
  email text unique not null,
  name text,
  organization text,
  domain text,
  crs text,
  reason text,
  user_agent text,
  referer text,
  status text not null default 'queued',
  created_at timestamptz not null default now()
);
alter table public.waitlist enable row level security;
-- no public policies — only service role inserts
