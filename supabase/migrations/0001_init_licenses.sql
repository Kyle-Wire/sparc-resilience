-- 0001_init_licenses.sql
-- SPARC v1 licensing schema: per-user subscription rows + standalone
-- license keys (for users who never created an auth account).
--
-- Conventions:
--   - All timestamps are timestamptz UTC.
--   - `plan` ∈ ('free','pro','team').
--   - `status` ∈ ('active','trialing','past_due','cancelled','expired').
--   - RLS is enabled on every table. Reads of `licenses` are limited
--     to the row's owning user. `license_keys` is NOT user-readable;
--     all access goes through the `verify-license` / `redeem-key`
--     Edge Functions (which use the service-role key).

-- ─── extensions ─────────────────────────────────────────────────
create extension if not exists "pgcrypto";

-- ─── licenses (one row per Supabase user) ───────────────────────
create table if not exists public.licenses (
  user_id              uuid primary key references auth.users(id) on delete cascade,
  plan                 text not null default 'free'
    check (plan in ('free','pro','team')),
  status               text not null default 'active'
    check (status in ('active','trialing','past_due','cancelled','expired')),
  current_period_end   timestamptz,
  ls_subscription_id   text,
  ls_customer_id       text,
  ls_variant_id        text,
  metadata             jsonb not null default '{}'::jsonb,
  created_at           timestamptz not null default now(),
  updated_at           timestamptz not null default now()
);

create index if not exists licenses_ls_subscription_idx
  on public.licenses (ls_subscription_id);

-- Auto-create a free `licenses` row whenever a new auth user is born.
create or replace function public.handle_new_user()
returns trigger language plpgsql security definer set search_path = public as $$
begin
  insert into public.licenses (user_id, plan, status)
  values (new.id, 'free', 'active')
  on conflict (user_id) do nothing;
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();

-- Touch updated_at on row update.
create or replace function public.set_updated_at()
returns trigger language plpgsql as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists licenses_set_updated_at on public.licenses;
create trigger licenses_set_updated_at
  before update on public.licenses
  for each row execute function public.set_updated_at();

-- ─── license_keys (standalone keys; may or may not link to a user) ─
create table if not exists public.license_keys (
  key                  text primary key,
  user_id              uuid references auth.users(id) on delete set null,
  plan                 text not null default 'pro'
    check (plan in ('free','pro','team')),
  status               text not null default 'active'
    check (status in ('active','disabled','expired','revoked')),
  ls_order_id          text,
  ls_license_key_id    text,
  activated_at         timestamptz,
  last_seen_at         timestamptz,
  expiry               timestamptz,
  metadata             jsonb not null default '{}'::jsonb,
  created_at           timestamptz not null default now(),
  updated_at           timestamptz not null default now()
);

create index if not exists license_keys_user_idx on public.license_keys (user_id);
create index if not exists license_keys_ls_order_idx on public.license_keys (ls_order_id);

drop trigger if exists license_keys_set_updated_at on public.license_keys;
create trigger license_keys_set_updated_at
  before update on public.license_keys
  for each row execute function public.set_updated_at();

-- ─── RLS ────────────────────────────────────────────────────────
alter table public.licenses     enable row level security;
alter table public.license_keys enable row level security;

-- A user can only read their own license row.
drop policy if exists licenses_select_own on public.licenses;
create policy licenses_select_own on public.licenses
  for select using (auth.uid() = user_id);

-- Nobody can write `licenses` from the client. Edge Functions use the
-- service role key (which bypasses RLS) for all mutations.
drop policy if exists licenses_no_client_write on public.licenses;
create policy licenses_no_client_write on public.licenses
  for all using (false) with check (false);

-- `license_keys` is opaque to clients. All access goes through Edge
-- Functions; no select/insert/update/delete policies are created.

-- ─── helper: canonical license JSON shape ───────────────────────
-- Returns the JSON the desktop expects from `verify-license`. Used by
-- the Edge Function via PostgREST.
create or replace function public.license_payload(
  p_plan text,
  p_status text,
  p_expiry timestamptz,
  p_email_confirmed boolean
) returns jsonb language sql immutable as $$
  select jsonb_build_object(
    'valid',           p_status in ('active','trialing'),
    'plan',            p_plan,
    'status',          p_status,
    'expiry',          p_expiry,
    'email_confirmed', coalesce(p_email_confirmed, true)
  );
$$;
