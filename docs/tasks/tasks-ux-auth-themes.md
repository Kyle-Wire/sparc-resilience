# Tasks: UX Overhaul — Auth, Themes, 4D Interactivity, Splash

Related PRD: docs/prd/prd-ux-auth-themes.md

## Tasks
- [ ] T01 — Install `@supabase/supabase-js`, create `.env` file
- [ ] T02 — Create `src/lib/supabase.ts` (client singleton)
- [ ] T03 — Create `src/stores/authStore.ts` (Zustand, session init/signIn/signOut)
- [ ] T04 — Rebuild `src/lib/theme.ts` (5 named presets, fix storage key)
- [ ] T05 — Create `src/hooks/useMouseParallax.ts`
- [ ] T06 — Update `vite.config.ts` to inject `__APP_VERSION__`
- [ ] T07 — Rewrite `src/components/layout/Splash.tsx` (real progress + version + parallax)
- [ ] T08 — Create `src/components/layout/LoginScreen.tsx` (email/password + parallax)
- [ ] T09 — Create `src/components/layout/AuthGate.tsx` (frosted overlay)
- [ ] T10 — Update `src/components/layout/Topbar.tsx` (user email/avatar)
- [ ] T11 — Update `src/components/pages/SettingsPage.tsx` (themes, account, parallax toggle)
- [ ] T12 — Update `src/App.tsx` (full Splash→Login→Shell flow)
- [ ] T13 — Add `vite-env.d.ts` type for `__APP_VERSION__`
