# PRD: UX Overhaul — Auth, Themes, 4D Interactivity, Splash

## Problem Statement
The SPARC Desktop app has no authentication, a broken theme system (key mismatch bug means themes don't persist), a hardcoded version on the splash screen with a fake loading bar, and no interactive UX polish. This PRD covers a full UX uplift.

## Goals
1. Supabase email+password auth with persistent sessions
2. Named theme presets (5 themes) that actually persist correctly
3. Splash screen shows real step-based loading progress + correct version
4. 4D mouse parallax effect on Splash, Login, and Shell (toggleable in Settings)
5. Frosted-glass auth gate overlay on all pages when unauthenticated
6. Account section in Settings + user email in Topbar
7. Sign-out returns to login screen in-app

## Non-Goals
- OAuth / Magic Link auth (future)
- Remote user profile database (user prefs stay in localStorage)
- Mobile / web version of the auth flow

## Design Decisions
- Supabase project: `dugemitlmtsuztpwxehy`, URL: `https://dugemitlmtsuztpwxehy.supabase.co`
- Anon key stored in `.env` as `VITE_SUPABASE_ANON_KEY`
- Session stored in localStorage by Supabase SDK (auto-refresh, 60-day tokens)
- Theme key unified to `sparc-theme` (fixes underscore/hyphen bug)
- Version injected at build time via `define: { __APP_VERSION__: pkg.version }` in vite.config.ts
- Unauthenticated users can browse all pages but see frosted overlay gate
- Settings page when unauthed: shows only login form + link to sparclabs.co

## Technical Approach

### New files
- `src/lib/supabase.ts` — Supabase client singleton
- `src/stores/authStore.ts` — Zustand store for session/user state
- `src/components/layout/LoginScreen.tsx` — Full-screen login UI with 4D parallax
- `src/components/layout/AuthGate.tsx` — Frosted overlay wrapper
- `src/hooks/useMouseParallax.ts` — Mouse-tracking parallax hook

### Modified files
- `src/lib/theme.ts` — Replace hue+tone with 5 named presets; fix storage key
- `src/components/layout/Splash.tsx` — Real progress steps, dynamic version, parallax
- `src/components/layout/Shell.tsx` — Pass parallax toggle setting down
- `src/components/layout/Topbar.tsx` — User avatar + email display
- `src/components/pages/SettingsPage.tsx` — Account section, theme presets, parallax toggle
- `src/App.tsx` — Wire Splash→Login→Shell flow; auth context
- `vite.config.ts` — Inject `__APP_VERSION__`

### Theme Presets
| Key | Name | Paper | Ink | Accent |
|-----|------|-------|-----|--------|
| `warm-paper` | Warm Paper | #f7f4ee | #1a1416 | #e73c25 (crimson) |
| `dark` | Dark | #141214 | #f0ebe4 | #e73c25 |
| `high-contrast` | High Contrast | #000000 | #ffffff | #ffff00 |
| `sparc-electric` | SPARC Electric | #0a0812 | #e8f4ff | #00d4ff (electric cyan) |
| `cool-slate` | Cool Slate | #f2f4f7 | #1e2535 | #5b7af0 (indigo) |

### App flow
```
App mounts
  → Splash renders (step 1: "Starting sidecar…")
  → useServer() resolves → step 2: "Restoring session…"  
  → authStore.init() → step 3: "Ready"
  → if session: show Shell
  → if no session: show LoginScreen
  → LoginScreen sign-in → set session → show Shell
  → Shell always renders; pages wrapped in <AuthGate>
  → AuthGate: if authed → render children; if not → render frosted overlay
  → SettingsPage when unauthed: only login form visible
```

## Acceptance Criteria
- [ ] Signing in with valid Supabase credentials opens the Shell
- [ ] Closing and reopening the app stays logged in
- [ ] Sign-out returns to LoginScreen without app restart
- [ ] Splash bar shows 3 distinct steps with labels
- [ ] Splash version matches `package.json` version
- [ ] All 5 theme presets apply correctly and persist across reload
- [ ] 4D parallax visible on Splash and Login; subtle in Shell
- [ ] Parallax can be toggled off in Settings and choice persists
- [ ] Unauthenticated page navigation shows frosted overlay gate
- [ ] Settings page when unauthed shows only login + sparclabs.co link

## Open Questions
- None — all resolved in design interview
