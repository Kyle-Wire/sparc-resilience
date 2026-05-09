# PRD: SPARC Desktop — Bootstrap Installer & Branded First-Run Experience

**Date:** 2026-05-09  
**Status:** Approved — ready for implementation  
**Target Version:** 1.1.0  
**Scope:** Build pipeline, Tauri/Rust host, React frontend (first-run wizard), CI/CD

---

## Problem Statement

The SPARC Desktop installer currently bundles the entire Python runtime, PyTorch, GeoPandas, and all ML dependencies into a single PyInstaller binary (~1 GB uncompressed, ~600 MB download). This creates three concrete problems:

1. **Download size friction** — a 600 MB installer is a significant barrier for trial users and makes every version update a large re-download
2. **No branded installer experience** — the NSIS/DMG installer is generic Tauri defaults; there is no SPARC visual identity at the moment of first contact
3. **Bundled deps are frozen** — updating a Python dependency (e.g. a NumPy security patch) requires a full new installer release; users must re-download 600 MB

The bootstrap approach solves all three: ship a ~38 MB installer containing only the Tauri shell, `uv`, and the React frontend. On first launch, a branded wizard downloads and installs the Python engine into `~/.sparc/env`. Subsequent engine-only updates happen silently on launch.

---

## Goals

1. Reduce installer download size from ~600 MB to ~38 MB
2. Deliver a branded first-run wizard with SPARC visual identity (color ramp, logo animation)
3. Decouple Python engine releases from app releases — engine updates happen silently without a new installer
4. Provide clear, recoverable failure handling if the download fails
5. Brand both the OS-level installer (NSIS/DMG) and the in-app first-run window
6. Publish compiled Python wheel to a public releases repo without exposing source code

---

## Non-Goals

- EULA / legal license text — deferred until legal text is ready
- User-selectable install path — fixed at `~/.sparc/env`
- Cloud/remote sidecar mode — this PRD is local-only; `~/.sparc/env` replaces the bundled binary
- Offline installation support — internet required for first-run engine download
- Uninstaller customization — OS handles uninstall; `~/.sparc/env` is left in place

---

## Design Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | Bundle `uv` as a Tauri sidecar binary | Reliable, pinned version, no extra network step; ~8 MB overhead negligible against 560 MB saved |
| 2 | Public `sparc-labs/sparc-releases` repo for wheel distribution | Private source stays private; compiled wheel (`.pyc`, no raw `.py`) is publicly downloadable; no auth token management in installer |
| 3 | Wheel URL baked at build time | Deterministic; installer always knows exactly what it expects; no GitHub API call at runtime |
| 4 | Fixed install path `~/.sparc/env` | Fewer failure modes; consistent location for support; matches Cursor/VS Code pattern |
| 5 | Silent engine upgrade on launch | First-run wizard is a one-time experience; updates should be invisible; brief status text on normal splash is sufficient |
| 6 | Inline error recovery (Retry + Quit) on download failure | Partial venv cleaned up on failure; wizard resumes from Downloading step on next launch |
| 7 | 4-step wizard: Welcome → Install Info → Downloading → Ready | EULA deferred; Install Location removed (fixed path); minimal steps = less friction |
| 8 | `splash-logo.png` loops with 6s pause on Downloading step | Matches existing brand asset; animated logo is more polished than a static progress bar alone |
| 9 | Branded NSIS header/sidebar + DMG background | Both OS installer moments carry SPARC identity; static image assets, no NSIS scripting complexity |
| 10 | Ships as version 1.1.0 | Semver minor bump is appropriate — this changes the delivery architecture, not the ML pipeline |

---

## Technical Approach

### Overview — What Changes

**Before (1.0.x):**
```
Installer (~600 MB)
└── sparc-sidecar binary (Python + all ML deps bundled by PyInstaller)
└── Tauri shell + React frontend
```

**After (1.1.0):**
```
Installer (~38 MB)
├── uv binary (Tauri sidecar, ~8 MB)
└── Tauri shell + React frontend

First launch:
  ~/.sparc/env/   ← created by uv, ~400 MB, persists across app updates
```

---

### Part 1 — Public Wheel Release Repo

#### 1.1 Create `sparc-labs/sparc-releases` GitHub repo
- Public repo, no source code
- Contains only: `README.md` (download instructions), `.github/workflows/` (receives wheel uploads)
- Release assets: `sparc-{VERSION}-py3-none-any.whl` per tag

#### 1.2 Build and publish wheel from private repo CI
Add a new step to `.github/workflows/build-desktop.yml` after the Python install step:

```yaml
- name: Build Python wheel (no source)
  run: python -m build --wheel --no-isolation

- name: Upload wheel to sparc-releases
  uses: actions/github-script@v7
  with:
    github-token: ${{ secrets.RELEASES_REPO_PAT }}
    script: |
      // Create release on sparc-labs/sparc-releases at matching tag
      // Upload .whl file as release asset
```

The wheel is built with `python -m build --wheel`. Source distribution is **not** built. The `.whl` contains compiled `.pyc` bytecode — no raw `.py` source.

`RELEASES_REPO_PAT` is a GitHub PAT with `repo` scope on `sparc-labs/sparc-releases` only, stored as a secret in the private repo.

#### 1.3 Wheel URL convention
```
https://github.com/sparc-labs/sparc-releases/releases/download/v{VERSION}/sparc-{VERSION}-py3-none-any.whl
```

This URL is injected into the Tauri build via an environment variable, exposed to the React frontend as `VITE_WHEEL_URL`, and stored in `Cargo.toml` env for the Rust layer.

---

### Part 2 — Bundle `uv` as Tauri Sidecar

#### 2.1 Add `uv` binaries to `src-tauri/binaries/`
CI downloads the correct `uv` binary for each target triple before building:

| Platform | Binary name |
|---|---|
| macOS arm64 | `uv-aarch64-apple-darwin` |
| macOS x64 | `uv-x86_64-apple-darwin` |
| Windows x64 | `uv-x86_64-pc-windows-msvc.exe` |

Version pinned in CI env var: `UV_VERSION: "0.7.x"` (latest stable at implementation time).

#### 2.2 Register `uv` in `tauri.conf.json`
```json
"bundle": {
  "externalBin": ["binaries/uv"]
}
```

#### 2.3 Expose `run_uv` Tauri command in `sidecar.rs`
```rust
#[tauri::command]
pub async fn run_uv(app: AppHandle, args: Vec<String>) -> Result<String, String>
```
- Resolves the `uv` sidecar binary path via `app.path().resource_dir()`
- Spawns with `stdout` + `stderr` piped, streams output lines back to frontend via Tauri events
- Returns combined stdout on success, error string on non-zero exit

---

### Part 3 — First-Run Detection & Routing

#### 3.1 Detection logic in `lib.rs` setup hook
On every launch, before spawning the existing Python sidecar:

```rust
let env_path = dirs::home_dir()
    .map(|h| h.join(".sparc").join("env"))
    .unwrap_or_default();

let env_ready = env_path.join(".sparc-version").exists()
    && std::fs::read_to_string(env_path.join(".sparc-version"))
        .unwrap_or_default()
        .trim() == env!("CARGO_PKG_VERSION");
```

- If `env_ready` → launch sidecar as today, show normal splash
- If not `env_ready` → open the `setup` window (a separate Tauri `WebviewWindow`) before the main window

#### 3.2 `.sparc-version` sentinel file
`uv` install step writes `~/.sparc/env/.sparc-version` containing the app version string on successful install. This is how the launch-time version mismatch check works — if the file is missing or the version doesn't match, the setup window (or silent upgrade) is triggered.

#### 3.3 Silent upgrade on launch (for existing installs)
When `.sparc-version` exists but contains an older version than the running app:
- Main window opens as normal (sidecar still launches with old env — backward compatible)
- A `upgrading_engine` event is emitted to the frontend
- Existing splash screen shows "Updating SPARC engine…" status text
- `run_uv(["pip", "install", "--upgrade", WHEEL_URL])` runs in background
- On completion, `.sparc-version` is updated; sidecar is restarted; status clears

---

### Part 4 — First-Run Wizard (React)

#### 4.1 New Tauri window: `setup`
Defined in `tauri.conf.json`:
```json
{
  "label": "setup",
  "url": "index.html#/setup",
  "title": "SPARC Setup",
  "width": 640,
  "height": 480,
  "resizable": false,
  "decorations": false,
  "center": true,
  "visible": false
}
```

`decorations: false` — the React component renders its own window chrome (close button, drag region) so the window has full design control.

#### 4.2 New route: `src/pages/Setup.tsx`
Rendered when `window.location.hash === "#/setup"` — no React Router needed, same pattern as existing single-page routing.

#### 4.3 Wizard steps

**Step 1 — Welcome**
- Full-bleed SPARC purple-to-gold gradient background (`#602468` → `#fbdd46`)
- `splash-logo.png` centered, static
- Headline: "Welcome to SPARC"
- Subheadline: "Spatial Analysis & Research Core"
- Single CTA button: "Get Started →" (SPARC crimson `#e73c25`)

**Step 2 — Install Info**
- White card on dark purple background
- Icon + "SPARC Engine" heading
- Body text: "SPARC will install its analysis engine (~400 MB) to your computer. This is a one-time setup that takes 1–3 minutes depending on your connection."
- Read-only path display: `~/.sparc/env` (monospace, muted)
- "Install Now" button → advances to Step 3

**Step 3 — Downloading**
- Dark background, `splash-logo.png` centered and animated:
  - CSS `@keyframes` fadeInOut — logo fades in over 1s, holds for 2s, fades out over 1s, **pauses 6 seconds**, repeats
- Below logo: status text line (e.g. "Downloading Python 3.11…", "Installing packages…", "Finalizing…")
- Progress bar — SPARC ramp gradient fill, width driven by percent received from Tauri events
- No skip/cancel button — installation must complete or fail cleanly

Progress events from `run_uv` Tauri command are parsed to extract percentage:
- `uv` emits lines like `Downloading sparc-1.1.0... [=====>    ] 47%`
- Frontend parses these to drive the progress bar

**Failure state (inline, same step):**
- Logo animation stops
- Status text turns SPARC crimson: "Download failed: [error message]"
- Two buttons appear: **"Try Again"** (re-runs install from scratch, clears partial venv) and **"Quit"**
- Partial `~/.sparc/env` is deleted before retry

**Step 4 — Ready**
- Full-bleed gradient (same as Step 1)
- Checkmark icon (SPARC gold)
- Headline: "SPARC is ready."
- Body: "Your analysis engine is installed and ready to use."
- "Launch SPARC" button → closes setup window, opens main window

#### 4.4 Animation spec
```css
@keyframes sparc-logo-pulse {
  0%   { opacity: 0; }
  10%  { opacity: 1; }          /* fade in over 1s */
  30%  { opacity: 1; }          /* hold 2s */
  40%  { opacity: 0; }          /* fade out over 1s */
  100% { opacity: 0; }          /* pause 6s (gap fills remaining 60% of cycle) */
}

.setup-logo {
  animation: sparc-logo-pulse 10s ease-in-out infinite;
}
```

Total cycle: 10s (1s in + 2s hold + 1s out + 6s pause = 10s).

---

### Part 5 — Branded OS Installer Assets

#### 5.1 NSIS (Windows) custom images
Configured in `tauri.conf.json` under `bundle.windows.nsis`:
```json
"headerImage": "assets/installer/nsis-header.bmp",
"sidebarImage": "assets/installer/nsis-sidebar.bmp"
```

Required sizes:
- Header image: 150 × 57 px, 24-bit BMP
- Sidebar image: 164 × 314 px, 24-bit BMP

Design: SPARC purple background, `splash-logo.png` or wordmark, no text overlay (NSIS adds its own).

#### 5.2 DMG (macOS) background
Configured in `tauri.conf.json` under `bundle.macOS.dmg`:
```json
"background": "assets/installer/dmg-background.png",
"windowSize": { "width": 660, "height": 400 }
```

Required size: 660 × 400 px (or 1320 × 800 px @2x for Retina), PNG.

Design: Dark SPARC purple background, arrow graphic from app icon to Applications folder (standard DMG pattern), SPARC wordmark in upper area, gold accent line.

**Asset creation:** These are static design files to be created by the team outside this PRD. Placeholder solid-color BMPs/PNGs are used during development so CI doesn't break.

---

### Part 6 — CI Changes

#### 6.1 Remove PyInstaller from build
Remove from `.github/workflows/build-desktop.yml`:
```yaml
pip install pyinstaller
# and the pnpm build:sidecar step
```

Replace `beforeBuildCommand` in `tauri.conf.json` (currently runs `pnpm build:sidecar`) with nothing — no sidecar to build.

#### 6.2 Download `uv` binary in CI before Tauri build
```yaml
- name: Download uv binary for sidecar bundling
  shell: bash
  run: |
    UV_VERSION="0.7.x"  # pin to tested version
    # Download platform-appropriate uv binary to src-tauri/binaries/
    # Rename to Tauri sidecar convention: uv-{target-triple}[.exe]
```

#### 6.3 Build and upload wheel to `sparc-labs/sparc-releases`
Only runs on macOS (wheel is `py3-none-any` — platform-independent Python):
```yaml
- name: Build Python wheel
  if: matrix.label == 'macOS-arm64'
  run: python -m build --wheel --no-isolation
```

#### 6.4 Update `pip-audit` step
Already fixed (`--skip-editable`). No additional changes needed.

---

## File Map — New & Changed Files

| File | Action | Notes |
|---|---|---|
| `sparc-desktop/src-tauri/binaries/uv-*` | New | Downloaded in CI; gitignored locally |
| `sparc-desktop/src-tauri/src/setup.rs` | New | `run_uv` Tauri command, first-run detection logic |
| `sparc-desktop/src-tauri/src/lib.rs` | Modified | Add setup window launch, import setup module |
| `sparc-desktop/src-tauri/tauri.conf.json` | Modified | Add `setup` window, `externalBin`, NSIS/DMG asset paths, remove `resources: ["binaries/*"]` |
| `sparc-desktop/src/pages/Setup.tsx` | New | 4-step wizard React component |
| `sparc-desktop/src/main.tsx` | Modified | Route `#/setup` to `Setup` page |
| `sparc-desktop/assets/installer/nsis-header.bmp` | New | Placeholder → real asset |
| `sparc-desktop/assets/installer/nsis-sidebar.bmp` | New | Placeholder → real asset |
| `sparc-desktop/assets/installer/dmg-background.png` | New | Placeholder → real asset |
| `.github/workflows/build-desktop.yml` | Modified | Remove PyInstaller, add uv download + wheel build/upload |
| `sparc-desktop/scripts/bump-version.sh` | No change | Version already propagated to `tauri.conf.json` → wheel URL is consistent |
| `pyproject.toml` | Minor | Add `[project.urls]` pointing to releases repo |
| `sparc-labs/sparc-releases` (new repo) | New repo | README only; assets uploaded by CI |

---

## Acceptance Criteria

- [ ] macOS DMG installer is ≤ 45 MB
- [ ] Windows NSIS installer is ≤ 45 MB
- [ ] On a clean machine (no Python, no SPARC), double-clicking the installer and completing the wizard results in a fully working SPARC app
- [ ] Wizard shows all 4 steps with correct SPARC branding (purple gradient, logo, crimson CTA)
- [ ] Logo animation on Step 3 loops with a visible 6-second pause between cycles
- [ ] Progress bar advances during download with no stalls > 2s on a normal connection
- [ ] Simulating a network failure during download shows the error state; clicking "Try Again" succeeds on a restored connection
- [ ] On a machine with an older `~/.sparc/env`, launching the new app upgrades silently without showing the wizard
- [ ] NSIS installer shows SPARC header and sidebar images (not Tauri defaults)
- [ ] macOS DMG shows SPARC background artwork
- [ ] `.sparc-version` sentinel file exists and contains `1.1.0` after successful install
- [ ] `sparc-labs/sparc-releases` repo has a public release with the `.whl` asset attached
- [ ] Wheel URL in the installed app matches the release tag (verified by inspecting compiled binary constants)
- [ ] `pnpm test` and `npx tsc --noEmit` both exit 0 after all changes

---

## Open Questions

- **Placeholder installer assets:** Who creates the final NSIS/DMG design assets? Until they exist, placeholder solid-color files allow CI to pass. Tracked separately from code work.
- **`uv` version pin:** Specific `uv` version to pin will be confirmed at implementation start (latest stable at that time).
- **`RELEASES_REPO_PAT` secret:** Needs to be created and added to the private repo's GitHub secrets before CI can upload wheels. One-time ops step.
- **Windows arm64:** Not in current build matrix. No change needed for this PRD.
