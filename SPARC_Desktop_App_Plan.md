# SPARC Labs Desktop Application — Development Plan

**Version 1.0 — April 2026**

---

## Executive Summary

This plan covers the transformation of SPARC from a Streamlit-based demo into a polished, brand-aligned desktop application. It spans four workstreams: a Tauri + Python sidecar shell, an Anthropic Claude integration for guided project setup, spatial visualization powered by deck.gl, and a design system rooted in the SPARC Labs brand guidelines.

The existing SPARC Python pipeline (`sparc/` package, CLI, all 13 domain templates) remains untouched. The desktop app wraps it; it does not rewrite it.

---

## 1. Architecture: Tauri v2 + Python Sidecar

### Why Tauri Over Electron

Tauri v2 uses the operating system's native webview (WebKit on macOS, WebView2 on Windows) instead of bundling Chromium, which keeps the app binary small (~5–15 MB vs. Electron's ~150 MB) and memory overhead low. This matters because the SPARC pipeline itself is memory-intensive — Stage 2 model training with LightGBM, torch-based Deep Kriging, and mgwr on 50,000+ spatial points already demands significant RAM. The UI layer should be lightweight.

### Python Sidecar Pattern

Tauri's sidecar system bundles an external binary alongside the app. The SPARC Python codebase gets compiled into a standalone executable via **PyInstaller** (`--onefile` mode), which is then registered in `tauri.conf.json` under `bundle.externalBin`. The React frontend invokes it through Tauri's shell plugin:

```
Frontend (React)
    │
    ▼
Tauri IPC (shell plugin)
    │
    ▼
Python sidecar (PyInstaller binary)
    │  ├── sparc validate --project project.yml
    │  ├── sparc run --project project.yml --stage 0
    │  ├── sparc run --project project.yml --stage 2
    │  └── ... (all existing CLI commands)
    │
    ▼
stdout/stderr → streamed back to frontend as progress events
```

**Key design decisions:**

- **Two communication modes.** Short commands (validate, init) use `Command.sidecar().execute()` for a one-shot call. Long-running pipeline stages use `Command.sidecar().spawn()` with streaming stdout/stderr captured as events — the frontend displays a live progress panel showing stage progression, fold counts, and metric updates.

- **FastAPI local server option.** For richer bidirectional communication (e.g., querying intermediate results, browsing saved runs), the sidecar can optionally start a FastAPI server on `localhost:8008`. The frontend communicates via HTTP. Tauri manages the server lifecycle — spawning on app launch, killing on window close.

- **Development mode.** During development, rather than recompiling the PyInstaller binary on every change, the Tauri dev server runs the Python code directly from the interpreter (pass `--dev-sidecar` flag). PyInstaller compilation only happens for `pnpm tauri build`.

### Project Structure

```
sparc-desktop/
├── src/                          # React frontend (TypeScript)
│   ├── components/
│   │   ├── layout/               # Shell, sidebar, topbar
│   │   ├── pipeline/             # Stage progress, config editor
│   │   ├── dag/                  # DAG visual editor (React Flow)
│   │   ├── map/                  # deck.gl spatial views
│   │   ├── chat/                 # LLM conversation panel
│   │   └── results/              # Charts, tables, diagnostics
│   ├── hooks/                    # useSidecar, useAnthropicChat, etc.
│   ├── lib/
│   │   ├── design-tokens.ts      # Brand palette, typography
│   │   └── ipc.ts                # Tauri command wrappers
│   ├── App.tsx
│   └── main.tsx
├── src-tauri/
│   ├── src/main.rs               # Tauri app entry, sidecar management
│   ├── capabilities/default.json # Shell permissions
│   ├── tauri.conf.json           # Bundle config, externalBin
│   ├── binaries/                 # PyInstaller-compiled SPARC
│   └── icons/                    # SPARC Labs logo (isometric cube)
├── sparc/                        # Existing Python package (symlinked or copied)
├── build-sidecar.py              # PyInstaller build script
├── package.json
├── tailwind.config.ts
├── tsconfig.json
└── vite.config.ts
```

### Build & Distribution

- **macOS:** `.dmg` installer. PyInstaller binary targets `aarch64-apple-darwin` (Apple Silicon) and `x86_64-apple-darwin` (Intel). Tauri handles universal binary creation.
- **Windows:** `.msi` or `.exe` installer via Tauri's WiX or NSIS bundler. PyInstaller binary targets `x86_64-pc-windows-msvc`.
- **Linux:** `.AppImage` or `.deb`. PyInstaller binary targets `x86_64-unknown-linux-gnu`.

The PyInstaller binary must be named with the target triple suffix (e.g., `sparc-sidecar-x86_64-apple-darwin`) per Tauri's sidecar convention.

---

## 2. Anthropic Claude Integration

### API vs. Subscription — What You Need to Know

**Your Pro subscription ($20/month) is for claude.ai only — it does not include API access.** The API and the consumer product are completely separate billing systems. To use Claude in the SPARC desktop app, you (and your users) need an Anthropic API key from `console.anthropic.com`, with prepaid credits.

Current API pricing (as of March 2026):

| Model | Input (per 1M tokens) | Output (per 1M tokens) | Best for |
|---|---|---|---|
| **Haiku 4.5** | $1 | $5 | Quick template matching, simple Q&A |
| **Sonnet 4.6** | $3 | $15 | DAG construction, physics reasoning |
| **Opus 4.6** | $5 | $25 | Complex domain reasoning (overkill for this use case) |

**Cost estimate for a typical SPARC project setup conversation:**

A full guided setup (domain selection → DAG construction → physics constraints) runs about 8–12 conversation turns. With a well-crafted system prompt (~3,000 tokens), user messages (~200 tokens each), and Claude responses (~500 tokens each), that's roughly:

- Input: ~10,000 tokens per session (including cached system prompt)
- Output: ~5,000 tokens per session
- **Per-session cost with Sonnet 4.6: ~$0.11**

That's about 11 cents per project setup. Even heavy iteration (30+ turns refining a DAG) stays under $0.50. Prompt caching can reduce this further — the system prompt with all 13 template definitions only needs to be sent fresh once, then cached at 90% discount for subsequent turns.

**For your own development and testing**, you'll want to set up an API account with $20–50 in prepaid credits. That gives you hundreds of test conversations. This is separate from your Pro subscription.

### User Experience: Bring Your Own Key

The app includes a Settings panel where users paste their Anthropic API key. The key is stored locally in the OS keychain via Tauri's secure storage plugin — never sent to any server other than `api.anthropic.com`. If no key is present, the LLM features are simply hidden and the user configures their project manually (the existing workflow).

### Integration Architecture

The LLM integration is a **guided conversation** scoped to three tasks, not a general-purpose chatbot. The conversation panel lives in a collapsible sidebar alongside the project configuration views.

```
┌─────────────────────────────────────────────────────┐
│  SPARC Labs                          [Settings] [≡] │
├──────────┬──────────────────────────────────────────┤
│          │                                          │
│  Chat    │        Main workspace                    │
│  Panel   │        (config / DAG editor / map)       │
│          │                                          │
│  ┌────┐  │                                          │
│  │User│  │                                          │
│  │msg │  │                                          │
│  └────┘  │                                          │
│  ┌────┐  │                                          │
│  │AI  │  │                                          │
│  │resp│  │                                          │
│  └────┘  │                                          │
│          │                                          │
│  [input] │                                          │
│          │                                          │
├──────────┴──────────────────────────────────────────┤
│  Stage 0 ● │ Stage 1 ○ │ Stage 2 ○ │ ...           │
└─────────────────────────────────────────────────────┘
```

### System Prompt Design

The system prompt is the core of the integration. It contains:

1. **Role definition.** Claude acts as a spatial analysis consultant helping configure a SPARC project. It knows the pipeline stages, the YAML schema, and the domain science.

2. **Template library.** Condensed descriptions of all 13 domain templates (UHI, ForceSMIP, groundwater, air quality, etc.) with their default predictors, physics priors, and DAG structures. ~1,500 tokens.

3. **YAML schema.** The `project.yml` JSON schema so Claude can generate valid configuration. ~800 tokens.

4. **User's data context.** Dynamically injected when the user uploads a CSV — column names, row count, coordinate range, and basic summary statistics. ~200 tokens.

5. **Output format instructions.** Claude responds conversationally but wraps structured outputs in JSON code blocks that the frontend parses:

```json
{
  "action": "suggest_template",
  "template": "uhi",
  "predictors": ["Pct_Canopy", "Pct_Impervious", "NDVI", "Albedo"],
  "reasoning": "Based on your columns and target variable..."
}
```

```json
{
  "action": "propose_dag_edges",
  "edges": [
    {"source": "Pct_Canopy", "target": "NDVI", "type": "causal"},
    {"source": "Pct_Canopy", "target": "AAT_z", "type": "causal"},
    {"source": "Elevation_m", "target": "AAT_z", "type": "confounder"}
  ]
}
```

```json
{
  "action": "suggest_physics",
  "monotonic_constraints": {"Pct_Canopy": -1, "Pct_Impervious": 1, "NDVI": -1},
  "variable_bounds": {"Pct_Canopy": [0, 100], "Pct_Impervious": [0, 100]},
  "combined_constraints": [{"vars": ["Pct_Canopy", "Pct_Impervious"], "max_sum": 100}]
}
```

The frontend watches for these JSON blocks and automatically updates the DAG editor, config form, or physics panel in real time as Claude responds.

### Three Conversation Modes

| Mode | Trigger | Claude's Job |
|---|---|---|
| **Domain Setup** | User creates new project | Match description to template, suggest predictors from CSV columns, configure CRS |
| **DAG Construction** | User opens DAG editor | Propose causal edges, explain confounder vs. mediator, respond to "should X cause Y?" questions |
| **Physics Constraints** | User opens physics panel | Suggest monotonic directions, variable bounds, delta caps, and combined constraints based on domain literature |

Each mode uses a slightly different system prompt suffix that primes Claude for that specific task. The conversation history carries across modes so context isn't lost.

### Model Selection

**Sonnet 4.6 is the right choice** for this integration. It's the balanced option — strong enough for causal reasoning and domain-specific physics suggestions, fast enough for conversational responsiveness (~1–2 seconds to first token), and cost-efficient at $3/$15 per MTok. Haiku would struggle with the nuance of DAG construction; Opus is overkill and slower.

### React Implementation

```typescript
// hooks/useAnthropicChat.ts — simplified
const useAnthropicChat = (systemPrompt: string) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const sendMessage = async (content: string) => {
    const apiKey = await getSecureKey('anthropic-api-key');
    const updated = [...messages, { role: 'user', content }];
    setMessages(updated);
    setIsLoading(true);

    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'x-api-key': apiKey },
      body: JSON.stringify({
        model: 'claude-sonnet-4-6',
        max_tokens: 1024,
        system: systemPrompt,
        messages: updated.map(m => ({ role: m.role, content: m.content })),
      }),
    });

    const data = await response.json();
    const assistantMsg = data.content[0].text;

    // Parse any structured JSON blocks for DAG/config updates
    const structuredActions = extractJsonBlocks(assistantMsg);
    structuredActions.forEach(action => dispatchAction(action));

    setMessages([...updated, { role: 'assistant', content: assistantMsg }]);
    setIsLoading(false);
  };

  return { messages, sendMessage, isLoading };
};
```

The API call goes directly from the Tauri webview to `api.anthropic.com` — no backend proxy needed. Tauri's CSP configuration allows this outbound request. The user's API key stays local.

---

## 3. Spatial Visualization & Mapping

### Why deck.gl

The SPARC pipeline outputs dense spatial datasets — 54,000+ points for the Brown UHI study, 71,000+ grid cells for ForceSMIP. Matplotlib generates static PNGs. The desktop app replaces these with interactive, GPU-accelerated WebGL map views using **deck.gl** with **react-map-gl** (MapLibre GL JS backend, which is free and open-source — no Mapbox token required for the base tiles).

deck.gl handles millions of points at 60fps through GPU instancing, which is critical for scrubbing between scenario interventions in real time.

### Map Views by Pipeline Stage

| Stage | Visualization | deck.gl Layer Type |
|---|---|---|
| **Stage 0** (Correlogram) | Spatial autocorrelation heatmap, bandwidth selection | `HeatmapLayer` |
| **Stage 2** (Model training) | Predicted vs. actual maps, residual maps | `ScatterplotLayer` with brand color ramp |
| **Stage 3** (Causal validation) | CATE spatial heterogeneity, treatment effect maps | `GeoJsonLayer` with diverging palette |
| **Stage 4** (Scenarios) | Intervention comparison maps, uncertainty bands | `ScatterplotLayer` with scenario toggle slider |

### Brand Color Ramp for Maps

The SPARC Labs brand palette (from the design guide page 9) translates directly into a continuous spatial color ramp:

```
#602468 → #9e337d → #e94d9b → #e94461 → #e73c25 → #e76c25 → #e79024 → #f0b632 → #fbdd46
(purple)   (magenta)  (pink)    (red)     (crimson)  (orange)  (amber)   (gold)    (yellow)
```

This is implemented as a D3 scale and a deck.gl color accessor:

```typescript
import { scaleLinear } from 'd3-scale';

const SPARC_RAMP = [
  [96, 36, 104],    // #602468
  [158, 51, 125],   // #9e337d
  [233, 77, 155],   // #e94d9b
  [233, 68, 97],    // #e94461
  [231, 60, 37],    // #e73c25
  [231, 108, 37],   // #e76c25
  [231, 144, 36],   // #e79024
  [240, 182, 50],   // #f0b632
  [251, 221, 70],   // #fbdd46
];

const colorScale = scaleLinear<number[]>()
  .domain(SPARC_RAMP.map((_, i) => i / (SPARC_RAMP.length - 1)))
  .range(SPARC_RAMP);
```

For diverging variables (cooling vs. warming), the ramp is split: purple–pink for cooling (negative), orange–yellow for warming (positive), with a neutral gray midpoint.

### DAG Visual Editor

The DAG editor uses **@xyflow/react** (formerly React Flow) to provide a drag-and-drop node/edge graph that maps directly to the DoWhy causal structure:

- **Nodes** represent variables from the dataset (rendered as rounded cards with the variable name and a sparkline of its distribution).
- **Edges** represent causal relationships, colored by type: blue for causal, gray for confounder, amber for mediator.
- **Claude integration:** When Claude suggests new edges via the chat panel, they appear as dashed "proposed" edges that the user can accept or reject with a click.
- **Export:** The graph serializes to the same JSON DAG format that `sparc/causal/` expects.

### Base Map

Use **MapLibre GL JS** via `react-map-gl/maplibre` — fully open-source, no API key required. Free tile providers include OpenFreeMap, Stadia Maps (free tier), or self-hosted PMTiles. The base map style should be minimal/muted (light gray streets, no labels clutter) to let the data layer dominate visually, consistent with the brand's "scientifically rigorous" tone.

---

## 4. Design System — Applying the Brand Guidelines

### Typography

**Primary typeface: Neue Haas Grotesk** (55 Roman for body, 65 Medium for headings). This is the commercial version of Helvetica Neue with optical refinements. Load as a web font in the Tauri webview. Fallback stack: `'Neue Haas Grotesk', 'Helvetica Neue', Helvetica, Arial, sans-serif`.

If licensing is a concern for distribution (Neue Haas Grotesk requires a commercial license), consider **Inter** or **Geist** as free alternatives that share similar proportions. However, the brand guide specifically calls for Neue Haas Grotesk, so licensing it for the desktop app is recommended.

### Color System

| Token | Hex | Usage |
|---|---|---|
| `--sparc-black` | `#000000` | Logo, primary text, headings |
| `--sparc-white` | `#ffffff` | Backgrounds, reversed text |
| `--sparc-purple` | `#602468` | Accent start, deep data values |
| `--sparc-magenta` | `#9e337d` | Secondary accent |
| `--sparc-pink` | `#e94d9b` | Highlights, interactive states |
| `--sparc-red` | `#e94461` | Alerts, high-intensity data |
| `--sparc-crimson` | `#e73c25` | Warnings, critical values |
| `--sparc-orange` | `#e76c25` | Warm accent |
| `--sparc-amber` | `#e79024` | Mid-range data values |
| `--sparc-gold` | `#f0b632` | Progress indicators |
| `--sparc-yellow` | `#fbdd46` | Light data values, success states |
| `--sparc-gray-100` | `#f5f5f5` | Background surfaces |
| `--sparc-gray-300` | `#d4d4d4` | Borders, grid lines |
| `--sparc-gray-600` | `#737373` | Secondary text |

### Grid-Paper Motif

The brand guide uses a fine grid-paper pattern on every page. In the app, this becomes a subtle CSS background on workspace panels:

```css
.workspace-panel {
  background-image:
    linear-gradient(var(--sparc-gray-300) 1px, transparent 1px),
    linear-gradient(90deg, var(--sparc-gray-300) 1px, transparent 1px);
  background-size: 24px 24px;
  background-position: -1px -1px;
  opacity: 0.04;  /* very subtle — data-first, texture second */
}
```

### Logo Usage Rules (from brand guide pages 6–8)

- Always black on white or white on black. Never on colored backgrounds.
- The isometric cube icon can be used standalone (e.g., favicon, window title bar) without the "SPARC LABS" wordmark.
- Window title bar: cube icon only. Sidebar header: full lockup (cube + wordmark).

### UI Tone

The brand guide's three pillars — **scientifically rigorous**, **climate focused**, **innovative** — translate to:

- Dense, information-rich layouts (no decorative whitespace for its own sake)
- Data tables and charts as first-class UI elements, not afterthoughts
- Restrained animation (progress bars, stage transitions) — nothing gratuitous
- The warm color ramp provides personality against the otherwise monochrome interface

---

## 5. Phased Delivery

### Phase 1: Shell + IPC Bridge (Weeks 1–4)

**Goal:** Feature parity with the Streamlit cloud demo, running as a native desktop app.

- [ ] Initialize Tauri v2 project with React + TypeScript + Vite
- [ ] Set up PyInstaller build script for the SPARC Python package
- [ ] Wire Tauri sidecar: `sparc validate`, `sparc run --stage N`
- [ ] Build basic UI: project creation, template selection, YAML config editor
- [ ] Implement data upload (CSV) with preview table
- [ ] Stage progress panel with streaming stdout capture
- [ ] CI/CD: GitHub Actions for building macOS/Windows/Linux installers

### Phase 2: Design System + LLM Integration (Weeks 5–8)

**Goal:** The app looks and feels like a SPARC Labs product, with Claude-powered project setup.

- [ ] Implement design token system (colors, typography, grid motif)
- [ ] Build the chat panel component with message history
- [ ] Craft system prompts for domain setup, DAG construction, physics constraints
- [ ] Implement `useAnthropicChat` hook with structured output parsing
- [ ] Settings panel with API key storage (Tauri secure storage)
- [ ] Integrate chat actions with config editor (auto-fill from Claude suggestions)
- [ ] Dark mode support with theme-aware brand colors

### Phase 3: DAG Editor + Spatial Visualization (Weeks 9–14)

**Goal:** Interactive causal graph editing and GPU-accelerated map views replace static outputs.

- [ ] Build DAG editor with @xyflow/react
- [ ] Wire Claude's proposed edges into the DAG editor as "suggestions"
- [ ] Implement deck.gl map component with SPARC brand color ramp
- [ ] Stage 2 results map (predicted vs. actual, residuals)
- [ ] Stage 4 scenario comparison view with intervention slider
- [ ] CATE heterogeneity map for Stage 3
- [ ] MapLibre base map with muted style
- [ ] GeoParquet data loading for large datasets

### Phase 4: Reports + Polish (Weeks 15–18)

**Goal:** Professional PDF/DOCX export and release-quality polish.

- [ ] Report template with SPARC Labs branding
- [ ] PDF generation (client-side or via Python sidecar)
- [ ] Results explorer: browse saved runs, compare across stages
- [ ] Keyboard shortcuts, accessibility audit
- [ ] Performance profiling (large dataset rendering, startup time)
- [ ] User documentation and onboarding flow
- [ ] Beta release for testing

---

## 6. Technology Stack Summary

| Layer | Technology | Purpose |
|---|---|---|
| **Desktop shell** | Tauri v2 (Rust) | Native window, file system access, sidecar management |
| **Frontend** | React 19 + TypeScript + Vite | UI components and state management |
| **Styling** | Tailwind CSS + custom design tokens | Brand-aligned design system |
| **Spatial viz** | deck.gl + react-map-gl + MapLibre GL JS | GPU-accelerated geospatial maps |
| **DAG editor** | @xyflow/react | Interactive causal graph editing |
| **Charts** | Recharts or D3 | Diagnostic plots, model performance |
| **LLM** | Anthropic Messages API (Sonnet 4.6) | Guided project setup, DAG assistance |
| **Backend** | Python 3.10+ (existing SPARC package) | Pipeline execution, all modeling |
| **Sidecar packaging** | PyInstaller | Bundle Python + deps into single binary |
| **Data format** | GeoParquet, CSV, project.yml | Spatial data interchange |

---

## 7. Cost Summary

| Item | Cost | Notes |
|---|---|---|
| **Anthropic API (development)** | ~$20–50 one-time | Prepaid credits for testing. Separate from Pro subscription. |
| **Anthropic API (per user session)** | ~$0.05–0.50 | Depends on conversation length. Users bring their own key. |
| **Neue Haas Grotesk license** | ~$50–200 | Desktop + web font license (one-time) |
| **MapLibre GL JS** | Free | Open-source, no API key needed |
| **deck.gl** | Free | Open-source (MIT license) |
| **Tauri** | Free | Open-source (MIT/Apache 2.0) |
| **Apple Developer Program** | $99/year | Required for macOS code signing + notarization |
| **Windows code signing cert** | ~$70–200/year | Optional but recommended for avoiding SmartScreen warnings |

---

*This plan is a living document. Each phase generates its own detailed implementation spec as work begins.*
